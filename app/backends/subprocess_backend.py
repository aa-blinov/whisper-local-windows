"""Out-of-process inference backend.

:class:`SubprocessBackend` implements the :class:`TranscriptionBackend`
Protocol but runs the actual :class:`RegistryBackend` in a worker
process spawned via ``multiprocessing.Process``.  Each public method
sends a command tuple over a duplex ``Pipe`` and blocks waiting for
the response.

Why
~~~
On Windows ``onnx_asr.load_model`` calls ``LoadLibrary`` for ORT op
kernels (Conv / MatMul / cuDNN / cuBLAS / …).  ``LoadLibrary`` for a
not-yet-resident DLL takes the **process-wide DLL loader-lock**;
while it's held, every other thread in the process — *including* the
Qt main-thread message pump — blocks on any Win32 syscall.  In a
single-process design that means the title bar can't be dragged, the
mouse wheel doesn't scroll, and Windows eventually marks the window
"(Not responding)".

By moving the model into a worker process, the loader-lock contention
stays inside *that* process; our Qt process keeps painting and
processing OS messages at full rate.

Pipe protocol
~~~~~~~~~~~~~
Every parent → worker message is a tuple ``(op, *args)``.  The worker
replies with one of:

- ``("ok", payload)`` — command succeeded; payload is the return value
- ``("error", str)`` — command raised; this client re-raises as
  RuntimeError on the calling thread

The worker also sends out-of-band ``("progress", current, total, desc)``
messages from tqdm during a HF download.  The parent's reader thread
peels those off the stream and dispatches them to the user-supplied
progress callback before queuing command responses.

Threading model
~~~~~~~~~~~~~~~
- A reader thread (``onnx-worker-reader``) loops on
  ``parent_conn.recv()`` and routes incoming messages.
- A send lock serialises ``_send_cmd`` calls so multiple Qt threads
  hitting the backend can't interleave commands and responses.
- Only one outstanding command at a time — Qt UI is single-writer for
  the backend in practice (transcribe / change_model don't overlap
  thanks to StateManager's queueing).
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
from typing import Any, Callable, Optional

import numpy as np

from app.backends.subprocess_worker import _worker_main


log = logging.getLogger(__name__)


# Per-command timeout caps.  Most commands return in milliseconds; the
# generous defaults are a safety net for genuinely long ops (model
# load + download from HuggingFace).
_FAST_TIMEOUT = 30.0       # status, current_model, set_progress_callback, …
_LOAD_TIMEOUT = 1800.0     # load(), change_model() — first-time download
_TRANSCRIBE_TIMEOUT = 600.0  # 10 min cap for very long audio files


class SubprocessBackend:
    """Drop-in replacement for :class:`RegistryBackend` that runs the
    inference pipeline in a worker process.

    Constructor arguments are forwarded verbatim to the worker's
    ``RegistryBackend(**kwargs)`` so callers don't need to know
    they're crossing a process boundary.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Start-method choice:
        #   - **Windows** must use ``spawn`` (fork unavailable).
        #   - **macOS dev (uv run)** uses ``spawn`` too — that's the
        #     OS default and it works against ``sys.executable`` =
        #     ``.venv/bin/python``.
        #   - **macOS frozen (.app via py2app)** must use ``fork``:
        #     ``sys.executable`` inside the bundle is the py2app
        #     launcher binary (``Contents/MacOS/Lazy to Text``),
        #     not a Python interpreter, so ``spawn`` fails with
        #     ``Worker pipe closed before init`` — the launcher
        #     can't handle the bootstrap argv multiprocessing
        #     passes to a Python child.  Fork inherits the
        #     already-imported runtime instead of re-execing, side-
        #     stepping the launcher entirely.  Apple deprecated
        #     fork-safety guarantees but our worker stays single-
        #     threaded until ``onnx_asr.load_model``, by which
        #     point the duplicated state is harmless.
        import sys as _sys

        if _sys.platform == "darwin" and getattr(_sys, "frozen", False):
            method = "fork"
        else:
            method = "spawn"
        try:
            multiprocessing.set_start_method(method, force=False)
        except (RuntimeError, AssertionError):
            pass

        self._parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        self._proc = multiprocessing.Process(
            target=_worker_main,
            args=(child_conn,),
            name="onnx-worker",
            daemon=True,
        )
        self._proc.start()

        # State for the reader thread + command pipeline.
        self._send_lock = threading.Lock()
        self._response_queue: "queue.Queue[tuple]" = queue.Queue()
        self._progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None
        self._shutdown = False

        # Status cache populated by ``status_change`` push messages
        # from the worker.  Removes the need for the parent to
        # ``_send_cmd(("status",))`` every 200 ms (RecordingController
        # poll) — that IPC round-trip would contend with the worker's
        # tqdm-progress + log forwarding stream and stutter the UI
        # during model loads.
        self._status_cache: str = "stopped"
        self._status_cache_lock = threading.Lock()

        # Active-EP cache populated by ``provider_change`` push
        # messages from the worker — see ``active_provider``. ``None``
        # means either no model is loaded yet or the underlying
        # backend doesn't surface the field.
        self._provider_cache: Optional[str] = None
        self._provider_cache_lock = threading.Lock()

        # Async init: don't block the caller waiting for the worker
        # to come up.  ``__init__`` returns immediately; the reader
        # thread captures the init ack into ``_init_event`` and any
        # subsequent ``_send_cmd`` blocks on that event before
        # sending its own command.  Why: Windows ``spawn`` re-execs
        # Python and re-imports onnx_asr (~3-7 s); doing that
        # synchronously here used to delay ``window.show()`` and the
        # user saw a blank screen for 5-10 s on app startup.
        self._init_event = threading.Event()
        self._init_error: Optional[str] = None

        self._reader_thread = threading.Thread(
            target=self._read_loop,
            name="onnx-worker-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # Fire the init command — non-blocking; the reader thread
        # will set ``_init_event`` when the worker acks.
        try:
            self._parent_conn.send(("init", kwargs))
        except (BrokenPipeError, OSError) as exc:  # pragma: no cover
            self._init_error = f"init send failed: {exc}"
            self._init_event.set()

    # ------------------------------------------------------------------ IPC

    def _read_loop(self) -> None:
        """Reader thread — splits the pipe stream into progress events
        (dispatched to the user callback), log records (re-emitted
        through the parent's logging), the init ack (consumed
        internally to flip ``_init_event``), and command responses
        (queued for ``_send_cmd``)."""
        while True:
            try:
                msg = self._parent_conn.recv()
            except (EOFError, OSError):
                # Worker process exited or the pipe was closed —
                # signal any waiting _send_cmd by enqueuing a sentinel
                # error so it doesn't block forever.  Also unblock
                # any caller still waiting on init.
                if not self._init_event.is_set():
                    self._init_error = "Worker pipe closed before init"
                    self._init_event.set()
                self._response_queue.put(
                    ("error", "Subprocess pipe closed unexpectedly")
                )
                return
            if not msg:
                self._response_queue.put(
                    ("error", "Subprocess sent empty message")
                )
                return

            if msg[0] == "progress":
                cb = self._progress_callback
                if cb is not None:
                    try:
                        cb(int(msg[1]), int(msg[2]), str(msg[3]))
                    except Exception as exc:  # pragma: no cover
                        log.warning("progress callback raised: %s", exc)
            elif msg[0] == "status_change":
                # Worker pushes this on every transition of its inner
                # backend.status() so the parent can answer
                # ``backend.status()`` from cache instead of doing an
                # IPC round-trip.
                try:
                    new_status = str(msg[1])
                except Exception:  # pragma: no cover — defensive
                    continue
                with self._status_cache_lock:
                    self._status_cache = new_status
            elif msg[0] == "provider_change":
                # Worker pushes the EP that ``onnx_asr.load_model`` is
                # actually using as soon as it knows (after the
                # session is built, including any retry-on-CPU
                # fallback). ``None`` is a valid value — model was
                # unloaded or never bound to a session.
                try:
                    new_provider = msg[1]
                except Exception:  # pragma: no cover — defensive
                    continue
                if new_provider is not None:
                    new_provider = str(new_provider)
                with self._provider_cache_lock:
                    self._provider_cache = new_provider
            elif msg[0] == "log":
                # Re-emit worker log records through the parent's
                # logging system so they land in app.log + Logs view.
                # Format: ("log", levelno, logger_name, message)
                try:
                    _, levelno, name, message = msg
                    logging.getLogger(name).log(int(levelno), "%s", message)
                except Exception as exc:  # pragma: no cover
                    log.warning("worker log forward failed: %s", exc)
            else:
                # First command response is the init ack (we sent
                # ``("init", …)`` immediately at construction).  Capture
                # it here so callers don't see it as the response to
                # their first ``_send_cmd``.
                if not self._init_event.is_set():
                    if msg[0] == "error":
                        self._init_error = str(msg[1])
                    self._init_event.set()
                else:
                    self._response_queue.put(msg)

    def _send_cmd(self, cmd: tuple, *, timeout: float = _FAST_TIMEOUT) -> Any:
        """Send a command and block until the worker replies.

        Blocks first on ``_init_event`` if the worker hasn't finished
        initialising yet — Windows ``spawn`` + heavy onnx_asr import
        can take 3-7 s after construction.  Callers running on the
        Qt main thread should keep that in mind; the autoload path
        deliberately runs on a daemon thread to avoid stalling the UI.

        Raises :class:`RuntimeError` if the worker reports an error
        (or never inits, or the pipe times out).  Thread-safe via
        ``_send_lock``.
        """
        if not self._init_event.is_set():
            if not self._init_event.wait(timeout=timeout):
                raise RuntimeError(
                    f"Subprocess worker failed to init within {timeout}s"
                )
        if self._init_error is not None:
            raise RuntimeError(f"Subprocess init failed: {self._init_error}")

        with self._send_lock:
            try:
                self._parent_conn.send(cmd)
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"Subprocess pipe broken: {exc}")
            try:
                kind, payload = self._response_queue.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError(
                    f"Subprocess command {cmd[0]!r} timed out after {timeout}s"
                )
            if kind == "error":
                raise RuntimeError(f"Subprocess error: {payload}")
            return payload

    # ------------------------------------------------------------------ TranscriptionBackend Protocol

    def status(self) -> str:
        # Served from the local cache populated by ``status_change``
        # push messages from the worker.  No IPC — RecordingController
        # polls this every 200 ms; an IPC round-trip per poll would
        # serialise against the worker's tqdm + log forwarding and
        # stutter the UI during a model load.
        with self._status_cache_lock:
            return self._status_cache

    def health_check(self) -> bool:
        # Same fast-path as ``status()`` — no IPC.
        return self.status() == "ready"

    def active_provider(self) -> Optional[str]:
        # Served from the local cache populated by ``provider_change``
        # push messages from the worker; no IPC.  ``None`` means
        # either no model is loaded yet (worker hasn't sent the first
        # ``provider_change`` since startup) or the inner backend
        # doesn't expose the field (legacy / fake doubles in tests).
        with self._provider_cache_lock:
            return self._provider_cache

    def current_model(self) -> str:
        return self._send_cmd(("current_model",), timeout=_FAST_TIMEOUT)

    def current_language(self) -> Optional[str]:
        return self._send_cmd(
            ("current_language",), timeout=_FAST_TIMEOUT,
        )

    def load(self) -> None:
        # ``load`` itself just spawns a daemon thread inside the
        # worker and returns immediately — the timeout only covers
        # the spawn handshake, not the actual model load.
        self._send_cmd(("load",), timeout=_FAST_TIMEOUT)

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,
    ) -> None:
        # Same rationale as ``load`` — fast handshake only.
        self._send_cmd(
            ("change_model", model, compute_type),
            timeout=_FAST_TIMEOUT,
        )

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000,
    ) -> Optional[str]:
        return self._send_cmd(
            ("transcribe", audio, sample_rate),
            timeout=_TRANSCRIBE_TIMEOUT,
        )

    def transcribe_file(self, path: str) -> Optional[str]:
        return self._send_cmd(
            ("transcribe_file", path),
            timeout=_TRANSCRIBE_TIMEOUT,
        )

    def cancel_load(self) -> None:
        try:
            self._send_cmd(("cancel_load",), timeout=_FAST_TIMEOUT)
        except Exception as exc:  # pragma: no cover — best-effort
            log.warning("cancel_load on subprocess raised: %s", exc)

    def update_inference_settings(self, settings: Any) -> None:
        try:
            self._send_cmd(
                ("update_inference_settings", settings),
                timeout=_FAST_TIMEOUT,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "update_inference_settings on subprocess raised: %s", exc,
            )

    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        # Worker always forwards via the pipe — we just remember the
        # local callback for the reader thread to invoke.
        self._progress_callback = callback

    def shutdown(self) -> None:
        """Tear down the worker process. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        # Politely ask the worker to drop the inner backend and exit
        # its receive loop, then wait briefly.  Fall back to terminate
        # if it doesn't quit on its own (stuck in C++ destructor).
        try:
            self._send_cmd(("shutdown_worker",), timeout=_FAST_TIMEOUT)
        except Exception as exc:  # pragma: no cover — pipe may be dead
            log.debug("shutdown_worker send failed: %s", exc)
        try:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                log.info("Worker still alive after 5s — terminating")
                self._proc.terminate()
                self._proc.join(timeout=2)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Worker process join/terminate raised: %s", exc)
        try:
            self._parent_conn.close()
        except Exception:
            pass
