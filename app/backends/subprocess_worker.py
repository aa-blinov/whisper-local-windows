"""Inference worker subprocess entry point.

This module is launched as a child process from
:class:`app.backends.subprocess_backend.SubprocessBackend`.  It hosts a
real :class:`app.backends.registry_backend.RegistryBackend` and
dispatches commands sent over a ``multiprocessing.Pipe`` from the parent
process.

Why run the model in a separate process
---------------------------------------
On Windows ``onnx_asr.load_model`` calls ``LoadLibrary`` for ORT op
kernels (Conv / MatMul / cuDNN / cuBLAS / …).  Each ``LoadLibrary`` for
a not-yet-resident DLL acquires the **process-wide DLL loader-lock**.
While that lock is held, every other thread in the process — including
Qt's main-thread message pump — blocks on any Win32 syscall.  The
visible symptoms in a single-process app are: window cannot be moved,
scroll doesn't react, "(Not responding)" badge.

Pushing the inference pipeline into a worker process means the loader-
lock contention happens in *that* process; our Qt process is free to
paint, drag the window, and respond to wheel events at its normal pace.

Pipe protocol
-------------
Every parent → worker message is a tuple whose first element is the
command name (``"status"``, ``"transcribe"``, …).  The worker replies
with one of:

- ``("ok", payload)``  — command succeeded, payload is the return value
- ``("error", message_string)`` — command raised; the parent re-raises
  as RuntimeError (`SubprocessBackend._send_cmd`)

Out-of-band messages (not in response to a command):

- ``("progress", current_bytes, total_bytes, desc)`` — fired from tqdm
  during a HuggingFace download.  The parent's reader thread routes
  these straight to the user's progress callback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Late-bound at module level so tests can monkey-patch them with
# fakes without touching the real implementations.  ``RegistryBackend``
# imports ``onnx_asr`` indirectly which is heavy; we want the test
# suite to run the dispatch logic without that cost.
try:
    from app.backends.registry_backend import RegistryBackend as _RegistryBackend
except Exception:  # pragma: no cover — defensive: tests still want to import
    _RegistryBackend = None  # type: ignore[assignment]

try:
    from app.backends._progress import set_progress_callback as _set_progress_callback
except Exception:  # pragma: no cover
    def _set_progress_callback(_cb):
        return None


if TYPE_CHECKING:  # pragma: no cover
    from multiprocessing.connection import Connection


def _worker_main(child_conn: "Connection") -> None:
    """Worker process main loop.

    Receives commands from the parent over ``child_conn`` and
    dispatches them to a single :class:`RegistryBackend` instance
    that lives for the lifetime of this worker.

    Returns when the parent closes the pipe (EOFError) or sends
    ``("shutdown_worker",)`` — both are normal exits.
    """
    log = logging.getLogger("subprocess_worker")
    log.info("Worker process started")

    # Forward all logging records from this worker process back to the
    # parent over the pipe so users see them in the Logs view.  pythonw
    # has no stdout/stderr — without this, every "Loading ONNX model
    # …" / progress / error message produced inside the worker
    # disappears into the void.
    class _PipeLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                # ``getMessage()`` formats with args; the parent only
                # needs the rendered message + level + logger name.
                child_conn.send((
                    "log",
                    record.levelno,
                    record.name,
                    record.getMessage(),
                ))
            except Exception:  # pragma: no cover — pipe broken
                pass

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Drop any default handlers (basicConfig may have attached one
    # writing to stderr on import) so we don't double-log.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_PipeLogHandler())

    # Forward backend progress events back to the parent over the pipe.
    # tqdm fires this from any thread inside ``onnx_asr.load_model``;
    # the parent's reader thread will dispatch to the user's progress
    # callback.
    def _forward_progress(current: int, total: int, desc: str) -> None:
        try:
            child_conn.send(
                ("progress", int(current), int(total), str(desc))
            )
        except Exception:  # pragma: no cover — pipe broken
            pass

    _set_progress_callback(_forward_progress)

    backend = None

    while True:
        try:
            cmd = child_conn.recv()
        except (EOFError, OSError):
            log.info("Pipe closed by parent — exiting")
            break
        if not cmd:
            break

        op = cmd[0]
        args = cmd[1:] if len(cmd) > 1 else ()

        try:
            if op == "init":
                # ``args[0]`` is the kwargs dict for RegistryBackend.
                kwargs = args[0]
                if _RegistryBackend is None:
                    raise RuntimeError(
                        "RegistryBackend unavailable in worker process"
                    )
                backend = _RegistryBackend(**kwargs)
                child_conn.send(("ok", None))
                continue

            if op == "shutdown_worker":
                # Tear down the inner backend cleanly, then exit the loop.
                if backend is not None:
                    try:
                        backend.shutdown()
                    except Exception as exc:  # pragma: no cover
                        log.warning("Inner shutdown raised: %s", exc)
                child_conn.send(("ok", None))
                log.info("Shutdown requested — exiting")
                break

            if backend is None:
                child_conn.send(
                    ("error", "Backend not initialised — send 'init' first")
                )
                continue

            if op == "status":
                child_conn.send(("ok", backend.status()))
            elif op == "health_check":
                child_conn.send(("ok", backend.health_check()))
            elif op == "current_model":
                child_conn.send(("ok", backend.current_model()))
            elif op == "current_language":
                child_conn.send(("ok", backend.current_language()))
            elif op == "load":
                backend.load()
                child_conn.send(("ok", None))
            elif op == "transcribe":
                audio, sample_rate = args
                child_conn.send(
                    ("ok", backend.transcribe(audio, sample_rate=sample_rate))
                )
            elif op == "transcribe_file":
                path = args[0]
                child_conn.send(("ok", backend.transcribe_file(path)))
            elif op == "change_model":
                model, compute_type = args
                backend.change_model(model, compute_type=compute_type)
                child_conn.send(("ok", None))
            elif op == "cancel_load":
                backend.cancel_load()
                child_conn.send(("ok", None))
            elif op == "update_inference_settings":
                settings = args[0]
                backend.update_inference_settings(settings)
                child_conn.send(("ok", None))
            elif op == "shutdown":
                # Non-terminal shutdown: stop the inner backend but keep
                # the worker alive (legacy compatibility — main process
                # always uses ``shutdown_worker`` to actually terminate).
                backend.shutdown()
                child_conn.send(("ok", None))
            else:
                child_conn.send(("error", f"Unknown command: {op!r}"))
        except Exception as exc:
            log.exception("Worker command %r failed", op)
            child_conn.send(
                ("error", f"{type(exc).__name__}: {exc}")
            )


# multiprocessing.spawn re-imports this module on Windows when starting
# the worker — avoid running anything at module import time.
if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(
        "subprocess_worker is not meant to be run directly — "
        "spawn it via SubprocessBackend"
    )
