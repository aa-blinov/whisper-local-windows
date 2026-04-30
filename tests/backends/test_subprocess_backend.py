"""Tests for SubprocessBackend — the IPC wrapper that hosts the
inference pipeline in a worker process.

The point of running ONNX inference in a worker process: when
``onnx_asr.load_model`` calls ``LoadLibrary`` for ORT op kernels
(Conv / MatMul / cuDNN / cuBLAS / …), it acquires the Win32 DLL
loader-lock.  In a single-process app that lock blocks every other
thread including the Qt main-thread message pump → window can't be
moved, scroll doesn't work, "(Not responding)".  Pushing the model
into a separate process means the loader-lock contention stays
inside that process; our Qt process is unaffected.

These tests don't spawn a real Python subprocess (slow + flaky in
CI).  They drive ``_worker_main`` synchronously through an
in-memory ``multiprocessing.Pipe`` and a real-thread reader, which
exercises the full command/response/progress flow.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---- _worker_main: command dispatch (in-process) ---------------------------


def _drive_worker(commands, *, init_kwargs=None):
    """Spin up ``_worker_main`` on a thread (NOT a subprocess) so we
    can exercise the dispatch logic without paying the spawn cost.

    Returns the parent-side ``Connection``, the worker thread handle,
    and a ``run_all()`` helper that ships every command + drains the
    matching responses.

    Streamed-command approach (vs. pre-load everything into the pipe
    buffer): Windows ``Pipe`` has a 64 KB buffer; large numpy audio
    arrays would deadlock if we tried to enqueue all commands before
    the worker drained the first batch.
    """
    from app.backends.subprocess_worker import _worker_main

    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)

    t = threading.Thread(target=_worker_main, args=(child_conn,), daemon=True)
    t.start()

    if init_kwargs is not None:
        parent_conn.send(("init", init_kwargs))

    def run_all(timeout: float = 5.0):
        """Send every command, drain responses, return the list of
        non-progress responses (init included if present)."""
        out = []
        # Drain init ack first if it was sent.
        if init_kwargs is not None:
            out.append(_recv_response(parent_conn, timeout))
        for cmd in commands:
            parent_conn.send(cmd)
            out.append(_recv_response(parent_conn, timeout))
        return out

    return parent_conn, t, run_all


def _recv_response(conn, timeout):
    """Read one command response from ``conn``, skipping out-of-band
    messages (``progress`` from tqdm, ``log`` from the worker's log
    forwarder, ``status_change`` from the status broadcaster)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(0.01, deadline - time.monotonic())
        if not conn.poll(timeout=remaining):
            raise TimeoutError(
                f"no response within {timeout}s"
            )
        msg = conn.recv()
        if msg and msg[0] in ("progress", "log", "status_change"):
            continue
        return msg


def _drain_responses(conn, count, *, timeout=5.0):
    """Pull ``count`` non-progress messages off the pipe (legacy helper
    kept for tests that pre-load all commands before driving — only
    safe for small payloads that fit in the pipe buffer)."""
    out = []
    for _ in range(count):
        out.append(_recv_response(conn, timeout))
    return out


def test_worker_init_creates_registry_backend(monkeypatch):
    """``init`` command must build a RegistryBackend with the kwargs
    forwarded from the parent — that's the whole "reuse existing
    chain in the worker" trick."""
    captured: dict = {}

    class _FakeBackend:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def status(self):
            return "stopped"

    import app.backends.subprocess_worker as worker_mod

    monkeypatch.setattr(worker_mod, "_RegistryBackend", _FakeBackend)

    parent, t, run_all = _drive_worker(
        [("shutdown_worker",)],
        init_kwargs={"model": "whisper-large-v3-turbo", "device": "auto"},
    )
    try:
        responses = run_all()
    finally:
        parent.close()
        t.join(timeout=2)

    assert responses[0] == ("ok", None)  # init ack
    assert captured["kwargs"] == {
        "model": "whisper-large-v3-turbo", "device": "auto",
    }


def test_worker_status_load_transcribe_dispatch(monkeypatch):
    """Standard read-side commands forward to the backend."""
    fake = MagicMock()
    fake.status.return_value = "ready"
    fake.health_check.return_value = True
    fake.current_model.return_value = "x/y"
    fake.current_language.return_value = "ru"
    fake.transcribe.return_value = "hello"

    import app.backends.subprocess_worker as worker_mod

    monkeypatch.setattr(
        worker_mod, "_RegistryBackend", lambda **_kw: fake,
    )

    # 1 second of mono float32 (4 KB) — comfortably fits the pipe
    # buffer with the streamed-command driver.
    audio = np.zeros(1000, dtype=np.float32)
    parent, t, run_all = _drive_worker(
        [
            ("status",),
            ("health_check",),
            ("current_model",),
            ("current_language",),
            ("load",),
            ("transcribe", audio, 16000),
            ("shutdown_worker",),
        ],
        init_kwargs={"model": "x"},
    )
    try:
        responses = run_all()
    finally:
        parent.close()
        t.join(timeout=2)

    # init ack + 7 commands
    assert responses[0] == ("ok", None)
    assert responses[1] == ("ok", "ready")
    assert responses[2] == ("ok", True)
    assert responses[3] == ("ok", "x/y")
    assert responses[4] == ("ok", "ru")
    assert responses[5] == ("ok", None)
    assert responses[6] == ("ok", "hello")
    fake.transcribe.assert_called_once()
    # Numpy array round-trips through pickle — verify shape preserved.
    audio_arg = fake.transcribe.call_args.args[0]
    assert audio_arg.shape == (1000,)


def test_worker_change_model_dispatch(monkeypatch):
    fake = MagicMock()

    import app.backends.subprocess_worker as worker_mod

    monkeypatch.setattr(
        worker_mod, "_RegistryBackend", lambda **_kw: fake,
    )

    parent, t, run_all = _drive_worker(
        [
            ("change_model", "new/model", "int8"),
            ("shutdown_worker",),
        ],
        init_kwargs={"model": "old/model"},
    )
    try:
        run_all()
    finally:
        parent.close()
        t.join(timeout=2)

    fake.change_model.assert_called_once_with(
        "new/model", compute_type="int8",
    )


def test_worker_uninitialised_returns_error(monkeypatch):
    """Sanity: commands sent before ``init`` get back an error tuple
    rather than crashing the worker — defensive against pipe protocol
    misuse."""
    parent, t, run_all = _drive_worker(
        [("status",), ("shutdown_worker",)],
    )
    try:
        responses = run_all()
    finally:
        parent.close()
        t.join(timeout=2)

    assert responses[0][0] == "error"
    assert "not initialised" in responses[0][1].lower()


def test_worker_command_exception_returns_error(monkeypatch):
    """Backend method raising must surface as an ``("error", str)``
    tuple, not bring the worker down."""
    fake = MagicMock()
    fake.status.side_effect = RuntimeError("simulated crash")

    import app.backends.subprocess_worker as worker_mod

    monkeypatch.setattr(
        worker_mod, "_RegistryBackend", lambda **_kw: fake,
    )

    parent, t, run_all = _drive_worker(
        [("status",), ("shutdown_worker",)],
        init_kwargs={"model": "x"},
    )
    try:
        responses = run_all()
    finally:
        parent.close()
        t.join(timeout=2)

    assert responses[0] == ("ok", None)  # init
    kind, payload = responses[1]
    assert kind == "error"
    assert "simulated crash" in payload


def test_worker_progress_callback_forwards_to_pipe(monkeypatch):
    """``set_progress_callback`` is called once at worker start with
    a function that pushes ``("progress", current, total, desc)``
    over the pipe.  When tqdm fires, the parent must see the event
    in the pipe stream."""
    captured_cb = []

    fake_set_cb = MagicMock(side_effect=lambda cb: captured_cb.append(cb))

    import app.backends.subprocess_worker as worker_mod

    monkeypatch.setattr(worker_mod, "_set_progress_callback", fake_set_cb)
    monkeypatch.setattr(
        worker_mod, "_RegistryBackend", lambda **_kw: MagicMock(),
    )

    parent, t, _run_all = _drive_worker(
        [],
        init_kwargs={"model": "x"},
    )
    try:
        # Drain init ack — registering the progress callback happens
        # at worker startup, before any command is processed, so the
        # init response is enough to know it's wired.
        ack = _recv_response(parent, 5.0)
        assert ack == ("ok", None)
        assert captured_cb, "worker did not call set_progress_callback"

        # Now invoke the worker's progress callback from the test
        # (simulating tqdm) and confirm it lands on the parent pipe.
        cb = captured_cb[0]
        cb(50, 100, "model.bin")

        assert parent.poll(timeout=2.0)
        msg = parent.recv()
        assert msg[0] == "progress"
        assert msg[1] == 50
        assert msg[2] == 100
        assert msg[3] == "model.bin"

        # Cleanly shut down the worker.
        parent.send(("shutdown_worker",))
        _recv_response(parent, 5.0)
    finally:
        parent.close()
        t.join(timeout=2)


# ---- SubprocessBackend client (mocked process) -----------------------------


class _FakeProcess:
    """Stand-in for ``multiprocessing.Process`` so tests don't fork."""

    def __init__(self, *_a, **_kw):
        self._alive = True

    def start(self):
        pass

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False

    def terminate(self):
        self._alive = False


class _ScriptedConn:
    """Bidirectional pipe stub.  Test scripts an ordered list of
    responses and provides a callback to invoke when ``send`` is
    called — keeps the test deterministic without race conditions."""

    def __init__(self, on_send):
        self._on_send = on_send
        self._lock = threading.Lock()
        self._inbox: list = []
        self._cond = threading.Condition(self._lock)
        self._closed = False

    # parent side
    def send(self, payload):
        self._on_send(payload, self)

    def recv(self):
        with self._cond:
            while not self._inbox and not self._closed:
                self._cond.wait()
            if self._closed and not self._inbox:
                raise EOFError()
            return self._inbox.pop(0)

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # test side: enqueue a response
    def push(self, msg):
        with self._cond:
            self._inbox.append(msg)
            self._cond.notify_all()


@pytest.fixture
def patched_subprocess(monkeypatch):
    """Wire SubprocessBackend so it doesn't actually spawn a process —
    each ``send`` invokes the test-supplied callback, which decides
    what to push onto the recv side.
    """
    import app.backends.subprocess_backend as mod

    sent: list = []

    def on_send(payload, conn):
        sent.append(payload)
        # Default: ack every command synchronously with ("ok", None).
        # Tests override by calling ``conn.push(...)`` themselves
        # before sending the trigger.
        if payload[0] == "init":
            conn.push(("ok", None))

    conn_holder = {}

    def fake_pipe(duplex=True):
        c = _ScriptedConn(on_send)
        conn_holder["conn"] = c
        return c, c  # parent and child are the same object — tests
                     # write to ``conn``, both sides see it

    # Production code switched from ``multiprocessing.Pipe()`` /
    # ``multiprocessing.Process()`` to ``ctx = multiprocessing.get_context("spawn")``
    # + ``ctx.Pipe()`` / ``ctx.Process()`` in the macOS-bundle fix
    # (commit d05d275).  The context object isn't the same as the
    # ``multiprocessing`` module — patches on the module never reach
    # the context's bound methods — so we have to intercept
    # ``get_context`` itself and return a fake whose ``Pipe`` /
    # ``Process`` we control.  Patches on the module stay too as a
    # safety net for any future code path that goes through the
    # bare module API directly.
    class _FakeCtx:
        Pipe = staticmethod(fake_pipe)
        Process = _FakeProcess

    monkeypatch.setattr(mod.multiprocessing, "Pipe", fake_pipe)
    monkeypatch.setattr(mod.multiprocessing, "Process", _FakeProcess)
    monkeypatch.setattr(
        mod.multiprocessing, "get_context", lambda _method=None: _FakeCtx
    )

    return {"sent": sent, "conn_holder": conn_holder}


def test_subprocess_backend_init_sends_init(patched_subprocess):
    from app.backends.subprocess_backend import SubprocessBackend

    SubprocessBackend(model="whisper-large-v3-turbo", device="auto")

    sent = patched_subprocess["sent"]
    assert sent[0][0] == "init"
    kwargs = sent[0][1]
    assert kwargs["model"] == "whisper-large-v3-turbo"
    assert kwargs["device"] == "auto"


def test_subprocess_backend_status_served_from_cache(patched_subprocess):
    """``status()`` no longer round-trips through the pipe — it reads
    a local cache populated by ``status_change`` push messages from
    the worker.  RecordingController polls status() every 200 ms; an
    IPC round-trip per poll contended with the worker's tqdm + log
    forwarding stream and stuttered the UI during a model load.

    Verifies:
    1. Initial status is ``"stopped"`` even before the worker sends
       anything (sane default — UI shows the right pill on boot).
    2. After a ``("status_change", "loading")`` arrives on the pipe,
       the next ``status()`` returns ``"loading"`` without any new
       command being sent.
    """
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]
    sent_before = list(patched_subprocess["sent"])

    # Default cache before any push.
    assert backend.status() == "stopped"

    # Worker pushes a status change.
    conn.push(("status_change", "loading"))

    # Reader thread is async — give it a moment.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if backend.status() == "loading":
            break
        time.sleep(0.01)

    assert backend.status() == "loading"
    # No new pipe send for the status() call itself — only the
    # init that fired in __init__.
    assert patched_subprocess["sent"] == sent_before

    backend.shutdown()


def test_subprocess_backend_health_check_served_from_cache(patched_subprocess):
    """``health_check()`` is just ``status() == "ready"`` and shares
    the same cache fast-path.  No IPC."""
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]

    # No "ready" yet — health_check should be False.
    assert backend.health_check() is False

    conn.push(("status_change", "ready"))

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if backend.health_check():
            break
        time.sleep(0.01)

    assert backend.health_check() is True
    backend.shutdown()


def test_subprocess_backend_progress_routes_to_callback(patched_subprocess):
    """When the worker pushes a ``("progress", ...)`` message onto the
    pipe, the registered callback must fire on the parent side."""
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]

    received: list = []
    backend.set_progress_callback(
        lambda c, t, d: received.append((c, t, d))
    )

    conn.push(("progress", 100, 1000, "weights.onnx"))

    # Reader thread is async; give it a moment.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not received:
        time.sleep(0.01)

    assert received == [(100, 1000, "weights.onnx")]
    backend.shutdown()


def test_subprocess_backend_error_response_raises(patched_subprocess):
    """An ``("error", "msg")`` from worker must surface as a
    RuntimeError on the calling thread — otherwise the parent is
    silently broken.

    Uses ``current_model()`` rather than ``status()`` because
    ``status()`` is now served from the local cache (no IPC) and
    can't surface a worker-side error.
    """
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]

    conn.push(("error", "RuntimeError: boom"))
    with pytest.raises(RuntimeError, match="boom"):
        backend.current_model()


def test_subprocess_backend_change_model_sync(patched_subprocess):
    """change_model forwards args correctly through the pipe."""
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]
    conn.push(("ok", None))

    backend.change_model("new/model", compute_type="int8")

    assert ("change_model", "new/model", "int8") in patched_subprocess["sent"]


def test_subprocess_backend_shutdown_terminates_process(patched_subprocess):
    """shutdown sends ``shutdown_worker`` and joins the process.
    Idempotent — calling twice doesn't blow up."""
    from app.backends.subprocess_backend import SubprocessBackend

    backend = SubprocessBackend(model="x")
    conn = patched_subprocess["conn_holder"]["conn"]
    conn.push(("ok", None))

    backend.shutdown()
    # Second call must not raise (process already dead).
    backend.shutdown()
