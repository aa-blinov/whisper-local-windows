"""Tests for StateManager.request_model_change non-blocking semantics.

The Qt main thread calls into ``request_model_change`` from
``_on_model_selected`` (Select-button click handler).  If the call
chain blocks for more than a few hundred milliseconds — e.g. on
``OnnxAsrBackend.change_model`` synchronously dropping the old
``self._model`` reference, which fires the ONNX session destructor
and CUDA-memory release on the caller — the window cannot be moved
or repainted and Windows marks it "(Not responding)".
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock


def _build_state_manager() -> "object":
    """Build a half-real StateManager with mock collaborators.

    Idle state — ``request_model_change`` hits the
    ``_execute_model_change`` path (not the queued/cancel paths).
    """
    from app.state_manager import StateManager

    sm = StateManager.__new__(StateManager)

    sm.audio_recorder = MagicMock()
    sm.audio_recorder.get_recording_status.return_value = False
    sm.backend = MagicMock()
    sm.backend.status.return_value = "ready"
    sm.backend.health_check.return_value = True
    sm.backend.current_model.return_value = "old/model"
    sm.clipboard_manager = MagicMock()
    sm.config_manager = MagicMock()
    sm.history_manager = None
    sm.history_update_callback = None
    sm.is_processing = False
    sm.is_model_loading = False
    sm.last_transcription = None
    sm._pending_model_change = None
    sm._state_lock = threading.Lock()
    sm.logger = logging.getLogger("test.state_manager_async_change")
    sm.audio_feedback = MagicMock()
    sm.system_tray = MagicMock()
    return sm


def test_request_model_change_does_not_block_on_slow_backend_change_model():
    """``request_model_change`` must return to the caller (the Qt
    main thread, in production) before ``backend.change_model``
    finishes.  Otherwise the synchronous ONNX-session destructor +
    CUDA-memory release can stall the main thread for hundreds of
    milliseconds and the window freezes / gets (Not responding)."""
    sm = _build_state_manager()

    # Simulate a slow backend.change_model — what happens in real
    # life when ``self._model = None`` triggers an ONNX session
    # destructor that takes 300+ ms.
    release = threading.Event()

    def slow_change_model(*_a, **_kw):
        release.wait(5)

    sm.backend.change_model = slow_change_model

    t0 = time.monotonic()
    result = sm.request_model_change("new/model")
    elapsed = time.monotonic() - t0

    try:
        assert result is True
        assert elapsed < 0.5, (
            f"request_model_change blocked for {elapsed:.2f}s — must "
            f"return immediately so the Qt main thread doesn't stall "
            f"on ONNX session teardown"
        )
        # Sanity: the worker did kick off (we're holding it on
        # ``release``, so it's blocked but alive).
        assert sm.is_model_loading is True
    finally:
        release.set()


def test_request_model_change_eventually_calls_backend_change_model():
    """The point of deferring change_model is just to move it off
    the main thread — it must still happen.  Otherwise the model
    never actually switches."""
    sm = _build_state_manager()
    # ``status() == "ready"`` keeps both the request_model_change
    # gate (which checks current_state via backend.status()) and the
    # watcher (polling for ready/error/stopped) happy.
    sm.backend.status.return_value = "ready"

    sm.request_model_change("new/model")

    # Give the worker a moment to call change_model (it runs on a
    # daemon thread).  In production this is bounded by
    # ``OnnxAsrBackend.change_model`` returning, which is fast once
    # the destructor moves to the worker.  3 s is a generous CI cap.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if sm.backend.change_model.called:
            break
        time.sleep(0.01)

    sm.backend.change_model.assert_called_once_with(
        "new/model", compute_type=None,
    )


def test_request_model_change_clears_loading_flag_on_backend_failure():
    """If ``backend.change_model`` raises, the worker must clear
    ``is_model_loading`` so the UI doesn't get stuck on the
    Loading pill forever."""
    sm = _build_state_manager()
    sm.backend.change_model.side_effect = RuntimeError("boom")

    sm.request_model_change("new/model")

    # Wait for the worker to run + handle the error.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if sm.is_model_loading is False:
            break
        time.sleep(0.01)

    assert sm.is_model_loading is False, (
        "is_model_loading must be cleared when backend.change_model raises"
    )
