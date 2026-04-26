"""Tests for StateManager.cancel_model_change.

Cancel-load is an escape hatch when the user mis-clicks a heavy model
card (or NeMo's cold-import deadlocks for half an hour). The state
manager turns the user-facing intent ("Cancel") into the right
sequence of side effects: tell the backend to drop its in-flight
load, clear the ``is_model_loading`` flag, and let the watcher thread
exit cleanly via the ``stopped`` status it now sees.
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock


def _build_state_manager(*, status: str = "loading") -> "object":
    """Build a half-real StateManager with mock collaborators.

    Mirrors the helper in ``test_state_manager_audio_feedback`` so the
    two suites stay in sync. ``status`` controls what
    ``backend.status()`` reports — most cancel tests want ``loading``.
    """
    from app.state_manager import StateManager

    sm = StateManager.__new__(StateManager)

    sm.audio_recorder = MagicMock()
    sm.audio_recorder.get_recording_status.return_value = False
    sm.backend = MagicMock()
    sm.backend.status.return_value = status
    sm.backend.health_check.return_value = status == "ready"
    sm.clipboard_manager = MagicMock()
    sm.config_manager = MagicMock()
    sm.history_manager = None
    sm.history_update_callback = None
    sm.is_processing = False
    sm.is_model_loading = status == "loading"
    sm.last_transcription = None
    sm._pending_model_change = None
    sm._state_lock = threading.Lock()
    sm.logger = logging.getLogger("test.state_manager_cancel")
    sm.audio_feedback = MagicMock()
    sm.system_tray = MagicMock()
    return sm


def test_cancel_model_change_calls_backend_cancel():
    """Cancel must reach the backend so its load thread learns it
    should discard the in-flight result."""
    sm = _build_state_manager(status="loading")

    cancelled = sm.cancel_model_change()

    assert cancelled is True
    sm.backend.cancel_load.assert_called_once()


def test_cancel_model_change_clears_loading_flag():
    """After cancel, ``is_model_loading`` must drop to False so the
    UI's recording-state poll flips the topbar back to idle and
    re-enables recording."""
    sm = _build_state_manager(status="loading")

    sm.cancel_model_change()

    assert sm.is_model_loading is False


def test_cancel_model_change_returns_false_when_not_loading():
    """If nothing is loading, cancel is a no-op — the topbar's
    cancel button can race with a load that already finished or
    failed and we don't want a stale click to mark a ready model
    as cancelled."""
    sm = _build_state_manager(status="ready")
    sm.is_model_loading = False

    cancelled = sm.cancel_model_change()

    assert cancelled is False
    sm.backend.cancel_load.assert_not_called()


def test_cancel_model_change_survives_backend_without_cancel_method():
    """Some test fakes (and any future inner backend that hasn't been
    updated yet) may not expose ``cancel_load``. The state manager
    must still clear the loading flag and not raise — the cancel
    button click can never crash."""
    sm = _build_state_manager(status="loading")
    # Backend has no cancel_load attribute at all.
    sm.backend = MagicMock(spec=["status", "health_check"])
    sm.backend.status.return_value = "loading"

    cancelled = sm.cancel_model_change()

    assert cancelled is True
    assert sm.is_model_loading is False


def test_watch_model_change_exits_when_status_goes_to_stopped():
    """The watcher polls ``backend.status()`` waiting for ``ready``
    or ``error``. Cancel makes the backend report ``stopped`` — the
    watcher must treat that as a clean exit (not a timeout) and
    drop the loading flag without logging a scary warning."""
    sm = _build_state_manager(status="loading")
    # First poll sees "loading", second sees "stopped" (cancel
    # happened in between).
    statuses = ["loading", "stopped"]
    sm.backend.status.side_effect = lambda: (
        statuses.pop(0) if statuses else "stopped"
    )

    # Run watcher synchronously with a short timeout.
    sm._watch_model_change("test-model", timeout=2.0)

    assert sm.is_model_loading is False
