"""Targeted tests for the start-sound-on-hotkey UX guarantee and the
transcription-pipeline threading contract.

The full StateManager has many collaborators; these tests use light Mock
stand-ins to verify:
- pressing the toggle hotkey while idle but unable-to-start still plays the
  start sound, so the user always hears that their keypress was received.
- the transcription pipeline runs on a dedicated worker thread so the hotkey
  listener is freed immediately after the user stops recording.
"""

import importlib.util
import sys
import threading
from unittest.mock import MagicMock

# Several native / heavy extensions may not be installed in all test
# environments.  Pre-inject stubs ONLY when the module is genuinely absent
# so we don't shadow a real installed package on dev machines.
#
# ``importlib.util.find_spec`` is used instead of the bare
# ``if _mod not in sys.modules`` guard to avoid replacing a real
# (but not-yet-imported) package with a Mock.  For dotted names like
# ``ruamel.yaml`` the call can raise ``ModuleNotFoundError`` when the
# parent has already been stubbed (it lacks a real ``__path__``), so we
# wrap it in a try/except and treat that as "not available".
def _is_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, AttributeError, ValueError):
        return False


for _mod in (
    "sounddevice",
    "pyperclip",
    "pynput",
    "pynput.keyboard",
):
    if not _is_available(_mod):
        sys.modules[_mod] = MagicMock()


def _build_state_manager(can_start: bool):
    from app.state_manager import StateManager

    sm = StateManager.__new__(StateManager)

    sm.audio_recorder = MagicMock()
    sm.audio_recorder.get_recording_status.return_value = False
    sm.backend = MagicMock()
    sm.backend.health_check.return_value = can_start
    sm.clipboard_manager = MagicMock()
    sm.config_manager = MagicMock()
    sm.history_manager = None
    sm.history_update_callback = None
    sm.is_processing = False
    sm.is_model_loading = False
    sm.last_transcription = None
    sm._pending_model_change = None

    import threading
    sm._state_lock = threading.Lock()

    import logging
    sm.logger = logging.getLogger("test.state_manager_audio_feedback")

    sm.audio_feedback = MagicMock()
    sm.system_tray = MagicMock()

    sm.can_start_recording = MagicMock(return_value=can_start)
    return sm


def test_toggle_plays_start_sound_when_unable_to_start():
    sm = _build_state_manager(can_start=False)

    sm.toggle_recording()

    sm.audio_feedback.play_start_sound.assert_called_once()
    # Must NOT have asked the recorder to start.
    sm.audio_recorder.start_recording.assert_not_called()


def test_toggle_starts_recording_when_able():
    sm = _build_state_manager(can_start=True)
    sm.audio_recorder.start_recording.return_value = True

    sm.toggle_recording()

    sm.audio_recorder.start_recording.assert_called_once()
    sm.audio_feedback.play_start_sound.assert_called_once()


def test_start_recording_plays_sound_before_opening_audio_recorder():
    """Reorder: sound first, recorder second.

    Opening the mic stream takes ~100 ms on Windows; if we play the sound
    afterwards, that latency stacks on top of the winsound first-call delay.
    Playing first means the start sound is queued before the mic open
    pressures the audio session.
    """
    sm = _build_state_manager(can_start=True)
    call_order: list[str] = []

    sm.audio_feedback.play_start_sound.side_effect = (
        lambda: call_order.append("sound")
    )
    sm.audio_recorder.start_recording.side_effect = lambda: (
        call_order.append("record") or True
    )

    sm.toggle_recording()

    assert call_order == ["sound", "record"], (
        f"expected sound to play before recorder open, got {call_order!r}"
    )


def test_start_recording_still_plays_sound_when_recorder_fails():
    """If start_recording() returns False, the user still got acoustic
    feedback that the hotkey was caught (because we played first)."""
    sm = _build_state_manager(can_start=True)
    sm.audio_recorder.start_recording.return_value = False

    sm.toggle_recording()

    sm.audio_feedback.play_start_sound.assert_called_once()


def test_toggle_does_not_play_start_sound_when_already_recording():
    """When was_recording=True, toggle stops recording — start sound is irrelevant
    here (stop sound fires from inside the transcription pipeline)."""
    sm = _build_state_manager(can_start=True)
    sm.audio_recorder.get_recording_status.return_value = True
    sm.audio_recorder.stop_recording.return_value = None  # short-circuits pipeline

    # Stub the pipeline so we don't drive the real transcription path.
    sm._transcription_pipeline = MagicMock()

    sm.toggle_recording()

    sm.audio_feedback.play_start_sound.assert_not_called()


# ---- prewarm-on-model-load tests -------------------------------------------


def test_set_model_loading_prewarns_audio_on_completion():
    """When model loading ends (False after True), prewarm() must be called.

    Windows releases idle audio devices after a few seconds. By the time a
    model finishes loading the device opened at startup is almost certainly
    closed. Without a re-warm the first recording-start click after model
    load is silently dropped.
    """
    sm = _build_state_manager(can_start=False)
    sm.is_model_loading = True   # pretend we were in model_loading

    sm.set_model_loading(False)

    sm.audio_feedback.prewarm.assert_called_once()


def test_set_model_loading_does_not_prewarm_when_loading_starts():
    """Entering model_loading must NOT trigger prewarm — no point warming the
    device before the backend starts a potentially multi-minute download."""
    sm = _build_state_manager(can_start=False)
    sm.is_model_loading = False

    sm.set_model_loading(True)

    sm.audio_feedback.prewarm.assert_not_called()


def test_set_model_loading_does_not_prewarm_if_state_unchanged():
    """Calling set_model_loading(False) when already False must be a no-op
    (no state change, no prewarm)."""
    sm = _build_state_manager(can_start=False)
    sm.is_model_loading = False   # already not loading

    sm.set_model_loading(False)

    sm.audio_feedback.prewarm.assert_not_called()


# ---- pipeline-threading tests -----------------------------------------------


def test_stop_recording_pipeline_runs_in_separate_thread():
    """The transcription pipeline must run on a worker thread so the
    hotkey-listener thread is freed immediately — the user can press
    the start hotkey again while Whisper is still transcribing."""
    sm = _build_state_manager(can_start=True)
    sm.audio_recorder.get_recording_status.return_value = True
    sm.audio_recorder.stop_recording.return_value = MagicMock()  # dummy audio

    caller_thread_id = threading.get_ident()
    pipeline_ran = threading.Event()
    pipeline_thread_ids: list[int] = []

    def capture_thread(audio_data, use_auto_enter=False):
        pipeline_thread_ids.append(threading.get_ident())
        pipeline_ran.set()

    sm._transcription_pipeline = capture_thread

    sm.stop_recording()

    assert pipeline_ran.wait(timeout=2), "pipeline did not run within 2 s"
    assert pipeline_thread_ids[0] != caller_thread_id, (
        "pipeline must run on a different thread than the caller"
    )


def test_stop_recording_sets_is_processing_before_pipeline_starts():
    """is_processing must be True before the worker thread touches it so the
    hotkey listener cannot start a second recording in the gap between
    stop_recording() returning and the thread setting is_processing itself."""
    sm = _build_state_manager(can_start=True)
    sm.audio_recorder.get_recording_status.return_value = True
    sm.audio_recorder.stop_recording.return_value = MagicMock()

    # Hold the pipeline thread so it cannot clear is_processing before we check.
    hold = threading.Event()
    sm._transcription_pipeline = MagicMock(side_effect=lambda *a, **k: hold.wait(5))

    sm.stop_recording()

    # is_processing is True synchronously — set before the thread starts.
    assert sm.is_processing is True
    hold.set()  # release the background thread so it can exit cleanly


def test_handle_max_duration_pipeline_runs_in_separate_thread():
    """handle_max_recording_duration_reached must also offload the pipeline
    to a worker thread — it's called from the audio-recorder callback, and
    blocking that thread would starve future audio frames."""
    sm = _build_state_manager(can_start=True)

    caller_thread_id = threading.get_ident()
    pipeline_ran = threading.Event()
    pipeline_thread_ids: list[int] = []

    def capture_thread(audio_data, use_auto_enter=False):
        pipeline_thread_ids.append(threading.get_ident())
        pipeline_ran.set()

    sm._transcription_pipeline = capture_thread

    sm.handle_max_recording_duration_reached(MagicMock())

    assert pipeline_ran.wait(timeout=2), "pipeline did not run within 2 s"
    assert pipeline_thread_ids[0] != caller_thread_id, (
        "pipeline must run on a different thread than the caller"
    )
