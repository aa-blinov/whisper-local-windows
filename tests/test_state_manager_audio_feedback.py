"""Targeted tests for the start-sound-on-hotkey UX guarantee.

The full StateManager has many collaborators; these tests use light Mock
stand-ins to verify just the hotkey acoustic-feedback contract:
- pressing the toggle hotkey while idle but unable-to-start still plays the
  start sound, so the user always hears that their keypress was received.
"""

from unittest.mock import MagicMock


def _build_state_manager(can_start: bool):
    from app.state_manager import StateManager

    sm = StateManager.__new__(StateManager)

    sm.audio_recorder = MagicMock()
    sm.audio_recorder.get_recording_status.return_value = False
    sm.whisper_engine = MagicMock()
    sm.clipboard_manager = MagicMock()
    sm.config_manager = MagicMock()
    sm.docker_backend_manager = None
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

    # Stub the pipeline so we don't drive Wyoming/transcription paths.
    sm._transcription_pipeline = MagicMock()

    sm.toggle_recording()

    sm.audio_feedback.play_start_sound.assert_not_called()
