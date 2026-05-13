"""Tests for the winsound-based AudioFeedback prewarm.

The first ``winsound.PlaySound`` call on a freshly-started Windows
process spends 100-300 ms opening the multimedia device; an SND_ASYNC
sound issued during that window is silently dropped. AudioFeedback
should fire a no-op silent play at construction so the user's first
real start sound is heard.

These tests cover the Windows-only branch of ``AudioFeedback`` —
``winsound`` is part of the Python stdlib but only available on
Windows. The macOS / Linux branch (``playsound3``) does not have the
silent-drop pathology and therefore the prewarm is a no-op there;
nothing to test on those platforms.
"""

import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="winsound prewarm only exists on Windows",
)


def test_silent_wav_is_a_valid_riff_buffer():
    from app.audio_feedback import _SILENT_WAV

    assert _SILENT_WAV[:4] == b"RIFF"
    assert _SILENT_WAV[8:12] == b"WAVE"
    # Must contain a fmt chunk and a data chunk.
    assert b"fmt " in _SILENT_WAV
    assert b"data" in _SILENT_WAV


def test_prewarm_calls_winsound_play_with_memory_flag(monkeypatch):
    import winsound

    calls: list[tuple] = []

    def fake_play(sound, flags):
        calls.append((sound, flags))

    monkeypatch.setattr(winsound, "PlaySound", fake_play)

    from app.audio_feedback import AudioFeedback

    AudioFeedback(
        enabled=True,
        start_sound="",
        stop_sound="",
        cancel_sound="",
    )

    # Prewarm runs on a daemon thread — give it a moment.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not calls:
        time.sleep(0.01)

    assert calls, "prewarm did not call winsound.PlaySound"
    sound, flags = calls[0]
    assert flags & winsound.SND_MEMORY, "prewarm should use SND_MEMORY"
    assert flags & winsound.SND_ASYNC, "prewarm should be non-blocking"


def test_prewarm_skipped_when_audio_feedback_disabled(monkeypatch):
    import winsound

    calls: list[tuple] = []

    def fake_play(sound, flags):
        calls.append((sound, flags))

    monkeypatch.setattr(winsound, "PlaySound", fake_play)

    from app.audio_feedback import AudioFeedback

    AudioFeedback(
        enabled=False,
        start_sound="",
        stop_sound="",
        cancel_sound="",
    )
    time.sleep(0.1)  # prewarm thread would have run by now if it was started

    assert calls == []


def test_prewarm_does_not_propagate_winsound_errors(monkeypatch):
    """A flaky audio device must not crash the app at startup."""
    import winsound

    def boom(sound, flags):
        raise OSError("audio device unavailable")

    monkeypatch.setattr(winsound, "PlaySound", boom)

    from app.audio_feedback import AudioFeedback

    # Constructor must not raise — prewarm runs in a daemon thread that
    # swallows the exception.
    AudioFeedback(
        enabled=True,
        start_sound="",
        stop_sound="",
        cancel_sound="",
    )
    # Give the thread a moment to attempt and swallow the error.
    time.sleep(0.1)


def test_prewarm_is_callable_explicitly(monkeypatch):
    """prewarm() is public so state_manager can call it when the model
    finishes loading to re-warm the audio device before the first click."""
    import winsound

    calls: list[tuple] = []

    def fake_play(sound, flags):
        calls.append((sound, flags))

    monkeypatch.setattr(winsound, "PlaySound", fake_play)

    from app.audio_feedback import AudioFeedback

    af = AudioFeedback(enabled=True, start_sound="", stop_sound="", cancel_sound="")
    # Wait for constructor prewarm, then clear.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not calls:
        time.sleep(0.01)
    calls.clear()

    # Explicit call — must fire another SND_MEMORY | SND_ASYNC play.
    af.prewarm()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not calls:
        time.sleep(0.01)

    assert calls, "explicit prewarm() did not call winsound.PlaySound"
    sound, flags = calls[0]
    assert flags & winsound.SND_MEMORY
    assert flags & winsound.SND_ASYNC


def test_prewarm_noop_when_disabled(monkeypatch):
    """Calling prewarm() on a disabled AudioFeedback must not play anything —
    the user deliberately opted out of audio."""
    import winsound

    calls: list[tuple] = []
    monkeypatch.setattr(winsound, "PlaySound", lambda s, f: calls.append(f))

    from app.audio_feedback import AudioFeedback

    af = AudioFeedback(enabled=False, start_sound="", stop_sound="", cancel_sound="")
    af.prewarm()
    time.sleep(0.1)

    assert calls == [], "prewarm() on disabled instance must not play"
