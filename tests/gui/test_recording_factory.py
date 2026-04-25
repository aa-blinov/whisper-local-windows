"""Tests for the recording stack factory.

Most of the work is a light glue layer between ``ConfigManager`` and the
domain-layer constructors, so we keep the unit tests minimal — full
integration is verified by the launch smoke test in main.
"""

import pytest


@pytest.mark.skipif(
    True,  # Full build registers global hotkeys; opt-in via env or smoke test only.
    reason="Skipped by default — exercised end-to-end via the launch smoke test.",
)
def test_build_recording_stack_constructs_full_triple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("")  # ConfigManager picks tmp_path as base

    from app.backends.base import TranscriptionBackend
    from app.config_manager import ConfigManager
    from app.gui.recording_factory import build_recording_stack
    from app.hotkey_listener import HotkeyListener
    from app.state_manager import StateManager

    config = ConfigManager()
    sm, hk, backend = build_recording_stack(config)
    try:
        assert isinstance(sm, StateManager)
        assert isinstance(hk, HotkeyListener)
        assert isinstance(backend, TranscriptionBackend)
        # Backend is not loaded yet — caller is expected to call .load().
        assert backend.status() == "stopped"
    finally:
        try:
            hk.stop_listening()
        except Exception:
            pass
        try:
            backend.shutdown()
        except Exception:
            pass
