"""Light tests for the recording stack factory.

Full integration is verified in the smoke-launch — these tests cover the
pure-Python helpers and a guarded build that may bail on platforms where
HotkeyListener cannot register (e.g. CI without Windows).
"""

import pytest


def test_wyoming_url_strips_http_prefix():
    from app.gui.recording_factory import _wyoming_url

    assert _wyoming_url("http://localhost:10300") == "localhost:10300"
    assert _wyoming_url("https://host:10300") == "host:10300"
    assert _wyoming_url("localhost:10300") == "localhost:10300"
    assert _wyoming_url("") == "localhost:10300"


def test_resolve_model_pair_maps_alias_to_canonical():
    from app.gui.recording_factory import _resolve_model_pair

    alias, canonical = _resolve_model_pair("large-v3")
    assert alias == "large-v3"
    assert canonical.endswith("/faster-whisper-large-v3")


def test_resolve_model_pair_handles_canonical_input():
    from app.gui.recording_factory import _resolve_model_pair

    alias, canonical = _resolve_model_pair("Systran/faster-whisper-large-v3")
    assert alias == "large-v3"
    assert canonical == "Systran/faster-whisper-large-v3"


def test_resolve_model_pair_falls_back_to_default_when_empty():
    from app.gui.recording_factory import _resolve_model_pair

    alias, canonical = _resolve_model_pair(None)
    assert alias == "large-v3"
    assert canonical


@pytest.mark.skipif(
    True,  # Full build registers global hotkeys; opt-in via env or smoke test only.
    reason="Skipped by default — exercised end-to-end via the launch smoke test.",
)
def test_build_recording_stack_constructs_state_manager_and_listener(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("")  # ConfigManager picks tmp_path as base
    from app.config_manager import ConfigManager
    from app.gui.recording_factory import build_recording_stack
    from app.hotkey_listener import HotkeyListener
    from app.state_manager import StateManager

    config = ConfigManager()
    sm, hk = build_recording_stack(config)
    try:
        assert isinstance(sm, StateManager)
        assert isinstance(hk, HotkeyListener)
    finally:
        try:
            hk.stop_listening()
        except Exception:
            pass
