"""Tests for ConfigManager — base-dir resolution + first-launch behaviour.

ConfigManager always reads / writes ``config.yaml`` at the project
root (resolved by walking up from CWD to the nearest
``pyproject.toml``). The previous PyInstaller-bundle paths
(``%APPDATA%/LazyToText/`` + bundled-defaults seed-copy) were
removed when the source-only distribution was adopted; this test
file now covers only the dev-mode resolver and the atomic-write
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_project(monkeypatch, tmp_path):
    """Each test gets a clean fake project root (``pyproject.toml``
    + empty CWD chdir) so the ``_resolve_base_dir`` walk hits a
    deterministic location instead of climbing up to the actual
    repo root and clobbering the developer's ``config.yaml``."""
    project_root = tmp_path / "fake_project"
    project_root.mkdir()
    (project_root / "pyproject.toml").touch()
    monkeypatch.chdir(project_root)
    return project_root


# ---- _resolve_base_dir ------------------------------------------------------


def test_dev_mode_uses_project_root(fake_project):
    """The resolver walks up from CWD until it finds a
    ``pyproject.toml``. Default behaviour preserved."""
    from app.config_manager import ConfigManager

    cm = ConfigManager()
    assert cm.base_dir == fake_project
    assert cm.config_path == fake_project / "config.yaml"


# ---- First-launch defaults --------------------------------------------------


def test_first_launch_writes_defaults_when_no_user_config(fake_project):
    """No existing config.yaml → write the in-code ``DEFAULT_CONFIG``
    to the project root."""
    from app.config_manager import ConfigManager, DEFAULT_CONFIG

    cm = ConfigManager()
    assert cm.config_path.exists()
    # Verify a known default field round-tripped.
    assert cm.get_setting("hotkey", "start_recording_hotkey") == \
        DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]


def test_subsequent_writes_round_trip_through_disk(fake_project):
    """``update_user_setting`` must persist the new value to the
    ``config.yaml`` on disk so the next launch picks it up. Async
    writer is synchronously drained via ``flush_pending_writes`` so
    the read-back below sees the new value."""
    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("hotkey", "start_recording_hotkey", "ctrl+f5")
    cm.flush_pending_writes()

    text = cm.config_path.read_text(encoding="utf-8")
    assert "ctrl+f5" in text
    assert cm.config_path.parent == fake_project


# ---- atomic write ----------------------------------------------------------


def test_write_leaves_no_tmp_file(fake_project):
    """Atomic write must clean up the ``.tmp`` staging file regardless
    of outcome — a leftover ``.tmp`` means the swap never completed
    and the previous file should still be intact."""
    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("audio", "channels", 2)
    cm.flush_pending_writes()

    tmp = cm.config_path.with_suffix(".tmp")
    assert not tmp.exists(), ".tmp staging file must be removed after a successful write"


def test_config_round_trips_through_yaml(fake_project):
    """Values written by ConfigManager must survive a YAML round-trip —
    read back from disk and compare to what was stored in-memory."""
    import yaml
    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("clipboard", "key_simulation_delay", 0.07)
    cm.update_user_setting("whisper", "language", "ru")
    cm.flush_pending_writes()

    on_disk = yaml.safe_load(cm.config_path.read_text(encoding="utf-8"))
    assert on_disk["clipboard"]["key_simulation_delay"] == pytest.approx(0.07)
    assert on_disk["whisper"]["language"] == "ru"
