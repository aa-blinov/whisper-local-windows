"""Tests for ConfigManager — base-dir resolution + first-launch behaviour.

The interesting code paths are:
  - frozen build → ``%APPDATA%/LazyToText/config.yaml``
  - dev build → project root ``config.yaml``
  - first launch frozen with bundled defaults available → seed-copy
  - first launch with no user / bundled config → write ``DEFAULT_CONFIG``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_appdata(tmp_path, monkeypatch):
    """Each test gets a clean ``APPDATA`` pointing at a fresh tmp dir
    so user-config writes don't bleed across tests."""
    appdata = tmp_path / "appdata"
    appdata.mkdir()
    monkeypatch.setenv("APPDATA", str(appdata))
    return appdata


# ---- _resolve_base_dir ------------------------------------------------------


def test_dev_mode_uses_project_root(monkeypatch, tmp_path):
    """In dev (non-frozen) the resolver walks up from CWD until it
    finds a ``pyproject.toml``. Default behaviour preserved."""
    from app.config_manager import ConfigManager

    # Make sure we're not seen as frozen.
    monkeypatch.delattr(sys, "frozen", raising=False)
    # Place a fake project root with pyproject.toml under tmp.
    project_root = tmp_path / "fake_project"
    project_root.mkdir()
    (project_root / "pyproject.toml").touch()
    monkeypatch.chdir(project_root)

    cm = ConfigManager()
    assert cm.base_dir == project_root
    assert cm.config_path == project_root / "config.yaml"


def test_frozen_mode_uses_appdata(fresh_appdata, monkeypatch):
    """Frozen build → ``%APPDATA%/LazyToText/`` so writes don't
    require admin in Program Files."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Program Files/LazyToText/LazyToText.exe", raising=False)

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    assert cm.base_dir == fresh_appdata / "LazyToText"
    assert cm.config_path == fresh_appdata / "LazyToText" / "config.yaml"


def test_frozen_mode_creates_user_dir_on_first_write(fresh_appdata, monkeypatch):
    """The user-config dir typically doesn't exist on first launch
    of an installed build. ConfigManager must create it before
    writing — otherwise the YAML write raises ``FileNotFoundError``."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Program Files/LazyToText/LazyToText.exe", raising=False)

    target_dir = fresh_appdata / "LazyToText"
    assert not target_dir.exists()

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    assert target_dir.exists()
    assert cm.config_path.exists()


def test_frozen_mode_falls_back_to_home_when_appdata_missing(
    monkeypatch, tmp_path,
):
    """Some sandboxed environments don't expose ``%APPDATA%``. Fall
    back to ``~/AppData/Roaming/LazyToText`` so the app still works."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/foo/LazyToText.exe", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fake_home")

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    expected = tmp_path / "fake_home" / "AppData" / "Roaming" / "LazyToText"
    assert cm.base_dir == expected


# ---- First-launch seeding --------------------------------------------------


def test_first_launch_writes_defaults_when_no_user_config(fresh_appdata, monkeypatch):
    """No existing user config + no bundled defaults file → write
    the in-code ``DEFAULT_CONFIG`` to the user dir. Same behaviour
    as dev mode on first launch."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/foo/LazyToText.exe", raising=False)

    from app.config_manager import ConfigManager, DEFAULT_CONFIG

    cm = ConfigManager()
    assert cm.config_path.exists()
    # Verify a known default field round-tripped.
    assert cm.get_setting("hotkey", "start_recording_hotkey") == \
        DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]


def test_first_launch_seeds_from_bundled_defaults_next_to_exe(
    fresh_appdata, monkeypatch, tmp_path,
):
    """Frozen build can ship a tweaked ``config.yaml`` next to the
    executable as a 'factory defaults'. On first launch — when the
    user dir is empty — copy that seed into the user dir so the
    user inherits any non-default values the build was shipped with."""
    exe_dir = tmp_path / "install"
    exe_dir.mkdir()
    bundled = exe_dir / "config.yaml"
    bundled.write_text(
        "hotkey:\n"
        "  start_recording_hotkey: ctrl+shift+r\n"
        "  stop_recording_hotkey: ctrl+shift+s\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "LazyToText.exe"), raising=False)

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    assert cm.get_setting("hotkey", "start_recording_hotkey") == "ctrl+shift+r"
    assert cm.get_setting("hotkey", "stop_recording_hotkey") == "ctrl+shift+s"
    # User config now exists at the appdata path.
    assert cm.config_path.exists()
    assert cm.config_path.parent == fresh_appdata / "LazyToText"


def test_subsequent_launch_reads_user_config_not_bundled(
    fresh_appdata, monkeypatch, tmp_path,
):
    """User config takes precedence over bundled defaults — once
    the user has saved their own settings, the build's seed
    shouldn't overwrite them."""
    user_dir = fresh_appdata / "LazyToText"
    user_dir.mkdir()
    user_config = user_dir / "config.yaml"
    user_config.write_text(
        "hotkey:\n  start_recording_hotkey: f1\n",
        encoding="utf-8",
    )

    exe_dir = tmp_path / "install"
    exe_dir.mkdir()
    bundled = exe_dir / "config.yaml"
    bundled.write_text(
        "hotkey:\n  start_recording_hotkey: ctrl+shift+r\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "LazyToText.exe"), raising=False)

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    # User-set value wins, bundled "factory" value doesn't overwrite.
    assert cm.get_setting("hotkey", "start_recording_hotkey") == "f1"


def test_subsequent_writes_go_to_user_config(fresh_appdata, monkeypatch):
    """Once a user config exists, ``update_user_setting`` writes
    to the user dir, not the (possibly read-only) install dir."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Program Files/LazyToText/LazyToText.exe", raising=False)

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("hotkey", "start_recording_hotkey", "ctrl+f5")
    # ``update_user_setting`` is async — the daemon writer hasn't
    # necessarily landed the YAML by the time we return.  Block here
    # so the file-on-disk read below sees the new value.
    cm.flush_pending_writes()

    # Read the file back to verify it landed at user config path.
    text = cm.config_path.read_text(encoding="utf-8")
    assert "ctrl+f5" in text
    assert cm.config_path.parent == fresh_appdata / "LazyToText"


# ---- atomic write ----------------------------------------------------------


def test_write_leaves_no_tmp_file(fresh_appdata, monkeypatch):
    """Atomic write must clean up the ``.tmp`` staging file regardless
    of outcome — a leftover ``.tmp`` means the swap never completed
    and the previous file should still be intact."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/foo/LazyToText.exe", raising=False)

    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("audio", "channels", 2)
    cm.flush_pending_writes()

    tmp = cm.config_path.with_suffix(".tmp")
    assert not tmp.exists(), ".tmp staging file must be removed after a successful write"


def test_config_round_trips_through_yaml(fresh_appdata, monkeypatch):
    """Values written by ConfigManager must survive a YAML round-trip —
    read back from disk and compare to what was stored in-memory."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/foo/LazyToText.exe", raising=False)

    import yaml
    from app.config_manager import ConfigManager

    cm = ConfigManager()
    cm.update_user_setting("clipboard", "key_simulation_delay", 0.07)
    cm.update_user_setting("whisper", "language", "ru")
    cm.flush_pending_writes()

    on_disk = yaml.safe_load(cm.config_path.read_text(encoding="utf-8"))
    assert on_disk["clipboard"]["key_simulation_delay"] == pytest.approx(0.07)
    assert on_disk["whisper"]["language"] == "ru"
