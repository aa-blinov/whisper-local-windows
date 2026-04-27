"""Tests for the AppController wiring Models view to config storage."""

from typing import Any, Dict, List, Optional, Tuple

import pytest


class FakeConfig:
    """Minimal stand-in for ConfigManager used in controller tests."""

    def __init__(self, initial: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._data: Dict[str, Dict[str, Any]] = {
            section: dict(values) for section, values in (initial or {}).items()
        }
        self.writes: List[Tuple[str, str, Any]] = []

    def get_setting(self, section: str, key: str) -> Any:
        return self._data.get(section, {}).get(key)

    def update_user_setting(self, section: str, key: str, value: Any) -> None:
        self._data.setdefault(section, {})[key] = value
        self.writes.append((section, key, value))


@pytest.fixture(autouse=True)
def _assume_cached(monkeypatch):
    """The controller now consults ``is_cached_for_info`` before
    restoring the persisted active card so a fresh install / cleared
    cache doesn't silently kick off a multi-gigabyte download. The
    bulk of the existing tests assume the persisted model is on disk;
    default the mock to True here and let the dedicated uncached-
    model test override it.
    """
    monkeypatch.setattr(
        "app.gui.controllers.app_controller.is_cached_for_info",
        lambda info: True,
    )


# ---- Init from config -------------------------------------------------------


def test_controller_sets_active_model_from_config_alias(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)

    assert window.models_view.active_alias() == "whisper-large-v3"


def test_controller_normalizes_canonical_model_in_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {"whisper": {"model": "onnx-community/whisper-large-v3"}}
    )

    AppController(config=config, window=window)

    assert window.models_view.active_alias() == "whisper-large-v3"


def test_controller_ignores_unknown_model_without_raising(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "totally-unknown-model"}})

    AppController(config=config, window=window)

    assert window.models_view.active_alias() is None


def test_controller_handles_missing_model_in_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})  # no whisper section at all

    AppController(config=config, window=window)

    assert window.models_view.active_alias() is None


def test_controller_skips_active_when_persisted_model_is_not_cached(
    qtbot, monkeypatch
):
    """If the persisted model isn't on disk yet (fresh install / cleared
    cache), don't restore it as Active — the green pill would advertise
    a ready state while the backend is empty, AND the Select/Download
    button stays hidden, leaving the user stuck. Force a deliberate
    Download click so progress is visible."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setattr(
        "app.gui.controllers.app_controller.is_cached_for_info",
        lambda info: False,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)

    assert window.models_view.active_alias() is None
    # Topbar should also reflect the no-model state.
    assert "No model" in window.topbar._model_pill.text() or window.topbar._model_pill.text() == "No model"


# ---- Selection ↔ persistence ------------------------------------------------


def test_controller_marks_view_loading_immediately_on_select(qtbot):
    """Clicking Download must paint the orange Loading pill on the
    card right away — without it, the user briefly sees the green
    Active pill (200 ms until the next state poll) before it flips
    to Loading, which looks like a flicker."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("whisper-large-v3-turbo")

    cards = {
        c.alias(): c
        for c in window.models_view.findChildren(
            __import__(
                "app.gui.widgets.model_card", fromlist=["ModelCard"]
            ).ModelCard
        )
    }
    assert cards["whisper-large-v3-turbo"].is_loading() is True


def test_controller_persists_selection_back_to_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("whisper-large-v3-turbo")

    assert ("whisper", "model", "whisper-large-v3-turbo") in config.writes
    assert window.models_view.active_alias() == "whisper-large-v3-turbo"


def test_controller_no_ops_when_selecting_already_active(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)
    config.writes.clear()

    window.models_view.model_selected.emit("whisper-large-v3")

    assert config.writes == []


# ---- Shortcuts ↔ config -----------------------------------------------------


def test_controller_prefills_shortcuts_view_from_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {
            "hotkey": {
                "start_recording_hotkey": "ctrl+f2",
                "stop_recording_hotkey": "ctrl+f3",
            },
            "clipboard": {"auto_paste": False},
        }
    )

    AppController(config=config, window=window)

    assert window.shortcuts_view.start_hotkey() == "ctrl+f2"
    assert window.shortcuts_view.stop_hotkey() == "ctrl+f3"
    assert window.shortcuts_view.auto_paste() is False


def test_controller_persists_shortcuts_on_save(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {
            "hotkey": {
                "start_recording_hotkey": "ctrl+f2",
                "stop_recording_hotkey": "ctrl+f3",
            },
            "clipboard": {"auto_paste": False},
        }
    )

    AppController(config=config, window=window)
    window.shortcuts_view.save_requested.emit(
        {
            "start_hotkey": "ctrl+alt+1",
            "stop_hotkey": "ctrl+alt+2",
            "auto_paste": True,
        }
    )

    assert ("hotkey", "start_recording_hotkey", "ctrl+alt+1") in config.writes
    assert ("hotkey", "stop_recording_hotkey", "ctrl+alt+2") in config.writes
    assert ("clipboard", "auto_paste", True) in config.writes


def test_controller_handles_missing_shortcut_sections(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})

    AppController(config=config, window=window)

    assert window.shortcuts_view.start_hotkey() == ""
    assert window.shortcuts_view.stop_hotkey() == ""
    assert window.shortcuts_view.auto_paste() is False


def test_reset_shortcuts_restores_defaults_in_config(qtbot):
    from app.config_manager import DEFAULT_CONFIG
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {
            "hotkey": {
                "start_recording_hotkey": "ctrl+x",
                "stop_recording_hotkey": "ctrl+y",
            },
            "clipboard": {"auto_paste": False},
        }
    )

    AppController(config=config, window=window)
    window.shortcuts_view.hotkeys_reset_requested.emit()

    expected_start = DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]
    expected_stop = DEFAULT_CONFIG["hotkey"]["stop_recording_hotkey"]

    assert ("hotkey", "start_recording_hotkey", expected_start) in config.writes
    assert ("hotkey", "stop_recording_hotkey", expected_stop) in config.writes
    # Per-card reset: ``auto_paste`` is no longer reset by the
    # hotkeys button — it's a separate setting and would have its
    # own reset path on the Clipboard card if we wanted one.
    assert not any(
        write[0] == "clipboard" and write[1] == "auto_paste"
        for write in config.writes
    )


def test_reset_shortcuts_updates_view_to_defaults(qtbot):
    from app.config_manager import DEFAULT_CONFIG
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {
            "hotkey": {
                "start_recording_hotkey": "ctrl+x",
                "stop_recording_hotkey": "ctrl+y",
            },
            "clipboard": {"auto_paste": False},
        }
    )

    AppController(config=config, window=window)
    window.shortcuts_view.hotkeys_reset_requested.emit()

    sv = window.shortcuts_view
    assert sv.start_hotkey() == DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]
    assert sv.stop_hotkey() == DEFAULT_CONFIG["hotkey"]["stop_recording_hotkey"]
    # auto_paste preserved (was False, still False) — hotkeys reset
    # doesn't touch it.
    assert sv.auto_paste() is False


def test_reset_does_not_re_emit_save_requested(qtbot):
    """Reset programmatically updates fields — must not feed back as a save."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"hotkey": {"start_recording_hotkey": "ctrl+x"}})

    AppController(config=config, window=window)

    # Capture only writes that happen AFTER the reset.
    writes_before = len(config.writes)
    window.shortcuts_view.hotkeys_reset_requested.emit()
    writes_during = len(config.writes) - writes_before

    # Reset writes exactly 3 settings: start, stop, cancel.
    # If save_requested re-fired from set_values, we'd see extras.
    assert writes_during == 3


# ---- Topbar sync ------------------------------------------------------------


def test_controller_syncs_topbar_model_on_init(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)

    assert "Large v3" in window.topbar._model_pill.text()


def test_controller_updates_topbar_on_model_select(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("vosk-ru-small")

    assert "Vosk" in window.topbar._model_pill.text()


def test_controller_clears_topbar_model_when_unknown(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "weird-custom-model"}})

    AppController(config=config, window=window)

    assert "no model" in window.topbar._model_pill.text().lower()


# ---- History ↔ manager ------------------------------------------------------


class FakeHistoryEntry:
    def __init__(self, text):
        self.timestamp = 0.0
        self.text = text
        self.duration = 1.0
        self.model = "whisper-large-v3"
        self.language = "ru"
        self.datetime_str = "00:00:00"
        self.short_text = text[:50]


class FakeHistory:
    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.max_entries = 1000
        self.cleared = False
        self.exported_to: list[str] = []
        self.export_returns: bool = True

    def get_entries(self):
        return list(self._entries)

    def clear_history(self):
        self._entries.clear()
        self.cleared = True

    def export_to_text(self, filepath: str) -> bool:
        self.exported_to.append(filepath)
        return self.export_returns


def test_controller_populates_history_view_from_manager(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("a"), FakeHistoryEntry("b")])

    AppController(config=config, window=window, history=history)

    table_model = window.history_view._source_model
    assert table_model.rowCount() == 2


def test_controller_clears_history_through_manager(qtbot, monkeypatch):
    """Clear is destructive — confirm via QMessageBox before
    forwarding to the manager. The test simulates clicking Yes."""
    from PySide6.QtWidgets import QMessageBox
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("a")])

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: QMessageBox.Yes,
    )

    AppController(config=config, window=window, history=history)
    window.history_view.clear_requested.emit()

    assert history.cleared is True
    assert window.history_view._source_model.rowCount() == 0


def test_controller_deletes_cached_model_after_confirm(qtbot, monkeypatch):
    """Yes on the confirmation dialog → delete is called with the
    matching ModelInfo, then ``refresh_cache_state`` is invoked so the
    Download/Select label and the Delete-button visibility update."""
    from PySide6.QtWidgets import QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.Yes)

    deleted: list = []

    def fake_delete(info):
        deleted.append(info.alias)
        return True

    monkeypatch.setattr(controller_module, "delete_cached_for_info", fake_delete)

    refreshed = {"called": False}

    def fake_refresh():
        refreshed["called"] = True

    monkeypatch.setattr(window.models_view, "refresh_cache_state", fake_refresh)

    AppController(config=config, window=window)
    window.models_view.model_delete_requested.emit("vosk-ru-small")

    assert deleted == ["vosk-ru-small"]
    assert refreshed["called"] is True


def test_controller_does_not_delete_when_user_cancels(qtbot, monkeypatch):
    """Cancel on the confirmation dialog → cache stays put and no
    refresh fires (UI was already correct)."""
    from PySide6.QtWidgets import QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.Cancel)

    deleted: list = []
    monkeypatch.setattr(
        controller_module,
        "delete_cached_for_info",
        lambda info: deleted.append(info.alias) or True,
    )

    AppController(config=config, window=window)
    window.models_view.model_delete_requested.emit("vosk-ru-small")

    assert deleted == []


def test_controller_delete_dialog_warns_about_shared_canonical(qtbot, monkeypatch):
    """``gigaam-v3-ctc`` and ``gigaam-v3-rnnt`` point at the same HF
    repo (``istupakov/gigaam-v3-onnx``); deleting one wipes weights for
    both decoders.  The confirm-dialog text must mention the sibling
    so the user isn't surprised when the other card flips back to
    'Download'."""
    from PySide6.QtWidgets import QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    captured: dict = {}

    def fake_question(parent, title, text, *args, **kwargs):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    monkeypatch.setattr(
        controller_module, "delete_cached_for_info", lambda info: True
    )

    AppController(config=config, window=window)
    window.models_view.model_delete_requested.emit("gigaam-v3-ctc")

    # Sibling alias mentioned somewhere in the dialog body.
    assert "gigaam-v3-rnnt" in captured["text"]


def test_controller_prefills_storage_path_from_config(qtbot, monkeypatch):
    """The Storage card needs to render the configured (or default)
    path on first paint — without this the user sees '(loading…)'
    forever even though the value is already in config."""
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": "D:/models"}})

    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )

    AppController(config=config, window=window)
    label = window.shortcuts_view.findChild(
        type(window.shortcuts_view._storage_path_label),
        "StoragePathLabel",
    )
    assert "D:/models" in label.text()


def test_controller_prefill_marks_default_when_config_empty(qtbot, monkeypatch):
    """Empty/missing ``storage.models_dir`` → label still shows the
    *resolved* default path AND a '(default)' marker — so the user
    knows where weights actually go even when nothing's overridden."""
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default-models"
    )

    AppController(config=config, window=window)
    label = window.shortcuts_view.findChild(
        type(window.shortcuts_view._storage_path_label),
        "StoragePathLabel",
    )
    text = label.text()
    assert "C:/default-models" in text
    assert "default" in text.lower()


def test_controller_storage_change_writes_config_and_updates_env(
    qtbot, monkeypatch,
):
    """Picking a folder via QFileDialog → controller writes
    ``storage.models_dir`` AND mirrors the new path into the live
    process environment so the next ``WhisperModel`` /
    ``gigaam.load_model`` call uses it without a restart.

    The info dialog no longer mentions restarting (it would be a
    lie now)."""
    import os
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    # Tell monkeypatch about both env vars up-front so it tracks
    # them and reverts on teardown — without this the production
    # code's direct ``os.environ`` writes leak into other tests.
    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    chosen = "D:/lazy-to-text-models"
    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: chosen,
    )
    info_calls: list = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **kw: info_calls.append((a, kw)),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    assert config._data.get("storage", {}).get("models_dir") == chosen
    # ``HF_HOME`` updated live so the next ``onnx_asr.load_model``
    # download routes through huggingface_hub into the new root.
    assert os.environ.get("HF_HOME") == chosen
    # Info dialog body must NOT mention restart/next-launch — that
    # wording is now a lie since the change applies live.
    assert info_calls, "expected QMessageBox.information to fire after change"
    args, _kwargs = info_calls[0]
    body_text = " ".join(str(a) for a in args).lower()
    assert "restart" not in body_text
    assert "next launch" not in body_text


def test_controller_storage_change_cancelled_writes_nothing(qtbot, monkeypatch):
    """User clicks Cancel on the folder picker → no config write,
    no info dialog."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": "D:/old"}})

    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **kw: "",
    )
    info_calls: list = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **kw: info_calls.append(a),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    assert config._data["storage"]["models_dir"] == "D:/old"
    assert info_calls == []


def test_controller_storage_change_offers_migration_when_old_has_weights(
    qtbot, monkeypatch, tmp_path,
):
    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")
    """Old root has cached weights → controller pops a Yes/No/Cancel
    prompt offering to move them. ``Yes`` triggers ``move_cached_dir``
    for both ``hub/`` and ``gigaam/`` (whichever exist) and writes
    the new path to config."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    old_root = tmp_path / "old-models"
    new_root = tmp_path / "new-models"
    (old_root / "hub").mkdir(parents=True)
    (old_root / "hub" / "model.bin").write_bytes(b"x" * 4096)

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    # Old config is empty → resolves to old_root (default).
    monkeypatch.setattr(
        controller_module, "get_models_root",
        lambda v: v or str(old_root),
    )
    # Folder picker returns the new path.
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: str(new_root),
    )
    # User clicks Yes on the migration prompt.
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: QMessageBox.Yes,
    )
    # Swallow the post-migration restart info dialog.
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    moves: list = []
    real_move = controller_module.move_cached_dir

    def tracking_move(src, dst):
        moves.append((src, dst))
        return real_move(src, dst)

    monkeypatch.setattr(controller_module, "move_cached_dir", tracking_move)

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    # Hub moved, gigaam absent so its move was a no-op (still
    # called — controller decides per-subdir).
    assert any("hub" in src for src, _ in moves)
    # Files actually moved on disk.
    assert (new_root / "hub" / "model.bin").exists()
    # New path written to config.
    assert config._data["storage"]["models_dir"] == str(new_root)


def test_controller_storage_change_no_prompt_when_old_root_is_empty(
    qtbot, monkeypatch, tmp_path,
):
    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")
    """Old root has nothing → skip the migration prompt entirely.
    The user only sees the standard 'restart required' info."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    new_root = tmp_path / "new-models"
    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    monkeypatch.setattr(
        controller_module, "get_models_root",
        lambda v: v or str(tmp_path / "definitely-empty"),
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: str(new_root),
    )

    question_calls: list = []

    def fake_question(*args, **kwargs):
        question_calls.append(args)
        return QMessageBox.Yes  # would say Yes if asked

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    # No migration prompt fired — old root was empty.
    assert question_calls == []
    # Path still written.
    assert config._data["storage"]["models_dir"] == str(new_root)


def test_controller_storage_change_no_on_migration_writes_config_only(
    qtbot, monkeypatch, tmp_path,
):
    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")
    """Old has weights, user clicks ``No`` on the migration prompt →
    config still updates (so future downloads go to new place) but
    nothing moves on disk."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    old_root = tmp_path / "old-models"
    new_root = tmp_path / "new-models"
    (old_root / "hub").mkdir(parents=True)
    (old_root / "hub" / "model.bin").write_bytes(b"x" * 1024)

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    monkeypatch.setattr(
        controller_module, "get_models_root",
        lambda v: v or str(old_root),
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: str(new_root),
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: QMessageBox.No,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    moves: list = []
    monkeypatch.setattr(
        controller_module, "move_cached_dir",
        lambda src, dst: moves.append((src, dst)),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    assert moves == []
    assert (old_root / "hub" / "model.bin").exists()
    assert config._data["storage"]["models_dir"] == str(new_root)


def test_controller_storage_change_cancel_on_migration_aborts(
    qtbot, monkeypatch, tmp_path,
):
    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")
    """Cancel on the migration prompt → don't write config either,
    so the user can pick a different folder without leaving a
    half-applied state."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    old_root = tmp_path / "old-models"
    new_root = tmp_path / "new-models"
    (old_root / "hub").mkdir(parents=True)
    (old_root / "hub" / "model.bin").write_bytes(b"x")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": "C:/initial"}})

    monkeypatch.setattr(
        controller_module, "get_models_root",
        lambda v: v if v != "C:/initial" else str(old_root),
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: str(new_root),
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: QMessageBox.Cancel,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    # Nothing changed — user can re-pick.
    assert config._data["storage"]["models_dir"] == "C:/initial"


def test_controller_storage_change_refreshes_model_card_cache_state(
    qtbot, monkeypatch,
):
    """After changing the models folder, ``models_view.refresh_cache_state``
    must be called so cards immediately show 'Select' instead of
    'Download' for models already present in the new directory."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_HOME", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": ""}})

    chosen = "D:/my-models"
    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        lambda *a, **kw: chosen,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    refresh_calls: list = []
    monkeypatch.setattr(
        window.models_view, "refresh_cache_state",
        lambda: refresh_calls.append(1),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_path_change_requested.emit()

    assert refresh_calls, (
        "refresh_cache_state must be called so model cards update "
        "Download→Select without requiring a restart"
    )


def test_controller_storage_reset_refreshes_model_card_cache_state(
    qtbot, monkeypatch,
):
    """Same contract for the Reset path: returning to the default
    directory must also trigger a cache-state refresh on all cards."""
    from PySide6.QtWidgets import QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_HOME", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": "D:/old"}})

    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    refresh_calls: list = []
    monkeypatch.setattr(
        window.models_view, "refresh_cache_state",
        lambda: refresh_calls.append(1),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_reset_requested.emit()

    assert refresh_calls, (
        "refresh_cache_state must be called on reset so model cards "
        "reflect the default directory's cache state immediately"
    )


def test_controller_prefills_hf_token_from_config(qtbot, monkeypatch):
    """The persisted token must paint the Settings field on first
    render — without that the user can't edit it (the field shows
    blank then gets overwritten by save) and shoulder-surfing risk
    via screenshare looks identical."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"huggingface": {"token": "hf_persisted"}})

    AppController(config=config, window=window)
    assert window.shortcuts_view.hf_token() == "hf_persisted"


def test_controller_writes_hf_token_to_config_and_env(qtbot, monkeypatch):
    """User edits the token field → controller writes to
    ``huggingface.token`` AND mirrors into ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` env vars so the next download picks
    it up without a restart."""
    import os
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"huggingface": {"token": ""}})

    AppController(config=config, window=window)
    window.shortcuts_view.hf_token_changed.emit("hf_brand_new")

    assert config._data["huggingface"]["token"] == "hf_brand_new"
    assert os.environ.get("HF_TOKEN") == "hf_brand_new"
    assert os.environ.get("HUGGING_FACE_HUB_TOKEN") == "hf_brand_new"


def test_controller_clearing_hf_token_removes_env(qtbot, monkeypatch):
    """Empty value in the field → drop env vars entirely so
    huggingface_hub doesn't try to use a stale token."""
    import os
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_TOKEN", "stale_value")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "stale_value")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"huggingface": {"token": "stale_value"}})

    AppController(config=config, window=window)
    window.shortcuts_view.hf_token_changed.emit("")

    assert config._data["huggingface"]["token"] == ""
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


# HF-token-warning UX was removed when GigaAM moved to its ONNX path
# (no more pyannote / gated weights).  The legacy
# ``test_controller_hf_token_change_refreshes_model_cards`` is gone
# with the feature.


def test_controller_prefills_cancel_hotkey_from_config(qtbot):
    """The controller paints whatever's in ``hotkey.cancel_recording_hotkey``
    into the new third field on first render."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({
        "hotkey": {
            "start_recording_hotkey": "ctrl+f2",
            "stop_recording_hotkey": "ctrl+f3",
            "cancel_recording_hotkey": "ctrl+f6",
        },
    })

    AppController(config=config, window=window)
    assert window.shortcuts_view.cancel_hotkey() == "ctrl+f6"


def test_controller_persists_cancel_hotkey_on_save(qtbot):
    """Save payload from the view carries ``cancel_hotkey``;
    controller writes it under ``hotkey.cancel_recording_hotkey``."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({
        "hotkey": {
            "start_recording_hotkey": "ctrl+f2",
            "stop_recording_hotkey": "ctrl+f3",
            "cancel_recording_hotkey": "",
        },
    })

    AppController(config=config, window=window)
    window.shortcuts_view.save_requested.emit({
        "start_hotkey": "ctrl+f2",
        "stop_hotkey": "ctrl+f3",
        "cancel_hotkey": "ctrl+alt+x",
        "auto_paste": True,
    })

    assert config._data["hotkey"]["cancel_recording_hotkey"] == "ctrl+alt+x"


def test_controller_reset_restores_cancel_hotkey_default(qtbot):
    """Reset to defaults populates all three hotkey fields, including
    cancel — otherwise a user who cleared it can't quickly get the
    default back."""
    from app.config_manager import DEFAULT_CONFIG
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({
        "hotkey": {
            "start_recording_hotkey": "ctrl+x",
            "stop_recording_hotkey": "ctrl+y",
            "cancel_recording_hotkey": "",
        },
    })

    AppController(config=config, window=window)
    window.shortcuts_view.hotkeys_reset_requested.emit()

    expected_default = DEFAULT_CONFIG["hotkey"]["cancel_recording_hotkey"]
    assert config._data["hotkey"]["cancel_recording_hotkey"] == expected_default
    assert window.shortcuts_view.cancel_hotkey() == expected_default


def test_controller_hf_token_clear_button_wipes_config_and_env(
    qtbot, monkeypatch,
):
    """Per-card 'Clear token' on the HF card must do the same as
    typing an empty value into the field: drop config + env, repaint
    the field, refresh the GigaAM warning banner."""
    import os
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"huggingface": {"token": "hf_persisted"}})

    AppController(config=config, window=window)
    # Sanity: the prefill plus init's apply-to-env populates env.
    assert os.environ.get("HF_TOKEN") == "hf_persisted"
    assert window.shortcuts_view.hf_token() == "hf_persisted"

    window.shortcuts_view.hf_token_reset_requested.emit()

    assert config._data["huggingface"]["token"] == ""
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ
    assert window.shortcuts_view.hf_token() == ""


def test_controller_storage_reset_clears_config_and_updates_env(
    qtbot, monkeypatch,
):
    """Reset clears ``storage.models_dir`` and mirrors the change
    into the live env: ``HF_HOME`` snaps back to the resolved
    default and ``GIGAAM_MODELS_DIR`` is removed entirely so GigaAM
    falls back to its library default ``~/.cache/gigaam``."""
    import os
    from PySide6.QtWidgets import QMessageBox
    import app.gui.controllers.app_controller as controller_module
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    # ``setenv`` to a placeholder forces monkeypatch to track the
    # variable; production code under test will overwrite it with
    # ``os.environ[...] = ...`` which monkeypatch can then restore on
    # teardown. ``delenv(raising=False)`` doesn't track unset vars
    # — leaks env changes into other tests.
    monkeypatch.setenv("HF_HOME", "")

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"storage": {"models_dir": "D:/old"}})

    monkeypatch.setattr(
        controller_module, "get_models_root", lambda v: v or "C:/default"
    )
    info_calls: list = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **kw: info_calls.append(a),
    )

    AppController(config=config, window=window)
    window.shortcuts_view.storage_reset_requested.emit()

    assert config._data["storage"]["models_dir"] == ""
    # HF_HOME → resolved default.
    assert os.environ.get("HF_HOME") == "C:/default"
    assert info_calls, "expected info dialog after reset"


def test_controller_clear_cancelled_keeps_entries(qtbot, monkeypatch):
    """If the user clicks Cancel on the confirm dialog, history must
    stay intact."""
    from PySide6.QtWidgets import QMessageBox
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("a"), FakeHistoryEntry("b")])

    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **kw: QMessageBox.Cancel,
    )

    AppController(config=config, window=window, history=history)
    window.history_view.clear_requested.emit()

    assert history.cleared is False
    assert window.history_view._source_model.rowCount() == 2


def test_controller_export_writes_through_manager(qtbot, monkeypatch):
    """Export should pop a save-as dialog and forward the chosen path
    to ``history_manager.export_to_text``."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("hi")])

    chosen_path = "C:/tmp/history-export.txt"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **kw: (chosen_path, "Text files (*.txt)"),
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **kw: None)

    AppController(config=config, window=window, history=history)
    window.history_view.export_requested.emit()

    assert history.exported_to == [chosen_path]


def test_controller_export_cancelled_does_not_call_manager(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("hi")])

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **kw: ("", ""),  # user clicked Cancel
    )

    AppController(config=config, window=window, history=history)
    window.history_view.export_requested.emit()

    assert history.exported_to == []


def test_controller_export_with_empty_history_skips_dialog(qtbot, monkeypatch):
    """Don't bother the user with a save-as dialog when there's
    nothing to write — just inform them."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([])

    save_called = []
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        lambda *a, **kw: (save_called.append(True), ("", ""))[1],
    )
    info_called = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda *a, **kw: info_called.append(True),
    )

    AppController(config=config, window=window, history=history)
    window.history_view.export_requested.emit()

    assert save_called == []  # save dialog never shown
    assert info_called  # informational popup shown instead
    assert history.exported_to == []


def test_controller_copy_writes_to_clipboard(qtbot):
    from PySide6.QtWidgets import QApplication

    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("abc")])

    AppController(config=config, window=window, history=history)
    window.history_view.copy_requested.emit("abc")

    assert QApplication.clipboard().text() == "abc"


def test_controller_works_without_history_manager(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    AppController(config=config, window=window)  # no history arg

    assert window.history_view._source_model.rowCount() == 0


# ---- Recording controller wiring -------------------------------------------


class FakeAudioRecorder:
    def __init__(self, peak: float = 0.4, rms: float = 0.15) -> None:
        self._peak = peak
        self._rms = rms
        self.test_calls = 0
        self.raise_on_test: Optional[Exception] = None

    def test_input_level(self, duration_s: float = 3.0) -> dict:
        self.test_calls += 1
        if self.raise_on_test is not None:
            raise self.raise_on_test
        return {
            "peak": self._peak,
            "rms": self._rms,
            "duration_s": duration_s,
        }


class FakeClipboardManager:
    def __init__(self) -> None:
        self.auto_paste_calls: list[bool] = []

    def update_auto_paste(self, enabled: bool) -> None:
        self.auto_paste_calls.append(bool(enabled))


class FakeStateManager:
    def __init__(
        self,
        audio_recorder: Optional[FakeAudioRecorder] = None,
        clipboard_manager: Optional[FakeClipboardManager] = None,
    ) -> None:
        self.audio_recorder = audio_recorder
        self.clipboard_manager = clipboard_manager


class FakeRecordingController:
    """Stand-in exposing the surface AppController consumes."""

    def __init__(
        self,
        model_change_returns: bool = True,
        state_manager: Optional[FakeStateManager] = None,
        current_state_value: str = "idle",
    ) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Bus(QObject):
            state_changed = Signal(str)
            history_updated = Signal()

        self._bus = _Bus()
        self.state_changed = self._bus.state_changed
        self.history_updated = self._bus.history_updated
        self.model_change_requests: list[str] = []
        self._model_change_returns = model_change_returns
        self.state_manager = state_manager
        self._current_state = current_state_value

    def request_model_change(self, canonical: str, compute_type=None) -> bool:
        self.model_change_requests.append((canonical, compute_type))
        return self._model_change_returns

    def cancel_model_change(self) -> bool:
        self.cancel_model_change_calls = (
            getattr(self, "cancel_model_change_calls", 0) + 1
        )
        return True

    def current_state(self) -> str:
        return self._current_state


def test_controller_updates_sidebar_recording_pill_on_state_change(qtbot):
    """Recording pill lives in the sidebar's bottom-left slot now —
    the controller fans recording state out to it the same way it
    used to fan it to the topbar's pill."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)

    pill = window.sidebar.recording_status._pill

    rec.state_changed.emit("recording")
    assert pill.property("state") == "recording"

    rec.state_changed.emit("idle")
    assert pill.property("state") == "idle"


def test_controller_routes_topbar_cancel_to_recording_controller(qtbot):
    """The topbar's Cancel button emits ``cancel_load_requested``; the
    AppController must wire that into ``recording.cancel_model_change``
    so a click actually stops the in-flight load."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)

    window.topbar.cancel_load_requested.emit()

    assert getattr(rec, "cancel_model_change_calls", 0) == 1


# ---- Cancel-load rollback (the half-clicked card bug) ---------------------


def test_cancel_after_select_reverts_active_card_to_previous(qtbot):
    """The user had Whisper Large v3 active. They click on a different
    card → the controller flips the new card to Active and starts the
    load. They click Cancel → the previously-active card must reclaim
    the green Active pill, otherwise a model the backend never loaded
    looks confirmed in the UI."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)
    assert window.models_view.active_alias() == "whisper-large-v3"

    # User clicks a different card.
    window.models_view.model_selected.emit("whisper-large-v3-turbo")
    assert window.models_view.active_alias() == "whisper-large-v3-turbo"

    # User clicks Cancel.
    window.topbar.cancel_load_requested.emit()

    assert window.models_view.active_alias() == "whisper-large-v3"


def test_cancel_after_select_clears_active_when_no_prior_card(qtbot):
    """Fresh install — no previously-active card. User clicks a card,
    decides they don't want it, hits Cancel. No card should be Active
    afterwards."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})  # no persisted model
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)
    assert window.models_view.active_alias() is None

    window.models_view.model_selected.emit("whisper-large-v3-turbo")
    assert window.models_view.active_alias() == "whisper-large-v3-turbo"

    window.topbar.cancel_load_requested.emit()

    assert window.models_view.active_alias() is None


def test_cancel_after_select_restores_topbar_pill(qtbot):
    """The topbar's display name shadows the active card. After a
    cancel, the previously-active model's display name must come
    back — otherwise the user sees ``Loading: <half-clicked>`` flip
    to ``Current model: <half-clicked>`` of a model the backend
    never actually loaded."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)
    # Snapshot: large-v3 display name is in the pill.
    initial_display = window.topbar._model_display_name

    window.models_view.model_selected.emit("whisper-large-v3-turbo")
    # Pill was just flipped to a different model.
    assert window.topbar._model_display_name != initial_display

    window.topbar.cancel_load_requested.emit()

    assert window.topbar._model_display_name == initial_display


def test_cancel_after_select_restores_config(qtbot):
    """``_on_model_selected`` writes the new alias to config
    immediately so a crash-on-load doesn't leave a half-applied
    state. Cancel must roll the config back too — otherwise next
    launch picks up the model the user explicitly cancelled."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)

    window.models_view.model_selected.emit("whisper-large-v3-turbo")
    assert config.get_setting("whisper", "model") == "whisper-large-v3-turbo"

    window.topbar.cancel_load_requested.emit()

    assert config.get_setting("whisper", "model") == "whisper-large-v3"


def test_download_progress_falls_back_to_size_mb_when_total_zero(qtbot):
    """NeMo's ``cloud.maybe_download_from_cloud`` streams via plain
    ``requests`` without a Content-Length header, so the tqdm bar
    fires with ``total=0``. Without a fallback the topbar pill
    shows raw bytes ('Loading: ... 1.3 GB') for the full 4-minute
    download instead of a percentage that climbs.

    The controller plugs ``ModelInfo.size_mb * 1 MB`` in as the
    fallback total so the bar still climbs 0% → 99% smoothly."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow
    from app.model_mapping import get_model

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})
    rec = FakeRecordingController()

    controller = AppController(config=config, window=window, recording=rec)
    # User clicks Parakeet — controller records the expected total
    # from the registry (Parakeet TDT v3 → 1200 MB).
    window.models_view.model_selected.emit("parakeet-tdt-v3")
    # Mirror the recording controller emitting model_loading so the
    # topbar pill renders its loading variant.
    window.topbar.set_recording_state("model_loading")
    # Half the expected size has streamed in; tqdm reports total=0.
    expected_total = get_model("parakeet-tdt-v3").size_mb * 1024 * 1024
    controller._on_download_progress(expected_total // 2, 0, "model.nemo")

    pill_text = window.topbar._model_pill.text()
    # 50% (or 49% — the topbar caps progress at 99 to avoid showing
    # 100% before ready). Either way "%" must appear, NOT raw bytes.
    assert "%" in pill_text, f"expected percentage, got: {pill_text}"
    assert "GB" not in pill_text and "MB" not in pill_text, (
        f"raw bytes leaked through, got: {pill_text}"
    )


def test_download_progress_preserves_real_total(qtbot):
    """Faster-whisper / GigaAM paths give tqdm a real total via
    ``huggingface_hub.snapshot_download``. The controller must NOT
    clobber a real total with the size_mb fallback."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})
    rec = FakeRecordingController()

    controller = AppController(config=config, window=window, recording=rec)
    window.models_view.model_selected.emit("whisper-large-v3")
    window.topbar.set_recording_state("model_loading")
    # 50 MB out of 200 MB — should render as 25%.
    controller._on_download_progress(50_000_000, 200_000_000, "model.bin")

    pill_text = window.topbar._model_pill.text()
    assert "25%" in pill_text, f"expected 25%, got: {pill_text}"


def test_download_progress_resets_expected_total_on_idle(qtbot):
    """After a successful load the expected-bytes fallback must
    drop back to zero — otherwise a future download whose tqdm
    actually does report total=0 (a small misc file in some
    other backend) would inherit Parakeet's 1200 MB fallback and
    show nonsense progress."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({})
    rec = FakeRecordingController()

    controller = AppController(config=config, window=window, recording=rec)
    window.models_view.model_selected.emit("parakeet-tdt-v3")
    assert controller._loading_expected_bytes > 0

    # Simulate the load finishing — recording state goes idle.
    rec.state_changed.emit("idle")

    assert controller._loading_expected_bytes == 0


def test_cancel_with_no_load_in_flight_does_not_revert(qtbot):
    """If cancel arrives while nothing is loading (the recording
    controller returns falsy), the rollback path must NOT fire —
    otherwise an unrelated cancel click would un-mark a happily-
    loaded card."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    class _NoCancelRec(FakeRecordingController):
        def cancel_model_change(self) -> bool:
            self.cancel_model_change_calls = (
                getattr(self, "cancel_model_change_calls", 0) + 1
            )
            return False  # nothing was loading

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})
    rec = _NoCancelRec()

    AppController(config=config, window=window, recording=rec)
    # Pretend the user picked "whisper-large-v3-turbo" earlier and it actually loaded —
    # then they hit Cancel idly with nothing in flight.
    window.models_view.model_selected.emit("whisper-large-v3-turbo")
    # Suppose loading completed; backend reports ready, etc. We
    # simulate that by clearing the snapshot the way a real
    # ``ready`` state would (the rollback target). This is the
    # behavioural assertion: the snapshot should NOT survive once
    # the load has succeeded.
    window.topbar.cancel_load_requested.emit()

    # The cancel reached the recording controller (one call).
    assert rec.cancel_model_change_calls == 1
    # But because cancel returned False, the active card stayed at
    # the just-selected one — no spurious rollback.
    assert window.models_view.active_alias() == "whisper-large-v3-turbo"


def test_controller_refreshes_history_on_history_updated(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("first")])
    rec = FakeRecordingController()

    AppController(config=config, window=window, history=history, recording=rec)

    # Simulate the history manager prepending a new entry (newest-first).
    history._entries.insert(0, FakeHistoryEntry("second"))
    rec.history_updated.emit()

    from PySide6.QtCore import Qt
    model = window.history_view._source_model
    assert model.rowCount() == 2
    assert model.data(model.index(0, 1), Qt.DisplayRole) == "second"


def test_controller_shows_toast_on_history_updated(qtbot):
    """A successful transcription is silent in the UI otherwise — the
    text just appears on the clipboard. Surfacing a confirmation
    toast with the latest entry's text gives the user a 'yes, the
    hotkey worked' moment."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    window.resize(800, 600)
    qtbot.addWidget(window)
    window.show()

    config = FakeConfig()
    history = FakeHistory([])
    rec = FakeRecordingController()

    AppController(config=config, window=window, history=history, recording=rec)

    history._entries.insert(0, FakeHistoryEntry("transcribed phrase"))
    rec.history_updated.emit()

    assert window.toast.isVisible()
    assert "transcribed phrase" in window.toast._body.text()


def test_controller_routes_model_select_through_recording_when_present(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)
    window.models_view.model_selected.emit("vosk-ru-small")

    # config still updated for persistence
    assert ("whisper", "model", "vosk-ru-small") in config.writes
    # compute_type written too — the registry tells us each card's preference
    assert ("whisper", "compute_type", "float16") in config.writes
    # AND recording stack was asked to actually switch (with compute_type)
    assert rec.model_change_requests == [
        ("alphacep/vosk-model-small-ru", "float16"),
    ]


def test_controller_skips_recording_call_when_recording_absent(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)  # no recording arg
    window.models_view.model_selected.emit("vosk-ru-small")

    # Should still write config and not crash.
    assert ("whisper", "model", "vosk-ru-small") in config.writes


def test_controller_locks_models_view_when_state_not_idle(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)

    rec.state_changed.emit("recording")
    assert window.models_view.is_locked() is True

    rec.state_changed.emit("processing")
    assert window.models_view.is_locked() is True

    rec.state_changed.emit("model_loading")
    assert window.models_view.is_locked() is True

    rec.state_changed.emit("idle")
    assert window.models_view.is_locked() is False


# ---- Tray wiring ----------------------------------------------------------


class FakeTrayIcon:
    """Stand-in exposing the surface AppController consumes."""

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Bus(QObject):
            show_requested = Signal()
            quit_requested = Signal()

        self._bus = _Bus()
        self.show_requested = self._bus.show_requested
        self.quit_requested = self._bus.quit_requested
        self.states: list[str] = []
        self.shown = False

    def setVisible(self, visible: bool) -> None:
        self.shown = bool(visible)

    def set_state(self, state: str) -> None:
        self.states.append(state)


def test_controller_with_tray_enables_close_to_tray_on_window(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    tray = FakeTrayIcon()

    AppController(config=config, window=window, tray=tray)

    assert window._close_to_tray is True


def test_controller_show_requested_brings_window_back(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.hide()
    assert not window.isVisible()

    config = FakeConfig()
    tray = FakeTrayIcon()
    AppController(config=config, window=window, tray=tray)

    tray.show_requested.emit()
    assert window.isVisible()


def test_controller_quit_requested_calls_request_quit_and_app_quit(qtbot, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    tray = FakeTrayIcon()

    quit_calls: list[None] = []
    original = window.request_quit
    window.request_quit = lambda: quit_calls.append(None) or original()

    app_quit_calls: list[None] = []
    monkeypatch.setattr(
        QApplication.instance(), "quit",
        lambda: app_quit_calls.append(None),
    )

    AppController(config=config, window=window, tray=tray)
    tray.quit_requested.emit()

    # Both must fire — closing the window alone does not end the app loop
    # when setQuitOnLastWindowClosed(False) is set for tray support.
    assert quit_calls == [None]
    assert app_quit_calls == [None]


def test_controller_forwards_recording_state_to_tray(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    rec = FakeRecordingController()
    tray = FakeTrayIcon()

    AppController(config=config, window=window, recording=rec, tray=tray)

    rec.state_changed.emit("recording")
    rec.state_changed.emit("processing")
    rec.state_changed.emit("idle")

    assert tray.states == ["recording", "processing", "idle"]


def test_controller_without_tray_does_not_enable_close_to_tray(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    AppController(config=config, window=window)
    assert window._close_to_tray is False


def test_controller_runs_mic_test_and_reports_result(qtbot):
    """Clicking 'Test microphone' should kick off a background capture
    and land the result back on the Settings view via Qt signals."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    recorder = FakeAudioRecorder(peak=0.42, rms=0.18)
    rec = FakeRecordingController(state_manager=FakeStateManager(recorder))

    AppController(config=config, window=window, recording=rec)

    window.shortcuts_view.test_mic_requested.emit()

    # Result text is purely descriptive now — the VU meter, frozen
    # at the captured peak level, IS the visual "how loud" answer.
    # Numeric peak/rms still surface in the tooltip for bug reports.
    qtbot.waitUntil(
        lambda: "Looks good"
        in window.shortcuts_view._test_mic_label.text(),
        timeout=2000,
    )
    assert recorder.test_calls == 1
    assert window.shortcuts_view._test_mic_btn.isEnabled()
    tooltip = window.shortcuts_view._test_mic_label.toolTip()
    assert "0.42" in tooltip and "0.18" in tooltip
    # Meter stays visible after the test, holding the peak level so
    # the user sees how loud they actually were. ``isHidden()`` is
    # the right check in unit tests — ``isVisible()`` requires the
    # parent chain to be on screen, but here we only assert the
    # widget's own visibility flag.
    assert not window.shortcuts_view._test_mic_meter.isHidden()


def test_controller_mic_test_blocked_during_recording(qtbot):
    """Don't try to grab the input while a recording is in flight —
    both would race for the device."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    recorder = FakeAudioRecorder()
    rec = FakeRecordingController(
        state_manager=FakeStateManager(recorder),
        current_state_value="recording",
    )

    AppController(config=config, window=window, recording=rec)

    window.shortcuts_view.test_mic_requested.emit()
    # No worker thread should have run; label should report a refusal.
    assert recorder.test_calls == 0
    text = window.shortcuts_view._test_mic_label.text()
    assert text  # any non-empty error message


def test_controller_mic_test_surfaces_recorder_errors(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    recorder = FakeAudioRecorder()
    recorder.raise_on_test = RuntimeError("device busy")
    rec = FakeRecordingController(state_manager=FakeStateManager(recorder))

    AppController(config=config, window=window, recording=rec)

    window.shortcuts_view.test_mic_requested.emit()
    qtbot.waitUntil(
        lambda: "device busy" in window.shortcuts_view._test_mic_label.text(),
        timeout=2000,
    )


def test_controller_pushes_auto_paste_to_live_clipboard_manager(qtbot):
    """Toggling Auto-paste in Settings must update the running
    ClipboardManager — otherwise the checkbox flips visually but the
    actual delivery keeps using the value it had at startup."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"clipboard": {"auto_paste": True}})
    clipboard = FakeClipboardManager()
    rec = FakeRecordingController(
        state_manager=FakeStateManager(clipboard_manager=clipboard)
    )

    AppController(config=config, window=window, recording=rec)

    # Toggle off, then on, via the checkbox.
    cb = window.shortcuts_view._auto_paste_cb
    cb.setChecked(False)
    cb.setChecked(True)

    # Most recent should be True (re-enabled), and at least one False
    # along the way (when disabled).
    assert clipboard.auto_paste_calls
    assert clipboard.auto_paste_calls[-1] is True
    assert False in clipboard.auto_paste_calls


def test_test_microphone_button_does_not_grab_focus(qtbot):
    """``setEnabled(False)`` during the 3-second test would chase
    focus to the next focusable widget (the Start hotkey edit) if
    the button had focus. NoFocus prevents the button from grabbing
    focus on click in the first place."""
    from PySide6.QtCore import Qt
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    assert view._test_mic_btn.focusPolicy() == Qt.NoFocus
    # Per-card reset / clear buttons replaced the old single
    # ``_reset_btn`` footer — same NoFocus discipline applies so
    # they don't steal focus when clicked.
    assert view._reset_hotkeys_btn.focusPolicy() == Qt.NoFocus
    assert view._clear_hf_token_btn.focusPolicy() == Qt.NoFocus


def test_controller_loads_persisted_inference_overrides_on_init(qtbot):
    """When the user toggles a per-model setting, it lands in
    ``config['model_overrides'][alias]`` — and the controller must
    re-hydrate that mapping back into the card's inline panel on the
    next launch."""
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({
        "whisper": {"model": "whisper-large-v3"},
        "model_overrides": {
            "whisper-large-v3": {
                "language": "ru",
                "vad_filter": False,
                "beam_size": 7,
                "temperature": 0.4,
                "initial_prompt": "Anthropic, Claude",
            },
        },
    })

    AppController(config=config, window=window)

    settings = window.models_view._cards["whisper-large-v3"].inference_settings()
    assert settings.language == "ru"
    assert settings.vad_filter is False
    assert settings.beam_size == 7
    assert settings.temperature == 0.4
    assert settings.initial_prompt == "Anthropic, Claude"


def test_controller_persists_inference_change_on_active_card(qtbot):
    """Tweaking a control fires
    ``ModelsView.inference_settings_changed`` and the controller
    writes the new dict into ``config['model_overrides'][alias]``
    so it survives a restart."""
    from app.inference_settings import InferenceSettings
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window)

    new = InferenceSettings(
        language="en", vad_filter=True, beam_size=3, temperature=0.2,
        initial_prompt=None,
    )
    window.models_view.inference_settings_changed.emit("whisper-large-v3", new)

    saved = config.get_setting("model_overrides", "whisper-large-v3")
    assert saved["language"] == "en"
    assert saved["beam_size"] == 3
    assert saved["temperature"] == 0.2
    assert saved["vad_filter"] is True


def test_controller_pushes_inference_settings_to_live_backend_on_change(qtbot):
    """Editing the panel of the *active* card should also push the
    fresh values into the running backend so the next transcribe
    honours them without waiting for a restart."""
    from app.inference_settings import InferenceSettings
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    class _LiveBackend:
        def __init__(self) -> None:
            self.received: list[InferenceSettings] = []

        def update_inference_settings(self, settings) -> None:
            self.received.append(settings)

    backend = _LiveBackend()

    class _StateManager:
        def __init__(self) -> None:
            self.backend = backend
            self.audio_recorder = None
            self.clipboard_manager = None

    rec = FakeRecordingController(state_manager=_StateManager())
    rec.state_manager = _StateManager()
    rec.state_manager.backend = backend

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "whisper-large-v3"}})

    AppController(config=config, window=window, recording=rec)

    new = InferenceSettings(language="ru", vad_filter=False, beam_size=4)
    window.models_view.inference_settings_changed.emit("whisper-large-v3", new)

    # Most recent push must match what we emitted.
    assert backend.received[-1] == new
