"""Tests for the AppController wiring Models view to config storage."""

from typing import Any, Dict, List, Optional, Tuple


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


# ---- Init from config -------------------------------------------------------


def test_controller_sets_active_model_from_config_alias(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)

    assert window.models_view.active_alias() == "large-v3"


def test_controller_normalizes_canonical_model_in_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig(
        {"whisper": {"model": "Systran/faster-whisper-large-v3"}}
    )

    AppController(config=config, window=window)

    assert window.models_view.active_alias() == "large-v3"


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


# ---- Selection ↔ persistence ------------------------------------------------


def test_controller_persists_selection_back_to_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("small")

    assert ("whisper", "model", "small") in config.writes
    assert window.models_view.active_alias() == "small"


def test_controller_no_ops_when_selecting_already_active(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)
    config.writes.clear()

    window.models_view.model_selected.emit("large-v3")

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


# ---- Topbar sync ------------------------------------------------------------


def test_controller_syncs_topbar_model_on_init(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)

    assert "Large v3" in window.topbar._model_pill.text()


def test_controller_updates_topbar_on_model_select(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("tiny")

    assert "Tiny" in window.topbar._model_pill.text()


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
        self.model = "large-v3"
        self.language = "ru"
        self.datetime_str = "00:00:00"
        self.short_text = text[:50]


class FakeHistory:
    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.cleared = False

    def get_entries(self):
        return list(self._entries)

    def clear_history(self):
        self._entries.clear()
        self.cleared = True


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


def test_controller_clears_history_through_manager(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("a")])

    AppController(config=config, window=window, history=history)
    window.history_view.clear_requested.emit()

    assert history.cleared is True
    assert window.history_view._source_model.rowCount() == 0


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


# ---- Backend status poller wiring -------------------------------------------


def test_controller_drives_topbar_status_from_fetcher(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    AppController(
        config=config,
        window=window,
        backend_status_fetcher=lambda: "running",
    )

    # Polling is async (worker thread), so wait for the topbar to update.
    qtbot.waitUntil(
        lambda: window.topbar._status_pill.property("status") == "running",
        timeout=2000,
    )


def test_controller_without_backend_fetcher_leaves_status_unknown(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()

    AppController(config=config, window=window)

    assert window.topbar._status_pill.property("status") == "unknown"
