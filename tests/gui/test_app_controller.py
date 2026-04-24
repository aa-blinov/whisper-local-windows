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
