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
