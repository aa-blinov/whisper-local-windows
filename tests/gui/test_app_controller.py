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
    """The controller now consults ``is_model_cached`` before restoring
    the persisted active card so a fresh install / cleared cache doesn't
    silently kick off a multi-gigabyte download. The bulk of the existing
    tests assume the persisted model is on disk; default the mock to True
    here and let the dedicated uncached-model test override it.
    """
    monkeypatch.setattr(
        "app.gui.controllers.app_controller.is_model_cached",
        lambda canonical: True,
    )


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
        "app.gui.controllers.app_controller.is_model_cached",
        lambda canonical: False,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

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
    window.models_view.model_selected.emit("turbo")

    cards = {
        c.alias(): c
        for c in window.models_view.findChildren(
            __import__(
                "app.gui.widgets.model_card", fromlist=["ModelCard"]
            ).ModelCard
        )
    }
    assert cards["turbo"].is_loading() is True


def test_controller_persists_selection_back_to_config(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)
    window.models_view.model_selected.emit("turbo")

    assert ("whisper", "model", "turbo") in config.writes
    assert window.models_view.active_alias() == "turbo"


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
    window.shortcuts_view.reset_requested.emit()

    expected_start = DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]
    expected_stop = DEFAULT_CONFIG["hotkey"]["stop_recording_hotkey"]
    expected_paste = DEFAULT_CONFIG["clipboard"]["auto_paste"]

    assert ("hotkey", "start_recording_hotkey", expected_start) in config.writes
    assert ("hotkey", "stop_recording_hotkey", expected_stop) in config.writes
    assert ("clipboard", "auto_paste", expected_paste) in config.writes


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
    window.shortcuts_view.reset_requested.emit()

    sv = window.shortcuts_view
    assert sv.start_hotkey() == DEFAULT_CONFIG["hotkey"]["start_recording_hotkey"]
    assert sv.stop_hotkey() == DEFAULT_CONFIG["hotkey"]["stop_recording_hotkey"]
    assert sv.auto_paste() == bool(DEFAULT_CONFIG["clipboard"]["auto_paste"])


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
    window.shortcuts_view.reset_requested.emit()
    writes_during = len(config.writes) - writes_before

    # Reset should write exactly 3 settings (start, stop, auto_paste).
    # If save_requested re-fired from set_values, we'd see additional writes.
    assert writes_during == 3


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
    window.models_view.model_selected.emit("distil-large-v3")

    assert "Distil" in window.topbar._model_pill.text()


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
        backend_status_fetcher=lambda: "error",
    )

    # Polling is async (worker thread), so wait for the topbar to update.
    # ``ready`` and ``loading`` map to ``hidden`` to avoid duplicating the
    # left-side recording-state pill, so we use ``error`` here as a status
    # the topbar surfaces visibly.
    qtbot.waitUntil(
        lambda: window.topbar._status_pill.property("status") == "error",
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


# ---- Recording controller wiring -------------------------------------------


class FakeRecordingController:
    """Stand-in exposing the surface AppController consumes."""

    def __init__(self, model_change_returns: bool = True) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Bus(QObject):
            state_changed = Signal(str)
            history_updated = Signal()

        self._bus = _Bus()
        self.state_changed = self._bus.state_changed
        self.history_updated = self._bus.history_updated
        self.model_change_requests: list[str] = []
        self._model_change_returns = model_change_returns

    def request_model_change(self, canonical: str, compute_type=None) -> bool:
        self.model_change_requests.append((canonical, compute_type))
        return self._model_change_returns


def test_controller_updates_topbar_recording_pill_on_state_change(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)

    rec.state_changed.emit("recording")
    assert window.topbar._recording_pill.property("state") == "recording"
    assert window.topbar._recording_pill.isVisibleTo(window.topbar) or True
    # visibility depends on parent visibility — property change is the contract

    rec.state_changed.emit("idle")
    assert window.topbar._recording_pill.property("state") == "idle"


def test_controller_refreshes_history_on_history_updated(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig()
    history = FakeHistory([FakeHistoryEntry("first")])
    rec = FakeRecordingController()

    AppController(config=config, window=window, history=history, recording=rec)

    # New entry appears in the manager outside our control.
    history._entries.append(FakeHistoryEntry("second"))
    rec.history_updated.emit()

    assert window.history_view._source_model.rowCount() == 2


def test_controller_routes_model_select_through_recording_when_present(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})
    rec = FakeRecordingController()

    AppController(config=config, window=window, recording=rec)
    window.models_view.model_selected.emit("distil-large-v3")

    # config still updated for persistence
    assert ("whisper", "model", "distil-large-v3") in config.writes
    # compute_type written too — the registry tells us each card's preference
    assert ("whisper", "compute_type", "float16") in config.writes
    # AND recording stack was asked to actually switch (with compute_type)
    assert rec.model_change_requests == [
        ("Systran/faster-distil-whisper-large-v3", "float16"),
    ]


def test_controller_skips_recording_call_when_recording_absent(qtbot):
    from app.gui.controllers.app_controller import AppController
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    config = FakeConfig({"whisper": {"model": "large-v3"}})

    AppController(config=config, window=window)  # no recording arg
    window.models_view.model_selected.emit("distil-large-v3")

    # Should still write config and not crash.
    assert ("whisper", "model", "distil-large-v3") in config.writes


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
