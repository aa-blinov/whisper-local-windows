"""Tests for the MainWindow shell."""

from PySide6.QtWidgets import QStackedWidget


def test_main_window_instantiates(qtbot):
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "Lazy to Text"


def test_main_window_has_sidebar(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.widgets.sidebar import Sidebar

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.sidebar, Sidebar)


def test_main_window_has_stack_with_view_per_nav_item(qtbot):
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.stack, QStackedWidget)
    assert window.stack.count() == len(window.sidebar.items())


def test_main_window_default_view_matches_default_sidebar_key(qtbot):
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    default_key = window.sidebar.active_key()
    assert window.stack.currentWidget() is window.get_view(default_key)


def test_main_window_switches_view_when_sidebar_changes(qtbot):
    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    window.sidebar.set_active("history")
    assert window.stack.currentWidget() is window.get_view("history")

    window.sidebar.set_active("logs")
    assert window.stack.currentWidget() is window.get_view("logs")


def test_main_window_get_view_raises_on_unknown_key(qtbot):
    import pytest

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    with pytest.raises(KeyError):
        window.get_view("nope")


def test_main_window_uses_models_view_for_models_key(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.views.models_view import ModelsView

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.get_view("models"), ModelsView)
    assert window.models_view is window.get_view("models")


def test_main_window_uses_logs_view_for_logs_key(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.views.logs_view import LogsView

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.get_view("logs"), LogsView)
    assert window.logs_view is window.get_view("logs")


def test_main_window_uses_shortcuts_view_for_shortcuts_key(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.views.shortcuts_view import ShortcutsView

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.get_view("shortcuts"), ShortcutsView)
    assert window.shortcuts_view is window.get_view("shortcuts")


def test_main_window_has_topbar(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.widgets.topbar import TopBar

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.topbar, TopBar)
    assert window.topbar.parent() is not None


def test_main_window_uses_history_view_for_history_key(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.views.history_view import HistoryView

    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.get_view("history"), HistoryView)
    assert window.history_view is window.get_view("history")


def test_main_window_no_placeholders_remain(qtbot):
    """All nav keys now map to real views."""
    from app.gui.main_window import MainWindow
    from app.gui.views.placeholder import PlaceholderView

    window = MainWindow()
    qtbot.addWidget(window)
    for key in window.sidebar.items():
        assert not isinstance(window.get_view(key), PlaceholderView), key


# ---- Close-to-tray behaviour -----------------------------------------------


def test_close_event_closes_normally_by_default(qtbot):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QCloseEvent

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_close_event_hides_when_close_to_tray_enabled(qtbot):
    from PySide6.QtGui import QCloseEvent

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.set_close_to_tray(True)

    event = QCloseEvent()
    window.closeEvent(event)

    assert not event.isAccepted()  # close was vetoed
    assert not window.isVisible()  # but the window was hidden


def test_close_event_emits_hidden_to_tray_signal(qtbot):
    from PySide6.QtGui import QCloseEvent

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.set_close_to_tray(True)

    with qtbot.waitSignal(window.hidden_to_tray, timeout=1000):
        window.closeEvent(QCloseEvent())


def test_request_quit_overrides_close_to_tray(qtbot):
    from PySide6.QtGui import QCloseEvent

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.set_close_to_tray(True)

    # Mark the window as quitting — the next close should be honoured.
    window.request_quit()

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()


def test_set_close_to_tray_can_be_disabled(qtbot):
    from PySide6.QtGui import QCloseEvent

    from app.gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    window.set_close_to_tray(True)
    window.set_close_to_tray(False)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
