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


def test_main_window_remaining_keys_still_use_placeholder(qtbot):
    from app.gui.main_window import MainWindow
    from app.gui.views.placeholder import PlaceholderView

    window = MainWindow()
    qtbot.addWidget(window)
    for key in ("shortcuts", "history"):
        assert isinstance(window.get_view(key), PlaceholderView)
