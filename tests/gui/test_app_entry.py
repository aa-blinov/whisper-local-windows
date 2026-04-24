"""Tests for the Qt application entry point helpers."""

from PySide6.QtWidgets import QApplication


def test_build_application_returns_app_and_main_window(qapp):
    from app.gui.app import build_application
    from app.gui.main_window import MainWindow

    app, window = build_application()
    assert isinstance(app, QApplication)
    assert isinstance(window, MainWindow)


def test_build_application_applies_theme(qapp):
    from app.gui.app import build_application
    from app.gui.theme import load_stylesheet

    app, _window = build_application()
    assert app.styleSheet() == load_stylesheet("dark")


def test_build_application_does_not_show_window(qapp):
    from app.gui.app import build_application

    _app, window = build_application()
    assert not window.isVisible()


def test_build_application_reuses_existing_qapplication(qapp):
    """Must not instantiate a second QApplication — Qt forbids it."""
    from app.gui.app import build_application

    app, _window = build_application()
    assert app is QApplication.instance()
