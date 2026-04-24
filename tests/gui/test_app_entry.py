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


def test_build_application_does_not_wire_controller_when_config_absent(qapp):
    from app.gui.app import build_application
    from app.gui.controllers.app_controller import AppController

    _app, window = build_application()
    assert window.findChildren(AppController) == []


def test_build_application_wires_controller_when_config_provided(qapp):
    from app.gui.app import build_application
    from app.gui.controllers.app_controller import AppController

    class StubConfig:
        def __init__(self):
            self._data = {"whisper": {"model": "large-v3"}}

        def get_setting(self, section, key):
            return self._data.get(section, {}).get(key)

        def update_user_setting(self, section, key, value):
            self._data.setdefault(section, {})[key] = value

    _app, window = build_application(config=StubConfig())
    controllers = window.findChildren(AppController)
    assert len(controllers) == 1
    assert window.models_view.active_alias() == "large-v3"


def test_build_application_does_not_install_log_bridge_by_default(qapp):
    from app.gui.app import build_application
    from app.gui.log_bridge import QtLogBridge

    _app, window = build_application()
    assert window.findChildren(QtLogBridge) == []


def test_build_application_installs_log_bridge_when_requested(qapp, qtbot):
    import logging

    from app.gui.app import build_application
    from app.gui.log_bridge import QtLogBridge

    _app, window = build_application(install_logs=True)
    bridges = window.findChildren(QtLogBridge)
    assert len(bridges) == 1

    bridge = bridges[0]
    try:
        logger = logging.getLogger("test.entry.logs")
        logger.setLevel(logging.DEBUG)
        with qtbot.waitSignal(bridge.line_received, timeout=1000) as blocker:
            logger.warning("bridge works")
        assert "bridge works" in blocker.args[0]
    finally:
        bridge.uninstall()
