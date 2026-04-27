"""Tests for the Qt application entry point helpers."""

import pytest
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


def test_build_application_wires_controller_when_config_provided(qapp, monkeypatch):
    from app.gui.app import build_application
    from app.gui.controllers.app_controller import AppController

    class StubConfig:
        def __init__(self):
            self._data = {"whisper": {"model": "whisper-large-v3"}}

        def get_setting(self, section, key):
            return self._data.get(section, {}).get(key)

        def update_user_setting(self, section, key, value):
            self._data.setdefault(section, {})[key] = value

    # Pretend the configured model is cached so the controller restores
    # it as the active card on startup.  The cache check itself is
    # tested separately in tests/test_utils.py.
    import app.gui.controllers.app_controller as controller_module

    monkeypatch.setattr(
        controller_module, "is_cached_for_info", lambda info: True
    )

    _app, window = build_application(config=StubConfig())
    controllers = window.findChildren(AppController)
    assert len(controllers) == 1
    assert window.models_view.active_alias() == "whisper-large-v3"


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


def test_build_application_sets_window_icon(qapp):
    from app.gui.app import build_application

    app, window = build_application()
    # The QApplication-level icon propagates to all windows on Windows so the
    # title bar / taskbar / Alt-Tab show the app's branding.
    assert not app.windowIcon().isNull()
    # MainWindow inherits the app icon by default.
    assert not window.windowIcon().isNull()


# ---- Persisted-model auto-load --------------------------------------------


class _FakeBackend:
    """Minimal backend stub: records load() and current_model()."""

    def __init__(self, model: str) -> None:
        self._model = model
        self.load_called = 0

    def current_model(self) -> str:
        return self._model

    def load(self) -> None:
        self.load_called += 1


def test_autoload_kicks_off_load_when_model_is_cached(monkeypatch):
    """When the persisted model is in the registry AND its weights are
    already on disk, the helper must call ``backend.load()`` so the
    backend transitions ``stopped → loading → ready`` in the
    background — otherwise the topbar shows the model name but the
    hotkey listener rejects every press with 'Model is not ready yet'."""
    import app.gui.app as app_module

    backend = _FakeBackend("whisper-large-v3-turbo")
    monkeypatch.setattr(app_module, "is_cached_for_info", lambda info: True)

    app_module._autoload_persisted_model(backend)
    assert backend.load_called == 1


def test_autoload_skips_load_when_model_not_cached(monkeypatch):
    """If the configured model isn't downloaded yet, don't auto-load —
    force the user to click Download deliberately so they see the
    progress bar and aren't surprised by a 1.5 GB silent transfer."""
    import app.gui.app as app_module

    backend = _FakeBackend("whisper-large-v3-turbo")
    monkeypatch.setattr(app_module, "is_cached_for_info", lambda info: False)

    app_module._autoload_persisted_model(backend)
    assert backend.load_called == 0


def test_autoload_falls_back_to_canonical_check_for_unknown_model(monkeypatch):
    """If the configured model isn't in the registry (legacy entry,
    user-pasted HF id), the helper still tries the lenient HF cache
    check and loads if anything is on disk."""
    import app.gui.app as app_module

    backend = _FakeBackend("some/unknown-model")
    monkeypatch.setattr(app_module, "is_model_cached", lambda canonical: True)
    # is_cached_for_info shouldn't even be reached for an unknown model.
    monkeypatch.setattr(
        app_module,
        "is_cached_for_info",
        lambda info: pytest.fail("should not call is_cached_for_info for unknown id"),
    )

    app_module._autoload_persisted_model(backend)
    assert backend.load_called == 1


def test_autoload_no_op_when_backend_is_none():
    """Sanity: helper must accept None backend without raising — main()
    can be called with backend=None during early shutdown / tests."""
    import app.gui.app as app_module

    # Must not raise.
    app_module._autoload_persisted_model(None)


def test_app_module_no_longer_imports_splash():
    """The splash module is gone (ONNX backends load in seconds, no
    GIL-blocking import to hide).  Make sure nothing in app.py
    still references it."""
    import inspect

    import app.gui.app as app_module

    src = inspect.getsource(app_module)
    assert "splash" not in src.lower(), (
        "app.py should not reference the splash anymore"
    )


# ---- AUMID icon registry registration --------------------------------------


def test_register_aumid_icon_writes_hkcu_entry_when_frozen(monkeypatch, tmp_path):
    """Without an HKCU\\AppUserModelId\\<id> entry pointing at our exe,
    Windows shows a generic document icon for taskbar entries grouped
    under our AppUserModelID — confirmed empirically on a fresh
    install. Registering at startup is a no-op on the source-run dev
    path (which uses python.exe as host) but mandatory on frozen
    builds."""
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    fake_exe = str(tmp_path / "LazyToText.exe")
    monkeypatch.setattr(sys, "executable", fake_exe, raising=False)

    captured: dict = {}

    class FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_create_key(_root, subkey, _reserved, _access):
        captured["subkey"] = subkey
        return FakeKey()

    def fake_set_value(_key, name, _reserved, _typ, value):
        captured.setdefault("values", {})[name] = (value, _typ)

    fake_winreg = type(sys)("winreg")
    fake_winreg.HKEY_CURRENT_USER = 0x80000001
    fake_winreg.KEY_SET_VALUE = 0x0002
    fake_winreg.REG_SZ = 1
    fake_winreg.REG_EXPAND_SZ = 2
    fake_winreg.CreateKeyEx = fake_create_key
    fake_winreg.SetValueEx = fake_set_value
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

    from app.gui.app import _register_aumid_icon

    _register_aumid_icon("LazyToText.App")

    assert captured.get("subkey") == "Software\\Classes\\AppUserModelId\\LazyToText.App"
    values = captured.get("values", {})
    assert values["DisplayName"][0] == "Lazy to Text"
    assert values["IconResource"][0] == f"{fake_exe},0"
    assert values["IconUri"][0] == fake_exe


def test_register_aumid_icon_noop_when_not_frozen(monkeypatch):
    """Source-run path uses python.exe as the host process — pointing
    Windows at python.exe's icon resource would be hostile. Function
    must short-circuit before touching the registry."""
    import sys

    monkeypatch.setattr(sys, "frozen", False, raising=False)

    fake_winreg = type(sys)("winreg")
    fake_winreg.HKEY_CURRENT_USER = 0x80000001
    fake_winreg.KEY_SET_VALUE = 0x0002
    fake_winreg.REG_SZ = 1
    fake_winreg.REG_EXPAND_SZ = 2

    touched: list = []
    fake_winreg.CreateKeyEx = lambda *_args, **_kw: touched.append("create")
    fake_winreg.SetValueEx = lambda *_args, **_kw: touched.append("set")
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

    from app.gui.app import _register_aumid_icon

    _register_aumid_icon("LazyToText.App")
    assert touched == [], (
        "should not write registry on source-run dev path"
    )


def test_register_aumid_icon_swallows_oserror(monkeypatch, tmp_path):
    """Registry writes can fail under restrictive group policies or
    locked-down enterprise installs. Don't bring the app down — the
    visual fallback is just a generic icon, not catastrophic."""
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "x.exe"), raising=False)

    fake_winreg = type(sys)("winreg")
    fake_winreg.HKEY_CURRENT_USER = 0x80000001
    fake_winreg.KEY_SET_VALUE = 0x0002
    fake_winreg.REG_SZ = 1
    fake_winreg.REG_EXPAND_SZ = 2

    def boom(*_args, **_kw):
        raise OSError("ERROR_ACCESS_DENIED")
    fake_winreg.CreateKeyEx = boom
    fake_winreg.SetValueEx = boom
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg)

    from app.gui.app import _register_aumid_icon

    # Must not raise — defensive against Group Policy / locked HKCU.
    _register_aumid_icon("LazyToText.App")
