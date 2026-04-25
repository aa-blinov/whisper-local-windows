"""Qt application entry point."""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.gui.controllers.app_controller import AppController
from app.gui.log_bridge import QtLogBridge
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme
from app.utils import resolve_asset_path


def _load_app_icon() -> QIcon:
    """Build a QIcon that includes both the multi-size .ico and the .png so
    Windows can pick the right resolution for the title bar, taskbar, and
    Alt-Tab switcher."""
    icon = QIcon()
    for asset in ("assets/tray_idle.ico", "assets/tray_idle.png"):
        path = resolve_asset_path(asset)
        if path and os.path.isfile(path):
            icon.addFile(path)
    return icon


def _force_window_icon(hwnd: int, ico_path: str) -> bool:
    """Bypass Qt and tell Windows directly which icon to use for this hWnd.

    Qt's setWindowIcon often fails to translate into a real WM_SETICON, so
    the taskbar / Alt-Tab keep the python.exe icon. We load the .ico via
    LoadImageW and push it through both WM_SETICON (per-window) AND
    SetClassLongPtr (per-window-class) so taskbar, Alt-Tab, and the title
    bar all pick up the same icon.
    """
    if sys.platform != "win32" or not hwnd or not ico_path:
        return False
    try:
        import ctypes
        from ctypes import c_void_p, c_wchar_p

        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        user32 = ctypes.windll.user32

        # Set up types only for return values that cross 32/64 bits — keep
        # parameters as Python ints to avoid sign-extension surprises.
        user32.LoadImageW.restype = c_void_p
        user32.SendMessageW.restype = c_void_p
        user32.SetClassLongPtrW.restype = c_void_p

        def load(size: int) -> int:
            return user32.LoadImageW(
                None,
                c_wchar_p(ico_path),
                IMAGE_ICON,
                size,
                size,
                LR_LOADFROMFILE,
            ) or 0

        h_small = load(16)
        h_big = load(32)
        if not (h_small or h_big):
            return False

        if h_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_small)
            user32.SetClassLongPtrW(hwnd, GCLP_HICONSM, h_small)
        if h_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_big)
            user32.SetClassLongPtrW(hwnd, GCLP_HICON, h_big)
        return True
    except Exception:
        return False


def _set_app_user_model_id(app_id: str = "LazyToText.App") -> int:
    """Tell Windows this process is its own app, not a hosted Python script.

    Without this, the taskbar / Alt-Tab / system tray group everything under
    Python's default AppUserModelID and use the Python interpreter's icon
    instead of the one we set via setWindowIcon. This must run before any
    window or QApplication is created.

    Returns the HRESULT from the Win32 call (0 on success, non-zero on
    failure), or -1 on platforms / pythons where the API is unavailable.
    """
    if sys.platform != "win32":
        return -1
    try:
        import ctypes

        hr = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return int(hr) if hr is not None else 0
    except Exception:
        # Old Windows / missing API — non-fatal, the taskbar just stays
        # grouped under Python.
        return -1


def build_application(
    argv: Optional[List[str]] = None,
    theme: str = "dark",
    config: Optional[Any] = None,
    history: Optional[Any] = None,
    backend_status_fetcher: Optional[Any] = None,
    recording: Optional[Any] = None,
    tray: Optional[Any] = None,
    install_logs: bool = False,
) -> Tuple[QApplication, MainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    # When a tray icon is present, the window can be hidden indefinitely; we
    # need to keep the app alive even when no top-level window is visible.
    # Without a tray the default behaviour (quit on last closed) is correct.
    if tray is not None:
        app.setQuitOnLastWindowClosed(False)
    icon = _load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    apply_theme(app, theme)

    window = MainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)
    if install_logs:
        bridge = QtLogBridge(parent=window)
        bridge.line_received.connect(window.logs_view.append_line)
        bridge.install()
    if config is not None:
        AppController(
            config=config,
            window=window,
            history=history,
            backend_status_fetcher=backend_status_fetcher,
            recording=recording,
            tray=tray,
        )
        if recording is not None:
            recording.setParent(window)
            recording.start()
    if tray is not None:
        tray.setVisible(True)
    return app, window


def main() -> int:
    import logging

    from app.config_manager import ConfigManager
    from app.gui.controllers.recording_controller import RecordingController
    from app.gui.recording_factory import build_recording_stack
    from app.gui.widgets.tray_icon import AppTrayIcon
    from app.instance_manager import try_acquire_single_instance
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    # Distinct AppUserModelID before any window is created so Windows uses
    # our icon in the taskbar instead of the Python interpreter's.
    _set_app_user_model_id()

    instance_handle = try_acquire_single_instance("LazyToTextQt")

    # Bring up QApplication regardless of branch — both the primary path
    # and the duplicate-warning dialog need our app icon to show in
    # taskbar / Alt-Tab instead of python.exe's snake.
    from PySide6.QtWidgets import QMessageBox

    qt_app = QApplication.instance() or QApplication(sys.argv)
    _early_icon = _load_app_icon()
    if not _early_icon.isNull():
        qt_app.setWindowIcon(_early_icon)

    if instance_handle is None:
        # Use an explicit QMessageBox instance + exec() rather than the
        # static QMessageBox.warning(None, ...) — the latter crashed with
        # an access violation when invoked early in the process lifetime.
        try:
            msg = QMessageBox()
            if not _early_icon.isNull():
                msg.setWindowIcon(_early_icon)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Lazy to Text")
            msg.setText(
                "Another copy of Lazy to Text is already running.\n\n"
                "Use its system tray icon to bring it back, or quit it first."
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.show()
            ico_path = resolve_asset_path("assets/tray_idle.ico")
            if ico_path and os.path.isfile(ico_path):
                _force_window_icon(int(msg.winId()), ico_path)
            msg.exec()
        except Exception:
            print(
                "Lazy to Text: another instance is already running.",
                file=sys.stderr,
            )
        return 0

    # We are the primary instance — bind the mutex handle so it survives.
    qt_app._instance_mutex = instance_handle  # type: ignore[attr-defined]

    config = ConfigManager()

    state_manager = None
    hotkey_listener = None
    backend = None
    recording_controller = None
    try:
        state_manager, hotkey_listener, backend = build_recording_stack(
            config_manager=config,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Recording stack failed to initialise — UI will run without "
            "hotkeys/transcription: %s", exc,
        )

    if state_manager is not None:
        recording_controller = RecordingController(
            state_manager=state_manager,
            hotkey_listener=hotkey_listener,
        )

    if backend is not None:
        # Kick off background model load right away so the UI status pill
        # transitions from "Loading model…" to "Model ready" without waiting
        # for the user's first keypress.
        backend.load()

    history = state_manager.history_manager if state_manager is not None else None
    backend_status_fetcher = backend.status if backend is not None else None

    # qt_app already exists from the single-instance gate above.
    tray: Optional[AppTrayIcon] = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = AppTrayIcon(parent=qt_app)
    else:
        logging.getLogger(__name__).warning(
            "System tray not available — close button will quit the app."
        )

    app, window = build_application(
        config=config,
        history=history,
        backend_status_fetcher=backend_status_fetcher,
        recording=recording_controller,
        tray=tray,
        install_logs=True,
    )
    window.show()

    # Qt's setWindowIcon doesn't reliably translate into Win32 WM_SETICON,
    # which means the taskbar and Alt-Tab fall back to python.exe's icon.
    # Push the icon explicitly via SendMessage(WM_SETICON) once the window
    # has its native handle.
    ico_path = resolve_asset_path("assets/tray_idle.ico")
    if ico_path and os.path.isfile(ico_path):
        _force_window_icon(int(window.winId()), ico_path)

    try:
        return app.exec()
    finally:
        if recording_controller is not None:
            recording_controller.shutdown()
        if backend is not None:
            backend.shutdown()
        if tray is not None:
            tray.setVisible(False)


if __name__ == "__main__":
    raise SystemExit(main())
