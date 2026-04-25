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
    from app.docker_backend_manager import DockerBackendManager
    from app.gui.controllers.recording_controller import RecordingController
    from app.gui.recording_factory import build_recording_stack
    from app.gui.widgets.tray_icon import AppTrayIcon
    from app.instance_manager import try_acquire_single_instance
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    # Distinct AppUserModelID before any window is created so Windows uses
    # our icon in the taskbar instead of the Python interpreter's.
    _set_app_user_model_id()

    # Single-instance check happens BEFORE we create QApplication. Showing
    # an early Qt dialog without a fully-initialised event loop crashes with
    # an access violation on Windows; instead, use the native Win32
    # MessageBoxW which is a synchronous OS-level dialog that doesn't need
    # any Qt machinery.
    instance_handle = try_acquire_single_instance("LazyToTextQt")
    if instance_handle is None:
        try:
            import ctypes
            MB_ICONWARNING = 0x00000030
            MB_OK = 0x00000000
            MB_TOPMOST = 0x00040000
            ctypes.windll.user32.MessageBoxW(
                0,
                "Another copy of Lazy to Text is already running.\n\n"
                "Use its system tray icon to bring it back, or quit it first.",
                "Lazy to Text",
                MB_ICONWARNING | MB_OK | MB_TOPMOST,
            )
        except Exception:
            # Fall back to stderr if even the native dialog fails.
            print(
                "Lazy to Text: another instance is already running.",
                file=sys.stderr,
            )
        return 0

    # We are the primary instance — bring up the QApplication. The mutex
    # handle must outlive this function or another launch could race in.
    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app._instance_mutex = instance_handle  # type: ignore[attr-defined]

    # Set the icon on the QApplication immediately, before any window or
    # secondary widget is created. Late setWindowIcon doesn't refresh the
    # taskbar reliably on Windows.
    _early_icon = _load_app_icon()
    if not _early_icon.isNull():
        qt_app.setWindowIcon(_early_icon)

    config = ConfigManager()
    docker = DockerBackendManager()

    state_manager = None
    hotkey_listener = None
    recording_controller = None
    try:
        state_manager, hotkey_listener = build_recording_stack(
            config_manager=config, docker_backend_manager=docker,
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

    history = state_manager.history_manager if state_manager is not None else None

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
        backend_status_fetcher=docker.status,
        recording=recording_controller,
        tray=tray,
        install_logs=True,
    )
    window.show()

    try:
        return app.exec()
    finally:
        if recording_controller is not None:
            recording_controller.shutdown()
        if tray is not None:
            tray.setVisible(False)


if __name__ == "__main__":
    raise SystemExit(main())
