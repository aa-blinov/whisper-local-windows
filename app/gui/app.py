"""Qt application entry point."""

from __future__ import annotations


# DLL-ordering workaround for a pyarrow ↔ Qt segfault on Windows.
#
# pyarrow's ``arrow.dll`` and Qt's runtime end up sharing some Windows-
# global state (CRT / OpenSSL / something else — the EventLog points at
# arrow.dll offset 0xbc5431 with exception 0xC0000005 every time). If
# Qt loads first and pyarrow comes later — through the NeMo backend
# pulling in lhotse → pyarrow on a worker thread — pyarrow segfaults
# the entire process during ``import pyarrow.arrow.dll``. Importing
# pyarrow FIRST puts arrow.dll into the loader's address space before
# Qt has a chance to claim conflicting slots, and the rest of the day
# is fine.
#
# Reproduced cleanly with ``scripts/diag_parakeet_with_qt.py`` (segfault
# inside ``import nemo.collections.asr``) vs ``diag_parakeet_qt_preimport``
# (ALL DONE). Wrapped in try/except so machines without pyarrow installed
# (a Whisper-only setup) still boot.
try:  # noqa: SIM105 — keep the explicit comment + import-time placement
    import pyarrow  # noqa: F401  (warmup-only, value unused)
except Exception as _pyarrow_exc:  # noqa: BLE001 — boot-time resilience
    # ImportError is the obvious case (pyarrow not installed in a
    # Whisper-only setup), but binary wheels can also raise
    # OSError / RuntimeError at import time when their DLL
    # dependencies are missing or shadowed by a conflicting load.
    # Letting any of those escape would crash the whole app at
    # import time, which is exactly the failure mode this pre-
    # import is supposed to prevent — fall through to stderr and
    # let the rest of the app boot, NeMo will surface a real error
    # later if it actually needed pyarrow.
    import sys as _sys
    print(
        f"[lazy-to-text] pyarrow pre-import skipped: "
        f"{type(_pyarrow_exc).__name__}: {_pyarrow_exc}",
        file=_sys.stderr,
    )
    del _pyarrow_exc, _sys


import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.gui.controllers.app_controller import AppController
from app.gui.log_bridge import QtLogBridge
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme
from app.utils import is_cached_for_info, is_model_cached, resolve_asset_path


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


def _register_aumid_icon(app_id: str = "LazyToText.App") -> None:
    """Bind a real icon to our AppUserModelID in the user's registry.

    Without an ``HKCU\\Software\\Classes\\AppUserModelId\\<id>`` entry
    pointing at the actual executable, Windows falls back to a generic
    document icon for taskbar / Alt-Tab entries grouped under our
    AppUserModelID — even though the .exe itself carries the proper
    icon resource and ``setWindowIcon`` was called on the Qt window.
    Confirmed empirically on a fresh ``%LOCALAPPDATA%\\Programs\\...``
    install: blank icon stayed blank across explorer restarts and
    icon-cache flushes until this registry entry was written.

    Frozen-only: source-run dev path uses python.exe as the host
    process and shouldn't fight Windows for that icon binding.
    Idempotent — overwrites stale values cheerfully on every launch
    so a moved install picks up the new path next time. Failures
    (locked HKCU under Group Policy, e.g.) are swallowed; falling
    back to a generic icon is annoying but not fatal.
    """
    if sys.platform != "win32":
        return
    if not getattr(sys, "frozen", False):
        return
    try:
        import winreg
    except ImportError:  # pragma: no cover — winreg is std-lib on Windows
        return
    exe_path = sys.executable
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            f"Software\\Classes\\AppUserModelId\\{app_id}",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Lazy to Text")
            winreg.SetValueEx(
                key, "IconResource", 0, winreg.REG_EXPAND_SZ, f"{exe_path},0",
            )
            winreg.SetValueEx(
                key, "IconUri", 0, winreg.REG_EXPAND_SZ, exe_path,
            )
    except OSError:
        # Locked HKCU / Group Policy — fall through with the generic
        # icon. Not worth crashing the app over.
        return


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
        # NB: ``main()`` is responsible for setting the root level + file
        # handler before this function runs so the recording stack's INFO
        # messages aren't lost. Here we just attach the UI bridge.
        bridge = QtLogBridge(parent=window)
        bridge.record_received.connect(window.logs_view.append_record)
        bridge.install()
    if config is not None:
        AppController(
            config=config,
            window=window,
            history=history,
            recording=recording,
            tray=tray,
        )
        if recording is not None:
            recording.setParent(window)
            recording.start()
    if tray is not None:
        tray.setVisible(True)
    return app, window


def _apply_hf_token(configured: Optional[str]) -> bool:
    """Mirror the user's HF token into the live process environment.

    huggingface_hub and pyannote both read ``HF_TOKEN`` (alongside
    ``HUGGING_FACE_HUB_TOKEN`` as a legacy alias). Setting just one
    is enough — huggingface_hub treats them as equivalent.

    Returns True iff a non-empty token was applied.
    """
    if configured and str(configured).strip():
        token = str(configured).strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return True
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return False


def _apply_storage_path(configured: Optional[str]) -> str:
    """Resolve and apply the user's chosen models directory to env vars.

    ``HF_HOME`` is always set (faster-whisper / huggingface_hub
    ignores the system-wide ``~/.cache/huggingface`` only when this is
    set). ``GIGAAM_MODELS_DIR`` is set ONLY when the user has
    explicitly picked a custom path — leaving it unset keeps GigaAM
    on its library default ``~/.cache/gigaam`` so existing installs
    don't have their already-downloaded ckpt files orphaned by the
    upgrade. Returns the resolved hub root for logging.

    Must run before any ``huggingface_hub`` or ``gigaam`` import: HF
    reads ``HF_HOME`` once at module load, GigaAM doesn't but its
    ``download_root`` is read per-call so the env var has to be in
    place by the time ``GigaamBackend.load`` runs.
    """
    from app.utils import get_models_root

    root = get_models_root(configured)
    os.environ["HF_HOME"] = root
    is_custom = bool(configured and str(configured).strip())
    if is_custom:
        os.environ["GIGAAM_MODELS_DIR"] = str(Path(root) / "gigaam")
    else:
        os.environ.pop("GIGAAM_MODELS_DIR", None)
    return root


def _autoload_persisted_model(backend) -> None:
    """Kick off the persisted model load if its weights are already on disk.

    Called once at startup, after the recording stack is built but
    before the main window is shown.  Returns immediately — the
    backend's ``load()`` spawns a daemon thread internally and the
    UI paints the loading state from the model card / topbar polling
    machinery.

    Behaviour:

    - ``backend is None``        → no-op (early shutdown / tests).
    - Model in registry, cached  → log + ``backend.load()``.
    - Model in registry, NOT     → log "skipping auto-load"; user has
      cached                       to click Download deliberately so
                                   they see the progress bar (a silent
                                   1.5 GB transfer would feel like a
                                   hang).
    - Unknown model id (raw HF   → fall back to the lenient HF cache
      path the user pasted)        check; load if anything's on disk.
    """
    import logging

    log = logging.getLogger(__name__)
    if backend is None:
        return

    from app.model_mapping import alias_for, get_model

    canonical = backend.current_model()
    try:
        info = get_model(alias_for(canonical))
    except KeyError:
        info = None
        cached = is_model_cached(canonical)
    else:
        cached = is_cached_for_info(info)

    if not cached:
        log.info(
            "Persisted model %s is not cached — skipping auto-load. "
            "Waiting for the user to pick a model.",
            canonical,
        )
        return

    display = info.display_name if info is not None else canonical
    log.info(
        "Persisted model %s is cached — kicking off background load.",
        display,
    )
    backend.load()


def main() -> int:
    import logging

    # Read the configured ``storage.models_dir`` (may be empty for
    # 'use the default') from config.yaml, then plant ``HF_HOME`` and
    # ``GIGAAM_MODELS_DIR`` BEFORE the libraries that need them get
    # imported. ConfigManager itself doesn't pull in HF/torch so we
    # can safely import it first.
    from app.config_manager import ConfigManager

    _early_config = ConfigManager()
    storage_root = _apply_storage_path(
        _early_config.get_setting("storage", "models_dir")
    )
    _apply_hf_token(_early_config.get_setting("huggingface", "token"))

    from app.gui.controllers.recording_controller import RecordingController
    from app.gui.recording_factory import build_recording_stack
    from app.gui.widgets.tray_icon import AppTrayIcon
    from app.instance_manager import try_acquire_single_instance
    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    # Distinct AppUserModelID before any window is created so Windows uses
    # our icon in the taskbar instead of the Python interpreter's.
    _set_app_user_model_id()
    # Bind the AppUserModelID to our exe's icon resource in the user's
    # registry — without this Windows shows a generic document icon
    # for taskbar entries grouped under this AUMID on fresh installs.
    _register_aumid_icon()

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

    # Set up the logging pipeline BEFORE building the recording stack so the
    # HotkeyListener / model-load messages from build_recording_stack reach
    # both the UI Logs view and logs/app.log. Without this, INFO records
    # emitted during stack construction are dropped by the default WARNING
    # root level and we lose the most useful diagnostic moment.
    import logging as _logging

    _logging.getLogger().setLevel(_logging.INFO)
    from app.utils import get_project_logs_path

    _log_path = os.path.join(get_project_logs_path(), "app.log")
    _file_handler = _logging.FileHandler(_log_path, encoding="utf-8")
    _file_handler.setLevel(_logging.INFO)
    _file_handler.setFormatter(
        _logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    _logging.getLogger().addHandler(_file_handler)

    # Bridge ``warnings.warn(...)`` into the logging pipeline so
    # NeMo / PyTorch / pyannote deprecation noise (and our own
    # ``DeprecationWarning`` etc.) lands in the same place as
    # everything else — both ``app.log`` and the Logs view.
    # Without this, those warnings only print to stderr and
    # disappear in a windowed build with no console.
    _logging.captureWarnings(True)

    # Reuse the early config — re-creating it would re-read the YAML
    # and just produce identical state, but the early one was made
    # before the logging file handler was attached, so log messages
    # from the load path went to stderr only. That's fine; we don't
    # need them in app.log.
    config = _early_config
    logging.getLogger(__name__).info(
        "Models root: %s (configured=%r)",
        storage_root,
        _early_config.get_setting("storage", "models_dir"),
    )

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

    _autoload_persisted_model(backend)

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
        recording=recording_controller,
        tray=tray,
        install_logs=True,
    )

    # Live CPU / RAM / GPU stats in the topbar — polls every 2 s and
    # pushes numbers straight to the widget via signal.
    from app.resource_monitor import ResourceMonitor

    resource_monitor = ResourceMonitor(parent=window)
    resource_monitor.metrics_updated.connect(window.topbar.set_resource_metrics)
    resource_monitor.start()

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
        try:
            resource_monitor.stop()
        except Exception:  # pragma: no cover — defensive
            pass
        if recording_controller is not None:
            recording_controller.shutdown()
        if backend is not None:
            backend.shutdown()
        if tray is not None:
            tray.setVisible(False)


if __name__ == "__main__":
    raise SystemExit(main())
