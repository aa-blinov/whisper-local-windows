"""Qt application entry point."""

from __future__ import annotations

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

    Sets ``HF_HOME`` so ``huggingface_hub`` (used by ``onnx-asr``)
    downloads weights into our managed root instead of the system-
    wide ``~/.cache/huggingface``.  Returns the resolved hub root
    for logging.

    Must run before any ``huggingface_hub`` import: HF reads
    ``HF_HOME`` once at module load.
    """
    from app.utils import get_models_root

    root = get_models_root(configured)
    os.environ["HF_HOME"] = root
    # Suppress the per-download warning about symlinks not being
    # available on Windows.  Symlinks require either admin rights or
    # Developer Mode to be enabled; neither is realistic for a
    # consumer dictation app.  The HF cache works fine without them
    # (just uses more disk for duplicated files), so the warning is
    # noise that clutters our Logs view.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return root


# Minimal valid ONNX model: float32[1] → Identity → float32[1].
# Generated once with raw protobuf wire encoding; verified with
# ``onnxruntime.InferenceSession(bytes, ...).run(...)``.
# Used as a dummy graph to force OnnxRuntime to initialise its
# execution-provider DLLs (DirectML, CUDA, CPU) at app startup so the
# first real model load doesn't acquire the Win32 DLL loader-lock on
# the main thread and freeze the window.
_WARMUP_ONNX_BYTES: bytes = (
    b'\x08\x08:4\n\x10\n\x01x\x12\x01y"\x08Identity'
    b'Z\x0f\n\x01x\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'b\x0f\n\x01y\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'B\x04\n\x00\x10\x0b'
)


def _do_onnx_asr_preimport() -> None:
    """Import onnx_asr and warm up OnnxRuntime provider DLLs.

    Split out from ``_preload_onnx_asr_async`` so tests can monkeypatch
    just this call without faking ``threading.Thread``.  Allowed to
    raise — the surrounding wrapper in ``_preload_onnx_asr_async``
    catches everything and logs at DEBUG level.

    Why the warmup session matters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``import onnx_asr`` loads the Python module but does *not* create
    any ONNX InferenceSession.  The first ``InferenceSession(...)`` call
    triggers OnnxRuntime to ``LoadLibrary`` its execution-provider DLLs
    (``DirectML.dll``, ``onnxruntime_providers_shared.dll``, CUDA libs,
    …).  Windows serialises all ``LoadLibrary`` calls through the
    process-wide DLL loader-lock.  While a worker thread holds that
    lock, the Qt main thread — which uses Win32 APIs for window
    management and rendering — cannot move the window, paint, or process
    any system messages.  The user sees "(Not responding)" and a frozen
    title bar for the entire duration of that first session creation.

    Creating a trivial 62-byte dummy session here, on a daemon thread
    *before* ``window.show()``, absorbs the lock contention in the
    background.  All subsequent ``InferenceSession`` calls (real model
    loads) re-use already-loaded DLLs; ``LoadLibrary`` for a resident
    DLL increments a ref-count in <1 ms and never parks the caller on
    the loader-lock.
    """
    import onnx_asr  # noqa: F401 — pulls in onnxruntime

    # Create throwaway InferenceSession(s) to force ORT provider DLLs
    # to load now.  We iterate over every available provider so that
    # CUDA (cublas/cuDNN), DirectML, TensorRT, etc. are each
    # initialised on this daemon thread — not on the Qt main thread
    # where they would hold the Win32 DLL loader-lock and freeze the
    # window.
    try:
        import onnxruntime as _ort
        import numpy as _np

        _opts = _ort.SessionOptions()
        _opts.log_severity_level = 4  # silence ORT — errors only
        _x = _np.array([0.0], dtype=_np.float32)

        for _provider in _ort.get_available_providers():
            # Always include CPU as the fallback so the session has a
            # usable provider even if the primary one is unavailable.
            # Avoid duplicating CPUExecutionProvider (ORT warns on that).
            _providers = (
                [_provider]
                if _provider == "CPUExecutionProvider"
                else [_provider, "CPUExecutionProvider"]
            )
            try:
                _sess = _ort.InferenceSession(
                    _WARMUP_ONNX_BYTES,
                    sess_options=_opts,
                    providers=_providers,
                )
                _sess.run(None, {"x": _x})
                del _sess
            except Exception:
                pass  # provider DLL missing / not supported — skip
    except Exception:
        pass  # warmup is best-effort; real error path is inside _do_load


def _preload_onnx_asr_async() -> "threading.Thread":
    """Kick off ``import onnx_asr`` + ORT provider warmup on a daemon thread.

    Why this exists: on Windows, the first OnnxRuntime ``InferenceSession``
    creation acquires the Win32 DLL loader-lock while loading provider
    DLLs (DirectML, CUDA, CPU).  That lock is process-wide and serialises
    ALL Win32 DLL operations — including the ones Qt uses internally for
    window management and rendering.  Result: the window cannot be moved
    and is marked "(Not responding)" for the duration of the DLL load.

    Starting this warmup on a daemon thread *before* ``window.show()``
    absorbs the contention in the background.  Real model loads after that
    re-use already-resident DLLs and skip the heavy lock.

    Returns the started ``Thread`` so callers can ``join()`` it when
    needed — ``main()`` joins it just before ``window.show()`` (the join
    is nearly instantaneous because the thread finishes during
    ``build_recording_stack``).
    """
    import logging
    import threading

    log = logging.getLogger(__name__)

    def _worker() -> None:
        # Swallow EVERYTHING — the preimport is a latency optimisation,
        # not part of the model-load contract.  If onnx_asr is missing
        # or broken, the user will discover it when they click a model
        # and the proper error path takes over; we must never crash a
        # background thread to the point that pytest / the user's
        # logs scream.
        try:
            _do_onnx_asr_preimport()
        except Exception as exc:  # noqa: BLE001 — intentionally broad
            log.debug("onnx_asr preimport failed (deferred to load): %s", exc)

    t = threading.Thread(
        target=_worker,
        name="onnx-asr-preimport",
        daemon=True,
    )
    t.start()
    return t


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
    # 'use the default') from config.yaml, then plant ``HF_HOME``
    # BEFORE huggingface_hub gets imported.  ConfigManager itself
    # doesn't pull in HF so we can safely import it first.
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

    # Kick off ``import onnx_asr`` on a daemon thread so its onnxruntime +
    # provider DLLs (and the loader-lock contention they trigger on
    # Windows) are absorbed in parallel with QApplication / window
    # construction.  Without this, the first model load can leave the
    # title bar marked "(Not responding)" until the import finishes.
    # Started after the single-instance gate so a second copy of the
    # app exits without paying the import cost.
    _onnx_preload_t = _preload_onnx_asr_async()

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

    # Block until the preimport + ORT provider warmup finishes.
    # Normally the thread completes during build_recording_stack (~1-2 s),
    # so this join returns immediately.  The timeout is a safety net:
    # if ORT hangs on a pathological install we don't block the user
    # indefinitely — they'll see the first model load freeze instead,
    # which is the pre-existing behaviour.  15 s is generous; the real
    # warmup session takes <2 s on this dev box.
    _onnx_preload_t.join(timeout=15)

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

    # Defer the persisted-model autoload until the Qt event loop has
    # ticked at least once — otherwise the loader-thread starts firing
    # tqdm-driven download_progress signals into the main thread queue
    # before Qt has had a chance to acknowledge to Windows that the
    # message pump is alive, and the title bar gets stamped with
    # "(Not responding)" even though the worker is doing the actual work.
    # Using singleShot(0, …) schedules the call onto the next event
    # loop iteration, after Qt has processed WM_PAINT / WM_NCCALCSIZE
    # / WM_SHOWWINDOW from window.show().  The user perceives no delay
    # — the loading pill appears within ~50 ms — but Windows now sees
    # a responsive process before the heavy load starts.
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, lambda: _autoload_persisted_model(backend))

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
