"""Qt application entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImageReader,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from app.gui.controllers.app_controller import AppController
from app.gui.log_bridge import QtLogBridge
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme, load_bundled_fonts
from app.utils import is_cached_for_info, is_model_cached, resolve_asset_path


def _add_ico_frames(icon: QIcon, ico_path: str) -> int:
    """Add every embedded frame of a Windows ICO file to ``icon``.

    Why we can't just ``QIcon.addFile(path.ico)``: Qt's ICO plugin on
    macOS only loads the file's *first* frame.  Our ``tray_idle.ico``
    bundles 7 sizes from 16 × 16 up to 256 × 256, but QIcon ends up
    holding the 16 × 16 entry only — so the macOS Dock (which wants
    128 × 128) up-scales it from 16 px and the user sees a blurry
    blob instead of the real logo.

    Iterating frames via ``QImageReader.jumpToImage`` lets us pull
    every embedded size into the QIcon's pixmap cache, and macOS
    then picks the closest match to the requested rendering size.

    Returns the number of frames successfully added.
    """
    reader = QImageReader(ico_path)
    if not reader.canRead():
        return 0
    frame_count = reader.imageCount() or 1
    added = 0
    for i in range(frame_count):
        if not reader.jumpToImage(i):
            break
        image = reader.read()
        if image.isNull():
            continue
        icon.addPixmap(QPixmap.fromImage(image))
        added += 1
    return added


# Accent gradient for the squircle Dock icon — matches the
# ``TOKENS.colors.accent`` / ``accent_hover`` pair from
# ``app.gui.theme``.  Vertical top-to-bottom gradient gives a subtle
# sheen that reads as "modern app" without overpromising depth.
_DOCK_GRADIENT_TOP = QColor("#5b8cff")
_DOCK_GRADIENT_BOTTOM = QColor("#7aa2ff")
# Apple's iOS / macOS app-icon shape is a *superellipse* (Lamé curve
# ``|x|^n + |y|^n = 1``), not a rounded rectangle: rounded rects
# join straight edges to circular corners with a visible curvature
# discontinuity (G1 only), while a superellipse has a smooth
# curvature derivative all the way around (G2).  ``n ≈ 4`` is the
# closest single-exponent superellipse to Apple's actual app-icon
# silhouette — slightly rounder than ``n = 5``, less square at the
# corners, and visually matches what shows up next to it in the
# Dock (Steam, Music, Calculator, …).
_DOCK_SUPERELLIPSE_N = 4.0
# Apple's app-icon design grid leaves a margin around the squircle.
# The published content area is 824 / 1024 ≈ 80 %, but the visible
# squircle in shipped system apps (Steam, Music, Calculator, …)
# sits at ~75 % of the canvas — eyeballed against neighbours in
# the Dock until ours matched their footprint.
_DOCK_ICON_OCCUPANCY = 0.75
# Heroicons microphone-solid is rendered at ~50 % of the icon width
# so it sits centred with comfortable padding — the proportion most
# Mac apps with single-glyph logos (Slack mic, Zoom mic, etc.) use.
_DOCK_FOREGROUND_RATIO = 0.50


def _build_superellipse_path(size: float, n: float = _DOCK_SUPERELLIPSE_N) -> QPainterPath:
    """Build a superellipse (Lamé curve) path inscribed in a
    ``size × size`` square — the actual macOS / iOS app-icon shape.

    Parametric form (centred on origin, half-width = a):
        x(t) = sgn(cos t) · a · |cos t|^(2/n)
        y(t) = sgn(sin t) · a · |sin t|^(2/n)

    n = 2  → a regular ellipse (circle when a = b)
    n = 4  → classic squircle, a touch boxier
    n = 5  → close to Apple's actual app-icon shape
    n = 6+ → approaches a square

    A 240-segment polyline is more than enough for the antialiased
    Dock thumbnail; the eye can't tell apart from a true G2 curve
    at any Dock size.
    """
    cx = size / 2.0
    cy = size / 2.0
    a = size / 2.0

    path = QPainterPath()
    steps = 240
    exp = 2.0 / n
    for i in range(steps + 1):
        t = 2.0 * math.pi * i / steps
        ct = math.cos(t)
        st = math.sin(t)
        x = cx + a * math.copysign(abs(ct) ** exp, ct)
        y = cy + a * math.copysign(abs(st) ** exp, st)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    path.closeSubpath()
    return path


def _render_white_glyph(svg_path: str, size: int) -> Optional[QPixmap]:
    """Rasterise a monochromatic SVG into a white-on-transparent
    pixmap suitable for compositing onto a coloured Dock background.

    The SVGs we ship (Heroicons) declare ``fill="currentColor"`` /
    ``stroke="currentColor"``, so ``QSvgRenderer`` paints them in
    whatever pen colour is active — black by default. We render
    once, then replace the result's black pixels with white via the
    ``CompositionMode_SourceIn`` trick: filling a rect over the
    rasterised glyph keeps only pixels where the original alpha
    channel was non-zero, so the glyph silhouette becomes a solid
    white shape.
    """
    if not svg_path or not os.path.isfile(svg_path):
        return None
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        return None
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(255, 255, 255))
    finally:
        painter.end()
    return pixmap


def _render_dock_icon_at(size: int) -> QPixmap:
    """Render the Dock-style squircle app icon at ``size × size`` px.

    Layers (bottom-up):
      1. Transparent margin matching Apple's 824/1024 design grid
         so the visible squircle sits at ~80 % of the canvas — same
         occupancy as system apps (Music, Calculator, Photo Booth).
      2. Superellipse (G2 squircle) filled with the accent gradient.
      3. Heroicons ``microphone-solid`` glyph in white, centred at
         ~50 % of the squircle width.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)

        inner_size = size * _DOCK_ICON_OCCUPANCY
        offset = (size - inner_size) / 2.0

        # Background squircle (true superellipse, not a rounded rect),
        # translated by ``offset`` so it sits centred inside the
        # design-grid margin.
        path = _build_superellipse_path(inner_size)
        path.translate(offset, offset)
        gradient = QLinearGradient(0, offset, 0, offset + inner_size)
        gradient.setColorAt(0.0, _DOCK_GRADIENT_TOP)
        gradient.setColorAt(1.0, _DOCK_GRADIENT_BOTTOM)
        painter.fillPath(path, gradient)

        # Foreground microphone glyph, sized relative to the inner
        # squircle (not the full canvas) so it stays at the visual
        # 50 % proportion users see in other Dock icons.
        glyph_size = int(inner_size * _DOCK_FOREGROUND_RATIO)
        glyph = _render_white_glyph(
            resolve_asset_path("gui/styles/icons/microphone-solid.svg"),
            glyph_size,
        )
        if glyph is not None:
            glyph_offset = (size - glyph_size) // 2
            painter.setClipPath(path)
            painter.drawPixmap(glyph_offset, glyph_offset, glyph)
    finally:
        painter.end()
    return pixmap


def _build_macos_dock_icon() -> QIcon:
    """Programmatically assemble a Mac-native Dock app icon.

    Renders the squircle + microphone composite at every Dock size
    macOS asks for (16 / 32 / 64 / 128 / 256 / 512 / 1024) so the
    same QIcon answers crisply on @1x and @2x density displays
    without re-rasterising on demand.
    """
    icon = QIcon()
    for size in (16, 32, 64, 128, 256, 512, 1024):
        icon.addPixmap(_render_dock_icon_at(size))
    return icon


def _load_app_icon() -> QIcon:
    """Build the QIcon used for the app's window / taskbar / Dock.

    macOS gets a procedurally-rendered squircle (gradient + Heroicons
    microphone) so the Dock and Cmd-Tab show a sharp, on-brand icon
    at every size the system asks for. Other platforms fall back to
    the bundled multi-resolution ICO + PNG so taskbar / Alt-Tab pick
    the right embedded size.
    """
    if sys.platform == "darwin":
        return _build_macos_dock_icon()

    icon = QIcon()
    ico_path = resolve_asset_path("assets/tray_idle.ico")
    if ico_path and os.path.isfile(ico_path):
        _add_ico_frames(icon, ico_path)
    png_path = resolve_asset_path("assets/tray_idle.png")
    if png_path and os.path.isfile(png_path):
        icon.addFile(png_path)
    return icon


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


from app.backends.onnx_backend import _WARMUP_ONNX_BYTES


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

        # Keep missing-CUDA-DLL noise off stderr during warmup.
        _ort.set_default_logger_severity(4)

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


def _autoload_persisted_model(backend, config=None) -> None:
    """Kick off a model load on startup, picking the best
    candidate available on disk.

    Called once at startup, after the recording stack is built
    but before the main window is shown.  Returns immediately —
    the backend's ``load()`` spawns a daemon thread internally
    and the UI paints the loading state from the model card /
    topbar polling machinery.

    Selection order:

    1. If the persisted model from ``config.yaml`` is cached →
       load that.  Honours the user's last explicit pick.
    2. Otherwise pick the first cached model in registry display
       order.  Avoids the "fresh restart, recording does
       nothing" pothole when ``config.yaml`` points at a model
       the user removed (or never downloaded — e.g. the default
       ``parakeet-tdt-v3`` on a Mac that's been working with
       ``gigaam-v3-ctc``).  We log the swap so the Logs view
       explains why a different model came up than what
       Settings currently reads.
    3. Nothing on disk at all → log "skipping auto-load" so
       Settings shows an empty state instead of pretending to
       load something that isn't there.
    """
    import logging

    log = logging.getLogger(__name__)
    if backend is None:
        return

    from app.model_mapping import MODELS, alias_for, get_model

    persisted_name = None
    if config is not None:
        raw = getattr(config, "get_setting", None)
        if callable(raw):
            try:
                candidate = raw("whisper", "model")
            except Exception:  # pragma: no cover — defensive
                candidate = None
            if isinstance(candidate, str) and candidate.strip():
                persisted_name = candidate.strip()

    requested = persisted_name or backend.current_model()
    try:
        alias = requested if persisted_name is not None else alias_for(requested)
        info = get_model(alias)
    except KeyError:
        info = None
        cached = is_model_cached(requested)
    else:
        cached = is_cached_for_info(info)

    if cached:
        display = info.display_name if info is not None else requested
        log.info(
            "Persisted model %s is cached — kicking off background load.",
            display,
        )
        backend.load()
        return

    # Persisted model isn't cached — look for any cached fallback
    # in registry order so the app still comes up with a working
    # backend instead of an idle "click Download" placeholder.
    for candidate in MODELS:
        if not is_cached_for_info(candidate):
            continue
        log.info(
            "Persisted model %s is not cached — falling back to "
            "cached %s (%s).  Pick a different model in Settings to "
            "override.",
            requested, candidate.display_name, candidate.alias,
        )
        change = getattr(backend, "change_model", None)
        if callable(change):
            try:
                change(candidate.alias)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("change_model fallback raised: %s", exc)
                return
        # Persist the fallback into config.yaml too — otherwise
        # ``_get_active_alias`` in the controller still reads the
        # uncached pick from disk and the Models tab paints the
        # wrong card as Active. ``config`` is optional so ad-hoc
        # callers (tests, future scripts) don't have to wire it.
        if config is not None:
            try:
                config.update_user_setting(
                    "whisper", "model", candidate.alias,
                )
                if candidate.compute_type:
                    config.update_user_setting(
                        "whisper", "compute_type",
                        candidate.compute_type,
                    )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "Failed to persist fallback model into config: %s",
                    exc,
                )
        backend.load()
        return

        log.info(
            "Persisted model %s is not cached and no other model is "
            "downloaded — skipping auto-load. Waiting for the user to "
            "pick a model.",
            requested,
        )


def main() -> int:
    import logging
    import multiprocessing

    # Frozen / py2app builds re-exec the bundle's launcher when
    # ``multiprocessing.spawn`` starts a worker.  Without
    # ``freeze_support`` that re-exec re-enters ``main()`` instead
    # of the worker's target, leaks dozens of half-started Qt
    # windows, and our ``Subprocess init failed: Worker pipe
    # closed before init`` warning fires because the worker
    # never made it to the ``init`` ack.  Calling here is a no-op
    # in dev (``uv run``) — it only takes effect when ``sys.frozen``
    # is set, which py2app does inside the bundle.
    multiprocessing.freeze_support()

    # Auto-install CUDA redistributables on Windows when an NVIDIA
    # GPU is present but the runtime DLLs are missing.  This keeps
    # ``uv run lazy-to-text-ui`` as the single entry point — no
    # manual ``pip install [cuda]`` required.  Fast no-op (<10 ms)
    # when CUDA already works or no GPU exists.
    try:
        from app.cuda_bootstrap import ensure_cuda
        ensure_cuda()
    except Exception:
        pass  # never block startup over bootstrap noise

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

    instance_handle = try_acquire_single_instance("LazyToTextQt")

    # Bring up QApplication regardless of branch — both the primary path
    # and the duplicate-warning dialog need our app icon to show in
    # taskbar / Alt-Tab instead of the Python interpreter's icon.
    from PySide6.QtWidgets import QMessageBox

    qt_app = QApplication.instance() or QApplication(sys.argv)
    # Register the bundled Inter font as early as possible so any QFont
    # resolution further down in the stack (icons, message boxes,
    # tooltips spun up before ``apply_theme`` runs) doesn't trigger the
    # ``qt.qpa.fonts: Replace uses of missing font family "Inter"``
    # warning. Idempotent if called again from ``apply_theme``.
    load_bundled_fonts()
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
            msg.exec()
        except Exception:
            print(
                "Lazy to Text: another instance is already running.",
                file=sys.stderr,
            )
        return 0

    # We are the primary instance — bind the lock handle so it survives.
    qt_app._instance_mutex = instance_handle  # type: ignore[attr-defined]

    # Spawn the inference worker as the very first thing after the
    # single-instance gate.  Windows ``multiprocessing.spawn`` re-execs
    # Python and re-imports onnx_asr (~3-7 s on a cold start); doing it
    # now lets that work overlap with logging + recording-stack +
    # MainWindow construction below, so by the time we hit
    # ``window.show()`` the worker has usually finished init and the
    # first ``backend.load()`` (autoload) ack-roundtrips in <50 ms
    # instead of stalling the UI.
    #
    # ``SubprocessBackend.__init__`` is async — it returns once the
    # child process is started + the init command has been written to
    # the pipe; the worker's ack is consumed by the reader thread and
    # any later ``_send_cmd`` blocks on the init event until ready.
    #
    # Inside a py2app .app on macOS we skip the subprocess entirely
    # and use the in-process ``RegistryBackend`` instead.  The
    # subprocess design exists to dodge Windows' DLL-loader-lock
    # while ``onnx_asr`` initialises ORT providers — that lock
    # doesn't exist on macOS, so the only thing we'd buy by
    # spawning a worker is a portable code path.  Spawning is also
    # the part that doesn't survive py2app: the spawn child re-execs
    # the bundle's launcher binary instead of a Python interpreter,
    # ``init_main_from_path`` then tries to ``runpy.run_path`` a
    # bootstrap path that isn't a script, and the worker dies
    # before its init ack with the cryptic ``Worker pipe closed
    # before init`` we kept seeing.  In-process is simpler, faster
    # to start, and entirely sufficient on macOS.
    _whisper_cfg = _early_config.get_whisper_config()
    _backend_kwargs = dict(
        model=_whisper_cfg.get("model") or "whisper-large-v3-turbo",
        device=str(_whisper_cfg.get("device", "auto")),
        compute_type=str(_whisper_cfg.get("compute_type", "float16")),
        language=_whisper_cfg.get("language") or None,
        beam_size=int(_whisper_cfg.get("beam_size", 5)),
    )
    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        from app.backends.registry_backend import RegistryBackend

        _early_backend = RegistryBackend(**_backend_kwargs)
    else:
        from app.backends.subprocess_backend import SubprocessBackend

        _early_backend = SubprocessBackend(**_backend_kwargs)

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
        # Pass the pre-spawned backend in so build_recording_stack
        # doesn't create a second one — the early-spawned worker has
        # already started initialising in parallel.
        state_manager, hotkey_listener, backend = build_recording_stack(
            config_manager=config,
            backend=_early_backend,
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

    # macOS Dock-click handling: when the user clicks the app's Dock
    # icon while the main window is hidden (close button → tray
    # path), Qt fires ``applicationStateChanged(ApplicationActive)``
    # but does NOT restore the window for us.  Listen for the
    # transition and unhide / raise / activate the window so the
    # Dock icon behaves like every other Mac app.  Skip on
    # Windows / Linux — the tray icon is the canonical restore
    # affordance there, not the taskbar / launcher button.
    if sys.platform == "darwin":
        from PySide6.QtCore import Qt as _Qt

        def _on_application_state_changed(state) -> None:
            if state != _Qt.ApplicationState.ApplicationActive:
                return
            if window.isVisible():
                # User just brought the existing window back to
                # focus — macOS handles the layering for us if the
                # window is visible.
                return
            window.showNormal()
            window.raise_()
            window.activateWindow()

        app.applicationStateChanged.connect(_on_application_state_changed)

    # Run the persisted-model autoload on a daemon thread so the Qt
    # main thread isn't blocked if the worker process is still finishing
    # its init when we get here.  ``backend.load()`` calls
    # ``SubprocessBackend._send_cmd`` which waits on the worker's
    # ``_init_event`` — that wait can take a couple of seconds on a
    # cold start (Windows ``spawn`` + onnx_asr re-import).  Doing it
    # off the main thread means the window is fully interactive
    # straight away; the loading pill / topbar progress paint as soon
    # as the worker is ready.
    import threading as _threading

    _threading.Thread(
        target=_autoload_persisted_model,
        args=(backend, config),
        daemon=True,
        name="autoload-model",
    ).start()

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


def main_cuda() -> int:
    """Entry point for GPU-accelerated launch (``lazy-to-text-ui-cuda``).

    On Windows: unconditionally installs ``onnxruntime-gpu`` + CUDA
    redistributables if they are missing, then delegates to ``main()``.
    On other platforms this is identical to ``main()``.
    """
    if sys.platform == "win32":
        from app.cuda_bootstrap import (
            _cuda_probe,
            _install_cuda_redist,
            _install_onnxruntime_gpu,
        )
        from app.utils import _try_inject_nvidia_pip_dll_paths

        _try_inject_nvidia_pip_dll_paths()
        if not _cuda_probe():
            print("Setting up GPU acceleration for NVIDIA...")
            if not _install_onnxruntime_gpu():
                print("Failed to install onnxruntime-gpu, falling back to CPU.")
            elif not _install_cuda_redist():
                print("Failed to install CUDA libraries, falling back to CPU.")
            else:
                _try_inject_nvidia_pip_dll_paths()
                if _cuda_probe():
                    print("GPU acceleration ready.")
                else:
                    print("GPU setup incomplete, falling back to CPU.")
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
