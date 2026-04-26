"""Startup splash screen + synchronous backend warmup.

Exists because Qt's main window goes unresponsive (un-draggable,
un-minimisable) during NeMo's cold-import + ``from_pretrained`` —
the GIL is held by the Python-heavy module loader, so even the
fact that the load runs in a worker thread doesn't help. The
fix is to pre-load the backend BEFORE creating the main window,
and pump ``QApplication.processEvents()`` from the main thread
during the wait so Qt can deliver paint / mouse / window-manager
events to a small splash widget.

The splash widget is movable, minimisable, and shows a live
progress percentage so the user has immediate visual feedback —
all of which the half-built main window couldn't provide while
``import nemo`` had the GIL.

Public surface
--------------
- ``make_splash(app_name)`` — build a QSplashScreen pixmap.
- ``wait_for_backend(splash, backend, display_name, app, timeout_s)``
  — pump events while the backend warms up; returns one of
  ``"ready"``, ``"error"``, ``"stopped"`` (cancel / shutdown), or
  ``"timeout"``.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen


# Splash visual sizing — picked to feel like a small launcher card,
# big enough that the model name + progress percentage both fit on a
# single line without being cropped at narrow DPI.
_SPLASH_W = 380
_SPLASH_H = 180

# Colour tokens — kept in sync (by hand) with app/gui/theme.py so the
# splash matches the dark theme of the main window.
_BG = QColor("#1a1d24")        # bg_secondary
_BORDER = QColor("#2d3140")    # border
_TEXT = QColor("#e6e8ec")      # text_primary
_TEXT_MUTED = QColor("#7d828d")  # text_muted
_ACCENT = QColor("#7c93ff")    # accent

_MESSAGE_FLAGS = Qt.AlignBottom | Qt.AlignLeft


def _format_size(num_bytes: int) -> str:
    n = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def make_splash(app_name: str = "Lazy to Text") -> QSplashScreen:
    """Build a small dark-themed splash widget.

    The pixmap is custom-painted (rather than loaded from disk) so we
    don't drag in another asset file just for boot — the splash is
    only on screen for ~30 s on a warm load.
    """
    pixmap = QPixmap(_SPLASH_W, _SPLASH_H)
    pixmap.fill(_BG)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)

        # Outline so the splash reads as a card, not a flat block.
        painter.setPen(_BORDER)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, 0, _SPLASH_W - 1, _SPLASH_H - 1)

        # Title — bigger, accent-coloured.
        title_font = QFont(painter.font())
        title_font.setPointSizeF(title_font.pointSizeF() + 6)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(_TEXT)
        painter.drawText(
            24, 0, _SPLASH_W - 48, 80,
            Qt.AlignVCenter | Qt.AlignLeft, app_name,
        )

        # Subtitle — muted hint.
        sub_font = QFont(painter.font())
        sub_font.setPointSizeF(sub_font.pointSizeF() - 6)
        sub_font.setBold(False)
        painter.setFont(sub_font)
        painter.setPen(_TEXT_MUTED)
        painter.drawText(
            24, 70, _SPLASH_W - 48, 24,
            Qt.AlignVCenter | Qt.AlignLeft,
            "Preparing model… you can move or minimise this window.",
        )
    finally:
        painter.end()

    splash = QSplashScreen(pixmap)
    # Stay-on-top so the user always sees progress; not modal.
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    return splash


def wait_for_backend(
    splash: QSplashScreen,
    backend,
    display_name: str,
    app: Optional[QApplication] = None,
    timeout_s: float = 1800.0,
) -> str:
    """Block on ``backend.load()`` while pumping Qt events through the
    splash. Returns the terminal status — one of:

    - ``"ready"``   — backend reported ready
    - ``"error"``   — backend reported error
    - ``"stopped"`` — backend reported stopped (cancel-load,
      shutdown, or any other non-error abort path)
    - ``"timeout"`` — wall-clock deadline exceeded without a
      terminal status

    Wires the backend's progress callback so the splash message shows
    a climbing percentage (or byte count when total is unknown — same
    fallback the topbar pill uses).
    """
    if app is None:
        app = QApplication.instance()
    assert app is not None, "QApplication must exist before wait_for_backend"

    # Mutable progress state captured by the closure below; updated
    # every time tqdm fires, read on the next splash repaint.
    state = {"current": 0, "total": 0, "desc": ""}

    def _on_progress(current: int, total: int, desc: str) -> None:
        state["current"] = int(current or 0)
        state["total"] = int(total or 0)
        state["desc"] = str(desc or "")

    set_cb = getattr(backend, "set_progress_callback", None)
    if set_cb is not None:
        try:
            set_cb(_on_progress)
        except Exception:  # pragma: no cover — defensive
            pass

    backend.load()

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_render = 0.0

    while time.monotonic() < deadline:
        status = backend.status()
        if status in ("ready", "error", "stopped"):
            # Detach the progress callback so the next backend (if the
            # routed swap rebuilds it) gets a clean slot.
            if set_cb is not None:
                try:
                    set_cb(None)
                except Exception:  # pragma: no cover — defensive
                    pass
            return "ready" if status == "ready" else (
                "error" if status == "error" else "stopped"
            )

        # Render at ~10 Hz — keeps the UI responsive without burning
        # CPU on splash repaints between actual progress updates.
        now = time.monotonic()
        if now - last_render >= 0.1:
            last_render = now
            current = state["current"]
            total = state["total"]
            if total > 0 and current >= 0:
                pct = int(min(99, max(0, current * 100 // total)))
                progress_text = f"{pct}%"
            elif current > 0:
                progress_text = _format_size(current)
            else:
                progress_text = ""
            message = f"Loading: {display_name}"
            if progress_text:
                message += f"  {progress_text}"
            splash.showMessage(message, _MESSAGE_FLAGS, _TEXT)

        app.processEvents()
        # Short sleep keeps the GIL share modest — the worker thread
        # needs to make progress, and if we processEvents() in a tight
        # loop we starve it.
        time.sleep(0.02)

    if set_cb is not None:
        try:
            set_cb(None)
        except Exception:  # pragma: no cover — defensive
            pass
    return "timeout"
