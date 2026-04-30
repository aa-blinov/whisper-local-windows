"""Qt-native system tray icon for the application.

Wraps QSystemTrayIcon with a small state machine (idle / recording /
processing / model_loading), a Show + Quit context menu, and signals
the AppController consumes:

- show_requested: fired by the Show menu item or a left-click on the
  tray icon — controller restores the main window.
- quit_requested: fired by the Quit menu item — controller asks for
  a real application quit (closeEvent honoured, tray torn down).

The state controls only the icon image; clicking the tray icon while
recording/processing does not change behaviour.

Per-platform icon strategy
--------------------------
- **Windows**: load the bundled multi-resolution PNG / ICO assets
  from ``app/assets/`` so the notification area picks the
  appropriate size + the recording / processing colour cues.
- **macOS**: render template icons procedurally with QPainter
  (pure black on transparent, marked as ``QIcon.setIsMask(True)``)
  so macOS auto-tints them per dark / light mode and inverts on
  click — the Apple HIG pattern used by Spotify, Slack, Bartender,
  etc.  We can't just template-flag the bundled cyan/red/yellow
  PNGs: ``setIsMask`` collapses any non-zero alpha pixel to the
  system fill colour, so a coloured source produces an unreadable
  glob.  Drawing a fresh shape per state is cheap and gives us
  sharp 22 × 22 / 44 × 44 (retina) renderings without needing to
  ship a second set of PNGs.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from app.utils import resolve_asset_path


_STATES = ("idle", "recording", "processing", "model_loading")
# Multiple paths per state — we add them all to the QIcon so Windows can
# pick the right resolution / format for the tray (.ico supports multi-size).
_STATE_ASSETS: Dict[str, List[str]] = {
    "idle": ["assets/tray_idle.ico", "assets/tray_idle.png"],
    "recording": ["assets/tray_recording.png"],
    "processing": ["assets/tray_processing.png"],
    # No dedicated icon for model_loading yet — share the processing one.
    "model_loading": ["assets/tray_processing.png"],
}

# Heroicons (MIT, https://heroicons.com) used for the macOS template
# variants — same family the sidebar already uses, so the menu-bar
# icon reads as part of the same product even when collapsed to
# 22 × 22.  Keys mirror ``_STATES``.
_TEMPLATE_SVGS: Dict[str, str] = {
    "idle": "gui/styles/icons/microphone.svg",         # outline mic
    "recording": "gui/styles/icons/microphone-solid.svg",  # filled mic
    "processing": "gui/styles/icons/arrow-path.svg",   # cycling arrows
    "model_loading": "gui/styles/icons/arrow-path.svg",
}


def _load_state_icon(paths: List[str]) -> QIcon:
    icon = QIcon()
    for asset in paths:
        path = resolve_asset_path(asset)
        if path and os.path.isfile(path):
            icon.addFile(path)
    return icon


def _render_svg_template(svg_path: str) -> QIcon:
    """Render a Heroicons SVG into a ``QIcon`` flagged as a macOS
    template image.

    macOS auto-tints template images per dark / light mode and inverts
    them on click — the standard menu-bar look.  Heroicons SVGs use
    ``stroke="currentColor"`` / ``fill="currentColor"`` and the
    QSvgRenderer's default pen is black, so a direct render onto a
    transparent canvas already produces the canonical
    black-on-transparent pixmap that ``setIsMask(True)`` expects.

    We rasterise at 22 / 44 px — the standard NSStatusItem image
    sizes for @1x and @2x (Retina) — so the system picks the sharp
    variant on the user's display.  Glyph inset of ~10 % matches
    what neighbours like the OpenAI / Fantastical menu-bar items
    use; tighter than that bumps into the slot edge, looser than
    that makes our icon look smaller than its peers (which is
    exactly the report this method's first iteration triggered).
    """
    icon = QIcon()
    if not svg_path or not os.path.isfile(svg_path):
        return icon
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        return icon
    for px in (22, 44):
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            inset = px * 0.10
            rect = QRectF(inset, inset, px - 2 * inset, px - 2 * inset)
            renderer.render(painter, rect)
        finally:
            painter.end()
        icon.addPixmap(pixmap)
    icon.setIsMask(True)
    return icon


def _render_template_icon(state: str) -> QIcon:
    """Build a macOS menu-bar template icon for ``state``.

    Picks the matching Heroicons SVG from ``_TEMPLATE_SVGS`` and
    rasterises it via :func:`_render_svg_template`.  Returns an
    empty QIcon if the SVG is missing — caller is responsible for
    falling back to the bundled tray PNGs.
    """
    rel = _TEMPLATE_SVGS.get(state)
    if not rel:
        return QIcon()
    return _render_svg_template(resolve_asset_path(rel))


class AppTrayIcon(QSystemTrayIcon):
    show_requested = Signal()
    quit_requested = Signal()

    # User-facing one-line tooltips per backend state.  Shown on
    # hover over the menu-bar / tray icon, complementing the
    # minimalist template shapes on macOS.
    _STATE_TOOLTIPS: Dict[str, str] = {
        "idle": "Lazy to Text — idle",
        "recording": "Lazy to Text — recording",
        "processing": "Lazy to Text — transcribing",
        "model_loading": "Lazy to Text — loading model",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip(self._STATE_TOOLTIPS["idle"])

        if sys.platform == "darwin":
            # Procedurally-rendered template icons that blend with the
            # macOS menu bar (auto-tinted, sharp on Retina).
            self._icons: Dict[str, QIcon] = {
                state: _render_template_icon(state) for state in _STATES
            }
        else:
            self._icons = {
                state: _load_state_icon(paths)
                for state, paths in _STATE_ASSETS.items()
            }
        self._state = "idle"
        self.setIcon(self._icons[self._state])

        # QMenu is a QWidget and QSystemTrayIcon is only a QObject — we can't
        # parent the menu to ourselves, so hold the reference manually so it
        # outlives the tray icon.
        self._menu = QMenu()
        show_action = self._menu.addAction("Show window")
        show_action.triggered.connect(self.show_requested.emit)
        self._menu.addSeparator()
        quit_action = self._menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_requested.emit)
        self.setContextMenu(self._menu)

        self.activated.connect(self._on_activated)

    # ---- public API ---------------------------------------------------------

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in _STATES:
            raise ValueError(
                f"state must be one of {_STATES}, got {state!r}"
            )
        if state == self._state:
            return
        self._state = state
        self.setIcon(self._icons[state])
        self.setToolTip(self._STATE_TOOLTIPS.get(state, "Lazy to Text"))

    # ---- internal -----------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        # Left-click (Trigger) and double-click both restore the window.
        # Context-menu (right-click) is handled by Qt itself.
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_requested.emit()
