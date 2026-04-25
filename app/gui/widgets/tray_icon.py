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
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from app.utils import resolve_asset_path


_STATES = ("idle", "recording", "processing", "model_loading")
_STATE_ASSETS = {
    "idle": "assets/tray_idle.png",
    "recording": "assets/tray_recording.png",
    "processing": "assets/tray_processing.png",
    # No dedicated icon for model_loading yet — share the processing one.
    "model_loading": "assets/tray_processing.png",
}


class AppTrayIcon(QSystemTrayIcon):
    show_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("Lazy to Text")

        self._icons: Dict[str, QIcon] = {
            state: QIcon(resolve_asset_path(path))
            for state, path in _STATE_ASSETS.items()
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

    # ---- internal -----------------------------------------------------------

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        # Left-click (Trigger) and double-click both restore the window.
        # Context-menu (right-click) is handled by Qt itself.
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_requested.emit()
