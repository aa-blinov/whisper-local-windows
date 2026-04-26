"""Main application window — shell with sidebar navigation and stacked views."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.views.history_view import HistoryView
from app.gui.views.logs_view import LogsView
from app.gui.views.models_view import ModelsView
from app.gui.views.placeholder import PlaceholderView
from app.gui.views.shortcuts_view import ShortcutsView
from app.gui.widgets.sidebar import Sidebar
from app.gui.widgets.toast import Toast
from app.gui.widgets.topbar import TopBar


class MainWindow(QMainWindow):
    hidden_to_tray = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lazy to Text")
        # Default size accommodates the Settings tab without scroll
        # bars; the minimum keeps the user from squishing the window
        # below the Storage card's natural height (any smaller and
        # form labels start rendering on top of their inputs).
        self.resize(1100, 780)
        self.setMinimumSize(900, 720)
        self._close_to_tray = False
        self._quitting = False

        central = QWidget(self)
        central.setObjectName("Central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.topbar = TopBar(parent=central)
        root.addWidget(self.topbar)

        body = QWidget(central)
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = Sidebar(parent=body)
        self.stack = QStackedWidget(body)

        body_layout.addWidget(self.sidebar)
        body_layout.addWidget(self.stack, 1)

        root.addWidget(body, 1)

        self._views: Dict[str, QWidget] = {}
        self.models_view = ModelsView(parent=self.stack)
        self.logs_view = LogsView(parent=self.stack)
        self.shortcuts_view = ShortcutsView(parent=self.stack)
        self.history_view = HistoryView(parent=self.stack)
        for key in self.sidebar.items():
            if key == "models":
                view: QWidget = self.models_view
            elif key == "logs":
                view = self.logs_view
            elif key == "shortcuts":
                view = self.shortcuts_view
            elif key == "history":
                view = self.history_view
            else:
                view = PlaceholderView(key.capitalize(), parent=self.stack)
            self._views[key] = view
            self.stack.addWidget(view)

        default_key = self.sidebar.active_key()
        if default_key in self._views:
            self.stack.setCurrentWidget(self._views[default_key])

        self.sidebar.nav_selected.connect(self._on_nav_selected)

        # Floating banner that shows after every successful
        # transcription. Parented to the central widget so it sits
        # above the views; positioned in the top-right by ``Toast``
        # itself.
        self.toast = Toast(parent=central)
        central.installEventFilter(self)

        # Ctrl+1..4 jump straight to the matching tab — same order as
        # the sidebar.
        self._shortcuts: list[QShortcut] = []
        nav_keys = list(self.sidebar.items())
        for index, key in enumerate(nav_keys[:9]):
            sc = QShortcut(
                QKeySequence(f"Ctrl+{index + 1}"),
                self,
            )
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda k=key: self._activate_nav(k))
            self._shortcuts.append(sc)

    def get_view(self, key: str) -> QWidget:
        if key not in self._views:
            raise KeyError(key)
        return self._views[key]

    def set_close_to_tray(self, enabled: bool) -> None:
        """When True, the window's close button hides to tray instead of
        quitting; the controller is expected to wire a tray icon that can
        bring the window back. Call ``request_quit`` to bypass the override
        for a real exit."""
        self._close_to_tray = bool(enabled)

    def request_quit(self) -> None:
        """Mark the next close as a real quit and close the window."""
        self._quitting = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if self._close_to_tray and not self._quitting:
            event.ignore()
            self.hide()
            self.hidden_to_tray.emit()
            return
        super().closeEvent(event)

    def _on_nav_selected(self, key: str) -> None:
        if key in self._views:
            self.stack.setCurrentWidget(self._views[key])

    def _activate_nav(self, key: str) -> None:
        """Switch to the named tab — used by the Ctrl+N keyboard
        shortcuts."""
        try:
            self.sidebar.set_active(key)
        except ValueError:
            pass

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        # Re-anchor the toast on resize so it sticks to the top-right
        # corner regardless of window size.
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Resize and watched is self.centralWidget():
            self.toast.parentResized()
        return super().eventFilter(watched, event)
