"""Main application window — shell with sidebar navigation and stacked views."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.views.logs_view import LogsView
from app.gui.views.models_view import ModelsView
from app.gui.views.placeholder import PlaceholderView
from app.gui.views.shortcuts_view import ShortcutsView
from app.gui.widgets.sidebar import Sidebar
from app.gui.widgets.topbar import TopBar


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lazy to Text")
        self.resize(1000, 660)
        self.setMinimumSize(860, 560)

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
        for key in self.sidebar.items():
            if key == "models":
                view: QWidget = self.models_view
            elif key == "logs":
                view = self.logs_view
            elif key == "shortcuts":
                view = self.shortcuts_view
            else:
                view = PlaceholderView(key.capitalize(), parent=self.stack)
            self._views[key] = view
            self.stack.addWidget(view)

        default_key = self.sidebar.active_key()
        if default_key in self._views:
            self.stack.setCurrentWidget(self._views[default_key])

        self.sidebar.nav_selected.connect(self._on_nav_selected)

    def get_view(self, key: str) -> QWidget:
        if key not in self._views:
            raise KeyError(key)
        return self._views[key]

    def _on_nav_selected(self, key: str) -> None:
        if key in self._views:
            self.stack.setCurrentWidget(self._views[key])
