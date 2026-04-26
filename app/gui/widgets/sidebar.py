"""Left-hand navigation sidebar for the main window."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.theme import icon_path
from app.gui.widgets.recording_status_widget import RecordingStatusWidget


NavItem = Tuple[str, str]
_DEFAULT_ITEMS: tuple[NavItem, ...] = (
    ("models", "Models"),
    ("shortcuts", "Settings"),
    ("history", "History"),
    ("logs", "Logs"),
)
# Filename (without ``.svg``) inside ``app/gui/styles/icons/`` for
# each nav key. Heroicons (outline, 24×24) — line-style works at
# 20 px sidebar size and ages better than custom icons.
_KEY_ICON_FILES: dict[str, str] = {
    "models": "models",
    "shortcuts": "settings",
    "history": "history",
    "logs": "logs",
}
_KEY_ROLE = Qt.UserRole + 1
_LABEL_ROLE = Qt.UserRole + 2


class Sidebar(QWidget):
    nav_selected = Signal(str)

    def __init__(
        self,
        items: Optional[Sequence[NavItem]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._list = QListWidget(self)
        self._list.setObjectName("SidebarList")
        self._list.setFrameShape(QListWidget.NoFrame)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        # Pixel-level wheel scrolling so the sidebar doesn't snap by
        # one item per notch.
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list.verticalScrollBar().setSingleStep(20)
        # Heroicons render best at ~20 px in a 14-px-text row.
        self._list.setIconSize(QSize(20, 20))
        # Stretch=1 so the nav list eats whatever vertical space the
        # bottom recording-status slot doesn't claim.
        layout.addWidget(self._list, 1)

        # Recording status slot pinned to the bottom-left of the
        # sidebar — recording pill stacked over the live VU meter,
        # always reserves its placeholder height even when idle.
        # Lives here (not in the topbar) so it doesn't overlap the
        # CPU / RAM / GPU resource graphs that occupy the topbar.
        self.recording_status = RecordingStatusWidget(self)
        layout.addWidget(self.recording_status)

        resolved = tuple(items) if items is not None else _DEFAULT_ITEMS
        for key, label in resolved:
            entry = QListWidgetItem(label)
            entry.setData(_KEY_ROLE, key)
            entry.setData(_LABEL_ROLE, label)
            icon_name = _KEY_ICON_FILES.get(key)
            if icon_name:
                path = icon_path(f"{icon_name}.svg")
                if path is not None and Path(path).is_file():
                    entry.setIcon(QIcon(path))
            self._list.addItem(entry)

        self._current_key = resolved[0][0] if resolved else ""
        if resolved:
            self._list.setCurrentRow(0)

        self._list.currentRowChanged.connect(self._on_row_changed)

    def items(self) -> List[str]:
        return [
            self._list.item(i).data(_KEY_ROLE) for i in range(self._list.count())
        ]

    def label_for(self, key: str) -> str:
        """Plain display label for a nav key (e.g. ``shortcuts`` ->
        ``Settings``) — strips the icon prefix used in the list row.
        Falls back to ``key.capitalize()`` if the key isn't known."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(_KEY_ROLE) == key:
                return item.data(_LABEL_ROLE) or item.text()
        return key.capitalize()

    def active_key(self) -> str:
        return self._current_key

    def set_active(self, key: str) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(_KEY_ROLE) == key:
                if self._list.currentRow() != i:
                    self._list.setCurrentRow(i)
                return
        raise ValueError(f"Unknown nav key: {key!r}")

    def _on_row_changed(self, row: int) -> None:
        if row < 0:
            return
        key = self._list.item(row).data(_KEY_ROLE)
        if key == self._current_key:
            return
        self._current_key = key
        self.nav_selected.emit(key)
