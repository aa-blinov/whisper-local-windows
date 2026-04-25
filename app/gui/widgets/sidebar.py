"""Left-hand navigation sidebar for the main window."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


NavItem = Tuple[str, str]
# Item-row display strings — a leading Unicode glyph (rendered by
# Segoe UI Symbol / Segoe UI Emoji on Windows) gives the eye a
# categorical anchor without bundling SVG icons. The plain label is
# preserved separately so ``label_for`` can return it without the
# decoration for places like the topbar section title.
_DEFAULT_ITEMS: tuple[NavItem, ...] = (
    ("models", "Models"),
    ("shortcuts", "Settings"),
    ("history", "History"),
    ("logs", "Logs"),
)
_KEY_ICONS: dict[str, str] = {
    "models": "◆",
    "shortcuts": "⚙",
    "history": "⏱",
    "logs": "≡",
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
        layout.addWidget(self._list)

        resolved = tuple(items) if items is not None else _DEFAULT_ITEMS
        for key, label in resolved:
            icon = _KEY_ICONS.get(key, "")
            display = f"{icon}  {label}" if icon else label
            entry = QListWidgetItem(display)
            entry.setData(_KEY_ROLE, key)
            entry.setData(_LABEL_ROLE, label)
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
