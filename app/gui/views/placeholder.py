"""Placeholder view used by the shell until real views are implemented."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderView(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName(f"View_{title}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self._title_label = QLabel(title, self)
        self._title_label.setProperty("role", "title")

        self._subtitle_label = QLabel("Coming soon.", self)
        self._subtitle_label.setProperty("role", "muted")

        layout.addWidget(self._title_label)
        layout.addWidget(self._subtitle_label)
        layout.addStretch(1)

    def title(self) -> str:
        return self._title_label.text()
