"""Application logs view."""

from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LogsView(QWidget):
    DEFAULT_MAX_LINES = 5000

    def __init__(
        self,
        max_lines: int = DEFAULT_MAX_LINES,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("LogsView")
        self._max_lines = max(1, int(max_lines))

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addStretch(1)

        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setObjectName("ClearLogsButton")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)

        root.addLayout(header)

        self._text = QPlainTextEdit(self)
        self._text.setObjectName("LogsTextArea")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self._max_lines)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        root.addWidget(self._text, 1)

    def append_line(self, text: str) -> None:
        self._text.appendPlainText(text)
        self._text.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        self._text.clear()
