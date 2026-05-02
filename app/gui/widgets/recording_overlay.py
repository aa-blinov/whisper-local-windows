"""Small always-on-top recording status overlay.

Unlike the in-window toast, this widget is a separate top-level tool
window so it can stay visible while the main app is hidden or unfocused.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class RecordingOverlay(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        flags = (
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
        )
        super().__init__(parent, flags)
        self.setObjectName("RecordingOverlay")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("RecordingOverlaySurface")
        self._surface.setProperty("role", "recording-overlay-surface")
        self._surface.setMinimumHeight(56)
        root.addWidget(self._surface)

        surface_layout = QHBoxLayout(self._surface)
        surface_layout.setContentsMargins(16, 12, 16, 12)
        surface_layout.setSpacing(12)

        self._dot = QFrame(self._surface)
        self._dot.setObjectName("RecordingOverlayDot")
        self._dot.setProperty("role", "recording-overlay-dot")
        self._dot.setFixedSize(14, 14)
        surface_layout.addWidget(self._dot, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)

        self._title = QLabel("", self._surface)
        self._title.setObjectName("RecordingOverlayTitle")
        self._title.setProperty("role", "recording-overlay-title")
        text_col.addWidget(self._title)

        self._body = QLabel("", self._surface)
        self._body.setObjectName("RecordingOverlayBody")
        self._body.setProperty("role", "recording-overlay-body")
        text_col.addWidget(self._body)

        surface_layout.addLayout(text_col, 1)

        self._state = "idle"
        self.hide()

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        state = (state or "").strip().lower()
        if state == "recording":
            self._state = "recording"
            self._title.setText("Recording")
            self._body.setText("Speak now")
            self._dot.setProperty("state", "recording")
            self._refresh_styles()
            self._show_overlay()
            return
        if state == "processing":
            self._state = "processing"
            self._title.setText("Processing")
            self._body.setText("Transcribing speech")
            self._dot.setProperty("state", "processing")
            self._refresh_styles()
            self._show_overlay()
            return
        self._state = "idle"
        self.hide()

    def _show_overlay(self) -> None:
        self.adjustSize()
        self._reposition()
        self.raise_()
        self.show()

    def _reposition(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        margin_top = 28
        x = rect.x() + max(0, (rect.width() - self.width()) // 2)
        y = rect.y() + margin_top
        self.move(x, y)

    def _refresh_styles(self) -> None:
        self.style().unpolish(self._dot)
        self.style().polish(self._dot)
        self.style().unpolish(self._surface)
        self.style().polish(self._surface)
