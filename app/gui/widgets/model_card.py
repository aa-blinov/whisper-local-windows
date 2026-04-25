"""Card widget that displays a single model entry."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.model_mapping import ModelInfo


def _format_size(size_mb: int) -> str:
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


class ModelCard(QFrame):
    select_requested = Signal(str)

    def __init__(self, info: ModelInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._info = info
        self._active = False
        self._locked = False
        self._loading = False

        self.setObjectName("ModelCard")
        self.setProperty("role", "card")
        self.setProperty("active", False)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel(info.display_name, self)
        title.setProperty("role", "heading")
        header.addWidget(title)

        header.addStretch(1)

        self._active_pill = QLabel("Active", self)
        self._active_pill.setProperty("role", "pill-active")
        self._active_pill.setProperty("state", "ready")
        self._active_pill.setAlignment(Qt.AlignCenter)
        self._active_pill.setVisible(False)
        header.addWidget(self._active_pill)

        root.addLayout(header)

        description = QLabel(info.description, self)
        description.setProperty("role", "muted")
        description.setWordWrap(True)
        root.addWidget(description)

        badges = QHBoxLayout()
        badges.setSpacing(6)
        for text in (
            f"speed: {info.speed}",
            f"quality: {info.quality}",
            f"size: {_format_size(info.size_mb)}",
            f"vram: {info.vram_gb:.1f} GB",
            f"lang: {info.languages}",
        ):
            badge = QLabel(text, self)
            badge.setProperty("role", "badge")
            badges.addWidget(badge)
        badges.addStretch(1)
        root.addLayout(badges)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._select_btn = QPushButton("Select", self)
        self._select_btn.setObjectName("SelectButton")
        self._select_btn.setProperty("role", "primary")
        self._select_btn.clicked.connect(
            lambda: self.select_requested.emit(self._info.alias)
        )
        footer.addWidget(self._select_btn)
        root.addLayout(footer)

    def alias(self) -> str:
        return self._info.alias

    def info(self) -> ModelInfo:
        return self._info

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.setProperty("active", self._active)
        self._active_pill.setVisible(self._active)
        self._select_btn.setVisible(not self._active)
        self._select_btn.setEnabled(not self._active and not self._locked)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        # Active cards keep Select hidden regardless; for inactive ones,
        # locking disables the button.
        if not self._active:
            self._select_btn.setEnabled(not self._locked)

    def is_loading(self) -> bool:
        return self._loading

    def set_loading(self, loading: bool) -> None:
        """Reflect backend load state on the active pill — swap 'Active' for
        'Loading…' with a different colour while the model is loading."""
        self._loading = bool(loading)
        if self._loading:
            self._active_pill.setText("Loading\u2026")
            self._active_pill.setProperty("state", "loading")
        else:
            self._active_pill.setText("Active")
            self._active_pill.setProperty("state", "ready")
        self._active_pill.style().unpolish(self._active_pill)
        self._active_pill.style().polish(self._active_pill)
