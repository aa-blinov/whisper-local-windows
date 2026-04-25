"""Top bar — shows app title, current model, and backend status."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)


_STATUS_VALUES = ("running", "stopped", "error", "unknown", "hidden")
_STATUS_DEFAULT_LABELS = {
    "running": "Backend running",
    "stopped": "Backend stopped",
    "error": "Backend error",
    "unknown": "Status unknown",
    "hidden": "",
}
_NO_MODEL_TEXT = "No model"

_RECORDING_STATES = ("idle", "recording", "processing", "model_loading")
_RECORDING_LABELS = {
    "recording": "● Recording",
    "processing": "Processing…",
    "model_loading": "Loading model…",
}


class TopBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        # Left: contextual section title (driven by sidebar selection).
        # The OS window title bar already shows the app name, so we use this
        # area for "where am I" navigation context instead.
        self._section_title = QLabel("", self)
        self._section_title.setObjectName("TopBarSectionTitle")
        self._section_title.setProperty("role", "section-title")
        layout.addWidget(self._section_title)

        layout.addStretch(1)

        self._recording_pill = QLabel("", self)
        self._recording_pill.setObjectName("TopBarRecordingPill")
        self._recording_pill.setProperty("role", "recording-pill")
        self._recording_pill.setProperty("state", "idle")
        self._recording_pill.setAlignment(Qt.AlignCenter)
        self._recording_pill.setVisible(False)
        layout.addWidget(self._recording_pill)

        self._model_pill = QLabel(_NO_MODEL_TEXT, self)
        self._model_pill.setObjectName("TopBarModelPill")
        self._model_pill.setProperty("role", "badge")
        self._model_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._model_pill)

        self._status_pill = QLabel(_STATUS_DEFAULT_LABELS["unknown"], self)
        self._status_pill.setObjectName("TopBarStatusPill")
        self._status_pill.setProperty("role", "status-pill")
        self._status_pill.setProperty("status", "unknown")
        self._status_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_pill)

    # ---- public API ---------------------------------------------------------

    def set_section_title(self, text: str) -> None:
        self._section_title.setText(text)

    def set_active_model(self, display_name: Optional[str]) -> None:
        if display_name:
            self._model_pill.setText(f"Model: {display_name}")
        else:
            self._model_pill.setText(_NO_MODEL_TEXT)

    def set_backend_status(
        self,
        status: str,
        label: Optional[str] = None,
    ) -> None:
        if status not in _STATUS_VALUES:
            raise ValueError(
                f"status must be one of {_STATUS_VALUES}, got {status!r}"
            )
        if status == "hidden":
            self._status_pill.setVisible(False)
            return
        self._status_pill.setVisible(True)
        self._status_pill.setProperty("status", status)
        self._status_pill.setText(label or _STATUS_DEFAULT_LABELS[status])
        self._status_pill.style().unpolish(self._status_pill)
        self._status_pill.style().polish(self._status_pill)

    def set_recording_state(self, state: str) -> None:
        if state not in _RECORDING_STATES:
            raise ValueError(
                f"state must be one of {_RECORDING_STATES}, got {state!r}"
            )
        if state == "idle":
            self._recording_pill.setVisible(False)
            self._recording_pill.setProperty("state", "idle")
        else:
            self._recording_pill.setText(_RECORDING_LABELS[state])
            self._recording_pill.setProperty("state", state)
            self._recording_pill.setVisible(True)
        self._recording_pill.style().unpolish(self._recording_pill)
        self._recording_pill.style().polish(self._recording_pill)
