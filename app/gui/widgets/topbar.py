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

from app.gui.widgets.resource_widget import ResourceWidget
from app.gui.widgets.vu_meter import VUMeter


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


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _format_progress(current: int, total: int) -> str:
    """Compact progress label suitable for appending to 'Loading model…'."""
    if total > 0 and current >= 0:
        pct = int(min(99, max(0, current * 100 // total)))
        return f"{pct}%"
    if current > 0:
        return _format_size(current)
    return ""


class TopBar(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        # Slightly taller so pills don't kiss the OS title bar above —
        # 44 px was just enough to fit a pill at all, with no
        # breathing room.
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # The sidebar already highlights the active section name and
        # the OS title bar shows "Lazy to Text", so the topbar's left
        # area stays empty — duplicating the label was just visual
        # noise.
        layout.addStretch(1)

        # Resource stats — sits at the far left of the right-side
        # cluster so it's always visible without competing with the
        # recording / model pills for attention. Updated by the
        # ``ResourceMonitor`` that the controller owns.
        self._resources = ResourceWidget(self)
        layout.addWidget(self._resources)

        # Slim live-input meter — visible only while a recording is in
        # flight. Sits next to the recording pill so the eye associates
        # the two.
        self._vu_meter = VUMeter(self)
        self._vu_meter.setVisible(False)
        layout.addWidget(self._vu_meter)

        self._recording_pill = QLabel("", self)
        self._recording_pill.setObjectName("TopBarRecordingPill")
        self._recording_pill.setProperty("role", "recording-pill")
        self._recording_pill.setProperty("state", "idle")
        self._recording_pill.setAlignment(Qt.AlignCenter)
        self._recording_pill.setVisible(False)
        layout.addWidget(self._recording_pill)
        self._recording_state = "idle"
        self._loading_progress_text = ""
        self._loading_elapsed_s = 0

        self._model_pill = QLabel(_NO_MODEL_TEXT, self)
        self._model_pill.setObjectName("TopBarModelPill")
        self._model_pill.setProperty("role", "model-pill")
        self._model_pill.setProperty("state", "empty")
        self._model_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._model_pill)

        self._status_pill = QLabel(_STATUS_DEFAULT_LABELS["unknown"], self)
        self._status_pill.setObjectName("TopBarStatusPill")
        self._status_pill.setProperty("role", "status-pill")
        self._status_pill.setProperty("status", "unknown")
        self._status_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_pill)

    # ---- public API ---------------------------------------------------------

    def _compose_label(self, state: str) -> str:
        base = _RECORDING_LABELS.get(state, "")
        if state != "model_loading":
            return base
        if self._loading_progress_text:
            return f"{base} {self._loading_progress_text}"
        if self._loading_elapsed_s > 0:
            return f"{base} {self._loading_elapsed_s}s"
        return base

    def set_active_model(self, display_name: Optional[str]) -> None:
        if display_name:
            self._model_pill.setText(f"Current model: {display_name}")
            self._model_pill.setProperty("state", "active")
        else:
            self._model_pill.setText(_NO_MODEL_TEXT)
            self._model_pill.setProperty("state", "empty")
        # ``setProperty`` doesn't trigger a style refresh on its own.
        self._model_pill.style().unpolish(self._model_pill)
        self._model_pill.style().polish(self._model_pill)

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
            self._loading_progress_text = ""
            self._loading_elapsed_s = 0
        else:
            if state != "model_loading":
                # Reset loading state inputs when the pill is repurposed
                # for a non-loading mode (recording / processing).
                self._loading_progress_text = ""
                self._loading_elapsed_s = 0
            self._recording_pill.setText(self._compose_label(state))
            self._recording_pill.setProperty("state", state)
            self._recording_pill.setVisible(True)
        self._recording_pill.style().unpolish(self._recording_pill)
        self._recording_pill.style().polish(self._recording_pill)
        self._recording_state = state
        # VU meter only matters while audio is actively flowing in.
        if state == "recording":
            self._vu_meter.setVisible(True)
        else:
            self._vu_meter.setVisible(False)
            self._vu_meter.reset()

    def set_input_level(self, level: float) -> None:
        """Push a fresh amplitude reading into the VU meter — called
        from a Qt-side polling timer that reads
        ``AudioRecorder.current_input_level`` while recording."""
        self._vu_meter.set_level(level)

    def set_resource_metrics(self, metrics: dict) -> None:
        """Forward a sample from ``ResourceMonitor`` into the live
        stats widget."""
        self._resources.set_metrics(metrics)

    def set_loading_progress(self, current: int, total: int) -> None:
        """Update the loading-state pill with download progress.

        ``current`` / ``total`` are byte counts; ``total == 0`` (unknown
        size) renders as ``Loading model… (12 MB)``, otherwise as
        ``Loading model… 35%``. Has no visible effect unless the pill is
        currently in ``model_loading`` state.
        """
        self._loading_progress_text = _format_progress(current, total)
        if self._recording_state == "model_loading":
            self._recording_pill.setText(self._compose_label("model_loading"))

    def set_loading_elapsed(self, seconds: int) -> None:
        """Update the loading pill with elapsed seconds.

        Used as a fallback when the backend is in ``model_loading`` but
        no tqdm progress has fired (cached-model deserialisation).
        Bytes-based progress, when available, takes priority over
        elapsed time inside ``_compose_label``.
        """
        self._loading_elapsed_s = max(0, int(seconds))
        if self._recording_state == "model_loading":
            self._recording_pill.setText(self._compose_label("model_loading"))
