"""Top bar — recording status, current/loading model, backend health, stats."""

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


# Sidebar's fixed width — duplicated here so the topbar can leave
# the matching gap on its left edge and continue the sidebar's
# right border line up through itself. Updating this without
# updating ``Sidebar.setFixedWidth`` would visually misalign the
# divider, so it's pinned alongside it.
_SIDEBAR_WIDTH_PX = 200


_STATUS_VALUES = ("running", "stopped", "error", "unknown", "hidden")
_STATUS_DEFAULT_LABELS = {
    "running": "Backend running",
    "stopped": "Backend stopped",
    "error": "Backend error",
    "unknown": "Status unknown",
    "hidden": "",
}
_NO_MODEL_TEXT = "No model"

# The recording pill no longer handles model-load progress — that
# moved into the model pill below to avoid showing two near-
# duplicate "Loading model…" indicators side by side.
_RECORDING_STATES = ("idle", "recording", "processing", "model_loading")
_RECORDING_LABELS = {
    "recording": "● Recording",
    "processing": "Processing…",
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
    """Compact progress label appended to the loading model pill."""
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
        # Slightly taller so pills don't kiss the OS title bar above.
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Outer layout has no padding so the gutter / divider extend
        # to the very top and bottom of the topbar — the divider
        # then lines up edge-to-edge with the sidebar's
        # ``border-right`` below.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Sidebar-aligned blank zone -------------------------------
        # Eats the same horizontal space as the sidebar so the
        # divider after it lands exactly on the sidebar's right
        # border line.
        left_gutter = QWidget(self)
        left_gutter.setFixedWidth(_SIDEBAR_WIDTH_PX)
        outer.addWidget(left_gutter)

        divider = QWidget(self)
        divider.setObjectName("TopBarDivider")
        divider.setFixedWidth(1)
        # ``WA_StyledBackground`` lets QSS's ``background-color``
        # rule actually paint — without it Qt treats the widget as
        # unstyled-bg and the line stays invisible.
        divider.setAttribute(Qt.WA_StyledBackground, True)
        outer.addWidget(divider)

        # Right-hand content lives in its own container so it can
        # carry its own padding without cropping the divider.
        content = QWidget(self)
        content.setObjectName("TopBarContent")
        layout = QHBoxLayout(content)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)

        # Resource stats — anchored to the leading edge of the
        # content area, immediately right of the divider.
        self._resources = ResourceWidget(content)
        layout.addWidget(self._resources)

        # Push the rest of the topbar (recording / model / status)
        # to the right.
        layout.addStretch(1)

        # Slim live-input meter — visible only while a recording is
        # in flight. Sits next to the recording pill so the eye
        # associates the two.
        self._vu_meter = VUMeter(content)
        self._vu_meter.setVisible(False)
        layout.addWidget(self._vu_meter)

        self._recording_pill = QLabel("", content)
        self._recording_pill.setObjectName("TopBarRecordingPill")
        self._recording_pill.setProperty("role", "recording-pill")
        self._recording_pill.setProperty("state", "idle")
        self._recording_pill.setAlignment(Qt.AlignCenter)
        self._recording_pill.setVisible(False)
        layout.addWidget(self._recording_pill)
        self._recording_state = "idle"

        # Model pill: empty / loading / active. ``loading`` shows
        # download progress + elapsed time inline so the topbar
        # doesn't double up on "Loading model…" labels.
        self._model_pill = QLabel(_NO_MODEL_TEXT, content)
        self._model_pill.setObjectName("TopBarModelPill")
        self._model_pill.setProperty("role", "model-pill")
        self._model_pill.setProperty("state", "empty")
        self._model_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._model_pill)
        self._model_state: str = "empty"
        self._model_display_name: Optional[str] = None
        self._loading_progress_text = ""
        self._loading_elapsed_s = 0

        self._status_pill = QLabel(_STATUS_DEFAULT_LABELS["unknown"], content)
        self._status_pill.setObjectName("TopBarStatusPill")
        self._status_pill.setProperty("role", "status-pill")
        self._status_pill.setProperty("status", "unknown")
        self._status_pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_pill)

        outer.addWidget(content, 1)

    # ---- public API ---------------------------------------------------------

    def set_active_model(self, display_name: Optional[str]) -> None:
        """Mark the model as the currently-loaded one (the green-blue
        accent). Call ``set_recording_state('model_loading')`` to
        flip the pill into its yellow loading variant — that's the
        single source of truth for the loading state, not a separate
        recording-pill label."""
        self._model_display_name = display_name
        if self._model_state == "loading":
            # Keep the loading variant visible — the controller will
            # call ``set_recording_state('idle')`` when ready, which
            # will also drop us out of loading on the model pill.
            self._render_model_pill()
            return
        self._render_model_pill(state_override="active" if display_name else "empty")

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
        # ``model_loading`` is reflected in the model pill, not in
        # the recording pill — the latter only shows actively-
        # recording / actively-processing states.
        if state in _RECORDING_LABELS:
            self._recording_pill.setText(_RECORDING_LABELS[state])
            self._recording_pill.setProperty("state", state)
            self._recording_pill.setVisible(True)
        else:
            self._recording_pill.setVisible(False)
            self._recording_pill.setProperty("state", "idle")
        self._recording_pill.style().unpolish(self._recording_pill)
        self._recording_pill.style().polish(self._recording_pill)
        self._recording_state = state

        # Model pill loading state mirrors the backend's
        # ``model_loading`` phase exactly.
        if state == "model_loading":
            self._render_model_pill(state_override="loading")
        else:
            # Drop loading text when leaving the loading state.
            if self._model_state == "loading":
                self._loading_progress_text = ""
                self._loading_elapsed_s = 0
                self._render_model_pill(
                    state_override="active" if self._model_display_name else "empty"
                )

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
        """Update the model-pill's loading variant with download
        progress. ``total == 0`` (unknown size) renders as e.g.
        ``Loading: Tiny (test)  12 MB``, otherwise ``35%``.

        No visible effect unless the model pill is in the
        ``loading`` state (i.e. ``set_recording_state('model_loading')``
        was called).
        """
        self._loading_progress_text = _format_progress(current, total)
        if self._model_state == "loading":
            self._render_model_pill()

    def set_loading_elapsed(self, seconds: int) -> None:
        """Update the model-pill's loading variant with an elapsed
        seconds counter — fallback when no tqdm progress is firing
        (cached-model deserialisation). Bytes-progress wins over
        elapsed inside ``_render_model_pill``.
        """
        self._loading_elapsed_s = max(0, int(seconds))
        if self._model_state == "loading":
            self._render_model_pill()

    # ---- internal -----------------------------------------------------------

    def _render_model_pill(self, state_override: Optional[str] = None) -> None:
        state = state_override if state_override is not None else self._model_state
        if state == "loading":
            name = self._model_display_name or "model"
            text = f"Loading: {name}"
            if self._loading_progress_text:
                text += f"  {self._loading_progress_text}"
            elif self._loading_elapsed_s > 0:
                text += f"  {self._loading_elapsed_s}s"
        elif state == "active" and self._model_display_name:
            text = f"Current model: {self._model_display_name}"
        else:
            text = _NO_MODEL_TEXT
            state = "empty"
        self._model_pill.setText(text)
        self._model_pill.setProperty("state", state)
        self._model_pill.style().unpolish(self._model_pill)
        self._model_pill.style().polish(self._model_pill)
        self._model_state = state
