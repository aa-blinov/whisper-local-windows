"""Recording status chip for the sidebar's bottom-left corner.

Same visual shape as the CPU / RAM / GPU chips in the topbar's
``ResourceWidget`` — a single rounded card with a muted label on
the top-left, the state value on the top-right, and a thin
progress bar (the live VU meter) running along the bottom.
Reads as a sibling of the resource graphs rather than a pair of
free-floating chips.

States
------
The chip is always present in the slot — same convention as the
resource graphs (they show 0% rather than disappearing). Only the
text and bar fill change with state:

- ``idle``          — value 'Idle'         (muted), bar at 0
- ``recording``     — value '● Recording'  (red),   bar live
- ``processing``    — value '● Processing' (accent), bar at 0
- ``model_loading`` — same look as idle (the topbar's model pill
  already shows the load state; we don't compete)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.vu_meter import VUMeter


_RECORDING_STATES = ("idle", "recording", "processing", "model_loading")

# Pill text per active state — model_loading collapses onto the
# idle look so we don't compete with the topbar's loading pill.
_PILL_LABELS = {
    "idle": "● Idle",
    "model_loading": "● Idle",
    "recording": "● Recording",
    "processing": "● Processing",
}
_PILL_VARIANTS = {
    "idle": "idle",
    "model_loading": "idle",
    "recording": "recording",
    "processing": "processing",
}

# Fixed height for the whole chip — matches the visual weight of
# ResourceWidget's 32-px chip plus a row for the VU bar.
_CHIP_HEIGHT_PX = 48


class RecordingStatusWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecordingStatusWidget")
        self.setFixedHeight(_CHIP_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        # Top row — STATUS label on the left, state value on the right.
        # Mirrors how ResourceWidget paints "CPU  12%".
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._label = QLabel("STATUS", self)
        self._label.setObjectName("RecordingStatusLabel")
        self._label.setProperty("role", "chip-label")
        row.addWidget(self._label)
        row.addStretch(1)

        self._pill = QLabel(_PILL_LABELS["idle"], self)
        self._pill.setObjectName("RecordingStatusPill")
        self._pill.setProperty("role", "chip-value")
        self._pill.setProperty("state", "idle")
        self._pill.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._pill)

        outer.addLayout(row)

        # VU meter sits below the row, full width of the chip.
        self._vu_meter = VUMeter(self)
        outer.addWidget(self._vu_meter)

    # ---- public API ---------------------------------------------------------

    def set_recording_state(self, state: str) -> None:
        if state not in _RECORDING_STATES:
            raise ValueError(
                f"state must be one of {_RECORDING_STATES}, got {state!r}"
            )
        self._pill.setText(_PILL_LABELS[state])
        self._pill.setProperty("state", _PILL_VARIANTS[state])
        self._pill.style().unpolish(self._pill)
        self._pill.style().polish(self._pill)

        # Bar runs only during active capture; idle / processing /
        # model_loading all sit at 0 — they share the same chip
        # placeholder look.
        if state != "recording":
            self._vu_meter.reset()

    def set_input_level(self, level: float) -> None:
        """Push a fresh amplitude reading into the VU meter."""
        self._vu_meter.set_level(level)
