"""Recording status slot for the sidebar's bottom-left corner.

Holds the recording-state pill stacked vertically over the live
VU meter. Replaces the equivalent two widgets that used to live in
the TopBar — they overlapped the resource graphs (CPU / RAM / GPU
bars) and pushed the model pill to the edge on narrow windows.

Sizing strategy
---------------
The widget reserves a fixed minimum height (placeholder) so that
toggling visibility of pill/meter content doesn't reflow the
sidebar's nav list. Idle state keeps the slot empty but the slot
itself stays in place.

States
------
- ``idle`` / ``model_loading`` — pill hidden, meter hidden, slot blank
- ``recording``                — pill 'Recording', meter live
- ``processing``               — pill 'Processing', meter hidden
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.vu_meter import VUMeter


_RECORDING_STATES = ("idle", "recording", "processing", "model_loading")
_RECORDING_LABELS = {
    "recording": "● Recording",
    "processing": "Processing…",
}

# Placeholder height — picked to fit the pill (~24 px) + spacing
# (~6 px) + VU meter (8 px) + outer margins, with a couple of pixels
# of breathing room. Constant so the sidebar nav list above doesn't
# jump when the slot's contents toggle visibility.
_SLOT_HEIGHT_PX = 56


class RecordingStatusWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecordingStatusWidget")
        self.setMinimumHeight(_SLOT_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 8)
        layout.setSpacing(4)

        self._pill = QLabel("", self)
        self._pill.setObjectName("RecordingStatusPill")
        self._pill.setProperty("role", "recording-pill")
        self._pill.setProperty("state", "idle")
        self._pill.setAlignment(Qt.AlignCenter)
        self._pill.setVisible(False)
        layout.addWidget(self._pill, 0, Qt.AlignHCenter)

        self._vu_meter = VUMeter(self)
        self._vu_meter.setVisible(False)
        layout.addWidget(self._vu_meter, 0, Qt.AlignHCenter)

        layout.addStretch(1)

    # ---- public API ---------------------------------------------------------

    def set_recording_state(self, state: str) -> None:
        if state not in _RECORDING_STATES:
            raise ValueError(
                f"state must be one of {_RECORDING_STATES}, got {state!r}"
            )
        if state in _RECORDING_LABELS:
            self._pill.setText(_RECORDING_LABELS[state])
            self._pill.setProperty("state", state)
            self._pill.setVisible(True)
        else:
            # idle / model_loading — slot stays empty, widget itself
            # remains in place as the placeholder.
            self._pill.setVisible(False)
            self._pill.setProperty("state", "idle")
        self._pill.style().unpolish(self._pill)
        self._pill.style().polish(self._pill)

        # Live VU meter only matters during active capture.
        if state == "recording":
            self._vu_meter.setVisible(True)
        else:
            self._vu_meter.setVisible(False)
            self._vu_meter.reset()

    def set_input_level(self, level: float) -> None:
        """Push a fresh amplitude reading into the VU meter."""
        self._vu_meter.set_level(level)
