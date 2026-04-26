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
The slot always shows its current state, the same way the
topbar's CPU / RAM / GPU resource graphs always show their
current values — hiding everything in idle made the slot look
like a blank hole. Pill is always visible with a state-driven
label; meter is always visible with its bar at 0 unless audio
is actively flowing in.

- ``idle``          — pill 'Idle' (muted),  meter visible, level 0
- ``recording``     — pill '● Recording',   meter live
- ``processing``    — pill 'Processing…',   meter visible, level 0
- ``model_loading`` — same look as idle (the actual loading state
  is reflected on the topbar's model pill; we don't compete)
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

# Pill text per active state. ``idle`` and ``model_loading`` both
# render as the muted Idle pill — the topbar already shows
# ``Loading: <model>`` during model_loading and we don't want to
# compete with it.
_PILL_LABELS = {
    "idle": "● Idle",
    "model_loading": "● Idle",
    "recording": "● Recording",
    "processing": "Processing…",
}
# Which property value the pill carries — drives the QSS variant.
_PILL_VARIANTS = {
    "idle": "idle",
    "model_loading": "idle",
    "recording": "recording",
    "processing": "processing",
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

        self._pill = QLabel(_PILL_LABELS["idle"], self)
        self._pill.setObjectName("RecordingStatusPill")
        self._pill.setProperty("role", "recording-pill")
        self._pill.setProperty("state", "idle")
        # Text inside the pill is centered (looks balanced inside the
        # rounded chip) but the pill chip itself hugs the left edge of
        # the slot — same column as the sidebar nav items above.
        self._pill.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._pill, 0, Qt.AlignLeft)

        self._vu_meter = VUMeter(self)
        layout.addWidget(self._vu_meter, 0, Qt.AlignLeft)

        layout.addStretch(1)

    # ---- public API ---------------------------------------------------------

    def set_recording_state(self, state: str) -> None:
        if state not in _RECORDING_STATES:
            raise ValueError(
                f"state must be one of {_RECORDING_STATES}, got {state!r}"
            )
        # Pill is always visible — placeholder semantics. Label and
        # variant change with state.
        self._pill.setText(_PILL_LABELS[state])
        self._pill.setProperty("state", _PILL_VARIANTS[state])
        self._pill.style().unpolish(self._pill)
        self._pill.style().polish(self._pill)

        # Meter is always visible too — its bar just sits at 0
        # whenever audio isn't actively being captured.
        if state != "recording":
            self._vu_meter.reset()

    def set_input_level(self, level: float) -> None:
        """Push a fresh amplitude reading into the VU meter."""
        self._vu_meter.set_level(level)
