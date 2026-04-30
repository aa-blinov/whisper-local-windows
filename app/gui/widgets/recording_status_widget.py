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

A second row above the STATUS line carries the active ONNX Runtime
EP (``Engine: CoreML / CUDA / CPU``) so the user sees both
"backend lifecycle" signals — what's running and on what — in one
cluster instead of having to glance at the topbar separately.
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

# Engine-pill defaults (no model loaded yet).
_NO_ENGINE_TEXT = "Engine: —"
# Provider names that count as "real" hardware acceleration. CPU
# is the muted fallback variant; everything else (including
# DirectML / ROCm / Azure for forward-compatibility) gets the
# accent green.
_ACCELERATOR_PROVIDERS = frozenset({
    "CUDA", "CoreML", "TensorRT", "DirectML", "ROCm", "Azure",
})

# Fixed height for the whole chip — matches the visual weight of
# ResourceWidget's 32-px chip plus a row for the VU bar plus the
# engine row above STATUS.
_CHIP_HEIGHT_PX = 72


class RecordingStatusWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("RecordingStatusWidget")
        self.setFixedHeight(_CHIP_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        # Top row — ENGINE label on the left, EP pill on the right.
        # Mirrors how ResourceWidget paints "CPU  12%".  Sits above
        # STATUS so the two backend signals (what's running, on
        # what) cluster together.
        engine_row = QHBoxLayout()
        engine_row.setContentsMargins(0, 0, 0, 0)
        engine_row.setSpacing(8)

        self._engine_label = QLabel("ENGINE", self)
        self._engine_label.setObjectName("EngineLabel")
        self._engine_label.setProperty("role", "chip-label")
        engine_row.addWidget(self._engine_label)
        engine_row.addStretch(1)

        # Same ``role="chip-value"`` as the STATUS pill below so the
        # two read as visually paired — just plain coloured text on
        # the chip background, no extra border / pill rectangle.
        self._engine_pill = QLabel("—", self)
        self._engine_pill.setObjectName("EnginePill")
        self._engine_pill.setProperty("role", "chip-value")
        self._engine_pill.setProperty("state", "idle")
        self._engine_pill.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._engine_pill.setToolTip(
            "ONNX Runtime execution provider used by the loaded model"
        )
        engine_row.addWidget(self._engine_pill)
        outer.addLayout(engine_row)

        # Middle row — STATUS label + recording-state pill.
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)

        self._label = QLabel("STATUS", self)
        self._label.setObjectName("RecordingStatusLabel")
        self._label.setProperty("role", "chip-label")
        status_row.addWidget(self._label)
        status_row.addStretch(1)

        self._pill = QLabel(_PILL_LABELS["idle"], self)
        self._pill.setObjectName("RecordingStatusPill")
        self._pill.setProperty("role", "chip-value")
        self._pill.setProperty("state", "idle")
        self._pill.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row.addWidget(self._pill)

        outer.addLayout(status_row)

        # VU meter sits below the rows, full width of the chip.
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

    def set_active_provider(self, provider: Optional[str]) -> None:
        """Update the engine pill with the EP currently backing the
        loaded model.  ``None`` (no model loaded yet) renders the
        muted ``● —`` placeholder.  Accelerator names (CoreML /
        CUDA / TensorRT / …) render in the accent green so a
        fallback to CPU is immediately visible to the user.

        Reuses the ``chip-value`` QSS role so the pill matches the
        STATUS pill's geometry — same font weight, same plain
        coloured text, no extra border.  Prefixes the provider name
        with the same ``●`` glyph the STATUS pill uses for visual
        rhythm: at the small font size, all-caps "CPU" reads taller
        than title-case "Idle" without the bullet so the rows look
        unbalanced; matching the bullet-prefix puts both rows on the
        same baseline.
        """
        if not provider:
            text = "● —"
            state = "idle"
        else:
            text = f"● {provider}"
            if provider in _ACCELERATOR_PROVIDERS:
                # Re-use the existing ``processing`` accent shade —
                # it's the closest match to "actively using the
                # accelerator" from the values already in dark.qss.
                state = "processing"
            else:
                state = "idle"
        self._engine_pill.setText(text)
        self._engine_pill.setProperty("state", state)
        self._engine_pill.style().unpolish(self._engine_pill)
        self._engine_pill.style().polish(self._engine_pill)
