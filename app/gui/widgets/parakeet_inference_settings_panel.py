"""Inline 'Inference settings' panel for NeMo-backed model cards.

Parakeet TDT v3 (and the Canary family) ship a much narrower
inference API than Whisper: language is auto-detected, decoding
is greedy-only, no temperature / initial-prompt knobs. The single
useful tunable is ``timestamps`` — when enabled, NeMo populates
the per-word and per-segment offsets on each result.

We keep the panel mounted in the same spot as
``InferenceSettingsPanel`` (the active model card) and emit the
same kind of ``settings_changed`` signal so the controller's
persistence + live-push path doesn't have to special-case the
backend kind.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFrame, QLabel, QVBoxLayout, QWidget

from app.inference_settings import ParakeetInferenceSettings


class ParakeetInferenceSettingsPanel(QFrame):
    settings_changed = Signal(ParakeetInferenceSettings)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ParakeetInferenceSettingsPanel")
        # Reuse the same QSS role as the Whisper panel — same outer
        # treatment (subtle bg, padding) so both panels look like
        # they belong on the active card.
        self.setProperty("role", "inference-panel")
        self.setFrameShape(QFrame.NoFrame)

        self._suspend_emit = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(8)

        title = QLabel("Inference settings", self)
        title.setProperty("role", "section-header")
        outer.addWidget(title)

        # The single knob: timestamps. Off by default — there's a
        # small but nonzero overhead per frame for the alignment
        # post-processing. Most dictation use-cases don't need them.
        self._timestamps = QCheckBox(
            "Include word + segment timestamps in output",
            self,
        )
        self._timestamps.setObjectName("NemoTimestampsCheckbox")
        self._timestamps.toggled.connect(self._on_changed)
        outer.addWidget(self._timestamps)

        hint = QLabel(
            "NeMo auto-detects the source language and uses greedy "
            "decoding — no language / beam / temperature knobs to "
            "tune. Toggle timestamps if you need the per-word "
            "offsets in the result.",
            self,
        )
        hint.setObjectName("NemoSettingsHint")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        outer.addWidget(hint)

    # ---- public API ---------------------------------------------------------

    def set_settings(self, settings: ParakeetInferenceSettings) -> None:
        """Mirror ``settings`` onto the controls without firing the
        ``settings_changed`` signal — used by the controller when it
        pre-fills the panel from config."""
        self._suspend_emit = True
        try:
            self._timestamps.setChecked(bool(settings.timestamps))
        finally:
            self._suspend_emit = False

    def values(self) -> ParakeetInferenceSettings:
        return ParakeetInferenceSettings(
            timestamps=self._timestamps.isChecked(),
        )

    # ---- internal -----------------------------------------------------------

    def _on_changed(self, *_args) -> None:
        if self._suspend_emit:
            return
        self.settings_changed.emit(self.values())
