"""Inline 'Inference settings' panel for the active model card.

Five controls — language / VAD filter / beam size / temperature /
initial prompt — laid out as two compact rows. Emits a single
``settings_changed`` signal whenever any field commits a change so
the controller can persist + push to the live backend in one shot.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.inference_settings import InferenceSettings


# Curated subset of the ~99 languages Whisper supports — top
# spoken languages first, ``Auto`` is the default. The dropdown
# shows the human label; the ``data`` field carries the ISO code
# (or ``None`` for auto-detect) that's plumbed into ``transcribe``.
_LANGUAGES: tuple[tuple[str, Optional[str]], ...] = (
    ("Auto-detect", None),
    ("Russian", "ru"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Portuguese", "pt"),
    ("Italian", "it"),
    ("Polish", "pl"),
    ("Dutch", "nl"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
)


class InferenceSettingsPanel(QFrame):
    settings_changed = Signal(InferenceSettings)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("InferenceSettingsPanel")
        self.setProperty("role", "inference-panel")
        self.setFrameShape(QFrame.NoFrame)

        # Programmatic updates suppress the signal so we don't
        # round-trip a save when the controller pre-fills the panel.
        self._suspend_emit = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(8)

        title = QLabel("Inference settings", self)
        title.setProperty("role", "section-header")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        # Row 0: Language | VAD
        grid.addWidget(QLabel("Language", self), 0, 0)
        self._language = QComboBox(self)
        self._language.setObjectName("LanguageCombo")
        for label, code in _LANGUAGES:
            self._language.addItem(label, code)
        self._language.currentIndexChanged.connect(self._on_changed)
        grid.addWidget(self._language, 0, 1)

        self._vad = QCheckBox("VAD filter (skip silence)", self)
        self._vad.setObjectName("VadFilterCheckbox")
        self._vad.toggled.connect(self._on_changed)
        grid.addWidget(self._vad, 0, 2, 1, 2)

        # Row 1: Beam size | Temperature
        grid.addWidget(QLabel("Beam size", self), 1, 0)
        self._beam = QSpinBox(self)
        self._beam.setObjectName("BeamSizeSpin")
        self._beam.setRange(1, 20)
        self._beam.valueChanged.connect(self._on_changed)
        grid.addWidget(self._beam, 1, 1)

        grid.addWidget(QLabel("Temperature", self), 1, 2)
        self._temperature = QDoubleSpinBox(self)
        self._temperature.setObjectName("TemperatureSpin")
        self._temperature.setRange(0.0, 1.0)
        self._temperature.setSingleStep(0.1)
        self._temperature.setDecimals(1)
        self._temperature.valueChanged.connect(self._on_changed)
        grid.addWidget(self._temperature, 1, 3)

        # Row 2: Initial prompt — full width
        grid.addWidget(QLabel("Initial prompt", self), 2, 0)
        self._prompt = QLineEdit(self)
        self._prompt.setObjectName("InitialPromptEdit")
        self._prompt.setPlaceholderText(
            "Custom vocabulary / context (optional)"
        )
        self._prompt.editingFinished.connect(self._on_changed)
        grid.addWidget(self._prompt, 2, 1, 1, 3)

        outer.addLayout(grid)

    # ---- public API ---------------------------------------------------------

    def set_settings(self, settings: InferenceSettings) -> None:
        """Mirror ``settings`` onto the controls without firing the
        ``settings_changed`` signal — used by the controller when it
        pre-fills the panel from config."""
        self._suspend_emit = True
        try:
            target = settings.language
            for i in range(self._language.count()):
                if self._language.itemData(i) == target:
                    self._language.setCurrentIndex(i)
                    break
            else:
                # Unknown code (e.g. ``"sw"``); fall back to Auto.
                self._language.setCurrentIndex(0)
            self._vad.setChecked(bool(settings.vad_filter))
            self._beam.setValue(int(settings.beam_size))
            self._temperature.setValue(float(settings.temperature))
            self._prompt.setText(settings.initial_prompt or "")
        finally:
            self._suspend_emit = False

    def set_enabled_for_engine(self, enabled: bool, reason: str = "") -> None:
        """Greys out every control — used for engines (GigaAM) that
        don't accept any of these knobs at inference time. Optional
        ``reason`` becomes the panel's tooltip so the user knows why."""
        for widget in (
            self._language,
            self._vad,
            self._beam,
            self._temperature,
            self._prompt,
        ):
            widget.setEnabled(bool(enabled))
        if reason:
            self.setToolTip(reason)
        else:
            self.setToolTip("")

    def values(self) -> InferenceSettings:
        return InferenceSettings(
            language=self._language.currentData(),
            vad_filter=self._vad.isChecked(),
            initial_prompt=(self._prompt.text().strip() or None),
            beam_size=int(self._beam.value()),
            temperature=float(self._temperature.value()),
        )

    # ---- internal -----------------------------------------------------------

    def _on_changed(self, *_args) -> None:
        if self._suspend_emit:
            return
        self.settings_changed.emit(self.values())
