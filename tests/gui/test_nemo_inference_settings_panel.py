"""Tests for the NeMo inference-settings panel."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox


def test_panel_starts_with_unchecked_timestamps(qtbot):
    """Default state — no timestamps. Off-by-default keeps the
    common path fast (small but nonzero post-processing cost
    for word/segment alignment)."""
    from app.gui.widgets.nemo_inference_settings_panel import (
        NemoInferenceSettingsPanel,
    )

    panel = NemoInferenceSettingsPanel()
    qtbot.addWidget(panel)
    cb = panel.findChild(QCheckBox, "NemoTimestampsCheckbox")
    assert cb is not None
    assert cb.isChecked() is False


def test_set_settings_pre_fills_panel(qtbot):
    from app.gui.widgets.nemo_inference_settings_panel import (
        NemoInferenceSettingsPanel,
    )
    from app.inference_settings import NemoInferenceSettings

    panel = NemoInferenceSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_settings(NemoInferenceSettings(timestamps=True))

    cb = panel.findChild(QCheckBox, "NemoTimestampsCheckbox")
    assert cb.isChecked() is True


def test_set_settings_does_not_re_emit(qtbot):
    """Programmatic prefill must not echo a save back through the
    signal — would cause a feedback loop on init when the
    controller paints persisted overrides into the panel."""
    from app.gui.widgets.nemo_inference_settings_panel import (
        NemoInferenceSettingsPanel,
    )
    from app.inference_settings import NemoInferenceSettings

    panel = NemoInferenceSettingsPanel()
    qtbot.addWidget(panel)

    emissions: list = []
    panel.settings_changed.connect(emissions.append)
    panel.set_settings(NemoInferenceSettings(timestamps=True))
    assert emissions == []


def test_toggling_checkbox_emits_settings_changed(qtbot):
    from app.gui.widgets.nemo_inference_settings_panel import (
        NemoInferenceSettingsPanel,
    )

    panel = NemoInferenceSettingsPanel()
    qtbot.addWidget(panel)
    cb = panel.findChild(QCheckBox, "NemoTimestampsCheckbox")

    with qtbot.waitSignal(panel.settings_changed, timeout=1000) as blocker:
        cb.setChecked(True)

    settings = blocker.args[0]
    assert settings.timestamps is True


def test_values_returns_current_state(qtbot):
    from app.gui.widgets.nemo_inference_settings_panel import (
        NemoInferenceSettingsPanel,
    )

    panel = NemoInferenceSettingsPanel()
    qtbot.addWidget(panel)
    cb = panel.findChild(QCheckBox, "NemoTimestampsCheckbox")
    cb.setChecked(True)

    assert panel.values().timestamps is True
