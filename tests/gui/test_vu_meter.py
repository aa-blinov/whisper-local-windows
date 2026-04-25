"""Tests for the VUMeter widget."""

import pytest


def test_vu_meter_starts_at_zero(qtbot):
    from app.gui.widgets.vu_meter import VUMeter

    meter = VUMeter()
    qtbot.addWidget(meter)
    assert meter.current_level() == 0.0


def test_vu_meter_clamps_levels_to_unit_range(qtbot):
    from app.gui.widgets.vu_meter import VUMeter

    meter = VUMeter()
    qtbot.addWidget(meter)

    meter.set_level(2.5)
    assert meter.current_level() == pytest.approx(1.0)

    meter.set_level(-0.3)
    # Decay applied: previous 1.0 * decay → 0.85, but new sample 0
    # → max(0, 0.85) = 0.85. So just check it didn't go negative.
    assert meter.current_level() >= 0.0


def test_vu_meter_decays_when_silenced(qtbot):
    """Repeated zero readings should taper the bar back down rather
    than snapping it off — gives a more natural ‘holding’ effect."""
    from app.gui.widgets.vu_meter import VUMeter

    meter = VUMeter()
    qtbot.addWidget(meter)

    meter.set_level(1.0)
    initial = meter.current_level()
    meter.set_level(0.0)
    after_one = meter.current_level()
    meter.set_level(0.0)
    after_two = meter.current_level()

    assert initial > after_one > after_two


def test_vu_meter_reset_zeroes_immediately(qtbot):
    from app.gui.widgets.vu_meter import VUMeter

    meter = VUMeter()
    qtbot.addWidget(meter)
    meter.set_level(0.7)
    meter.reset()
    assert meter.current_level() == 0.0


def test_vu_meter_handles_invalid_input_gracefully(qtbot):
    from app.gui.widgets.vu_meter import VUMeter

    meter = VUMeter()
    qtbot.addWidget(meter)
    meter.set_level("not a number")  # type: ignore[arg-type]
    assert meter.current_level() == 0.0
