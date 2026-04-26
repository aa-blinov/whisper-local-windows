"""Tests for the RecordingStatusWidget — a fixed-height container in
the sidebar's bottom-left slot that holds the recording pill stacked
vertically over the live VU meter.

Replaces the recording-pill + VU meter combo that used to live in the
TopBar (where it overlapped the resource graphs). The widget always
occupies its placeholder slot so the sidebar layout doesn't jump
when audio capture starts/stops; the contents inside just show or
hide based on state.
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QLabel


def _label_by_name(widget, name: str) -> QLabel:
    return widget.findChild(QLabel, name)


def test_widget_has_recording_pill_and_vu_meter(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget
    from app.gui.widgets.vu_meter import VUMeter

    w = RecordingStatusWidget()
    qtbot.addWidget(w)

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill is not None

    meter = w.findChild(VUMeter)
    assert meter is not None


def test_widget_has_fixed_placeholder_height(qtbot):
    """The slot in the sidebar must reserve constant vertical space so
    the navigation list doesn't jump when audio starts/stops. A
    ``setMinimumHeight`` >= 50 px is enough for pill + meter + spacing."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)

    # Either a fixed height or a minimum that won't collapse.
    min_h = w.minimumHeight()
    assert min_h >= 50, (
        f"placeholder must reserve >=50 px height, got {min_h}"
    )


def test_idle_state_hides_pill_and_meter_but_keeps_widget(qtbot):
    """In idle the pill text and the VU bar are hidden — but the
    widget itself stays in place (it is the placeholder)."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("idle")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert not pill.isVisibleTo(w)
    assert not w._vu_meter.isVisibleTo(w)
    # Widget itself is still visible — the placeholder slot.
    assert w.isVisible()


def test_recording_state_shows_pill_and_meter(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("recording")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "recording"
    assert "recording" in pill.text().lower()
    assert w._vu_meter.isVisibleTo(w)


def test_processing_state_shows_pill_hides_meter(qtbot):
    """During Processing the audio buffer is already captured — the
    VU meter has nothing live to display, so hide it. The pill
    still indicates the work-in-progress."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("processing")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "processing"
    assert "processing" in pill.text().lower()
    assert not w._vu_meter.isVisibleTo(w)


def test_model_loading_state_keeps_pill_and_meter_hidden(qtbot):
    """``model_loading`` is reflected in the topbar's model pill, not
    here — the recording status slot stays empty so the user isn't
    confronted with two competing indicators."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("model_loading")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert not pill.isVisibleTo(w)
    assert not w._vu_meter.isVisibleTo(w)


def test_back_to_idle_hides_pill_again(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("recording")
    w.set_recording_state("idle")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert not pill.isVisibleTo(w)
    assert not w._vu_meter.isVisibleTo(w)


def test_set_recording_state_rejects_unknown(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    with pytest.raises(ValueError):
        w.set_recording_state("snoozing")


def test_set_input_level_forwards_to_meter(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.set_recording_state("recording")
    w.set_input_level(0.6)
    assert w._vu_meter.current_level() == 0.6


def test_meter_resets_when_leaving_recording_state(qtbot):
    """Same contract the topbar previously held — leaving the
    recording state must zero the meter so a stale reading from the
    last capture doesn't linger when the next one starts."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.set_recording_state("recording")
    w.set_input_level(0.6)
    w.set_recording_state("idle")
    # current_level() should drop back to 0 after reset().
    assert w._vu_meter.current_level() == 0.0
