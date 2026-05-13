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


def test_widget_has_status_label_pill_and_vu_meter(qtbot):
    """Chip layout — same shape as ResourceWidget's CPU/RAM blocks:
    a muted ``STATUS`` label on the top-left, the state value on the
    top-right, and a thin progress bar (the VU meter) underneath."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget
    from app.gui.widgets.vu_meter import VUMeter

    w = RecordingStatusWidget()
    qtbot.addWidget(w)

    label = _label_by_name(w, "RecordingStatusLabel")
    assert label is not None
    assert "STATUS" in label.text().upper()

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill is not None

    meter = w.findChild(VUMeter)
    assert meter is not None


def test_widget_has_fixed_placeholder_height(qtbot):
    """Chip reserves constant vertical space so the sidebar nav
    list above doesn't jump when state changes."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)

    min_h = w.minimumHeight()
    assert min_h >= 44, (
        f"chip must reserve >=44 px height, got {min_h}"
    )


def test_idle_state_shows_muted_pill_and_zeroed_meter(qtbot):
    """Same idea as the CPU / RAM / GPU graphs in the topbar —
    the slot always shows its current state, even when nothing
    is happening. Idle = visible muted pill + visible meter sat at 0,
    NOT a blank placeholder. Hiding everything was confusing
    (looked like the slot wasn't initialised)."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("idle")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "idle"
    # Some non-empty label so the user can read 'Idle' / 'Ready' / etc.
    assert pill.text().strip() != ""
    # Meter visible too — its bar just sits at 0 in the idle state.
    assert w._vu_meter.isVisibleTo(w)
    assert w._vu_meter.current_level() == 0.0


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


def test_processing_state_shows_pill_keeps_meter_at_zero(qtbot):
    """During Processing the audio buffer is already captured — the
    VU meter has nothing live to show, so it sits at zero. The
    meter stays visible (placeholder), unlike the previous design
    where it disappeared."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("processing")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "processing"
    assert "processing" in pill.text().lower()
    # Meter still visible (placeholder) but reset to 0.
    assert w._vu_meter.isVisibleTo(w)
    assert w._vu_meter.current_level() == 0.0


def test_model_loading_state_shows_explicit_loading_label(qtbot):
    """``model_loading`` must surface explicitly in the chip — the
    user needs a reason for hotkey presses to be rejected with
    "Model is not ready yet". Painting it the same as Idle (the
    previous behaviour, leftover from when the topbar carried the
    only loading indicator) caused exactly that confusion. We now
    render the accent variant + a dedicated label."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("model_loading")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    # Same accent shade as ``processing`` — both communicate
    # "backend is busy, hold off".
    assert pill.property("state") == "processing"
    assert "loading" in pill.text().lower()
    assert w._vu_meter.isVisibleTo(w)
    assert w._vu_meter.current_level() == 0.0


def test_back_to_idle_returns_pill_to_idle_appearance(qtbot):
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("recording")
    w.set_recording_state("idle")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "idle"
    # Meter still visible, reset to 0.
    assert w._vu_meter.isVisibleTo(w)
    assert w._vu_meter.current_level() == 0.0


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


def test_label_left_value_right_in_chip(qtbot):
    """Mirrors ResourceWidget's block layout — STATUS label hugs the
    left edge of the top row, the state value hugs the right edge,
    so the user reads it as 'STATUS: Idle' the same way they read
    'CPU: 12%' next to it."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.resize(200, 60)
    w.show()
    qtbot.waitExposed(w)

    label = _label_by_name(w, "RecordingStatusLabel")
    pill = _label_by_name(w, "RecordingStatusPill")

    # Label sits to the left of the value pill within the same row.
    assert label.x() < pill.x(), (
        f"STATUS label x={label.x()} should be left of value x={pill.x()}"
    )
    # And both share the top row — y coordinate within a few px.
    assert abs(label.y() - pill.y()) <= 4


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
    # Visibility check intentionally omitted — qtbot.addWidget() doesn't
    # call show() and isVisibleTo would always be False here. The level
    # reset is the actual behaviour under test; visibility is exercised
    # in test_idle_state_shows_muted_pill_and_zeroed_meter where the
    # widget is explicitly shown.
    assert w._vu_meter.current_level() == 0.0
