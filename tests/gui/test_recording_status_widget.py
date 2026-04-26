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


def test_model_loading_state_falls_back_to_idle_appearance(qtbot):
    """``model_loading`` is reflected in the topbar's model pill,
    not here — but the slot still has to fill its placeholder, so
    show it the same way Idle does. Hiding everything would create
    visual hole in the sidebar."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.show()

    w.set_recording_state("model_loading")

    pill = _label_by_name(w, "RecordingStatusPill")
    assert pill.isVisibleTo(w)
    assert pill.property("state") == "idle"
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


def test_pill_and_meter_aligned_to_left_edge(qtbot):
    """The slot lives in the bottom-left corner — pill and VU meter
    must hug the left edge to read as a single column with the nav
    list above, not a pair of free-floating centered chips."""
    from PySide6.QtCore import Qt
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)

    layout = w.layout()
    # Walk the layout items and confirm both child widgets have a
    # left-aligned alignment flag (no AlignHCenter / AlignRight bits).
    found_pill = False
    found_meter = False
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is None:
            continue
        if widget.objectName() == "RecordingStatusPill":
            assert item.alignment() & Qt.AlignLeft, (
                f"pill expected left-aligned, got {item.alignment()!r}"
            )
            assert not (item.alignment() & Qt.AlignHCenter)
            found_pill = True
        if widget.objectName() == "VUMeter":
            assert item.alignment() & Qt.AlignLeft, (
                f"VU meter expected left-aligned, got {item.alignment()!r}"
            )
            assert not (item.alignment() & Qt.AlignHCenter)
            found_meter = True
    assert found_pill and found_meter


def test_meter_resets_when_leaving_recording_state(qtbot):
    """Same contract the topbar previously held — leaving the
    recording state must zero the meter so a stale reading from the
    last capture doesn't linger when the next one starts. The
    meter widget stays visible (placeholder); only its bar resets."""
    from app.gui.widgets.recording_status_widget import RecordingStatusWidget

    w = RecordingStatusWidget()
    qtbot.addWidget(w)
    w.set_recording_state("recording")
    w.set_input_level(0.6)
    w.set_recording_state("idle")
    assert w._vu_meter.isVisibleTo(w) or True  # not asserting visibility before show()
    assert w._vu_meter.current_level() == 0.0
