"""Tests for the TopBar widget."""

import pytest

from PySide6.QtWidgets import QLabel


def _label_by_name(widget, name: str) -> QLabel:
    return widget.findChild(QLabel, name)


def test_topbar_has_model_and_status_labels(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    assert _label_by_name(bar, "TopBarModelPill") is not None
    assert _label_by_name(bar, "TopBarStatusPill") is not None


def test_topbar_does_not_duplicate_window_title(qtbot):
    """The window's title bar already shows the app name; the topbar should
    not repeat it."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    assert _label_by_name(bar, "TopBarTitle") is None


def test_topbar_has_section_title_label(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    assert _label_by_name(bar, "TopBarSectionTitle") is not None


def test_set_section_title_updates_label(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_section_title("Models")
    assert _label_by_name(bar, "TopBarSectionTitle").text() == "Models"

    bar.set_section_title("History")
    assert _label_by_name(bar, "TopBarSectionTitle").text() == "History"


def test_topbar_default_model_pill_text_when_no_model(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    pill = _label_by_name(bar, "TopBarModelPill")
    assert "no model" in pill.text().lower()


def test_set_active_model_updates_pill(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    bar.set_active_model("Large v3")

    pill = _label_by_name(bar, "TopBarModelPill")
    assert "Large v3" in pill.text()


def test_set_active_model_none_clears(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    bar.set_active_model("Large v3")
    bar.set_active_model(None)

    pill = _label_by_name(bar, "TopBarModelPill")
    assert "no model" in pill.text().lower()


def test_default_backend_status_is_unknown(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    pill = _label_by_name(bar, "TopBarStatusPill")
    assert pill.property("status") == "unknown"


def test_set_backend_status_updates_property_and_label(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    bar.set_backend_status("running", "Docker up")

    pill = _label_by_name(bar, "TopBarStatusPill")
    assert pill.property("status") == "running"
    assert "Docker up" in pill.text()


def test_set_backend_status_uses_default_label_when_omitted(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    bar.set_backend_status("stopped")

    pill = _label_by_name(bar, "TopBarStatusPill")
    assert "stopped" in pill.text().lower()


def test_set_backend_status_rejects_unknown_value(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    with pytest.raises(ValueError):
        bar.set_backend_status("nuclear")


# ---- Recording state indicator ---------------------------------------------


def _recording_pill(bar) -> QLabel:
    return _label_by_name(bar, "TopBarRecordingPill")


def test_recording_pill_exists_and_hidden_by_default(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    pill = _recording_pill(bar)
    assert pill is not None
    assert not pill.isVisibleTo(bar)


def test_set_recording_state_idle_keeps_pill_hidden(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_recording_state("idle")
    assert not _recording_pill(bar).isVisibleTo(bar)


def test_set_recording_state_recording_shows_pill(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_recording_state("recording")
    pill = _recording_pill(bar)
    assert pill.isVisibleTo(bar)
    assert pill.property("state") == "recording"
    assert "recording" in pill.text().lower()


def test_set_recording_state_processing_shows_pill(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_recording_state("processing")
    pill = _recording_pill(bar)
    assert pill.isVisibleTo(bar)
    assert pill.property("state") == "processing"
    assert "processing" in pill.text().lower()


def test_set_recording_state_model_loading_shows_pill(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_recording_state("model_loading")
    pill = _recording_pill(bar)
    assert pill.isVisibleTo(bar)
    assert pill.property("state") == "model_loading"
    assert "model" in pill.text().lower()


def test_set_recording_state_back_to_idle_hides_pill(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_recording_state("recording")
    bar.set_recording_state("idle")
    assert not _recording_pill(bar).isVisibleTo(bar)


def test_set_recording_state_rejects_unknown(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    with pytest.raises(ValueError):
        bar.set_recording_state("snoozing")


def test_topbar_set_loading_elapsed_appends_seconds_when_no_progress(qtbot):
    """While the backend is in model_loading state but no tqdm progress
    has fired (cached model deserialisation), the elapsed-seconds
    counter is the only signal that the wait is advancing."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_recording_state("model_loading")

    bar.set_loading_elapsed(5)
    text = _recording_pill(bar).text()
    assert "5" in text and "s" in text.lower()


def test_topbar_loading_progress_takes_priority_over_elapsed(qtbot):
    """Once download bytes start arriving, the percentage is more
    informative than the elapsed counter."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_recording_state("model_loading")

    bar.set_loading_elapsed(7)
    bar.set_loading_progress(35, 100)
    text = _recording_pill(bar).text()
    assert "35%" in text
    assert "7s" not in text


def test_topbar_back_to_idle_clears_elapsed_state(qtbot):
    """Once loading ends, the elapsed counter must reset so the next
    loading session doesn't start at a stale number."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_recording_state("model_loading")
    bar.set_loading_elapsed(8)

    bar.set_recording_state("idle")
    bar.set_recording_state("model_loading")
    text = _recording_pill(bar).text()
    assert "8" not in text
