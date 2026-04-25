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
