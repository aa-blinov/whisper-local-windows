"""Tests for the TopBar widget."""

import pytest

from PySide6.QtWidgets import QLabel


def _label_by_name(widget, name: str) -> QLabel:
    return widget.findChild(QLabel, name)


def test_topbar_has_title_model_and_status_labels(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    assert _label_by_name(bar, "TopBarTitle") is not None
    assert _label_by_name(bar, "TopBarModelPill") is not None
    assert _label_by_name(bar, "TopBarStatusPill") is not None


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
