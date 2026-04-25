"""Tests for the Sidebar navigation widget."""

import pytest


DEFAULT_KEYS = ("models", "shortcuts", "history", "logs")


def test_sidebar_instantiates_with_default_items(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.items() == list(DEFAULT_KEYS)


def test_sidebar_default_active_is_first_item(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.active_key() == "models"


def test_sidebar_set_active_changes_selection(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    sidebar.set_active("history")
    assert sidebar.active_key() == "history"


def test_sidebar_set_active_rejects_unknown_key(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    with pytest.raises(ValueError):
        sidebar.set_active("does-not-exist")


def test_sidebar_emits_signal_when_user_clicks_item(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    with qtbot.waitSignal(sidebar.nav_selected, timeout=1000) as blocker:
        sidebar.set_active("shortcuts")

    assert blocker.args == ["shortcuts"]


def test_sidebar_does_not_emit_signal_when_setting_same_key(qtbot):
    """Setting active to the already-active key should not re-emit."""
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)

    emissions: list[str] = []
    sidebar.nav_selected.connect(emissions.append)

    sidebar.set_active("models")  # already default
    assert emissions == []


def test_sidebar_custom_items(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    items = [("a", "Alpha"), ("b", "Beta")]
    sidebar = Sidebar(items=items)
    qtbot.addWidget(sidebar)

    assert sidebar.items() == ["a", "b"]
    assert sidebar.active_key() == "a"


def test_sidebar_label_for_returns_display_label(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    # 'shortcuts' is the internal key; display label is 'Settings'.
    assert sidebar.label_for("shortcuts") == "Settings"
    assert sidebar.label_for("models") == "Models"


def test_sidebar_label_for_falls_back_for_unknown_key(qtbot):
    from app.gui.widgets.sidebar import Sidebar

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    assert sidebar.label_for("nope") == "Nope"
