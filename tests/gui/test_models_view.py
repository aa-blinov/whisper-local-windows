"""Tests for the ModelsView."""

import pytest


def test_models_view_creates_one_card_per_registry_entry(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard
    from app.model_mapping import MODELS

    view = ModelsView()
    qtbot.addWidget(view)

    cards = view.findChildren(ModelCard)
    assert {c.alias() for c in cards} == {m.alias for m in MODELS}


def test_models_view_custom_registry(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard
    from app.model_mapping import get_model

    subset = (get_model("tiny"), get_model("large-v3"))
    view = ModelsView(models=subset)
    qtbot.addWidget(view)

    cards = view.findChildren(ModelCard)
    assert [c.alias() for c in cards] == ["tiny", "large-v3"]


def test_models_view_set_active_marks_correct_card(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)

    view.set_active("large-v3")
    assert view.active_alias() == "large-v3"

    cards = {c.alias(): c for c in view.findChildren(ModelCard)}
    assert cards["large-v3"].is_active() is True
    for alias, card in cards.items():
        if alias != "large-v3":
            assert card.is_active() is False


def test_models_view_set_active_switches_cleanly(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)

    view.set_active("tiny")
    view.set_active("large-v3")

    cards = {c.alias(): c for c in view.findChildren(ModelCard)}
    assert cards["tiny"].is_active() is False
    assert cards["large-v3"].is_active() is True


def test_models_view_set_active_rejects_unknown(qtbot):
    from app.gui.views.models_view import ModelsView

    view = ModelsView()
    qtbot.addWidget(view)
    with pytest.raises(KeyError):
        view.set_active("does-not-exist")


def test_models_view_emits_model_selected_when_card_emits(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)

    card = next(c for c in view.findChildren(ModelCard) if c.alias() == "small")

    with qtbot.waitSignal(view.model_selected, timeout=1000) as blocker:
        card.select_requested.emit(card.alias())

    assert blocker.args == ["small"]


def test_models_view_starts_with_no_active(qtbot):
    from app.gui.views.models_view import ModelsView

    view = ModelsView()
    qtbot.addWidget(view)
    assert view.active_alias() is None
