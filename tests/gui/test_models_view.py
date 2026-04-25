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

    subset = (get_model("turbo"), get_model("large-v3"))
    view = ModelsView(models=subset)
    qtbot.addWidget(view)

    cards = view.findChildren(ModelCard)
    assert [c.alias() for c in cards] == ["turbo", "large-v3"]


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

    view.set_active("turbo")
    view.set_active("large-v3")

    cards = {c.alias(): c for c in view.findChildren(ModelCard)}
    assert cards["turbo"].is_active() is False
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

    card = next(c for c in view.findChildren(ModelCard) if c.alias() == "distil-large-v3")

    with qtbot.waitSignal(view.model_selected, timeout=1000) as blocker:
        card.select_requested.emit(card.alias())

    assert blocker.args == ["distil-large-v3"]


def test_models_view_starts_with_no_active(qtbot):
    from app.gui.views.models_view import ModelsView

    view = ModelsView()
    qtbot.addWidget(view)
    assert view.active_alias() is None


# ---- Lock state ------------------------------------------------------------


def test_models_view_starts_unlocked(qtbot):
    from app.gui.views.models_view import ModelsView

    view = ModelsView()
    qtbot.addWidget(view)
    assert view.is_locked() is False


def test_set_locked_disables_all_select_buttons(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)

    view.set_locked(True)
    assert view.is_locked() is True

    for card in view.findChildren(ModelCard):
        select_btn = next(
            b for b in card.findChildren(__import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton)
            if b.objectName() == "SelectButton"
        )
        assert not select_btn.isEnabled()


def test_models_view_set_loading_progress_propagates_to_active_card(qtbot):
    """Download progress events arriving from the backend should bubble
    down to whichever card is currently flagged as loading, so the
    user sees percentage advance on the model they just clicked."""
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)
    view.set_active("turbo")
    view.set_loading(True)

    view.set_loading_progress(50, 100)

    cards = {c.alias(): c for c in view.findChildren(ModelCard)}
    assert "50%" in cards["turbo"]._active_pill.text()
    # Inactive cards' pills are hidden — text doesn't matter to the
    # user, but we don't want to crash trying to update them either.


def test_models_view_refresh_cache_state_propagates_to_all_cards(qtbot, monkeypatch):
    """When the cache state for any model changes (e.g. a download just
    finished), the view's refresh_cache_state must update every card so
    freshly-downloaded models flip from 'Download' to 'Select'."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard
    from PySide6.QtWidgets import QPushButton

    cache_status = {"cached": False}
    monkeypatch.setattr(
        model_card_module,
        "is_model_cached",
        lambda c: cache_status["cached"],
    )

    view = ModelsView()
    qtbot.addWidget(view)

    # Initially, every Select button should advertise Download.
    for card in view.findChildren(ModelCard):
        btn = next(
            b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
        )
        assert btn.text() == "Download"

    cache_status["cached"] = True
    view.refresh_cache_state()

    for card in view.findChildren(ModelCard):
        btn = next(
            b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
        )
        assert btn.text() == "Select"


def test_set_locked_false_re_enables_buttons_for_inactive_cards(qtbot):
    from app.gui.views.models_view import ModelsView
    from app.gui.widgets.model_card import ModelCard

    view = ModelsView()
    qtbot.addWidget(view)
    view.set_active("large-v3")

    view.set_locked(True)
    view.set_locked(False)

    cards = {c.alias(): c for c in view.findChildren(ModelCard)}
    # Active card's Select stays hidden/disabled (active state).
    assert cards["large-v3"].is_active() is True
    # Inactive cards must be clickable again.
    select_btn = next(
        b for b in cards["turbo"].findChildren(__import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton)
        if b.objectName() == "SelectButton"
    )
    assert select_btn.isEnabled()
