"""Tests for the ModelCard widget."""

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton


def _make_info():
    from app.model_mapping import ModelInfo

    return ModelInfo(
        alias="large-v3",
        canonical="Systran/faster-whisper-large-v3",
        display_name="Large v3",
        size_mb=3000,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Latest large model.",
    )


def test_model_card_exposes_alias_and_info(qtbot):
    from app.gui.widgets.model_card import ModelCard

    info = _make_info()
    card = ModelCard(info)
    qtbot.addWidget(card)
    assert card.alias() == "large-v3"
    assert card.info() is info


def test_model_card_renders_display_name(qtbot):
    from app.gui.widgets.model_card import ModelCard

    info = _make_info()
    card = ModelCard(info)
    qtbot.addWidget(card)

    from PySide6.QtWidgets import QLabel

    labels = card.findChildren(QLabel)
    texts = [label.text() for label in labels]
    assert any("Large v3" == text for text in texts)


def test_model_card_renders_description(qtbot):
    from app.gui.widgets.model_card import ModelCard

    info = _make_info()
    card = ModelCard(info)
    qtbot.addWidget(card)

    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in card.findChildren(QLabel)]
    assert any(info.description in text for text in texts)


def test_model_card_shows_metadata_badges(qtbot):
    from app.gui.widgets.model_card import ModelCard

    info = _make_info()
    card = ModelCard(info)
    qtbot.addWidget(card)

    from PySide6.QtWidgets import QLabel

    texts = [label.text() for label in card.findChildren(QLabel)]
    assert any(info.speed in text for text in texts)
    assert any(info.quality in text for text in texts)
    assert any(("GB" in text or "MB" in text) for text in texts)


def test_model_card_defaults_to_inactive(qtbot):
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    assert card.is_active() is False


def test_model_card_set_active_reflects_state(qtbot):
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    card.set_active(True)
    assert card.is_active() is True
    assert card.property("active") is True

    card.set_active(False)
    assert card.is_active() is False
    assert card.property("active") is False


def test_model_card_hides_select_button_when_active(qtbot):
    """Active model should not offer a Select button."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    card.set_active(True)
    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )
    assert not select_btn.isVisible() or not select_btn.isEnabled()


def test_model_card_default_loading_state_is_off(qtbot):
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    assert card.is_loading() is False


def test_model_card_active_loading_swaps_pill_text(qtbot):
    """While loading, the Active pill must say 'Loading…' so it stops
    contradicting the topbar status pill."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)

    # Default: pill says Active
    assert card._active_pill.text() == "Active"

    card.set_loading(True)
    assert card.is_loading() is True
    assert "loading" in card._active_pill.text().lower()
    assert card._active_pill.property("state") == "loading"

    card.set_loading(False)
    assert card.is_loading() is False
    assert card._active_pill.text() == "Active"
    assert card._active_pill.property("state") == "ready"


def test_model_card_badges_carry_category_attribute(qtbot):
    """Each metadata badge carries a ``cat`` property so the QSS can
    style speed/quality/compute/lang differently — without it every
    pill looks identical and the eye can't tell them apart."""
    from PySide6.QtWidgets import QLabel
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    badges = [
        lbl for lbl in card.findChildren(QLabel)
        if lbl.property("role") == "badge"
    ]
    cats = {lbl.property("cat") for lbl in badges}
    assert cats == {"speed", "quality", "size", "vram", "compute", "lang"}


def test_model_card_speed_quality_badges_carry_value_for_styling(qtbot):
    """The QSS ``[cat='speed'][value='fast']`` selector tints fast
    speed badges green; without the ``value`` property nothing
    matches and the highlight never appears."""
    from PySide6.QtWidgets import QLabel
    from app.gui.widgets.model_card import ModelCard
    from app.model_mapping import ModelInfo

    info = ModelInfo(
        alias="test-fast",
        canonical="fake/canonical",
        display_name="Test fast",
        size_mb=1000,
        vram_gb=4.0,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description="x",
    )
    card = ModelCard(info)
    qtbot.addWidget(card)

    badges = {
        lbl.property("cat"): lbl
        for lbl in card.findChildren(QLabel)
        if lbl.property("role") == "badge"
    }
    assert badges["speed"].property("value") == "fast"
    assert badges["quality"].property("value") == "excellent"
    # Resource / technical badges should not carry a discrete value
    # property — they're styled purely by category.
    for cat in ("size", "vram", "compute", "lang"):
        assert not badges[cat].property("value")


def test_model_card_select_button_does_not_grab_focus(qtbot):
    """Clicking Download/Select must not put focus on the button.

    The card hides the Select button as soon as the model becomes
    active, and Qt moves focus to the next button in the tab order
    (the Download button on the next card). The scroll area then
    scrolls to keep that newly-focused button visible — i.e. clicking
    Download on the first card jumps the whole list downwards. Stopping
    the button from grabbing focus on click prevents the chase entirely."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )
    assert select_btn.focusPolicy() == Qt.NoFocus


def test_model_card_loading_progress_updates_pill_text(qtbot):
    """While loading, the active pill should show download progress as a
    percentage so the user can see the model fetch advancing."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)

    # Default loading pill before any progress
    assert card._active_pill.text() == "Loading\u2026"

    card.set_loading_progress(50, 100)
    assert "50%" in card._active_pill.text()

    card.set_loading_progress(35, 100)
    assert "35%" in card._active_pill.text()


def test_model_card_loading_elapsed_shows_seconds_when_no_progress(qtbot):
    """Cached model loads (CTranslate2 deserialisation) take ~15 seconds
    without ever firing a tqdm progress event. Without an elapsed
    counter the pill just sits at 'Loading…' and looks frozen — show
    seconds instead so the user can see the wait advancing."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)

    card.set_loading_elapsed(5)
    text = card._active_pill.text()
    assert "5" in text and ("s" in text.lower() or "сек" in text.lower())

    card.set_loading_elapsed(12)
    assert "12" in card._active_pill.text()


def test_model_card_loading_progress_takes_priority_over_elapsed(qtbot):
    """Once download bytes start arriving, the percentage is more
    informative than elapsed seconds — the pill should show '%' even
    if the elapsed counter is also being pushed."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)

    card.set_loading_elapsed(7)
    card.set_loading_progress(50, 100)
    assert "50%" in card._active_pill.text()
    assert "7" not in card._active_pill.text()


def test_model_card_set_loading_false_clears_elapsed(qtbot):
    """Returning to ready state must wipe both progress and elapsed
    state, not just the progress text."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)
    card.set_loading_elapsed(7)

    card.set_loading(False)
    assert card._active_pill.text() == "Active"

    card.set_loading(True)
    # Re-entering loading must not resurrect the stale elapsed text.
    assert card._active_pill.text() == "Loading\u2026"


def test_model_card_loading_progress_falls_back_to_size(qtbot):
    """When the total file size is unknown (Hugging Face streaming bars
    sometimes have total=0), fall back to a byte counter so the user
    still sees movement instead of a stuck 'Loading…'."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)

    card.set_loading_progress(2_500_000, 0)
    text = card._active_pill.text()
    assert "MB" in text or "KB" in text or "GB" in text


def test_model_card_loading_progress_ignored_when_not_loading(qtbot):
    """Stale progress events arriving after the model finished must not
    repaint the green Active pill with download bytes."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    # Not in loading state.

    card.set_loading_progress(50, 100)
    assert card._active_pill.text() == "Active"


def test_model_card_set_loading_false_resets_pill_text(qtbot):
    """Returning to ready state must wipe any leftover progress text."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.set_active(True)
    card.set_loading(True)
    card.set_loading_progress(50, 100)

    card.set_loading(False)
    assert card._active_pill.text() == "Active"


def test_model_card_inactive_card_ignores_loading(qtbot):
    """Loading should only affect the currently-active card's pill, not
    the inactive ones."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    # No set_active(True) — card is inactive.

    card.set_loading(True)
    # Inactive cards: pill is hidden, no visible text changes needed.
    assert not card._active_pill.isVisibleTo(card)


def test_model_card_button_says_download_when_not_cached(qtbot, monkeypatch):
    """Uncached models advertise the action as 'Download' so the user knows
    the click will fetch weights from the network."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_model_cached", lambda c: False)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )
    assert select_btn.text() == "Download"


def test_model_card_button_says_select_when_cached(qtbot, monkeypatch):
    """Once weights are on disk, the action button switches to 'Select'."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_model_cached", lambda c: True)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )
    assert select_btn.text() == "Select"


def test_model_card_refresh_cache_state_picks_up_new_state(qtbot, monkeypatch):
    """After a download finishes, calling refresh_cache_state should flip
    the button label without needing to rebuild the card."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    cache_status = {"cached": False}
    monkeypatch.setattr(
        model_card_module,
        "is_model_cached",
        lambda c: cache_status["cached"],
    )

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )
    assert select_btn.text() == "Download"

    cache_status["cached"] = True
    card.refresh_cache_state()
    assert select_btn.text() == "Select"


def test_model_card_emits_select_signal_with_alias(qtbot):
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()  # buttons must be visible for qtbot.mouseClick
    qtbot.waitExposed(card)

    select_btn = next(
        b for b in card.findChildren(QPushButton) if b.objectName() == "SelectButton"
    )

    with qtbot.waitSignal(card.select_requested, timeout=1000) as blocker:
        qtbot.mouseClick(select_btn, Qt.LeftButton)

    assert blocker.args == ["large-v3"]
