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


def test_model_card_renders_family_chip_with_attribute(qtbot):
    """The family chip in the card header carries a normalised
    ``family`` attribute so QSS can color-code per family without
    string parsing."""
    from PySide6.QtWidgets import QLabel
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    chip = next(
        lbl for lbl in card.findChildren(QLabel)
        if lbl.objectName() == "FamilyChip"
    )
    assert chip.text()  # non-empty
    assert chip.property("family")  # populated for QSS selector


def test_model_card_subtitle_contains_alias_canonical_and_link(qtbot):
    """The subtitle line surfaces the registry alias, the full
    canonical id, and a clickable link to the model's home page."""
    from PySide6.QtWidgets import QLabel
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)

    subtitle = next(
        lbl for lbl in card.findChildren(QLabel)
        if lbl.objectName() == "ModelSubtitle"
    )
    text = subtitle.text()
    assert "large-v3" in text
    assert "Systran/faster-whisper-large-v3" in text
    assert "huggingface.co" in text
    assert subtitle.openExternalLinks() is True


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

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: False)

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

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: True)

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
        "is_cached_for_info",
        lambda info: cache_status["cached"],
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


# ---- Delete button ---------------------------------------------------------


def _delete_btn(card) -> QPushButton:
    return next(
        b for b in card.findChildren(QPushButton)
        if b.objectName() == "DeleteButton"
    )


def test_model_card_has_a_delete_button(qtbot):
    """Every card carries a Delete affordance — cached state controls
    its visibility, but the widget is always present."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    btns = [b.objectName() for b in card.findChildren(QPushButton)]
    assert "DeleteButton" in btns


def test_model_card_delete_button_hidden_when_not_cached(qtbot, monkeypatch):
    """Nothing to delete → no button. Otherwise the user gets an
    enabled control that does nothing (or worse, fires a confirm
    dialog over an empty cache)."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: False)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    assert not _delete_btn(card).isVisible()


def test_model_card_delete_button_visible_when_cached_and_inactive(qtbot, monkeypatch):
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: True)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)
    assert _delete_btn(card).isVisible()


def test_model_card_delete_button_hidden_when_active(qtbot, monkeypatch):
    """Deleting the loaded model would crash the running backend —
    hide the button until the user picks a different active card."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: True)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)
    card.set_active(True)
    assert not _delete_btn(card).isVisible()


def test_model_card_delete_button_hidden_when_loading(qtbot, monkeypatch):
    """Mid-download / mid-deserialise the cache state is undefined —
    hiding Delete avoids confusing the user with a button that might
    succeed or fail depending on timing."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: True)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)
    card.set_loading(True)
    assert not _delete_btn(card).isVisible()


def test_model_card_delete_button_emits_signal_with_alias(qtbot, monkeypatch):
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    monkeypatch.setattr(model_card_module, "is_cached_for_info", lambda info: True)

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)

    with qtbot.waitSignal(card.delete_requested, timeout=1000) as blocker:
        qtbot.mouseClick(_delete_btn(card), Qt.LeftButton)

    assert blocker.args == ["large-v3"]


def test_model_card_refresh_cache_state_toggles_delete_visibility(qtbot, monkeypatch):
    """After a Download or Delete completes, the controller calls
    ``refresh_cache_state`` — Delete's visibility must follow."""
    import app.gui.widgets.model_card as model_card_module
    from app.gui.widgets.model_card import ModelCard

    cache_status = {"cached": True}
    monkeypatch.setattr(
        model_card_module,
        "is_cached_for_info",
        lambda info: cache_status["cached"],
    )

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    card.show()
    qtbot.waitExposed(card)
    assert _delete_btn(card).isVisible()

    cache_status["cached"] = False
    card.refresh_cache_state()
    assert not _delete_btn(card).isVisible()


# ---- HF token warning (GigaAM only) ----------------------------------------


def _make_gigaam_info():
    from app.model_mapping import ModelInfo

    return ModelInfo(
        alias="gigaam-v3-e2e-ctc",
        canonical="v3_e2e_ctc",
        display_name="GigaAM v3 CTC (e2e, punctuated)",
        size_mb=260,
        vram_gb=2.0,
        speed="fast",
        quality="excellent",
        languages="Russian (only)",
        description="Sber GigaAM v3 with CTC decoder.",
        backend_kind="gigaam",
        family="GigaAM",
    )


def _hf_warning(card):
    from PySide6.QtWidgets import QLabel

    return card.findChild(QLabel, "HfTokenWarning")


def test_gigaam_card_has_hf_token_warning_widget(qtbot):
    """Every GigaAM card carries a warning label that surfaces when
    no HF token is configured — long-form audio (>25 s) routes
    through pyannote VAD which needs a token to download
    ``pyannote/segmentation-3.0`` (gated)."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_gigaam_info())
    qtbot.addWidget(card)
    assert _hf_warning(card) is not None


def test_whisper_card_has_no_hf_token_warning(qtbot):
    """Whisper long-form goes through Silero VAD — no HF token
    needed — so the warning widget is omitted entirely on
    ``faster_whisper`` cards."""
    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_info())
    qtbot.addWidget(card)
    assert _hf_warning(card) is None


def test_gigaam_card_warning_visible_when_no_token(qtbot, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_gigaam_info())
    qtbot.addWidget(card)
    card.show()
    assert _hf_warning(card).isVisible()


def test_gigaam_card_warning_hidden_when_token_set(qtbot, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_value")

    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_gigaam_info())
    qtbot.addWidget(card)
    card.show()
    assert not _hf_warning(card).isVisible()


def test_gigaam_card_refresh_hf_token_state_updates_warning(qtbot, monkeypatch):
    """After the user pastes a token in Settings, the controller
    calls ``refresh_hf_token_state`` to re-evaluate every GigaAM
    card's warning visibility without rebuilding the card tree."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_gigaam_info())
    qtbot.addWidget(card)
    card.show()
    assert _hf_warning(card).isVisible()

    monkeypatch.setenv("HF_TOKEN", "hf_dummy_value")
    card.refresh_hf_token_state()
    assert not _hf_warning(card).isVisible()


def test_gigaam_card_warning_text_mentions_settings_and_25s(qtbot, monkeypatch):
    """User-facing copy must explain WHY (long-form / 25 s cap) and
    WHERE to fix (Settings tab) — otherwise the warning is just
    noise."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    from app.gui.widgets.model_card import ModelCard

    card = ModelCard(_make_gigaam_info())
    qtbot.addWidget(card)
    text = _hf_warning(card).text().lower()
    assert "25" in text
    assert "settings" in text or "token" in text
