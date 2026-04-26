"""Tests for the TopBar widget."""

import pytest

from PySide6.QtWidgets import QLabel, QPushButton


def _label_by_name(widget, name: str) -> QLabel:
    return widget.findChild(QLabel, name)


def _button_by_name(widget, name: str) -> QPushButton:
    return widget.findChild(QPushButton, name)


def test_topbar_has_model_pill(qtbot):
    """The topbar's model pill is what tells the user which weights
    are loaded right now. Status pill is gone (Docker-era artifact)."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)

    assert _label_by_name(bar, "TopBarModelPill") is not None
    # Backend status pill removed alongside the BackendStatusPoller.
    assert _label_by_name(bar, "TopBarStatusPill") is None
    # Recording pill + VU meter moved to the sidebar's
    # RecordingStatusWidget so they don't overlap the resource graphs.
    assert _label_by_name(bar, "TopBarRecordingPill") is None
    from app.gui.widgets.vu_meter import VUMeter
    assert bar.findChild(VUMeter) is None


def test_topbar_does_not_duplicate_window_title(qtbot):
    """The window's title bar already shows the app name; the topbar should
    not repeat it."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    assert _label_by_name(bar, "TopBarTitle") is None


def test_topbar_no_longer_duplicates_section_label(qtbot):
    """The sidebar already highlights the active section; repeating
    its name in the topbar was just visual noise."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    assert _label_by_name(bar, "TopBarSectionTitle") is None


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


# ---- Recording state indicator ---------------------------------------------
# (Recording pill / VU meter moved to RecordingStatusWidget — see its
# dedicated test module. The topbar still accepts the same state vocab
# so the controller can fan it out, but only acts on ``model_loading``.)


def test_set_recording_state_model_loading_flips_model_pill(qtbot):
    """``model_loading`` flips the topbar's model pill into its
    yellow loading variant. The recording-pill side of the state
    is the sidebar widget's job, not the topbar's."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_active_model("Large v3 Turbo (int8)")
    bar.set_recording_state("model_loading")

    model_pill = _label_by_name(bar, "TopBarModelPill")
    assert model_pill.property("state") == "loading"
    assert "loading" in model_pill.text().lower()
    assert "Large v3 Turbo (int8)" in model_pill.text()


def test_model_pill_returns_to_active_when_loading_ends(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_active_model("Large v3 Turbo (int8)")
    bar.set_recording_state("model_loading")
    bar.set_recording_state("idle")

    pill = _label_by_name(bar, "TopBarModelPill")
    assert pill.property("state") == "active"
    assert "Current model" in pill.text()


def test_set_recording_state_rejects_unknown(qtbot):
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    with pytest.raises(ValueError):
        bar.set_recording_state("snoozing")


def test_topbar_set_loading_elapsed_appends_seconds_when_no_progress(qtbot):
    """While the backend is in model_loading state but no tqdm progress
    has fired (cached model deserialisation), the elapsed-seconds
    counter is the only signal that the wait is advancing.

    Now lives on the model pill, not the recording pill.
    """
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_active_model("Large v3 Turbo (int8)")
    bar.set_recording_state("model_loading")

    bar.set_loading_elapsed(5)
    text = _label_by_name(bar, "TopBarModelPill").text()
    assert "5" in text and "s" in text.lower()


def test_topbar_loading_progress_takes_priority_over_elapsed(qtbot):
    """Once download bytes start arriving, the percentage is more
    informative than the elapsed counter."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_active_model("Large v3 Turbo (int8)")
    bar.set_recording_state("model_loading")

    bar.set_loading_elapsed(7)
    bar.set_loading_progress(35, 100)
    text = _label_by_name(bar, "TopBarModelPill").text()
    assert "35%" in text
    assert "7s" not in text


def test_topbar_back_to_idle_clears_elapsed_state(qtbot):
    """Once loading ends, the elapsed counter must reset so the next
    loading session doesn't start at a stale number."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_active_model("Large v3 Turbo (int8)")
    bar.set_recording_state("model_loading")
    # Elapsed value chosen so its digit doesn't appear in the
    # model display name — otherwise we can't tell whether the
    # counter cleared or just happens to repeat a digit from the
    # name. ``Large v3 Turbo (int8)`` contains 3 and 8, so 4 is safe.
    bar.set_loading_elapsed(4)

    bar.set_recording_state("idle")
    bar.set_recording_state("model_loading")
    text = _label_by_name(bar, "TopBarModelPill").text()
    assert "4" not in text


# ---- Cancel-load button -----------------------------------------------------


def test_cancel_button_exists_and_hidden_by_default(qtbot):
    """Cancel only makes sense while a load is in flight; the button
    stays out of sight until ``set_recording_state('model_loading')``
    flips it on."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    button = _button_by_name(bar, "TopBarCancelLoadButton")
    assert button is not None
    assert not button.isVisibleTo(bar)


def test_cancel_button_visible_during_model_loading(qtbot):
    """When the backend goes into ``model_loading``, surface a cancel
    button next to the loading pill so the user can back out of a
    mis-clicked heavy model card without waiting for the deadlocked
    NeMo import to finish."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_active_model("Parakeet TDT v3 (multilingual)")
    bar.set_recording_state("model_loading")

    button = _button_by_name(bar, "TopBarCancelLoadButton")
    assert button.isVisibleTo(bar)


def test_cancel_button_hidden_again_after_loading_ends(qtbot):
    """Once loading finishes (or is itself cancelled), the button
    disappears so the topbar isn't littered with a stale Cancel
    that does nothing."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_active_model("Parakeet TDT v3 (multilingual)")
    bar.set_recording_state("model_loading")
    bar.set_recording_state("idle")

    button = _button_by_name(bar, "TopBarCancelLoadButton")
    assert not button.isVisibleTo(bar)


def test_cancel_button_emits_signal_on_click(qtbot):
    """The button is purely UI — its click emits a ``cancel_load_requested``
    signal that the AppController routes to the recording controller's
    ``cancel_model_change``. Keeps the topbar ignorant of the domain
    layer."""
    from app.gui.widgets.topbar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.show()
    bar.set_active_model("Parakeet TDT v3 (multilingual)")
    bar.set_recording_state("model_loading")

    button = _button_by_name(bar, "TopBarCancelLoadButton")
    with qtbot.waitSignal(bar.cancel_load_requested, timeout=500):
        button.click()
