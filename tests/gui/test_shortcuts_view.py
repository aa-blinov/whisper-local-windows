"""Tests for the ShortcutsView."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLineEdit, QPushButton


def _start_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StartHotkeyEdit")


def _stop_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StopHotkeyEdit")


def _auto_paste_cb(view) -> QCheckBox:
    return view.findChild(QCheckBox, "AutoPasteCheckbox")


def _reset_hotkeys_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "ResetHotkeysButton")


def _clear_hf_token_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "ClearHfTokenButton")


def test_shortcuts_view_has_expected_widgets(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    assert _start_edit(view) is not None
    assert _stop_edit(view) is not None
    assert _auto_paste_cb(view) is not None
    # Each card now owns its own reset/clear button — the global
    # footer button is gone.
    assert _reset_hotkeys_btn(view) is not None
    assert _clear_hf_token_btn(view) is not None


def test_global_reset_footer_button_no_longer_exists(qtbot):
    """Per-card buttons replaced the catch-all 'Reset to defaults'
    footer; that button used to mislead by only resetting hotkeys
    + auto_paste. Asserting it's gone keeps the migration honest."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    assert view.findChild(QPushButton, "ResetShortcutsButton") is None


def test_save_button_no_longer_exists(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    save_btn = view.findChild(QPushButton, "SaveShortcutsButton")
    assert save_btn is None


def test_set_values_prefills_fields(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_values(
        start_hotkey="ctrl+alt+r",
        stop_hotkey="ctrl+alt+s",
        auto_paste=True,
    )

    assert _start_edit(view).text() == "ctrl+alt+r"
    assert _stop_edit(view).text() == "ctrl+alt+s"
    assert _auto_paste_cb(view).isChecked() is True


def test_set_values_handles_false_auto_paste(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=False,
    )

    assert _auto_paste_cb(view).isChecked() is False


def test_getters_return_current_values(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
    )

    assert view.start_hotkey() == "ctrl+f2"
    assert view.stop_hotkey() == "ctrl+f3"
    assert view.auto_paste() is True


def test_set_values_does_not_emit_save_requested(qtbot):
    """Programmatic prefill must not trigger persistence."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    emissions: list[dict] = []
    view.save_requested.connect(emissions.append)

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
    )

    assert emissions == []


def test_editing_start_hotkey_emits_save_requested_on_finish(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(start_hotkey="ctrl+f2", stop_hotkey="ctrl+f3", auto_paste=False)

    start = _start_edit(view)
    start.clear()
    qtbot.keyClicks(start, "ctrl+alt+1")

    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        # editingFinished fires on focus loss / Enter
        start.editingFinished.emit()

    payload = blocker.args[0]
    assert payload["start_hotkey"] == "ctrl+alt+1"
    assert payload["stop_hotkey"] == "ctrl+f3"
    assert payload["auto_paste"] is False


def test_editing_stop_hotkey_emits_save_requested(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(start_hotkey="ctrl+f2", stop_hotkey="ctrl+f3", auto_paste=False)

    stop = _stop_edit(view)
    stop.clear()
    qtbot.keyClicks(stop, "ctrl+alt+2")

    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        stop.editingFinished.emit()

    assert blocker.args[0]["stop_hotkey"] == "ctrl+alt+2"


def test_toggling_auto_paste_emits_save_requested_immediately(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(start_hotkey="ctrl+f2", stop_hotkey="ctrl+f3", auto_paste=False)

    cb = _auto_paste_cb(view)
    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        cb.setChecked(True)

    assert blocker.args[0]["auto_paste"] is True


def test_reset_hotkeys_button_emits_hotkeys_reset_requested(qtbot):
    """Per-card 'Reset to defaults' inside the Hotkeys card emits
    the new card-scoped signal. Replaces the global ``reset_requested``
    that used to also reset auto-paste."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    btn = _reset_hotkeys_btn(view)
    with qtbot.waitSignal(view.hotkeys_reset_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


def test_clear_hf_token_button_emits_request(qtbot):
    """The HF card's 'Clear token' button emits its own signal —
    controller wipes the token from config + env."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    btn = _clear_hf_token_btn(view)
    with qtbot.waitSignal(view.hf_token_reset_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


# ---- Microphone permission banner -----------------------------------------


def test_mic_banner_switches_to_restart_after_grant(qtbot, monkeypatch):
    """Granting microphone access mid-session must refresh the banner
    out of the initial 'Allow microphone access' state and into the
    post-grant 'Restart now' prompt."""
    import app.gui.views.shortcuts_view as shortcuts_module
    from app.gui.views.shortcuts_view import ShortcutsView

    status = {"value": "not_determined"}
    monkeypatch.setattr(shortcuts_module, "microphone_authorization_status", lambda: status["value"])

    view = ShortcutsView()
    qtbot.addWidget(view)

    assert view._mic_state == "not_determined"
    assert view._mic_banner_button.text() == "Allow microphone access"

    status["value"] = "authorized"
    view._on_mic_request_completed(True)

    qtbot.waitUntil(
        lambda: view._mic_state == "granted"
        and view._mic_banner_button.text() == "Restart now",
        timeout=1000,
    )


# ---- Storage card ----------------------------------------------------------


def _change_storage_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "ChangeStorageButton")


def _reset_storage_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "ResetStorageButton")


def _storage_path_label(view):
    from PySide6.QtWidgets import QLabel

    return view.findChild(QLabel, "StoragePathLabel")


def test_shortcuts_view_has_storage_widgets(qtbot):
    """Settings tab carries a Storage card so the user can move
    downloaded weights off the system drive without editing
    ``config.yaml`` by hand."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    assert _change_storage_btn(view) is not None
    assert _reset_storage_btn(view) is not None
    assert _storage_path_label(view) is not None


def test_set_storage_path_updates_label_with_path(qtbot):
    """``set_storage_path(path, is_default=False)`` shows the chosen
    directory verbatim — no truncation. The user copy-pastes this
    into Explorer to verify the weights ended up where they expected."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_storage_path("D:/lazy-to-text-models", is_default=False)
    label = _storage_path_label(view)
    assert "D:/lazy-to-text-models" in label.text()


def test_set_storage_path_marks_default_explicitly(qtbot):
    """When the user hasn't picked a custom path, the label still
    shows the resolved default path AND a ``(default)`` marker so it's
    clear nothing is overridden."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_storage_path("C:/project/models", is_default=True)
    text = _storage_path_label(view).text()
    assert "C:/project/models" in text
    assert "default" in text.lower()


def _storage_size_label(view):
    from PySide6.QtWidgets import QLabel
    return view.findChild(QLabel, "StorageSizeLabel")


def _open_storage_btn(view):
    from PySide6.QtWidgets import QPushButton
    return view.findChild(QPushButton, "OpenStorageButton")


def test_storage_card_has_size_label(qtbot):
    """The Storage card carries a label that shows total bytes used
    by downloaded weights — without it the user has no idea how much
    disk the cache eats and whether they should move it to a bigger
    drive.  Starts blank until the controller computes the size."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    label = _storage_size_label(view)
    assert label is not None


def test_set_storage_size_updates_label(qtbot):
    """``set_storage_size(text)`` writes the human-readable size into
    the label.  The controller does the formatting (bytes → ``"3.4 GB"``)
    so the view stays free of locale rules."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_storage_size("3.4 GB")
    text = _storage_size_label(view).text()
    assert "3.4 GB" in text


def test_storage_card_has_open_folder_button(qtbot):
    """A direct-to-Explorer button is the user's escape hatch — once
    they know how much is used, they want to inspect / clean up."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    btn = _open_storage_btn(view)
    assert btn is not None


def test_open_folder_button_emits_request(qtbot):
    """Same delegation pattern as Change/Reset — the view emits, the
    controller actually opens the folder (so we can mock subprocess
    in tests)."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    btn = _open_storage_btn(view)
    with qtbot.waitSignal(view.storage_open_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


def test_change_storage_button_emits_request(qtbot):
    """The view delegates path-picking to the controller — the
    button itself just emits a request signal and the controller
    opens the QFileDialog. Keeps the view free of dialogs and easier
    to test."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    btn = _change_storage_btn(view)
    with qtbot.waitSignal(view.storage_path_change_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


def test_reset_storage_button_emits_request(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    # Reset is disabled until the user has actually customised the
    # path — set a non-default before clicking.
    view.set_storage_path("D:/elsewhere", is_default=False)

    btn = _reset_storage_btn(view)
    with qtbot.waitSignal(view.storage_reset_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


def test_reset_storage_button_disabled_when_already_on_default(qtbot):
    """No point clicking Reset when there's nothing to reset to —
    fire the disabled state from ``set_storage_path`` so the user
    doesn't get an info dialog saying 'already on default'."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    view.set_storage_path("C:/project/models", is_default=True)
    assert not _reset_storage_btn(view).isEnabled()

    view.set_storage_path("D:/elsewhere", is_default=False)
    assert _reset_storage_btn(view).isEnabled()


# ---- Hugging Face card -----------------------------------------------------


def _hf_token_edit(view):
    return view.findChild(QLineEdit, "HfTokenEdit")


def test_shortcuts_view_has_hf_token_widgets(qtbot):
    """Settings tab carries a Hugging Face card with a token field
    so users don't have to set ``HF_TOKEN`` env var by hand for
    GigaAM long-form."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    assert _hf_token_edit(view) is not None


def test_hf_token_field_is_password_masked(qtbot):
    """Tokens are sensitive — render as bullets, not plaintext, so
    the value isn't shoulder-surfed during a screenshare."""
    from PySide6.QtWidgets import QLineEdit
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    assert _hf_token_edit(view).echoMode() == QLineEdit.Password


def test_set_hf_token_prefills_field(qtbot):
    """The view's own setter — used by the controller on init to
    paint the persisted value into the field without firing the
    save signal back."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.set_hf_token("hf_persisted_value")
    assert _hf_token_edit(view).text() == "hf_persisted_value"


def test_hf_token_field_emits_signal_on_edit(qtbot):
    """Edit → focus loss → controller saves. Same auto-save pattern
    as the hotkey fields."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    edit = _hf_token_edit(view)
    edit.setText("hf_new_value")

    with qtbot.waitSignal(view.hf_token_changed, timeout=1000) as blocker:
        edit.editingFinished.emit()
    assert blocker.args == ["hf_new_value"]


def test_set_hf_token_does_not_re_emit(qtbot):
    """Programmatic prefill via ``set_hf_token`` must not echo back
    a signal — would cause an infinite save loop on init."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    emissions: list = []
    view.hf_token_changed.connect(emissions.append)
    view.set_hf_token("hf_persisted_value")
    assert emissions == []


# ---- Cancel-recording hotkey -----------------------------------------------


def _cancel_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "CancelHotkeyEdit")


def test_shortcuts_view_has_cancel_hotkey_field(qtbot):
    """The Hotkeys card carries a third row for the "discard buffer
    without transcribing" hotkey — the runtime supports it via
    ``StateManager.cancel_active_recording`` but until now it was
    never exposed to the user."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    assert _cancel_edit(view) is not None


def test_set_values_prefills_cancel_hotkey(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
        cancel_hotkey="ctrl+f6",
    )
    assert _cancel_edit(view).text() == "ctrl+f6"


def test_cancel_hotkey_returns_via_getter(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
        cancel_hotkey="ctrl+shift+x",
    )
    assert view.cancel_hotkey() == "ctrl+shift+x"


def test_save_payload_includes_cancel_hotkey(qtbot):
    """Edits to the cancel field must surface through the same
    ``save_requested`` payload the controller already listens on —
    otherwise a typed value never reaches config."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=False,
        cancel_hotkey="ctrl+f6",
    )

    edit = _cancel_edit(view)
    edit.clear()
    qtbot.keyClicks(edit, "ctrl+alt+x")

    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        edit.editingFinished.emit()

    assert blocker.args[0]["cancel_hotkey"] == "ctrl+alt+x"


def test_set_values_handles_legacy_callers_without_cancel_kw(qtbot):
    """Callers that still pass only the old three kwargs must keep
    working — controller tests + any external code shouldn't need a
    flag day to upgrade. Cancel field stays empty in that case."""
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
    )
    assert _cancel_edit(view).text() == ""
