"""Tests for the ShortcutsView."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLineEdit, QPushButton


def _start_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StartHotkeyEdit")


def _stop_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StopHotkeyEdit")


def _auto_paste_cb(view) -> QCheckBox:
    return view.findChild(QCheckBox, "AutoPasteCheckbox")


def _reset_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "ResetShortcutsButton")


def test_shortcuts_view_has_expected_widgets(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    assert _start_edit(view) is not None
    assert _stop_edit(view) is not None
    assert _auto_paste_cb(view) is not None
    assert _reset_btn(view) is not None


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


def test_reset_button_emits_reset_requested(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    btn = _reset_btn(view)
    with qtbot.waitSignal(view.reset_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


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
