"""Tests for the ShortcutsView."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLineEdit, QPushButton


def _start_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StartHotkeyEdit")


def _stop_edit(view) -> QLineEdit:
    return view.findChild(QLineEdit, "StopHotkeyEdit")


def _auto_paste_cb(view) -> QCheckBox:
    return view.findChild(QCheckBox, "AutoPasteCheckbox")


def _save_btn(view) -> QPushButton:
    return view.findChild(QPushButton, "SaveShortcutsButton")


def test_shortcuts_view_has_expected_widgets(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)

    assert _start_edit(view) is not None
    assert _stop_edit(view) is not None
    assert _auto_paste_cb(view) is not None
    assert _save_btn(view) is not None


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


def test_save_button_emits_save_requested_with_values(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=True,
    )

    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        qtbot.mouseClick(_save_btn(view), Qt.LeftButton)

    payload = blocker.args[0]
    assert payload == {
        "start_hotkey": "ctrl+f2",
        "stop_hotkey": "ctrl+f3",
        "auto_paste": True,
    }


def test_save_reflects_user_edits(qtbot):
    from app.gui.views.shortcuts_view import ShortcutsView

    view = ShortcutsView()
    qtbot.addWidget(view)
    view.show()

    view.set_values(
        start_hotkey="ctrl+f2",
        stop_hotkey="ctrl+f3",
        auto_paste=False,
    )

    start = _start_edit(view)
    start.clear()
    qtbot.keyClicks(start, "ctrl+alt+1")

    _auto_paste_cb(view).setChecked(True)

    with qtbot.waitSignal(view.save_requested, timeout=1000) as blocker:
        qtbot.mouseClick(_save_btn(view), Qt.LeftButton)

    payload = blocker.args[0]
    assert payload["start_hotkey"] == "ctrl+alt+1"
    assert payload["auto_paste"] is True
