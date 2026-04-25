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
