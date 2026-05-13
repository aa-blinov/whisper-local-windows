"""Tests for the Toast confirmation banner."""

from PySide6.QtWidgets import QWidget


def test_toast_starts_hidden(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    qtbot.addWidget(parent)
    toast = Toast(parent=parent)
    assert not toast.isVisible()


def test_toast_show_message_displays_text_and_becomes_visible(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    parent.resize(800, 600)
    qtbot.addWidget(parent)
    parent.show()

    toast = Toast(parent=parent)
    toast.show_message("hello world", duration_ms=2000)

    assert toast.isVisible()
    assert "hello world" in toast._body.text()


def test_toast_show_message_truncates_long_previews(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    toast = Toast(parent=parent)
    long_text = "a" * 200
    toast.show_message(long_text)

    text = toast._body.text()
    assert text.endswith("…")
    assert len(text) < len(long_text)


def test_toast_ignores_empty_text(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.show()
    toast = Toast(parent=parent)
    toast.show_message("")
    assert not toast.isVisible()
    toast.show_message("   \n  ")
    assert not toast.isVisible()


def test_toast_auto_hides_after_duration(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    parent.resize(800, 600)
    qtbot.addWidget(parent)
    parent.show()

    toast = Toast(parent=parent)
    toast.show_message("ping", duration_ms=600)

    assert toast.isVisible()
    qtbot.waitUntil(lambda: not toast.isVisible(), timeout=2000)


def test_toast_hide_message_stops_timer_and_hides(qtbot):
    from app.gui.widgets.toast import Toast

    parent = QWidget()
    parent.resize(800, 600)
    qtbot.addWidget(parent)
    parent.show()

    toast = Toast(parent=parent)
    toast.show_message("ping", duration_ms=10000)
    toast.hide_message()
    assert not toast.isVisible()
    assert not toast._timer.isActive()
