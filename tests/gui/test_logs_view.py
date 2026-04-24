"""Tests for the LogsView widget."""

from PySide6.QtWidgets import QPlainTextEdit, QPushButton


def _text(view) -> str:
    return view.findChild(QPlainTextEdit).toPlainText()


def test_logs_view_starts_empty(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    assert _text(view) == ""


def test_logs_view_textbox_is_read_only(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    textbox = view.findChild(QPlainTextEdit)
    assert textbox.isReadOnly()


def test_append_line_adds_single_line(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_line("hello")
    assert _text(view).strip() == "hello"


def test_append_line_accumulates(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    for line in ("one", "two", "three"):
        view.append_line(line)
    content = _text(view)
    assert "one" in content and "two" in content and "three" in content
    assert content.index("one") < content.index("two") < content.index("three")


def test_clear_empties_textbox(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_line("before clear")
    view.clear()
    assert _text(view) == ""


def test_clear_button_triggers_clear(qtbot):
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_line("data")

    clear_btn = next(
        b for b in view.findChildren(QPushButton) if b.objectName() == "ClearLogsButton"
    )
    clear_btn.click()
    assert _text(view) == ""


def test_append_line_respects_buffer_cap(qtbot):
    """When cap is reached, oldest lines should fall off."""
    from app.gui.views.logs_view import LogsView

    view = LogsView(max_lines=3)
    qtbot.addWidget(view)
    for line in ("a", "b", "c", "d"):
        view.append_line(line)

    content = _text(view)
    assert "a" not in content
    assert "b" in content and "c" in content and "d" in content
