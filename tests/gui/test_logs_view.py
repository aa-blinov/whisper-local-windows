"""Tests for the LogsView widget."""

from PySide6.QtWidgets import QCheckBox, QPlainTextEdit, QPushButton


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


def test_logs_view_wraps_long_lines(qtbot):
    """Long log lines should wrap to widget width, not push a horizontal scrollbar."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    textbox = view.findChild(QPlainTextEdit)
    assert textbox.lineWrapMode() == QPlainTextEdit.WidgetWidth


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


def test_append_record_renders_message_text(qtbot):
    """Structured records still surface their message text in the
    plain-text view (so ``toPlainText`` can be searched / copied)."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_record("12:00:01", "INFO", "app.state_manager", "switching model")
    content = _text(view)
    assert "switching model" in content
    assert "INFO" in content
    assert "app.state_manager" in content


def test_append_record_emits_inline_color_for_levels(qtbot):
    """Each record renders inline-styled HTML with a level-specific
    colour so the eye can pick errors out of a busy stream."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_record("12:00:01", "ERROR", "app.state_manager", "boom")

    textbox = view.findChild(QPlainTextEdit)
    html = textbox.document().toHtml()
    assert "[ERROR]" in html
    # Danger / red tint must appear somewhere in the styled span.
    assert "fca5a5" in html.lower()


def test_append_record_skips_network_loggers_by_default(qtbot):
    """``httpx`` floods the view with HTTP debug noise during model
    loads; hide it unless the user opts in via the toggle."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_record("12:00:01", "INFO", "httpx", "GET /foo 200")
    view.append_record("12:00:02", "INFO", "app.state_manager", "important")

    content = _text(view)
    assert "GET /foo" not in content
    assert "important" in content


def test_append_record_shows_network_when_toggle_on(qtbot):
    """Flipping the 'Show network logs' checkbox should let httpx
    records through."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    toggle = next(
        b for b in view.findChildren(QCheckBox) if b.objectName() == "ShowNetworkLogs"
    )
    toggle.setChecked(True)

    view.append_record("12:00:01", "INFO", "httpx", "GET /foo 200")
    assert "GET /foo" in _text(view)


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
