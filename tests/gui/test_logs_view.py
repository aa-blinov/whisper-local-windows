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


def test_logs_view_has_search_field(qtbot):
    from PySide6.QtWidgets import QLineEdit
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    assert view.findChild(QLineEdit, "LogsSearchEdit") is not None


def test_logs_search_filters_buffered_records(qtbot):
    """Typing in the search field hides records that don't match;
    re-rendering uses the in-memory ring buffer so previously-shown
    entries can come back instantly when the filter clears."""
    from PySide6.QtWidgets import QLineEdit, QPlainTextEdit
    from app.gui.views.logs_view import LogsView

    # search_debounce_ms=0 so the timer fires on the next event-loop tick
    view = LogsView(search_debounce_ms=0)
    qtbot.addWidget(view)
    view.append_record("12:00:00", "INFO", "app.state_manager", "model loaded")
    view.append_record("12:00:01", "WARNING", "app.audio_recorder", "low gain")
    view.append_record("12:00:02", "INFO", "app.state_manager", "transcribed")

    text = view.findChild(QPlainTextEdit, "LogsTextArea").toPlainText()
    assert "model loaded" in text
    assert "low gain" in text

    search = view.findChild(QLineEdit, "LogsSearchEdit")
    search.setText("low")
    qtbot.wait(50)  # allow debounce timer to fire

    after = view.findChild(QPlainTextEdit, "LogsTextArea").toPlainText()
    assert "low gain" in after
    assert "model loaded" not in after
    assert "transcribed" not in after

    search.setText("")
    qtbot.wait(50)
    restored = view.findChild(QPlainTextEdit, "LogsTextArea").toPlainText()
    assert "model loaded" in restored
    assert "transcribed" in restored


def test_logs_search_matches_logger_name(qtbot):
    from PySide6.QtWidgets import QLineEdit, QPlainTextEdit
    from app.gui.views.logs_view import LogsView

    view = LogsView(search_debounce_ms=0)
    qtbot.addWidget(view)
    view.append_record("12:00:00", "INFO", "app.state_manager", "alpha")
    view.append_record("12:00:01", "INFO", "app.audio_recorder", "beta")

    search = view.findChild(QLineEdit, "LogsSearchEdit")
    search.setText("audio")
    qtbot.wait(50)
    text = view.findChild(QPlainTextEdit, "LogsTextArea").toPlainText()
    assert "beta" in text
    assert "alpha" not in text


# ---- Debounce ---------------------------------------------------------------


def test_logs_search_debounce_does_not_rerender_immediately(qtbot):
    """After typing, the textbox must NOT be filtered until the debounce
    timer fires.  Every keystroke rebuilding 5 000 HTML lines is the
    bug we're fixing."""
    from app.gui.views.logs_view import LogsView

    DEBOUNCE_MS = 120
    view = LogsView(search_debounce_ms=DEBOUNCE_MS)
    qtbot.addWidget(view)

    view.append_record("12:00:00", "INFO", "app.state_manager", "hello world")
    view.append_record("12:00:01", "INFO", "app.state_manager", "other line")

    # Simulate keystroke — timer is now armed but hasn't fired yet.
    view._on_search_changed("hello")

    # Immediately after: both records must still be visible (no rerender yet).
    assert "other line" in view._text.toPlainText(), (
        "_rerender must not fire synchronously on each keystroke"
    )

    # After the debounce window: only the matching record should survive.
    qtbot.wait(DEBOUNCE_MS + 60)
    assert "other line" not in view._text.toPlainText()
    assert "hello world" in view._text.toPlainText()


def test_logs_search_debounce_rapid_keystrokes_single_rerender(qtbot):
    """Five rapid keystrokes must produce exactly one rerender, not five.

    We verify this by checking that the textbox stays unfiltered right up
    until the debounce window, then filters correctly in one shot.
    """
    from app.gui.views.logs_view import LogsView

    DEBOUNCE_MS = 120
    view = LogsView(search_debounce_ms=DEBOUNCE_MS)
    qtbot.addWidget(view)

    for i in range(20):
        view.append_record("12:00:00", "INFO", "app.x", f"msg {i}")

    # Simulate rapid typing — each call restarts the timer.
    for prefix in ("e", "er", "err", "erro", "error"):
        view._on_search_changed(prefix)

    # Still unfiltered (timer keeps being restarted, hasn't fired).
    assert view._text.toPlainText() != "", (
        "Rapid keystrokes must not rerender until the debounce fires"
    )

    # Wait for the debounce to settle.
    qtbot.wait(DEBOUNCE_MS + 60)
    # "error" query matches nothing in "msg N" → textbox should be empty.
    assert view._text.toPlainText().strip() == ""


def test_logs_search_debounce_timer_is_single_shot(qtbot):
    """The debounce QTimer must be single-shot so it doesn't keep
    re-rendering the log on a fixed interval after the user stops typing."""
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    assert view._search_timer.isSingleShot()


def test_logs_clear_drops_buffered_records(qtbot):
    """Clear should wipe both the visible textbox AND the buffer
    feeding the filter — otherwise a stale search re-rendered from
    the buffer would resurrect the cleared entries."""
    from PySide6.QtWidgets import QPlainTextEdit
    from app.gui.views.logs_view import LogsView

    view = LogsView()
    qtbot.addWidget(view)
    view.append_record("12:00:00", "INFO", "app.state_manager", "alpha")
    view.clear()
    # Re-trigger render via toggle — anything buffered would appear.
    view._on_toggle_network(True)
    text = view.findChild(QPlainTextEdit, "LogsTextArea").toPlainText()
    assert "alpha" not in text
