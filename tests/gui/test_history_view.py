"""Tests for the HistoryView and its backing table model."""

from dataclasses import dataclass

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QTableView


@dataclass
class FakeEntry:
    timestamp: float
    text: str
    duration: float
    model: str
    language: str

    @property
    def datetime_str(self) -> str:
        return "00:00:00 01.01.2026"

    @property
    def short_text(self) -> str:
        return self.text[:50]


def _make_entries(n: int = 3):
    return [
        FakeEntry(
            timestamp=float(i),
            text=f"entry text {i}",
            duration=float(i + 1),
            model="large-v3",
            language="ru",
        )
        for i in range(n)
    ]


# ---- Table model ------------------------------------------------------------


def test_history_model_reports_row_and_column_counts(qtbot):
    from app.gui.views.history_view import HistoryTableModel

    entries = _make_entries(3)
    model = HistoryTableModel(entries)
    assert model.rowCount() == 3
    assert model.columnCount() == 5


def test_history_model_row_zero_contains_first_entry(qtbot):
    from app.gui.views.history_view import HistoryTableModel

    entries = _make_entries(3)
    model = HistoryTableModel(entries)

    row_zero_text = model.data(model.index(0, 1), Qt.DisplayRole)
    assert row_zero_text == "entry text 0"


def test_history_model_headers_exposed(qtbot):
    from app.gui.views.history_view import HistoryTableModel

    model = HistoryTableModel([])
    headers = [
        model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
        for i in range(model.columnCount())
    ]
    assert headers[0].lower().startswith("time")
    assert "text" in headers[1].lower()
    assert "model" in headers[2].lower()
    assert "lang" in headers[3].lower()
    assert "duration" in headers[4].lower()


def test_history_model_set_entries_replaces_data(qtbot):
    from app.gui.views.history_view import HistoryTableModel

    model = HistoryTableModel([])
    assert model.rowCount() == 0

    model.set_entries(_make_entries(2))
    assert model.rowCount() == 2


def test_history_model_entry_at_returns_original(qtbot):
    from app.gui.views.history_view import HistoryTableModel

    entries = _make_entries(3)
    model = HistoryTableModel(entries)
    assert model.entry_at(0) is entries[0]
    assert model.entry_at(2) is entries[2]
    with pytest.raises(IndexError):
        model.entry_at(99)


# ---- View -------------------------------------------------------------------


def _table(view) -> QTableView:
    return view.findChild(QTableView, "HistoryTable")


def test_history_view_has_expected_widgets(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)

    assert _table(view) is not None
    assert view.findChild(QLineEdit, "HistorySearchEdit") is not None
    assert view.findChild(QPushButton, "CopyEntryButton") is not None
    assert view.findChild(QPushButton, "ClearHistoryButton") is not None
    assert view.findChild(QLabel, "HistoryCountLabel") is not None


def test_set_entries_populates_table(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)

    view.set_entries(_make_entries(3))
    table = _table(view)
    assert table.model().rowCount() == 3


def test_search_filter_narrows_rows(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)

    entries = [
        FakeEntry(0.0, "apple pie", 1.0, "m", "en"),
        FakeEntry(1.0, "banana bread", 1.0, "m", "en"),
        FakeEntry(2.0, "cherry cake", 1.0, "m", "en"),
    ]
    view.set_entries(entries)

    search = view.findChild(QLineEdit, "HistorySearchEdit")
    search.setText("banana")

    table = _table(view)
    proxy = table.model()
    assert proxy.rowCount() == 1
    assert proxy.data(proxy.index(0, 1), Qt.DisplayRole) == "banana bread"


def test_count_label_reflects_visible_rows(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)

    view.set_entries(_make_entries(3))

    label = view.findChild(QLabel, "HistoryCountLabel")
    assert "3" in label.text()

    search = view.findChild(QLineEdit, "HistorySearchEdit")
    search.setText("entry text 1")
    assert "1" in label.text()


def test_clear_button_emits_clear_requested(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    view.show()

    btn = view.findChild(QPushButton, "ClearHistoryButton")
    with qtbot.waitSignal(view.clear_requested, timeout=1000):
        qtbot.mouseClick(btn, Qt.LeftButton)


def test_copy_button_emits_copy_requested_with_text(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    view.show()

    entries = _make_entries(3)
    view.set_entries(entries)

    table = _table(view)
    table.selectRow(1)

    btn = view.findChild(QPushButton, "CopyEntryButton")
    with qtbot.waitSignal(view.copy_requested, timeout=1000) as blocker:
        qtbot.mouseClick(btn, Qt.LeftButton)

    assert blocker.args == ["entry text 1"]


def test_copy_button_does_not_emit_when_no_selection(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    view.show()
    view.set_entries(_make_entries(3))

    emissions: list[str] = []
    view.copy_requested.connect(emissions.append)

    btn = view.findChild(QPushButton, "CopyEntryButton")
    qtbot.mouseClick(btn, Qt.LeftButton)

    assert emissions == []


def test_history_model_exposes_full_text_via_tooltip(qtbot):
    """Long transcripts are ellipsised in the table, so the tooltip
    must carry the full text — quick hover-peek without opening the
    detail dialog."""
    from app.gui.views.history_view import HistoryTableModel

    long_text = "a long transcription that overflows the column width " * 4
    entry = FakeEntry(
        timestamp=0.0, text=long_text, duration=1.0,
        model="large-v3", language="ru",
    )
    model = HistoryTableModel([entry])
    text_index = model.index(0, 1)  # Text column
    assert model.data(text_index, Qt.ToolTipRole) == long_text


def test_history_detail_dialog_shows_full_entry(qtbot):
    """The detail dialog must surface every field — full text, time,
    model, language, duration — so the user can read what got
    transcribed without round-tripping through the clipboard."""
    from app.gui.views.history_view import HistoryDetailDialog

    entry = FakeEntry(
        timestamp=0.0,
        text="full transcription body that's too long for the table cell",
        duration=12.5,
        model="large-v3",
        language="ru",
    )
    dialog = HistoryDetailDialog(entry)
    qtbot.addWidget(dialog)
    assert dialog._text.toPlainText() == entry.text
    rendered = " ".join(
        lbl.text() for lbl in dialog.findChildren(QLabel)
    )
    assert "large-v3" in rendered
    assert "ru" in rendered
    assert "12.5" in rendered


def test_history_detail_dialog_copy_button_copies_text(qtbot):
    """The dialog's Copy button must put the full text on the
    clipboard so the user can paste it elsewhere."""
    from PySide6.QtWidgets import QApplication
    from app.gui.views.history_view import HistoryDetailDialog

    entry = FakeEntry(
        timestamp=0.0, text="transcribed words", duration=1.0,
        model="large-v3", language="ru",
    )
    dialog = HistoryDetailDialog(entry)
    qtbot.addWidget(dialog)

    QApplication.clipboard().clear()
    btn = next(
        b for b in dialog.findChildren(QPushButton)
        if b.objectName() == "DetailCopyButton"
    )
    btn.click()
    assert QApplication.clipboard().text() == "transcribed words"


def test_history_view_double_click_opens_detail(qtbot, monkeypatch):
    """Double-clicking a row must surface the full transcript, not
    just toggle selection — that's how the user reads long entries
    when the table cell ellipsises."""
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)

    captured: list = []

    def fake_open(self, entry):
        captured.append(entry)

    monkeypatch.setattr(HistoryView, "_open_detail_for_entry", fake_open)

    entries = _make_entries(2)
    view.set_entries(entries)

    # Simulate double-click on first row — bypass mouse mechanics by
    # invoking the slot directly via the underlying signal.
    proxy_index = view._proxy.index(0, 1)
    view._table.doubleClicked.emit(proxy_index)

    assert len(captured) == 1
    assert captured[0] is entries[0]


def test_history_view_uses_pixel_scroll_mode(qtbot):
    """History table should scroll smoothly per pixel, not per row,
    matching the rest of the UI's scroll feel."""
    from PySide6.QtWidgets import QAbstractItemView
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    assert view._table.verticalScrollMode() == QAbstractItemView.ScrollPerPixel


def test_history_view_shows_empty_state_when_no_entries(qtbot):
    """A blank table is unfriendly — surface a 'press the hotkey'
    placeholder when there's nothing to show."""
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    view.set_entries([])
    assert view._stack.currentWidget() is view._empty_state


def test_history_view_swaps_to_table_when_entries_arrive(qtbot):
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    view.set_entries(_make_entries(2))
    assert view._stack.currentWidget() is view._table
