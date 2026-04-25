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


def test_history_view_uses_pixel_scroll_mode(qtbot):
    """History table should scroll smoothly per pixel, not per row,
    matching the rest of the UI's scroll feel."""
    from PySide6.QtWidgets import QAbstractItemView
    from app.gui.views.history_view import HistoryView

    view = HistoryView()
    qtbot.addWidget(view)
    assert view._table.verticalScrollMode() == QAbstractItemView.ScrollPerPixel
