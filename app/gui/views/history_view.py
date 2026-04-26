"""History view — searchable table of past transcriptions."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
# Qt is imported above for the alignment flags used by the empty state.

from app.gui.smooth_scroll import apply_smooth_scroll
from app.model_mapping import alias_for
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)


_HEADERS = ("Time", "Text", "Model", "Language", "Duration")


class HistoryTableModel(QAbstractTableModel):
    def __init__(self, entries: Optional[Sequence[Any]] = None) -> None:
        super().__init__()
        self._entries: List[Any] = list(entries or [])

    # ---- Qt overrides -------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(_HEADERS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return _HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        entry = self._entries[index.row()]
        col = index.column()
        # Tooltip on the Text column carries the full transcription so
        # the user can hover-peek long entries without opening the
        # detail dialog. The Model column tooltip surfaces the full
        # canonical id (since the displayed value is the short alias).
        if role == Qt.ToolTipRole:
            if col == 1:
                return getattr(entry, "text", "")
            if col == 2:
                return getattr(entry, "model", "")
            return None
        if role != Qt.DisplayRole:
            return None
        if col == 0:
            return getattr(entry, "datetime_str", "")
        if col == 1:
            return getattr(entry, "text", "")
        if col == 2:
            # Map the canonical Hugging Face / engine id back to the
            # short registry alias (``large-v3``, ``gigaam-v2-ctc``).
            # Falls through to the original string for unregistered
            # canonicals.
            return alias_for(getattr(entry, "model", ""))
        if col == 3:
            return getattr(entry, "language", "")
        if col == 4:
            duration = getattr(entry, "duration", 0.0)
            return f"{duration:.1f}s"
        return None

    # ---- public API ---------------------------------------------------------

    def set_entries(self, entries: Sequence[Any]) -> None:
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()

    def entry_at(self, row: int) -> Any:
        if row < 0 or row >= len(self._entries):
            raise IndexError(row)
        return self._entries[row]


class HistoryDetailDialog(QDialog):
    """Modal dialog showing the full transcription text + metadata.

    The history table ellipsises long entries in the Text column, so
    this dialog is what the user opens (via double-click) when they
    actually want to read or copy the whole transcript.
    """

    def __init__(self, entry: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("HistoryDetailDialog")
        self.setWindowTitle("Transcription details")
        self.setModal(True)
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # ---- meta strip ----------------------------------------------------
        meta = QHBoxLayout()
        meta.setSpacing(8)
        for caption, value in (
            ("Time", str(getattr(entry, "datetime_str", ""))),
            ("Model", str(getattr(entry, "model", ""))),
            ("Language", str(getattr(entry, "language", ""))),
            ("Duration", f"{float(getattr(entry, 'duration', 0.0)):.1f}s"),
        ):
            chip = QLabel(f"{caption}: {value}", self)
            chip.setProperty("role", "badge")
            meta.addWidget(chip)
        meta.addStretch(1)
        layout.addLayout(meta)

        # ---- full text body ------------------------------------------------
        self._text = QPlainTextEdit(self)
        self._text.setObjectName("DetailText")
        self._text.setReadOnly(True)
        self._text.setPlainText(str(getattr(entry, "text", "")))
        layout.addWidget(self._text, 1)

        # ---- footer buttons ------------------------------------------------
        buttons = QHBoxLayout()
        copy_btn = QPushButton("Copy text", self)
        copy_btn.setObjectName("DetailCopyButton")
        copy_btn.setProperty("role", "primary")
        copy_btn.clicked.connect(self._on_copy_clicked)
        buttons.addWidget(copy_btn)
        buttons.addStretch(1)
        close_btn = QPushButton("Close", self)
        close_btn.setObjectName("DetailCloseButton")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _on_copy_clicked(self) -> None:
        QApplication.clipboard().setText(self._text.toPlainText())


class HistoryView(QWidget):
    clear_requested = Signal()
    copy_requested = Signal(str)
    export_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("HistoryView")

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(10)

        controls = QHBoxLayout()
        self._search = QLineEdit(self)
        self._search.setObjectName("HistorySearchEdit")
        self._search.setPlaceholderText("Search transcriptions…")
        self._search.textChanged.connect(self._on_search_changed)
        controls.addWidget(self._search, 1)

        self._copy_btn = QPushButton("Copy", self)
        self._copy_btn.setObjectName("CopyEntryButton")
        self._copy_btn.clicked.connect(self._on_copy_clicked)
        controls.addWidget(self._copy_btn)

        self._export_btn = QPushButton("Export", self)
        self._export_btn.setObjectName("ExportHistoryButton")
        self._export_btn.clicked.connect(self.export_requested.emit)
        controls.addWidget(self._export_btn)

        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setObjectName("ClearHistoryButton")
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        controls.addWidget(self._clear_btn)

        root.addLayout(controls)

        self._source_model = HistoryTableModel([])
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._source_model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setFilterKeyColumn(1)  # Text column

        # Wrap the table in a card so it reads as a defined surface
        # against the view background instead of floating with no
        # boundaries. The QStackedWidget swaps between the table card
        # and the empty-state placeholder when the entry count crosses
        # zero.
        self._stack = QStackedWidget(self)

        table_card = QFrame(self._stack)
        table_card.setProperty("role", "card")
        table_card.setFrameShape(QFrame.NoFrame)
        card_layout = QVBoxLayout(table_card)
        card_layout.setContentsMargins(2, 2, 2, 2)
        card_layout.setSpacing(0)

        self._table = QTableView(table_card)
        self._table.setObjectName("HistoryTable")
        self._table.setModel(self._proxy)
        self._table.setFrameShape(QTableView.NoFrame)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        apply_smooth_scroll(self._table)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setStretchLastSection(False)
        card_layout.addWidget(self._table)
        self._stack.addWidget(table_card)

        empty = QFrame(self._stack)
        empty.setObjectName("HistoryEmptyState")
        empty.setProperty("role", "card")
        empty.setFrameShape(QFrame.NoFrame)
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(40, 60, 40, 60)
        empty_layout.setSpacing(8)
        empty_layout.addStretch(1)
        title = QLabel("No transcriptions yet", empty)
        title.setProperty("role", "empty-title")
        title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(title)
        hint = QLabel(
            "Press the Start hotkey (Ctrl+F2 by default) and speak — "
            "transcriptions will land here.",
            empty,
        )
        hint.setProperty("role", "empty-hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        empty_layout.addWidget(hint)
        empty_layout.addStretch(2)
        self._empty_state = empty
        self._stack.addWidget(empty)
        # Pre-fetch the table-card reference so ``_update_empty_state``
        # can swap between them by widget identity.
        self._table_card = table_card

        root.addWidget(self._stack, 1)
        self._update_empty_state()

        footer = QHBoxLayout()
        self._count_label = QLabel("0 entries", self)
        self._count_label.setObjectName("HistoryCountLabel")
        self._count_label.setProperty("role", "muted")
        footer.addWidget(self._count_label)
        footer.addStretch(1)
        root.addLayout(footer)

        self._proxy.rowsInserted.connect(self._refresh_count)
        self._proxy.rowsRemoved.connect(self._refresh_count)
        self._proxy.modelReset.connect(self._refresh_count)
        self._proxy.layoutChanged.connect(self._refresh_count)

        # Double-click on any row opens the full-text detail dialog.
        self._table.doubleClicked.connect(self._on_row_double_clicked)

    # ---- public API ---------------------------------------------------------

    def set_entries(self, entries: Sequence[Any]) -> None:
        self._source_model.set_entries(entries)
        self._refresh_count()
        self._update_empty_state()

    # ---- internal -----------------------------------------------------------

    def _on_search_changed(self, text: str) -> None:
        self._proxy.setFilterFixedString(text)
        self._refresh_count()

    def _refresh_count(self, *_args) -> None:
        count = self._proxy.rowCount()
        self._count_label.setText(f"{count} entries")

    def _update_empty_state(self) -> None:
        if self._source_model.rowCount() == 0:
            self._stack.setCurrentWidget(self._empty_state)
        else:
            self._stack.setCurrentWidget(self._table_card)

    def _on_copy_clicked(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        source_index = self._proxy.mapToSource(rows[0])
        entry = self._source_model.entry_at(source_index.row())
        text = getattr(entry, "text", "")
        if text:
            self.copy_requested.emit(text)

    def _on_row_double_clicked(self, proxy_index) -> None:
        if not proxy_index.isValid():
            return
        source_index = self._proxy.mapToSource(proxy_index)
        try:
            entry = self._source_model.entry_at(source_index.row())
        except IndexError:
            return
        self._open_detail_for_entry(entry)

    def _open_detail_for_entry(self, entry: Any) -> None:
        """Pop the detail dialog for the given entry. Extracted so
        tests can monkeypatch the dialog opening without bringing up
        a real modal window."""
        dialog = HistoryDetailDialog(entry, self)
        dialog.exec()
