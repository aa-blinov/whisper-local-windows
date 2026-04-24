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
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
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
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        entry = self._entries[index.row()]
        col = index.column()
        if col == 0:
            return getattr(entry, "datetime_str", "")
        if col == 1:
            return getattr(entry, "text", "")
        if col == 2:
            return getattr(entry, "model", "")
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


class HistoryView(QWidget):
    clear_requested = Signal()
    copy_requested = Signal(str)
    export_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("HistoryView")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("History", self)
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        root.addLayout(header)

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

        self._table = QTableView(self)
        self._table.setObjectName("HistoryTable")
        self._table.setModel(self._proxy)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setStretchLastSection(False)
        root.addWidget(self._table, 1)

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

    # ---- public API ---------------------------------------------------------

    def set_entries(self, entries: Sequence[Any]) -> None:
        self._source_model.set_entries(entries)
        self._refresh_count()

    # ---- internal -----------------------------------------------------------

    def _on_search_changed(self, text: str) -> None:
        self._proxy.setFilterFixedString(text)
        self._refresh_count()

    def _refresh_count(self, *_args) -> None:
        count = self._proxy.rowCount()
        self._count_label.setText(f"{count} entries")

    def _on_copy_clicked(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        source_index = self._proxy.mapToSource(rows[0])
        entry = self._source_model.entry_at(source_index.row())
        text = getattr(entry, "text", "")
        if text:
            self.copy_requested.emit(text)
