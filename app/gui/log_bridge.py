"""Forward stdlib logging records into a Qt signal for UI consumption."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


class _SignalHandler(logging.Handler):
    def __init__(
        self,
        on_line: Callable[[str], None],
        on_record: Callable[[str, str, str, str], None],
    ) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        self._on_line = on_line
        self._on_record = on_record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Force the formatter to compute ``asctime`` on the record
            # so the structured callback can reuse it without a second
            # strftime call.
            line = self.format(record)
            asctime = getattr(record, "asctime", "") or ""
            self._on_record(
                asctime, record.levelname, record.name, record.getMessage()
            )
            self._on_line(line)
        except Exception:  # pragma: no cover — defensive
            self.handleError(record)


class QtLogBridge(QObject):
    # Legacy signal — formatted single-string line. Kept so any
    # external listener / test that already wired ``line_received``
    # keeps working.
    line_received = Signal(str)
    # Structured signal — ``(asctime, levelname, logger_name, message)``.
    # The LogsView listens to this so it can colour records by level
    # and filter by logger name.
    record_received = Signal(str, str, str, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._handler = _SignalHandler(self._emit_line, self._emit_record)

    def handler(self) -> logging.Handler:
        return self._handler

    def install(self, logger: Optional[logging.Logger] = None) -> None:
        (logger or logging.getLogger()).addHandler(self._handler)

    def uninstall(self, logger: Optional[logging.Logger] = None) -> None:
        (logger or logging.getLogger()).removeHandler(self._handler)

    def _emit_line(self, text: str) -> None:
        self.line_received.emit(text)

    def _emit_record(
        self, asctime: str, level: str, name: str, message: str
    ) -> None:
        self.record_received.emit(asctime, level, name, message)
