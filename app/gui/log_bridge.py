"""Forward stdlib logging records into a Qt signal for UI consumption."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


class _SignalHandler(logging.Handler):
    def __init__(self, on_line: Callable[[str], None]) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        self._on_line = on_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._on_line(self.format(record))
        except Exception:  # pragma: no cover — defensive
            self.handleError(record)


class QtLogBridge(QObject):
    line_received = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._handler = _SignalHandler(self._emit_line)

    def handler(self) -> logging.Handler:
        return self._handler

    def install(self, logger: Optional[logging.Logger] = None) -> None:
        (logger or logging.getLogger()).addHandler(self._handler)

    def uninstall(self, logger: Optional[logging.Logger] = None) -> None:
        (logger or logging.getLogger()).removeHandler(self._handler)

    def _emit_line(self, text: str) -> None:
        self.line_received.emit(text)
