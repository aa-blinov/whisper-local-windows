"""Poll a backend status source and emit canonical TopBar status values."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal


log = logging.getLogger(__name__)


_StatusFetcher = Callable[[], str]


def _map_status(raw: str) -> Tuple[str, str]:
    """Translate DockerBackendManager status strings into TopBar vocabulary."""
    if raw == "running":
        return "running", "Backend running"
    if raw == "stopped":
        return "stopped", "Backend stopped"
    if raw == "not_found":
        return "stopped", "Container not found"
    if raw == "error":
        return "error", "Docker unavailable"
    return "unknown", f"Status: {raw}"


class BackendStatusPoller(QObject):
    status_changed = Signal(str, str)  # canonical, label

    DEFAULT_INTERVAL_MS = 3000

    def __init__(
        self,
        fetcher: _StatusFetcher,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._fetcher = fetcher
        self._last_canonical: Optional[str] = None
        self._timer = QTimer(self)
        self._timer.setInterval(max(100, int(interval_ms)))
        self._timer.timeout.connect(self.tick)

    def start(self) -> None:
        self.tick()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def tick(self) -> None:
        try:
            raw = self._fetcher()
        except Exception as exc:
            log.warning("Backend status fetcher raised: %s", exc)
            canonical, label = "error", f"Backend error: {exc}"
        else:
            canonical, label = _map_status(raw)

        if canonical == self._last_canonical:
            return
        self._last_canonical = canonical
        self.status_changed.emit(canonical, label)
