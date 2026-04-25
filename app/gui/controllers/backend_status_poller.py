"""Poll a backend status source on a worker thread and surface the result via signals.

Running the fetcher on the UI thread is a problem when it talks to a slow or
unresponsive Docker daemon — the event loop freezes for the duration of every
poll. This module offloads the call to a single-worker thread pool and uses a
queued Qt signal to deliver the result back to the main thread.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

from PySide6.QtCore import QObject, QTimer, Signal


log = logging.getLogger(__name__)


_StatusFetcher = Callable[[], str]


def _map_status(raw: str) -> Tuple[str, str]:
    """Translate ``TranscriptionBackend.status()`` values into TopBar vocabulary.

    The backend exposes ``stopped | loading | ready | error``; the topbar
    pill speaks ``running | stopped | error | unknown``. Loading is
    rendered as a ``stopped`` pill with a hint label so the user can see
    "Loading model…" without it looking like a green-light "ready".
    """
    if raw == "ready":
        return "running", "Model ready"
    if raw == "loading":
        return "stopped", "Loading model\u2026"
    if raw == "error":
        return "error", "Backend error"
    if raw == "stopped":
        return "stopped", "Model not loaded"
    return "unknown", f"Status: {raw}"


class BackendStatusPoller(QObject):
    status_changed = Signal(str, str)  # canonical, label

    # Internal: emitted by the worker thread, consumed by the main thread.
    # Connected via Qt.AutoConnection — auto-promotes to QueuedConnection
    # because emitter and receiver live on different threads.
    _fetch_completed = Signal(str, str)

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
        self._in_flight = False
        self._stopped = False

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="backend-status"
        )

        self._timer = QTimer(self)
        self._timer.setInterval(max(100, int(interval_ms)))
        self._timer.timeout.connect(self.tick)

        self._fetch_completed.connect(self._on_fetch_completed)

    # ---- public API ---------------------------------------------------------

    def start(self) -> None:
        self.tick()
        self._timer.start()

    def stop(self) -> None:
        self._stopped = True
        self._timer.stop()
        self._executor.shutdown(wait=False)

    def tick(self) -> None:
        """Schedule a fetch on the worker thread.

        Returns immediately. If a previous tick is still running, this call is
        a no-op so slow fetchers can't pile up a backlog of pending pulls.
        """
        if self._stopped or self._in_flight:
            return
        self._in_flight = True
        try:
            self._executor.submit(self._fetch_in_thread)
        except RuntimeError:
            # Executor already shut down — treat as no-op.
            self._in_flight = False

    # ---- worker-side helpers (run off the UI thread) -----------------------

    def _fetch_in_thread(self) -> None:
        try:
            raw = self._fetcher()
        except Exception as exc:
            log.warning("Backend status fetcher raised: %s", exc)
            canonical, label = "error", f"Backend error: {exc}"
        else:
            canonical, label = _map_status(raw)
        # Cross-thread emit — Qt queues this onto the main thread automatically.
        self._fetch_completed.emit(canonical, label)

    # ---- main-thread slot ---------------------------------------------------

    def _on_fetch_completed(self, canonical: str, label: str) -> None:
        self._in_flight = False
        if self._stopped:
            return
        if canonical == self._last_canonical:
            return
        self._last_canonical = canonical
        self.status_changed.emit(canonical, label)
