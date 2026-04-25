"""Bridge between the StateManager domain stack and the Qt UI.

Wraps a pre-built ``StateManager`` (and optional ``HotkeyListener``) into a
``QObject`` that exposes Qt signals for state transitions and history updates.
The controller owns no audio/Wyoming/clipboard logic itself — it only marshals
events from non-Qt threads (transcription pipeline, global hotkey thread)
back onto the Qt main thread via signals.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

from PySide6.QtCore import QObject, QTimer, Signal


log = logging.getLogger(__name__)


class _StateManagerLike(Protocol):
    history_update_callback: Any  # Optional[Callable[[], None]]

    def get_current_state(self) -> str: ...
    def request_model_change(self, new_model_size: str) -> bool: ...
    def shutdown(self) -> None: ...


class _HotkeyListenerLike(Protocol):
    def stop_listening(self) -> None: ...


class RecordingController(QObject):
    state_changed = Signal(str)
    history_updated = Signal()

    DEFAULT_POLL_INTERVAL_MS = 200

    def __init__(
        self,
        state_manager: _StateManagerLike,
        hotkey_listener: Optional[_HotkeyListenerLike] = None,
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._state_manager = state_manager
        self._hotkey_listener = hotkey_listener
        self._last_state: Optional[str] = None
        self._shutdown_done = False

        self._timer = QTimer(self)
        self._timer.setInterval(max(50, int(poll_interval_ms)))
        self._timer.timeout.connect(self._poll)

        # StateManager will invoke this from the transcription thread; the
        # signal connection is auto-queued onto the main thread.
        self._state_manager.history_update_callback = self._on_history_update

    # ---- public API ---------------------------------------------------------

    @property
    def state_manager(self) -> _StateManagerLike:
        return self._state_manager

    def start(self) -> None:
        self._poll()
        self._timer.start()

    def current_state(self) -> Optional[str]:
        return self._last_state

    def request_model_change(self, new_model_size: str) -> bool:
        return self._state_manager.request_model_change(new_model_size)

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._timer.stop()
        if self._hotkey_listener is not None:
            try:
                self._hotkey_listener.stop_listening()
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("Hotkey listener stop raised: %s", exc)
        try:
            self._state_manager.shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("StateManager shutdown raised: %s", exc)
        # Drop the callback so StateManager no longer holds a reference back.
        self._state_manager.history_update_callback = None

    # ---- internal -----------------------------------------------------------

    def _poll(self) -> None:
        if self._shutdown_done:
            return
        try:
            state = self._state_manager.get_current_state()
        except Exception as exc:
            log.warning("StateManager.get_current_state raised: %s", exc)
            return
        if state == self._last_state:
            return
        self._last_state = state
        self.state_changed.emit(state)

    def _on_history_update(self) -> None:
        # Called on the transcription pipeline thread. The signal connection
        # is queued cross-thread, so subscribers see it on the Qt main thread.
        self.history_updated.emit()
