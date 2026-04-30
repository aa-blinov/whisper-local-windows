"""Bridge between the StateManager domain stack and the Qt UI.

Wraps a pre-built ``StateManager`` (and optional ``HotkeyListener``) into a
``QObject`` that exposes Qt signals for state transitions and history updates.
The controller owns no audio/transcription/clipboard logic itself — it only marshals
events from non-Qt threads (transcription pipeline, global hotkey thread)
back onto the Qt main thread via signals.
"""

from __future__ import annotations

import logging
import threading
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
    # Emitted while the backend is downloading model weights from
    # Hugging Face. ``current`` and ``total`` are byte counts (or 0 when
    # unknown), ``desc`` is the file description from huggingface_hub
    # (e.g. ``model.bin``). Progress comes from a non-Qt thread inside
    # tqdm — Qt auto-queues the signal cross-thread.
    download_progress = Signal(int, int, str)
    # File-transcription results, dispatched from the transcribe view.
    # The ``path`` argument lets the view ignore stale results if the
    # user picked a second file before the first finished.
    file_transcribed = Signal(str, str)            # (path, text)
    file_transcription_failed = Signal(str, str)   # (path, message)

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

        # Wire backend download progress (if backend supports it) through
        # to a Qt signal so the UI can show a percentage.
        self._wire_backend_progress()

    # ---- public API ---------------------------------------------------------

    @property
    def state_manager(self) -> _StateManagerLike:
        return self._state_manager

    def start(self) -> None:
        self._poll()
        self._timer.start()

    def current_state(self) -> Optional[str]:
        return self._last_state

    def request_model_change(
        self,
        new_model_size: str,
        compute_type: Optional[str] = None,
    ) -> bool:
        """Switch the active backend model. ``compute_type`` is forwarded to
        the state manager; ``None`` keeps the current setting."""
        return self._state_manager.request_model_change(
            new_model_size, compute_type=compute_type
        )

    def cancel_model_change(self) -> bool:
        """Forward the topbar's Cancel click to the state manager.

        Thin pass-through: cancel is a one-shot intent, no Qt-side
        bookkeeping needed. Falls back gracefully when the underlying
        state manager doesn't expose ``cancel_model_change`` (older
        fakes in tests) so the cancel button never crashes the UI.
        """
        target = getattr(self._state_manager, "cancel_model_change", None)
        if target is None:
            return False
        try:
            return bool(target())
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("StateManager.cancel_model_change raised: %s", exc)
            return False

    def list_input_devices(self) -> list:
        recorder = getattr(self._state_manager, "audio_recorder", None)
        if recorder is None or not hasattr(recorder, "list_input_devices"):
            return []
        try:
            return recorder.list_input_devices()
        except Exception:
            return []

    def current_input_device(self):
        recorder = getattr(self._state_manager, "audio_recorder", None)
        return getattr(recorder, "device", None) if recorder is not None else None

    def set_input_device(self, raw):
        """Switch the recorder's input device. Accepts ``None`` (system
        default), an int index, or a substring of the device name."""
        recorder = getattr(self._state_manager, "audio_recorder", None)
        if recorder is None or not hasattr(recorder, "set_device"):
            return None
        return recorder.set_device(raw)

    def active_provider(self) -> Optional[str]:
        """Pretty-printed ONNX Runtime EP that the loaded model is
        actually using (e.g. ``"CoreML"``, ``"CUDA"``, ``"CPU"``), or
        ``None`` while the model is unloaded / loading / errored.
        Proxies the inner backend so the UI can stay agnostic of
        which transport (subprocess vs in-process) is wired up.
        """
        backend = getattr(self._state_manager, "backend", None)
        if backend is None:
            return None
        getter = getattr(backend, "active_provider", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:  # pragma: no cover — defensive
            return None

    def transcribe_file_async(self, path: str) -> None:
        """Run ``backend.transcribe_file(path)`` on a worker thread and
        emit ``file_transcribed`` / ``file_transcription_failed`` with
        the result.

        Returns immediately so the UI thread isn't blocked.  If the
        backend isn't loaded (or the inner doesn't expose
        ``transcribe_file``), emits ``file_transcription_failed`` with
        a friendly error message.
        """
        backend = getattr(self._state_manager, "backend", None)
        if backend is None:
            self.file_transcription_failed.emit(
                path, "Backend unavailable.",
            )
            return
        if not getattr(backend, "health_check", lambda: False)():
            self.file_transcription_failed.emit(
                path,
                "Model isn't ready yet — wait for it to finish loading.",
            )
            return
        target = getattr(backend, "transcribe_file", None)
        if target is None:
            self.file_transcription_failed.emit(
                path,
                "This backend doesn't support file transcription.",
            )
            return

        def _run() -> None:
            try:
                text = target(path)
            except Exception as exc:  # pragma: no cover — defensive
                log.exception("transcribe_file raised for %s", path)
                self.file_transcription_failed.emit(path, str(exc))
                return
            if not text:
                self.file_transcription_failed.emit(
                    path,
                    "Transcription returned an empty result — "
                    "the file may be corrupt or the model didn't hear any speech.",
                )
                return
            self.file_transcribed.emit(path, text)

        threading.Thread(
            target=_run,
            daemon=True,
            name=f"transcribe-file:{path[-32:]}",
        ).start()

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

    def _wire_backend_progress(self) -> None:
        """Install a callback on the backend that re-emits progress as a
        Qt signal. Called once at construction; safe if the backend
        doesn't support ``set_progress_callback`` (no-op)."""
        backend = getattr(self._state_manager, "backend", None)
        if backend is None or not hasattr(backend, "set_progress_callback"):
            return
        try:
            backend.set_progress_callback(self._on_backend_progress)
        except Exception as exc:
            log.debug("set_progress_callback failed: %s", exc)

    def _on_backend_progress(self, current: int, total: int, desc: str) -> None:
        # Fired from a non-Qt thread inside tqdm.update. Qt auto-queues
        # signal emission to the main thread.
        self.download_progress.emit(int(current), int(total), str(desc))
