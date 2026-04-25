"""Controller wiring views to the domain layer (config, state, backends)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

# QApplication is imported above for the clipboard helper; reuse it for
# explicit ``quit()`` calls from tray actions.

from app.gui.controllers.backend_status_poller import BackendStatusPoller
from app.gui.main_window import MainWindow
from app.model_mapping import alias_for, canonical_for, get_model
from app.utils import is_model_cached


log = logging.getLogger(__name__)


class _ConfigLike(Protocol):
    def get_setting(self, section: str, key: str) -> Any: ...
    def update_user_setting(self, section: str, key: str, value: Any) -> None: ...


class _HistoryLike(Protocol):
    def get_entries(self) -> list: ...
    def clear_history(self) -> None: ...


class _RecordingLike(Protocol):
    state_changed: Any
    history_updated: Any

    def request_model_change(self, canonical: str) -> bool: ...


class _TrayLike(Protocol):
    show_requested: Any
    quit_requested: Any

    def set_state(self, state: str) -> None: ...


class AppController(QObject):
    def __init__(
        self,
        config: _ConfigLike,
        window: MainWindow,
        history: Optional[_HistoryLike] = None,
        backend_status_fetcher: Optional[Callable[[], str]] = None,
        recording: Optional[_RecordingLike] = None,
        tray: Optional[_TrayLike] = None,
    ) -> None:
        super().__init__(parent=window)
        self._config = config
        self._window = window
        self._history = history
        self._recording = recording
        self._tray = tray
        self._poller: Optional[BackendStatusPoller] = None
        # Drives the elapsed-seconds counter shown in the loading pill
        # while the backend is in model_loading. Started/stopped from
        # ``_on_recording_state_changed``.
        self._loading_elapsed_s = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(1000)
        self._loading_timer.timeout.connect(self._on_loading_tick)
        self._wire_models()
        self._wire_shortcuts()
        self._wire_history()
        if backend_status_fetcher is not None:
            self._wire_backend_status(backend_status_fetcher)
        if recording is not None:
            self._wire_recording(recording)
        if tray is not None:
            self._wire_tray(tray)

    def _wire_models(self) -> None:
        view = self._window.models_view
        raw = self._config.get_setting("whisper", "model")
        active_info = None
        if isinstance(raw, str) and raw:
            alias = alias_for(raw)
            try:
                info = get_model(alias)
            except KeyError:
                log.warning(
                    "Model %r from config is not in the registry — leaving inactive",
                    raw,
                )
            else:
                # Only restore the persisted "active" state if the
                # weights are already on disk. If they aren't, we'd be
                # showing the green Active pill while the backend has
                # nothing loaded — and the Select/Download button stays
                # hidden, leaving the user with no way to trigger the
                # download. Force a deliberate click in that case.
                if is_model_cached(info.canonical):
                    view.set_active(alias)
                    active_info = info
                else:
                    log.info(
                        "Persisted model %s is not cached — leaving "
                        "inactive until the user clicks Download.",
                        info.canonical,
                    )

        self._sync_topbar_model(active_info)

        view.model_selected.connect(self._on_model_selected)

    def _on_model_selected(self, alias: str) -> None:
        if self._window.models_view.active_alias() == alias:
            return

        info = None
        try:
            info = get_model(alias)
        except KeyError:
            pass

        self._config.update_user_setting("whisper", "model", alias)
        if info is not None:
            self._config.update_user_setting(
                "whisper", "compute_type", info.compute_type
            )
        self._window.models_view.set_active(alias)
        # Paint the Loading pill on the card immediately — otherwise
        # there is a ~200 ms window where the green Active pill flashes
        # before the recording-state poll catches up and switches it to
        # Loading. The state poll's later set_loading(True) is
        # idempotent.
        self._window.models_view.set_loading(True)
        self._sync_topbar_model(info)
        if self._recording is not None:
            canonical = info.canonical if info else canonical_for(alias)
            compute_type = info.compute_type if info else None
            try:
                self._recording.request_model_change(canonical, compute_type)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("request_model_change raised: %s", exc)

    def _sync_topbar_model(self, info) -> None:
        self._window.topbar.set_active_model(info.display_name if info else None)

    def _wire_shortcuts(self) -> None:
        view = self._window.shortcuts_view
        start = self._config.get_setting("hotkey", "start_recording_hotkey") or ""
        stop = self._config.get_setting("hotkey", "stop_recording_hotkey") or ""
        auto_paste = bool(self._config.get_setting("clipboard", "auto_paste"))
        view.set_values(start_hotkey=start, stop_hotkey=stop, auto_paste=auto_paste)

        # Populate the microphone dropdown if a recording stack is wired in.
        if self._recording is not None and hasattr(self._recording, "list_input_devices"):
            try:
                devices = self._recording.list_input_devices()
                current = self._recording.current_input_device()
            except Exception:
                devices, current = [], None
            view.set_devices(devices, current)

        view.save_requested.connect(self._on_shortcuts_save)
        view.reset_requested.connect(self._on_shortcuts_reset)

    def _on_shortcuts_save(self, payload: dict) -> None:
        self._config.update_user_setting(
            "hotkey", "start_recording_hotkey", payload["start_hotkey"]
        )
        self._config.update_user_setting(
            "hotkey", "stop_recording_hotkey", payload["stop_hotkey"]
        )
        self._config.update_user_setting(
            "clipboard", "auto_paste", payload["auto_paste"]
        )
        if "device" in payload:
            self._config.update_user_setting("audio", "device", payload["device"])
            if self._recording is not None and hasattr(self._recording, "set_input_device"):
                try:
                    self._recording.set_input_device(payload["device"])
                except Exception as exc:
                    log.warning("Failed to switch input device: %s", exc)

    def _on_shortcuts_reset(self) -> None:
        from app.config_manager import DEFAULT_CONFIG

        defaults_hotkey = DEFAULT_CONFIG.get("hotkey", {})
        defaults_clipboard = DEFAULT_CONFIG.get("clipboard", {})
        start = defaults_hotkey.get("start_recording_hotkey", "")
        stop = defaults_hotkey.get("stop_recording_hotkey", "")
        auto_paste = bool(defaults_clipboard.get("auto_paste", True))

        self._config.update_user_setting(
            "hotkey", "start_recording_hotkey", start
        )
        self._config.update_user_setting(
            "hotkey", "stop_recording_hotkey", stop
        )
        self._config.update_user_setting("clipboard", "auto_paste", auto_paste)
        # set_values uses the suspend-emit guard so this won't fire save_requested.
        self._window.shortcuts_view.set_values(
            start_hotkey=start, stop_hotkey=stop, auto_paste=auto_paste
        )

    def _wire_history(self) -> None:
        view = self._window.history_view
        if self._history is not None:
            view.set_entries(self._history.get_entries())
            view.clear_requested.connect(self._on_history_clear)
        view.copy_requested.connect(self._on_history_copy)

    def _on_history_clear(self) -> None:
        if self._history is None:
            return
        self._history.clear_history()
        self._window.history_view.set_entries(self._history.get_entries())

    def _on_history_copy(self, text: str) -> None:
        QApplication.clipboard().setText(text)

    def _wire_backend_status(self, fetcher: Callable[[], str]) -> None:
        self._poller = BackendStatusPoller(fetcher=fetcher, parent=self)
        self._poller.status_changed.connect(self._window.topbar.set_backend_status)
        self._poller.start()

    def _wire_recording(self, recording: _RecordingLike) -> None:
        recording.state_changed.connect(self._window.topbar.set_recording_state)
        recording.state_changed.connect(self._on_recording_state_changed)
        recording.history_updated.connect(self._on_history_updated_signal)
        # Optional: download progress (only the real RecordingController
        # exposes it; fakes in tests can omit it).
        progress_signal = getattr(recording, "download_progress", None)
        if progress_signal is not None:
            try:
                progress_signal.connect(self._on_download_progress)
            except Exception:  # pragma: no cover — defensive
                pass

    def _on_download_progress(self, current: int, total: int, _desc: str) -> None:
        # Mirror progress in two places: the small pill in the topbar
        # and the larger pill on the active model card. The card is
        # where the user just clicked, so it's the most discoverable
        # spot to surface byte-by-byte feedback.
        try:
            self._window.topbar.set_loading_progress(int(current), int(total))
        except Exception:  # pragma: no cover — defensive
            pass
        try:
            self._window.models_view.set_loading_progress(int(current), int(total))
        except Exception:  # pragma: no cover — defensive
            pass

    def _on_recording_state_changed(self, state: str) -> None:
        # Block destructive interactions while not idle.
        self._window.models_view.set_locked(state != "idle")
        # Reflect the model-loading state on the active card's pill so it
        # doesn't say "Active" while the topbar shows "Loading model…".
        self._window.models_view.set_loading(state == "model_loading")
        # While loading, tick an elapsed-seconds counter so the pill has
        # something to show even when no tqdm download progress fires
        # (cached models deserialise silently for ~15 s).
        if state == "model_loading":
            self._loading_elapsed_s = 0
            if not self._loading_timer.isActive():
                self._loading_timer.start()
        else:
            self._loading_timer.stop()
            self._loading_elapsed_s = 0
        # When the backend transitions back to idle, a download (if any)
        # has finished — refresh per-card cache state so the button on
        # the previously-undownloaded model switches to "Select".
        if state == "idle":
            try:
                self._window.models_view.refresh_cache_state()
            except Exception:  # pragma: no cover — defensive
                pass
        if self._tray is not None:
            self._tray.set_state(state)

    def _on_loading_tick(self) -> None:
        self._loading_elapsed_s += 1
        try:
            self._window.topbar.set_loading_elapsed(self._loading_elapsed_s)
        except Exception:  # pragma: no cover — defensive
            pass
        try:
            self._window.models_view.set_loading_elapsed(
                self._loading_elapsed_s
            )
        except Exception:  # pragma: no cover — defensive
            pass

    def _wire_tray(self, tray: _TrayLike) -> None:
        self._window.set_close_to_tray(True)
        tray.show_requested.connect(self._on_tray_show)
        tray.quit_requested.connect(self._on_tray_quit)

    def _on_tray_show(self) -> None:
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()

    def _on_tray_quit(self) -> None:
        # Close the main window (will accept thanks to request_quit's flag)
        # and then explicitly tell the QApplication to leave its event loop.
        # ``setQuitOnLastWindowClosed(False)`` is set when a tray is present,
        # so the app would otherwise stay alive forever after the window
        # disappears.
        self._window.request_quit()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_history_updated_signal(self) -> None:
        if self._history is None:
            return
        self._window.history_view.set_entries(self._history.get_entries())
