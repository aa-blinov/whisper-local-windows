"""Controller wiring views to the domain layer (config, state, backends)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from app.gui.controllers.backend_status_poller import BackendStatusPoller
from app.gui.main_window import MainWindow
from app.model_mapping import alias_for, canonical_for, get_model


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


class AppController(QObject):
    def __init__(
        self,
        config: _ConfigLike,
        window: MainWindow,
        history: Optional[_HistoryLike] = None,
        backend_status_fetcher: Optional[Callable[[], str]] = None,
        recording: Optional[_RecordingLike] = None,
    ) -> None:
        super().__init__(parent=window)
        self._config = config
        self._window = window
        self._history = history
        self._recording = recording
        self._poller: Optional[BackendStatusPoller] = None
        self._wire_models()
        self._wire_shortcuts()
        self._wire_history()
        if backend_status_fetcher is not None:
            self._wire_backend_status(backend_status_fetcher)
        if recording is not None:
            self._wire_recording(recording)

    def _wire_models(self) -> None:
        view = self._window.models_view
        raw = self._config.get_setting("whisper", "model")
        active_info = None
        if isinstance(raw, str) and raw:
            alias = alias_for(raw)
            try:
                active_info = get_model(alias)
            except KeyError:
                log.warning(
                    "Model %r from config is not in the registry — leaving inactive",
                    raw,
                )
            else:
                view.set_active(alias)

        self._sync_topbar_model(active_info)

        view.model_selected.connect(self._on_model_selected)

    def _on_model_selected(self, alias: str) -> None:
        if self._window.models_view.active_alias() == alias:
            return
        self._config.update_user_setting("whisper", "model", alias)
        self._window.models_view.set_active(alias)
        try:
            self._sync_topbar_model(get_model(alias))
        except KeyError:
            self._sync_topbar_model(None)
        if self._recording is not None:
            try:
                self._recording.request_model_change(canonical_for(alias))
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

    def _on_recording_state_changed(self, state: str) -> None:
        # Block destructive interactions while not idle.
        self._window.models_view.set_locked(state != "idle")

    def _on_history_updated_signal(self) -> None:
        if self._history is None:
            return
        self._window.history_view.set_entries(self._history.get_entries())
