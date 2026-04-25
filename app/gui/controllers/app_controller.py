"""Controller wiring views to the domain layer (config, state, backends)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

import threading

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

# QApplication is imported above for the clipboard helper; reuse it for
# explicit ``quit()`` calls from tray actions.

from app.gui.controllers.backend_status_poller import BackendStatusPoller
from app.gui.main_window import MainWindow
from app.model_mapping import alias_for, canonical_for, get_model
from app.utils import is_cached_for_info


log = logging.getLogger(__name__)


class _ConfigLike(Protocol):
    def get_setting(self, section: str, key: str) -> Any: ...
    def update_user_setting(self, section: str, key: str, value: Any) -> None: ...


class _HistoryLike(Protocol):
    def get_entries(self) -> list: ...
    def clear_history(self) -> None: ...
    def export_to_text(self, filepath: str) -> bool: ...


class _RecordingLike(Protocol):
    state_changed: Any
    history_updated: Any

    def request_model_change(self, canonical: str) -> bool: ...


class _TrayLike(Protocol):
    show_requested: Any
    quit_requested: Any

    def set_state(self, state: str) -> None: ...


class AppController(QObject):
    # Worker threads emit these to push results back onto the main
    # Qt thread (auto-queued thanks to the cross-thread connection).
    _mic_test_completed = Signal(float, float)
    _mic_test_failed = Signal(str)

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
        # Mic test worker bookkeeping.
        self._mic_test_in_progress = False
        self._mic_test_completed.connect(self._on_mic_test_completed)
        self._mic_test_failed.connect(self._on_mic_test_failed)
        # Polls the live audio level for the topbar VU meter while the
        # recording pipeline is in the ``recording`` state. ~30 Hz feels
        # alive without burning CPU.
        self._vu_timer = QTimer(self)
        self._vu_timer.setInterval(33)
        self._vu_timer.timeout.connect(self._on_vu_tick)
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
                if is_cached_for_info(info):
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
        view.test_mic_requested.connect(self._on_test_mic_requested)

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

    def _on_test_mic_requested(self) -> None:
        if self._mic_test_in_progress:
            return
        if self._recording is None:
            self._window.shortcuts_view.show_mic_test_error(
                "no recording stack"
            )
            return
        # Don't poke the device while transcription is running — both
        # would try to open the same input simultaneously.
        try:
            current = self._recording.current_state()
        except Exception:
            current = None
        if current and current != "idle":
            self._window.shortcuts_view.show_mic_test_error(
                "wait until current operation finishes"
            )
            return
        recorder = self._resolve_audio_recorder()
        if recorder is None:
            self._window.shortcuts_view.show_mic_test_error(
                "audio recorder unavailable"
            )
            return

        self._mic_test_in_progress = True
        self._window.shortcuts_view.show_mic_test_running()

        def worker():
            try:
                result = recorder.test_input_level(3.0)
            except Exception as exc:  # pragma: no cover — surfaces in UI
                self._mic_test_failed.emit(str(exc))
                return
            self._mic_test_completed.emit(
                float(result.get("peak", 0.0)),
                float(result.get("rms", 0.0)),
            )

        threading.Thread(
            target=worker, daemon=True, name="mic-test"
        ).start()

    def _on_mic_test_completed(self, peak: float, rms: float) -> None:
        self._mic_test_in_progress = False
        self._window.shortcuts_view.show_mic_test_result(peak, rms)

    def _on_mic_test_failed(self, reason: str) -> None:
        self._mic_test_in_progress = False
        self._window.shortcuts_view.show_mic_test_error(reason)

    def _resolve_audio_recorder(self):
        """The state_manager owns the AudioRecorder; reach for it
        through whichever recording-controller protocol the controller
        was given."""
        sm = getattr(self._recording, "state_manager", None)
        if sm is None:
            return None
        return getattr(sm, "audio_recorder", None)

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
            view.export_requested.connect(self._on_history_export)
        view.copy_requested.connect(self._on_history_copy)

    def _on_history_clear(self) -> None:
        if self._history is None:
            return
        # Wipes the on-disk history file too — confirm before doing
        # anything irreversible.
        entries = self._history.get_entries()
        if not entries:
            return
        answer = QMessageBox.question(
            self._window,
            "Clear history?",
            f"Delete all {len(entries)} transcriptions? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self._history.clear_history()
        self._window.history_view.set_entries(self._history.get_entries())

    def _on_history_export(self) -> None:
        if self._history is None:
            return
        entries = self._history.get_entries()
        if not entries:
            QMessageBox.information(
                self._window,
                "Nothing to export",
                "Your history is empty — record a transcription first.",
            )
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self._window,
            "Export history",
            "transcription_history.txt",
            "Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        try:
            ok = bool(self._history.export_to_text(path))
        except Exception as exc:
            log.warning("History export raised: %s", exc)
            ok = False
        if ok:
            QMessageBox.information(
                self._window,
                "History exported",
                f"Saved {len(entries)} transcriptions to:\n{path}",
            )
        else:
            QMessageBox.warning(
                self._window,
                "Export failed",
                "Could not write the history file. Check the destination "
                "path and permissions.",
            )

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
        # Live VU meter follows the recording state — start polling as
        # soon as we enter "recording", stop the moment we leave.
        if state == "recording":
            if not self._vu_timer.isActive():
                self._vu_timer.start()
        else:
            self._vu_timer.stop()
            try:
                self._window.topbar.set_input_level(0.0)
            except Exception:  # pragma: no cover — defensive
                pass
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

    def _on_vu_tick(self) -> None:
        recorder = self._resolve_audio_recorder()
        if recorder is None:
            return
        getter = getattr(recorder, "current_input_level", None)
        if getter is None:
            return
        try:
            level = float(getter())
        except Exception:  # pragma: no cover — defensive
            return
        try:
            self._window.topbar.set_input_level(level)
        except Exception:  # pragma: no cover — defensive
            pass

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
        entries = self._history.get_entries()
        self._window.history_view.set_entries(entries)
        # Pop a confirmation toast for the most recent entry — gives
        # the user a visible "yes, the hotkey worked" moment that
        # was missing from the silent clipboard-paste flow.
        if entries:
            try:
                latest_text = getattr(entries[0], "text", "") or ""
            except Exception:  # pragma: no cover — defensive
                latest_text = ""
            if latest_text:
                try:
                    self._window.toast.show_message(latest_text)
                except Exception:  # pragma: no cover — defensive
                    pass
