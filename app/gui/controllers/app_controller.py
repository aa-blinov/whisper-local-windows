"""Controller wiring views to the domain layer (config, state, backends)."""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

import threading

from pathlib import Path

from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

# QApplication is imported above for the clipboard helper; reuse it for
# explicit ``quit()`` calls from tray actions.

from app.gui.main_window import MainWindow
from app.inference_settings import InferenceSettings, NemoInferenceSettings
from app.model_mapping import MODELS, alias_for, canonical_for, get_model
from app.utils import (
    cached_models_size,
    delete_cached_for_info,
    get_models_root,
    is_cached_for_info,
    move_cached_dir,
)


log = logging.getLogger(__name__)


def _apply_hf_token_to_env(configured: Optional[str]) -> bool:
    """Mirror the user's HF token into the live environment.

    huggingface_hub reads ``HF_TOKEN`` (and the legacy alias
    ``HUGGING_FACE_HUB_TOKEN``) on every download — setting them
    here makes the change effective without a restart.

    Returns True iff a non-empty token was applied.
    """
    import os as _os

    if configured and str(configured).strip():
        token = str(configured).strip()
        _os.environ["HF_TOKEN"] = token
        _os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return True
    _os.environ.pop("HF_TOKEN", None)
    _os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return False


def _apply_env_for_models_root(configured: str) -> str:
    """Mirror the user's chosen models root into the live process
    environment so the next ``onnx_asr.load_model(...)`` call's
    ``huggingface_hub`` download picks it up without a restart.

    Mirrors the startup logic in ``app.gui.app._apply_storage_path``
    — kept in lockstep so 'change live' and 'apply on next launch'
    end up at the same env state.

    Returns the resolved root for logging.
    """
    import os as _os

    root = get_models_root(configured)
    _os.environ["HF_HOME"] = root
    return root


def _human_size(num_bytes: int) -> str:
    """Compact human size for status / dialog text. KB/MB/GB to one
    decimal — close enough for 'will this fit?' reasoning."""
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


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
        recording: Optional[_RecordingLike] = None,
        tray: Optional[_TrayLike] = None,
    ) -> None:
        super().__init__(parent=window)
        self._config = config
        self._window = window
        self._history = history
        self._recording = recording
        self._tray = tray
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
        # Independent polling timer for the mic-test VU meter on the
        # Settings card. Separate from the recording one so a mic
        # test can run while the recording state is anything other
        # than ``recording`` (idle, processing, model_loading, …)
        # without us trying to feed two meters from one tick with
        # divergent on/off conditions.
        self._mic_test_vu_timer = QTimer(self)
        self._mic_test_vu_timer.setInterval(33)
        self._mic_test_vu_timer.timeout.connect(self._on_mic_test_vu_tick)
        # Snapshot of the pre-click state, captured in
        # ``_on_model_selected`` and consumed by
        # ``_on_cancel_load_requested`` to roll the active card +
        # config + topbar pill back when the user bails out of a load.
        # Cleared once the load actually succeeds (in
        # ``_on_recording_state_changed`` when state goes to idle and
        # the backend reports ``ready``).
        self._pre_select_snapshot: Optional[dict] = None
        # Expected weights size for the model currently loading, in
        # bytes. Set from ``ModelInfo.size_mb`` at the moment of click
        # (and at app startup for the persisted active model). Used as
        # a fallback ``total`` in ``_on_download_progress`` when the
        # backend's tqdm fires with ``total=0`` — NeMo's downloader
        # streams via plain ``requests`` with no Content-Length, so
        # without this the topbar would show raw bytes the entire
        # download instead of a climbing percentage. Reset once the
        # load settles (state goes idle).
        self._loading_expected_bytes: int = 0
        self._wire_models()
        self._wire_shortcuts()
        self._wire_history()
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

        # Pre-populate the fallback total for an auto-loading
        # persisted model — same rationale as ``_on_model_selected``.
        if active_info is not None:
            self._loading_expected_bytes = (
                int(active_info.size_mb) * 1024 * 1024
            )

        self._sync_topbar_model(active_info)

        # Pre-fill every card's inference panel from saved per-alias
        # overrides so the user sees the values they last picked
        # when activating a card.
        for alias in self._all_known_aliases():
            view.set_inference_settings(
                alias, self._load_inference_settings(alias)
            )
        if active_info is not None:
            # Push the *active* card's overrides into the live backend
            # so the first transcription respects the saved values.
            self._push_inference_to_backend(
                self._load_inference_settings(active_info.alias)
            )

        view.model_selected.connect(self._on_model_selected)
        view.model_delete_requested.connect(self._on_model_delete_requested)
        view.inference_settings_changed.connect(
            self._on_inference_settings_changed
        )

    def _on_model_selected(self, alias: str) -> None:
        if self._window.models_view.active_alias() == alias:
            return

        info = None
        try:
            info = get_model(alias)
        except KeyError:
            pass

        # Snapshot the pre-click state before we mutate anything. If
        # the user clicks Cancel before this load finishes, the
        # cancel handler restores the active card / config / topbar
        # pill from this dict — without it the half-clicked card
        # stays marked Active forever.
        prev_alias = self._window.models_view.active_alias()
        prev_info = None
        if prev_alias:
            try:
                prev_info = get_model(prev_alias)
            except KeyError:
                prev_info = None
        self._pre_select_snapshot = {
            "alias": prev_alias,
            "info": prev_info,
            "config_model": self._config.get_setting("whisper", "model"),
            "config_compute": self._config.get_setting(
                "whisper", "compute_type"
            ),
        }

        self._config.update_user_setting("whisper", "model", alias)
        if info is not None:
            self._config.update_user_setting(
                "whisper", "compute_type", info.compute_type
            )
            # Pre-populate the fallback total so the topbar can show
            # percentages even when the backend's tqdm doesn't carry
            # a Content-Length (NeMo's case — see field doc).
            self._loading_expected_bytes = (
                int(info.size_mb) * 1024 * 1024
            )
        else:
            self._loading_expected_bytes = 0
        self._window.models_view.set_active(alias)
        # Paint the Loading pill on the card immediately — otherwise
        # there is a ~200 ms window where the green Active pill flashes
        # before the recording-state poll catches up and switches it to
        # Loading. The state poll's later set_loading(True) is
        # idempotent.
        self._window.models_view.set_loading(True)
        # Push this card's persisted inference overrides into the
        # live backend so the first transcription on the new model
        # honours the saved values.
        self._push_inference_to_backend(self._load_inference_settings(alias))
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

    def _on_model_delete_requested(self, alias: str) -> None:
        """Confirm with the user, then drop the cached weights for
        ``alias`` and refresh the Models view so the buttons reflect
        the new state.

        Several aliases can share the same Hugging Face canonical
        (``turbo`` and ``turbo-int8`` both point at
        ``deepdml/faster-whisper-large-v3-turbo-ct2``). Deleting the
        cache for one wipes weights for the other too — surface that
        in the confirmation text so the user isn't surprised when
        the sibling card flips back to 'Download'."""
        try:
            info = get_model(alias)
        except KeyError:
            log.warning("delete requested for unknown alias %s — ignoring", alias)
            return

        siblings = [
            m.alias for m in MODELS
            if m.canonical == info.canonical and m.alias != alias
        ]
        sibling_note = ""
        if siblings:
            sibling_note = (
                "\n\nThis cache is shared with other variants and they "
                f"will also flip to 'Download' after deletion: "
                f"{', '.join(siblings)}."
            )

        answer = QMessageBox.question(
            self._window,
            "Delete cached model?",
            (
                f"Remove the downloaded weights for "
                f"{info.display_name} ({info.size_mb / 1000:.1f} GB)?"
                "\n\nYou'll need to download them again the next time "
                "this model is selected."
                f"{sibling_note}"
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return

        if delete_cached_for_info(info):
            log.info("Deleted cached weights for %s (%s)", alias, info.canonical)
        else:
            log.warning(
                "Delete returned False for %s — see earlier log for the "
                "filesystem error, or the cache was already empty",
                alias,
            )
        self._window.models_view.refresh_cache_state()

    def _wire_shortcuts(self) -> None:
        view = self._window.shortcuts_view
        start = self._config.get_setting("hotkey", "start_recording_hotkey") or ""
        stop = self._config.get_setting("hotkey", "stop_recording_hotkey") or ""
        cancel = self._config.get_setting("hotkey", "cancel_recording_hotkey") or ""
        auto_paste = bool(self._config.get_setting("clipboard", "auto_paste"))
        view.set_values(
            start_hotkey=start,
            stop_hotkey=stop,
            auto_paste=auto_paste,
            cancel_hotkey=cancel,
        )

        # Populate the microphone dropdown if a recording stack is wired in.
        if self._recording is not None and hasattr(self._recording, "list_input_devices"):
            try:
                devices = self._recording.list_input_devices()
                current = self._recording.current_input_device()
            except Exception:
                devices, current = [], None
            view.set_devices(devices, current)

        view.save_requested.connect(self._on_shortcuts_save)
        view.hotkeys_reset_requested.connect(self._on_hotkeys_reset)
        view.hf_token_reset_requested.connect(self._on_hf_token_reset)
        view.test_mic_requested.connect(self._on_test_mic_requested)

        # Storage card — render the resolved path on first paint so
        # the user sees where weights actually live, even when they
        # haven't picked a custom path yet.
        view.storage_path_change_requested.connect(self._on_storage_path_change)
        view.storage_reset_requested.connect(self._on_storage_reset)
        self._refresh_storage_path()

        # Hugging Face card — paint the persisted token + apply to
        # env (in case ``app.py``'s startup hook missed something).
        view.hf_token_changed.connect(self._on_hf_token_changed)
        persisted_token = self._config.get_setting("huggingface", "token") or ""
        view.set_hf_token(persisted_token)
        _apply_hf_token_to_env(persisted_token)

    def _refresh_storage_path(self) -> None:
        """Push the resolved storage path into the Settings card. The
        view shows ``(default)`` after the path when nothing's been
        overridden — same source-of-truth (``get_models_root``) the
        rest of the app uses at startup."""
        configured = self._config.get_setting("storage", "models_dir")
        is_default = not (configured and str(configured).strip())
        resolved = get_models_root(configured)
        self._window.shortcuts_view.set_storage_path(
            resolved, is_default=is_default,
        )

    def _on_storage_path_change(self) -> None:
        """User clicked Change….

        Flow:
          1. Open the folder picker; bail on Cancel.
          2. If the pick equals the current root → no-op.
          3. Sum cached weights at the old root. If non-zero, ask
             Yes/No/Cancel about migrating them. Cancel here aborts
             the whole change so the user can re-pick without leaving
             config in a half-applied state.
          4. On Yes — block UI with a wait cursor and call
             ``move_cached_dir`` for ``hub/`` and ``gigaam/`` in
             sequence; the helper handles intra- vs cross-volume
             internally and refuses to overwrite existing dirs.
          5. Write the new path to config and pop a single info
             dialog summarising what moved + the restart caveat
             (env vars are baked at startup).
        """
        configured = self._config.get_setting("storage", "models_dir") or ""
        old_root = get_models_root(configured)
        start_dir = configured or str(Path(old_root).parent)
        chosen = QFileDialog.getExistingDirectory(
            self._window,
            "Choose models folder",
            start_dir,
        )
        if not chosen:
            return  # Cancelled at the folder picker.

        try:
            same = Path(chosen).resolve() == Path(old_root).resolve()
        except OSError:
            same = chosen == old_root
        if same:
            return  # Picked the same folder — nothing to do.

        old_size = cached_models_size(old_root)
        move_outcomes: list[tuple[str, dict]] = []

        if old_size > 0:
            answer = QMessageBox.question(
                self._window,
                "Move existing weights?",
                (
                    f"You have {_human_size(old_size)} of cached models at:\n"
                    f"{old_root}\n\n"
                    f"Move them to the new location?\n{chosen}\n\n"
                    "Yes — relocate now (intra-drive is instant; "
                    "across drives can take several minutes for large "
                    "caches).\n"
                    "No  — leave them in place; new downloads go to "
                    "the new folder.\n"
                    "Cancel — go back without changing anything."
                ),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                return  # Bail without writing config.
            if answer == QMessageBox.Yes:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    # Single-engine app — only the HF hub subtree
                    # exists.  The legacy ``gigaam`` subtree (used by
                    # the old gigaam-Python backend) was retired with
                    # the ONNX-only refactor.
                    result = move_cached_dir(
                        str(Path(old_root) / "hub"),
                        str(Path(chosen) / "hub"),
                    )
                    move_outcomes.append(("hub", result))
                finally:
                    QApplication.restoreOverrideCursor()

        self._config.update_user_setting("storage", "models_dir", chosen)
        # Mirror into the live process env — both backends read these
        # at every load, so the change takes effect on the very next
        # ``Download`` click without a restart.
        _apply_env_for_models_root(chosen)
        self._refresh_storage_path()
        # Refresh every model card's cache state so the Download ↔ Select
        # button reflects the new directory immediately — without this the
        # cards keep showing "Download" even when the chosen folder already
        # contains the model weights.
        self._window.models_view.refresh_cache_state()

        # Build a user-friendly summary so they know what landed
        # where and what didn't.
        summary_lines = [f"Models folder set to:\n{chosen}\n"]
        if move_outcomes:
            for name, result in move_outcomes:
                if result.get("moved"):
                    summary_lines.append(
                        f"  • {name}: moved {_human_size(int(result['bytes']))}"
                    )
                else:
                    reason = result.get("reason", "no source")
                    if "missing" in reason or "same" in reason:
                        # Don't bother surfacing 'gigaam: source missing'
                        # — that's the normal case for Whisper-only
                        # users and would clutter the dialog.
                        continue
                    summary_lines.append(f"  • {name}: skipped ({reason})")
            summary_lines.append("")
        elif old_size > 0:
            summary_lines.append(
                f"Existing {_human_size(old_size)} of weights left at:\n"
                f"{old_root}\n"
            )
        summary_lines.append(
            "New downloads will land at the new location immediately."
        )

        QMessageBox.information(
            self._window,
            "Models folder updated",
            "\n".join(summary_lines),
        )

    def _on_storage_reset(self) -> None:
        """Reset Storage to default and mirror that into the live
        env so subsequent loads/downloads use the default path."""
        self._config.update_user_setting("storage", "models_dir", "")
        resolved = _apply_env_for_models_root("")
        self._refresh_storage_path()
        self._window.models_view.refresh_cache_state()
        QMessageBox.information(
            self._window,
            "Models folder reset",
            (
                f"Models folder set to default:\n{resolved}\n\n"
                "New downloads will land there immediately."
            ),
        )

    def _on_hf_token_changed(self, token: str) -> None:
        """Persist the new token, mirror into env, and re-evaluate
        every GigaAM card's warning so the user gets immediate
        feedback that the warning has cleared."""
        token = (token or "").strip()
        self._config.update_user_setting("huggingface", "token", token)
        _apply_hf_token_to_env(token)
        self._window.models_view.refresh_hf_token_state()

    def _on_shortcuts_save(self, payload: dict) -> None:
        self._config.update_user_setting(
            "hotkey", "start_recording_hotkey", payload["start_hotkey"]
        )
        self._config.update_user_setting(
            "hotkey", "stop_recording_hotkey", payload["stop_hotkey"]
        )
        # ``cancel_hotkey`` may be missing if a legacy view emits the
        # old payload shape — default to empty rather than raising.
        self._config.update_user_setting(
            "hotkey",
            "cancel_recording_hotkey",
            payload.get("cancel_hotkey", ""),
        )
        self._config.update_user_setting(
            "clipboard", "auto_paste", payload["auto_paste"]
        )
        # Push the new auto-paste flag into the running ClipboardManager
        # too — without this the checkbox visually toggles but the
        # actual delivery path keeps the value it had at startup.
        clipboard = self._resolve_clipboard_manager()
        if clipboard is not None and hasattr(clipboard, "update_auto_paste"):
            try:
                clipboard.update_auto_paste(bool(payload["auto_paste"]))
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("Failed to push auto_paste live: %s", exc)
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
        # Start the live-level poll right when the worker thread
        # spins up — first audio block lands ~50 ms in, which is
        # before the worker's first level update could otherwise be
        # picked up.
        if not self._mic_test_vu_timer.isActive():
            self._mic_test_vu_timer.start()

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
        self._mic_test_vu_timer.stop()
        self._window.shortcuts_view.show_mic_test_result(peak, rms)

    def _on_mic_test_failed(self, reason: str) -> None:
        self._mic_test_in_progress = False
        self._mic_test_vu_timer.stop()
        self._window.shortcuts_view.show_mic_test_error(reason)

    def _on_mic_test_vu_tick(self) -> None:
        """Poll the live audio level while a mic test is running and
        push it into the Settings-card VU meter. Mirrors the topbar
        VU pump but routed to a different sink."""
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
            self._window.shortcuts_view.set_mic_test_level(level)
        except Exception:  # pragma: no cover — defensive
            pass

    # ---- Inference-settings plumbing ---------------------------------------

    def _all_known_aliases(self) -> list:
        from app.model_mapping import aliases
        return aliases()

    def _settings_class_for(self, alias: str):
        """Pick the inference-settings dataclass that matches the
        model's backend kind. Whisper → ``InferenceSettings`` (5
        knobs); Parakeet/NeMo-style → ``NemoInferenceSettings`` (just
        ``timestamps``); unknown alias defaults to Whisper since that's
        the most common case for raw HF ids."""
        try:
            info = get_model(alias)
        except KeyError:
            return InferenceSettings
        if getattr(info, "onnx_family", "") == "parakeet":
            return NemoInferenceSettings
        return InferenceSettings

    def _load_inference_settings(self, alias: str):
        try:
            raw = self._config.get_setting("model_overrides", alias)
        except Exception:
            raw = None
        return self._settings_class_for(alias).from_mapping(raw)

    def _save_inference_settings(self, alias: str, settings) -> None:
        self._config.update_user_setting(
            "model_overrides", alias, settings.to_mapping()
        )

    def _push_inference_to_backend(self, settings) -> None:
        if self._recording is None:
            return
        sm = getattr(self._recording, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm is not None else None
        target = getattr(backend, "update_inference_settings", None)
        if target is None:
            return
        try:
            target(settings)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Live inference-settings push raised: %s", exc)

    def _on_inference_settings_changed(self, alias: str, settings) -> None:
        # Persist for next launch.
        self._save_inference_settings(alias, settings)
        # Live-apply only if the change was made on the *active*
        # card — otherwise the user is configuring something they
        # haven't picked yet.
        if self._window.models_view.active_alias() == alias:
            self._push_inference_to_backend(settings)

    def _resolve_audio_recorder(self):
        """The state_manager owns the AudioRecorder; reach for it
        through whichever recording-controller protocol the controller
        was given."""
        sm = getattr(self._recording, "state_manager", None)
        if sm is None:
            return None
        return getattr(sm, "audio_recorder", None)

    def _resolve_clipboard_manager(self):
        """Same indirection for the ClipboardManager so settings can
        push live updates (auto_paste toggle) without the user having
        to restart the app for the change to take effect."""
        sm = getattr(self._recording, "state_manager", None)
        if sm is None:
            return None
        return getattr(sm, "clipboard_manager", None)

    def _on_hotkeys_reset(self) -> None:
        """Reset only the three hotkey fields to their built-in
        defaults — leaves auto-paste, storage, HF token alone.
        Per-card reset replaces the old global 'Reset to defaults'
        footer button which conflated unrelated settings."""
        from app.config_manager import DEFAULT_CONFIG

        defaults_hotkey = DEFAULT_CONFIG.get("hotkey", {})
        start = defaults_hotkey.get("start_recording_hotkey", "")
        stop = defaults_hotkey.get("stop_recording_hotkey", "")
        cancel = defaults_hotkey.get("cancel_recording_hotkey", "")

        self._config.update_user_setting(
            "hotkey", "start_recording_hotkey", start
        )
        self._config.update_user_setting(
            "hotkey", "stop_recording_hotkey", stop
        )
        self._config.update_user_setting(
            "hotkey", "cancel_recording_hotkey", cancel
        )
        # Refresh the three fields without disturbing the rest of the
        # form (current auto-paste / device / HF token / storage path
        # all stay where they are). ``set_values`` is suspend-guarded
        # so this won't bounce ``save_requested`` back at us.
        view = self._window.shortcuts_view
        view.set_values(
            start_hotkey=start,
            stop_hotkey=stop,
            auto_paste=view.auto_paste(),
            cancel_hotkey=cancel,
        )

    def _on_hf_token_reset(self) -> None:
        """Per-card 'Clear token' on the Hugging Face card. Routes
        through the same path the textbox-edit handler uses so the
        env var, model-card warnings, and view all update."""
        self._on_hf_token_changed("")
        self._window.shortcuts_view.set_hf_token("")

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


    def _wire_recording(self, recording: _RecordingLike) -> None:
        recording.state_changed.connect(self._window.topbar.set_recording_state)
        # Sidebar bottom-left slot owns the recording pill + VU meter
        # since they moved out of the topbar (where they overlapped
        # the resource graphs). Same state vocab, fanned out.
        recording.state_changed.connect(
            self._window.sidebar.recording_status.set_recording_state
        )
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
        # Topbar's Cancel button → recording controller's cancel.
        # Wrapped in a thin lambda so we can keep recording strictly
        # typed against the protocol (which only declares request_*).
        self._window.topbar.cancel_load_requested.connect(
            self._on_cancel_load_requested
        )

    def _on_cancel_load_requested(self) -> None:
        if self._recording is None:
            return
        target = getattr(self._recording, "cancel_model_change", None)
        if target is None:
            return
        try:
            cancelled = bool(target())
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("cancel_model_change forward raised: %s", exc)
            return

        # If nothing was actually loading (cancel raced a finished
        # load), don't roll back — the active card / config reflect a
        # successfully loaded model that we shouldn't undo.
        if not cancelled:
            return

        snapshot = self._pre_select_snapshot
        self._pre_select_snapshot = None
        if snapshot is None:
            return

        prev_alias = snapshot.get("alias")
        prev_info = snapshot.get("info")
        if prev_alias:
            self._window.models_view.set_active(prev_alias)
        else:
            self._window.models_view.set_active(None)
        self._sync_topbar_model(prev_info)
        # Reset the fallback total to the previous active model's
        # size (or 0 if there was no prior active card) — a future
        # progress event from a stale tqdm bar shouldn't render
        # a percentage scaled to the cancelled model.
        if prev_info is not None:
            self._loading_expected_bytes = (
                int(prev_info.size_mb) * 1024 * 1024
            )
        else:
            self._loading_expected_bytes = 0

        # Roll back config too, otherwise next launch picks the
        # cancelled model up as the persisted active one.
        prev_model = snapshot.get("config_model")
        if prev_model is not None:
            self._config.update_user_setting("whisper", "model", prev_model)
        else:
            self._config.update_user_setting("whisper", "model", "")
        prev_compute = snapshot.get("config_compute")
        if prev_compute is not None:
            self._config.update_user_setting(
                "whisper", "compute_type", prev_compute
            )

    def _on_download_progress(self, current: int, total: int, _desc: str) -> None:
        # Mirror progress in two places: the small pill in the topbar
        # and the larger pill on the active model card. The card is
        # where the user just clicked, so it's the most discoverable
        # spot to surface byte-by-byte feedback.
        # When the backend's tqdm fires without a usable total (NeMo's
        # streaming download has no Content-Length), substitute the
        # expected size we cached at click time. Real totals (Whisper /
        # GigaAM via huggingface_hub.snapshot_download) take priority.
        effective_total = int(total)
        if effective_total <= 0 and self._loading_expected_bytes > 0:
            effective_total = self._loading_expected_bytes
        try:
            self._window.topbar.set_loading_progress(
                int(current), effective_total
            )
        except Exception:  # pragma: no cover — defensive
            pass
        try:
            self._window.models_view.set_loading_progress(
                int(current), effective_total
            )
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
                self._window.sidebar.recording_status.set_input_level(0.0)
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
            # Clear the fallback total so a future load whose tqdm
            # legitimately reports total=0 (e.g. a small misc file
            # in some unrelated download) doesn't inherit a stale
            # 1.2 GB expectation from the previous Parakeet load.
            self._loading_expected_bytes = 0
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
            self._window.sidebar.recording_status.set_input_level(level)
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
        if not entries:
            return
        # Prepend the newest entry (index 0) without resetting the whole
        # table model — beginInsertRows preserves scroll position and
        # selection when the user is reading history while transcribing.
        max_entries = getattr(self._history, "max_entries", 0)
        self._window.history_view.prepend_entry(entries[0], max_entries)
        # Pop a confirmation toast for the most recent entry — gives
        # the user a visible "yes, the hotkey worked" moment that
        # was missing from the silent clipboard-paste flow.
        try:
            latest_text = getattr(entries[0], "text", "") or ""
        except Exception:  # pragma: no cover — defensive
            latest_text = ""
        if latest_text:
            try:
                self._window.toast.show_message(latest_text)
            except Exception:  # pragma: no cover — defensive
                pass
