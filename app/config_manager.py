import copy
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import yaml


# macOS reserves ``Ctrl+F1`` through ``Ctrl+F7`` for keyboard
# navigation (Move focus to menu bar / Dock / window / toolbar /
# floating window …). Pynput never sees the events because macOS
# captures them first, so the same defaults that work on Windows
# silently fail on Mac. Pick free-of-conflict combos per platform.
if sys.platform == "darwin":
    _DEFAULT_START_HOTKEY = "ctrl+f8"
    _DEFAULT_STOP_HOTKEY = "ctrl+f9"
    _DEFAULT_CANCEL_HOTKEY = "ctrl+f10"
else:
    _DEFAULT_START_HOTKEY = "ctrl+f2"
    _DEFAULT_STOP_HOTKEY = "ctrl+f3"
    _DEFAULT_CANCEL_HOTKEY = "ctrl+f6"

DEFAULT_CONFIG: Dict[str, Any] = {
    "whisper": {
        # Active model alias (or full HF canonical id). The
        # registry maps user-friendly aliases like
        # ``whisper-large-v3-turbo`` to the actual HF repo path.
        # See ``app/model_mapping.py``.
        "model": "whisper-large-v3-turbo",
        "language": "auto",
        "beam_size": 5,
    },
    "hotkey": {
        # Defaults chosen per-platform — see the constants above.
        # Windows: ``ctrl+f2`` / ``ctrl+f3`` / ``ctrl+f6`` (the
        # original muscle-memory set; nothing else uses them on
        # Win 10/11). macOS: ``ctrl+f8`` / ``ctrl+f9`` / ``ctrl+f10``
        # (lower F-keys are claimed by the system's keyboard
        # navigation). 'cancel' discards the current buffer
        # without transcribing — empty string disables the binding.
        "start_recording_hotkey": _DEFAULT_START_HOTKEY,
        "stop_recording_hotkey": _DEFAULT_STOP_HOTKEY,
        "cancel_recording_hotkey": _DEFAULT_CANCEL_HOTKEY,
    },
    "audio": {
        "channels": 1,
        "dtype": "float32",
        "max_duration": 300,
    },
    "clipboard": {
        "auto_paste": True,
        "preserve_clipboard": False,
        "key_simulation_delay": 0.05,
    },
    "logging": {
        "level": "INFO",
        "file": {
            "enabled": True,
            "filename": "app.log",
            "rotation": {
                "enabled": True,
                "max_bytes": 1048576,
                "backup_count": 5,
                "encoding": "utf-8",
            },
        },
        "console": {"enabled": True, "level": "WARNING"},
    },
    "audio_feedback": {
        "enabled": True,
        "start_sound": "assets/sounds/record_start.wav",
        "stop_sound": "assets/sounds/record_stop.wav",
        "cancel_sound": "assets/sounds/record_cancel.wav",
    },
    "system_tray": {"enabled": True, "tooltip": "Lazy to text"},
    "history": {
        "enabled": True,
        "max_entries": 1000,
        "auto_cleanup_days": 30,
    },
    "storage": {
        # Where downloaded model weights live. Empty string = use the
        # built-in default (``<project>/models`` in dev,
        # ``<exe-dir>/models`` when frozen). Settings → Storage card
        # writes here. Both Whisper (HF hub) and GigaAM (.ckpt) live
        # under this root, in ``hub/`` and ``gigaam/`` subdirs
        # respectively.
        "models_dir": "",
    },
    "huggingface": {
        # Optional HF API token. Required ONLY for GigaAM long-form
        # audio (>25 s) which routes through pyannote VAD; the
        # underlying model ``pyannote/segmentation-3.0`` is gated and
        # needs an authenticated download once. Empty = no token.
        # Settings → "Hugging Face" card writes here.
        "token": "",
    },
}


class ConfigManager:
    def __init__(self, config_filename: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = self._resolve_base_dir()
        self.config_path = self.base_dir / config_filename
        self.config: Dict[str, Any] = {}
        # Serialises concurrent ``_write_config_file`` calls — the
        # async writer thread we spawn from ``update_user_setting``
        # would otherwise race itself when the user clicks Select on
        # a model card (two writes, model + compute_type, fired ~5 ms
        # apart on the Qt main thread).
        self._write_lock = threading.Lock()
        self._load_or_create()
        self.logger.info(f"Configuration loaded: {self.config_path}")

    def _resolve_base_dir(self) -> Path:
        """Where ``config.yaml`` is read from and written to.

        Walks up from CWD to the nearest ``pyproject.toml`` so a
        developer running ``uv run …`` from anywhere in the repo
        still reads the project's ``config.yaml``. Falls back to CWD
        when no marker is found (e.g. running from ``%LOCALAPPDATA%``
        without the source tree alongside).
        """
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents]:
            if (p / 'pyproject.toml').exists():
                return p
        return cwd

    def _load_or_create(self):
        path = self.config_path
        if not path.exists():
            self.logger.warning("config.yaml not found, creating with defaults")
            self.config = DEFAULT_CONFIG.copy()
            self._write_config_file()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.config = self._fill_defaults(data, DEFAULT_CONFIG)
        except Exception as e:
            self.logger.error(f"Failed to load config.yaml: {e}. Recreating defaults.")
            self.config = DEFAULT_CONFIG.copy()
            self._write_config_file()

    def _fill_defaults(
        self, current: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, dv in defaults.items():
            cv = current.get(k)
            if isinstance(dv, dict):
                if isinstance(cv, dict):
                    result[k] = self._fill_defaults(cv, dv)
                else:
                    result[k] = dv
            else:
                result[k] = dv if cv is None else cv
        # Keep extra keys from current (do not prune)
        for extra_k, extra_v in current.items():
            if extra_k not in result:
                result[extra_k] = extra_v
        return result

    def flush_pending_writes(self, timeout: float = 5.0) -> bool:
        """Block until any in-flight async writer thread has finished.

        Daemon threads spawned by ``update_user_setting`` write the
        YAML file at their own pace.  Production code rarely needs
        this — the in-memory ``self.config`` already reflects the
        change before the disk catches up — but tests that read the
        file back immediately after a write call this to drain the
        queue.  Returns True iff the writer completed within
        ``timeout``; False on timeout (the caller can decide whether
        to error out or just retry).
        """
        if self._write_lock.acquire(timeout=timeout):
            self._write_lock.release()
            return True
        return False

    def _write_config_file_locked(self):
        """Run on the writer daemon thread — serialised by ``_write_lock``.

        Takes a snapshot of ``self.config`` *under* the lock to defend
        against the Qt main thread mutating the dict mid-serialisation
        when several ``update_user_setting`` calls fire in quick
        succession (Select-button click writes both ``whisper.model``
        and ``whisper.compute_type`` ~5 ms apart).
        """
        with self._write_lock:
            snapshot = copy.deepcopy(self.config)
            self._write_config_file(snapshot)

    def _write_config_file(self, payload: Dict[str, Any] | None = None):
        """Synchronous YAML write + atomic rename.

        ``payload`` defaults to ``self.config`` for the legacy code path
        (``_load_or_create`` after seeding defaults).  When called from
        the writer thread, the caller passes a deep-copied snapshot so
        the dict can't change under our feet during ``yaml.safe_dump``.
        """
        if payload is None:
            payload = self.config
        # Write to a .tmp sibling first, then rename atomically so a crash
        # mid-write never leaves a half-written (corrupted) config file.
        tmp = self.config_path.with_name(self.config_path.name + ".tmp")
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    payload, f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            os.replace(tmp, self.config_path)
            self.logger.info("Saved configuration to %s", self.config_path)
        except Exception as e:
            self.logger.error("Error writing configuration: %s", e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


    # --- Public accessors (return live copies) ---
    def get_setting(self, section: str, key: str):
        return self.config.get(section, {}).get(key)

    def update_user_setting(self, section: str, key: str, value: Any):
        if section not in self.config or not isinstance(self.config[section], dict):
            self.config[section] = {}
        old = self.config[section].get(key)
        if old == value:
            return
        self.config[section][key] = value
        # YAML serialise + atomic os.replace can take 100-500 ms when
        # the file is on a slow disk or under antivirus inspection.
        # Calling this from the Qt main thread (model-card Select
        # click triggers two writes back-to-back) freezes the UI
        # noticeably — wheel-scroll events back up while the main
        # thread is blocked on libc writes.  Spawn a daemon writer
        # so the caller returns immediately; the lock inside
        # ``_write_config_file_locked`` keeps concurrent writers
        # from racing.
        threading.Thread(
            target=self._write_config_file_locked,
            daemon=True,
            name="config-write",
        ).start()
        self.logger.info(f"Updated setting {section}.{key}: {old} -> {value}")

    # For backward compatibility with existing code expecting dict copies
    def get_whisper_config(self) -> Dict[str, Any]:
        return self.config.get("whisper", {}).copy()

    def get_hotkey_config(self) -> Dict[str, Any]:
        return self.config.get("hotkey", {}).copy()

    def get_audio_config(self) -> Dict[str, Any]:
        return self.config.get("audio", {}).copy()

    def get_clipboard_config(self) -> Dict[str, Any]:
        return self.config.get("clipboard", {}).copy()

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get("logging", {}).copy()

    def get_system_tray_config(self) -> Dict[str, Any]:
        return self.config.get("system_tray", {}).copy()

    def get_audio_feedback_config(self) -> Dict[str, Any]:
        return self.config.get("audio_feedback", {}).copy()

    def get_history_config(self) -> Dict[str, Any]:
        return self.config.get("history", {}).copy()

    # Deprecated method kept for minimal surface compatibility (no-op now)
    def print_stop_instructions_based_on_config(self):
        pass
