import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

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
        "start_recording_hotkey": "ctrl+f2",
        "stop_recording_hotkey": "ctrl+f3",
        # 'Discard the current buffer without transcribing'. Empty
        # string means no global key is bound — the runtime feature
        # works through ``StateManager.cancel_active_recording`` but
        # nothing fires it. ctrl+f6 sits in the same row as f2/f3
        # without colliding with Alt+F4 (close window) or F5
        # (refresh) muscle memory the way ctrl+f4 / ctrl+f5 would.
        "cancel_recording_hotkey": "ctrl+f6",
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
        self._load_or_create()
        self.logger.info(f"Configuration loaded: {self.config_path}")

    def _resolve_base_dir(self) -> Path:
        """Where ``config.yaml`` is read from and written to.

        Frozen build → ``%APPDATA%/LazyToText/`` so the file is
        per-user, writable without admin even when the binary is
        installed in ``Program Files``. Falls back to
        ``~/AppData/Roaming/LazyToText`` if ``APPDATA`` is unset
        (sandboxed shells / unusual envs).

        Dev build → walks up from CWD to the nearest
        ``pyproject.toml`` so a developer running ``uv run …`` from
        anywhere in the repo still reads the project's
        ``config.yaml``.
        """
        if getattr(sys, 'frozen', False):  # PyInstaller frozen
            return self._user_config_dir()
        cwd = Path.cwd()
        for p in [cwd, *cwd.parents]:
            if (p / 'pyproject.toml').exists():
                return p
        return cwd

    @staticmethod
    def _user_config_dir() -> Path:
        """Per-user config directory on Windows.

        Honours ``%APPDATA%`` (the canonical Roaming path); falls
        back to ``~/AppData/Roaming/LazyToText`` when the env var
        isn't exposed (rare).
        """
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "LazyToText"
        return Path.home() / "AppData" / "Roaming" / "LazyToText"

    @staticmethod
    def _bundled_defaults_path() -> Path:
        """Where the build's 'factory defaults' ``config.yaml`` lives.

        PyInstaller layout depends on the version: classic onedir
        builds put datas alongside the exe, PyInstaller 6+ puts
        them under ``_internal/``. ``sys._MEIPASS`` is the most
        reliable hint when present. Returns the first candidate
        that exists; falls back to the exe-dir candidate for the
        'no bundled config' path so callers' ``.is_file()`` check
        cleanly returns False.
        """
        exe_dir = Path(sys.executable).resolve().parent
        meipass = getattr(sys, "_MEIPASS", None)
        candidates = [exe_dir / "config.yaml"]
        if meipass:
            candidates.append(Path(meipass) / "config.yaml")
        candidates.append(exe_dir / "_internal" / "config.yaml")
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    def _load_or_create(self):
        path = self.config_path
        if not path.exists():
            seeded_from_bundle = False
            if getattr(sys, 'frozen', False):
                bundled = self._bundled_defaults_path()
                if bundled.is_file():
                    # Seed-copy 'factory defaults' shipped with the
                    # build into the user dir on first launch — lets
                    # a customised installer ship pre-tweaked
                    # settings without requiring write access to
                    # the install dir at runtime.
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(bundled, path)
                        seeded_from_bundle = True
                        self.logger.info(
                            "Seeded user config from bundled defaults: %s -> %s",
                            bundled, path,
                        )
                    except OSError as exc:
                        self.logger.warning(
                            "Failed to seed bundled defaults from %s: %s "
                            "— falling back to in-code DEFAULT_CONFIG",
                            bundled, exc,
                        )
            if seeded_from_bundle:
                # Re-enter the load path to pick up bundled values
                # through the regular YAML reader + defaults merge.
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    self.config = self._fill_defaults(data, DEFAULT_CONFIG)
                    self._migrate_legacy_whisper_section()
                    return
                except Exception as exc:
                    self.logger.warning(
                        "Bundled config %s unreadable (%s); using DEFAULT_CONFIG",
                        path, exc,
                    )
            self.logger.warning("config.yaml not found, creating with defaults")
            self.config = DEFAULT_CONFIG.copy()
            self._write_config_file()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.config = self._fill_defaults(data, DEFAULT_CONFIG)
            self._migrate_legacy_whisper_section()
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

    def _write_config_file(self):
        # Write to a .tmp sibling first, then rename atomically so a crash
        # mid-write never leaves a half-written (corrupted) config file.
        tmp = self.config_path.with_name(self.config_path.name + ".tmp")
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.config, f,
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

    # --- Migration from legacy schema (model_size / whisper_model / whisper_url) ---
    def _migrate_legacy_whisper_section(self, write_if_changed: bool = False):
        """Strip / map fields that existed in the pre-in-process
        Docker / Wyoming era of this app:

          - ``model_size`` / ``whisper_model``  → ``model`` (current
            schema), via the registry's canonical mapping.
          - ``whisper_url`` / ``backend_mode`` / ``local_url`` /
            ``external_url`` → dropped. We always run the inference
            backend in-process now, so URLs and the local/external
            switch have no consumers.
        """
        wh = self.config.get("whisper", {})
        if not isinstance(wh, dict):  # sanity
            return
        legacy_model_size = wh.pop("model_size", None)
        legacy_model = wh.pop("whisper_model", None)
        # Older configs carried these — drop them silently. They're
        # not read anywhere anymore.
        wh.pop("whisper_url", None)
        wh.pop("backend_mode", None)
        wh.pop("local_url", None)
        wh.pop("external_url", None)
        changed = False
        # Determine canonical model
        if legacy_model_size or legacy_model:
            from app.model_mapping import canonical_for
            # priority: explicit whisper_model if non-empty else map model_size
            candidate = legacy_model if legacy_model else legacy_model_size
            if isinstance(candidate, str) and candidate:
                wh["model"] = canonical_for(candidate)
                changed = True
        if "model" not in wh:
            wh["model"] = DEFAULT_CONFIG["whisper"]["model"]
            changed = True
        # Ensure required keys exist
        for k in ("beam_size", "language"):
            if k not in wh:
                wh[k] = DEFAULT_CONFIG["whisper"][k]
                changed = True
        if changed:
            self.config["whisper"] = wh
            self.logger.info("Migrated legacy whisper config -> new schema")
            if write_if_changed:
                self._write_config_file()

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
        self._write_config_file()
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
