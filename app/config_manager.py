import logging
import sys
from pathlib import Path
from typing import Any, Dict
from ruamel.yaml import YAML

DEFAULT_CONFIG: Dict[str, Any] = {
    "whisper": {
        "backend_mode": "local", # "local" or "external"
        "model": "Systran/faster-distil-whisper-base",
        "language": "auto",
        "beam_size": 5,
        "local_url": "http://localhost:10300",
        "external_url": "http://remote-host:10300",
    },
    "hotkey": {
        "start_recording_hotkey": "ctrl+f2",
        "stop_recording_hotkey": "ctrl+f3",
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
}


class ConfigManager:
    def __init__(self, config_filename: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.yaml = YAML()
        self.base_dir = self._resolve_base_dir()
        self.config_path = self.base_dir / config_filename
        self.config: Dict[str, Any] = {}
        self._load_or_create()
        self.logger.info(f"Configuration loaded: {self.config_path}")

    def _resolve_base_dir(self) -> Path:
        if getattr(sys, 'frozen', False):  # PyInstaller frozen
            try:
                return Path(sys.executable).resolve().parent
            except Exception:
                return Path.cwd()
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
                data = self.yaml.load(f) or {}
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
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                self.yaml.dump(self.config, f)
            self.logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error writing configuration: {e}")

    # --- Migration from legacy schema (model_size / whisper_model / whisper_url) ---
    def _migrate_legacy_whisper_section(self, write_if_changed: bool = False):
        wh = self.config.get("whisper", {})
        if not isinstance(wh, dict):  # sanity
            return
        legacy_model_size = wh.pop("model_size", None)
        legacy_model = wh.pop("whisper_model", None)
        legacy_url = wh.pop("whisper_url", None)
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
        # backend mode determination
        if legacy_url:
            # If user changed url from default local one -> treat as external
            default_local = DEFAULT_CONFIG["whisper"]["local_url"]
            if legacy_url != default_local:
                wh["backend_mode"] = "external"
                wh["external_url"] = legacy_url
                changed = True
            else:
                wh["backend_mode"] = "local"
                wh["local_url"] = legacy_url
                changed = True
        # Ensure required keys exist
        for k in ("backend_mode", "local_url", "external_url", "beam_size", "language"):
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

    # Deprecated method kept for minimal surface compatibility (no-op now)
    def print_stop_instructions_based_on_config(self):
        pass
