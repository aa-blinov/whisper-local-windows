"""Factory: build the StateManager + HotkeyListener stack from config.

Mirrors the wiring previously done inside the legacy AppContext: instantiate
audio recorder, whisper engine, clipboard manager, audio feedback, then a
StateManager, then a HotkeyListener. Centralised here so the new Qt entry
point (and integration tests) can compose the recording subsystem in one call.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.audio_feedback import AudioFeedback
from app.audio_recorder import AudioRecorder
from app.clipboard_manager import ClipboardManager
from app.config_manager import ConfigManager
from app.hotkey_listener import HotkeyListener
from app.model_mapping import ALIAS_TO_MODEL, alias_for
from app.state_manager import StateManager
from app.whisper_engine import WhisperEngine


log = logging.getLogger(__name__)


def _wyoming_url(http_or_tcp_url: str) -> str:
    """Strip http(s):// for Wyoming TCP usage."""
    if not http_or_tcp_url:
        return "localhost:10300"
    if http_or_tcp_url.startswith("http://"):
        return http_or_tcp_url[len("http://"):]
    if http_or_tcp_url.startswith("https://"):
        return http_or_tcp_url[len("https://"):]
    return http_or_tcp_url


def _resolve_model_pair(raw: Optional[str]) -> Tuple[str, str]:
    """Return (alias, canonical) for whatever the config stored."""
    if raw and raw in ALIAS_TO_MODEL:
        return raw, ALIAS_TO_MODEL[raw]
    if raw:
        return alias_for(raw), raw
    return "large-v3", "Systran/faster-whisper-large-v3"


def build_recording_stack(
    config_manager: ConfigManager,
    docker_backend_manager=None,
) -> Tuple[StateManager, HotkeyListener]:
    """Build the full domain stack and start the global hotkey listener.

    Returns ``(state_manager, hotkey_listener)``. The HotkeyListener begins
    listening immediately (its ``__init__`` calls ``start_listening``), so the
    caller must keep both objects alive for the lifetime of the app.
    """
    whisper_cfg = config_manager.get_whisper_config()
    audio_cfg = config_manager.get_audio_config()
    clipboard_cfg = config_manager.get_clipboard_config()
    feedback_cfg = config_manager.get_audio_feedback_config()
    hotkey_cfg = config_manager.get_hotkey_config()

    backend_mode = whisper_cfg.get("backend_mode", "local")
    base_url_key = "local_url" if backend_mode == "local" else "external_url"
    base_url = whisper_cfg.get(base_url_key) or whisper_cfg.get("local_url") or "localhost:10300"
    wyoming_url = _wyoming_url(base_url)

    alias, canonical = _resolve_model_pair(whisper_cfg.get("model"))

    audio_feedback = AudioFeedback(
        enabled=bool(feedback_cfg.get("enabled", True)),
        start_sound=str(feedback_cfg.get("start_sound", "")),
        stop_sound=str(feedback_cfg.get("stop_sound", "")),
        cancel_sound=str(feedback_cfg.get("cancel_sound", "")),
    )

    audio_recorder = AudioRecorder(
        channels=int(audio_cfg.get("channels", 1)),
        dtype=str(audio_cfg.get("dtype", "float32")),
        max_duration=int(audio_cfg.get("max_duration", 300)),
    )

    clipboard_manager = ClipboardManager(
        key_simulation_delay=float(clipboard_cfg.get("key_simulation_delay", 0.05)),
        auto_paste=bool(clipboard_cfg.get("auto_paste", True)),
        preserve_clipboard=bool(clipboard_cfg.get("preserve_clipboard", False)),
    )

    whisper_engine = WhisperEngine(
        base_url=wyoming_url,
        model_size=alias,
        language=whisper_cfg.get("language") or None,
        beam_size=int(whisper_cfg.get("beam_size", 5)),
        remote_model=canonical,
    )

    state_manager = StateManager(
        audio_recorder=audio_recorder,
        whisper_engine=whisper_engine,
        clipboard_manager=clipboard_manager,
        config_manager=config_manager,
        system_tray=None,
        audio_feedback=audio_feedback,
        docker_backend_manager=docker_backend_manager,
    )

    # Recorder needs to call back into state_manager when max duration is hit.
    audio_recorder.on_max_duration_reached = (
        state_manager.handle_max_recording_duration_reached
    )

    hotkey_listener = HotkeyListener(
        state_manager=state_manager,
        start_recording_hotkey=hotkey_cfg.get("start_recording_hotkey", "ctrl+f2"),
        stop_recording_hotkey=hotkey_cfg.get("stop_recording_hotkey", "ctrl+f3"),
    )

    log.info(
        "Recording stack built: model=%s url=%s backend_mode=%s",
        alias, wyoming_url, backend_mode,
    )
    return state_manager, hotkey_listener
