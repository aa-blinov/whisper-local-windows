"""Factory: build the StateManager + HotkeyListener + backend stack from config.

Replaces the legacy AppContext setup. Composes the audio recorder, clipboard
manager, audio feedback, and an in-process speech-to-text backend (currently
``FasterWhisperBackend``) into a ``StateManager``, then wraps it in a
``HotkeyListener``. Returns the backend separately so the caller can hand it
to the status poller and trigger ``backend.load()`` once the UI is up.
"""

from __future__ import annotations

import logging
from typing import Tuple

from app.audio_feedback import AudioFeedback
from app.audio_recorder import AudioRecorder
from app.backends.base import TranscriptionBackend
from app.backends.faster_whisper_backend import FasterWhisperBackend
from app.clipboard_manager import ClipboardManager
from app.config_manager import ConfigManager
from app.hotkey_listener import HotkeyListener
from app.model_mapping import canonical_for
from app.state_manager import StateManager


log = logging.getLogger(__name__)


def build_recording_stack(
    config_manager: ConfigManager,
) -> Tuple[StateManager, HotkeyListener, TranscriptionBackend]:
    """Build the full domain stack and start the global hotkey listener.

    Returns ``(state_manager, hotkey_listener, backend)``. The HotkeyListener
    begins listening immediately (its ``__init__`` calls ``start_listening``),
    so the caller must keep all three objects alive for the lifetime of the
    app. The returned ``backend`` is constructed in the ``stopped`` state —
    call ``backend.load()`` to begin loading the configured model.
    """
    whisper_cfg = config_manager.get_whisper_config()
    audio_cfg = config_manager.get_audio_config()
    clipboard_cfg = config_manager.get_clipboard_config()
    feedback_cfg = config_manager.get_audio_feedback_config()
    hotkey_cfg = config_manager.get_hotkey_config()

    # Resolve the model name. Accept either an alias from the registry
    # (e.g. ``large-v3``) or a full Hugging Face id; faster-whisper handles
    # both via huggingface_hub.
    raw_model = whisper_cfg.get("model") or "large-v3"
    canonical = canonical_for(raw_model)

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
        device=audio_cfg.get("device"),
    )

    clipboard_manager = ClipboardManager(
        key_simulation_delay=float(clipboard_cfg.get("key_simulation_delay", 0.05)),
        auto_paste=bool(clipboard_cfg.get("auto_paste", True)),
        preserve_clipboard=bool(clipboard_cfg.get("preserve_clipboard", False)),
    )

    backend: TranscriptionBackend = FasterWhisperBackend(
        model=canonical,
        device=str(whisper_cfg.get("device", "auto")),
        compute_type=str(whisper_cfg.get("compute_type", "float16")),
        language=whisper_cfg.get("language") or None,
        beam_size=int(whisper_cfg.get("beam_size", 5)),
    )

    state_manager = StateManager(
        audio_recorder=audio_recorder,
        backend=backend,
        clipboard_manager=clipboard_manager,
        config_manager=config_manager,
        system_tray=None,
        audio_feedback=audio_feedback,
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
        "Recording stack built: model=%s device=%s compute_type=%s",
        canonical,
        whisper_cfg.get("device", "auto"),
        whisper_cfg.get("compute_type", "float16"),
    )
    return state_manager, hotkey_listener, backend
