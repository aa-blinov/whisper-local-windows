"""Tests for the in-process faster-whisper backend.

The real ``faster_whisper.WhisperModel`` would download a multi-GB model
on first call; tests monkey-patch it with a MagicMock so they stay fast
and deterministic.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest


def _wait(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ---- Construction & initial state ------------------------------------------


def test_initial_status_is_stopped():
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(model="tiny")
    assert backend.status() == "stopped"
    assert backend.current_model() == "tiny"
    assert backend.health_check() is False


def test_implements_transcription_backend_protocol():
    """FasterWhisperBackend must satisfy the public Protocol surface."""
    from app.backends.base import TranscriptionBackend
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(model="tiny")
    # runtime_checkable Protocol — checks attribute presence at runtime.
    assert isinstance(backend, TranscriptionBackend)


# ---- Loading ---------------------------------------------------------------


def test_load_transitions_through_loading_to_ready(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    monkeypatch.setattr(
        "faster_whisper.WhisperModel",
        lambda *args, **kwargs: MagicMock(),
    )

    backend = FasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    assert backend.health_check() is True


def test_load_failure_transitions_to_error(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    def boom(*args, **kwargs):
        raise RuntimeError("model not found")

    monkeypatch.setattr("faster_whisper.WhisperModel", boom)

    backend = FasterWhisperBackend(model="bad-model")
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_load_is_idempotent(monkeypatch):
    """Calling load() while a previous load is in flight must not double-load."""
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    call_count = {"n": 0}

    def slow(*args, **kwargs):
        call_count["n"] += 1
        time.sleep(0.05)
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", slow)

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    backend.load()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert call_count["n"] == 1


def test_load_passes_device_and_compute_type(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    captured: dict = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", fake)

    backend = FasterWhisperBackend(model="medium", device="cuda", compute_type="float16")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert captured["args"] == ("medium",)
    assert captured["kwargs"]["device"] == "cuda"
    assert captured["kwargs"]["compute_type"] == "float16"


# ---- Transcription ---------------------------------------------------------


def test_transcribe_returns_concatenated_segment_text(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    seg1 = SimpleNamespace(text="Hello ")
    seg2 = SimpleNamespace(text="world.")
    info = SimpleNamespace(language="en")

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([seg1, seg2]), info)
    monkeypatch.setattr(
        "faster_whisper.WhisperModel", lambda *a, **kw: fake_model,
    )

    backend = FasterWhisperBackend(model="tiny", language="en", beam_size=7)
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(16000, dtype=np.float32)
    result = backend.transcribe(audio)

    assert result == "Hello world."
    fake_model.transcribe.assert_called_once()
    _args, kwargs = fake_model.transcribe.call_args
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 7


def test_transcribe_returns_none_when_not_loaded():
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(model="tiny")
    audio = np.zeros(16000, dtype=np.float32)

    assert backend.transcribe(audio) is None


def test_transcribe_returns_none_when_model_returned_empty(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([]), SimpleNamespace())
    monkeypatch.setattr(
        "faster_whisper.WhisperModel", lambda *a, **kw: fake_model,
    )

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_swallows_runtime_errors(monkeypatch):
    """A failed transcribe call must not crash the recording pipeline."""
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("CUDA OOM")
    monkeypatch.setattr(
        "faster_whisper.WhisperModel", lambda *a, **kw: fake_model,
    )

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


# ---- Model swap ------------------------------------------------------------


def test_change_model_loads_new_model(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    loaded: list[str] = []

    def factory(model_name, *args, **kwargs):
        loaded.append(model_name)
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", factory)

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.change_model("base")
    assert _wait(lambda: backend.current_model() == "base")
    assert _wait(lambda: backend.status() == "ready")

    assert loaded == ["tiny", "base"]


def test_change_model_to_same_name_is_noop(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    call_count = {"n": 0}

    def factory(*args, **kwargs):
        call_count["n"] += 1
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", factory)

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.change_model("tiny")
    time.sleep(0.05)

    assert call_count["n"] == 1


# ---- Shutdown --------------------------------------------------------------


def test_shutdown_resets_state_to_stopped(monkeypatch):
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    monkeypatch.setattr(
        "faster_whisper.WhisperModel", lambda *a, **kw: MagicMock(),
    )

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.shutdown()
    assert backend.status() == "stopped"


def test_shutdown_during_load_does_not_overwrite_state(monkeypatch):
    """If shutdown happens while a load is in flight, the late load result
    must not flip the status back to ``ready``."""
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    def slow(*args, **kwargs):
        time.sleep(0.1)
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", slow)

    backend = FasterWhisperBackend(model="tiny")
    backend.load()
    backend.shutdown()
    time.sleep(0.2)  # let the background load complete

    assert backend.status() == "stopped"
