"""Tests for OnnxParakeetBackend.

``onnx-asr`` is an optional heavy dependency; the tests install a fake
module so the suite runs without it. Real inference is tested manually
or in a dedicated integration test.
"""

from __future__ import annotations

import sys
import threading
import time
import types
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


def _install_fake_onnx_asr(monkeypatch, recognize_fn=None) -> MagicMock:
    """Install a fake ``onnx_asr`` module with a ``load_model`` function.

    The returned fake model calls ``recognize_fn(audio)`` if supplied,
    otherwise returns the string ``"fake transcription"``.
    """
    fake_model = MagicMock()
    if recognize_fn is not None:
        fake_model.recognize.side_effect = recognize_fn
    else:
        fake_model.recognize.return_value = "fake transcription"

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=fake_model)

    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)
    return fake_model


# ---- Construction & initial state ------------------------------------------


def test_initial_status_is_stopped():
    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend(model="istupakov/parakeet-tdt-0.6b-v3-onnx")
    assert backend.status() == "stopped"
    assert backend.current_model() == "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert backend.health_check() is False


def test_current_language_is_none():
    """Auto-detect — same as NeMo Parakeet, no source-language knob."""
    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    assert backend.current_language() is None


def test_implements_transcription_backend_protocol():
    from app.backends.base import TranscriptionBackend
    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    assert isinstance(backend, TranscriptionBackend)


# ---- Loading ---------------------------------------------------------------


def test_load_transitions_through_loading_to_ready(monkeypatch):
    from app.backends.onnx_backend import OnnxParakeetBackend

    _install_fake_onnx_asr(monkeypatch)

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    assert backend.health_check() is True


def test_load_failure_transitions_to_error(monkeypatch):
    """``load_model()`` raises — backend lands on ``error``, not stuck
    at ``loading``."""
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(
        side_effect=RuntimeError("download failed")
    )
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_import_error_transitions_to_error(monkeypatch):
    """``onnx-asr`` not installed → backend lands on ``error``."""
    # Remove any cached fake/real module so the importer re-resolves.
    monkeypatch.delitem(sys.modules, "onnx_asr", raising=False)

    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "onnx_asr":
            raise ImportError("No module named 'onnx_asr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_load_is_idempotent_when_already_loading(monkeypatch):
    """Calling ``load()`` twice should not spawn a second thread or
    reset the state — the first load is already in progress."""
    started = threading.Event()
    finish = threading.Event()

    def slow_loader(_model_id):
        started.set()
        finish.wait(timeout=2.0)
        return MagicMock()

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(side_effect=slow_loader)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert started.wait(2.0)
    assert backend.status() == "loading"

    backend.load()  # second call — must not reset to stopped/start a new thread
    assert backend.status() == "loading"

    finish.set()
    assert _wait(lambda: backend.status() == "ready")
    # Only one actual load_model call.
    assert fake_module.load_model.call_count == 1


def test_load_is_noop_when_already_ready(monkeypatch):
    """``load()`` when already ``ready`` must be a no-op."""
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    count_before = fake_module.load_model.call_count

    backend.load()
    time.sleep(0.05)
    assert fake_module.load_model.call_count == count_before


# ---- Model swap ------------------------------------------------------------


def test_change_model_triggers_reload(monkeypatch):
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend(model="model-a")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    fake_module.load_model.reset_mock()

    backend.change_model("model-b")
    assert _wait(lambda: backend.status() == "ready")
    assert backend.current_model() == "model-b"
    fake_module.load_model.assert_called_once()


def test_change_model_same_noop(monkeypatch):
    """Calling ``change_model`` with the current model when already ready
    must not trigger a reload."""
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend(model="model-a")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    count_before = fake_module.load_model.call_count

    backend.change_model("model-a")
    time.sleep(0.05)
    assert fake_module.load_model.call_count == count_before


def test_change_model_accepts_compute_type_for_api_parity(monkeypatch):
    """``compute_type`` is meaningless for ONNX but accepted for API parity
    with faster-whisper — must not raise."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.change_model("istupakov/parakeet-tdt-0.6b-v3-onnx", compute_type="float16")
    # Same model, already ready — no exception, stays ready.
    assert backend.status() == "ready"


# ---- Transcription ---------------------------------------------------------


def test_transcribe_returns_none_when_not_ready():
    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_returns_string_from_model(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "fake transcription"


def test_transcribe_handles_object_with_text_attribute(monkeypatch):
    """``recognize()`` may return an object with ``.text`` instead of
    a bare string — handle both."""
    result_obj = MagicMock()
    result_obj.text = "from-object"

    fake_module = types.ModuleType("onnx_asr")
    fake_model = MagicMock()
    fake_model.recognize.return_value = result_obj
    fake_module.load_model = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "from-object"


def test_transcribe_returns_none_on_empty_string(monkeypatch):
    fake_module = types.ModuleType("onnx_asr")
    fake_model = MagicMock()
    fake_model.recognize.return_value = "   "
    fake_module.load_model = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_returns_none_when_model_returns_none(monkeypatch):
    fake_module = types.ModuleType("onnx_asr")
    fake_model = MagicMock()
    fake_model.recognize.return_value = None
    fake_module.load_model = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_short_audio_calls_recognize_once(monkeypatch):
    """Audio <= 25 s must be passed to ``recognize()`` as a single call
    — no chunking for short recordings."""
    chunks_received: list = []

    def capture_recognize(audio):
        chunks_received.append(len(audio))
        return "ok"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture_recognize)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(24 * 16000, dtype=np.float32)  # 24 s
    backend.transcribe(audio)
    assert len(chunks_received) == 1
    assert chunks_received[0] == len(audio)


def test_transcribe_long_audio_splits_into_chunks(monkeypatch):
    """Audio > 25 s must be split into multiple calls, each ≤ 25 s."""
    chunks_received: list = []

    def capture_recognize(audio):
        chunks_received.append(len(audio))
        return "chunk"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture_recognize)

    from app.backends.onnx_backend import OnnxParakeetBackend, _CHUNK_SAMPLES

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    # 60 s audio → 3 chunks of 25 s / 25 s / 10 s.
    audio = np.zeros(60 * 16000, dtype=np.float32)
    backend.transcribe(audio)

    assert len(chunks_received) == 3
    assert chunks_received[0] == _CHUNK_SAMPLES
    assert chunks_received[1] == _CHUNK_SAMPLES
    assert chunks_received[2] == 10 * 16000  # remainder


def test_transcribe_joins_multi_chunk_results(monkeypatch):
    """Multiple chunks must be joined with a space into one string."""
    call_count = 0

    def per_chunk(audio):
        nonlocal call_count
        call_count += 1
        return f"part{call_count}"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=per_chunk)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(60 * 16000, dtype=np.float32)
    text = backend.transcribe(audio)
    assert text == "part1 part2 part3"


def test_transcribe_skips_empty_chunks_in_join(monkeypatch):
    """Chunks where ``recognize`` returns empty / None must not add
    empty tokens to the joined result."""
    results = ["hello", "", "world"]
    idx = [0]

    def rotating(_audio):
        r = results[idx[0] % len(results)]
        idx[0] += 1
        return r

    _install_fake_onnx_asr(monkeypatch, recognize_fn=rotating)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(75 * 16000, dtype=np.float32)
    text = backend.transcribe(audio)
    assert text == "hello world"


def test_transcribe_normalises_multichannel_input(monkeypatch):
    """Stereo (2-channel) input must be downmixed to mono before inference."""
    received: list = []

    def capture(audio):
        received.append(audio.ndim)
        return "ok"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    stereo = np.zeros((16000, 2), dtype=np.float32)
    backend.transcribe(stereo)
    assert received[0] == 1, "model must receive 1-D (mono) audio"


# ---- Lifecycle -------------------------------------------------------------


def test_shutdown_is_idempotent(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()
    backend.shutdown()
    assert backend.status() == "stopped"


def test_shutdown_blocks_subsequent_load(monkeypatch):
    """``load()`` after ``shutdown()`` must be a permanent no-op."""
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.shutdown()
    backend.load()
    assert backend.status() == "stopped"
    fake_module.load_model.assert_not_called()


def test_transcribe_returns_none_after_shutdown(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


# ---- Cancel-load -----------------------------------------------------------


def test_cancel_load_returns_status_to_stopped(monkeypatch):
    """Cancelling a slow load must flip status back to ``stopped``
    immediately without waiting for the download to finish."""
    started = threading.Event()
    finish = threading.Event()

    def slow_loader(_model_id):
        started.set()
        finish.wait(timeout=2.0)
        return MagicMock()

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(side_effect=slow_loader)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert started.wait(2.0)
    assert backend.status() == "loading"

    backend.cancel_load()
    assert backend.status() == "stopped"

    finish.set()
    assert _wait(lambda: backend.status() == "stopped")
    assert backend._model is None


def test_cancel_load_is_noop_when_not_loading(monkeypatch):
    """Idempotent — cancel when idle or already ready must not raise."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.cancel_load()  # not loading yet
    assert backend.status() == "stopped"

    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.cancel_load()  # already ready
    assert backend.status() == "ready"


def test_load_after_cancel_resumes_normally(monkeypatch):
    """After a cancel the user may click the model again; a fresh
    ``load()`` must reset the cancel flag and reach ``ready``."""
    started = threading.Event()
    finish = threading.Event()

    def slow_loader(_model_id):
        if not started.is_set():
            started.set()
            finish.wait(timeout=2.0)
        return MagicMock()

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(side_effect=slow_loader)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxParakeetBackend

    backend = OnnxParakeetBackend()
    backend.load()
    assert started.wait(2.0)
    backend.cancel_load()
    finish.set()
    assert _wait(lambda: backend.status() == "stopped")

    backend.load()
    assert _wait(lambda: backend.status() == "ready")
