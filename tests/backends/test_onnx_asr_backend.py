"""Tests for OnnxAsrBackend — the unified ONNX backend for Whisper,
GigaAM and Parakeet model families.

``onnx-asr`` is an optional heavy dependency in test envs; the suite
installs a fake module so it runs without the real package.
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


def _install_fake_onnx_asr(
    monkeypatch,
    recognize_fn=None,
    recognize_return: object = "fake transcription",
):
    """Install a fake ``onnx_asr`` module with a ``load_model`` function.

    Returns ``(fake_module, fake_model)`` so tests can assert on both
    ``onnx_asr.load_model`` arguments and ``model.recognize`` arguments.
    """
    fake_model = MagicMock()
    if recognize_fn is not None:
        fake_model.recognize.side_effect = recognize_fn
    else:
        fake_model.recognize.return_value = recognize_return

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(return_value=fake_model)

    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)
    return fake_module, fake_model


# ---- Construction & initial state ------------------------------------------


def test_initial_status_is_stopped():
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="onnx-community/whisper-large-v3-turbo")
    assert backend.status() == "stopped"
    assert backend.current_model() == "onnx-community/whisper-large-v3-turbo"
    assert backend.health_check() is False


def test_implements_transcription_backend_protocol():
    from app.backends.base import TranscriptionBackend
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    assert isinstance(backend, TranscriptionBackend)


# ---- Family-aware language reporting ---------------------------------------


def test_parakeet_family_reports_no_language():
    """Parakeet TDT v3 auto-detects across 25 languages — current_language()
    must be None so history rows aren't tagged with a wrong code."""
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="istupakov/parakeet-tdt-0.6b-v3-onnx", family="parakeet"
    )
    assert backend.current_language() is None


def test_gigaam_family_reports_russian():
    """GigaAM is Russian-only — must always report 'ru'."""
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="istupakov/gigaam-v3-onnx", family="gigaam"
    )
    assert backend.current_language() == "ru"


def test_whisper_family_reports_user_language():
    """Whisper respects the user's language setting; 'auto' / None means
    auto-detect (return None)."""
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="en",
    )
    assert backend.current_language() == "en"

    backend_auto = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="auto",
    )
    assert backend_auto.current_language() is None


# ---- Loading ---------------------------------------------------------------


def test_load_transitions_through_loading_to_ready(monkeypatch):
    from app.backends.onnx_backend import OnnxAsrBackend

    _install_fake_onnx_asr(monkeypatch)

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    assert backend.health_check() is True


def test_load_passes_model_name_to_load_model(monkeypatch):
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="onnx-community/whisper-base")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_module.load_model.assert_called_once()
    args, _ = fake_module.load_model.call_args
    assert args[0] == "onnx-community/whisper-base"


def test_load_passes_quantization_kwarg_when_requested(monkeypatch):
    """``quantization='int8'`` must reach onnx_asr.load_model so the
    int8 variant is selected."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x", quantization="int8")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    _args, kwargs = fake_module.load_model.call_args
    assert kwargs.get("quantization") == "int8"


def test_load_does_not_pass_quantization_when_none(monkeypatch):
    """If quantization is None (default), don't pass it — let onnx_asr
    pick its own default."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    _args, kwargs = fake_module.load_model.call_args
    assert "quantization" not in kwargs


def test_load_passes_cuda_provider_when_device_cuda(monkeypatch):
    """``device='cuda'`` must result in providers=['CUDAExecutionProvider',
    'CPUExecutionProvider'] being passed to load_model — the second
    entry gives ONNX Runtime an automatic CPU fallback if CUDA fails."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x", device="cuda")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    _args, kwargs = fake_module.load_model.call_args
    providers = kwargs.get("providers")
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_load_passes_cpu_provider_when_device_cpu(monkeypatch):
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x", device="cpu")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    _args, kwargs = fake_module.load_model.call_args
    assert kwargs.get("providers") == ["CPUExecutionProvider"]


def test_load_omits_providers_when_device_auto(monkeypatch):
    """``device='auto'`` lets ONNX Runtime pick — don't pass providers
    so it uses its built-in auto-discovery."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x", device="auto")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    _args, kwargs = fake_module.load_model.call_args
    assert "providers" not in kwargs


def test_load_failure_transitions_to_error(monkeypatch):
    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(side_effect=RuntimeError("download failed"))
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_import_error_transitions_to_error(monkeypatch):
    """``onnx-asr`` not installed → backend lands on ``error``."""
    monkeypatch.delitem(sys.modules, "onnx_asr", raising=False)

    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "onnx_asr":
            raise ImportError("No module named 'onnx_asr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "error")


def test_load_falls_back_to_cpu_when_cuda_provider_missing(monkeypatch):
    """If the user picked ``device='cuda'`` but the first load_model
    raises an error mentioning CUDA / providers, the backend retries
    once with CPU provider only — better degraded mode than a hard
    fail in the UI."""
    fake_module = types.ModuleType("onnx_asr")
    fake_model = MagicMock()
    call_count = {"n": 0}

    def flaky_load(model_name, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError(
                "[E:onnxruntime] CUDAExecutionProvider not available"
            )
        return fake_model

    fake_module.load_model = MagicMock(side_effect=flaky_load)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x", device="cuda")
    backend.load()
    assert _wait(lambda: backend.status() == "ready"), (
        "expected CPU fallback after CUDA provider failure"
    )

    # Two load_model attempts: first CUDA (failed), second CPU (succeeded).
    assert fake_module.load_model.call_count == 2
    second_call_kwargs = fake_module.load_model.call_args_list[1].kwargs
    assert second_call_kwargs.get("providers") == ["CPUExecutionProvider"]


# ---- Model swap ------------------------------------------------------------


def test_change_model_triggers_reload(monkeypatch):
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="model-a")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    fake_module.load_model.reset_mock()

    backend.change_model("model-b")
    assert _wait(lambda: backend.status() == "ready")
    assert backend.current_model() == "model-b"
    fake_module.load_model.assert_called_once()


def test_change_model_accepts_compute_type_for_api_parity(monkeypatch):
    """``compute_type`` is passed by the routed-backend façade — accept
    and ignore (ONNX precision lives in the model repo / quantization
    parameter, not on a per-call kwarg)."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.change_model("x", compute_type="float16")
    assert backend.status() == "ready"


# ---- Transcription ---------------------------------------------------------


def test_transcribe_returns_none_when_not_ready():
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_returns_string_from_recognize(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "fake transcription"


def test_transcribe_handles_object_with_text_attribute(monkeypatch):
    """``recognize()`` returns either a bare string or a result-like
    object with a ``.text`` attribute — handle both."""
    result_obj = MagicMock()
    result_obj.text = "from-object"
    _install_fake_onnx_asr(monkeypatch, recognize_return=result_obj)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) == "from-object"


def test_transcribe_returns_none_on_empty_string(monkeypatch):
    _install_fake_onnx_asr(monkeypatch, recognize_return="   ")

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_passes_language_for_whisper(monkeypatch):
    """For Whisper family, the user-selected language must be passed to
    ``model.recognize(language=...)`` so Whisper picks the right
    language model variant."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="ru",
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    fake_model.recognize.assert_called()
    _args, kwargs = fake_model.recognize.call_args
    assert kwargs.get("language") == "ru"


def test_transcribe_omits_language_for_whisper_when_auto(monkeypatch):
    """``language='auto'`` (or None) means auto-detect — don't pass
    language to recognize()."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="auto",
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    _args, kwargs = fake_model.recognize.call_args
    assert "language" not in kwargs


def test_transcribe_omits_language_for_non_whisper_families(monkeypatch):
    """GigaAM and Parakeet ignore the language kwarg — don't pass it to
    avoid clutter / unexpected onnx_asr behaviour."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    for family in ("gigaam", "parakeet"):
        fake_model.recognize.reset_mock()
        backend = OnnxAsrBackend(model="x", family=family, language="ru")
        backend.load()
        assert _wait(lambda: backend.status() == "ready")
        backend.transcribe(np.zeros(16000, dtype=np.float32))
        _args, kwargs = fake_model.recognize.call_args
        assert "language" not in kwargs, (
            f"{family} backend must not forward language kwarg"
        )


def test_transcribe_short_audio_calls_recognize_once(monkeypatch):
    chunks: list = []

    def capture(audio, **_kwargs):
        chunks.append(len(audio))
        return "ok"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(24 * 16000, dtype=np.float32)
    backend.transcribe(audio)
    assert chunks == [24 * 16000]


def test_transcribe_long_audio_splits_into_chunks(monkeypatch):
    chunks: list = []

    def capture(audio, **_kwargs):
        chunks.append(len(audio))
        return "chunk"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture)

    from app.backends.onnx_backend import OnnxAsrBackend, _CHUNK_SAMPLES

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(60 * 16000, dtype=np.float32)
    backend.transcribe(audio)

    assert len(chunks) == 3
    assert chunks[0] == _CHUNK_SAMPLES
    assert chunks[1] == _CHUNK_SAMPLES
    assert chunks[2] == 10 * 16000


def test_transcribe_joins_multi_chunk_results(monkeypatch):
    counter = {"n": 0}

    def per_chunk(_audio, **_kwargs):
        counter["n"] += 1
        return f"part{counter['n']}"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=per_chunk)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(60 * 16000, dtype=np.float32)
    text = backend.transcribe(audio)
    assert text == "part1 part2 part3"


def test_transcribe_normalises_multichannel_input(monkeypatch):
    received: list = []

    def capture(audio, **_kwargs):
        received.append(audio.ndim)
        return "ok"

    _install_fake_onnx_asr(monkeypatch, recognize_fn=capture)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    stereo = np.zeros((16000, 2), dtype=np.float32)
    backend.transcribe(stereo)
    assert received[0] == 1


# ---- Lifecycle -------------------------------------------------------------


def test_shutdown_is_idempotent(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()
    backend.shutdown()
    assert backend.status() == "stopped"


def test_shutdown_blocks_subsequent_load(monkeypatch):
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.shutdown()
    backend.load()
    assert backend.status() == "stopped"
    fake_module.load_model.assert_not_called()


def test_transcribe_returns_none_after_shutdown(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


# ---- Cancel-load -----------------------------------------------------------


def test_cancel_load_returns_status_to_stopped(monkeypatch):
    started = threading.Event()
    finish = threading.Event()

    def slow_loader(_model_id, **_kwargs):
        started.set()
        finish.wait(timeout=2.0)
        return MagicMock()

    fake_module = types.ModuleType("onnx_asr")
    fake_module.load_model = MagicMock(side_effect=slow_loader)
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_module)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert started.wait(2.0)
    assert backend.status() == "loading"

    backend.cancel_load()
    assert backend.status() == "stopped"

    finish.set()
    assert _wait(lambda: backend.status() == "stopped")
    assert backend._model is None


def test_cancel_load_is_noop_when_not_loading(monkeypatch):
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.cancel_load()
    assert backend.status() == "stopped"

    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.cancel_load()
    assert backend.status() == "ready"


# ---- Inference settings (timestamps) --------------------------------------


def test_with_timestamps_called_when_settings_enable_it(monkeypatch):
    """When the user enables timestamps in inference settings, the backend
    must call ``model.with_timestamps()`` to wrap the model before running
    recognise (it is a chainable adapter, not a bool kwarg)."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)
    fake_model.with_timestamps = MagicMock(return_value=fake_model)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import NemoInferenceSettings

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(NemoInferenceSettings(timestamps=True))
    backend.transcribe(np.zeros(16000, dtype=np.float32))

    fake_model.with_timestamps.assert_called()


def test_with_timestamps_not_called_when_disabled(monkeypatch):
    _, fake_model = _install_fake_onnx_asr(monkeypatch)
    fake_model.with_timestamps = MagicMock(return_value=fake_model)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    fake_model.with_timestamps.assert_not_called()
