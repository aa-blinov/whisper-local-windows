"""Tests for the NemoBackend (NVIDIA NeMo / Parakeet TDT v3).

NeMo's heavy dependency tree (PyTorch Lightning, hydra, lhotse, …) gets
mocked so the suite stays fast and runnable even without ``nemo_toolkit``
installed in the dev env.
"""

from __future__ import annotations

import sys
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


def _install_fake_nemo(monkeypatch, asr_model_class=None) -> MagicMock:
    """Install a fake ``nemo.collections.asr`` module exposing
    ``models.ASRModel.from_pretrained``.

    The real package layout is ``nemo.collections.asr.models.ASRModel``;
    callers usually do ``import nemo.collections.asr as nemo_asr`` then
    ``nemo_asr.models.ASRModel.from_pretrained(...)``. Stub all three
    levels so ``from_pretrained`` returns whatever the test set up.
    """
    fake_asr_class = asr_model_class or MagicMock(
        from_pretrained=MagicMock(return_value=MagicMock())
    )
    fake_models = types.ModuleType("nemo.collections.asr.models")
    fake_models.ASRModel = fake_asr_class
    fake_asr = types.ModuleType("nemo.collections.asr")
    fake_asr.models = fake_models
    fake_collections = types.ModuleType("nemo.collections")
    fake_collections.asr = fake_asr
    fake_root = types.ModuleType("nemo")
    fake_root.collections = fake_collections
    monkeypatch.setitem(sys.modules, "nemo", fake_root)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", fake_asr)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", fake_models)
    return fake_asr_class


# ---- Construction & initial state ------------------------------------------


def test_initial_status_is_stopped():
    from app.backends.nemo_backend import NemoBackend

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    assert backend.status() == "stopped"
    assert backend.current_model() == "nvidia/parakeet-tdt-0.6b-v3"
    assert backend.health_check() is False


def test_current_language_is_none_for_auto_detect():
    """Parakeet TDT v3 auto-detects the spoken language across 25
    languages and exposes no source-lang knob — we report ``None``
    so history entries don't get tagged with a wrong code."""
    from app.backends.nemo_backend import NemoBackend

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    assert backend.current_language() is None


def test_implements_transcription_backend_protocol():
    from app.backends.base import TranscriptionBackend
    from app.backends.nemo_backend import NemoBackend

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    assert isinstance(backend, TranscriptionBackend)


# ---- Loading ---------------------------------------------------------------


def test_load_transitions_through_loading_to_ready(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    _install_fake_nemo(monkeypatch)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    assert backend.health_check() is True


def test_load_failure_transitions_to_error(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(
        side_effect=RuntimeError("download failed")
    )
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_load_passes_model_name_to_loader(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=MagicMock())
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    fake_class.from_pretrained.assert_called_once()
    _args, kwargs = fake_class.from_pretrained.call_args
    assert kwargs.get("model_name") == "nvidia/parakeet-tdt-0.6b-v3"


def test_load_switches_to_local_attention_for_long_audio(monkeypatch):
    """The default full-attention layout caps Parakeet at ~24 minutes
    of audio. Switching to ``rel_pos_local_attn`` extends that to
    several hours at a small accuracy cost — worth it because
    ``transcribe`` doesn't know in advance how long the input is, and
    paying the cost once at load is simpler than re-loading on demand."""
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    fake_model.change_attention_model.assert_called_once()
    args, kwargs = fake_model.change_attention_model.call_args
    # Either positional or keyword; check both shapes.
    flat = list(args) + [v for v in kwargs.values()]
    assert "rel_pos_local_attn" in flat
    assert any(isinstance(v, list) and v == [256, 256] for v in flat)


# ---- Model swap -----------------------------------------------------------


def test_change_model_triggers_reload(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=MagicMock())
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    fake_class.from_pretrained.reset_mock()

    backend.change_model("nvidia/canary-1b-v2")
    assert _wait(lambda: backend.status() == "ready")
    assert backend.current_model() == "nvidia/canary-1b-v2"
    fake_class.from_pretrained.assert_called_once()
    _args, kwargs = fake_class.from_pretrained.call_args
    assert kwargs.get("model_name") == "nvidia/canary-1b-v2"


def test_change_model_accepts_compute_type_for_api_parity(monkeypatch):
    """``compute_type`` is meaningless to NeMo (precision lives on the
    model card / launch flags) but the routed-backend façade calls
    every backend with the same shape — accept and ignore."""
    from app.backends.nemo_backend import NemoBackend

    _install_fake_nemo(monkeypatch)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.change_model("nvidia/parakeet-tdt-0.6b-v3", compute_type="float16")
    # No swap (same model, already ready) and no exception raised.
    assert backend.current_model() == "nvidia/parakeet-tdt-0.6b-v3"


# ---- Transcription --------------------------------------------------------


def test_transcribe_returns_text_from_first_output(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(
        return_value=[MagicMock(text="hello world")]
    )
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(16000, dtype=np.float32)
    text = backend.transcribe(audio, sample_rate=16000)
    assert text == "hello world"


def test_transcribe_returns_none_when_not_ready():
    from app.backends.nemo_backend import NemoBackend

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    audio = np.zeros(16000, dtype=np.float32)
    assert backend.transcribe(audio) is None


def test_transcribe_handles_string_output(monkeypatch):
    """Some test fakes — and possibly older NeMo versions — return
    plain strings instead of ``Hypothesis`` objects with ``.text``.
    Accept both."""
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(return_value=["plain string output"])
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "plain string output"


def test_transcribe_returns_none_on_empty_output(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(return_value=[])
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_passes_timestamps_flag_when_settings_enable_it(monkeypatch):
    """``update_inference_settings`` flips a per-call kwarg —
    ``timestamps=True`` makes NeMo populate the .timestamp dict on
    each output. Off by default (small but nonzero overhead)."""
    from app.backends.nemo_backend import NemoBackend
    from app.inference_settings import NemoInferenceSettings

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(
        return_value=[MagicMock(text="ok")]
    )
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(NemoInferenceSettings(timestamps=True))
    backend.transcribe(np.zeros(16000, dtype=np.float32))

    fake_model.transcribe.assert_called()
    _args, kwargs = fake_model.transcribe.call_args
    assert kwargs.get("timestamps") is True


def test_transcribe_omits_timestamps_when_settings_disable_it(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(
        return_value=[MagicMock(text="ok")]
    )
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    _args, kwargs = fake_model.transcribe.call_args
    # Either omitted or explicitly False — both are fine.
    assert kwargs.get("timestamps", False) is False


# ---- Lifecycle ------------------------------------------------------------


def test_shutdown_is_idempotent(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    _install_fake_nemo(monkeypatch)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()
    backend.shutdown()  # second call must not raise
    assert backend.status() == "stopped"


def test_shutdown_blocks_subsequent_load(monkeypatch):
    from app.backends.nemo_backend import NemoBackend

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=MagicMock())
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.shutdown()
    backend.load()
    # ``load`` is a no-op after shutdown.
    assert backend.status() == "stopped"
    fake_class.from_pretrained.assert_not_called()
