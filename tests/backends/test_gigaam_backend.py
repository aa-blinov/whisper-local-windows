"""Tests for the GigaAM backend.

GigaAM ships its own model loader; tests monkey-patch ``gigaam.load_model``
with a MagicMock so we never download real weights or pull in the heavy
PyTorch dependency tree.
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


def _install_fake_gigaam(monkeypatch, load_model=None) -> MagicMock:
    """Install a fake ``gigaam`` module exposing ``load_model``."""
    fake = types.ModuleType("gigaam")
    fake.load_model = load_model or MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "gigaam", fake)
    return fake


# ---- Construction & initial state ------------------------------------------


def test_initial_status_is_stopped():
    from app.backends.gigaam_backend import GigaamBackend

    backend = GigaamBackend(model="v2_ctc")
    assert backend.status() == "stopped"
    assert backend.current_model() == "v2_ctc"
    assert backend.health_check() is False


def test_implements_transcription_backend_protocol():
    from app.backends.base import TranscriptionBackend
    from app.backends.gigaam_backend import GigaamBackend

    backend = GigaamBackend(model="v2_ctc")
    assert isinstance(backend, TranscriptionBackend)


# ---- Loading ---------------------------------------------------------------


def test_load_transitions_through_loading_to_ready(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    _install_fake_gigaam(monkeypatch)

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    assert backend.health_check() is True


def test_load_failure_transitions_to_error(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    def boom(_name):
        raise RuntimeError("model file missing")

    _install_fake_gigaam(monkeypatch, load_model=boom)

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "error")
    assert backend.health_check() is False


def test_load_passes_model_name_to_loader(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    load_model = MagicMock(return_value=MagicMock())
    _install_fake_gigaam(monkeypatch, load_model=load_model)

    backend = GigaamBackend(model="v2_rnnt")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    load_model.assert_called_once()
    args, _kwargs = load_model.call_args
    assert args[0] == "v2_rnnt"


# ---- Transcription ---------------------------------------------------------


def test_transcribe_returns_model_text(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    fake_model = MagicMock()
    fake_model.transcribe.return_value = "привет мир"
    _install_fake_gigaam(
        monkeypatch, load_model=MagicMock(return_value=fake_model)
    )

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    audio = np.zeros(16000, dtype=np.float32)
    result = backend.transcribe(audio, sample_rate=16000)
    assert result == "привет мир"


def test_transcribe_returns_none_when_not_loaded():
    from app.backends.gigaam_backend import GigaamBackend

    backend = GigaamBackend(model="v2_ctc")
    audio = np.zeros(16000, dtype=np.float32)
    assert backend.transcribe(audio) is None


def test_transcribe_returns_none_when_model_returned_empty(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    fake_model = MagicMock()
    fake_model.transcribe.return_value = "   "  # only whitespace
    _install_fake_gigaam(
        monkeypatch, load_model=MagicMock(return_value=fake_model)
    )

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


def test_transcribe_swallows_runtime_errors(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("decoder crashed")
    _install_fake_gigaam(
        monkeypatch, load_model=MagicMock(return_value=fake_model)
    )

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) is None


# ---- Model swap ------------------------------------------------------------


def test_change_model_loads_new_model(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    load_model = MagicMock(return_value=MagicMock())
    _install_fake_gigaam(monkeypatch, load_model=load_model)

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.change_model("v2_rnnt")
    assert _wait(lambda: backend.current_model() == "v2_rnnt")
    assert _wait(lambda: backend.status() == "ready")
    # Loader called twice: initial + change.
    assert load_model.call_count == 2


def test_change_model_to_same_name_is_noop(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    load_model = MagicMock(return_value=MagicMock())
    _install_fake_gigaam(monkeypatch, load_model=load_model)

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.change_model("v2_ctc")
    time.sleep(0.05)

    assert load_model.call_count == 1


# ---- Shutdown --------------------------------------------------------------


def test_shutdown_resets_state_to_stopped(monkeypatch):
    from app.backends.gigaam_backend import GigaamBackend

    _install_fake_gigaam(monkeypatch)

    backend = GigaamBackend(model="v2_ctc")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.shutdown()
    assert backend.status() == "stopped"
