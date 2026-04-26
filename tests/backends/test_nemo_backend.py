"""Tests for the NemoBackend (NVIDIA NeMo / Parakeet TDT v3).

NeMo's heavy dependency tree (PyTorch Lightning, hydra, lhotse, …) gets
mocked so the suite stays fast and runnable even without ``nemo_toolkit``
installed in the dev env.
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


def test_load_patches_legacy_np_sctypes_shim(monkeypatch):
    """NeMo / lhotse / older librosa code paths call ``np.sctypes``,
    which NumPy 2.0 removed with a hard AttributeError. The backend
    has to put the attribute back as a compat shim, otherwise
    ``transcribe`` raises and the user sees ``Transcription returned
    0 characters`` for every recording."""
    import numpy as np

    # Pretend np.sctypes is missing — matches NumPy 2.0+ actual state.
    monkeypatch.delattr(np, "sctypes", raising=False)

    _install_fake_nemo(monkeypatch)

    from app.backends.nemo_backend import NemoBackend

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert hasattr(np, "sctypes"), "shim was not installed during load"
    # Spot-check the buckets NeMo actually indexes into.
    assert np.float32 in np.sctypes["float"]
    assert np.int32 in np.sctypes["int"]


def test_import_attribute_error_transitions_to_error(monkeypatch):
    """NeMo's exp_manager touches ``signal.SIGKILL`` at class-definition
    time, which is POSIX-only — on Windows the import graph used to die
    with an unhandled ``AttributeError`` and the worker thread silently
    exited while ``_status`` stayed at ``loading``. The watcher then
    sat through the full timeout (8+ minutes) reporting heartbeats on a
    dead thread.

    The fix: ``_do_load`` catches ``Exception`` (not just
    ``ImportError``) on the import line, so any failure mode promotes
    cleanly to ``error`` and the UI gets a real failure signal."""
    from app.backends.nemo_backend import NemoBackend

    # Wipe any cached real/fake nemo modules so the importer is forced
    # to re-resolve and run our raising loader below.
    for mod in [
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
    ]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "nemo.collections.asr":
            raise AttributeError(
                "module 'signal' has no attribute 'SIGKILL'"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

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


def test_transcribe_handles_tuple_of_lists_output(monkeypatch):
    """Parakeet TDT v3 ``model.transcribe([wav])`` (greedy + beam
    shapes) returns a tuple of two parallel lists:
    ``(['greedy text'], ['beam text'])``. The first list is the
    primary hypothesis we want; the second is the alternative the
    decoder kept around.

    Confirmed in production with a real recording — we logged the
    raw shape as ``raw type=tuple, value preview=(['Теперь все работает.'],
    ['Теперь все работает.'])`` and lost it to the previous
    extractor. Must unwrap recursively."""
    from app.backends.nemo_backend import NemoBackend

    fake_model = MagicMock()
    fake_model.transcribe = MagicMock(
        return_value=(["Теперь все работает."], ["Теперь все работает."])
    )
    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(return_value=fake_model)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    text = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "Теперь все работает."


def test_extract_text_recursively_unwraps_nested_iterables():
    """Direct unit test for the helper — covers shapes the
    transcribe-level test doesn't (single tuple, deeply nested
    list, mix of str and Hypothesis-like objects)."""
    from app.backends.nemo_backend import NemoBackend

    # Bare tuple of strings — first element wins.
    assert NemoBackend._extract_text(("hello", "world")) == "hello"
    # Nested list of lists.
    assert NemoBackend._extract_text([["nested"]]) == "nested"
    # Hypothesis-like at the top level.
    h = MagicMock(text="from-hypothesis")
    assert NemoBackend._extract_text(h) == "from-hypothesis"
    # Hypothesis nested inside a tuple+list.
    assert NemoBackend._extract_text(([h], [h])) == "from-hypothesis"
    # Empty — None.
    assert NemoBackend._extract_text(()) is None
    assert NemoBackend._extract_text([[]]) is None
    assert NemoBackend._extract_text(None) is None


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


# ---- Cancel-load (user clicked Cancel during a slow download) -------------


def test_cancel_load_returns_status_to_stopped(monkeypatch):
    """The user mis-clicked a model and needs an out. ``cancel_load``
    flips status back to ``stopped`` immediately so the watcher in
    ``StateManager`` exits and the loading pill drops away — without
    waiting for the underlying download/import to actually finish
    (Python can't safely interrupt a foreign thread)."""
    from app.backends.nemo_backend import NemoBackend

    started = threading.Event()
    finish = threading.Event()

    def slow_loader(*_args, **_kwargs):
        started.set()
        # Hold the load thread inside ``from_pretrained`` until the
        # test releases it — gives us a window where the backend is
        # in the ``loading`` state for cancel_load to act on.
        finish.wait(timeout=2.0)
        return MagicMock()

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(side_effect=slow_loader)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert started.wait(2.0)
    assert backend.status() == "loading"

    backend.cancel_load()
    assert backend.status() == "stopped"
    # Now let the slow loader finish — its result must be discarded
    # because we cancelled.
    finish.set()
    # Brief settle window so the load thread can exit.
    assert _wait(lambda: backend.status() == "stopped")
    assert backend._model is None


def test_cancel_load_is_noop_when_not_loading(monkeypatch):
    """Idempotent — calling cancel when there's nothing to cancel
    must not raise or thrash state. Important because the UI cancel
    button can race with a load that already finished."""
    from app.backends.nemo_backend import NemoBackend

    _install_fake_nemo(monkeypatch)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    # No load yet — status is "stopped".
    backend.cancel_load()
    assert backend.status() == "stopped"

    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    # Already ready — cancel must not flip status back to stopped.
    backend.cancel_load()
    assert backend.status() == "ready"


def test_load_after_cancel_resumes_normally(monkeypatch):
    """After a cancel, the user might pick the same model again;
    a fresh load() call must reset the cancel flag and proceed to
    ready, otherwise the second click would silently no-op."""
    from app.backends.nemo_backend import NemoBackend

    started = threading.Event()
    finish = threading.Event()

    def slow_loader(*_args, **_kwargs):
        if not started.is_set():
            started.set()
            finish.wait(timeout=2.0)
        return MagicMock()

    fake_class = MagicMock()
    fake_class.from_pretrained = MagicMock(side_effect=slow_loader)
    _install_fake_nemo(monkeypatch, asr_model_class=fake_class)

    backend = NemoBackend(model="nvidia/parakeet-tdt-0.6b-v3")
    backend.load()
    assert started.wait(2.0)
    backend.cancel_load()
    finish.set()
    assert _wait(lambda: backend.status() == "stopped")

    # Second load — must succeed (cancel flag reset).
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
