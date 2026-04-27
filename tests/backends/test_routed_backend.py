"""Tests for the RoutedBackend that switches between concrete engines.

Uses a fake ``ModelInfo`` registry and fake backends so the routing
logic is verified without touching faster-whisper or gigaam at all.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


class _FakeBackend:
    """Records every method call so tests can assert delegation."""

    def __init__(self, model: str, kind: str, **kwargs):
        self.model = model
        self.kind = kind
        self.kwargs = kwargs
        self.shutdown_called = False
        self.load_called = False
        self.cancel_load_called = False
        self.changed_to: list = []
        self.transcribe_calls: list = []
        self._status = "stopped"
        self._progress_cb = None

    def status(self) -> str:
        return self._status

    def health_check(self) -> bool:
        return self._status == "ready"

    def load(self) -> None:
        self.load_called = True
        self._status = "ready"

    def change_model(self, model: str, compute_type: Optional[str] = None) -> None:
        self.changed_to.append((model, compute_type))
        self.model = model

    def current_model(self) -> str:
        return self.model

    def current_language(self) -> Optional[str]:
        return None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        self.transcribe_calls.append((audio, sample_rate))
        return f"transcribed-by-{self.kind}"

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._status = "stopped"

    def set_progress_callback(self, callback) -> None:
        self._progress_cb = callback

    def cancel_load(self) -> None:
        self.cancel_load_called = True
        self._status = "stopped"


@pytest.fixture
def patch_builders(monkeypatch):
    """Replace ``RoutedBackend``'s backend builders with fakes."""
    from app.backends import routed_backend as mod

    created: list[_FakeBackend] = []

    def build_fw(model, **kwargs):
        backend = _FakeBackend(model, "faster_whisper", **kwargs)
        created.append(backend)
        return backend

    def build_gigaam(model, **kwargs):
        backend = _FakeBackend(model, "gigaam", **kwargs)
        created.append(backend)
        return backend

    def build_nemo(model, **kwargs):
        backend = _FakeBackend(model, "nemo", **kwargs)
        created.append(backend)
        return backend

    def build_onnx_parakeet(model, **kwargs):
        backend = _FakeBackend(model, "onnx_parakeet", **kwargs)
        created.append(backend)
        return backend

    monkeypatch.setattr(mod, "_build_faster_whisper", build_fw)
    monkeypatch.setattr(mod, "_build_gigaam", build_gigaam)
    monkeypatch.setattr(mod, "_build_nemo", build_nemo)
    monkeypatch.setattr(mod, "_build_onnx_parakeet", build_onnx_parakeet)
    return created


@pytest.fixture
def patch_registry(monkeypatch):
    """Replace ``get_model`` with a tiny fake registry."""
    from app.backends import routed_backend as mod
    from app.model_mapping import ModelInfo

    registry = {
        "fw-model": ModelInfo(
            alias="fw-model",
            canonical="fake/fw-canonical",
            display_name="FW",
            size_mb=100,
            vram_gb=2.0,
            speed="fast",
            quality="basic",
            languages="multilingual",
            description="x",
            backend_kind="faster_whisper",
        ),
        "gigaam-model": ModelInfo(
            alias="gigaam-model",
            canonical="v2_ctc",
            display_name="Gigaam",
            size_mb=100,
            vram_gb=2.0,
            speed="fast",
            quality="basic",
            languages="Russian (only)",
            description="x",
            backend_kind="gigaam",
        ),
        "onnx-model": ModelInfo(
            alias="onnx-model",
            canonical="fake/onnx-weights",
            display_name="OnnxParakeet",
            size_mb=100,
            vram_gb=2.0,
            speed="fast",
            quality="excellent",
            languages="25 langs incl. Russian, Ukrainian",
            description="x",
            compute_type="float32",
            backend_kind="onnx_parakeet",
            family="Parakeet",
        ),
    }

    def fake_get_model(alias: str):
        if alias not in registry:
            raise KeyError(alias)
        return registry[alias]

    monkeypatch.setattr(mod, "get_model", fake_get_model)
    return registry


# ---- Initial backend selection --------------------------------------------


def test_initial_kind_picked_from_model(patch_builders, patch_registry):
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="gigaam-model")
    assert backend.current_kind() == "gigaam"
    assert len(patch_builders) == 1
    assert patch_builders[0].kind == "gigaam"


def test_unknown_model_falls_back_to_faster_whisper(patch_builders, patch_registry):
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="some-bare-hf-id/repo")
    assert backend.current_kind() == "faster_whisper"


# ---- Delegation ------------------------------------------------------------


def test_status_load_transcribe_shutdown_delegate(patch_builders, patch_registry):
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="fw-model")
    inner = patch_builders[0]

    backend.load()
    assert inner.load_called

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(inner.transcribe_calls) == 1
    assert backend.status() == "ready"

    backend.shutdown()
    assert inner.shutdown_called


# ---- Same-kind change ------------------------------------------------------


def test_change_model_within_same_kind_delegates(patch_builders, patch_registry):
    """When the new model uses the same backend kind, just forward
    ``change_model`` to the existing inner — no rebuild."""
    from app.backends.routed_backend import RoutedBackend

    # Add another faster_whisper model to the registry.
    from app.model_mapping import ModelInfo
    patch_registry["fw-other"] = ModelInfo(
        alias="fw-other", canonical="fake/other", display_name="Other",
        size_mb=100, vram_gb=2.0, speed="fast", quality="basic",
        languages="multilingual", description="x",
        backend_kind="faster_whisper",
    )

    backend = RoutedBackend(model="fw-model")
    inner = patch_builders[0]

    backend.change_model("fw-other", compute_type="int8_float16")

    # Still one backend created, one change_model call recorded.
    assert len(patch_builders) == 1
    assert inner.changed_to == [("fake/other", "int8_float16")]
    assert backend.current_kind() == "faster_whisper"


# ---- Cross-kind change -----------------------------------------------------


def test_change_model_across_kinds_rebuilds_backend(patch_builders, patch_registry):
    """Switching from faster-whisper to gigaam must shut down the old
    backend and stand up a fresh one of the new kind."""
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="fw-model")
    fw_inner = patch_builders[0]

    backend.change_model("gigaam-model")

    assert fw_inner.shutdown_called
    assert len(patch_builders) == 2
    assert patch_builders[1].kind == "gigaam"
    assert backend.current_kind() == "gigaam"
    assert backend.current_model() == "v2_ctc"


def test_progress_callback_forwarded_to_new_backend(patch_builders, patch_registry):
    """A progress callback registered on the routed backend should be
    re-applied to a freshly-built inner so tqdm bars from the new
    engine still feed the UI."""
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="fw-model")
    callback = MagicMock()
    backend.set_progress_callback(callback)

    backend.change_model("gigaam-model")

    new_inner = patch_builders[1]
    assert new_inner._progress_cb is callback


# ---- Cancel-load forwarding -----------------------------------------------


def test_cancel_load_forwards_to_inner(patch_builders, patch_registry):
    """The user clicked Cancel on the topbar during a load. The
    routed backend just hands the request through to whichever
    engine is currently loading."""
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="fw-model")
    inner = patch_builders[0]
    inner._status = "loading"  # simulate in-flight load

    backend.cancel_load()
    assert inner.cancel_load_called is True


def test_onnx_parakeet_kind_routes_to_onnx_builder(patch_builders, patch_registry):
    """Selecting a model with ``backend_kind='onnx_parakeet'`` must
    build an ``OnnxParakeetBackend`` via ``_build_onnx_parakeet``, not
    fall through to the faster-whisper default."""
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="onnx-model")
    assert backend.current_kind() == "onnx_parakeet"
    assert len(patch_builders) == 1
    assert patch_builders[0].kind == "onnx_parakeet"
    assert patch_builders[0].model == "fake/onnx-weights"


def test_cross_kind_switch_fw_to_onnx_parakeet(patch_builders, patch_registry):
    """Changing from faster_whisper to onnx_parakeet must shut down the
    old backend and build a new one of the correct kind."""
    from app.backends.routed_backend import RoutedBackend

    backend = RoutedBackend(model="fw-model")
    fw_inner = patch_builders[0]

    backend.change_model("onnx-model")

    assert fw_inner.shutdown_called
    assert len(patch_builders) == 2
    assert patch_builders[1].kind == "onnx_parakeet"
    assert backend.current_kind() == "onnx_parakeet"


def test_cancel_load_silent_when_inner_lacks_method(patch_builders, patch_registry):
    """Older fakes / future engines might not implement ``cancel_load``
    yet — the routed backend must not raise in that case so the
    cancel button stays harmless."""
    from app.backends.routed_backend import RoutedBackend

    class _NoCancelBackend:
        def status(self) -> str:
            return "loading"

        def health_check(self) -> bool:
            return False

        def load(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def transcribe(self, *_args, **_kwargs):
            return None

        def current_model(self) -> str:
            return "x"

        def current_language(self):
            return None

    backend = RoutedBackend(model="fw-model")
    backend._inner = _NoCancelBackend()  # type: ignore[assignment]
    backend.cancel_load()  # must not raise
