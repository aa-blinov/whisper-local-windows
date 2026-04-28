"""Tests for the RegistryBackend façade.

Now that the app is ONNX-only, RegistryBackend just builds an
``OnnxAsrBackend`` and forwards every method to it — but it still
encapsulates registry lookups so callers don't have to.

These tests use a fake ``_build_onnx_asr`` so the suite never imports
``onnx_asr`` for real.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


class _FakeBackend:
    """Records every method call so tests can assert delegation."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._family = kwargs.get("onnx_family", "auto")
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

    def change_model(
        self, model: str,
        compute_type: Optional[str] = None,
        load_id: Optional[str] = None,
    ) -> None:
        self.changed_to.append((model, compute_type))
        self.model = model
        if load_id is not None:
            self.kwargs["load_id"] = load_id

    def current_model(self) -> str:
        return self.model

    def current_language(self) -> Optional[str]:
        return None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000):
        self.transcribe_calls.append((audio, sample_rate))
        return f"transcribed-by-onnx-{self._family}"

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._status = "stopped"

    def set_progress_callback(self, callback) -> None:
        self._progress_cb = callback

    def cancel_load(self) -> None:
        self.cancel_load_called = True
        self._status = "stopped"


@pytest.fixture
def patch_builder(monkeypatch):
    """Replace the inner-backend builder with a recording fake."""
    from app.backends import registry_backend as mod

    created: list[_FakeBackend] = []

    def build(model, **kwargs):
        backend = _FakeBackend(model, **kwargs)
        created.append(backend)
        return backend

    monkeypatch.setattr(mod, "_build_onnx_asr", build)
    return created


@pytest.fixture
def patch_registry(monkeypatch):
    """Replace ``get_model`` with a tiny fake registry covering every
    onnx_family."""
    from app.backends import registry_backend as mod
    from app.model_mapping import ModelInfo

    registry = {
        "whisper-model": ModelInfo(
            alias="whisper-model",
            canonical="onnx-community/whisper-base",
            display_name="Whisper",
            size_mb=145,
            vram_gb=1.0,
            speed="fast",
            quality="good",
            languages="multilingual",
            description="x",
            compute_type="float16",
            backend_kind="onnx_asr",
            family="Whisper",
            onnx_family="whisper",
        ),
        "gigaam-model": ModelInfo(
            alias="gigaam-model",
            canonical="istupakov/gigaam-v3-onnx",
            display_name="GigaAM",
            size_mb=290,
            vram_gb=2.5,
            speed="medium",
            quality="excellent",
            languages="Russian (only)",
            description="x",
            compute_type="float16",
            backend_kind="onnx_asr",
            family="GigaAM",
            onnx_family="gigaam",
        ),
        "parakeet-model": ModelInfo(
            alias="parakeet-model",
            canonical="istupakov/parakeet-tdt-0.6b-v3-onnx",
            display_name="Parakeet",
            size_mb=1200,
            vram_gb=2.0,
            speed="fast",
            quality="excellent",
            languages="multilingual",
            description="x",
            compute_type="float32",
            backend_kind="onnx_asr",
            family="Parakeet",
            onnx_family="parakeet",
        ),
    }

    def fake_get_model(alias: str):
        if alias not in registry:
            raise KeyError(alias)
        return registry[alias]

    monkeypatch.setattr(mod, "get_model", fake_get_model)
    return registry


# ---- Initial backend selection --------------------------------------------


def test_initial_constructs_one_inner_backend(patch_builder, patch_registry):
    """``RegistryBackend.__init__`` resolves the alias and builds
    exactly one ``OnnxAsrBackend`` — no eager rebuilds, no double
    instantiation from a stray ``change_model`` call inside __init__."""
    from app.backends.registry_backend import RegistryBackend

    RegistryBackend(model="whisper-model")
    assert len(patch_builder) == 1


def test_initial_uses_canonical_from_registry(patch_builder, patch_registry):
    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="parakeet-model")
    assert backend.current_model() == "istupakov/parakeet-tdt-0.6b-v3-onnx"
    assert patch_builder[0].kwargs.get("onnx_family") == "parakeet"


def test_unknown_model_falls_back_with_auto_family(patch_builder, patch_registry):
    """Unknown model id should still build a backend (with onnx_family='auto')
    so a power user pasting a bare HF id keeps working."""
    from app.backends.registry_backend import RegistryBackend

    RegistryBackend(model="some-bare-hf-id/repo")
    assert patch_builder[0].kwargs.get("onnx_family") == "auto"
    assert patch_builder[0].model == "some-bare-hf-id/repo"


# ---- Delegation ------------------------------------------------------------


def test_status_load_transcribe_shutdown_delegate(patch_builder, patch_registry):
    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    inner = patch_builder[0]

    backend.load()
    assert inner.load_called

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert len(inner.transcribe_calls) == 1
    assert backend.status() == "ready"

    backend.shutdown()
    assert inner.shutdown_called


# ---- Same-family change ----------------------------------------------------


def test_change_model_same_family_delegates_without_rebuild(
    patch_builder, patch_registry,
):
    """When the new model has the same onnx_family, just forward
    ``change_model`` to the existing inner — no rebuild."""
    from app.backends.registry_backend import RegistryBackend
    from app.model_mapping import ModelInfo

    # Add a second whisper model so we can swap within the family.
    patch_registry["whisper-other"] = ModelInfo(
        alias="whisper-other",
        canonical="onnx-community/whisper-large-v3-turbo",
        display_name="Whisper Turbo",
        size_mb=1620, vram_gb=4.0,
        speed="fast", quality="excellent",
        languages="multilingual", description="x",
        compute_type="float16",
        backend_kind="onnx_asr",
        family="Whisper Turbo",
        onnx_family="whisper",
    )

    backend = RegistryBackend(model="whisper-model")
    inner = patch_builder[0]

    backend.change_model("whisper-other", compute_type="int8")

    assert len(patch_builder) == 1, "no new backend should be created"
    assert inner.changed_to == [("onnx-community/whisper-large-v3-turbo", "int8")]


# ---- Cross-family change ---------------------------------------------------


def test_change_model_across_families_rebuilds_backend(
    patch_builder, patch_registry,
):
    """Switching from Whisper to GigaAM (different onnx_family) must
    shut down the old inner and build a new one with the right family."""
    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    whisper_inner = patch_builder[0]

    backend.change_model("gigaam-model")

    assert whisper_inner.shutdown_called
    assert len(patch_builder) == 2
    assert patch_builder[1].kwargs.get("onnx_family") == "gigaam"
    assert backend.current_model() == "istupakov/gigaam-v3-onnx"


def test_progress_callback_forwarded_to_new_backend(patch_builder, patch_registry):
    """A progress callback registered on the routed backend should be
    re-applied to a freshly-built inner so HF download bars from the
    new model still feed the UI."""
    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    callback = MagicMock()
    backend.set_progress_callback(callback)

    backend.change_model("gigaam-model")

    new_inner = patch_builder[1]
    assert new_inner._progress_cb is callback


# ---- Cancel-load forwarding -----------------------------------------------


def test_cancel_load_forwards_to_inner(patch_builder, patch_registry):
    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    inner = patch_builder[0]
    inner._status = "loading"

    backend.cancel_load()
    assert inner.cancel_load_called is True


def test_cancel_load_silent_when_inner_lacks_method(patch_builder, patch_registry):
    """Older fakes / partial test stubs might not implement ``cancel_load``
    — must not raise."""
    from app.backends.registry_backend import RegistryBackend

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

    backend = RegistryBackend(model="whisper-model")
    backend._inner = _NoCancelBackend()  # type: ignore[assignment]
    backend.cancel_load()  # must not raise


# ---- compute_type → quantization mapping -----------------------------------


def test_compute_type_int8_maps_to_quantization_int8(patch_builder, patch_registry):
    """Legacy ``compute_type='int8'`` must map to onnx-asr's
    ``quantization='int8'``."""
    from app.backends.registry_backend import RegistryBackend

    RegistryBackend(model="whisper-model", compute_type="int8")
    assert patch_builder[0].kwargs.get("quantization") == "int8"


def test_compute_type_float16_maps_to_no_quantization(
    patch_builder, patch_registry,
):
    from app.backends.registry_backend import RegistryBackend

    RegistryBackend(model="whisper-model", compute_type="float16")
    assert patch_builder[0].kwargs.get("quantization") is None


def test_compute_type_int8_float16_maps_to_int8_quantization(
    patch_builder, patch_registry,
):
    """Legacy CTranslate2 mode ``int8_float16`` collapses to plain
    ``int8`` for onnx-asr (closest valid match)."""
    from app.backends.registry_backend import RegistryBackend

    RegistryBackend(model="whisper-model", compute_type="int8_float16")
    assert patch_builder[0].kwargs.get("quantization") == "int8"


# ---- Cross-family async shutdown -------------------------------------------


def test_change_model_cross_family_does_not_block_caller_on_shutdown(
    patch_builder, patch_registry,
):
    """``change_model`` across families must not wait for the old
    backend's ``shutdown()`` to finish before returning.

    Why: ONNX session destruction is bounded by GIL + DLL
    contention and on Windows can stall the calling thread for
    several hundred milliseconds.  When the caller is the Qt
    main thread (model card "Switch" click), that stall paints
    the window as "(Not responding)".  We defer the shutdown to
    a worker thread and return immediately.
    """
    import threading
    import time

    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    whisper_inner = patch_builder[0]

    # Make the old backend's shutdown intentionally slow.
    shutdown_started = threading.Event()
    release = threading.Event()

    def slow_shutdown() -> None:
        shutdown_started.set()
        release.wait(5)
        whisper_inner._status = "stopped"

    whisper_inner.shutdown = slow_shutdown  # type: ignore[assignment]

    t0 = time.monotonic()
    backend.change_model("gigaam-model")
    elapsed = time.monotonic() - t0
    try:
        assert elapsed < 0.5, (
            f"change_model blocked for {elapsed:.2f}s — old-backend "
            f"shutdown must run on a worker thread"
        )
        # Sanity: the shutdown was actually started in the background
        # (just not waited on).
        assert shutdown_started.wait(2.0), (
            "shutdown worker did not start"
        )
        # New backend was already built and the swap is complete.
        assert len(patch_builder) == 2
        assert backend.current_model() == "istupakov/gigaam-v3-onnx"
    finally:
        release.set()


def test_change_model_cross_family_eventually_calls_old_shutdown(
    patch_builder, patch_registry,
):
    """The shutdown must still happen — just on a worker thread.
    Otherwise the old ONNX session leaks GPU memory + file handles."""
    import time

    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    whisper_inner = patch_builder[0]

    backend.change_model("gigaam-model")

    # Give the worker up to 3 s to finish.  In practice it returns in
    # a few ms because _FakeBackend.shutdown is trivial; the cap is
    # only a CI safety net.
    deadline = time.monotonic() + 3.0
    while not whisper_inner.shutdown_called and time.monotonic() < deadline:
        time.sleep(0.01)
    assert whisper_inner.shutdown_called, (
        "old backend's shutdown was never called — leak"
    )


def test_change_model_cross_family_swallows_old_shutdown_error(
    patch_builder, patch_registry,
):
    """If the old backend's shutdown raises (corrupted ONNX session,
    flaky DLL), the worker thread must catch and log it — never
    propagate to the caller (who has already moved on)."""
    import time

    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    whisper_inner = patch_builder[0]

    def boom() -> None:
        raise RuntimeError("simulated ONNX session destructor crash")

    whisper_inner.shutdown = boom  # type: ignore[assignment]

    # Must not raise on the caller thread.
    backend.change_model("gigaam-model")

    # Give the worker a moment to run the failing shutdown so any
    # uncaught exception would surface in the test runner's thread
    # exception handler.
    time.sleep(0.1)

    assert backend.current_model() == "istupakov/gigaam-v3-onnx"


def test_change_model_cross_family_uses_named_worker_thread(
    patch_builder, patch_registry,
):
    """The shutdown thread must have a recognisable name so it shows
    up in psutil / Logs view as a known background task — not as
    ``Thread-N`` that nobody can attribute when debugging hangs."""
    import threading
    import time

    from app.backends.registry_backend import RegistryBackend

    backend = RegistryBackend(model="whisper-model")
    whisper_inner = patch_builder[0]

    captured_name: dict[str, str] = {}
    barrier = threading.Event()

    def capture_shutdown() -> None:
        captured_name["name"] = threading.current_thread().name
        captured_name["daemon"] = (
            "yes" if threading.current_thread().daemon else "no"
        )
        barrier.set()

    whisper_inner.shutdown = capture_shutdown  # type: ignore[assignment]

    backend.change_model("gigaam-model")

    assert barrier.wait(3.0), "shutdown worker did not start"
    # Wait a beat for current_thread().name to be readable (it's set
    # in the worker before we capture it, so this is always already
    # done at this point — but be explicit).
    time.sleep(0.01)

    assert "shutdown" in captured_name["name"].lower(), (
        f"expected 'shutdown' in worker thread name, "
        f"got {captured_name['name']!r}"
    )
    assert captured_name["daemon"] == "yes", (
        "shutdown worker must be a daemon — interpreter must be allowed "
        "to exit even if the worker is mid-tear-down"
    )
