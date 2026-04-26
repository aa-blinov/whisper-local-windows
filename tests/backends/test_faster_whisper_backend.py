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


def test_load_passes_hf_home_as_download_root(monkeypatch, tmp_path):
    """``HF_HOME`` is the source of truth for the user-configured
    storage path. faster-whisper's ``WhisperModel`` accepts a
    ``download_root`` that's forwarded straight to
    ``huggingface_hub.snapshot_download(cache_dir=...)``. Reading the
    env var at construction time + passing it explicitly is what
    makes a runtime path change take effect on the next model load
    without restarting the process — same dynamic-path pattern we
    already use for GigaAM."""
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    monkeypatch.setenv("HF_HOME", str(tmp_path))

    captured: list = []

    def fake_whisper(*args, **kwargs):
        captured.append((args, kwargs))
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", fake_whisper)

    backend = FasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert captured, "WhisperModel was not constructed"
    _args, kwargs = captured[0]
    expected = str(tmp_path / "hub")
    assert kwargs.get("download_root") == expected, (
        f"expected download_root={expected!r}, got {kwargs.get('download_root')!r}"
    )


def test_load_omits_download_root_when_hf_home_unset(monkeypatch):
    """No ``HF_HOME`` → don't pass ``download_root`` so faster-whisper
    falls back to its / huggingface_hub's default cache path. Lets
    users who haven't customised storage keep their existing
    downloads."""
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    monkeypatch.delenv("HF_HOME", raising=False)

    captured: list = []

    def fake_whisper(*args, **kwargs):
        captured.append((args, kwargs))
        return MagicMock()

    monkeypatch.setattr("faster_whisper.WhisperModel", fake_whisper)

    backend = FasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    assert captured
    _args, kwargs = captured[0]
    assert "download_root" not in kwargs


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


# ---- Download progress (tqdm patch) ----------------------------------------


def test_tqdm_patch_fires_progress_callback_when_enabled():
    """The patch routes tqdm.update calls through our callback so the UI
    can mirror download progress."""
    from app.backends.faster_whisper_backend import (
        FasterWhisperBackend,
        _install_tqdm_progress,
    )

    _install_tqdm_progress()

    captured: list[tuple[int, int, str]] = []
    FasterWhisperBackend.set_progress_callback(
        lambda current, total, desc: captured.append((current, total, desc))
    )
    try:
        import tqdm.auto

        # Use bytes counts above the 1 MB filter threshold so the bar
        # reaches the callback.
        bar = tqdm.auto.tqdm(total=10_000_000, desc="model.bin", disable=False)
        bar.update(5_000_000)
        bar.close()
    finally:
        FasterWhisperBackend.set_progress_callback(None)

    assert any(c == 5_000_000 and t == 10_000_000 for (c, t, _) in captured), captured


def test_tqdm_patch_fires_progress_callback_even_when_disabled():
    """PyQt apps run without a TTY, so tqdm's ``disable=None`` resolves
    to ``True`` and vanilla ``update()`` becomes a no-op that never
    increments ``self.n``. The patch must still report accurate progress
    in that case — otherwise the topbar shows ``Loading model… 0%``
    forever even while bytes are arriving."""
    from app.backends.faster_whisper_backend import (
        FasterWhisperBackend,
        _install_tqdm_progress,
    )

    _install_tqdm_progress()

    captured: list[tuple[int, int, str]] = []
    FasterWhisperBackend.set_progress_callback(
        lambda current, total, desc: captured.append((current, total, desc))
    )
    try:
        import tqdm.auto

        bar = tqdm.auto.tqdm(total=10_000_000, desc="model.bin", disable=True)
        bar.update(5_000_000)
        bar.update(2_500_000)
        bar.close()
    finally:
        FasterWhisperBackend.set_progress_callback(None)

    # At least one callback must report a non-zero current — otherwise
    # the UI thinks 0 bytes have arrived.
    assert any(c > 0 for (c, _, _) in captured), captured
    # Final cumulative count should reach the bytes we fed in.
    assert any(c == 7_500_000 and t == 10_000_000 for (c, t, _) in captured), captured


def test_tqdm_patch_skips_trivial_bars_to_avoid_jumps():
    """Hugging Face creates a separate tqdm bar per file in a snapshot
    download (config.json, tokenizer.json, vocabulary.txt, model.bin,
    …). The small ones flash 0%→99% in milliseconds; if we report
    every bar, the UI sees the percentage drop back to 0% each time a
    new file starts. Skip bars whose total is below the trivial-file
    threshold so only the actual weights show up."""
    from app.backends.faster_whisper_backend import (
        FasterWhisperBackend,
        _install_tqdm_progress,
    )

    _install_tqdm_progress()

    captured: list[tuple[int, int, str]] = []
    FasterWhisperBackend.set_progress_callback(
        lambda current, total, desc: captured.append((current, total, desc))
    )
    try:
        import tqdm.auto

        # Tiny bar (5 KB) — should be ignored.
        tiny = tqdm.auto.tqdm(total=5_000, desc="config.json", disable=True)
        tiny.update(2_500)
        tiny.update(2_500)
        tiny.close()

        # Real weights bar (75 MB) — must fire.
        weights = tqdm.auto.tqdm(total=75_000_000, desc="model.bin", disable=True)
        weights.update(15_000_000)
        weights.close()
    finally:
        FasterWhisperBackend.set_progress_callback(None)

    totals = {t for (_, t, _) in captured}
    # Small bar must not have produced any callback (no 5_000 entry).
    assert 5_000 not in totals
    # Large bar must have fired.
    assert 75_000_000 in totals


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
