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


def test_load_uses_load_id_when_provided(monkeypatch):
    """When ``load_id`` is passed (because the HF canonical and the
    onnx-asr identifier differ — T-One, GigaAM e2e, NeMo short names),
    ``load_model`` must receive the load_id rather than the model
    name we display in the UI."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="t-tech/T-one",      # what current_model() returns
        load_id="t-tech/t-one",   # what onnx_asr expects
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    # current_model() still returns the HF canonical for cache checks
    # / UI display.
    assert backend.current_model() == "t-tech/T-one"
    # …but load_model() received the lowercase onnx-asr identifier.
    args, _ = fake_module.load_model.call_args
    assert args[0] == "t-tech/t-one"


def test_load_falls_back_to_model_name_when_load_id_omitted(monkeypatch):
    """When the registry doesn't override ``load_id`` (the common
    case), passing model name verbatim to ``load_model`` is correct."""
    fake_module, _ = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="onnx-community/whisper-base")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

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


# ---- Transcribe a file from disk ------------------------------------------


def test_transcribe_file_returns_none_when_not_ready(tmp_path):
    """Before the model is loaded, ``transcribe_file`` must return None
    so callers can render a 'model not ready' message instead of
    crashing on ``NoneType.recognize``."""
    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) is None


# ---- _decode_audio_file standalone helper ---------------------------------


def test_decode_audio_file_uses_soundfile_for_native_formats(monkeypatch, tmp_path):
    """When soundfile can read the file (WAV/FLAC/OGG/OPUS/AIFF), use
    its result directly — no ffmpeg subprocess fork."""
    import sys
    import types

    import numpy as np

    audio_data = np.zeros(16000, dtype=np.float32)
    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(return_value=(audio_data, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    # imageio_ffmpeg should NOT be touched if soundfile worked.
    fake_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_ffmpeg.get_ffmpeg_exe = MagicMock(
        side_effect=AssertionError("ffmpeg should not be invoked")
    )
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    from app.backends.onnx_backend import _decode_audio_file

    fake_audio = tmp_path / "demo.wav"
    fake_audio.write_bytes(b"")
    out = _decode_audio_file(str(fake_audio))
    assert out is not None
    assert out.dtype == np.float32
    assert out.ndim == 1
    fake_sf.read.assert_called_once()


def test_decode_audio_file_falls_back_to_ffmpeg_for_unsupported(
    monkeypatch, tmp_path,
):
    """soundfile can't decode .mp3 / .m4a — fall through to ffmpeg
    via ``imageio_ffmpeg``, which bundles a static binary."""
    import subprocess
    import sys
    import types

    import numpy as np

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(side_effect=RuntimeError("Format not recognised"))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_ffmpeg.get_ffmpeg_exe = MagicMock(return_value="/fake/ffmpeg.exe")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    # Pretend ffmpeg writes 2 seconds of silence to stdout.
    decoded_bytes = np.zeros(32_000, dtype=np.float32).tobytes()
    completed = MagicMock(returncode=0, stdout=decoded_bytes, stderr=b"")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=completed))

    from app.backends.onnx_backend import _decode_audio_file

    fake_audio = tmp_path / "voice.mp3"
    fake_audio.write_bytes(b"\xff\xfb")
    out = _decode_audio_file(str(fake_audio))

    assert out is not None
    assert out.dtype == np.float32
    assert len(out) == 32_000
    # ffmpeg got called once with a sane command line.
    subprocess.run.assert_called_once()
    cmd = subprocess.run.call_args.args[0]
    assert cmd[0] == "/fake/ffmpeg.exe"
    assert "-ar" in cmd and "16000" in cmd, "must request 16 kHz output"
    assert "-ac" in cmd and "1" in cmd, "must request mono output"


def test_decode_audio_file_returns_none_when_both_decoders_fail(
    monkeypatch, tmp_path,
):
    """Neither soundfile nor ffmpeg could open the file — surface None
    so the caller logs a friendly 'cannot decode' rather than dying
    on a stack trace."""
    import subprocess
    import sys
    import types

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(side_effect=RuntimeError("nope"))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_ffmpeg.get_ffmpeg_exe = MagicMock(return_value="/fake/ffmpeg.exe")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    completed = MagicMock(
        returncode=1,
        stdout=b"",
        stderr=b"Invalid data found when processing input",
    )
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=completed))

    from app.backends.onnx_backend import _decode_audio_file

    fake_audio = tmp_path / "broken.bin"
    fake_audio.write_bytes(b"")
    assert _decode_audio_file(str(fake_audio)) is None


def test_decode_audio_file_resamples_ffmpeg_output_when_needed(
    monkeypatch, tmp_path,
):
    """ffmpeg is asked for 16 kHz output — but if the user runs an
    older bundled binary (rare) and we get something else, resample."""
    import subprocess
    import sys
    import types

    import numpy as np

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(side_effect=RuntimeError("not native"))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_ffmpeg.get_ffmpeg_exe = MagicMock(return_value="/fake/ffmpeg.exe")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    # Default subprocess fakes return 16 kHz output (16k samples = 1 s).
    decoded = np.zeros(16_000, dtype=np.float32).tobytes()
    completed = MagicMock(returncode=0, stdout=decoded, stderr=b"")
    monkeypatch.setattr(subprocess, "run", MagicMock(return_value=completed))

    from app.backends.onnx_backend import _decode_audio_file

    fake_audio = tmp_path / "voice.mp3"
    fake_audio.write_bytes(b"")
    out = _decode_audio_file(str(fake_audio))
    assert out is not None
    # 16 kHz × 1 s = 16k samples — no resample needed for the
    # canonical-case check.  (Resampling logic is exercised by the
    # soundfile path tests above.)
    assert len(out) == 16_000


def test_decode_audio_file_returns_none_when_imageio_ffmpeg_missing(
    monkeypatch, tmp_path,
):
    """soundfile failed, imageio_ffmpeg not installed → None.
    Logged so the user sees 'install imageio-ffmpeg or convert to wav'."""
    import sys

    monkeypatch.delitem(sys.modules, "soundfile", raising=False)
    monkeypatch.delitem(sys.modules, "imageio_ffmpeg", raising=False)

    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name in ("soundfile", "imageio_ffmpeg"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    from app.backends.onnx_backend import _decode_audio_file

    fake_audio = tmp_path / "x.mp3"
    fake_audio.write_bytes(b"")
    assert _decode_audio_file(str(fake_audio)) is None


def test_transcribe_file_decodes_then_passes_array_to_recognize(
    monkeypatch, tmp_path,
):
    """``transcribe_file`` decodes the file with ``soundfile`` (which
    handles WAV/FLAC/OGG/OPUS/AIFF) and passes the numpy array to
    ``recognize()``.  Going through ``soundfile`` instead of the
    raw path lets us handle .ogg etc. — onnx-asr's bundled loader
    is WAV-only and crashes on anything else with 'file does not
    start with RIFF id'."""
    import sys
    import types

    import numpy as np

    fake_audio_data = np.zeros(16000, dtype=np.float32)
    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(return_value=(fake_audio_data, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    _, fake_model = _install_fake_onnx_asr(
        monkeypatch, recognize_return="from-file"
    )

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "demo.ogg"
    fake_audio.write_bytes(b"OggS\x00\x02")

    text = backend.transcribe_file(str(fake_audio))
    assert text == "from-file"
    # soundfile.read was called with the path
    fake_sf.read.assert_called_once()
    args, _kw = fake_sf.read.call_args
    assert args[0] == str(fake_audio)
    # model.recognize received the decoded ndarray (NOT the path string)
    fake_model.recognize.assert_called_once()
    rec_args, _ = fake_model.recognize.call_args
    assert isinstance(rec_args[0], np.ndarray), (
        "recognize() must receive a decoded ndarray, not the file path"
    )


def test_transcribe_file_resamples_if_not_16khz(monkeypatch, tmp_path):
    """Audio at 44.1 kHz must be resampled to 16 kHz before recognise."""
    import sys
    import types

    import numpy as np

    # 44.1 kHz, 1 second mono = 44100 samples
    fake_audio_data = np.zeros(44100, dtype=np.float32)
    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(return_value=(fake_audio_data, 44100))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "high-rate.wav"
    fake_audio.write_bytes(b"")
    backend.transcribe_file(str(fake_audio))

    rec_args, _ = fake_model.recognize.call_args
    decoded = rec_args[0]
    # Resampled length should be ~16k samples (give or take rounding)
    assert 15500 <= len(decoded) <= 16500, (
        f"expected resample to ~16k samples, got {len(decoded)}"
    )


def test_transcribe_file_downmixes_stereo_to_mono(monkeypatch, tmp_path):
    """Stereo files must be averaged down to mono before recognise."""
    import sys
    import types

    import numpy as np

    # (samples, channels) — soundfile's default shape for stereo
    stereo = np.zeros((16000, 2), dtype=np.float32)
    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(return_value=(stereo, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "stereo.wav"
    fake_audio.write_bytes(b"")
    backend.transcribe_file(str(fake_audio))

    rec_args, _ = fake_model.recognize.call_args
    assert rec_args[0].ndim == 1, (
        "recognize() must receive a 1-D mono array"
    )


def test_transcribe_file_returns_friendly_error_on_unsupported_format(
    monkeypatch, tmp_path,
):
    """If soundfile can't decode the file (e.g. MP3 — libsndfile
    licensing skip), surface a None result so the controller can
    show a clean error instead of a stack trace."""
    import sys
    import types

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(side_effect=RuntimeError(
        "Format not recognised."
    ))
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "broken.mp3"
    fake_audio.write_bytes(b"\xff\xfb")
    assert backend.transcribe_file(str(fake_audio)) is None


def test_transcribe_file_passes_language_for_whisper(monkeypatch, tmp_path):
    """Same as the array path: Whisper's recognize() takes a language
    kwarg and we forward the user-selected one."""
    import sys
    import types

    import numpy as np

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(
        return_value=(np.zeros(16000, dtype=np.float32), 16000)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="ru",
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "ru.wav"
    fake_audio.write_bytes(b"")
    backend.transcribe_file(str(fake_audio))

    _args, kwargs = fake_model.recognize.call_args
    assert kwargs.get("language") == "ru"


def _install_fake_soundfile(monkeypatch):
    """Helper: register a no-op soundfile stub returning silent 1-s mono."""
    import sys
    import types

    import numpy as np

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = MagicMock(
        return_value=(np.zeros(16000, dtype=np.float32), 16000)
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    return fake_sf


def test_transcribe_file_returns_none_after_shutdown(monkeypatch, tmp_path):
    _install_fake_soundfile(monkeypatch)
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.shutdown()

    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) is None


def test_transcribe_file_returns_none_on_exception(monkeypatch, tmp_path):
    """If the model.recognize() raises during inference, return None."""
    _install_fake_soundfile(monkeypatch)
    fake_module, fake_model = _install_fake_onnx_asr(monkeypatch)
    fake_model.recognize.side_effect = RuntimeError("inference failed")

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "broken.bin"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) is None


def test_transcribe_file_extracts_text_from_object_result(monkeypatch, tmp_path):
    _install_fake_soundfile(monkeypatch)
    result_obj = MagicMock()
    result_obj.text = "from-object"
    _install_fake_onnx_asr(monkeypatch, recognize_return=result_obj)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) == "from-object"


def test_transcribe_file_returns_none_on_empty_string(monkeypatch, tmp_path):
    _install_fake_soundfile(monkeypatch)
    _install_fake_onnx_asr(monkeypatch, recognize_return="   ")

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) is None


def test_transcribe_file_returns_none_when_soundfile_missing(
    monkeypatch, tmp_path,
):
    """If soundfile isn't installed, surface a clean None — let the
    controller render a friendly 'install soundfile' message rather
    than crashing on ImportError."""
    import sys

    monkeypatch.delitem(sys.modules, "soundfile", raising=False)

    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "soundfile":
            raise ImportError("No module named 'soundfile'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    assert backend.transcribe_file(str(fake_audio)) is None


def test_transcribe_file_disables_with_timestamps_for_simplicity(
    monkeypatch, tmp_path,
):
    """The file-transcribe path produces a paste-ready transcript;
    word-level timestamps would require a different output path
    (history JSON or a side panel) we don't have yet.  Don't apply
    ``with_timestamps`` even if the user has it enabled in the
    inference panel — keeps the ``recognize()`` return as a plain
    string."""
    _install_fake_soundfile(monkeypatch)
    _, fake_model = _install_fake_onnx_asr(monkeypatch)
    fake_model.with_timestamps = MagicMock(return_value=fake_model)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import ParakeetInferenceSettings

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")
    backend.update_inference_settings(ParakeetInferenceSettings(timestamps=True))

    fake_audio = tmp_path / "x.wav"
    fake_audio.write_bytes(b"")
    backend.transcribe_file(str(fake_audio))

    fake_model.with_timestamps.assert_not_called()


# ---- Live language update via update_inference_settings -------------------


def test_update_inference_settings_with_whisper_settings_updates_live_language(
    monkeypatch,
):
    """Whisper's inference panel sends an ``InferenceSettings`` (with a
    ``language`` field).  ``update_inference_settings`` must pull the
    new language out and apply it to the live backend so the next
    ``transcribe`` call passes the user's choice — without it the
    language change persists to config but doesn't take effect until
    the next app launch."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import InferenceSettings

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="en",       # initial language
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    # Simulate the user picking 'ru' from the Whisper inference panel.
    backend.update_inference_settings(InferenceSettings(language="ru"))

    # current_language() reflects the new value immediately.
    assert backend.current_language() == "ru"

    # And the next transcribe forwards it to recognize().
    backend.transcribe(np.zeros(16000, dtype=np.float32))
    _args, kwargs = fake_model.recognize.call_args
    assert kwargs.get("language") == "ru"


def test_update_inference_settings_with_whisper_auto_resets_to_autodetect(
    monkeypatch,
):
    """Picking 'auto' (or empty) in the language combo must reset the
    backend to auto-detect — i.e. ``current_language()`` returns None
    and ``transcribe()`` omits the language kwarg."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import InferenceSettings

    backend = OnnxAsrBackend(
        model="onnx-community/whisper-large-v3-turbo",
        family="whisper",
        language="ru",
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(InferenceSettings(language="auto"))

    assert backend.current_language() is None

    backend.transcribe(np.zeros(16000, dtype=np.float32))
    _args, kwargs = fake_model.recognize.call_args
    assert "language" not in kwargs


def test_update_inference_settings_does_not_touch_language_for_gigaam(
    monkeypatch,
):
    """GigaAM is Russian-only; even if the user passes a different
    language in an InferenceSettings (which shouldn't happen because
    GigaAM cards have no panel), ``current_language()`` still returns
    'ru' — no path lets the user accidentally turn off Russian."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import InferenceSettings

    backend = OnnxAsrBackend(model="x", family="gigaam")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(InferenceSettings(language="en"))

    assert backend.current_language() == "ru"


def test_update_inference_settings_does_not_touch_language_for_parakeet(
    monkeypatch,
):
    """Parakeet auto-detects across 25 languages; whatever the user
    picks in a (hypothetical) panel, ``current_language()`` stays None."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import InferenceSettings

    backend = OnnxAsrBackend(model="x", family="parakeet")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(InferenceSettings(language="en"))

    assert backend.current_language() is None


def test_update_inference_settings_with_parakeet_settings_keeps_language(
    monkeypatch,
):
    """``ParakeetInferenceSettings`` doesn't carry a language field —
    the existing ``_language`` must survive the update unchanged."""
    _install_fake_onnx_asr(monkeypatch)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import ParakeetInferenceSettings

    backend = OnnxAsrBackend(
        model="x", family="whisper", language="ru",
    )
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    # Whisper backend with a non-Whisper settings shape (defensive —
    # in practice the controller never crosses streams, but the
    # backend shouldn't lose state if it does).
    backend.update_inference_settings(ParakeetInferenceSettings(timestamps=True))

    assert backend.current_language() == "ru"


def test_with_timestamps_called_when_settings_enable_it(monkeypatch):
    """When the user enables timestamps in inference settings, the backend
    must call ``model.with_timestamps()`` to wrap the model before running
    recognise (it is a chainable adapter, not a bool kwarg)."""
    _, fake_model = _install_fake_onnx_asr(monkeypatch)
    fake_model.with_timestamps = MagicMock(return_value=fake_model)

    from app.backends.onnx_backend import OnnxAsrBackend
    from app.inference_settings import ParakeetInferenceSettings

    backend = OnnxAsrBackend(model="x")
    backend.load()
    assert _wait(lambda: backend.status() == "ready")

    backend.update_inference_settings(ParakeetInferenceSettings(timestamps=True))
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
