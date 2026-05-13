"""Unified ONNX inference backend for every supported ASR model family.

Uses ``onnx-asr`` with ONNX-exported weights from HuggingFace.  One
backend class drives Whisper, GigaAM v3 and Parakeet TDT v3 by
switching a small ``family`` parameter; ONNX Runtime handles
everything below — no NeMo, PyTorch, Lightning, or CTranslate2 in
the dependency tree.

Supported families
------------------
- ``whisper``  — ``onnx-community/whisper-*``, configurable language
- ``gigaam``   — ``istupakov/gigaam-v3-onnx``, Russian only
- ``parakeet`` — ``istupakov/parakeet-tdt-0.6b-v3-onnx``, multilingual

Lifecycle
---------
    stopped ──load()──▶ loading ──ok──▶ ready
                               └──err──▶ error

    ready ──change_model()──▶ loading ──▶ ready / error
    *     ──shutdown()──▶ stopped  (terminal)

Provider selection
------------------
``device='auto'`` (default) is platform-aware: on macOS it stages
``CoreMLExecutionProvider`` (Neural Engine + GPU) ahead of CPU; on
Windows / Linux it leaves the choice to onnx-asr / ORT (which picks
CUDA when ``onnxruntime-gpu`` is installed).  ``device='cuda'`` and
``device='coreml'`` are explicit overrides — both come with a CPU
fallback baked into the providers list, plus a runtime retry on CPU
if the accelerator fails at session-create time (handles the case
where the GPU library is installed but the driver / hardware is
missing). ``device='cpu'`` pins to CPU.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Callable, Optional

import numpy as np

from app.backends._progress import install_tqdm_progress, set_progress_callback
from app.inference_settings import InferenceSettings, ParakeetInferenceSettings
from app.utils import _try_inject_nvidia_pip_dll_paths

# If the user installed NVIDIA CUDA libraries via pip (e.g.
# nvidia-cublas-cu12, nvidia-cudnn-cu12, …) but they are not on the
# system PATH, onnxruntime's C++ provider DLL cannot find them.
# We inject the pip package bin/ directories into PATH before the
# first onnxruntime import so CUDA works out of the box on Windows.
_try_inject_nvidia_pip_dll_paths()

log = logging.getLogger(__name__)

# Minimal valid ONNX model: float32[1] -> Identity -> float32[1].
# Generated once with raw protobuf wire encoding; verified with
# ``onnxruntime.InferenceSession(bytes, ...).run(...)``.
# Used as a dummy graph to probe whether an execution provider
# (CUDA, DirectML, etc.) is actually functional without loading
# a real model.
_WARMUP_ONNX_BYTES: bytes = (
    b'\x08\x08:4\n\x10\n\x01x\x12\x01y"\x08Identity'
    b'Z\x0f\n\x01x\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'b\x0f\n\x01y\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'B\x04\n\x00\x10\x0b'
)

# Cache for the Windows/Linux ``device='auto'`` provider probe so we
# only pay the dummy-session cost once per process.
_AUTO_PROVIDER_CACHE: Optional[list] = None

# 25 s @ 16 kHz.  Whisper / GigaAM / Parakeet ONNX models are trained on
# sequences up to ~30 s; 25 s gives a comfortable margin and keeps chunk
# results coherent for conversational-pace dictation.
_CHUNK_SAMPLES: int = 25 * 16_000

# Which family-specific tweaks apply.  See ``current_language`` and
# ``transcribe`` for how each is honoured.
_FAMILIES = ("whisper", "gigaam", "parakeet", "auto")


class OnnxAsrBackend:
    """ONNX Runtime inference for any onnx-asr supported model.

    Parameters
    ----------
    model
        HuggingFace repo id (or short name accepted by onnx-asr) of the
        ONNX-exported weights.
    family
        Which ASR family this is.  Controls language reporting and
        whether Whisper-only ``recognize(language=...)`` is used.
    language
        For Whisper: source language code (``"en"``, ``"ru"``, …) or
        ``"auto"`` / ``None`` for auto-detect.  Ignored for other families.
    device
        ``"auto"`` lets onnx-asr pick a provider; ``"cuda"`` requests
        CUDA with CPU fallback; ``"cpu"`` pins to CPU.
    quantization
        Passed verbatim to ``onnx_asr.load_model(quantization=…)``.
        ``"int8"`` / ``"fp16"`` / ``None`` (default).
    """

    def __init__(
        self,
        model: str,
        family: str = "auto",
        language: Optional[str] = None,
        device: str = "auto",
        quantization: Optional[str] = None,
        load_id: Optional[str] = None,
        prefer_cpu_provider: bool = False,
    ) -> None:
        if family not in _FAMILIES:
            raise ValueError(
                f"family must be one of {_FAMILIES}, got {family!r}"
            )

        self._lock = threading.Lock()
        self._model_name = model
        # ``load_id`` is what we pass to onnx_asr.load_model; it can
        # differ from ``model`` (the HF canonical we display + cache
        # against) when onnx-asr knows the model under a different
        # name.  Defaults to ``model`` for the common case.
        self._load_id = load_id or model
        self._family = family
        self._language = language
        self._device = device
        self._quantization = quantization
        # Skip the accelerator EP entirely for models known to break
        # ORT's CoreML / CUDA adapters at session-init time.  Set by
        # the registry from ``ModelInfo.prefer_cpu_provider`` — see
        # ``_resolve_providers`` for the full rationale.
        self._prefer_cpu_provider = bool(prefer_cpu_provider)

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False
        # Set to True when the user clicks Cancel during a load.  The
        # worker keeps running (Python cannot safely interrupt a thread
        # blocked on a network read or ONNX init), but its result is
        # discarded at the publish step.  Reset on each fresh ``load()``.
        self._cancel_requested = False
        # Pretty-printed name of the ONNX Runtime EP actually used by
        # the loaded model's ``InferenceSession`` (``"CUDA"``,
        # ``"CoreML"``, ``"CPU"``, …) — populated from
        # ``_detect_active_provider`` after a successful ``load_model``
        # so the UI can display the real EP rather than the requested
        # one (which lies after a CPU retry).  ``None`` until first
        # successful load.
        self._active_provider: Optional[str] = None
        self._inference_settings = ParakeetInferenceSettings()

    # ---- public API ---------------------------------------------------------

    def current_model(self) -> str:
        with self._lock:
            return self._model_name

    def current_language(self) -> Optional[str]:
        """What language code to stamp history rows with.

        - ``gigaam``  → always ``"ru"`` (Russian-only model)
        - ``parakeet`` → ``None`` (auto-detect across 25 langs)
        - ``whisper`` → user setting, or ``None`` if ``"auto"``/empty
        - ``auto``    → user setting if set, else None
        """
        if self._family == "gigaam":
            return "ru"
        if self._family == "parakeet":
            return None
        # whisper / auto
        lang = self._language
        if not lang or lang == "auto":
            return None
        return lang

    def status(self) -> str:
        with self._lock:
            return self._status

    def health_check(self) -> bool:
        return self.status() == "ready"

    def active_provider(self) -> Optional[str]:
        """Pretty-printed name of the EP actually backing the loaded
        model — ``"CUDA"`` / ``"CoreML"`` / ``"CPU"`` / ``"TensorRT"`` /
        ``"DirectML"`` / etc., or ``None`` while the model is unloaded
        / loading / errored.  This is the *real* EP after any retry-
        on-CPU fallback, not the EP we initially requested."""
        with self._lock:
            return self._active_provider

    def load(self) -> None:
        """Begin loading the model in a background thread (idempotent)."""
        with self._lock:
            if self._shutdown:
                return
            if self._status in ("loading", "ready"):
                return
            self._status = "loading"
            self._cancel_requested = False
            target_model = self._model_name

        thread = threading.Thread(
            target=self._do_load,
            args=(target_model,),
            daemon=True,
            name="onnx-asr-load",
        )
        with self._lock:
            self._load_thread = thread
        thread.start()

    def cancel_load(self) -> None:
        """Abandon an in-flight load, flipping status back to ``stopped``."""
        with self._lock:
            if self._shutdown:
                return
            if self._status != "loading":
                return
            log.info("OnnxAsr load cancelled by user")
            self._cancel_requested = True
            self._model = None
            self._status = "stopped"
            self._active_provider = None

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,  # API parity, ignored
        load_id: Optional[str] = None,
        prefer_cpu_provider: Optional[bool] = None,
    ) -> None:
        """Switch to a different ONNX model. Triggers a background reload.

        ``load_id`` mirrors the constructor knob — pass an explicit
        loader identifier when the new model's onnx-asr id differs
        from its HF canonical (T-One, GigaAM e2e, NeMo short names).
        Defaults to ``model``.

        ``prefer_cpu_provider=None`` means "leave the existing flag
        as-is" — the in-family swap path in ``RegistryBackend`` keeps
        the same ``ModelInfo.prefer_cpu_provider`` for both old and
        new model anyway (Whisper-base → Whisper-turbo: both False;
        GigaAM CTC → GigaAM RNN-T: both True), so unspecified is
        safe.  Pass an explicit value when crossing the boundary.
        """
        del compute_type
        requested_load_id = load_id or model
        with self._lock:
            if self._shutdown:
                return
            if (
                model == self._model_name
                and requested_load_id == self._load_id
                and self._status == "ready"
            ):
                return
            self._model_name = model
            self._load_id = requested_load_id
            if prefer_cpu_provider is not None:
                self._prefer_cpu_provider = bool(prefer_cpu_provider)
            self._model = None
            self._status = "stopped"
            self._active_provider = None
        self.load()

    def update_inference_settings(self, settings) -> None:
        """Apply a new settings bundle.  Effect is per-call: the next
        ``transcribe`` reads ``self._inference_settings``.  No reload.

        Accepts either ``InferenceSettings`` (Whisper card) or
        ``ParakeetInferenceSettings`` (Parakeet card).  When the
        bundle carries a ``language`` field (Whisper), update the
        live ``_language`` so the next ``transcribe`` honours the
        user's choice without an app restart — this is what wires the
        Whisper inference panel to the running model.
        """
        with self._lock:
            self._inference_settings = settings
            # Whisper's panel sends an InferenceSettings with a
            # ``language`` attribute.  Apply it live so the next
            # transcribe call passes the new value.  Other families
            # (gigaam / parakeet) ignore self._language anyway, so
            # writing it here is harmless even if the user pushed a
            # cross-shape settings object by accident.
            new_language = getattr(settings, "language", None)
            if isinstance(settings, InferenceSettings):
                # ``InferenceSettings.language`` may be ``None`` (auto)
                # or an explicit string — either way it is the new
                # source of truth.  Don't read ``getattr`` here so a
                # future field with the same name on an unrelated
                # object can't silently overwrite our state.
                self._language = new_language

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16_000
    ) -> Optional[str]:
        """Transcribe mono/stereo ``float32`` audio.

        Audio longer than ``_CHUNK_SAMPLES`` is split into 25 s chunks
        and the results are joined with a space.  Empty chunks (model
        returned None / empty) are skipped.
        """
        with self._lock:
            if self._shutdown:
                return None
            if self._status != "ready" or self._model is None:
                log.warning(
                    "transcribe() called but backend status is %r",
                    self._status,
                )
                return None
            model = self._model
            settings = self._inference_settings

        try:
            buf = np.asarray(audio, dtype=np.float32)
            if buf.ndim > 1:
                buf = buf.mean(axis=-1)
            if sample_rate != 16_000:
                buf = _resample(buf, sample_rate, 16_000)

            recognise_model = model
            if getattr(settings, "timestamps", False):
                # ``with_timestamps`` returns a wrapper that yields
                # word-level offsets.  No-op if the model doesn't have
                # the method (defensive for fakes / older onnx-asr).
                wrap = getattr(model, "with_timestamps", None)
                if callable(wrap):
                    recognise_model = wrap()

            kwargs: dict = {}
            if self._family == "whisper":
                lang = self.current_language()
                if lang is not None:
                    kwargs["language"] = lang

            if len(buf) <= _CHUNK_SAMPLES:
                return _transcribe_chunk(recognise_model, buf, kwargs)

            parts: list[str] = []
            for start in range(0, len(buf), _CHUNK_SAMPLES):
                chunk = buf[start : start + _CHUNK_SAMPLES]
                text = _transcribe_chunk(recognise_model, chunk, kwargs)
                if text:
                    parts.append(text)
            return " ".join(parts) or None
        except Exception as exc:
            log.error(
                "OnnxAsr transcription failed: %s", exc, exc_info=True
            )
            return None

    def transcribe_file(self, path: str) -> Optional[str]:
        """Transcribe an audio file from disk.

        Decodes the file via :func:`_decode_audio_file` (soundfile for
        WAV / FLAC / OGG / OPUS / AIFF — fast in-process; ffmpeg
        subprocess for everything else: MP3 / M4A / AAC / WMA / WebM
        / MOV …) and feeds the resulting numpy array to
        ``model.recognize``.

        For Whisper family the user-selected language is forwarded
        (same as the ``transcribe`` array path).  Timestamps are
        intentionally not applied here — the file path produces a
        paste-ready transcript and a side panel for word offsets
        doesn't exist yet.

        Returns ``None`` on any failure (backend not ready, decode
        failure, model error).  Errors are logged.
        """
        with self._lock:
            if self._shutdown:
                return None
            if self._status != "ready" or self._model is None:
                log.warning(
                    "transcribe_file() called but backend status is %r",
                    self._status,
                )
                return None
            model = self._model

        audio = _decode_audio_file(path)
        if audio is None:
            return None

        kwargs: dict = {}
        if self._family == "whisper":
            lang = self.current_language()
            if lang is not None:
                kwargs["language"] = lang

        try:
            return _transcribe_chunk(model, audio, kwargs)
        except Exception as exc:
            log.error(
                "OnnxAsr transcribe_file failed for %s: %s",
                path, exc, exc_info=True,
            )
            return None

    def shutdown(self) -> None:
        """Release resources; move to ``stopped`` (terminal). Idempotent."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._model = None
            self._status = "stopped"
            self._active_provider = None

    @staticmethod
    def set_progress_callback(
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        """Register the UI's HF download progress callback."""
        set_progress_callback(callback)

    # ---- internal -----------------------------------------------------------

    def _build_load_kwargs(self, providers: Optional[list[str]]) -> dict:
        kwargs: dict = {}
        if self._quantization is not None:
            kwargs["quantization"] = self._quantization
        if providers is not None:
            kwargs["providers"] = providers
        return kwargs

    def _resolve_providers(self) -> Optional[list]:
        """Map ``self._device`` to an ONNX Runtime providers list.

        Returns a list that ``onnx_asr.load_model`` accepts directly
        (``Sequence[str | tuple[str, dict]]``).  CPU is appended as a
        fallback after every accelerator entry so ORT can still build
        the session if the primary provider's session-create fails
        (driver missing, unsupported op for this model, etc.).
        Returning ``None`` lets onnx-asr / ORT pick from whatever is
        registered in the installed ``onnxruntime`` wheel.

        ``prefer_cpu_provider`` short-circuits the accelerator branches
        for models we know will fail mid-compilation:  GigaAM v3 CTC
        and RNN-T burn ~75 s in ``MLModel.compileModelAtURL`` before
        ORT raises ``HandleNegativeAxis … axis 2 is not in valid
        range`` and the retry-on-CPU path kicks in.  The wait is the
        same on Windows + CUDA for the same models (CUDA EP also
        rejects the CTC op).  Skipping straight to CPU saves the user
        a 75-second freeze on every model load — same eventual
        outcome, no wasted compilation.  Honoured for ``auto``,
        ``cuda`` and ``coreml`` selections;  an explicit ``cpu`` is
        already CPU.
        """
        if self._prefer_cpu_provider and self._device != "cpu":
            return ["CPUExecutionProvider"]
        if self._device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self._device == "coreml":
            return [_coreml_provider_entry(), "CPUExecutionProvider"]
        if self._device == "cpu":
            return ["CPUExecutionProvider"]
        # ``auto``: prefer CoreML on Apple Silicon (it is the only
        # accelerator path that actually exercises the Neural Engine
        # / GPU on macOS); on Windows / Linux let ORT pick — it will
        # use CUDA if ``onnxruntime-gpu`` is installed and a working
        # NVIDIA driver is present, otherwise CPU.
        if sys.platform == "darwin":
            return [_coreml_provider_entry(), "CPUExecutionProvider"]
        return _resolve_auto_providers_non_darwin()


    def _do_load(self, model_name: str) -> None:
        install_tqdm_progress()

        log.info("Importing onnx_asr…")
        try:
            import onnx_asr  # type: ignore[import]
        except ImportError as exc:
            extras = (
                "[gpu,hub]" if sys.platform == "win32" else "[cpu,hub]"
            )
            log.error(
                "onnx-asr is not installed: %s\n"
                "Install it with:  pip install \"onnx-asr%s\"",
                exc, extras,
            )
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._status = "error"
            return

        providers = self._resolve_providers()
        load_id = self._load_id
        log.info(
            "Loading ONNX model %s (load_id=%s, family=%s, providers=%s, quantization=%s)…",
            model_name, load_id, self._family, providers, self._quantization,
        )
        forced_provider: Optional[str] = None
        if providers == ["CPUExecutionProvider"]:
            forced_provider = "CPU"

        try:
            model = onnx_asr.load_model(
                load_id, **self._build_load_kwargs(providers)
            )
        except Exception as exc:
            # Fallback path: an accelerator provider (CUDA / CoreML)
            # was requested or auto-selected but failed at
            # session-create time — retry with CPU only so the user
            # ends up with a working backend instead of an error pill.
            #
            # We don't try to classify the exception any further:
            # ONNX Runtime surfaces accelerator-specific failures in
            # at least three different shapes — "CUDAExecutionProvider
            # not available …", CoreML's "model_builder.cc … Unable
            # to get shape for output …" (unsupported op), TensorRT
            # build errors, etc. — and a string-keyword match
            # inevitably misses one. Retrying on CPU when an
            # accelerator was the request is always safe: CPU is the
            # universal EP, so the second attempt either succeeds
            # (best outcome — model is at least usable, just slower)
            # or fails for a model-level reason that the first
            # attempt would have hit anyway.
            wants_accelerator = self._device in ("cuda", "coreml", "auto")
            if (
                wants_accelerator
                and not self._shutdown
                and not self._cancel_requested
            ):
                log.warning(
                    "Accelerator provider failed for this model (%s) — "
                    "retrying on CPU",
                    exc,
                )
                try:
                    model = onnx_asr.load_model(
                        load_id,
                        **self._build_load_kwargs(["CPUExecutionProvider"]),
                    )
                    # Mark the active provider as CPU explicitly — the
                    # detection probe below would also resolve to CPU,
                    # but stamping it here keeps the source of truth
                    # close to the retry that produced it.
                    forced_provider = "CPU"
                except Exception as cpu_exc:
                    log.error(
                        "CPU fallback also failed for %s: %s",
                        model_name, cpu_exc, exc_info=True,
                    )
                    with self._lock:
                        if (
                            not self._shutdown
                            and not self._cancel_requested
                            and self._model_name == model_name
                        ):
                            self._model = None
                            self._status = "error"
                    return
            else:
                log.error(
                    "Failed to load ONNX model %s: %s", model_name, exc,
                    exc_info=True,
                )
                with self._lock:
                    if (
                        not self._shutdown
                        and not self._cancel_requested
                        and self._model_name == model_name
                    ):
                        self._model = None
                        self._status = "error"
                return

        # Probe the loaded model for the EP its InferenceSession actually
        # bound to.  The retry-on-CPU branch already pre-stamps
        # ``forced_provider``; otherwise we sniff the model object
        # (encoder / model attribute → ``session.get_providers()[0]``)
        # so the UI shows the real EP rather than the requested one.
        detected_provider = forced_provider or _detect_active_provider(model)

        with self._lock:
            if self._shutdown:
                return
            if self._cancel_requested:
                log.info(
                    "Discarding loaded ONNX model %s — cancelled by user",
                    model_name,
                )
                return
            if self._model_name != model_name:
                log.info(
                    "Discarding loaded ONNX model %s — name changed to %s",
                    model_name, self._model_name,
                )
                return
            self._model = model
            self._status = "ready"
            self._active_provider = detected_provider
            log.info(
                "OnnxAsr model %s ready (provider=%s)",
                model_name, detected_provider or "CPU (fallback/unknown)",
            )


def _resolve_auto_providers_non_darwin() -> list:
    """Return a provider list for ``device='auto'`` on Windows / Linux.

    Probes CUDA with a dummy session; if it works we return
    ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` so the model
    loads on the GPU. If CUDA is unavailable (missing driver / DLL) we
    return ``["CPUExecutionProvider"]`` and skip the noisy TensorRT/CUDA
    fallback dance entirely.

    TensorRT is ignored for ``auto`` because it requires a separate SDK
    install and almost never works out-of-the-box.
    """
    global _AUTO_PROVIDER_CACHE
    if _AUTO_PROVIDER_CACHE is not None:
        return _AUTO_PROVIDER_CACHE

    try:
        import onnxruntime as ort
        import numpy as np
    except Exception:
        _AUTO_PROVIDER_CACHE = ["CPUExecutionProvider"]
        return _AUTO_PROVIDER_CACHE

    # Suppress ORT's default C++ logger so missing-CUDA-DLL messages
    # don't spam stderr during the probe (or during any later session
    # creation in this process).
    ort.set_default_logger_severity(4)

    available = ort.get_available_providers()

    # Never auto-pick TensorRT — it needs a separate SDK and spams
    # the console with missing-cublas errors when the DLLs aren't
    # present.
    if "CUDAExecutionProvider" not in available:
        _AUTO_PROVIDER_CACHE = ["CPUExecutionProvider"]
        return _AUTO_PROVIDER_CACHE

    opts = ort.SessionOptions()
    opts.log_severity_level = 4  # silence ORT console spam
    x = np.array([0.0], dtype=np.float32)

    try:
        sess = ort.InferenceSession(
            _WARMUP_ONNX_BYTES,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        sess.run(None, {"x": x})
        # ORT silently falls back to CPU when the CUDA DLLs are
        # missing; session creation succeeds but get_providers()
        # reveals the real EP.  Only cache CUDA when the session
        # actually bound to it.
        if sess.get_providers() and sess.get_providers()[0] == "CUDAExecutionProvider":
            _AUTO_PROVIDER_CACHE = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return _AUTO_PROVIDER_CACHE
        # GPU driver says CUDA is available, but the runtime DLLs
        # (cublasLt64_12.dll, cudnn64_9.dll, …) are not on PATH.
        # Log a one-time pointer so the user knows how to enable GPU.
        _warn_missing_cuda_redist()
        _AUTO_PROVIDER_CACHE = ["CPUExecutionProvider"]
        return _AUTO_PROVIDER_CACHE
    except Exception:
        _warn_missing_cuda_redist()
        _AUTO_PROVIDER_CACHE = ["CPUExecutionProvider"]
        return _AUTO_PROVIDER_CACHE


_CUDA_REDIST_WARNED: bool = False


def _warn_missing_cuda_redist() -> None:
    """Log a one-time hint when a GPU is present but CUDA runtime DLLs
    are missing so the model falls back to CPU."""
    global _CUDA_REDIST_WARNED
    if _CUDA_REDIST_WARNED:
        return
    _CUDA_REDIST_WARNED = True
    log.warning(
        "NVIDIA GPU detected, but CUDA runtime libraries are missing "
        "(cublasLt64_12.dll, cudnn64_9.dll, …).  "
        "The model will run on CPU.  "
        "To enable GPU acceleration, install the pip CUDA packages:\n"
        "    uv pip install --extra cuda\n"
        "or on non-uv workflows:\n"
        "    pip install \"lazy-to-text[cuda]\""
    )


# ---- module-level helpers (testable without a backend instance) ------------


def _transcribe_chunk(model, audio: np.ndarray, kwargs: dict) -> Optional[str]:
    """Run ``model.recognize(audio, **kwargs)`` and extract text.

    Handles both shapes onnx-asr returns:
    - bare ``"string"``
    - object with ``.text`` attribute (an ``OnnxAsrResult``)
    """
    result = model.recognize(audio, **kwargs)
    if result is None:
        return None
    if isinstance(result, str):
        text = result
    else:
        text_attr = getattr(result, "text", None)
        if isinstance(text_attr, str):
            text = text_attr
        else:
            text = str(result)
    return text.strip() or None


def _decode_audio_file(path: str) -> Optional[np.ndarray]:
    """Decode an audio file to a 16 kHz mono ``float32`` array.

    Two-tier decoder:

    1. **soundfile** (libsndfile) — handles WAV / FLAC / OGG / OPUS /
       AIFF in-process.  Fast, no subprocess fork.  Used as the fast
       path for the formats it understands.
    2. **ffmpeg** — bundled via ``imageio-ffmpeg`` (a static binary,
       ~70 MB on Windows).  Used as a fallback for everything else:
       MP3 / M4A / AAC / WMA / WebM / MOV / FLV / 3GP / Matroska /
       any other container ffmpeg can demultiplex.

    Returns ``None`` on any failure with a useful log line so the
    user sees ``cannot decode`` rather than a stack trace.
    """
    sf_audio = _try_decode_with_soundfile(path)
    if sf_audio is not None:
        return sf_audio
    return _try_decode_with_ffmpeg(path)


def _try_decode_with_soundfile(path: str) -> Optional[np.ndarray]:
    try:
        import soundfile as sf  # type: ignore[import]
    except ImportError:
        return None
    try:
        buf, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        log.debug("soundfile cannot decode %s: %s — trying ffmpeg", path, exc)
        return None
    buf = np.asarray(buf, dtype=np.float32)
    if buf.ndim > 1:
        buf = buf.mean(axis=-1)
    if sample_rate != 16_000:
        buf = _resample(buf, int(sample_rate), 16_000)
    return buf


def _try_decode_with_ffmpeg(path: str) -> Optional[np.ndarray]:
    try:
        import imageio_ffmpeg  # type: ignore[import]
    except ImportError as exc:
        log.error(
            "Cannot decode %s — soundfile rejected the format and "
            "imageio-ffmpeg is not installed (would handle MP3 / M4A / "
            "WMA / WebM etc.).  Install with: pip install imageio-ffmpeg.  "
            "Or convert the file to .wav / .flac / .ogg manually.  "
            "Original ImportError: %s",
            path, exc,
        )
        return None

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        log.error(
            "imageio-ffmpeg failed to locate its bundled binary: %s", exc,
        )
        return None

    cmd = [
        ffmpeg_exe,
        "-nostdin",                 # don't try to read keyboard input
        "-i", path,
        "-f", "f32le",              # raw 32-bit-float little-endian
        "-acodec", "pcm_f32le",
        "-ar", "16000",             # resample to 16 kHz
        "-ac", "1",                 # downmix to mono
        "-loglevel", "error",       # silence the banner / progress noise
        "pipe:1",
    ]
    # Hide the ffmpeg console window on Windows — without this, every
    # decode flashes a black cmd window in the user's face for a few
    # seconds.  No-op on POSIX (the flag doesn't exist there).
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,            # 10 min cap for very long files
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out decoding %s (>10 min)", path)
        return None
    except Exception as exc:
        log.error("ffmpeg subprocess failed for %s: %s", path, exc)
        return None

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        log.error(
            "ffmpeg returned %d decoding %s: %s",
            proc.returncode, path, stderr[:500],
        )
        return None

    if not proc.stdout:
        log.error("ffmpeg produced no audio output for %s", path)
        return None

    # ``np.frombuffer`` gives a read-only view onto the bytes buffer;
    # copy so downstream chunking can index/slice freely.
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        return None
    return audio


def _resample(
    audio: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Resample ``audio`` from ``src_rate`` to ``dst_rate``.

    Prefers ``scipy.signal.resample_poly``; falls back to
    ``numpy.interp`` if scipy isn't available.  The fallback is fine
    for speech-band audio at common ratios (44100 → 16000 etc.).
    """
    if src_rate == dst_rate:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore[import]

        g = gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(
            np.float32
        )
    except ImportError:
        n_samples = int(len(audio) * dst_rate / src_rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)


def _coreml_provider_entry() -> tuple[str, dict]:
    """CoreML provider entry tuned for Apple Silicon.

    Provider options
    ~~~~~~~~~~~~~~~~
    - ``ModelFormat='MLProgram'`` — the modern Core ML 5 container
      (macOS 12+).  ORT 1.18+ defaults to it but we pin explicitly
      so a future ORT release that flips the default back to the
      legacy ``NeuralNetwork`` format doesn't change our behaviour
      silently.
    - ``MLComputeUnits='ALL'`` — let CoreML's dispatcher route ops
      between Neural Engine, GPU and CPU per-op.  Apple's
      recommendation for mixed-workload models like ASR.
    - ``RequireStaticInputShapes='0'`` / ``EnableOnSubgraphs='0'``
      — current ORT defaults, pinned defensively so a future
      version change doesn't break dynamic-shape models that today
      load fine.

    Known model compatibility (empirical + ORT issue tracker as
    of ORT 1.25)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Whisper Large v3 / Turbo, Canary 1B v2 (Transformer ED) —
      load successfully on CoreML.
    - GigaAM v3 CTC / RNN-T (Conformer with static shapes) —
      expected to work; not exhaustively tested on every Mac.
    - Parakeet TDT v3 — unstable on CoreML, see ORT issue #26355
      ("CoreML execution provider fails during inference with the
      Parakeet CTC ASR model").
    - T-One (Russian Conformer-CTC) — fails at session-create with
      ``axis 2 is not in valid range [-2,1]`` (unsupported op).
    - Vosk-RU / Vosk Small (Zipformer streaming) — fails at
      session-create with ``Unable to get shape for output:
      tmp_9`` (CoreML EP can't handle streaming chunk dynamic
      shapes).

    The retry-on-CPU branch in :meth:`OnnxAsrBackend._do_load`
    catches every accelerator failure and silently re-loads the
    model on CPU, so the failure modes above surface to the user
    only as a one-line "Accelerator provider failed … retrying on
    CPU" warning in the Logs view; the Engine pill ends up
    reflecting the real EP (CPU).
    """
    return (
        "CoreMLExecutionProvider",
        {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0",
            "EnableOnSubgraphs": "0",
        },
    )


# Maps ORT's verbose provider names to short labels for the UI pill.
_PROVIDER_PRETTY = {
    "CUDAExecutionProvider": "CUDA",
    "CoreMLExecutionProvider": "CoreML",
    "CPUExecutionProvider": "CPU",
    "TensorrtExecutionProvider": "TensorRT",
    "DmlExecutionProvider": "DirectML",
    "AzureExecutionProvider": "Azure",
    "ROCMExecutionProvider": "ROCm",
}


def _pretty_provider(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _PROVIDER_PRETTY.get(raw, raw.replace("ExecutionProvider", ""))


def _detect_active_provider(model) -> Optional[str]:
    """Return the EP name actually backing ``model``'s InferenceSession."""
    if model is None:
        return None

    # Log internal structure to find the session in wrapped adapters
    log.info("Probing model of type %s. dir() contents: %s", type(model), dir(model))

    # Handle onnx-asr adapters (e.g. TextResultsAsrAdapter) that wrap the real model.
    # These often have a .model attribute containing the actual ASR object.
    if hasattr(model, "model") and not hasattr(model, "get_providers"):
        log.info("Unwrapping model adapter: %s", type(model))
        model = model.model

    # Probe order: common attribute names used by onnx-asr
    probe_attrs = (
        "encoder", "_encoder",
        "model", "_model",
        "decoder", "_decoder",
        "session", "_session",
        "inference_session"
    )

    # 1. Check if the model itself is the session
    if hasattr(model, "get_providers"):
        try:
            providers = model.get_providers()
            if providers:
                res = _pretty_provider(providers[0])
                log.info("Found provider on model root: %s", res)
                return res
        except Exception:
            pass

    # 2. Check attributes
    for attr in probe_attrs:
        sess = getattr(model, attr, None)
        if sess is None:
            continue

        log.info("Probing attribute '%s' for session...", attr)

        # Some onnx-asr models wrap the session in another object
        # that has a 'session' attribute.
        if not hasattr(sess, "get_providers") and hasattr(sess, "session"):
            log.info("Attribute '%s' is a wrapper, using .session", attr)
            sess = sess.session

        get_providers = getattr(sess, "get_providers", None)
        if not callable(get_providers):
            continue
        try:
            providers = get_providers()
            if providers:
                res = _pretty_provider(providers[0])
                log.info("Found provider on attribute '%s': %s", attr, res)
                return res
        except Exception as e:
            log.info("Failed to get providers from '%s': %s", attr, e)
            continue

    log.info("No session/provider found in model %s", type(model))
    return None

