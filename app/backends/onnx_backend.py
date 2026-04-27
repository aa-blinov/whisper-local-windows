"""Unified ONNX inference backend for every supported ASR model family.

Uses ``onnx-asr`` (``pip install "onnx-asr[gpu,hub]"``) with ONNX-exported
weights from HuggingFace.  One backend class drives Whisper, GigaAM v3
and Parakeet TDT v3 by switching a small ``family`` parameter; ONNX
Runtime handles everything below — no NeMo, PyTorch, Lightning, or
CTranslate2 in the dependency tree.

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
``device='auto'`` (default) lets ONNX Runtime pick.  ``device='cuda'``
explicitly requests CUDA with a CPU fallback in the providers list,
plus a runtime retry-on-CPU if the first load fails with a CUDA-related
error (handles the case where ``onnxruntime-gpu`` is installed but no
NVIDIA driver is present).  ``device='cpu'`` pins to CPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

from app.backends._progress import install_tqdm_progress, set_progress_callback
from app.inference_settings import NemoInferenceSettings

log = logging.getLogger(__name__)

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

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False
        # Set to True when the user clicks Cancel during a load.  The
        # worker keeps running (Python cannot safely interrupt a thread
        # blocked on a network read or ONNX init), but its result is
        # discarded at the publish step.  Reset on each fresh ``load()``.
        self._cancel_requested = False
        self._inference_settings = NemoInferenceSettings()

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

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,  # API parity, ignored
        load_id: Optional[str] = None,
    ) -> None:
        """Switch to a different ONNX model. Triggers a background reload.

        ``load_id`` mirrors the constructor knob — pass an explicit
        loader identifier when the new model's onnx-asr id differs
        from its HF canonical (T-One, GigaAM e2e, NeMo short names).
        Defaults to ``model``.
        """
        del compute_type
        with self._lock:
            if self._shutdown:
                return
            if model == self._model_name and self._status == "ready":
                return
            self._model_name = model
            self._load_id = load_id or model
            self._model = None
            self._status = "stopped"
        self.load()

    def update_inference_settings(self, settings: NemoInferenceSettings) -> None:
        """Apply a new settings bundle.  Effect is per-call: the next
        ``transcribe`` reads ``self._inference_settings``.  No reload."""
        with self._lock:
            self._inference_settings = settings

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

    def shutdown(self) -> None:
        """Release resources; move to ``stopped`` (terminal). Idempotent."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._model = None
            self._status = "stopped"

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

    def _resolve_providers(self) -> Optional[list[str]]:
        """Map ``self._device`` to an ONNX Runtime providers list.

        ``auto`` returns None so onnx-asr / ORT pick from what's installed.
        """
        if self._device == "cuda":
            # CPU is a fallback if CUDA fails at session-create time
            # (e.g. driver missing) — ORT will use the first viable
            # provider in the list.
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self._device == "cpu":
            return ["CPUExecutionProvider"]
        return None

    def _do_load(self, model_name: str) -> None:
        install_tqdm_progress()

        log.info("Importing onnx_asr…")
        try:
            import onnx_asr  # type: ignore[import]
        except ImportError as exc:
            log.error(
                "onnx-asr is not installed: %s\n"
                "Install it with:  pip install \"onnx-asr[gpu,hub]\"",
                exc,
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
        try:
            model = onnx_asr.load_model(
                load_id, **self._build_load_kwargs(providers)
            )
        except Exception as exc:
            # Fallback path: user asked for CUDA but the CUDA provider
            # isn't actually available.  Retry with CPU only so the user
            # ends up with a working backend instead of an error pill.
            if (
                self._device == "cuda"
                and _is_cuda_provider_error(exc)
                and not self._shutdown
                and not self._cancel_requested
            ):
                log.warning(
                    "CUDA provider unavailable (%s) — retrying on CPU",
                    exc,
                )
                try:
                    model = onnx_asr.load_model(
                        load_id,
                        **self._build_load_kwargs(["CPUExecutionProvider"]),
                    )
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
            log.info("OnnxAsr model %s ready", model_name)


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


def _is_cuda_provider_error(exc: BaseException) -> bool:
    """Heuristic: does the exception text suggest a CUDA-provider issue?

    onnx_asr.load_model raises plain ``RuntimeError`` with messages like
    ``[E:onnxruntime] CUDAExecutionProvider not available …`` when the
    requested provider isn't loadable.  We use a string match because
    ORT doesn't expose a typed exception for this case.
    """
    msg = str(exc).lower()
    keywords = ("cuda", "cudaexecutionprovider", "provider", "tensorrt")
    return any(k in msg for k in keywords)


# ---- backwards-compat alias ------------------------------------------------
# A short period after the rename, ``OnnxParakeetBackend`` still gets
# imported from older revisions of model-mapping / tests.  Keeping the
# alias avoids a breaking rename.
OnnxParakeetBackend = OnnxAsrBackend
