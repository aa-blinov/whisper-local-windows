"""In-process ONNX backend for Parakeet TDT 0.6B v3.

Uses ``onnx-asr`` (``pip install "onnx-asr[cpu,hub]"``) with the
ONNX-exported Parakeet weights on HuggingFace:
  ``istupakov/parakeet-tdt-0.6b-v3-onnx``

vs NeMo backend
---------------
+ Cold import:  ~1–2 s  (vs 30–90 s for NeMo + PyTorch + Lightning)
+ Model load:   ~3–5 s  (vs 10–15 s)
+ First infer:  no CUDA JIT warmup stall
+ Dependencies: NumPy + ONNX Runtime only (no PyTorch, Lightning, Hydra)
- Max chunk:    25 s     — longer audio is split automatically
- Language:     same 25-language Parakeet support, auto-detect

Lifecycle
---------
    stopped ──load()──▶ loading ──ok──▶ ready
                               └──err──▶ error

    ready ──change_model()──▶ loading ──▶ ready / error
    *     ──shutdown()──▶ stopped  (terminal)
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# 25 s @ 16 kHz. The onnx-asr models are trained on sequences up to
# ~30 s; 25 s gives a comfortable margin and still keeps chunk results
# coherent for conversational-pace dictation.
_CHUNK_SAMPLES: int = 25 * 16_000


class OnnxParakeetBackend:
    """ONNX Runtime inference for Parakeet TDT 0.6B v3.

    ``model`` is the HuggingFace repo id that goes into the storage/
    download path and is shown in the UI.  The ``onnx_asr.load_model``
    call receives it verbatim — ``onnx-asr`` forwards HF repo ids to
    ``huggingface_hub.snapshot_download`` automatically.
    """

    def __init__(
        self,
        model: str = "istupakov/parakeet-tdt-0.6b-v3-onnx",
        device: str = "auto",  # accepted for API parity; passed to load_model
    ) -> None:
        self._lock = threading.Lock()
        self._model_name = model
        self._device = device

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False
        # Set to True when the user clicks Cancel during a load. The
        # worker keeps running (Python cannot safely interrupt a thread
        # blocked on a network read or ONNX init), but its result is
        # discarded at the publish step. Reset on each fresh ``load()``.
        self._cancel_requested = False

    # ---- public API ---------------------------------------------------------

    def current_model(self) -> str:
        with self._lock:
            return self._model_name

    def current_language(self) -> Optional[str]:
        # Parakeet TDT v3 auto-detects among 25 languages — no knob to expose.
        return None

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
            name="onnx-parakeet-load",
        )
        with self._lock:
            self._load_thread = thread
        thread.start()

    def cancel_load(self) -> None:
        """Abandon an in-flight load.

        Flips status to ``stopped`` immediately so the state-watcher
        in ``StateManager`` exits and the loading pill drops away.
        The load worker still completes its download (Python cannot
        safely interrupt a thread blocked on I/O), but its result is
        discarded thanks to ``_cancel_requested``.

        Idempotent — calling cancel when nothing is loading is a no-op.
        """
        with self._lock:
            if self._shutdown:
                return
            if self._status != "loading":
                return
            log.info("OnnxParakeet load cancelled by user")
            self._cancel_requested = True
            self._model = None
            self._status = "stopped"

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,  # accepted for API parity, unused
    ) -> None:
        """Switch to a different ONNX model. Triggers a background reload."""
        del compute_type  # ONNX precision is baked into the weights
        with self._lock:
            if self._shutdown:
                return
            if model == self._model_name and self._status == "ready":
                return
            self._model_name = model
            self._model = None
            self._status = "stopped"
        self.load()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16_000
    ) -> Optional[str]:
        """Transcribe mono/stereo ``float32`` audio.

        Audio longer than ``_CHUNK_SAMPLES`` (25 s) is split into
        equal chunks and the results are joined with a space. Chunks
        whose ``recognize()`` call returns empty / None are silently
        skipped so a quiet section in the middle doesn't leave a gap.
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

        try:
            buf = np.asarray(audio, dtype=np.float32)
            if buf.ndim > 1:
                # Downmix multi-channel to mono
                buf = buf.mean(axis=-1)

            if sample_rate != 16_000:
                buf = _resample(buf, sample_rate, 16_000)

            if len(buf) <= _CHUNK_SAMPLES:
                return _transcribe_chunk(model, buf)

            parts: list[str] = []
            for start in range(0, len(buf), _CHUNK_SAMPLES):
                chunk = buf[start : start + _CHUNK_SAMPLES]
                text = _transcribe_chunk(model, chunk)
                if text:
                    parts.append(text)
            return " ".join(parts) or None
        except Exception as exc:
            log.error(
                "OnnxParakeet transcription failed: %s", exc, exc_info=True
            )
            return None

    def shutdown(self) -> None:
        """Release resources and move to ``stopped`` (terminal). Idempotent."""
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
        """Forward to FasterWhisperBackend's tqdm hook.

        HuggingFace downloads go through ``huggingface_hub`` which
        the existing ``_install_tqdm_progress`` shim already intercepts.
        Registering via FasterWhisperBackend keeps the mechanism in
        one place without duplicating it.
        """
        from app.backends.faster_whisper_backend import FasterWhisperBackend as _FW

        _FW.set_progress_callback(callback)

    # ---- internal -----------------------------------------------------------

    def _do_load(self, model_name: str) -> None:
        from app.backends.faster_whisper_backend import _install_tqdm_progress

        _install_tqdm_progress()

        log.info("Importing onnx_asr…")
        try:
            import onnx_asr  # type: ignore[import]
        except ImportError as exc:
            log.error(
                "onnx-asr is not installed: %s\n"
                "Install it with:  pip install \"onnx-asr[cpu,hub]\"",
                exc,
            )
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._status = "error"
            return

        log.info("onnx_asr imported, loading model %s…", model_name)
        try:
            model = onnx_asr.load_model(model_name)
        except Exception as exc:
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
            log.info("OnnxParakeet model %s ready", model_name)


# ---- module-level helpers (testable without a backend instance) ------------


def _transcribe_chunk(model, audio: np.ndarray) -> Optional[str]:
    """Run ``model.recognize()`` on one audio chunk and extract the text.

    Handles two result shapes seen in ``onnx-asr``:
    - ``"bare string"``
    - ``obj`` with a ``.text`` attribute (e.g. an ``OnnxAsrResult``)
    """
    result = model.recognize(audio)
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

    Prefers ``scipy.signal.resample_poly`` (high-quality anti-aliased
    resampler); falls back to ``numpy.interp`` if scipy is not available.
    The fallback is adequate for converting common rates (e.g. 44100 →
    16000) without audible artefacts in the frequency range that matters
    for speech.
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
