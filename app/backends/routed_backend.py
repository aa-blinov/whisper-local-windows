"""Backend façade.

The app is now ONNX-only — every model loads via ``OnnxAsrBackend``.
The original purpose of this class was to route between heterogeneous
engines (faster-whisper / GigaAM-Python / NeMo).  Now that all engines
collapsed into one (onnx-asr), the class is a thin wrapper that
preserves the public ``TranscriptionBackend`` shape while the rest of
the app catches up.

Why keep it instead of using ``OnnxAsrBackend`` directly?

- ``change_model`` looks up the new model's ``onnx_family`` from the
  registry and rebuilds the inner backend with the correct family —
  the bare backend has no way to know that.
- It centralises construction kwargs (device, language, quantization)
  so call sites that don't care about the registry can stay simple.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

from app.backends.base import TranscriptionBackend
from app.model_mapping import alias_for, canonical_for, get_model


log = logging.getLogger(__name__)


def _build_onnx_asr(
    model: str,
    onnx_family: str = "auto",
    language: Optional[str] = None,
    device: str = "auto",
    quantization: Optional[str] = None,
    load_id: Optional[str] = None,
    **_ignored,
) -> TranscriptionBackend:
    # Late import: keeps onnx-asr off the module-load path of tests
    # that patch this builder (no need to even have onnxruntime
    # installed in the test env).
    from app.backends.onnx_backend import OnnxAsrBackend

    return OnnxAsrBackend(
        model=model,
        family=onnx_family,
        language=language,
        device=device,
        quantization=quantization,
        load_id=load_id,
    )


class RoutedBackend:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,  # accepted for API parity, ignored
    ) -> None:
        del beam_size  # unused (no beam search knob in ONNX path)

        self._lock = threading.Lock()
        # Resolve compute_type → quantization mapping for onnx-asr.
        quantization = _quantization_for(compute_type)
        self._kwargs = dict(
            device=device,
            language=language,
            quantization=quantization,
        )
        self._progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None

        canonical, onnx_family, load_id = self._resolve_for(model)
        self._inner = _build_onnx_asr(
            canonical,
            onnx_family=onnx_family,
            load_id=load_id,
            **self._kwargs,
        )

    # ---- public API ---------------------------------------------------------

    def current_kind(self) -> str:
        # Single-engine app — always ``onnx_asr``.  Kept for callers
        # that still inspect the kind (settings UI, telemetry).
        return "onnx_asr"

    def status(self) -> str:
        return self._inner.status()

    def health_check(self) -> bool:
        return self._inner.health_check()

    def load(self) -> None:
        self._inner.load()

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,
    ) -> None:
        canonical, onnx_family, load_id = self._resolve_for(model)

        # Same family — let the inner backend swap models without a
        # rebuild.  The onnx-asr session can be re-pointed to a new
        # repo within the same family (Whisper-base → Whisper-turbo)
        # without throwing away the family-specific configuration.
        current_family = getattr(self._inner, "_family", None)
        if current_family == onnx_family:
            self._inner.change_model(
                canonical, compute_type=compute_type, load_id=load_id,
            )
            return

        # Different model or different family — tear down, rebuild.
        log.info(
            "Switching ONNX model: %s (family=%s) → %s (family=%s)",
            self._inner.current_model(), current_family,
            canonical, onnx_family,
        )
        try:
            self._inner.shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Old backend shutdown raised: %s", exc)

        if compute_type is not None:
            self._kwargs["quantization"] = _quantization_for(compute_type)

        new_inner = _build_onnx_asr(
            canonical,
            onnx_family=onnx_family,
            load_id=load_id,
            **self._kwargs,
        )
        if self._progress_callback is not None:
            try:
                new_inner.set_progress_callback(self._progress_callback)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "set_progress_callback on new backend raised: %s", exc,
                )

        with self._lock:
            self._inner = new_inner

        new_inner.load()

    def current_model(self) -> str:
        return self._inner.current_model()

    def current_language(self) -> Optional[str]:
        return self._inner.current_language()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        return self._inner.transcribe(audio, sample_rate=sample_rate)

    def transcribe_file(self, path: str) -> Optional[str]:
        target = getattr(self._inner, "transcribe_file", None)
        if target is None:
            return None
        return target(path)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def cancel_load(self) -> None:
        target = getattr(self._inner, "cancel_load", None)
        if target is None:
            return
        try:
            target()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("cancel_load on inner raised: %s", exc)

    def update_inference_settings(self, settings) -> None:
        target = getattr(self._inner, "update_inference_settings", None)
        if target is None:
            return
        try:
            target(settings)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "update_inference_settings on inner raised: %s", exc,
            )

    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        self._progress_callback = callback
        try:
            self._inner.set_progress_callback(callback)
        except AttributeError:
            pass

    # ---- helpers ------------------------------------------------------------

    def _resolve_for(self, model: str) -> tuple[str, str, Optional[str]]:
        """Map an alias / canonical id to ``(canonical, onnx_family,
        onnx_load_id)``.

        ``onnx_load_id`` is what to pass to ``onnx_asr.load_model`` —
        usually the same as ``canonical`` (the HF repo path), but
        overridden in the registry when onnx-asr knows the model under
        a different identifier (T-One, GigaAM e2e variants, NeMo
        short names).  Falls back to ``("…", "auto", None)`` for
        unknown ids so a bare Hugging Face repo path still loads.
        """
        try:
            info = get_model(alias_for(model))
        except KeyError:
            return canonical_for(model), "auto", None
        return info.canonical, info.onnx_family, info.onnx_load_id


def _quantization_for(compute_type: Optional[str]) -> Optional[str]:
    """Translate the legacy ``compute_type`` knob to onnx-asr's
    ``quantization`` parameter.

    onnx-asr only accepts ``"int8"``, ``"fp16"``, or ``None`` — the
    nuanced CTranslate2 modes (``int8_float16`` etc.) collapse to the
    nearest valid value.
    """
    if compute_type in (None, "float16", "float32"):
        return None
    if compute_type in ("int8", "int8_float16"):
        return "int8"
    return None
