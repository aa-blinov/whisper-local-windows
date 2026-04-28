"""Registry-aware backend façade.

A thin wrapper around :class:`OnnxAsrBackend` that resolves model
aliases through the application registry (``app.model_mapping``)
before forwarding calls.  It exists so the rest of the app can pass
human-friendly aliases (``parakeet-tdt-v3``, ``whisper-large-v3-turbo``,
``t-one``) and stay agnostic of three details:

- the actual HuggingFace canonical (``istupakov/parakeet-tdt-…-onnx``)
- the onnx-asr load identifier when it differs from the canonical
  (``nemo-parakeet-tdt-0.6b-v3`` short name, lowercase ``t-tech/t-one``,
  ``gigaam-v3-e2e-rnnt`` decoder picker)
- the family-specific runtime knobs (Whisper takes a ``language`` kwarg,
  Parakeet doesn't, GigaAM is RU-only, etc.)

Why a class and not just a free function: ``change_model`` needs to
re-do the lookup for the *new* alias, and family transitions
(Whisper → GigaAM) want to rebuild the inner backend so we don't carry
stale family-specific state.  Keeping the inner reference in ``self``
makes that bookkeeping cheap.

The class also collapses the legacy ``compute_type`` knob (``int8`` /
``int8_float16`` / ``float16`` / ``float32``) into onnx-asr's smaller
``quantization`` vocabulary (``"int8"`` / ``None``) at one place, so
config changes don't have to be plumbed through.
"""

from __future__ import annotations

import logging
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


class RegistryBackend:
    """Backend that looks up alias → ``(canonical, family, load_id)``
    in the registry and forwards to an ``OnnxAsrBackend`` instance.

    The class implements ``TranscriptionBackend`` Protocol verbatim;
    callers that already have a canonical HF id can pass it through
    unchanged (we fall back gracefully via ``canonical_for``).
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,  # accepted for API parity, ignored
    ) -> None:
        del beam_size  # unused (no beam search knob in ONNX path)

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

        # Different family — tear down, rebuild.
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
