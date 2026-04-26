"""Multi-engine routing backend.

Implements ``TranscriptionBackend`` but doesn't run inference itself —
it owns one inner backend at a time (``FasterWhisperBackend`` or
``GigaamBackend``) and delegates every call. When the user picks a
model whose ``backend_kind`` differs from the current inner, this
class shuts the old inner down and stands up a fresh one of the new
kind so the rest of the app (StateManager, status poller, recording
controller) doesn't need to know engines come and go underneath.

Building the concrete backends lives behind module-level functions
(``_build_faster_whisper`` / ``_build_gigaam``) so tests can monkey-
patch the construction without spinning up real model-loading
threads.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

import numpy as np

from app.backends.base import TranscriptionBackend
from app.model_mapping import alias_for, canonical_for, get_model


log = logging.getLogger(__name__)


def _build_faster_whisper(model: str, **kwargs) -> TranscriptionBackend:
    # Late import: avoids pulling CTranslate2 + huggingface_hub in
    # tests that monkey-patch the builder.
    from app.backends.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend(model=model, **kwargs)


def _build_gigaam(model: str, **kwargs) -> TranscriptionBackend:
    from app.backends.gigaam_backend import GigaamBackend

    # GigaamBackend has a narrower constructor than FasterWhisperBackend.
    # Drop kwargs that don't apply (compute_type, beam_size).
    accepted = {
        k: v for k, v in kwargs.items() if k in ("device", "language")
    }
    return GigaamBackend(model=model, **accepted)


def _build_nemo(model: str, **kwargs) -> TranscriptionBackend:
    from app.backends.nemo_backend import NemoBackend

    # NemoBackend takes only ``device``; ``compute_type`` /
    # ``beam_size`` / ``language`` are NeMo-internal or not
    # exposed through the public transcribe API.
    accepted = {k: v for k, v in kwargs.items() if k in ("device",)}
    return NemoBackend(model=model, **accepted)


class RoutedBackend:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> None:
        self._lock = threading.Lock()
        # Keep the construction kwargs around so the next inner
        # backend (after a kind switch) gets the same configuration.
        self._kwargs = dict(
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
        )
        self._progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None

        kind, canonical = self._resolve_for(model)
        self._kind = kind
        self._inner = self._build(canonical, kind)

    # ---- public API ---------------------------------------------------------

    def current_kind(self) -> str:
        with self._lock:
            return self._kind

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
        new_kind, canonical = self._resolve_for(model)

        with self._lock:
            current_kind = self._kind

        if new_kind == current_kind:
            # Same engine — let the inner backend handle the swap.
            self._inner.change_model(canonical, compute_type=compute_type)
            return

        # Cross-engine swap — tear down old, build fresh.
        log.info(
            "Switching backend kind: %s → %s for model %s",
            current_kind, new_kind, canonical,
        )
        try:
            self._inner.shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Old backend shutdown raised: %s", exc)

        if compute_type is not None:
            # Persist the new compute_type so a subsequent same-kind
            # rebuild picks it up.
            self._kwargs["compute_type"] = compute_type

        new_inner = self._build(canonical, new_kind)
        if self._progress_callback is not None:
            try:
                new_inner.set_progress_callback(self._progress_callback)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("set_progress_callback on new backend raised: %s", exc)

        with self._lock:
            self._inner = new_inner
            self._kind = new_kind

        new_inner.load()

    def current_model(self) -> str:
        return self._inner.current_model()

    def current_language(self) -> Optional[str]:
        return self._inner.current_language()

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        return self._inner.transcribe(audio, sample_rate=sample_rate)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def cancel_load(self) -> None:
        """Forward cancel-load to the inner backend.

        Silent if the inner doesn't expose ``cancel_load`` (older fakes
        in tests; future engines that haven't been updated yet) — the
        topbar's cancel button must never raise from a click. The
        inner backends themselves are idempotent when status isn't
        ``loading``, so it's safe to call this blindly from the UI.
        """
        target = getattr(self._inner, "cancel_load", None)
        if target is None:
            return
        try:
            target()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("cancel_load on inner raised: %s", exc)

    def update_inference_settings(self, settings) -> None:
        """Forward per-model overrides to the inner backend if it
        accepts them. GigaAM's backend ignores the call (its engine
        has no inference-time tunables) — kept silent so the
        controller can call this blindly after every model swap."""
        target = getattr(self._inner, "update_inference_settings", None)
        if target is None:
            return
        try:
            target(settings)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("update_inference_settings on inner raised: %s", exc)

    def set_progress_callback(
        self,
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        """Forward the callback to the current inner backend AND
        cache it so a subsequent kind-switching rebuild can re-attach
        it to the new inner."""
        self._progress_callback = callback
        try:
            self._inner.set_progress_callback(callback)
        except AttributeError:
            # Inner backend doesn't expose progress hooks — fine.
            pass

    # ---- helpers ------------------------------------------------------------

    def _resolve_for(self, model: str) -> tuple[str, str]:
        """Map an alias / canonical id to ``(kind, canonical)``.

        Falls back to ``faster_whisper`` for unknown ids so a bare
        Hugging Face repo path still works through the existing
        path.
        """
        try:
            info = get_model(alias_for(model))
        except KeyError:
            return "faster_whisper", canonical_for(model)
        return info.backend_kind, info.canonical

    def _build(self, canonical: str, kind: str) -> TranscriptionBackend:
        if kind == "gigaam":
            return _build_gigaam(canonical, **self._kwargs)
        if kind == "nemo":
            return _build_nemo(canonical, **self._kwargs)
        return _build_faster_whisper(canonical, **self._kwargs)
