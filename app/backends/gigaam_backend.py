"""In-process GigaAM backend.

Wraps Sber's ``gigaam`` package — a Russian-only acoustic model with
RNN-T / CTC decoders. Mirrors the lifecycle of ``FasterWhisperBackend``:
model loads on a background thread, ``transcribe`` runs synchronously
from whatever thread the recording pipeline supplies the audio.

State machine matches the rest of the backends::

    stopped ──load()──▶ loading ──ok──▶ ready
                               └──err──▶ error

    ready ──change_model()──▶ loading ──▶ ready / error
    *     ──shutdown()──▶ stopped (terminal)
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Callable, Optional

import numpy as np


log = logging.getLogger(__name__)


class GigaamBackend:
    def __init__(
        self,
        model: str = "v2_ctc",
        device: str = "auto",
        language: Optional[str] = "ru",
    ) -> None:
        self._lock = threading.Lock()
        self._model_name = model
        self._device = device
        # GigaAM is Russian-only — exposing the language attribute keeps
        # the contract uniform across backends so ``state_manager`` can
        # tag history entries the same way regardless of engine.
        self._language = language

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False

    # ---- public API ---------------------------------------------------------

    def current_model(self) -> str:
        with self._lock:
            return self._model_name

    def current_language(self) -> Optional[str]:
        with self._lock:
            return self._language

    def status(self) -> str:
        with self._lock:
            return self._status

    def health_check(self) -> bool:
        return self.status() == "ready"

    def load(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            if self._status in ("loading", "ready"):
                return
            self._status = "loading"
            target_model = self._model_name

        thread = threading.Thread(
            target=self._do_load,
            args=(target_model,),
            daemon=True,
            name=f"gigaam-load-{target_model}",
        )
        with self._lock:
            self._load_thread = thread
        thread.start()

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,  # accepted for API parity
    ) -> None:
        """Switch to a different GigaAM model. ``compute_type`` is
        ignored — GigaAM doesn't expose a CT2-style precision knob;
        the kwarg only exists so the backend interface matches
        ``FasterWhisperBackend``."""
        del compute_type  # unused, see docstring
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
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[str]:
        with self._lock:
            if self._shutdown:
                return None
            if self._status != "ready" or self._model is None:
                log.warning(
                    "transcribe() called but backend status is %r", self._status
                )
                return None
            model = self._model

        # GigaAM's ``transcribe`` only accepts a path on disk — it
        # internally re-reads the WAV with ``soundfile`` to a tensor.
        # Spool the captured numpy buffer to a temp file, hand the
        # path over, then clean up. ``delete=False`` is mandatory on
        # Windows: NamedTemporaryFile holds an open handle that
        # ``soundfile`` would fail to re-open.
        try:
            import soundfile as sf  # provided as a gigaam dep
        except ImportError as exc:
            log.error("soundfile is required for GigaAM transcribe: %s", exc)
            return None

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            # Cast to float32 mono in case the recorder handed us
            # something else; gigaam expects 16 kHz PCM.
            buf = np.asarray(audio, dtype=np.float32)
            if buf.ndim > 1:
                buf = buf.mean(axis=1)
            sf.write(tmp_path, buf, int(sample_rate), subtype="PCM_16")

            result = model.transcribe(tmp_path)
        except Exception as exc:
            log.error("GigaAM transcription failed: %s", exc, exc_info=True)
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # ``transcribe`` returns ``TranscriptionResult`` with ``.text``
        # and ``.words``; the legacy code path expected a plain string,
        # so unwrap. Tolerate both shapes for safety.
        text = getattr(result, "text", None)
        if text is None and isinstance(result, str):
            text = result
        if text is None:
            return None
        text = str(text).strip()
        return text or None

    def shutdown(self) -> None:
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
        """GigaAM doesn't expose a download progress hook — its loader
        prints to stderr. Kept on the API for symmetry with
        ``FasterWhisperBackend.set_progress_callback`` so the
        ``RoutedBackend`` facade can forward callbacks blindly."""
        del callback  # no-op

    # ---- internal -----------------------------------------------------------

    def _do_load(self, model_name: str) -> None:
        # Install the same tqdm progress hook the FasterWhisper backend
        # uses BEFORE importing gigaam — gigaam binds ``from tqdm
        # import tqdm`` at module-import time, so the patched
        # subclass needs to be in place first or the download bar
        # never reaches our callback.
        from app.backends.faster_whisper_backend import _install_tqdm_progress

        _install_tqdm_progress()

        # Lazy import: GigaAM pulls in PyTorch which is heavy.
        # Importing only when we actually need it keeps the default
        # startup path light.
        try:
            import gigaam  # type: ignore
        except ImportError as exc:
            log.error("gigaam package is not installed: %s", exc)
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._model = None
                    self._status = "error"
            return

        log.info("Loading GigaAM model %s (device=%s)", model_name, self._device)
        try:
            model = gigaam.load_model(model_name)
        except Exception as exc:
            log.error(
                "Failed to load GigaAM model %s: %s",
                model_name, exc, exc_info=True,
            )
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._model = None
                    self._status = "error"
            return

        with self._lock:
            if self._shutdown:
                return
            if self._model_name != model_name:
                log.info(
                    "Discarding loaded GigaAM model %s — name changed to %s",
                    model_name, self._model_name,
                )
                return
            self._model = model
            self._status = "ready"
            log.info("GigaAM model %s ready", model_name)
