"""In-process faster-whisper backend.

Replaces the previous Wyoming/Docker setup. The backend wraps
``faster_whisper.WhisperModel`` directly: model load happens on a background
thread (so the UI doesn't freeze for the 3-15 seconds it takes), and
``transcribe`` calls the loaded model synchronously from whatever thread the
recording pipeline runs on.

State machine::

    stopped ──load()──▶ loading ──ok──▶ ready
                               └──err──▶ error

    ready ──change_model()──▶ loading ──▶ ready / error
    *     ──shutdown()──▶ stopped (terminal)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


def _register_cuda_dll_dirs() -> None:
    """Add ``nvidia-*`` wheel ``bin`` directories to the Windows DLL search.

    ``ctranslate2`` (the engine behind faster-whisper) loads
    ``cublas64_12.dll`` / ``cudnn64_9.dll`` at inference time. The Python
    ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` wheels ship those DLLs
    inside ``site-packages/nvidia/<pkg>/bin``, but Python does not put that
    directory on ``%PATH%`` automatically. Register each present
    sub-package's ``bin`` folder via ``os.add_dll_directory`` so the
    runtime resolution succeeds without a system-wide CUDA install.
    Idempotent — safe to call multiple times.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    # ``nvidia`` is a PEP 420 namespace package — ``__file__`` is None,
    # ``__path__`` lists every site-packages directory that contributes
    # ``nvidia/<sub>/`` (typically just one).
    roots = list(getattr(nvidia, "__path__", []) or [])
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    for nvidia_root in roots:
        if not os.path.isdir(nvidia_root):
            continue
        for sub in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, sub, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except (FileNotFoundError, OSError) as exc:
                    log.debug("Skipping CUDA dll dir %s: %s", bin_dir, exc)


class FasterWhisperBackend:
    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> None:
        self._lock = threading.Lock()
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size

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
            name=f"fw-load-{target_model}",
        )
        with self._lock:
            self._load_thread = thread
        thread.start()

    def change_model(self, model: str) -> None:
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
            language = self._language
            beam_size = self._beam_size

        # Belt-and-suspenders: ctranslate2 lazy-loads cuBLAS / cuDNN on the
        # first inference call rather than at model construction. Re-register
        # the DLL dirs here too in case load() ran in a context where the
        # nvidia-* wheels weren't yet importable.
        _register_cuda_dll_dirs()

        try:
            segments, _info = model.transcribe(
                audio, language=language, beam_size=beam_size,
            )
            text = "".join(segment.text for segment in segments).strip()
            return text or None
        except Exception as exc:
            log.error("Transcription failed: %s", exc, exc_info=True)
            return None

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._model = None
            self._status = "stopped"

    # ---- internal -----------------------------------------------------------

    def _do_load(self, model_name: str) -> None:
        # Make CUDA DLLs from the ``nvidia-*-cu12`` wheels resolvable BEFORE
        # we import the engine. Without this, GPU inference falls over with
        # ``Library cublas64_12.dll is not found or cannot be loaded``.
        _register_cuda_dll_dirs()

        # Lazy import: keeps ``import app.backends`` free of the heavy
        # CTranslate2 / cuDNN dependency chain until somebody actually loads
        # a backend.
        from faster_whisper import WhisperModel

        with self._lock:
            device = self._device
            compute_type = self._compute_type

        log.info(
            "Loading faster-whisper model %s (device=%s, compute_type=%s)",
            model_name, device, compute_type,
        )
        try:
            model = WhisperModel(
                model_name, device=device, compute_type=compute_type,
            )
        except Exception as exc:
            log.error("Failed to load model %s: %s", model_name, exc, exc_info=True)
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._model = None
                    self._status = "error"
            return

        with self._lock:
            if self._shutdown:
                # Backend was torn down while loading — discard the result.
                return
            if self._model_name != model_name:
                # change_model was called mid-load; the new request will
                # have started its own load.
                log.info(
                    "Discarding loaded model %s — name changed to %s",
                    model_name, self._model_name,
                )
                return
            self._model = model
            self._status = "ready"
            log.info("faster-whisper model %s ready", model_name)
