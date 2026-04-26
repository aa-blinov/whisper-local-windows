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

            duration_s = len(buf) / max(1, int(sample_rate))
            # GigaAM's plain ``transcribe`` is documented as good up to
            # 25 seconds. Longer clips have to go through
            # ``transcribe_longform``, which uses pyannote VAD to
            # split the audio into <25 s segments. Use a small
            # safety margin so we never feed transcribe a buffer
            # right at the edge.
            if duration_s > 24.0:
                text = self._transcribe_longform(model, tmp_path)
            else:
                result = model.transcribe(tmp_path)
                text = self._extract_text(result)
        except Exception as exc:
            log.error("GigaAM transcription failed: %s", exc, exc_info=True)
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if text is None:
            return None
        text = str(text).strip()
        return text or None

    @staticmethod
    def _extract_text(result) -> Optional[str]:
        """``model.transcribe`` returns ``TranscriptionResult`` with
        ``.text`` / ``.words``; the legacy fakes return a plain
        string. Accept both."""
        text = getattr(result, "text", None)
        if text is None and isinstance(result, str):
            text = result
        return text

    def _transcribe_longform(self, model, wav_path: str) -> Optional[str]:
        """Route long captures through ``transcribe_longform``.

        The method needs the ``gigaam[longform]`` extras
        (``pyannote-audio``) installed AND the user has to have
        accepted the gated ``pyannote/segmentation-3.0`` model on
        Hugging Face with ``HF_TOKEN`` set. Falls back to plain
        ``transcribe`` (which will likely truncate) if the longform
        path isn't available — better partial output than nothing.
        """
        longform = getattr(model, "transcribe_longform", None)
        if longform is None:
            log.warning(
                "GigaAM model has no transcribe_longform — falling "
                "back to plain transcribe (audio may be truncated)."
            )
            return self._extract_text(model.transcribe(wav_path))
        try:
            segments = longform(wav_path)
        except Exception as exc:
            log.error(
                "transcribe_longform failed (deps missing or "
                "pyannote/segmentation-3.0 not accepted on HF?): %s",
                exc,
            )
            return self._extract_text(model.transcribe(wav_path))
        # Each segment exposes ``.text`` / ``.start`` / ``.end``;
        # strings stay supported for fakes.
        chunks: list[str] = []
        for seg in segments or []:
            piece = self._extract_text(seg)
            if piece:
                chunks.append(str(piece).strip())
        return " ".join(c for c in chunks if c)

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
        """GigaAM's own loader downloads from Sber's CDN via plain
        ``urllib`` (no tqdm), so progress for the .ckpt fetch can't
        flow through the shared hook either way. But pyannote's
        long-form deps (downloaded on first long-audio capture) DO
        use ``huggingface_hub`` + tqdm — forward to FasterWhisper's
        module-level ``_progress_callback`` so those at least
        surface in the UI. Same pattern as ``NemoBackend``."""
        from app.backends.faster_whisper_backend import (
            FasterWhisperBackend as _FW,
        )

        _FW.set_progress_callback(callback)

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

        # Honour ``GIGAAM_MODELS_DIR`` (set at startup from the
        # configured ``storage.models_dir``); when unset, omit
        # ``download_root`` entirely so the library uses its own
        # default ``~/.cache/gigaam`` — keeps existing installs
        # finding their weights after an upgrade.
        load_kwargs: dict = {}
        env_root = os.environ.get("GIGAAM_MODELS_DIR")
        if env_root:
            load_kwargs["download_root"] = env_root

        log.info("Loading GigaAM model %s (device=%s)", model_name, self._device)
        try:
            model = gigaam.load_model(model_name, **load_kwargs)
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
