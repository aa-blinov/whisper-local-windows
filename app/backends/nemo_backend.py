"""In-process NVIDIA NeMo backend.

Wraps ``nemo_toolkit[asr]`` so the app can run NVIDIA's recent
multilingual ASR releases — Parakeet TDT 0.6B v3 (25 European
languages incl. Russian) plus future siblings like Canary v2.

Lifecycle matches the other backends:

    stopped ──load()──▶ loading ──ok──▶ ready
                               └──err──▶ error

    ready ──change_model()──▶ loading ──▶ ready / error
    *     ──shutdown()──▶ stopped (terminal)

NeMo models register downloads against the same Hugging Face hub
cache as faster-whisper (``HF_HOME``), so the user's Storage-tab
path automatically applies — no extra plumbing.

Inference quirks worth knowing
------------------------------
- ``ASRModel.transcribe`` only takes file paths in older releases.
  We spool the captured ``numpy`` buffer to a temporary WAV the
  same way ``GigaamBackend`` does. ``delete=False`` is mandatory
  on Windows: ``NamedTemporaryFile`` holds an open handle that
  ``soundfile`` would fail to re-open.
- Auto language detection — Parakeet TDT v3 picks one of 25
  languages on its own. We expose ``current_language()`` as
  ``None`` so history rows aren't tagged with a hard-coded code.
- Greedy decoding only — no beam-size knob to expose. The
  inference settings panel for NeMo cards is therefore minimal:
  one toggle for ``timestamps`` (output the per-word/segment
  offsets in the result object).
- Long-form attention. The default full-attention layout caps
  Parakeet at ~24 minutes per call. We switch to
  ``rel_pos_local_attn`` with ``[256, 256]`` context once at load
  time — extends the limit to a few hours at a tiny accuracy
  cost, which is the right trade for a dictation app where we
  don't know the recording length in advance.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Callable, Optional

import numpy as np

from app.inference_settings import NemoInferenceSettings


log = logging.getLogger(__name__)


class NemoBackend:
    def __init__(
        self,
        model: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "auto",
    ) -> None:
        self._lock = threading.Lock()
        self._model_name = model
        self._device = device

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False
        # Inference-time tunables. Cached here so ``transcribe`` can
        # apply them without re-reading from config on every call.
        self._inference_settings = NemoInferenceSettings()

    # ---- public API ---------------------------------------------------------

    def current_model(self) -> str:
        with self._lock:
            return self._model_name

    def current_language(self) -> Optional[str]:
        # Parakeet TDT v3 auto-detects language across 25 supported
        # tongues — no source-language knob to expose. Reporting
        # ``None`` matches the contract used elsewhere ("the model
        # decides, don't stamp history with a guess").
        return None

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
            name=f"nemo-load-{target_model}",
        )
        with self._lock:
            self._load_thread = thread
        thread.start()

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,  # accepted for API parity
    ) -> None:
        """Switch to a different NeMo model. ``compute_type`` is
        ignored — NeMo's model precision is baked into the released
        weights (Parakeet TDT v3 ships fp32 in safetensors); the
        kwarg only exists so the routed-backend façade can call
        every backend with the same shape."""
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

    def update_inference_settings(self, settings: NemoInferenceSettings) -> None:
        """Apply a new settings bundle. Effect is per-call: the next
        ``transcribe`` reads ``self._inference_settings``. No model
        reload needed."""
        with self._lock:
            self._inference_settings = settings

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
            settings = self._inference_settings

        # NeMo's ``transcribe`` API stabilised around file-path inputs
        # — newer versions accept arrays but the path overload is the
        # safe denominator across releases. Spool to a tempfile, call,
        # clean up.
        try:
            import soundfile as sf  # transitive dep of NeMo
        except ImportError as exc:
            log.error("soundfile is required for NeMo transcribe: %s", exc)
            return None

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            buf = np.asarray(audio, dtype=np.float32)
            if buf.ndim > 1:
                buf = buf.mean(axis=1)
            sf.write(tmp_path, buf, int(sample_rate), subtype="PCM_16")

            kwargs: dict = {}
            if getattr(settings, "timestamps", False):
                kwargs["timestamps"] = True

            result = model.transcribe([tmp_path], **kwargs)
        except Exception as exc:
            log.error("NeMo transcription failed: %s", exc, exc_info=True)
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        text = self._extract_text(result)
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
        """NeMo defers HF downloads to ``huggingface_hub`` which the
        existing ``_install_tqdm_progress`` (faster-whisper backend)
        already hooks. Kept for API parity with the other backends so
        ``RoutedBackend`` can forward callbacks blindly."""
        del callback  # no-op — progress flows through the shared tqdm hook

    # ---- internal -----------------------------------------------------------

    @staticmethod
    def _extract_text(result) -> Optional[str]:
        """``ASRModel.transcribe`` returns ``List[Hypothesis]`` in
        recent releases; older / mocked variants may yield a list
        of strings or a single string. Accept all three shapes."""
        if result is None:
            return None
        if isinstance(result, str):
            return result
        if not result:
            # Empty list / tuple — no transcription produced.
            return None
        first = result[0] if hasattr(result, "__getitem__") else result
        text = getattr(first, "text", None)
        if text is None and isinstance(first, str):
            text = first
        return text

    def _do_load(self, model_name: str) -> None:
        # Same tqdm progress hook the other backends install before
        # importing their loader; NeMo's HF-backed downloads will
        # bubble up through huggingface_hub which ``_install_tqdm_progress``
        # already wraps.
        from app.backends.faster_whisper_backend import _install_tqdm_progress

        _install_tqdm_progress()

        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except ImportError as exc:
            log.error("nemo_toolkit is not installed: %s", exc)
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._model = None
                    self._status = "error"
            return

        log.info("Loading NeMo model %s (device=%s)", model_name, self._device)
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
            )
        except Exception as exc:
            log.error(
                "Failed to load NeMo model %s: %s",
                model_name, exc, exc_info=True,
            )
            with self._lock:
                if not self._shutdown and self._model_name == model_name:
                    self._model = None
                    self._status = "error"
            return

        # Switch to local attention for long-audio support. Cheap on
        # short clips (the model was trained with both layouts) and
        # the only way to handle multi-minute dictation without a
        # second reload.
        try:
            model.change_attention_model(
                "rel_pos_local_attn",
                att_context_size=[256, 256],
            )
        except Exception as exc:
            log.warning(
                "change_attention_model failed for %s — proceeding with "
                "default attention (long-form audio may be capped): %s",
                model_name, exc,
            )

        with self._lock:
            if self._shutdown:
                return
            if self._model_name != model_name:
                log.info(
                    "Discarding loaded NeMo model %s — name changed to %s",
                    model_name, self._model_name,
                )
                return
            self._model = model
            self._status = "ready"
            log.info("NeMo model %s ready", model_name)
