"""In-process faster-whisper backend.

Wraps ``faster_whisper.WhisperModel`` directly: model load happens on a
background thread (so the UI doesn't freeze for the 3-15 seconds it takes),
and ``transcribe`` calls the loaded model synchronously from whatever thread
the recording pipeline runs on.

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
from typing import Callable, Optional

import numpy as np


log = logging.getLogger(__name__)


_cuda_dlls_registered = False
_tqdm_patched = False
# Module-level callback called by the custom tqdm. Set by
# ``FasterWhisperBackend.set_progress_callback``. Signature:
# ``callback(current_bytes: int, total_bytes: int, desc: str) -> None``.
_progress_callback: Optional["Callable[[int, int, str], None]"] = None  # type: ignore


def _install_tqdm_progress() -> None:
    """Patch ``tqdm`` so huggingface_hub's download bars report into our
    progress callback.

    huggingface_hub uses ``tqdm.tqdm`` for download progress. We subclass
    it, override ``update`` and ``refresh`` to forward ``(n, total, desc)``
    to ``_progress_callback`` whenever it's set. The patch is applied to
    both ``tqdm`` and ``tqdm.auto`` (huggingface_hub imports from one or
    the other depending on the call site). Idempotent.
    """
    global _tqdm_patched
    if _tqdm_patched:
        return
    try:
        import tqdm as _tqdm
        import tqdm.auto as _tqdm_auto
    except ImportError:
        return

    base_cls = _tqdm.tqdm

    class _ProgressTqdm(base_cls):  # type: ignore[misc, valid-type]
        def update(self, n=1):
            ret = super().update(n)
            # PyQt apps usually run without a TTY, so huggingface_hub's
            # tqdm gets ``disable=None`` which auto-resolves to ``True``.
            # When disabled, vanilla tqdm short-circuits ``update()`` and
            # leaves ``self.n`` at zero — meaning our callback would see
            # 0 bytes forever. Mirror the count ourselves in that case so
            # the progress callback reports accurate bytes.
            if getattr(self, "disable", False) and n:
                try:
                    self.n = (self.n or 0) + n
                except (TypeError, ValueError):
                    pass
            self._fire()
            return ret

        def refresh(self, *args, **kwargs):
            ret = super().refresh(*args, **kwargs)
            self._fire()
            return ret

        def close(self):
            self._fire()
            return super().close()

        def _fire(self):
            cb = _progress_callback
            if cb is None:
                return
            # ``disable=True`` makes tqdm.__init__ return early before
            # ``self.desc`` is assigned — and PyQt apps run without a TTY,
            # which auto-disables every bar huggingface_hub creates. Use
            # getattr so we still report progress instead of swallowing
            # an AttributeError silently.
            total = int(getattr(self, "total", 0) or 0)
            # Hugging Face spawns one tqdm bar per file (config.json,
            # tokenizer.json, vocabulary.txt, model.bin, …). The small
            # ones complete in milliseconds, so the user sees the bar
            # bounce 0%→99%→0%→99%→0%→% as each file is touched. Skip
            # bars whose total weight is trivial — only the actual
            # weights file is worth surfacing in the UI.
            if 0 < total < 1_000_000:
                return
            try:
                cb(
                    int(getattr(self, "n", 0) or 0),
                    total,
                    str(getattr(self, "desc", "") or ""),
                )
            except Exception:
                pass

    _tqdm.tqdm = _ProgressTqdm
    _tqdm_auto.tqdm = _ProgressTqdm
    _tqdm_patched = True


def _register_cuda_dll_dirs() -> None:
    """Make CUDA DLLs from ``nvidia-*-cu12`` wheels resolvable on Windows.

    ``ctranslate2`` lazy-loads ``cublas64_12.dll`` / ``cudnn64_9.dll`` etc.
    via a plain ``LoadLibrary`` call, which does NOT search directories
    added via ``os.add_dll_directory`` (that flag is only honoured by
    ``LoadLibraryEx`` with ``LOAD_LIBRARY_SEARCH_USER_DIRS``). To cover
    all cases we do three things for each ``nvidia/<pkg>/bin`` directory:

    1. ``os.add_dll_directory`` — covers anything using LoadLibraryEx
    2. Prepend to ``PATH`` — covers the legacy LoadLibrary search order
    3. Pre-load the DLLs into the process via ``ctypes.WinDLL`` from the
       absolute path so subsequent ``LoadLibrary("cublas64_12.dll")``
       calls return the already-loaded module handle.

    Idempotent.
    """
    global _cuda_dlls_registered
    if _cuda_dlls_registered or sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    # ``nvidia`` is a PEP 420 namespace package — ``__path__`` lists the
    # site-packages directories that contribute ``nvidia/<sub>/``.
    roots = list(getattr(nvidia, "__path__", []) or [])
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]

    bin_dirs: list[str] = []
    for nvidia_root in roots:
        if not os.path.isdir(nvidia_root):
            continue
        for sub in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)

    if not bin_dirs:
        return

    # 1) AddDllDirectory for LoadLibraryEx-style loads.
    for bin_dir in bin_dirs:
        try:
            os.add_dll_directory(bin_dir)
        except (FileNotFoundError, OSError) as exc:
            log.debug("add_dll_directory(%s) failed: %s", bin_dir, exc)

    # 2) Prepend to PATH for legacy LoadLibrary resolution.
    existing = os.environ.get("PATH", "")
    new_path = os.pathsep.join(bin_dirs + ([existing] if existing else []))
    os.environ["PATH"] = new_path

    # 3) Eagerly load the runtime DLLs that ctranslate2 will need so
    # subsequent ``LoadLibrary("cublas64_12.dll")`` calls find the already-
    # loaded module by name. Order matters — cuBLAS depends on cublasLt,
    # cuDNN on the cuDNN sub-libs.
    try:
        import ctypes
    except ImportError:
        ctypes = None  # type: ignore[assignment]

    if ctypes is not None:
        candidates = [
            "cublasLt64_12.dll",
            "cublas64_12.dll",
            "cudnn64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_adv64_9.dll",
        ]
        for bin_dir in bin_dirs:
            for name in candidates:
                full = os.path.join(bin_dir, name)
                if not os.path.isfile(full):
                    continue
                try:
                    ctypes.WinDLL(full)
                    log.debug("Preloaded %s", full)
                except OSError as exc:
                    log.debug("Preload %s failed: %s", full, exc)

    _cuda_dlls_registered = True
    log.info("Registered CUDA DLL directories: %s", bin_dirs)


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
        # Inference-time tunables that ``transcribe`` plumbs into
        # ``WhisperModel.transcribe``. Overridable via
        # ``update_inference_settings`` so the user's per-model
        # settings on the active card take effect without a reload.
        self._vad_filter: bool = True
        self._initial_prompt: Optional[str] = None
        self._temperature: float = 0.0

        self._model = None
        self._status = "stopped"
        self._load_thread: Optional[threading.Thread] = None
        self._shutdown = False
        # Set to True when the user clicks Cancel while a load is in
        # flight. The load worker keeps running until ``WhisperModel``
        # returns (Python can't safely interrupt a foreign thread)
        # but its result is discarded at the publish step. Reset on
        # each fresh ``load()``.
        self._cancel_requested = False

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
            self._cancel_requested = False
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

    def cancel_load(self) -> None:
        """Abandon an in-flight load — see ``NemoBackend.cancel_load``
        for the full rationale. Idempotent; only acts when status is
        ``loading``."""
        with self._lock:
            if self._shutdown:
                return
            if self._status != "loading":
                return
            log.info("faster-whisper model load cancelled by user")
            self._cancel_requested = True
            self._model = None
            self._status = "stopped"

    def change_model(
        self,
        model: str,
        compute_type: Optional[str] = None,
    ) -> None:
        """Switch to a different model and/or compute_type.

        Passing ``compute_type=None`` keeps the current setting; otherwise a
        change in either field triggers a reload.
        """
        with self._lock:
            if self._shutdown:
                return
            same_model = model == self._model_name
            same_compute = compute_type is None or compute_type == self._compute_type
            if same_model and same_compute and self._status == "ready":
                return
            self._model_name = model
            if compute_type is not None:
                self._compute_type = compute_type
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
            vad_filter = self._vad_filter
            initial_prompt = self._initial_prompt
            temperature = self._temperature

        # Belt-and-suspenders: ctranslate2 lazy-loads cuBLAS / cuDNN on the
        # first inference call rather than at model construction. Re-register
        # the DLL dirs here too in case load() ran in a context where the
        # nvidia-* wheels weren't yet importable.
        _register_cuda_dll_dirs()

        kwargs = {
            "language": language,
            "beam_size": beam_size,
            "temperature": temperature,
            "vad_filter": vad_filter,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        try:
            segments, _info = model.transcribe(audio, **kwargs)
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

    def update_inference_settings(self, settings) -> None:
        """Push a fresh ``InferenceSettings`` instance into the
        backend — applied on the next ``transcribe`` call without a
        reload (the parameters are passed straight to
        ``WhisperModel.transcribe``)."""
        with self._lock:
            self._language = getattr(settings, "language", None)
            self._beam_size = int(getattr(settings, "beam_size", 5))
            self._vad_filter = bool(getattr(settings, "vad_filter", True))
            self._initial_prompt = getattr(settings, "initial_prompt", None)
            self._temperature = float(
                getattr(settings, "temperature", 0.0)
            )

    @staticmethod
    def set_progress_callback(
        callback: Optional[Callable[[int, int, str], None]],
    ) -> None:
        """Register a function called whenever a download progress bar
        ticks. Signature ``(current_bytes, total_bytes, desc) -> None``.

        Stored module-globally because the custom ``tqdm`` subclass is
        installed once per process — it doesn't know which backend
        instance it belongs to.
        """
        global _progress_callback
        _progress_callback = callback

    # ---- internal -----------------------------------------------------------

    def _do_load(self, model_name: str) -> None:
        # Make CUDA DLLs from the ``nvidia-*-cu12`` wheels resolvable BEFORE
        # we import the engine. Without this, GPU inference falls over with
        # ``Library cublas64_12.dll is not found or cannot be loaded``.
        _register_cuda_dll_dirs()
        # Install our custom tqdm subclass before huggingface_hub imports
        # ``from tqdm import tqdm`` so its download progress bars go through
        # our callback.
        _install_tqdm_progress()

        # Lazy import: keeps ``import app.backends`` free of the heavy
        # CTranslate2 / cuDNN dependency chain until somebody actually loads
        # a backend.
        from faster_whisper import WhisperModel

        with self._lock:
            device = self._device
            compute_type = self._compute_type

        # Honour ``HF_HOME`` at the moment of load — passing it as
        # ``download_root`` makes the user's Storage-tab path change
        # apply to the very next model load, no restart needed
        # (faster-whisper forwards this straight to
        # ``huggingface_hub.snapshot_download(cache_dir=...)``).
        # When unset, omit the kwarg entirely so the default
        # ``~/.cache/huggingface/hub`` location is used.
        load_kwargs: dict = {"device": device, "compute_type": compute_type}
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            load_kwargs["download_root"] = os.path.join(hf_home, "hub")

        log.info(
            "Loading faster-whisper model %s (device=%s, compute_type=%s, "
            "download_root=%s)",
            model_name, device, compute_type, load_kwargs.get("download_root"),
        )
        try:
            model = WhisperModel(model_name, **load_kwargs)
        except Exception as exc:
            log.error("Failed to load model %s: %s", model_name, exc, exc_info=True)
            with self._lock:
                # If the user cancelled while WhisperModel was running,
                # don't promote a partial-download failure to "error"
                # — leave the cancelled state alone.
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
                # Backend was torn down while loading — discard the result.
                return
            if self._cancel_requested:
                log.info(
                    "Discarding loaded model %s — cancelled by user",
                    model_name,
                )
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
