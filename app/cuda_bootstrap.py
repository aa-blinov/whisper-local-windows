"""Auto-install NVIDIA CUDA redistributables for Windows + NVIDIA GPU users.

Called once at app startup before the Qt event loop spins up.  If CUDA
already works the function returns in <10 ms.  If the user has an NVIDIA
GPU but the runtime DLLs are missing we either:

1. silently inject the pip-package paths (already installed case), or
2. ask interactively whether to download & install them (~1 GB).

Both paths keep the single ``uv run lazy-to-text-ui`` entry point — no
manual ``pip install [cuda]`` required.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ANSI colours — only emitted when stdout is a real TTY so headless / CI
# logs stay clean.
_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}


def _c(key: str) -> str:
    return _C[key] if sys.stdout.isatty() else ""


# Minimal valid ONNX model used for the CUDA probe.
_WARMUP_ONNX_BYTES: bytes = (
    b'\x08\x08:4\n\x10\n\x01x\x12\x01y"\x08Identity'
    b'Z\x0f\n\x01x\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'b\x0f\n\x01y\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01'
    b'B\x04\n\x00\x10\x0b'
)


def _nvidia_pip_bin_dirs() -> list[str]:
    """Return bin/ directories of pip-installed nvidia-* packages."""
    dirs: list[str] = []
    for sp in sys.path:
        if not sp or not os.path.isdir(sp):
            continue
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for pkg in base.iterdir():
            bin_dir = pkg / "bin"
            if bin_dir.is_dir():
                dirs.append(str(bin_dir))
    return dirs


def _cuda_probe() -> bool:
    """Return True only when an InferenceSession actually binds to CUDA."""
    try:
        import onnxruntime as ort
        import numpy as np
    except Exception:
        return False

    try:
        import warnings

        # Suppress the default C++ logger so missing-DLL noise doesn't
        # hit stderr while we probe.  Also silence the Python-side
        # ``UserWarning: Specified provider ... is not in available
        # provider names`` when the CPU-only wheel is installed.
        ort.set_default_logger_severity(4)
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sess = ort.InferenceSession(
                _WARMUP_ONNX_BYTES,
                sess_options=opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            sess.run(None, {"x": np.array([0.0], dtype=np.float32)})
        providers = sess.get_providers()
        return bool(providers and providers[0] == "CUDAExecutionProvider")
    except Exception:
        return False


def _cuda_advertised() -> bool:
    """ORT says CUDAExecutionProvider is registered (onnxruntime-gpu installed)."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _has_nvidia_gpu() -> bool:
    """Use NVML (nvidia-ml-py) to detect a physical NVIDIA GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False


def _find_uv() -> Optional[str]:
    """Return the 'uv' executable path, or None if not on PATH."""
    for name in ("uv", "uv.exe"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _run_install_cmd(pkgs: list[str], description: str) -> bool:
    """Install ``pkgs`` via uv (preferred) or pip."""
    uv = _find_uv()
    if uv:
        cmd = [uv, "pip", "install"] + pkgs
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + pkgs

    print(
        f"{_c('cyan')}{description} ({'uv' if uv else 'pip'}) — "
        f"one-time download...{_c('reset')}"
    )
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(
            f"{_c('red')}Installation failed (exit {exc.returncode}).{_c('reset')}\n"
            f"You can retry manually later with:\n"
            f"    uv pip install --extra cuda"
        )
        return False


def _install_onnxruntime_gpu() -> bool:
    """Replace the CPU-only onnxruntime with the GPU wheel."""
    return _run_install_cmd(
        ["onnxruntime-gpu"],
        "Installing GPU-accelerated ONNX Runtime (~300 MB)",
    )


def _install_cuda_redist() -> bool:
    """Install NVIDIA CUDA runtime libraries (cublas, cuDNN, …)."""
    return _run_install_cmd(
        [
            "nvidia-cublas-cu12>=12.0",
            "nvidia-cuda-runtime-cu12>=12.0",
            "nvidia-cudnn-cu12>=9.0",
            "nvidia-cufft-cu12>=11.0",
        ],
        "Installing CUDA runtime libraries (~1 GB)",
    )


def ensure_cuda() -> None:
    """Entry point called from ``app.gui.app:main`` before Qt starts.

    Fast-path: if CUDA already works, return immediately.
    """
    if sys.platform != "win32":
        return

    # Inject pip-package bin/ directories into PATH before the very first
    # ``import onnxruntime``.  If the user already ran
    # ``uv pip install --extra cuda`` but hasn't restarted their shell,
    # this makes the DLLs discoverable without a restart.
    from app.utils import _try_inject_nvidia_pip_dll_paths

    _try_inject_nvidia_pip_dll_paths()

    # Fast path — CUDA already functional (common for dev machines with
    # system-wide CUDA Toolkit, or after the PATH injection above).
    if _cuda_probe():
        return

    # No physical NVIDIA GPU — nothing to accelerate.
    if not _has_nvidia_gpu():
        return

    # GPU есть, но CUDA не работает.  Либо стоит CPU-only onnxruntime,
    # либо onnxruntime-gpu установлен, но runtime DLL отсутствуют.
    print(
        f"\n{_c('yellow')}NVIDIA GPU detected, but GPU acceleration is not ready.{_c('reset')}\n"
        f"The model will run on CPU (slower) until GPU libraries are installed.\n"
    )

    if not sys.stdin.isatty():
        # Headless / CI — can't prompt; leave a log pointer.
        log.warning(
            "NVIDIA GPU present, CUDA libraries missing.  "
            "Install with: uv pip install --extra cuda"
        )
        return

    try:
        answer = input(
            f"{_c('bold')}Install GPU support automatically (~1.3 GB download)? [Y/n]: {_c('reset')}"
        )
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer.strip().lower() not in ("", "y", "yes"):
        print(
            f"{_c('yellow')}Skipped.  You can install later with:{_c('reset')}\n"
            f"    uv pip install --extra cuda\n"
        )
        log.warning("User declined CUDA auto-install; continuing on CPU.")
        return

    # 1) Убедиться, что стоит onnxruntime-gpu, а не CPU-версия.
    try:
        import importlib.metadata as _metadata
        _metadata.version("onnxruntime")
        _has_cpu = True
    except Exception:
        _has_cpu = False

    try:
        _metadata.version("onnxruntime-gpu")
        _has_gpu_wheel = True
    except Exception:
        _has_gpu_wheel = False

    if _has_cpu and not _has_gpu_wheel:
        if not _install_onnxruntime_gpu():
            log.warning("Failed to install onnxruntime-gpu; continuing on CPU.")
            return

    # 2) CUDA runtime DLL (cublas, cuDNN, …).
    if not _nvidia_pip_bin_dirs():
        if not _install_cuda_redist():
            log.warning("CUDA auto-install failed; continuing on CPU.")
            return

    # Re-inject the freshly installed paths and re-probe.
    _try_inject_nvidia_pip_dll_paths()
    if _cuda_probe():
        print(
            f"{_c('green')}GPU acceleration enabled — the model will load on CUDA.{_c('reset')}\n"
        )
        log.info("CUDA ready; GPU enabled.")
    else:
        print(
            f"{_c('yellow')}Packages installed but CUDA still not functional. "
            f"A process restart may be required.{_c('reset')}\n"
        )
        log.warning("CUDA packages installed but probe still fails; needs restart.")
