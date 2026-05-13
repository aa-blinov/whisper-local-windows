"""Tests for ``app.cuda_bootstrap`` — the Windows+NVIDIA GPU auto-setup flow.

All real network / hardware probes are mocked out so the suite runs
cleanly on CI boxes without a GPU or CUDA installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_ort(cuda_works: bool = True):
    """Return a minimal fake ``onnxruntime`` module."""
    fake = types.ModuleType("onnxruntime")
    fake.set_default_logger_severity = lambda _lvl: None
    opts = MagicMock()
    fake.SessionOptions = lambda: opts

    class _FakeSession:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, *_a, **_kw):
            return [None]

        @staticmethod
        def get_providers():
            if cuda_works:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    fake.InferenceSession = _FakeSession
    fake.get_available_providers = lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return fake


def _make_fake_np():
    fake = types.ModuleType("numpy")
    fake.float32 = "float32"
    fake.array = lambda *a, **kw: [0.0]
    return fake


# ---------------------------------------------------------------------------
# _nvidia_pip_bin_dirs
# ---------------------------------------------------------------------------


def test_nvidia_pip_bin_dirs_finds_bin_subdirectories(tmp_path, monkeypatch):
    # Isolate from the real venv so we only see the fake tree.
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    base = tmp_path / "nvidia"
    (base / "cublas" / "bin").mkdir(parents=True)
    (base / "cudnn" / "bin").mkdir(parents=True)
    (base / "cudnn" / "lib").mkdir(parents=True)  # should be ignored

    from app.cuda_bootstrap import _nvidia_pip_bin_dirs

    dirs = _nvidia_pip_bin_dirs()
    assert len(dirs) == 2
    assert any("cublas" in d and "bin" in d for d in dirs)
    assert any("cudnn" in d and "bin" in d for d in dirs)


def test_nvidia_pip_bin_dirs_returns_empty_when_none_present(monkeypatch):
    monkeypatch.setattr(sys, "path", [])

    from app.cuda_bootstrap import _nvidia_pip_bin_dirs

    assert _nvidia_pip_bin_dirs() == []


# ---------------------------------------------------------------------------
# _cuda_probe
# ---------------------------------------------------------------------------


def test_cuda_probe_returns_true_when_session_binds_to_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", _make_fake_ort(cuda_works=True))
    monkeypatch.setitem(sys.modules, "numpy", _make_fake_np())

    from app.cuda_bootstrap import _cuda_probe

    assert _cuda_probe() is True


def test_cuda_probe_returns_false_when_session_fails(monkeypatch):
    fake_ort = _make_fake_ort(cuda_works=False)
    fake_ort.InferenceSession = MagicMock(side_effect=RuntimeError("CUDA missing"))
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "numpy", _make_fake_np())

    from app.cuda_bootstrap import _cuda_probe

    assert _cuda_probe() is False


def test_cuda_probe_returns_false_when_import_fails(monkeypatch):
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    from app.cuda_bootstrap import _cuda_probe

    assert _cuda_probe() is False


# ---------------------------------------------------------------------------
# _cuda_advertised
# ---------------------------------------------------------------------------


def test_cuda_advertised_true_when_cuda_in_available_providers(monkeypatch):
    fake_ort = _make_fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    from app.cuda_bootstrap import _cuda_advertised

    assert _cuda_advertised() is True


def test_cuda_advertised_false_when_only_cpu(monkeypatch):
    fake_ort = _make_fake_ort()
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    from app.cuda_bootstrap import _cuda_advertised

    assert _cuda_advertised() is False


# ---------------------------------------------------------------------------
# _has_nvidia_gpu
# ---------------------------------------------------------------------------


def test_has_nvidia_gpu_true_when_nvml_reports_devices(monkeypatch):
    fake_nvml = types.ModuleType("pynvml")
    fake_nvml.nvmlInit = lambda: None
    fake_nvml.nvmlDeviceGetCount = lambda: 2
    fake_nvml.nvmlShutdown = lambda: None
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    from app.cuda_bootstrap import _has_nvidia_gpu

    assert _has_nvidia_gpu() is True


def test_has_nvidia_gpu_false_when_nvml_unavailable(monkeypatch):
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    real_import = __import__

    def raising_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", raising_import)

    from app.cuda_bootstrap import _has_nvidia_gpu

    assert _has_nvidia_gpu() is False


# ---------------------------------------------------------------------------
# _find_uv
# ---------------------------------------------------------------------------


def test_find_uv_returns_path_when_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name in ("uv", "uv.exe") else None)

    from app.cuda_bootstrap import _find_uv

    assert _find_uv() == "/usr/bin/uv"


def test_find_uv_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)

    from app.cuda_bootstrap import _find_uv

    assert _find_uv() is None


# ---------------------------------------------------------------------------
# _run_install_cmd
# ---------------------------------------------------------------------------


def test_run_install_cmd_uses_uv_when_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "uv.exe" if name in ("uv", "uv.exe") else None)
    captured: list[list[str]] = []

    def capture_subprocess(cmd, **kwargs):
        captured.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", capture_subprocess)

    from app.cuda_bootstrap import _run_install_cmd

    assert _run_install_cmd(["onnxruntime-gpu"], "Installing GPU wheel") is True
    assert captured
    assert captured[0][0] == "uv.exe"
    assert captured[0][1] == "pip"
    assert captured[0][2] == "install"
    assert "onnxruntime-gpu" in captured[0]


def test_run_install_cmd_falls_back_to_pip_when_uv_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    captured: list[list[str]] = []

    def capture_subprocess(cmd, **kwargs):
        captured.append(cmd)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", capture_subprocess)

    from app.cuda_bootstrap import _run_install_cmd

    assert _run_install_cmd(["pkg"], "Installing") is True
    assert captured[0][0] == sys.executable
    assert captured[0][1] == "-m"
    assert captured[0][2] == "pip"


def test_run_install_cmd_returns_false_on_failure(monkeypatch):
    from subprocess import CalledProcessError

    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: (_ for _ in ()).throw(CalledProcessError(1, "pip")),
    )

    from app.cuda_bootstrap import _run_install_cmd

    assert _run_install_cmd(["pkg"], "Installing") is False


# ---------------------------------------------------------------------------
# ensure_cuda — high-level orchestration
# ---------------------------------------------------------------------------


def _patch_ensure_cuda_deps(monkeypatch, *, platform="win32", cuda_works=False, has_gpu=False, tty=False):
    """Common monkeypatch set for ensure_cuda tests."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    from app import cuda_bootstrap as _mod
    from app import utils as _utils_mod

    monkeypatch.setattr(_mod, "_cuda_probe", lambda: cuda_works)
    monkeypatch.setattr(_mod, "_has_nvidia_gpu", lambda: has_gpu)
    monkeypatch.setattr(_utils_mod, "_try_inject_nvidia_pip_dll_paths", lambda: None)
    monkeypatch.setattr(_mod, "_nvidia_pip_bin_dirs", lambda: [])

    # By default pretend no packages are installed.
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: (_ for _ in ()).throw(Exception("not installed")),
    )
    return _mod


def test_ensure_cuda_short_circuits_on_non_windows(monkeypatch, caplog):
    """macOS / Linux must bail out immediately — no prompts, no installs."""
    _patch_ensure_cuda_deps(monkeypatch, platform="darwin", cuda_works=False, has_gpu=True)

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()
    # No errors, no side effects.


def test_ensure_cuda_short_circuits_when_cuda_already_works(monkeypatch):
    """Fast path: CUDA functional → nothing to do."""
    _patch_ensure_cuda_deps(monkeypatch, platform="win32", cuda_works=True, has_gpu=True)

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()  # must not raise or prompt


def test_ensure_cuda_short_circuits_when_no_gpu(monkeypatch):
    """No NVIDIA hardware → silently continue on CPU."""
    _patch_ensure_cuda_deps(monkeypatch, platform="win32", cuda_works=False, has_gpu=False)

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()


def test_ensure_cuda_warns_and_returns_when_not_tty(monkeypatch, caplog):
    """Headless / CI: can't prompt, so log a pointer and continue."""
    _patch_ensure_cuda_deps(
        monkeypatch, platform="win32", cuda_works=False, has_gpu=True, tty=False
    )

    from app.cuda_bootstrap import ensure_cuda

    with caplog.at_level("WARNING"):
        ensure_cuda()

    assert "uv pip install --extra cuda" in caplog.text


def test_ensure_cuda_prompts_and_skips_when_user_declines(monkeypatch, capsys):
    """User types 'n' → skip install, show helpful fallback message."""
    _patch_ensure_cuda_deps(
        monkeypatch, platform="win32", cuda_works=False, has_gpu=True, tty=True
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()
    captured = capsys.readouterr()
    assert "Skipped" in captured.out or "skipped" in captured.out.lower()


def test_ensure_cuda_installs_gpu_wheel_when_only_cpu_present(monkeypatch):
    """CPU-only onnxruntime installed, user says yes → swap to GPU wheel."""
    _patch_ensure_cuda_deps(
        monkeypatch, platform="win32", cuda_works=False, has_gpu=True, tty=True
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    # Pretend CPU-only onnxruntime is present, GPU wheel is not.
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: (
            "1.0"
            if name == "onnxruntime"
            else (_ for _ in ()).throw(Exception("not installed"))
        ),
    )

    from app import cuda_bootstrap as _mod

    installed: list[str] = []
    monkeypatch.setattr(_mod, "_install_onnxruntime_gpu", lambda: installed.append("gpu") or True)
    monkeypatch.setattr(_mod, "_install_cuda_redist", lambda: installed.append("redist") or True)

    # After install the probe should succeed.
    probe_results = [False, True]  # first fails, second succeeds
    monkeypatch.setattr(_mod, "_cuda_probe", lambda: probe_results.pop(0))

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()
    assert "gpu" in installed
    assert "redist" in installed


def test_ensure_cuda_installs_redist_when_gpu_wheel_already_present(monkeypatch):
    """onnxruntime-gpu already installed, but runtime DLLs missing → only
    install the redistributables."""
    _patch_ensure_cuda_deps(
        monkeypatch, platform="win32", cuda_works=False, has_gpu=True, tty=True
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    # Pretend onnxruntime-gpu is already present.
    monkeypatch.setattr(
        "importlib.metadata.version",
        lambda name: (
            "1.0"
            if name == "onnxruntime-gpu"
            else (_ for _ in ()).throw(Exception("not installed"))
        ),
    )

    from app import cuda_bootstrap as _mod

    installed: list[str] = []
    monkeypatch.setattr(_mod, "_install_onnxruntime_gpu", lambda: installed.append("gpu") or True)
    monkeypatch.setattr(_mod, "_install_cuda_redist", lambda: installed.append("redist") or True)

    probe_results = [False, True]
    monkeypatch.setattr(_mod, "_cuda_probe", lambda: probe_results.pop(0))

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()
    assert "gpu" not in installed  # already there, no reinstall needed
    assert "redist" in installed


def test_ensure_cuda_surfaces_restart_hint_when_probe_still_fails(monkeypatch, capsys):
    """Packages installed but CUDA still not functional → tell user to restart."""
    _patch_ensure_cuda_deps(
        monkeypatch, platform="win32", cuda_works=False, has_gpu=True, tty=True
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    from app import cuda_bootstrap as _mod

    monkeypatch.setattr(_mod, "_install_onnxruntime_gpu", lambda: True)
    monkeypatch.setattr(_mod, "_install_cuda_redist", lambda: True)
    # Probe still fails even after install.
    monkeypatch.setattr(_mod, "_cuda_probe", lambda: False)

    from app.cuda_bootstrap import ensure_cuda

    ensure_cuda()
    captured = capsys.readouterr()
    assert "restart" in captured.out.lower()
