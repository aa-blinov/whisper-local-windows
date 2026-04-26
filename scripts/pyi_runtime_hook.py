"""PyInstaller runtime hook — applied before any user code in the frozen exe.

PyInstaller wires this in via the spec file's ``runtime_hooks=[...]``
parameter. It runs in the bootloader phase, before
``lazy-to-text-ui.py`` (the entry script) is invoked, so any patch
applied here is in place before NumPy / NeMo / lhotse / pyarrow
get imported.

Three compatibility shims are needed for our Windows + NumPy 2.x +
windowed-build target:

1. ``signal.SIGKILL = signal.SIGTERM`` — NeMo's exp_manager touches
   ``signal.SIGKILL`` at class-definition time, but SIGKILL is
   POSIX-only. Without the shim, the very first
   ``import nemo.collections.asr`` inside the frozen exe raises
   ``AttributeError`` and the bootloader exits non-zero.
2. ``np.sctypes`` — removed in NumPy 2.0, but lhotse / older librosa
   code paths still touch it. Without the shim, transcribe fails
   with the same AttributeError every recording.
3. ``sys.stdout`` / ``sys.stderr`` fallback — windowed PyInstaller
   builds (``console=False``) get ``sys.stdout`` and ``sys.stderr``
   set to ``None``. GigaAM's loader / tqdm-using libraries call
   ``sys.stdout.write(...)`` and crash with
   ``'NoneType' object has no attribute 'write'``. Replace ``None``
   streams with a discarding writer so library code that prints
   doesn't crash the load path.

The shims are also applied at build time (top of ``lazy_to_text.spec``)
so PyInstaller's hidden-imports analysis can ``find_spec('nemo.*')``
without crashing.
"""

import signal
import sys


if not hasattr(signal, "SIGKILL"):
    # SIGTERM is the closest POSIX equivalent that exists on Windows;
    # the value is never actually delivered (NeMo's fault-tolerance
    # subsystem isn't exercised by our single-process inference path),
    # so all that matters is the dataclass default has SOMETHING that
    # passes ``isinstance(..., signal.Signals)``.
    signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]


try:
    import numpy as _np

    if not hasattr(_np, "sctypes"):
        _np.sctypes = {  # type: ignore[attr-defined]
            "int": [_np.int8, _np.int16, _np.int32, _np.int64],
            "uint": [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
            "float": [_np.float16, _np.float32, _np.float64],
            "complex": [_np.complex64, _np.complex128],
            "others": [bool, object, bytes, str, _np.void],
        }
except ImportError:
    # NumPy not installed — Whisper-only build with neither GigaAM
    # nor NeMo. Nothing else here is going to work anyway, but
    # don't crash the bootloader on the way out.
    pass


class _NullStream:
    """Discards everything written to it — replacement for ``sys.stdout``
    / ``sys.stderr`` when PyInstaller's bootloader nulls them out in
    windowed builds. Library code (gigaam / tqdm / urllib download
    progress) doesn't get to crash on ``None.write(...)``."""

    encoding = "utf-8"

    def write(self, _data) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError("no fileno on null stream")


if sys.stdout is None:
    sys.stdout = _NullStream()  # type: ignore[assignment]
if sys.stderr is None:
    sys.stderr = _NullStream()  # type: ignore[assignment]
