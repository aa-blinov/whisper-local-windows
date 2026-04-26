"""PyInstaller runtime hook — applied before any user code in the frozen exe.

PyInstaller wires this in via the spec file's ``runtime_hooks=[...]``
parameter. It runs in the bootloader phase, before
``lazy-to-text-ui.py`` (the entry script) is invoked, so any patch
applied here is in place before NumPy / NeMo / lhotse / pyarrow
get imported.

Two compatibility shims are needed for our Windows + NumPy 2.x
target:

1. ``signal.SIGKILL = signal.SIGTERM`` — NeMo's exp_manager touches
   ``signal.SIGKILL`` at class-definition time, but SIGKILL is
   POSIX-only. Without the shim, the very first
   ``import nemo.collections.asr`` inside the frozen exe raises
   ``AttributeError`` and the bootloader exits non-zero.
2. ``np.sctypes`` — removed in NumPy 2.0, but lhotse / older librosa
   code paths still touch it. Without the shim, transcribe fails
   with the same AttributeError every recording.

The shims are also applied at build time (top of ``lazy_to_text.spec``)
so PyInstaller's hidden-imports analysis can ``find_spec('nemo.*')``
without crashing.
"""

import signal


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
