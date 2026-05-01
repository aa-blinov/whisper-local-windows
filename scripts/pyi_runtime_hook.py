"""PyInstaller runtime hook — applied before any user code in the frozen exe.

PyInstaller wires this in via the spec file's ``runtime_hooks=[…]``
parameter.  It runs in the bootloader phase, before
``lazy-to-text-ui.py`` (the entry script) is invoked, so any patch
applied here is in place before NumPy / huggingface_hub / onnx_asr
/ Qt get imported.

Two compatibility shims for the Windows + windowed-build target:

1. ``sys.stdout`` / ``sys.stderr`` fallback — windowed PyInstaller
   builds (``console=False``) have ``sys.stdout`` and ``sys.stderr``
   set to ``None``.  ``huggingface_hub`` (and any tqdm-using
   library) calls ``sys.stdout.write(…)`` for download progress
   and crashes the bootloader with
   ``'NoneType' object has no attribute 'write'``.  Replace the
   ``None`` streams with a discarding writer so library code that
   prints doesn't take down the launch.

2. ``CREATE_NO_WINDOW`` for child processes on Windows — pyperclip
   / pyautogui spawn helper processes via ``subprocess.Popen``;
   without the flag each spawn flashes a visible ``cmd.exe`` window
   in front of the user (especially noticeable during transcribe —
   every autopaste blinks).  We monkey-patch ``Popen.__init__`` so
   the flag is added by default; explicit callers that pass their
   own ``creationflags`` override us, which is the correct
   precedence.

The legacy NeMo / lhotse / GigaAM-Python shims (``signal.SIGKILL``
re-creation, ``np.sctypes`` rebuild) were retired with the
migration to ``onnx-asr`` — none of those wheels are in the
dependency set anymore.
"""

import sys


class _NullStream:
    """Discards everything written to it.

    Drop-in for ``sys.stdout`` / ``sys.stderr`` when PyInstaller's
    bootloader nulls them out in windowed builds.  Library code
    (``huggingface_hub`` download progress, ``tqdm`` bars,
    ``urllib`` warnings) never gets to crash on
    ``None.write(…)``.
    """

    encoding = "utf-8"

    def write(self, _data) -> int:
        return 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # Some libraries probe ``fileno()`` to decide whether to use
        # OS-level pipes vs Python-level streams.  Raising OSError
        # tells them "no underlying fd, fall back to pure-Python
        # buffering".  Do NOT return a fake fd — the consumer will
        # try to ``os.write(fd, …)`` and corrupt unrelated handles.
        raise OSError("no fileno on null stream")


if sys.stdout is None:
    sys.stdout = _NullStream()  # type: ignore[assignment]
if sys.stderr is None:
    sys.stderr = _NullStream()  # type: ignore[assignment]


# Suppress subprocess console flashes on Windows.
if sys.platform == "win32":
    import subprocess as _subprocess

    _CREATE_NO_WINDOW = 0x08000000
    _orig_popen_init = _subprocess.Popen.__init__

    def _silent_popen_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "creationflags" not in kwargs or kwargs["creationflags"] is None:
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        return _orig_popen_init(self, *args, **kwargs)

    _subprocess.Popen.__init__ = _silent_popen_init  # type: ignore[assignment]
