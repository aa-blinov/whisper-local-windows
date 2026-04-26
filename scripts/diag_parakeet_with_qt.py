"""Diagnostic: same as ``diag_parakeet.py`` but with PySide6
QApplication created BEFORE the NeMo import — to test if Qt's
DLL graph (Qt6Core, Qt6Gui, OpenGL, ...) clashes with pyarrow's
arrow.dll on Windows.

The earlier diag (no Qt) succeeds. The production app, which boots
Qt before letting the NemoBackend worker run its import, segfaults
inside arrow.dll. Isolate Qt as the cause.
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    t0 = time.monotonic()

    def stamp(msg: str) -> None:
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    import signal as _signal
    if not hasattr(_signal, "SIGKILL"):
        _signal.SIGKILL = _signal.SIGTERM  # type: ignore[attr-defined]

    stamp("creating PySide6 QApplication (mirrors app startup)...")
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    stamp("QApplication created")

    stamp("importing nemo_toolkit...")
    import nemo.collections.asr as nemo_asr  # type: ignore

    stamp(f"nemo_toolkit imported (v{getattr(nemo_asr, '__version__', '?')})")

    stamp("calling ASRModel.from_pretrained for Parakeet TDT v3...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
    )
    stamp("from_pretrained returned")

    stamp("ALL DONE - Parakeet loaded with QApplication alive")
    # Don't actually enter the Qt event loop — we just wanted to
    # exercise the DLL load order.
    return 0


if __name__ == "__main__":
    sys.exit(main())
