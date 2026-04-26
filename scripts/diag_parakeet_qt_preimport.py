"""Diagnostic: pre-import pyarrow BEFORE QApplication, then NeMo.

If this passes ALL DONE, the fix for the production segfault is to
add the same pre-import to ``app/gui/app.py`` before
``QApplication([])``.
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

    stamp("pre-importing pyarrow (force arrow.dll load BEFORE Qt)...")
    try:
        import pyarrow  # noqa: F401
        stamp(f"pyarrow {pyarrow.__version__} imported")
    except ImportError as exc:
        stamp(f"pyarrow not available: {exc}")

    stamp("creating PySide6 QApplication...")
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
    stamp("ALL DONE - Qt + NeMo coexist after pyarrow pre-import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
