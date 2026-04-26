"""Diagnostic: same as ``diag_parakeet.py`` but with NVML initialised
BEFORE the NeMo import.

Hypothesis: the production app's pyarrow segfault during
``import nemo.collections.asr`` is caused by the CUDA runtime that
NVML loads on startup (via ``app.resource_monitor.ResourceMonitor.start``)
clashing with the CUDA libs lhotse / pyarrow expect at import time.

If this script segfaults, the hypothesis holds — fix is to either
delay NVML init until after NeMo loads, or do NeMo in a subprocess.

If this script reaches ALL DONE, the cause is something else
(maybe Qt's OpenGL / D3D context, audio driver, …).
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
        stamp("patched signal.SIGKILL = SIGTERM (Windows fallback)")

    # Initialize NVML the same way the resource monitor does — this is
    # the suspected trigger for the production segfault. If pyarrow
    # crashes after NVML has loaded the CUDA driver, the hypothesis
    # holds.
    stamp("initialising NVML (same as resource_monitor)...")
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        stamp(f"NVML initialised, watching {device_count} GPU(s)")
    except Exception as exc:
        stamp(f"NVML init failed (continuing without it): {exc}")

    stamp("python startup OK; importing nemo_toolkit (cold)...")
    import nemo.collections.asr as nemo_asr  # type: ignore

    stamp(f"nemo_toolkit imported (version: {getattr(nemo_asr, '__version__', '?')})")

    stamp("calling ASRModel.from_pretrained for Parakeet TDT v3...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
    )
    stamp("from_pretrained returned")

    stamp("trying change_attention_model(rel_pos_local_attn, [256, 256])...")
    try:
        model.change_attention_model(
            "rel_pos_local_attn",
            att_context_size=[256, 256],
        )
        stamp("change_attention_model OK")
    except Exception as exc:
        stamp(f"change_attention_model raised: {exc!r}")

    stamp("ALL DONE - Parakeet TDT v3 is loaded and ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
