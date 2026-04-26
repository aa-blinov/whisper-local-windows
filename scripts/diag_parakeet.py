"""Diagnostic: time the cold-import of NeMo and the Parakeet TDT v3
download, the same way ``NemoBackend._do_load`` does it but without
GUI / threading / state machinery.

Each phase prints its wall-clock duration so we can tell which step
blocks the in-app load — the cold import (lhotse / hydra / lightning /
torch CUDA init / numba JIT) or the actual ``from_pretrained`` (which
internally calls ``huggingface_hub.snapshot_download``).
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    t0 = time.monotonic()

    def stamp(msg: str) -> None:
        # Flush so log lines stream out in real time even when stdout
        # is captured by the test runner / background bash wrapper.
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    # Mirror the production patch — see NemoBackend._patch_signal_for_windows.
    # NeMo's exp_manager touches ``signal.SIGKILL`` at class-definition time,
    # which is POSIX-only; without this stub the import dies on Windows.
    import signal as _signal

    if not hasattr(_signal, "SIGKILL"):
        _signal.SIGKILL = _signal.SIGTERM  # type: ignore[attr-defined]
        stamp("patched signal.SIGKILL = SIGTERM (Windows fallback)")

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

    stamp("ALL DONE — Parakeet TDT v3 is loaded and ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
