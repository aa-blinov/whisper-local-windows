"""Central model registry.

Single source of truth mapping user-facing aliases to canonical Hugging Face
model IDs, together with rich metadata used by the UI (size, VRAM, speed,
quality tier, language support, description, recommended ``compute_type``).

Public API:
- ``ModelInfo``: metadata for a single model preset
- ``MODELS``: ordered tuple of all supported presets
- ``aliases()``: list of aliases in display order
- ``get_model(alias)``: ``ModelInfo`` lookup, ``KeyError`` if unknown
- ``ALIAS_TO_MODEL`` / ``MODEL_TO_ALIAS``: backward-compatible mappings
- ``canonical_for(x)`` / ``alias_for(x)``: string helpers

Several aliases can share the same canonical Hugging Face id and only
differ by ``compute_type`` (``float16`` vs ``int8_float16``) — that's how
"quantized" cards are exposed to the user without re-uploading weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


_SPEED_VALUES = ("fast", "medium", "slow")
_QUALITY_VALUES = ("basic", "good", "excellent")
# What ctranslate2 accepts for ``compute_type``. We restrict to the four
# values that actually make sense for inference.
_COMPUTE_VALUES = ("float32", "float16", "int8_float16", "int8")


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    canonical: str
    display_name: str
    size_mb: int
    vram_gb: float
    speed: str
    quality: str
    languages: str
    description: str
    compute_type: str = "float16"

    def __post_init__(self) -> None:
        if self.speed not in _SPEED_VALUES:
            raise ValueError(
                f"speed must be one of {_SPEED_VALUES}, got {self.speed!r}"
            )
        if self.quality not in _QUALITY_VALUES:
            raise ValueError(
                f"quality must be one of {_QUALITY_VALUES}, got {self.quality!r}"
            )
        if self.compute_type not in _COMPUTE_VALUES:
            raise ValueError(
                f"compute_type must be one of {_COMPUTE_VALUES}, got {self.compute_type!r}"
            )


MODELS: Tuple[ModelInfo, ...] = (
    # ---- TEMPORARY: tiny preset for testing the download progress UI -------
    # Remove this entry once verified.
    ModelInfo(
        alias="tiny",
        canonical="Systran/faster-whisper-tiny",
        display_name="Tiny (test)",
        size_mb=75,
        vram_gb=1.0,
        speed="fast",
        quality="basic",
        languages="multilingual",
        description="Temporary card for testing the download progress UI.",
        compute_type="float16",
    ),
    # ---- Distilled / turbo (faster, near-large quality) ---------------------
    ModelInfo(
        alias="turbo",
        canonical="deepdml/faster-whisper-large-v3-turbo-ct2",
        display_name="Large v3 Turbo",
        size_mb=1620,
        vram_gb=6.0,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description="Distilled large-v3 — much faster than the full model with similar quality.",
        compute_type="float16",
    ),
    ModelInfo(
        alias="turbo-int8",
        canonical="deepdml/faster-whisper-large-v3-turbo-ct2",
        display_name="Large v3 Turbo (int8)",
        size_mb=1620,
        vram_gb=3.5,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description="Quantized turbo — half the VRAM, slight quality dip. Great on 4–6 GB GPUs.",
        compute_type="int8_float16",
    ),
    ModelInfo(
        alias="distil-large-v3",
        canonical="Systran/faster-distil-whisper-large-v3",
        display_name="Distil Large v3",
        size_mb=1510,
        vram_gb=5.0,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description="6× faster than large-v3, ~1% WER drop. English-leaning.",
        compute_type="float16",
    ),
    # ---- Full large-v3 ------------------------------------------------------
    ModelInfo(
        alias="large-v3",
        canonical="Systran/faster-whisper-large-v3",
        display_name="Large v3",
        size_mb=3145,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Latest large model. Best overall quality.",
        compute_type="float16",
    ),
    ModelInfo(
        alias="large-v3-int8",
        canonical="Systran/faster-whisper-large-v3",
        display_name="Large v3 (int8)",
        size_mb=3145,
        vram_gb=5.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Quantized large-v3 — same accuracy on most prompts, half the VRAM.",
        compute_type="int8_float16",
    ),
    # ---- Russian fine-tunes -------------------------------------------------
    ModelInfo(
        alias="large-v3-ru",
        canonical="bzikst/faster-whisper-large-v3-russian",
        display_name="Large v3 — Russian fine-tune",
        size_mb=3090,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="Russian (fine-tuned)",
        description="large-v3 fine-tuned on Common Voice RU — WER 6.39 vs 9.84.",
        compute_type="float16",
    ),
    ModelInfo(
        alias="large-v3-ru-int8",
        canonical="bzikst/faster-whisper-large-v3-russian",
        display_name="Large v3 — Russian (int8)",
        size_mb=3090,
        vram_gb=5.0,
        speed="slow",
        quality="excellent",
        languages="Russian (fine-tuned)",
        description="Quantized Russian fine-tune. Best Russian quality on a 6 GB GPU.",
        compute_type="int8_float16",
    ),
)


_BY_ALIAS = {m.alias: m for m in MODELS}


def aliases() -> List[str]:
    return [m.alias for m in MODELS]


def get_model(alias: str) -> ModelInfo:
    if alias not in _BY_ALIAS:
        raise KeyError(alias)
    return _BY_ALIAS[alias]


# ``ALIAS_TO_MODEL`` maps alias → canonical. ``MODEL_TO_ALIAS`` is the
# reverse — first alias wins when several presets share a canonical id.
ALIAS_TO_MODEL = {m.alias: m.canonical for m in MODELS}

MODEL_TO_ALIAS: dict[str, str] = {}
for _m in MODELS:
    MODEL_TO_ALIAS.setdefault(_m.canonical, _m.alias)


def canonical_for(name: str) -> str:
    return ALIAS_TO_MODEL.get(name, name)


def alias_for(canonical: str) -> str:
    return MODEL_TO_ALIAS.get(canonical, canonical)
