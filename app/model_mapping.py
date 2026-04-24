"""Central model registry.

Single source of truth mapping user-facing aliases to canonical Hugging Face
model IDs, together with rich metadata used by the UI (size, VRAM, speed,
quality tier, language support, description).

Public API:
- ``ModelInfo``: metadata for a single model
- ``MODELS``: ordered tuple of all supported models
- ``aliases()``: list of aliases in display order
- ``get_model(alias)``: ``ModelInfo`` lookup, ``KeyError`` if unknown
- ``ALIAS_TO_MODEL`` / ``MODEL_TO_ALIAS``: backward-compatible mappings
- ``canonical_for(x)`` / ``alias_for(x)``: string helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


_SPEED_VALUES = ("fast", "medium", "slow")
_QUALITY_VALUES = ("basic", "good", "excellent")


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

    def __post_init__(self) -> None:
        if self.speed not in _SPEED_VALUES:
            raise ValueError(
                f"speed must be one of {_SPEED_VALUES}, got {self.speed!r}"
            )
        if self.quality not in _QUALITY_VALUES:
            raise ValueError(
                f"quality must be one of {_QUALITY_VALUES}, got {self.quality!r}"
            )


MODELS: Tuple[ModelInfo, ...] = (
    ModelInfo(
        alias="tiny",
        canonical="Systran/faster-whisper-tiny",
        display_name="Tiny",
        size_mb=75,
        vram_gb=1.0,
        speed="fast",
        quality="basic",
        languages="multilingual",
        description="Smallest model. Fast and light, good for quick drafts.",
    ),
    ModelInfo(
        alias="base",
        canonical="Systran/faster-whisper-base",
        display_name="Base",
        size_mb=145,
        vram_gb=1.0,
        speed="fast",
        quality="basic",
        languages="multilingual",
        description="Lightweight general-purpose model with decent accuracy.",
    ),
    ModelInfo(
        alias="small",
        canonical="Systran/faster-whisper-small",
        display_name="Small",
        size_mb=480,
        vram_gb=2.0,
        speed="medium",
        quality="good",
        languages="multilingual",
        description="Balanced speed and quality for everyday transcription.",
    ),
    ModelInfo(
        alias="medium",
        canonical="Systran/faster-whisper-medium",
        display_name="Medium",
        size_mb=1500,
        vram_gb=5.0,
        speed="medium",
        quality="good",
        languages="multilingual",
        description="Higher accuracy, still reasonable on modern GPUs.",
    ),
    ModelInfo(
        alias="large-v2",
        canonical="Systran/faster-whisper-large-v2",
        display_name="Large v2",
        size_mb=3000,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Large multilingual model with excellent quality.",
    ),
    ModelInfo(
        alias="large-v3",
        canonical="Systran/faster-whisper-large-v3",
        display_name="Large v3",
        size_mb=3000,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Latest large model. Best overall quality.",
    ),
)


_BY_ALIAS = {m.alias: m for m in MODELS}


def aliases() -> List[str]:
    return [m.alias for m in MODELS]


def get_model(alias: str) -> ModelInfo:
    if alias not in _BY_ALIAS:
        raise KeyError(alias)
    return _BY_ALIAS[alias]


ALIAS_TO_MODEL = {m.alias: m.canonical for m in MODELS}

MODEL_TO_ALIAS: dict[str, str] = {}
for _m in MODELS:
    MODEL_TO_ALIAS.setdefault(_m.canonical, _m.alias)


def canonical_for(name: str) -> str:
    return ALIAS_TO_MODEL.get(name, name)


def alias_for(canonical: str) -> str:
    return MODEL_TO_ALIAS.get(canonical, canonical)
