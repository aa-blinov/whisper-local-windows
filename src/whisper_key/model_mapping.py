"""Central model mapping utilities.

We keep a single source of truth for mapping between short aliases presented to the user
and canonical model identifiers expected by the backend.

Public helpers:
- ALIAS_TO_MODEL: dict alias->canonical
- MODEL_TO_ALIAS: reverse map (first alias wins)
- canonical_for(x): returns canonical model id for alias or canonical if already canonical
- alias_for(canonical): returns preferred alias for canonical, or canonical if unknown
"""
from __future__ import annotations

ALIAS_TO_MODEL = {
    # "tiny.en": "Systran/faster-whisper-tiny.en",
    # "tiny": "Systran/faster-whisper-tiny",
    # "base.en": "Systran/faster-whisper-base.en",
    # "base": "Systran/faster-whisper-base",
    # "small.en": "Systran/faster-whisper-small.en",
    # "small": "Systran/faster-whisper-small",
    # "medium.en": "Systran/faster-whisper-medium.en",
    # "medium": "Systran/faster-whisper-medium",
    # "large-v1": "Systran/faster-whisper-large-v1",
    # "large-v2": "Systran/faster-whisper-large-v2",
    # "large-v3": "Systran/faster-whisper-large-v3",
    # "large": "Systran/faster-whisper-large-v3",
    # "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    # "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    # "distil-small.en": "Systran/faster-distil-whisper-small.en",
    # "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    # "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    # "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

MODEL_TO_ALIAS = {}
for a, c in ALIAS_TO_MODEL.items():  # first alias wins if duplicates
    MODEL_TO_ALIAS.setdefault(c, a)


def canonical_for(name: str) -> str:
    return ALIAS_TO_MODEL.get(name, name)


def alias_for(canonical: str) -> str:
    return MODEL_TO_ALIAS.get(canonical, canonical)
