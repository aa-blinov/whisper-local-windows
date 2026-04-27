"""Central model registry.

Single source of truth mapping user-facing aliases to canonical Hugging
Face model IDs, together with rich metadata used by the UI (size, VRAM,
speed, quality tier, language support, description, recommended
``compute_type``).

The app is **ONNX-only** — every model here is an ONNX-exported variant
loaded by ``OnnxAsrBackend``.  Older heterogeneous backends (NeMo,
GigaAM-Python, faster-whisper) were dropped; the ONNX equivalents
provide identical accuracy at a fraction of the install size.

Public API:
- ``ModelInfo``: metadata for a single model preset
- ``MODELS``: ordered tuple of all supported presets
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
# Precision label for the model card UI.  ONNX picks precision via
# the ``quantization`` parameter to ``onnx_asr.load_model`` (one of
# ``int8`` / ``fp16`` / ``None``).  The mapping happens in
# ``OnnxAsrBackend.__init__`` based on this hint.
_COMPUTE_VALUES = ("float32", "float16", "int8")
# Single-engine app: every model goes through ``OnnxAsrBackend``.  The
# ``family`` field tells the backend which onnx-asr behaviour to use
# (Whisper takes a language kwarg, others don't; GigaAM is RU-only,
# Parakeet auto-detects, …).
BACKEND_KINDS = ("onnx_asr",)
ONNX_FAMILIES = ("whisper", "gigaam", "parakeet")
# Visual grouping shown on the card.
FAMILIES = (
    "Whisper",
    "Whisper Turbo",
    "GigaAM",
    "Parakeet",
    "T-One",
    "Vosk",
    "Canary",
)


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
    backend_kind: str = "onnx_asr"
    family: str = "Whisper"
    # Which onnx-asr family adapter to use.  Drives backend behaviour
    # (language passing, language reporting).  Independent of the UI
    # ``family`` label which is purely cosmetic.
    onnx_family: str = "whisper"

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
        if self.backend_kind not in BACKEND_KINDS:
            raise ValueError(
                f"backend_kind must be one of {BACKEND_KINDS}, got {self.backend_kind!r}"
            )
        if self.family not in FAMILIES:
            raise ValueError(
                f"family must be one of {FAMILIES}, got {self.family!r}"
            )
        if self.onnx_family not in ONNX_FAMILIES:
            raise ValueError(
                f"onnx_family must be one of {ONNX_FAMILIES}, got {self.onnx_family!r}"
            )


MODELS: Tuple[ModelInfo, ...] = (
    # ---- Whisper Turbo (large-v3 distilled, multilingual) ------------------
    ModelInfo(
        alias="whisper-large-v3-turbo",
        canonical="onnx-community/whisper-large-v3-turbo",
        display_name="Whisper Large v3 Turbo",
        size_mb=1620,
        vram_gb=4.0,
        speed="fast",
        quality="excellent",
        languages="multilingual",
        description=(
            "OpenAI Whisper Large v3 Turbo — distilled large-v3, near-large "
            "quality at 6× speed.  Best general-purpose multilingual model."
        ),
        compute_type="float16",
        family="Whisper Turbo",
        onnx_family="whisper",
    ),
    # ---- Whisper Large v3 (full) -------------------------------------------
    ModelInfo(
        alias="whisper-large-v3",
        canonical="onnx-community/whisper-large-v3",
        display_name="Whisper Large v3",
        size_mb=3145,
        vram_gb=6.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="OpenAI Whisper Large v3 — best raw multilingual quality.",
        compute_type="float16",
        family="Whisper",
        onnx_family="whisper",
    ),
    # ---- Whisper Base (tiny, fast on CPU) ----------------------------------
    ModelInfo(
        alias="whisper-base",
        canonical="onnx-community/whisper-base",
        display_name="Whisper Base",
        size_mb=145,
        vram_gb=1.0,
        speed="fast",
        quality="good",
        languages="multilingual",
        description=(
            "OpenAI Whisper Base — small and fast, runs comfortably on CPU. "
            "Quality dips on accented speech but fine for clean dictation."
        ),
        compute_type="float16",
        family="Whisper",
        onnx_family="whisper",
    ),
    # ---- GigaAM v3 (Sber, Russian-only, ONNX) ------------------------------
    # GigaAM v3 e2e variants include built-in punctuation and
    # normalisation in the output, which matters for the clipboard-paste
    # flow (we don't have a separate punctuator).
    ModelInfo(
        alias="gigaam-v3-ctc",
        canonical="istupakov/gigaam-v3-onnx",
        display_name="GigaAM v3 CTC (Russian, punctuated)",
        size_mb=260,
        vram_gb=2.0,
        speed="fast",
        quality="excellent",
        languages="Russian (only)",
        description=(
            "Sber GigaAM v3 with CTC decoder — fast Russian transcription "
            "with built-in punctuation."
        ),
        compute_type="float16",
        family="GigaAM",
        onnx_family="gigaam",
    ),
    ModelInfo(
        alias="gigaam-v3-rnnt",
        canonical="istupakov/gigaam-v3-onnx",
        display_name="GigaAM v3 RNN-T (Russian, punctuated)",
        size_mb=290,
        vram_gb=2.5,
        speed="medium",
        quality="excellent",
        languages="Russian (only)",
        description=(
            "Sber GigaAM v3 with RNN-T decoder — best Russian quality with "
            "built-in punctuation.  Recommended for Russian speakers."
        ),
        compute_type="float16",
        family="GigaAM",
        onnx_family="gigaam",
    ),
    # ---- Parakeet TDT v3 (NVIDIA, multilingual, ONNX) ----------------------
    ModelInfo(
        alias="parakeet-tdt-v3",
        canonical="istupakov/parakeet-tdt-0.6b-v3-onnx",
        display_name="Parakeet TDT v3 (multilingual)",
        size_mb=1200,
        vram_gb=2.0,
        speed="fast",
        quality="excellent",
        languages="25 langs incl. Russian, Ukrainian",
        description=(
            "NVIDIA Parakeet TDT 0.6B v3 — 25 European languages with "
            "auto-detect.  Fastest multilingual ASR on the HF leaderboard."
        ),
        compute_type="float32",
        family="Parakeet",
        onnx_family="parakeet",
    ),
    # ---- T-One (T-Tech, Russian, Conformer-CTC, ONNX) ---------------------
    # 71.7M params, trained on 80k hours of Russian (57.9k of telephony).
    # WER 8.63% on call-center / 6.20% on other Russian telephony — beats
    # Whisper large-v3 (19.39%) on real-world speech with noise/codecs.
    # Apache 2.0.  Built-in KenLM beam search → strong on punctuation.
    # Uses ``gigaam`` onnx_family because the runtime behaviour matches:
    # Russian-only, no language kwarg passed to recognize().
    ModelInfo(
        alias="t-one",
        canonical="t-tech/T-one",
        display_name="T-One (Russian, telephony-tuned)",
        size_mb=290,
        vram_gb=1.5,
        speed="fast",
        quality="excellent",
        languages="Russian (only)",
        description=(
            "T-Tech T-One — Russian Conformer-CTC trained on 80k h of "
            "speech (mostly telephony).  Crushes Whisper on call-center / "
            "noisy audio (8.63 % WER vs 19.39 %).  Built-in KenLM beam "
            "search yields strong punctuation."
        ),
        compute_type="float16",
        family="T-One",
        onnx_family="gigaam",
    ),
    # ---- Vosk Russian (alphacep, Zipformer2 RNN-T, ONNX) ------------------
    # The lightweight option.  ``vosk-model-small-ru`` is ~30 MB,
    # ``vosk-model-ru`` is ~50 MB; both run comfortably on CPU.  Useful
    # for low-spec laptops or as a quick fallback when a heavier model
    # is mid-download.  WER 6.1 % on Common Voice ru.  Apache 2.0.
    ModelInfo(
        alias="vosk-ru-small",
        canonical="alphacep/vosk-model-small-ru",
        display_name="Vosk Small (Russian, 30 MB)",
        size_mb=30,
        vram_gb=0.5,
        speed="fast",
        quality="good",
        languages="Russian (only)",
        description=(
            "Vosk small Russian (Zipformer2 RNN-T) — ultra-lightweight, "
            "~30 MB, runs easily on CPU.  Quality dips on accented speech "
            "but fine for clean dictation."
        ),
        compute_type="float16",
        family="Vosk",
        onnx_family="gigaam",
    ),
    ModelInfo(
        alias="vosk-ru",
        canonical="alphacep/vosk-model-ru",
        display_name="Vosk (Russian, 50 MB)",
        size_mb=50,
        vram_gb=1.0,
        speed="fast",
        quality="excellent",
        languages="Russian (only)",
        description=(
            "Vosk Russian (Zipformer2 RNN-T) — 6.1 % WER on Common Voice "
            "ru, ~50 MB.  Best speed/size/quality balance on CPU."
        ),
        compute_type="float16",
        family="Vosk",
        onnx_family="gigaam",
    ),
    # ---- NVIDIA Canary 1B v2 (multilingual, ONNX) -------------------------
    # 1B-param transformer encoder-decoder; 25 languages incl. Russian.
    # Larger and slightly slower than Parakeet TDT, but stronger on
    # short utterances.  Auto-detects language; behaves like Parakeet
    # for our purposes (no language kwarg, current_language() → None).
    ModelInfo(
        alias="canary-1b-v2",
        canonical="istupakov/canary-1b-v2-onnx",
        display_name="Canary 1B v2 (multilingual)",
        size_mb=2000,
        vram_gb=4.0,
        speed="medium",
        quality="excellent",
        languages="25 langs incl. Russian, Ukrainian",
        description=(
            "NVIDIA Canary 1B v2 — multilingual transformer encoder-"
            "decoder, 25 languages with auto-detect.  Stronger than "
            "Parakeet on short utterances; heavier (1 B params)."
        ),
        compute_type="float16",
        family="Canary",
        onnx_family="parakeet",
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


def model_url(info: ModelInfo) -> str:
    """Resolve the canonical web home for a model card's link icon.

    Every supported model is now hosted on Hugging Face, so we always
    return the HF repo URL.
    """
    return f"https://huggingface.co/{info.canonical}"
