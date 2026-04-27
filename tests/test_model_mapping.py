"""Tests for the ONNX-only model registry."""

import pytest


# ---- ModelInfo dataclass ----------------------------------------------------


def test_model_info_exposes_required_fields():
    from app.model_mapping import ModelInfo

    info = ModelInfo(
        alias="whisper-large-v3",
        canonical="onnx-community/whisper-large-v3",
        display_name="Whisper Large v3",
        size_mb=3000,
        vram_gb=6.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Highest-quality multilingual Whisper.",
    )
    assert info.alias == "whisper-large-v3"
    assert info.canonical == "onnx-community/whisper-large-v3"
    assert info.display_name == "Whisper Large v3"
    assert info.size_mb == 3000
    assert info.vram_gb == 6.0
    assert info.speed == "slow"
    assert info.quality == "excellent"
    assert info.languages == "multilingual"
    assert info.description
    # Defaults
    assert info.backend_kind == "onnx_asr"
    assert info.onnx_family == "whisper"


def test_model_info_rejects_invalid_speed():
    from app.model_mapping import ModelInfo

    with pytest.raises(ValueError):
        ModelInfo(
            alias="x",
            canonical="x",
            display_name="X",
            size_mb=1,
            vram_gb=0.1,
            speed="warp",  # invalid
            quality="good",
            languages="multilingual",
            description="",
        )


def test_model_info_rejects_invalid_quality():
    from app.model_mapping import ModelInfo

    with pytest.raises(ValueError):
        ModelInfo(
            alias="x",
            canonical="x",
            display_name="X",
            size_mb=1,
            vram_gb=0.1,
            speed="fast",
            quality="perfect",  # invalid
            languages="multilingual",
            description="",
        )


def test_model_info_rejects_invalid_onnx_family():
    from app.model_mapping import ModelInfo

    with pytest.raises(ValueError):
        ModelInfo(
            alias="x",
            canonical="x",
            display_name="X",
            size_mb=1,
            vram_gb=0.1,
            speed="fast",
            quality="good",
            languages="multilingual",
            description="",
            onnx_family="bogus",
        )


def test_model_info_is_frozen():
    from app.model_mapping import ModelInfo

    info = ModelInfo(
        alias="x",
        canonical="x",
        display_name="X",
        size_mb=1,
        vram_gb=0.1,
        speed="fast",
        quality="good",
        languages="multilingual",
        description="",
    )
    with pytest.raises((AttributeError, Exception)):
        info.alias = "y"  # type: ignore[misc]


# ---- Registry ---------------------------------------------------------------


def test_registry_contains_core_models():
    """The post-ONNX-only lineup: a Whisper turbo, a small Whisper for
    CPU users, GigaAM for Russian, Parakeet for multilingual."""
    from app.model_mapping import MODELS, aliases

    expected = {
        "whisper-large-v3-turbo",
        "vosk-ru-small",
        "gigaam-v3-rnnt",
        "parakeet-tdt-v3",
    }
    assert expected.issubset(set(aliases()))
    assert len(MODELS) == len(aliases())


def test_every_registry_entry_uses_onnx_asr_backend():
    """The single-engine invariant: every model goes through onnx_asr."""
    from app.model_mapping import MODELS

    for info in MODELS:
        assert info.backend_kind == "onnx_asr", (
            f"{info.alias} should use onnx_asr backend, got {info.backend_kind!r}"
        )


def test_every_registry_entry_has_a_valid_onnx_family():
    from app.model_mapping import MODELS, ONNX_FAMILIES

    for info in MODELS:
        assert info.onnx_family in ONNX_FAMILIES, (
            f"{info.alias} has invalid onnx_family {info.onnx_family!r}"
        )


def test_registry_preserves_order_between_models_and_aliases():
    from app.model_mapping import MODELS, aliases

    assert [m.alias for m in MODELS] == aliases()


def test_get_model_returns_info_by_alias():
    from app.model_mapping import ModelInfo, get_model

    info = get_model("whisper-large-v3-turbo")
    assert isinstance(info, ModelInfo)
    assert info.alias == "whisper-large-v3-turbo"
    assert info.canonical == "onnx-community/whisper-large-v3-turbo"


def test_get_model_raises_on_unknown_alias():
    from app.model_mapping import get_model

    with pytest.raises(KeyError):
        get_model("not-a-model")


# ---- Backward compatibility -------------------------------------------------


def test_alias_to_model_derived_from_registry():
    from app.model_mapping import ALIAS_TO_MODEL, MODELS

    for m in MODELS:
        assert ALIAS_TO_MODEL[m.alias] == m.canonical


def test_canonical_for_returns_canonical_for_known_alias():
    from app.model_mapping import canonical_for

    assert canonical_for("whisper-large-v3-turbo") == (
        "onnx-community/whisper-large-v3-turbo"
    )


def test_canonical_for_passes_through_unknown_names():
    from app.model_mapping import canonical_for

    assert canonical_for("custom/model-id") == "custom/model-id"


def test_alias_for_returns_alias_for_known_canonical():
    from app.model_mapping import alias_for

    assert alias_for("onnx-community/whisper-large-v3-turbo") == (
        "whisper-large-v3-turbo"
    )


def test_alias_for_passes_through_unknown_canonicals():
    from app.model_mapping import alias_for

    assert alias_for("unknown/model") == "unknown/model"


def test_every_registry_entry_carries_a_known_family():
    from app.model_mapping import FAMILIES, MODELS

    for info in MODELS:
        assert info.family in FAMILIES, (
            f"{info.alias} has unknown family {info.family!r}"
        )


def test_model_url_points_at_hf_repo():
    """Every model is hosted on Hugging Face now — the URL should
    always be the HF repo path."""
    from app.model_mapping import get_model, model_url

    info = get_model("whisper-large-v3-turbo")
    assert model_url(info) == (
        "https://huggingface.co/onnx-community/whisper-large-v3-turbo"
    )

    gigaam = get_model("gigaam-v3-rnnt")
    assert model_url(gigaam) == "https://huggingface.co/istupakov/gigaam-v3-onnx"

    parakeet = get_model("parakeet-tdt-v3")
    assert model_url(parakeet) == (
        "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx"
    )
