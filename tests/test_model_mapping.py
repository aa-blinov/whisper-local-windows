"""Tests for the extended model registry."""

import pytest


# ---- ModelInfo dataclass ----------------------------------------------------


def test_model_info_exposes_required_fields():
    from app.model_mapping import ModelInfo

    info = ModelInfo(
        alias="large-v3",
        canonical="Systran/faster-whisper-large-v3",
        display_name="Large v3",
        size_mb=3000,
        vram_gb=10.0,
        speed="slow",
        quality="excellent",
        languages="multilingual",
        description="Highest-quality multilingual model.",
    )
    assert info.alias == "large-v3"
    assert info.canonical == "Systran/faster-whisper-large-v3"
    assert info.display_name == "Large v3"
    assert info.size_mb == 3000
    assert info.vram_gb == 10.0
    assert info.speed == "slow"
    assert info.quality == "excellent"
    assert info.languages == "multilingual"
    assert info.description


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
    from app.model_mapping import MODELS, aliases

    # Lineup is the post-cleanup powerful presets — large-v3 family
    # plus turbo / distil and Russian fine-tunes.
    expected = {
        "turbo",
        "distil-large-v3",
        "large-v3",
        "large-v3-int8",
        "large-v3-ru",
        "large-v3-ru-int8",
    }
    assert expected.issubset(set(aliases()))
    assert len(MODELS) == len(aliases())


def test_registry_preserves_order_between_models_and_aliases():
    from app.model_mapping import MODELS, aliases

    assert [m.alias for m in MODELS] == aliases()


def test_get_model_returns_info_by_alias():
    from app.model_mapping import ModelInfo, get_model

    info = get_model("large-v3")
    assert isinstance(info, ModelInfo)
    assert info.alias == "large-v3"
    assert info.canonical == "Systran/faster-whisper-large-v3"


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

    assert canonical_for("large-v3") == "Systran/faster-whisper-large-v3"


def test_canonical_for_passes_through_unknown_names():
    from app.model_mapping import canonical_for

    assert canonical_for("custom/model-id") == "custom/model-id"


def test_alias_for_returns_alias_for_known_canonical():
    from app.model_mapping import alias_for

    assert alias_for("Systran/faster-whisper-large-v3") == "large-v3"


def test_alias_for_passes_through_unknown_canonicals():
    from app.model_mapping import alias_for

    assert alias_for("unknown/model") == "unknown/model"


def test_every_registry_entry_carries_a_known_family():
    from app.model_mapping import FAMILIES, MODELS

    for info in MODELS:
        assert info.family in FAMILIES, (
            f"{info.alias} has unknown family {info.family!r}"
        )


def test_model_url_for_faster_whisper_points_at_hf_repo():
    from app.model_mapping import get_model, model_url

    info = get_model("large-v3")
    assert model_url(info) == (
        "https://huggingface.co/Systran/faster-whisper-large-v3"
    )


def test_model_url_for_gigaam_points_at_github():
    """GigaAM doesn't ship via Hugging Face — its weights come from
    Sber's CDN. The closest "model home" the user can browse is the
    project's GitHub README, so all GigaAM cards link there."""
    from app.model_mapping import get_model, model_url

    info = get_model("gigaam-v2-ctc")
    assert model_url(info) == "https://github.com/salute-developers/GigaAM"

    info_rnnt = get_model("gigaam-v2-rnnt")
    assert model_url(info_rnnt) == "https://github.com/salute-developers/GigaAM"
