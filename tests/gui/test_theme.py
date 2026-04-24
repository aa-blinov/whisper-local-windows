"""Tests for the theme tokens and QSS loader."""

import pytest


def test_tokens_expose_required_color_keys():
    from app.gui.theme import TOKENS

    required = {
        "bg_primary",
        "bg_secondary",
        "bg_elevated",
        "accent",
        "text_primary",
        "text_secondary",
        "border",
        "success",
        "danger",
        "warning",
    }
    assert required.issubset(TOKENS.colors.keys())


def test_tokens_expose_spacing_and_radius():
    from app.gui.theme import TOKENS

    for key in ("xs", "sm", "md", "lg", "xl"):
        assert key in TOKENS.spacing, f"missing spacing.{key}"
        assert isinstance(TOKENS.spacing[key], int)

    for key in ("sm", "md", "lg"):
        assert key in TOKENS.radius
        assert isinstance(TOKENS.radius[key], int)


def test_tokens_expose_font_family_and_sizes():
    from app.gui.theme import TOKENS

    assert isinstance(TOKENS.fonts["family"], str)
    assert TOKENS.fonts["family"]
    for key in ("size_title", "size_body", "size_small"):
        assert key in TOKENS.fonts
        assert isinstance(TOKENS.fonts[key], int)


def test_load_stylesheet_returns_non_empty_string_for_dark_theme():
    from app.gui.theme import load_stylesheet

    qss = load_stylesheet("dark")
    assert isinstance(qss, str)
    assert qss.strip()


def test_load_stylesheet_resolves_all_token_placeholders():
    """No unresolved {{token}} markers should remain after loading."""
    from app.gui.theme import load_stylesheet

    qss = load_stylesheet("dark")
    assert "{{" not in qss
    assert "}}" not in qss


def test_load_stylesheet_substitutes_actual_colors():
    from app.gui.theme import TOKENS, load_stylesheet

    qss = load_stylesheet("dark")
    assert TOKENS.colors["bg_primary"] in qss


def test_load_stylesheet_raises_for_unknown_theme():
    from app.gui.theme import load_stylesheet

    with pytest.raises(ValueError):
        load_stylesheet("does-not-exist")


def test_load_stylesheet_default_is_dark():
    from app.gui.theme import load_stylesheet

    assert load_stylesheet() == load_stylesheet("dark")


def test_apply_theme_sets_stylesheet_on_qapplication(qapp):
    from app.gui.theme import apply_theme, load_stylesheet

    apply_theme(qapp, "dark")
    assert qapp.styleSheet() == load_stylesheet("dark")
