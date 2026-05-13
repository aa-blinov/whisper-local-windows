"""Unit tests for the pure hotkey-string validator.

The validator is pulled out of ``ShortcutsView`` so we can exercise
its accept / reject decisions without spinning up Qt.  Coverage
matrix:

- valid combinations (modifier + letter, F-keys with / without
  modifiers, named keys like ``space`` / ``esc``)
- malformed inputs (empty when not allowed, trailing ``+``,
  duplicate modifier, unknown modifier, lone letter)
- cross-field conflicts (Stop == Start in non-toggle mode,
  Cancel duplicating either)
- toggle-mode neutralises the Stop / Cancel conflict checks
"""

from __future__ import annotations

import pytest

from app.gui.views._hotkey_validation import (
    find_hotkey_conflicts,
    validate_all,
    validate_hotkey,
)


# ---- Single-field validation ---------------------------------------------


@pytest.mark.parametrize(
    "combo",
    [
        "ctrl+f2",
        "Ctrl+F2",  # case-insensitive
        "ctrl+shift+a",
        "alt+space",
        "cmd+v",
        "f5",  # F-key alone — fine
        "f24",
        "ctrl+1",
        "ctrl+enter",
    ],
)
def test_valid_combinations_accepted(combo: str) -> None:
    assert validate_hotkey(combo) is None, f"expected {combo!r} to validate"


@pytest.mark.parametrize(
    ("combo", "needle"),
    [
        ("", "empty"),
        ("ctrl+", "empty segment"),
        ("+ctrl+f2", "empty segment"),
        ("ctrl+ctrl+f2", "duplicate modifier"),
        ("foo+f2", "not a recognised modifier"),
        ("ctrl+blah", "not a recognised key"),
        ("a", "modifier"),
        ("1", "modifier"),
        # ``ctrl+ctrl`` parses as one modifier (ctrl) + final-token
        # ctrl, and ctrl-on-the-final-position is a more useful
        # error than "duplicate modifier" — the user typed two
        # modifiers and forgot the key.
        ("ctrl+ctrl", "modifier"),
    ],
)
def test_invalid_combinations_carry_helpful_message(
    combo: str, needle: str,
) -> None:
    err = validate_hotkey(combo)
    assert err is not None, f"expected {combo!r} to fail"
    assert needle.lower() in err.lower(), (
        f"expected error for {combo!r} to mention {needle!r}, got: {err!r}"
    )


def test_empty_is_allowed_when_flagged() -> None:
    """The Cancel field is optional — empty means 'no binding'."""
    assert validate_hotkey("", allow_empty=True) is None
    assert validate_hotkey("   ", allow_empty=True) is None


def test_lone_modifier_rejected() -> None:
    assert "modifier" in (validate_hotkey("ctrl") or "").lower()


def test_duplicate_modifier_aliases_rejected() -> None:
    err = validate_hotkey("ctrl+control+f2")
    assert err is not None
    assert "duplicate modifier" in err.lower()


@pytest.mark.parametrize("key", ["space", "enter", "esc", "tab", "left"])
def test_lone_named_key_rejected(key: str) -> None:
    """Named keys like ``space`` or ``enter`` are too "hot" to
    register as a global hotkey on their own — would block normal
    typing in every other app. Only F1..F24 are exempt."""
    err = validate_hotkey(key)
    assert err is not None, f"expected lone {key!r} to fail"
    assert "modifier" in err.lower() or "alone" in err.lower()


@pytest.mark.parametrize(
    "combo",
    [
        "alt+space",
        "ctrl+enter",
        "ctrl+return",
        "ctrl+pageup",
        "shift+tab",
        "cmd+left",
    ],
)
def test_named_key_with_modifier_accepted(combo: str) -> None:
    assert validate_hotkey(combo) is None


def test_windows_push_to_talk_accepts_right_side_modifiers() -> None:
    assert (
        validate_hotkey(
            "right_alt", push_to_talk=True, platform="win32",
        )
        is None
    )
    assert (
        validate_hotkey(
            "right_control", push_to_talk=True, platform="win32",
        )
        is None
    )


def test_windows_push_to_talk_rejects_fn() -> None:
    err = validate_hotkey("fn", push_to_talk=True, platform="win32")
    assert err is not None
    assert "recognised key" in err.lower()


# ---- Cross-field conflicts -----------------------------------------------


def test_no_conflict_when_all_fields_distinct() -> None:
    errors = find_hotkey_conflicts(
        "ctrl+f2", "ctrl+f3", "ctrl+f6", mode="two_keys",
    )
    assert errors == {}


def test_stop_equals_start_flagged_outside_toggle_mode() -> None:
    errors = find_hotkey_conflicts(
        "ctrl+f2", "ctrl+f2", "ctrl+f6", mode="two_keys",
    )
    assert "stop" in errors
    assert "Start" in errors["stop"]


def test_cancel_equals_start_flagged() -> None:
    errors = find_hotkey_conflicts(
        "ctrl+f2", "ctrl+f3", "ctrl+f2", mode="two_keys",
    )
    assert "cancel" in errors
    assert "Start" in errors["cancel"]


def test_cancel_equals_stop_flagged() -> None:
    errors = find_hotkey_conflicts(
        "ctrl+f2", "ctrl+f3", "ctrl+f3", mode="two_keys",
    )
    assert "cancel" in errors


def test_toggle_mode_disables_stop_and_cancel_conflict_checks() -> None:
    """In toggle mode start == stop is intentional and cancel is
    expected to be empty — no conflicts to flag."""
    errors = find_hotkey_conflicts(
        "ctrl+f2", "ctrl+f2", "", mode="toggle",
    )
    assert errors == {}


def test_conflicts_compared_case_insensitively() -> None:
    errors = find_hotkey_conflicts(
        "Ctrl+F2", "ctrl+f2", "", mode="two_keys",
    )
    assert "stop" in errors


# ---- validate_all aggregator ---------------------------------------------


def test_validate_all_combines_field_errors_and_conflicts() -> None:
    errors = validate_all(
        start="ctrl+f2",
        stop="ctrl+f2",
        cancel="ctrl+f6",
        mode="two_keys",
    )
    # No per-field errors → conflict surfaces.
    assert "stop" in errors


def test_validate_all_does_not_emit_conflict_when_field_already_invalid() -> None:
    """If a field fails to parse, we surface the parse error and
    don't pile on a duplicate-of-itself conflict — first error wins
    per field."""
    errors = validate_all(
        start="not a real hotkey",
        stop="not a real hotkey",
        cancel="",
        mode="two_keys",
    )
    assert "start" in errors
    # Stop also doesn't parse, so it's flagged for the same reason —
    # not for duplicating Start.
    assert "stop" in errors
    assert "matches Start" not in errors["stop"]


def test_validate_all_skips_stop_check_in_toggle_mode() -> None:
    """In toggle mode the Stop field is allowed to be anything (or
    empty) — the UI muts it and re-uses Start.  Validate_all
    shouldn't surface a Stop error in that case."""
    errors = validate_all(
        start="ctrl+f2",
        stop="",                # would normally fail "empty"
        cancel="",
        mode="toggle",
    )
    assert "stop" not in errors


def test_validate_all_clears_when_everything_is_valid() -> None:
    assert validate_all(
        start="ctrl+f2",
        stop="ctrl+f3",
        cancel="ctrl+f6",
        mode="two_keys",
    ) == {}
