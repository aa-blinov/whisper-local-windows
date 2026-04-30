"""Pure-Python validator for hotkey combination strings.

Used by ``ShortcutsView`` to surface inline feedback when the user
types something that ``HotkeyListener`` won't be able to register
(unrecognised modifier, missing modifier on a letter shortcut,
duplicate binding across Start / Stop / Cancel fields). Lives in its
own module so tests can import it without spinning up Qt.

The accepted vocabulary mirrors what ``hotkey_listener._convert_*``
methods understand: same modifier aliases, same named-key set
(``f1..f24``, ``space``, ``enter``, ``esc``, arrows, etc.), same
case-insensitive comparison.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional


_MODIFIERS = frozenset({
    "ctrl", "control",
    "shift",
    "alt", "option",
    "cmd", "command", "win", "windows", "super",
})

_NAMED_KEYS = frozenset({
    "space", "enter", "return",
    "esc", "escape",
    "tab", "backspace", "delete", "insert",
    "home", "end", "page_up", "page_down", "pageup", "pagedown",
    "up", "down", "left", "right",
    *(f"f{i}" for i in range(1, 25)),
})


def _is_function_key(key: str) -> bool:
    """``f1`` … ``f24`` (lowercase)."""
    if len(key) < 2 or key[0] != "f":
        return False
    rest = key[1:]
    if not rest.isdigit():
        return False
    n = int(rest)
    return 1 <= n <= 24


def _is_letter(key: str) -> bool:
    return len(key) == 1 and "a" <= key <= "z"


def _is_digit(key: str) -> bool:
    return len(key) == 1 and "0" <= key <= "9"


def validate_hotkey(text: str, *, allow_empty: bool = False) -> Optional[str]:
    """Return ``None`` if ``text`` is a valid hotkey combination,
    otherwise a human-readable error string.

    ``allow_empty=True`` makes the empty / whitespace-only input
    accepted (used for the optional Cancel field — empty means
    "no binding").
    """
    raw = (text or "").strip()
    if not raw:
        return None if allow_empty else "Hotkey cannot be empty"

    parts = [p.strip().lower() for p in raw.split("+")]
    if any(not p for p in parts):
        return f"Empty segment in {raw!r} — remove the extra '+' or trailing space"
    if len(parts) < 1:
        return f"Cannot parse {raw!r}"

    # Last segment is the key; everything before it must be a modifier.
    *mods, key = parts

    # Duplicate modifiers
    if len(mods) != len(set(mods)):
        return f"Duplicate modifier in {raw!r}"

    # Validate every modifier
    for m in mods:
        if m not in _MODIFIERS:
            return f"{m!r} is not a recognised modifier (use ctrl, shift, alt, cmd)"

    # The trailing token must be a real key, not a modifier.
    if key in _MODIFIERS:
        return (
            f"{key!r} is a modifier — the combination needs a final key "
            "(letter, digit, F-key or named key like 'space')"
        )

    if not (
        _is_letter(key)
        or _is_digit(key)
        or _is_function_key(key)
        or key in _NAMED_KEYS
    ):
        return f"{key!r} is not a recognised key"

    # Letters / digits without a modifier would intercept normal
    # typing globally — almost never what the user wants.
    if not mods and (_is_letter(key) or _is_digit(key)):
        return (
            "Add at least one modifier (ctrl, alt, shift, cmd) — a "
            "lone letter / digit would steal every keystroke"
        )

    return None


def find_hotkey_conflicts(
    start: str,
    stop: str,
    cancel: str,
    *,
    toggle_mode: bool,
) -> Dict[str, str]:
    """Cross-field conflict check.

    Returns a mapping ``field_name -> error_message`` where
    ``field_name`` is one of ``"start"``, ``"stop"``, ``"cancel"``.

    In toggle mode ``stop`` is expected to equal ``start`` (the
    HotkeyListener registers a single toggle binding) and ``cancel``
    is expected to be empty (the UI mutes it) — so neither pair is
    flagged.
    """
    errors: Dict[str, str] = {}
    if toggle_mode:
        return errors

    s = start.strip().lower()
    p = stop.strip().lower()
    c = cancel.strip().lower()

    if s and p and s == p:
        errors["stop"] = (
            f"Stop matches Start ({s}) — enable the toggle checkbox if you "
            "want one hotkey to do both"
        )
    if c and s and c == s:
        errors["cancel"] = f"Cancel matches Start ({s})"
    if c and p and c == p and "cancel" not in errors:
        errors["cancel"] = f"Cancel matches Stop ({p})"

    return errors


def validate_all(
    *,
    start: str,
    stop: str,
    cancel: str,
    toggle_mode: bool,
) -> Mapping[str, str]:
    """One-stop check returning a ``field_name -> error`` map for
    every problem ShortcutsView should surface.  Combines
    :func:`validate_hotkey` per-field with
    :func:`find_hotkey_conflicts`. First-error-wins per field — we
    don't accumulate multiple messages on the same input.
    """
    errors: Dict[str, str] = {}

    err = validate_hotkey(start)
    if err:
        errors["start"] = err

    if not toggle_mode:
        err = validate_hotkey(stop)
        if err:
            errors["stop"] = err

    err = validate_hotkey(cancel, allow_empty=True)
    if err:
        errors["cancel"] = err

    # Cross-field conflicts: only meaningful when each field
    # individually parsed correctly.
    if "start" not in errors and "stop" not in errors and "cancel" not in errors:
        errors.update(find_hotkey_conflicts(
            start, stop, cancel, toggle_mode=toggle_mode,
        ))

    return errors
