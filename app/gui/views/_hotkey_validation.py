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

import sys
from typing import Dict, Mapping, Optional


_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "cmd": "cmd",
    "command": "cmd",
    "win": "cmd",
    "windows": "cmd",
    "super": "cmd",
}
_MODIFIERS = frozenset(_MODIFIER_ALIASES.values())

# Solo keys allowed only in push-to-talk mode.  The listener
# tracks press + release events directly there, so we don't have
# to worry about a held key swallowing an active typing session
# (the user is voluntarily holding it).  Right-side modifiers
# are the safest pick: they're rarely the target of an existing
# shortcut on any platform, and the user keeps a free left-hand
# modifier for normal Cmd+letter / Ctrl+letter shortcuts.
_NON_WINDOWS_PTT_SOLO_KEYS = frozenset({
    "right_cmd", "cmd_r", "right_command", "command_r",
    "right_win", "win_r", "right_windows", "windows_r",
    "right_super", "super_r",
    "right_option", "option_r", "right_alt", "alt_r",
    "right_shift", "shift_r",
    "right_ctrl", "ctrl_r", "right_control", "control_r",
    "fn",
})

_WINDOWS_PTT_SOLO_KEYS = frozenset({
    "right_cmd", "cmd_r", "right_command", "command_r",
    "right_win", "win_r", "right_windows", "windows_r",
    "right_super", "super_r",
    "right_window",
    "right_option", "option_r", "right_alt", "alt_r",
    "right_menu",
    "right_shift", "shift_r",
    "right_ctrl", "ctrl_r", "right_control", "control_r",
})

_NAMED_KEY_ALIASES = {
    "return": "enter",
    "esc": "esc",
    "escape": "esc",
    "pageup": "page_up",
    "pagedown": "page_down",
    "del": "delete",
}

_NAMED_KEYS = frozenset({
    "space", "enter",
    "esc",
    "tab", "backspace", "delete", "insert",
    "home", "end", "page_up", "page_down",
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


def _ptt_solo_keys_for_platform(platform: str) -> frozenset[str]:
    if platform == "win32":
        return _WINDOWS_PTT_SOLO_KEYS
    return _NON_WINDOWS_PTT_SOLO_KEYS


def is_push_to_talk_solo_key(
    text: str, *, platform: str | None = None
) -> bool:
    """Return whether ``text`` is one of the platform's supported
    solo push-to-talk modifier keys.

    Kept separate from :func:`validate_hotkey` so other UI code can
    branch on the active binding shape without re-running the full
    validation routine.
    """
    raw = (text or "").strip().lower()
    if not raw:
        return False
    resolved_platform = platform or sys.platform
    return raw in _ptt_solo_keys_for_platform(resolved_platform)


def validate_hotkey(
    text: str,
    *,
    allow_empty: bool = False,
    push_to_talk: bool = False,
    platform: str | None = None,
) -> Optional[str]:
    """Return ``None`` if ``text`` is a valid hotkey combination,
    otherwise a human-readable error string.

    ``allow_empty=True`` makes the empty / whitespace-only input
    accepted (used for the optional Cancel field — empty means
    "no binding").

    ``push_to_talk=True`` switches to the more permissive
    platform-specific vocabulary that allows a solo right-side
    modifier (``right_cmd`` / ``right_alt`` / …; plus ``fn`` on
    non-Windows platforms). Those bindings are only safe when the
    listener tracks press + release events, because the user is
    deliberately holding the key for the recording duration. In
    any other mode a solo modifier would silently swallow the next
    keystroke and is rejected.
    """
    raw = (text or "").strip()
    if not raw:
        return None if allow_empty else "Hotkey cannot be empty"
    resolved_platform = platform or sys.platform

    # Push-to-talk solo modifier shortcut: accept the special
    # vocabulary up-front, before normal parsing splits on ``+``.
    # ``right_cmd`` doesn't have a ``+`` separator, and the modifier
    # check below would reject it because nothing precedes the key.
    if push_to_talk and is_push_to_talk_solo_key(
        raw, platform=resolved_platform
    ):
        return None

    raw_parts = [p.strip().lower() for p in raw.split("+")]
    if any(not p for p in raw_parts):
        return f"Empty segment in {raw!r} — remove the extra '+' or trailing space"
    if len(raw_parts) < 1:
        return f"Cannot parse {raw!r}"

    parts = [
        _MODIFIER_ALIASES.get(part, _NAMED_KEY_ALIASES.get(part, part))
        for part in raw_parts
    ]

    # Last segment is the key; everything before it must be a modifier.
    *mods, key = parts
    *raw_mods, raw_key = raw_parts

    # Duplicate modifiers
    if len(mods) != len(set(mods)):
        return f"Duplicate modifier in {raw!r}"

    # Validate every modifier
    for raw_modifier, m in zip(raw_mods, mods):
        if m not in _MODIFIERS:
            return (
                f"{raw_modifier!r} is not a recognised modifier "
                "(use ctrl, shift, alt, cmd)"
            )

    # The trailing token must be a real key, not a modifier.
    if key in _MODIFIERS:
        return (
            f"{raw_key!r} is a modifier — the combination needs a final key "
            "(letter, digit, F-key or named key like 'space')"
        )

    if not (
        _is_letter(key)
        or _is_digit(key)
        or _is_function_key(key)
        or key in _NAMED_KEYS
    ):
        return f"{raw_key!r} is not a recognised key"

    # Anything other than an F-key without a modifier would
    # intercept normal typing globally.  ``space`` would block
    # every space character system-wide; ``a`` would block every
    # letter ``a``; ``enter`` would block submit-on-Enter
    # everywhere.  F1..F24 are the only keys exempt — they're
    # essentially never used in plain text input.
    if not mods and not _is_function_key(key):
        return (
            "Add a modifier (ctrl, alt, shift, cmd) — a lone key "
            "would be intercepted every time you press it in any "
            "app. Only F1..F24 may be used alone."
        )

    return None


def find_hotkey_conflicts(
    start: str,
    stop: str,
    cancel: str,
    *,
    mode: str = "two_keys",
) -> Dict[str, str]:
    """Cross-field conflict check.

    Returns a mapping ``field_name -> error_message`` where
    ``field_name`` is one of ``"start"``, ``"stop"``, ``"cancel"``.

    ``mode``:
      - ``"two_keys"`` (default):  flags Stop == Start as needing
        toggle mode, plus standard Cancel collisions.
      - ``"toggle"`` / ``"push_to_talk"``:  ``stop`` is unused;
        we only check Cancel against Start.
    """
    errors: Dict[str, str] = {}

    s = start.strip().lower()
    p = stop.strip().lower()
    c = cancel.strip().lower()

    if mode == "two_keys":
        if s and p and s == p:
            errors["stop"] = (
                f"Stop matches Start ({s}) — switch to Toggle mode if you "
                "want one hotkey to do both"
            )
        if c and p and c == p:
            errors["cancel"] = f"Cancel matches Stop ({p})"

    if c and s and c == s and "cancel" not in errors:
        errors["cancel"] = f"Cancel matches Start ({s})"

    return errors


def validate_all(
    *,
    start: str,
    stop: str,
    cancel: str,
    mode: str = "two_keys",
    platform: str | None = None,
) -> Mapping[str, str]:
    """One-stop check returning a ``field_name -> error`` map for
    every problem ShortcutsView should surface.  Combines
    :func:`validate_hotkey` per-field with
    :func:`find_hotkey_conflicts`. First-error-wins per field — we
    don't accumulate multiple messages on the same input.

    ``mode``:
      - ``"two_keys"``:  Start + Stop + (optional) Cancel
      - ``"toggle"``:    only Start + (optional) Cancel
      - ``"push_to_talk"``:  only Start (with the looser "solo
        modifier" vocabulary) + (optional) Cancel
    """
    errors: Dict[str, str] = {}

    err = validate_hotkey(
        start,
        push_to_talk=(mode == "push_to_talk"),
        platform=platform,
    )
    if err:
        errors["start"] = err

    if mode == "two_keys":
        err = validate_hotkey(stop, platform=platform)
        if err:
            errors["stop"] = err

    err = validate_hotkey(cancel, allow_empty=True, platform=platform)
    if err:
        errors["cancel"] = err

    # Cross-field conflicts: only meaningful when each field
    # individually parsed correctly.
    if "start" not in errors and "stop" not in errors and "cancel" not in errors:
        errors.update(find_hotkey_conflicts(
            start, stop, cancel, mode=mode,
        ))

    return errors
