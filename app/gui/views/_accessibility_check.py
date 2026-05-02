"""Compatibility wrappers for the macOS keyboard-access probes.

Historically this module exposed a single
``is_accessibility_trusted()`` helper backed by
``AXIsProcessTrusted()``. The app now distinguishes between:

- listening to global key events (hotkeys)
- posting synthetic key events (auto-paste)

The canonical implementation lives in :mod:`app.macos_permissions`;
this module keeps the old import path alive for the Settings view and
older tests.
"""

from __future__ import annotations

from typing import Optional

from app.macos_permissions import (
    is_listen_event_access_trusted,
    is_post_event_access_trusted,
    open_accessibility_settings,
    request_listen_event_access,
    request_post_event_access,
)

def is_accessibility_trusted() -> Optional[bool]:
    """Backward-compatible alias for the listen-event permission."""
    return is_listen_event_access_trusted()


def request_accessibility_access() -> bool:
    """Ask macOS to grant hotkey-listening permission."""
    return request_listen_event_access()


__all__ = [
    "is_accessibility_trusted",
    "is_post_event_access_trusted",
    "open_accessibility_settings",
    "request_accessibility_access",
    "request_post_event_access",
]
