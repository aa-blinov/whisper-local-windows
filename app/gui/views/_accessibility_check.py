"""macOS-only Accessibility-permission probe used by the Settings
banner.

``pynput`` listens for global keyboard events through ``CGEventTap``
which silently returns no events at all when the host process isn't
on the system's Accessibility allow-list. The warning ``pynput``
prints lands in ``logs/app.log`` only — users almost never see it.
This helper exposes the same check synchronously so the Settings
view can show a banner explaining why hotkeys aren't firing.

The call goes through ``ApplicationServices.AXIsProcessTrusted``,
which is part of pyobjc and shipped transitively via ``pynput`` on
macOS — no extra dependency to declare.
"""

from __future__ import annotations

import sys
from typing import Optional


def is_accessibility_trusted() -> Optional[bool]:
    """``True`` when the process can read global keyboard events,
    ``False`` when the user hasn't granted Accessibility access yet,
    ``None`` on platforms where the question doesn't apply
    (Windows / Linux: hotkeys work without an extra permission so
    we never need to surface a banner)."""
    if sys.platform != "darwin":
        return None
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:  # pragma: no cover — pyobjc is mac-only
        return None
    try:
        return bool(AXIsProcessTrusted())
    except Exception:  # pragma: no cover — defensive
        return None


def open_accessibility_settings() -> bool:
    """Launch the System Settings panel that lists Accessibility
    clients so the user can toggle our process on without hunting
    through nested settings panes.

    Returns ``True`` on success.  No-op on non-macOS platforms.
    """
    if sys.platform != "darwin":
        return False
    import subprocess

    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
    try:
        subprocess.Popen(["open", url])
        return True
    except Exception:  # pragma: no cover — defensive
        return False
