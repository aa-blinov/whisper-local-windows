"""macOS-only TCC permission probes for keyboard event access.

Apple now distinguishes between two keyboard-related capabilities:

- listening to global key events (used for global hotkeys)
- posting synthetic key events (used for auto-paste / auto-enter)

Both live under the broader Accessibility / Input Monitoring family,
but they are not the same gate and should not be conflated in the UI
or in runtime checks.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def is_listen_event_access_trusted() -> Optional[bool]:
    """Return whether macOS allows the current process to read global
    keyboard events.

    Our macOS hotkey path uses ``NSEvent.addGlobalMonitor...``. Apple
    documents key-event monitoring there in terms of
    ``AXIsProcessTrusted()``, so that Accessibility trust is the
    primary signal we should surface in the UI.  The newer CoreGraphics
    preflight API can still be useful as a fallback, but on some
    systems it produces false negatives for AppKit-based monitors.
    """
    if not _is_darwin():
        return None
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:  # pragma: no cover - mac-only dependency path
        AXIsProcessTrusted = None

    if AXIsProcessTrusted is not None:
        try:
            return bool(AXIsProcessTrusted())
        except Exception:
            pass

    try:
        from Quartz import CGPreflightListenEventAccess
    except ImportError:
        CGPreflightListenEventAccess = None

    if CGPreflightListenEventAccess is not None:
        try:
            return bool(CGPreflightListenEventAccess())
        except Exception:
            pass

    return None


def request_listen_event_access() -> bool:
    """Ask macOS to grant global keyboard-listening access.

    Returns the live boolean result from the system API when
    available. ``True`` means access is already granted; ``False``
    means it is still missing (the request may or may not have shown
    a prompt). Callers that need a guaranteed visible path should
    fall back to opening System Settings when ``False`` comes back.
    """
    if not _is_darwin():
        return False
    try:
        from ApplicationServices import (
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
    except ImportError:
        AXIsProcessTrustedWithOptions = None
        kAXTrustedCheckOptionPrompt = None

    if (
        AXIsProcessTrustedWithOptions is not None
        and kAXTrustedCheckOptionPrompt is not None
    ):
        try:
            return bool(
                AXIsProcessTrustedWithOptions(
                    {kAXTrustedCheckOptionPrompt: True}
                )
            )
        except Exception:
            pass

    try:
        from Quartz import CGRequestListenEventAccess
    except ImportError:
        CGRequestListenEventAccess = None

    if CGRequestListenEventAccess is not None:
        try:
            return bool(CGRequestListenEventAccess())
        except Exception:
            pass
    return False


def is_post_event_access_trusted() -> Optional[bool]:
    """Return whether macOS allows the current process to drive
    auto-paste keystrokes.

    The app's primary macOS auto-paste path uses the Accessibility
    API (``AXUIElementPostKeyboardEvent``) against the active
    application, so the same "trusted accessibility client" gate is
    the most relevant signal to surface in the UI.
    """
    if not _is_darwin():
        return None
    return is_listen_event_access_trusted()


def request_post_event_access() -> bool:
    """Ask macOS to grant synthetic-keyboard-event access.

    The primary runtime path goes through Accessibility APIs, so ask
    for that trust first. If the dedicated CoreGraphics post-event
    request API is available too, keep it as a best-effort fallback.
    """
    if not _is_darwin():
        return False
    if request_listen_event_access():
        return True
    try:
        from Quartz import CGRequestPostEventAccess
    except ImportError:
        return False
    try:
        return bool(CGRequestPostEventAccess())
    except Exception:  # pragma: no cover - defensive
        return False


def open_accessibility_settings() -> bool:
    """Open the macOS Accessibility privacy pane."""
    if not _is_darwin():
        return False
    url = (
        "x-apple.systempreferences:com.apple.preference.security?"
        "Privacy_Accessibility"
    )
    try:
        subprocess.Popen(["open", url])
        return True
    except Exception:  # pragma: no cover - defensive
        return False
