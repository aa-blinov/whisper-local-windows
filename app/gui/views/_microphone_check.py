"""macOS-only microphone-permission probe used by the Settings
banner.

Same shape as :mod:`_accessibility_check` — wraps the relevant
TCC API behind a tiny pure-Python module so the UI can render
state-aware banners without importing pyobjc directly.

API
---
- :func:`microphone_authorization_status` returns one of
  ``"authorized"`` / ``"denied"`` / ``"restricted"`` /
  ``"not_determined"`` on macOS, or ``None`` everywhere else.
- :func:`request_microphone_access` triggers the system prompt
  the first time it's called for ``not_determined``; on every
  subsequent call macOS routes the user to System Settings (or
  silently ignores if already authorised).
- :func:`open_microphone_settings` opens the System Settings
  pane that lists Microphone clients — used after the user has
  hit the system "denied" wall and needs to flip the toggle
  manually.

The AVFoundation framework comes in via the
``pyobjc-framework-AVFoundation`` extra dependency declared
under ``project.optional-dependencies.darwin`` in
``pyproject.toml``.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional


# Mirrors the C ``AVAuthorizationStatus`` enum.
_STATUS_BY_INT = {
    0: "not_determined",
    1: "restricted",
    2: "denied",
    3: "authorized",
}


def microphone_authorization_status() -> Optional[str]:
    """Return the current macOS Microphone TCC status.

    ``"authorized"`` — green light, ``sounddevice`` opens fine.
    ``"denied"``     — user previously hit Don't Allow; prompt
                       won't show again, only System Settings can
                       flip it.
    ``"restricted"`` — parental / MDM lockdown; user can't change.
    ``"not_determined"`` — never asked yet; the next
                          ``request_microphone_access`` call will
                          surface the system prompt.
    ``None``         — non-macOS platforms (no per-app gate).
    """
    if sys.platform != "darwin":
        return None
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except ImportError:  # pragma: no cover — pyobjc-AVFoundation missing
        return None
    try:
        status_int = AVCaptureDevice.authorizationStatusForMediaType_(
            AVMediaTypeAudio,
        )
    except Exception:  # pragma: no cover — defensive
        return None
    return _STATUS_BY_INT.get(int(status_int))


def request_microphone_access(
    on_result: Optional[Callable[[bool], None]] = None,
) -> bool:
    """Trigger the macOS microphone prompt for the first-time
    case (status == ``"not_determined"``).

    On any other status — and on non-macOS — this is a no-op
    returning ``False`` to make the caller fall back to
    :func:`open_microphone_settings`.

    The prompt itself fires asynchronously: macOS shows the
    system dialog and calls ``on_result(granted: bool)`` from a
    background thread once the user clicks Allow / Don't Allow.
    Callers that need a UI refresh on completion should pass an
    ``on_result`` callback (the GUI thread should never block on
    the prompt — the user might leave the dialog up indefinitely).
    """
    if sys.platform != "darwin":
        return False
    if microphone_authorization_status() != "not_determined":
        return False
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except ImportError:  # pragma: no cover
        return False

    def _completion(granted: bool) -> None:
        if on_result is not None:
            try:
                on_result(bool(granted))
            except Exception:  # pragma: no cover — defensive
                pass

    try:
        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio, _completion,
        )
        return True
    except Exception:  # pragma: no cover — defensive
        return False


def open_microphone_settings() -> bool:
    """Launch the System Settings pane that lists Microphone
    clients so the user can toggle our process on after a deny /
    restrict.  No-op on non-macOS platforms.

    Returns ``True`` on success.
    """
    if sys.platform != "darwin":
        return False
    import subprocess

    url = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
    try:
        subprocess.Popen(["open", url])
        return True
    except Exception:  # pragma: no cover — defensive
        return False
