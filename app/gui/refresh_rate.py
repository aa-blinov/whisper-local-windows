"""Display refresh-rate helpers.

A handful of UI subsystems benefit from running at the user's native
monitor rate instead of a hard-coded 60 Hz:

- ``smooth_scroll`` — wheel-driven scroll animation; visibly stutters
  at 60 Hz / 144 Hz mismatch.
- VU meter — at 30 Hz the audio-level bar jumps in chunks on a
  144 Hz display.

The helpers below detect the primary screen's refresh rate once
(per call), clamp it to a sane band, and convert it to a millisecond
tick interval.  Both lookups are cheap and Qt-thread-safe; calling
them from a constructor is fine.

Clamps:

- floor 60 Hz — some monitors briefly report 30 Hz on wake / sleep,
  which would degenerate the VU meter to a slideshow.
- ceil 240 Hz — capping CPU burn on rare 360 / 480 Hz displays.  The
  visual difference between 240 and 480 Hz is imperceptible for our
  workloads (scroll easing, audio level fade).
"""

from __future__ import annotations


def display_refresh_rate() -> int:
    """Primary screen refresh rate, clamped to ``[60, 240]`` Hz.

    Falls back to ``60`` whenever Qt has no primary screen (headless
    test, sandbox env) or raises during the lookup.
    """
    try:
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
    except Exception:
        return 60
    if screen is None:
        return 60
    try:
        rate = float(screen.refreshRate())
    except Exception:
        return 60
    rounded = int(round(rate))
    return max(60, min(240, rounded))


def tick_interval_ms(max_rate: int | None = None) -> int:
    """Per-tick interval that aligns with the display's frame cadence.

    ``max_rate`` lets callers opt into a lower ceiling — e.g. the VU
    meter passes ``max_rate=60`` because the underlying audio buffer
    is only sampled every ~10 ms, so >60 Hz updates show duplicate
    frames anyway.

    Returns the rounded ``1000 / rate`` so we don't generate negative
    or fractional intervals.  Minimum 4 ms (240 Hz cap of the helper).
    """
    rate = display_refresh_rate()
    if max_rate is not None:
        rate = min(rate, int(max_rate))
    rate = max(1, rate)  # ultimate safety
    return max(1, int(round(1000 / rate)))
