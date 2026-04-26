"""Browser-like smooth scrolling for Qt scroll areas.

Qt's QPropertyAnimation is capped at ~60 updates/s regardless of the
display refresh rate. On 144 Hz monitors this produces visible
stutter because only 60 distinct scroll positions are generated per
second while the display can show 144.

This module replaces the animation with a QTimer at 7 ms (≈144 Hz)
that recalculates the scroll position from wall-clock time on every
tick. The easing is computed from elapsed milliseconds, not frame
count, so the curve shape is identical at any refresh rate — only
the number of steps changes (more steps = smoother on high-Hz
displays).

Usage::

    from app.gui.smooth_scroll import apply_smooth_scroll
    apply_smooth_scroll(my_scroll_area)
"""

from __future__ import annotations

import time

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QAbstractScrollArea

_DURATION_MS = 180.0   # total animation length (wall-clock ms)
_PX_PER_NOTCH = 100.0  # pixels per standard wheel notch (angleDelta = 120)
# 16 ms matches Windows' default timer resolution (15.6 ms) so the timer
# fires at a steady 60 Hz without jitter. Sub-16ms intervals cause
# irregular firing (7ms → 15ms → 7ms) which produces visible tearing.
# DWM vsync-composites Qt widget frames to the display rate automatically.
_TICK_MS = 16


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


class _SmoothScrollFilter(QObject):
    def __init__(self, area: QAbstractScrollArea) -> None:
        super().__init__(area)
        self._area = area
        self._start: float = 0.0
        self._target: float = 0.0
        self._t0: float = 0.0   # wall-clock start of current animation (seconds)

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ timer

    def _tick(self) -> None:
        elapsed_ms = (time.monotonic() - self._t0) * 1000.0
        t = min(1.0, elapsed_ms / _DURATION_MS)
        eased = _ease_out_cubic(t)

        bar = self._area.verticalScrollBar()
        bar.setValue(int(self._start + (self._target - self._start) * eased))

        if t >= 1.0:
            self._timer.stop()

    # ----------------------------------------------------------- event filter

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        angle = event.angleDelta().y()
        if angle == 0:
            return False

        delta_px = -angle * _PX_PER_NOTCH / 120.0
        bar = self._area.verticalScrollBar()

        if self._timer.isActive():
            # Re-anchor from wherever the animation is right now so rapid
            # flicks accumulate into the running motion.
            elapsed_ms = (time.monotonic() - self._t0) * 1000.0
            t = min(1.0, elapsed_ms / _DURATION_MS)
            current = self._start + (self._target - self._start) * _ease_out_cubic(t)
            self._start = current
            self._target = current + delta_px
            self._timer.stop()
        else:
            self._start = float(bar.value())
            self._target = self._start + delta_px

        lo, hi = float(bar.minimum()), float(bar.maximum())
        self._target = max(lo, min(hi, self._target))
        self._t0 = time.monotonic()
        self._timer.start()
        return True  # consume — prevent Qt's own jump-scroll


def apply_smooth_scroll(area: QAbstractScrollArea) -> None:
    """Enable high-refresh-rate smooth scrolling on *area*."""
    from PySide6.QtWidgets import QAbstractItemView
    if isinstance(area, QAbstractItemView):
        area.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    f = _SmoothScrollFilter(area)
    area.viewport().installEventFilter(f)
