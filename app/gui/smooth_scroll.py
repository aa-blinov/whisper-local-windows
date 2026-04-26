"""Browser-like smooth scrolling for Qt scroll areas.

Implements the "FixedStep + COSINE" algorithm from qfluentwidgets
(zhiyiYo/PyQt-Fluent-Widgets).  Each wheel notch is broken into
``_STEPS_TOTAL`` equal-interval ticks whose per-tick displacement
follows a cosine bell curve:

    sub = (cos(x * π / m) + 1) / (2*m) * delta

where x = |step_done - m| and m = _STEPS_TOTAL / 2.  The curve
integrates exactly to ``delta`` so the total scroll distance per
notch is preserved.

Key properties:
- Ease-in-out feel: slow at start and end, peak in the middle.
- Accumulation: rapid notches append independent items to the queue;
  each tick sums contributions from all active items, so fast
  flicking naturally builds momentum.
- Consistent 16 ms ticks: matches the Windows default timer
  resolution, preventing the jitter that causes visual tearing.

Usage::

    from app.gui.smooth_scroll import apply_smooth_scroll
    apply_smooth_scroll(my_scroll_area)
"""

from __future__ import annotations

from collections import deque
from math import cos, pi

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QAbstractScrollArea

_FPS = 60
_DURATION_MS = 400            # total animation length per notch
_STEPS_TOTAL = int(_FPS * _DURATION_MS / 1000)   # 24
_PX_PER_NOTCH = 100.0         # base pixels per standard notch (angleDelta=120)
_STEP_RATIO = 1.5             # multiplier — 100 * 1.5 = 150 px/notch
_TICK_MS = int(1000 / _FPS)   # 16 ms — one Windows timer tick


def _sub_delta(delta: float, steps_left: int) -> float:
    """Cosine bell contribution for this step."""
    m = _STEPS_TOTAL / 2
    x = abs(_STEPS_TOTAL - steps_left - m)
    return (cos(x * pi / m) + 1) / (2 * m) * delta


class _SmoothScrollFilter(QObject):
    def __init__(self, area: QAbstractScrollArea) -> None:
        super().__init__(area)
        self._area = area
        # Queue of [remaining_delta_px, steps_left] pairs.
        # Each wheel notch appends one item; all active items contribute
        # to every tick, then items are dropped when steps_left reaches 0.
        self._queue: deque[list[float]] = deque()

        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ timer

    def _tick(self) -> None:
        if not self._queue:
            self._timer.stop()
            return

        total = 0.0
        for item in self._queue:
            step = _sub_delta(item[0], int(item[1]))
            total += step
            item[1] -= 1

        # Drop exhausted items.
        while self._queue and self._queue[0][1] <= 0:
            self._queue.popleft()

        bar = self._area.verticalScrollBar()
        bar.setValue(max(bar.minimum(), min(bar.maximum(), int(bar.value() + total))))

    # ----------------------------------------------------------- event filter

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        angle = event.angleDelta().y()
        if angle == 0:
            return False

        delta_px = -angle * _PX_PER_NOTCH * _STEP_RATIO / 120.0
        self._queue.append([delta_px, float(_STEPS_TOTAL)])

        if not self._timer.isActive():
            self._timer.start()
        return True


def apply_smooth_scroll(area: QAbstractScrollArea) -> None:
    """Enable qfluentwidgets-style smooth scrolling on *area*."""
    from PySide6.QtWidgets import QAbstractItemView
    if isinstance(area, QAbstractItemView):
        area.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    f = _SmoothScrollFilter(area)
    area.viewport().installEventFilter(f)
