"""Browser-like smooth scrolling for Qt scroll areas.

Qt's default wheel scroll is per-step and instant — each notch of the
mouse wheel jumps the scrollbar by ``singleStep`` pixels with no
animation. Browsers animate the scroll over ~200 ms with a decelerating
curve and accumulate rapid notches into a single ongoing animation so
fast flicks feel fluid.

``apply_smooth_scroll(area)`` installs a lightweight event filter on
the viewport that:

1. Intercepts ``Wheel`` events before Qt's handler sees them.
2. Converts ``angleDelta`` to a pixel target (100 px per notch —
   matches Chrome's default on Windows).
3. Starts (or re-targets) a ``QPropertyAnimation`` on the vertical
   scrollbar's ``value`` property with an ``OutCubic`` easing curve.
4. Accumulates rapid wheel ticks into the running animation's end
   value, so fast flicks build momentum instead of resetting.

Usage::

    from app.gui.smooth_scroll import apply_smooth_scroll
    apply_smooth_scroll(my_scroll_area)
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation
from PySide6.QtWidgets import QAbstractScrollArea

_DURATION_MS = 180
_PX_PER_NOTCH = 100  # pixels scrolled per standard wheel notch (angleDelta=120)


class _SmoothScrollFilter(QObject):
    def __init__(self, area: QAbstractScrollArea) -> None:
        super().__init__(area)
        self._area = area
        self._anim = QPropertyAnimation(area.verticalScrollBar(), b"value", self)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setDuration(_DURATION_MS)
        self._target: float = 0.0

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False

        angle = event.angleDelta().y()
        if angle == 0:
            return False

        bar = self._area.verticalScrollBar()
        delta_px = -angle * _PX_PER_NOTCH / 120.0

        # Accumulate into running animation so rapid flicks build momentum.
        if self._anim.state() == QPropertyAnimation.State.Running:
            self._target = float(self._anim.endValue()) + delta_px
        else:
            self._target = float(bar.value()) + delta_px

        self._target = max(float(bar.minimum()), min(float(bar.maximum()), self._target))

        self._anim.stop()
        self._anim.setStartValue(bar.value())
        self._anim.setEndValue(int(self._target))
        self._anim.start()
        return True  # event consumed — Qt won't do its own jump-scroll


def apply_smooth_scroll(area: QAbstractScrollArea) -> None:
    """Enable browser-like smooth scrolling on *area*.

    ``setVerticalScrollMode`` only exists on ``QAbstractItemView``
    subclasses (tables, lists, trees) — ``QScrollArea`` already
    moves its content per-pixel by default. We only need the
    animation filter here.
    """
    from PySide6.QtWidgets import QAbstractItemView
    if isinstance(area, QAbstractItemView):
        area.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    f = _SmoothScrollFilter(area)
    area.viewport().installEventFilter(f)
