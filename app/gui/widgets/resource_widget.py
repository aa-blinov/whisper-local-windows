"""Compact GPU/CPU/RAM stats widget for the topbar.

Custom-painted three-row mini display. Each row is a labelled
horizontal bar showing one metric. GPU row hides itself if NVML
isn't available on the host. A tooltip carries the full breakdown
(absolute MB, percentages) on hover.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget


_BG = QColor("#252932")          # bg_elevated — chip surface
_BORDER = QColor("#2d3140")      # border
_TRACK = QColor("#1a1d24")       # darker than chip — bar background
_TEXT = QColor("#b8bcc6")        # text_secondary
_TEXT_MUTED = QColor("#7d828d")  # text_muted

# Per-bar fill colour.
_GREEN = QColor("#4ade80")       # success
_AMBER = QColor("#f59e0b")       # warning
_RED = QColor("#ef4444")         # danger


def _fill_color_for(percent: float) -> QColor:
    if percent < 60:
        return _GREEN
    if percent < 85:
        return _AMBER
    return _RED


class ResourceWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResourceWidget")
        # Three labelled rows × 12 px each + padding fits 38–40 px.
        # Width covers two columns (label + bar) — 200 px is enough
        # for the longest GPU label ("GPU 8.0/8.0 GB").
        self.setFixedSize(200, 38)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._cpu_percent: float = 0.0
        self._ram_percent: float = 0.0
        self._ram_used_mb: float = 0.0
        self._ram_total_mb: float = 0.0
        self._gpu_vram_percent: Optional[float] = None
        self._gpu_vram_used_mb: float = 0.0
        self._gpu_vram_total_mb: float = 0.0
        self._gpu_util_percent: Optional[float] = None

        # Slightly smaller font so three rows fit cleanly.
        self._font = QFont(self.font())
        self._font.setPointSizeF(max(8.0, self._font.pointSizeF() - 1.0))

        self._refresh_tooltip()

    # ---- public API ---------------------------------------------------------

    def set_metrics(self, m: Dict[str, Any]) -> None:
        self._cpu_percent = float(m.get("cpu_percent", 0.0))
        self._ram_percent = float(m.get("ram_percent", 0.0))
        self._ram_used_mb = float(m.get("ram_used_mb", 0.0))
        self._ram_total_mb = float(m.get("ram_total_mb", 0.0))
        if "gpu_vram_used_mb" in m and "gpu_vram_total_mb" in m and m["gpu_vram_total_mb"]:
            self._gpu_vram_used_mb = float(m["gpu_vram_used_mb"])
            self._gpu_vram_total_mb = float(m["gpu_vram_total_mb"])
            self._gpu_vram_percent = (
                100.0 * self._gpu_vram_used_mb / self._gpu_vram_total_mb
            )
        else:
            self._gpu_vram_percent = None
        if "gpu_util_percent" in m:
            self._gpu_util_percent = float(m["gpu_util_percent"])
        else:
            self._gpu_util_percent = None

        self._refresh_tooltip()
        self.update()

    # ---- painting -----------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self._font)

        rect = self.rect()
        radius = 6
        # Chip background + border so the widget reads as a single
        # piece next to the recording / model pills, not as bare
        # paint on the topbar surface.
        painter.setPen(_BORDER)
        painter.setBrush(_BG)
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), radius, radius)

        rows = self._row_specs()
        if not rows:
            return

        # Lay rows out vertically inside the chip with even spacing.
        padding = 5
        gap = 2
        total_h = rect.height() - 2 * padding
        row_h = (total_h - gap * (len(rows) - 1)) / len(rows)

        fm = QFontMetrics(self._font)
        # Reserve a small label column so bars line up between rows.
        label_w = max(fm.horizontalAdvance(label) for label, _, _ in rows) + 4

        for i, (label, value_text, percent) in enumerate(rows):
            y = padding + i * (row_h + gap)
            row_rect = QRect(
                padding,
                int(y),
                rect.width() - 2 * padding,
                int(row_h),
            )
            self._paint_row(painter, fm, row_rect, label_w, label, value_text, percent)

    def _paint_row(
        self,
        painter: QPainter,
        fm: QFontMetrics,
        rect: QRect,
        label_w: int,
        label: str,
        value_text: str,
        percent: float,
    ) -> None:
        # Label on the left.
        painter.setPen(_TEXT_MUTED)
        painter.drawText(
            QRect(rect.x(), rect.y(), label_w, rect.height()),
            Qt.AlignVCenter | Qt.AlignLeft,
            label,
        )

        bar_x = rect.x() + label_w
        bar_y = rect.y() + (rect.height() - 6) // 2
        # Reserve space on the right for the numeric value.
        value_w = fm.horizontalAdvance(value_text) + 4
        bar_w = rect.right() - bar_x - value_w
        if bar_w < 20:
            bar_w = 20

        # Track.
        painter.setPen(Qt.NoPen)
        painter.setBrush(_TRACK)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, 6, 3, 3)

        # Fill.
        clamped = max(0.0, min(percent, 100.0))
        fill_w = int(bar_w * clamped / 100.0)
        if fill_w > 0:
            painter.setBrush(_fill_color_for(clamped))
            painter.drawRoundedRect(bar_x, bar_y, fill_w, 6, 3, 3)

        # Numeric value on the right.
        painter.setPen(_TEXT)
        painter.drawText(
            QRect(
                rect.right() - value_w,
                rect.y(),
                value_w,
                rect.height(),
            ),
            Qt.AlignVCenter | Qt.AlignRight,
            value_text,
        )

    # ---- helpers ------------------------------------------------------------

    def _row_specs(self) -> list[tuple[str, str, float]]:
        rows: list[tuple[str, str, float]] = []
        if self._gpu_vram_percent is not None:
            used_gb = self._gpu_vram_used_mb / 1024.0
            total_gb = self._gpu_vram_total_mb / 1024.0
            rows.append(
                ("GPU", f"{used_gb:.1f}/{total_gb:.1f} GB", self._gpu_vram_percent)
            )
        rows.append(("CPU", f"{self._cpu_percent:.0f}%", self._cpu_percent))
        rows.append(
            (
                "RAM",
                f"{self._ram_used_mb / 1024.0:.1f}/{self._ram_total_mb / 1024.0:.1f} GB",
                self._ram_percent,
            )
        )
        return rows

    def _refresh_tooltip(self) -> None:
        parts = []
        if self._gpu_vram_percent is not None:
            parts.append(
                "GPU VRAM: "
                f"{self._gpu_vram_used_mb / 1024.0:.2f} / "
                f"{self._gpu_vram_total_mb / 1024.0:.2f} GB "
                f"({self._gpu_vram_percent:.0f}%)"
            )
            if self._gpu_util_percent is not None:
                parts.append(f"GPU util: {self._gpu_util_percent:.0f}%")
        parts.append(f"CPU: {self._cpu_percent:.0f}%")
        parts.append(
            "RAM: "
            f"{self._ram_used_mb / 1024.0:.2f} / "
            f"{self._ram_total_mb / 1024.0:.2f} GB "
            f"({self._ram_percent:.0f}%)"
        )
        self.setToolTip("\n".join(parts))
