"""Compact GPU/CPU/RAM stats widget for the topbar.

Three horizontal mini-blocks (GPU / CPU / RAM) laid out in a row.
Each block has a tiny label, a thin progress bar, and a numeric
readout. Layout the painter does itself — Qt QSS can't draw
labelled bars cleanly, so this widget owns its rendering. GPU
block hides itself if NVML isn't reporting a device.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget


_BG = QColor("#252932")          # bg_elevated — chip surface
_BORDER = QColor("#2d3140")      # border
_TRACK = QColor("#1a1d24")       # bar background
_TEXT = QColor("#b8bcc6")        # text_secondary — value
_TEXT_MUTED = QColor("#7d828d")  # text_muted — label

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
        # One row with up to 4 blocks (CPU / RAM / GPU util / VRAM).
        # On machines without an NVIDIA driver only the first two
        # render. 460×32 px keeps every value readable without
        # truncation.
        self.setFixedSize(460, 32)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._cpu_percent: float = 0.0
        self._ram_percent: float = 0.0
        self._ram_used_mb: float = 0.0
        self._ram_total_mb: float = 0.0
        self._gpu_vram_percent: Optional[float] = None
        self._gpu_vram_used_mb: float = 0.0
        self._gpu_vram_total_mb: float = 0.0
        self._gpu_util_percent: Optional[float] = None

        self._font = QFont(self.font())
        self._font.setPointSizeF(max(8.0, self._font.pointSizeF() - 1.0))

        self._refresh_tooltip()

    # ---- public API ---------------------------------------------------------

    def set_metrics(self, m: Dict[str, Any]) -> None:
        self._cpu_percent = float(m.get("cpu_percent", 0.0))
        self._ram_percent = float(m.get("ram_percent", 0.0))
        self._ram_used_mb = float(m.get("ram_used_mb", 0.0))
        self._ram_total_mb = float(m.get("ram_total_mb", 0.0))
        if (
            "gpu_vram_used_mb" in m
            and "gpu_vram_total_mb" in m
            and m["gpu_vram_total_mb"]
        ):
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
        painter.setPen(_BORDER)
        painter.setBrush(_BG)
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), radius, radius)

        blocks = self._block_specs()
        if not blocks:
            return

        outer_pad = 8
        between = 14
        usable = rect.width() - 2 * outer_pad - between * (len(blocks) - 1)
        block_w = usable // len(blocks)

        x = rect.x() + outer_pad
        for label, value_text, percent in blocks:
            block_rect = QRect(x, rect.y(), block_w, rect.height())
            self._paint_block(painter, block_rect, label, value_text, percent)
            x += block_w + between

    def _paint_block(
        self,
        painter: QPainter,
        rect: QRect,
        label: str,
        value_text: str,
        percent: float,
    ) -> None:
        fm = QFontMetrics(self._font)

        # Top half: label (left, muted) + value (right, bright).
        top_h = rect.height() // 2
        top_rect = QRect(rect.x(), rect.y(), rect.width(), top_h)
        painter.setPen(_TEXT_MUTED)
        painter.drawText(
            top_rect, Qt.AlignBottom | Qt.AlignLeft, label
        )
        painter.setPen(_TEXT)
        painter.drawText(
            top_rect, Qt.AlignBottom | Qt.AlignRight, value_text
        )

        # Bottom half: thin progress bar.
        bar_h = 5
        bar_y = rect.y() + top_h + 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(_TRACK)
        painter.drawRoundedRect(rect.x(), bar_y, rect.width(), bar_h, 2, 2)
        clamped = max(0.0, min(percent, 100.0))
        fill_w = int(rect.width() * clamped / 100.0)
        if fill_w > 0:
            painter.setBrush(_fill_color_for(clamped))
            painter.drawRoundedRect(rect.x(), bar_y, fill_w, bar_h, 2, 2)

    # ---- helpers ------------------------------------------------------------

    def _block_specs(self) -> List[Tuple[str, str, float]]:
        # Order matters — most-relevant-to-everyone first
        # (CPU, RAM), then GPU breakdown (utilisation + VRAM).
        # GPU blocks drop out when NVML isn't available so the
        # widget collapses to the two-block CPU/RAM layout cleanly.
        blocks: List[Tuple[str, str, float]] = [
            ("CPU", f"{self._cpu_percent:.0f}%", self._cpu_percent),
            (
                "RAM",
                f"{self._ram_used_mb / 1024.0:.1f}/{self._ram_total_mb / 1024.0:.1f} GB",
                self._ram_percent,
            ),
        ]
        if self._gpu_util_percent is not None:
            blocks.append(
                ("GPU", f"{self._gpu_util_percent:.0f}%", self._gpu_util_percent)
            )
        if self._gpu_vram_percent is not None:
            used_gb = self._gpu_vram_used_mb / 1024.0
            total_gb = self._gpu_vram_total_mb / 1024.0
            blocks.append(
                ("VRAM", f"{used_gb:.1f}/{total_gb:.1f} GB", self._gpu_vram_percent)
            )
        return blocks

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
