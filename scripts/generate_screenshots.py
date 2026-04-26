"""Render the Qt UI offscreen and dump per-section PNG screenshots.

Run with::

    QT_QPA_PLATFORM=offscreen uv run python scripts/generate_screenshots.py

Drops files into ``docs/screenshots/``. Used to populate the README.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from pathlib import Path

# NB: do NOT force offscreen here — that platform plugin has no link to the
# native font subsystem on Windows, so every label renders as tofu (□).
# We use the real "windows" platform and rely on widget.grab() to capture
# off-screen anyway.
os.environ.pop("QT_QPA_PLATFORM", None)

# Project root must be on sys.path so we can import app.* directly.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.gui.app import build_application  # noqa: E402
from app.gui.log_bridge import QtLogBridge  # noqa: E402
from app.history_manager import TranscriptionEntry  # noqa: E402


OUT_DIR = _ROOT / "docs" / "screenshots"
WINDOW_SIZE = QSize(1100, 720)


class _FakeHistory:
    """Stand-in for HistoryManager with a realistic-looking dataset."""

    def __init__(self) -> None:
        now = time.time()
        self._entries = [
            TranscriptionEntry(
                timestamp=now - 60,
                text="Quick reminder: ship the PySide migration this week.",
                duration=3.4,
                model="large-v3",
                language="en",
            ),
            TranscriptionEntry(
                timestamp=now - 1200,
                text=(
                    "Customer feedback summary: the Logs tab finally shows "
                    "what happens during transcription, and the auto-save "
                    "Shortcuts tab is much friendlier."
                ),
                duration=11.7,
                model="large-v3",
                language="en",
            ),
            TranscriptionEntry(
                timestamp=now - 3600,
                text="Action item: reproduce the duplicate-launch warning on a clean install.",
                duration=4.9,
                model="large-v3",
                language="en",
            ),
            TranscriptionEntry(
                timestamp=now - 7200,
                text="Daily standup notes — keeping the in-process backend warm at 1.4 GB VRAM, fine for now.",
                duration=6.2,
                model="medium",
                language="en",
            ),
            TranscriptionEntry(
                timestamp=now - 86400,
                text="Yesterday I tried the turbo-int8 build for quick replies — surprisingly usable on a 6 GB GPU.",
                duration=4.0,
                model="turbo-int8",
                language="en",
            ),
        ]

    def get_entries(self):
        return list(self._entries)

    def clear_history(self):
        self._entries.clear()


class _FakeConfig:
    def __init__(self) -> None:
        self._data = {
            "whisper": {"model": "large-v3"},
            "hotkey": {
                "start_recording_hotkey": "ctrl+f2",
                "stop_recording_hotkey": "ctrl+f3",
            },
            "clipboard": {"auto_paste": True},
        }

    def get_setting(self, section: str, key: str):
        return self._data.get(section, {}).get(key)

    def update_user_setting(self, section: str, key: str, value):
        self._data.setdefault(section, {})[key] = value


def _seed_logs(window) -> None:
    """Push a handful of realistic-looking log lines into the LogsView."""
    bridge = QtLogBridge(parent=window)
    bridge.line_received.connect(window.logs_view.append_line)

    logger = logging.getLogger("lazy_to_text")
    logger.addHandler(bridge.handler())
    logger.setLevel(logging.DEBUG)
    try:
        logger.info("Audio feedback enabled...")
        logger.info("Recording stack built: model=large-v3 kind=faster_whisper device=auto compute_type=float16")
        logger.info("Primary instance acquired mutex LazyToTextQt_SingleInstance")
        logger.info("Hotkeys registered: ctrl+f2 (start) / ctrl+f3 (stop)")
        logger.info("Backend running on tcp://localhost:10300")
        logger.warning("Container log buffer at 80% — consider rotating soon.")
        logger.info("Transcription complete (3.4s) — 'Quick reminder: ship the PySide migration...'")
        logger.debug("[Pipeline] Audio data memory freed")
    finally:
        logger.removeHandler(bridge.handler())


def _seed_resource_metrics(window) -> None:
    """Push a realistic synthetic sample into the topbar's resource
    widget. Without this the screenshot is captured before the real
    ``ResourceMonitor`` has produced its first sample (it polls every
    2 s), so CPU/RAM render as 0% — and on machines without an NVIDIA
    driver the GPU/VRAM blocks would also be hidden, leaving a half-
    empty bar that misrepresents what the app actually shows in use."""
    window.topbar.set_resource_metrics(
        {
            "cpu_percent": 23.0,
            "ram_percent": 38.7,
            "ram_used_mb": 12_700.0,
            "ram_total_mb": 32_768.0,
            "gpu_util_percent": 67.0,
            "gpu_vram_used_mb": 4_300.0,
            "gpu_vram_total_mb": 8_192.0,
        }
    )


def _capture(window, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    pixmap = window.grab()
    pixmap.save(str(path), "PNG")
    print(f"  wrote {path.relative_to(_ROOT)} ({pixmap.width()}x{pixmap.height()})")
    return path


def main() -> int:
    config = _FakeConfig()
    history = _FakeHistory()

    app, window = build_application(
        config=config,
        history=history,
        install_logs=False,
    )
    window.resize(WINDOW_SIZE)
    window.show()
    app.processEvents()

    _seed_logs(window)
    _seed_resource_metrics(window)

    # Force one initial paint so the layout settles before we grab.
    for _ in range(5):
        app.processEvents()

    sections = ("models", "shortcuts", "history", "logs")
    for key in sections:
        window.sidebar.set_active(key)
        # Repeatedly drain events; offscreen platform sometimes needs a few
        # ticks before the layout reflects the new active view.
        for _ in range(8):
            app.processEvents()
        _capture(window, key)

    # Restore default and capture a "hero" shot showing Models from a fresh
    # angle.
    window.sidebar.set_active("models")
    for _ in range(8):
        app.processEvents()
    _capture(window, "hero")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
