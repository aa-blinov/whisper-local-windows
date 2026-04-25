"""Application logs view — colour-coded by level + logger source."""

from __future__ import annotations

import html
from typing import Optional

from PySide6.QtGui import QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Loggers that are *technically* informative but flood the view with
# HTTP noise during model loads. Hidden by default; the
# "Show network logs" checkbox brings them back.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "requests",
    "asyncio",
)


# Colours map to design-token values from theme.py — kept here as
# literals because QPlainTextEdit's HTML rendering ignores QSS.
_COLOR_TIMESTAMP = "#7d828d"      # text_muted
_COLOR_NAME_OWN = "#7aa2ff"        # accent_hover
_COLOR_NAME_OTHER = "#7d828d"      # text_muted
_COLOR_MESSAGE = "#f5f6f8"         # text_primary
_COLOR_MESSAGE_MUTED = "#b8bcc6"   # text_secondary
_COLOR_INFO = "#86efac"            # success-tint
_COLOR_DEBUG = "#7d828d"           # text_muted
_COLOR_WARNING = "#fbbf24"         # warning
_COLOR_ERROR = "#fca5a5"           # danger-tint
_COLOR_CRITICAL = "#fca5a5"


def _is_noisy(name: str) -> bool:
    """Match top-level package — ``httpx`` matches both ``httpx`` and
    ``httpx._client``, etc."""
    head = name.split(".", 1)[0]
    return head in _NOISY_LOGGERS


def _level_color(level: str) -> str:
    return {
        "DEBUG": _COLOR_DEBUG,
        "INFO": _COLOR_INFO,
        "WARNING": _COLOR_WARNING,
        "ERROR": _COLOR_ERROR,
        "CRITICAL": _COLOR_CRITICAL,
    }.get(level.upper(), _COLOR_INFO)


def _format_record_html(asctime: str, level: str, name: str, message: str) -> str:
    """Render a single log line as inline-styled HTML.

    QPlainTextEdit accepts HTML via ``appendHtml`` but doesn't honour
    QSS, so colours are baked in as inline styles using the same
    palette as theme.py.
    """
    name_color = _COLOR_NAME_OWN
    if _is_noisy(name) or not name.startswith("app."):
        name_color = _COLOR_NAME_OTHER

    msg_color = _COLOR_MESSAGE
    if level.upper() == "DEBUG":
        msg_color = _COLOR_MESSAGE_MUTED

    parts = [
        f'<span style="color:{_COLOR_TIMESTAMP}">{html.escape(asctime)}</span>',
        f'<span style="color:{_level_color(level)};font-weight:600">'
        f'[{html.escape(level)}]</span>',
        f'<span style="color:{name_color}">{html.escape(name)}:</span>',
        f'<span style="color:{msg_color}">{html.escape(message)}</span>',
    ]
    # Single non-breaking-space-joined paragraph; QPlainTextEdit
    # converts each appendHtml call into one block.
    return "&nbsp;".join(parts)


class LogsView(QWidget):
    DEFAULT_MAX_LINES = 5000

    def __init__(
        self,
        max_lines: int = DEFAULT_MAX_LINES,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("LogsView")
        self._max_lines = max(1, int(max_lines))
        self._show_network = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)

        self._network_toggle = QCheckBox("Show network logs", self)
        self._network_toggle.setObjectName("ShowNetworkLogs")
        self._network_toggle.setChecked(False)
        self._network_toggle.toggled.connect(self._on_toggle_network)
        header.addWidget(self._network_toggle)

        header.addStretch(1)

        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setObjectName("ClearLogsButton")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)

        root.addLayout(header)

        self._text = QPlainTextEdit(self)
        self._text.setObjectName("LogsTextArea")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(self._max_lines)
        # Wrap to widget width — log lines from the recording pipeline can be
        # 200+ chars, and a horizontal scrollbar makes them effectively
        # invisible.
        self._text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._text.setWordWrapMode(QTextOption.WrapAnywhere)
        # Smooth wheel scrolling — default jumps a couple of lines per
        # notch which feels stuttery in a tall log.
        self._text.verticalScrollBar().setSingleStep(20)
        root.addWidget(self._text, 1)

    # ---- public API ---------------------------------------------------------

    def append_line(self, text: str) -> None:
        """Legacy entry point — keeps existing tests / callers working
        when the bridge only emits formatted strings. Renders without
        colour or filtering since we don't know the level here."""
        self._text.appendPlainText(text)
        self._text.moveCursor(QTextCursor.End)

    def append_record(
        self, asctime: str, level: str, name: str, message: str
    ) -> None:
        """Render a structured log record with colours + filtering."""
        if _is_noisy(name) and not self._show_network:
            return
        self._text.appendHtml(_format_record_html(asctime, level, name, message))
        self._text.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        self._text.clear()

    # ---- internal -----------------------------------------------------------

    def _on_toggle_network(self, checked: bool) -> None:
        self._show_network = bool(checked)