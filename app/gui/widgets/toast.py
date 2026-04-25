"""Transient notification banner overlaid on the main window.

Used to confirm that a finished transcription landed on the
clipboard — successful runs were silent before, leaving the user
unsure whether the hotkey actually did anything.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


_DEFAULT_DURATION_MS = 3000
_MAX_PREVIEW_CHARS = 80


def _truncate(text: str, limit: int = _MAX_PREVIEW_CHARS) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class Toast(QFrame):
    """Top-right floating banner that auto-hides after a few seconds.

    The widget is parented to a host (typically ``MainWindow``) and
    positions itself in the top-right corner via ``move``. It does
    nothing until ``show_message`` is called, then re-arms its
    auto-hide timer on every subsequent call so back-to-back
    transcriptions keep extending the banner instead of stacking
    multiple toasts.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setProperty("role", "toast")
        self.setFrameShape(QFrame.NoFrame)
        # The toast floats above sibling widgets; turn off mouse
        # interaction so clicks pass through to whatever is beneath.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        self._title = QLabel("Copied", self)
        self._title.setObjectName("ToastTitle")
        self._title.setProperty("role", "toast-title")
        layout.addWidget(self._title)

        self._body = QLabel("", self)
        self._body.setObjectName("ToastBody")
        self._body.setProperty("role", "toast-body")
        layout.addWidget(self._body, 1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

        self.hide()

    # ---- public API ---------------------------------------------------------

    def show_message(
        self,
        text: str,
        duration_ms: int = _DEFAULT_DURATION_MS,
    ) -> None:
        """Display a confirmation banner with a preview of ``text``.

        Re-arms the auto-hide timer if the toast is already visible.
        """
        preview = _truncate(text)
        if not preview:
            return
        self._body.setText(preview)
        self.adjustSize()
        self._reposition()
        self.raise_()
        self.show()
        self._timer.start(max(500, int(duration_ms)))

    def hide_message(self) -> None:
        self._timer.stop()
        self.hide()

    # ---- internal -----------------------------------------------------------

    def _reposition(self) -> None:
        host = self.parentWidget()
        if host is None:
            return
        margin = 16
        # Bottom-right corner so the banner never lands on top of the
        # search bar / first-row content of whatever view is active.
        # Same convention as Slack / Discord / VS Code notifications.
        x = host.width() - self.width() - margin
        y = host.height() - self.height() - margin
        self.move(max(margin, x), max(0, y))

    # Re-anchor when the host resizes.
    def parentResized(self) -> None:  # pragma: no cover — convenience hook
        self._reposition()
