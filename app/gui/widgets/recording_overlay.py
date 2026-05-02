"""Small always-on-top recording status overlay.

Unlike the in-window toast, this widget is a separate top-level tool
window so it can stay visible while the main app is hidden or unfocused.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class RecordingOverlay(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        flags = (
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
        )
        super().__init__(parent, flags)
        self.setObjectName("RecordingOverlay")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WA_MacAlwaysShowToolWindow, True)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("RecordingOverlaySurface")
        self._surface.setProperty("role", "recording-overlay-surface")
        self._surface.setMinimumHeight(56)
        root.addWidget(self._surface)

        surface_layout = QHBoxLayout(self._surface)
        surface_layout.setContentsMargins(16, 12, 16, 12)
        surface_layout.setSpacing(12)

        self._dot = QFrame(self._surface)
        self._dot.setObjectName("RecordingOverlayDot")
        self._dot.setProperty("role", "recording-overlay-dot")
        self._dot.setFixedSize(14, 14)
        surface_layout.addWidget(self._dot, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)

        self._title = QLabel("", self._surface)
        self._title.setObjectName("RecordingOverlayTitle")
        self._title.setProperty("role", "recording-overlay-title")
        text_col.addWidget(self._title)

        self._body = QLabel("", self._surface)
        self._body.setObjectName("RecordingOverlayBody")
        self._body.setProperty("role", "recording-overlay-body")
        text_col.addWidget(self._body)

        surface_layout.addLayout(text_col, 1)

        self._state = "idle"
        self.hide()

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        state = (state or "").strip().lower()
        if state == "recording":
            self._state = "recording"
            self._title.setText("Recording")
            self._body.setText("Speak now")
            self._dot.setProperty("state", "recording")
            self._refresh_styles()
            self._show_overlay()
            return
        if state == "processing":
            self._state = "processing"
            self._title.setText("Processing")
            self._body.setText("Transcribing speech")
            self._dot.setProperty("state", "processing")
            self._refresh_styles()
            self._show_overlay()
            return
        self._state = "idle"
        self.hide()

    def _show_overlay(self) -> None:
        self.adjustSize()
        self._reposition()
        self.show()
        if sys.platform == "darwin":
            self._apply_mac_window_behaviors()

    def _apply_mac_window_behaviors(self) -> None:
        """Apply native macOS behaviors for visibility in fullscreen apps."""
        if hasattr(self, "_mac_behaviors_applied"):
            return

        # Skip if not on a live Cocoa display (e.g. during headless tests)
        if QGuiApplication.platformName() != "cocoa":
            return

        try:
            import objc
            from AppKit import (
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSStatusWindowLevel,
            )

            # PySide6 winId() on macOS is the NSView pointer.
            view_id = int(self.winId())
            if not view_id:
                return

            # Wrap as objc object and find its window.
            ns_view = objc.objc_object(c_void_p=view_id)
            ns_window = ns_view.window()

            if ns_window:
                # 1. Allow window to float over fullscreen apps
                # 2. Allow window to appear on all Spaces/desktops
                # 3. Ensure it moves to the active space immediately
                from AppKit import (
                    NSWindowCollectionBehaviorMoveToActiveSpace,
                    NSWindowCollectionBehaviorIgnoresCycle,
                    NSWindowCollectionBehaviorStationary,
                )
                ns_window.setCollectionBehavior_(
                    NSWindowCollectionBehaviorCanJoinAllSpaces
                    | NSWindowCollectionBehaviorFullScreenAuxiliary
                    | NSWindowCollectionBehaviorMoveToActiveSpace
                    | NSWindowCollectionBehaviorIgnoresCycle
                    | NSWindowCollectionBehaviorStationary
                )
                
                # NSScreenSaverWindowLevel (1000) is very high, usually 
                # reserved for screen savers and system overlays. 
                # This ensures we are above the Notch, Menu Bar, and 
                # Fullscreen app shields.
                from AppKit import NSScreenSaverWindowLevel
                ns_window.setLevel_(NSScreenSaverWindowLevel)

                # Prevent the window from being hidden when the app is inactive
                ns_window.setHidesOnDeactivate_(False)
                ns_window.setCanHide_(False)

            self._mac_behaviors_applied = True
        except Exception:
            # Silently fail if native hooks aren't available
            pass

    def _reposition(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        margin_top = 28
        x = rect.x() + max(0, (rect.width() - self.width()) // 2)
        y = rect.y() + margin_top
        self.move(x, y)

    def _refresh_styles(self) -> None:
        self.style().unpolish(self._dot)
        self.style().polish(self._dot)
        self.style().unpolish(self._surface)
        self.style().polish(self._surface)
