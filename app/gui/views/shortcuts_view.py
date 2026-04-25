"""Shortcuts view — global hotkeys and auto-paste toggle.

Edits persist immediately when a field commits (editingFinished on a
QLineEdit, toggled on the QCheckBox); there is no Save button. A Reset
button asks the controller to restore default values.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ShortcutsView(QWidget):
    save_requested = Signal(dict)
    reset_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ShortcutsView")

        # Suppresses save_requested emission while we are populating fields
        # programmatically (e.g. controller prefilling from config).
        self._suspend_emit = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        title = QLabel("Shortcuts", self)
        title.setProperty("role", "title")
        root.addWidget(title)

        hint = QLabel(
            "Global hotkeys and paste behaviour. Changes save automatically.",
            self,
        )
        hint.setProperty("role", "muted")
        root.addWidget(hint)

        card = QFrame(self)
        card.setProperty("role", "card")
        form = QFormLayout(card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        self._start_edit = QLineEdit(card)
        self._start_edit.setObjectName("StartHotkeyEdit")
        self._start_edit.setPlaceholderText("e.g. ctrl+f2")
        self._start_edit.editingFinished.connect(self._emit_save)
        form.addRow("Start recording", self._start_edit)

        self._stop_edit = QLineEdit(card)
        self._stop_edit.setObjectName("StopHotkeyEdit")
        self._stop_edit.setPlaceholderText("e.g. ctrl+f3")
        self._stop_edit.editingFinished.connect(self._emit_save)
        form.addRow("Stop recording", self._stop_edit)

        self._auto_paste_cb = QCheckBox("Auto-paste transcription", card)
        self._auto_paste_cb.setObjectName("AutoPasteCheckbox")
        self._auto_paste_cb.toggled.connect(self._on_auto_paste_toggled)
        form.addRow("", self._auto_paste_cb)

        root.addWidget(card)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._reset_btn = QPushButton("Reset to defaults", self)
        self._reset_btn.setObjectName("ResetShortcutsButton")
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        footer.addWidget(self._reset_btn)
        root.addLayout(footer)

        root.addStretch(1)

    # ---- public API ---------------------------------------------------------

    def set_values(
        self,
        start_hotkey: str,
        stop_hotkey: str,
        auto_paste: bool,
    ) -> None:
        # Programmatic update — must not feed back into save_requested.
        self._suspend_emit = True
        try:
            self._start_edit.setText(start_hotkey)
            self._stop_edit.setText(stop_hotkey)
            self._auto_paste_cb.setChecked(bool(auto_paste))
        finally:
            self._suspend_emit = False

    def start_hotkey(self) -> str:
        return self._start_edit.text().strip()

    def stop_hotkey(self) -> str:
        return self._stop_edit.text().strip()

    def auto_paste(self) -> bool:
        return self._auto_paste_cb.isChecked()

    def values(self) -> Dict[str, Any]:
        return {
            "start_hotkey": self.start_hotkey(),
            "stop_hotkey": self.stop_hotkey(),
            "auto_paste": self.auto_paste(),
        }

    # ---- internal -----------------------------------------------------------

    def _emit_save(self) -> None:
        if self._suspend_emit:
            return
        self.save_requested.emit(self.values())

    def _on_auto_paste_toggled(self, _checked: bool) -> None:
        self._emit_save()
