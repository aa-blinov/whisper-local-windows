"""Shortcuts view — global hotkeys and auto-paste toggle."""

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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ShortcutsView")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        title = QLabel("Shortcuts", self)
        title.setProperty("role", "title")
        root.addWidget(title)

        hint = QLabel("Global hotkeys and paste behaviour.", self)
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
        form.addRow("Start recording", self._start_edit)

        self._stop_edit = QLineEdit(card)
        self._stop_edit.setObjectName("StopHotkeyEdit")
        self._stop_edit.setPlaceholderText("e.g. ctrl+f3")
        form.addRow("Stop recording", self._stop_edit)

        self._auto_paste_cb = QCheckBox("Auto-paste transcription", card)
        self._auto_paste_cb.setObjectName("AutoPasteCheckbox")
        form.addRow("", self._auto_paste_cb)

        root.addWidget(card)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._save_btn = QPushButton("Save", self)
        self._save_btn.setObjectName("SaveShortcutsButton")
        self._save_btn.setProperty("role", "primary")
        self._save_btn.clicked.connect(self._emit_save)
        footer.addWidget(self._save_btn)
        root.addLayout(footer)

        root.addStretch(1)

    # ---- public API ---------------------------------------------------------

    def set_values(
        self,
        start_hotkey: str,
        stop_hotkey: str,
        auto_paste: bool,
    ) -> None:
        self._start_edit.setText(start_hotkey)
        self._stop_edit.setText(stop_hotkey)
        self._auto_paste_cb.setChecked(bool(auto_paste))

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
        self.save_requested.emit(self.values())
