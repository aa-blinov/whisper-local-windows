"""Shortcuts view — global hotkeys, microphone, and auto-paste toggle.

Edits persist immediately when a field commits (editingFinished on a
QLineEdit, currentIndexChanged on the QComboBox, toggled on the
QCheckBox); there is no Save button. A Reset button asks the controller
to restore default values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    test_mic_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ShortcutsView")

        # Suppresses save_requested emission while we are populating fields
        # programmatically (e.g. controller prefilling from config).
        self._suspend_emit = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(10)

        hint = QLabel(
            "Microphone, global hotkeys, paste behaviour. Changes save automatically.",
            self,
        )
        hint.setProperty("role", "muted")
        root.addWidget(hint)

        card = QFrame(self)
        card.setProperty("role", "card")
        form = QFormLayout(card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(10)

        self._device_combo = QComboBox(card)
        self._device_combo.setObjectName("MicrophoneCombo")
        # Long device names ("Микрофон (Razer BlackShark V2 Pro 2.4 …)") need
        # a wider popup than the combo box itself, otherwise they're cut off.
        self._device_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._device_combo.view().setMinimumWidth(420)
        self._device_combo.setStyleSheet(
            "QComboBox QAbstractItemView { min-width: 420px; }"
        )
        # Populated later via set_devices(); placeholder until then.
        self._device_combo.addItem("System default", None)
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        form.addRow("Microphone", self._device_combo)

        # Quick verifier: capture ~3 s, report peak/RMS so the user
        # knows the chosen device is actually picking up sound.
        mic_test_row = QHBoxLayout()
        mic_test_row.setSpacing(8)
        self._test_mic_btn = QPushButton("Test microphone", card)
        self._test_mic_btn.setObjectName("TestMicrophoneButton")
        self._test_mic_btn.clicked.connect(self.test_mic_requested.emit)
        mic_test_row.addWidget(self._test_mic_btn)
        self._test_mic_label = QLabel("", card)
        self._test_mic_label.setObjectName("MicrophoneTestResult")
        self._test_mic_label.setProperty("role", "muted")
        mic_test_row.addWidget(self._test_mic_label, 1)
        form.addRow("", mic_test_row)

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

    def set_devices(
        self,
        devices: List[Tuple[int, str]],
        current: Optional[int] = None,
    ) -> None:
        """Populate the microphone dropdown. ``devices`` is a list of
        ``(index, name)`` tuples. ``current`` is the index to preselect, or
        ``None`` for system default."""
        self._suspend_emit = True
        try:
            self._device_combo.clear()
            self._device_combo.addItem("System default", None)
            for idx, name in devices:
                self._device_combo.addItem(f"[{idx}] {name}", idx)

            if current is not None:
                for i in range(self._device_combo.count()):
                    if self._device_combo.itemData(i) == current:
                        self._device_combo.setCurrentIndex(i)
                        break
        finally:
            self._suspend_emit = False

    def start_hotkey(self) -> str:
        return self._start_edit.text().strip()

    def stop_hotkey(self) -> str:
        return self._stop_edit.text().strip()

    def auto_paste(self) -> bool:
        return self._auto_paste_cb.isChecked()

    def device_index(self) -> Optional[int]:
        return self._device_combo.currentData()

    def values(self) -> Dict[str, Any]:
        return {
            "start_hotkey": self.start_hotkey(),
            "stop_hotkey": self.stop_hotkey(),
            "auto_paste": self.auto_paste(),
            "device": self.device_index(),
        }

    # ---- internal -----------------------------------------------------------

    def _emit_save(self) -> None:
        if self._suspend_emit:
            return
        self.save_requested.emit(self.values())

    def _on_auto_paste_toggled(self, _checked: bool) -> None:
        self._emit_save()

    def _on_device_changed(self, _idx: int) -> None:
        self._emit_save()

    # ---- mic test feedback --------------------------------------------------

    def show_mic_test_running(self) -> None:
        self._test_mic_btn.setEnabled(False)
        self._test_mic_label.setText("Listening… speak now (3 s)")
        self._test_mic_label.setProperty("role", "muted")
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)

    def show_mic_test_result(self, peak: float, rms: float) -> None:
        self._test_mic_btn.setEnabled(True)
        if peak < 0.01:
            text = (
                f"Silence detected (peak {peak:.3f}). "
                "Check the device or speak louder."
            )
            role = "test-result-bad"
        elif peak < 0.08:
            text = (
                f"Quiet input (peak {peak:.3f}, rms {rms:.3f}). "
                "Audible but on the low side."
            )
            role = "test-result-warn"
        else:
            text = (
                f"Looks good — peak {peak:.3f}, rms {rms:.3f}."
            )
            role = "test-result-good"
        self._test_mic_label.setText(text)
        self._test_mic_label.setProperty("role", role)
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)

    def show_mic_test_error(self, reason: str) -> None:
        self._test_mic_btn.setEnabled(True)
        self._test_mic_label.setText(f"Test failed: {reason}")
        self._test_mic_label.setProperty("role", "test-result-bad")
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)
