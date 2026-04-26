"""Settings view — global hotkeys, microphone, and auto-paste toggle.

Three independent cards (Audio input / Hotkeys / Clipboard) make the
panel easier to scan than a single flat form. Edits persist
immediately on commit (``editingFinished`` for line edits,
``currentIndexChanged`` for the combo, ``toggled`` for the checkbox);
there is no Save button. A Reset button restores defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _make_section_card(title: str, parent: QWidget) -> tuple[QFrame, QFormLayout]:
    """Build a card-styled QFrame with a section title and an empty
    QFormLayout ready for rows.

    Returns ``(card, form)`` so the caller can keep adding rows. The
    title sits inside the card, above the form, with consistent
    padding.
    """
    card = QFrame(parent)
    card.setProperty("role", "card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 16, 20, 16)
    layout.setSpacing(12)

    header = QLabel(title, card)
    header.setProperty("role", "section-header")
    layout.addWidget(header)

    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(10)
    form.setLabelAlignment(form.labelAlignment())  # default left
    form.setFormAlignment(form.formAlignment())
    form.setRowWrapPolicy(QFormLayout.DontWrapRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    layout.addLayout(form)
    return card, form


class ShortcutsView(QWidget):
    save_requested = Signal(dict)
    reset_requested = Signal()
    test_mic_requested = Signal()
    # Storage card — view delegates path-picking to the controller so
    # QFileDialog stays out of the widget code (cleaner tests).
    storage_path_change_requested = Signal()
    storage_reset_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ShortcutsView")

        # Suppresses save_requested emission while we are populating fields
        # programmatically (e.g. controller prefilling from config).
        self._suspend_emit = False

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 22)
        root.setSpacing(14)

        hint = QLabel(
            "Microphone, global hotkeys, paste behaviour. "
            "Changes save automatically.",
            self,
        )
        hint.setProperty("role", "muted")
        root.addWidget(hint)

        # ---- Audio input card -------------------------------------------
        audio_card, audio_form = _make_section_card("Audio input", self)

        self._device_combo = QComboBox(audio_card)
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
        audio_form.addRow("Microphone", self._device_combo)

        # Quick verifier: capture ~3 s, report peak/RMS so the user
        # knows the chosen device is actually picking up sound.
        mic_test_row = QHBoxLayout()
        mic_test_row.setSpacing(10)
        self._test_mic_btn = QPushButton("Test microphone", audio_card)
        self._test_mic_btn.setObjectName("TestMicrophoneButton")
        # Without ``NoFocus`` clicking the button puts keyboard focus
        # on it; once we disable it for the 3-second test, Qt chases
        # focus to the next focusable widget — the Start hotkey
        # QLineEdit — and the cursor lands inside it. Annoying.
        self._test_mic_btn.setFocusPolicy(Qt.NoFocus)
        self._test_mic_btn.clicked.connect(self.test_mic_requested.emit)
        mic_test_row.addWidget(self._test_mic_btn)
        self._test_mic_label = QLabel("", audio_card)
        self._test_mic_label.setObjectName("MicrophoneTestResult")
        self._test_mic_label.setProperty("role", "muted")
        self._test_mic_label.setWordWrap(True)
        mic_test_row.addWidget(self._test_mic_label, 1)
        audio_form.addRow("", mic_test_row)
        root.addWidget(audio_card)

        # ---- Hotkeys card -----------------------------------------------
        hotkeys_card, hotkeys_form = _make_section_card("Hotkeys", self)

        self._start_edit = QLineEdit(hotkeys_card)
        self._start_edit.setObjectName("StartHotkeyEdit")
        self._start_edit.setPlaceholderText("e.g. ctrl+f2")
        self._start_edit.editingFinished.connect(self._emit_save)
        hotkeys_form.addRow("Start recording", self._start_edit)

        self._stop_edit = QLineEdit(hotkeys_card)
        self._stop_edit.setObjectName("StopHotkeyEdit")
        self._stop_edit.setPlaceholderText("e.g. ctrl+f3")
        self._stop_edit.editingFinished.connect(self._emit_save)
        hotkeys_form.addRow("Stop recording", self._stop_edit)
        root.addWidget(hotkeys_card)

        # ---- Clipboard card ---------------------------------------------
        clipboard_card, clipboard_form = _make_section_card("Clipboard", self)

        self._auto_paste_cb = QCheckBox(
            "Auto-paste transcription into the focused window",
            clipboard_card,
        )
        self._auto_paste_cb.setObjectName("AutoPasteCheckbox")
        self._auto_paste_cb.toggled.connect(self._on_auto_paste_toggled)
        # Single full-width row — no left label needed for a checkbox
        # whose own text already describes it.
        clipboard_form.addRow(self._auto_paste_cb)
        root.addWidget(clipboard_card)

        # ---- Storage card -----------------------------------------------
        # User-pickable models directory — both Whisper (HF hub) and
        # GigaAM weights live under this root. Empty config value =
        # use the default ``<project>/models`` (or ``<exe>/models``
        # when frozen).
        #
        # Built by hand instead of via ``_make_section_card`` because
        # this card mixes a label-row with a button-row and a hint
        # paragraph; QFormLayout's spanning-row layout shrinks rows
        # whose label column is empty, squashing the buttons. A
        # straight QVBoxLayout sidesteps that entirely.
        storage_card = QFrame(self)
        storage_card.setObjectName("StorageCard")
        storage_card.setProperty("role", "card")
        storage_v = QVBoxLayout(storage_card)
        storage_v.setContentsMargins(20, 16, 20, 16)
        storage_v.setSpacing(10)

        storage_header = QLabel("Storage", storage_card)
        storage_header.setProperty("role", "section-header")
        storage_v.addWidget(storage_header)

        # Path row — leading caption + selectable label, all on one line.
        storage_path_row = QHBoxLayout()
        storage_path_row.setSpacing(16)
        storage_caption = QLabel("Models folder", storage_card)
        storage_path_row.addWidget(storage_caption)

        self._storage_path_label = QLabel("(loading…)", storage_card)
        self._storage_path_label.setObjectName("StoragePathLabel")
        self._storage_path_label.setProperty("role", "muted")
        self._storage_path_label.setWordWrap(True)
        self._storage_path_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        storage_path_row.addWidget(self._storage_path_label, 1)
        storage_v.addLayout(storage_path_row)

        # Button row — flush left, stretch on the right. Wrapped in a
        # QWidget rather than added as a bare QHBoxLayout because the
        # outer VBox doesn't reliably pick up the layout's sizeHint
        # in this nesting (hint label was rendering on top of the
        # button row's bottom edge).
        storage_btn_widget = QWidget(storage_card)
        storage_btn_widget.setObjectName("StorageButtonRow")
        storage_btn_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        storage_btn_row = QHBoxLayout(storage_btn_widget)
        storage_btn_row.setContentsMargins(0, 0, 0, 0)
        storage_btn_row.setSpacing(10)
        self._change_storage_btn = QPushButton("Change…", storage_btn_widget)
        self._change_storage_btn.setObjectName("ChangeStorageButton")
        self._change_storage_btn.setFocusPolicy(Qt.NoFocus)
        self._change_storage_btn.clicked.connect(
            self.storage_path_change_requested.emit
        )
        storage_btn_row.addWidget(self._change_storage_btn)

        self._reset_storage_btn = QPushButton(
            "Reset to default", storage_btn_widget,
        )
        self._reset_storage_btn.setObjectName("ResetStorageButton")
        self._reset_storage_btn.setFocusPolicy(Qt.NoFocus)
        # Disabled until a custom path is set — see ``set_storage_path``.
        self._reset_storage_btn.setEnabled(False)
        self._reset_storage_btn.clicked.connect(
            self.storage_reset_requested.emit
        )
        storage_btn_row.addWidget(self._reset_storage_btn)
        storage_btn_row.addStretch(1)
        # Match the wrapper's height to the buttons' sizeHint so the
        # outer VBox can't squish it below the button height.
        storage_btn_widget.setMinimumHeight(
            self._change_storage_btn.sizeHint().height(),
        )
        storage_v.addWidget(storage_btn_widget)

        storage_hint = QLabel(
            "Changes apply on next launch. Already-downloaded weights "
            "stay where they are.",
            storage_card,
        )
        storage_hint.setObjectName("StorageHint")
        storage_hint.setProperty("role", "muted")
        storage_hint.setWordWrap(True)
        storage_v.addWidget(storage_hint)
        root.addWidget(storage_card)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._reset_btn = QPushButton("Reset to defaults", self)
        self._reset_btn.setObjectName("ResetShortcutsButton")
        self._reset_btn.setFocusPolicy(Qt.NoFocus)
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

    def set_storage_path(self, path: str, is_default: bool) -> None:
        """Update the Storage card's path display.

        ``path`` is the *resolved* absolute path — not the raw config
        value. ``is_default`` toggles a ``(default)`` marker and
        disables the Reset button (no point resetting when we're
        already on the default).
        """
        if is_default:
            self._storage_path_label.setText(f"{path}  (default)")
            self._reset_storage_btn.setEnabled(False)
        else:
            self._storage_path_label.setText(path)
            self._reset_storage_btn.setEnabled(True)

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
