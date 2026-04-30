"""Settings view — global hotkeys, microphone, and auto-paste toggle.

Three independent cards (Audio input / Hotkeys / Clipboard) make the
panel easier to scan than a single flat form. Edits persist
immediately on commit (``editingFinished`` for line edits,
``currentIndexChanged`` for the combo, ``toggled`` for the checkbox);
there is no Save button. A Reset button restores defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.gui.smooth_scroll import apply_smooth_scroll
from app.gui.views._accessibility_check import (
    is_accessibility_trusted,
    open_accessibility_settings,
)
from app.gui.views._hotkey_validation import validate_all

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
    QScrollArea,
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
    test_mic_requested = Signal()
    # Per-card reset signals — granular replacements for the old
    # single ``reset_requested`` footer button. Each card now owns
    # its own affordance so the user can revert one section without
    # nuking unrelated state.
    hotkeys_reset_requested = Signal()
    hf_token_reset_requested = Signal()
    # Storage card — view delegates path-picking to the controller so
    # QFileDialog stays out of the widget code (cleaner tests).
    storage_path_change_requested = Signal()
    storage_reset_requested = Signal()
    # 'Open folder' shortcut — the controller spawns Explorer
    # (subprocess + path stays out of the view).
    storage_open_requested = Signal()
    # Hugging Face card — fired on focus loss after the user edits
    # the token field. Controller persists + applies to env.
    hf_token_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ShortcutsView")

        # Suppresses save_requested emission while we are populating fields
        # programmatically (e.g. controller prefilling from config).
        self._suspend_emit = False

        # Snapshot of the user's Stop hotkey taken just before we
        # mirror the Start value over it on toggle-mode entry, so
        # un-ticking can restore exactly what was there before.
        # Only set on user-driven toggle (``_on_toggle_mode_changed``)
        # — ``set_values`` skips it because the values come from
        # config and don't need a "previous" copy.
        self._previous_stop_hotkey: Optional[str] = None

        # Outer layout = top hint pinned + scrollable card stack.
        # Without the scroll area Qt tried to fit every card into
        # whatever vertical space the window had; once we pushed
        # past 4-5 cards Qt started squishing form rows below
        # their min-height and labels rendered on top of inputs.
        # Mirrors the Models view's pattern exactly.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hint_wrapper = QWidget(self)
        hint_wrapper_layout = QVBoxLayout(hint_wrapper)
        # Bottom margin separates the description from the first card
        # so the text doesn't kiss the card border on scroll — without
        # this the muted hint visually merged with the dark card frame.
        hint_wrapper_layout.setContentsMargins(28, 22, 28, 14)
        hint_wrapper_layout.setSpacing(0)
        hint = QLabel(
            "Microphone, global hotkeys, paste behaviour. "
            "Changes save automatically.",
            hint_wrapper,
        )
        hint.setProperty("role", "muted")
        hint_wrapper_layout.addWidget(hint)
        outer.addWidget(hint_wrapper)

        scroll = QScrollArea(self)
        scroll.setObjectName("ShortcutsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        apply_smooth_scroll(scroll)
        outer.addWidget(scroll, 1)

        scroll_content = QWidget(scroll)
        scroll_content.setObjectName("ShortcutsScrollContent")
        scroll.setWidget(scroll_content)
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(28, 14, 28, 22)
        root.setSpacing(14)

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

        # Microphone row: dropdown + Test button inline. Same trick
        # as the HF card — kills the empty space a button-on-its-
        # own-row left next to the dropdown.
        from app.gui.widgets.vu_meter import VUMeter

        mic_input_row = QHBoxLayout()
        mic_input_row.setSpacing(10)
        mic_input_row.addWidget(self._device_combo, 1)
        self._test_mic_btn = QPushButton("Test microphone", audio_card)
        self._test_mic_btn.setObjectName("TestMicrophoneButton")
        # Without ``NoFocus`` clicking the button puts keyboard focus
        # on it; once we disable it for the 3-second test, Qt chases
        # focus to the next focusable widget — the Start hotkey
        # QLineEdit — and the cursor lands inside it. Annoying.
        self._test_mic_btn.setFocusPolicy(Qt.NoFocus)
        self._test_mic_btn.clicked.connect(self.test_mic_requested.emit)
        mic_input_row.addWidget(self._test_mic_btn)
        audio_form.addRow("Microphone", mic_input_row)

        # Result row: live VU meter (visible only during / after a
        # test) + the textual result. Sits below the input row —
        # collapses to a thin empty strip when nothing is running,
        # blooms into a meter + verdict line during / after a test.
        mic_result_row = QHBoxLayout()
        mic_result_row.setSpacing(10)
        self._test_mic_meter = VUMeter(audio_card)
        self._test_mic_meter.setObjectName("MicrophoneTestMeter")
        self._test_mic_meter.setVisible(False)
        mic_result_row.addWidget(self._test_mic_meter)

        self._test_mic_label = QLabel("", audio_card)
        self._test_mic_label.setObjectName("MicrophoneTestResult")
        self._test_mic_label.setProperty("role", "muted")
        self._test_mic_label.setWordWrap(True)
        mic_result_row.addWidget(self._test_mic_label, 1)
        audio_form.addRow("", mic_result_row)
        root.addWidget(audio_card)

        # ---- Hotkeys card -----------------------------------------------
        hotkeys_card, hotkeys_form = _make_section_card("Hotkeys", self)

        # macOS-only: warn the user when the process hasn't been
        # added to System Settings → Privacy & Security →
        # Accessibility. Without that, ``pynput``'s CGEventTap
        # silently returns no events at all and hotkeys "don't
        # work" with no on-screen explanation. Banner sits at the
        # top of the Hotkeys card so it's seen the moment the user
        # looks at hotkey settings; ``_refresh_accessibility_banner``
        # toggles it visible / hidden based on the current trusted
        # state.
        self._accessibility_banner = QFrame(hotkeys_card)
        self._accessibility_banner.setObjectName("AccessibilityWarningBanner")
        self._accessibility_banner.setProperty("role", "warning-banner")
        banner_layout = QHBoxLayout(self._accessibility_banner)
        banner_layout.setContentsMargins(12, 10, 12, 10)
        banner_layout.setSpacing(12)
        banner_text = QLabel(
            "macOS hasn't granted Accessibility access yet — global "
            "hotkeys won't fire until you add this app's terminal / "
            "IDE under System Settings → Privacy & Security → "
            "Accessibility, then restart it.",
            self._accessibility_banner,
        )
        banner_text.setWordWrap(True)
        banner_text.setProperty("role", "warning-banner-text")
        banner_layout.addWidget(banner_text, 1)
        self._open_accessibility_btn = QPushButton(
            "Open Accessibility settings", self._accessibility_banner,
        )
        self._open_accessibility_btn.setObjectName("OpenAccessibilityButton")
        self._open_accessibility_btn.setFocusPolicy(Qt.NoFocus)
        self._open_accessibility_btn.clicked.connect(open_accessibility_settings)
        banner_layout.addWidget(self._open_accessibility_btn, 0)
        self._accessibility_banner.setVisible(False)
        # The form's row spans both columns — the banner runs full
        # card width, not nested under the field column.
        hotkeys_form.addRow(self._accessibility_banner)
        # Re-check at every paint of the Settings tab; permissions
        # don't update live anyway (Mac requires a relaunch), but a
        # quick refresh here covers the case where the user clicked
        # the button, granted access, came back without restarting,
        # and we still flag it correctly.
        self._refresh_accessibility_banner()

        # Toggle-mode switch: when checked, the Start hotkey doubles
        # as the Stop hotkey — pressing it again stops the recording.
        # The HotkeyListener already supports this when start == stop,
        # but exposing it as an explicit checkbox is much friendlier
        # than asking the user to type the same combination into two
        # fields. Cancel is greyed out in this mode too — the toggle
        # flow is "one key, one job", and an extra cancel binding
        # adds friction without earning its keep.
        self._toggle_mode_cb = QCheckBox(
            "One hotkey for recording — press once to start, again to stop",
            hotkeys_card,
        )
        self._toggle_mode_cb.setObjectName("ToggleHotkeyCheckbox")
        self._toggle_mode_cb.setToolTip(
            "When on, the same hotkey starts and stops a recording. "
            "Stop and Cancel fields below become read-only — only the "
            "Start field is in use."
        )
        self._toggle_mode_cb.toggled.connect(self._on_toggle_mode_changed)
        hotkeys_form.addRow("", self._toggle_mode_cb)

        self._start_edit = QLineEdit(hotkeys_card)
        self._start_edit.setObjectName("StartHotkeyEdit")
        self._start_edit.setPlaceholderText("e.g. ctrl+f2")
        self._start_edit.editingFinished.connect(self._emit_save)
        # While toggle-mode is on, the Stop field mirrors Start —
        # listen for live edits to keep them in sync visually.
        self._start_edit.textChanged.connect(self._mirror_start_into_stop)
        hotkeys_form.addRow("Start recording", self._start_edit)

        self._stop_edit = QLineEdit(hotkeys_card)
        self._stop_edit.setObjectName("StopHotkeyEdit")
        self._stop_edit.setPlaceholderText("e.g. ctrl+f3")
        self._stop_edit.editingFinished.connect(self._emit_save)
        hotkeys_form.addRow("Stop recording", self._stop_edit)

        # "Discard buffer without transcribing" — the runtime has
        # always supported this (StateManager.cancel_active_recording)
        # but the hotkey was never exposed. Optional — empty value
        # means no global key, the feature simply isn't bound.
        self._cancel_edit = QLineEdit(hotkeys_card)
        self._cancel_edit.setObjectName("CancelHotkeyEdit")
        self._cancel_edit.setPlaceholderText("e.g. ctrl+f6 — leave empty to disable")
        self._cancel_edit.editingFinished.connect(self._emit_save)
        hotkeys_form.addRow("Cancel recording", self._cancel_edit)

        # "Reset to defaults" lives inside the card now (next to its
        # owned content) instead of a footer at the bottom of the
        # whole tab — matches the Storage card's button placement
        # and means each card's reset only touches its own settings.
        hotkeys_btn_row = QHBoxLayout()
        hotkeys_btn_row.setSpacing(10)
        hotkeys_btn_row.addStretch(1)
        self._reset_hotkeys_btn = QPushButton("Reset to defaults", hotkeys_card)
        self._reset_hotkeys_btn.setObjectName("ResetHotkeysButton")
        self._reset_hotkeys_btn.setFocusPolicy(Qt.NoFocus)
        self._reset_hotkeys_btn.clicked.connect(
            self.hotkeys_reset_requested.emit
        )
        hotkeys_btn_row.addWidget(self._reset_hotkeys_btn)
        hotkeys_card.layout().addLayout(hotkeys_btn_row)

        # Hint goes into the card's OUTER VBox, not the form — adding
        # it as a labelless form-row would offset it to the field
        # column (under the inputs), inconsistent with the Storage
        # card's hint which sits flush-left across the full card.
        hotkeys_hint = QLabel(
            "Cancel discards the current buffer instead of transcribing.",
            hotkeys_card,
        )
        hotkeys_hint.setObjectName("HotkeysHint")
        hotkeys_hint.setProperty("role", "muted")
        hotkeys_hint.setWordWrap(True)
        hotkeys_card.layout().addWidget(hotkeys_hint)
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

        # Used-space row: tells the user how much disk the cache eats
        # so they can decide whether to move it to a bigger drive.
        # The controller computes this asynchronously (cached_models_size
        # walks the entire HF hub subtree).
        storage_size_row = QHBoxLayout()
        storage_size_row.setSpacing(16)
        storage_size_caption = QLabel("Used", storage_card)
        storage_size_row.addWidget(storage_size_caption)
        self._storage_size_label = QLabel("…", storage_card)
        self._storage_size_label.setObjectName("StorageSizeLabel")
        self._storage_size_label.setProperty("role", "muted")
        self._storage_size_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        storage_size_row.addWidget(self._storage_size_label, 1)
        storage_v.addLayout(storage_size_row)

        # Button row — flush left, stretch on the right. Wrapped in a
        # QWidget rather than added as a bare QHBoxLayout because the
        # outer VBox doesn't reliably pick up the layout's sizeHint
        # in this nesting (hint label was rendering on top of the
        # button row's bottom edge).
        storage_btn_widget = QWidget(storage_card)
        storage_btn_widget.setObjectName("StorageButtonRow")
        storage_btn_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Without this the global ``QWidget { background-color:
        # bg_primary }`` rule paints a dark slab around the buttons
        # that's visibly different from the card's elevated bg —
        # makes the row look like its own button-coloured strip.
        storage_btn_widget.setStyleSheet("background: transparent;")
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

        # Open-folder shortcut — once the user knows the size, the
        # natural next step is "show me what's inside" (delete stale
        # downloads, free up space, copy weights to a backup, …).
        # Cheaper than implementing a built-in cache browser.
        self._open_storage_btn = QPushButton(
            "Open folder", storage_btn_widget,
        )
        self._open_storage_btn.setObjectName("OpenStorageButton")
        self._open_storage_btn.setFocusPolicy(Qt.NoFocus)
        self._open_storage_btn.clicked.connect(
            self.storage_open_requested.emit
        )
        storage_btn_row.addWidget(self._open_storage_btn)
        storage_btn_row.addStretch(1)
        # Match the wrapper's height to the buttons' sizeHint so the
        # outer VBox can't squish it below the button height.
        storage_btn_widget.setMinimumHeight(
            self._change_storage_btn.sizeHint().height(),
        )
        storage_v.addWidget(storage_btn_widget)

        storage_hint = QLabel(
            "New downloads land here immediately. Already-downloaded "
            "weights stay in their current folder unless you choose "
            "to move them.",
            storage_card,
        )
        storage_hint.setObjectName("StorageHint")
        storage_hint.setProperty("role", "muted")
        storage_hint.setWordWrap(True)
        storage_v.addWidget(storage_hint)
        root.addWidget(storage_card)

        # ---- Hugging Face card ------------------------------------------
        # Optional API token, only relevant for GigaAM long-form
        # audio (>25 s) which routes through pyannote VAD —
        # ``pyannote/segmentation-3.0`` is gated and needs an HF
        # account that's accepted the model card. Built by hand
        # rather than via ``_make_section_card`` for the same
        # reason as the Storage card (form-row layout + helper
        # widgets clash on spanning rows).
        hf_card = QFrame(self)
        hf_card.setObjectName("HfCard")
        hf_card.setProperty("role", "card")
        hf_v = QVBoxLayout(hf_card)
        hf_v.setContentsMargins(20, 16, 20, 16)
        hf_v.setSpacing(10)

        hf_header = QLabel("Hugging Face", hf_card)
        hf_header.setProperty("role", "section-header")
        hf_v.addWidget(hf_header)

        # Token field + Clear on a single row to avoid the wide
        # empty rectangle a stretch-aligned button row used to
        # leave next to the input. Scope is field-level (just the
        # token), so inline placement is clear without the extra
        # vertical real estate.
        hf_row = QHBoxLayout()
        hf_row.setSpacing(10)
        hf_caption = QLabel("API token", hf_card)
        hf_row.addWidget(hf_caption)
        self._hf_token_edit = QLineEdit(hf_card)
        self._hf_token_edit.setObjectName("HfTokenEdit")
        self._hf_token_edit.setEchoMode(QLineEdit.Password)
        self._hf_token_edit.setPlaceholderText("hf_…")
        self._hf_token_edit.setClearButtonEnabled(True)
        self._hf_token_edit.editingFinished.connect(self._on_hf_token_finished)
        hf_row.addWidget(self._hf_token_edit, 1)
        self._clear_hf_token_btn = QPushButton("Clear", hf_card)
        self._clear_hf_token_btn.setObjectName("ClearHfTokenButton")
        self._clear_hf_token_btn.setFocusPolicy(Qt.NoFocus)
        self._clear_hf_token_btn.clicked.connect(
            self.hf_token_reset_requested.emit
        )
        hf_row.addWidget(self._clear_hf_token_btn)
        hf_v.addLayout(hf_row)

        hf_hint = QLabel(
            "Optional. Used when downloading gated or private "
            "Hugging Face models — the app passes it to "
            "<code>huggingface_hub</code> on every fetch.<br>"
            'Get one at '
            '<a href="https://huggingface.co/settings/tokens" '
            'style="color:#7aa2ff;text-decoration:none">'
            'huggingface.co/settings/tokens</a>.',
            hf_card,
        )
        hf_hint.setObjectName("HfHint")
        hf_hint.setProperty("role", "muted")
        hf_hint.setWordWrap(True)
        hf_hint.setTextFormat(Qt.RichText)
        hf_hint.setOpenExternalLinks(True)
        hf_hint.setTextInteractionFlags(Qt.TextBrowserInteraction)
        hf_v.addWidget(hf_hint)
        root.addWidget(hf_card)

        # Footer reset button retired — each card now owns its own
        # 'Reset' / 'Clear' affordance. The previous global button
        # was misleading: it advertised 'Reset to defaults' but
        # only touched hotkeys + auto_paste, leaving Storage / HF
        # untouched. Per-card buttons make the scope explicit.
        root.addStretch(1)

    # ---- public API ---------------------------------------------------------

    def set_values(
        self,
        start_hotkey: str,
        stop_hotkey: str,
        auto_paste: bool,
        cancel_hotkey: str = "",
    ) -> None:
        # Programmatic update — must not feed back into save_requested.
        # ``cancel_hotkey`` is keyword-only with a default so callers
        # written before the field existed keep working unchanged.
        self._suspend_emit = True
        try:
            self._start_edit.setText(start_hotkey)
            self._stop_edit.setText(stop_hotkey)
            self._auto_paste_cb.setChecked(bool(auto_paste))
            self._cancel_edit.setText(cancel_hotkey or "")
            # If the persisted config has the same combination for
            # start and stop, the user is implicitly in toggle mode —
            # tick the checkbox so the UI matches.  Empty start ==
            # empty stop should NOT auto-tick (that's "no hotkey
            # configured at all", not toggle).
            same = bool(
                start_hotkey
                and start_hotkey.strip().lower() == stop_hotkey.strip().lower()
            )
            self._toggle_mode_cb.setChecked(same)
            self._apply_toggle_mode(same)
        finally:
            self._suspend_emit = False
        # Run validation once the suspend flag is back down so the
        # invalid-border / tooltip state matches the freshly-loaded
        # values.  Doing it inside the suspend block would skip the
        # repaint triggered by the property change.
        self._refresh_hotkey_validation()

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
        # In toggle-mode the stop combo is implicitly the start
        # combo — return it so the persisted config keeps both
        # fields in sync (HotkeyListener relies on equality to
        # decide whether to bind a single toggle handler).
        if self._toggle_mode_cb.isChecked():
            return self._start_edit.text().strip()
        return self._stop_edit.text().strip()

    def cancel_hotkey(self) -> str:
        # Toggle-mode disables Cancel functionally — the user wanted
        # a "one key, one job" recording flow. Returning an empty
        # string here propagates through ``values()`` /
        # ``save_requested`` so the persisted config drops the
        # binding and the HotkeyListener stops registering it. The
        # field text itself is preserved on screen so un-ticking
        # restores the previous value transparently.
        if self._toggle_mode_cb.isChecked():
            return ""
        return self._cancel_edit.text().strip()

    def auto_paste(self) -> bool:
        return self._auto_paste_cb.isChecked()

    def device_index(self) -> Optional[int]:
        return self._device_combo.currentData()

    def set_hf_token(self, token: str) -> None:
        """Programmatic prefill of the HF token field — used by the
        controller on init. Won't echo a ``hf_token_changed`` signal
        back so we don't re-save what we just loaded."""
        self._suspend_emit = True
        try:
            self._hf_token_edit.setText(token or "")
        finally:
            self._suspend_emit = False

    def hf_token(self) -> str:
        return self._hf_token_edit.text().strip()

    def _on_hf_token_finished(self) -> None:
        if self._suspend_emit:
            return
        self.hf_token_changed.emit(self.hf_token())

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

    def set_storage_size(self, text: str) -> None:
        """Render the human-readable used-space string in the Storage card.

        The controller does the formatting (bytes → ``"3.4 GB"``) so
        this view stays free of locale rules and unit thresholds.
        ``""`` blanks the label (used while the worker is computing).
        """
        self._storage_size_label.setText(text or "…")

    def values(self) -> Dict[str, Any]:
        return {
            "start_hotkey": self.start_hotkey(),
            "stop_hotkey": self.stop_hotkey(),
            "cancel_hotkey": self.cancel_hotkey(),
            "auto_paste": self.auto_paste(),
            "device": self.device_index(),
        }

    # ---- internal -----------------------------------------------------------

    def _emit_save(self) -> None:
        if self._suspend_emit:
            return
        # Run validation alongside every save so red-border / tooltip
        # state stays in sync with whatever's currently typed.  We
        # still emit ``save_requested`` even when fields are invalid
        # — backend writes a warning to the Logs view, the UI
        # carries the visual feedback, and the user can keep typing
        # to fix it without the controller getting stuck on a
        # partial edit.
        self._refresh_hotkey_validation()
        self.save_requested.emit(self.values())

    def _refresh_hotkey_validation(self) -> None:
        """Run :func:`validate_all` over the current field values
        and toggle the ``invalid`` Qt property + tooltip on each
        QLineEdit. Pure UI shuffle — no signals."""
        errors = validate_all(
            start=self._start_edit.text(),
            stop=self._stop_edit.text(),
            cancel=self._cancel_edit.text(),
            toggle_mode=self._toggle_mode_cb.isChecked(),
        )
        for field_name, edit in (
            ("start", self._start_edit),
            ("stop", self._stop_edit),
            ("cancel", self._cancel_edit),
        ):
            err = errors.get(field_name)
            edit.setProperty("invalid", bool(err))
            edit.setToolTip(err or "")
            # ``setProperty`` on a styled widget needs an
            # unpolish/polish cycle for Qt to repaint with the new
            # selector match.
            edit.style().unpolish(edit)
            edit.style().polish(edit)

    def _on_auto_paste_toggled(self, _checked: bool) -> None:
        self._emit_save()

    def _refresh_accessibility_banner(self) -> None:
        """Show / hide the macOS Accessibility warning banner based
        on whether the current process can read global keyboard
        events. ``None`` (non-macOS) keeps it hidden — Win / Linux
        don't have the equivalent permission gate.
        """
        trusted = is_accessibility_trusted()
        # ``True``  → permission granted, hide banner.
        # ``False`` → not granted, show banner.
        # ``None``  → not on macOS, banner irrelevant.
        self._accessibility_banner.setVisible(trusted is False)

    def showEvent(self, event):  # noqa: N802 — Qt naming
        """Re-check Accessibility every time the Settings tab
        becomes visible. Permission changes need a process restart
        to take effect on Mac, but a fresh ``AXIsProcessTrusted``
        call is cheap and covers the case where the user opened
        Settings, hit the Accessibility button, granted access,
        and is now back in our window without an app restart —
        the banner can at least disappear."""
        super().showEvent(event)
        self._refresh_accessibility_banner()

    def _on_toggle_mode_changed(self, checked: bool) -> None:
        """User flipped 'Use one hotkey for both start and stop'.

        Mirror Start into Stop and lock the Stop / Cancel fields,
        but snapshot Stop's original text first so un-ticking
        restores the user's previous binding instead of leaving
        Stop frozen at the Start value. Cancel doesn't need this —
        we never overwrite its text in toggle mode, only ignore it
        in ``cancel_hotkey()``.
        """
        if checked:
            # Capture the value the user had before we mirror Start
            # in.  Skipped if we're already in toggle mode (would
            # snapshot a value that's already a mirror of Start).
            if self._previous_stop_hotkey is None:
                self._previous_stop_hotkey = self._stop_edit.text()
        else:
            if self._previous_stop_hotkey is not None:
                self._stop_edit.setText(self._previous_stop_hotkey)
                self._previous_stop_hotkey = None
        self._apply_toggle_mode(checked)
        self._emit_save()

    def _apply_toggle_mode(self, checked: bool) -> None:
        """Lock / unlock the Stop and Cancel fields per ``checked``.
        Pure UI shuffle — no signal emission.

        Read-only (rather than disabled) makes it obvious that the
        fields are *deactivated by toggle-mode*, not broken. The
        ``muted="true"`` Qt property switches the field's QSS to
        ``color.bg_elevated`` background + ``text_muted`` foreground
        so the visual reads as "currently inactive" rather than a
        normal editable input.

        - **Stop** mirrors the Start value live (so the user sees
          which hotkey actually stops a recording in toggle mode).
        - **Cancel** keeps its previously-configured value so a
          later un-tick restores it, but is muted — toggle mode is a
          "one key for everything recording" UX and the cancel
          escape hatch would muddy that.
        """
        for field in (self._stop_edit, self._cancel_edit):
            field.setReadOnly(checked)
            field.setProperty("muted", checked)
            # ``setProperty`` on a styled widget needs an
            # unpolish/polish cycle before Qt picks up the new
            # selector match.
            field.style().unpolish(field)
            field.style().polish(field)
        if checked:
            self._stop_edit.setText(self._start_edit.text())

    def _mirror_start_into_stop(self, new_text: str) -> None:
        """Keep the Stop field synced with Start while toggle-mode
        is on. No-op when toggle-mode is off — that's the regular
        two-independent-fields path.

        Bypasses ``_suspend_emit`` because this is a UI mirror, not
        a programmatic load — we explicitly want the user's keystroke
        in Start to ripple through and persist.
        """
        if self._toggle_mode_cb.isChecked():
            self._stop_edit.setText(new_text)

    def _on_device_changed(self, _idx: int) -> None:
        self._emit_save()

    # ---- mic test feedback --------------------------------------------------

    def show_mic_test_running(self) -> None:
        self._test_mic_btn.setEnabled(False)
        self._test_mic_label.setText("Listening… speak now (3 s)")
        self._test_mic_label.setProperty("role", "muted")
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)
        # Show the live VU meter for the duration of the test.
        # Controller starts a polling timer to feed it through
        # ``set_mic_test_level`` — without that the bar would just
        # sit at zero.
        self._test_mic_meter.reset()
        self._test_mic_meter.setVisible(True)

    def set_mic_test_level(self, level: float) -> None:
        """Push a fresh amplitude reading into the mic-test VU meter.
        Called by the controller while a test is running. No-op when
        the meter is hidden so a stray late tick can't paint over a
        finished result."""
        if self._test_mic_meter.isVisible():
            self._test_mic_meter.set_level(level)

    def show_mic_test_result(self, peak: float, rms: float) -> None:
        self._test_mic_btn.setEnabled(True)
        # Keep the meter visible and frozen at the peak amplitude
        # — the bar IS the visual "how loud were you" answer, no
        # numeric % needed in the text. ``set_level`` once + no
        # follow-up calls = the peak-and-decay envelope just holds
        # the value indefinitely (decay only fires on subsequent
        # ``set_level`` calls). Reset happens on the next test.
        self._test_mic_meter.setVisible(True)
        self._test_mic_meter.set_level(max(0.0, min(1.0, peak)))

        # Plain English result: descriptive only, no jargon, no
        # mystery numbers. Power users / bug-reporters still get
        # the raw 0–1 ``peak`` and ``rms`` via the tooltip.
        if peak < 0.01:
            text = "No sound detected — check the selected microphone."
            role = "test-result-bad"
        elif peak < 0.08:
            text = (
                "Very quiet. Speak louder or raise the input "
                "level in Windows sound settings."
            )
            role = "test-result-warn"
        else:
            text = "Looks good."
            role = "test-result-good"
        self._test_mic_label.setText(text)
        self._test_mic_label.setToolTip(
            f"peak={peak:.3f}, rms={rms:.3f}\n"
            "(amplitude on a 0–1 scale; peak = loudest sample, "
            "rms = average power)"
        )
        self._test_mic_label.setProperty("role", role)
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)

    def show_mic_test_error(self, reason: str) -> None:
        self._test_mic_btn.setEnabled(True)
        self._test_mic_meter.setVisible(False)
        self._test_mic_meter.reset()
        self._test_mic_label.setText(f"Test failed: {reason}")
        self._test_mic_label.setProperty("role", "test-result-bad")
        self._test_mic_label.style().unpolish(self._test_mic_label)
        self._test_mic_label.style().polish(self._test_mic_label)
