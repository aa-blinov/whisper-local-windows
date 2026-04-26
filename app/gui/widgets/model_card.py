"""Card widget that displays a single model entry."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _has_hf_token() -> bool:
    """True if either ``HF_TOKEN`` or its legacy alias
    ``HUGGING_FACE_HUB_TOKEN`` is set to a non-empty value."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return True
    return False

from app.gui.widgets.flow_layout import FlowLayout
from app.gui.widgets.inference_settings_panel import InferenceSettingsPanel
from app.inference_settings import InferenceSettings
from app.model_mapping import ModelInfo, model_url
from app.utils import is_cached_for_info


def _format_size(size_mb: int) -> str:
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


def _format_bytes(num_bytes: int) -> str:
    """Compact byte counter used inside the loading pill."""
    if num_bytes <= 0:
        return ""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _format_loading_progress(current: int, total: int) -> str:
    """Pill label for the active card while a model is downloading.

    Returns an empty string when neither bytes nor totals are known —
    the caller falls back to the default 'Loading…' marker.
    """
    if total > 0 and current >= 0:
        pct = int(min(99, max(0, current * 100 // total)))
        return f"Loading {pct}%"
    if current > 0:
        return f"Loading {_format_bytes(current)}"
    return ""


def _compute_label(compute_type: str) -> str:
    """Pretty short label for the ``compute_type`` badge."""
    return {
        "float32": "fp32",
        "float16": "fp16",
        "int8_float16": "int8 + fp16",
        "int8": "int8",
    }.get(compute_type, compute_type)


class ModelCard(QFrame):
    select_requested = Signal(str)
    # Emitted when the user clicks Delete on a cached model — arg is
    # the alias. The controller is responsible for confirming with
    # the user before actually wiping the cache, then calling
    # ``refresh_cache_state`` so the button visibility and label
    # update.
    delete_requested = Signal(str)
    # Emitted when the user changes anything in the inline inference
    # settings panel — args: ``(alias, InferenceSettings)`` so the
    # controller can route to per-model config + push live to the
    # backend without having to map widgets back to models.
    inference_settings_changed = Signal(str, InferenceSettings)

    def __init__(self, info: ModelInfo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._info = info
        self._active = False
        self._locked = False
        self._loading = False
        # Pill-text inputs — both reset on every loading transition.
        # ``_loading_progress_text`` wins when present (download %),
        # ``_loading_elapsed_s`` is the fallback for cached loads.
        self._loading_progress_text: str = ""
        self._loading_elapsed_s: int = 0

        self.setObjectName("ModelCard")
        self.setProperty("role", "card")
        self.setProperty("active", False)
        self.setFrameShape(QFrame.NoFrame)
        # Variable vertical size — badges wrap onto a second line on
        # narrow windows, so the card has to grow to fit them.
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

        # Soft drop shadow makes the card "float" off the dark surface
        # — Qt QSS has no box-shadow so this is the only way to add
        # depth. Keep blur generous and offset small so the effect is
        # subtle, not theatrical.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)

        # Family chip — visual grouping (Whisper / Turbo / Distil /
        # Russian / GigaAM). Coloured per family via QSS. Qt QSS
        # doesn't honour ``text-transform: uppercase`` so we
        # upper-case the text in Python; ``unpolish/polish`` forces
        # the engine to re-evaluate the compound selector
        # ``[role="family-chip"][family="…"]`` against the freshly-
        # set property.
        family_chip = QLabel(info.family.upper(), self)
        family_chip.setObjectName("FamilyChip")
        family_chip.setProperty("role", "family-chip")
        family_chip.setProperty(
            "family", info.family.lower().replace(" ", "-")
        )
        family_chip.setAlignment(Qt.AlignCenter)
        family_chip.style().unpolish(family_chip)
        family_chip.style().polish(family_chip)
        header.addWidget(family_chip)

        title = QLabel(info.display_name, self)
        title.setProperty("role", "heading")
        header.addWidget(title)

        header.addStretch(1)

        self._active_pill = QLabel("Active", self)
        self._active_pill.setProperty("role", "pill-active")
        self._active_pill.setProperty("state", "ready")
        self._active_pill.setAlignment(Qt.AlignCenter)
        self._active_pill.setVisible(False)
        header.addWidget(self._active_pill)

        root.addLayout(header)

        # Subtitle line: alias · canonical · external-link affordance.
        # ``QLabel`` with ``openExternalLinks`` is the cheapest way to
        # get a clickable URL inside the card without a button.
        subtitle = QLabel(
            f'<span style="color:#7aa2ff">{info.alias}</span>'
            f'<span style="color:#7d828d">  ·  </span>'
            f'<span style="color:#b8bcc6">{info.canonical}</span>'
            f'  <a href="{model_url(info)}" '
            f'style="color:#7aa2ff;text-decoration:none">↗</a>',
            self,
        )
        subtitle.setObjectName("ModelSubtitle")
        subtitle.setProperty("role", "muted")
        subtitle.setOpenExternalLinks(True)
        subtitle.setTextFormat(Qt.RichText)
        subtitle.setTextInteractionFlags(
            Qt.TextBrowserInteraction
        )
        subtitle.setToolTip(f"Open {model_url(info)}")
        root.addWidget(subtitle)

        description = QLabel(info.description, self)
        description.setProperty("role", "muted")
        description.setWordWrap(True)
        root.addWidget(description)

        # FlowLayout wraps badges to a new line when the card is too
        # narrow to fit them all in one row — without this, narrow
        # windows clipped the right side of the card (and the
        # Download button) at the scroll viewport's edge.
        badges = FlowLayout(spacing=6)
        # Each badge carries a ``cat`` property (and ``value`` where
        # the value is one of a known set) so the QSS can give the
        # three categories distinct visual weights:
        #   - speed / quality → tinted (green / blue) when the value
        #     is the desirable one ("fast", "excellent")
        #   - size / vram → solid neutral pills (current default)
        #   - compute / lang → outlined-only, transparent bg
        badge_specs = (
            ("speed", info.speed, info.speed),
            ("quality", info.quality, info.quality),
            ("size", _format_size(info.size_mb), ""),
            ("vram", f"{info.vram_gb:.1f} GB", ""),
            ("compute", _compute_label(info.compute_type), ""),
            ("lang", info.languages, ""),
        )
        for cat, display_value, qss_value in badge_specs:
            badge = QLabel(f"{cat}: {display_value}", self)
            badge.setProperty("role", "badge")
            badge.setProperty("cat", cat)
            if qss_value:
                badge.setProperty("value", qss_value)
            badges.addWidget(badge)
        root.addLayout(badges)

        # Inline inference settings — only relevant for engines that
        # actually accept transcribe-time tunables. GigaAM is
        # end-to-end (Russian-only, deterministic, no prompt) so it
        # gets no panel; inserting a greyed-out one looked like a
        # rendering bug. faster-whisper-backed cards get the full
        # panel, hidden until the card is active.
        self._settings_panel: Optional[InferenceSettingsPanel] = None
        if info.backend_kind != "gigaam":
            self._settings_panel = InferenceSettingsPanel(self)
            self._settings_panel.setVisible(False)
            self._settings_panel.settings_changed.connect(
                lambda s: self.inference_settings_changed.emit(
                    self._info.alias, s
                )
            )
            root.addWidget(self._settings_panel)

        # HF-token warning — only on GigaAM cards, since GigaAM's
        # long-form path (>25 s captures) routes through pyannote
        # VAD which needs a token to download the gated
        # ``pyannote/segmentation-3.0`` weights. Whisper's
        # long-form is Silero VAD, no token required, so the
        # widget is omitted entirely on faster_whisper cards.
        self._hf_warning: Optional[QLabel] = None
        if info.backend_kind == "gigaam":
            self._hf_warning = QLabel(
                "⚠ Long-form audio (>25 s) needs a Hugging Face "
                "token — set one in Settings → Hugging Face.",
                self,
            )
            self._hf_warning.setObjectName("HfTokenWarning")
            self._hf_warning.setProperty("role", "warning")
            self._hf_warning.setWordWrap(True)
            root.addWidget(self._hf_warning)
            self.refresh_hf_token_state()

        footer = QHBoxLayout()
        footer.addStretch(1)
        # Delete sits to the LEFT of Select — destructive action stays
        # visually subordinate to the primary one. Hidden by default;
        # visibility is recomputed every time the cache / active /
        # loading state changes (see ``_refresh_delete_visibility``).
        self._delete_btn = QPushButton("Delete", self)
        self._delete_btn.setObjectName("DeleteButton")
        self._delete_btn.setProperty("role", "danger")
        self._delete_btn.setFocusPolicy(Qt.NoFocus)
        self._delete_btn.setVisible(False)
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._info.alias)
        )
        footer.addWidget(self._delete_btn)

        self._select_btn = QPushButton("Select", self)
        self._select_btn.setObjectName("SelectButton")
        self._select_btn.setProperty("role", "primary")
        # Without NoFocus, clicking puts keyboard focus on the button.
        # When the card transitions to Active immediately afterwards,
        # the button is hidden — Qt then chases focus to the next
        # focusable widget (the Download button on the card below) and
        # the QScrollArea scrolls to bring it into view, jumping the
        # entire models list. NoFocus keeps clicks working but stops
        # the focus dance.
        self._select_btn.setFocusPolicy(Qt.NoFocus)
        self._select_btn.clicked.connect(
            lambda: self.select_requested.emit(self._info.alias)
        )
        footer.addWidget(self._select_btn)
        root.addLayout(footer)

        # Initial Download/Select label based on whether the canonical is
        # already cached on disk. Refreshable via ``refresh_cache_state``.
        self.refresh_cache_state()

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        layout = self.layout()
        if layout is None:
            return -1
        margins = self.contentsMargins()
        inner = width - margins.left() - margins.right()
        return layout.heightForWidth(inner) + margins.top() + margins.bottom()

    def alias(self) -> str:
        return self._info.alias

    def info(self) -> ModelInfo:
        return self._info

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.setProperty("active", self._active)
        self._active_pill.setVisible(self._active)
        self._select_btn.setVisible(not self._active)
        self._select_btn.setEnabled(not self._active and not self._locked)
        # Inference panel visible only on the active card (and only
        # on cards that actually have one — GigaAM doesn't).
        if self._settings_panel is not None:
            self._settings_panel.setVisible(self._active)
        self._refresh_delete_visibility()
        self.style().unpolish(self)
        self.style().polish(self)

    def set_inference_settings(self, settings: InferenceSettings) -> None:
        """Pre-fill the inline panel from the controller (called when
        the card becomes active and the controller has loaded the
        per-alias overrides out of config). No-op on engines that
        don't have a panel (GigaAM)."""
        if self._settings_panel is not None:
            self._settings_panel.set_settings(settings)

    def inference_settings(self) -> Optional[InferenceSettings]:
        if self._settings_panel is None:
            return None
        return self._settings_panel.values()

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        # Active cards keep Select hidden regardless; for inactive ones,
        # locking disables the button.
        if not self._active:
            self._select_btn.setEnabled(not self._locked)

    def is_loading(self) -> bool:
        return self._loading

    def refresh_hf_token_state(self) -> None:
        """Recompute warning visibility from the current process env.
        Called by the controller after the user pastes a token in
        Settings; no-op on cards that don't carry the warning
        (Whisper)."""
        if self._hf_warning is None:
            return
        self._hf_warning.setVisible(not _has_hf_token())

    def refresh_cache_state(self) -> None:
        """Recompute whether the underlying model is downloaded and update
        the action button label (``Download`` vs ``Select``) plus the
        Delete button's visibility."""
        cached = is_cached_for_info(self._info)
        self._select_btn.setText("Select" if cached else "Download")
        self._refresh_delete_visibility()

    def _refresh_delete_visibility(self) -> None:
        """Delete is shown only when (a) weights are on disk, (b) the
        card isn't currently the active model — yanking the cache out
        from under a loaded backend would crash the next transcribe —
        and (c) we aren't mid-load, when the cache state is undefined.
        """
        cached = is_cached_for_info(self._info)
        self._delete_btn.setVisible(
            cached and not self._active and not self._loading
        )

    def set_loading(self, loading: bool) -> None:
        """Reflect backend load state on the active pill — swap 'Active' for
        'Loading…' with a different colour while the model is loading."""
        self._loading = bool(loading)
        # Both pill-text inputs reset every transition so a fresh load
        # never inherits stale numbers from a previous one.
        self._loading_progress_text = ""
        self._loading_elapsed_s = 0
        if self._loading:
            self._active_pill.setText("Loading\u2026")
            self._active_pill.setProperty("state", "loading")
        else:
            self._active_pill.setText("Active")
            self._active_pill.setProperty("state", "ready")
        self._active_pill.style().unpolish(self._active_pill)
        self._active_pill.style().polish(self._active_pill)
        self._refresh_delete_visibility()

    def set_loading_progress(self, current: int, total: int) -> None:
        """Update the active pill with download progress while the card
        is in the loading state. Ignored when the card isn't loading so
        stale events arriving after the model finishes can't repaint
        the green Active pill with stale byte counts."""
        if not self._loading:
            return
        self._loading_progress_text = _format_loading_progress(
            int(current), int(total)
        )
        self._refresh_loading_pill()

    def set_loading_elapsed(self, seconds: int) -> None:
        """Update the elapsed-seconds counter shown in the loading pill
        when no byte progress is available — used for cached model
        loads where CTranslate2 deserialises weights silently."""
        if not self._loading:
            return
        self._loading_elapsed_s = max(0, int(seconds))
        self._refresh_loading_pill()

    def _refresh_loading_pill(self) -> None:
        """Recompute the loading pill text from progress + elapsed inputs.

        Order of precedence: download percentage / bytes win over
        elapsed seconds; both win over the static 'Loading…' marker."""
        if not self._loading:
            return
        if self._loading_progress_text:
            self._active_pill.setText(self._loading_progress_text)
        elif self._loading_elapsed_s > 0:
            self._active_pill.setText(f"Loading {self._loading_elapsed_s}s")
        else:
            self._active_pill.setText("Loading\u2026")
