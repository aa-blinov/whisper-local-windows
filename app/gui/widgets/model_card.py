"""Card widget that displays a single model entry."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
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
from app.gui.widgets.parakeet_inference_settings_panel import (
    ParakeetInferenceSettingsPanel,
)
from app.inference_settings import InferenceSettings, ParakeetInferenceSettings
from app.model_mapping import ModelInfo, model_url


class _CacheWorkerSignals(QObject):
    """Signals carrier for ``_CacheCheckWorker``.

    ``QRunnable`` cannot itself hold signals (it doesn't inherit
    ``QObject``), so the canonical PySide6 pattern is a tiny
    ``QObject`` companion created on the main thread — Qt then
    routes the emitted signal back via a queued connection.

    The ``request_id`` int travels with the boolean result so
    ``_apply_cache_result`` can discard stale responses from workers
    that were superseded by a later ``refresh_cache_state()`` call.
    """

    result = Signal(bool, int)  # (cached, request_id)


class _CacheCheckWorker(QRunnable):
    """Run ``is_cached_for_info`` in a thread-pool thread.

    Emits ``signals.result`` with the boolean outcome and the
    originating ``request_id`` so the card can ignore results that
    arrived out-of-order (an older worker finishing after a newer one).
    """

    def __init__(self, info: ModelInfo, request_id: int) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._info = info
        self._request_id = request_id
        self.signals = _CacheWorkerSignals()

    def run(self) -> None:  # called by QThreadPool on a worker thread
        from app.utils import is_cached_for_info  # lazy — gets patched version in tests

        cached = is_cached_for_info(self._info)
        self.signals.result.emit(cached, self._request_id)


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
    # settings panel. Args: ``(alias, settings_object)``. The settings
    # object is either ``InferenceSettings`` (faster-whisper cards)
    # or ``ParakeetInferenceSettings`` (NeMo cards) — controller dispatches
    # on the alias's backend kind. Declared as ``object`` because
    # PySide signals can't express a sum type and the consumer only
    # uses duck-typed ``.to_mapping()``.
    inference_settings_changed = Signal(str, object)

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
        # Last result from the async cache check.  None = check not yet
        # complete; False = not cached; True = cached on disk.
        self._cached: Optional[bool] = None
        # Monotonically increasing counter: bumped on every
        # refresh_cache_state() call.  Workers embed this id at
        # dispatch time; _apply_cache_result silently drops any
        # result whose id doesn't match the current value (stale
        # worker from a superseded request).
        self._cache_request_id: int = 0

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

        # Inline inference settings — choice of panel keyed on
        # onnx_family:
        #   - ``whisper``: full Whisper panel (language, beam,
        #     temperature, …). Some knobs (beam) don't translate to
        #     the onnx-asr Whisper path but the panel stays useful
        #     for language selection.
        #   - ``parakeet``: minimal panel (timestamps toggle — the
        #     only inference-time tunable onnx-asr exposes for Parakeet).
        #   - ``gigaam``: no panel; e2e model with no per-call knobs.
        self._settings_panel: Optional[QWidget] = None
        onnx_family = getattr(info, "onnx_family", "auto")
        if onnx_family == "whisper":
            self._settings_panel = InferenceSettingsPanel(self)
        elif onnx_family == "parakeet":
            self._settings_panel = ParakeetInferenceSettingsPanel(self)
        if self._settings_panel is not None:
            self._settings_panel.setVisible(False)
            self._settings_panel.settings_changed.connect(
                lambda s: self.inference_settings_changed.emit(
                    self._info.alias, s
                )
            )
            root.addWidget(self._settings_panel)

        # HF-token warning is no longer needed: the legacy GigaAM-Python
        # path used pyannote VAD (gated weights) for long audio.  ONNX
        # GigaAM has its own internal segmentation, no token required.
        self._hf_warning: Optional[QLabel] = None

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

        self._select_btn = QPushButton("Download", self)
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
        new_active = bool(active)
        if new_active == self._active:
            return  # no state change — skip widget updates and style recalc
        self._active = new_active
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

    def set_inference_settings(self, settings) -> None:
        """Pre-fill the inline panel from the controller (called when
        the card becomes active and the controller has loaded the
        per-alias overrides out of config). No-op on engines that
        don't have a panel (GigaAM). ``settings`` is the dataclass
        appropriate for this card's backend (Whisper /
        NeMo) — caller is responsible for sending the right type."""
        if self._settings_panel is not None:
            self._settings_panel.set_settings(settings)

    def inference_settings(self):
        """Return the panel's current values (``InferenceSettings``
        for Whisper, ``ParakeetInferenceSettings`` for NeMo) or ``None``
        on cards that don't have a panel."""
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
        """Schedule an async disk check for this model.

        Launches a ``_CacheCheckWorker`` on Qt's global thread pool so
        the filesystem walk never blocks the main thread.  Each call
        increments ``_cache_request_id``; workers carry that id and
        ``_apply_cache_result`` discards any result whose id is stale
        (i.e. a slower earlier worker finishing after a faster newer
        one).
        """
        self._cache_request_id += 1
        worker = _CacheCheckWorker(self._info, self._cache_request_id)
        worker.signals.result.connect(self._apply_cache_result)
        QThreadPool.globalInstance().start(worker)

    def _apply_cache_result(self, cached: bool, request_id: int) -> None:
        """Slot — called on the main thread by the queued connection
        when the thread-pool worker has finished its disk check.

        Results from superseded requests (stale workers) are silently
        dropped so they cannot overwrite a more recent cache state.
        """
        if request_id != self._cache_request_id:
            return  # stale — a newer request has already landed
        self._cached = cached
        self._select_btn.setText("Select" if cached else "Download")
        self._refresh_delete_visibility()

    def _refresh_delete_visibility(self) -> None:
        """Update Delete button visibility from the last known cache state.

        Uses ``self._cached`` — set by the async worker — so this method
        never touches the filesystem.  Called from ``_apply_cache_result``,
        ``set_active``, and ``set_loading`` to keep the button in sync
        whenever active/loading state changes.

        Delete is shown only when (a) weights are on disk, (b) the card
        isn't currently the active model, and (c) we aren't mid-load.
        """
        cached = self._cached or False  # None → unknown → treat as not cached
        self._delete_btn.setVisible(
            cached and not self._active and not self._loading
        )

    def set_loading(self, loading: bool) -> None:
        """Reflect backend load state on the active pill — swap 'Active' for
        'Loading…' with a different colour while the model is loading."""
        new_loading = bool(loading)
        if new_loading == self._loading:
            return  # no state change \u2014 skip pill update and style recalc
        self._loading = new_loading
        # Reset pill-text inputs only on actual state transitions so a
        # fresh load never inherits stale numbers from a previous one, but
        # repeated set_loading(True) calls don't wipe in-progress text.
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
        # Deliberately no ``_refresh_loading_pill()`` — the pill text
        # only ever changes on real download progress, not on elapsed
        # ticks.  The elapsed counter lives on the topbar pill so the
        # same number doesn't appear in two places at once.

    def _refresh_loading_pill(self) -> None:
        """Recompute the loading pill text from progress + elapsed inputs.

        Order of precedence: download percentage / bytes win over
        elapsed seconds; both win over the static 'Loading…' marker."""
        if not self._loading:
            return
        if self._loading_progress_text:
            self._active_pill.setText(self._loading_progress_text)
        else:
            # No elapsed-seconds branch on the card \u2014 topbar pill
            # owns that display.
            self._active_pill.setText("Loading\u2026")
