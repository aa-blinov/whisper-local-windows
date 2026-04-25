"""Models browser view — scrollable grid of ModelCards."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.gui.widgets.model_card import ModelCard
from app.model_mapping import MODELS, ModelInfo


class ModelsView(QWidget):
    model_selected = Signal(str)

    def __init__(
        self,
        models: Optional[Sequence[ModelInfo]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ModelsView")
        resolved: Sequence[ModelInfo] = tuple(models) if models is not None else MODELS

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(10)

        # Section title lives in the TopBar; a hint here is enough context.
        hint = QLabel(
            "Pick a model. Quality scales with size; speed is the opposite.",
            self,
        )
        hint.setProperty("role", "muted")
        root.addWidget(hint)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Smaller wheel step → smooth pixel-ish scrolling instead of a
        # whole-card jump per notch.
        scroll.verticalScrollBar().setSingleStep(20)

        content = QWidget(scroll)
        content.setObjectName("ModelsScrollContent")
        cards_layout = QVBoxLayout(content)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(12)

        self._cards: Dict[str, ModelCard] = {}
        for info in resolved:
            card = ModelCard(info, parent=content)
            card.select_requested.connect(self.model_selected.emit)
            cards_layout.addWidget(card)
            self._cards[info.alias] = card

        cards_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self._active_alias: Optional[str] = None
        self._locked = False

    def active_alias(self) -> Optional[str]:
        return self._active_alias

    def set_active(self, alias: str) -> None:
        if alias not in self._cards:
            raise KeyError(alias)
        for card_alias, card in self._cards.items():
            card.set_active(card_alias == alias)
        self._active_alias = alias

    def is_locked(self) -> bool:
        return self._locked

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        for card in self._cards.values():
            card.set_locked(self._locked)

    def set_loading(self, loading: bool) -> None:
        """Mark the currently active card as loading — its pill swaps from
        'Active' to 'Loading…' until the backend reports ready."""
        for card in self._cards.values():
            card.set_loading(loading)

    def set_loading_progress(self, current: int, total: int) -> None:
        """Forward backend download progress to every card. Each card
        ignores the update unless it's currently in the loading state,
        so only the active-and-loading card actually repaints."""
        for card in self._cards.values():
            card.set_loading_progress(current, total)

    def set_loading_elapsed(self, seconds: int) -> None:
        """Forward the elapsed-seconds tick to every card. Used for
        cached model loads where no tqdm progress fires; the active
        card surfaces it so the user sees the wait advancing."""
        for card in self._cards.values():
            card.set_loading_elapsed(seconds)

    def refresh_cache_state(self) -> None:
        """Re-check the on-disk cache for every card. Called after a model
        finishes downloading so the freshly-downloaded card switches its
        button from "Download" to "Select"."""
        for card in self._cards.values():
            card.refresh_cache_state()
