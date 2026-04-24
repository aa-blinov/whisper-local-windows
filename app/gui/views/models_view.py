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
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        title = QLabel("Models", self)
        title.setProperty("role", "title")
        root.addWidget(title)

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

    def active_alias(self) -> Optional[str]:
        return self._active_alias

    def set_active(self, alias: str) -> None:
        if alias not in self._cards:
            raise KeyError(alias)
        for card_alias, card in self._cards.items():
            card.set_active(card_alias == alias)
        self._active_alias = alias
