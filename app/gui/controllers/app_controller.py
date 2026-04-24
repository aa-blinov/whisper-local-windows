"""Controller wiring views to the domain layer (config, state, backends)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from PySide6.QtCore import QObject

from app.gui.main_window import MainWindow
from app.model_mapping import alias_for, get_model


log = logging.getLogger(__name__)


class _ConfigLike(Protocol):
    def get_setting(self, section: str, key: str) -> Any: ...
    def update_user_setting(self, section: str, key: str, value: Any) -> None: ...


class AppController(QObject):
    def __init__(self, config: _ConfigLike, window: MainWindow) -> None:
        super().__init__(parent=window)
        self._config = config
        self._window = window
        self._wire_models()

    def _wire_models(self) -> None:
        view = self._window.models_view
        raw = self._config.get_setting("whisper", "model")
        if isinstance(raw, str) and raw:
            alias = alias_for(raw)
            try:
                get_model(alias)
            except KeyError:
                log.warning(
                    "Model %r from config is not in the registry — leaving inactive",
                    raw,
                )
            else:
                view.set_active(alias)

        view.model_selected.connect(self._on_model_selected)

    def _on_model_selected(self, alias: str) -> None:
        if self._window.models_view.active_alias() == alias:
            return
        self._config.update_user_setting("whisper", "model", alias)
        self._window.models_view.set_active(alias)
