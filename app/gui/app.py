"""Qt application entry point."""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

from PySide6.QtWidgets import QApplication

from app.gui.controllers.app_controller import AppController
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme


def build_application(
    argv: Optional[List[str]] = None,
    theme: str = "dark",
    config: Optional[Any] = None,
) -> Tuple[QApplication, MainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    apply_theme(app, theme)

    window = MainWindow()
    if config is not None:
        AppController(config=config, window=window)
    return app, window


def main() -> int:
    from app.config_manager import ConfigManager

    config = ConfigManager()
    app, window = build_application(config=config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
