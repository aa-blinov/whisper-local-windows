"""Qt application entry point."""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Tuple

from PySide6.QtWidgets import QApplication

from app.gui.controllers.app_controller import AppController
from app.gui.log_bridge import QtLogBridge
from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme


def build_application(
    argv: Optional[List[str]] = None,
    theme: str = "dark",
    config: Optional[Any] = None,
    history: Optional[Any] = None,
    install_logs: bool = False,
) -> Tuple[QApplication, MainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    apply_theme(app, theme)

    window = MainWindow()
    if install_logs:
        bridge = QtLogBridge(parent=window)
        bridge.line_received.connect(window.logs_view.append_line)
        bridge.install()
    if config is not None:
        AppController(config=config, window=window, history=history)
    return app, window


def main() -> int:
    from app.config_manager import ConfigManager
    from app.history_manager import HistoryManager

    config = ConfigManager()
    history_cfg = config.get_history_config()
    history = HistoryManager(
        max_entries=int(history_cfg.get("max_entries", 1000)),
    )
    app, window = build_application(
        config=config,
        history=history,
        install_logs=True,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
