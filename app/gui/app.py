"""Qt application entry point."""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow
from app.gui.theme import apply_theme


def build_application(
    argv: Optional[List[str]] = None,
    theme: str = "dark",
) -> Tuple[QApplication, MainWindow]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    apply_theme(app, theme)

    window = MainWindow()
    return app, window


def main() -> int:
    app, window = build_application()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
