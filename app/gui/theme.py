"""Design tokens and QSS loader for the Qt UI.

Tokens live in Python so widgets can reference them directly. The QSS files
use ``{{group.key}}`` placeholders that are substituted at load time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from PySide6.QtWidgets import QApplication


def _styles_dir() -> Path:
    """Locate the directory containing per-theme .qss files.

    In a PyInstaller bundle the .py modules live in the frozen PYZ archive,
    so ``Path(__file__).parent`` does not resolve to a real folder. The
    spec drops the bundled stylesheets under ``sys._MEIPASS / gui / styles``;
    fall back to that location when frozen.
    """
    if getattr(sys, "frozen", False):
        meipass = Path(getattr(sys, "_MEIPASS", ""))
        return meipass / "gui" / "styles"
    return Path(__file__).parent / "styles"


_STYLES_DIR = _styles_dir()


@dataclass(frozen=True)
class _Tokens:
    colors: Dict[str, str] = field(default_factory=dict)
    spacing: Dict[str, int] = field(default_factory=dict)
    radius: Dict[str, int] = field(default_factory=dict)
    fonts: Dict[str, object] = field(default_factory=dict)


TOKENS = _Tokens(
    colors={
        # Slightly bluer, more saturated dark base — gives the surface
        # a "designed" feel instead of pure neutral gray.
        "bg_primary": "#0f1115",
        "bg_secondary": "#1a1d24",
        "bg_elevated": "#252932",
        "bg_hover": "#2d323d",
        "accent": "#5b8cff",
        "accent_hover": "#7aa2ff",
        "text_primary": "#f5f6f8",
        "text_secondary": "#b8bcc6",
        "text_muted": "#7d828d",
        "border": "#2d3140",
        "border_focus": "#5b8cff",
        "success": "#4ade80",
        "danger": "#ef4444",
        "warning": "#f59e0b",
    },
    spacing={
        "xs": 4,
        "sm": 8,
        "md": 12,
        "lg": 16,
        "xl": 24,
    },
    radius={
        # Plumper rounded corners read as more modern. Cards float at
        # ``lg``, primary surfaces at ``md``, badges and pills at ``sm``.
        "sm": 8,
        "md": 12,
        "lg": 16,
        "xl": 20,
    },
    fonts={
        "family": "Segoe UI",
        "size_title": 22,
        "size_heading": 17,
        "size_body": 13,
        "size_small": 11,
    },
)


def _build_substitutions() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in TOKENS.colors.items():
        result[f"color.{key}"] = value
    for key, value in TOKENS.spacing.items():
        result[f"space.{key}"] = f"{value}px"
    for key, value in TOKENS.radius.items():
        result[f"radius.{key}"] = f"{value}px"
    for key, value in TOKENS.fonts.items():
        if isinstance(value, int):
            result[f"font.{key}"] = f"{value}px"
        else:
            result[f"font.{key}"] = str(value)
    # Forward-slash absolute path so QSS ``url(...)`` rules can
    # reference bundled SVG icons (checkbox indicator etc.). QSS
    # treats backslashes as escape characters, so always use
    # ``as_posix``.
    result["path.styles_dir"] = _STYLES_DIR.resolve().as_posix()
    return result


def load_stylesheet(theme: str = "dark") -> str:
    path = _STYLES_DIR / f"{theme}.qss"
    if not path.exists():
        raise ValueError(f"Unknown theme: {theme!r} (looked for {path})")

    qss = path.read_text(encoding="utf-8")
    for token, value in _build_substitutions().items():
        qss = qss.replace(f"{{{{{token}}}}}", value)
    return qss


def apply_theme(app: QApplication, theme: str = "dark") -> None:
    app.setStyleSheet(load_stylesheet(theme))
