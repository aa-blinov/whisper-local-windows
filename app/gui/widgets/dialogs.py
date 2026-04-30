"""App-icon-branded replacements for ``QMessageBox`` static helpers.

``QMessageBox.question(...)`` on macOS draws the system "?" template
ignoring ``setWindowIcon`` and similar overrides — the user sees a
generic question-mark instead of the app icon. Apple HIG recommends
the app icon for confirmation / info / warning alerts (the system
critical icon is reserved for hard errors).

This module provides ``confirm`` / ``notify`` thin wrappers that
build a ``QMessageBox`` instance, set its pixmap to the live
``QApplication.windowIcon`` and run ``exec`` synchronously.  Drop-in
replacements for ``QMessageBox.question`` /
``QMessageBox.information`` / ``QMessageBox.warning`` everywhere
the project showed a system-template glyph.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget


_APP_ICON_PIXMAP_SIZE = QSize(64, 64)


def _app_icon_pixmap() -> Optional[QPixmap]:
    """Render the currently-installed app icon at 64 × 64 for use as
    the ``QMessageBox`` icon pixmap.  Returns ``None`` if no
    QApplication has been created yet (defensive — every real
    caller runs after ``main()`` has built the app)."""
    app = QApplication.instance()
    if app is None:
        return None
    icon: QIcon = app.windowIcon()
    if icon.isNull():
        return None
    pixmap = icon.pixmap(_APP_ICON_PIXMAP_SIZE)
    if pixmap.isNull():
        return None
    return pixmap


def _stamp_app_icon(msg: QMessageBox) -> None:
    """Replace the system template glyph (Question / Information /
    Warning) on ``msg`` with the live app icon — no-op when the
    icon isn't available."""
    pixmap = _app_icon_pixmap()
    if pixmap is not None:
        msg.setIconPixmap(pixmap)


def confirm(
    parent: Optional[QWidget],
    title: str,
    text: str,
    *,
    default_yes: bool = False,
    yes_label: str = "Yes",
    cancel_label: str = "Cancel",
) -> bool:
    """Drop-in for ``QMessageBox.question`` that shows the app icon.

    Returns ``True`` when the user clicks the affirmative button,
    ``False`` for cancel / close.  ``default_yes`` controls which
    button is highlighted as the default (Enter target); the
    cancel button is keyboard-default otherwise so accidental
    Enter presses don't fire destructive operations.
    """
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    yes_btn = msg.addButton(yes_label, QMessageBox.AcceptRole)
    cancel_btn = msg.addButton(cancel_label, QMessageBox.RejectRole)
    msg.setDefaultButton(yes_btn if default_yes else cancel_btn)
    _stamp_app_icon(msg)
    msg.exec()
    return msg.clickedButton() is yes_btn


def confirm_three_way(
    parent: Optional[QWidget],
    title: str,
    text: str,
    *,
    yes_label: str,
    no_label: str,
    cancel_label: str = "Cancel",
) -> str:
    """Drop-in for ``QMessageBox.question`` with Yes / No / Cancel.

    Returns ``"yes"`` / ``"no"`` / ``"cancel"`` — matches the way
    Storage's "Move existing weights?" prompt branches.  Cancel is
    the keyboard default, in line with the rule that destructive /
    irreversible options should never be the Enter target.
    """
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    yes_btn = msg.addButton(yes_label, QMessageBox.AcceptRole)
    no_btn = msg.addButton(no_label, QMessageBox.NoRole)
    cancel_btn = msg.addButton(cancel_label, QMessageBox.RejectRole)
    msg.setDefaultButton(cancel_btn)
    _stamp_app_icon(msg)
    msg.exec()
    clicked = msg.clickedButton()
    if clicked is yes_btn:
        return "yes"
    if clicked is no_btn:
        return "no"
    return "cancel"


def notify(
    parent: Optional[QWidget],
    title: str,
    text: str,
    *,
    kind: str = "info",
) -> None:
    """Drop-in for ``QMessageBox.information`` / ``.warning``.

    ``kind`` is informational only (``"info"`` / ``"warning"``) —
    we always paint the app icon, but the value lets future
    callers theme the OK button or trigger different system
    sounds without touching every callsite.
    """
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setDefaultButton(QMessageBox.Ok)
    _stamp_app_icon(msg)
    msg.exec()
