"""Main application window — shell with sidebar navigation and stacked views."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.views.history_view import HistoryView
from app.gui.views.logs_view import LogsView
from app.gui.views.models_view import ModelsView
from app.gui.views.placeholder import PlaceholderView
from app.gui.views.shortcuts_view import ShortcutsView
from app.gui.views.transcribe_view import TranscribeView
from app.gui.widgets.sidebar import Sidebar
from app.gui.widgets.toast import Toast
from app.gui.widgets.topbar import TopBar


class MainWindow(QMainWindow):
    hidden_to_tray = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lazy to Text")
        # Settings + Models views both wrap their card stacks in a
        # QScrollArea, so we can drop back to a sane minimum that
        # fits a 1366×768 laptop with Windows scaling on. Default
        # has breathing room for 4-5 cards visible without
        # scrolling on most desktops.
        self.resize(1100, 780)
        self.setMinimumSize(900, 620)
        self._close_to_tray = False
        self._quitting = False

        central = QWidget(self)
        central.setObjectName("Central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.topbar = TopBar(parent=central)
        root.addWidget(self.topbar)

        body = QWidget(central)
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = Sidebar(parent=body)
        self.stack = QStackedWidget(body)

        body_layout.addWidget(self.sidebar)
        body_layout.addWidget(self.stack, 1)

        root.addWidget(body, 1)

        self._views: Dict[str, QWidget] = {}
        self.models_view = ModelsView(parent=self.stack)
        self.transcribe_view = TranscribeView(parent=self.stack)
        self.logs_view = LogsView(parent=self.stack)
        self.shortcuts_view = ShortcutsView(parent=self.stack)
        self.history_view = HistoryView(parent=self.stack)
        for key in self.sidebar.items():
            if key == "models":
                view: QWidget = self.models_view
            elif key == "transcribe":
                view = self.transcribe_view
            elif key == "logs":
                view = self.logs_view
            elif key == "shortcuts":
                view = self.shortcuts_view
            elif key == "history":
                view = self.history_view
            else:
                view = PlaceholderView(key.capitalize(), parent=self.stack)
            self._views[key] = view
            self.stack.addWidget(view)

        default_key = self.sidebar.active_key()
        if default_key in self._views:
            self.stack.setCurrentWidget(self._views[default_key])

        self.sidebar.nav_selected.connect(self._on_nav_selected)

        # Floating banner that shows after every successful
        # transcription. Parented to the central widget so it sits
        # above the views; positioned in the top-right by ``Toast``
        # itself.
        self.toast = Toast(parent=central)
        central.installEventFilter(self)

        # Ctrl+1..4 jump straight to the matching tab — same order as
        # the sidebar.
        self._shortcuts: list[QShortcut] = []
        nav_keys = list(self.sidebar.items())
        for index, key in enumerate(nav_keys[:9]):
            sc = QShortcut(
                QKeySequence(f"Ctrl+{index + 1}"),
                self,
            )
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda k=key: self._activate_nav(k))
            self._shortcuts.append(sc)

        # Standard "App / About / Settings / Quit" menu — visible
        # under the Apple logo on macOS, in a regular top menu bar
        # on Windows / Linux.
        self._install_app_menu()

    def get_view(self, key: str) -> QWidget:
        if key not in self._views:
            raise KeyError(key)
        return self._views[key]

    def set_close_to_tray(self, enabled: bool) -> None:
        """When True, the window's close button hides to tray instead of
        quitting; the controller is expected to wire a tray icon that can
        bring the window back. Call ``request_quit`` to bypass the override
        for a real exit."""
        self._close_to_tray = bool(enabled)

    def request_quit(self) -> None:
        """Mark the next close as a real quit and close the window."""
        self._quitting = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt naming)
        if self._close_to_tray and not self._quitting:
            event.ignore()
            self.hide()
            self.hidden_to_tray.emit()
            return
        super().closeEvent(event)

    def _on_nav_selected(self, key: str) -> None:
        if key in self._views:
            self.stack.setCurrentWidget(self._views[key])

    def _activate_nav(self, key: str) -> None:
        """Switch to the named tab — used by the Ctrl+N keyboard
        shortcuts."""
        try:
            self.sidebar.set_active(key)
        except ValueError:
            pass

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        # Re-anchor the toast on resize so it sticks to the top-right
        # corner regardless of window size.
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Resize and watched is self.centralWidget():
            self.toast.parentResized()
        return super().eventFilter(watched, event)

    def _install_app_menu(self) -> None:
        """Wire the standard ``About / Settings / Quit`` menu items.

        On macOS, Qt promotes any QAction whose ``MenuRole`` is set
        to ``AboutRole / PreferencesRole / QuitRole`` into the global
        Application menu (the one under the Apple logo) regardless
        of which submenu we attach them to — so the host menu's
        title is irrelevant on Mac. On Windows / Linux the same
        actions live in a regular ``Lazy to Text`` top-menu.

        ``Ctrl+,`` and ``Ctrl+Q`` are auto-translated to ``Cmd+,``
        and ``Cmd+Q`` on macOS by Qt's portable key sequence layer
        — matching what every other Mac app shows next to those
        menu items.
        """
        menu_bar = self.menuBar()
        app_menu = menu_bar.addMenu("&Lazy to Text")

        about_action = QAction("&About Lazy to Text", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self._show_about_dialog)
        app_menu.addAction(about_action)

        app_menu.addSeparator()

        settings_action = QAction("&Settings…", self)
        # Ctrl+, → Cmd+, on macOS via Qt's portable key sequence
        # layer.  ``ApplicationShortcut`` so the binding works even
        # when focus is in a child widget (e.g. log search box).
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.setShortcutContext(Qt.ApplicationShortcut)
        settings_action.setMenuRole(QAction.MenuRole.PreferencesRole)
        settings_action.triggered.connect(self._open_settings_view)
        app_menu.addAction(settings_action)

        app_menu.addSeparator()

        quit_action = QAction("&Quit Lazy to Text", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.setShortcutContext(Qt.ApplicationShortcut)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self._quit_application)
        app_menu.addAction(quit_action)

    def _open_settings_view(self) -> None:
        """Bring the Settings tab to the front.

        Maps to the sidebar's ``shortcuts`` nav key — the historical
        name from when the view only held hotkey bindings; it has
        since grown into the full settings surface (storage, HF
        token, audio feedback, …) but we kept the key stable to
        avoid migrating the persisted ``Sidebar.active_key`` for
        existing users.

        If the window was hidden to tray when ``Cmd+,`` fires, we
        un-hide and raise it so the user actually sees what they
        opened.
        """
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()
        self._activate_nav("shortcuts")

    def _quit_application(self) -> None:
        """``Cmd+Q`` / menu ``Quit`` — full app exit, bypassing tray.

        Mirrors what the tray's Quit menu does (see
        ``_tray_mixin._on_tray_quit``): mark the next window close
        as a real quit (so ``closeEvent`` doesn't fall back to
        hide-to-tray), then drop out of Qt's event loop. Without
        the explicit ``app.quit()`` we'd just hide the window —
        ``setQuitOnLastWindowClosed(False)`` is set when the tray
        is alive, so closing the last window doesn't end the
        process by itself.
        """
        self.request_quit()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _show_about_dialog(self) -> None:
        """``About Lazy to Text`` info dialog (app icon + version).

        Pulls the version string from the installed package
        metadata when possible — when running from a wheel /
        ``pip install -e .`` install ``importlib.metadata`` knows
        the canonical version.  Falls back to a hardcoded string
        when running from a raw checkout where no distribution
        metadata exists yet.
        """
        from app.gui.widgets.dialogs import notify

        try:
            from importlib.metadata import PackageNotFoundError
            from importlib.metadata import version as _pkg_version

            try:
                version = _pkg_version("lazy-to-text")
            except PackageNotFoundError:
                version = "0.0.1"
        except Exception:
            version = "0.0.1"

        notify(
            self,
            "About Lazy to Text",
            (
                f"Lazy to Text {version}\n\n"
                "Local-first speech-to-text — Whisper / Parakeet / GigaAM\n"
                "via ONNX Runtime.  Apple Silicon goes through CoreML\n"
                "(Neural Engine + GPU); NVIDIA goes through CUDA / "
                "TensorRT.\n\n"
                "https://github.com/aa-blinov/lazy-to-text"
            ),
            kind="info",
        )
