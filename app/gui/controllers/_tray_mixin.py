"""System-tray wiring as a mixin.

Hide-to-tray + Show / Quit menu actions.  Tiny but pulled out for
consistency with the other domain mixins so ``AppController``'s
``__init__`` stays a one-screen overview of which slices fire.

Expects the host class to provide ``self._window`` (main window).
The tray instance is passed into ``_wire_tray`` directly so this
mixin doesn't have to assume which attribute holds it.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication


class TrayMixin:
    """Tray-icon slice of ``AppController``."""

    def _wire_tray(self, tray) -> None:
        self._window.set_close_to_tray(True)
        tray.show_requested.connect(self._on_tray_show)
        tray.quit_requested.connect(self._on_tray_quit)

    def _on_tray_show(self) -> None:
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()

    def _on_tray_quit(self) -> None:
        # Close the main window (will accept thanks to request_quit's flag)
        # and then explicitly tell the QApplication to leave its event loop.
        # ``setQuitOnLastWindowClosed(False)`` is set when a tray is present,
        # so the app would otherwise stay alive forever after the window
        # disappears.
        self._window.request_quit()
        app = QApplication.instance()
        if app is not None:
            app.quit()
