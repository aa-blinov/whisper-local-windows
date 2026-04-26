"""Tests for the startup splash-screen helpers.

The splash screen is the simplest fix for the "main window unresponsive
during NeMo cold-import" complaint — show a movable / minimisable
splash widget BEFORE creating the main window, then synchronously load
the model in the main thread while pumping ``QApplication.processEvents()``
so Qt can deliver mouse / window-manager events to the splash.

Once the model is ready (or errors out), close the splash and show the
main window — which now appears with a fully-loaded backend and never
freezes during startup.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional


class _FakeBackend:
    """Minimal backend stub satisfying the surface ``wait_for_backend``
    needs: ``status()``, ``load()``, ``set_progress_callback``."""

    def __init__(self, eventual_status: str = "ready", delay_s: float = 0.05):
        self._status = "stopped"
        self._eventual = eventual_status
        self._delay = delay_s
        self._cb: Optional[Callable[[int, int, str], None]] = None
        self.load_called = False

    def status(self) -> str:
        return self._status

    def set_progress_callback(self, callback) -> None:
        self._cb = callback

    def load(self) -> None:
        self.load_called = True
        self._status = "loading"

        def _worker():
            # Fire a couple of progress ticks so the splash has
            # something to render. Re-read ``self._cb`` each time
            # because ``wait_for_backend`` clears it on completion —
            # without the re-read we'd race a None into the call.
            cb = self._cb
            if cb is not None:
                try:
                    cb(0, 100_000_000, "loading")
                except Exception:
                    pass
            time.sleep(self._delay / 4)
            cb = self._cb
            if cb is not None:
                try:
                    cb(50_000_000, 100_000_000, "loading")
                except Exception:
                    pass
            time.sleep(self._delay)
            self._status = self._eventual

        threading.Thread(target=_worker, daemon=True).start()


def test_make_splash_returns_widget(qtbot):
    """Smoke test — make_splash returns a QSplashScreen we can show()
    without crashing."""
    from PySide6.QtWidgets import QSplashScreen

    from app.gui.splash import make_splash

    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    assert isinstance(splash, QSplashScreen)
    splash.show()
    assert splash.isVisible()


def test_wait_for_backend_returns_ready_when_status_ready(qtbot):
    from PySide6.QtWidgets import QApplication

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="ready", delay_s=0.05)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    result = wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    assert result == "ready"
    assert backend.load_called is True


def test_wait_for_backend_returns_error_on_failure(qtbot):
    from PySide6.QtWidgets import QApplication

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="error", delay_s=0.05)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    result = wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    assert result == "error"


def test_wait_for_backend_returns_timeout_when_load_never_finishes(qtbot):
    """Defensive — if the backend gets stuck in 'loading' forever, the
    splash helper has to give up rather than block the app start."""
    from PySide6.QtWidgets import QApplication

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="loading", delay_s=10.0)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    result = wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=0.3,
    )

    assert result == "timeout"


def test_wait_for_backend_pumps_process_events(qtbot):
    """The whole point of the splash is that Qt keeps processing
    events while the model loads. Verify a queued signal fires
    during the wait."""
    from PySide6.QtCore import QObject, QTimer, Signal
    from PySide6.QtWidgets import QApplication

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="ready", delay_s=0.3)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    fired: List[str] = []

    class _Bus(QObject):
        ping = Signal()

    bus = _Bus()
    bus.ping.connect(lambda: fired.append("tick"))
    # Schedule the signal to fire 50 ms in — well before the load
    # finishes, but it can only fire if processEvents is being pumped.
    QTimer.singleShot(50, bus.ping.emit)

    result = wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    assert result == "ready"
    assert fired == ["tick"], (
        "Qt timer didn't fire — processEvents is not being pumped"
    )


def test_wait_for_backend_updates_splash_with_progress(qtbot):
    """Backend progress callbacks should reach the splash message —
    user sees '… 50%' rather than a frozen 'Loading' string for
    minutes."""
    from PySide6.QtWidgets import QApplication

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="ready", delay_s=0.2)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    seen_messages: List[str] = []
    original_show_message = splash.showMessage

    def capture_show_message(msg, *args, **kwargs):
        seen_messages.append(msg)
        return original_show_message(msg, *args, **kwargs)

    splash.showMessage = capture_show_message  # type: ignore[assignment]

    wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    # Some message should have included a percentage after the
    # progress callback fired.
    assert any("%" in m for m in seen_messages), (
        f"expected a percentage in splash messages, got {seen_messages!r}"
    )
