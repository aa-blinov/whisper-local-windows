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
        self.cancel_called = False

    def status(self) -> str:
        return self._status

    def set_progress_callback(self, callback) -> None:
        self._cb = callback

    def cancel_load(self) -> None:
        self.cancel_called = True
        self._status = "stopped"

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
            # Don't overwrite "stopped" if cancel was requested.
            if self._status == "loading":
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


def test_wait_for_backend_shows_cancel_button_when_backend_supports_it(qtbot):
    """A Cancel button must appear on the splash when the backend exposes
    ``cancel_load()``, and clicking it must call ``cancel_load`` and
    return ``'stopped'`` from ``wait_for_backend``."""
    from PySide6.QtWidgets import QApplication, QPushButton

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    # Backend that takes a long time — so the button has a chance to fire
    backend = _FakeBackend(eventual_status="ready", delay_s=10.0)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    # Schedule a click on the Cancel button 100 ms after the splash appears
    def _click_cancel():
        btns = splash.findChildren(QPushButton)
        for btn in btns:
            if btn.text() == "Cancel":
                btn.click()
                return

    from PySide6.QtCore import QTimer
    QTimer.singleShot(100, _click_cancel)

    result = wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    assert result == "stopped"
    assert backend.cancel_called is True


def test_wait_for_backend_no_cancel_button_when_backend_lacks_cancel_load(qtbot):
    """Backends without ``cancel_load`` must not get a Cancel button —
    the button is opt-in to avoid calling a non-existent method."""
    from PySide6.QtWidgets import QApplication, QPushButton

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="ready", delay_s=0.05)
    # Shadow the inherited method with None on the instance so
    # ``getattr(backend, "cancel_load", None)`` returns None.
    backend.cancel_load = None  # type: ignore[assignment]

    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    btns = splash.findChildren(QPushButton)
    cancel_btns = [b for b in btns if b.text() == "Cancel"]
    assert cancel_btns == [], "no Cancel button expected when backend lacks cancel_load"


def test_wait_for_backend_updates_splash_with_progress(qtbot):
    """Backend progress callbacks should reach the status label —
    user sees '… 50%' rather than a frozen 'Loading' string for
    minutes."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QLabel

    from app.gui.splash import make_splash, wait_for_backend

    app = QApplication.instance()
    backend = _FakeBackend(eventual_status="ready", delay_s=0.2)
    splash = make_splash("Lazy to Text")
    qtbot.addWidget(splash)
    splash.show()

    # Collect label texts while the splash is live. QTimer fires inside
    # wait_for_backend because processEvents() is pumped every 20 ms.
    seen_messages: List[str] = []
    _active = [True]  # mutable flag — set to False after wait_for_backend returns

    def _capture() -> None:
        if not _active[0]:
            return  # wait_for_backend already finished — don't touch the splash
        try:
            for lbl in splash.findChildren(QLabel):
                t = lbl.text()
                if t and t not in seen_messages:
                    seen_messages.append(t)
        except RuntimeError:
            return  # C++ object already deleted — stop rescheduling
        QTimer.singleShot(40, _capture)

    QTimer.singleShot(40, _capture)

    wait_for_backend(
        splash=splash,
        backend=backend,
        display_name="Test Model",
        app=app,
        timeout_s=2.0,
    )

    # Stop the recurring capture timer — any pending tick will see _active=False
    # and return immediately without touching the (already-closed) splash.
    _active[0] = False

    # Some captured text should include a percentage after the progress
    # callback fired.
    assert any("%" in m for m in seen_messages), (
        f"expected a percentage in splash messages, got {seen_messages!r}"
    )
