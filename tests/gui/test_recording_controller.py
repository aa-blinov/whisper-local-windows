"""Tests for the RecordingController."""

import threading


class FakeStateManager:
    """Stand-in for StateManager exposing only the surface we depend on."""

    def __init__(self, initial: str = "idle") -> None:
        self._state = initial
        self.history_update_callback = None
        self.shutdown_called = False
        self.model_change_requests: list[str] = []

    def get_current_state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state

    def fire_history_update(self) -> None:
        if self.history_update_callback is not None:
            self.history_update_callback()

    def request_model_change(
        self, new_model_size: str, compute_type=None,
    ) -> bool:
        self.model_change_requests.append((new_model_size, compute_type))
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeHotkeyListener:
    def __init__(self) -> None:
        self.stop_called = False
        self.active = True

    def stop_listening(self) -> None:
        self.stop_called = True
        self.active = False

    def is_active(self) -> bool:
        return self.active


# ---- Polling ---------------------------------------------------------------


def test_polls_initial_state_and_emits(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager("idle")
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    with qtbot.waitSignal(rc.state_changed, timeout=1000) as blocker:
        rc.start()

    assert blocker.args == ["idle"]
    rc.shutdown()


def test_emits_only_on_state_change(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager("idle")
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    emissions: list[str] = []
    rc.state_changed.connect(emissions.append)

    rc.start()
    qtbot.wait(80)  # several polls — all read "idle"

    sm.set_state("recording")
    qtbot.waitUntil(lambda: emissions[-1:] == ["recording"], timeout=500)

    sm.set_state("processing")
    qtbot.waitUntil(lambda: emissions[-1:] == ["processing"], timeout=500)

    sm.set_state("idle")
    qtbot.waitUntil(lambda: emissions[-1:] == ["idle"], timeout=500)

    assert emissions == ["idle", "recording", "processing", "idle"]
    rc.shutdown()


def test_current_state_returns_last_seen(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager("idle")
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)
    rc.start()

    qtbot.waitUntil(lambda: rc.current_state() == "idle", timeout=500)

    sm.set_state("recording")
    qtbot.waitUntil(lambda: rc.current_state() == "recording", timeout=500)
    rc.shutdown()


# ---- History forward -------------------------------------------------------


def test_history_callback_forwards_to_signal(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    with qtbot.waitSignal(rc.history_updated, timeout=1000):
        sm.fire_history_update()

    rc.shutdown()


def test_history_callback_from_worker_thread_marshals_to_main(qtbot):
    """StateManager invokes history_update_callback on the transcription
    pipeline thread; the resulting Qt signal slot must run on main."""
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    main_thread = threading.current_thread()
    received_on: list[threading.Thread] = []

    rc.history_updated.connect(lambda: received_on.append(threading.current_thread()))

    def fire_from_worker():
        sm.fire_history_update()

    worker = threading.Thread(target=fire_from_worker)
    worker.start()
    worker.join()

    qtbot.waitUntil(lambda: len(received_on) == 1, timeout=1000)
    assert received_on[0] is main_thread
    rc.shutdown()


# ---- Model change pass-through ---------------------------------------------


def test_request_model_change_proxies_to_state_manager(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    rc.request_model_change("Systran/faster-whisper-tiny")
    rc.request_model_change("Systran/faster-whisper-large-v3", "int8_float16")

    assert sm.model_change_requests == [
        ("Systran/faster-whisper-tiny", None),
        ("Systran/faster-whisper-large-v3", "int8_float16"),
    ]
    rc.shutdown()


# ---- Lifecycle -------------------------------------------------------------


def test_shutdown_stops_timer_and_hotkey_listener_and_state_manager(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    hk = FakeHotkeyListener()
    rc = RecordingController(
        state_manager=sm,
        hotkey_listener=hk,
        poll_interval_ms=10,
    )
    rc.start()
    qtbot.wait(30)

    rc.shutdown()

    assert sm.shutdown_called is True
    assert hk.stop_called is True
    assert not rc._timer.isActive()


def test_shutdown_is_idempotent(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)
    rc.start()
    rc.shutdown()
    rc.shutdown()  # must not raise

    assert sm.shutdown_called is True


def test_shutdown_disconnects_history_callback_from_state_manager(qtbot):
    from app.gui.controllers.recording_controller import RecordingController

    sm = FakeStateManager()
    rc = RecordingController(state_manager=sm, poll_interval_ms=10)

    rc.shutdown()

    # After shutdown the controller must no longer hold the callback.
    assert sm.history_update_callback is None
