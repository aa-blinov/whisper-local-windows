"""Tests for the BackendStatusPoller."""


def test_tick_emits_initial_status(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "stopped")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()
    canonical, _label = blocker.args
    assert canonical == "stopped"


def test_tick_dedupes_unchanged_status(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "stopped")
    emissions: list[tuple[str, str]] = []
    poller.status_changed.connect(lambda s, l: emissions.append((s, l)))

    with qtbot.waitSignal(poller.status_changed, timeout=1000):
        poller.tick()

    # Second and third ticks return the same value — no further emission.
    poller.tick()
    qtbot.wait(150)
    poller.tick()
    qtbot.wait(150)

    assert len(emissions) == 1


def test_tick_emits_when_status_changes(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    states = iter(["error", "error", "stopped"])
    poller = BackendStatusPoller(fetcher=lambda: next(states))
    emissions: list[str] = []
    poller.status_changed.connect(lambda s, _l: emissions.append(s))

    with qtbot.waitSignal(poller.status_changed, timeout=1000):
        poller.tick()
    poller.tick()
    qtbot.wait(150)
    with qtbot.waitSignal(poller.status_changed, timeout=1000):
        poller.tick()

    assert emissions == ["error", "stopped"]


def test_tick_runs_fetcher_off_main_thread(qtbot):
    """The fetcher must execute on a worker thread, not the UI thread."""
    import threading

    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    main_thread = threading.current_thread()
    seen_thread: list[threading.Thread] = []

    def probe() -> str:
        seen_thread.append(threading.current_thread())
        return "ready"

    poller = BackendStatusPoller(fetcher=probe)
    with qtbot.waitSignal(poller.status_changed, timeout=1000):
        poller.tick()

    assert seen_thread, "fetcher was never called"
    assert seen_thread[0] is not main_thread


def test_slow_fetcher_does_not_block_ui_thread(qtbot):
    """A long fetcher call must not freeze the event loop between ticks."""
    import time

    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    def slow_fetcher() -> str:
        time.sleep(0.4)
        return "ready"

    poller = BackendStatusPoller(fetcher=slow_fetcher)
    poller.tick()

    start = time.monotonic()
    qtbot.wait(50)
    elapsed = time.monotonic() - start
    assert elapsed < 0.2

    qtbot.waitSignal(poller.status_changed, timeout=1500).wait()


def test_overlapping_ticks_do_not_pile_up(qtbot):
    """A second tick fired while a fetch is in flight must be a no-op."""
    import time

    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    call_count = {"n": 0}

    def slow_fetcher() -> str:
        call_count["n"] += 1
        time.sleep(0.2)
        return "ready"

    poller = BackendStatusPoller(fetcher=slow_fetcher)
    poller.tick()
    poller.tick()  # should be skipped — previous still running
    poller.tick()

    qtbot.waitSignal(poller.status_changed, timeout=2000).wait()
    qtbot.wait(100)

    assert call_count["n"] == 1


def test_map_loading_renders_as_hidden(qtbot):
    """Loading is shown by the recording-state pill on the left, so the
    backend status pill on the right hides itself instead of duplicating."""
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "loading")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, _label = blocker.args
    assert canonical == "hidden"


def test_map_ready_renders_as_hidden(qtbot):
    """No need to advertise 'all good' — keep the topbar quiet when the
    backend is ready."""
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "ready")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, _label = blocker.args
    assert canonical == "hidden"


def test_map_backend_error_becomes_error(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "error")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, _label = blocker.args
    assert canonical == "error"


def test_map_stopped_renders_as_stopped(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "stopped")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, label = blocker.args
    assert canonical == "stopped"
    assert "not loaded" in label.lower() or "model" in label.lower()


def test_fetcher_exception_surfaces_as_error(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    def boom() -> str:
        raise RuntimeError("backend crashed")

    poller = BackendStatusPoller(fetcher=boom)
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, label = blocker.args
    assert canonical == "error"
    assert "backend crashed" in label or "error" in label.lower()


def test_start_kicks_off_immediate_tick(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "stopped", interval_ms=10000)
    try:
        with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
            poller.start()
        assert blocker.args[0] == "stopped"
    finally:
        poller.stop()


def test_stop_halts_polling(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "running", interval_ms=5)
    poller.start()
    poller.stop()
    assert not poller._timer.isActive()
