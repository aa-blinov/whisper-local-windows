"""Tests for the BackendStatusPoller."""


def test_tick_emits_initial_status(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "running")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()
    canonical, _label = blocker.args
    assert canonical == "running"


def test_tick_dedupes_unchanged_status(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "running")
    emissions: list[tuple[str, str]] = []
    poller.status_changed.connect(lambda s, l: emissions.append((s, l)))

    poller.tick()
    poller.tick()
    poller.tick()

    assert len(emissions) == 1


def test_tick_emits_when_status_changes(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    states = iter(["running", "running", "stopped"])
    poller = BackendStatusPoller(fetcher=lambda: next(states))
    emissions: list[str] = []
    poller.status_changed.connect(lambda s, _l: emissions.append(s))

    poller.tick()
    poller.tick()
    poller.tick()

    assert emissions == ["running", "stopped"]


def test_map_docker_not_found_becomes_stopped(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "not_found")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, label = blocker.args
    assert canonical == "stopped"
    assert "not found" in label.lower() or "container" in label.lower()


def test_map_docker_error_becomes_error(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "error")
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, _label = blocker.args
    assert canonical == "error"


def test_fetcher_exception_surfaces_as_error(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    def boom() -> str:
        raise RuntimeError("docker crashed")

    poller = BackendStatusPoller(fetcher=boom)
    with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
        poller.tick()

    canonical, label = blocker.args
    assert canonical == "error"
    assert "docker crashed" in label or "error" in label.lower()


def test_start_kicks_off_immediate_tick(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "running", interval_ms=10000)
    try:
        with qtbot.waitSignal(poller.status_changed, timeout=1000) as blocker:
            poller.start()
        assert blocker.args[0] == "running"
    finally:
        poller.stop()


def test_stop_halts_polling(qtbot):
    from app.gui.controllers.backend_status_poller import BackendStatusPoller

    poller = BackendStatusPoller(fetcher=lambda: "running", interval_ms=5)
    poller.start()
    poller.stop()
    assert not poller._timer.isActive()
