"""Tests for the ResourceMonitor.

The collector touches ``psutil`` and ``pynvml`` directly; tests
sample synchronously via ``ResourceMonitor.sample()`` so we don't
need a Qt event loop to verify the contract.
"""


def test_resource_monitor_sample_includes_cpu_ram_keys(qtbot):
    from app.resource_monitor import ResourceMonitor

    monitor = ResourceMonitor()
    snapshot = monitor.sample()
    # CPU / RAM are always present (psutil ships on every platform
    # the project supports).
    assert "cpu_percent" in snapshot
    assert "ram_percent" in snapshot
    assert "ram_used_mb" in snapshot
    assert "ram_total_mb" in snapshot
    assert snapshot["ram_total_mb"] > 0
    assert 0 <= snapshot["ram_percent"] <= 100


def test_resource_monitor_emits_metrics_updated_signal(qtbot):
    from app.resource_monitor import ResourceMonitor

    monitor = ResourceMonitor(interval_ms=500)
    received: list[dict] = []
    monitor.metrics_updated.connect(received.append)
    try:
        monitor.start()
        # ``start`` fires one sample synchronously before arming
        # the timer, so we should see at least one record.
        assert received, "expected at least one immediate sample"
        assert "cpu_percent" in received[0]
    finally:
        monitor.stop()
