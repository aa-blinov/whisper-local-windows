"""Tests for the ResourceWidget."""


def test_resource_widget_starts_with_neutral_metrics(qtbot):
    from app.gui.widgets.resource_widget import ResourceWidget

    widget = ResourceWidget()
    qtbot.addWidget(widget)
    # No metrics pushed yet — defaults to zero, no GPU row.
    assert widget._cpu_percent == 0.0
    assert widget._gpu_vram_percent is None


def test_resource_widget_picks_up_full_metrics(qtbot):
    from app.gui.widgets.resource_widget import ResourceWidget

    widget = ResourceWidget()
    qtbot.addWidget(widget)
    widget.set_metrics(
        {
            "cpu_percent": 32.0,
            "ram_percent": 47.5,
            "ram_used_mb": 7600.0,
            "ram_total_mb": 16000.0,
            "gpu_vram_used_mb": 4096.0,
            "gpu_vram_total_mb": 8192.0,
            "gpu_util_percent": 78.0,
        }
    )
    assert widget._cpu_percent == 32.0
    assert widget._ram_percent == 47.5
    assert abs(widget._gpu_vram_percent - 50.0) < 0.01
    assert widget._gpu_util_percent == 78.0
    assert "GPU VRAM" in widget.toolTip()


def test_resource_widget_drops_gpu_row_when_no_nvml(qtbot):
    """Machines without an NVIDIA driver omit the ``gpu_*`` keys —
    the widget should suppress its GPU row instead of showing 0/0."""
    from app.gui.widgets.resource_widget import ResourceWidget

    widget = ResourceWidget()
    qtbot.addWidget(widget)
    widget.set_metrics(
        {
            "cpu_percent": 18.0,
            "ram_percent": 33.0,
            "ram_used_mb": 5300.0,
            "ram_total_mb": 16000.0,
        }
    )
    assert widget._gpu_vram_percent is None
    rows = widget._row_specs()
    assert all(label != "GPU" for label, _, _ in rows)


def test_resource_widget_color_thresholds():
    from app.gui.widgets.resource_widget import _fill_color_for, _GREEN, _AMBER, _RED

    assert _fill_color_for(20.0) is _GREEN
    assert _fill_color_for(70.0) is _AMBER
    assert _fill_color_for(95.0) is _RED
