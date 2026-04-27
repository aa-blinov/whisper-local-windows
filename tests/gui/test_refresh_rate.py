"""Tests for the display-refresh-rate helper used by the smooth-scroll
filter and the VU-meter timer to match the user's actual monitor."""

from __future__ import annotations

from unittest.mock import MagicMock


def _patch_primary_screen(monkeypatch, rate: float | None) -> None:
    """Make ``QGuiApplication.primaryScreen()`` report a fake refresh rate.

    Pass ``None`` to simulate a headless / unknown-screen environment
    (the helper must fall back gracefully)."""
    from PySide6.QtGui import QGuiApplication

    if rate is None:
        monkeypatch.setattr(
            QGuiApplication, "primaryScreen", staticmethod(lambda: None)
        )
        return

    fake_screen = MagicMock()
    fake_screen.refreshRate = MagicMock(return_value=rate)
    monkeypatch.setattr(
        QGuiApplication, "primaryScreen", staticmethod(lambda: fake_screen)
    )


def test_returns_60_when_screen_unavailable(qapp, monkeypatch):
    """No screen → safe fallback of 60 Hz so timers don't degenerate
    to 0 or NaN intervals."""
    _patch_primary_screen(monkeypatch, None)

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 60


def test_returns_60_for_60hz_displays(qapp, monkeypatch):
    """Most laptops + integrated panels — keep the legacy default."""
    _patch_primary_screen(monkeypatch, 60.0)

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 60


def test_returns_144_for_144hz_displays(qapp, monkeypatch):
    """144 Hz gaming monitors — promote scrolling / VU meter to native
    rate so animations don't stutter at 60 / 144 ratios."""
    _patch_primary_screen(monkeypatch, 143.998)  # real-world rounding

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 144


def test_clamps_low_rates_up_to_60(qapp, monkeypatch):
    """Some screens report 30 Hz briefly during sleep / wake — don't
    drop our timers to that, the UI would feel chunky."""
    _patch_primary_screen(monkeypatch, 30.0)

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 60


def test_clamps_high_rates_to_240(qapp, monkeypatch):
    """480 Hz exotic displays exist — cap at 240 to keep CPU burn
    bounded.  Twelve-frame-per-second VU updates are imperceptibly
    different from twenty regardless of native rate."""
    _patch_primary_screen(monkeypatch, 480.0)

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 240


def test_handles_qt_exceptions_gracefully(qapp, monkeypatch):
    """If Qt raises (no QGuiApplication, screen disconnected mid-call),
    fall back rather than blowing up early in app startup."""
    from PySide6.QtGui import QGuiApplication

    def boom() -> None:
        raise RuntimeError("display server died")

    monkeypatch.setattr(QGuiApplication, "primaryScreen", staticmethod(boom))

    from app.gui.refresh_rate import display_refresh_rate

    assert display_refresh_rate() == 60


# ---- Convenience: tick_interval_ms ----------------------------------------


def test_tick_interval_ms_matches_display_rate_on_high_refresh(
    qapp, monkeypatch,
):
    """``tick_interval_ms()`` returns ``round(1000 / refresh_rate)``
    so smooth_scroll / VU updates align with the native frame cadence."""
    _patch_primary_screen(monkeypatch, 144.0)

    from app.gui.refresh_rate import tick_interval_ms

    # 1000 / 144 ≈ 6.94 → round to 7
    assert tick_interval_ms() == 7


def test_tick_interval_ms_returns_16_at_60hz(qapp, monkeypatch):
    _patch_primary_screen(monkeypatch, 60.0)

    from app.gui.refresh_rate import tick_interval_ms

    # 1000 / 60 ≈ 16.67 → round to 17
    assert tick_interval_ms() == 17


def test_tick_interval_ms_max_rate_caps_to_60_for_vu_meter(qapp, monkeypatch):
    """``tick_interval_ms(max_rate=60)`` lets callers opt into a
    lower-rate cap — the VU meter doesn't benefit from >60 Hz updates
    because audio levels are sampled at ~10 ms anyway."""
    _patch_primary_screen(monkeypatch, 144.0)

    from app.gui.refresh_rate import tick_interval_ms

    # Capped at 60 Hz → 17 ms tick even on 144 Hz display
    assert tick_interval_ms(max_rate=60) == 17
