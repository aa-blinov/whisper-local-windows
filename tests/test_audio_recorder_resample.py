"""Tests for AudioRecorder._resample_to.

Covers: correct output length, dtype, no-op path, stereo, edge cases, and
signal integrity (DC + low-frequency sine preserved through resampling).
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample(audio, src, dst):
    from app.audio_recorder import AudioRecorder
    return AudioRecorder._resample_to(audio, src, dst)


def _sine(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    """Pure sine at freq_hz, float32, mono."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Output shape & dtype
# ---------------------------------------------------------------------------

def test_resample_48k_to_16k_length():
    """48 kHz → 16 kHz must yield exactly 1/3 of the original samples."""
    audio = np.zeros(48_000, dtype=np.float32)  # 1 second
    out = _resample(audio, 48_000, 16_000)
    assert len(out) == 16_000


def test_resample_44100_to_16k_length():
    """44.1 kHz → 16 kHz: non-integer ratio handled correctly."""
    audio = np.zeros(44_100, dtype=np.float32)  # 1 second
    out = _resample(audio, 44_100, 16_000)
    # Allow ±1 sample rounding
    assert abs(len(out) - 16_000) <= 1


def test_resample_output_is_float32():
    """Output must always be float32 regardless of input dtype."""
    audio = np.zeros(48_000, dtype=np.float64)
    out = _resample(audio, 48_000, 16_000)
    assert out.dtype == np.float32


def test_resample_noop_same_rate():
    """Same source and target rate returns the array as float32 unchanged."""
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _resample(audio, 16_000, 16_000)
    assert out.dtype == np.float32
    np.testing.assert_array_almost_equal(out, audio)


def test_resample_empty_returns_empty_float32():
    audio = np.array([], dtype=np.float32)
    out = _resample(audio, 48_000, 16_000)
    assert out.dtype == np.float32
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Stereo (2-D input)
# ---------------------------------------------------------------------------

def test_resample_stereo_preserves_channel_count():
    """2-D (frames × channels) input must stay 2-D with correct frame count."""
    audio = np.zeros((48_000, 2), dtype=np.float32)
    out = _resample(audio, 48_000, 16_000)
    assert out.ndim == 2
    assert out.shape == (16_000, 2)
    assert out.dtype == np.float32


def test_resample_stereo_channels_are_independent():
    """Each channel must be resampled independently — L≠R must stay L≠R."""
    rng = np.random.default_rng(42)
    left = rng.uniform(-1, 1, 4_800).astype(np.float32)
    right = rng.uniform(-1, 1, 4_800).astype(np.float32)
    audio = np.stack([left, right], axis=1)

    out_stereo = _resample(audio, 48_000, 16_000)
    out_left = _resample(left, 48_000, 16_000)
    out_right = _resample(right, 48_000, 16_000)

    np.testing.assert_array_almost_equal(out_stereo[:, 0], out_left, decimal=5)
    np.testing.assert_array_almost_equal(out_stereo[:, 1], out_right, decimal=5)


# ---------------------------------------------------------------------------
# Signal integrity
# ---------------------------------------------------------------------------

def test_resample_dc_signal_preserved():
    """A constant (DC) signal must stay constant after resampling."""
    audio = np.full(48_000, 0.5, dtype=np.float32)
    out = _resample(audio, 48_000, 16_000)
    # Trim edges where the FIR filter ramps up/down
    interior = out[100:-100]
    assert np.allclose(interior, 0.5, atol=1e-4), (
        f"DC not preserved: mean={interior.mean():.6f}, std={interior.std():.6f}"
    )


def test_resample_low_freq_sine_preserved():
    """A 440 Hz sine at 48 kHz must survive resampling to 16 kHz intact.

    440 Hz is well below the 8 kHz Nyquist of the 16 kHz target, so both
    amplitude and frequency must be preserved.  We check via correlation
    with the expected 440 Hz signal at 16 kHz.
    """
    src = _sine(440.0, 1.0, 48_000)
    out = _resample(src, 48_000, 16_000)
    ref = _sine(440.0, 1.0, 16_000)

    # Trim FIR filter edge artefacts (first/last ~5 ms)
    trim = 80
    out_trim = out[trim:-trim]
    ref_trim = ref[trim : trim + len(out_trim)]

    # Normalise both and correlate — expect near-perfect alignment
    out_n = out_trim / (np.max(np.abs(out_trim)) + 1e-9)
    ref_n = ref_trim / (np.max(np.abs(ref_trim)) + 1e-9)
    corr = float(np.corrcoef(out_n, ref_n)[0, 1])
    assert corr > 0.999, f"440 Hz sine corrupted after resampling (corr={corr:.4f})"


def test_resample_high_freq_sine_attenuated():
    """A 10 kHz sine (above the 8 kHz Nyquist of 16 kHz target) must be
    suppressed — resample_poly applies an anti-aliasing filter that
    np.interp did not, so aliasing artefacts are eliminated."""
    src = _sine(10_000.0, 0.5, 48_000)
    out = _resample(src, 48_000, 16_000)
    # After proper anti-aliasing, amplitude of a 10 kHz source must be small
    interior = out[100:-100]
    assert np.max(np.abs(interior)) < 0.1, (
        f"10 kHz signal not attenuated (max={np.max(np.abs(interior)):.4f})"
    )
