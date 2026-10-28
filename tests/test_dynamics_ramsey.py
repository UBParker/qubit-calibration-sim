"""Tests for the Ramsey Bloch animation (viz/dynamics_ramsey.py)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from simulator.device import TRUE_DEVICE
from viz import dynamics_ramsey
from viz.dynamics_ramsey import make_ramsey_animation, ramsey_frames

T2_STAR = TRUE_DEVICE.T2ramsey


@pytest.fixture(scope="module")
def frames():
    return ramsey_frames(detuning_hz=0.3e6, max_idle_ns=20_000.0)


def test_ramsey_frames_starts_at_minus_y(frames):
    v0 = frames.vectors[0]
    assert frames.idle_ns[0] == 0.0
    assert v0[1] < -0.98
    assert abs(v0[0]) < 0.03 and abs(v0[2]) < 0.03


def test_ramsey_frames_equatorial_length_follows_the_true_envelope(frames):
    """The drawn vector is the shot average over the quasi-static offset, so it shrinks as the
    device's envelope exp(-t/T2echo - (t/Tq)^2), not as exp(-t/T2*)."""
    t = frames.idle_ns
    length = np.hypot(frames.vectors[:, 0], frames.vectors[:, 1])
    expected = length[0] * TRUE_DEVICE.ramsey_envelope(t)
    np.testing.assert_allclose(length, expected, rtol=0.05)
    assert t[-1] == pytest.approx(20_000.0)
    # Over 20 us the vector loses a visible fraction of its length.
    assert length[-1] / length[0] == pytest.approx(float(TRUE_DEVICE.ramsey_envelope(20_000.0)), rel=0.05)
    # The envelope (0.588 at 20 us) sits well above the exponential (0.525): fails if the GIF falls
    # back to a single-shot exponential decay.
    assert length[-1] / length[0] > 1.08 * np.exp(-20_000.0 / T2_STAR)


def test_ramsey_frames_precess_at_the_detuning(frames):
    t = frames.idle_ns
    phase = np.unwrap(np.arctan2(frames.vectors[:, 1], frames.vectors[:, 0]))
    rate_hz = abs(np.polyfit(t, phase, 1)[0]) / (2 * np.pi) * 1e9
    assert rate_hz == pytest.approx(0.3e6, rel=0.01)


def test_ramsey_frames_precess_right_handed_for_positive_detuning(frames):
    """Drive above f_01: from -y the vector turns toward +x, a right-handed turn about +z."""
    assert frames.detuning_hz > 0
    assert frames.vectors[1, 0] > 0.1
    assert frames.vectors[1, 0] > frames.vectors[0, 0]


def test_ramsey_frames_z_relaxes_toward_ground(frames):
    z = frames.vectors[:, 2]
    expected = 1.0 - np.exp(-frames.idle_ns / TRUE_DEVICE.T1)
    np.testing.assert_allclose(z, expected, atol=0.03)


def test_ramsey_fringe_matches_vector(frames):
    """The second x90 maps y onto z, so P(e) = (1 - y) / 2 up to leakage."""
    assert frames.p_excited.shape == frames.idle_ns.shape
    np.testing.assert_allclose(frames.p_excited, (1 - frames.vectors[:, 1]) / 2, atol=0.02)
    assert frames.p_excited[0] > 0.97


@pytest.mark.slow
def test_make_ramsey_animation_writes_small_gif(tmp_path, monkeypatch):
    # A short idle, a low frame floor and a short hold keep the render to a few frames.
    monkeypatch.setattr(dynamics_ramsey, "MIN_FRAMES", 8)
    monkeypatch.setattr(dynamics_ramsey, "HOLD_S", 0.2)
    out = make_ramsey_animation(tmp_path / "bloch-ramsey.gif", max_idle_ns=500.0)
    assert out == tmp_path / "bloch-ramsey.gif"
    assert out.exists()
    assert out.stat().st_size < 3_000_000
    with Image.open(out) as im:
        assert im.n_frames >= 8
        assert im.info.get("duration") == 50
        assert im.info.get("loop") == 0
