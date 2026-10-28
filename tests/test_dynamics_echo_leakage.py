"""Tests for the echo animation (viz/dynamics_echo.py) and the leakage figure (viz/dynamics_leakage.py)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from calibration_utils.style import KET_E, theme_context  # noqa: E402
from viz.dynamics_echo import CAPTION, ECHO_TAU_NS, echo_ensemble, make_echo_animation  # noqa: E402
from viz.dynamics_leakage import (  # noqa: E402
    SHORT_LENGTH_NS,
    SHORT_PI_AMPLITUDE,
    leakage_numbers,
    plot_leakage,
    pulse_populations,
)


def _mean_length(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(vectors.mean(axis=0)))


def test_echo_ensemble_refocuses():
    times = np.array([0.0, ECHO_TAU_NS, 2 * ECHO_TAU_NS])
    vecs = echo_ensemble(times, n_spins=24, spread_hz=50e3, seed=0)
    assert vecs.shape == (3, 24, 3)
    # Every spin starts on -y after the x90.
    np.testing.assert_allclose(vecs[0], np.tile([0.0, -1.0, 0.0], (24, 1)), atol=1e-12)
    # Half way (just before the x180) the fan has spread; at the echo time it is back.
    assert _mean_length(vecs[1]) < 0.5
    assert _mean_length(vecs[2]) > 0.95
    # The x180 about +x maps -y to +y, so the echo forms on +y.
    np.testing.assert_allclose(vecs[2].mean(axis=0), [0.0, 1.0, 0.0], atol=1e-9)
    # All vectors stay on the equator.
    np.testing.assert_allclose(vecs[..., 2], 0.0, atol=1e-12)


def test_echo_ensemble_is_seeded():
    times = np.linspace(0, 2 * ECHO_TAU_NS, 7)
    a = echo_ensemble(times, n_spins=8, spread_hz=50e3, seed=3)
    b = echo_ensemble(times, n_spins=8, spread_hz=50e3, seed=3)
    np.testing.assert_array_equal(a, b)


def test_leakage_numbers():
    n = leakage_numbers()
    assert n["p_f_40"] == pytest.approx(9e-7, rel=0.2)
    assert n["residual_g_40"] == pytest.approx(0.0028, rel=0.2)
    assert n["x90_phase_error_40"] == pytest.approx(0.024, rel=0.2)
    assert n["p_f_10"] > 100 * n["p_f_40"]


def test_short_pulse_is_a_pi_rotation():
    _, pops = pulse_populations(SHORT_LENGTH_NS, SHORT_LENGTH_NS / 5, SHORT_PI_AMPLITUDE)
    # Most of the population ends in |e>; what is missing went to |f> or stayed in |g>.
    assert pops[-1, 1] > 0.94
    assert pops[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_plot_leakage_renders(theme):
    with theme_context(theme):
        fig = plot_leakage(theme)
        try:
            text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
            assert "40 ns" in text and "10 ns" in text
            # The 10 ns pulse has pi area but ends near P(e) 0.95, so nothing calls it a pi pulse.
            assert "π area" in text and "π pulse" not in text
            assert f"{KET_E} after pulse: 95." in text
            # The amplitudes are the device's true pi amplitude (and 4 times it), not fitted values.
            assert "true π amplitude, 147.14 mV" in text and "4 × true π amplitude, 588.56 mV" in text
            assert "calibrated" not in text
            fig.canvas.draw()
        finally:
            plt.close(fig)


def test_caption_text():
    assert CAPTION == (
        "Illustration: static detunings, 50 kHz rms. The simulated device has 5.1 kHz rms, "
        "refocused the same way."
    )


@pytest.mark.slow
def test_make_echo_animation(tmp_path):
    path = make_echo_animation(tmp_path)
    assert path == tmp_path / "bloch-echo.gif"
    assert path.stat().st_size < 1_000_000
    with Image.open(path) as im:
        assert im.n_frames > 50
        # Wide enough that the 9.5 pt strip labels are about 13 px at native size.
        assert im.size == (560, 620)
