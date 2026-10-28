"""Tests for the Rabi Bloch animation (viz/dynamics_rabi.py)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from simulator.device import TRUE_PI_AMPLITUDE  # noqa: E402
from viz.bloch import MAX_GIF_BYTES  # noqa: E402
from viz.dynamics_rabi import HOLD_FRAMES, make_rabi_animation, rabi_frames  # noqa: E402


def test_rabi_frames():
    frames = rabi_frames(initial_amp=0.1, pi_amp=TRUE_PI_AMPLITUDE)
    guess, true_pi = frames["guess"], frames["true_pi"]

    # Both start from pure |g> at the north pole.
    np.testing.assert_allclose(guess.vectors[0], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(true_pi.vectors[0], [0, 0, 1], atol=1e-9)

    # The true pi amplitude lands on |e>; the 0.1 V starting guess under-rotates.
    assert true_pi.vectors[-1, 2] < -0.99
    assert -0.8 < guess.vectors[-1, 2] < 0.0
    # Phase-0 pulses rotate |g> through -y.
    assert true_pi.vectors[len(true_pi.vectors) // 2, 1] < -0.5
    assert true_pi.populations[-1, 1] > 0.99


# Rendering takes about 9 s, so it runs in CI with the other slow tests, not in the default run.
@pytest.mark.slow
def test_rabi_gif(tmp_path):
    out = make_rabi_animation(tmp_path / "bloch-rabi.gif")
    assert out == tmp_path / "bloch-rabi.gif"
    assert out.stat().st_size < MAX_GIF_BYTES
    with Image.open(out) as gif:
        assert gif.n_frames > 20
        gif.seek(gif.n_frames - 1)
        # Pillow merges the identical held frames into one long final frame.
        assert gif.info["duration"] >= HOLD_FRAMES * 50
