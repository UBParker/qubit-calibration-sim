import numpy as np
import pytest
import qutip as qt

from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.transmon import N_LEVELS, Idle, Pulse, Transmon, bloch_vector, level_populations

GROUND = qt.basis(N_LEVELS, 0).proj()


def pi_pulse(amplitude: float = TRUE_PI_AMPLITUDE) -> Pulse:
    return Pulse(amplitude=amplitude, length=40.0, sigma=8.0)


def test_trajectory_endpoint_matches_evolve():
    tr = Transmon(TRUE_DEVICE)
    segments = [pi_pulse(), Idle(1000.0)]
    times, states = tr.trajectory(segments, TRUE_DEVICE.f_01, samples_per_segment=30, rho0=GROUND)
    expected = tr.evolve(segments, TRUE_DEVICE.f_01, rho0=GROUND)
    assert len(times) == len(states) == 1 + 2 * 30
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(1040.0)
    assert np.all(np.diff(times) > 0)
    assert np.max(np.abs(states[-1].full() - expected.full())) < 1e-6


def test_pi_pulse_ends_at_south_pole():
    tr = Transmon(TRUE_DEVICE)
    _, states = tr.trajectory([pi_pulse()], TRUE_DEVICE.f_01, rho0=GROUND)
    assert bloch_vector(states[0])[2] == pytest.approx(1.0)
    assert bloch_vector(states[-1])[2] < -0.99


def test_x90_lands_on_equator():
    tr = Transmon(TRUE_DEVICE)
    _, states = tr.trajectory([pi_pulse(TRUE_PI_AMPLITUDE / 2)], TRUE_DEVICE.f_01, rho0=GROUND)
    v = bloch_vector(states[-1])
    assert abs(v[2]) < 0.02
    assert np.linalg.norm(v) > 0.98


def test_populations_sum_to_one():
    tr = Transmon(TRUE_DEVICE)
    _, states = tr.trajectory([pi_pulse(), Idle(500.0)], TRUE_DEVICE.f_01)
    for rho in states:
        p = level_populations(rho)
        assert p.shape == (3,)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(p > -1e-9)


def test_bloch_vector_convention():
    g, e = qt.basis(N_LEVELS, 0), qt.basis(N_LEVELS, 1)
    plus_x = (g + e).unit()
    plus_y = (g + 1j * e).unit()
    np.testing.assert_allclose(bloch_vector(g.proj()), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(bloch_vector(e.proj()), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(bloch_vector(plus_x.proj()), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(bloch_vector(plus_y.proj()), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(bloch_vector(qt.basis(N_LEVELS, 2).proj()), [0, 0, 0], atol=1e-12)
