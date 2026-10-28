"""The quasi-static frequency noise model: derived parameters, the shot average, and the echo.

Each shot sees a fixed qubit frequency offset drawn from N(0, sigma_f^2) on top of white Lindblad
dephasing (simulator/device.py, simulator/transmon.py). The shot-averaged Ramsey envelope is then
exp(-t/T2echo - (t/Tq)^2), which a Hahn echo refocuses to exp(-t/T2echo).
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import qutip as qt

from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.qpu import SimulatedQPU
from simulator.transmon import (
    N_LEVELS,
    Idle,
    Pulse,
    Transmon,
    bloch_vector,
    quasistatic_offsets,
    saturation_excited_population,
)

X90 = Pulse(TRUE_PI_AMPLITUDE / 2, 40.0, 8.0)
X180 = Pulse(TRUE_PI_AMPLITUDE, 40.0, 8.0)
GROUND = qt.basis(N_LEVELS, 0).proj()


def test_derived_noise_parameters():
    d = TRUE_DEVICE
    # 1/T2echo = 1/(2 T1) + 1/Tphi_white
    tphi = 1.0 / (1.0 / d.T2echo - 1.0 / (2 * d.T1))
    # E(T2*) = 1/e with E(t) = exp(-t/T2echo - (t/Tq)^2)  =>  Tq = T2* / sqrt(1 - T2*/T2echo)
    tq = d.T2ramsey / math.sqrt(1.0 - d.T2ramsey / d.T2echo)
    # exp(-(2 pi sigma t)^2 / 2) = exp(-(t/Tq)^2)  =>  sigma = sqrt(2) / (2 pi Tq), Tq in s
    sigma = math.sqrt(2) / (2 * math.pi * tq * 1e-9)
    assert d.Tphi_white == pytest.approx(tphi, rel=1e-12)
    assert d.T_quasistatic == pytest.approx(tq, rel=1e-12)
    assert d.sigma_f_quasistatic == pytest.approx(sigma, rel=1e-12)
    # Literal cross-check of the formulas above.
    assert d.Tphi_white == pytest.approx(175_060, rel=1e-4)
    assert d.T_quasistatic == pytest.approx(43_841, rel=1e-4)
    assert d.sigma_f_quasistatic == pytest.approx(5134, rel=1e-3)
    assert d.ramsey_envelope(d.T2ramsey) == pytest.approx(math.exp(-1), rel=1e-9)


def test_no_quasistatic_noise_when_t2star_equals_t2echo():
    d = replace(TRUE_DEVICE, T2ramsey=TRUE_DEVICE.T2echo)
    assert d.sigma_f_quasistatic == 0.0
    offsets, weights = quasistatic_offsets(d)
    assert offsets.tolist() == [0.0]
    assert weights.tolist() == [1.0]
    with pytest.raises(ValueError):
        replace(TRUE_DEVICE, T2ramsey=70_000.0).T_quasistatic


def test_quadrature_reproduces_gaussian_average():
    offsets, weights = quasistatic_offsets(TRUE_DEVICE)
    assert weights.sum() == pytest.approx(1)
    assert np.allclose(offsets, -offsets[::-1])
    tq = TRUE_DEVICE.T_quasistatic
    for t in np.linspace(0, 100_000, 41):
        average = np.sum(weights * np.cos(2 * np.pi * offsets * t * 1e-9))
        assert abs(average - np.exp(-((t / tq) ** 2))) < 1e-4


def test_superoperator_path_matches_mesolve():
    tr = Transmon(TRUE_DEVICE)
    seq = [X90, Idle(10_000.0), replace(X90, phase=1.234)]
    f = TRUE_DEVICE.f_01 + 3e5
    a = tr.evolve_superop(seq, f, f01_offset=4e3).full()
    b = tr.evolve(seq, f, f01_offset=4e3).full()
    # The residual is mesolve's default tolerance in evolve, not the superoperator path: measured
    # 1.18e-6 on the g-f coherence, while the superoperator path agrees with a mesolve at
    # atol 1e-12, rtol 1e-10 to 9e-11. Populations agree to 3e-8.
    assert np.max(np.abs(a - b)) < 2e-6
    assert np.max(np.abs(np.diag(a) - np.diag(b))) < 1e-7


def _equatorial_ratios(device, t):
    tr = Transmon(device)
    f = device.f_01

    def length(rho):
        v = bloch_vector(rho)
        return math.hypot(v[0], v[1])

    start = length(tr.shot_averaged_state([X90], f, rho0=GROUND))
    ramsey = length(tr.shot_averaged_state([X90, Idle(t)], f, rho0=GROUND)) / start
    echo = length(tr.shot_averaged_state([X90, Idle(t / 2), X180, Idle(t / 2)], f, rho0=GROUND)) / start
    return ramsey, echo


@pytest.mark.parametrize("t", [20_000.0, 60_000.0, 120_000.0])
def test_echo_gain_is_emergent(t):
    ramsey, echo = _equatorial_ratios(TRUE_DEVICE, t)
    assert ramsey == pytest.approx(float(TRUE_DEVICE.ramsey_envelope(t)), abs=2e-3)
    assert echo == pytest.approx(math.exp(-t / TRUE_DEVICE.T2echo), abs=2e-3)


def test_no_echo_gain_without_quasistatic_noise():
    # Control: with no quasi-static noise, Ramsey and echo decay alike.
    ramsey, echo = _equatorial_ratios(replace(TRUE_DEVICE, T2ramsey=TRUE_DEVICE.T2echo), 60_000.0)
    assert ramsey == pytest.approx(echo, abs=2e-3)


def test_short_sequences_skip_the_average():
    f = TRUE_DEVICE.f_01
    got = SimulatedQPU().populations([[X180]], f)[:, 0]
    tr = Transmon(TRUE_DEVICE)
    want = tr.populations(tr.evolve([X180], f))
    assert np.array_equal(got, want)


def test_saturation_line_uses_markovian_t2():
    d = TRUE_DEVICE
    amplitude = 2e-3
    f = d.f_01 + np.linspace(-3e6, 3e6, 6001)
    signal = saturation_excited_population(d, f, amplitude) - d.thermal_excited_population
    above = f[signal >= signal.max() / 2]
    fwhm = above.max() - above.min()
    omega = 2 * np.pi * d.rabi_hz_per_volt * amplitude * 1e-9  # rad/ns
    t1, t2 = d.T1, d.T2echo
    expected = math.sqrt(1 + omega**2 * t1 * t2) / (math.pi * t2) * 1e9  # Hz
    assert fwhm == pytest.approx(expected, rel=0.02)
