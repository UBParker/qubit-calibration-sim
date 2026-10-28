"""Time-domain readout (simulator/readout.py, SimulatedQPU.measure_*): the closed-form window sums
against the explicit field and QUA's demodulation formula, noise, time of flight and ADC offsets."""

import copy
import dataclasses
import inspect
import warnings

import numpy as np
import pytest

from calibration_utils.common import readout_settings
from simulator.device import TRUE_DEVICE
from simulator.qpu import SimulatedQPU
from simulator.readout import (
    ADC_RAW_PER_VOLT,
    IQ_DEMOD,
    ROTATED_IQ_DEMOD,
    TWO_PI_NS,
    cavity_field,
    decay_window_sums,
    demod_window,
    expand_weights,
    field_rate,
    steady_field,
)

d = TRUE_DEVICE
RATE = TWO_PI_NS * d.kappa / 2  # field relaxation rate, 1/ns


@pytest.fixture
def ro(calibrated_state):
    return readout_settings(calibrated_state)


@pytest.fixture
def aligned(ro):
    """Window opens when the signal arrives; configured offsets equal the true ones."""
    return ro.replace(time_of_flight=237, input_offsets=tuple(d.adc_offsets))


def window_mean_factor(lam, L):
    """W = 1 - (1 - e^{-λL}) / (L (1 - e^{-λ})): the 1 ns-sampled window mean of b over b_ss."""
    return 1 - (1 - np.exp(-lam * L)) / (L * (1 - np.exp(-lam)))


def test_steady_state_is_the_notch():
    f_r = d.f_res_g
    for df in (-3e6, -0.7e6, 0.0, 0.4e6, 2e6):
        f = f_r + df
        b = cavity_field(np.array([20_000.0]), f, f_r, f_r, np.inf, 1e9, d.kappa)[0]
        want = 1 / (1 + 2j * (f - f_r) / d.kappa)
        assert abs(b - want) < 1e-12
        assert abs(RATE / field_rate(f, f_r, d.kappa) - want) < 1e-12


def test_window_average_closed_form():
    f_r = d.f_res_g
    for L in (500, 1000, 4000):
        theta = np.arange(L, dtype=float)
        lam = field_rate(f_r, f_r, d.kappa)
        mean = cavity_field(theta, f_r, f_r, f_r, np.inf, L, d.kappa).mean()
        assert abs(mean - steady_field(f_r, f_r, d.kappa) * window_mean_factor(lam, L)) < 1e-12


def test_ring_up_and_ring_down_rate():
    f_r = d.f_res_g
    L = 1000
    up = np.arange(10.0, 900.0)
    down = np.arange(1100.0, 1900.0)
    b_up = cavity_field(up, f_r, f_r, f_r, np.inf, L, d.kappa)
    b_down = cavity_field(down, f_r, f_r, f_r, np.inf, L, d.kappa)
    slope_up = np.polyfit(up, np.log(np.abs(b_up - 1.0)), 1)[0]
    slope_down = np.polyfit(down, np.log(np.abs(b_down)), 1)[0]
    assert slope_up == pytest.approx(-RATE, rel=1e-9)
    assert slope_down == pytest.approx(-RATE, rel=1e-9)


def test_decay_is_continuous_and_relaxes_to_g():
    t_d = 300.5
    f = d.f_res_g - d.chi
    # ±1e-10 ns: the smooth slope |db/dt| ~ 2e-3 per ns alone moves b by 4e-13 over that step.
    pair = cavity_field(np.array([t_d - 1e-10, t_d + 1e-10]), f, d.f_res_e, d.f_res_g, t_d, 4000, d.kappa)
    assert abs(pair[1] - pair[0]) < 1e-12
    late = cavity_field(np.array([2000.0]), f, d.f_res_e, d.f_res_g, t_d, 4000, d.kappa)[0]
    assert abs(late - steady_field(f, d.f_res_g, d.kappa)) < 1e-3


def test_decay_window_sums_matches_explicit():
    rng = np.random.default_rng(0)
    decay_times = [0, 0.3, 17.6, 399.5, 500.2, 999.9, 1000, 1100.5, 5000, 1e9, np.inf]
    f = d.f_res_g - 0.3e6
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for theta0 in (0.0, -213.0, 63.0):
            for L, N in ((1000, 1000), (1000, 1200), (400, 1000)):
                K = rng.normal(size=N) + 1j * rng.normal(size=N)
                theta = theta0 + np.arange(N, dtype=float)
                closed = decay_window_sums(K, theta0, f, d.f_res_e, d.f_res_g, d.kappa, L, decay_times)
                explicit = np.array([cavity_field(theta, f, d.f_res_e, d.f_res_g, t, L, d.kappa) @ K for t in decay_times])
                worst = max(worst, float(np.max(np.abs(closed - explicit)) / np.max(np.abs(explicit))))
    assert worst < 1e-10


def test_default_weights_recover_baseband(aligned):
    qpu = SimulatedQPU()
    L = aligned.pulse_length
    for df in (0.0, 0.4e6, 200e6):
        f = d.f_res_g + df
        settings = aligned.with_frequency(f)
        lam = field_rate(f, d.f_res_g, d.kappa)
        b_ss = steady_field(f, d.f_res_g, d.kappa)
        f_if = f - settings.lo_frequency
        want = d.readout_gain * settings.amplitude * (1 - d.kappa_ext_ratio * b_ss * window_mean_factor(lam, L))
        want *= np.exp(-1j * TWO_PI_NS * f_if * d.time_of_flight)
        got = qpu.expected_iq(0, settings, decay=False)
        assert abs(got - want) / abs(want) < 1e-12


def _explicit_outputs(settings, demod, f_res):
    """QM's formula: (1/L) Σ_n (v1[n] P1[n] + v2[n] P2[n]) on traces built from the field."""
    n = max(settings.weights[name][0].size for dm in demod for name, _ in dm)
    theta = settings.time_of_flight - d.time_of_flight + np.arange(n)
    omega = TWO_PI_NS * settings.intermediate_frequency
    b = cavity_field(theta, settings.frequency, f_res, f_res, np.inf, settings.pulse_length, d.kappa)
    u = ((theta >= 0) & (theta < settings.pulse_length)).astype(float)
    s = d.readout_gain * settings.amplitude * (u - d.kappa_ext_ratio * b) * np.exp(1j * omega * theta)
    v = {1: s.real + d.adc_offsets[0] - settings.input_offsets[0], 2: s.imag + d.adc_offsets[1] - settings.input_offsets[1]}
    phi = omega * (settings.time_of_flight + np.arange(n))
    outs = []
    for dm in demod:
        total = 0.0
        for name, analog_input in dm:
            wc, ws = settings.weights[name]
            total += np.sum(v[analog_input] * (wc * np.cos(phi) + ws * np.sin(phi)))
        outs.append(total / settings.pulse_length)
    return np.array(outs)


def test_explicit_demod_formula_matches_kernel(calibrated_state):
    state = copy.deepcopy(calibrated_state)
    state["qubits"]["q1"]["resonator"]["operations"]["readout"]["integration_weights_angle"] = 0.7
    settings = readout_settings(state).with_frequency(d.f_res_g - 0.2e6)  # tof 236, offsets as configured
    for demod in (IQ_DEMOD, ROTATED_IQ_DEMOD):
        window = demod_window(settings, d, demod)
        for f_res in (d.f_res_g, d.f_res_e):
            got = window.static_outputs(f_res)
            want = _explicit_outputs(settings, demod, f_res)
            assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


def test_rotated_weights_rotate_iq(calibrated_state):
    qpu = SimulatedQPU()
    for a in (-2.1, 0.3, 1.2):
        state = copy.deepcopy(calibrated_state)
        state["qubits"]["q1"]["resonator"]["operations"]["readout"]["integration_weights_angle"] = a
        settings = readout_settings(state)
        for s in (0, 1):
            plain = qpu.expected_iq(s, settings, IQ_DEMOD)
            rotated = qpu.expected_iq(s, settings, ROTATED_IQ_DEMOD)
            assert abs(rotated - plain * np.exp(-1j * a)) < 1e-12 * abs(plain)


def test_weights_expand_in_4ns_groups():
    assert np.array_equal(expand_weights([(1.0, 8), (-0.5, 4)], 16), [1] * 8 + [-0.5] * 4 + [0] * 4)
    assert np.array_equal(expand_weights([0.2, 0.4], 8), [0.2] * 4 + [0.4] * 4)
    with pytest.raises(ValueError):
        expand_weights([(1.0, 6)], 8)


def test_noise_statistics(calibrated_state, ro):
    device = dataclasses.replace(TRUE_DEVICE, adc_sample_noise=0.038, thermal_excited_population=0)
    qpu = SimulatedQPU(device, seed=3)
    sigma = 0.038 / np.sqrt(1000)
    I, Q = qpu.measure_shots(np.array([1.0, 0, 0]), ro, 40_000)
    assert np.std(I) == pytest.approx(sigma, rel=0.02)
    assert np.std(Q) == pytest.approx(sigma, rel=0.02)
    assert abs(np.corrcoef(I, Q)[0, 1]) < 0.02
    short_state = copy.deepcopy(calibrated_state)
    short_state["qubits"]["q1"]["resonator"]["operations"]["readout"]["length"] = 248  # on the 4 ns grid
    I_short, _ = qpu.measure_shots(np.array([1.0, 0, 0]), readout_settings(short_state), 40_000)
    assert np.std(I_short) / np.std(I) == pytest.approx(np.sqrt(1000 / 248), abs=0.05)
    pops = np.tile([[1.0], [0.0], [0.0]], (1, 2000))
    I_avg, _ = qpu.measure_averaged(pops, ro, 400)
    assert np.std(I_avg) == pytest.approx(sigma / 20, rel=0.03)


def test_trace_path_matches_shot_path(ro):
    """3000 shots materialised sample by sample with noise and demodulated with QM's formula, against
    3000 shots of measure_shots, for prepared |e> (decay in the window included)."""
    rng = np.random.default_rng(11)
    n_shots, chunk = 3000, 500
    n = ro.pulse_length
    theta = ro.time_of_flight - d.time_of_flight + np.arange(n)
    omega = TWO_PI_NS * ro.intermediate_frequency
    phi = omega * (ro.time_of_flight + np.arange(n))
    carrier = np.exp(1j * omega * theta)
    u = ((theta >= 0) & (theta < n)).astype(float)
    ga = d.readout_gain * ro.amplitude
    weights = {name: wc * np.cos(phi) + ws * np.sin(phi) for name, (wc, ws) in ro.weights.items()}
    offsets = np.array(d.adc_offsets) - np.array(ro.input_offsets)
    t_decay = rng.exponential(d.T1, n_shots)
    out = np.empty((n_shots, 2))
    for start in range(0, n_shots, chunk):
        idx = range(start, start + chunk)
        s = np.array([ga * (u - d.kappa_ext_ratio * cavity_field(theta, ro.frequency, d.f_res_e, d.f_res_g, t_decay[i], n, d.kappa)) * carrier for i in idx])
        v1 = s.real + offsets[0] + rng.normal(0, d.adc_sample_noise, s.shape)
        v2 = s.imag + offsets[1] + rng.normal(0, d.adc_sample_noise, s.shape)
        for k, dm in enumerate(IQ_DEMOD):
            out[start : start + chunk, k] = sum((v1 if inp == 1 else v2) @ weights[name] for name, inp in dm) / n
    I, Q = SimulatedQPU(seed=5).measure_shots(np.array([0.0, 1.0, 0.0]), ro, n_shots)
    for k, shots in enumerate((I, Q)):
        se = np.hypot(np.std(out[:, k]), np.std(shots)) / np.sqrt(n_shots)
        assert abs(np.mean(out[:, k]) - np.mean(shots)) < 3 * se
        assert np.std(shots) == pytest.approx(np.std(out[:, k]), rel=0.05)


def test_time_of_flight_misalignment(aligned):
    qpu = SimulatedQPU()
    settings = aligned.with_frequency(d.f_res_g + 200e6)
    ratio = abs(qpu.expected_iq(0, settings.replace(time_of_flight=24), decay=False)) / abs(qpu.expected_iq(0, settings, decay=False))
    assert ratio == pytest.approx((1000 - 213) / 1000, abs=2e-3)


def test_adc_offset_barely_reaches_demod(aligned):
    qpu = SimulatedQPU()
    exact = qpu.expected_iq(0, aligned, decay=False)
    unset = qpu.expected_iq(0, aligned.replace(input_offsets=(0.0, 0.0)), decay=False)
    assert abs(unset - exact) < 1e-3 * abs(exact)


def test_averaged_e_includes_decay(ro):
    quiet = SimulatedQPU(dataclasses.replace(TRUE_DEVICE, adc_sample_noise=0.0), seed=8)
    I, Q = quiet.measure_shots(np.array([0.0, 1.0, 0.0]), ro, 200_000)
    want = quiet.expected_iq(1, ro, decay=True)
    assert abs(np.mean(I) - want.real) < 3 * np.std(I) / np.sqrt(I.size)
    assert abs(np.mean(Q) - want.imag) < 3 * np.std(Q) / np.sqrt(Q.size)
    g, e_static = quiet.expected_iq(0, ro), quiet.expected_iq(1, ro, decay=False)
    assert 0.008 <= abs(want - e_static) / abs(e_static - g) <= 0.013


def test_adc_trace_units_and_window(ro):
    settings = ro.replace(time_of_flight=24, amplitude=0.1, input_offsets=(0.0, 0.0)).with_frequency(d.f_res_g + 5e6)
    trace = SimulatedQPU(seed=4).adc_trace(settings, n_avg=1000)
    assert trace.averaged.shape == trace.single_run.shape == (2, 1000)
    assert trace.single_run.dtype == np.int16
    assert trace.single_run.min() >= -2048 and trace.single_run.max() <= 2047
    volts = -trace.averaged / ADC_RAW_PER_VOLT
    se = d.adc_sample_noise / np.sqrt(1000 * 200)
    for k in (0, 1):
        assert abs(volts[k, :200].mean() - d.adc_offsets[k]) < 4 * se
    z = volts[0] + 1j * volts[1]
    rms = lambda x: np.sqrt(np.mean(np.abs(x) ** 2))  # noqa: E731
    assert rms(z[213:]) >= 5 * rms(z[:200])


def test_qpu_has_no_old_readout():
    assert not hasattr(SimulatedQPU, "_state_responses")
    assert "readout_frequency" not in inspect.signature(SimulatedQPU.run_sequences).parameters
    assert not hasattr(TRUE_DEVICE, "adc_noise")
