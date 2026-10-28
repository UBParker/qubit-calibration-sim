import copy

import numpy as np
import pytest
import xarray as xr

from calibration_utils import ramsey
from calibration_utils.common import INITIAL_STATE_PATH, NodeContext, load_node, load_state, x90_pulse
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.qpu import SimulatedQPU
from simulator.transmon import Idle
from tests.conftest import apply_setup

node = load_node("06a_ramsey.py")

SEEDS = (1, 2)
F_01_ERRORS = (+150e3, -150e3)

# Relative bias of the 06a exponential fit, 1/decay / T2ramsey - 1, on noise-free data. The device's
# shot-averaged envelope is exp(-t/T2E - (t/Tq)^2) and T2ramsey is its 1/e time, so the exponential
# fit cannot match it: over the default 50 us window it reads low, over a 25 us window high.
# Measured -9.14 % and +15.53 % (prototype -9.1 % and +15.5 %); each band is the measured value
# +- 1.5 percentage points, kept inside the spec's (-10.5 %, -7.5 %) and (+13 %, +18 %).
BIAS_50US_BAND = (-0.105, -0.076)
BIAS_25US_BAND = (0.140, 0.170)


def _calibrated_state() -> dict:
    """The tests/conftest.py ``calibrated_state`` fixture, as a plain function for a module fixture."""
    state = copy.deepcopy(load_state(INITIAL_STATE_PATH))
    q = state["qubits"]["q1"]
    q["f_01"] = TRUE_DEVICE.f_01
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    return state


@pytest.fixture(scope="module")
def ramsey_runs() -> dict:
    """One 06a run per (seed, f_01 error), shared by the closure and bias tests."""
    runs = {}
    for seed in SEEDS:
        for err in F_01_ERRORS:
            state = _calibrated_state()
            state["qubits"]["q1"]["f_01"] = TRUE_DEVICE.f_01 + err
            ctx = NodeContext(state=state, qpu=SimulatedQPU(seed=seed), save_outputs=False)
            runs[(seed, err)] = (node.run(ctx), ctx)
    return runs


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("f_01_error", F_01_ERRORS)
def test_ramsey_closure(ramsey_runs, seed, f_01_error):
    fit, ctx = ramsey_runs[(seed, f_01_error)]
    assert fit["success"]
    # The sign matters: f_drive - f_01 must come out as the error we put in, not its mirror image.
    assert fit["freq_offset"] == pytest.approx(f_01_error, abs=20e3)
    assert ctx.qubit["f_01"] == pytest.approx(TRUE_DEVICE.f_01, abs=20e3)
    assert fit["T2ramsey"] == pytest.approx(TRUE_DEVICE.T2ramsey, rel=0.15)
    assert ctx.qubit["T2ramsey"] == fit["T2ramsey"]


def test_ramsey_t2star_biased_low_by_the_model_mismatch(ramsey_runs):
    """The mismatch the closure tolerance absorbs: the exponential fit to the non-exponential
    envelope reads T2* low in every run, by more than twice its own statistical error. The bias is
    real, not noise."""
    for (seed, err), (fit, _) in ramsey_runs.items():
        bias = fit["T2ramsey"] / TRUE_DEVICE.T2ramsey - 1
        assert -0.14 <= bias <= -0.04, (seed, err, bias)
        assert TRUE_DEVICE.T2ramsey - fit["T2ramsey"] > 2 * fit["T2ramsey_error"], (seed, err)


def _noise_free_bias(params) -> float:
    """06a's + sign sequences on the true device, noise-free populations, QM's fit on P(e)."""
    times = node.idle_times_ns(params)
    detuning = params.frequency_detuning_in_mhz * 1e6
    state = _calibrated_state()
    sequences = [[x90_pulse(state), Idle(float(t)), x90_pulse(state, phase=-2 * np.pi * detuning * t * 1e-9)] for t in times]
    p_e = SimulatedQPU().populations(sequences, TRUE_DEVICE.f_01)[1]
    p = ramsey._fit_one_sign(xr.DataArray(p_e, dims="idle_time", coords={"idle_time": times}))
    return 1 / p["decay"] / TRUE_DEVICE.T2ramsey - 1


def test_exponential_fit_bias_noise_free():
    low, high = BIAS_50US_BAND
    assert low <= _noise_free_bias(node.Parameters()) <= high
    low, high = BIAS_25US_BAND
    assert low <= _noise_free_bias(node.Parameters(max_wait_time_ns=25_000)) <= high


def test_ramsey_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    f_01 = ctx.qubit["f_01"]
    node.update_state(ctx, {"success": False, "freq_offset": 1e6, "T2ramsey": 1.0})
    assert ctx.qubit["f_01"] == f_01
    assert ctx.qubit["T2ramsey"] is None
