import pytest

from calibration_utils import power_rabi as analysis
from calibration_utils.common import load_node
from simulator.device import TRUE_PI_AMPLITUDE

node = load_node("04b_power_rabi.py")
ROUGH_PI_AMPLITUDE = 0.1  # state_initial.json starting guess


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_power_rabi_closure(calibrated_state, make_ctx, seed):
    calibrated_state["qubits"]["q1"]["xy"]["operations"]["x180"]["amplitude"] = ROUGH_PI_AMPLITUDE
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert fit["opt_amp"] == pytest.approx(TRUE_PI_AMPLITUDE, rel=0.02)
    assert ctx.qubit["xy"]["operations"]["x180"]["amplitude"] == fit["opt_amp"]


def test_power_rabi_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    calibrated_state["qubits"]["q1"]["xy"]["operations"]["x180"]["amplitude"] = ROUGH_PI_AMPLITUDE
    ctx = make_ctx(calibrated_state)
    node.update_state(ctx, {"success": False, "opt_amp": 1.0})
    assert ctx.qubit["xy"]["operations"]["x180"]["amplitude"] == ROUGH_PI_AMPLITUDE


def test_power_rabi_rejects_noise(calibrated_state, make_ctx):
    """With the drive detuned far from the qubit there is no Rabi oscillation; the fit must fail."""
    calibrated_state["qubits"]["q1"]["f_01"] += 300e6
    ctx = make_ctx(calibrated_state)
    fit = node.run(ctx)
    assert not fit["success"]
    assert fit["refined_from"] is None
    assert ctx.qubit["xy"]["operations"]["x180"]["amplitude"] == TRUE_PI_AMPLITUDE


def test_power_rabi_reports_n_periods(calibrated_state, make_ctx):
    fit = node.run(make_ctx(calibrated_state, seed=0))
    assert fit["n_periods"] >= 0.9
    assert fit["refined_from"] is None


def test_power_rabi_gate_rejects_short_sweep(calibrated_state, make_ctx):
    """A sweep that stops at 1.2 x the pi amplitude covers about 0.6 periods and must be rejected."""
    ctx = make_ctx(calibrated_state, seed=0)
    node.run(ctx)
    ds = ctx.results[node.NODE_NAME]["dataset"].sel(amp_prefactor=slice(0, 1.2))
    fit = analysis.fit_raw_data(ds, TRUE_PI_AMPLITUDE)
    assert fit["gates"]["n_periods"] is False
    assert fit["success"] is False


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_power_rabi_refines_from_rough_guess(calibrated_state, make_ctx, seed):
    """From the 0.1 V starting guess the first sweep covers under 0.75 periods; one refine pass fixes it."""
    calibrated_state["qubits"]["q1"]["xy"]["operations"]["x180"]["amplitude"] = ROUGH_PI_AMPLITUDE
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert fit["refined_from"]["n_periods"] < 0.75
    assert fit["n_periods"] >= 0.75
    assert abs(fit["opt_amp"] / TRUE_PI_AMPLITUDE - 1) < 0.02
    assert ctx.qubit["xy"]["operations"]["x180"]["amplitude"] == fit["opt_amp"]
    assert ctx.results[node.NODE_NAME]["first_pass_dataset"] is not None
    assert analysis.prefactor_vs_state(fit) == pytest.approx(fit["opt_amp"] / ROUGH_PI_AMPLITUDE)
