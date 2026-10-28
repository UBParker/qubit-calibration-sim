import copy

import numpy as np
import pytest

import calibration_utils.resonator_spectroscopy as rs
from calibration_utils.common import QUBIT, load_node
from simulator.device import TRUE_DEVICE

node = load_node("02a_resonator_spectroscopy.py")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_resonator_spectroscopy_closure(rough_state, make_ctx, seed):
    ctx = make_ctx(rough_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    # A tenth of the resonator linewidth (kappa = 1.4 MHz).
    assert fit["frequency"] == pytest.approx(TRUE_DEVICE.f_res_g, abs=TRUE_DEVICE.kappa / 10)
    assert ctx.qubit["resonator"]["f_01"] == fit["frequency"]


def test_resonator_spectroscopy_failed_fit_leaves_state_untouched(rough_state, make_ctx):
    ctx = make_ctx(rough_state)
    before = ctx.snapshot()
    node.update_state(ctx, {"success": False, "frequency": 1.0})
    assert ctx.state == before


def test_resonator_spectroscopy_no_dip_in_sweep_fails(rough_state, make_ctx):
    # Guess 30 MHz above the resonator: the 20 MHz sweep sees only noise, so the node must refuse.
    rough_state["qubits"]["q1"]["resonator"]["f_01"] = TRUE_DEVICE.f_res_g + 30e6
    ctx = make_ctx(rough_state)
    before = ctx.snapshot()
    fit = node.run(ctx)
    assert not fit["success"]
    assert ctx.state == before


# The 1 us window holds the resonator's ring-up (time constant 2/(2 pi kappa) = 227 ns), so the
# windowed spectrum is not a notch line shape: the notch fit returns kappa about 30 % high and a
# depth of about 0.63 (true 0.8). Bands are the measured seeds 0 to 4 plus or minus 0.03.
KAPPA_RATIO_BAND = (1.27, 1.36)  # measured 1.306 to 1.331
NOTCH_DEPTH_BAND = (0.58, 0.66)  # measured 0.618 to 0.625


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_resonator_kappa_closure(rough_state, make_ctx, seed):
    fit = node.run(make_ctx(rough_state, seed))
    assert KAPPA_RATIO_BAND[0] < fit["kappa"] / TRUE_DEVICE.kappa < KAPPA_RATIO_BAND[1]
    assert NOTCH_DEPTH_BAND[0] < fit["notch_depth"] < NOTCH_DEPTH_BAND[1]


def test_kappa_bias_is_ring_up(rough_state, make_ctx):
    # A 4 us readout leaves little of the ring-up in the window: noiseless 1.061 for |g>, plus about
    # 1 % from the thermal |e> dip.
    rough_state["qubits"]["q1"]["resonator"]["operations"]["readout"]["length"] = 4000
    fit = node.run(make_ctx(rough_state, 1))
    assert 1.02 < fit["kappa"] / TRUE_DEVICE.kappa < 1.10


def test_dip_fwhm_is_not_kappa(rough_state, make_ctx):
    # For a deep notch the |S21| dip is narrower than kappa.
    fit = node.run(make_ctx(rough_state, 0))
    assert fit["fwhm"] < 0.85 * fit["kappa"]


def test_kappa_nan_on_failed_fit(rough_state, make_ctx):
    # Same no-dip setup as test_resonator_spectroscopy_no_dip_in_sweep_fails.
    rough_state["qubits"]["q1"]["resonator"]["f_01"] = TRUE_DEVICE.f_res_g + 30e6
    fit = node.run(make_ctx(rough_state))
    assert not fit["success"]
    assert np.isnan(fit["kappa"])
    assert np.isnan(fit["kappa_error"])
    assert np.isnan(fit["notch_depth"])
    assert fit["notch_params"] == {}


def test_notch_params_reproduce_data(rough_state, make_ctx):
    # notch_params (SI units) drawn with notch_amplitude must match |IQ| to within the noise,
    # estimated from the residual scatter far from the resonance.
    ctx = make_ctx(rough_state, 0)
    fit = node.run(ctx)
    ds = ctx.results[node.NODE_NAME]["dataset"]
    p = fit["notch_params"]
    assert set(p) == {"a", "depth", "center", "kappa"}
    assert p["kappa"] == fit["kappa"] and p["depth"] == fit["notch_depth"]
    x = ds.detuning.values
    residual = ds.IQ_abs.sel(qubit=QUBIT).values - rs.notch_amplitude(x, **p)
    noise = np.std(residual[np.abs(x - p["center"]) > 3 * p["kappa"]])
    assert np.std(residual) < 1.2 * noise
    assert np.max(np.abs(residual)) < 5 * noise


def test_notch_fit_failure_after_lorentzian_success(rough_state, make_ctx, monkeypatch):
    # The Lorentzian fit succeeds but the notch fit raises: the kappa keys are NaN, notch_params
    # is empty and the Lorentzian results, including success, are unchanged.
    reference = node.run(make_ctx(copy.deepcopy(rough_state), 0))
    real_curve_fit = rs.curve_fit

    def failing_for_notch(f, *args, **kwargs):
        if f is rs.notch_amplitude:
            raise RuntimeError("forced notch fit failure")
        return real_curve_fit(f, *args, **kwargs)

    monkeypatch.setattr(rs, "curve_fit", failing_for_notch)
    fit = node.run(make_ctx(rough_state, 0))
    assert fit["success"] and reference["success"]
    for key in ("frequency", "frequency_error", "fwhm", "snr", "fit_params"):
        assert fit[key] == reference[key]
    assert np.isnan(fit["kappa"]) and np.isnan(fit["kappa_error"]) and np.isnan(fit["notch_depth"])
    assert fit["notch_params"] == {}
