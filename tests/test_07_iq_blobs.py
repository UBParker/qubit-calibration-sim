import numpy as np
import pytest

from calibration_utils.common import load_node
from calibration_utils.iq_blobs import demod_to_volts

node = load_node("07_iq_blobs.py")


def hardware_fidelity(ds, readout: dict) -> float:
    """Assign shots the way the OPX would with the stored state: demodulate with the rotated
    weights of qua.config.build_config (I_rot = I cos a + Q sin a), then compare with the threshold."""
    a = readout["integration_weights_angle"]
    threshold = demod_to_volts(readout["threshold"], readout["length"])
    d = ds.sel(qubit="q1")
    Ig_rot = d.I_g.values * np.cos(a) + d.Q_g.values * np.sin(a)
    Ie_rot = d.I_e.values * np.cos(a) + d.Q_e.values * np.sin(a)
    return 0.5 * (np.mean(Ig_rot < threshold) + np.mean(Ie_rot > threshold))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_iq_blobs_closure(calibrated_state, make_ctx, seed):
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert 0.95 < fit["fidelity"] < 0.995
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    assert gg + ge == pytest.approx(1) and eg + ee == pytest.approx(1)
    assert fit["fidelity"] == pytest.approx((gg + ee) / 2)

    readout = ctx.qubit["resonator"]["operations"]["readout"]
    assert ctx.qubit["resonator"]["readout_fidelity"] == fit["fidelity"]
    ds = ctx.results[node.NODE_NAME]["dataset"]
    assert hardware_fidelity(ds, readout) == pytest.approx(fit["fidelity"], abs=1e-3)


def test_iq_blobs_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    fit = {"success": False, "iw_angle": 1.0, "ge_threshold": 1e-3, "fidelity": 0.5}
    node.update_state(ctx, fit)
    readout = ctx.qubit["resonator"]["operations"]["readout"]
    assert readout["integration_weights_angle"] == 0.0
    assert readout["threshold"] is None
    assert ctx.qubit["resonator"]["readout_fidelity"] is None


@pytest.mark.parametrize("seed", [1, 2])
def test_threshold_on_rotated_demod_output(calibrated_state, make_ctx, seed):
    """Shots demodulated by the simulator with the stored rotated weights (qua.config) and compared
    with the stored threshold, as the OPX does with readout_iq(rotated=True), reproduce the fidelity."""
    from calibration_utils.common import readout_settings, x180_pulse
    from simulator.qpu import SimulatedQPU
    from simulator.readout import ROTATED_IQ_DEMOD

    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    readout = ctx.qubit["resonator"]["operations"]["readout"]
    threshold = demod_to_volts(readout["threshold"], readout["length"])
    qpu = SimulatedQPU(seed=seed)
    shots = {"drive_frequency": ctx.qubit["f_01"], "readout": readout_settings(ctx.state), "n_shots": 4000, "demod": ROTATED_IQ_DEMOD}
    Ig_rot, _ = qpu.single_shots([], **shots)
    Ie_rot, _ = qpu.single_shots([x180_pulse(ctx.state)], **shots)
    hardware = 0.5 * (np.mean(Ig_rot < threshold) + np.mean(Ie_rot > threshold))
    assert hardware == pytest.approx(fit["fidelity"], abs=0.01)
