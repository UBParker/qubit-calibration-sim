import pytest

from calibration_utils.common import load_node
from simulator.device import TRUE_DEVICE

node = load_node("03a_qubit_spectroscopy.py")

ROUGH_F01 = 5.12e9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_qubit_spectroscopy_closure(calibrated_state, make_ctx, seed):
    calibrated_state["qubits"]["q1"]["f_01"] = ROUGH_F01
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert fit["frequency"] == pytest.approx(TRUE_DEVICE.f_01, abs=200e3)
    assert ctx.qubit["f_01"] == fit["frequency"]


def test_qubit_spectroscopy_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    calibrated_state["qubits"]["q1"]["f_01"] = ROUGH_F01
    ctx = make_ctx(calibrated_state)
    node.update_state(ctx, {"success": False, "frequency": 1.0})
    assert ctx.qubit["f_01"] == ROUGH_F01
