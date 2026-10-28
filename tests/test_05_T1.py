import pytest

from calibration_utils.common import load_node
from simulator.device import TRUE_DEVICE

node = load_node("05_T1.py")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_T1_closure(calibrated_state, make_ctx, seed):
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert fit["T1"] == pytest.approx(TRUE_DEVICE.T1, rel=0.10)
    assert ctx.qubit["T1"] == fit["T1"]


def test_T1_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    node.update_state(ctx, {"success": False, "T1": 1.0})
    assert ctx.qubit["T1"] is None
