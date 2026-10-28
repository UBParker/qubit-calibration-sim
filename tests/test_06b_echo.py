import inspect

import pytest

from calibration_utils.common import load_node
from simulator.device import TRUE_DEVICE
from simulator.qpu import SimulatedQPU
from simulator.transmon import Transmon

node = load_node("06b_echo.py")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_echo_closure(calibrated_state, make_ctx, seed):
    ctx = make_ctx(calibrated_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    assert fit["T2echo"] == pytest.approx(TRUE_DEVICE.T2echo, rel=0.15)
    assert ctx.qubit["T2echo"] == fit["T2echo"]


def test_echo_idle_times_are_valid_qua_waits():
    cycles = node.half_idle_cycles(node.Parameters())
    assert cycles.min() >= 4
    assert (node.idle_times_ns(node.Parameters()) == 2 * 4 * cycles).all()


def test_echo_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    node.update_state(ctx, {"success": False, "T2echo": 1.0})
    assert ctx.qubit["T2echo"] is None


def test_echo_runs_without_a_separate_dephasing_rate():
    """06b calls the QPU as 06a does; the echo gain comes from the sequence, not a second rate."""
    for fn in (SimulatedQPU.run_sequences, SimulatedQPU.populations, Transmon.__init__):
        assert "dephasing" not in inspect.signature(fn).parameters, fn.__qualname__
