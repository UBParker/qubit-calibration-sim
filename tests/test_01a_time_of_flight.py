"""01a time of flight (setup step): closure from the rough 24 ns, QM's two update branches, reruns,
temporary overrides, raw units, and the case of a window that opens after the signal arrives.

Measured over seeds 0 to 199 from 24 ns: stored 236 ns in 74 runs and 240 ns in 126 (true 237), edge
212 to 216 ns into the window (true 213), offsets within 0.40 mV of the truth; a rerun on the result
moves the time of flight by 0 (190 runs) or 4 ns (10).
"""

import copy

import numpy as np
import pytest

from calibration_utils import time_of_flight as analysis
from calibration_utils.common import QUBIT, load_node
from qua.config import build_config
from simulator.device import TRUE_DEVICE

node = load_node("01a_time_of_flight.py")


@pytest.fixture
def setup_state(initial_state):
    res = initial_state["qubits"][QUBIT]["resonator"]
    assert res["time_of_flight"] == 24 and res["opx_input_I"]["offset"] == 0.0  # state_initial.json
    return initial_state


def _resonator(ctx):
    return ctx.qubit["resonator"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_time_of_flight_closure(setup_state, make_ctx, seed):
    ctx = make_ctx(setup_state, seed)
    fit = node.run(ctx)
    assert fit["success"]
    res = _resonator(ctx)
    assert res["time_of_flight"] % 4 == 0
    assert abs(res["time_of_flight"] - TRUE_DEVICE.time_of_flight) <= 4
    assert abs(res["opx_input_I"]["offset"] - TRUE_DEVICE.adc_offsets[0]) < 0.5e-3
    assert abs(res["opx_input_Q"]["offset"] - TRUE_DEVICE.adc_offsets[1]) < 0.5e-3


def test_explicit_start_branch(setup_state, make_ctx):
    params = node.Parameters(time_of_flight_in_ns=100)
    ctx = make_ctx(setup_state, 4)
    fit = node.run(ctx, params)
    assert fit["success"]
    assert _resonator(ctx)["time_of_flight"] == 100 + fit["tof_to_add"]
    assert abs(_resonator(ctx)["time_of_flight"] - TRUE_DEVICE.time_of_flight) <= 4


def test_rerun_on_calibrated_state(setup_state, make_ctx):
    ctx = make_ctx(setup_state, 5)
    node.run(ctx)
    first = copy.deepcopy(_resonator(ctx))
    node.run(ctx)
    second = _resonator(ctx)
    assert second["time_of_flight"] - first["time_of_flight"] in (0, 4, 8)
    for key in ("opx_input_I", "opx_input_Q"):
        assert abs(second[key]["offset"] - first[key]["offset"]) < 0.5e-3


def test_temporary_overrides_are_reverted(setup_state, make_ctx):
    """The readout length and amplitude the node plays with never persist; the export (the
    program and config it ran with) has them."""
    import calibration_utils.common as common

    ctx = make_ctx(setup_state, 6)
    before = ctx.snapshot()
    readout_before = copy.deepcopy(before["qubits"][QUBIT]["resonator"]["operations"]["readout"])
    captured = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(node, "export_qua", lambda name, prog, state: captured.append(copy.deepcopy(state)))
        mp.setattr(node, "save_figure", lambda *args, **kwargs: [])
        mp.setattr(common, "save_figure", lambda *args, **kwargs: [])
        ctx.save_outputs = True
        node.run(ctx)
        ctx.save_outputs = False
    assert _resonator(ctx)["operations"]["readout"] == readout_before
    (exported,) = captured
    assert exported == node.patched_state(before, node.Parameters())
    readout = exported["qubits"][QUBIT]["resonator"]["operations"]["readout"]
    assert readout["length"] == 1000 and readout["amplitude"] == 0.1
    assert build_config(exported)["pulses"]["readout_pulse"]["length"] == 1000


def test_failed_fit_leaves_state_untouched(setup_state, make_ctx):
    ctx = make_ctx(setup_state)
    before = ctx.snapshot()
    node.update_state(ctx, node.Parameters(), {"success": False, "tof_to_add": 200.0, "offset_I_to_add": 1.0, "offset_Q_to_add": 1.0})
    assert ctx.state == before


def test_raw_units(setup_state, make_ctx):
    ctx = make_ctx(setup_state, 7)
    raw = node.run_simulated(ctx, node.Parameters())
    single = raw.adc_single_runI.values
    assert np.issubdtype(single.dtype, np.integer)
    assert single.min() >= -2048 and single.max() <= 2047
    volts = analysis.process_raw_dataset(raw)
    np.testing.assert_array_equal(volts.adcI.values, -raw.adcI.values / 2**12)
    np.testing.assert_array_equal(volts.adc_single_runQ.values, -raw.adc_single_runQ.values / 2**12)


def test_late_window_cannot_find_edge(setup_state, make_ctx):
    """A window that opens after the signal arrived has no edge in it: QM's detector adds 0 or 4 ns
    and the stored value stays at least the late start, so run 01a from a time of flight at or below
    the true delay."""
    setup_state["qubits"][QUBIT]["resonator"]["time_of_flight"] = 300
    ctx = make_ctx(setup_state, 8)
    fit = node.run(ctx)
    assert fit["tof_to_add"] in (0, 4)
    assert _resonator(ctx)["time_of_flight"] >= 300


@pytest.mark.slow
def test_qm_code_default_amplitude_off_with_success_flag(setup_state):
    """At QM's 0.03 V code default, 100 averages and this simulator's assumed ADC noise, the
    stored time of flight is more than 4 ns from the true 237 ns with the success flag set.
    Measured over seeds 0 to 49: 50 of 50; stored values 24 to 136 ns. The
    README states this count."""
    from calibration_utils.common import NodeContext
    from simulator.qpu import SimulatedQPU

    off_with_success = 0
    for seed in range(50):
        ctx = NodeContext(state=copy.deepcopy(setup_state), qpu=SimulatedQPU(seed=seed), save_outputs=False)
        fit = node.run(ctx, node.Parameters(readout_amplitude_in_v=0.03))
        tof = _resonator(ctx)["time_of_flight"]
        if fit["success"] and abs(tof - TRUE_DEVICE.time_of_flight) > 4:
            off_with_success += 1
    assert off_with_success == 50
