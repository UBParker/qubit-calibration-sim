"""Shared fixtures.

``calibrated_state`` holds the true device values for everything a node's prerequisites would
have calibrated, so each node's closure test runs in isolation. ``initial_state`` is
state_initial.json as committed (time of flight 24 ns, zero ADC offsets), what the 01a setup step
starts from. ``rough_state`` is the graph's starting point: state_initial.json after the 01a setup
step (``after_setup``), as calibrations/90_calibration_graph_bringup.py run_bringup leaves it.
"""

import copy
import logging

import matplotlib.pyplot as plt
import pytest

from calibration_utils.common import INITIAL_STATE_PATH, NodeContext, load_state
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.qpu import SimulatedQPU


logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def _close_stray_figures():
    # A test that fails before its own plt.close leaks a figure; close it here so tests that
    # assert no open figures do not depend on test order.
    plt.close("all")
    yield


def apply_setup(q: dict) -> None:
    """The 01a setup values as a calibrated state holds them: time of flight on the 4 ns grid just
    below the true 237 ns, ADC offsets exact."""
    q["resonator"]["time_of_flight"] = 236
    q["resonator"]["opx_input_I"] = {"offset": TRUE_DEVICE.adc_offsets[0]}
    q["resonator"]["opx_input_Q"] = {"offset": TRUE_DEVICE.adc_offsets[1]}


def after_setup(state: dict, seed: int = 0) -> dict:
    """``state`` after the 01a setup step on a fresh simulator (seed 0: time of flight 240 ns,
    offsets within 0.4 mV of the truth). Returns a new state."""
    from calibration_utils.common import load_node

    ctx = NodeContext(state=copy.deepcopy(state), qpu=SimulatedQPU(seed=seed), save_outputs=False)
    fit = load_node("01a_time_of_flight.py").run(ctx)
    assert fit["success"]
    return ctx.state


@pytest.fixture
def initial_state() -> dict:
    return load_state(INITIAL_STATE_PATH)


@pytest.fixture
def rough_state(initial_state) -> dict:
    return after_setup(initial_state)


@pytest.fixture
def calibrated_state(rough_state) -> dict:
    state = copy.deepcopy(rough_state)
    q = state["qubits"]["q1"]
    q["f_01"] = TRUE_DEVICE.f_01
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    return state


@pytest.fixture
def make_ctx():
    def _make(state: dict, seed: int = 1234) -> NodeContext:
        return NodeContext(state=state, qpu=SimulatedQPU(seed=seed), save_outputs=False)

    return _make


@pytest.fixture(scope="session")
def graph_run_2026() -> NodeContext:
    """The full bring-up graph run once per session: seed 2026 from the rough state.

    Built as the 90 graph's ``__main__`` builds it (setup step 01a, then the graph) (``ctx.reference = build_reference(state)`` on
    the rough state before the graph runs), but with ``save_outputs=False``, so nothing is written:
    nodes skip ``export_qua`` and ``save_figure``, and ``run_graph`` itself never calls
    ``save_state`` (only the script's ``__main__`` does). Shared by every test that asks for it:
    tests that mutate the context, its state, results or history must ``copy.deepcopy`` it first.
    """
    from calibration_utils.common import load_node
    from simulator.reference import build_reference

    graph = load_node("90_calibration_graph_bringup.py")
    ctx = NodeContext(state=load_state(INITIAL_STATE_PATH), qpu=SimulatedQPU(seed=2026), save_outputs=False)
    ctx.reference = build_reference(ctx.state)
    graph.run_bringup(ctx)  # the 01a setup step, then the graph
    return ctx


def _strip_timestamp(script: str) -> str:
    return "\n".join(line for line in script.split("\n") if not line.startswith("# Single QUA script generated at"))


def assert_export_uses_pre_update_state(node_module, ctx: NodeContext, params=None) -> dict:
    """Run ``node_module.run(ctx, params)`` and check its QUA export used the state from before the run.

    ``export_qua`` only runs when ``ctx.save_outputs`` is set, so the run sets it for its duration
    with ``export_qua`` swapped for a recorder and ``save_figure`` for a no-op: nothing is written.
    Asserts one export, with the node's name, a state equal to ``ctx.snapshot()`` taken before the
    run, and a program whose generated script equals one built by ``create_qua_program`` from that
    snapshot. Returns the fit. Usage in a node test::

        from tests.conftest import assert_export_uses_pre_update_state
        assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(rough_state), 0))
    """
    import calibration_utils.common as common
    from qm import QuantumMachinesManager, generate_qua_script

    from qua.config import build_config

    before = ctx.snapshot()
    params = params if params is not None else node_module.Parameters()
    captured = []

    def record_export(node_name, program, state):
        captured.append((node_name, program, copy.deepcopy(state)))

    saved = ctx.save_outputs
    with pytest.MonkeyPatch.context() as mp:
        for target in (node_module, common):
            if hasattr(target, "export_qua"):
                mp.setattr(target, "export_qua", record_export)
            if hasattr(target, "save_figure"):
                mp.setattr(target, "save_figure", lambda *args, **kwargs: [])
        ctx.save_outputs = True
        try:
            fit = node_module.run(ctx, params)
        finally:
            ctx.save_outputs = saved

    assert len(captured) == 1, f"{node_module.NODE_NAME} exported {len(captured)} programs"
    name, program, state = captured[0]
    assert name == node_module.NODE_NAME
    assert state == before, f"{node_module.NODE_NAME} exported a state that differs from the state before the run"
    QuantumMachinesManager.set_capabilities_offline()
    config = build_config(before)
    got = _strip_timestamp(generate_qua_script(program, config))
    want = _strip_timestamp(generate_qua_script(node_module.create_qua_program(before, params), config))
    assert got == want, f"{node_module.NODE_NAME} exported a program not built from the state before the run"
    return fit
