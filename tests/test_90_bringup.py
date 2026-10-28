"""End-to-end closure test: rough guesses in, the full bring-up graph, the true device out."""

import copy

import pytest

from calibration_utils.common import load_node
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.reference import CLOSURE_TOLERANCES as TOL

graph = load_node("90_calibration_graph_bringup.py")


def test_bringup_recovers_true_device(rough_state, make_ctx):
    ctx = make_ctx(rough_state, 7)
    _assert_recovers_true_device(ctx, graph.run_graph(ctx))


def test_bringup_recovers_true_device_seed_2026(graph_run_2026):
    # The shared session run (tests/conftest.py); read only, so no copy.
    ctx = graph_run_2026
    _assert_recovers_true_device(ctx, {record.name: record.fit for record in ctx.history})


def _assert_recovers_true_device(ctx, outcomes):
    assert list(outcomes) == [name.removesuffix(".py") for name in graph.NODES]
    assert all(fit["success"] for fit in outcomes.values()), {k: v["success"] for k, v in outcomes.items()}

    q = ctx.qubit
    low, high = TOL["fidelity"]["window"]
    # 08a reads out at the g/e midpoint plus QM's trailing-window offset; 01a ran before the graph.
    assert q["resonator"]["f_01"] == pytest.approx(TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi, abs=TOL["readout.frequency"]["abs"])
    assert q["chi"] == pytest.approx(-TRUE_DEVICE.chi, abs=TOL["chi"]["abs"])
    assert q["resonator"]["time_of_flight"] == pytest.approx(TRUE_DEVICE.time_of_flight, abs=TOL["time_of_flight"]["abs"])
    for key, true in zip(("opx_input_I", "opx_input_Q"), TRUE_DEVICE.adc_offsets):
        assert q["resonator"][key]["offset"] == pytest.approx(true, abs=TOL["adc_offset"]["abs"])
    assert q["f_01"] == pytest.approx(TRUE_DEVICE.f_01, abs=TOL["f_01"]["abs"])  # after the Ramsey correction
    assert q["xy"]["operations"]["x180"]["amplitude"] == pytest.approx(
        TRUE_PI_AMPLITUDE, rel=TOL["x180.amplitude"]["rel"]
    )
    assert low < q["resonator"]["readout_fidelity"] < high
    assert q["T1"] == pytest.approx(TRUE_DEVICE.T1, rel=TOL["T1"]["rel"])
    assert q["T2ramsey"] == pytest.approx(TRUE_DEVICE.T2ramsey, rel=TOL["T2ramsey"]["rel"])
    assert q["T2echo"] == pytest.approx(TRUE_DEVICE.T2echo, rel=TOL["T2echo"]["rel"])


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(30))
def test_bringup_succeeds_across_seeds(rough_state, make_ctx, seed):
    ctx = make_ctx(rough_state, seed)
    outcomes = graph.run_graph(ctx)

    assert list(outcomes) == [name.removesuffix(".py") for name in graph.NODES]
    assert all(fit["success"] for fit in outcomes.values()), {k: v["success"] for k, v in outcomes.items()}


def test_graph_stops_at_first_failure(rough_state, make_ctx, monkeypatch):
    ctx = make_ctx(rough_state)
    calls = []

    class FailingNode:
        NODE_NAME = "fails"

        @staticmethod
        def run(ctx):
            calls.append("fails")
            return {"success": False}

    class NeverRuns:
        NODE_NAME = "never"

        @staticmethod
        def run(ctx):
            calls.append("never")
            return {"success": True}

    modules = {"a.py": FailingNode, "b.py": NeverRuns}
    monkeypatch.setattr(graph, "load_node", lambda filename: modules[filename])
    outcomes = graph.run_graph(ctx, ["a.py", "b.py"])
    assert calls == ["fails"]
    assert list(outcomes) == ["fails"]


def test_subset_run_sets_graph_nodes_to_requested_nodes(rough_state, make_ctx):
    ctx = make_ctx(rough_state)
    graph.run_graph(ctx, ["02a_resonator_spectroscopy.py", "03a_qubit_spectroscopy.py"])
    assert ctx.graph_nodes == ["02a_resonator_spectroscopy", "03a_qubit_spectroscopy"]


def test_full_run_sets_graph_nodes_to_all_eight(graph_run_2026):
    assert graph_run_2026.graph_nodes == [name.removesuffix(".py") for name in graph.NODES]
    assert len(graph_run_2026.graph_nodes) == 8


def test_history_records_each_node(rough_state, make_ctx, graph_run_2026):
    import xarray as xr

    initial = copy.deepcopy(rough_state)
    assert make_ctx(rough_state).history == []
    # The shared seed-2026 run, deep-copied because the end of this test mutates the state.
    ctx = copy.deepcopy(graph_run_2026)

    names = [name.removesuffix(".py") for name in graph.NODES]
    assert [record.name for record in ctx.history] == names
    assert len(ctx.history) == 8
    # The graph starts from state_initial.json after the 01a setup step of this run (seed 2026).
    first = ctx.history[0].before["qubits"]["q1"]["resonator"]
    for key in ("time_of_flight", "opx_input_I", "opx_input_Q"):
        initial["qubits"]["q1"]["resonator"][key] = copy.deepcopy(first[key])
    assert ctx.history[0].before == initial
    for previous, record in zip(ctx.history, ctx.history[1:]):
        assert record.before == previous.after
    assert ctx.history[-1].after == ctx.state
    for record in ctx.history:
        assert record.success is True
        assert record.fit is ctx.results[record.name]["fit"]
        assert isinstance(record.dataset, xr.Dataset)
        assert record.before is not ctx.state and record.after is not ctx.state
    # Snapshots are deep copies: later mutation of the live state must not reach them.
    ctx.state["qubits"]["q1"]["f_01"] = -1.0
    assert ctx.history[-1].after["qubits"]["q1"]["f_01"] != -1.0


def test_history_stops_with_failure(rough_state, make_ctx, monkeypatch):
    ctx = make_ctx(rough_state)

    class FailingNode:
        NODE_NAME = "fails"

        @staticmethod
        def run(ctx):
            return {"success": False}

    class NeverRuns:
        NODE_NAME = "never"

        @staticmethod
        def run(ctx):
            return {"success": True}

    modules = {"a.py": FailingNode, "b.py": NeverRuns}
    monkeypatch.setattr(graph, "load_node", lambda filename: modules[filename])
    graph.run_graph(ctx, ["a.py", "b.py"])
    assert len(ctx.history) == 1
    record = ctx.history[0]
    assert record.name == "fails"
    assert record.success is False
    assert record.fit == {"success": False}
    assert record.dataset is None
