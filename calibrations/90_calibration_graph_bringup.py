# %% {Imports}
"""
        FIXED-FREQUENCY TRANSMON BRING-UP GRAPH
Runs the single-qubit calibration nodes in sequence, each node reading the state written by the
previous one, starting from the rough guesses in state_initial.json and writing the result to state.json.

Adapted from Quantum Machines' qua-libs graph (BSD-3-Clause), keeping QM's node order for the nodes
implemented here:
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/90_calibration_graph_bringup_fixed_frequency_transmon.py
The eight nodes here are 02a, 03a, 04b, 08a, 07, 06a, 05 and 06b. QM's graph also runs Rabi chevron,
readout power optimization (08b), error-amplified power Rabi, DRAG and randomized benchmarking; those
are not implemented yet.

Before the graph, run_setup runs the setup step 01a_time_of_flight (time of flight and ADC offsets
from the raw ADC trace). QM numbers that node first but leaves it out of this graph, which assumes it
was run when the setup was wired; here it runs from the rough 24 ns of state_initial.json, outside the
graph's node list and history, so the graph order stays QM's.

The graph stops at the first node whose fit fails, because every later node depends on the
parameters that node was meant to set. This is a choice made here, not QM's behaviour: QUAlibrate's
BasicOrchestrator drops a failed qubit from later nodes when skip_failed=True and keeps it otherwise
(https://github.com/qua-platform/qualibrate-core/blob/main/qualibrate/orchestration/basic_orchestrator.py).
"""

from __future__ import annotations

import logging
import sys

from calibration_utils.common import (
    INITIAL_STATE_PATH,
    NODE_ORDER,
    SETUP_NODES,
    NodeContext,
    NodeRecord,
    load_node,
    load_state,
    save_state,
)
from calibration_utils.iq_blobs import demod_to_volts
from simulator.qpu import SimulatedQPU
from simulator.reference import build_reference

log = logging.getLogger("bringup")

NODES = [f"{name}.py" for name in NODE_ORDER]


def run_setup(ctx: NodeContext) -> dict:
    """Run the setup steps (01a) on ``ctx`` before the graph. Returns {node_name: fit}.

    While a setup step runs, ``ctx.graph_nodes`` holds the setup steps, so its figure shows the true
    value when ``ctx.reference`` is set; it is reset to None afterwards. Setup steps are not added to
    ``ctx.history`` (the graph's record). Raises RuntimeError if a setup fit fails.
    """
    outcomes = {}
    for name in SETUP_NODES:
        node = load_node(f"{name}.py")
        ctx.graph_nodes = [name]
        try:
            fit = node.run(ctx)
        finally:
            ctx.graph_nodes = None
        outcomes[name] = fit
        if not fit["success"]:
            raise RuntimeError(f"setup step {name} failed; the bring-up graph needs its time of flight and offsets")
    return outcomes


def run_bringup(ctx: NodeContext) -> dict:
    """Setup steps, then the graph: what the script's ``__main__`` and viz.build_all run."""
    run_setup(ctx)
    return run_graph(ctx)


def run_graph(ctx: NodeContext, nodes: list[str] = NODES) -> dict:
    """Run nodes in order. Returns {node_name: fit}; stops after the first failed fit.

    Appends a NodeRecord per node run to ``ctx.history``. A node that fails may not have stored a
    dataset in ``ctx.results``, in which case the record's dataset is None.
    """
    outcomes = {}
    modules = [load_node(filename) for filename in nodes]
    ctx.graph_nodes = [node.NODE_NAME for node in modules]
    for node in modules:
        before = ctx.snapshot()
        fit = node.run(ctx)
        outcomes[node.NODE_NAME] = fit
        dataset = ctx.results.get(node.NODE_NAME, {}).get("dataset")
        ctx.history.append(NodeRecord(node.NODE_NAME, before, ctx.snapshot(), fit, dataset, bool(fit["success"])))
        if not fit["success"]:
            log.error("%s failed; stopping the graph so later nodes do not run on a bad state", node.NODE_NAME)
            break
    return outcomes


def summary_table(state: dict) -> str:
    q = state["qubits"]["q1"]
    ro = q["resonator"]["operations"]["readout"]
    rows = [
        ("Resonator frequency", q["resonator"]["f_01"] / 1e9, "GHz", "{:.6f}"),
        ("Qubit frequency", q["f_01"] / 1e9, "GHz", "{:.6f}"),
        ("x180 amplitude", q["xy"]["operations"]["x180"]["amplitude"], "V", "{:.4f}"),
        ("Readout fidelity", q["resonator"]["readout_fidelity"], "", "{:.3f}"),
        ("Readout threshold", None if ro["threshold"] is None else demod_to_volts(ro["threshold"], ro["length"]) * 1e3, "mV", "{:.3f}"),
        ("T2*", None if q["T2ramsey"] is None else q["T2ramsey"] / 1e3, "us", "{:.1f}"),
        ("T1", None if q["T1"] is None else q["T1"] / 1e3, "us", "{:.1f}"),
        ("T2 echo", None if q["T2echo"] is None else q["T2echo"] / 1e3, "us", "{:.1f}"),
    ]
    lines = []
    for name, value, unit, fmt in rows:
        shown = "uncalculated" if value is None else f"{fmt.format(value)} {unit}"
        lines.append(f"  {name:<20} {shown}".rstrip())
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    ctx = NodeContext(state=load_state(INITIAL_STATE_PATH), qpu=SimulatedQPU(seed=seed))
    ctx.reference = build_reference(ctx.state)
    outcomes = run_bringup(ctx)
    save_state(ctx.state)
    passed = sum(fit["success"] for fit in outcomes.values())
    print(f"\n{passed}/{len(NODES)} nodes succeeded. Calibrated state written to state.json:\n")
    print(summary_table(ctx.state))
