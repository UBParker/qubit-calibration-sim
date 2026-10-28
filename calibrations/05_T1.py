# %% {Imports}
from dataclasses import dataclass

import numpy as np
from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, play, program, save, stream_processing, wait

from calibration_utils import T1 as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    iq_dataset,
    iq_projection,
    outlier_mask_of,
    readout_settings,
    save_figure,
    store_figure_inputs,
    x180_pulse,
)
from calibration_utils.style import FigureContext, Marks
from qua.config import CLOCK_CYCLE_NS
from qua.macros import readout_iq, thermalize
from simulator.transmon import Idle

NODE_NAME = "05_T1"

description = """
        T1 MEASUREMENT
Excites the qubit with x180, idles for a variable time, then reads out. The longer the idle, the closer the signal
returns to its |g> value. An exponential fit to that relaxation gives T1.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/05_T1.py

Prerequisites:
    - Readout frequency calibrated (node 02a).
    - Qubit frequency and x180 amplitude calibrated (nodes 03a and 04b).

State update:
    - Energy relaxation time: qubit.T1
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 200
    min_wait_time_ns: int = 16
    max_wait_time_ns: int = 200_000
    num_points: int = 61


def idle_times_ns(params: Parameters) -> np.ndarray:
    """Sweep points rounded to whole clock cycles, as QUA wait() requires."""
    t = np.linspace(params.min_wait_time_ns, params.max_wait_time_ns, params.num_points)
    return (np.round(t / CLOCK_CYCLE_NS) * CLOCK_CYCLE_NS).astype(int)


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    idle_cycles = [int(t) // CLOCK_CYCLE_NS for t in idle_times_ns(params)]
    with program() as prog:
        n = declare(int)
        t = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_each_(t, idle_cycles):
                play("x180", "qubit")
                wait(t, "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
                thermalize(state)
        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(idle_cycles)).average().save("I")
            Q_st.buffer(len(idle_cycles)).average().save("Q")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters):
    """Stand-in for qm.execute + fetching: the same sequence, played on the simulated QPU."""
    times = idle_times_ns(params)
    sequences = [[x180_pulse(ctx.state), Idle(float(t))] for t in times]
    I, Q = ctx.qpu.run_sequences(sequences, drive_frequency=ctx.qubit["f_01"], n_avg=params.num_shots, readout=readout_settings(ctx.state))
    return iq_dataset(I, Q, "idle_time", times, {"long_name": "idle time", "units": "ns"})


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["T1"] = fit["T1"]


# %% {Figure inputs}
def figure_inputs(ctx: NodeContext, before: dict, fit: dict, ds) -> tuple[Marks, FigureContext]:
    """Marks (previous, fitted and true T1 in ns) and the Layout C context for the figure.

    The IQ axis and centre come from the same projection the analysis used, so the 07 blob means
    land on this node's signal scale. ``ctx.reference`` is read here, for plotting only.
    """
    _, axis, centre = iq_projection(ds, "idle_time", first_point_excited=True, exclude=outlier_mask_of(ds))
    context = figure_context(ctx, NODE_NAME, before, fit, axis, centre)
    # Ruling V: a standalone run shows no true value, so only a graph run with a reference sets it.
    true = (ctx.reference or {}).get("T1") if ctx.graph_nodes is not None else None
    previous = before["qubits"][QUBIT].get("T1")
    marks = Marks(previous=previous, fit=fit["T1"] if fit["success"] else None, true=true)
    return marks, context


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    prog = create_qua_program(ctx.state, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    update_state(ctx, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    marks, context = figure_inputs(ctx, before, fit, ds)
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        export_qua(NODE_NAME, prog, before)  # the state the program was built from
        save_figure(lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context), NODE_NAME)
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
