# %% {Imports}
from dataclasses import dataclass

import numpy as np
from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, play, program, save, stream_processing, wait

from calibration_utils import echo as analysis
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
    x90_pulse,
    x180_pulse,
)
from calibration_utils.style import Marks
from qua.config import CLOCK_CYCLE_NS
from qua.macros import readout_iq, thermalize
from simulator.transmon import Idle

NODE_NAME = "06b_echo"

description = """
        HAHN ECHO (T2 echo)
Hahn echo (Hahn 1950, https://doi.org/10.1103/PhysRev.80.580): x90, half the idle time, x180, the other half, -x90, readout, over a sweep of total idle times. The x180 in
the middle refocuses quasi-static frequency noise, so the coherence decays with T2echo instead of T2*. The final pulse
is -x90, as in QM's node: the net rotation is then x180 and a fully refocused qubit ends in |e>, decaying towards the
50/50 mixture as coherence is lost. An exponential fit to that decay gives T2echo.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/06b_echo.py

Simulator note: the simulated qubit has a quasi-static frequency offset per shot (simulator/device.py) on top of
white dephasing. The x180 refocuses the offset, so T2echo exceeds T2* because of the sequence, not a separate setting;
the white part is not refocused, so the decay stays exponential.

Prerequisites:
    - Readout frequency calibrated (node 02a).
    - x180 amplitude calibrated (node 04b) and qubit frequency refined by Ramsey (node 06a).

State update:
    - Echo coherence time: qubit.T2echo
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 1000
    min_wait_time_ns: int = 32
    max_wait_time_ns: int = 200_000
    num_points: int = 81


def half_idle_cycles(params: Parameters) -> np.ndarray:
    """Clock cycles waited on each side of the x180. QUA wait() needs whole cycles, at least 4 (16 ns)."""
    t_half = np.linspace(params.min_wait_time_ns, params.max_wait_time_ns, params.num_points) / 2
    return np.maximum(np.round(t_half / CLOCK_CYCLE_NS), 4).astype(int)


def idle_times_ns(params: Parameters) -> np.ndarray:
    """Total idle time between the two x90 pulses (both halves), as QM's node defines idle_time."""
    return 2 * CLOCK_CYCLE_NS * half_idle_cycles(params)


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    idle_cycles = [int(t) for t in half_idle_cycles(params)]
    with program() as prog:
        n = declare(int)
        t = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_each_(t, idle_cycles):
                play("x90", "qubit")
                wait(t, "qubit")
                play("x180", "qubit")
                wait(t, "qubit")
                play("-x90", "qubit")
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
    sequences = [
        [x90_pulse(ctx.state), Idle(t / 2), x180_pulse(ctx.state), Idle(t / 2), x90_pulse(ctx.state, phase=np.pi)]
        for t in times.astype(float)
    ]
    I, Q = ctx.qpu.run_sequences(
        sequences, drive_frequency=ctx.qubit["f_01"], n_avg=params.num_shots, readout=readout_settings(ctx.state)
    )
    return iq_dataset(I, Q, "idle_time", times, {"long_name": "total idle time", "units": "ns"})


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["T2echo"] = fit["T2echo"]


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
    # Same projection as process_raw_dataset, for the P(e) axis from the 07 blob means.
    _, axis, centre = iq_projection(ds, "idle_time", first_point_excited=True, exclude=outlier_mask_of(ds))
    context = figure_context(ctx, NODE_NAME, before, fit, axis, centre)
    marks = Marks(
        previous=before["qubits"][QUBIT].get("T2echo"),
        fit=fit["T2echo"] if fit["success"] else None,
        # Ruling V: a standalone run shows no true value, so only a graph run with a reference sets it.
        true=ctx.reference.get("T2echo") if ctx.reference and ctx.graph_nodes is not None else None,
    )
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        export_qua(NODE_NAME, prog, before)
        save_figure(lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context), NODE_NAME)
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
