# %% {Imports}
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, play, program, save, stream_processing, update_frequency

from calibration_utils import readout_frequency_optimization as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    readout_settings,
    save_figure,
    store_figure_inputs,
    x180_pulse,
)
from calibration_utils.style import Marks
from qua.macros import readout_iq, thermalize

NODE_NAME = "08a_readout_frequency_optimization"

description = """
        READOUT FREQUENCY OPTIMIZATION
Sweeps the readout frequency around its current value and, at each point, measures the averaged I/Q after
preparing |g> (passive reset) and |e> (reset, x180). The readout frequency becomes the one that maximises the
distance D between the two averaged I/Q points (after QM's 5-point rolling mean), and the dispersive shift chi is
half the distance between the |e> and |g> resonance dips.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/08a_readout_frequency_optimization.py

Prerequisites:
    - Readout resonator frequency (node 02a), qubit frequency (03a) and x180 amplitude (04b) calibrated.

State update:
    - Readout frequency: qubit.resonator.f_01
    - Dispersive shift: qubit.chi, and qubit.extras["chi_source"] = this node's name
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 100
    frequency_span_in_mhz: float = 10.0
    frequency_step_in_mhz: float = 0.1


def detunings_hz(params: Parameters) -> np.ndarray:
    """QM's sweep, np.arange(-span/2, span/2, step), in whole Hz (QUA IFs are integers)."""
    span, step = params.frequency_span_in_mhz * 1e6, params.frequency_step_in_mhz * 1e6
    return np.round(np.arange(-span / 2, span / 2, step)).astype(int)


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    res = state["qubits"][QUBIT]["resonator"]
    dfs = detunings_hz(params)
    if_start = int(round(res["f_01"] - res["LO_frequency"])) + int(dfs[0])
    if_step = int(dfs[1] - dfs[0])
    with program() as prog:
        n = declare(int)
        df = declare(int)
        I_g, Q_g, I_e, Q_e = (declare(fixed) for _ in range(4))
        n_st = declare_stream()
        I_g_st, Q_g_st, I_e_st, Q_e_st = (declare_stream() for _ in range(4))
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_(df, if_start, df < if_start + len(dfs) * if_step, df + if_step):
                update_frequency("resonator", df)
                # Ground state: passive reset, then measure with no pulse
                thermalize(state)
                readout_iq(I_g, Q_g)
                save(I_g, I_g_st)
                save(Q_g, Q_g_st)
                # Excited state: passive reset, x180, then measure
                thermalize(state)
                play("x180", "qubit")
                align("qubit", "resonator")
                readout_iq(I_e, Q_e)
                save(I_e, I_e_st)
                save(Q_e, Q_e_st)
        with stream_processing():
            n_st.save("n")
            I_g_st.buffer(len(dfs)).average().save("I_g")
            Q_g_st.buffer(len(dfs)).average().save("Q_g")
            I_e_st.buffer(len(dfs)).average().save("I_e")
            Q_e_st.buffer(len(dfs)).average().save("Q_e")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters) -> xr.Dataset:
    """Stand-in for qm.execute + fetching: averaged I/Q per detuning for each preparation."""
    res = ctx.qubit["resonator"]
    dfs = detunings_hz(params)
    frequencies = res["f_01"] + dfs
    common = {"drive_frequency": ctx.qubit["f_01"], "readout": readout_settings(ctx.state), "n_avg": params.num_shots, "frequencies": frequencies}
    I_g, Q_g = ctx.qpu.run_sequences([[] for _ in dfs], **common)
    I_e, Q_e = ctx.qpu.run_sequences([[x180_pulse(ctx.state)] for _ in dfs], **common)
    detuning = xr.DataArray(dfs, dims="detuning", attrs={"long_name": "readout frequency detuning", "units": "Hz"})
    dims = ("qubit", "detuning")
    ds = xr.Dataset(
        {name: (dims, np.asarray(v)[None, :]) for name, v in (("I_g", I_g), ("Q_g", Q_g), ("I_e", I_e), ("Q_e", Q_e))},
        coords={"qubit": [QUBIT], "detuning": detuning},
    )
    return ds.assign_coords(full_freq=("detuning", frequencies, {"long_name": "readout frequency", "units": "Hz"}))


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["resonator"]["f_01"] = fit["optimal_frequency"]
        ctx.qubit["chi"] = fit["chi"]
        ctx.qubit.setdefault("extras", {})["chi_source"] = NODE_NAME


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    prog = create_qua_program(ctx.state, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    if ctx.save_outputs:
        export_qua(NODE_NAME, prog, ctx.state)  # the program as run, with the state it ran with
    update_state(ctx, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    # Figures only, in detuning from the frequency the node started at.
    f_before = before["qubits"][QUBIT]["resonator"]["f_01"]
    true = (ctx.reference or {}).get("readout.frequency") if ctx.graph_nodes is not None else None
    marks = Marks(
        previous=0.0,
        fit=fit["optimal_detuning"] if fit["success"] else None,
        true=None if true is None else true - f_before,
    )
    context = figure_context(ctx, NODE_NAME, before, fit)
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        save_figure(
            lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context), NODE_NAME
        )
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
