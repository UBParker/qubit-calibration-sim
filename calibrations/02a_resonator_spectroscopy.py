# %% {Imports}
from dataclasses import dataclass

import numpy as np
from qm.qua import declare, declare_stream, fixed, for_, program, save, stream_processing, update_frequency, wait

from calibration_utils import resonator_spectroscopy as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    iq_dataset,
    readout_settings,
    save_figure,
    store_figure_inputs,
)
from calibration_utils.style import Marks
from qua.config import CLOCK_CYCLE_NS
from qua.macros import readout_iq

NODE_NAME = "02a_resonator_spectroscopy"

description = """
        1D RESONATOR SPECTROSCOPY
Sweeps the readout intermediate frequency across the resonator with the qubit undriven, in its thermal state. Each
point is one demodulated I/Q pair. A Lorentzian dip fitted to |S21| locates the resonance, and the fitted centre
becomes the new readout frequency.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/02a_resonator_spectroscopy.py

Prerequisites:
    - Time of flight and ADC offsets calibrated (the 01a setup step, run before the graph).
    - Readout pulse amplitude and length set in the state.

State update:
    - Readout resonator frequency: qubit.resonator.f_01
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 100
    frequency_span_in_mhz: float = 20.0
    frequency_step_in_mhz: float = 0.05
    # Wait after each readout so the resonator rings down before the next pulse. 1/kappa is
    # about 0.1 us for a 1-2 MHz linewidth, so 1 us leaves well under 1e-3 of the photons.
    depletion_time_ns: int = 1000


def detunings_hz(params: Parameters) -> np.ndarray:
    """Symmetric sweep around the current readout frequency, in whole Hz (QUA IFs are integers)."""
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
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_(df, if_start, df < if_start + len(dfs) * if_step, df + if_step):
                update_frequency("resonator", df)
                readout_iq(I, Q)
                # No qubit pulse, so no thermalization: only the resonator needs to empty.
                wait(params.depletion_time_ns // CLOCK_CYCLE_NS, "resonator")
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(dfs)).average().save("I")
            Q_st.buffer(len(dfs)).average().save("Q")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters):
    """Stand-in for qm.execute + fetching: the same sweep, measured on the simulated QPU."""
    res = ctx.qubit["resonator"]
    dfs = detunings_hz(params)
    frequencies = res["f_01"] + dfs
    I, Q = ctx.qpu.resonator_spectroscopy(frequencies, readout=readout_settings(ctx.state), n_avg=params.num_shots)
    ds = iq_dataset(I, Q, "detuning", dfs, {"long_name": "readout frequency detuning", "units": "Hz"})
    ds = ds.assign_coords(full_freq=("detuning", frequencies, {"long_name": "readout frequency", "units": "Hz"}))
    ds.attrs["readout_frequency_guess"] = float(res["f_01"])
    return ds


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["resonator"]["f_01"] = fit["frequency"]


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    prog = create_qua_program(ctx.state, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    if ctx.save_outputs:
        # Exported with the state the program was built from, before the update.
        export_qua(NODE_NAME, prog, ctx.state)
    update_state(ctx, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    # Figures only: the reference is read here, after the fit and the state update.
    marks = Marks(
        previous=before["qubits"][QUBIT]["resonator"]["f_01"],
        fit=fit["frequency"] if fit["success"] else None,
        true=ctx.reference.get("resonator.f_01") if ctx.reference else None,
    )
    context = figure_context(ctx, NODE_NAME, before, fit)
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        save_figure(
            lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context),
            NODE_NAME,
        )
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
