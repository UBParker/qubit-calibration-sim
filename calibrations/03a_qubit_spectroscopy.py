# %% {Imports}
from dataclasses import dataclass

import numpy as np
from qm.qua import align, amp, declare, declare_stream, fixed, for_, play, program, save, stream_processing, update_frequency

from calibration_utils import qubit_spectroscopy as analysis
from calibration_utils.common import (
    NodeContext,
    export_qua,
    QUBIT,
    figure_context,
    iq_dataset,
    iq_projection,
    outlier_mask_of,
    readout_settings,
    save_figure,
    store_figure_inputs,
)
from calibration_utils.style import Marks
from qua.macros import readout_iq, thermalize

NODE_NAME = analysis.NODE_NAME

description = """
        QUBIT SPECTROSCOPY
At each frequency in a sweep, plays a long, weak saturation pulse on the qubit and then reads the resonator. Near the
qubit transition the pulse leaves the qubit in a g/e mixture, which shows up as a change in the readout signal. A
Lorentzian fit to that line gives the qubit frequency and its width. If the line is too faint to find, the drive can
be raised with operation_amplitude_factor.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/03a_qubit_spectroscopy.py

Prerequisites:
    - Readout frequency calibrated (node 02a).
    - A starting value of qubit.f_01 that lies inside the swept span.

State update:
    - Qubit g-e transition frequency: qubit.f_01
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 100
    # The rough guess is a few MHz off; 20 MHz covers it with margin. The 100 kHz step puts
    # about 6 points across the ~0.63 MHz power-broadened line (see operation_amplitude_factor).
    frequency_span_in_mhz: float = 20.0
    frequency_step_in_mhz: float = 0.1
    # Pre-factor on the saturation amplitude in the state. At 1.0 (2 mV, Rabi rate ~0.36 MHz)
    # the drive saturation parameter Omega^2 T1 T2 is ~15 000 (T2 = T2echo, the Markovian T2 of the
    # Bloch equations): the line is fully saturated (P_e ~ 0.5, maximal contrast) and power
    # broadened to FWHM = sqrt(1 + s) / (pi T2) ~ 0.63 MHz,
    # narrow enough to centre to a few kHz. The simulator has no AC Stark shift, so the
    # broadening does not bias the centre.
    operation_amplitude_factor: float = 1.0


def detunings_hz(params: Parameters) -> np.ndarray:
    """Drive detunings from the current f_01 guess, in whole Hz as QUA update_frequency() takes."""
    span = params.frequency_span_in_mhz * 1e6
    step = params.frequency_step_in_mhz * 1e6
    return np.round(np.arange(-span / 2, span / 2, step)).astype(int)


def saturation_amplitude(state: dict, params: Parameters) -> float:
    sat = state["qubits"][QUBIT]["xy"]["operations"]["saturation"]
    return sat["amplitude"] * params.operation_amplitude_factor


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    q = state["qubits"][QUBIT]
    dfs = detunings_hz(params)
    qubit_if = int(round(q["f_01"] - q["xy"]["LO_frequency"]))
    if_start, if_step = qubit_if + int(dfs[0]), int(dfs[1] - dfs[0])
    if_stop = qubit_if + int(dfs[-1]) + if_step
    with program() as prog:
        n = declare(int)
        f_if = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_(f_if, if_start, f_if < if_stop, f_if + if_step):
                update_frequency("qubit", f_if)
                play("saturation" * amp(params.operation_amplitude_factor), "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
                thermalize(state)
        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(dfs)).average().save("I")
            Q_st.buffer(len(dfs)).average().save("Q")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters):
    """Stand-in for qm.execute + fetching: the same sweep, played on the simulated QPU."""
    dfs = detunings_hz(params)
    drive_frequencies = ctx.qubit["f_01"] + dfs
    I, Q = ctx.qpu.saturation_spectroscopy(
        drive_frequencies,
        drive_amplitude=saturation_amplitude(ctx.state, params),
        n_avg=params.num_shots,
        readout=readout_settings(ctx.state),
    )
    ds = iq_dataset(I, Q, "detuning", dfs, {"long_name": "qubit drive detuning", "units": "Hz"})
    ds = ds.assign_coords(full_freq=("detuning", drive_frequencies))
    ds.full_freq.attrs = {"long_name": "qubit drive frequency", "units": "Hz"}
    return ds


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["f_01"] = fit["frequency"]


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    prog = create_qua_program(ctx.state, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    if ctx.save_outputs:
        export_qua(NODE_NAME, prog, ctx.state)
    update_state(ctx, fit)
    # Figures only: the true value comes from ctx.reference, which the fit above never sees.
    marks = Marks(
        previous=before["qubits"][QUBIT]["f_01"],
        fit=fit["frequency"] if fit["success"] else None,
        # Ruling V: a standalone run shows no true value, so only a graph run with a reference sets it.
        true=ctx.reference.get("f_01") if ctx.reference and ctx.graph_nodes is not None else None,
    )
    _, axis, centre = iq_projection(ds, "detuning", orientation="peak", exclude=outlier_mask_of(ds))
    context = figure_context(ctx, NODE_NAME, before, fit, axis, centre)
    if ctx.save_outputs:
        save_figure(
            lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context),
            NODE_NAME,
        )
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
