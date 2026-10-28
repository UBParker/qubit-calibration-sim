# %% {Imports}
import copy
from dataclasses import dataclass, replace

import numpy as np
from qm.qua import align, declare, declare_stream, fixed, for_, for_each_, play, program, save, stream_processing

from calibration_utils import power_rabi as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    iq_dataset,
    readout_settings,
    save_figure,
    store_figure_inputs,
    x180_pulse,
)
from calibration_utils.style import Marks
from qua.macros import readout_iq, thermalize

NODE_NAME = "04b_power_rabi"

description = """
        POWER RABI
Sweeps the x180 amplitude, as a multiple of its present value, and reads the qubit after each pulse. The excited
population oscillates with amplitude. A sinusoidal fit gives the frequency and phase of that oscillation, and the pi
amplitude is placed at the fit's first maximum.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/04b_power_rabi.py
Only the single-pulse sweep is implemented; QM's node can also repeat the pulse N times to amplify small amplitude errors.

Sweep coverage and the refine pass (this repo's choice, not QM behaviour):
QUA's amplitude_scale must lie in [-2, 2), so from a rough starting amplitude the sweep can cover less than a full Rabi
period. Like QM's analysis, the fit is rejected when it spans fewer than 0.75 periods. QM's node stops there and leaves
recentring the sweep to the user. This node instead runs one refine pass when coverage is the only failed gate: it
sweeps again around the provisional pi amplitude from the first pass (on a working copy of the state, not written to
the state) and fits again. The state update uses the refine pass.

Prerequisites:
    - Readout frequency (node 02a) and qubit frequency (node 03a) calibrated.
    - Gate length chosen: qubit.xy.operations["x180"].length.

State update:
    - The x180 pulse amplitude: qubit.xy.operations["x180"].amplitude
      (x90 is generated as half of it in qua.config, so it follows automatically).
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 100
    # Pre-factor of the current x180 amplitude; QUA's amplitude_scale must lie in [-2, 2).
    min_amp_factor: float = 0.001
    max_amp_factor: float = 1.99
    amp_factor_step: float = 0.01


def amp_prefactors(params: Parameters) -> np.ndarray:
    return np.arange(params.min_amp_factor, params.max_amp_factor, params.amp_factor_step)


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    amps = [float(a) for a in amp_prefactors(params)]
    with program() as prog:
        n = declare(int)
        a = declare(fixed)
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            with for_each_(a, amps):
                play("x180", "qubit", amplitude_scale=a)
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
                thermalize(state)
        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(amps)).average().save("I")
            Q_st.buffer(len(amps)).average().save("Q")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters):
    """Stand-in for qm.execute + fetching: the same sequence, played on the simulated QPU."""
    amps = amp_prefactors(params)
    sequences = [[x180_pulse(ctx.state, amplitude_scale=float(a))] for a in amps]
    I, Q = ctx.qpu.run_sequences(sequences, drive_frequency=ctx.qubit["f_01"], n_avg=params.num_shots, readout=readout_settings(ctx.state))
    return iq_dataset(I, Q, "amp_prefactor", amps, {"long_name": "pulse amplitude prefactor"})


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        ctx.qubit["xy"]["operations"]["x180"]["amplitude"] = fit["opt_amp"]


# %% {Run}
def _measure(ctx: NodeContext, state: dict, params: Parameters):
    """One sweep and fit around the x180 amplitude held in ``state``."""
    current_amplitude = state["qubits"][QUBIT]["xy"]["operations"]["x180"]["amplitude"]
    ds = analysis.process_raw_dataset(run_simulated(replace(ctx, state=state), params), current_amplitude)
    return ds, analysis.fit_raw_data(ds, current_amplitude)


def _only_coverage_failed(fit: dict) -> bool:
    return not fit["success"] and all(ok for gate, ok in fit["gates"].items() if gate != "n_periods")


def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    ds, fit = _measure(ctx, ctx.state, params)
    first_ds, first_fit = None, None
    if _only_coverage_failed(fit):
        # Refine pass: recentre the sweep on the provisional pi amplitude, in a working copy only.
        first_ds, first_fit = ds, fit
        run_state = copy.deepcopy(ctx.state)
        run_state["qubits"][QUBIT]["xy"]["operations"]["x180"]["amplitude"] = first_fit["opt_amp"]
        ds, fit = _measure(ctx, run_state, params)
    fit = {**fit, "refined_from": first_fit}
    analysis.log_fitted_results(fit)
    update_state(ctx, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds, "first_pass_dataset": first_ds}
    # Figures only: the reference never reaches the fit above.
    reference = ctx.reference or {}
    marks = Marks(
        previous=before["qubits"][QUBIT]["xy"]["operations"]["x180"]["amplitude"],
        fit=fit["opt_amp"] if fit["success"] else None,
        true=reference.get("x180.amplitude"),
    )
    context = figure_context(ctx, NODE_NAME, before, fit)
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        # The program and config as the node is entered (the state before the update). A refine
        # pass replays the same program with the x180 amplitude set to the provisional pi value.
        export_qua(NODE_NAME, create_qua_program(before, params), before)
        save_figure(
            lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context, first_pass=first_ds),
            NODE_NAME,
        )
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
