# %% {Imports}
import dataclasses
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qm.qua import align, declare, declare_stream, fixed, for_, play, program, save, stream_processing

from calibration_utils import iq_blobs as analysis
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

NODE_NAME = "07_iq_blobs"

description = """
        IQ BLOBS
Collects single-shot I/Q points for two preparations: the qubit left in |g> after waiting for it to thermalize, and
the qubit after an x180, nominally in |e>. From the two clouds the analysis finds:
    - the angle that rotates the integration weights so the g/e separation lies along I;
    - the threshold on I that misassigns the fewest shots;
    - the confusion matrix and assignment fidelity. Both include x180 errors and |e> decay during the readout
      window, not only readout noise.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/07_iq_blobs.py

Prerequisites:
    - Readout frequency calibrated (node 02a).
    - Qubit frequency and x180 amplitude calibrated (nodes 03a and 04b).

State update:
    - Integration-weight rotation angle: qubit.resonator.operations["readout"].integration_weights_angle
    - g/e discrimination threshold: qubit.resonator.operations["readout"].threshold
    - Assignment fidelity: qubit.resonator.readout_fidelity
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 4000


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    with program() as prog:
        n = declare(int)
        I_g, Q_g, I_e, Q_e = (declare(fixed) for _ in range(4))
        n_st = declare_stream()
        I_g_st, Q_g_st, I_e_st, Q_e_st = (declare_stream() for _ in range(4))
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
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
            I_g_st.buffer(params.num_shots).save("I_g")
            Q_g_st.buffer(params.num_shots).save("Q_g")
            I_e_st.buffer(params.num_shots).save("I_e")
            Q_e_st.buffer(params.num_shots).save("Q_e")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters) -> xr.Dataset:
    """Stand-in for qm.execute + fetching: unaveraged shots for each preparation, over (qubit, n_runs)."""
    readout = readout_settings(ctx.state)
    shots = {"drive_frequency": ctx.qubit["f_01"], "readout": readout, "n_shots": params.num_shots}
    I_g, Q_g = ctx.qpu.single_shots([], **shots)
    I_e, Q_e = ctx.qpu.single_shots([x180_pulse(ctx.state)], **shots)
    n_runs = xr.DataArray(np.arange(1, params.num_shots + 1), dims="n_runs", attrs={"long_name": "number of shots"})
    dims = ("qubit", "n_runs")
    return xr.Dataset(
        {name: (dims, np.asarray(v)[None, :]) for name, v in (("I_g", I_g), ("Q_g", Q_g), ("I_e", I_e), ("Q_e", Q_e))},
        coords={"qubit": [QUBIT], "n_runs": n_runs},
        attrs={"readout_length": readout.pulse_length},
    )


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        readout = ctx.qubit["resonator"]["operations"]["readout"]
        # Shots were demodulated with the unrotated weights, so this is the absolute angle (QM's node
        # subtracts from the previous angle because it measures with the already-rotated weights).
        readout["integration_weights_angle"] = fit["iw_angle"]
        # Stored in demodulation units, as QUA compares it with the raw dual_demod output.
        readout["threshold"] = analysis.volts_to_demod(fit["ge_threshold"], readout["length"])
        ctx.qubit["resonator"]["readout_fidelity"] = fit["fidelity"]


# %% {Run}
def reference_at_tone(ctx: NodeContext, before: dict) -> dict | None:
    """Figures only: ``ctx.reference`` with the weight angle and threshold of the simulator reference
    taken at this run's readout tone (they turn with the readout frequency); fidelity stays the
    reference at the g/e midpoint. None in standalone runs (Ruling V) or without a reference."""
    if ctx.reference is None or ctx.graph_nodes is None:
        return ctx.reference
    from simulator.reference import blob_reference_at_tone

    tone = before["qubits"][QUBIT]["resonator"]["f_01"]
    at_tone = blob_reference_at_tone(before, tone)
    return {**ctx.reference, "ge_threshold": at_tone["ge_threshold"], "iw_angle": at_tone["iw_angle"], "reference_tone": tone}


def threshold_marks(ctx: NodeContext, before: dict, fit: dict, reference: dict | None = None) -> Marks:
    """Thresholds in volts for the histogram: previous (from the state), fitted, simulator reference."""
    readout = before["qubits"][QUBIT]["resonator"]["operations"]["readout"]
    previous = readout["threshold"]
    # Standalone runs draw no reference (Ruling V).
    reference = (reference or {}).get("ge_threshold") if ctx.graph_nodes is not None else None
    return Marks(
        previous=None if previous is None else analysis.demod_to_volts(previous, readout["length"]),
        fit=fit["ge_threshold"] if fit["success"] else None,
        true=reference,
        true_label="simulator reference",
    )


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
    context = figure_context(ctx, NODE_NAME, before, fit)
    at_tone = reference_at_tone(ctx, before)
    if at_tone is not None:
        context = dataclasses.replace(context, reference=at_tone)
    marks = threshold_marks(ctx, before, fit, at_tone)
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
