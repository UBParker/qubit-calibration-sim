# %% {Imports}
import copy
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qm.qua import declare, declare_stream, for_, measure, program, reset_if_phase, save, stream_processing, wait

from calibration_utils import time_of_flight as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    readout_settings,
    save_figure,
    store_figure_inputs,
)
from calibration_utils.style import Marks
from qua.config import CLOCK_CYCLE_NS

NODE_NAME = "01a_time_of_flight"

description = """
        TIME OF FLIGHT
Plays the readout pulse and records the raw ADC trace of both analog inputs, averaged over the shots,
with the integration window opening at the configured time of flight. The trace shows when the signal
arrives at the ADC (the time of flight to add) and the DC offset of each input (the offsets to add).

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/01a_time_of_flight.py

Run as a setup step before the bring-up graph (calibrations/90_calibration_graph_bringup.py), as QM's
fixed-frequency graph assumes it was run.

The readout amplitude defaults to 0.1 V, the value in QM's docstring; QM's code default is 0.03 V. At
0.03 V and 100 averages the per-sample SNR in this simulator is too low for QM's threshold detector,
while at 0.1 V every run lands within 4 ns of the true delay. The simulated readout is linear, so the
higher amplitude has no side effect.

Prerequisites:
    - None; run with a time_of_flight at or below the true delay (the detector cannot see an edge that
      arrives before the window opens).

State update:
    - resonator.time_of_flight (time_of_flight_in_ns + delay if that parameter is set, else += delay)
    - resonator.opx_input_I.offset and opx_input_Q.offset (+= the measured mean of each averaged trace)
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 100
    time_of_flight_in_ns: int | None = None  # None: keep the state's value
    readout_amplitude_in_v: float = 0.1  # QM's code default is 0.03 V (its docstring says 0.1 V); see description
    readout_length_in_ns: int = 1000
    depletion_time_ns: int = 1000


def patched_state(state: dict, params: Parameters) -> dict:
    """A copy with QM's temporary overrides (tracked_updates): time of flight, readout length and
    amplitude. The program and its export use it; the state update writes to the unpatched state."""
    out = copy.deepcopy(state)
    res = out["qubits"][QUBIT]["resonator"]
    if params.time_of_flight_in_ns is not None:
        res["time_of_flight"] = int(params.time_of_flight_in_ns)
    res["operations"]["readout"]["length"] = int(params.readout_length_in_ns)
    res["operations"]["readout"]["amplitude"] = float(params.readout_amplitude_in_v)
    return out


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    """The program as QM's node writes it, for the patched state (the config it runs with)."""
    with program() as prog:
        n = declare(int)
        n_st = declare_stream()
        adc_st = declare_stream(adc_trace=True)
        with for_(n, 0, n < params.num_shots, n + 1):
            reset_if_phase("resonator")
            measure("readout", "resonator", adc_stream=adc_st)
            wait(params.depletion_time_ns // CLOCK_CYCLE_NS, "resonator")
            save(n, n_st)
        with stream_processing():
            n_st.save("n")
            adc_st.input1().average().save("adcI")
            adc_st.input2().average().save("adcQ")
            adc_st.input1().save("adc_single_runI")
            adc_st.input2().save("adc_single_runQ")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters) -> xr.Dataset:
    """Stand-in for qm.execute + fetching: the raw ADC traces in raw units over (qubit, readout_time)."""
    readout = readout_settings(patched_state(ctx.state, params))
    trace = ctx.qpu.adc_trace(readout=readout, n_avg=params.num_shots)
    dims = ("qubit", "readout_time")
    data = {
        "adcI": trace.averaged[0],
        "adcQ": trace.averaged[1],
        "adc_single_runI": trace.single_run[0],
        "adc_single_runQ": trace.single_run[1],
    }
    t = xr.DataArray(np.asarray(trace.readout_time), dims="readout_time", attrs={"long_name": "readout time", "units": "ns"})
    return xr.Dataset(
        {name: (dims, np.asarray(v)[None, :]) for name, v in data.items()},
        coords={"qubit": [QUBIT], "readout_time": t},
        attrs={
            "readout_length": readout.pulse_length,
            "readout_amplitude": readout.amplitude,
            "num_shots": params.num_shots,
            "time_of_flight": readout.time_of_flight,
        },
    )


# %% {Update_state}
def update_state(ctx: NodeContext, params: Parameters, fit: dict) -> None:
    """QM's two branches: an explicit time_of_flight_in_ns plus the delay, else the stored value plus it."""
    if not fit["success"]:
        return
    res = ctx.qubit["resonator"]
    delay = int(fit["tof_to_add"])
    if params.time_of_flight_in_ns is not None:
        res["time_of_flight"] = int(params.time_of_flight_in_ns) + delay
    else:
        res["time_of_flight"] = int(res.get("time_of_flight", 24)) + delay
    for key, add in (("opx_input_I", "offset_I_to_add"), ("opx_input_Q", "offset_Q_to_add")):
        entry = res.setdefault(key, {"offset": 0.0})
        entry["offset"] = float(entry.get("offset", 0.0)) + fit[add]


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    ran_with = patched_state(before, params)
    prog = create_qua_program(ran_with, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    update_state(ctx, params, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    context = figure_context(ctx, NODE_NAME, before, fit)
    # The true edge in the window, for figures only (a graph or setup run with a reference).
    true_tof = (ctx.reference or {}).get("time_of_flight") if ctx.graph_nodes is not None else None
    window_start = ds.attrs["time_of_flight"]
    marks = Marks(
        previous=None,
        fit=fit["delay"] if fit["success"] else None,
        true=None if true_tof is None else true_tof - window_start,
    )
    store_figure_inputs(ctx, NODE_NAME, marks, context)
    if ctx.save_outputs:
        export_qua(NODE_NAME, prog, ran_with)  # the program with the patched state it ran with
        save_figure(lambda theme: analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context), NODE_NAME)
    return fit


if __name__ == "__main__":
    from calibration_utils.common import load_state, save_state
    from simulator.qpu import SimulatedQPU

    ctx = NodeContext(state=load_state(), qpu=SimulatedQPU())
    print(run(ctx))
    save_state(ctx.state)
