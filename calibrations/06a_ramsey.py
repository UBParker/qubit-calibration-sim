# %% {Imports}
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qm.qua import (
    align,
    assign,
    declare,
    declare_stream,
    else_,
    fixed,
    for_,
    for_each_,
    frame_rotation_2pi,
    if_,
    play,
    program,
    reset_frame,
    save,
    stream_processing,
    wait,
)

from calibration_utils import ramsey as analysis
from calibration_utils.common import (
    QUBIT,
    NodeContext,
    export_qua,
    figure_context,
    readout_settings,
    save_figure,
    store_figure_inputs,
    x90_pulse,
)
from calibration_utils.style import Marks
from qua.config import CLOCK_CYCLE_NS
from qua.macros import readout_iq, thermalize
from simulator.transmon import Idle

NODE_NAME = "06a_ramsey"

description = """
        RAMSEY WITH VIRTUAL DETUNING
Ramsey sequence (Ramsey 1950, https://doi.org/10.1103/PhysRev.78.695): x90, variable idle, x90, readout. The drive stays on the current qubit frequency. Instead, the frame
of the second x90 is advanced in phase in proportion to the idle time, which produces the same fringes a real drive
detuning would. The sweep runs once with a positive and once with a negative virtual detuning; comparing the two
fringe frequencies gives the sign of the remaining frequency error. A damped-cosine fit returns that error, which
corrects the qubit frequency, and T2*. The damped-cosine model is exponential; the simulated envelope is not.

Adapted from Quantum Machines' qua-libs node of the same name (BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibrations/1Q_calibrations/06a_ramsey.py

Prerequisites:
    - Readout frequency calibrated (node 02a).
    - Qubit frequency and x180 amplitude calibrated (nodes 03a and 04b).

State update:
    - Qubit frequency, shifted by the fitted error: qubit.f_01
    - Ramsey dephasing time T2*: qubit.T2ramsey
"""


# %% {Node parameters}
@dataclass
class Parameters:
    num_shots: int = 200
    # QM defaults to 1 MHz; 0.3 MHz gives about 16 samples per fringe, so the oscillation is visible
    # in the plot, while still well above the frequency error left by 03a (about 10 kHz).
    frequency_detuning_in_mhz: float = 0.3
    min_wait_time_ns: int = 16
    max_wait_time_ns: int = 50_000
    # Linear sweep (QM defaults to log): the FFT frequency guess in QM's fit assumes even spacing.
    num_points: int = 250


DETUNING_SIGNS = [-1, 1]


def idle_times_ns(params: Parameters) -> np.ndarray:
    """Sweep points rounded to whole clock cycles, as QUA wait() requires."""
    t = np.linspace(params.min_wait_time_ns, params.max_wait_time_ns, params.num_points)
    return (np.round(t / CLOCK_CYCLE_NS) * CLOCK_CYCLE_NS).astype(int)


def virtual_detuning_turns(params: Parameters) -> list[float]:
    """Frame phase, in turns, that a +detuning accumulates over each idle time, wrapped to [0, 1).

    With the defaults the unwrapped phase reaches 0.3 MHz * 50 us = 15 turns, past the [-8, 8) range of a QUA
    fixed (4.28 fixed point). Wrapping modulo one turn first keeps every value the program
    assigns inside the range; the frame rotation is unchanged, since frames are periodic in turns.
    Rounded to 1e-12 turns so the exported literals read cleanly (15.000000000000002 % 1 would
    print as 1.8e-15).
    """
    detuning = params.frequency_detuning_in_mhz * 1e6
    return [round(float(detuning * 1e-9 * t) % 1.0, 12) % 1.0 for t in idle_times_ns(params)]


# %% {Create_QUA_program}
def create_qua_program(state: dict, params: Parameters):
    idle_cycles = [int(t) // CLOCK_CYCLE_NS for t in idle_times_ns(params)]
    phase_turns = virtual_detuning_turns(params)
    with program() as prog:
        n = declare(int)
        t = declare(int)
        phase = declare(fixed)
        detuning_sign = declare(int)
        virtual_detuning_phase = declare(fixed)
        I = declare(fixed)
        Q = declare(fixed)
        n_st, I_st, Q_st = declare_stream(), declare_stream(), declare_stream()
        with for_(n, 0, n < params.num_shots, n + 1):
            save(n, n_st)
            # t (clock cycles) and the phase a +detuning accumulates over 4*t ns, in turns modulo 1
            with for_each_((t, phase), (idle_cycles, phase_turns)):
                with for_each_(detuning_sign, DETUNING_SIGNS):
                    # Frame rotations accumulate, so start every shot from a fresh frame.
                    reset_frame("qubit")
                    with if_(detuning_sign == 1):
                        assign(virtual_detuning_phase, phase)
                    with else_():
                        assign(virtual_detuning_phase, -phase)
                    play("x90", "qubit")
                    frame_rotation_2pi(virtual_detuning_phase, "qubit")
                    wait(t, "qubit")
                    play("x90", "qubit")
                    align("qubit", "resonator")
                    readout_iq(I, Q)
                    save(I, I_st)
                    save(Q, Q_st)
                    thermalize(state)
        with stream_processing():
            n_st.save("n")
            I_st.buffer(len(DETUNING_SIGNS)).buffer(len(idle_cycles)).average().save("I")
            Q_st.buffer(len(DETUNING_SIGNS)).buffer(len(idle_cycles)).average().save("Q")
    return prog


# %% {Execute (simulated)}
def run_simulated(ctx: NodeContext, params: Parameters):
    """Stand-in for qm.execute + fetching: the same sequence, played on the simulated QPU.

    The QUA program keeps the drive on f_01 and rotates the frame of the second x90, so here too
    the drive stays at f_01 and the virtual detuning is a phase on the second pulse. QM's analysis
    reads detuning sign +1 as a drive sitting +detuning above f_01, which in this simulator's
    rotating frame (H = -(f_drive - f_01) n) is a pulse phase of -2*pi*detuning*tau. That
    equivalence was checked against a real drive offset: same fringe frequency, up to a constant
    phase from the first pulse's length.
    """
    times = idle_times_ns(params)
    detuning = params.frequency_detuning_in_mhz * 1e6
    sequences = [
        [x90_pulse(ctx.state), Idle(float(t)), x90_pulse(ctx.state, phase=-sign * 2 * np.pi * detuning * t * 1e-9)]
        for t in times
        for sign in DETUNING_SIGNS
    ]
    I, Q = ctx.qpu.run_sequences(sequences, drive_frequency=ctx.qubit["f_01"], n_avg=params.num_shots, readout=readout_settings(ctx.state))
    # Same (idle_time, detuning_signs) layout as the buffered streams.
    shape = (1, len(times), len(DETUNING_SIGNS))
    return xr.Dataset(
        {"I": (("qubit", "idle_time", "detuning_signs"), np.reshape(I, shape)), "Q": (("qubit", "idle_time", "detuning_signs"), np.reshape(Q, shape))},
        coords={
            "qubit": [QUBIT],
            "idle_time": xr.DataArray(times, dims="idle_time", attrs={"long_name": "idle time", "units": "ns"}),
            "detuning_signs": xr.DataArray(DETUNING_SIGNS, dims="detuning_signs", attrs={"long_name": "detuning signs"}),
        },
        attrs={"detuning_hz": detuning},
    )


# %% {Update_state}
def update_state(ctx: NodeContext, fit: dict) -> None:
    if fit["success"]:
        # freq_offset = f_drive - f_01, and the drive sat at the old f_01.
        ctx.qubit["f_01"] -= fit["freq_offset"]
        ctx.qubit["T2ramsey"] = fit["T2ramsey"]


# %% {Run}
def run(ctx: NodeContext, params: Parameters | None = None) -> dict:
    params = params or Parameters()
    before = ctx.snapshot()
    prog = create_qua_program(ctx.state, params)
    ds = analysis.process_raw_dataset(run_simulated(ctx, params))
    fit = analysis.fit_raw_data(ds)
    analysis.log_fitted_results(fit)
    if ctx.save_outputs:
        # Exported with the state the program ran on, before the update below.
        export_qua(NODE_NAME, prog, ctx.state)
    update_state(ctx, fit)
    ctx.results[NODE_NAME] = {"fit": fit, "dataset": ds}
    f_before = before["qubits"][QUBIT]["f_01"]
    marks = Marks(
        previous=f_before,
        # Only an accepted fit has a fitted value; a rejected one is not applied.
        fit=f_before - fit["freq_offset"] if fit["success"] else None,
        true=ctx.reference.get("f_01") if ctx.reference else None,
    )
    axis, centre = analysis.iq_axis(ds)
    context = figure_context(ctx, NODE_NAME, before, fit, axis, centre)
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
