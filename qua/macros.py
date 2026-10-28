"""Small QUA building blocks shared by the calibration programs.

QUAlibrate nodes call equivalent QUAM methods (``qubit.resonator.measure``,
``qubit.reset``); these plain macros play that role against the config in
:mod:`qua.config`.
"""

from qm.qua import align, dual_demod, measure, wait

from qua.config import CLOCK_CYCLE_NS


def readout_iq(I, Q, rotated: bool = False):
    """Measure the resonator and demodulate into I and Q."""
    if rotated:
        measure(
            "readout",
            "resonator",
            dual_demod.full("rotated_cos", "rotated_sin", I),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
        )
    else:
        measure("readout", "resonator", dual_demod.full("cos", "sin", I), dual_demod.full("minus_sin", "cos", Q))


def thermalize(state: dict, qubit: str = "q1"):
    """Passive reset: wait several T1 for the qubit to relax before the next shot."""
    cycles = int(state["qubits"][qubit]["thermalization_time"] // CLOCK_CYCLE_NS)
    wait(cycles, "resonator")
    align()
