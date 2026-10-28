"""A stand-in for SimulatedQPU that records what a node asks for and returns zeros."""

from __future__ import annotations

import inspect

import numpy as np

from simulator.qpu import SimulatedQPU

RECORDED_METHODS = ("run_sequences", "resonator_spectroscopy", "saturation_spectroscopy", "single_shots", "adc_trace")
# The argument that sets how many points a method returns, in order of preference.
_SIZE_ARGUMENTS = ("sequences", "frequencies", "drive_frequencies", "n_shots")


class RecordingQPU:
    """Stands in for SimulatedQPU in run_simulated: records each call's method and keyword arguments, returns zeros.

    Arguments are bound with ``inspect.signature(SimulatedQPU.<method>)``, so a recorded call is
    always by name and the stub follows SimulatedQPU's signatures as fixes 01, 07 and 08 change
    them. Only arguments the node passes are recorded (no defaults filled in). Keywords the real
    method lacks are recorded too, so the check sees them.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, method: str):
        if method not in RECORDED_METHODS:
            raise AttributeError(method)
        signature = inspect.signature(getattr(SimulatedQPU, method))

        def call(*args, **kwargs):
            known = {k: v for k, v in kwargs.items() if k in signature.parameters}
            extra = {k: v for k, v in kwargs.items() if k not in signature.parameters}
            bound = signature.bind(None, *args, **known)  # raises TypeError on a missing argument
            recorded = {k: v for k, v in bound.arguments.items() if k != "self"}
            recorded.update(extra)
            self.calls.append((method, recorded))
            if method == "adc_trace":
                from simulator.qpu import AdcTrace

                n = recorded["readout"].pulse_length + recorded["readout"].smearing
                return AdcTrace(np.arange(n), np.zeros((2, n)), np.zeros((2, n), dtype=np.int16))
            size = next((recorded[k] for k in _SIZE_ARGUMENTS if k in recorded), None)
            if size is None:
                raise TypeError(f"{method}: no argument sets the number of points (looked for {_SIZE_ARGUMENTS})")
            n = size if isinstance(size, int) else len(size)
            return np.zeros(n), np.zeros(n)

        return call
