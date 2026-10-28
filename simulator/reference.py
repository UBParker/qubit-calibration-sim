"""Reference values that figures draw next to each node's result.

Two kinds, labelled differently on figures:
    - true values: parameters stored on the hidden :class:`simulator.device.Device`;
    - simulator reference: readout fidelity, threshold and weight angle, which the device does not
      store, estimated here from many single shots with the true pi pulse.

Only plotting reads these, through ``NodeContext.reference``. Calibration code never does, and
``tests/test_reference.py::test_fits_ignore_reference`` checks that fits do not change when the
reference is set.

Units: the true values follow state.json (frequencies in Hz, amplitudes in volts, times in ns).
The readout reference follows the 07_iq_blobs fit dict, not state.json: ``ge_threshold`` is in
volts (state.json stores the threshold in demodulation units, see ``iq_blobs.volts_to_demod``),
``iw_angle`` is in radians and ``fidelity`` is a fraction.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE, Device
from simulator.qpu import SimulatedQPU
from simulator.transmon import Pulse


# Closure tolerances the bring-up test (tests/test_90_bringup.py) asserts on the calibrated state,
# keyed like the reference. "abs" is in state units (Hz), "rel" a fraction of the true value, and
# "window" an absolute (low, high) range on the stored readout fidelity. Figures shade the same
# bands; calibration code never reads them.
CLOSURE_TOLERANCES: dict[str, dict] = {
    # 08a sets the final readout frequency, so the bring-up tests it against the g/e midpoint
    # f_res_g - chi (02a's per-node test keeps its own kappa/10 band on the |g> dip). The band is
    # QM's trailing-window offset (+0.2 MHz) plus 4.3 standard deviations of the pick scatter
    # (93 kHz at 100 shots, 300 runs); the g/e separation at the band edge is 0.85 of its maximum.
    "readout.frequency": {"abs": 600e3},
    "chi": {"abs": 150e3},
    "f_01": {"abs": 20e3},
    "x180.amplitude": {"rel": 0.03},
    "fidelity": {"window": (0.95, 0.995)},
    # The 06a exponential fit is biased low by about 9 % against the 1/e time of the non-exponential
    # envelope (tests/test_06a_ramsey.py pins it); the 15 % band has to absorb that, leaving about
    # 6 % for statistics.
    "T2ramsey": {"rel": 0.15},
    "T1": {"rel": 0.15},
    "T2echo": {"rel": 0.15},
    # 01a (setup step): QM rounds the edge to the 4 ns clock; offsets per analog input, in volts.
    "time_of_flight": {"abs": 4.0},
    "adc_offset": {"abs": 0.5e-3},
}


def true_values(device: Device = TRUE_DEVICE) -> dict[str, float]:
    """Device parameters keyed like the state entries the nodes calibrate.

    ``T2ramsey`` is the 1/e time of the shot-averaged Ramsey envelope exp(-t/T2echo - (t/Tq)^2),
    not the time constant of an exponential.

    ``x180.amplitude`` is TRUE_PI_AMPLITUDE, which belongs to TRUE_DEVICE and the 40 ns,
    sigma = 8 ns pulse; it is not recomputed for another ``device``.
    """
    return {
        "resonator.f_01": float(device.f_res_g),
        "readout.frequency": float(device.f_res_g - device.chi),
        "chi": float(-device.chi),  # QM's sign: (f_res_e - f_res_g) / 2
        "f_01": float(device.f_01),
        "x180.amplitude": float(TRUE_PI_AMPLITUDE),
        "T1": float(device.T1),
        "T2ramsey": float(device.T2ramsey),
        "T2echo": float(device.T2echo),
        "time_of_flight": float(device.time_of_flight),
        "adc_offset_I": float(device.adc_offsets[0]),
        "adc_offset_Q": float(device.adc_offsets[1]),
    }


def readout_reference(state: dict, n_shots: int = 100_000, seed: int = 0) -> dict[str, float]:
    """High-shot IQ blobs with the true pi pulse and the true readout tone, analysed as 07_iq_blobs does.

    The simulator reference is a property of the simulated device, not of the run (Ruling AB), so
    the readout tone sits at the true g/e midpoint (``TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi``,
    where 08a's separation peaks) and the
    drive at the true qubit frequency with TRUE_PI_AMPLITUDE. The other readout settings (readout
    amplitude and length) and the x180 length and sigma come from ``state``. Miscalibrated
    frequencies or x180 amplitude in ``state`` therefore do not enter the reference.
    """
    return _readout_blob_fit(state, TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi, n_shots, seed)


def blob_reference_at_tone(state: dict, readout_frequency: float, n_shots: int = 100_000, seed: int = 0) -> dict[str, float]:
    """The same high-shot 07 analysis read out at ``readout_frequency`` (the run's own tone).

    For figures only: the weight angle and threshold turn with the readout frequency (25 deg for the
    144 kHz between 08a's seed-2026 pick and the g/e midpoint), so the 07 card compares them with
    this rather than with ``readout_reference``'s midpoint tone.
    """
    return _readout_blob_fit(state, readout_frequency, n_shots, seed)


def _readout_blob_fit(state: dict, readout_frequency: float, n_shots: int, seed: int) -> dict[str, float]:
    """The 07 analysis of true-pi-pulse single shots read out at ``readout_frequency``."""
    # Imported here so the simulator package does not depend on calibration_utils at import time
    # (calibration_utils.common imports simulator.qpu).
    from calibration_utils.common import QUBIT, readout_settings
    from calibration_utils.iq_blobs import fit_raw_data

    qubit = state["qubits"][QUBIT]
    x180 = qubit["xy"]["operations"]["x180"]
    # Ideal alignment: the true time of flight and the true ADC offsets (device properties, Ruling AB).
    settings = readout_settings(state).with_frequency(readout_frequency).replace(
        time_of_flight=int(TRUE_DEVICE.time_of_flight), input_offsets=tuple(TRUE_DEVICE.adc_offsets)
    )
    readout = {"readout": settings, "drive_frequency": TRUE_DEVICE.f_01, "n_shots": n_shots}
    qpu = SimulatedQPU(seed=seed)
    I_g, Q_g = qpu.single_shots([], **readout)
    I_e, Q_e = qpu.single_shots([Pulse(TRUE_PI_AMPLITUDE, x180["length"], x180["sigma"])], **readout)
    dims = ("qubit", "n_runs")
    ds = xr.Dataset(
        {name: (dims, np.asarray(v)[None, :]) for name, v in (("I_g", I_g), ("Q_g", Q_g), ("I_e", I_e), ("Q_e", Q_e))},
        coords={"qubit": [QUBIT], "n_runs": np.arange(1, n_shots + 1)},
    )
    fit = fit_raw_data(ds)
    return {key: float(fit[key]) for key in ("fidelity", "ge_threshold", "iw_angle")}


def readout_levels(state: dict, qubit: str = "q1") -> tuple[complex, complex]:
    """Noiseless averaged I + jQ (volts) for a prepared |g> and |e> at the state's readout settings.

    Truth-only helper for plotting (for example the ideal blob centres or signal levels on a
    node figure). It is the simulator's own readout (``SimulatedQPU.expected_iq`` for the true
    device) with the readout settings of ``qubit`` in ``state``: pure states, decay of |e> during the
    window included (it is part of every measured mean), no thermal population, no noise. Fits must
    not use it; calibration code only sees measured data.
    """
    from calibration_utils.common import readout_settings

    settings = readout_settings(state, qubit)
    qpu = SimulatedQPU()
    return qpu.expected_iq(0, settings, decay=True), qpu.expected_iq(1, settings, decay=True)


def build_reference(state: dict) -> dict[str, float]:
    """True device values plus the simulator readout reference, for ``NodeContext.reference``."""
    return {**true_values(), **readout_reference(state)}
