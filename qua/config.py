"""QM configuration dictionary generated from the calibrated state.

This follows the OPX+ configuration format of QM's single fixed-frequency transmon example in qua-libs
(BSD-3-Clause, https://github.com/qua-platform/qua-libs/blob/main/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon/configuration.py),
including its integration-weight names: one IQ-mixed drive
element for the qubit, one IQ-mixed measured element for the readout resonator, Gaussian
control pulses, and a constant readout pulse with cosine/sine integration weights. In
QUAlibrate this dictionary is produced by ``machine.generate_config()`` from QUAM.
"""

from __future__ import annotations

import numpy as np

CLOCK_CYCLE_NS = 4


def gaussian_samples(amplitude: float, length: int, sigma: float) -> list[float]:
    t = np.arange(length)
    centre = (length - 1) / 2
    g = np.exp(-((t - centre) ** 2) / (2 * sigma**2))
    edge = np.exp(-(centre**2) / (2 * sigma**2))
    return list(amplitude * (g - edge) / (1 - edge))


def build_config(state: dict, qubit: str = "q1") -> dict:
    q = state["qubits"][qubit]
    xy = q["xy"]
    res = q["resonator"]
    x180 = xy["operations"]["x180"]
    sat = xy["operations"]["saturation"]
    ro = res["operations"]["readout"]
    angle = ro.get("integration_weights_angle", 0.0) or 0.0
    ro_len = int(ro["length"])

    qubit_if = q["f_01"] - xy["LO_frequency"]
    res_if = res["f_01"] - res["LO_frequency"]

    return {
        "controllers": {
            "con1": {
                "analog_outputs": {i: {"offset": 0.0} for i in range(1, 5)},
                "digital_outputs": {},
                "analog_inputs": {
                    1: {"offset": float(res.get("opx_input_I", {}).get("offset", 0.0)), "gain_db": 0},
                    2: {"offset": float(res.get("opx_input_Q", {}).get("offset", 0.0)), "gain_db": 0},
                },
            }
        },
        "elements": {
            "qubit": {
                "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2), "lo_frequency": xy["LO_frequency"], "mixer": "mixer_qubit"},
                "intermediate_frequency": qubit_if,
                "operations": {
                    "x180": "x180_pulse",
                    "x90": "x90_pulse",
                    "-x90": "-x90_pulse",
                    "y90": "y90_pulse",
                    "saturation": "saturation_pulse",
                },
            },
            "resonator": {
                "mixInputs": {"I": ("con1", 3), "Q": ("con1", 4), "lo_frequency": res["LO_frequency"], "mixer": "mixer_resonator"},
                "intermediate_frequency": res_if,
                "operations": {"readout": "readout_pulse"},
                "outputs": {"out1": ("con1", 1), "out2": ("con1", 2)},
                "time_of_flight": int(res.get("time_of_flight", 24)),
                "smearing": int(res.get("smearing", 0)),
            },
        },
        "pulses": {
            **{
                name: {
                    "operation": "control",
                    "length": int(x180["length"]),
                    "waveforms": waves,
                }
                for name, waves in {
                    "x180_pulse": {"I": "x180_wf", "Q": "zero_wf"},
                    "x90_pulse": {"I": "x90_wf", "Q": "zero_wf"},
                    "-x90_pulse": {"I": "minus_x90_wf", "Q": "zero_wf"},
                    "y90_pulse": {"I": "zero_wf", "Q": "x90_wf"},
                }.items()
            },
            "saturation_pulse": {
                "operation": "control",
                "length": int(sat["length"]),
                "waveforms": {"I": "saturation_wf", "Q": "zero_wf"},
            },
            "readout_pulse": {
                "operation": "measurement",
                "length": ro_len,
                "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                    "rotated_cos": "rotated_cosine_weights",
                    "rotated_sin": "rotated_sine_weights",
                    "rotated_minus_sin": "rotated_minus_sine_weights",
                },
                "digital_marker": "ON",
            },
        },
        "waveforms": {
            "zero_wf": {"type": "constant", "sample": 0.0},
            "saturation_wf": {"type": "constant", "sample": sat["amplitude"]},
            "readout_wf": {"type": "constant", "sample": ro["amplitude"]},
            "x180_wf": {"type": "arbitrary", "samples": gaussian_samples(x180["amplitude"], int(x180["length"]), x180["sigma"])},
            "x90_wf": {"type": "arbitrary", "samples": gaussian_samples(x180["amplitude"] / 2, int(x180["length"]), x180["sigma"])},
            "minus_x90_wf": {"type": "arbitrary", "samples": gaussian_samples(-x180["amplitude"] / 2, int(x180["length"]), x180["sigma"])},
        },
        "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
        "integration_weights": {
            "cosine_weights": {"cosine": [(1.0, ro_len)], "sine": [(0.0, ro_len)]},
            "sine_weights": {"cosine": [(0.0, ro_len)], "sine": [(1.0, ro_len)]},
            "minus_sine_weights": {"cosine": [(0.0, ro_len)], "sine": [(-1.0, ro_len)]},
            "rotated_cosine_weights": {"cosine": [(np.cos(angle), ro_len)], "sine": [(-np.sin(angle), ro_len)]},
            "rotated_sine_weights": {"cosine": [(np.sin(angle), ro_len)], "sine": [(np.cos(angle), ro_len)]},
            "rotated_minus_sine_weights": {"cosine": [(-np.sin(angle), ro_len)], "sine": [(-np.cos(angle), ro_len)]},
        },
        "mixers": {
            "mixer_qubit": [{"intermediate_frequency": qubit_if, "lo_frequency": xy["LO_frequency"], "correction": (1, 0, 0, 1)}],
            "mixer_resonator": [{"intermediate_frequency": res_if, "lo_frequency": res["LO_frequency"], "correction": (1, 0, 0, 1)}],
        },
    }
