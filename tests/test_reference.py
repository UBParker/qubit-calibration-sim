"""Reference values for figures, and the guard that calibration code never reads them."""

import copy
import math

import numpy as np
import pytest

from calibration_utils.common import REPO_ROOT, load_node
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE, Device
from simulator.reference import build_reference, readout_reference, true_values

NODE_FILES = [
    "01a_time_of_flight.py",
    "02a_resonator_spectroscopy.py",
    "03a_qubit_spectroscopy.py",
    "04b_power_rabi.py",
    "07_iq_blobs.py",
    "06a_ramsey.py",
    "05_T1.py",
    "06b_echo.py",
    "08a_readout_frequency_optimization.py",
]

# Assignment fidelity from the seed-2026 bring-up (time-domain readout at the |g> resonance:
# ring-up and decay in the 1 us window give 0.961).
BRINGUP_FIDELITY = 0.961


def test_true_values_match_device():
    assert true_values() == {
        "resonator.f_01": TRUE_DEVICE.f_res_g,
        "readout.frequency": TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi,
        "chi": -TRUE_DEVICE.chi,
        "f_01": TRUE_DEVICE.f_01,
        "x180.amplitude": TRUE_PI_AMPLITUDE,
        "T1": TRUE_DEVICE.T1,
        "T2ramsey": TRUE_DEVICE.T2ramsey,
        "T2echo": TRUE_DEVICE.T2echo,
        "time_of_flight": TRUE_DEVICE.time_of_flight,
        "adc_offset_I": TRUE_DEVICE.adc_offsets[0],
        "adc_offset_Q": TRUE_DEVICE.adc_offsets[1],
    }
    other = true_values(Device(T1=10_000.0, f_01=5.0e9))
    assert other["T1"] == 10_000.0
    assert other["f_01"] == 5.0e9


def test_node_context_reference_defaults_to_none(calibrated_state, make_ctx):
    assert make_ctx(calibrated_state).reference is None


def test_readout_reference_close_to_bringup(calibrated_state):
    ref = readout_reference(calibrated_state, n_shots=20_000, seed=0)
    assert set(ref) == {"fidelity", "ge_threshold", "iw_angle"}
    assert all(isinstance(v, float) for v in ref.values())
    assert ref["fidelity"] == pytest.approx(BRINGUP_FIDELITY, abs=0.01)


def test_build_reference_is_union(calibrated_state, monkeypatch):
    import simulator.reference as reference

    fake = {"fidelity": 0.5, "ge_threshold": 1.0, "iw_angle": 0.1}
    monkeypatch.setattr(reference, "readout_reference", lambda state: fake)
    assert build_reference(calibrated_state) == {**true_values(), **fake}


def _same(a, b) -> bool:
    """Equality that treats NaN as equal to NaN, through nested dicts, lists and arrays."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_same_is_nan_aware():
    assert _same({"x": float("nan"), "y": [1.0, np.nan]}, {"x": float("nan"), "y": [1.0, np.nan]})
    assert not _same({"x": 1.0}, {"x": 2.0})
    assert not _same({"x": float("nan")}, {"x": 1.0})


@pytest.mark.parametrize("filename", NODE_FILES)
def test_fits_ignore_reference(filename, calibrated_state, make_ctx, monkeypatch):
    """Fits do not change when the reference is set. Truth-only helpers (simulator.reference's
    true_values, readout_reference, readout_levels and SimulatedQPU.expected_iq) never appear in a node."""
    source = (REPO_ROOT / "calibrations" / filename).read_text()
    assert "expected_iq" not in source and "readout_levels" not in source
    import simulator.reference as reference

    # Small readout reference keeps this fast; the values only need to be present, not precise.
    monkeypatch.setattr(reference, "readout_reference", lambda state: readout_reference(state, n_shots=2_000))
    ref = build_reference(calibrated_state)
    # calibrated_state holds the true values, so a node reading the real reference instead of the
    # state would not change its fit. A perturbed sentinel with the same keys exposes that leak.
    sentinel = {key: value * 1.37 + 0.123 for key, value in ref.items()}
    node = load_node(filename)

    def run_with(reference):
        ctx = make_ctx(copy.deepcopy(calibrated_state), seed=7)
        ctx.reference = reference
        return node.run(ctx), ctx.state

    fit_none, state_none = run_with(None)
    for label, reference in (("sentinel", sentinel), ("real", ref)):
        fit, state = run_with(reference)
        assert _same(fit, fit_none), f"{node.NODE_NAME} fit changed with the {label} reference"
        assert _same(state, state_none), f"{node.NODE_NAME} state update changed with the {label} reference"


def test_readout_reference_uses_true_resonator_frequency(rough_state):
    # Ruling AB: the simulator reference belongs to the device, not to the run, so the readout tone
    # sits at the true |g> resonator frequency whatever the state stores.
    import simulator.reference as reference

    at_true = copy.deepcopy(rough_state)
    at_true["qubits"]["q1"]["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    assert rough_state["qubits"]["q1"]["resonator"]["f_01"] != TRUE_DEVICE.f_res_g
    ref_rough = readout_reference(rough_state, n_shots=5_000, seed=3)
    ref_true = readout_reference(at_true, n_shots=5_000, seed=3)
    assert ref_rough == ref_true
    # The old behaviour (tone at the state's frequency) gives a very different weight angle.
    detuned = reference._readout_blob_fit(
        rough_state, rough_state["qubits"]["q1"]["resonator"]["f_01"], n_shots=5_000, seed=3
    )
    angle = math.remainder(detuned["iw_angle"] - ref_true["iw_angle"], 2 * math.pi)
    assert abs(angle) > math.radians(45)


def test_readout_levels_match_expected_iq(calibrated_state):
    from calibration_utils.common import readout_settings
    from simulator.qpu import SimulatedQPU
    from simulator.reference import readout_levels

    level_g, level_e = readout_levels(calibrated_state)
    assert isinstance(level_g, complex) and isinstance(level_e, complex)
    ro = readout_settings(calibrated_state)
    qpu = SimulatedQPU()
    assert level_g == qpu.expected_iq(0, ro, decay=True) and level_e == qpu.expected_iq(1, ro, decay=True)
    assert abs(level_e - level_g) > 0
    # Follows the state's readout settings, not the true device.
    shifted = copy.deepcopy(calibrated_state)
    shifted["qubits"]["q1"]["resonator"]["f_01"] += 2e6
    assert readout_levels(shifted) != (level_g, level_e)
    # A 07 fit's blob means sit close to these levels (both include decay in the readout window).
    fit = load_node("07_iq_blobs.py").run(_ctx(calibrated_state))
    g_fit = complex(*fit["g_mean_iq"])
    e_fit = complex(*fit["e_mean_iq"])
    span = abs(level_e - level_g)
    assert abs(g_fit - level_g) < 0.1 * span
    assert abs(e_fit - level_e) < 0.1 * span


def _ctx(state):
    from calibration_utils.common import NodeContext
    from simulator.qpu import SimulatedQPU

    return NodeContext(state=copy.deepcopy(state), qpu=SimulatedQPU(seed=0), save_outputs=False)
