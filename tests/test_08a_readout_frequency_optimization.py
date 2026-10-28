"""08a readout frequency optimization: closure on the g/e midpoint, QM's trailing-window offset,
the chi estimate, and the 07 fidelity gain.

Measured over seeds 0 to 299 from the calibrated state (readout at f_res_g): pick minus the true
midpoint f_res_g - chi has mean +201 kHz, sd 93 kHz, range -50 to +450 kHz (spec prototype +208,
sd 91); chi estimates -650 to -450 kHz (true -550 kHz in QM's sign).
"""

import copy

import numpy as np
import pytest
import xarray as xr

from calibration_utils import readout_frequency_optimization as analysis
from calibration_utils.common import QUBIT, load_node, readout_settings, x180_pulse
from simulator.device import TRUE_DEVICE
from simulator.qpu import SimulatedQPU
from simulator.reference import CLOSURE_TOLERANCES
from tests.conftest import assert_export_uses_pre_update_state

node = load_node("08a_readout_frequency_optimization.py")
iq_blobs = load_node("07_iq_blobs.py")
MIDPOINT = TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi
BAND = CLOSURE_TOLERANCES["readout.frequency"]["abs"]


def _separation(state, frequencies) -> np.ndarray:
    """Noiseless |<g> - <e>| for pure states, |e> decay in the window included."""
    qpu = SimulatedQPU()
    readout = readout_settings(state)
    return np.array([abs(qpu.expected_iq(1, readout, frequency=f) - qpu.expected_iq(0, readout, frequency=f)) for f in frequencies])


@pytest.fixture(scope="module")
def runs():
    """08a from the calibrated state on seeds 1, 2, 3: (state before, state after, fit) each."""
    from tests.conftest import apply_setup
    from calibration_utils.common import INITIAL_STATE_PATH, NodeContext, load_state
    from simulator.device import TRUE_PI_AMPLITUDE

    state = load_state(INITIAL_STATE_PATH)
    q = state["qubits"][QUBIT]
    q["f_01"] = TRUE_DEVICE.f_01
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    out = {}
    for seed in (1, 2, 3):
        ctx = NodeContext(state=copy.deepcopy(state), qpu=SimulatedQPU(seed=seed), save_outputs=False)
        fit = node.run(ctx)
        out[seed] = (state, ctx.state, fit)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_readout_frequency_closure(runs, seed):
    before, after, fit = runs[seed]
    assert fit["success"]
    f_new = after["qubits"][QUBIT]["resonator"]["f_01"]
    assert abs(f_new - MIDPOINT) <= BAND
    scan = MIDPOINT + np.arange(-1.5e6, 1.5e6 + 1, 5e3)
    assert _separation(before, [f_new])[0] >= 0.84 * _separation(before, scan).max()
    assert after["qubits"][QUBIT]["extras"]["chi_source"] == node.NODE_NAME


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_chi_estimate(runs, seed):
    _, after, fit = runs[seed]
    assert abs(fit["chi"] + TRUE_DEVICE.chi) <= CLOSURE_TOLERANCES["chi"]["abs"]
    assert after["qubits"][QUBIT]["chi"] == fit["chi"]


def test_separation_at_band_edge(calibrated_state):
    """The closure band edge keeps 0.85 of the maximum separation (the spec's prototype figure)."""
    scan = MIDPOINT + np.arange(-1.5e6, 1.5e6 + 1, 5e3)
    peak = _separation(calibrated_state, scan).max()
    edges = _separation(calibrated_state, [MIDPOINT - BAND, MIDPOINT + BAND])
    assert np.all(edges >= 0.84 * peak) and np.all(edges <= 0.87 * peak)


def test_true_optimum_is_the_midpoint(calibrated_state):
    scan = MIDPOINT + np.arange(-1.5e6, 1.5e6 + 1, 5e3)
    assert abs(scan[np.argmax(_separation(calibrated_state, scan))] - MIDPOINT) <= 10e3


def test_qm_trailing_window_offset_noise_free(calibrated_state):
    """Pins QM's behaviour, not a simulator property: on noiseless data the trailing 5-point mean
    puts the pick exactly 2 steps above the maximum of D."""
    params = node.Parameters()
    dfs = node.detunings_hz(params)
    f0 = calibrated_state["qubits"][QUBIT]["resonator"]["f_01"]
    qpu = SimulatedQPU()
    readout = readout_settings(calibrated_state)
    pops = qpu.populations([[], [x180_pulse(calibrated_state)]], calibrated_state["qubits"][QUBIT]["f_01"])
    levels = np.array([[qpu.expected_iq(s, readout, frequency=f0 + df) for s in range(3)] for df in dfs])  # (n, 3)
    z_g, z_e = levels @ pops[:, 0], levels @ pops[:, 1]
    dims = ("qubit", "detuning")
    ds = xr.Dataset(
        {"I_g": (dims, z_g.real[None]), "Q_g": (dims, z_g.imag[None]), "I_e": (dims, z_e.real[None]), "Q_e": (dims, z_e.imag[None])},
        coords={"qubit": [QUBIT], "detuning": dfs, "full_freq": ("detuning", f0 + dfs)},
    )
    ds = analysis.process_raw_dataset(ds)
    fit = analysis.fit_raw_data(ds)
    D = ds.D.sel(qubit=QUBIT).values
    step = dfs[1] - dfs[0]
    assert fit["optimal_detuning"] == dfs[np.argmax(D)] + 2 * step


def test_fidelity_improves_after_08a(runs):
    """07 at 08a's pick beats 07 at f_res_g: measured +0.0061, +0.0114, +0.0055 (prototype about +0.009)."""
    from calibration_utils.common import NodeContext

    gains = []
    for seed, (before, after, _) in runs.items():
        fid = []
        for state in (before, after):
            ctx = NodeContext(state=copy.deepcopy(state), qpu=SimulatedQPU(seed=seed), save_outputs=False)
            fid.append(iq_blobs.run(ctx)["fidelity"])
        gains.append(fid[1] - fid[0])
    assert np.mean(gains) >= 0.005, gains


def test_failed_fit_leaves_state_untouched(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    before = ctx.snapshot()
    node.update_state(ctx, {"success": False, "optimal_frequency": 1.0, "chi": 1.0})
    assert ctx.state == before


def test_08a_export_uses_pre_update_state(calibrated_state, make_ctx):
    fit = assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(calibrated_state), 0))
    assert fit["success"]
