"""Each node's QUA program and its run_simulated describe the same experiment.

For every node and both states, the program is traced (qua/trace.py) and turned into the
SimulatedQPU calls it implies (calibration_utils/qua_bridge.py); run_simulated is run against a
recording stub and its calls must match: same methods in order, same sweep, pulses (amplitude,
phase, length), waits, frequencies (to 1 Hz) and readout. What the simulator knowingly ignores
is a key of SIMULATOR_APPROXIMATIONS, pinned per node here:

    01a  readout_empty_resonator, undriven_qubit_stays_thermal
    02a  integer_if, readout_empty_resonator, undriven_qubit_stays_thermal
    03a  integer_if, passive_reset_complete, readout_empty_resonator, saturation_steady_state
    04b  continuous_envelope, passive_reset_complete, readout_empty_resonator
    05   continuous_envelope, passive_reset_complete, readout_empty_resonator
    06a  continuous_envelope, passive_reset_complete, readout_empty_resonator
    06b  continuous_envelope, passive_reset_complete, readout_empty_resonator
    07   continuous_envelope, passive_reset_complete, readout_empty_resonator, shot_groups_batched,
         undriven_qubit_stays_thermal
    08a  as 07, plus integer_if (the readout frequency sweep)

02a gets undriven_qubit_stays_thermal because the qubit idles (never driven) while the resonator
rings down between readouts; those idles are dropped.
"""

import copy

import numpy as np
import pytest
from qm import QuantumMachinesManager

from calibration_utils.common import REPO_ROOT, NodeContext, load_node
from calibration_utils.qua_bridge import calls_from_program
from simulator.qpu import SimulatedQPU
from tests.qua_check import assert_calls_match
from tests.recording_qpu import RECORDED_METHODS, RecordingQPU

NODE_FILES = sorted(p.name for p in (REPO_ROOT / "calibrations").glob("[0-8]*.py"))

EXPECTED_APPROXIMATIONS = {
    "01a_time_of_flight.py": {"readout_empty_resonator", "undriven_qubit_stays_thermal"},
    "08a_readout_frequency_optimization.py": {
        "readout_empty_resonator",
        "integer_if",
        "passive_reset_complete",
        "continuous_envelope",
        "undriven_qubit_stays_thermal",
        "shot_groups_batched",
    },
    "02a_resonator_spectroscopy.py": {"readout_empty_resonator", "integer_if", "undriven_qubit_stays_thermal"},
    "03a_qubit_spectroscopy.py": {"readout_empty_resonator", "integer_if", "passive_reset_complete", "saturation_steady_state"},
    "04b_power_rabi.py": {"readout_empty_resonator", "passive_reset_complete", "continuous_envelope"},
    "05_T1.py": {"readout_empty_resonator", "passive_reset_complete", "continuous_envelope"},
    "06a_ramsey.py": {"readout_empty_resonator", "passive_reset_complete", "continuous_envelope"},
    "06b_echo.py": {"readout_empty_resonator", "passive_reset_complete", "continuous_envelope"},
    "07_iq_blobs.py": {
        "readout_empty_resonator",
        "passive_reset_complete",
        "continuous_envelope",
        "undriven_qubit_stays_thermal",
        "shot_groups_batched",
    },
}

# Simulator-only keyword arguments (SIMULATOR_ONLY_KWARGS) each node passes; none today.
EXPECTED_SIMULATOR_ONLY: dict[str, dict] = {}


@pytest.fixture(scope="module", autouse=True)
def offline_capabilities():
    QuantumMachinesManager.set_capabilities_offline()


def _state(name, rough_state, calibrated_state):
    return rough_state if name == "rough" else calibrated_state


def _recorded(node, state, params):
    qpu = RecordingQPU()
    node.run_simulated(NodeContext(state=copy.deepcopy(state), qpu=qpu, save_outputs=False), params)
    return qpu.calls


@pytest.mark.parametrize("filename", NODE_FILES)
@pytest.mark.parametrize("state_name", ["rough", "calibrated"])
def test_program_matches_simulator_calls(filename, state_name, rough_state, calibrated_state):
    node = load_node(filename)
    state = _state(state_name, rough_state, calibrated_state)
    params = node.Parameters()
    # 01a runs its program with QM's temporary overrides (readout length and amplitude): the config
    # it runs with is built from that patched state.
    ran_with = node.patched_state(state, params) if hasattr(node, "patched_state") else state
    expected = calls_from_program(node.create_qua_program(ran_with, params), ran_with)
    simulator_only = assert_calls_match(_recorded(node, state, params), expected, node=filename)
    assert simulator_only == EXPECTED_SIMULATOR_ONLY.get(filename, {})
    keys = {a.key for call in expected for a in call.approximations}
    assert keys == EXPECTED_APPROXIMATIONS[filename]


def test_every_node_is_pinned():
    assert set(EXPECTED_APPROXIMATIONS) == set(NODE_FILES)


def test_recording_qpu_mirrors_simulated_qpu():
    import inspect

    for method in RECORDED_METHODS:
        assert callable(getattr(SimulatedQPU, method)), method
    qpu = RecordingQPU()
    params = [p for p in inspect.signature(SimulatedQPU.single_shots).parameters if p != "self"]
    I, Q = qpu.single_shots([], 5e9, "readout", 3, new_knob=1)
    assert qpu.calls == [("single_shots", {**dict(zip(params, [[], 5e9, "readout", 3])), "new_knob": 1})]
    assert I.shape == Q.shape == (3,)
    with pytest.raises(TypeError):
        qpu.run_sequences([[]], 5e9)  # missing arguments fail as on the real QPU


def test_recording_qpu_names_a_missing_size_argument(monkeypatch):
    import tests.recording_qpu as recording

    monkeypatch.setattr(recording, "_SIZE_ARGUMENTS", ())
    with pytest.raises(TypeError, match="no argument sets the number of points"):
        RecordingQPU().single_shots([], 5e9, "readout", 3)


# ---------------------------------------------------------------- the check catches mutations
def test_check_catches_a_swapped_pulse(calibrated_state, monkeypatch):
    from calibration_utils.common import x90_pulse

    t1 = load_node("05_T1.py")
    params = t1.Parameters()
    expected = calls_from_program(t1.create_qua_program(calibrated_state, params), calibrated_state)
    monkeypatch.setattr(t1, "x180_pulse", lambda state, *args, **kwargs: x90_pulse(state))
    with pytest.raises(AssertionError, match="amplitude"):
        assert_calls_match(_recorded(t1, calibrated_state, params), expected)


def test_check_catches_a_flipped_frame_sign(calibrated_state):
    from calibration_utils.common import readout_settings, x90_pulse
    from simulator.transmon import Idle

    ramsey = load_node("06a_ramsey.py")
    params = ramsey.Parameters()
    expected = calls_from_program(ramsey.create_qua_program(calibrated_state, params), calibrated_state)

    def run_simulated_flipped(ctx, params):  # 06a's run_simulated with the frame sign flipped
        times = ramsey.idle_times_ns(params)
        detuning = params.frequency_detuning_in_mhz * 1e6
        sequences = [
            [x90_pulse(ctx.state), Idle(float(t)), x90_pulse(ctx.state, phase=+sign * 2 * np.pi * detuning * t * 1e-9)]
            for t in times
            for sign in ramsey.DETUNING_SIGNS
        ]
        ctx.qpu.run_sequences(sequences, drive_frequency=ctx.qubit["f_01"], n_avg=params.num_shots, readout=readout_settings(ctx.state))

    qpu = RecordingQPU()
    run_simulated_flipped(NodeContext(state=calibrated_state, qpu=qpu, save_outputs=False), params)
    with pytest.raises(AssertionError, match="phase"):
        assert_calls_match(qpu.calls, expected)


def test_check_catches_a_wrong_sweep(calibrated_state):
    t1 = load_node("05_T1.py")
    expected = calls_from_program(t1.create_qua_program(calibrated_state, t1.Parameters()), calibrated_state)
    with pytest.raises(AssertionError, match="60 sequences, the program implies 61"):
        assert_calls_match(_recorded(t1, calibrated_state, t1.Parameters(num_points=60)), expected)


def test_check_catches_an_unregistered_kwarg(calibrated_state):
    t1 = load_node("05_T1.py")
    params = t1.Parameters()
    expected = calls_from_program(t1.create_qua_program(calibrated_state, params), calibrated_state)
    (method, kwargs), = _recorded(t1, calibrated_state, params)
    with pytest.raises(AssertionError, match="dephasing"):
        assert_calls_match([(method, {**kwargs, "dephasing": "echo"})], expected)


def test_check_catches_a_wrong_readout_frequency(calibrated_state):
    t1 = load_node("05_T1.py")
    params = t1.Parameters()
    expected = calls_from_program(t1.create_qua_program(calibrated_state, params), calibrated_state)
    (method, kwargs), = _recorded(t1, calibrated_state, params)
    wrong = kwargs["readout"].with_frequency(kwargs["readout"].frequency + 5.0)
    with pytest.raises(AssertionError, match="readout frequency"):
        assert_calls_match([(method, {**kwargs, "readout": wrong})], expected)


def test_check_catches_a_wrong_time_of_flight(calibrated_state):
    t1 = load_node("05_T1.py")
    params = t1.Parameters()
    expected = calls_from_program(t1.create_qua_program(calibrated_state, params), calibrated_state)
    (method, kwargs), = _recorded(t1, calibrated_state, params)
    with pytest.raises(AssertionError, match="time_of_flight"):
        assert_calls_match([(method, {**kwargs, "readout": kwargs["readout"].replace(time_of_flight=24)})], expected)
