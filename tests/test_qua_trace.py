"""Unit tests for qua/trace.py (the QUA protobuf evaluator) and calibration_utils/qua_bridge.py.

Programs are built inline with qm.qua against the config from qua/config.py.
"""

import importlib.metadata

import numpy as np
import pytest
from qm import QuantumMachinesManager
from qm.qua import (
    Cast,
    align,
    amp,
    assign,
    declare,
    declare_stream,
    else_,
    fixed,
    for_,
    for_each_,
    frame_rotation,
    frame_rotation_2pi,
    if_,
    measure,
    pause,
    play,
    program,
    reset_frame,
    save,
    strict_timing_,
    stream_processing,
    wait,
)

from calibration_utils.common import load_node
from qua.config import build_config
from qua.macros import readout_iq
from qua.trace import (
    AlignEvent,
    MeasureEvent,
    PlayEvent,
    SaveEvent,
    StreamSpec,
    UnsupportedQuaError,
    trace_program,
)


@pytest.fixture(scope="module", autouse=True)
def offline_capabilities():
    QuantumMachinesManager.set_capabilities_offline()


@pytest.fixture
def config(calibrated_state):
    return build_config(calibrated_state)


def _shot_loop(n_shots: int = 1):
    """Context manager body helper: the outer averaging loop every node uses."""
    n = declare(int)
    return n, for_(n, 0, n < n_shots, n + 1)


# ---------------------------------------------------------------- pinned qm-qua surface
PINNED_ONEOFS = {
    "AnyStatement": "statement_oneof",
    "AnyScalarExpression": "expression_oneof",
    "PlayStatement": "pulseType",
    "MeasureProcess": "process",
    "AnalogMeasureProcess": "process",
    "AnalogProcessTarget": "processTarget",
}
PINNED_FIELDS = {
    "PlayStatement": ["qe", "namedPulse", "amp", "duration", "condition", "chirp", "truncate", "timestampLabel", "targetInput", "port_condition"],
    "AmpMultiplier": ["v0", "v1", "v2", "v3"],
    "MeasureStatement": ["qe", "pulse", "amp", "measureProcesses", "streamAs", "timestampLabel"],
    "AnalogMeasureProcess.DualDemodIntegration": ["integration1", "integration2", "target", "elementOutput1", "elementOutput2"],
    "ForStatement": ["init", "condition", "update", "body"],
    "ForEachStatement": ["iterator", "body"],
    "ForEachStatement.VariableWithValues": ["variable", "array"],
    "IfStatement": ["condition", "body", "else", "elseifs"],
    "AssignmentStatement": ["target", "expression"],
    "WaitStatement": ["qe", "time"],
    "AlignStatement": ["qe"],
    "ZRotationStatement": ["qe", "value"],
    "ResetFrameStatement": ["qe"],
    "UpdateFrequencyStatement": ["qe", "value", "units", "keepPhase"],
    "SaveStatement": ["tag", "source"],
    "LibFunctionExpression": ["libraryName", "functionName", "arguments"],
    "VarDeclaration": ["name", "type", "size", "dim", "value"],
    "LiteralExpression": ["value", "type"],
    "Script": ["variables", "body"],
}
PINNED_ENUMS = {
    "Type": {"INT", "BOOL", "REAL"},
    "BinaryExpression.BinaryOperator": {"ADD", "SUB", "MULT", "DIV", "AND", "OR", "LT", "LET", "GT", "GET", "EQ"},
    "UpdateFrequencyStatement.Units": {"Hz"},
}
SURFACE_MESSAGE = "qm-qua schema changed; re-verify qua/trace.py against the new inc_qua_pb2 before bumping the pin"


def _nested(root, dotted):
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


def test_qm_qua_surface_is_pinned():
    from qm.grpc.qm.pb import inc_qua_pb2
    from qm.program import Program

    qp = inc_qua_pb2.QuaProgram
    assert importlib.metadata.version("qm-qua").startswith("1.4."), SURFACE_MESSAGE
    assert isinstance(Program.__dict__["qua_program"], property), SURFACE_MESSAGE
    assert {"script", "resultAnalysis"} <= set(qp.DESCRIPTOR.fields_by_name), SURFACE_MESSAGE
    assert "model" in qp.DESCRIPTOR.fields_by_name["resultAnalysis"].message_type.fields_by_name, SURFACE_MESSAGE
    for message, oneof in PINNED_ONEOFS.items():
        assert oneof in _nested(qp, message).DESCRIPTOR.oneofs_by_name, f"{SURFACE_MESSAGE}: {message}.{oneof}"
    for message, fields in PINNED_FIELDS.items():
        have = set(_nested(qp, message).DESCRIPTOR.fields_by_name)
        assert set(fields) <= have, f"{SURFACE_MESSAGE}: {message} lacks {set(fields) - have}"
    for enum, names in PINNED_ENUMS.items():
        assert names <= set(_nested(qp, enum).keys()), f"{SURFACE_MESSAGE}: {enum}"


# ---------------------------------------------------------------- trace
def test_shot_loop_collapsed(config):
    with program() as prog:
        n = declare(int)
        t = declare(int)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 7, n + 1):
            with for_each_(t, [4, 8, 12]):
                play("x180", "qubit")
                wait(t, "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(3).average().save("I")
            Q_st.buffer(3).average().save("Q")
    trace = trace_program(prog, config)
    assert trace.shots == 7
    measures = [e for e in trace.events if isinstance(e, MeasureEvent)]
    assert len(measures) == 3
    assert measures[0].demods[0].weight1 == "cos" and measures[0].demods[1].weight1 == "minus_sin"


def test_state_carried_between_shots_raises(config):
    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 2, n + 1):
            frame_rotation_2pi(0.1, "qubit")  # no reset_frame: the second shot plays at 0.2 turns
            play("x90", "qubit")
    with pytest.raises(UnsupportedQuaError, match="shot passes differ"):
        trace_program(prog, config)


def test_shot_loop_required(config):
    with program() as prog:
        play("x180", "qubit")
    with pytest.raises(UnsupportedQuaError, match="outer averaging loop"):
        trace_program(prog, config)


def _assert_unsupported(prog, config, construct):
    with pytest.raises(UnsupportedQuaError) as err:
        trace_program(prog, config)
    assert construct in str(err.value)
    assert "line" in str(err.value)


def test_unsupported_statement_raises(config):
    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 2, n + 1):
            with strict_timing_():
                play("x180", "qubit")
    _assert_unsupported(prog, config, "strictTiming")

    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 2, n + 1):
            pause()
    _assert_unsupported(prog, config, "pause")

    with program() as prog:
        n = declare(int)
        flag = declare(bool, value=True)
        with for_(n, 0, n < 2, n + 1):
            play("x180", "qubit", condition=flag)
    _assert_unsupported(prog, config, "condition")

    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 2, n + 1):
            play("x180" * amp(1, 0, 0, 1), "qubit")
    _assert_unsupported(prog, config, "amp matrix")


def test_expressions(config):
    with program() as prog:
        n = declare(int)
        k = declare(int, value=7)
        x = declare(fixed)
        arr = declare(int, value=[10, 20, 30])
        out = [declare(int) for _ in range(5)]
        fx = declare(fixed)
        flag = declare(bool)
        st = declare_stream()
        with for_(n, 0, n < 1, n + 1):
            assign(out[0], k + 3)
            assign(out[1], k - 10)
            assign(out[2], k * 2)
            assign(out[3], -k / 2)  # integer division toward zero: -3
            assign(out[4], arr[k - 6])
            assign(fx, Cast.mul_fixed_by_int(0.25, 3))
            assign(flag, (k > 6) & (k <= 7))
            with if_(k == 8):
                assign(x, 1.0)
            with else_():
                assign(x, -0.5)
            for v in (*out, fx, flag, x):
                save(v, st)
        with stream_processing():
            st.save("s")
    values = [e.value for e in trace_program(prog, config).events if isinstance(e, SaveEvent)]
    assert values == [10, -3, 14, -3, 20, 0.75, True, -0.5]


def test_unsupported_stream_operator_raises(config):
    with program() as prog:
        n = declare(int)
        st = declare_stream()
        with for_(n, 0, n < 1, n + 1):
            save(n, st)
        with stream_processing():
            st.save_all("n")
    with pytest.raises(UnsupportedQuaError, match="stream processing"):
        trace_program(prog, config)


def test_stream_processing_shapes(calibrated_state, config):
    from qua.trace import parse_stream_processing

    ramsey = load_node("06a_ramsey.py")
    streams = parse_stream_processing(ramsey.create_qua_program(calibrated_state, ramsey.Parameters()).qua_program)
    assert streams["I"] == StreamSpec("I", "r2", True, (250, 2))
    assert streams["n"] == StreamSpec("n", "r1", False, ())
    blobs = load_node("07_iq_blobs.py")
    trace = trace_program(blobs.create_qua_program(calibrated_state, blobs.Parameters()), config)
    assert trace.streams["I_g"].average is False
    assert trace.streams["I_g"].buffer_shape == (4000,)


def _old_ramsey_frames_program(detuning_turns_per_ns: float, idle_cycles):
    """06a's frame pattern before the in-range rewrite: phase = detuning * 4t, unwrapped."""
    with program() as prog:
        n = declare(int)
        t = declare(int)
        sign = declare(int)
        phase = declare(fixed)
        with for_(n, 0, n < 1, n + 1):
            with for_each_(t, idle_cycles):
                with for_each_(sign, [-1, 1]):
                    reset_frame("qubit")
                    with if_(sign == 1):
                        assign(phase, Cast.mul_fixed_by_int(detuning_turns_per_ns, 4 * t))
                    with else_():
                        assign(phase, Cast.mul_fixed_by_int(-detuning_turns_per_ns, 4 * t))
                    play("x90", "qubit")
                    frame_rotation_2pi(phase, "qubit")
                    wait(t, "qubit")
                    play("x90", "qubit")
    return prog


def test_fixed_wrap_keeps_frames(config):
    idle_cycles = [4, 5000, 9000, 12500]  # 0.0003 * 4 * 12500 = 15 turns, past the 4.28 range
    trace = trace_program(_old_ramsey_frames_program(0.0003, idle_cycles), config)
    assert trace.max_abs_fixed > 8  # the wrap fired
    frames = [e.frame_turns for e in trace.events if isinstance(e, PlayEvent)][1::2]
    expected = [(sign * 0.0003 * 4 * t) % 1 for t in idle_cycles for sign in (-1, 1)]
    for got, want in zip(frames, expected, strict=True):
        assert abs(((got - want) + 0.5) % 1 - 0.5) < 1e-12


def test_ramsey_frame_phase_stays_in_fixed_range(calibrated_state, config):
    # Owner decision on 06: the assigned virtual detuning phase never leaves the 4.28 range.
    ramsey = load_node("06a_ramsey.py")
    params = ramsey.Parameters()
    trace = trace_program(ramsey.create_qua_program(calibrated_state, params), config)
    assert trace.max_abs_fixed < 8
    detuning = params.frequency_detuning_in_mhz * 1e6
    times = ramsey.idle_times_ns(params)
    frames = [e.frame_turns for e in trace.events if isinstance(e, PlayEvent)][1::2]
    expected = [(sign * detuning * 1e-9 * t) % 1 for t in times for sign in ramsey.DETUNING_SIGNS]
    for got, want in zip(frames, expected, strict=True):
        assert abs(((got - want) + 0.5) % 1 - 0.5) < 1e-12


def test_frame_and_reset(config):
    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 1, n + 1):
            frame_rotation_2pi(0.25, "qubit")
            play("x90", "qubit")
            frame_rotation(np.pi, "qubit")
            play("x90", "qubit")
            reset_frame("qubit")
            play("x90", "qubit")
    frames = [e.frame_turns for e in trace_program(prog, config).events if isinstance(e, PlayEvent)]
    assert frames == pytest.approx([0.25, 0.75, 0.0], abs=1e-12)


def test_implicit_align_on_loop_entry(config):
    with program() as prog:
        n = declare(int)
        t = declare(int)
        I, Q = declare(fixed), declare(fixed)
        with for_(n, 0, n < 1, n + 1):
            with for_each_(t, [4]):
                play("x180", "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
    events = trace_program(prog, config).events
    # first the shot loop's own entry, then the for_each_ body's
    assert events[1] == AlignEvent(("qubit", "resonator"), implicit=True)


def test_update_frequency_sets_if(config):
    from qm.qua import update_frequency

    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 1, n + 1):
            play("x90", "qubit")
            update_frequency("qubit", 123_456)
            play("x90", "qubit")
    plays = [e for e in trace_program(prog, config).events if isinstance(e, PlayEvent)]
    assert plays[0].if_hz == config["elements"]["qubit"]["intermediate_frequency"]
    assert plays[1].if_hz == 123_456


def test_max_events_guard(config):
    with program() as prog:
        n = declare(int)
        t = declare(int)
        with for_(n, 0, n < 1, n + 1):
            with for_(t, 0, t < 1000, t + 1):
                play("x90", "qubit")
    with pytest.raises(UnsupportedQuaError, match="max_events"):
        trace_program(prog, config, max_events=100)


def test_measure_with_unsupported_process_raises(config):
    from qm.qua import demod

    with program() as prog:
        n = declare(int)
        I = declare(fixed)
        with for_(n, 0, n < 1, n + 1):
            measure("readout", "resonator", demod.full("cos", I))
    _assert_unsupported(prog, config, "demodIntegration")


# ---------------------------------------------------------------- bridge
def test_pulse_library_matches_config(calibrated_state, config):
    from calibration_utils.qua_bridge import PulseLibrary

    lib = PulseLibrary.from_state(calibrated_state, config)
    want = {"x180": (1.0, 0.0), "x90": (0.5, 0.0), "-x90": (0.5, np.pi), "y90": (0.5, np.pi / 2)}
    for op, (factor, angle) in want.items():
        assert lib.factors[op][0] == factor, op
        assert lib.factors[op][1] == pytest.approx(angle, abs=1e-12), op
    f = calibrated_state["qubits"]["q1"]["f_01"]
    # The latent y90 sign: a Gaussian on Q is lab phase +pi/2, simulator phase -pi/2.
    assert lib.pulse("y90", None, 0.0, f).phase == pytest.approx(-np.pi / 2, abs=1e-12)
    assert lib.pulse("x90", None, 0.25, f).phase == pytest.approx(-np.pi / 2, abs=1e-12)
    assert lib.pulse("-x90", None, 0.0, f).phase == np.pi
    x180 = calibrated_state["qubits"]["q1"]["xy"]["operations"]["x180"]
    assert lib.pulse("x180", 0.37, 0.0, f).amplitude == x180["amplitude"] * 0.37  # bit-equal
    sat = lib.pulse("saturation", 0.5, 0.0, f)
    assert sat.amplitude == calibrated_state["qubits"]["q1"]["xy"]["operations"]["saturation"]["amplitude"] * 0.5
    assert sat.length == 20_000


def test_pulse_docstring_states_y90_sign():
    # Controller decision on 06: the Pulse docstring states the simulator's sign convention.
    from simulator.transmon import Pulse

    doc = " ".join(Pulse.__doc__.split())
    assert "minus the lab phase" in doc
    assert "y90" in doc and "-pi/2" in doc


def test_simulator_demod(config):
    from calibration_utils.qua_bridge import simulator_demod
    from qua.trace import Demod, UnsupportedQuaError

    def dd(w1, w2, o1="out1", o2="out2"):
        return Demod("dual_demod.full", w1, w2, o1, o2, "v1")

    assert simulator_demod(config, dd("cos", "sin")) == (("cos", 1), ("sin", 2))
    assert simulator_demod(config, dd("minus_sin", "cos")) == (("minus_sin", 1), ("cos", 2))
    assert simulator_demod(config, dd("rotated_cos", "rotated_sin", "out2", "out1")) == (("rotated_cos", 2), ("rotated_sin", 1))
    with pytest.raises(UnsupportedQuaError, match="not defined"):
        simulator_demod(config, dd("cos", "nonexistent"))


def test_qubit_drive_after_measure_without_reset_raises(calibrated_state, config):
    from calibration_utils.qua_bridge import sequences_from_trace

    with program() as prog:
        n = declare(int)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 1, n + 1):
            for _ in range(2):
                play("x180", "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
        with stream_processing():
            I_st.buffer(2).average().save("I")
            Q_st.buffer(2).average().save("Q")
    with pytest.raises(UnsupportedQuaError, match="no passive reset"):
        sequences_from_trace(trace_program(prog, config), calibrated_state, config)


def test_second_measure_of_driven_qubit_without_reset_raises(calibrated_state, config):
    from calibration_utils.qua_bridge import sequences_from_trace

    with program() as prog:
        n = declare(int)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 1, n + 1):
            play("x180", "qubit")
            align("qubit", "resonator")
            for _ in range(2):  # the second readout sees the post-measurement state of a driven qubit
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
            wait(62_500, "resonator")  # reset before the next pass, so only the second readout is at fault
            align()
        with stream_processing():
            I_st.buffer(2).average().save("I")
            Q_st.buffer(2).average().save("Q")
    with pytest.raises(UnsupportedQuaError, match="no passive reset"):
        sequences_from_trace(trace_program(prog, config), calibrated_state, config)


def test_measure_saved_to_no_stream_raises(calibrated_state, config):
    from calibration_utils.qua_bridge import calls_from_trace

    with program() as prog:
        n = declare(int)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 1, n + 1):
            readout_iq(I, Q)
            save(I, I_st)
            save(Q, Q_st)
        with stream_processing():
            I_st.average().save("I")  # Q_st is never saved, so the measure feeds I only
    with pytest.raises(UnsupportedQuaError, match="no stream"):
        calls_from_trace(trace_program(prog, config), calibrated_state, config)


def test_wait_below_four_cycles_raises(config):
    with program() as prog:
        n = declare(int)
        with for_(n, 0, n < 1, n + 1):
            wait(2, "qubit")
    _assert_unsupported(prog, config, "wait of 2 cycles")


def test_qubit_drive_overlapping_readout_raises(calibrated_state, config):
    from calibration_utils.qua_bridge import sequences_from_trace

    with program() as prog:
        n = declare(int)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 1, n + 1):
            play("x180", "qubit")
            readout_iq(I, Q)  # no align: the readout starts while the pulse plays
            save(I, I_st)
            save(Q, Q_st)
        with stream_processing():
            I_st.average().save("I")
            Q_st.average().save("Q")
    with pytest.raises(UnsupportedQuaError, match="overlaps readout"):
        sequences_from_trace(trace_program(prog, config), calibrated_state, config)


def _ramsey_program(config, idle_cycles, detuning_hz, virtual: bool, frame_sign: int = 1, offset_hz: float = 0.15e6):
    """x90, wait, x90 with the drive offset_hz above the qubit, plus detuning_hz either as a real
    drive offset or as a virtual one (a frame advancing frame_sign * detuning_hz before the second x90)."""
    from qm.qua import update_frequency

    qubit_if = config["elements"]["qubit"]["intermediate_frequency"] + offset_hz
    with program() as prog:
        n = declare(int)
        t = declare(int)
        phi = declare(fixed)
        I, Q = declare(fixed), declare(fixed)
        I_st, Q_st = declare_stream(), declare_stream()
        with for_(n, 0, n < 1, n + 1):
            update_frequency("qubit", int(round(qubit_if + (0 if virtual else detuning_hz))))
            with for_each_(t, idle_cycles):
                if virtual:
                    reset_frame("qubit")
                    assign(phi, Cast.mul_fixed_by_int(frame_sign * detuning_hz * 1e-9, 4 * t))
                play("x90", "qubit")
                if virtual:
                    frame_rotation_2pi(phi, "qubit")
                wait(t, "qubit")
                play("x90", "qubit")
                align("qubit", "resonator")
                readout_iq(I, Q)
                save(I, I_st)
                save(Q, Q_st)
                wait(62_500, "resonator")
                align()
        with stream_processing():
            I_st.buffer(len(idle_cycles)).average().save("I")
            Q_st.buffer(len(idle_cycles)).average().save("Q")
    return prog


def test_virtual_detuning_equals_real_detuning(calibrated_state, config):
    """Pins "simulator phase = -(lab phase)": a frame advancing at +delta acts as a drive delta above.

    The drive sits 0.15 MHz above the qubit in every program, so the fringe frequency tells the
    sign: 0.35 MHz for the real and the virtual +0.2 MHz, 0.05 MHz for the flipped frame. (At zero
    offset the fringes for +delta and -delta would be the same cosine.)"""
    from calibration_utils.qua_bridge import sequences_from_trace
    from simulator.qpu import SimulatedQPU

    idle_cycles = list(range(4, 1254, 50))  # 25 points, 16 ns to 5 us
    delta = 0.2e6
    qpu = SimulatedQPU()

    def p_e(prog):
        records = sequences_from_trace(trace_program(prog, config), calibrated_state, config)
        return qpu.populations([r.sequence for r in records], records[0].drive_frequency)[1]

    real = p_e(_ramsey_program(config, idle_cycles, delta, virtual=False))
    virtual = p_e(_ramsey_program(config, idle_cycles, delta, virtual=True))
    flipped = p_e(_ramsey_program(config, idle_cycles, delta, virtual=True, frame_sign=-1))
    # residual: precession during the 40 ns pulses, about 2 pi * 0.2 MHz * 40 ns = 0.05 rad
    assert np.max(np.abs(real - virtual)) < 0.06
    assert np.max(np.abs(real - flipped)) > 0.5
