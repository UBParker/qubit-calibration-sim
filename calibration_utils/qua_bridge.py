"""From a QUA program's trace to the SimulatedQPU calls that program implies.

``qua.trace.trace_program`` says what each element is told to do; this module turns that into
qubit pulse sequences on per-element timelines (``sequences_from_trace``) and then into calls of
``simulator.qpu.SimulatedQPU`` (``calls_from_trace``). ``tests/test_qua_single_source.py`` compares
those calls with the ones each node's ``run_simulated`` makes, so the program and the simulation are
checked to describe the same experiment.

It reads only the trace, the state and ``qua.config.build_config(state)``: never ``simulator.device``
or ``ctx.reference`` (truth isolation, tests/test_reference.py).

Every difference the simulator knowingly makes from the program is a named entry in
``SIMULATOR_APPROXIMATIONS``; the per-node set is pinned in tests/test_qua_single_source.py.

Adapting to SimulatedQPU signature changes (fixes 01, 07 and 08 change them): the mapping from a
group of measurements to one QPU method and its keyword arguments lives only in ``_qpu_call`` and
``SIMULATOR_ONLY_KWARGS`` below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from qua.config import gaussian_samples
from qua.trace import AlignEvent, Demod, MeasureEvent, PlayEvent, ProgramTrace, SaveEvent, UnsupportedQuaError, WaitEvent, trace_program
from simulator.readout import IQ_DEMOD, ReadoutSettings
from simulator.transmon import Idle, Pulse

QUBIT_ELEMENT = "qubit"
READOUT_ELEMENT = "resonator"
CLOCK_CYCLE_NS = 4

SIMULATOR_APPROXIMATIONS: dict[str, str] = {
    "passive_reset_complete": "A qubit wait of at least thermalization_time after a measurement returns the qubit to the "
    "exact thermal state; the residual exp(-t/T1) excitation (0.55 % for 250 us at T1 = 48 us) is dropped.",
    "undriven_qubit_stays_thermal": "Idles on a qubit that has not been driven since its reset are dropped, so it stays "
    "in the thermal state (the Lindblad model relaxes toward |g>, see 01's follow-up).",
    "saturation_steady_state": "A constant qubit drive is replaced by the steady-state Bloch solution; its length is ignored.",
    "readout_empty_resonator": "Every measure starts with an empty resonator: photons a previous readout leaves "
    "(exp(-2 pi kappa t) after a wait t) are dropped. The readout itself is simulated from the config's pulse, "
    "integration weights, time of flight, smearing and analog-input offsets (simulator/readout.py).",
    "integer_if": "update_frequency sets integer Hz; nodes pass the unrounded frequency (difference below 1 Hz).",
    "continuous_envelope": "Gaussian pulses are simulated as the continuous envelope with the state's amplitude, length "
    "and sigma; the config's 1 ns samples are checked to match it, not played sample by sample.",
    "shot_groups_batched": "Measurements interleaved within one shot (07: ground, then excited) are simulated as one "
    "call per stream group, all shots of the first group before the second. Shots are independent, so only the "
    "order of random draws differs; the residual excitation one shot leaves for the next is already dropped by "
    "passive_reset_complete.",
}

# Keyword arguments a node passes to SimulatedQPU that no QUA program can determine. The check
# ignores exactly these names (and the per-node test pins which node passes which value).
# Empty since fix 01 removed 06b's dephasing="echo"; fix 08 may add entries (name: reason).
SIMULATOR_ONLY_KWARGS: dict[str, str] = {}


@dataclass(frozen=True)
class ConstantDrive:
    """A constant-waveform qubit pulse (the config's saturation pulse)."""

    operation: str
    amplitude: float  # volts: waveform sample times amplitude_scale
    length: float  # ns
    drive_frequency: float


@dataclass(frozen=True)
class Approximation:
    key: str  # registry key
    detail: str  # instance detail, e.g. "saturation pulse 20000 ns treated as steady state"


@dataclass(frozen=True)
class QPUCall:
    method: str  # "run_sequences" | "resonator_spectroscopy" | "saturation_spectroscopy" | "single_shots" | "adc_trace"
    kwargs: dict  # exactly the keyword arguments of that SimulatedQPU method
    # Per result array, the stream names it fills, e.g. (("I",), ("Q",)); for adc_trace the arrays are
    # averaged input 1, averaged input 2, single run input 1, single run input 2.
    streams: tuple[tuple[str, ...], ...]
    approximations: tuple[Approximation, ...]


@dataclass(frozen=True)
class MeasureRecord:
    """One measurement: the qubit sequence since its reset and the readout that ends it."""

    sequence: tuple  # Pulse, Idle and ConstantDrive segments, leading idles dropped
    drive_frequency: float  # qubit drive frequency (LO + IF) of the sequence's pulses
    readout: ReadoutSettings  # the config's readout with the IF and amplitude at measure time
    demod: tuple  # simulator.readout Demod per output, in measure order
    outputs: dict  # demod target -> (stream tag, index into demod)
    approximations: tuple[Approximation, ...]
    adc_stream: str | None = None  # result tag of the raw ADC trace (measure(..., adc_stream=...))


# ---------------------------------------------------------------- pulses
def _normalise_phase(phase: float) -> float:
    """Into (-pi, pi]."""
    phase = math.remainder(phase, 2 * math.pi)
    return math.pi if phase <= -math.pi else phase


def _waveform(config: dict, name: str, length: int) -> np.ndarray:
    wf = config["waveforms"][name]
    if wf["type"] == "constant":
        return np.full(length, float(wf["sample"]))
    if wf["type"] == "arbitrary":
        samples = np.asarray(wf["samples"], dtype=float)
        if samples.size != length:
            raise ValueError(f"waveform {name} has {samples.size} samples, pulse length {length}")
        return samples
    raise ValueError(f"waveform type {wf['type']!r} not modelled")


class PulseLibrary:
    """Simulator pulses for the config's qubit operations, built from the state and checked against the config.

    ``factors[op] = (factor, angle)``: the operation's complex waveform is ``factor * e^{i angle}``
    times x180's. ``constants[op] = (sample, length)`` for constant-waveform operations.
    """

    def __init__(self, x180: dict, factors: dict, constants: dict):
        self.x180 = x180
        self.factors = factors
        self.constants = constants

    @classmethod
    def from_state(cls, state: dict, config: dict, qubit: str = "q1") -> PulseLibrary:
        x180 = state["qubits"][qubit]["xy"]["operations"]["x180"]
        element = config["elements"][QUBIT_ELEMENT]
        pulses = config["pulses"]
        ref_pulse = pulses[element["operations"]["x180"]]
        length = int(ref_pulse["length"])
        if length != x180["length"]:
            raise ValueError("config x180 length differs from the state")
        w180 = _waveform(config, ref_pulse["waveforms"]["I"], length) + 1j * _waveform(config, ref_pulse["waveforms"]["Q"], length)
        expected = np.asarray(gaussian_samples(x180["amplitude"], length, x180["sigma"]))
        if np.max(np.abs(w180 - expected)) > 1e-12:
            raise ValueError("config x180 waveform is not the state's Gaussian")
        mask = np.abs(w180) > 1e-6 * np.max(np.abs(w180))
        factors, constants = {}, {}
        for op, pulse_name in element["operations"].items():
            pulse = pulses[pulse_name]
            wf_i, wf_q = (config["waveforms"][pulse["waveforms"][k]] for k in ("I", "Q"))
            if wf_i["type"] == "constant" and wf_q["type"] == "constant":
                if wf_q["sample"] != 0:
                    raise ValueError(f"constant pulse {op} with a Q component is not modelled")
                constants[op] = (float(wf_i["sample"]), int(pulse["length"]))
                continue
            if int(pulse["length"]) != length:
                raise ValueError(f"config waveform for {op} is not a scaled x180 (length {pulse['length']})")
            w = _waveform(config, pulse["waveforms"]["I"], length) + 1j * _waveform(config, pulse["waveforms"]["Q"], length)
            ratio = w[mask] / w180[mask]
            c = complex(ratio.mean())
            if np.max(np.abs(ratio - c)) > 1e-12 or np.max(np.abs(w[~mask])) > 1e-6 * np.max(np.abs(w180)):
                raise ValueError(f"config waveform for {op} is not a scaled x180")
            factors[op] = (round(abs(c), 12), float(np.angle(c)))
        return cls(dict(x180), factors, constants)

    def pulse(self, operation: str, amplitude_scale: float | None, frame_turns: float, drive_frequency: float, duration_cycles: int | None = None) -> Pulse | ConstantDrive:
        if operation in self.constants:
            sample, length = self.constants[operation]
            amplitude = sample if amplitude_scale is None else sample * amplitude_scale
            if duration_cycles is not None:
                length = CLOCK_CYCLE_NS * duration_cycles
            return ConstantDrive(operation, amplitude, float(length), drive_frequency)
        if operation not in self.factors:
            raise UnsupportedQuaError(f"qubit operation {operation!r} is not in the config")
        if duration_cycles is not None:
            raise UnsupportedQuaError(f"play({operation!r}, duration=...) stretches an arbitrary waveform; not modelled")
        factor, angle = self.factors[operation]
        amplitude = self.x180["amplitude"] * factor
        if amplitude_scale is not None:
            amplitude = amplitude * amplitude_scale
        # Simulator phase = -(lab phase): see simulator.transmon.Pulse.
        phase = _normalise_phase(-(angle + 2 * math.pi * frame_turns))
        return Pulse(amplitude, self.x180["length"], self.x180["sigma"], phase, None)


# ---------------------------------------------------------------- readout
def simulator_demod(config: dict, demod: Demod, element: str = READOUT_ELEMENT) -> tuple:
    """``dual_demod.full(w1, w2, target)`` as the simulator's Demod: ((w1, input of output1), (w2, input of output2)).

    The element's ``outputs`` map ("out1", "out2") to controller analog inputs; the weights are looked
    up by name in the config the program runs with (simulator.readout.ReadoutSettings.from_config).
    """
    outputs = config["elements"][element]["outputs"]
    pulse = config["pulses"][config["elements"][element]["operations"]["readout"]]
    terms = []
    for weight, output in ((demod.weight1, demod.output1), (demod.weight2, demod.output2)):
        if weight not in pulse["integration_weights"]:
            raise UnsupportedQuaError(f"integration weight {weight!r} is not defined for the readout pulse")
        terms.append((weight, int(outputs[output][1])))
    return tuple(terms)


# ---------------------------------------------------------------- timelines
class _Timeline:
    """Per-element clocks in ns, and the qubit's segments since its last reset."""

    def __init__(self, state: dict, config: dict, lib: PulseLibrary):
        self.config = config
        self.lib = lib
        self.elements = tuple(config["elements"])
        self.clock = {el: 0.0 for el in self.elements}
        self.segments: list = []
        self.freqs: list = []  # drive frequency per segment (None for idles)
        self.measured = False
        self.driven_since_reset = False  # a measurement of a driven qubit leaves an unmodelled state
        self.idle_since_measure = 0.0
        self.pending: set[Approximation] = set()
        self.thermalization = state["qubits"]["q1"]["thermalization_time"]
        self.qubit_lo = config["elements"][QUBIT_ELEMENT]["mixInputs"]["lo_frequency"]
        self.qubit_if = config["elements"][QUBIT_ELEMENT]["intermediate_frequency"]
        self.records: list[dict] = []
        self.open: dict | None = None
        self.readout = ReadoutSettings.from_config(config, READOUT_ELEMENT)

    def _qubit_idle(self, duration: float) -> None:
        if duration <= 0:
            return
        self.segments.append(Idle(float(duration)))
        self.freqs.append(None)
        if self.measured:
            self.idle_since_measure += duration
            if self.idle_since_measure >= self.thermalization:
                self.segments, self.freqs = [], []
                self.measured = False
                self.driven_since_reset = False
                self.pending.add(
                    Approximation("passive_reset_complete", f"{self.idle_since_measure:.0f} ns qubit wait after a measurement")
                )

    def play(self, e: PlayEvent) -> None:
        if e.element != QUBIT_ELEMENT:
            raise UnsupportedQuaError(f"play on element {e.element!r}: only the qubit drive is simulated")
        if self.measured:
            raise UnsupportedQuaError("no passive reset between measurements: the post-measurement state is not modelled")
        self.qubit_if = e.if_hz
        frequency = self.qubit_lo + e.if_hz
        seg = self.lib.pulse(e.operation, e.amplitude_scale, e.frame_turns, frequency, e.duration_cycles)
        self.segments.append(seg)
        self.freqs.append(frequency)
        self.driven_since_reset = True
        self.clock[QUBIT_ELEMENT] += seg.length

    def wait(self, e: WaitEvent) -> None:
        for el in e.elements or self.elements:
            self.clock[el] += CLOCK_CYCLE_NS * e.cycles
            if el == QUBIT_ELEMENT:
                self._qubit_idle(CLOCK_CYCLE_NS * e.cycles)

    def align(self, e: AlignEvent) -> None:
        els = e.elements or self.elements
        t = max(self.clock[el] for el in els)
        for el in els:
            gap = t - self.clock[el]
            self.clock[el] = t
            if el == QUBIT_ELEMENT:
                self._qubit_idle(gap)

    def measure(self, e: MeasureEvent) -> None:
        if e.element != READOUT_ELEMENT:
            raise UnsupportedQuaError(f"measure on element {e.element!r}: only the resonator readout is simulated")
        self._close()
        if self.measured and self.driven_since_reset:
            raise UnsupportedQuaError(
                "no passive reset between measurements: the post-measurement state of a driven qubit is not modelled"
            )
        t_m, t_q = self.clock[READOUT_ELEMENT], self.clock[QUBIT_ELEMENT]
        if t_q > t_m:
            raise UnsupportedQuaError(f"qubit drive overlaps readout: the qubit is busy until {t_q:.0f} ns, the readout starts at {t_m:.0f} ns")
        self._qubit_idle(t_m - t_q)
        self.clock[QUBIT_ELEMENT] = t_m
        approx = {Approximation("readout_empty_resonator", "the resonator is empty when the readout starts")} | self.pending
        first = next((i for i, s in enumerate(self.segments) if not isinstance(s, Idle)), len(self.segments))
        if first > 0:
            approx.add(Approximation("undriven_qubit_stays_thermal", f"{sum(s.duration for s in self.segments[:first]):.0f} ns of undriven qubit idle dropped"))
        sequence, freqs = self.segments[first:], self.freqs[first:]
        pulse_freqs = [f for f in freqs if f is not None]
        if any(isinstance(s, Pulse) for s in sequence):
            approx.add(Approximation("continuous_envelope", "Gaussian pulses simulated as the continuous envelope"))
        if len(set(pulse_freqs)) > 1:
            # Pulses of one sequence at different frequencies: each segment carries its own.
            current = None
            fixed = []
            for seg, f in zip(sequence, freqs):
                current = f if f is not None else current
                fixed.append(replace(seg, drive_frequency=current) if current is not None else seg)
            sequence = fixed
        drive_frequency = pulse_freqs[0] if pulse_freqs else self.qubit_lo + self.qubit_if
        res = self.config["elements"][READOUT_ELEMENT]
        pulse_name = res["operations"][e.operation]
        pulse = self.config["pulses"][pulse_name]
        wf_i, wf_q = (self.config["waveforms"][pulse["waveforms"][k]] for k in ("I", "Q"))
        if wf_i["type"] != "constant" or wf_q["type"] != "constant" or wf_q["sample"] != 0:
            raise UnsupportedQuaError(f"readout pulse {pulse_name} is not a constant I waveform")
        sample = float(wf_i["sample"])
        if e.if_hz != res["intermediate_frequency"] or any(f != self.qubit_lo + self.config["elements"][QUBIT_ELEMENT]["intermediate_frequency"] for f in pulse_freqs):
            approx.add(Approximation("integer_if", "frequencies set by update_frequency in integer Hz"))
        readout = self.readout.replace(
            intermediate_frequency=float(e.if_hz),
            amplitude=sample if e.amplitude_scale is None else sample * e.amplitude_scale,
        )
        self.open = {
            "sequence": tuple(sequence),
            "drive_frequency": drive_frequency,
            "readout": readout,
            "demod": tuple(simulator_demod(self.config, d) for d in e.demods),
            "outputs": {d.target: [None, k] for k, d in enumerate(e.demods)},
            "approximations": approx,
            "adc_stream": e.adc_stream,
        }
        self.records.append(self.open)
        self.clock[READOUT_ELEMENT] += int(pulse["length"])
        self.segments, self.freqs = [], []
        self.measured = True
        self.idle_since_measure = 0.0
        self.pending = set()

    def save(self, e: SaveEvent) -> None:
        if self.open is not None and e.variable in self.open["outputs"]:
            self.open["outputs"][e.variable][0] = e.tag

    def _close(self) -> None:
        if self.open is not None:
            unsaved = [t for t, (tag, _) in self.open["outputs"].items() if tag is None]
            if unsaved:
                raise UnsupportedQuaError(f"measure target(s) {unsaved} never saved to a stream")
        self.open = None

    def run(self, events) -> list[dict]:
        self.records = []
        handlers = {PlayEvent: self.play, WaitEvent: self.wait, AlignEvent: self.align, MeasureEvent: self.measure, SaveEvent: self.save}
        for e in events:
            handlers[type(e)](e)
        self._close()
        return self.records


def sequences_from_trace(trace: ProgramTrace, state: dict, config: dict) -> list[MeasureRecord]:
    """The measurements of one shot pass, in program order.

    The timeline is walked over two consecutive passes: the first starts from the thermal state,
    the second from whatever the first left behind. Collapsing the averaging loop is only valid
    when both passes give the same sequences; their approximations are combined.
    """
    lib = PulseLibrary.from_state(state, config)
    timeline = _Timeline(state, config, lib)
    first = timeline.run(trace.events)
    second = timeline.run(trace.events)
    if [r["sequence"] for r in first] != [r["sequence"] for r in second]:
        raise UnsupportedQuaError("shot passes differ: the qubit state one pass leaves changes the next pass's sequences")
    records = []
    for a, b in zip(first, second):
        approx = tuple(sorted(a["approximations"] | b["approximations"], key=lambda x: (x.key, x.detail)))
        outputs = {target: (tag, k) for target, (tag, k) in a["outputs"].items()}
        records.append(MeasureRecord(a["sequence"], a["drive_frequency"], a["readout"], a["demod"], outputs, approx, a["adc_stream"]))
    return records


# ---------------------------------------------------------------- dispatch
def _common(values, name: str, group: tuple):
    values = list(values)
    if any(v != values[0] for v in values):
        raise UnsupportedQuaError(f"{name} differs between the measurements of stream group {group}")
    return values[0]


def _qpu_call(
    group: tuple[str, ...], records: list[MeasureRecord], trace: ProgramTrace, base: ReadoutSettings
) -> tuple[str, dict, list[Approximation]]:
    """The SimulatedQPU method and keyword arguments for one stream group. The only place that
    knows SimulatedQPU's signatures (with SIMULATOR_ONLY_KWARGS)."""
    specs = [trace.streams[name] for name in group]
    if records[0].adc_stream is not None:
        if len(records) != 1:
            raise UnsupportedQuaError(f"adc stream group {group}: one raw-trace measurement per shot is modelled")
        return "adc_trace", {"readout": records[0].readout, "n_avg": trace.shots}, []
    shapes = {s.buffer_shape for s in specs}
    if len(shapes) != 1:
        raise UnsupportedQuaError(f"streams {group} have different buffer shapes")
    shape = shapes.pop()
    averaged = [s.average for s in specs]
    extra: list[Approximation] = []
    readout_amplitude = _common((r.readout.amplitude for r in records), "readout amplitude", group)
    demod = _common((r.demod for r in records), "demodulation", group)

    def readout(per_point_frequency: bool = False):
        kwargs = {}
        settings = [r.readout for r in records]
        frequencies = [s.frequency for s in settings]
        if per_point_frequency and len(set(frequencies)) > 1:
            # A readout update_frequency inside the sweep (08a): the settings must agree apart from
            # the frequency, which goes to ``frequencies``; ``readout`` keeps the config's IF.
            settings = [s.with_frequency(base.frequency) for s in settings]
            kwargs["frequencies"] = np.array(frequencies)
        kwargs["readout"] = _common(settings, "readout settings (frequency, amplitude, timing)", group)
        if demod != IQ_DEMOD:
            kwargs["demod"] = demod  # SimulatedQPU's default is the unrotated cos/sin pair
        return kwargs

    if all(averaged):
        if len(records) != math.prod(shape):
            raise UnsupportedQuaError(f"stream group {group}: {len(records)} measurements per pass, buffer shape {shape}")
        no_qubit_play = not any(isinstance(e, PlayEvent) and e.element == QUBIT_ELEMENT for e in trace.events)
        if no_qubit_play and all(len(r.sequence) == 0 for r in records):
            if demod != IQ_DEMOD:
                raise UnsupportedQuaError("resonator spectroscopy with weights other than cos/sin is not dispatched")
            # The sweep is in `frequencies`; the settings carry the config IF the program starts from.
            kwargs = {
                "frequencies": np.array([r.readout.frequency for r in records]),
                "readout": base.replace(amplitude=readout_amplitude),
                "n_avg": trace.shots,
            }
            return "resonator_spectroscopy", kwargs, extra
        if all(len(r.sequence) == 1 and isinstance(r.sequence[0], ConstantDrive) for r in records):
            drives = [r.sequence[0] for r in records]
            kwargs = {
                "drive_frequencies": np.array([d.drive_frequency for d in drives]),
                "drive_amplitude": _common((d.amplitude for d in drives), "saturation amplitude", group),
                **readout(),
                "n_avg": trace.shots,
            }
            lengths = sorted({d.length for d in drives})
            extra.append(Approximation("saturation_steady_state", f"saturation pulse {lengths[0]:.0f} ns treated as steady state"))
            return "saturation_spectroscopy", kwargs, extra
        if any(isinstance(s, ConstantDrive) for r in records for s in r.sequence):
            raise UnsupportedQuaError(f"stream group {group}: a constant drive mixed with other segments is not modelled")
        sequences = [list(r.sequence) for r in records]
        drive_frequencies = [r.drive_frequency for r in records]
        if len(set(drive_frequencies)) > 1:
            sequences = [[replace(s, drive_frequency=f) if s.drive_frequency is None else s for s in seq] for seq, f in zip(sequences, drive_frequencies)]
        kwargs = {"sequences": sequences, "drive_frequency": drive_frequencies[0], **readout(True), "n_avg": trace.shots}
        return "run_sequences", kwargs, extra
    if not any(averaged):
        if shape != (trace.shots,) or len(records) != 1:
            raise UnsupportedQuaError(f"raw stream group {group}: expected one measurement per pass and buffer({trace.shots})")
        (r,) = records
        kwargs = {
            "sequence": list(r.sequence),
            "drive_frequency": r.drive_frequency,
            **readout(),
            "n_shots": trace.shots,
        }
        return "single_shots", kwargs, extra
    raise UnsupportedQuaError(f"stream group {group} mixes averaged and raw streams")


def calls_from_trace(trace: ProgramTrace, state: dict, config: dict) -> list[QPUCall]:
    records = sequences_from_trace(trace, state, config)
    names_by_tag: dict[str, list[str]] = {}
    for spec in trace.streams.values():
        names_by_tag.setdefault(spec.tag, []).append(spec.name)
    groups: dict[tuple[str, ...], list[MeasureRecord]] = {}
    arrays: dict[tuple[str, ...], tuple] = {}
    for r in records:
        if r.adc_stream is not None:
            # Raw ADC traces: SimulatedQPU.adc_trace returns (averaged, single_run), each (2, n).
            if r.outputs or r.sequence:
                raise UnsupportedQuaError("an adc_stream measurement with demodulation or a driven qubit is not modelled")
            specs = [spec for spec in trace.streams.values() if spec.tag == r.adc_stream]
            if not specs:
                raise UnsupportedQuaError(f"adc stream {r.adc_stream} is never saved")
            slots = {(True, 1): [], (True, 2): [], (False, 1): [], (False, 2): []}
            for spec in specs:
                if spec.buffer_shape or spec.adc_input is None:
                    raise UnsupportedQuaError(f"adc stream {spec.name}: only input1()/input2(), averaged or plain, is modelled")
                slots[(spec.average, spec.adc_input)].append(spec.name)
            key = tuple(spec.name for spec in specs)
            streams = tuple(tuple(slots[k]) for k in ((True, 1), (True, 2), (False, 1), (False, 2)))
            if arrays.setdefault(key, streams) != streams:
                raise UnsupportedQuaError(f"stream group {key}: the adc streams change between measurements")
            groups.setdefault(key, []).append(r)
            continue
        unused = [tag for tag, _ in r.outputs.values() if tag not in names_by_tag]
        if unused:
            raise UnsupportedQuaError(f"measure saved to result tag(s) {unused} that no stream processing saves: no stream to fill")
        key = tuple(name for tag, _ in r.outputs.values() for name in names_by_tag[tag])
        # Which returned array feeds which stream: array k is the k-th demod output of the measure.
        by_array: list[list[str]] = [[] for _ in r.demod]
        for tag, k in r.outputs.values():
            by_array[k].extend(names_by_tag[tag])
        streams = tuple(tuple(names) for names in by_array)
        if arrays.setdefault(key, streams) != streams:
            raise UnsupportedQuaError(f"stream group {key}: the demodulation changes between measurements")
        groups.setdefault(key, []).append(r)
    calls = []
    for key, group_records in groups.items():
        method, kwargs, extra = _qpu_call(key, group_records, trace, ReadoutSettings.from_config(config, READOUT_ELEMENT))
        approx = {a for r in group_records for a in r.approximations} | set(extra)
        if len(groups) > 1:
            approx.add(Approximation("shot_groups_batched", f"{len(groups)} stream groups per shot simulated one after another"))
        unknown = {a.key for a in approx} - set(SIMULATOR_APPROXIMATIONS)
        if unknown:
            raise KeyError(f"approximation keys {unknown} are not in SIMULATOR_APPROXIMATIONS")
        calls.append(QPUCall(method, kwargs, arrays[key], tuple(sorted(approx, key=lambda a: (a.key, a.detail)))))
    return calls


def calls_from_program(prog, state: dict) -> list[QPUCall]:
    """trace_program(prog, build_config(state)) then calls_from_trace."""
    from qua.config import build_config

    config = build_config(state)
    return calls_from_trace(trace_program(prog, config), state, config)
