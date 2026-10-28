"""A deterministic evaluator of a QUA program's protobuf: what each element is told to do, in order.

``trace_program(prog, config)`` walks ``prog.qua_program`` (the ``QuaProgram`` protobuf message
that ``qm.program.Program`` wraps, QM serialization API:
https://docs.quantum-machines.co/1.4.1/docs/API_references/serialization/) with protobuf's own
reflection (``WhichOneof``, ``HasField``) and returns one pass of the outer averaging loop as a
list of events. It knows nothing about the simulator or the device; ``calibration_utils.qua_bridge``
turns a trace into simulator calls.

The message schema (oneof and field names) is QM's generated, undocumented code in
``qm/grpc/qm/pb/inc_qua_pb2.py``; tested on qm-qua 1.4.1 only, pinned by
``tests/test_qua_trace.py::test_qm_qua_surface_is_pinned``. Enum values are read through the
message's own class so this module imports nothing from ``qm.grpc``.

QUA semantics relied on (QUA API reference, https://docs.quantum-machines.co/latest/docs/API_references/qua/dsl_main/,
and Timing in QUA, https://docs.quantum-machines.co/latest/docs/Guides/timing_in_qua/) [vendor]:
wait and play durations are in 4 ns clock cycles; ``frame_rotation_2pi`` adds to the element's
frame phase in turns, ``reset_frame`` zeroes it; ``update_frequency`` sets an integer IF in Hz;
there is an implicit align of the participating elements on entering a flow-control body.
"""

from __future__ import annotations

from dataclasses import dataclass

# Two's-complement 4.28 fixed point: values live in [-8, 8). QM documents the range, not what an
# overflow does; a wrap by 16 is assumed. Frames are taken modulo one turn, so a wrap never moves a
# frame. Quantisation to 2^-28 is not emulated (the trace computes in float64).
FIXED_RANGE = 8.0

SUPPORTED_CASTS = {("cast", "mul_fixed_by_int"), ("cast", "mul_int_by_fixed"), ("cast", "to_int"), ("cast", "to_fixed")}


class UnsupportedQuaError(Exception):
    """A QUA construct the trace does not model. The message names the construct and the program's loc string."""


@dataclass(frozen=True)
class Demod:
    kind: str  # "dual_demod.full" only
    weight1: str  # integration weight name, e.g. "cos"
    weight2: str
    output1: str  # element output, e.g. "out1"
    output2: str
    target: str  # QUA variable name, e.g. "v3"


@dataclass(frozen=True)
class PlayEvent:
    element: str
    operation: str
    amplitude_scale: float | None  # amp.v0; None when absent
    duration_cycles: int | None  # None when absent
    frame_turns: float  # element frame phase in turns at play time, wrapped to [0, 1)
    if_hz: float  # element IF at play time (config value until update_frequency)


@dataclass(frozen=True)
class WaitEvent:
    elements: tuple[str, ...]  # () means all elements
    cycles: int


@dataclass(frozen=True)
class AlignEvent:
    elements: tuple[str, ...]  # () means all elements, as align() with no arguments
    implicit: bool = False  # inserted at flow-control entry


@dataclass(frozen=True)
class MeasureEvent:
    element: str
    operation: str
    amplitude_scale: float | None
    demods: tuple[Demod, ...]
    if_hz: float
    adc_stream: str | None = None  # result tag of measure(..., adc_stream=s), e.g. "atr_r1"


@dataclass(frozen=True)
class SaveEvent:
    tag: str  # result tag, e.g. "r2"
    variable: str  # source variable name ("" for a literal)
    value: float | int | bool | None  # None for measure targets (no value until simulated)


Event = PlayEvent | WaitEvent | AlignEvent | MeasureEvent | SaveEvent


@dataclass(frozen=True)
class StreamSpec:
    name: str  # saved name, e.g. "I"
    tag: str  # e.g. "r2"
    average: bool
    buffer_shape: tuple[int, ...]  # () for a plain save, (61,) or (250, 2)
    adc_input: int | None = None  # raw ADC trace stream: s.input1() is 1, s.input2() is 2


@dataclass(frozen=True)
class ProgramTrace:
    shots: int  # iterations of the outer averaging loop
    shot_variable: str
    events: tuple[Event, ...]  # one pass of the shot-loop body, in program order
    streams: dict[str, StreamSpec]
    max_abs_fixed: float = 0.0  # largest |value| assigned to a fixed variable, before any 4.28 wrap


# ---------------------------------------------------------------- stream processing
def _list_value(lv) -> list:
    out = []
    for v in lv.values:
        kind = v.WhichOneof("kind")
        if kind == "string_value":
            out.append(v.string_value)
        elif kind == "list_value":
            out.append(_list_value(v.list_value))
        else:
            raise UnsupportedQuaError(f"stream processing: unexpected value kind {kind!r}")
    return out


def parse_stream_processing(qua_program) -> dict[str, StreamSpec]:
    """Saved streams from ``resultAnalysis.model``: ``['save', name, chain]`` entries.

    The chain is ``['average', x]`` (outermost only), nested ``['buffer', 'n', x]`` and finally
    ``['@re', '0', tag]``. A raw ADC trace stream (``declare_stream(adc_trace=True)``, then
    ``.input1()`` or ``.input2()``) wraps the tag as
    ``['@macro_adc_trace', ['@macro_input', 'k', ['@re', '0', tag]]]``. The outermost buffer is the last applied, so it is the first axis of the
    fetched array: ``s.buffer(2).buffer(250)`` is ``['buffer', '250', ['buffer', '2', ...]]`` and
    has shape (250, 2). This list format is undocumented; pinned by tests/test_qua_trace.py.
    """
    streams: dict[str, StreamSpec] = {}
    for entry in qua_program.resultAnalysis.model:
        item = _list_value(entry)
        if len(item) != 3 or item[0] != "save" or not isinstance(item[2], list):
            raise UnsupportedQuaError(f"stream processing: unsupported operator {item[:1]!r} (only save)")
        name, chain = item[1], item[2]
        average = False
        if chain and chain[0] == "average":
            average, chain = True, chain[1]
        shape: list[int] = []
        while chain and chain[0] == "buffer":
            shape.append(int(chain[1]))
            chain = chain[2]
        adc_input = None
        if chain and chain[0] == "@macro_adc_trace":
            inner = chain[1]
            if not (isinstance(inner, list) and len(inner) == 3 and inner[0] == "@macro_input" and inner[1] in ("1", "2")):
                raise UnsupportedQuaError(f"stream processing: adc trace stream {name!r} without input1() or input2()")
            adc_input, chain = int(inner[1]), inner[2]
        if not (len(chain) == 3 and chain[0] == "@re" and chain[1] == "0"):
            raise UnsupportedQuaError(f"stream processing: unsupported operator {chain[:1]!r} in stream {name!r}")
        streams[name] = StreamSpec(name, chain[2], average, tuple(shape), adc_input)
    return streams


# ---------------------------------------------------------------- evaluator
def _wrap_fixed(value: float) -> float:
    return (value + FIXED_RANGE) % (2 * FIXED_RANGE) - FIXED_RANGE


def _int_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class _Evaluator:
    def __init__(self, qua_program, config: dict, max_events: int):
        self.qp = qua_program
        self.config = config
        self.max_events = max_events
        self.events: list[Event] = []
        self.steps = 0
        self.types: dict[str, str] = {}
        self.values: dict[str, object] = {}
        self.arrays: dict[str, list] = {}
        self.unknown: set[str] = set()  # measure targets: value only known after simulation
        self.frames = {el: 0.0 for el in config["elements"]}
        self.ifs = {el: config["elements"][el]["intermediate_frequency"] for el in config["elements"]}
        self.max_abs_fixed = 0.0
        self._elements: dict[int, tuple] = {}
        type_name = type(qua_program).Type.Name
        for var in qua_program.script.variables:
            vtype = type_name(var.type)
            literals = [self._parse_literal(lit.value, vtype) for lit in var.value]
            self.types[var.name] = vtype
            if var.dim == 1:
                self.arrays[var.name] = literals
            elif var.dim == 0:
                default = {"INT": 0, "REAL": 0.0, "BOOL": False}[vtype]
                self.values[var.name] = literals[0] if literals else default
            else:
                raise UnsupportedQuaError(f"variable {var.name} has dim {var.dim}")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse_literal(text: str, vtype: str):
        if vtype == "INT":
            return int(text)
        if vtype == "REAL":
            return float(text)
        return text in ("true", "True")

    @staticmethod
    def _fail(construct: str, msg) -> UnsupportedQuaError:
        loc = getattr(msg, "loc", "") or "(no loc)"
        return UnsupportedQuaError(f"{construct} is not supported by qua/trace.py, at {loc}")

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        self._step()

    def _step(self) -> None:
        self.steps += 1
        if self.steps > self.max_events:
            raise UnsupportedQuaError(f"trace exceeds max_events ({self.max_events})")

    def _assign(self, name: str, value, msg) -> None:
        if name not in self.types:
            raise self._fail(f"assignment to undeclared variable {name}", msg)
        vtype = self.types[name]
        if vtype == "REAL":
            value = float(value)
            self.max_abs_fixed = max(self.max_abs_fixed, abs(value))
            value = _wrap_fixed(value)
        elif vtype == "INT":
            value = int(value)
        else:
            value = bool(value)
        self.values[name] = value
        self.unknown.discard(name)

    def _value(self, name: str, msg):
        if name in self.unknown:
            raise self._fail(f"reading measure target {name} (its value comes from the simulation)", msg)
        if name not in self.values:
            raise self._fail(f"reference to unknown variable {name}", msg)
        return self.values[name]

    # -------------------------------------------------------------- expressions
    def expr(self, e):
        kind = e.WhichOneof("expression_oneof")
        if kind == "literal":
            return self._parse_literal(e.literal.value, type(self.qp).Type.Name(e.literal.type))
        if kind == "variable":
            return self._value(e.variable.name, e.variable)
        if kind == "arrayCell":
            arr = self.arrays.get(e.arrayCell.arrayVar.name)
            if arr is None:
                raise self._fail(f"unknown array {e.arrayCell.arrayVar.name}", e.arrayCell)
            index = self.expr(e.arrayCell.index)
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(arr):
                raise self._fail(f"array index {index!r} out of bounds", e.arrayCell)
            return arr[index]
        if kind == "binaryOperation":
            return self._binary(e.binaryOperation)
        if kind == "libFunction":
            return self._lib(e.libFunction)
        raise self._fail(f"expression {kind}", getattr(e, kind) if kind else e)

    def _binary(self, b):
        op = type(b).BinaryOperator.Name(b.op)
        left, right = self.expr(b.left), self.expr(b.right)
        if op == "ADD":
            return left + right
        if op == "SUB":
            return left - right
        if op == "MULT":
            return left * right
        if op == "DIV":
            if isinstance(left, int) and isinstance(right, int):
                return _int_div(left, right)
            return left / right
        if op == "LT":
            return left < right
        if op == "LET":
            return left <= right
        if op == "GT":
            return left > right
        if op == "GET":
            return left >= right
        if op == "EQ":
            return left == right
        if op == "AND":
            return bool(left) and bool(right)
        if op == "OR":
            return bool(left) or bool(right)
        raise self._fail(f"binary operator {op}", b)

    def _lib(self, f):
        key = (f.libraryName, f.functionName)
        if key not in SUPPORTED_CASTS:
            raise self._fail(f"library function {f.libraryName}.{f.functionName}", f)
        args = []
        for a in f.arguments:
            if a.WhichOneof("argument_oneof") != "scalar":
                raise self._fail(f"array argument to {f.functionName}", f)
            args.append(self.expr(a.scalar))
        name = f.functionName
        if name in ("mul_fixed_by_int", "mul_int_by_fixed"):
            return float(args[0] * args[1])
        if name == "to_fixed":
            return float(args[0])
        value = args[0]
        if isinstance(value, float) and value != int(value):
            raise self._fail("Cast.to_int of a non-integer fixed value (rounding not modelled)", f)
        return int(value)

    # -------------------------------------------------------------- statements
    def run_block(self, collection) -> None:
        for s in collection.statements:
            self.statement(s)

    @staticmethod
    def elements_in(collection) -> set[str]:
        """Every element a statement collection refers to, recursively."""
        found: set[str] = set()

        def visit(msg):
            for desc, value in msg.ListFields():
                if desc.message_type is None:
                    continue
                if desc.message_type.name == "QuantumElementReference":
                    items = value if desc.is_repeated else [value]
                    found.update(item.name for item in items)
                    continue
                for item in value if desc.is_repeated else [value]:
                    visit(item)

        visit(collection)
        return found

    def _elements_cached(self, collection) -> set[str]:
        # upb returns the same wrapper object for a submessage while its parent lives, so the id is
        # a stable key; the object is kept in the cache so the id cannot be reused.
        hit = self._elements.get(id(collection))
        if hit is None:
            hit = self._elements[id(collection)] = (collection, self.elements_in(collection))
        return hit[1]

    def _enter_body(self, *collections) -> None:
        elements = set().union(*(self._elements_cached(c) for c in collections))
        if len(elements) >= 2:
            self._emit(AlignEvent(tuple(sorted(elements)), implicit=True))

    def statement(self, s) -> None:
        kind = s.WhichOneof("statement_oneof")
        handler = getattr(self, f"_st_{kind}", None)
        if handler is None:
            raise self._fail(f"statement {kind}", getattr(s, kind) if kind else s)
        handler(getattr(s, kind))

    def _st_for(self, st) -> None:
        self.run_block(st.init)
        while self.expr(st.condition):
            self._step()
            self._enter_body(st.body)
            self.run_block(st.body)
            self.run_block(st.update)

    def _st_forEach(self, st) -> None:
        arrays = []
        for it in st.iterator:
            arr = self.arrays.get(it.array.name)
            if arr is None:
                raise self._fail(f"for_each_ over unknown array {it.array.name}", st)
            arrays.append((it.variable.name, arr))
        lengths = {len(a) for _, a in arrays}
        if len(lengths) != 1:
            raise self._fail("for_each_ over arrays of different lengths", st)
        for i in range(lengths.pop()):
            self._step()
            for name, arr in arrays:
                self._assign(name, arr[i], st)
            self._enter_body(st.body)
            self.run_block(st.body)

    def _st_if(self, st) -> None:
        if len(st.elseifs):
            raise self._fail("elif_", st)
        branch = st.body if self.expr(st.condition) else getattr(st, "else")
        self._enter_body(st.body, getattr(st, "else"))
        self.run_block(branch)

    def _st_assign(self, st) -> None:
        if st.HasField("target"):
            if st.target.WhichOneof("target") != "variable":
                raise self._fail("assignment to an array cell", st)
            name = st.target.variable.name
        else:
            name = st.variable.name
        self._assign(name, self.expr(st.expression), st)

    def _amp(self, st, construct: str) -> float | None:
        if not st.HasField("amp"):
            return None
        if st.amp.HasField("v1") or st.amp.HasField("v2") or st.amp.HasField("v3"):
            raise self._fail(f"{construct} amp matrix (amp(v00, v01, v10, v11))", st)
        scale = float(self.expr(st.amp.v0))
        if not -2 <= scale < 2:
            raise self._fail(f"{construct} amplitude_scale {scale} outside [-2, 2)", st)
        return scale

    def _st_play(self, st) -> None:
        if st.WhichOneof("pulseType") != "namedPulse":
            raise self._fail(f"play pulse type {st.WhichOneof('pulseType')}", st)
        for name in ("condition", "chirp", "truncate", "port_condition", "pulse"):
            if st.HasField(name):
                raise self._fail(f"play with {name}", st)
        for name in ("timestampLabel", "targetInput"):
            if getattr(st, name):
                raise self._fail(f"play with {name}", st)
        duration = int(self.expr(st.duration)) if st.HasField("duration") else None
        el = st.qe.name
        self._emit(PlayEvent(el, st.namedPulse.name, self._amp(st, "play"), duration, self.frames[el], self.ifs[el]))

    def _st_wait(self, st) -> None:
        cycles = int(self.expr(st.time))
        # Minimum wait is 4 cycles (16 ns) before QOP 3.3 (Timing in QUA,
        # https://docs.quantum-machines.co/latest/docs/Guides/timing_in_qua/ [vendor]).
        if cycles < 4:
            raise self._fail(f"wait of {cycles} cycles (under the 4-cycle minimum)", st)
        self._emit(WaitEvent(tuple(q.name for q in st.qe), cycles))

    def _st_align(self, st) -> None:
        self._emit(AlignEvent(tuple(q.name for q in st.qe)))

    def _st_measure(self, st) -> None:
        if st.timestampLabel:
            raise self._fail("measure with timestamp_stream", st)
        if len(st.outputs):
            raise self._fail("measure with legacy outputs", st)
        demods = []
        for proc in st.measureProcesses:
            if proc.WhichOneof("process") != "analog":
                raise self._fail(f"measure process {proc.WhichOneof('process')}", st)
            kind = proc.analog.WhichOneof("process")
            if kind != "dualDemodIntegration":
                raise self._fail(f"measure process {kind}", st)
            dd = proc.analog.dualDemodIntegration
            target_kind = dd.target.WhichOneof("processTarget")
            if target_kind != "scalarProcess" or dd.target.scalarProcess.WhichOneof("target") != "variable":
                raise self._fail(f"dual_demod target {target_kind} (only dual_demod.full into a scalar)", st)
            target = dd.target.scalarProcess.variable.name
            demods.append(
                Demod("dual_demod.full", dd.integration1.name, dd.integration2.name, dd.elementOutput1, dd.elementOutput2, target)
            )
        el = st.qe.name
        self._emit(MeasureEvent(el, st.pulse.name, self._amp(st, "measure"), tuple(demods), self.ifs[el], st.streamAs or None))
        self.unknown.update(d.target for d in demods)

    def _st_save(self, st) -> None:
        kind = st.source.WhichOneof("source")
        if kind == "variable":
            name = st.source.variable.name
            value = None if name in self.unknown else self._value(name, st)
        elif kind == "literal":
            name = ""
            value = self.expr(_literal_expr(st.source.literal, self))
        else:
            raise self._fail(f"save from {kind}", st)
        self._emit(SaveEvent(st.tag, name, value))

    def _st_zRotation(self, st) -> None:
        el = st.qe.name
        self.frames[el] = (self.frames[el] + float(self.expr(st.value))) % 1.0

    def _st_resetPhase(self, st) -> None:
        """reset_if_phase: no event. The simulator's demodulation oscillator is always referenced to
        the readout pulse start (simulator/readout.py), which is what reset_if_phase gives."""

    def _st_resetFrame(self, st) -> None:
        self.frames[st.qe.name] = 0.0

    def _st_updateFrequency(self, st) -> None:
        if type(st).Units.Name(st.units) != "Hz":
            raise self._fail(f"update_frequency units {type(st).Units.Name(st.units)}", st)
        if st.keepPhase:
            raise self._fail("update_frequency with keep_phase=True", st)
        self.ifs[st.qe.name] = int(self.expr(st.value))

    # -------------------------------------------------------------- the shot loop
    def shot_loop(self, streams: dict[str, StreamSpec]):
        """Check the body is one outer ``for_(n, 0, n < N, n + 1)``; return (N, n, the for statement)."""
        error = UnsupportedQuaError("expected one outer averaging loop for_(n, 0, n < N, n + 1)")
        body = self.qp.script.body.statements
        if len(body) != 1 or body[0].WhichOneof("statement_oneof") != "for":
            raise error
        loop = getattr(body[0], "for")
        init, update, cond = loop.init.statements, loop.update.statements, loop.condition
        try:
            (init_st,) = init
            (update_st,) = update
            n = init_st.assign.target.variable.name
            ok = (
                self.expr(init_st.assign.expression) == 0
                and init_st.assign.expression.WhichOneof("expression_oneof") == "literal"
                and cond.binaryOperation.left.variable.name == n
                and type(cond.binaryOperation).BinaryOperator.Name(cond.binaryOperation.op) == "LT"
                and cond.binaryOperation.right.WhichOneof("expression_oneof") == "literal"
                and update_st.assign.target.variable.name == n
                and type(update_st.assign.expression.binaryOperation).BinaryOperator.Name(update_st.assign.expression.binaryOperation.op) == "ADD"
                and update_st.assign.expression.binaryOperation.left.variable.name == n
                and self.expr(update_st.assign.expression.binaryOperation.right) == 1
            )
        except (ValueError, UnsupportedQuaError):
            ok = False
        if not ok:
            raise error
        shots = int(self.expr(cond.binaryOperation.right))
        # n may only be saved to plain (not averaged) streams; anything else would make passes differ.
        not_averaged = {s.tag for s in streams.values() if not s.average}
        averaged = {s.tag for s in streams.values() if s.average}
        if self._uses_variable(loop.body, n, not_averaged - averaged):
            raise error
        return shots, n, loop

    @staticmethod
    def _uses_variable(collection, name: str, allowed_save_tags: set[str]) -> bool:
        def visit(msg) -> bool:
            if msg.DESCRIPTOR.name == "SaveStatement" and msg.tag in allowed_save_tags:
                return False
            if msg.DESCRIPTOR.name == "VarRefExpression" and msg.name == name:
                return True
            for desc, value in msg.ListFields():
                if desc.message_type is None:
                    continue
                for item in value if desc.is_repeated else [value]:
                    if visit(item):
                        return True
            return False

        return visit(collection)


def _literal_expr(literal, ev: _Evaluator):
    """Wrap a LiteralExpression so ``_Evaluator.expr`` can read it."""
    expr = type(ev.qp).AnyScalarExpression()
    expr.literal.CopyFrom(literal)
    return expr


def trace_program(prog, config: dict, *, max_events: int = 200_000) -> ProgramTrace:
    """One pass of ``prog``'s outer averaging loop, evaluated with ``n = 0``.

    Collapsing the loop is exact for the simulator, whose shots are independent and identically
    prepared; the averaging over N shots happens in ``SimulatedQPU``.
    """
    qp = prog.qua_program
    streams = parse_stream_processing(qp)
    ev = _Evaluator(qp, config, max_events)
    shots, n, loop = ev.shot_loop(streams)
    ev.run_block(loop.init)
    ev._enter_body(loop.body)
    ev.run_block(loop.body)
    first = list(ev.events)
    if shots > 1:
        # A second pass (n = 1) must tell the elements exactly the same thing, or the passes are
        # not identical (a frame or frequency carried over from the previous shot, for example).
        ev.events = []
        ev._assign(n, 1, loop)
        ev._enter_body(loop.body)
        ev.run_block(loop.body)

        def comparable(events):
            return [e for e in events if not (isinstance(e, SaveEvent) and e.variable == n)]

        if comparable(ev.events) != comparable(first):
            raise UnsupportedQuaError("shot passes differ: the second pass of the averaging loop is not the first repeated")
    return ProgramTrace(shots, n, tuple(first), streams, ev.max_abs_fixed)
