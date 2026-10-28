"""Compare the SimulatedQPU calls a node makes with the calls its QUA program implies."""

from __future__ import annotations

import math

import numpy as np

from calibration_utils.qua_bridge import SIMULATOR_APPROXIMATIONS, SIMULATOR_ONLY_KWARGS, QPUCall
from simulator.transmon import Idle, Pulse

FREQUENCY_TOLERANCE_HZ = 1.0  # integer_if: QUA IFs are whole Hz
AMPLITUDE_RTOL = 1e-12
PHASE_ATOL = 1e-9

FREQUENCY_KWARGS = {"frequencies", "drive_frequencies", "drive_frequency"}
AMPLITUDE_KWARGS = {"drive_amplitude"}
SEQUENCE_KWARGS = {"sequences", "sequence"}
EXACT_KWARGS = {"demod"}  # simulator.readout Demod tuples: weight names and analog inputs


def _wrap(phase: float) -> float:
    phase = math.remainder(phase, 2 * math.pi)
    return math.pi if phase <= -math.pi else phase


def canonical_segment(seg) -> tuple:
    """A negative amplitude at phase 0 and a positive one at phase pi are the same drive."""
    if isinstance(seg, Pulse):
        phase = _wrap(seg.phase + (math.pi if seg.amplitude < 0 else 0.0))
        return ("pulse", abs(seg.amplitude), seg.length, seg.sigma, phase, seg.drive_frequency)
    if isinstance(seg, Idle):
        return ("idle", seg.duration, seg.drive_frequency)
    return (type(seg).__name__, seg)


def _close_frequency(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= FREQUENCY_TOLERANCE_HZ


def _segments_match(got, want) -> str | None:
    g, w = canonical_segment(got), canonical_segment(want)
    if g[0] != w[0]:
        return "segment kind"
    if g[0] == "pulse":
        if not math.isclose(g[1], w[1], rel_tol=AMPLITUDE_RTOL):
            return "amplitude"
        if g[2] != w[2] or g[3] != w[3]:
            return "length or sigma"
        if abs(_wrap(g[4] - w[4])) > PHASE_ATOL:
            return "phase"
        if not _close_frequency(g[5], w[5]):
            return "pulse drive_frequency"
        return None
    if g[0] == "idle":
        if g[1] != w[1]:
            return "idle duration"
        if not _close_frequency(g[2], w[2]):
            return "idle drive_frequency"
        return None
    return None if g == w else "segment"


def _compare_sequence(where: str, got, want) -> None:
    if len(got) != len(want):
        raise AssertionError(f"{where}: {len(got)} segments, the program implies {len(want)}:\n  got  {[canonical_segment(s) for s in got]}\n  want {[canonical_segment(s) for s in want]}")
    for k, (g, w) in enumerate(zip(got, want)):
        field = _segments_match(g, w)
        if field:
            raise AssertionError(f"{where}, segment {k}: {field} differs\n  got  {canonical_segment(g)}\n  want {canonical_segment(w)}")


def _compare_readout(where: str, got, want) -> None:
    """simulator.readout.ReadoutSettings: frequency to 1 Hz (integer_if), amplitude to 1e-12, the rest
    exact, and every integration weight sample equal."""
    if abs(got.frequency - want.frequency) > FREQUENCY_TOLERANCE_HZ:
        raise AssertionError(f"{where}: readout frequency {got.frequency!r}, the program implies {want.frequency!r}")
    if got.lo_frequency != want.lo_frequency:
        raise AssertionError(f"{where}: readout LO {got.lo_frequency!r}, the program implies {want.lo_frequency!r}")
    if not math.isclose(got.amplitude, want.amplitude, rel_tol=AMPLITUDE_RTOL):
        raise AssertionError(f"{where}: readout amplitude {got.amplitude!r}, the program implies {want.amplitude!r}")
    for field in ("pulse_length", "time_of_flight", "smearing", "input_offsets"):
        if getattr(got, field) != getattr(want, field):
            raise AssertionError(f"{where}: readout {field} {getattr(got, field)!r}, the program implies {getattr(want, field)!r}")
    if set(got.weights) != set(want.weights):
        raise AssertionError(f"{where}: integration weights {sorted(got.weights)}, the program implies {sorted(want.weights)}")
    for key in want.weights:
        if not all(np.array_equal(g, w) for g, w in zip(got.weights[key], want.weights[key])):
            raise AssertionError(f"{where}: integration weight {key!r} differs from the program's config")


def _compare_value(where: str, name: str, got, want) -> None:
    if name == "readout":
        _compare_readout(where, got, want)
        return
    if name in EXACT_KWARGS:
        if got != want:
            raise AssertionError(f"{where}: {got!r}, the program implies {want!r}")
        return
    if name in SEQUENCE_KWARGS:
        if name == "sequence":
            _compare_sequence(where, got, want)
            return
        if len(got) != len(want):
            raise AssertionError(f"{where}: {len(got)} sequences, the program implies {len(want)}")
        for i, (g, w) in enumerate(zip(got, want)):
            _compare_sequence(f"{where}, sequence {i}", g, w)
        return
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape:
        raise AssertionError(f"{where}: shape {g.shape}, the program implies {w.shape}")
    if name in FREQUENCY_KWARGS:
        worst = float(np.max(np.abs(g - w))) if g.size else 0.0
        if worst > FREQUENCY_TOLERANCE_HZ:
            raise AssertionError(f"{where}: frequency differs by up to {worst:.3f} Hz (tolerance {FREQUENCY_TOLERANCE_HZ} Hz)")
        return
    if name in AMPLITUDE_KWARGS:
        if not np.allclose(g, w, rtol=AMPLITUDE_RTOL, atol=0):
            raise AssertionError(f"{where}: amplitude {got!r}, the program implies {want!r}")
        return
    if not np.array_equal(g, w):
        raise AssertionError(f"{where}: {got!r}, the program implies {want!r}")


def assert_calls_match(recorded: list[tuple[str, dict]], expected: list[QPUCall], node: str = "") -> dict[str, object]:
    """Raise AssertionError naming the call, keyword and sequence index on the first difference.

    Returns the simulator-only keyword arguments the node passed (``SIMULATOR_ONLY_KWARGS``), so
    the caller can pin which node passes which.
    """
    label = f"{node}: " if node else ""
    for call in expected:
        unknown = {a.key for a in call.approximations} - set(SIMULATOR_APPROXIMATIONS)
        if unknown:
            raise AssertionError(f"{label}approximation keys {sorted(unknown)} are not registered in SIMULATOR_APPROXIMATIONS")
    got_methods = [m for m, _ in recorded]
    want_methods = [c.method for c in expected]
    if got_methods != want_methods:
        raise AssertionError(f"{label}calls {got_methods}, the program implies {want_methods}")
    simulator_only: dict[str, object] = {}
    for i, ((method, kwargs), call) in enumerate(zip(recorded, expected)):
        where = f"{label}call {i} ({method})"
        extra = set(kwargs) - set(call.kwargs)
        for name in sorted(extra):
            if name not in SIMULATOR_ONLY_KWARGS:
                raise AssertionError(f"{where}: keyword {name}={kwargs[name]!r} has no counterpart in the program")
            simulator_only[name] = kwargs[name]
        missing = set(call.kwargs) - set(kwargs)
        if missing:
            raise AssertionError(f"{where}: keywords {sorted(missing)} the program determines were not passed")
        for name, want in call.kwargs.items():
            _compare_value(f"{where}, keyword {name}", name, kwargs[name], want)
    return simulator_only
