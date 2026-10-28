"""Optional, seeded noise channels for SimulatedQPU. Off unless a NoiseModel is passed.

Units: Hz, ns, volts, as simulator/device.py; drift rates per second of wall-clock time.

The channels and where their numbers come from:

- Slow frequency drift (``SlowDrift`` on ``NoiseModel.frequency_drift``). Low-frequency (1/f-type)
  noise from microscopic fluctuators is generic in solid-state qubits (Paladino et al. 2014,
  https://doi.org/10.1103/RevModPhys.86.361). Schlör et al. 2019
  (https://doi.org/10.1103/PhysRevLett.123.190502) tracked a transmon for 19 hours and saw its
  frequency switch telegraphically between stationary points, by 5 to 140 kHz, explained by a
  near-resonant TLS whose own frequency switches. TLS frequencies also diffuse (Klimov et al. 2018,
  https://doi.org/10.1103/PhysRevLett.121.090502). The static limit and echo refocusing follow
  Ithier et al. 2005 (https://doi.org/10.1103/PhysRevB.72.134519) and Krantz et al. 2019
  (https://doi.org/10.1063/1.5089550). The linear and random-walk trajectories and their rates are
  synthetic, chosen inside the range Schlör et al. observed; they are not measured for any device.
- A coupled two-level defect (``TLSDefect``), modelled exactly in the Lindblad framework by
  ``simulator.transmon.Transmon``. Review: Müller, Cole, Lisenfeld 2019
  (https://doi.org/10.1088/1361-6633/ab3a7e). Strongly coupled TLS split the spectroscopy line
  (Schreier et al. 2008, https://doi.org/10.1103/PhysRevB.77.180502, as cited by Müller et al.);
  weakly coupled ones add Lorentzian resonances to 1/T1(f) (Barends et al. 2013,
  https://doi.org/10.1103/PhysRevLett.111.080502; Klimov et al. 2018, supplement S3).
- Quasiparticle bursts from ionizing radiation (``Bursts``): Vepsäläinen et al. 2020
  (https://doi.org/10.1038/s41586-020-2619-8) and McEwen et al. 2022
  (https://doi.org/10.1038/s41567-021-01432-8): events about every 10 s, decay errors only, recovery
  with a time constant of about 25 ms. Shielding underground lowers the rate (Cardani et al. 2021,
  https://doi.org/10.1038/s41467-021-23032-z). The peak extra relaxation rate (2 per µs) is an
  estimate from McEwen et al.'s 24 of 26 qubits decaying in a 1 µs idle, not a fitted value.
- Readout glitches (``ReadoutGlitches``). [assumption, no source]: a stand-in for corrupted
  samples, ADC overflow or stray pickup; no peer-reviewed rate exists in a usable form.
- Drive amplitude drift (``SlowDrift`` on ``NoiseModel.amplitude_drift``). [assumption, no source]:
  room-temperature electronics and amplifier gain drift with temperature; no measured number is used.

Readout-side channels (glitches, and the burst's effect on decay during the readout window) live in
their own functions below (``glitch_kicks_averaged``, ``glitch_kicks_shots``, ``readout_t1``) so a
different readout model can reuse them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SlowDrift:
    """A slowly varying offset x(t), t in seconds of simulated wall-clock time since the QPU was created.

    Used in Hz for the qubit frequency and as a fraction for the drive amplitude.
    kind "linear": x = rate * t.  "random_walk": Wiener process, Var[x(t + dt) - x(t)] = diffusion**2 * dt.
    "telegraph": x toggles between 0 and jump at each switch; switches at switch_times_s (sorted) plus
    Poisson switches at switch_rate_per_s. "step": x = jump for t >= switch_times_s[0], else 0.
    """

    kind: Literal["linear", "random_walk", "telegraph", "step"]
    rate: float = 0.0  # per second
    diffusion: float = 0.0  # per sqrt(second)
    jump: float = 0.0
    switch_times_s: tuple[float, ...] = ()
    switch_rate_per_s: float = 0.0


@dataclass(frozen=True)
class TLSDefect:
    """A two-level defect coupled to the transmon by g (a† σ- + a σ+).

    Müller, Cole, Lisenfeld 2019 (https://doi.org/10.1088/1361-6633/ab3a7e). Swap spectroscopy there
    found TLS lifetimes between 12 ns and 6 µs; Klimov et al. 2018
    (https://doi.org/10.1103/PhysRevLett.121.090502) report couplings of 50 to 500 kHz.

    Approximation: averaged readout (SimulatedQPU.measure_averaged) decays |e> in the readout window
    with the bare device T1, not the TLS-limited one; single shots use ``tls_effective_T1``.
    """

    detuning_hz: float  # f_TLS - f_01
    coupling_hz: float  # g / 2π
    T1_ns: float
    T2_ns: float  # must be <= 2 T1_ns


@dataclass(frozen=True)
class Bursts:
    """Quasiparticle bursts: extra relaxation peak_rate_per_ns * exp(-(t - t_e) / recovery_ns) after each event.

    McEwen et al. 2022 (https://doi.org/10.1038/s41567-021-01432-8): events about every 10 s,
    recovery with a time constant of about 25 ms, decay errors only.

    Approximation: averaged readout (SimulatedQPU.measure_averaged) ignores the burst's extra rate
    inside the readout window (bare T1 there); single shots include it.
    """

    event_times_s: tuple[float, ...] = ()
    rate_per_s: float = 0.0  # Poisson events; McEwen et al. 2022 report about 0.1 per second
    peak_rate_per_ns: float = 2e-3  # 2 per µs, estimate from McEwen et al. 2022 Fig. 1d
    recovery_ns: float = 25e6  # 25 ms


@dataclass(frozen=True)
class ReadoutGlitches:
    """Rare shots displaced by amplitude_v at a uniformly random phase. [assumption, no source]

    The kick is on the demodulated I/Q. 1 V is 50 mV scaled by the factor 20 of the readout-gain change
    in fix 07 (time-domain readout), so the kick keeps its size against the blob separation (about
    6 mV) and the noise; it is larger than any window average the ±0.5 V ADC could produce, which is
    what makes it a stand-in for corrupted data rather than a physical signal."""

    probability_per_shot: float
    amplitude_v: float = 1.0


@dataclass(frozen=True)
class NoiseModel:
    frequency_drift: SlowDrift | None = None
    amplitude_drift: SlowDrift | None = None
    tls: TLSDefect | None = None
    bursts: Bursts | None = None
    glitches: ReadoutGlitches | None = None
    # Every QUA program here averages with the shot loop outside the sweep loop. "sweep_outer" plays
    # all n_avg shots of one point before the next point; tests only.
    loop_order: Literal["averages_outer", "sweep_outer"] = "averages_outer"
    # Wall-clock time per shot outside the pulse sequence: state_initial.json thermalization_time
    # (250 µs) plus the 1 µs readout. Fix 06 (QUA single source) can derive these from the program.
    shot_overhead_ns: float = 251_000.0
    resonator_shot_ns: float = 2_000.0  # 02a: 1 µs readout plus 1 µs depletion, no thermalization
    saturation_pulse_ns: float = 20_000.0  # 03a saturation pulse length in state_initial.json
    amplitude_bins: int = 8

    @property
    def enabled(self) -> bool:
        return any(c is not None for c in (self.frequency_drift, self.amplitude_drift, self.tls, self.bursts, self.glitches))


@dataclass
class NoiseRecord:
    """What the channels did during one QPU call, for tests and truth helpers (never for fits)."""

    method: str
    times_ns: np.ndarray  # (n_shots_per_point, n_points) start time of each shot
    f01_offset_hz: np.ndarray  # same shape; drift only (the quasi-static part is not recorded)
    amplitude_scale: np.ndarray  # same shape; 1 + amplitude drift
    extra_relaxation_per_ns: np.ndarray  # same shape; burst rate at each shot
    glitches: np.ndarray  # (n_points,) glitch count per point, or (n_shots,) bool for single shots


# ---------------------------------------------------------------- shot clock
def shot_times(durations_ns: np.ndarray, n_avg: int, clock_ns: float, loop_order: str) -> np.ndarray:
    """Start time (ns) of every shot, shape (n_avg, n_points), for per-shot durations ``durations_ns``.

    "averages_outer" (QM's order): pass n plays every point once, t[n, j] = clock + n T_pass + Σ_{i<j} d_i.
    "sweep_outer": all shots of point j before point j + 1, t[n, j] = clock + Σ_{i<j} n_avg d_i + n d_j.
    """
    d = np.asarray(durations_ns, dtype=float)
    before = np.concatenate([[0.0], np.cumsum(d)[:-1]])
    n = np.arange(n_avg)[:, None]
    if loop_order == "averages_outer":
        return clock_ns + n * d.sum() + before[None, :]
    if loop_order == "sweep_outer":
        return clock_ns + n_avg * before[None, :] + n * d[None, :]
    raise ValueError(f"unknown loop_order {loop_order!r}")


# ---------------------------------------------------------------- slow drift
def _switch_times(explicit: tuple[float, ...], rate: float, t_max: float, rng: np.random.Generator, state: dict) -> np.ndarray:
    """Explicit plus lazily generated Poisson event times up to t_max, kept in ``state``."""
    poisson = state.setdefault("poisson", [])
    if rate > 0:
        last = state.get("poisson_until", 0.0)
        t = poisson[-1] if poisson else 0.0
        while last < t_max:
            t = t + rng.exponential(1.0 / rate)
            poisson.append(t)
            last = t
        state["poisson_until"] = last
    return np.sort(np.concatenate([np.asarray(explicit, dtype=float), np.asarray(poisson, dtype=float)]))


def sample_drift(drift: SlowDrift, times_s: np.ndarray, rng: np.random.Generator, state: dict) -> np.ndarray:
    """Drift values at ``times_s`` (any shape; evaluated in sorted order).

    ``state`` carries continuity across calls: for "random_walk" the last time and value (start 0
    at t = 0), for "telegraph" the Poisson switch times generated so far. "linear" and "step" need
    no state and no RNG.
    """
    t = np.asarray(times_s, dtype=float)
    if drift.kind == "linear":
        return drift.rate * t
    if drift.kind == "step":
        if not drift.switch_times_s:
            return np.zeros_like(t)
        return np.where(t >= drift.switch_times_s[0], drift.jump, 0.0)
    if drift.kind == "telegraph":
        switches = _switch_times(drift.switch_times_s, drift.switch_rate_per_s, float(t.max(initial=0.0)), rng, state)
        n_switches = np.searchsorted(switches, t, side="right")
        return np.where(n_switches % 2 == 1, drift.jump, 0.0)
    if drift.kind == "random_walk":
        flat = t.ravel()
        order = np.argsort(flat, kind="stable")
        ts = flat[order]
        t_last = state.get("t_last", 0.0)
        x_last = state.get("x_last", 0.0)
        if ts.size and ts[0] < t_last:
            raise ValueError("random_walk drift sampled before its last sampled time")
        dt = np.diff(np.concatenate([[t_last], ts]))
        steps = rng.normal(size=ts.size) * drift.diffusion * np.sqrt(dt)
        values = x_last + np.cumsum(steps)
        if ts.size:
            state["t_last"], state["x_last"] = float(ts[-1]), float(values[-1])
        out = np.empty_like(flat)
        out[order] = values
        return out.reshape(t.shape)
    raise ValueError(f"unknown drift kind {drift.kind!r}")


# ---------------------------------------------------------------- bursts
def burst_rate(bursts: Bursts, times_s: np.ndarray, rng: np.random.Generator, state: dict) -> np.ndarray:
    """Extra relaxation rate (1/ns) at ``times_s``: Σ over events t_e <= t of peak exp(-(t - t_e) / recovery).

    Explicit event times plus lazily generated Poisson events (``rate_per_s``), kept in ``state``.
    """
    t = np.asarray(times_s, dtype=float)
    events = _switch_times(bursts.event_times_s, bursts.rate_per_s, float(t.max(initial=0.0)), rng, state)
    rate = np.zeros_like(t)
    for t_e in events:
        after = t >= t_e
        rate = rate + np.where(after, bursts.peak_rate_per_ns * np.exp(-np.clip(t - t_e, 0, None) * 1e9 / bursts.recovery_ns), 0.0)
    return rate


BURST_LEVEL_EXPONENTS = (0.0, -0.5, -1.0, -1.5, -2.0)  # levels peak * 10^x the populations are computed at
# Rates below peak * 10^-2.25 (1.1e-5 per ns, about 0.54 / T1 for T1 = 48 µs) count as no burst, so
# the tail of each burst below that is dropped; the rest snaps to the nearest half decade, so a
# shot's extra rate is off by up to a factor 10^0.25 = 1.8.
BURST_IGNORE_EXPONENT = -2.25


def burst_levels(bursts: Bursts, rates: np.ndarray) -> np.ndarray:
    """Snap each rate (in log) to the nearest of the five levels; 0 for rates below the ignore cut."""
    rates = np.asarray(rates, dtype=float)
    peak = bursts.peak_rate_per_ns
    out = np.zeros_like(rates)
    hit = rates >= peak * 10**BURST_IGNORE_EXPONENT
    if np.any(hit):
        exps = np.array(BURST_LEVEL_EXPONENTS)
        logs = np.log10(rates[hit] / peak)
        nearest = exps[np.argmin(np.abs(logs[:, None] - exps[None, :]), axis=1)]
        out[hit] = peak * 10**nearest
    return out


# ---------------------------------------------------------------- readout-side channels
def glitch_kicks_averaged(glitches: ReadoutGlitches, n_avg: int, n_points: int, rng: np.random.Generator):
    """Glitch count per averaged point, k_j ~ Binomial(n_avg, p), and the complex kick to add to each
    point's mean: Σ amplitude e^{iθ} / n_avg with θ uniform."""
    counts = rng.binomial(n_avg, glitches.probability_per_shot, size=n_points)
    kick = np.zeros(n_points, dtype=complex)
    for j in np.flatnonzero(counts):
        theta = rng.uniform(0, 2 * np.pi, counts[j])
        kick[j] = glitches.amplitude_v * np.exp(1j * theta).sum() / n_avg
    return counts, kick


def glitch_kicks_shots(glitches: ReadoutGlitches, n_shots: int, rng: np.random.Generator):
    """Which single shots glitch (probability p each) and their complex kick amplitude e^{iθ}."""
    hit = rng.random(n_shots) < glitches.probability_per_shot
    kick = np.zeros(n_shots, dtype=complex)
    kick[hit] = glitches.amplitude_v * np.exp(1j * rng.uniform(0, 2 * np.pi, int(hit.sum())))
    return hit, kick


def readout_t1(t1_ns: float, extra_relaxation_per_ns: np.ndarray) -> np.ndarray:
    """Per-shot T1 during the readout window: 1 / (1/T1 + Γx)."""
    return 1.0 / (1.0 / t1_ns + np.asarray(extra_relaxation_per_ns, dtype=float))


# ---------------------------------------------------------------- truth helpers (tests and figures only)
def run_mean_offset_hz(record: NoiseRecord) -> float:
    """Mean drift offset over all shots of the call."""
    return float(np.mean(record.f01_offset_hz))


def effective_ramsey_t2(device, record: NoiseRecord, t_max_ns: float = 200_000.0) -> float:
    """1/e time of device.ramsey_envelope(t) * |mean over shots of exp(2πi δ t)|, δ = record.f01_offset_hz
    minus its mean (fix 01's envelope times the drift's characteristic function). Bisection on 0..t_max_ns."""
    delta = np.ravel(record.f01_offset_hz) - np.mean(record.f01_offset_hz)

    def envelope(t: float) -> float:
        return float(device.ramsey_envelope(t)) * abs(np.mean(np.exp(2j * np.pi * delta * t * 1e-9)))

    lo, hi = 0.0, t_max_ns
    if envelope(hi) > math.exp(-1):
        return math.inf
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if envelope(mid) > math.exp(-1):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tls_effective_T1(device, tls: TLSDefect) -> float:
    """Klimov et al. 2018 (https://doi.org/10.1103/PhysRevLett.121.090502, suppl. S3) weak-coupling T1:
    1/T1 + 2 g² Γ / (Γ² + Δ²) in angular units, Γ = 1/T2_TLS + 1/T2echo (TLS plus Markovian qubit
    decoherence), Δ = 2π detuning_hz. Valid for Γ > 2π g (weak coupling)."""
    g = 2 * math.pi * tls.coupling_hz * 1e-9
    gamma = 1.0 / tls.T2_ns + 1.0 / device.T2echo
    delta = 2 * math.pi * tls.detuning_hz * 1e-9
    return 1.0 / (1.0 / device.T1 + 2 * g**2 * gamma / (gamma**2 + delta**2))


def dressed_qubit_frequency(device, tls: TLSDefect) -> float:
    """Qubit-like eigenfrequency of [[f_01, g], [g, f_01 + Δ]]: f_01 + Δ/2 - sign(Δ) sqrt(g² + Δ²/4).
    NaN for Δ == 0 (equal mixing, no qubit-like branch)."""
    delta, g = tls.detuning_hz, tls.coupling_hz
    if delta == 0:
        return float("nan")
    return device.f_01 + delta / 2 - math.copysign(math.sqrt(g**2 + delta**2 / 4), delta)
