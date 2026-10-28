"""What the calibration nodes talk to in place of an OPX and a fridge.

Each method corresponds to the data a QUA program would stream back: I and Q per sweep
point, either averaged over ``n_avg`` shots or as raw single shots. Populations come from
:mod:`simulator.transmon`; the dispersive readout turns them into I/Q with shot noise.

Optional noise channels (:mod:`simulator.noise`: slow frequency and amplitude drift, a coupled TLS,
quasiparticle bursts, readout glitches) act only when a ``NoiseModel`` with a channel set is passed.
Without one every method runs exactly the noise-free code and draws exactly the same random numbers;
only the simulated wall clock (``clock_ns``) advances. Each noise channel draws from its own
generator, spawned from the seed, so switching a channel on never changes the base draws.

Known limitations of the noise path: 07 plays its two preparations as two calls (all |g> shots,
then all |e> shots) while the QUA program interleaves them per shot, so a slow channel reaches the
|e> block half a run later than it would. Saturation spectroscopy ignores bursts (a 20 µs
saturation pulse during a burst still ends near P_e = 0.5 minus the extra relaxation; not modelled)
and does not combine a TLS with drift (``NotImplementedError``). Single-shot readout with a TLS uses
the weak-coupling ``tls_effective_T1`` for decay during the readout window.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.interpolate import CubicSpline

from simulator.device import TRUE_DEVICE, Device
from simulator.readout import (
    ADC_RAW_PER_VOLT,
    IQ_DEMOD,
    TWO_PI_NS,
    DemodWindow,
    ReadoutSettings,
    cavity_field,
    demod_window,
    window_theta,
)
from simulator.noise import (
    NoiseModel,
    NoiseRecord,
    burst_levels,
    burst_rate,
    glitch_kicks_averaged,
    glitch_kicks_shots,
    readout_t1,
    sample_drift,
    shot_times,
    tls_effective_T1,
)
from simulator.transmon import (
    QUASISTATIC_MIN_PHASE_RAD,
    Pulse,
    Segment,
    Transmon,
    quasistatic_offsets,
    saturation_excited_population,
    sequence_duration,
    steady_state_excited_population,
)

MAX_DRIFT_GRID_NODES = 160  # offset-grid size above which _noisy_populations refuses (runtime guard)
MAX_DISTINCT_OFFSETS = 8  # up to this many distinct drift offsets per point are averaged exactly
# Clean-path wall clock per shot, the NoiseModel defaults (the clean path never builds a NoiseModel).
CLEAN_SHOT_OVERHEAD_NS = 251_000.0
CLEAN_RESONATOR_SHOT_NS = 2_000.0
CLEAN_SATURATION_PULSE_NS = 20_000.0


@dataclasses.dataclass(frozen=True)
class AdcTrace:
    readout_time: np.ndarray  # ns from the start of the recorded window, shape (n,)
    averaged: np.ndarray  # raw units, shape (2, n): inputs 1 and 2, float
    single_run: np.ndarray  # raw units, shape (2, n): int16, the last shot


def notch_s21(frequency, f0: float, kappa: float, kappa_ext_ratio: float):
    """Transmission past a notch-type resonator: a Lorentzian dip with a phase roll.

    The ideal notch model of Probst et al. 2015 (https://doi.org/10.1063/1.4907935), without
    cable delay, impedance mismatch or background.
    """
    return 1 - kappa_ext_ratio / (1 + 2j * (np.asarray(frequency) - f0) / kappa)


class SimulatedQPU:
    def __init__(self, device: Device = TRUE_DEVICE, seed: int | None = None, noise: NoiseModel | None = None):
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        # Separate streams per channel, so a channel never changes the base draws of self.rng.
        self._drift_rng, self._amp_rng, self._burst_rng, self._glitch_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
        )
        self.clock_ns = 0.0
        self.records: list[NoiseRecord] = []
        self._noise_state: dict = {"frequency": {}, "amplitude": {}, "bursts": {}}
        self._transmon = Transmon(device, tls=noise.tls if noise is not None else None)
        self._burst_transmons: dict[float, Transmon] = {}
        self._windows: dict = {}

    @property
    def _noisy(self) -> bool:
        return self.noise is not None and self.noise.enabled

    # ------------------------------------------------------------------ readout (simulator.readout)
    def _window(self, readout: ReadoutSettings, demod, frequency: float | None = None) -> tuple[DemodWindow, np.ndarray]:
        """The demod window for (settings, demod, frequency) and its static |g>, |e>, |f> outputs (3, n_out),
        cached: kernels and static sums are built once per frequency, never per shot."""
        frequency = readout.frequency if frequency is None else float(frequency)
        weights = tuple(readout.weights[n][k].tobytes() for d in demod for n, _ in d for k in (0, 1))
        key = (readout, weights, demod, frequency)
        cached = self._windows.get(key)
        if cached is None:
            if len(self._windows) > 4096:
                self._windows.clear()
            window = demod_window(readout, self.device, demod, frequency)
            d = self.device
            static = np.stack([window.static_outputs(f) for f in (d.f_res_g, d.f_res_e, d.f_res_f)])
            self._windows[key] = cached = (window, static)
        return cached

    def _noise(self, cov: np.ndarray, n: int) -> np.ndarray:
        """(n, n_out) Gaussian noise with covariance ``cov`` (positive semi-definite; zero allowed)."""
        w, v = np.linalg.eigh(cov)
        factor = v * np.sqrt(np.clip(w, 0, None))
        return self.rng.standard_normal((n, cov.shape[0])) @ factor.T

    def _draw_states(self, populations: np.ndarray, n_shots: int) -> np.ndarray:
        """States 0, 1, 2 per shot from (3,) populations, or from per-shot (n_shots, 3) populations."""
        pops = np.clip(np.asarray(populations, dtype=float), 0, None)
        if pops.ndim == 1 or np.all(pops == pops[0]):
            p = pops if pops.ndim == 1 else pops[0]
            return self.rng.choice(3, size=n_shots, p=p / p.sum())
        pops = pops / pops.sum(axis=1, keepdims=True)
        cdf = np.cumsum(pops, axis=1)
        cdf /= cdf[:, -1:]
        u = self.rng.random(n_shots)
        return (u[:, None] >= cdf).sum(axis=1)

    def measure_shots(self, populations, readout: ReadoutSettings, n_shots: int, demod=IQ_DEMOD, frequency=None, t1=None):
        """Single-shot demod outputs (volts) for shots drawn from ``populations`` (3,) or (n_shots, 3).

        |e> shots decay to |g> at t_d ~ Exp(T1) (``t1`` overrides T1, scalar or per shot) after the
        pulse reaches the resonator; the field relaxes to the |g> response from then on
        (``simulator.readout.decay_window_sums``). Returns one array per demod output.
        """
        window, static = self._window(readout, demod, frequency)
        states = self._draw_states(populations, n_shots)
        t_decay = self.rng.exponential(self.device.T1 if t1 is None else t1, size=n_shots)
        out = static[states]
        decays = (states == 1) & (t_decay < window.t_end)
        if np.any(decays):
            out[decays] = window.decay_outputs(t_decay[decays])
        out = out + self._noise(window.cov, n_shots)
        return tuple(out.T)

    def measure_averaged(self, populations, readout: ReadoutSettings, n_avg: int, demod=IQ_DEMOD, frequencies=None, t1=None):
        """Averaged demod outputs (volts) per point, populations (3, n_points).

        Per point a multinomial over states and per-shot decay times for its |e> shots; the mean of the
        noiseless outputs plus noise with the single-shot covariance over n_avg. ``frequencies``
        (n_points,) sets the readout frequency per point (02a, 08a); otherwise all points share
        ``readout.frequency`` and one kernel.

        Approximation: the |e> shots decay in the window with ``t1`` or the bare device T1, not with a
        TLS-limited T1 or a quasiparticle burst's extra rate; those channels act on the populations
        before the readout only (single_shots does use them in the window).
        """
        p = np.clip(np.asarray(populations, dtype=float), 0, None)
        p = p / p.sum(axis=0)
        n_points = p.shape[1]
        counts = np.stack([self.rng.multinomial(n_avg, p[:, k]) for k in range(n_points)], axis=1)
        n_e = counts[1]
        t_decay = self.rng.exponential(self.device.T1 if t1 is None else t1, size=int(n_e.sum()))
        point = np.repeat(np.arange(n_points), n_e)
        if frequencies is None:
            windows = [self._window(readout, demod)] * n_points
        else:
            windows = [self._window(readout, demod, f) for f in np.asarray(frequencies, dtype=float)]
        n_out = windows[0][0].K.shape[0]
        total = np.zeros((n_points, n_out))
        if frequencies is None:
            window, static = windows[0]
            total += counts[0][:, None] * static[0] + counts[2][:, None] * static[2]
            late = t_decay >= window.t_end
            np.add.at(total, point[late], static[1])
            if np.any(~late):
                np.add.at(total, point[~late], window.decay_outputs(t_decay[~late]))
        else:
            for k, (window, static) in enumerate(windows):
                total[k] = counts[0, k] * static[0] + counts[2, k] * static[2]
                td = t_decay[point == k]
                late = td >= window.t_end
                total[k] += late.sum() * static[1]
                if np.any(~late):
                    total[k] += window.decay_outputs(td[~late]).sum(axis=0)
        # One covariance for every point: sigma^2 Re(K K^H)/L^2 is frequency independent only when the
        # weights satisfy P1^2 + P2^2 = 1 on every sample (the cos/sin pairs of the config); with other
        # weights and per-point frequencies this is an approximation.
        mean = total / n_avg + self._noise(windows[0][0].cov / n_avg, n_points)
        return tuple(mean.T)

    def expected_iq(self, state: int, readout: ReadoutSettings, demod=IQ_DEMOD, frequency=None, decay: bool = True) -> complex:
        """Noiseless mean output (first + i second demod output, volts) for a pure state 0, 1, 2.

        For state 1 with ``decay`` the average over t_d ~ Exp(T1) uses the midpoint rule on 0.5 ns
        steps over the window plus the survival weight exp(-t_end/T1) on the no-decay value.
        Truth-only helper for reference values, figures and tests; nodes never call it.
        """
        window, static = self._window(readout, demod, frequency)
        out = static[state]
        if state == 1 and decay and window.t_end > 0:
            T1 = self.device.T1
            t = np.arange(0.25, window.t_end, 0.5)
            w = np.exp(-(t - 0.25) / T1) * (1 - np.exp(-0.5 / T1))
            out = (w[:, None] * window.decay_outputs(t)).sum(axis=0) + np.exp(-t.size * 0.5 / T1) * static[1]
        return complex(out[0], out[1] if out.size > 1 else 0.0)

    def adc_trace(self, readout: ReadoutSettings, n_avg: int, populations=None) -> AdcTrace:
        """The raw ADC record of ``pulse_length + smearing`` samples from the configured time of flight.

        Averaged over n_avg shots (each shot's state and decay time drawn, its field from
        ``cavity_field``) plus the offset difference and noise σ_s / sqrt(n_avg) per sample and input;
        and one extra single run with full noise, rounded and clipped to the 12-bit range (int16).
        Raw units: raw = -2^12 V. ``populations`` defaults to the thermal state.
        """
        d = self.device
        if populations is None:
            p_e = d.thermal_excited_population
            populations = np.array([1 - p_e, p_e, 0.0])
        n = readout.pulse_length + readout.smearing
        theta = window_theta(readout, d, n)
        carrier = np.exp(1j * TWO_PI_NS * readout.intermediate_frequency * theta)
        f = readout.frequency
        ga = d.readout_gain * readout.amplitude
        drive = ((theta >= 0) & (theta < readout.pulse_length)).astype(float)

        def signal(f_before, f_after=None, t_switch=np.inf):
            f_after = f_before if f_after is None else f_after
            b = cavity_field(theta, f, f_before, f_after, t_switch, readout.pulse_length, d.kappa)
            return ga * (drive - d.kappa_ext_ratio * b) * carrier

        static = [signal(d.f_res_g), signal(d.f_res_e), signal(d.f_res_f)]
        offsets = (np.array(d.adc_offsets) - np.array(readout.input_offsets))[:, None]

        def shots(k: int) -> np.ndarray:
            states = self._draw_states(populations, k)
            t_decay = self.rng.exponential(d.T1, size=k)
            total = np.zeros(n, complex)
            for s in (0, 2):
                total += np.count_nonzero(states == s) * static[s]
            for td in t_decay[states == 1]:
                total += static[1] if td >= theta[-1] + 1 else signal(d.f_res_e, d.f_res_g, td)
            return total

        mean = shots(n_avg) / n_avg
        averaged = np.stack([mean.real, mean.imag]) + offsets + self.rng.normal(0, d.adc_sample_noise / np.sqrt(n_avg), (2, n))
        one = shots(1)
        single = np.stack([one.real, one.imag]) + offsets + self.rng.normal(0, d.adc_sample_noise, (2, n))
        raw_single = np.clip(np.round(-ADC_RAW_PER_VOLT * single), -2048, 2047).astype(np.int16)
        return AdcTrace(readout_time=np.arange(n), averaged=-ADC_RAW_PER_VOLT * averaged, single_run=raw_single)

    # ------------------------------------------------------------------ experiments
    def resonator_spectroscopy(self, frequencies, readout: ReadoutSettings, n_avg: int):
        """Sweep the readout tone with the qubit left in its thermal state."""
        frequencies = np.asarray(frequencies, dtype=float)
        p_e = self.device.thermal_excited_population
        populations = np.tile(np.array([[1 - p_e], [p_e], [0.0]]), (1, frequencies.size))
        if not self._noisy:
            self._advance_clock(np.full(frequencies.size, CLEAN_RESONATOR_SHOT_NS), n_avg)
            return self.measure_averaged(populations, readout, n_avg, frequencies=frequencies)
        record = self._start_record("resonator_spectroscopy", np.full(frequencies.size, self.noise.resonator_shot_ns), n_avg)
        I, Q = self.measure_averaged(populations, readout, n_avg, frequencies=frequencies)
        return self._finish_averaged(record, I, Q, n_avg)

    def saturation_spectroscopy(self, drive_frequencies, drive_amplitude: float, readout: ReadoutSettings, n_avg: int):
        """Long weak drive at each frequency, then readout at a fixed frequency."""
        drive_frequencies = np.asarray(drive_frequencies, dtype=float)
        if not self._noisy:
            self._advance_clock(np.full(drive_frequencies.size, CLEAN_SHOT_OVERHEAD_NS + CLEAN_SATURATION_PULSE_NS), n_avg)
            p_e = saturation_excited_population(self.device, drive_frequencies, drive_amplitude)
            populations = np.stack([1 - p_e, p_e, np.zeros_like(p_e)])
            return self.measure_averaged(populations, readout, n_avg)
        noise = self.noise
        record = self._start_record(
            "saturation_spectroscopy", np.full(drive_frequencies.size, noise.shot_overhead_ns + noise.saturation_pulse_ns), n_avg
        )
        if noise.tls is not None:
            if noise.frequency_drift is not None or noise.amplitude_drift is not None:
                raise NotImplementedError("saturation spectroscopy with a TLS and a drift is not modelled")
            p_e = steady_state_excited_population(self.device, noise.tls, drive_frequencies, drive_amplitude)
        elif noise.frequency_drift is not None or noise.amplitude_drift is not None:
            # Exact per shot: the Bloch line at each shot's frequency offset and amplitude scale.
            p_e = saturation_excited_population(
                self.device, drive_frequencies[None, :] - record.f01_offset_hz, drive_amplitude * record.amplitude_scale
            ).mean(axis=0)
        else:
            p_e = saturation_excited_population(self.device, drive_frequencies, drive_amplitude)
        populations = np.stack([1 - p_e, p_e, np.zeros_like(p_e)])
        I, Q = self.measure_averaged(populations, readout, n_avg)
        return self._finish_averaged(record, I, Q, n_avg)

    def populations(self, sequences: list[list[Segment]], drive_frequency: float) -> np.ndarray:
        """Noise-free (3, n_sequences) populations at the end of each pulse sequence, shot-averaged
        over the quasi-static qubit frequency offset (``Transmon.shot_averaged_state``)."""
        transmon = self._transmon
        return np.stack(
            [transmon.populations(transmon.shot_averaged_state(seq, drive_frequency)) for seq in sequences], axis=1
        )

    def run_sequences(
        self,
        sequences: list[list[Segment]],
        drive_frequency: float,
        readout: ReadoutSettings,
        n_avg: int,
        frequencies: np.ndarray | None = None,
    ):
        """Averaged I/Q after each pulse sequence. ``frequencies`` (one per sequence, Hz) reads each
        point out at its own frequency (08a); otherwise all points use ``readout.frequency``."""
        if not self._noisy:
            self._advance_clock(np.array([CLEAN_SHOT_OVERHEAD_NS + sequence_duration(seq) for seq in sequences]), n_avg)
            return self.measure_averaged(self.populations(sequences, drive_frequency), readout, n_avg, frequencies=frequencies)
        durations = np.array([self.noise.shot_overhead_ns + sequence_duration(seq) for seq in sequences])
        record = self._start_record("run_sequences", durations, n_avg)
        populations = self._noisy_populations(sequences, drive_frequency, record)
        I, Q = self.measure_averaged(populations, readout, n_avg, frequencies=frequencies)
        return self._finish_averaged(record, I, Q, n_avg)

    def single_shots(self, sequence: list[Segment], drive_frequency: float, readout: ReadoutSettings, n_shots: int, demod=IQ_DEMOD):
        """Raw single-shot I/Q after one sequence (``measure_shots``). Shots in |e> can decay to |g>
        during the readout window, which smears them along the line between the two blobs. Each
        shot's state is drawn from the shot-averaged populations, which is exact for independent
        per-shot offsets: the probability of each outcome is the average over the offset of its
        probability."""
        if not self._noisy:
            self._advance_clock(np.array([CLEAN_SHOT_OVERHEAD_NS + sequence_duration(sequence)]), n_shots)
            pops = self.populations([sequence], drive_frequency)[:, 0]
            return self.measure_shots(pops, readout, n_shots, demod=demod)
        noise = self.noise
        # One row per shot: a single sweep point played n_shots times.
        record = self._start_record("single_shots", np.array([noise.shot_overhead_ns + sequence_duration(sequence)]), n_shots)
        per_shot = self._noisy_populations([sequence], drive_frequency, record, per_shot=True)[:, :, 0]  # (n_shots, 3)
        t1_base = self._readout_t1()
        rates = record.extra_relaxation_per_ns[:, 0]
        t1 = t1_base if not np.any(rates) else readout_t1(t1_base, rates)
        outputs = self.measure_shots(per_shot, readout, n_shots, demod=demod, t1=t1)
        record.glitches = np.zeros(n_shots, dtype=bool)
        if noise.glitches is not None:
            record.glitches, kick = glitch_kicks_shots(noise.glitches, n_shots, self._glitch_rng)
            outputs = (outputs[0] + kick.real, outputs[1] + kick.imag, *outputs[2:])
        return outputs

    def _readout_t1(self) -> float:
        """T1 during the single-shot readout window; with a TLS the weak-coupling ``tls_effective_T1``.
        Averaged data (measure_averaged) decay with the bare device T1 in the window: the TLS and
        burst channels act on their populations only."""
        if self.noise is not None and self.noise.tls is not None:
            return tls_effective_T1(self.device, self.noise.tls)
        return self.device.T1

    # ------------------------------------------------------------------ noise channels
    def _advance_clock(self, durations_ns: np.ndarray, n_avg: int) -> None:
        """Clean path: only the wall clock moves (default shot durations; no random draws)."""
        self.clock_ns = float(self.clock_ns + n_avg * np.sum(durations_ns))

    def _start_record(self, method: str, durations_ns: np.ndarray, n_avg: int) -> NoiseRecord:
        """Shot start times for this call, the drift and burst values at each shot, and the clock advance."""
        noise = self.noise
        times = shot_times(durations_ns, n_avg, self.clock_ns, noise.loop_order)
        self.clock_ns = float(self.clock_ns + n_avg * np.sum(durations_ns))
        t_s = times * 1e-9
        f_off = (
            sample_drift(noise.frequency_drift, t_s, self._drift_rng, self._noise_state["frequency"])
            if noise.frequency_drift is not None
            else np.zeros_like(times)
        )
        scale = (
            1.0 + sample_drift(noise.amplitude_drift, t_s, self._amp_rng, self._noise_state["amplitude"])
            if noise.amplitude_drift is not None
            else np.ones_like(times)
        )
        rates = (
            burst_rate(noise.bursts, t_s, self._burst_rng, self._noise_state["bursts"])
            if noise.bursts is not None
            else np.zeros_like(times)
        )
        record = NoiseRecord(method, times, f_off, scale, rates, np.zeros(times.shape[1], dtype=int))
        self.records.append(record)
        return record

    def _finish_averaged(self, record: NoiseRecord, I: np.ndarray, Q: np.ndarray, n_avg: int):
        """Readout glitches on averaged points (from their own generator)."""
        if self.noise.glitches is None:
            return I, Q
        counts, kick = glitch_kicks_averaged(self.noise.glitches, n_avg, I.size, self._glitch_rng)
        record.glitches = counts
        return I + kick.real, Q + kick.imag

    def _transmon_for(self, rate: float) -> Transmon:
        """The transmon with ``rate`` (1/ns) of extra relaxation, cached per burst level."""
        if rate == 0.0:
            return self._transmon
        if rate not in self._burst_transmons:
            self._burst_transmons[rate] = Transmon(self.device, tls=self.noise.tls, extra_relaxation_per_ns=rate)
        return self._burst_transmons[rate]

    def _offset_average(self, transmon: Transmon, seq: list[Segment], drive_frequency: float, deltas: np.ndarray) -> np.ndarray:
        """Populations (3,) averaged over the shots' drift offsets ``deltas`` and the quasi-static noise.

        A small phase spread runs once at the mean offset. Up to MAX_DISTINCT_OFFSETS distinct
        offsets (telegraph, step, no drift) are averaged exactly. Otherwise the single-offset
        populations are evaluated on a grid of offsets (multiples of h, so pulse superoperators are
        reused across points), interpolated with cubic splines and averaged over shots and
        Gauss-Hermite nodes; interpolation error about (2 pi h T)^4 / 384 <= 4e-4 of the fringe
        amplitude. The grid reaches half a sigma past the outermost Gauss-Hermite node.
        """
        duration = sequence_duration(seq)
        spread = 2 * np.pi * float(deltas.max() - deltas.min()) * duration * 1e-9
        if spread < QUASISTATIC_MIN_PHASE_RAD:
            return transmon.populations(transmon.shot_averaged_state(seq, drive_frequency, f01_offset=float(deltas.mean())))
        values, counts = np.unique(deltas, return_counts=True)
        if values.size <= MAX_DISTINCT_OFFSETS:
            pops = [transmon.populations(transmon.shot_averaged_state(seq, drive_frequency, f01_offset=float(v))) for v in values]
            return (np.array(pops) * counts[:, None]).sum(axis=0) / counts.sum()
        sigma_f = self.device.sigma_f_quasistatic
        nodes, w_k = quasistatic_offsets(self.device)  # offsets in Hz, weights summing to 1
        reach = float(np.abs(nodes).max()) + 0.5 * sigma_f
        h = min(2e3, 1.0 / (10 * duration * 1e-9))
        lo = np.floor((deltas.min() - reach) / h) - 1
        hi = np.ceil((deltas.max() + reach) / h) + 1
        grid = np.arange(lo, hi + 1) * h
        if grid.size > MAX_DRIFT_GRID_NODES:
            raise ValueError(
                f"frequency_drift: offset grid of {grid.size} nodes for a {duration:.0f} ns sequence "
                f"(limit {MAX_DRIFT_GRID_NODES}); lower the drift or shorten the sequence"
            )
        g = np.array([transmon.populations(transmon.evolve_superop(seq, drive_frequency, f01_offset=float(d))) for d in grid])
        spline = CubicSpline(grid, g, axis=0)
        points = (deltas[:, None] + nodes[None, :]).ravel()
        weights = np.tile(w_k, deltas.size) / deltas.size
        return (spline(points) * weights[:, None]).sum(axis=0)

    def _noisy_populations(self, sequences, drive_frequency: float, record: NoiseRecord, per_shot: bool = False) -> np.ndarray:
        """Populations (3, n_points) with the drift, amplitude and burst channels, from ``record``.

        Amplitude drift: each point's shots split into ``amplitude_bins`` equal-count groups sorted by
        scale, each group at its mean scale. Bursts: per-shot rates snapped to five levels, each level
        a cached Transmon with that extra relaxation, mixed by shot fraction. A TLS rides on the
        transmon instances. With ``per_shot`` the result is (n_shots, 3, n_points), each shot
        carrying its own group's populations.
        """
        noise = self.noise
        n_shots, n_points = record.times_ns.shape
        if noise.bursts is not None:
            levels = burst_levels(noise.bursts, record.extra_relaxation_per_ns)
        else:
            levels = np.zeros((n_shots, n_points))
        out = np.zeros((n_shots, 3, n_points)) if per_shot else np.zeros((3, n_points))
        for j, seq in enumerate(sequences):
            if noise.amplitude_drift is not None:
                order = np.argsort(record.amplitude_scale[:, j], kind="stable")
                groups = [idx for idx in np.array_split(order, min(noise.amplitude_bins, n_shots)) if idx.size]
            else:
                groups = [np.arange(n_shots)]
            for idx in groups:
                if noise.amplitude_drift is not None:
                    scale = float(record.amplitude_scale[idx, j].mean())
                    seq_g = [dataclasses.replace(s, amplitude=s.amplitude * scale) if isinstance(s, Pulse) else s for s in seq]
                else:
                    seq_g = seq
                for level in np.unique(levels[idx, j]):
                    sub = idx[levels[idx, j] == level]
                    pops = self._offset_average(self._transmon_for(float(level)), seq_g, drive_frequency, record.f01_offset_hz[sub, j])
                    if per_shot:
                        out[sub, :, j] = pops
                    else:
                        out[:, j] += pops * sub.size / n_shots
        return out
