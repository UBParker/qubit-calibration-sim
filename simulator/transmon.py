"""Three-level transmon dynamics with QuTiP.

The qubit is modelled as a Duffing oscillator (Koch et al. 2007, https://doi.org/10.1103/PhysRevA.76.042319)
truncated to |g>, |e>, |f> in the frame rotating at the drive frequency:

    H / hbar = 2*pi * [ -detuning * n + (anharmonicity / 2) * n (n - 1)
                        + (rabi(t) / 2) * (a e^{-i phase} + a^dag e^{i phase}) ]

with detuning = f_drive - f_01. Decay and dephasing enter as Lindblad collapse operators
(Lindblad 1976, https://doi.org/10.1007/BF01608499): decay sqrt(1/T1) a and white dephasing
sqrt(2/Tphi_white) n, so 1/T2echo = 1/(2 T1) + 1/Tphi_white (Krantz et al. 2019,
https://doi.org/10.1063/1.5089550). The master equation
is solved with QuTiP (Lambert et al. 2026, https://doi.org/10.1016/j.physrep.2025.10.001). Keeping |f> means strong
or short pulses leak out of the qubit subspace, the problem DRAG pulses were introduced to fix
(Motzoi et al. 2009, https://doi.org/10.1103/PhysRevLett.103.110501).

On top of the white dephasing, each shot sees a static qubit frequency offset delta drawn from
N(0, sigma_f^2) (``Device.sigma_f_quasistatic``), constant during the shot and independent between
shots. This is slow (1/f-type) noise in the static limit, which gives a Gaussian Ramsey decay that
a Hahn echo refocuses (Ithier et al. 2005, https://doi.org/10.1103/PhysRevB.72.134519; reviewed in
Krantz et al. 2019, https://doi.org/10.1063/1.5089550). The shot-averaged Ramsey coherence is
exp(-t/T2echo - (t/Tq)^2). ``Transmon.shot_averaged_state`` averages the density matrix over delta
with Gauss-Hermite quadrature (QUASISTATIC_NODES nodes). Sequences whose phase spread
2 pi sigma_f * duration is below QUASISTATIC_MIN_PHASE_RAD run once at delta = 0; the population
error from skipping is below that spread squared over 2.

Optionally (``Transmon(device, tls=...)``) the transmon couples to one two-level defect (TLS) by
g (a^dag sigma- + a sigma+), with the TLS's own T1 and T2, in a 6-dimensional Hilbert space (three
transmon levels times two TLS levels). Review: Muller, Cole, Lisenfeld 2019,
https://doi.org/10.1088/1361-6633/ab3a7e. No weak-coupling approximation is made, so the weak-coupling
relaxation Lorentzian of Klimov et al. 2018 (https://doi.org/10.1103/PhysRevLett.121.090502, suppl. S3)
and the strong-coupling split line both emerge. ``extra_relaxation_per_ns`` adds transmon relaxation
on top of 1/T1, for quasiparticle bursts (McEwen et al. 2022, https://doi.org/10.1038/s41567-021-01432-8).

Times are in ns and frequencies in Hz; QuTiP is fed GHz-ns units internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import qutip as qt

from simulator.device import Device

if TYPE_CHECKING:
    from simulator.noise import TLSDefect

N_LEVELS = 3
QUASISTATIC_NODES = 15  # Gauss-Hermite nodes; E[cos(aZ)] to 1e-4 for a <= 3.97
QUASISTATIC_MIN_PHASE_RAD = 0.01  # below this spread a sequence runs once at delta = 0


@dataclass(frozen=True)
class Pulse:
    """A Gaussian drive pulse, matching the waveform used in the QUA config.

    ``phase`` is the rotation axis in the frame rotating at the drive, as the drive term
    (a e^{-i phase} + a^dag e^{i phase}) defines it: 0 is the x axis, pi/2 the y axis of that
    frame. It is minus the lab phase of the played pulse (the angle of the QUA waveform I + iQ
    plus 2 pi times the element's frame phase in turns), because moving a lab drive
    cos(w t + phi_L) into the rotating frame keeps a e^{i phi_L} + a^dag e^{-i phi_L}. So the
    config's y90 (the Gaussian on the Q channel, lab phase +pi/2) is phase -pi/2 here, and
    frame_rotation_2pi(x) before a pulse subtracts 2 pi x from its phase. Pinned by
    tests/test_qua_trace.py (test_pulse_library_matches_config, test_virtual_detuning_equals_real_detuning).
    """

    amplitude: float  # volts
    length: float  # ns
    sigma: float  # ns
    phase: float = 0.0  # radians; 0 = x axis, pi/2 = y axis of the drive frame; see the class docstring for the sign
    drive_frequency: float | None = None  # Hz; None means the sequence default


@dataclass(frozen=True)
class Idle:
    duration: float  # ns
    drive_frequency: float | None = None


Segment = Pulse | Idle


def quasistatic_offsets(device: Device) -> tuple[np.ndarray, np.ndarray]:
    """Qubit frequency offsets (Hz) and weights (sum 1) for the shot average over N(0, sigma_f^2)."""
    sigma = device.sigma_f_quasistatic
    if sigma == 0.0:
        return np.array([0.0]), np.array([1.0])
    x, w = np.polynomial.hermite_e.hermegauss(QUASISTATIC_NODES)
    return sigma * x, w / w.sum()


def sequence_duration(segments: list[Segment]) -> float:
    """Total duration in ns: pulse lengths plus idle durations (negative idles count as 0)."""
    return float(sum(seg.length if isinstance(seg, Pulse) else max(seg.duration, 0.0) for seg in segments))


def gaussian_envelope(t: np.ndarray, length: float, sigma: float) -> np.ndarray:
    """Gaussian truncated to [0, length] and shifted so it starts and ends at zero."""
    centre = length / 2
    g = np.exp(-((t - centre) ** 2) / (2 * sigma**2))
    edge = np.exp(-(centre**2) / (2 * sigma**2))
    return (g - edge) / (1 - edge)


class Transmon:
    def __init__(self, device: Device, tls: TLSDefect | None = None, extra_relaxation_per_ns: float = 0.0):
        self.device = device
        self.tls = tls
        if tls is None:
            a = qt.destroy(N_LEVELS)
            eye = qt.qeye(N_LEVELS)
        else:
            if tls.T2_ns > 2 * tls.T1_ns:
                raise ValueError("TLS T2 must not exceed 2 T1")
            a = qt.tensor(qt.destroy(N_LEVELS), qt.qeye(2))
            eye = qt.tensor(qt.qeye(N_LEVELS), qt.qeye(2))
        self.a = a
        self.n = a.dag() * a
        self._anh = 0.5 * self.n * (self.n - eye)
        self.c_ops = [np.sqrt(1.0 / device.T1) * a, np.sqrt(2.0 / device.Tphi_white) * self.n]
        if extra_relaxation_per_ns > 0:
            # Bursts add relaxation and leave the white dephasing alone.
            self.c_ops.append(np.sqrt(extra_relaxation_per_ns) * a)
        if tls is None:
            self.sm = None
            self.n_total = self.n
            self.projectors = [qt.basis(N_LEVELS, k).proj() for k in range(N_LEVELS)]
        else:
            self.sm = qt.tensor(qt.qeye(N_LEVELS), qt.destroy(2))
            self.nt = self.sm.dag() * self.sm
            self.n_total = self.n + self.nt
            self.c_ops.append(np.sqrt(1.0 / tls.T1_ns) * self.sm)
            if tls.T2_ns < 2 * tls.T1_ns:
                tphi = 1.0 / (1.0 / tls.T2_ns - 1.0 / (2 * tls.T1_ns))
                self.c_ops.append(np.sqrt(2.0 / tphi) * self.nt)
            self.projectors = [qt.tensor(qt.basis(N_LEVELS, k).proj(), qt.qeye(2)) for k in range(N_LEVELS)]
        self._pulse_cache: dict[tuple, qt.Qobj] = {}

    def thermal_state(self) -> qt.Qobj:
        """Transmon thermal state; with a TLS, tensored with the TLS ground state (its thermalization
        is far shorter than the 250 µs between shots)."""
        p = self.device.thermal_excited_population
        if self.tls is None:
            return (1 - p) * self.projectors[0] + p * self.projectors[1]
        transmon = (1 - p) * qt.basis(N_LEVELS, 0).proj() + p * qt.basis(N_LEVELS, 1).proj()
        return qt.tensor(transmon, qt.basis(2, 0).proj())

    def qubit_state(self, rho: qt.Qobj) -> qt.Qobj:
        """The transmon's reduced density matrix (``rho`` itself without a TLS)."""
        return rho if self.tls is None else rho.ptrace(0)

    def _static_h(self, drive_frequency: float, f01_offset: float = 0.0) -> qt.Qobj:
        """Static part of H; ``f01_offset`` (Hz) is the shot's qubit frequency offset delta."""
        detuning_ghz = (drive_frequency - self.device.f_01 - f01_offset) * 1e-9
        anh_ghz = self.device.anharmonicity * 1e-9
        h = 2 * np.pi * (-detuning_ghz * self.n + anh_ghz * self._anh)
        if self.tls is not None:
            # The TLS frequency does not move with the qubit offset delta.
            tls_detuning_ghz = (drive_frequency - (self.device.f_01 + self.tls.detuning_hz)) * 1e-9
            g_ghz = self.tls.coupling_hz * 1e-9
            h = h + 2 * np.pi * (-tls_detuning_ghz * self.nt + g_ghz * (self.a.dag() * self.sm + self.a * self.sm.dag()))
        return h

    def _idle_propagator(self, drive_frequency: float, duration: float, f01_offset: float = 0.0) -> qt.Qobj:
        # Constant generator, so the exact propagator is a matrix exponential. Cheaper
        # and more accurate than integrating fast phase rotations over tens of us.
        return (qt.liouvillian(self._static_h(drive_frequency, f01_offset), self.c_ops) * duration).expm()

    def _pulse_hamiltonian(
        self, seg: Pulse, drive_frequency: float, f01_offset: float = 0.0
    ) -> tuple[list, np.ndarray]:
        """Time-dependent Hamiltonian of a pulse and the grid its envelope is sampled on."""
        rabi_ghz = self.device.rabi_hz_per_volt * seg.amplitude * 1e-9
        drive_op = 0.5 * (self.a * np.exp(-1j * seg.phase) + self.a.dag() * np.exp(1j * seg.phase))
        tlist = np.linspace(0.0, seg.length, max(int(seg.length * 4), 50))
        envelope = gaussian_envelope(tlist, seg.length, seg.sigma)
        # The controller amplitude sets the peak Rabi rate of the envelope.
        coeff = qt.coefficient(2 * np.pi * rabi_ghz * envelope, tlist=tlist)
        return [self._static_h(drive_frequency, f01_offset), [drive_op, coeff]], tlist

    def evolve(
        self, segments: list[Segment], drive_frequency: float, rho0: qt.Qobj | None = None, f01_offset: float = 0.0
    ) -> qt.Qobj:
        """One shot at qubit frequency offset ``f01_offset`` (Hz): mesolve for pulses, exact idles."""
        rho = self.thermal_state() if rho0 is None else rho0
        for seg in segments:
            f_drive = seg.drive_frequency if seg.drive_frequency is not None else drive_frequency
            if isinstance(seg, Idle):
                if seg.duration <= 0:
                    continue
                propagator = self._idle_propagator(f_drive, seg.duration, f01_offset)
                rho = qt.vector_to_operator(propagator * qt.operator_to_vector(rho))
                continue
            h, tlist = self._pulse_hamiltonian(seg, f_drive, f01_offset)
            rho = qt.mesolve(h, rho, tlist, c_ops=self.c_ops).final_state
        return rho

    def pulse_superoperator(self, seg: Pulse, drive_frequency: float, f01_offset: float = 0.0) -> qt.Qobj:
        """Superoperator of one pulse, cached on (amplitude, length, sigma, drive_frequency - f01_offset).

        Only the phase-0 pulse is integrated. A pulse with phase phi is S_phi = to_super(U) S_0
        to_super(U^dag) with U = exp(i phi n). This is exact: U a U^dag = e^{-i phi} a, so U turns the
        phase-0 drive term into the phase-phi one; U commutes with the static Hamiltonian, which is
        diagonal in n; and it leaves both dissipators unchanged (a only picks up a global phase, n is
        invariant).

        With a TLS, U = exp(i phi n_total), n_total = n + sigma+ sigma-: U a U^dag = e^{-i phi} a and
        U sigma- U^dag = e^{-i phi} sigma-, so U also commutes with the exchange term a^dag sigma- +
        a sigma+, and the TLS dissipators are unchanged. The cache key then adds drive_frequency, since
        the TLS detuning in the drive frame depends on the drive frequency itself.
        """
        key = (seg.amplitude, seg.length, seg.sigma, drive_frequency - f01_offset)
        if self.tls is not None:
            key = (*key, drive_frequency)
        s0 = self._pulse_cache.get(key)
        if s0 is None:
            phase0 = Pulse(seg.amplitude, seg.length, seg.sigma)
            h, _ = self._pulse_hamiltonian(phase0, drive_frequency, f01_offset)
            s0 = qt.propagator(h, seg.length, c_ops=self.c_ops, options={"atol": 1e-10, "rtol": 1e-8})
            self._pulse_cache[key] = s0
        if seg.phase == 0:
            return s0
        u = (1j * seg.phase * self.n_total).expm()
        return qt.to_super(u) * s0 * qt.to_super(u.dag())

    def evolve_superop(
        self, segments: list[Segment], drive_frequency: float, rho0: qt.Qobj | None = None, f01_offset: float = 0.0
    ) -> qt.Qobj:
        """Same result as :meth:`evolve`, built as a product of superoperators (pulses cached)."""
        rho = self.thermal_state() if rho0 is None else rho0
        vec = qt.operator_to_vector(rho)
        for seg in segments:
            f_drive = seg.drive_frequency if seg.drive_frequency is not None else drive_frequency
            if isinstance(seg, Idle):
                if seg.duration <= 0:
                    continue
                vec = self._idle_propagator(f_drive, seg.duration, f01_offset) * vec
                continue
            vec = self.pulse_superoperator(seg, f_drive, f01_offset) * vec
        return qt.vector_to_operator(vec)

    def shot_averaged_state(
        self, segments: list[Segment], drive_frequency: float, rho0: qt.Qobj | None = None, f01_offset: float = 0.0
    ) -> qt.Qobj:
        """Density matrix averaged over the quasi-static qubit frequency offset.

        Short sequences (phase spread below QUASISTATIC_MIN_PHASE_RAD) run once at delta = 0 through
        :meth:`evolve`; longer ones sum :meth:`evolve_superop` over the Gauss-Hermite nodes.
        ``f01_offset`` (Hz, default 0) shifts every node, for a slow drift on top of the quasi-static noise.
        """
        spread = 2 * np.pi * self.device.sigma_f_quasistatic * sequence_duration(segments) * 1e-9
        if spread < QUASISTATIC_MIN_PHASE_RAD:
            return self.evolve(segments, drive_frequency, rho0, f01_offset=f01_offset)
        offsets, weights = quasistatic_offsets(self.device)
        total = None
        for d, w in zip(offsets, weights):
            rho = float(w) * self.evolve_superop(segments, drive_frequency, rho0, f01_offset=f01_offset + float(d))
            total = rho if total is None else total + rho
        return total

    def trajectory(
        self,
        segments: list[Segment],
        drive_frequency: float,
        samples_per_segment: int = 60,
        rho0: qt.Qobj | None = None,
        f01_offset: float = 0.0,
    ) -> tuple[np.ndarray, list[qt.Qobj]]:
        """Like :meth:`evolve`, but also returns the intermediate states.

        Returns times in ns from the start of the sequence and the density matrix at each time,
        starting with rho0 at t = 0 and adding samples_per_segment evenly spaced points per
        segment. Pulses are sampled through the mesolve output grid; idles apply the exact
        propagator of duration / samples_per_segment repeatedly. Zero-length idles add nothing.
        """
        if samples_per_segment < 1:
            raise ValueError("samples_per_segment must be at least 1")
        rho = self.thermal_state() if rho0 is None else rho0
        t0 = 0.0
        times = [t0]
        states = [rho]
        for seg in segments:
            f_drive = seg.drive_frequency if seg.drive_frequency is not None else drive_frequency
            if isinstance(seg, Idle):
                if seg.duration <= 0:
                    continue
                step = seg.duration / samples_per_segment
                propagator = self._idle_propagator(f_drive, step, f01_offset)
                vec = qt.operator_to_vector(rho)
                for k in range(1, samples_per_segment + 1):
                    vec = propagator * vec
                    times.append(t0 + k * step)
                    states.append(qt.vector_to_operator(vec))
                rho = states[-1]
                t0 += seg.duration
                continue
            h, _ = self._pulse_hamiltonian(seg, f_drive, f01_offset)
            sample_t = np.linspace(0.0, seg.length, samples_per_segment + 1)
            result = qt.mesolve(h, rho, sample_t, c_ops=self.c_ops)
            times.extend(t0 + sample_t[1:])
            states.extend(result.states[1:])
            rho = states[-1]
            t0 += seg.length
        return np.array(times), states

    def populations(self, rho: qt.Qobj) -> np.ndarray:
        return np.array([qt.expect(p, rho).real for p in self.projectors])


def saturation_excited_population(device: Device, drive_frequency: np.ndarray, amplitude: float) -> np.ndarray:
    """Steady-state |e> population under continuous weak drive (two-level Bloch equations,
    Bloch 1946, https://doi.org/10.1103/PhysRev.70.460).

    P_e = (Omega^2 T1 T2 / 2) / (1 + (Delta T2)^2 + Omega^2 T1 T2), with Omega and Delta
    in angular units and T2 the Markovian T2 of the Bloch equations (``device.T2echo``). Gives a
    power-broadened Lorentzian centred on f_01, with half width sqrt(1 + Omega^2 T1 T2) / T2
    (angular), the power dependence measured by Schuster et al. 2005,
    https://doi.org/10.1103/PhysRevLett.94.123602. The static frequency offset then convolves the
    line with a Gaussian of rms sigma_f (a Voigt profile), done here with the same Gauss-Hermite
    average as the pulse sequences.
    """
    omega = 2 * np.pi * device.rabi_hz_per_volt * amplitude * 1e-9  # rad/ns
    t1, t2 = device.T1, device.T2echo
    s = omega**2 * t1 * t2
    f = np.asarray(drive_frequency, dtype=float)
    offsets, weights = quasistatic_offsets(device)
    p_e = 0.0
    for d, w in zip(offsets, weights):
        delta = 2 * np.pi * (f - device.f_01 - d) * 1e-9
        p_e = p_e + w * 0.5 * s / (1 + (delta * t2) ** 2 + s)
    p_th = device.thermal_excited_population
    return p_th + (1 - 2 * p_th) * p_e


def steady_state_excited_population(
    device: Device, tls: TLSDefect | None, drive_frequencies, amplitude: float, f01_offset: float = 0.0
) -> np.ndarray:
    """Steady-state |e> population under continuous drive, from the full master equation
    (``qt.steadystate``), for a transmon that may carry a TLS.

    H = static H (with the TLS) + 2 pi (Omega / 2) (a + a^dag), Omega = rabi_hz_per_volt * amplitude.
    Returns p_th + (1 - 2 p_th) P_e, as :func:`saturation_excited_population` does. No quasi-static
    average: the 5 kHz Gaussian on a line wider than 0.5 MHz changes P_e by under 1e-3.
    """
    transmon = Transmon(device, tls=tls)
    omega_ghz = device.rabi_hz_per_volt * amplitude * 1e-9
    drive = 2 * np.pi * 0.5 * omega_ghz * (transmon.a + transmon.a.dag())
    p_e = []
    for f in np.atleast_1d(np.asarray(drive_frequencies, dtype=float)):
        rho = qt.steadystate(transmon._static_h(float(f), f01_offset) + drive, transmon.c_ops)
        p_e.append(qt.expect(transmon.projectors[1], rho).real)
    p_th = device.thermal_excited_population
    return p_th + (1 - 2 * p_th) * np.array(p_e)


def bloch_vector(rho: qt.Qobj) -> np.ndarray:
    """Bloch vector (x, y, z) of the g/e block of rho, with |g> at the north pole.

    x = 2 Re(rho_eg), y = 2 Im(rho_eg), z = rho_gg - rho_ee. Population in |f> shortens the
    vector rather than being projected onto the sphere, so leakage shows as |v| < 1.
    """
    assert rho.shape == (N_LEVELS, N_LEVELS), "bloch_vector takes a 3-level transmon state (use Transmon.qubit_state)"
    m = rho.full()
    return np.array([2 * m[1, 0].real, 2 * m[1, 0].imag, (m[0, 0] - m[1, 1]).real])


def level_populations(rho: qt.Qobj) -> np.ndarray:
    """Populations of |g>, |e>, |f>: the real diagonal of rho."""
    return np.real(np.diag(rho.full()))[:N_LEVELS]
