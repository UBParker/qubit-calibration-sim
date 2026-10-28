"""Time-domain readout: resonator field, signal chain to the ADC, and QUA's demodulation.

The chain, per shot:

1. Field. The readout resonator's normalised intra-resonator field b obeys db/dt = -λ b + (κ/2) u(t)
   with λ = 2π [κ/2 + i (f - f_r)], the single-mode input-output equation of Gardiner & Collett 1985
   (https://doi.org/10.1103/PhysRevA.31.3761). f_r is the dispersively shifted resonator frequency of
   the qubit state (Blais et al. 2021, https://doi.org/10.1103/RevModPhys.93.025005); an |e> shot
   switches to the |g> frequency at its decay time, the mean-field limit of the quantum-trajectory
   picture of Gambetta et al. 2008 (https://doi.org/10.1103/PhysRevA.77.012112). The transmitted
   field past the notch is z = G A (u - r b), whose steady state is ``notch_s21`` (Probst et al. 2015,
   https://doi.org/10.1063/1.4907935).
2. Delay. The signal reaches the ADC ``Device.time_of_flight`` ns after the pulse leaves the DAC; the
   OPX records from the configured ``time_of_flight`` on, so a window that opens early holds noise only.
3. ADC. Inputs 1 and 2 carry Re and Im of z e^{iωθ} at the readout IF, plus the true DC offsets minus
   the configured ``analog_inputs`` offsets, plus white noise of rms ``Device.adc_sample_noise`` per
   sample and input. Raw units are -2^12 V (QM's 01a analysis), 12 bits over ±0.5 V.
4. Demodulation. ``dual_demod.full(w1, w2, target)`` sums input 1 against w1 and input 2 against w2 as
   QM documents it (https://docs.quantum-machines.co/latest/docs/Guides/demod/ [vendor]), with the
   oscillator referenced to the pulse start, and is returned in volts (QUA's value times 2^12 / L).
5. Demodulation is linear, so every output is Re Σ_n z(θ_n) K_n / L with a kernel K from the
   weights: the window sums are computed in closed form from prefix sums (exact, no ODE, no per-sample
   arrays per shot), and the noise enters with the covariance the weights imply. Raw traces are only
   materialised where a node streams them (``SimulatedQPU.adc_trace``).

Left out: resonator nonlinearity (Kerr), photon-number-dependent dispersive shift and the breakdown of
the dispersive approximation, measurement-induced state transitions, measurement-induced dephasing and
AC Stark shift during readout, Purcell filtering, amplifier bandwidth and the ADC anti-aliasing filter
(the edge at the ADC is instantaneous), and residual photons from a previous measurement (every readout
starts with an empty resonator).

Times are in ns, frequencies in Hz, voltages in volts. No import of ``qua`` or ``calibration_utils``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

TWO_PI_NS = 2 * np.pi * 1e-9  # Hz -> rad/ns
ADC_FULL_SCALE_V = 0.5  # OPX+ input range is +/-0.5 V (QM 01a description)
ADC_RAW_PER_VOLT = 2**12  # QM 01a analysis: volts = -raw / 2**12
DemodTerm = tuple[str, int]  # (integration-weight name, analog input 1 or 2)
Demod = tuple[DemodTerm, ...]  # one term is demod.full, two terms dual_demod.full
IQ_DEMOD: tuple[Demod, Demod] = ((("cos", 1), ("sin", 2)), (("minus_sin", 1), ("cos", 2)))
ROTATED_IQ_DEMOD: tuple[Demod, Demod] = (
    (("rotated_cos", 1), ("rotated_sin", 2)),
    (("rotated_minus_sin", 1), ("rotated_cos", 2)),
)


def expand_weights(segments, length_ns: int) -> np.ndarray:
    """QM weights as [(value, duration_ns), ...] (each duration a multiple of 4) or a flat list of
    values, one per 4 ns. Returns one value per ns sample; padded with zeros or truncated to length_ns.
    Raises ValueError for a duration that is not a multiple of 4."""
    values: list[float] = []
    for item in segments:
        if isinstance(item, (tuple, list)):
            value, duration = item
            if int(duration) % 4 or int(duration) != duration:
                raise ValueError(f"integration weight segment of {duration} ns is not a multiple of 4 ns")
            values.extend([float(value)] * int(duration))
        else:
            values.extend([float(item)] * 4)
    out = np.zeros(int(length_ns))
    n = min(len(values), int(length_ns))
    out[:n] = values[:n]
    return out


def _weights_length(segments) -> int:
    return sum(int(item[1]) if isinstance(item, (tuple, list)) else 4 for item in segments)


@dataclass(frozen=True)
class ReadoutSettings:
    lo_frequency: float  # Hz
    intermediate_frequency: float  # Hz
    amplitude: float  # V, constant readout waveform sample on I (Q must be zero)
    pulse_length: int  # ns
    time_of_flight: int  # ns, configured (>= 24, QM docs)
    smearing: int  # ns
    input_offsets: tuple[float, float]  # V, configured analog_inputs offsets of out1, out2
    # name -> (cosine, sine) per-sample arrays (expanded from 4 ns groups), length = weight length in ns
    weights: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @property
    def frequency(self) -> float:
        return self.lo_frequency + self.intermediate_frequency

    def with_frequency(self, frequency: float) -> ReadoutSettings:
        """Same LO, new IF."""
        return dataclasses.replace(self, intermediate_frequency=float(frequency) - self.lo_frequency)

    def replace(self, **changes) -> ReadoutSettings:
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_config(cls, config: dict, element: str = "resonator", operation: str = "readout") -> ReadoutSettings:
        el = config["elements"][element]
        pulse = config["pulses"][el["operations"][operation]]
        wf_i, wf_q = (config["waveforms"][pulse["waveforms"][k]] for k in ("I", "Q"))
        if wf_i["type"] != "constant" or wf_q["type"] != "constant" or float(wf_q["sample"]) != 0.0:
            raise NotImplementedError("only a constant readout waveform on I (zero Q) is modelled")
        weights = {}
        for name, weight_name in pulse.get("integration_weights", {}).items():
            w = config["integration_weights"][weight_name]
            n = max(_weights_length(w["cosine"]), _weights_length(w["sine"]))
            weights[name] = (expand_weights(w["cosine"], n), expand_weights(w["sine"], n))
        offsets = []
        for port in ("out1", "out2"):
            controller, index = el["outputs"][port]
            offsets.append(float(config["controllers"][controller]["analog_inputs"][index].get("offset", 0.0)))
        return cls(
            lo_frequency=float(el["mixInputs"]["lo_frequency"]),
            intermediate_frequency=float(el["intermediate_frequency"]),
            amplitude=float(wf_i["sample"]),
            pulse_length=int(pulse["length"]),
            time_of_flight=int(el["time_of_flight"]),
            smearing=int(el.get("smearing", 0)),
            input_offsets=(offsets[0], offsets[1]),
            weights=weights,
        )


# ---------------------------------------------------------------- field
def field_rate(frequency, f_res, kappa) -> complex:
    """λ in 1/ns: 2π [κ/2 + i (f - f_res)] with frequencies in Hz."""
    return TWO_PI_NS * (kappa / 2 + 1j * (np.asarray(frequency, float) - f_res))


def steady_field(frequency, f_res, kappa):
    """b_ss = 1/(1 + 2i (f - f_res)/κ)."""
    return 1.0 / (1.0 + 2j * (np.asarray(frequency, float) - f_res) / kappa)


def cavity_field(theta, frequency, f_before, f_after, t_switch, pulse_length, kappa) -> np.ndarray:
    """Normalised intra-resonator field b(theta), b_ss = 1/(1 + 2j (f - f_res)/kappa) under steady drive.

    Drive on for 0 <= theta < pulse_length; f_res = f_before until t_switch, f_after from then on;
    b = 0 for theta < 0. Exact piecewise solution of db/dt = -lam b + (2 pi kappa/2 1e-9) u.
    """
    theta = np.asarray(theta, float)
    b = np.zeros(theta.shape, complex)
    edges = {0.0, float(pulse_length)}
    if np.isfinite(t_switch) and t_switch > 0:
        edges.add(float(t_switch))
    edges = sorted(edges) + [np.inf]
    start, drive_rate = 0j, TWO_PI_NS * kappa / 2
    for a, e in zip(edges[:-1], edges[1:]):
        lam = field_rate(frequency, f_before if a < t_switch else f_after, kappa)
        b_ss = drive_rate / lam if a < pulse_length else 0j
        m = (theta >= a) & (theta < e)
        b[m] = b_ss + (start - b_ss) * np.exp(-lam * (theta[m] - a))
        if np.isfinite(e):
            start = b_ss + (start - b_ss) * np.exp(-lam * (e - a))
    return b


# ---------------------------------------------------------------- window and kernels
def window_theta(readout: ReadoutSettings, device, n_samples: int) -> np.ndarray:
    """Field time of each sample of a window that opens at the configured time of flight:
    θ_n = τ_set + n - τ (the pulse reaches the ADC at θ = 0)."""
    return readout.time_of_flight - float(device.time_of_flight) + np.arange(n_samples, dtype=float)


def demod_kernel(readout: ReadoutSettings, device, demod: Demod, frequency: float | None = None):
    """(K, P1, P2) for one demod output over the weight window; K complex, P1 and P2 real.

    P_k[n] = w_c cos φ_n + w_s sin φ_n on input k, φ_n = ω (τ_set + n) (oscillator referenced to the
    pulse start, as ``reset_if_phase`` gives), and K[n] = exp(i ω θ_n) (P1[n] - i P2[n]), so that
    Σ (v1 P1 + v2 P2) = Re Σ z K for v1 + i v2 = z exp(i ω θ).
    """
    frequency = readout.frequency if frequency is None else float(frequency)
    omega = TWO_PI_NS * (frequency - readout.lo_frequency)
    n = max(readout.weights[name][0].size for name, _ in demod)
    phi = omega * (readout.time_of_flight + np.arange(n, dtype=float))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    P = {1: np.zeros(n), 2: np.zeros(n)}
    for name, analog_input in demod:
        wc, ws = readout.weights[name]
        P[analog_input][: wc.size] += wc * cos_phi[: wc.size] + ws * sin_phi[: ws.size]
    theta = window_theta(readout, device, n)
    K = np.exp(1j * omega * theta) * (P[1] - 1j * P[2])
    return K, P[1], P[2]


def decay_window_sums(K, theta0, frequency, f_e, f_g, kappa, pulse_length, t_decay) -> np.ndarray:
    """Σ_n b_n K_n for |e> shots decaying to |g> at t_decay (field time), exact, vectorised.

    Prefix sums of K, K e^{-λ_e θ} and K e^{-λ_g θ} give every piece of the piecewise field. The
    factor exp(λ a) grows as exp(π κ 1e-9 a); with a <= t_end it stays below exp(10) for windows
    under 2 µs at κ = 1.4 MHz, and the assertion stops a window long enough to overflow.
    """
    K = np.asarray(K, complex)
    N = K.size
    theta = theta0 + np.arange(N, dtype=float)
    t_end = theta[-1] + 1.0
    assert TWO_PI_NS * kappa / 2 * (t_end + abs(theta0)) < 600, "window too long for the closed-form sums"
    td = np.minimum(np.asarray(t_decay, float), t_end)  # a decay after the window changes nothing in it
    L = float(pulse_length)
    drive_rate = TWO_PI_NS * kappa / 2
    lam_e, lam_g = field_rate(frequency, f_e, kappa), field_rate(frequency, f_g, kappa)
    bss_e, bss_g = drive_rate / lam_e, drive_rate / lam_g

    def cum(x):
        return np.concatenate([[0j], np.cumsum(x)])

    P0, Pe, Pg = cum(K), cum(K * np.exp(-lam_e * theta)), cum(K * np.exp(-lam_g * theta))

    def idx(a):
        return np.clip(np.ceil(np.asarray(a, float) - theta0), 0, N).astype(int)

    def seg(P, a, b):
        return P[idx(b)] - P[idx(a)]

    def driven(P_lam, lam, b_ss, b_start, a, b):
        return b_ss * seg(P0, a, b) + (b_start - b_ss) * np.exp(lam * a) * seg(P_lam, a, b)

    def free(P_lam, lam, b_start, a, b):
        return b_start * np.exp(lam * a) * seg(P_lam, a, b)

    zero = np.zeros_like(td)
    out = np.empty(td.shape, complex)
    early = td < L
    a = td[early]
    b_td = bss_e * (1 - np.exp(-lam_e * a))
    b_L = bss_g + (b_td - bss_g) * np.exp(-lam_g * (L - a))
    out[early] = (
        driven(Pe, lam_e, bss_e, 0j, zero[early], a) + driven(Pg, lam_g, bss_g, b_td, a, L) + free(Pg, lam_g, b_L, L, t_end)
    )
    a = td[~early]
    b_L = bss_e * (1 - np.exp(-lam_e * L))
    b_td = b_L * np.exp(-lam_e * (a - L))
    out[~early] = driven(Pe, lam_e, bss_e, 0j, zero[~early], L) + free(Pe, lam_e, b_L, L, a) + free(Pg, lam_g, b_td, a, t_end)
    return out


# ---------------------------------------------------------------- per-window model
@dataclass(frozen=True)
class DemodWindow:
    """Everything one (settings, demod, frequency) needs to turn a shot's state into demod outputs."""

    K: np.ndarray  # (n_out, N) complex kernels
    theta0: float  # field time of sample 0
    frequency: float  # readout frequency, Hz
    drive_sum: np.ndarray  # (n_out,) Σ over samples with 0 <= θ < L of K
    offset_term: np.ndarray  # (n_out,) Σ (Δo1 P1 + Δo2 P2) / L, volts
    cov: np.ndarray  # (n_out, n_out) single-shot noise covariance, volts²
    scale: float  # G A / L
    r: float  # κ_ext / κ
    device: object
    pulse_length: int

    def outputs_from_sums(self, sums: np.ndarray) -> np.ndarray:
        """Noiseless outputs from field sums Σ b K, shape (..., n_out) -> real (..., n_out)."""
        return (self.scale * (self.drive_sum - self.r * sums)).real + self.offset_term

    def static_outputs(self, f_res: float) -> np.ndarray:
        """(n_out,) noiseless outputs with the resonator at f_res for the whole shot."""
        d = self.device
        theta = self.theta0 + np.arange(self.K.shape[1], dtype=float)
        b = cavity_field(theta, self.frequency, f_res, f_res, np.inf, self.pulse_length, d.kappa)
        return self.outputs_from_sums(self.K @ b)

    def decay_outputs(self, t_decay) -> np.ndarray:
        """(n, n_out) noiseless outputs of |e> shots that decay to |g> at t_decay (field time)."""
        d = self.device
        sums = np.stack(
            [decay_window_sums(k, self.theta0, self.frequency, d.f_res_e, d.f_res_g, d.kappa, self.pulse_length, t_decay) for k in self.K],
            axis=-1,
        )
        return self.outputs_from_sums(sums)

    @property
    def t_end(self) -> float:
        return self.theta0 + self.K.shape[1]


def demod_window(readout: ReadoutSettings, device, demod: tuple[Demod, ...], frequency: float | None = None) -> DemodWindow:
    frequency = readout.frequency if frequency is None else float(frequency)
    parts = [demod_kernel(readout, device, d, frequency) for d in demod]
    n = max(k.size for k, _, _ in parts)
    K = np.zeros((len(parts), n), complex)
    P1 = np.zeros((len(parts), n))
    P2 = np.zeros((len(parts), n))
    for i, (k, p1, p2) in enumerate(parts):
        K[i, : k.size], P1[i, : p1.size], P2[i, : p2.size] = k, p1, p2
    theta0 = readout.time_of_flight - float(device.time_of_flight)
    theta = theta0 + np.arange(n, dtype=float)
    L = readout.pulse_length
    driven = (theta >= 0) & (theta < L)
    d_o1 = device.adc_offsets[0] - readout.input_offsets[0]
    d_o2 = device.adc_offsets[1] - readout.input_offsets[1]
    offset_term = (d_o1 * P1.sum(axis=1) + d_o2 * P2.sum(axis=1)) / L
    cov = device.adc_sample_noise**2 * (K @ K.conj().T).real / L**2
    return DemodWindow(
        K=K,
        theta0=theta0,
        frequency=frequency,
        drive_sum=K[:, driven].sum(axis=1),
        offset_term=offset_term,
        cov=cov,
        scale=device.readout_gain * readout.amplitude / L,
        r=device.kappa_ext_ratio,
        device=device,
        pulse_length=L,
    )
