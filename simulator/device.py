"""The hidden "true" device.

Calibration nodes never import this module directly. They only see noisy I/Q data
returned by :class:`simulator.qpu.SimulatedQPU`, the same way a real experiment only
sees what comes back from the controller. Tests import it to check that the
calibration recovered the truth (closure tests).

Units: frequencies in Hz, times in ns, amplitudes in volts at the controller output, ADC voltages
in volts, raw ADC in units of 2^-12 V.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Device:
    # Transmon
    f_01: float = 5.1237e9  # qubit 0-1 transition frequency
    anharmonicity: float = -212.0e6  # f_12 - f_01
    T1: float = 48_000.0  # energy relaxation time
    T2ramsey: float = 31_000.0  # T2*: 1/e time of the shot-averaged Ramsey envelope (quasi-static noise plus white dephasing)
    T2echo: float = 62_000.0  # Markovian T2 = 1/(1/(2 T1) + 1/Tphi_white); the Hahn echo refocuses the quasi-static part
    thermal_excited_population: float = 0.012

    # Drive line: Rabi frequency (Hz) produced per volt of drive amplitude
    rabi_hz_per_volt: float = 180.0e6

    # Readout resonator (notch-type, dispersively coupled)
    f_res_g: float = 7.04125e9  # resonator frequency with qubit in |g>
    # Dispersive shift magnitude; |g> and |e> resonator frequencies differ by 2*chi, as in
    # Blais et al. 2021 (https://doi.org/10.1103/RevModPhys.93.025005).
    chi: float = 0.55e6  # f_res_e = f_res_g - 2*chi
    kappa: float = 1.4e6  # total linewidth (FWHM)
    kappa_ext_ratio: float = 0.8  # kappa_ext / kappa
    # Volts at the ADC per volt of readout amplitude at full transmission: 10 mV at the 0.05 V
    # readout, about 40 LSB of the 12-bit ±0.5 V ADC (a gain of 1e-2 would put it at 2 LSB).
    readout_gain: float = 0.2

    # Signal chain to the ADC (simulator.readout)
    time_of_flight: float = 237.0  # DAC to ADC delay; chosen, deliberately not on the 4 ns grid
    adc_offsets: tuple[float, float] = (1.2e-3, -0.8e-3)  # true DC offsets at analog inputs 1 and 2 (volts)
    # rms white noise per 1 ns sample and input (volts); I/Q noise over a 1 µs window is
    # 0.038 / sqrt(1000) = 1.2 mV, 0.12 of the full-transmission signal G A.
    adc_sample_noise: float = 0.038

    @property
    def f_res_e(self) -> float:
        return self.f_res_g - 2 * self.chi

    @property
    def f_res_f(self) -> float:
        return self.f_res_g - 4 * self.chi

    @property
    def Tphi_white(self) -> float:
        """Markovian pure dephasing time: 1/T2echo = 1/(2 T1) + 1/Tphi_white
        (Krantz et al. 2019, https://doi.org/10.1063/1.5089550)."""
        if self.T2echo >= 2 * self.T1:
            raise ValueError("T2echo must be below 2 T1")
        return 1.0 / (1.0 / self.T2echo - 1.0 / (2 * self.T1))

    @property
    def T_quasistatic(self) -> float:
        """Gaussian decay time Tq of the quasi-static part, from E(T2ramsey) = 1/e with
        E(t) = exp(-t/T2echo - (t/Tq)^2). math.inf when T2ramsey == T2echo (no quasi-static noise).

        Slow noise in the static limit gives a Gaussian Ramsey decay that an echo refocuses
        (Ithier et al. 2005, https://doi.org/10.1103/PhysRevB.72.134519)."""
        if self.T2ramsey > self.T2echo:
            raise ValueError("T2ramsey must not exceed T2echo")
        if self.T2ramsey == self.T2echo:
            return math.inf
        return self.T2ramsey / math.sqrt(1.0 - self.T2ramsey / self.T2echo)

    @property
    def sigma_f_quasistatic(self) -> float:
        """Shot-to-shot rms qubit frequency offset in Hz: Tq = sqrt(2) / (2 pi sigma_f), because
        the Gaussian average of cos(2 pi delta t) is exp(-(2 pi sigma_f t)^2 / 2)."""
        tq = self.T_quasistatic
        return 0.0 if math.isinf(tq) else math.sqrt(2) / (2 * math.pi * tq * 1e-9)

    def ramsey_envelope(self, t_ns):
        """Shot-averaged Ramsey coherence exp(-t/T2echo - (t/Tq)^2), t in ns."""
        t = np.asarray(t_ns, dtype=float)
        return np.exp(-t / self.T2echo - (t / self.T_quasistatic) ** 2)


TRUE_DEVICE = Device()

# Pi-pulse amplitude of TRUE_DEVICE for the 40 ns, sigma = 8 ns Gaussian in state_initial.json,
# found by maximising the simulated excited population. Matches pi / (2 pi * rabi_hz_per_volt *
# envelope area) to 0.1%. Only tests use it; calibration code has to find it with 04b_power_rabi.
TRUE_PI_AMPLITUDE = 0.14714
