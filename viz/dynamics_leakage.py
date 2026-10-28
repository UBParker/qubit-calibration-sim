"""Leakage into |f> during a Gaussian pi pulse, simulated with the three-level transmon.

Compares a 40 ns, sigma 8 ns pi pulse at the simulated device's true pi amplitude
(TRUE_PI_AMPLITUDE, 147.14 mV, not a value fitted by 04b_power_rabi) with a 10 ns, sigma 2 ns
one (the same length / 5 ratio). The short pulse's amplitude is TRUE_PI_AMPLITUDE scaled by
the ratio of envelope areas, so both have pulse area pi (Ruling D). Because the truncated Gaussian keeps
its shape when length and sigma scale together, that ratio is exactly 40 / 10 = 4. Area pi is a
full pi rotation only for a two-level system: the 10 ns pulse ends with P(e) about 0.95, and no
amplitude does better than 0.952 in this model, because leakage to |f> and the drive-induced
shift of the qubit line grow as the pulse gets shorter. The figure titles say "π area" for that
reason rather than calling both pulses pi pulses.

Both runs switch decay off by clearing the transmon's collapse operators, so the figure isolates
coherent leakage (the spectral weight of a short pulse reaching the f_12 transition, the problem
DRAG addresses: Motzoi et al. 2009, https://doi.org/10.1103/PhysRevLett.103.110501). Each run
starts in |g> rather than the thermal state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import qutip as qt
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator

from calibration_utils.style import (
    KET_E,
    KET_F,
    KET_G,
    MARGIN_RIGHT,
    MARGIN_TOP,
    fit_axes,
    palette,
    place_legend,
    save_themed,
    title_block,
    value_chip,
)
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.transmon import N_LEVELS, Pulse, Transmon, bloch_vector, gaussian_envelope, level_populations

LONG_LENGTH_NS, LONG_SIGMA_NS = 40.0, 8.0
SHORT_LENGTH_NS, SHORT_SIGMA_NS = 10.0, 2.0
SHORT_PI_AMPLITUDE = TRUE_PI_AMPLITUDE * LONG_LENGTH_NS / SHORT_LENGTH_NS
SAMPLES = 160  # trajectory samples per pulse
LOG_FLOOR = 1e-9


def _coherent_transmon() -> Transmon:
    """TRUE_DEVICE with decay and dephasing switched off (no collapse operators)."""
    tr = Transmon(TRUE_DEVICE)
    tr.c_ops = []
    return tr


def _ground() -> qt.Qobj:
    return qt.basis(N_LEVELS, 0).proj()


def pulse_populations(length: float, sigma: float, amplitude: float, phase: float = 0.0):
    """Times (ns) and g/e/f populations, shape (N, 3), through one resonant pulse from |g>."""
    tr = _coherent_transmon()
    times, states = tr.trajectory(
        [Pulse(amplitude, length, sigma, phase)],
        TRUE_DEVICE.f_01,
        samples_per_segment=SAMPLES,
        rho0=_ground(),
    )
    return times, np.array([level_populations(rho) for rho in states])


def leakage_numbers() -> dict[str, float]:
    """Leakage figures of merit for the 40 ns pulse, plus |f> after the 10 ns pulse.

    ``x90_phase_error_40`` is the angle between the x90 end state and -y in the equatorial
    plane, from the state's off-axis (x) component: atan2(|x|, -y).
    """
    _, long_pi = pulse_populations(LONG_LENGTH_NS, LONG_SIGMA_NS, TRUE_PI_AMPLITUDE)
    _, short_pi = pulse_populations(SHORT_LENGTH_NS, SHORT_SIGMA_NS, SHORT_PI_AMPLITUDE)
    tr = _coherent_transmon()
    _, states = tr.trajectory(
        [Pulse(TRUE_PI_AMPLITUDE / 2, LONG_LENGTH_NS, LONG_SIGMA_NS)], TRUE_DEVICE.f_01, rho0=_ground()
    )
    x, y, _ = bloch_vector(states[-1])
    return {
        "p_f_40": float(long_pi[-1, 2]),
        "residual_g_40": float(long_pi[-1, 0]),
        "x90_phase_error_40": float(np.arctan2(abs(x), -y)),
        "p_f_10": float(short_pi[-1, 2]),
        "residual_g_10": float(short_pi[-1, 0]),
        "p_e_10": float(short_pi[-1, 1]),
        "peak_f_10": float(short_pi[:, 2].max()),
        "peak_f_40": float(long_pi[:, 2].max()),
    }


_SUPERSCRIPT = str.maketrans(
    "-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
)


def _sci(value: float) -> str:
    """``8.9×10⁻⁷`` in plain Unicode, so the digits stay in the text font."""
    exp = int(np.floor(np.log10(value)))
    mant = value / 10**exp
    if round(mant, 1) >= 10.0:
        mant, exp = mant / 10.0, exp + 1
    return f"{mant:.1f}×10{str(exp).translate(_SUPERSCRIPT)}"


def plot_leakage(theme: str) -> Figure:
    """Two panels: g/e/f populations through the 40 ns and 10 ns pi-area pulses, |f> on a log strip."""
    p = palette(theme)
    nums = leakage_numbers()
    fig, axes, strips = fit_axes(theme, ncols=2, residuals=True, figsize=(10.8, 5.4))
    w, h = fig.get_size_inches()
    # fit_axes has no public height_ratios or top-margin option, so adjust the GridSpec it stores
    # for fit_layout. fit_layout only changes left, right and wspace, so these settings survive.
    gs = fig._style_layout["gs"]
    gs.set_height_ratios([2.1, 1])
    gs.update(top=1 - (MARGIN_TOP + 0.2) / h)
    title_block(fig, "pulse leakage", "A shorter pulse leaks far more out of the qubit", "", theme)
    fig.text(
        1 - MARGIN_RIGHT / w,
        1 - 0.56 / h,
        f"simulated, decay off, start in {KET_G}",
        color=p.muted,
        fontsize=8,
        ha="right",
        va="center",
    )

    runs = (
        (LONG_LENGTH_NS, LONG_SIGMA_NS, TRUE_PI_AMPLITUDE, "true π amplitude"),
        (SHORT_LENGTH_NS, SHORT_SIGMA_NS, SHORT_PI_AMPLITUDE, f"{LONG_LENGTH_NS / SHORT_LENGTH_NS:.0f} × true π amplitude"),
    )
    for ax, strip, (length, sigma, amp, amp_name) in zip(axes, strips, runs, strict=True):
        t, pops = pulse_populations(length, sigma, amp)
        env = gaussian_envelope(t, length, sigma)
        ax.fill_between(
            t,
            0,
            env,
            color=p.muted,
            alpha=0.14 if theme == "dark" else 0.1,
            linewidth=0,
            label="pulse envelope",
            zorder=0,
        )
        ax.plot(t, pops[:, 0], color=p.ground, label=KET_G)
        ax.plot(t, pops[:, 1], color=p.excited, label=KET_E)
        ax.plot(t, pops[:, 2], color=p.leak, label=KET_F)
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlim(0, length)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("population")
        ax.set_title(f"{length:.0f} ns, σ = {sigma:.0f} ns, π area: {amp_name}, {amp * 1e3:.2f} mV", pad=24)

        # Values under the floor are left out rather than clamped, so no flat line at the floor.
        p_f = np.where(pops[:, 2] >= LOG_FLOOR, pops[:, 2], np.nan)
        strip.semilogy(t, p_f, color=p.leak)
        strip.set_ylim(LOG_FLOOR, 1)
        strip.yaxis.set_major_locator(FixedLocator([1e-8, 1e-5, 1e-2, 1.0]))
        strip.yaxis.set_minor_locator(NullLocator())
        strip.set_ylabel(f"{KET_F}, log")
        strip.set_xlabel("time in pulse (ns)")
        strip.grid(True, which="major", axis="y")
        p_end = pops[-1, 2]
        strip.plot([t[-1]], [p_end], "o", color=p.leak, markersize=4, zorder=4, clip_on=False)
        strip.annotate(
            f"end {_sci(p_end)}",
            xy=(t[-1], p_end),
            xytext=(-7, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            color=p.leak,
            fontsize=8,
        )

    value_chip(
        axes[0],
        [
            f"{KET_F} after pulse: {_sci(nums['p_f_40'])}",
            f"residual {KET_G}: {nums['residual_g_40'] * 100:.2f}%",
            f"x90 phase error: {nums['x90_phase_error_40']:.3f} rad",
        ],
        theme,
        loc="center left",
    )
    value_chip(
        axes[1],
        [
            f"{KET_E} after pulse: {nums['p_e_10'] * 100:.1f}%",
            f"{KET_F} after pulse: {_sci(nums['p_f_10'])}",
            f"{nums['p_f_10'] / nums['p_f_40']:,.0f}× the 40 ns pulse",
            f"residual {KET_G}: {nums['residual_g_10'] * 100:.1f}%",
        ],
        theme,
        loc="center left",
    )
    place_legend(axes[0], theme)
    return fig


def save_leakage(out_dir: Path) -> list[Path]:
    """Write ``leakage-light.png`` and ``leakage-dark.png`` to ``out_dir``."""
    return save_themed(plot_leakage, "leakage", out_dir)
