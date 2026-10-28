"""Ramsey on the Bloch sphere: x90, a detuned idle, then x90, simulated with the Lindblad model.

The qubit starts in |g> and the x90 leaves it at -y. The drive sits ``detuning_hz`` above f_01,
so in the drive frame the vector precesses about z at the detuning during the idle. The drawn
vector is the shot average over the simulator's quasi-static qubit frequency offset, so its
equatorial length follows the device's envelope E(t) = exp(-t/T2echo - (t/Tq)^2)
(``Device.ramsey_envelope``), not exp(-t/T2*), while T1 lifts it back toward |g>. A second x90
maps y onto z, so the excited population read out after it is P(e) = (1 - y) / 2: the Ramsey
fringe 06a_ramsey fits.

Every drawn vector and fringe point comes from ``simulator.transmon.Transmon``, averaged over the
Gauss-Hermite offsets of ``quasistatic_offsets`` (idles use the exact Liouvillian propagator,
pulses the cached pulse superoperators); only the faint envelope is the analytic E(t) for the
simulated device's true parameters (T2* = ``TRUE_DEVICE.T2ramsey`` is its 1/e time, not the value
06a_ramsey fits), drawn for comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from calibration_utils.style import theme_context
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.transmon import (
    N_LEVELS,
    Idle,
    Pulse,
    Transmon,
    bloch_vector,
    level_populations,
    quasistatic_offsets,
)
from viz.bloch import bloch_axes, draw_state, save_gif

THEME = "dark"
FPS = 20
FRAMES_PER_TURN = 36  # animation frames per precession turn
MIN_FRAMES = 24
HOLD_S = 1.0  # the last frame is held this long before the loop restarts
TRAIL_TURNS = 0.35  # bright trail length, as a fraction of one precession turn

# The x90 shape of the QUA config: a 40 ns Gaussian with sigma 8 ns at half the true pi amplitude.
X90 = Pulse(amplitude=TRUE_PI_AMPLITUDE / 2, length=40.0, sigma=8.0)


@dataclass(frozen=True)
class RamseyFrames:
    """Simulated Ramsey sequence sampled along the idle.

    ``idle_ns`` holds the idle times, ``vectors`` the (N, 3) Bloch vectors after the first x90
    and that much idle, and ``p_excited`` the |e> population after the second x90.
    """

    idle_ns: np.ndarray
    vectors: np.ndarray
    p_excited: np.ndarray
    detuning_hz: float


def _n_frames(detuning_hz: float, max_idle_ns: float) -> int:
    turns = abs(detuning_hz) * max_idle_ns * 1e-9
    return max(MIN_FRAMES, math.ceil(turns * FRAMES_PER_TURN) + 1)


def ramsey_frames(
    detuning_hz: float = 0.3e6, max_idle_ns: float = 20_000.0, n_samples: int | None = None
) -> RamseyFrames:
    """Run x90, idle, x90 in the simulator at a drive ``detuning_hz`` above f_01.

    ``n_samples`` idle times (default: one per animation frame) span 0 to ``max_idle_ns``.
    """
    n = _n_frames(detuning_hz, max_idle_ns) if n_samples is None else n_samples
    device = TRUE_DEVICE
    f_drive = device.f_01 + detuning_hz
    tr = Transmon(device)
    ground = qt.basis(N_LEVELS, 0).proj()

    # Shot average over the quasi-static offset: each Gauss-Hermite node is one fixed-offset run.
    averaged: list[qt.Qobj] | None = None
    p_excited = np.zeros(n)
    for d, w in zip(*quasistatic_offsets(device)):
        d, w = float(d), float(w)
        after_x90 = tr.evolve_superop([X90], f_drive, rho0=ground, f01_offset=d)
        idle_ns, states = tr.trajectory(
            [Idle(max_idle_ns)], f_drive, samples_per_segment=n - 1, rho0=after_x90, f01_offset=d
        )
        weighted = [w * rho for rho in states]
        averaged = weighted if averaged is None else [a + b for a, b in zip(averaged, weighted)]
        p_excited += w * np.array(
            [level_populations(tr.evolve_superop([X90], f_drive, rho0=rho, f01_offset=d))[1] for rho in states]
        )
    vectors = np.array([bloch_vector(rho) for rho in averaged])
    return RamseyFrames(idle_ns=idle_ns, vectors=vectors, p_excited=p_excited, detuning_hz=detuning_hz)


def make_ramsey_animation(out: Path, detuning_hz: float = 0.3e6, max_idle_ns: float = 20_000) -> Path:
    """Render the Ramsey animation to ``out`` as a GIF and return the path."""
    data = ramsey_frames(detuning_hz, float(max_idle_ns))
    t_us = data.idle_ns * 1e-3
    vec = data.vectors
    n = len(t_us)
    trail_len = max(2, round(TRAIL_TURNS * FRAMES_PER_TURN))
    hold = round(HOLD_S * FPS)
    t2_us = TRUE_DEVICE.T2ramsey * 1e-3

    with theme_context(THEME) as p:
        fig = plt.figure(figsize=(8.4, 4.4))
        fig.set_facecolor(p.bg)

        fig.text(
            0.035,
            0.935,
            f"x90, idle, x90 with {abs(detuning_hz) / 1e6:.1f} MHz detuning; "
            f"vector shrinks with T2* = {t2_us:.0f} µs (true value)",
            color=p.fg,
            fontsize=12,
            fontweight="semibold",
            ha="left",
            va="center",
        )
        fig.text(
            0.035,
            0.875,
            "Simulated: three-level transmon, Lindblad master equation, averaged over a quasi-static\n"
            "frequency offset, starting in the ground state. Shown in the drive frame.",
            color=p.muted,
            fontsize=8.5,
            ha="left",
            va="center",
        )

        ax3 = bloch_axes(fig, (0.0, 0.02, 0.48, 0.80), theme=THEME)
        # Readouts sit in the lower left corner, clear of the |e> label under the sphere.
        clock = fig.text(0.025, 0.135, "", color=p.fg, fontsize=10.5, ha="left", va="center")
        length_text = fig.text(
            0.025, 0.065, "", color=p.muted, fontsize=8.5, ha="left", va="center", linespacing=1.5
        )

        # Muted path of everything traced so far; it grows each frame.
        path = Line3DCollection([], colors=[p.fit], linewidths=0.7, alpha=0.28)
        path.set_zorder(4.5)
        ax3.add_collection3d(path, autolim=False)
        segs = np.stack([vec[:-1], vec[1:]], axis=1)

        ax = fig.add_axes((0.575, 0.15, 0.39, 0.62))
        ax.set_facecolor(p.bg)
        ax.set_xlim(0.0, t_us[-1])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("idle time (µs)")
        ax.set_ylabel("P(e) after second x90")
        ax.set_yticks([0.0, 0.5, 1.0])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=p.grid, lw=0.6)
        ax.set_axisbelow(True)

        t_dense = np.linspace(0.0, t_us[-1], 400)
        amp = 0.5 * np.hypot(vec[0, 0], vec[0, 1]) * TRUE_DEVICE.ramsey_envelope(t_dense * 1e3)
        for sign in (1, -1):
            ax.plot(t_dense, 0.5 + sign * amp, color=p.muted, lw=0.9, ls=(0, (4, 3)), alpha=0.55)
        ax.axhline(0.5, color=p.reference, lw=0.6, alpha=0.5)
        ax.text(
            1.0,
            1.04,
            f"dashed: 0.5 ± 0.5 E(t), true envelope, T2* = {t2_us:.0f} µs (1/e)",
            transform=ax.transAxes,
            color=p.muted,
            fontsize=8,
            ha="right",
            va="bottom",
        )
        (trace,) = ax.plot([], [], color=p.fit, lw=1.6, solid_capstyle="round", clip_on=False)
        (head,) = ax.plot([], [], "o", color=p.fit, ms=5, mec=p.bg, mew=0.8, clip_on=False)
        length0 = np.hypot(vec[0, 0], vec[0, 1])

        dynamic: list[Artist] = []

        def draw(frame: int) -> list[Artist]:
            k = min(frame, n - 1)
            for artist in dynamic:
                artist.remove()
            dynamic.clear()
            path.set_segments(segs[:k])
            trail = vec[max(0, k - trail_len) : k + 1]
            dynamic.extend(draw_state(ax3, vec[k], p.fit, trail=trail, alpha_trail=0.9))
            trace.set_data(t_us[: k + 1], data.p_excited[: k + 1])
            head.set_data([t_us[k]], [data.p_excited[k]])
            clock.set_text(f"idle  {t_us[k]:5.2f} µs")
            length_text.set_text(
                f"equatorial length  {np.hypot(vec[k, 0], vec[k, 1]):.2f}\n"
                f"true envelope gives  {length0 * float(TRUE_DEVICE.ramsey_envelope(t_us[k] * 1e3)):.2f}"
            )
            return [*dynamic, path, trace, head, clock, length_text]

        anim = FuncAnimation(fig, draw, frames=n + hold, blit=False)
        try:
            return save_gif(anim, Path(out), fps=FPS)
        finally:
            plt.close(fig)
