"""Rabi Bloch animation: the starting-guess pulse against a pulse at the true pi amplitude.

Both spheres run the same 40 ns, sigma 8 ns Gaussian from ``state_initial.json`` on resonance
with the true device, starting from pure |g> rather than the slightly thermal ground state so
the picture is clean. The left sphere uses the 100 mV amplitude stored in ``state_initial.json``
and under-rotates; the right one uses ``TRUE_PI_AMPLITUDE`` (147.14 mV), the simulated device's
true pi amplitude rather than a value fitted by 04b_power_rabi, and lands on |e>. The full
three-level open-system simulation drives both, so the final populations printed under each
sphere are simulated values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import qutip as qt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from calibration_utils.style import FONT_FAMILY, KET_G, palette, theme_context  # noqa: E402
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE  # noqa: E402
from simulator.transmon import (  # noqa: E402
    N_LEVELS,
    Pulse,
    Transmon,
    bloch_vector,
    gaussian_envelope,
    level_populations,
)
from viz.bloch import bloch_axes, draw_state, save_gif  # noqa: E402

PULSE_LENGTH = 40.0  # ns, the x180 length in state_initial.json
PULSE_SIGMA = 8.0  # ns
SAMPLES = 50  # trajectory points during the pulse, one animation frame each
FPS = 20
LEAD_FRAMES = 8  # hold on |g> before the pulse starts
HOLD_FRAMES = 24  # hold the final state so the printed values can be read (1.2 s)
THEME = "dark"


@dataclass(frozen=True)
class RabiTrajectory:
    amplitude: float  # volts
    times: np.ndarray  # ns, shape (N,)
    vectors: np.ndarray  # Bloch vectors, shape (N, 3)
    populations: np.ndarray  # P(g), P(e), P(f), shape (N, 3)


def _trajectory(amplitude: float) -> RabiTrajectory:
    transmon = Transmon(TRUE_DEVICE)
    rho0 = qt.basis(N_LEVELS, 0).proj()
    pulse = Pulse(amplitude=amplitude, length=PULSE_LENGTH, sigma=PULSE_SIGMA)
    times, states = transmon.trajectory(
        [pulse], TRUE_DEVICE.f_01, samples_per_segment=SAMPLES, rho0=rho0
    )
    vectors = np.array([bloch_vector(rho) for rho in states])
    populations = np.array([level_populations(rho) for rho in states])
    return RabiTrajectory(amplitude, times, vectors, populations)


def rabi_frames(
    initial_amp: float = 0.1, pi_amp: float = TRUE_PI_AMPLITUDE
) -> dict[str, RabiTrajectory]:
    """Simulate both pulses from pure |g>. Keys are ``"guess"`` and ``"true_pi"``."""
    return {"guess": _trajectory(initial_amp), "true_pi": _trajectory(pi_amp)}


def make_rabi_animation(
    out: Path, initial_amp: float = 0.1, pi_amp: float = TRUE_PI_AMPLITUDE
) -> Path:
    """Render the side-by-side Rabi GIF to ``out`` and return the path.

    The right panel is headed "true π amplitude" when ``pi_amp`` is ``TRUE_PI_AMPLITUDE`` (the
    default) and "π amplitude" for any other value.
    """
    frames = rabi_frames(initial_amp, pi_amp)
    p = palette(THEME)
    right = "true π amplitude" if pi_amp == TRUE_PI_AMPLITUDE else "π amplitude"
    panels = (
        (frames["guess"], p.guess, "starting guess", 0.05),
        (frames["true_pi"], p.fit, right, 0.53),
    )
    n_pulse = len(frames["guess"].times)
    env_t = np.linspace(0.0, PULSE_LENGTH, 241)
    env_shape = gaussian_envelope(env_t, PULSE_LENGTH, PULSE_SIGMA)
    amp_max = max(initial_amp, pi_amp) * 1.18

    with theme_context(THEME):
        fig = plt.figure(figsize=(8.0, 5.0), facecolor=p.bg)
        fig.text(
            0.5,
            0.958,
            f"Power Rabi: same {PULSE_LENGTH:.0f} ns pulse, two amplitudes",
            ha="center",
            va="center",
            color=p.fg,
            fontsize=14,
            fontweight="semibold",
        )
        fig.text(
            0.5,
            0.03,
            f"Simulated 3-level transmon, resonant drive, starting from pure {KET_G}",
            ha="center",
            va="center",
            color=p.muted,
            fontsize=8.5,
            family=FONT_FAMILY,
            fontweight="normal",
        )

        live = []
        for traj, color, name, left in panels:
            w = 0.42
            fig.text(
                left + w / 2,
                0.893,
                f"{name}, {traj.amplitude * 1e3:.2f} mV",
                ha="center",
                va="center",
                color=color,
                fontsize=11.5,
                fontweight="medium",
            )
            ax = bloch_axes(fig, (left, 0.275, w, 0.545), theme=THEME)

            # Pulse envelope strip with a cursor that follows the simulation time.
            env_ax = fig.add_axes((left + 0.07, 0.155, w - 0.14, 0.075))
            env_ax.set_facecolor(p.bg)
            env_ax.set_xlim(0.0, PULSE_LENGTH)
            env_ax.set_ylim(0.0, amp_max)
            env_ax.set_axis_off()
            env_v = traj.amplitude * env_shape
            env_ax.plot(env_t, env_v, color=color, lw=1.0, alpha=0.45)
            env_ax.axhline(0.0, color=p.grid, lw=0.8)
            fill = [env_ax.fill_between(env_t[:1], env_v[:1], color=color, alpha=0.35, lw=0)]
            cursor = env_ax.axvline(0.0, color=p.fg, lw=1.0, alpha=0.8)
            env_ax.text(
                1.0,
                -0.12,
                f"{PULSE_LENGTH:.0f} ns",
                transform=env_ax.transAxes,
                ha="right",
                va="top",
                color=p.muted,
                fontsize=8,
            )
            env_ax.text(
                0.0,
                -0.12,
                "t = 0",
                transform=env_ax.transAxes,
                ha="left",
                va="top",
                color=p.muted,
                fontsize=8,
            )

            readout = fig.text(
                left + w / 2,
                0.085,
                "",
                ha="center",
                va="center",
                color=p.fg,
                fontsize=12,
            )
            live.append(
                {
                    "traj": traj,
                    "color": color,
                    "ax": ax,
                    "env_ax": env_ax,
                    "env_v": env_v,
                    "fill": fill,
                    "cursor": cursor,
                    "readout": readout,
                    "state": [],
                }
            )

        def draw(frame: int):
            k = min(max(frame - LEAD_FRAMES, 0), n_pulse - 1)
            changed = []
            for panel in live:
                traj = panel["traj"]
                for artist in panel["state"]:
                    artist.remove()
                trail = traj.vectors[: k + 1] if k > 0 else None
                panel["state"] = draw_state(panel["ax"], traj.vectors[k], panel["color"], trail)
                t = traj.times[k]
                panel["fill"][0].remove()
                upto = env_t <= t + 1e-9
                panel["fill"][0] = panel["env_ax"].fill_between(
                    env_t[upto], panel["env_v"][upto], color=panel["color"], alpha=0.35, lw=0
                )
                panel["cursor"].set_xdata([t, t])
                panel["readout"].set_text(f"P(e) = {traj.populations[k, 1]:.3f}")
                changed.extend(panel["state"])
            return changed

        total = LEAD_FRAMES + n_pulse + HOLD_FRAMES
        anim = FuncAnimation(fig, draw, frames=total, blit=False)
        try:
            return save_gif(anim, Path(out), fps=FPS)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/media/bloch-rabi.gif")
    print(make_rabi_animation(target))
