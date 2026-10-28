"""Hahn echo on the Bloch sphere, drawn as an ensemble of static detunings.

This is an illustration, not a simulator run. Each spin has a fixed frequency offset drawn from a
normal distribution, and the picture is computed directly from rotations: an x90 takes |g> to -y,
each spin precesses about z at its own rate for tau, an x180 about +x maps (x, y, z) to
(x, -y, -z), and the same precession for another tau brings every spin back to +y at 2 tau. That
refocusing of static (low-frequency) detunings is the Hahn echo (Hahn 1950,
https://doi.org/10.1103/PhysRev.80.580). The simulator now has the same kind of static offset
(``Device.sigma_f_quasistatic``, 5.1 kHz rms) and refocuses it in 06b; this illustration exaggerates
the spread about tenfold (50 kHz rms) so the fan-out is visible, which the on-figure caption says.

Precession follows the simulator's sign convention: a spin whose frequency sits delta above the
frame rotates by -2 pi delta t about z (``simulator.transmon``'s H = 2 pi delta n).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from calibration_utils.style import theme_context
from viz.bloch import bloch_axes, draw_state, save_gif

CAPTION = (
    "Illustration: static detunings, 50 kHz rms. The simulated device has 5.1 kHz rms, "
    "refocused the same way."
)
ECHO_TAU_NS = 8_000.0  # free precession before and after the x180
GIF_NAME = "bloch-echo.gif"
FPS = 20
# 560 x 620 px at the 100 dpi GIF_DPI. The smallest text (strip ticks and label) is 9.5 pt, about
# 13 px, so it reads at native size.
FIG_W, FIG_H = 5.6, 6.2
TICK_PT = 9.5

# Frames drawn per stage: x90, each free precession, x180, and the final hold (1 s at FPS).
# Pulses are instantaneous in model time; they get frames so the eye can follow.
N_X90, N_FREE, N_X180, N_HOLD = 10, 30, 15, 20

_MINUS_Y = np.array([0.0, -1.0, 0.0])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _precess(vecs: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate each row of ``vecs`` about +z by the matching angle."""
    c, s = np.cos(angles), np.sin(angles)
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    return np.stack([c * x - s * y, s * x + c * y, z], axis=1)


def _detunings(n_spins: int, spread_hz: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, spread_hz, n_spins)


def echo_ensemble(
    times_ns: np.ndarray,
    n_spins: int = 24,
    spread_hz: float = 50e3,
    seed: int = 0,
    tau_ns: float = ECHO_TAU_NS,
) -> np.ndarray:
    """Bloch vectors of each spin after the x90, shape (len(times_ns), n_spins, 3).

    Times run from the end of the x90 (all spins on -y). The x180 acts at ``tau_ns``; a time equal
    to ``tau_ns`` gives the state just before it, and the echo forms on +y at 2 ``tau_ns``.
    """
    rate = -2.0 * np.pi * _detunings(n_spins, spread_hz, seed) * 1e-9  # rad per ns
    start = np.tile(_MINUS_Y, (n_spins, 1))
    flip = _rot_x(np.pi)
    out = np.empty((len(times_ns), n_spins, 3))
    for i, t in enumerate(np.asarray(times_ns, dtype=float)):
        if t <= tau_ns:
            out[i] = _precess(start, rate * t)
        else:
            out[i] = _precess(_precess(start, rate * tau_ns) @ flip.T, rate * (t - tau_ns))
    return out


@dataclass(frozen=True)
class _Frame:
    vecs: np.ndarray  # (n_spins, 3)
    t_ns: float  # model time since the x90
    stage: str


def _frames(n_spins: int, spread_hz: float, seed: int) -> list[_Frame]:
    tau = ECHO_TAU_NS
    frames: list[_Frame] = []
    g = np.tile([0.0, 0.0, 1.0], (n_spins, 1))
    # Each stage starts one step after the previous stage's last frame, so no frame repeats.
    for a in np.linspace(0.0, np.pi / 2, N_X90):
        frames.append(_Frame(g @ _rot_x(a).T, 0.0, "x90 pulse"))
    t_first = np.linspace(0.0, tau, N_FREE + 1)[1:]
    for t, v in zip(t_first, echo_ensemble(t_first, n_spins, spread_hz, seed), strict=True):
        frames.append(_Frame(v, t, "free precession"))
    at_tau = echo_ensemble(np.array([tau]), n_spins, spread_hz, seed)[0]
    for a in np.linspace(0.0, np.pi, N_X180 + 1)[1:]:
        frames.append(_Frame(at_tau @ _rot_x(a).T, tau, "x180 pulse"))
    # Every time here is past tau, so echo_ensemble applies the x180 first.
    t_second = np.linspace(tau, 2 * tau, N_FREE + 1)[1:]
    for t, v in zip(t_second, echo_ensemble(t_second, n_spins, spread_hz, seed), strict=True):
        frames.append(_Frame(v, t, "refocusing"))
    last = frames[-1]
    frames.extend(_Frame(last.vecs, 2 * tau, "echo") for _ in range(N_HOLD))
    return frames


def _spokes(vecs: np.ndarray) -> list[np.ndarray]:
    return [np.stack([np.zeros(3), v]) for v in vecs]


def _mean_length(vecs: np.ndarray) -> float:
    return float(np.linalg.norm(vecs.mean(axis=0)))


def make_echo_animation(out: Path, n_spins: int = 24, spread_hz: float = 50e3, seed: int = 0) -> Path:
    """Render the echo illustration to ``out/bloch-echo.gif`` (dark theme) and return its path."""
    theme = "dark"
    frames = _frames(n_spins, spread_hz, seed)
    tau_us = ECHO_TAU_NS / 1e3
    curve_t = np.linspace(0.0, 2 * ECHO_TAU_NS, 400)
    curve_len = np.linalg.norm(echo_ensemble(curve_t, n_spins, spread_hz, seed).mean(axis=1), axis=1)

    with theme_context(theme) as p:
        fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=p.bg)
        ax = bloch_axes(fig, (0.02, 0.27, 0.96, 0.68), theme)

        fig.text(0.05, 0.955, "HAHN ECHO", color=p.muted, fontsize=9, fontweight="medium", va="center")
        stage = fig.text(0.05, 0.92, "", color=p.fg, fontsize=12.5, fontweight="semibold", va="center")
        clock = fig.text(0.95, 0.955, "", color=p.muted, fontsize=9.5, ha="right", va="center")
        readout = fig.text(0.95, 0.92, "", color=p.fg, fontsize=11.5, ha="right", va="center")

        # Ensemble: thin, semi-transparent spokes with small tip dots, updated in place.
        spokes = Line3DCollection(_spokes(frames[0].vecs), colors=[to_rgba(p.fit, 0.45)], linewidths=0.9)
        spokes.set_zorder(5)
        ax.add_collection3d(spokes)
        tips = ax.scatter([0], [0], [0], s=7, color=to_rgba(p.fit, 0.85), linewidths=0, depthshade=False)
        tips.set_zorder(5.5)

        # Strip: mean vector length against model time.
        sx = fig.add_axes((0.14, 0.135, 0.78, 0.11))
        sx.set_facecolor(p.bg)
        sx.plot(curve_t / 1e3, curve_len, color=p.grid, lw=1.2)
        (done,) = sx.plot([], [], color=p.fg, lw=1.6)
        (cursor,) = sx.plot([], [], "o", color=p.fg, markersize=4)
        for x in (tau_us, 2 * tau_us):
            sx.axvline(x, color=p.muted, lw=0.7, ls=(0, (2, 2)))
        sx.set_xlim(0, 2 * tau_us * 1.02)
        sx.set_ylim(0, 1.05)
        sx.set_yticks([0, 1])
        sx.set_xticks([0, tau_us, 2 * tau_us])
        sx.set_xticklabels(["0", f"x180 at {tau_us:.0f} µs", f"echo at {2 * tau_us:.0f} µs"])
        sx.tick_params(labelsize=TICK_PT)
        sx.set_ylabel("mean\nlength", fontsize=TICK_PT, labelpad=4)
        sx.grid(False)

        # One sentence per line; the words are exactly CAPTION.
        fig.text(
            0.5, 0.04, CAPTION.replace(". ", ".\n", 1), color=p.muted, fontsize=9, ha="center", va="center"
        )

        mean_artists: list = []

        def update(i: int):
            fr = frames[i]
            for artist in mean_artists:
                artist.remove()
            mean_artists.clear()
            spokes.set_segments(_spokes(fr.vecs))
            tips._offsets3d = (fr.vecs[:, 0], fr.vecs[:, 1], fr.vecs[:, 2])
            mean = fr.vecs.mean(axis=0)
            mean_artists.extend(draw_state(ax, mean, p.fg))
            length = _mean_length(fr.vecs)
            stage.set_text(fr.stage)
            clock.set_text("" if fr.stage == "x90 pulse" else f"t = {fr.t_ns / 1e3:.1f} µs after x90")
            readout.set_text(f"mean length {length:.2f}")
            if fr.stage == "x90 pulse":
                done.set_data([], [])
                cursor.set_data([], [])
            else:
                shown = curve_t <= fr.t_ns + 1e-6
                done.set_data(curve_t[shown] / 1e3, curve_len[shown])
                cursor.set_data([fr.t_ns / 1e3], [length])
            return []

        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
        try:
            path = save_gif(anim, Path(out) / GIF_NAME, fps=FPS)
        finally:
            plt.close(fig)
    return path
