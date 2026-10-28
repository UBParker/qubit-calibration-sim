"""Bloch sphere drawing helpers for the README GIFs.

Convention matches ``simulator.transmon.bloch_vector``: x = 2 Re(rho_eg), y = 2 Im(rho_eg),
z = rho_gg - rho_ee, so |g> sits at the north pole and |e> at the south pole. The frame is
right-handed, so a phase-0 (x) pulse carries |g> down through -y. The fixed view looks at the
sphere mostly along +x with a slight tilt from -y, which shows the y-z plane nearly face on and
makes that rotation read as a clean arc toward the viewer's left.

Call these helpers inside ``calibration_utils.style.theme_context(theme)`` so the fonts and
colours match the static figures. GIFs use the dark theme by default.

Depth cues come from drawing order, not occlusion (mplot3d has none): the parts of the sphere
facing away from the viewer are drawn fainter, and the state arrow always sits on top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.animation import AbstractMovieWriter, Animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from calibration_utils.style import KET_E, KET_F, KET_G, palette

VIEW_ELEV = 28.0  # degrees above the equator
VIEW_AZIM = -40.0  # degrees; 0 puts the camera on +x looking back at the origin
GIF_DPI = 100
MAX_GIF_BYTES = 3_000_000

# Draw order. computed_zorder is off, so these fixed levels decide what sits on top.
_Z_DISC, _Z_BACK, _Z_AXES, _Z_FRONT, _Z_LABEL, _Z_TRAIL, _Z_ARROW = 0, 1, 2, 3, 4, 5, 6

_N_CIRCLE = 181  # points per wireframe circle


def _view_dir(elev: float, azim: float) -> np.ndarray:
    """Unit vector from the origin toward the viewer for matplotlib's elev and azim."""
    e, a = math.radians(elev), math.radians(azim)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _circle(u: np.ndarray, v: np.ndarray, n: int = _N_CIRCLE) -> np.ndarray:
    """Points on the circle cos(t) u + sin(t) v, closed (first point repeated at the end)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.outer(np.cos(t), u) + np.outer(np.sin(t), v)


def _depth_line(
    ax: Axes3D,
    pts: np.ndarray,
    color: str,
    lw: float,
    front_alpha: float,
    back_alpha: float,
    view: np.ndarray,
) -> None:
    """Draw a 3D polyline, splitting segments into a front and a fainter back layer."""
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    facing = (segs.mean(axis=1) @ view) >= 0.0
    for mask, alpha, z in ((~facing, back_alpha, _Z_BACK), (facing, front_alpha, _Z_FRONT)):
        if mask.any():
            coll = Line3DCollection(
                segs[mask], colors=[to_rgba(color, alpha)], linewidths=lw, capstyle="round"
            )
            coll.set_zorder(z)
            ax.add_collection3d(coll)


def bloch_axes(
    fig: Figure,
    rect_or_subplotspec: int | Sequence[float] | SubplotSpec | tuple[int, int, int],
    theme: str = "dark",
    *,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
) -> Axes3D:
    """Add a styled Bloch sphere to ``fig`` and return its 3D axes.

    ``rect_or_subplotspec`` is a 3-digit subplot code (``111``), a ``(rows, cols, index)``
    tuple, a ``SubplotSpec``, or a figure-fraction rectangle ``(left, bottom, width, height)``.
    ``elev`` and ``azim`` set the fixed camera; the silhouette and the front/back split of the
    wireframe are computed from the same angles, so do not call ``view_init`` afterwards.
    """
    p = palette(theme)
    if isinstance(rect_or_subplotspec, (int, SubplotSpec)):
        ax = fig.add_subplot(rect_or_subplotspec, projection="3d", proj_type="ortho")
    elif len(rect_or_subplotspec) == 3:
        ax = fig.add_subplot(*rect_or_subplotspec, projection="3d", proj_type="ortho")
    else:
        ax = fig.add_axes(rect_or_subplotspec, projection="3d", proj_type="ortho")

    ax.computed_zorder = False
    ax.set_facecolor(p.bg)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.42)

    view = _view_dir(elev, azim)
    ex, ey, ez = np.eye(3)

    # Silhouette disc: the great circle facing the viewer, filled with the raised panel colour.
    # Under the orthographic view it projects to exactly the sphere's outline.
    s1 = np.cross(view, ez)
    s1 /= np.linalg.norm(s1)
    s2 = np.cross(view, s1)
    rim = _circle(s1, s2)
    disc = Poly3DCollection([rim], facecolors=[to_rgba(p.panel, 0.9)], edgecolors="none")
    disc.set_zorder(_Z_DISC)
    ax.add_collection3d(disc)

    # Faint wireframe: latitudes every 30 degrees and meridians every 30 degrees.
    for lat in (-60, -30, 30, 60):
        r, h = math.cos(math.radians(lat)), math.sin(math.radians(lat))
        pts = _circle(r * ex, r * ey) + h * ez
        _depth_line(ax, pts, p.grid, 0.6, 0.9, 0.45, view)
    for lon in (30, 60, 120, 150):
        c, s = math.cos(math.radians(lon)), math.sin(math.radians(lon))
        _depth_line(ax, _circle(c * ex + s * ey, ez), p.grid, 0.6, 0.9, 0.45, view)

    # Equator and the two principal meridians (x-z and y-z planes), one step stronger.
    for u, v in ((ex, ey), (ex, ez), (ey, ez)):
        _depth_line(ax, _circle(u, v), p.reference, 0.8, 0.75, 0.3, view)

    # Silhouette ring: crisp outline of the sphere.
    ring = Line3DCollection(
        [rim],
        colors=[to_rgba(p.muted, 0.85)],
        linewidths=1.0,
    )
    ring.set_zorder(_Z_FRONT)
    ax.add_collection3d(ring)

    # Axes through the centre, thin and muted, ending at the sphere surface.
    for axis in (ex, ey, ez):
        seg = np.stack([-axis, axis])
        ax.plot(*seg.T, color=p.muted, lw=0.6, alpha=0.55, zorder=_Z_AXES)

    # Pole and axis-tip markers, then labels just outside the sphere.
    tips = np.array([ez, -ez, ex, ey])
    ax.scatter(*tips.T, s=7, color=p.muted, depthshade=False, linewidths=0, zorder=_Z_FRONT)
    # Under the orthographic camera a pole projects cos(elev) sphere radii from the centre,
    # while the rim stays at 1. Scaling the pole offset by 1/cos(elev) keeps the ket labels a
    # fixed screen gap outside the rim at any elevation.
    pole = (1.0 + 0.07) / max(math.cos(math.radians(elev)), 0.2)
    labels = (
        (ez * pole, KET_G, p.fg, 12.0, "normal", "bottom"),
        (-ez * pole, KET_E, p.fg, 12.0, "normal", "top"),
        (ex * 1.2, "+x", p.muted, 9.5, "medium", "center"),
        (ey * 1.2, "+y", p.muted, 9.5, "medium", "center"),
    )
    for pos, text, color, size, weight, va in labels:
        ax.text(
            *pos,
            text,
            color=color,
            fontsize=size,
            fontweight=weight,
            ha="center",
            va=va,
            zorder=_Z_LABEL,
        )
    return ax


class _Arrow3D(FancyArrowPatch):
    """A 2D arrow whose ends are projected from 3D data coordinates at draw time."""

    def __init__(self, start: np.ndarray, end: np.ndarray, **kwargs) -> None:
        super().__init__((0.0, 0.0), (0.0, 0.0), **kwargs)
        self._xyz = (np.asarray(start, float), np.asarray(end, float))

    def do_3d_projection(self, renderer=None) -> float:
        (x0, y0, z0), (x1, y1, z1) = self._xyz
        xs, ys, zs = proj3d.proj_transform(
            np.array([x0, x1]), np.array([y0, y1]), np.array([z0, z1]), self.axes.get_proj()
        )
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def draw_state(
    ax: Axes3D,
    vec: np.ndarray,
    color: str,
    trail: np.ndarray | None = None,
    alpha_trail: float = 0.6,
) -> list[Artist]:
    """Draw the state arrow from the origin to ``vec`` and an optional fading trail.

    ``trail`` is an (N, 3) array of earlier Bloch vectors, oldest first; it fades from
    transparent at the oldest point to ``alpha_trail`` at the newest. Returns the artists added,
    so an animation can remove them (``artist.remove()``) before drawing the next frame.
    """
    vec = np.asarray(vec, dtype=float)
    artists: list[Artist] = []

    if trail is not None and len(trail) >= 2:
        pts = np.asarray(trail, dtype=float)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        ramp = np.linspace(0.0, 1.0, len(segs))
        rgba = np.tile(to_rgba(color), (len(segs), 1))
        rgba[:, 3] = alpha_trail * ramp
        widths = 0.8 + 1.6 * ramp
        trail_coll = Line3DCollection(segs, colors=rgba, linewidths=widths, capstyle="butt")
        trail_coll.set_zorder(_Z_TRAIL)
        ax.add_collection3d(trail_coll)
        artists.append(trail_coll)

    if np.linalg.norm(vec) > 1e-3:
        arrow = _Arrow3D(
            np.zeros(3),
            vec,
            arrowstyle="-|>,head_length=0.55,head_width=0.28",
            mutation_scale=14,
            color=color,
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
            capstyle="round",
            joinstyle="round",
            zorder=_Z_ARROW,
        )
        ax.add_artist(arrow)
        artists.append(arrow)

    # A soft halo plus a bright core at the tip, so the state reads even when foreshortened.
    halo = ax.scatter(*vec, s=110, color=color, alpha=0.18, linewidths=0, depthshade=False)
    core = ax.scatter(*vec, s=16, color=color, linewidths=0, depthshade=False)
    halo.set_zorder(_Z_ARROW)
    core.set_zorder(_Z_ARROW + 0.1)
    artists.extend([halo, core])
    return artists


def _sci(value: float) -> str:
    """Format a positive value as mathtext ``m × 10^e`` with one decimal in the mantissa.

    The digits sit in ``\\mathdefault`` so they stay in the text font, as matplotlib's own log
    tick labels do; only the × comes from the mathtext fontset.
    """
    exp = math.floor(math.log10(value))
    mant = value / 10**exp
    if round(mant, 1) >= 10.0:
        mant, exp = mant / 10.0, exp + 1
    if exp == 0:
        return f"{value:.2f}"
    return rf"$\mathdefault{{{mant:.1f}\times10^{{{exp}}}}}$"


LEAK_FLOOR = 1e-5


def _leak_label(p_f: float) -> str:
    return _sci(float(p_f)) if p_f >= LEAK_FLOOR else r"$\mathdefault{<10^{-5}}$"


def leakage_bar(ax: Axes, p_f: float, theme: str) -> list[Artist]:
    """Set up ``ax`` as a vertical log-scale bar of the |f> population from 1e-5 to 1.

    Call once per axes: it styles the axes and adds the static track and |f> header. The value
    label sits below the bar; values under the floor show as an empty bar labelled "< 10^-5".
    Returns ``[bar, value_label]``; animate with :func:`update_leakage_bar`.
    """
    p = palette(theme)
    ax.set_facecolor(p.bg)
    ax.set_yscale("log")
    ax.set_ylim(LEAK_FLOOR, 1.0)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_yticks([1e-4, 1e-2, 1.0])
    ax.set_yticklabels(["10⁻⁴", "10⁻²", "1"])
    ax.minorticks_off()
    ax.tick_params(axis="y", colors=p.muted, labelsize=7, length=2, width=0.5)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(p.grid)
    ax.grid(False)

    # Static track behind the bar so the full scale is visible even at tiny leakage.
    ax.bar(0, 1.0 - LEAK_FLOOR, bottom=LEAK_FLOOR, width=0.5, color=p.panel, zorder=0)
    ax.text(
        0.5,
        1.02,
        KET_F,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=p.leak,
        fontsize=9,
    )
    (bar,) = ax.bar(0, 0.0, bottom=LEAK_FLOOR, width=0.5, color=p.leak, zorder=1).patches
    value = ax.text(
        0.5,
        -0.04,
        "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=p.fg,
        fontsize=7.5,
    )
    artists: list[Artist] = [bar, value]
    update_leakage_bar(artists, p_f)
    return artists


def update_leakage_bar(artists: Sequence[Artist], p_f: float) -> list[Artist]:
    """Set the bar height and value label returned by :func:`leakage_bar` to ``p_f``."""
    bar, value = artists
    bar.set_height(max(float(p_f), LEAK_FLOOR) - LEAK_FLOOR)
    value.set_text(_leak_label(p_f))
    return list(artists)


PALETTE_SAMPLES = 5  # evenly spaced frames, first and last included, that build the shared palette
PALETTE_COLOURS = 255  # leaves index 255 free for Pillow's transparent delta pixels


def quantize_frames(frames: Sequence[Image.Image]) -> list[Image.Image]:
    """Map RGB frames onto one shared palette, without dithering.

    The palette comes from median cut over a montage of ``PALETTE_SAMPLES`` evenly spaced frames
    (first and last included). One palette keeps palette indices equal wherever the picture is
    unchanged, so Pillow's frame differencing finds only the real changes; per-frame adaptive
    palettes made a Ramsey GIF 8.8 MB against 0.7 MB. Median cut with no dithering is
    deterministic, so reruns give identical bytes. 255 colours keep one index free, which
    ``optimize=True`` uses to mark unchanged pixels transparent.
    """
    rgb = [f.convert("RGB") for f in frames]
    picks = sorted({round(i) for i in np.linspace(0, len(rgb) - 1, min(PALETTE_SAMPLES, len(rgb)))})
    w, h = rgb[0].size
    montage = Image.new("RGB", (w, h * len(picks)))
    for row, i in enumerate(picks):
        montage.paste(rgb[i], (0, row * h))
    pal = montage.quantize(colors=PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in rgb]


class _GifWriter(AbstractMovieWriter):
    """Collects each frame as an RGB image and writes a shared-palette GIF on finish.

    Frames are rendered the way matplotlib's PillowWriter renders them (savefig to raw RGBA at
    the writer dpi) but kept in this class's own list, so nothing depends on PillowWriter
    internals. RGBA is flattened to RGB; the figures are opaque, so nothing is lost.
    """

    def setup(self, fig, outfile, dpi=None) -> None:
        super().setup(fig, outfile, dpi=dpi)
        self.frames: list[Image.Image] = []

    def grab_frame(self, **savefig_kwargs) -> None:
        buf = BytesIO()
        self.fig.savefig(buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        im = Image.frombuffer("RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1)
        self.frames.append(im.convert("RGB"))

    def finish(self) -> None:
        frames = quantize_frames(self.frames)
        frames[0].save(
            self.outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=True,
        )


def save_gif(anim: Animation, path: Path, fps: int = 20) -> Path:
    """Write ``anim`` as a looping GIF at a fixed fps and dpi, with one shared palette.

    This is the one GIF writer for the README animations. Frames share a palette from
    :func:`quantize_frames`, and Pillow stores each frame after the first as the pixels that
    changed (the rest transparent). Identical consecutive frames merge into one longer frame.

    The output is byte-identical across runs: Pillow writes no timestamps, and fps, dpi and
    frame content are all fixed. If the file exceeds ``MAX_GIF_BYTES`` (3 MB) it is deleted
    and ``ValueError`` is raised.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=_GifWriter(fps=fps), dpi=GIF_DPI)
    size = path.stat().st_size
    if size > MAX_GIF_BYTES:
        path.unlink()
        raise ValueError(f"{path.name} is {size / 1e6:.2f} MB, over the 3 MB GIF budget")
    return path
