r"""One visual system for every static figure in the repo.

Two themes, ``"light"`` and ``"dark"``, share the same semantic roles: measured data is a quiet
neutral, the fit is the one saturated accent, and the previous value, true value, populations and
pass/fail states each get their own colour. The categorical hues follow the Okabe and Ito
colour-blind-safe set (https://jfly.uni-koeln.de/color/), shifted lighter for the dark theme so
they keep their contrast. Text colours are checked against WCAG 2 contrast in tests/test_style.py.

Typography is Inter (SIL OFL 1.1, bundled in assets/fonts/) with DejaVu Sans as the fallback.
The HTML report reuses the hex values from :func:`palette`.

Figure text convention, the same in every node (use the constants below):

- microseconds as "µs" (:data:`US`, U+00B5), never "us";
- kets as :data:`KET_G`, :data:`KET_E`, :data:`KET_F` (built by :func:`ket`), never "|g>" or a
  hand-typed U+27E9;
- Greek letters as letters: "κ" (:data:`KAPPA`), "χ" (:data:`CHI`), "σ" (:data:`SIGMA`), never
  "kappa";
- a value and its unit joined by a no-break space (:data:`NBSP`, as ``format_change`` does), so a
  unit never wraps onto its own line.

Inter has µ, κ, χ and σ but no ket bracket, and the DejaVu Sans U+27E9 it used to fall back to
is so shallow that at figure size it reads as ")". A ket is therefore a mathtext fragment,
``$\mathdefault{|g}\rangle$``: the bar and the letter stay upright in the text font (Inter, at
the text's own weight) and only the angle bracket comes from the mathtext fontset, Computer
Modern (``mathtext.fontset = "cm"``), whose bracket is tall and clearly angled. Any other
mathtext in a figure wraps its digits in ``\mathdefault`` so they stay in Inter too. Since a
ket holds a ``$`` pair, a string holding a ket must not hold an unpaired literal ``$``.
(tests/test_harmonise.py renders all of them with warnings as errors.)
Test strings that assert ASCII text keep ASCII; figure text uses these forms.

Usage. Every helper below reads fonts, sizes and colours from rcParams, so call them inside
``theme_context(theme)``. ``save_themed`` already does that, so the usual pattern is a
``make_fig(theme)`` function passed to ``save_themed``. A node figure is built as::

    fig, axes, res = fit_axes(theme)
    axes[0].plot(x, y, "o", label="data")
    axes[0].plot(xs, model, label="fit")
    mark_values(axes[0], Marks(previous=..., fit=..., true=...), theme)
    value_chip(axes[0], ["f = ... MHz"], theme)
    place_legend(axes[0], theme)  # the one way node figures place legends
    plot_residuals(res[0], x, (y - model_at_x) / sigma, theme)
    offset_formatter(axes[0], "x", 1e6, "MHz", name="f")  # frequency axes: offsets, not 10 digits
    title_block(fig, NODE_NAME, "what it finds", "headline value", theme)

``save_themed`` runs ``fit_layout``, which widens the left margin and column gaps to fit the
measured tick and axis labels, so long labels never clip.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib

# Figures are only ever written to files. Pin Agg so the saved bytes do not depend on the
# platform's default backend (the macOS backend shifts tight bounding boxes by a pixel).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator, Formatter, Locator, MaxNLocator
from matplotlib.transforms import Bbox, blended_transform_factory

THEMES: tuple[str, str] = ("light", "dark")

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")
FONT_FAMILY = ["Inter", "DejaVu Sans"]
DPI = 160

# Figure geometry in inches. A fixed layout (instead of tight or constrained layout) keeps every
# node figure aligned the same way. The top band holds the title block (node name at 0.26 in,
# finding at 0.56 in from the top) and, below it, the legend row placed by place_legend.
PANEL_W, PANEL_H, RESID_H, NO_RESID_H = 4.6, 2.9, 0.95, 3.2
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM, MARGIN_TOP = 0.78, 0.28, 0.58, 1.12
GAP_W, GAP_H = 0.72, 0.12
LABEL_PAD = 0.1  # clearance in inches kept left of the outermost y label by fit_layout
TITLE_X = 0.22  # left inset of the title block in inches

# Figure text constants (see the convention in the module docstring).
US = "µs"
NBSP = " "


def ket(label: str) -> str:
    """Figure text for the ket ``|label>`` with an angled bracket (see the module docstring)."""
    return rf"$\mathdefault{{|{label}}}\rangle$"


KET_G, KET_E, KET_F = ket("g"), ket("e"), ket("f")
_KET_MARKUP = re.compile(r"\$\\mathdefault\{\|(\w+)\}\\rangle\$")


def plain(text: str) -> str:
    """``text`` with each :func:`ket` written as "|g⟩", for counting characters, not for figures."""
    return _KET_MARKUP.sub(r"|\1⟩", text)


def _is_math(text: str) -> bool:
    """True if matplotlib draws ``text`` as mathtext: an even, non-zero count of unescaped ``$``."""
    n = text.count("$") - text.count(r"\$")
    return n > 0 and n % 2 == 0
KAPPA, CHI, SIGMA = "κ", "χ", "σ"


@dataclass(frozen=True)
class Palette:
    """Hex colours for one theme. Role names are semantic, not hue names."""

    bg: str  # figure and axes background
    panel: str  # raised surfaces: value chip, residual bands
    fg: str  # primary text
    muted: str  # secondary text, tick labels, spines
    grid: str  # grid lines and hairline borders
    data: str  # measured points
    fit: str  # fitted model and the fitted value
    guess: str  # previous (pre-calibration) value
    truth: str  # stored device value ("true value")
    reference: str  # simulator reference, never called truth
    ground: str  # |g> population
    excited: str  # |e> population
    leak: str  # |f> and higher, leakage
    ok: str  # passed gate
    fail: str  # failed gate
    data2: str  # measured points of a second series (for example 06a's minus detuning)
    fit2: str  # fitted model of that second series


_PALETTES = {
    "light": Palette(
        bg="#FFFFFF",
        panel="#F4F6F8",
        fg="#1A1F26",
        muted="#5D6673",
        grid="#E3E7EC",
        data="#5B6B7F",
        fit="#0B6FB8",
        guess="#C77C02",
        truth="#00896A",
        reference="#8A93A0",
        ground="#3E8ED0",
        excited="#D2551C",
        leak="#B04E92",
        ok="#1D8A4E",
        fail="#C7362B",
        data2="#8B7BB5",
        fit2="#6247AA",
    ),
    "dark": Palette(
        bg="#0D1117",
        panel="#161B22",
        fg="#E6EDF3",
        muted="#9AA5B1",
        grid="#262D36",
        data="#A3B0BF",
        fit="#4FA8F0",
        guess="#F0B03C",
        truth="#33C79B",
        reference="#7D8793",
        ground="#79BDF2",
        excited="#FF8A4C",
        leak="#E386CC",
        ok="#3FB966",
        fail="#F2594B",
        data2="#8F84B8",
        fit2="#A48BF0",
    ),
}


def palette(theme: str) -> Palette:
    """Return the palette for ``"light"`` or ``"dark"``."""
    try:
        return _PALETTES[theme]
    except KeyError:
        raise ValueError(f"unknown theme {theme!r}, expected one of {THEMES}") from None


_fonts_registered = False


def _quiet_fallback_weight(record: logging.LogRecord) -> bool:
    """Drop matplotlib's notice that the DejaVu Sans fallback has no medium or semibold weight.

    The fallback only supplies the few glyphs Inter lacks; the nearest weight
    is fine for them, and the notice would otherwise print in every process that draws text.
    """
    msg = record.msg if isinstance(record.msg, str) else ""
    return not (msg.startswith("findfont: Failed to find font weight") and "DejaVu Sans" in record.args)


logging.getLogger("matplotlib.font_manager").addFilter(_quiet_fallback_weight)


def register_fonts() -> bool:
    """Add the bundled Inter files to matplotlib's font manager once. Returns True if present."""
    global _fonts_registered
    if not _fonts_registered:
        for name in FONT_FILES:
            path = FONT_DIR / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                _fonts_registered = True
    return _fonts_registered


def _rc(theme: str) -> dict:
    p = palette(theme)
    return {
        # The family list itself (not the generic "sans-serif") turns on per-glyph fallback, so
        # glyphs Inter lacks come from DejaVu Sans without a warning.
        "font.family": FONT_FAMILY,
        "font.sans-serif": FONT_FAMILY,
        "font.size": 9.0,
        # Computer Modern supplies the ket's angle bracket (see ket()); text inside mathtext
        # uses \mathdefault, so it stays in Inter.
        "mathtext.fontset": "cm",
        "text.color": p.fg,
        "figure.facecolor": p.bg,
        "figure.edgecolor": p.bg,
        "figure.dpi": 100,
        "savefig.facecolor": p.bg,
        "savefig.edgecolor": p.bg,
        "savefig.dpi": DPI,
        "axes.facecolor": p.bg,
        "axes.edgecolor": p.muted,
        "axes.linewidth": 0.6,
        "axes.labelcolor": p.fg,
        "axes.labelsize": 9.0,
        "axes.labelweight": "normal",
        "axes.labelpad": 5.0,
        "axes.titlesize": 9.5,
        "axes.titleweight": "medium",
        "axes.titlecolor": p.fg,
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=[p.data, p.fit, p.guess, p.truth, p.excited, p.leak]),
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-4, 5),
        "grid.color": p.grid,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "xtick.labelcolor": p.muted,
        "ytick.labelcolor": p.muted,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "patch.edgecolor": p.grid,
        "legend.frameon": False,
        "legend.fontsize": 8.0,
        "legend.labelcolor": p.fg,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.6,
        "image.cmap": "viridis",
    }


@contextmanager
def theme_context(theme: str) -> Iterator[Palette]:
    """Apply the theme's rcParams inside the block and yield its palette.

    All other helpers in this module assume they run inside this context.
    """
    register_fonts()
    with plt.rc_context(_rc(theme)):
        yield palette(theme)


@dataclass
class Marks:
    """Values to mark on an x axis. A value is drawn only if it is not None and finite."""

    previous: float | None = None
    fit: float | None = None
    true: float | None = None
    true_label: str = "true value"


def fit_axes(
    theme: str,
    ncols: int = 1,
    residuals: bool = True,
    figsize: tuple[float, float] | None = None,
    left: float | None = None,
) -> tuple[Figure, list[Axes], list[Axes | None]]:
    """Create a row of main axes, optionally with a residual row below (height 3:1, shared x).

    ``left`` fixes the left margin in inches; by default ``fit_layout`` (run by ``save_themed``)
    widens it to fit the measured y labels. Gaps between panels are GAP_W and GAP_H inches for
    any ``figsize``.
    """
    fig, axes, res_axes = _grid_axes(theme, ncols, residuals, figsize, left)
    return fig, axes, res_axes


def _grid_axes(
    theme: str,
    ncols: int,
    residuals: bool,
    figsize: tuple[float, float] | None,
    left: float | None,
    top: float = MARGIN_TOP,
    right: float = MARGIN_RIGHT,
    right_clear: float = MARGIN_RIGHT,
    bottom: float = MARGIN_BOTTOM,
    width_ratios: Sequence[float] | None = None,
    residual_rows: int = 1,
    strips: list[list[Axes]] | None = None,
) -> tuple[Figure, list[Axes], list[Axes | None]]:
    """``fit_axes`` with the margins (inches), column widths and residual rows as parameters.

    ``right`` is everything right of the last column; ``right_clear`` is the empty part of it
    next to the column, which ``fit_layout`` widens when a right-hand axis needs more room.
    ``width_ratios`` sets relative column widths (their mean width is the panel width); with
    ``residual_rows`` above 1 each column gets that many residual strips sharing x, collected in
    ``strips`` (one list per column, top to bottom) when it is given.
    """
    p = palette(theme)
    if residual_rows < 1:
        raise ValueError("residual_rows must be at least 1")
    n_resid = residual_rows if residuals else 0
    nrows = 1 + n_resid
    if width_ratios is not None:
        width_ratios = [float(r) for r in width_ratios]
        if len(width_ratios) != ncols or min(width_ratios) <= 0:
            raise ValueError(f"width_ratios needs {ncols} positive values, got {width_ratios}")
    left_in = MARGIN_LEFT if left is None else left
    if figsize is None:
        panel_h = PANEL_H + n_resid * RESID_H + (n_resid - 1) * GAP_H if residuals else NO_RESID_H
        figsize = (
            left_in + right + ncols * PANEL_W + (ncols - 1) * GAP_W,
            bottom + top + panel_h,
        )
    w, h = figsize
    # GridSpec spacing is a fraction of the mean axes size, so derive it from the actual sizes.
    panel_w = (w - left_in - right - (ncols - 1) * GAP_W) / ncols
    row_h = (h - top - bottom - (nrows - 1) * GAP_H) / nrows
    if panel_w <= 0 or row_h <= 0:
        raise ValueError(f"figsize {figsize} leaves no room for {ncols} panel(s)")
    fig = plt.figure(figsize=figsize, facecolor=p.bg)
    gs = fig.add_gridspec(
        nrows,
        ncols,
        height_ratios=[3] + [1] * n_resid if residuals else None,
        width_ratios=width_ratios,
        left=left_in / w,
        right=1 - right / w,
        bottom=bottom / h,
        top=1 - top / h,
        wspace=GAP_W / panel_w,
        hspace=GAP_H / row_h,
    )
    fig._style_layout = {  # read by fit_layout and the node_layout axes locators
        "gs": gs,
        "ncols": ncols,
        "left": left_in,
        "left_fixed": left is not None,
        "panel_w": panel_w,  # mean column width
        "gap": GAP_W,
        "right_base": right,
        "right": right,
        "right_clear": right_clear,
        "top": top,
        "bottom": bottom,
    }
    axes: list[Axes] = []
    res_axes: list[Axes | None] = []
    for col in range(ncols):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(p.bg)
        axes.append(ax)
        col_strips: list[Axes] = []
        for row in range(1, nrows):
            rax = fig.add_subplot(gs[row, col], sharex=ax)
            rax.set_facecolor(p.bg)
            col_strips.append(rax)
        if col_strips:
            ax.tick_params(labelbottom=False)
            for rax in col_strips[:-1]:
                rax.tick_params(labelbottom=False)
        res_axes.append(col_strips[0] if col_strips else None)
        if strips is not None:
            strips.append(col_strips)
    return fig, axes, res_axes


def fit_layout(fig: Figure) -> None:
    """Widen the margins and column gaps of a ``fit_axes`` figure to fit its y labels.

    Left y labels set the left margin and the gaps; a right-hand secondary axis (for example
    ``population_axis``) widens the gap after its column or the right margin. Axes placed in a
    subgridspec of a layout cell are measured for that cell's column. Panel sizes stay fixed; the
    figure grows wider instead. ``save_themed`` calls this right before saving, so it
    sees the final ticks and labels. Figures not made by ``fit_axes`` or ``node_layout`` are left
    alone.
    """
    layout = getattr(fig, "_style_layout", None)
    if layout is None:
        return
    renderer = fig.canvas.get_renderer()
    ncols = layout["ncols"]
    over_left = [0.0] * ncols
    over_right = [0.0] * ncols
    for ax in fig.axes:
        spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
        if spec is not None:
            # Axes in a subgridspec of a layout cell (for example two residual strips under one
            # main axes) count for the column of that cell.
            spec = spec.get_topmost_subplotspec()
        if spec is None or spec.get_gridspec() is not layout["gs"]:
            continue
        col = spec.colspan.start
        box = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer)
        if tight is not None:
            over_left[col] = max(over_left[col], (box.x0 - tight.x0) / fig.dpi)
        for child in ax.child_axes:  # secondary axes on the right edge
            child_tight = child.get_tightbbox(renderer)
            if child_tight is not None:
                last = spec.colspan.stop - 1
                over_right[last] = max(over_right[last], (child_tight.x1 - box.x1) / fig.dpi)
    left = layout["left"] if layout["left_fixed"] else max(MARGIN_LEFT, over_left[0] + LABEL_PAD)
    gap = max([GAP_W] + [over_left[c] + over_right[c - 1] + LABEL_PAD + 0.1 for c in range(1, ncols)])
    right_base = layout.get("right_base", MARGIN_RIGHT)
    right_clear = layout.get("right_clear", MARGIN_RIGHT)
    right = right_base + max(0.0, over_right[-1] + LABEL_PAD - right_clear)
    if left == layout["left"] and gap == layout["gap"] and right == layout.get("right", MARGIN_RIGHT):
        return
    w, h = fig.get_size_inches()
    panel_w = layout["panel_w"]
    new_w = left + right + ncols * panel_w + (ncols - 1) * gap
    fig.set_size_inches(new_w, h, forward=False)
    layout["gs"].update(left=left / new_w, right=1 - right / new_w, wspace=gap / panel_w)
    layout.update(left=left, gap=gap, right=right)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def mark_values(
    ax: Axes,
    marks: Marks,
    theme: str,
    scale: float = 1.0,
    *,
    previous_label: str = "previous value",
    fit_label: str = "fitted value",
    true_label: str | None = None,
) -> None:
    """Mark the previous, fitted and true values (each times ``scale``) on the x axis.

    Each value gets a full-height vertical line plus a small downward triangle on the top edge
    of the axes, which is also its legend entry. The fitted-value line is thin and translucent so
    it reads as a marker, not as the fit curve. The legend labels default to "previous value",
    "fitted value" and ``marks.true_label``; the keywords replace them (for example
    ``fit_label="rejected fit"``).
    """
    p = palette(theme)
    specs = (
        # value, label, colour, dash, line width, line alpha
        (marks.previous, previous_label, p.guess, (0, (4, 3)), 1.2, 0.9),
        (marks.fit, fit_label, p.fit, "-", 1.0, 0.6),
        (marks.true, marks.true_label if true_label is None else true_label, p.truth, (0, (1, 2.2)), 1.5, 0.95),
    )
    top = ax.get_xaxis_transform()  # x in data, y in axes fraction
    for value, label, colour, dash, lw, alpha in specs:
        if not _finite(value):
            continue
        x = value * scale
        ax.axvline(x, color=colour, linestyle=dash, linewidth=lw, alpha=alpha, zorder=1.5)
        ax.plot(
            [x],
            [1.0],
            transform=top,
            marker="v",
            markersize=5.5,
            markeredgewidth=0,
            color=colour,
            linestyle=dash,
            linewidth=lw,
            label=label,
            clip_on=False,
            zorder=3,
        )


def place_legend(
    ax: Axes, theme: str, ncols: int | None = None, title: str | None = None
) -> Legend | None:
    """Put the legend in one row above ``ax``, under the title block, clear of the data.

    This is the one way node figures place legends. ``ncols`` defaults to one column per entry
    (a single row). ``title`` is a panel title drawn at the left of the same row (use it instead
    of ``ax.set_title``, which would collide with the legend); the legend then starts after it.
    When the legend wraps to several rows the title sits on the first (top) row. On the main
    axes of a ``node_layout(top_extra=...)`` figure the row is lifted by ``top_extra`` inches,
    above a top twin axis. Returns None when ``ax`` has no labelled artists (the title is still
    drawn).
    """
    p = palette(theme)
    fig_w, fig_h = ax.figure.get_size_inches()
    ax_w = ax.get_position().width * fig_w
    ax_h = ax.get_position().height * fig_h
    y_anchor = 1.0 + _top_extra(ax) / ax_h
    title_style = dict(color=p.fg, fontsize=8.0, fontweight="semibold", ha="left", annotation_clip=False)
    x_anchor = 0.0
    if title:
        # The legend box has its own border padding (about 0.06 in) before the first handle.
        x_anchor = (_measure(ax, title, 8.0, "semibold") + 0.18 - 0.06) / ax_w
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        if title:
            ax.annotate(
                title, xy=(0.0, y_anchor), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", va="bottom", **title_style,
            )
        return None
    legend = ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(x_anchor, y_anchor),
        ncols=ncols or len(handles),
        frameon=False,
        borderaxespad=0.45,
        borderpad=0.3,
        handlelength=1.7,
        handletextpad=0.5,
        columnspacing=1.3,
        fontsize=8.0,
        labelcolor=p.fg,
    )
    if title:
        # Centred on the first legend entry (legends fill column by column, so entry 0 is on the
        # top row). Drawn after the legend, which places its entries at draw time.
        ax.annotate(
            title, xy=(0.0, 0.5), xycoords=("axes fraction", legend.get_texts()[0]),
            va="center", zorder=legend.get_zorder() + 0.1, **title_style,
        )
    return legend


def _top_extra(ax: Axes) -> float:
    """``top_extra`` inches of a ``node_layout`` figure if ``ax`` is one of its main axes, else 0."""
    layout = getattr(ax.figure, "_style_layout", None)
    spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
    if layout is None or spec is None:
        return 0.0
    spec = spec.get_topmost_subplotspec()
    if spec.get_gridspec() is not layout["gs"] or spec.rowspan.start != 0:
        return 0.0
    return float(layout.get("top_extra", 0.0))


class _OffsetLocator(Locator):
    """Ticks at round values of (value - offset) / unit_scale rather than of the raw value."""

    def __init__(self, offset: float, unit_scale: float):
        self.offset = offset
        self.unit_scale = unit_scale
        self._inner = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])

    def tick_values(self, vmin, vmax):
        lo, hi = ((v - self.offset) / self.unit_scale for v in (vmin, vmax))
        return np.asarray(self._inner.tick_values(lo, hi)) * self.unit_scale + self.offset

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)


class _OffsetFormatter(Formatter):
    """Tick labels as (value - offset) / unit_scale with just enough decimals for the step."""

    def __init__(self, offset: float, unit_scale: float):
        self.offset = offset
        self.unit_scale = unit_scale
        self._decimals = 0

    def _scaled(self, value: float) -> float:
        return (value - self.offset) / self.unit_scale

    def format_ticks(self, values):
        scaled = [self._scaled(v) for v in values]
        steps = [abs(b - a) for a, b in zip(scaled, scaled[1:]) if b != a]
        step = min(steps) if steps else 1.0
        self._decimals = next(
            (d for d in range(7) if all(abs(round(v, d) - v) <= step * 1e-3 for v in scaled)), 6
        )
        return [self(v) for v in values]

    def __call__(self, value, pos=None):
        text = f"{self._scaled(value):.{self._decimals}f}"
        if float(text) == 0:
            text = text.lstrip("-")
        return text.replace("-", "−")


def offset_formatter(
    ax: Axes,
    axis: str = "x",
    unit_scale: float = 1e6,
    unit: str = "MHz",
    name: str = "f",
    offset: float | None = None,
    offset_scale: float = 1e9,
    offset_unit: str = "GHz",
    digits: int = 4,
) -> float:
    """Show an axis of large raw values (Hz) as offsets, e.g. ticks in kHz from 5.1234 GHz.

    Ticks fall on round values in ``unit``. Call after plotting. The offset defaults to the
    centre of the current limits rounded to ``digits`` decimals of ``offset_unit``. The axis
    label becomes
    "<name> − <offset> <offset_unit> (<unit>)"; for a shared x axis it goes on the lowest axes.
    Returns the offset in raw units.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    if offset is None:
        quantum = offset_scale * 10.0**-digits
        offset = round((lo + hi) / 2 / quantum) * quantum
    label = f"{name} − {offset / offset_scale:.{digits}f} {offset_unit} ({unit})"
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(_OffsetLocator(offset, unit_scale))
    target.set_major_formatter(_OffsetFormatter(offset, unit_scale))
    if axis == "y":
        ax.set_ylabel(label)
    else:
        siblings = ax.get_shared_x_axes().get_siblings(ax)
        lowest = min(siblings, key=lambda a: a.get_position().y0)
        for a in siblings:
            a.set_xlabel(label if a is lowest else "")
    return offset


_CHIP_LOCS = ("upper right", "upper left", "lower right", "lower left", "center right", "center left")


def value_chip(ax: Axes, lines: Sequence[str], theme: str, loc: str = "upper right") -> AnchoredText:
    """Put a small rounded box with one result per line in a corner of ``ax``."""
    if loc not in _CHIP_LOCS:
        raise ValueError(f"unsupported loc {loc!r}")
    p = palette(theme)
    chip = AnchoredText(
        "\n".join(lines),
        loc=loc,
        frameon=True,
        pad=0.55,
        borderpad=0.7,
        prop={"color": p.fg, "size": 8.0, "linespacing": 1.45, "family": plt.rcParams["font.family"]},
    )
    chip.patch.set_boxstyle("round,pad=0.0,rounding_size=0.35")
    chip.patch.set_facecolor(p.panel)
    chip.patch.set_edgecolor(p.grid)
    chip.patch.set_linewidth(0.8)
    chip.set_zorder(5)
    ax.add_artist(chip)
    return chip


def plot_residuals(ax: Axes, x, residuals_over_sigma, theme: str) -> None:
    """Plot normalised residuals over shaded 1 and 2 sigma bands."""
    p = palette(theme)
    x = np.asarray(x, dtype=float)
    r = np.asarray(residuals_over_sigma, dtype=float)
    ax.axhspan(-2, 2, color=p.panel, linewidth=0, zorder=0)
    ax.axhspan(-1, 1, color=p.grid, linewidth=0, zorder=0.1, alpha=0.75)
    ax.axhline(0, color=p.muted, linewidth=0.7, zorder=0.2)
    ax.plot(x, r, "o", color=p.data, markersize=2.8, markeredgewidth=0, zorder=2)
    finite = r[np.isfinite(r)]
    top = max(3.0, math.ceil(float(np.max(np.abs(finite)))) + 0.5) if finite.size else 3.0
    ax.set_ylim(-top, top)
    ax.set_yticks([-2, 0, 2])
    ax.grid(False)
    ax.set_ylabel("resid. / σ")


def title_block(
    fig: Figure, node_name: str, finds: str, headline: str, theme: str, status: str = "ok"
) -> None:
    """Write a two-line header: short node label (small, muted), then finding and result.

    The headline is in the fit colour, or in ``palette.fail`` when ``status`` is ``"fail"`` (for
    example "fit failed"). Positions are in inches from the top left, so ``fit_layout`` widening
    the figure keeps them.
    """
    if status not in ("ok", "fail"):
        raise ValueError(f"status must be 'ok' or 'fail', got {status!r}")
    p = palette(theme)
    h = fig.get_size_inches()[1]
    inch_x = blended_transform_factory(fig.dpi_scale_trans, fig.transFigure)
    fig.text(
        TITLE_X,
        1 - 0.26 / h,
        NODE_SHORT.get(node_name, node_name),
        transform=inch_x,
        color=p.muted,
        fontsize=7.5,
        fontweight="medium",
        va="center",
        ha="left",
    )
    title = fig.text(
        TITLE_X,
        1 - 0.56 / h,
        finds,
        transform=inch_x,
        color=p.fg,
        fontsize=12.5,
        fontweight="semibold",
        va="center",
        ha="left",
    )
    if headline:
        # Anchored to the right edge of the title text, so it follows the title's real width.
        fig.add_artist(
            Annotation(
                headline,
                xy=(1.0, 0.5),
                xycoords=title,
                xytext=(9, 0),
                textcoords="offset points",
                color=p.fail if status == "fail" else p.fit,
                fontsize=12.5,
                fontweight="medium",
                va="center",
                ha="left",
            )
        )


def save_themed(make_fig: Callable[[str], Figure], name: str, out_dir: Path) -> list[Path]:
    """Render ``make_fig(theme)`` for each theme and write ``<out_dir>/<name>-<theme>.png``.

    Each render runs inside ``theme_context`` and passes through ``fit_layout`` before saving.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for theme in THEMES:
        with theme_context(theme) as p:
            fig = make_fig(theme)
            try:
                fit_layout(fig)
                path = out_dir / f"{name}-{theme}.png"
                fig.savefig(path, dpi=DPI, facecolor=p.bg, metadata={"Software": None})
            finally:
                plt.close(fig)
        paths.append(path)
    return paths


# Layout C: the node figure page. From the top: title block, progress strip, pulse sequence, the
# legend row, then the main row (with residuals) and a column of cards on the right. Positions
# are in inches from the figure edges and follow fit_layout when it widens the figure.

PROGRESS_TOP, PROGRESS_H = 0.84, 0.44  # pill row plus a status caption under it
SEQ_TOP, SEQ_H = 1.46, 0.86
NODE_MAIN_TOP = SEQ_TOP + SEQ_H + 0.5  # leaves the legend row above the main axes
CARDS_W, CARDS_GAP = 2.55, 0.5
CARDS_CLEAR = 0.25  # part of CARDS_GAP a right-hand axis label may not use
PILL_H, PILL_PAD, PILL_GAP, PILL_DOT_R = 0.27, 0.11, 0.1, 0.045  # progress pills, inches
STRIP_INSET = 0.04  # space left of the first pill and right of the last, inches
CARD_PAD, CARD_GAP, CARD_ACCENT = 0.13, 0.13, 0.045  # cards, inches
MIN_TEXT = 6.5  # smallest font size the strips and cards shrink to

# Short display labels for the progress strip and title kicker; node names stay the keys.
NODE_SHORT = {
    "01a_time_of_flight": "01a time of flight",
    "08a_readout_frequency_optimization": "08a readout freq.",
    "02a_resonator_spectroscopy": "02a resonator",
    "03a_qubit_spectroscopy": "03a qubit",
    "04b_power_rabi": "04b power Rabi",
    "07_iq_blobs": "07 IQ blobs",
    "06a_ramsey": "06a Ramsey",
    "05_T1": "05 T1",
    "06b_echo": "06b echo",
}

_STATUS_WORD = {"ok": "passed", "fail": "failed", "not run": "not run"}
_CARD_TAG = {"ok": "pass", "fail": "fail", "caveat": "caveat", "info": None}


@dataclass(frozen=True)
class SeqBlock:
    """One block of a pulse-sequence schematic. ``swept`` is the block's sweep range text (idle or drive)."""

    kind: Literal["drive", "idle", "readout", "wait"]
    label: str
    swept: str | None = None
    width: float = 1.0
    line: Literal["qubit", "resonator"] = "qubit"


@dataclass(frozen=True)
class Card:
    """One side card: a short title, a few lines, and an optional status accent."""

    title: str
    lines: tuple[str, ...]
    status: Literal["ok", "fail", "info", "caveat"] | None = None


@dataclass
class FigureContext:
    """What a node figure knows about the run it belongs to (built by ``figure_context``)."""

    node_order: list[str]
    statuses: dict[str, str]
    before: dict | None
    after: dict | None
    reference: dict | None
    population_levels: tuple[float, float] | None
    in_graph: bool = False  # True when the node runs inside run_graph (ctx.graph_nodes is set)


@dataclass
class NodeLayout:
    """The axes of a Layout C node figure, as made by :func:`node_layout`.

    ``residual_axes`` holds the first residual strip of each column (None without residuals);
    ``residual_strips`` holds all of them, one list per column, top to bottom.
    """

    fig: Figure
    progress_ax: Axes
    sequence_ax: Axes
    main_axes: list[Axes]
    residual_axes: list[Axes | None]
    cards_ax: Axes | None
    residual_strips: list[list[Axes]] = field(default_factory=list)


def _inch_axes(
    fig: Figure,
    place: Callable[[float, float, dict], tuple[float, float, float, float]],
    inch_x: bool = False,
) -> Axes:
    """Add axes whose box ``place(w, h, layout)`` gives in inches as (x0, top, width, height).

    ``top`` is measured down from the figure top. The box is recomputed at every draw, so it
    follows ``fit_layout`` when that widens the figure.
    """

    def bounds() -> Bbox:
        w, h = fig.get_size_inches()
        x0, top, width, height = place(w, h, fig._style_layout)
        return Bbox.from_bounds(x0 / w, 1 - (top + height) / h, width / w, height / h)

    def locate(target: Axes, _renderer) -> Bbox:
        box = bounds()
        if inch_x:  # keep 1 data unit = 1 inch on x, so drawings do not stretch when widened
            target.set_xlim(0, box.width * fig.get_size_inches()[0])
        return box

    ax = fig.add_axes(bounds().bounds)
    ax.set_axes_locator(locate)
    ax.set_axis_off()
    return ax


def node_layout(
    theme: str,
    ncols: int = 1,
    residuals: bool = True,
    cards: bool = True,
    figsize: tuple[float, float] | None = None,
    top_extra: float = 0.0,
    bottom_extra: float = 0.0,
    width_ratios: Sequence[float] | None = None,
    residual_rows: int = 1,
) -> NodeLayout:
    """Create a Layout C figure: progress strip, pulse sequence, main row, cards column.

    The main row follows ``fit_axes`` (3:1 residual row with shared x, ``place_legend`` above
    the main axes, margins fitted by ``fit_layout`` in ``save_themed``). The progress strip runs
    the full width under the title block, the sequence strip spans the main columns, and the
    cards column sits right of the main row from the sequence strip down.

    ``top_extra`` and ``bottom_extra`` (inches) reserve room above the main axes for a top twin
    axis (for example a prefactor axis; ``place_legend`` lifts the legend row by the same amount)
    and below the lowest axes for a bottom twin axis (for example a GHz axis). Without a
    ``figsize`` the figure grows by them and panel sizes stay the same. ``width_ratios`` sets
    relative column widths. ``residual_rows=2`` stacks two residual strips per column, sharing x
    with the main axes (see ``NodeLayout.residual_strips``).
    """
    if top_extra < 0 or bottom_extra < 0:
        raise ValueError("top_extra and bottom_extra must be non-negative")
    right = MARGIN_RIGHT + (CARDS_GAP + CARDS_W if cards else 0.0)
    # A right-hand P(e) axis may use part of the gap, but its label keeps CARDS_CLEAR + LABEL_PAD
    # inches from the cards.
    clear = CARDS_GAP - CARDS_CLEAR if cards else MARGIN_RIGHT
    strips: list[list[Axes]] = []
    fig, axes, res_axes = _grid_axes(
        theme,
        ncols,
        residuals,
        figsize,
        None,
        top=NODE_MAIN_TOP + top_extra,
        right=right,
        right_clear=clear,
        bottom=MARGIN_BOTTOM + bottom_extra,
        width_ratios=width_ratios,
        residual_rows=residual_rows,
        strips=strips,
    )
    fig._style_layout["top_extra"] = float(top_extra)

    def progress_box(w, h, layout):
        return TITLE_X, PROGRESS_TOP, w - TITLE_X - MARGIN_RIGHT, PROGRESS_H

    def sequence_box(w, h, layout):
        span = layout["ncols"] * layout["panel_w"] + (layout["ncols"] - 1) * layout["gap"]
        return layout["left"], SEQ_TOP, span, SEQ_H

    def cards_box(w, h, layout):
        return w - MARGIN_RIGHT - CARDS_W, SEQ_TOP, CARDS_W, h - SEQ_TOP - layout["bottom"]

    progress_ax = _inch_axes(fig, progress_box, inch_x=True)
    sequence_ax = _inch_axes(fig, sequence_box)
    cards_ax = _inch_axes(fig, cards_box) if cards else None
    return NodeLayout(fig, progress_ax, sequence_ax, axes, res_axes, cards_ax, strips)


def _ax_inches(ax: Axes) -> tuple[float, float]:
    """Width and height of ``ax`` in inches, from its current position."""
    w, h = ax.figure.get_size_inches()
    pos = ax.get_position()
    return pos.width * w, pos.height * h


def _measure(ax: Axes, text: str, size: float, weight: str = "normal") -> float:
    """Rendered width of ``text`` in inches at ``size`` points in the current theme font."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    prop = font_manager.FontProperties(family=plt.rcParams["font.family"], size=size, weight=weight)
    width, _, _ = renderer.get_text_width_height_descent(text, prop, ismath=_is_math(text))
    return width / renderer.dpi


def mix(color_a: str, color_b: str, t: float) -> str:
    """``t`` of ``color_a`` over ``color_b`` as an opaque hex colour (t = 0 gives ``color_b``).

    For tints: ``mix(p.fit, p.bg, 0.2)`` is a light fit-coloured fill on the theme background.
    """
    a, b = np.array(to_rgb(color_a)), np.array(to_rgb(color_b))
    return to_hex(t * a + (1 - t) * b)


_mix = mix  # older name, kept for existing callers


def _inch_canvas(ax: Axes) -> tuple[float, float]:
    """Turn ``ax`` into a blank drawing area whose data units are inches; return its size."""
    w, h = _ax_inches(ax)
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return w, h


def progress_strip(
    ax: Axes,
    node_order: list[str],
    current: str | None,
    statuses: dict[str, str],
    theme: str,
    labels: dict[str, str] | None = None,
) -> list[Artist]:
    """Draw one pill per node in graph order, with a status dot and ``current`` outlined.

    Each pill holds a status dot (gid ``name``; ``palette.ok`` for ``"ok"``, ``palette.fail``
    for ``"fail"``) and the node's short label (``labels``, default :data:`NODE_SHORT`). ok and
    fail pills get a light tint of their status colour; ``"not run"`` is a muted dashed pill;
    nodes absent from ``statuses`` are plain. The status is also written under each pill, so it
    does not rely on colour alone. The pill itself has gid ``name + ":pill"``. Pills fill the
    axes width at call time and are sized in inches, so their caps stay round.
    """
    p = palette(theme)
    w, h = _inch_canvas(ax)
    n = len(node_order)
    artists: list[Artist] = []
    if n == 0:
        return artists
    names = {name: (labels or {}).get(name, NODE_SHORT.get(name, name)) for name in node_order}
    size = 9.0
    dot_room = 2 * PILL_DOT_R + 0.08  # dot plus its gap before the label
    gaps = (n - 1) * PILL_GAP

    def text_widths(sz: float) -> list[float]:
        return [_measure(ax, names[name], sz, "semibold") for name in node_order]

    text_w = text_widths(size)
    usable = w - 2 * STRIP_INSET  # pills keep the same inset on both ends
    room = usable - gaps - (2 * PILL_PAD + dot_room) * n
    if sum(text_w) > room:
        size = max(MIN_TEXT, size * room / sum(text_w))
        text_w = text_widths(size)
    natural = [tw + 2 * PILL_PAD + dot_room for tw in text_w]
    fill = (usable - gaps) / n
    if fill >= max(natural):
        # Equal pills when every label fits; a lone pill keeps a modest width.
        widths = [min(fill, max(max(natural), 1.7))] * n
    else:
        extra = max(0.0, (usable - gaps - sum(natural)) / n)
        widths = [nat + extra for nat in natural]
    y0 = h - PILL_H - 0.04
    mid_y = y0 + PILL_H / 2
    centres: list[float] = []
    x = STRIP_INSET
    for name, width in zip(node_order, widths):
        status = statuses.get(name)
        dot_edge = None
        if status in ("ok", "fail"):
            colour = p.ok if status == "ok" else p.fail
            face, edge, text, dash, dot = _mix(colour, p.bg, 0.12), colour, p.fg, "solid", colour
        elif status == "not run":
            face, edge, text, dash, dot = p.bg, p.muted, p.muted, (0, (2.5, 2)), p.bg
            dot_edge = p.muted
        else:
            face, edge, text, dash, dot = p.panel, p.grid, p.muted, "solid", p.grid
        is_current = name == current
        pill = FancyBboxPatch(
            (x, y0),
            width,
            PILL_H,
            boxstyle=f"round,pad=0,rounding_size={PILL_H / 2}",
            facecolor=face,
            edgecolor=p.fg if is_current else edge,
            linewidth=1.6 if is_current else 0.8,
            linestyle="solid" if is_current else dash,
            zorder=2,
            gid=f"{name}:pill",
        )
        ax.add_patch(pill)
        dot_x = x + PILL_PAD + PILL_DOT_R
        marker = Circle(
            (dot_x, mid_y),
            PILL_DOT_R,
            facecolor=dot,
            edgecolor=dot_edge or dot,
            linewidth=0.8,
            zorder=3,
            gid=name,
        )
        ax.add_patch(marker)
        text_x = (x + PILL_PAD + dot_room + x + width - PILL_PAD) / 2
        label = ax.text(
            text_x,
            mid_y,
            names[name],
            color=text,
            fontsize=size,
            fontweight="semibold" if is_current else "medium",
            ha="center",
            va="center_baseline",
            zorder=4,
        )
        artists += [pill, marker, label]
        word = _STATUS_WORD.get(status or "")
        if word:
            caption = ax.text(
                x + width / 2,
                y0 - 0.075,
                word,
                color=p.fail if status == "fail" else p.muted,
                fontsize=7.0,
                ha="center",
                va="top",
                zorder=4,
            )
            artists.append(caption)
        centres.append(x + width / 2)
        x += width + PILL_GAP
    if n > 1:
        (link,) = ax.plot([centres[0], centres[-1]], [mid_y] * 2, color=p.grid, linewidth=1.0, zorder=1)
        artists.append(link)
    return artists


def pulse_sequence(
    ax: Axes, blocks: list[SeqBlock], theme: str, to_scale: bool = False
) -> list[Artist]:
    """Draw a two-line pulse schematic: qubit drive line above, resonator line below.

    Blocks run left to right in time on a shared clock, each on its own line. Drive pulses are
    tinted fit-coloured boxes, readout a neutral box, ``wait`` (for example thermalize) a dashed
    box, and idles a bare stretch of line. Any block with ``swept`` set (an idle, or a drive
    whose amplitude or frequency is swept) gets a bracket above it with its ``swept`` text.
    Unless ``to_scale``, a "not to scale" caption goes top right.
    """
    p = palette(theme)
    ax.set_axis_off()
    total = sum(b.width for b in blocks)
    lead = 0.35
    ax.set_xlim(0, total + 2 * lead)
    ax.set_ylim(0, 1)
    rows = {"qubit": (0.45, 0.72), "resonator": (0.06, 0.33)}  # (baseline, box top), fractions
    label_x = blended_transform_factory(ax.transAxes, ax.transData)
    artists: list[Artist] = []
    for line, (base, _top) in rows.items():
        (wire,) = ax.plot([0, total + 2 * lead], [base, base], color=p.muted, linewidth=0.8, zorder=1)
        name = ax.annotate(
            line,
            xy=(0, base),
            xycoords=label_x,
            xytext=(-6, 0),
            textcoords="offset points",
            color=p.muted,
            fontsize=7.5,
            ha="right",
            va="center_baseline",
            annotation_clip=False,
        )
        artists += [wire, name]
    inset = min(0.07, 0.1 * min((b.width for b in blocks), default=1.0))
    x = lead
    for block in blocks:
        base, top = rows[block.line]
        x0, x1 = x, x + block.width
        mid = (x0 + x1) / 2
        if block.kind in ("drive", "readout", "wait"):
            if block.kind == "drive":
                face, edge, text, dash = _mix(p.fit, p.bg, 0.2), p.fit, p.fg, "solid"
            elif block.kind == "readout":
                face, edge, text, dash = p.panel, p.data, p.fg, "solid"
            else:
                face, edge, text, dash = p.bg, p.muted, p.muted, (0, (2.5, 2))
            box = FancyBboxPatch(
                (x0 + inset, base),
                block.width - 2 * inset,
                top - base,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
                linestyle=dash,
                zorder=2,
            )
            ax.add_patch(box)
            label = ax.text(
                mid,
                (base + top) / 2,
                block.label,
                color=text,
                fontsize=8.0,
                ha="center",
                va="center_baseline",
                zorder=3,
            )
            artists += [box, label]
        elif block.label:  # idle: a bare stretch of line
            idle = ax.text(
                mid, base + 0.05, block.label, color=p.muted, fontsize=8.0,
                ha="center", va="bottom", zorder=3,
            )
            artists.append(idle)
        if block.swept:  # any swept block (idle, drive, ...) gets a bracket and its range
            lo, hi = top + 0.02, top + 0.08
            (bracket,) = ax.plot(
                [x0 + inset, x0 + inset, x1 - inset, x1 - inset],
                [lo, hi, hi, lo],
                color=p.fit,
                linewidth=0.9,
                solid_joinstyle="miter",
                zorder=3,
            )
            sweep = ax.text(
                mid, hi + 0.04, block.swept, color=p.fit, fontsize=7.5,
                ha="center", va="bottom", zorder=3,
            )
            artists += [bracket, sweep]
        x = x1
    if not to_scale:
        caption = ax.text(
            1.0, 1.0, "not to scale", transform=ax.transAxes, color=p.muted,
            fontsize=7.0, fontstyle="italic", ha="right", va="top",
        )
        artists.append(caption)
    return artists


def _wrap(ax: Axes, lines: Sequence[str], size: float, avail: float) -> list[str]:
    """Wrap ``lines`` at spaces so each fits ``avail`` inches at ``size`` points."""
    out: list[str] = []
    for line in lines:
        current = ""
        for word in line.split(" "):
            trial = f"{current} {word}" if current else word
            if current and _measure(ax, trial, size) > avail:
                out.append(current)
                current = word
            else:
                current = trial
        out.append(current)
    return out


def cards_column(ax: Axes, cards: list[Card], theme: str) -> list[Artist]:
    """Stack ``cards`` from the top of ``ax``: small muted title, then one line per entry.

    A card with a status gets a coloured accent bar (ok, fail, info, caveat) and, except for
    info, a status word top right. Text shrinks (down to MIN_TEXT points) and then wraps at
    spaces so every card stays inside the axes.
    """
    p = palette(theme)
    w, h = _inch_canvas(ax)
    artists: list[Artist] = []
    if not cards:
        return artists
    accent_of = {"ok": p.ok, "fail": p.fail, "info": p.fit, "caveat": p.guess}
    title_size, body_size, spacing, title_lead = 7.5, 8.5, 1.5, 1.75

    def inner_left(card: Card) -> float:
        return CARD_PAD + CARD_ACCENT + 0.02  # same text inset with or without an accent bar

    def widest(scale: float) -> float:
        """Largest ratio of needed to available width over all cards."""
        worst = 0.0
        for card in cards:
            avail = w - inner_left(card) - CARD_PAD
            tag = _CARD_TAG.get(card.status or "")
            title_w = _measure(ax, card.title.upper(), title_size * scale, "medium")
            if tag:
                title_w += 0.12 + _measure(ax, tag, title_size * scale, "medium")
            # Body lines wrap at spaces, so only the longest single word forces a smaller size.
            words = [wd for ln in card.lines for wd in ln.split(" ")]
            need = max([title_w] + [_measure(ax, wd, body_size * scale) for wd in words])
            worst = max(worst, need / avail)
        return worst

    def layout(scale: float) -> tuple[list[list[str]], list[float]]:
        wrapped = [_wrap(ax, c.lines, body_size * scale, w - inner_left(c) - CARD_PAD) for c in cards]
        line_h = body_size * scale * spacing / 72
        title_h = title_size * scale * title_lead / 72
        sizes = [2 * CARD_PAD + title_h + len(ls) * line_h - 0.04 for ls in wrapped]
        return wrapped, sizes

    floor = MIN_TEXT / body_size
    scale = max(floor, min(1.0, 1.0 / max(widest(1.0), 1e-9)))
    wrapped, sizes = layout(scale)
    total = sum(sizes) + CARD_GAP * (len(cards) - 1)
    if total > h and scale > floor:
        scale = max(floor, scale * h / total)
        wrapped, sizes = layout(scale)
    top = h
    for card, lines, card_h in zip(cards, wrapped, sizes):
        y0 = top - card_h
        box = FancyBboxPatch(
            (0.01, y0), w - 0.02, card_h, boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=p.panel, edgecolor=p.grid, linewidth=0.8, zorder=1,
        )
        ax.add_patch(box)
        artists.append(box)
        if card.status in accent_of:
            bar = FancyBboxPatch(
                (CARD_PAD - 0.03, y0 + CARD_PAD - 0.02), CARD_ACCENT, card_h - 2 * CARD_PAD + 0.04,
                boxstyle=f"round,pad=0,rounding_size={CARD_ACCENT / 2}",
                facecolor=accent_of[card.status], edgecolor="none", zorder=2,
            )
            ax.add_patch(bar)
            artists.append(bar)
        left = inner_left(card)
        y = top - CARD_PAD
        title = ax.text(
            left, y, card.title.upper(), color=p.muted, fontsize=title_size * scale,
            fontweight="medium", ha="left", va="top", zorder=3,
        )
        artists.append(title)
        tag = _CARD_TAG.get(card.status or "")
        if tag:
            tag_text = ax.text(
                w - CARD_PAD, y, tag, color=accent_of[card.status], fontsize=title_size * scale,
                fontweight="medium", ha="right", va="top", zorder=3,
            )
            artists.append(tag_text)
        y -= title_size * scale * title_lead / 72
        for line in lines:
            body = ax.text(
                left, y, line, color=p.fg, fontsize=body_size * scale, ha="left", va="top", zorder=3
            )
            artists.append(body)
            y -= body_size * scale * spacing / 72
        top = y0 - CARD_GAP
    return artists


def format_change(
    label: str, before: float | None, after: float | None, scale: float, unit: str, fmt: str
) -> str:
    """State-change line ``"label: before -> after unit"``; a missing value reads "uncalculated".

    Values are multiplied by ``scale`` and formatted with ``fmt``. The unit follows the last
    number after a no-break space (U+00A0), so it never wraps onto its own line:
    ``("T1", None, 47700.0, 1e-3, US, "{:.1f}")`` gives ``"T1: uncalculated -> 47.7\u00a0µs"``.
    NaN and infinity also read "uncalculated".
    """

    def show(value: float | None) -> str | None:
        return fmt.format(value * scale) if _finite(value) else None

    b, a = show(before), show(after)
    suffix = f"{NBSP}{unit}" if unit else ""
    if a is not None:
        return f"{label}: {b or 'uncalculated'} -> {a}{suffix}"
    if b is not None:
        return f"{label}: {b}{suffix} -> uncalculated"
    return f"{label}: uncalculated -> uncalculated"


def population_axis(ax: Axes, level_g: float, level_e: float, theme: str) -> Axes:
    """Add a right-hand P(e) axis: P(e) = (s - level_g) / (level_e - level_g).

    ``level_g`` and ``level_e`` are the measured 07 blob means projected onto this node's signal
    axis (measured, not truth). The mapping is for plotting only; fits keep their own inputs.
    ``fit_layout`` makes room for the axis labels on the right.
    """
    if not (math.isfinite(level_g) and math.isfinite(level_e)) or level_e == level_g:
        raise ValueError("level_g and level_e must be finite and different")
    p = palette(theme)
    span = level_e - level_g

    def to_population(s):
        return (np.asarray(s, dtype=float) - level_g) / span

    def to_signal(pe):
        return level_g + np.asarray(pe, dtype=float) * span

    sec = ax.secondary_yaxis("right", functions=(to_population, to_signal))
    sec.spines["right"].set_visible(True)
    sec.spines["right"].set_color(p.muted)
    sec.spines["right"].set_linewidth(0.6)
    sec.yaxis.set_major_locator(FixedLocator([0.0, 0.25, 0.5, 0.75, 1.0]))  # physical values only
    sec.tick_params(axis="y", colors=p.muted, labelcolor=p.muted, labelsize=8.0)
    sec.set_ylabel("P(e), from 07 blob means", color=p.fg, fontsize=9.0)
    return sec
