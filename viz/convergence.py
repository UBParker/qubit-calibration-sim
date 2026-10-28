"""Bring-up convergence: every calibrated parameter, node by node, against its reference.

The figure reads a finished (or stopped) graph run, ``ctx.history`` from ``run_graph``, and the
reference dict from ``simulator.reference.build_reference``. It reruns nothing. On top sits the
node chain, coloured by each node's outcome; below it one row per parameter shares the same
columns: the rough guess from the state before the first node, then the stored value after each
node. The y axis is the error against the reference in natural units, with the closure-test
tolerance shaded and zero meaning "matches the reference".

Errors are against the true value stored on the simulated device, except readout fidelity, which
the device does not store and is compared with the simulator reference (high-shot IQ blobs with
the true pi pulse). Tolerances come from ``simulator.reference.CLOSURE_TOLERANCES``, the same
values tests/test_90_bringup.py asserts.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from calibration_utils.common import QUBIT, NodeRecord
from calibration_utils.style import (
    NODE_SHORT,
    PILL_GAP,
    PILL_H,
    PROGRESS_H,
    PROGRESS_TOP,
    STRIP_INSET,
    TITLE_X,
    _mix,
    palette,
    progress_strip,
    save_themed,
    theme_context,
    title_block,
)
from simulator.device import TRUE_DEVICE
from simulator.reference import CLOSURE_TOLERANCES

FIGURE_NAME = "bringup-convergence"
KAPPA = TRUE_DEVICE.kappa  # true resonator linewidth (FWHM), the natural unit for f_res errors


@dataclass(frozen=True)
class Parameter:
    """One calibrated state entry and how its error is shown.

    ``error(value, reference)`` gives the error in ``unit``. ``setters`` are the nodes that write
    the entry; the last one that ran gives the final value. ``ticks`` overrides the default y
    ticks (band edges and zero).
    """

    key: str  # reference and CLOSURE_TOLERANCES key
    name: str
    subtitle: str
    unit: str
    read: Callable[[dict], float | None]
    error: Callable[[float, float], float]
    fmt: str  # error format
    setters: tuple[str, ...]
    reference_kind: str = "true value"
    ticks: tuple[float, ...] | None = None


def _qubit(state: dict) -> dict:
    return state["qubits"][QUBIT]


def _relative(value: float, ref: float) -> float:
    return 100.0 * (value / ref - 1.0)


PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "readout.frequency", "Readout frequency",
        f"f_ro, error vs the g/e midpoint in κ (κ = {KAPPA / 1e6:g} MHz; 02a finds the |g⟩ dip, 08a the midpoint)", "κ",
        lambda s: _qubit(s)["resonator"]["f_01"], lambda v, r: (v - r) / KAPPA, "{:+.3f}",
        ("02a_resonator_spectroscopy", "08a_readout_frequency_optimization"),
    ),
    Parameter(
        "f_01", "Qubit frequency", "f_01, error vs true value (kHz)", "kHz",
        lambda s: _qubit(s)["f_01"], lambda v, r: (v - r) / 1e3, "{:+.2f}",
        ("03a_qubit_spectroscopy", "06a_ramsey"),
    ),
    Parameter(
        "x180.amplitude", "π pulse amplitude", "x180, error vs true value (%)", "%",
        lambda s: _qubit(s)["xy"]["operations"]["x180"]["amplitude"], _relative, "{:+.2f}",
        ("04b_power_rabi",),
    ),
    Parameter(
        "fidelity", "Readout fidelity", "F, error vs simulator reference (pp)", "pp",
        lambda s: _qubit(s)["resonator"]["readout_fidelity"], lambda v, r: 100.0 * (v - r), "{:+.2f}",
        ("07_iq_blobs",), reference_kind="simulator reference", ticks=(-4.0, 0.0, 4.0),
    ),
    Parameter(
        "T2ramsey", "Ramsey dephasing", "T2*, error vs true value (%)", "%",
        lambda s: _qubit(s)["T2ramsey"], _relative, "{:+.1f}", ("06a_ramsey",),
    ),
    Parameter(
        "T1", "Energy relaxation", "T1, error vs true value (%)", "%",
        lambda s: _qubit(s)["T1"], _relative, "{:+.1f}", ("05_T1",),
    ),
    Parameter(
        "T2echo", "Echo dephasing", "T2 echo, error vs true value (%)", "%",
        lambda s: _qubit(s)["T2echo"], _relative, "{:+.1f}", ("06b_echo",),
    ),
)


def band(par: Parameter, ref: float) -> tuple[float, float]:
    """The closure tolerance of ``par`` as (low, high) errors in ``par.unit``."""
    tol = CLOSURE_TOLERANCES[par.key]
    if "abs" in tol:
        edges = (ref - tol["abs"], ref + tol["abs"])
    elif "rel" in tol:
        edges = (ref * (1 - tol["rel"]), ref * (1 + tol["rel"]))
    else:
        edges = tol["window"]
    return par.error(edges[0], ref), par.error(edges[1], ref)


def band_text(par: Parameter, ref: float) -> str:
    """The tolerance as the closure test states it, e.g. "±20 kHz" or "0.95 to 0.995 window"."""
    tol = CLOSURE_TOLERANCES[par.key]
    if "abs" in tol:
        text = f"±{tol['abs'] / 1e3:g} kHz"
        if par.unit == "κ":
            text += f" = ±{band(par, ref)[1]:.2g} κ"
        return text
    if "rel" in tol:
        return f"±{100 * tol['rel']:g} %"
    low, high = tol["window"]
    return f"{low:g} to {high:g} window"


@dataclass
class Row:
    """One parameter's track. Indices are positions in ``node_order``.

    ``values`` holds the error after each node (None where the node did not run or the state has
    no value yet), ``guess`` the error of the rough guess before the first node, ``first_set`` the
    first node that set the parameter and ``changed`` every node that changed it. ``final`` is the
    error of the last stored value and ``final_value`` that value itself.
    """

    key: str
    guess: float | None
    values: list[float | None]
    first_set: int | None
    changed: list[int]
    final: float | None
    final_value: float | None
    band: tuple[float, float]

    @property
    def measured(self) -> bool:
        return self.first_set is not None

    @property
    def within(self) -> bool:
        return self.measured and self.final is not None and self.band[0] <= self.final <= self.band[1]


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def _error(par: Parameter, ref: float, state: dict) -> float | None:
    value = par.read(state)
    return par.error(value, ref) if _finite(value) else None


def convergence_rows(history: list[NodeRecord], reference: dict, node_order: list[str]) -> list[Row]:
    """Errors against the reference for every parameter in :data:`PARAMETERS`, node by node."""
    by_name = {record.name: record for record in history}
    rows: list[Row] = []
    for par in PARAMETERS:
        ref = reference[par.key]
        err = partial(_error, par, ref)
        guess = err(history[0].before) if history else None
        values: list[float | None] = []
        changed: list[int] = []
        for i, name in enumerate(node_order):
            record = by_name.get(name)
            if record is None:
                values.append(None)
                continue
            before, after = par.read(record.before), par.read(record.after)
            if _finite(after) and after != before:
                changed.append(i)
            values.append(err(record.after))
        measured = bool(history and changed)
        final = err(history[-1].after) if measured else None
        final_value = par.read(history[-1].after) if measured else None
        first = changed[0] if changed else None
        rows.append(Row(par.key, guess, values, first, changed, final, final_value, band(par, ref)))
    return rows


def _statuses(history: list[NodeRecord], node_order: list[str]) -> dict[str, str]:
    done = {record.name: ("ok" if record.success else "fail") for record in history}
    return {name: done.get(name, "not run") for name in node_order}


# Page geometry, inches. The node track must stay at least about 7.8 in wide so the eight strip
# labels fit at the strip's smallest text size.
FIG_W = 9.8
LEFT = 0.62  # y tick labels
GUESS_W = 0.78  # rough-guess column left of the node columns
RIGHT = 0.25
LEGEND_Y = 1.56  # legend row centre, from the top
ROWS_TOP = 2.1  # top of the first panel
ROW_H, ROW_TITLE = 0.56, 0.34  # panel height and the title line above each panel
EDGE = 0.92  # off-scale markers sit at this fraction of the half-range
FOOT = 0.95  # below the last panel: footnote, up to four lines
SMALL = 7.5  # smallest text on the page, points
VALUE_LABEL_ROOM = 0.6  # inches right of a point needed for its value label; less puts it left


def _figure_height() -> float:
    n = len(PARAMETERS)
    return ROWS_TOP + n * ROW_H + (n - 1) * ROW_TITLE + FOOT


def _add_axes(fig: Figure, x0: float, top: float, width: float, height: float) -> Axes:
    w, h = fig.get_size_inches()
    return fig.add_axes((x0 / w, 1 - (top + height) / h, width / w, height / h))


def _limits(row: Row) -> float:
    """Half-range of the y axis: twice the tolerance, wider only for a fitted value outside it.

    A window tolerance gets 2.4 times its larger edge, which leaves room for the "F <edge>"
    labels outside the band.
    """
    tol = max(abs(row.band[0]), abs(row.band[1]))
    span = (2.4 if "window" in CLOSURE_TOLERANCES[row.key] else 2.0) * tol
    fitted = [abs(v) for v in row.values[row.first_set:] if v is not None] if row.measured else []
    if fitted and max(fitted) > 0.85 * span:
        span = min(1.2 * max(fitted), 10.0 * tol)
    return span


def _guess_off_scale(row: Row) -> bool:
    return row.guess is not None and abs(row.guess) > EDGE * _limits(row)


def _tick_formatter(fmt: str) -> FuncFormatter:
    def show(value, _pos=None):
        text = fmt.replace("+", "").format(value)
        if float(text) == 0:
            return "0"
        return text.rstrip("0").rstrip(".").replace("-", "−")

    return FuncFormatter(show)


def _compact(value: float) -> str:
    """A signed value with about two significant figures, for the rough guess."""
    size = abs(value)
    fmt = "{:+.0f}" if size >= 10 else "{:+.1f}" if size >= 1 else "{:+.2f}"
    return fmt.format(value).replace("-", "−")


def _signed(fmt: str, value: float) -> str:
    return fmt.format(value).replace("-", "−")


def plot_convergence(
    history: list[NodeRecord], reference: dict, theme: str, node_order: list[str]
) -> Figure:
    """Draw the bring-up convergence page for ``history`` (complete or stopped early).

    One strip axes (label ``"progress strip"``) and one panel per parameter (label
    ``"panel:<key>"``). Nodes in ``node_order`` absent from ``history`` show as "not run"; a
    parameter no node set says "not measured".
    """
    with theme_context(theme):
        return _plot(history, reference, theme, node_order)


def _plot(history: list[NodeRecord], reference: dict, theme: str, node_order: list[str]) -> Figure:
    p = palette(theme)
    rows = convergence_rows(history, reference, node_order)
    statuses = _statuses(history, node_order)
    fig = plt.figure(figsize=(FIG_W, _figure_height()), facecolor=p.bg)
    h = fig.get_size_inches()[1]

    # Node chain. Panels share its columns: x data units are inches from the panel's left edge.
    track_x0 = LEFT + GUESS_W
    strip = _add_axes(fig, track_x0, PROGRESS_TOP, FIG_W - RIGHT - track_x0, PROGRESS_H)
    strip.set_label("progress strip")
    artists = progress_strip(strip, node_order, None, statuses, theme)
    pills = {a.get_gid(): a for a in artists if (a.get_gid() or "").endswith(":pill")}
    spans = [
        (GUESS_W + pill.get_x(), GUESS_W + pill.get_x() + pill.get_width())
        for pill in (pills[f"{name}:pill"] for name in node_order)
    ]
    centres = [(a + b) / 2 for a, b in spans]
    pill_mid = PROGRESS_TOP + STRIP_INSET + PILL_H / 2  # as progress_strip places the pills
    guess_fx = (LEFT + GUESS_W / 2) / FIG_W
    fig.text(guess_fx, 1 - pill_mid / h, "rough guess", color=p.guess, fontsize=8.0,
             fontweight="medium", ha="center", va="center")
    fig.text(guess_fx, 1 - (pill_mid + 0.25) / h, "state_initial", color=p.muted, fontsize=7.0,
             ha="center", va="center")

    n_pass = sum(s == "ok" for s in statuses.values())
    n_run = sum(s != "not run" for s in statuses.values())
    n_within = sum(row.within for row in rows)
    n_measured = sum(row.measured for row in rows)
    headline = f"{n_pass} of {len(node_order)} nodes passed, {n_within} of {len(rows)} parameters within tolerance"
    title_block(fig, "90 bring-up graph", "Bring-up convergence", headline, theme)

    panel_w = FIG_W - RIGHT - LEFT
    # Not-run columns are shaded across the pill and half the gap on each side, so neighbours join.
    not_run = [
        (a - PILL_GAP / 2, b + PILL_GAP / 2) for (a, b), name in zip(spans, node_order) if statuses[name] == "not run"
    ]
    ran = [i for i, name in enumerate(node_order) if statuses[name] != "not run"]
    for k, (par, row) in enumerate(zip(PARAMETERS, rows)):
        top = ROWS_TOP + k * (ROW_H + ROW_TITLE)
        ax = _add_axes(fig, LEFT, top, panel_w, ROW_H)
        ax.set_label(f"panel:{par.key}")
        ax.set_xlim(0.0, panel_w)
        pending = [name for name in par.setters if name in statuses and statuses[name] != "ok"]
        last_ran = centres[ran[-1]] if ran else None
        _draw_row(ax, par, row, reference[par.key], centres, not_run, last_ran, pending, statuses, theme)

    guesses = [_guess_off_scale(row) for row in rows if row.guess is not None]
    handles = []
    if not all(guesses):
        handles.append((Line2D([], [], color=p.guess, linestyle=(0, (4, 3)), marker="D", markersize=4.5,
                               markerfacecolor=p.bg, markeredgecolor=p.guess, linewidth=1.2), "rough guess"))
    if any(guesses):
        handles.append((Line2D([], [], color=p.guess, linestyle=(0, (4, 3)), marker="v", markersize=6,
                               linewidth=1.2), "rough guess, off scale"))
    handles += [
        (Line2D([], [], color=p.fit, linewidth=1.6), "stored value after each node"),
        (Line2D([], [], color=p.fit, marker="o", markersize=7, markerfacecolor=p.fit,
                markeredgecolor=p.bg, markeredgewidth=1.4, linestyle="none"), "first set"),
        (Patch(facecolor=_mix(p.truth, p.bg, 0.16), edgecolor="none"), "closure tolerance"),
        (Line2D([], [], color=p.truth, linestyle=(0, (1, 2.2)), linewidth=1.5), "true value (zero error)"),
    ]
    fig.legend(
        [hd for hd, _ in handles], [lb for _, lb in handles],
        loc="center left", bbox_to_anchor=((TITLE_X - 0.06) / FIG_W, 1 - LEGEND_Y / h), ncols=len(handles),
        frameon=False, fontsize=SMALL, handlelength=1.8, handletextpad=0.45, columnspacing=1.0,
        labelcolor=p.fg,
    )
    lines = [
        "Error: stored value minus the true value of the simulated device, as a difference or a percent of it.",
        (
            "Readout fidelity is not a device parameter; it is compared with the simulator reference "
            "(high-shot IQ blobs, true π pulse, tone at the g/e midpoint)."
        ),
        "Shaded bands: the tolerances the closure test asserts. Rough guess: the state before the first node.",
    ]
    failed = [name for name in node_order if statuses[name] == "fail"]
    unmeasured = f"{len(rows) - n_measured} of {len(rows)} parameters were not measured."
    if failed:
        lines.append(f"The graph stopped at {NODE_SHORT.get(failed[0], failed[0])}, whose fit failed; {unmeasured}")
    elif n_run < len(node_order):
        lines.append(f"The history ends after {n_run} of {len(node_order)} nodes; {unmeasured}")
    fig.text(
        TITLE_X / FIG_W, (FOOT - 0.14) / h, "\n".join(lines), color=p.muted, fontsize=SMALL,
        ha="left", va="top", linespacing=1.45,
    )
    return fig


def _break_mark(ax: Axes, top: bool, colour: str) -> None:
    """Two short slashes across the y spine at the end a value runs off."""
    w, h = ax.figure.get_size_inches()
    pos = ax.get_position()
    ax_w, ax_h = pos.width * w, pos.height * h
    dx, dy, gap = 0.04 / ax_w, 0.025 / ax_h, 0.03 / ax_h  # inches to axes fraction
    yc = (1 + EDGE) / 2 if top else (1 - EDGE) / 2  # level of the off-scale marker
    for shift in (-gap, gap):
        ax.plot([-dx, dx], [yc + shift - dy, yc + shift + dy], color=colour, linewidth=0.9,
                transform=ax.transAxes, clip_on=False, zorder=6, solid_capstyle="butt")


def _draw_row(
    ax: Axes,
    par: Parameter,
    row: Row,
    ref: float,
    centres: list[float],
    not_run: list[tuple[float, float]],
    last_ran: float | None,
    pending: list[str],
    statuses: dict[str, str],
    theme: str,
) -> None:
    p = palette(theme)
    zero_colour = p.reference if par.reference_kind == "simulator reference" else p.truth
    guess_x = GUESS_W / 2
    span = _limits(row)
    ax.set_ylim(-span, span)
    ax.set_facecolor(p.bg)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(p.grid)
    ax.grid(False)
    ax.xaxis.set_major_locator(NullLocator())
    lo, hi = row.band
    ticks = par.ticks if par.ticks is not None else sorted({lo, 0.0, hi})
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(_tick_formatter(par.fmt))
    ax.tick_params(axis="y", length=2.5, pad=3, labelsize=SMALL)

    # Column guides, then the not-run columns and the tolerance band.
    for x in [guess_x, *centres]:
        ax.axvline(x, color=p.grid, linewidth=0.6, zorder=0.2)
    for x0, x1 in not_run:
        ax.axvspan(x0, x1, color=p.panel, linewidth=0, zorder=0.1)
    ax.axhspan(lo, hi, color=_mix(zero_colour, p.bg, 0.16), linewidth=0, zorder=0.3)
    ax.axhline(0.0, color=zero_colour, linestyle=(0, (1, 2.2)), linewidth=1.3, zorder=0.5)
    window = CLOSURE_TOLERANCES[par.key].get("window")
    if window:
        # The window edges in absolute terms at the right end, just outside the band: above the
        # top edge and below the bottom one. The y range is at least twice the larger edge, so
        # both labels stay inside the axes (tests/test_convergence.py checks this).
        for value, edge, va, dy in ((window[1], hi, "bottom", 1), (window[0], lo, "top", -1)):
            ax.annotate(
                f"F {value:g}", xy=(1.0, edge), xycoords=("axes fraction", "data"), xytext=(-2, dy),
                textcoords="offset points", color=p.muted, fontsize=SMALL, ha="right", va=va, zorder=6,
                gid=f"window:{value:g}",
            )

    # Title line: name and unit on the left, final error on the right.
    title = ax.annotate(
        par.name, xy=(0, 1), xycoords="axes fraction", xytext=(-(LEFT - TITLE_X) * 72, 5),
        textcoords="offset points", color=p.fg, fontsize=8.5, fontweight="semibold",
        ha="left", va="bottom", annotation_clip=False,
    )
    ax.annotate(
        par.subtitle, xy=(1, 0.5), xycoords=title, xytext=(7, 0), textcoords="offset points",
        color=p.muted, fontsize=SMALL, ha="left", va="center", annotation_clip=False,
    )
    if row.measured:
        inside = row.within
        if window:
            word = "inside" if inside else "outside"
            readout = f"F {row.final_value:.4f}, {word} the {band_text(par, ref)}"
        else:
            word = "within" if inside else "outside"
            readout = f"final {_signed(par.fmt, row.final)} {par.unit}, {word} {band_text(par, ref)}"
        colour = p.ok if inside else p.fail
        if pending:
            # A later setter did not run (or failed): the value is provisional.
            name = pending[0]
            readout = readout.replace("final ", "", 1) + f"; {NODE_SHORT.get(name, name)} {_status_word(statuses[name])}"
            colour = p.muted
        ax.annotate(
            readout, xy=(1, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
            color=colour, fontsize=8.0, fontweight="medium", ha="right", va="bottom", annotation_clip=False,
        )

    # The rough guess, held (at the axis edge when off scale) until the first node sets it.
    guess, first = row.guess, row.first_set
    edge = EDGE * span
    shown = None
    if guess is not None:
        shown = max(-edge, min(edge, guess))
        off = abs(guess) > edge
        end_x = centres[first] if first is not None else (last_ran if last_ran is not None else guess_x)
        ax.plot([guess_x, end_x], [shown, shown], color=p.guess, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        if off:
            ax.plot([guess_x], [shown], marker="^" if guess > 0 else "v", markersize=6.5,
                    color=p.guess, linestyle="none", zorder=4)
            _break_mark(ax, guess > 0, p.muted)
        else:
            ax.plot([guess_x], [guess], marker="D", markersize=5, markerfacecolor=p.bg,
                    markeredgecolor=p.guess, markeredgewidth=1.3, linestyle="none", zorder=4)
        # An off-scale guess sits at the axis edge, so its label goes toward zero to stay on the
        # panel; an on-scale one goes away from zero so the dotted zero line does not cross it.
        down = (guess > 0) == off
        ax.annotate(
            f"{_compact(guess)} {par.unit}", xy=(guess_x, shown), xytext=(0, -5 if down else 5),
            textcoords="offset points", color=p.guess, fontsize=SMALL, ha="center",
            va="top" if down else "bottom", annotation_clip=False, gid=f"guess:{par.key}",
        )

    if not row.measured:
        mid = (centres[0] + centres[-1]) / 2
        ax.text(mid, 0.0, "not measured", color=p.muted, fontsize=8.0, fontstyle="italic",
                ha="center", va="center", zorder=5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": p.bg, "edgecolor": "none"})
        return

    if shown is not None and row.values[first] is not None:
        # The setting node's step, from the held guess to its fitted value.
        ax.plot([centres[first]] * 2, [shown, max(-span, min(span, row.values[first]))],
                color=p.fit, linewidth=1.6, zorder=3, solid_capstyle="butt")

    # Stored value after each node, from the node that first set it.
    xs, ys = [], []
    for i in range(first, len(centres)):
        v = row.values[i]
        if v is None:
            break
        xs.append(centres[i])
        ys.append(max(-span, min(span, v)))
    ax.plot(xs, ys, color=p.fit, linewidth=1.6, drawstyle="steps-post", zorder=3)
    for i in row.changed:
        v = row.values[i]
        if v is None:
            continue
        y = max(-span, min(span, v))
        big = i == first
        ax.plot([centres[i]], [y], marker="o", markersize=7 if big else 5.5, color=p.fit,
                markeredgecolor=p.bg, markeredgewidth=1.4 if big else 0.8, linestyle="none", zorder=5)
        # Labels go right of the point, or left of it in the last column so they stay inside, and
        # on the far side of the point from the zero line (above a positive error, below a
        # negative one) so the dotted true-value line never runs through them.
        left = ax.get_xlim()[1] - centres[i] < VALUE_LABEL_ROOM
        below = v < 0
        ax.annotate(
            _signed(par.fmt, v), xy=(centres[i], y), xytext=(-6 if left else 6, -5 if below else 5),
            textcoords="offset points", color=p.fit, fontsize=SMALL, ha="right" if left else "left",
            va="top" if below else "bottom", zorder=6, annotation_clip=False, gid=f"value:{par.key}:{i}",
        )


def _status_word(status: str) -> str:
    return {"fail": "failed", "not run": "not run"}.get(status, status)


def save_convergence(
    history: list[NodeRecord], reference: dict, node_order: list[str], out_dir: Path
) -> list[Path]:
    """Write ``<out_dir>/bringup-convergence-<theme>.png`` for both themes."""
    return save_themed(
        lambda theme: plot_convergence(history, reference, theme, node_order), FIGURE_NAME, out_dir
    )
