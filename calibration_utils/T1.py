"""Analysis for 05_T1, using QM's own fit routine from qualibration-libs."""

from __future__ import annotations

import logging
import math

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.analysis import fit_decay_exp
from qualibration_libs.analysis.models import decay_exp

from calibration_utils import style
from calibration_utils.common import (
    QUBIT,
    decay_fit_gates,
    decay_residual,
    fit_with_masking,
    new_gate_reason,
    outlier_summary,
    project_on_iq_axis,
    sweep_range_key,
)
from calibration_utils.robust import iq_line_outliers

log = logging.getLogger(__name__)


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Flag averaged points off the g-e line (``outlier``), then project the rest onto it."""
    outlier = iq_line_outliers((ds.I + 1j * ds.Q).sel(qubit=QUBIT).values)
    ds = ds.assign(outlier=("idle_time", outlier))
    return ds.assign(signal=project_on_iq_axis(ds, "idle_time", first_point_excited=True, exclude=outlier))


def fit_raw_data(ds: xr.Dataset) -> dict:
    """Fit a*exp(decay*t) + offset. QM's decay_exp convention: decay < 0, T1 = -1/decay.

    Off-line points (``ds.outlier``) and residual outliers are masked around QM's fitter
    (``fit_with_masking``). Gates (``decay_fit_gates``): converged, relative_error, sampled, outliers.
    """

    def reproject(mask):
        return project_on_iq_axis(ds, "idle_time", first_point_excited=True, exclude=mask)

    raw, mask, signal = fit_with_masking(ds, "idle_time", reproject, lambda d: fit_decay_exp(d, "idle_time"), decay_residual)
    if mask.any():
        ds["signal"], ds["outlier"] = signal, ("idle_time", mask)
    fit = raw.sel(qubit=QUBIT)
    decay = float(fit.sel(fit_vals="decay"))
    decay_var = float(fit.sel(fit_vals="decay_decay"))
    T1 = -1.0 / decay if decay < 0 else float("nan")
    T1_err = np.sqrt(abs(decay_var)) / decay**2 if decay < 0 else float("nan")
    gates, n_sampled = decay_fit_gates(T1, T1_err, ds.idle_time.values, mask, MAX_RELATIVE_ERROR)
    return {
        "T1": T1,
        "T1_error": T1_err,
        "fit_params": {k: float(fit.sel(fit_vals=k)) for k in ("a", "offset", "decay")},
        # sweep span in ns, for the pulse diagram and the coverage note (t_max in units of T1)
        sweep_range_key("idle_time"): (float(ds.idle_time.values.min()), float(ds.idle_time.values.max())),
        "n_sampled": n_sampled,
        **outlier_summary(mask),
        "gates": gates,
        "success": all(gates.values()),
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info("05_T1 %s: T1 = %.1f +/- %.1f us", status, fit["T1"] / 1e3, fit["T1_error"] / 1e3)


NODE_NAME = "05_T1"
SIGNAL_SCALE = 1e3  # the fit works in volts; the figure plots mV
TIME_SCALE = 1e-3  # idle times are in ns; the figure plots µs
RANGE_KEY = sweep_range_key("idle_time")
MAX_RELATIVE_ERROR = 0.2  # fit_raw_data rejects T1_error / T1 at or above this


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _fit_finite(fit: dict) -> bool:
    """True when T1 and the model parameters are finite numbers, accepted or not."""
    params = fit.get("fit_params", {})
    return _finite(fit.get("T1"), *(params.get(k) for k in ("a", "offset", "decay")))


def _fit_ok(fit: dict) -> bool:
    return bool(fit.get("success")) and _fit_finite(fit)


def _coverage(fit: dict) -> float | None:
    """Longest idle time in units of the fitted T1, or None without an accepted fit."""
    span = fit.get(RANGE_KEY)
    if span is None or not _fit_ok(fit):
        return None
    return span[1] / fit["T1"]


def _us(value_ns: float, digits: int = 1) -> str:
    return f"{value_ns * TIME_SCALE:.{digits}f} {style.US}"


def _duration(value_ns: float) -> str:
    return f"{value_ns:.0f} ns" if value_ns < 1000 else f"{value_ns * TIME_SCALE:g} {style.US}"


def _t1_text(fit: dict) -> str:
    """"T1 = 47.8 ± 1.7 µs", without a ± when the error is not a finite number."""
    if _finite(fit.get("T1_error")):
        return f"T1 = {fit['T1'] * TIME_SCALE:.1f} ± {_us(fit['T1_error'])}"
    return f"T1 = {_us(fit['T1'])}, error uncalculated"


def _rejection_reason(fit: dict) -> str:
    error = fit.get("T1_error")
    new_gate = new_gate_reason(fit)
    if new_gate:
        return new_gate
    if fit["T1"] <= 0:
        return "T1 not positive"
    if _finite(error):
        return f"error {100 * error / fit['T1']:.0f} % of T1 (limit {100 * MAX_RELATIVE_ERROR:.0f} %)"
    return "error uncalculated"


def _minus(text: str) -> str:
    """U+2212 for the minus signs of formatted numbers (the repo convention for figure text)."""
    return text.replace("-", "−")


def _sigma_text(diff: float, error: float) -> str:
    return _minus(f", {diff / error:+.1f} {style.SIGMA}") if _finite(error) and error > 0 else ""


def _in_graph(context: style.FigureContext | None) -> bool:
    return context is not None and context.in_graph


def sequence_blocks(state: dict | None, fit: dict) -> list[style.SeqBlock]:
    """x180, a swept idle, then readout. ``state`` is unused: the schematic needs no device values."""
    span = fit.get(RANGE_KEY)
    swept = f"τ: {_duration(span[0])} to {_duration(span[1])}" if span else "τ swept"
    return [
        style.SeqBlock("drive", "x180", width=1.0),
        style.SeqBlock("idle", "idle τ", swept=swept, width=3.2),
        style.SeqBlock("readout", "readout", width=1.5, line="resonator"),
    ]


def _t1_of(state: dict | None) -> float | None:
    if not state:
        return None
    return state["qubits"][QUBIT].get("T1")


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph runs with a reference only) and state change."""
    ok = _fit_ok(fit)
    if ok:
        params = fit["fit_params"]
        measured = [
            _t1_text(fit),
            _minus(f"A = {params['a'] * SIGNAL_SCALE:.3f} mV, B = {params['offset'] * SIGNAL_SCALE:.3f} mV"),
        ]
        # Sweep coverage is in the plot's chip ("t_max = ... T1"), not repeated here.
    elif _fit_finite(fit):
        measured = ["rejected fit", f"T1 = {_us(fit['T1'])} (not stored)", _rejection_reason(fit)]
    else:
        measured = ["fit failed", "T1 uncalculated"]
    cards = [style.Card("Measured", tuple(measured), "ok" if ok else "fail")]

    # Ruling V: a standalone run draws no true-value card.
    true_t1 = (context.reference or {}).get("T1") if _in_graph(context) else None
    if _finite(true_t1):
        lines = [f"T1 = {_us(true_t1)}"]
        if ok:
            diff = fit["T1"] - true_t1
            lines += [
                _minus(f"fit - true: {diff * TIME_SCALE:+.1f} {style.US}"),
                _minus(f"{100 * diff / true_t1:+.1f} %") + f"{_sigma_text(diff, fit['T1_error'])}",
            ]
        else:
            lines.append("difference uncalculated")
        cards.append(style.Card("True value", tuple(lines)))

    if context is not None:
        change = style.format_change("T1", _t1_of(context.before), _t1_of(context.after), TIME_SCALE, style.US, "{:.1f}")
    else:
        change = f"T1 set to {_us(fit['T1'])}" if ok else "T1 unchanged"
    cards.append(style.Card("State change", (change,)))
    return cards


def _chip_lines(fit: dict, marks: style.Marks | None, context: style.FigureContext | None) -> list[str]:
    """Sweep coverage and, in a graph run, the difference from the true value; failures say so."""
    span = fit.get(RANGE_KEY)
    if not _fit_ok(fit):
        head = ["rejected fit", _rejection_reason(fit)] if _fit_finite(fit) else ["fit failed"]
        return head + ([f"t_max = {_us(span[1], 0)}"] if span else [])
    lines = []
    coverage = _coverage(fit)
    if coverage is not None:
        lines.append(f"t_max = {coverage:.1f} T1 ({_us(span[1], 0)})")
    true = marks.true if marks is not None else None
    if _in_graph(context) and _finite(true):
        diff = fit["T1"] - true
        lines.append(_minus(f"vs true value: {100 * diff / true:+.1f} %") + _sigma_text(diff, fit["T1_error"]))
    return lines


def _edge_markers(ax, marks: style.Marks | None, show_true: bool, theme: str) -> None:
    """Previous and true T1 as triangles on the top edge only.

    A full-height line would hide the 1/e drop line, which sits within a few µs of both.
    The fitted T1 is the drop line itself; the chip and cards carry the numbers.
    """
    if marks is None:
        return
    p = style.palette(theme)
    specs = ((marks.previous, "previous value", p.guess), (marks.true if show_true else None, marks.true_label, p.truth))
    for value, label, colour in specs:
        if not _finite(value):
            continue
        ax.plot(
            [value * TIME_SCALE], [1.0], transform=ax.get_xaxis_transform(), marker="v", markersize=7.0,
            markeredgewidth=0, color=colour, linestyle="none", label=label, clip_on=False, zorder=6,
        )


def _plot_main(ax, res, ds: xr.Dataset, fit: dict, marks, context, theme: str) -> None:
    p = style.palette(theme)
    t_ns = ds.idle_time.values.astype(float)
    x = t_ns * TIME_SCALE
    y = ds.signal.sel(qubit=QUBIT).values * SIGNAL_SCALE
    ax.plot(x, y, "o", color=p.data, markersize=3.4, markeredgewidth=0, label="data", zorder=3)
    ok, drawable = _fit_ok(fit), _fit_finite(fit)
    lo, hi = float(np.min(y)), float(np.max(y))
    if drawable:
        params = fit["fit_params"]
        a, b, decay = params["a"] * SIGNAL_SCALE, params["offset"] * SIGNAL_SCALE, params["decay"]
        if ok:
            lo = min(lo, b)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)  # never clipped at the top: P(e) may read above 1
    span = float(np.ptp(x))
    ax.set_xlim(x.min() - 0.02 * span, x.max() + 0.02 * span)
    t_dense = np.linspace(t_ns.min(), t_ns.max(), 400)
    if ok:
        ax.plot(
            t_dense * TIME_SCALE, decay_exp(t_dense, a, b, decay), color=p.fit, linewidth=1.8,
            label="fit A·exp(−t/T1) + B", zorder=2,
        )
        ax.axhline(b, color=p.muted, linewidth=1.0, linestyle=(0, (4, 3)), label="asymptote B", zorder=1.2)
        x_e, y_e = fit["T1"] * TIME_SCALE, b + a / math.e
        # Drop line from the 1/e point to the time axis; T1 is read off where it lands.
        ax.plot([x_e, x_e], [ax.get_ylim()[0], y_e], color=p.fit, linewidth=1.0, alpha=0.8, zorder=1.4)
        ax.plot(
            [x_e], [y_e], "o", markersize=6.5, markerfacecolor=p.bg, markeredgecolor=p.fit, markeredgewidth=1.4,
            linestyle="none", gid="one_over_e", label="1/e point", zorder=4,
        )
        # Below left of the point: the curve and data run above and to the right of it.
        ax.annotate(
            f"1/e point: B + A/e\nt = T1 = {_us(fit['T1'])}",
            xy=(x_e, y_e), xytext=(-8, -9), textcoords="offset points", color=p.fg, fontsize=8.0,
            ha="right", va="top", linespacing=1.35, zorder=5,
        )
        resid = y - decay_exp(t_ns, a, b, decay)
        sigma = float(np.std(resid))
        style.plot_residuals(res, x, resid / sigma if sigma > 0 else np.zeros_like(resid), theme)
        res.annotate(
            f"{style.SIGMA} = std of residuals = {sigma * 1e3:.1f} µV", xy=(1.0, 1.0), xycoords="axes fraction",
            xytext=(-4, -3), textcoords="offset points", color=p.muted, fontsize=7.0, ha="right", va="top",
        )
    else:
        if drawable:  # finite but rejected: show what was rejected, not an accepted-looking fit
            ax.plot(
                t_dense * TIME_SCALE, decay_exp(t_dense, a, b, decay), color=p.fail, linewidth=1.2,
                linestyle=(0, (4, 3)), label="rejected fit", zorder=2,
            )
        res.set_yticks([])
        res.grid(False)
        res.text(
            0.5, 0.5, "no residuals without an accepted fit", transform=res.transAxes, color=p.muted,
            fontsize=7.5, ha="center", va="center",
        )
    _edge_markers(ax, marks, _in_graph(context), theme)
    ax.set_ylabel("signal, rotated I (mV)")
    res.set_xlabel(f"idle time τ ({style.US})")
    lines = _chip_lines(fit, marks, context)
    if lines:
        style.value_chip(ax, lines, theme, loc="upper right")


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Layout C figure: progress strip, pulse sequence, decay with fit and residuals, cards.

    The fit model is A exp(-t/T1) + B (QM's decay_exp with decay = -1/T1). The asymptote B is
    drawn as a line and the 1/e point sits at B + A/e, t = T1: T1 is the 1/e time measured from
    B, not from zero. With ``context.population_levels`` a right-hand P(e) axis is added; the
    first point can map slightly above 1 (the 07 blob means include decay during readout), so
    the axis is not clipped at 1. The true value is shown only in a graph run
    (``context.in_graph``), as a top-edge triangle, in the chip and in its card.
    """
    with style.theme_context(theme):
        layout = style.node_layout(theme)
        fig = layout.fig
        ax, res = layout.main_axes[0], layout.residual_axes[0]
        ax.set_gid("main")
        _plot_main(ax, res, ds, fit, marks, context, theme)
        if context is not None and context.population_levels is not None:
            level_g, level_e = context.population_levels
            style.population_axis(ax, level_g * SIGNAL_SCALE, level_e * SIGNAL_SCALE, theme)
        style.place_legend(ax, theme)
        if context is not None:
            order, statuses = context.node_order, context.statuses
        else:
            order, statuses = [NODE_NAME], {NODE_NAME: "ok" if fit.get("success") else "fail"}
        style.progress_strip(layout.progress_ax, order, NODE_NAME, statuses, theme)
        state = context.before if context is not None else None
        style.pulse_sequence(layout.sequence_ax, sequence_blocks(state, fit), theme)
        style.cards_column(layout.cards_ax, node_cards(fit, context), theme)
        ok = _fit_ok(fit)
        headline = _t1_text(fit) if ok else ("rejected fit" if _fit_finite(fit) else "fit failed")
        style.title_block(fig, NODE_NAME, "Energy relaxation time", headline, theme, status="ok" if ok else "fail")
    return fig
