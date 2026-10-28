"""Analysis for 06b_echo, using QM's own fit routine from qualibration-libs."""

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
from simulator.device import Device

log = logging.getLogger(__name__)

NODE_NAME = "06b_echo"
MAX_RELATIVE_ERROR = 0.2  # fit_raw_data rejects T2echo_error / T2echo at or above this


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    # The sequence ends with -x90, so a fully refocused qubit ends in |e> (net rotation x180):
    # the shortest idle time is the most excited point and the signal decays towards the
    # 50/50 mixture as coherence is lost. Points off the g-e line are flagged (``outlier``) and
    # left out of the projection axis.
    outlier = iq_line_outliers((ds.I + 1j * ds.Q).sel(qubit=QUBIT).values)
    ds = ds.assign(outlier=("idle_time", outlier))
    return ds.assign(signal=project_on_iq_axis(ds, "idle_time", first_point_excited=True, exclude=outlier))


def fit_raw_data(ds: xr.Dataset) -> dict:
    """Fit a*exp(decay*t) + offset over the total idle time. QM's decay_exp convention:
    decay < 0, T2echo = -1/decay. Outliers are masked around QM's fitter (``fit_with_masking``);
    gates from ``decay_fit_gates``: converged, relative_error, sampled, outliers."""

    def reproject(mask):
        return project_on_iq_axis(ds, "idle_time", first_point_excited=True, exclude=mask)

    raw, mask, signal = fit_with_masking(ds, "idle_time", reproject, lambda d: fit_decay_exp(d, "idle_time"), decay_residual)
    if mask.any():
        ds["signal"], ds["outlier"] = signal, ("idle_time", mask)
    fit = raw.sel(qubit=QUBIT)
    decay = float(fit.sel(fit_vals="decay"))
    decay_var = float(fit.sel(fit_vals="decay_decay"))
    T2echo = -1.0 / decay if decay < 0 else float("nan")
    T2echo_err = float(np.sqrt(abs(decay_var)) / decay**2) if decay < 0 else float("nan")
    gates, n_sampled = decay_fit_gates(T2echo, T2echo_err, ds.idle_time.values, mask, MAX_RELATIVE_ERROR)
    return {
        "T2echo": T2echo,
        "T2echo_error": T2echo_err,
        "fit_params": {k: float(fit.sel(fit_vals=k)) for k in ("a", "offset", "decay")},
        "success": all(gates.values()),
        # Total idle time swept, in ns, for the pulse diagram (Ruling Y).
        sweep_range_key("idle_time"): (float(ds.idle_time.min()), float(ds.idle_time.max())),
        "n_sampled": n_sampled,
        **outlier_summary(mask),
        "gates": gates,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info("06b_echo %s: T2echo = %.1f +/- %.1f us", status, fit["T2echo"] / 1e3, fit["T2echo_error"] / 1e3)


# ---------------------------------------------------------------- figure (Layout C)
US = style.US


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _qubit(state: dict | None) -> dict:
    return ((state or {}).get("qubits") or {}).get(QUBIT) or {}


def _measured(context: style.FigureContext | None, key: str) -> float | None:
    """A coherence time measured by an earlier node (06a T2ramsey, 05 T1), read from the state
    before this node ran. Never from the reference: a standalone run without them gets None."""
    if context is None:
        return None
    value = _qubit(context.before).get(key)
    return float(value) if _finite(value) and value > 0 else None


def _curve_ok(fit: dict) -> bool:
    """The fit has finite parameters, so its curve can be drawn (accepted or rejected)."""
    params = fit.get("fit_params") or {}
    return bool(
        _finite(fit.get("T2echo"))
        and fit["T2echo"] > 0
        and all(_finite(params.get(k)) for k in ("a", "offset", "decay"))
        and params["a"] != 0
    )


def _fit_ok(fit: dict) -> bool:
    return bool(fit.get("success")) and _curve_ok(fit)


def _us(ns: float, digits: int = 1) -> str:
    return f"{ns / 1e3:.{digits}f}{style.NBSP}{US}"


def _value_err(fit: dict) -> str:
    """Value and error, "59.5 ± 2.4 µs (stat.)", or just the value when the error is not finite."""
    err = fit.get("T2echo_error")
    if _finite(err):
        return f"{fit['T2echo'] / 1e3:.1f} ± {_us(err)} (stat.)"
    return _us(fit["T2echo"])


def _range_us(lo_ns: float, hi_ns: float) -> str:
    def show(ns: float) -> str:
        us = ns / 1e3
        return f"{us:.2g}" if us < 1 else f"{us:.0f}"

    return f"{show(lo_ns)} to {show(hi_ns)}{style.NBSP}{US}"


def sequence_blocks(state: dict, fit: dict) -> list[style.SeqBlock]:
    """Hahn echo: x90, t/2, x180, t/2, -x90, readout. The two idles have equal widths, so the x180
    sits at the exact midpoint of the free evolution. ``fit["idle_time_range"]`` is the swept
    total idle time t in ns (Ruling Y); each half idle spans half of it."""
    sweep = fit.get(sweep_range_key("idle_time"))
    swept = _range_us(sweep[0] / 2, sweep[1] / 2) if sweep else "swept"
    return [
        style.SeqBlock("drive", "x90", width=0.75),
        style.SeqBlock("idle", "t/2", swept=swept, width=2.3),
        style.SeqBlock("drive", "x180", width=0.75),
        style.SeqBlock("idle", "t/2", swept=swept, width=2.3),
        style.SeqBlock("drive", "-x90", width=0.75),
        style.SeqBlock("readout", "readout", width=1.4, line="resonator"),
    ]


def _ratios(fit: dict, context: style.FigureContext | None) -> list[str]:
    """T2E / T2* (06a) and T2E / 2T1 (05), from the measured values; "uncalculated" without them."""
    t2e = fit["T2echo"] if _fit_ok(fit) else None
    t2s, t1 = _measured(context, "T2ramsey"), _measured(context, "T1")
    if not t2e:
        return ["T2E / T2*: uncalculated", "T2E / 2T1: uncalculated"]
    star = f"T2E / T2* = {t2e / t2s:.2f} (T2* {_us(t2s)}, 06a)" if t2s else "T2E / T2*: uncalculated (no 06a T2*)"
    ceil = f"T2E / 2T1 = {t2e / (2 * t1):.2f} (T1 {_us(t1)}, 05)" if t1 else "T2E / 2T1: uncalculated (no 05 T1)"
    return [star, ceil]


def _caveat_lines(context: style.FigureContext | None) -> tuple[str, str]:
    """Two lines: the echo gain is emergent (the x180 refocuses the simulator's quasi-static
    noise), and the white dephasing it cannot refocus makes the decay exponential by construction.

    With T1, T2*, T2echo in the reference (plotting may read it), the noise size and the white
    dephasing time come from ``Device`` (simulator/device.py)."""
    ref = (context.reference if context is not None else None) or {}
    if all(_finite(ref.get(k)) for k in ("T1", "T2ramsey", "T2echo")):
        d = Device(T1=ref["T1"], T2ramsey=ref["T2ramsey"], T2echo=ref["T2echo"])
        return (
            f"Emergent: the x180 refocuses the {d.sigma_f_quasistatic / 1e3:.1f}{style.NBSP}kHz rms quasi-static noise",
            f"White dephasing (Tphi {_us(d.Tphi_white, 0)}) is not refocused: exponential by construction",
        )
    return (
        "Emergent: the x180 refocuses the simulator's quasi-static noise",
        "White dephasing is not refocused: exponential by construction",
    )


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph run with a reference), state change, and the echo caveat
    (the gain over T2* is emergent; the remaining white dephasing is exponential by construction)."""
    ok = _fit_ok(fit)
    if ok:
        measured = [f"T2echo (T2E) = {_value_err(fit)}"]
    elif _curve_ok(fit):
        measured = [f"rejected fit: T2echo {_value_err(fit)}"]
        if new_gate_reason(fit):
            measured.append(new_gate_reason(fit))
    else:
        measured = ["T2echo: fit failed"]
    cards = [style.Card("Measured", tuple(measured + _ratios(fit, context)), "ok" if ok else "fail")]
    ref = (context.reference if context is not None else None) or {}
    if context is not None and context.in_graph and _finite(ref.get("T2echo")):
        true = ref["T2echo"]
        lines = [f"T2echo = {_us(true)}"]
        err = fit.get("T2echo_error")
        if ok:
            diff = fit["T2echo"] - true
            sigma = f", {diff / err:+.1f} {style.SIGMA}" if _finite(err) and err > 0 else ""
            # U+2212 for negative numbers, the repo convention for figure text.
            lines.append(
                f"fit − true: {diff / 1e3:+.1f}{style.NBSP}{US} ({100 * diff / true:+.1f} %{sigma})".replace("-", "−")
            )
        cards.append(style.Card("True value", tuple(lines)))
    if context is not None:
        before = _qubit(context.before).get("T2echo")
        after = _qubit(context.after).get("T2echo")
        cards.append(style.Card("State change", (style.format_change("T2echo", before, after, 1e-3, US, "{:.1f}"),)))
    cards.append(style.Card("Caveat", _caveat_lines(context), "caveat"))
    return cards


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Echo decay on a normalised coherence axis, (s - B) / A with the fitted A and B, between
    the T2* envelope exp(-t/T2*) (06a fit, floor) and the T1 limit exp(-t/2T1) (05 fit, ceiling).

    T2* and T1 are the measured values in the state before this node ran (``context.before``);
    without them both curves are left out and the ratios read "uncalculated". With
    ``context.population_levels`` a right-hand P(e) axis is added. A failed or rejected fit falls
    back to the projected signal in mV; a rejected fit with finite parameters is drawn faded and
    labelled "rejected fit".
    """
    p = style.palette(theme)
    marks = marks or style.Marks()
    lay = style.node_layout(theme)
    fig, ax, rax = lay.fig, lay.main_axes[0], lay.residual_axes[0]

    t_ns = ds.idle_time.values.astype(float)
    t_us = t_ns / 1e3
    signal = ds.signal.sel(qubit=QUBIT).values
    ok = _fit_ok(fit)
    rejected = not ok and _curve_ok(fit)
    params = fit.get("fit_params") or {}
    t2e = fit.get("T2echo")
    t2s, t1 = (_measured(context, "T2ramsey"), _measured(context, "T1")) if ok else (None, None)
    dense_ns = np.linspace(0.0, t_ns.max(), 400)
    dense_us = dense_ns / 1e3

    if ok:
        a, b = params["a"], params["offset"]
        y = (signal - b) / a
        model = decay_exp(t_ns, a, b, params["decay"])
        if t2s and t1:
            ax.fill_between(
                dense_us,
                np.exp(-dense_ns / t2s),
                np.exp(-dense_ns / (2 * t1)),
                color=style.mix(p.muted, p.bg, 0.1),
                linewidth=0,
                zorder=0.5,
            )
        if t2s:
            ax.plot(dense_us, np.exp(-dense_ns / t2s), color=p.muted, linestyle=(0, (4, 2.5)),
                    linewidth=1.2, label="T2* floor (06a)", zorder=1.6)
        if t1:
            ax.plot(dense_us, np.exp(-dense_ns / (2 * t1)), color=p.muted, linestyle=(0, (1, 2)),
                    linewidth=1.5, label="2T1 ceiling (05)", zorder=1.6)
        ax.plot(t_us, y, "o", color=p.data, markersize=3.2, markeredgewidth=0, label="data", zorder=2)
        ax.plot(dense_us, np.exp(-dense_ns / t2e), color=p.fit, label="fit", zorder=3)
        ax.plot([t2e / 1e3], [math.exp(-1)], "o", color=p.fit, markersize=5, markeredgewidth=0, zorder=4)
        ax.annotate("1/e", xy=(t2e / 1e3, math.exp(-1)), xytext=(-5, -5), textcoords="offset points",
                    color=p.fit, fontsize=8.0, ha="right", va="top", zorder=4)
        ax.set_ylabel("echo coherence, (s − B) / A")
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        ax.set_ylim(min(-0.12, lo - 0.05), max(1.12, hi + 0.05))
        resid = signal - model
        sigma = float(np.std(resid))
        style.plot_residuals(rax, t_us, resid / sigma if sigma > 0 else np.full_like(resid, np.nan), theme)
        rax.set_ylabel(f"resid. / {style.SIGMA}\n({style.SIGMA}: resid. std)", fontsize=7.5, linespacing=1.2)
        level_map = (b, a)  # signal s -> (s - B) / A
    else:
        y = signal * 1e3
        ax.plot(t_us, y, "o", color=p.data, markersize=3.2, markeredgewidth=0, label="data", zorder=2)
        if rejected:
            curve = decay_exp(dense_ns, params["a"], params["offset"], params["decay"]) * 1e3
            ax.plot(dense_us, curve, color=p.fail, linestyle=(0, (4, 2.5)), linewidth=1.2, alpha=0.8,
                    label="rejected fit", zorder=3)
        ax.set_ylabel("signal, rotated I (mV)")
        rax.text(0.5, 0.5, f"no residuals ({'rejected' if rejected else 'failed'} fit)", transform=rax.transAxes,
                 color=p.muted, fontsize=7.5, ha="center", va="center")
        rax.set_yticks([])
        rax.grid(False)
        level_map = (0.0, 1e-3)  # signal s (V) -> mV
    # Marks.fit is None for a fit that did not succeed, so no fitted-value line is drawn then.
    style.mark_values(ax, marks, theme, scale=1e-3)
    rax.set_xlabel(f"total idle time t ({US})")
    ax.set_xlim(0, t_us.max() * 1.02)

    if ok:
        chip = [f"T2E = {_value_err(fit)}"] + [line.split(" (")[0] for line in _ratios(fit, context)]
        if context is not None and context.in_graph and _finite(marks.true):  # ruling V
            diff, err = t2e - marks.true, fit["T2echo_error"]
            sig = f", {diff / err:+.1f} {style.SIGMA}" if _finite(err) and err > 0 else ""
            chip.append(f"vs {marks.true_label}: " + f"{100 * diff / marks.true:+.1f} %{sig}".replace("-", "−"))
    elif rejected:
        chip = [f"rejected fit: T2E {_value_err(fit)}", "no T2* or 2T1 bounds"]
    else:
        chip = ["fit failed: no T2* or 2T1 bounds"]
    style.value_chip(ax, chip, theme)

    if context is not None and context.population_levels is not None:
        offset, scale = level_map
        level_g, level_e = ((level - offset) / scale for level in context.population_levels)
        style.population_axis(ax, level_g, level_e, theme)

    n_entries = len(ax.get_legend_handles_labels()[0])
    style.place_legend(ax, theme, ncols=math.ceil(n_entries / 2) if n_entries > 4 else None, title="Echo decay")
    if context is None:
        order, statuses = [NODE_NAME], {NODE_NAME: "ok" if fit.get("success") else "fail"}
    else:
        order, statuses = context.node_order, context.statuses
    style.progress_strip(lay.progress_ax, order, NODE_NAME, statuses, theme)
    state = context.before if context is not None else {}
    style.pulse_sequence(lay.sequence_ax, sequence_blocks(state, fit), theme)
    style.cards_column(lay.cards_ax, node_cards(fit, context), theme)
    if ok:
        headline = f"T2echo = {_value_err(fit).replace(' (stat.)', '')}"
    else:
        headline = "rejected fit" if rejected else "fit failed"
        if rejected and new_gate_reason(fit):
            headline = f"rejected fit: {new_gate_reason(fit)}"
    style.title_block(fig, NODE_NAME, "Hahn echo coherence", headline, theme, status="ok" if ok else "fail")
    return fig
