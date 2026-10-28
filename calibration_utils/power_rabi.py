"""Analysis for 04b_power_rabi, using QM's own fit routine from qualibration-libs."""

from __future__ import annotations

import logging
import math

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.analysis import fit_oscillation
from qualibration_libs.analysis.models import oscillation
from scipy.optimize import curve_fit

from calibration_utils import style
from calibration_utils.common import (
    MAX_OUTLIER_FRACTION,
    QUBIT,
    fit_with_masking,
    iq_projection,
    outlier_mask_of,
    outlier_summary,
    project_on_iq_axis,
    sweep_range_key,
)
from calibration_utils.robust import iq_line_outliers

log = logging.getLogger(__name__)

NODE_NAME = "04b_power_rabi"

# Quality gates. fit_oscillation returns some sinusoid even for pure noise, so a pi amplitude is
# only accepted when the oscillation clearly stands out of the scatter and the fitted phase agrees
# with the physics (|g> at zero amplitude, so the signal starts near its minimum).
MIN_OSC_AMP_SNR = 3.0  # fitted amplitude / robust residual scatter
MAX_PHASE_ERROR = np.pi / 4  # allowed |phi - pi|, i.e. pi point within 1/8 period of 1/(2f)
# Minimum number of Rabi periods the sweep must span; QM's 04b analysis uses the same value
# ("below this the frequency is unconstrained"):
# https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibration_utils/power_rabi/analysis.py
MIN_N_PERIODS = 0.75


def process_raw_dataset(ds: xr.Dataset, current_amplitude: float) -> xr.Dataset:
    """Add the projected signal and the absolute pulse amplitude in volts (QM's ``full_amp``).
    Points off the g-e line are flagged (``outlier``) and left out of the projection axis."""
    outlier = iq_line_outliers((ds.I + 1j * ds.Q).sel(qubit=QUBIT).values)
    ds = ds.assign(outlier=("amp_prefactor", outlier))
    ds = ds.assign(signal=project_on_iq_axis(ds, "amp_prefactor", first_point_excited=False, exclude=outlier))
    full_amp = xr.DataArray(ds.amp_prefactor.values * current_amplitude, dims="amp_prefactor")
    return ds.assign_coords(full_amp=full_amp.assign_attrs(long_name="pulse amplitude", units="V"))


def _prefactor_error(x: np.ndarray, y: np.ndarray, params: tuple[float, float, float, float], x_pi: float) -> float:
    """1 sigma statistical error of x_pi from the covariance of a least-squares refit.

    fit_oscillation returns no parameter errors, so the same model is refitted from its values
    with scipy's curve_fit, and the (f, phi) covariance is propagated through
    x_pi = ((-phi) mod 2 pi) / (2 pi f): dx/df = -x_pi / f, dx/dphi = -1 / (2 pi f).
    """
    if not (all(math.isfinite(v) for v in params) and math.isfinite(x_pi)):
        return float("nan")
    try:
        popt, pcov = curve_fit(oscillation, x, y, p0=list(params))
    except (RuntimeError, ValueError):
        return float("nan")
    a, f, phi = popt[0], popt[1], popt[2]
    if not (math.isfinite(f) and f > 0 and math.isfinite(phi)):
        return float("nan")
    if a < 0:  # a cos(u) = (-a) cos(u + pi): the same curve, with a >= 0 as in QM's model
        phi += np.pi
    # Jacobian at the refit's own x_pi; reject a refit that moved to another solution.
    x_refit = float(np.mod(-phi, 2 * np.pi) / (2 * np.pi * f))
    if abs(x_refit - x_pi) > 0.01 / f:  # more than 1 % of a period from the QM fit
        return float("nan")
    jac = np.array([0.0, -x_refit / f, -1.0 / (2 * np.pi * f), 0.0])
    var = float(jac @ pcov @ jac)
    return math.sqrt(var) if var >= 0 and math.isfinite(var) else float("nan")


def fit_raw_data(ds: xr.Dataset, current_amplitude: float) -> dict:
    """Fit a*cos(2*pi*f*x + phi) + offset over the amplitude prefactor x (QM's oscillation model, a >= 0).

    The signal grows with excited population, so the pi pulse is the first maximum of the cosine
    at positive x: 2*pi*f*x + phi = 2*pi*k  ->  x_pi = ((-phi) mod 2*pi) / (2*pi*f). For a clean
    Rabi curve starting in |g>, phi = pi and x_pi = 1/(2f), the half period QM's node reports.
    Keeping the phase means a small offset in the drive (or a thermal start) shifts the answer
    the way the data does, and a phase far from pi flags a fit that is not a Rabi oscillation.

    ``opt_amp_error`` (V) is statistical only (see ``_prefactor_error``); on seed scans the fit
    sits systematically above the true pi amplitude by a few tenths of a percent.
    ``amp_prefactor_range`` is the swept prefactor range (Ruling Y).
    """
    def reproject(mask):
        return project_on_iq_axis(ds, "amp_prefactor", first_point_excited=False, exclude=mask)

    def residual_of(raw, data):
        params = [float(raw.sel(qubit=QUBIT, fit_vals=k)) for k in ("a", "f", "phi", "offset")]
        if not all(np.isfinite(params)):
            return None
        return data.sel(qubit=QUBIT).values - oscillation(data.amp_prefactor.values, *params)

    raw, mask, signal = fit_with_masking(ds, "amp_prefactor", reproject, lambda d: fit_oscillation(d, "amp_prefactor"), residual_of)
    if mask.any():
        ds["signal"], ds["outlier"] = signal, ("amp_prefactor", mask)
    x_all = ds.amp_prefactor.values
    x = x_all[~mask]
    y = ds.signal.sel(qubit=QUBIT).values[~mask]
    fit = raw.sel(qubit=QUBIT)
    a, f, phi, offset = (float(fit.sel(fit_vals=k)) for k in ("a", "f", "phi", "offset"))

    if np.isfinite(f) and f > 0 and np.isfinite(phi):
        prefactor = float(np.mod(-phi, 2 * np.pi) / (2 * np.pi * f))
        phase_error = float(abs(np.angle(np.exp(1j * (phi - np.pi)))))
    else:
        prefactor, phase_error = float("nan"), float("nan")

    residual = y - oscillation(x, a, f, phi, offset)
    scatter = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-15
    snr = abs(a) / scatter

    n_periods = float(f * (x_all.max() - x_all.min()))
    gates = {
        "in_range": bool(np.isfinite(prefactor) and x_all.min() <= prefactor <= x_all.max()),
        "snr": bool(snr >= MIN_OSC_AMP_SNR),
        "phase": bool(phase_error <= MAX_PHASE_ERROR),
        "n_periods": bool(np.isfinite(n_periods) and n_periods >= MIN_N_PERIODS),
        "outliers": bool(mask.mean() <= MAX_OUTLIER_FRACTION),
    }
    success = all(gates.values())
    return {
        "opt_amp_prefactor": prefactor,
        "opt_amp": prefactor * current_amplitude,
        "opt_amp_error": _prefactor_error(x, y, (a, f, phi, offset), prefactor) * current_amplitude,
        "half_period_prefactor": 1 / (2 * f) if f > 0 else float("nan"),
        "osc_amp_snr": snr,
        "phase_error": phase_error,
        "n_periods": n_periods,
        "gates": gates,
        "fit_params": {"a": a, "f": f, "phi": phi, "offset": offset},
        sweep_range_key("amp_prefactor"): (float(x_all.min()), float(x_all.max())),
        **outlier_summary(mask),
        "success": success,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "04b_power_rabi %s: x180 amplitude = %.2f +/- %.2f mV statistical (x%.3f), SNR %.1f, phase error %.2f rad, %.2f periods%s",
        status,
        fit["opt_amp"] * 1e3,
        fit["opt_amp_error"] * 1e3,
        prefactor_vs_state(fit),
        fit["osc_amp_snr"],
        fit["phase_error"],
        fit["n_periods"],
        _refine_note(fit.get("refined_from")),
    )


def prefactor_vs_state(fit: dict) -> float:
    """The pi amplitude as a multiple of the x180 amplitude in the state before the node ran.

    After a refine pass, ``opt_amp_prefactor`` is relative to the provisional base of the second
    sweep; the base the node started from is recovered from the first-pass fit.
    """
    first = fit.get("refined_from")
    if first is None:
        return fit["opt_amp_prefactor"]
    return fit["opt_amp"] * first["opt_amp_prefactor"] / first["opt_amp"]


def _refine_note(first: dict | None) -> str:
    if first is None:
        return ""
    return ", refined from %.2f mV (%.2f periods)" % (first["opt_amp"] * 1e3, first["n_periods"])


# ---------------------------------------------------------------- figure (Layout C)
# Room (inches) above the main axes for the top prefactor axis; place_legend lifts the legend row by it.
TOP_EXTRA = 0.42
PM = "±"


def _minus(text: str) -> str:
    """U+2212 for the minus signs of formatted numbers (the repo convention for figure text)."""
    return text.replace("-", "−")


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _x180(state: dict | None) -> float | None:
    try:
        return float(state["qubits"][QUBIT]["xy"]["operations"]["x180"]["amplitude"])
    except (TypeError, KeyError):
        return None


def _sweep_base(ds: xr.Dataset) -> float:
    """The amplitude (V) that the sweep in ``ds`` multiplies by its prefactor."""
    return float(ds.full_amp.values[-1] / ds.amp_prefactor.values[-1])


def _stored_amplitude(ds: xr.Dataset, fit: dict) -> float:
    """The x180 amplitude (V) stored before the node ran: the base of the first sweep."""
    first = fit.get("refined_from")
    if first is not None and _finite(first.get("opt_amp")) and first.get("opt_amp_prefactor"):
        return float(first["opt_amp"] / first["opt_amp_prefactor"])
    return _sweep_base(ds)


def _with_x180(amplitude: float) -> dict:
    return {"qubits": {QUBIT: {"xy": {"operations": {"x180": {"amplitude": amplitude}}}}}}


def reference_levels(ds: xr.Dataset, state: dict) -> tuple[float, float]:
    """Signal levels (V) of a prepared |g> and |e> on the IQ axis of ``ds``: simulator reference.

    ``simulator.reference.readout_levels`` (pure states, no noise) at the state's readout
    settings, projected the way ``process_raw_dataset`` projects the data. Plotting only.
    """
    from simulator.reference import readout_levels

    _, axis, centre = iq_projection(ds, "amp_prefactor", first_point_excited=False, exclude=outlier_mask_of(ds))
    level_g, level_e = (float(((z - centre) * np.conj(axis)).real) for z in readout_levels(state))
    return level_g, level_e


def _raw_peak_prefactor(x: np.ndarray, y: np.ndarray, f: float) -> float:
    """QM's fit-free pi estimate: the point in the first Rabi period farthest from the first sample.

    Mirrors ``_raw_pi_prefactor`` in QM's power_rabi analysis.py:
    https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibration_utils/power_rabi/analysis.py
    """
    if not (_finite(f) and f > 0):
        return float("nan")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    window = xs <= xs[0] + 1.0 / abs(f)
    xw, yw = xs[window], ys[window]
    return float(xw[int(np.nanargmax(np.abs(yw - yw[0])))])


def sequence_blocks(state: dict, fit: dict) -> list[style.SeqBlock]:
    """Pulse diagram: one x180 scaled by the swept prefactor, readout, then thermalize.

    The swept range comes from ``fit["amp_prefactor_range"]`` (Ruling Y). After a refine pass
    the second sweep multiplies the provisional pi amplitude from the first pass, not the stored
    x180 amplitude, and the swept text says so.
    """
    op = state["qubits"][QUBIT]["xy"]["operations"]["x180"]
    lo, hi = fit.get(sweep_range_key("amp_prefactor"), (float("nan"), float("nan")))
    base = "provisional pi amp" if fit.get("refined_from") is not None else "x180 amp"
    swept = f"{lo:.2f} to {hi:.2f} × {base}" if _finite(lo) and _finite(hi) else f"swept × {base}"
    length = op.get("length")
    label = f"x180 × a, {length:.0f} ns" if _finite(length) else "x180 × a"
    return [
        style.SeqBlock("drive", label, swept=swept, width=1.7),
        style.SeqBlock("readout", "readout", width=1.4, line="resonator"),
        style.SeqBlock("wait", "thermalize", width=1.3),
    ]


def _gate_line(name: str, value: float, limit: str, ok: bool, fmt: str) -> str:
    shown = fmt.format(value) if _finite(value) else "uncalculated"
    return f"{name} {shown} ({limit}): {'pass' if ok else 'fail'}"


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph runs with a reference), state change and fit gates.

    ``opt_amp_error`` is statistical only: on seed scans the fit sits a few tenths of a percent
    above the true pi amplitude, more than this error explains.
    """
    ok = bool(fit["success"])
    opt, err = fit["opt_amp"], fit.get("opt_amp_error")
    cards: list[style.Card] = []

    if _finite(opt):
        line = f"x180 amplitude {opt * 1e3:.2f}"
        line += f" {PM} {err * 1e3:.2f} mV (stat.)" if _finite(err) else " mV"
        measured = [line if ok else f"rejected fit: {opt * 1e3:.2f} mV"]
        factor = prefactor_vs_state(fit)
        if _finite(factor):
            measured.append(f"{factor:.3f} × the amplitude stored before the node ran")
        if not ok:
            measured.append("gates failed: state not updated")
    else:
        measured = ["fit failed: no pi amplitude"]
    cards.append(style.Card("Measured", tuple(measured), "ok" if ok else "fail"))

    reference = context.reference if context is not None else None
    if context is not None and context.in_graph and reference is not None and _finite(reference.get("x180.amplitude")):
        true = reference["x180.amplitude"]
        lines = [f"x180 amplitude {true * 1e3:.2f} mV"]
        if _finite(opt):
            diff = opt - true
            lines.append(_minus(f"{'fit' if ok else 'rejected fit'} - true {diff / true * 100:+.2f} %"))
            line = f"{diff * 1e3:+.2f} mV"
            if _finite(err) and err > 0:
                line += f", {diff / err:+.1f} {style.SIGMA} (stat.)"
            lines.append(_minus(line))
        cards.append(style.Card("True value", tuple(lines), "info"))

    before = _x180(context.before) if context is not None else None
    after = _x180(context.after) if context is not None else None
    change = [style.format_change("x180 amplitude", before, after, 1e3, "mV", "{:.2f}")]
    if _finite(before) and _finite(after) and before:
        change.append("unchanged" if before == after else _minus(f"correction {(after / before - 1) * 100:+.1f} %"))
    cards.append(style.Card("State change", tuple(change), "info"))

    gates = fit["gates"]
    lines = [
        _gate_line("periods", fit["n_periods"], f"min {MIN_N_PERIODS}", gates["n_periods"], "{:.2f}"),
        _gate_line("oscillation SNR", fit["osc_amp_snr"], f"min {MIN_OSC_AMP_SNR:.0f}", gates["snr"], "{:.1f}"),
        _gate_line("phase error", fit["phase_error"], f"max {MAX_PHASE_ERROR:.2f}", gates["phase"], "{:.2f} rad"),
        f"pi inside the sweep: {'pass' if gates['in_range'] else 'fail'}",
    ]
    if not gates.get("outliers", True):  # only a failed outlier gate adds a line; clean figures stay as they were
        lines.append(
            f"outliers {100 * fit['outlier_fraction']:.0f} % of points off the g-e line "
            f"(max {100 * MAX_OUTLIER_FRACTION:.0f} %): fail"
        )
    first = fit.get("refined_from")
    if first is not None:
        lines.append(
            f"first pass {first['n_periods']:.2f} periods, under {MIN_N_PERIODS}: "
            f"swept again around {first['opt_amp'] * 1e3:.1f} mV"
        )
    cards.append(style.Card("Fit gates", tuple(lines), "ok" if ok else "fail"))
    return cards


def _standalone_context(ds: xr.Dataset, fit: dict) -> style.FigureContext:
    """The context of a bare run: this node alone, state change from the stored amplitude."""
    stored = _stored_amplitude(ds, fit)
    after = fit["opt_amp"] if fit["success"] else stored
    return style.FigureContext(
        node_order=[NODE_NAME],
        statuses={NODE_NAME: "ok" if fit["success"] else "fail"},
        before=_with_x180(stored),
        after=_with_x180(after),
        reference=None,
        population_levels=None,
        in_graph=False,
    )


def _project(dataset: xr.Dataset, axis: complex, centre: complex) -> np.ndarray:
    z = (dataset.I + 1j * dataset.Q).sel(qubit=QUBIT).values
    return ((z - centre) * np.conj(axis)).real


def _model(x, params: dict):
    return oscillation(x, params["a"], params["f"], params["phi"], params["offset"])


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
    first_pass: xr.Dataset | None = None,
) -> Figure:
    """Layout C figure: the Rabi swing over pulse amplitude in mV, top axis in prefactor.

    The top axis is the prefactor of the x180 amplitude stored before the node ran, so it reads
    the same as ``prefactor_vs_state`` in both passes. ``first_pass`` is
    ``ctx.results[NODE]["first_pass_dataset"]``: when a refine pass ran (``fit["refined_from"]``)
    it is drawn in the second-series colours with its fit, projected on the final pass's IQ axis
    so both share one y scale. Without the dataset only the cards mention the first pass.
    Prepared |g> and |e> levels (``reference_levels``) and the true value are drawn only in a
    graph run (``context.in_graph``) with ``context.reference`` set (ruling V). The true value is a
    top-edge triangle only: it sits within about a millivolt of the fit, and a full-height line
    would hide the fitted-value line. A fit with ``success`` False is drawn as a "rejected fit" when its values are finite.
    Call inside ``style.theme_context(theme)``.
    """
    p = style.palette(theme)
    marks = marks or style.Marks()
    lay = style.node_layout(theme, ncols=1, residuals=True, cards=True, top_extra=TOP_EXTRA)
    fig, ax, rax = lay.fig, lay.main_axes[0], lay.residual_axes[0]
    ok = bool(fit["success"])
    params = fit["fit_params"]
    fitted = all(_finite(params[k]) for k in params)
    stored = _stored_amplitude(ds, fit)
    first = fit.get("refined_from")
    card_context = context if context is not None else _standalone_context(ds, fit)

    # Progress strip, pulse sequence, cards.
    style.progress_strip(lay.progress_ax, card_context.node_order, NODE_NAME, card_context.statuses, theme)
    before = context.before if context is not None and context.before is not None else None
    if before is None:  # bare run: rebuild what the diagram needs
        before = {"qubits": {QUBIT: {"xy": {"operations": {"x180": {"amplitude": stored}}}}}}
    style.pulse_sequence(lay.sequence_ax, sequence_blocks(before, fit), theme)
    style.cards_column(lay.cards_ax, node_cards(fit, card_context), theme)

    # Data, both passes.
    x = ds.amp_prefactor.values
    amp_mv = ds.full_amp.values * 1e3
    y = ds.signal.sel(qubit=QUBIT).values * 1e3
    _, axis, centre = iq_projection(ds, "amp_prefactor", first_point_excited=False, exclude=outlier_mask_of(ds))
    if first is not None and first_pass is not None:
        projected = _project(first_pass, axis, centre)
        ax.plot(first_pass.full_amp.values * 1e3, projected * 1e3, "o", mfc="none", mec=p.data2, mew=0.8,
                ms=3.2, alpha=0.85, zorder=1.8, gid="first pass data",
                label="first pass")  # its periods are in the fit-gates card
        fp = first["fit_params"]
        if all(_finite(fp[k]) for k in fp):
            # The first fit used that pass's own projection; map it onto this one (both are
            # projections of nearly collinear I/Q points, so the map is linear).
            own = first_pass.signal.sel(qubit=QUBIT).values
            slope, intercept = np.polyfit(own, projected, 1)
            px = first_pass.amp_prefactor.values
            xs = np.linspace(px.min(), px.max(), 400)
            ax.plot(xs * _sweep_base(first_pass) * 1e3, (slope * _model(xs, fp) + intercept) * 1e3,
                    color=p.fit2, lw=1.2, ls=(0, (4, 2)), zorder=1.9, gid="first pass fit")
    ax.plot(amp_mv, y, "o", color=p.data, ms=3.0, mew=0, zorder=2.5, gid="data",
            label="refine pass" if first is not None else "data")
    if fitted:
        xd = np.linspace(x.min(), x.max(), 500)
        ax.plot(xd * _sweep_base(ds) * 1e3, _model(xd, params) * 1e3, color=p.fit if ok else p.fail,
                lw=1.8, ls="-" if ok else (0, (5, 2)), zorder=3, gid="fit", label="fit" if ok else "rejected fit")
        raw = _raw_peak_prefactor(x, y, params["f"])
        if _finite(raw):
            i = int(np.argmin(np.abs(x - raw)))
            ax.plot([amp_mv[i]], [y[i]], "D", ms=6.5, mfc="none", mec=p.excited, mew=1.2, zorder=4,
                    label="raw peak (QM)")

    # Simulator-reference levels (graph runs with a reference only, ruling V).
    in_graph = context is not None and context.in_graph
    if in_graph and context.reference is not None and context.after is not None:
        level_g, level_e = (v * 1e3 for v in reference_levels(ds, context.after))
        # Labels where the data is far from each level and clear of the previous-value and pi
        # lines near the middle: |e> under its line at the high-amplitude end, |g> above its
        # line just right of the middle.
        for level, ket, x_frac, ha, above in (
            (level_g, style.KET_G, 0.53, "left", True),
            (level_e, style.KET_E, 0.99, "right", False),
        ):
            ax.axhline(level, color=p.reference, lw=1.0, ls=(0, (6, 3)), zorder=1.2)
            ax.annotate(f"prepared {ket}, simulator reference", xy=(x_frac, level),
                        xycoords=("axes fraction", "data"), xytext=(0, 3 if above else -3),
                        textcoords="offset points", ha=ha, va="bottom" if above else "top",
                        fontsize=7.0, color=p.muted, zorder=5)
    # Zero-amplitude end: a thermal start, not a prepared |g>. Text sits in the empty upper left.
    ax.annotate(f"about {style.KET_G}\n(thermal start)", xy=(amp_mv[0], y[:3].mean()), xycoords="data",
                xytext=(0.03, 0.42), textcoords="axes fraction", fontsize=7.0, color=p.muted,
                ha="left", va="bottom", linespacing=1.3,
                arrowprops={"arrowstyle": "-", "color": p.muted, "lw": 0.6, "shrinkA": 2, "shrinkB": 3})

    # Axes, marks, top prefactor axis, legend, residuals.
    hi = amp_mv.max()
    if first is not None and first_pass is not None:
        hi = max(hi, first_pass.full_amp.values.max() * 1e3)
    ax.set_xlim(0.0, hi * 1.02)
    # An accepted fit only: a rejected value is neither "fitted value" nor a correction.
    # The true value is not passed to mark_values: its full-height line would cover the fitted one.
    shown = style.Marks(previous=marks.previous, fit=marks.fit if ok else None, true=None)
    style.mark_values(ax, shown, theme, scale=1e3)
    if in_graph and _finite(marks.true):
        ax.plot([marks.true * 1e3], [1.0], transform=ax.get_xaxis_transform(), marker="v", markersize=7.0,
                markeredgewidth=0, color=p.truth, linestyle="none", label=marks.true_label, clip_on=False,
                zorder=6, gid="true value")
    if ok:
        _correction_note(ax, shown, p)
    ax.margins(y=0.14)
    ax.set_ylabel("rotated I signal (mV)")
    scale_mv = stored * 1e3
    twin = ax.secondary_xaxis("top", functions=(lambda v: np.asarray(v) / scale_mv, lambda v: np.asarray(v) * scale_mv))
    twin.set_xlabel(f"prefactor of the stored x180 amplitude ({scale_mv:.1f} mV)", fontsize=8.0, color=p.muted, labelpad=3)
    twin.tick_params(axis="x", colors=p.muted, labelcolor=p.muted, labelsize=7.5)
    twin.spines["top"].set_visible(True)
    twin.spines["top"].set_color(p.muted)
    twin.spines["top"].set_linewidth(0.6)
    n_entries = len(ax.get_legend_handles_labels()[1])
    style.place_legend(ax, theme, ncols=n_entries if n_entries <= 5 else math.ceil(n_entries / 2))

    if fitted:
        resid = y - _model(x, params) * 1e3
        sigma = float(np.std(resid)) or 1.0
        style.plot_residuals(rax, amp_mv, resid / sigma, theme)
        rax.set_ylabel(f"resid. / {style.SIGMA}\n{style.SIGMA} = std(resid.)", fontsize=7.5)
    else:
        rax.set_yticks([])
        rax.text(0.5, 0.5, "fit failed: no residuals", transform=rax.transAxes, color=p.fail,
                 fontsize=8.0, ha="center", va="center")
    rax.set_xlabel("x180 pulse amplitude (mV)")

    if ok:
        headline = f"{fit['opt_amp'] * 1e3:.2f} mV"
    elif _finite(fit["opt_amp"]):
        headline = f"rejected fit, {fit['opt_amp'] * 1e3:.2f} mV"
    else:
        headline = "fit failed"
    style.title_block(fig, NODE_NAME, "x180 pi amplitude", headline, theme, status="ok" if ok else "fail")
    return fig


def _correction_note(ax, marks: style.Marks, p: style.Palette) -> None:
    """The percent change from the previous to the fitted value.

    An arrow near the top when the two are apart; otherwise text beside the lines near the bottom
    of the axes, clear of the raw-peak marker at the top of the swing.
    """
    if not (_finite(marks.previous) and _finite(marks.fit)) or marks.previous == 0:
        return
    x0, x1 = marks.previous * 1e3, marks.fit * 1e3
    pct = (marks.fit / marks.previous - 1) * 100
    trans = ax.get_xaxis_transform()
    lo, hi = ax.get_xlim()
    if abs(x1 - x0) < 0.06 * (hi - lo):
        ax.annotate(_minus(f"{pct:+.1f} % from previous"), xy=(max(x0, x1), 0.06), xycoords=trans, xytext=(5, 0),
                    textcoords="offset points", color=p.guess, fontsize=7.5, ha="left", va="center")
        return
    ax.annotate("", xy=(x1, 0.92), xytext=(x0, 0.92), xycoords=trans, textcoords=trans,
                arrowprops={"arrowstyle": "->", "color": p.guess, "lw": 1.0, "shrinkA": 0, "shrinkB": 0})
    ax.text((x0 + x1) / 2, 0.94, _minus(f"{pct:+.1f} %"), transform=trans, color=p.guess, fontsize=7.5,
            ha="center", va="bottom")
