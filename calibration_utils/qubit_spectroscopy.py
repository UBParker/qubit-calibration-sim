"""Analysis for 03a_qubit_spectroscopy.

Peak finding follows QM's node (``qualibration_libs.analysis.peaks_dips`` on a rotated quadrature).
QM stops there and reports the position of the highest sample, which quantises the result to the
sweep step. qualibration-libs ships the ``lorentzian_peak`` model but no fit routine for it, so the
peak is then refined with ``scipy.optimize.curve_fit`` on that model, seeded by ``peaks_dips``.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from qualibration_libs.analysis import peaks_dips
from qualibration_libs.analysis.models import lorentzian_peak
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from calibration_utils import style
from calibration_utils.common import (
    MAX_OUTLIER_FRACTION,
    QUBIT,
    SECOND_PEAK_MIN_FRACTION,
    SECOND_PEAK_MIN_Z,
    add_iq_derived,
    new_gate_reason,
    outlier_summary,
    project_on_iq_axis,
    sweep_range_key,
)
from calibration_utils.robust import isolated_spikes, residual_peak

log = logging.getLogger(__name__)

# Fit quality gates. The centre must be known to a small fraction of a typical
# power-broadened linewidth for the next nodes (Rabi, Ramsey) to start on resonance.
MAX_CENTER_ERROR_HZ = 50e3


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Add amplitude, phase and the projected signal. Only a few points sit on the qubit line, so
    the projection is signed so its largest excursion from the median baseline is a peak."""
    ds = add_iq_derived(ds)
    # Isolated spikes (readout glitches) are flagged and left out of the projection axis.
    outlier = isolated_spikes((ds.I + 1j * ds.Q).sel(qubit=QUBIT).values)
    ds = ds.assign(outlier=("detuning", outlier))
    return ds.assign(signal=project_on_iq_axis(ds, "detuning", orientation="peak", exclude=outlier))


def _drive_frequency_offset(ds: xr.Dataset) -> float:
    """Absolute drive frequency at zero detuning (the qubit frequency the sweep was centred on)."""
    return float(ds.full_freq.values[0] - ds.detuning.values[0])


def fit_raw_data(ds: xr.Dataset) -> dict:
    """Locate the qubit line with ``peaks_dips``, then fit a Lorentzian peak around it.

    QM's ``lorentzian_peak`` is offset + amplitude / (1 + ((x - center) / width)^2), so its
    ``width`` is the half width at half maximum; the FWHM is twice that.

    Isolated spikes (``ds.outlier``) are left out of the peak search and the fit. A strongly
    coupled TLS splits the line (Müller, Cole, Lisenfeld 2019,
    https://doi.org/10.1088/1361-6633/ab3a7e), which a single Lorentzian fits to one branch or
    between them; ``second_peak`` rejects a fit whose residual still holds a significant peak.
    Gates, in order: found, converged, center_error, inside, width, outliers, second_peak.
    """
    detuning_all = ds.detuning.values.astype(float)
    mask = np.asarray(ds.outlier.values, dtype=bool) if "outlier" in ds else np.zeros(detuning_all.size, dtype=bool)
    kept_signal = ds.signal if not mask.any() else ds.signal.isel(detuning=np.flatnonzero(~mask))
    detuning = detuning_all[~mask]
    signal = kept_signal.sel(qubit=QUBIT).values
    step = float(np.diff(detuning_all)[0])
    span = float(detuning_all[-1] - detuning_all[0])
    # Swept range in Hz, for the pulse schematic (sequence_blocks); not a fit result.
    sweep_range = {sweep_range_key("detuning"): (float(detuning_all.min()), float(detuning_all.max()))}

    failed = {
        "frequency": float("nan"),
        "frequency_error": float("nan"),
        "detuning": float("nan"),
        "fwhm": float("nan"),
        "fit_params": None,
        "success": False,
        **sweep_range,
        **outlier_summary(mask),
    }

    peak = peaks_dips(kept_signal, "detuning", prominence_factor=5).sel(qubit=QUBIT)
    position, fwhm_guess = float(peak.position), float(peak.width)
    if not (np.isfinite(position) and np.isfinite(fwhm_guess)):
        log.warning("03a_qubit_spectroscopy: no peak above the noise floor")
        return {**failed, "gates": {"found": False}}

    offset_guess = float(np.median(signal))
    p0 = [float(signal.max() - offset_guess), position, max(fwhm_guess / 2, step), offset_guess]
    bounds = ([0.0, detuning_all[0], step / 4, -np.inf], [np.inf, detuning_all[-1], span, np.inf])
    try:
        popt, pcov = curve_fit(lorentzian_peak, detuning, signal, p0=p0, bounds=bounds)
    except (RuntimeError, ValueError) as err:
        log.warning("03a_qubit_spectroscopy: Lorentzian fit did not converge (%s)", err)
        return {**failed, "gates": {"found": True, "converged": False}}

    amplitude, center, hwhm, offset = (float(v) for v in popt)
    center_err = float(np.sqrt(abs(pcov[1, 1])))
    fwhm = 2 * hwhm
    z, fraction, i_peak = residual_peak(signal - lorentzian_peak(detuning, *popt), amplitude)
    gates = {
        "found": True,
        "converged": True,
        "center_error": bool(np.isfinite(center_err) and center_err < MAX_CENTER_ERROR_HZ),
        "inside": bool(detuning_all[0] < center < detuning_all[-1]),
        "width": bool(step < fwhm < span / 2),
        "outliers": bool(mask.mean() <= MAX_OUTLIER_FRACTION),
        "second_peak": not (z >= SECOND_PEAK_MIN_Z and fraction >= SECOND_PEAK_MIN_FRACTION),
    }
    return {
        "frequency": _drive_frequency_offset(ds) + center,
        "frequency_error": center_err,
        "detuning": center,
        "fwhm": fwhm,
        "fit_params": {"amplitude": amplitude, "center": center, "width": hwhm, "offset": offset},
        "success": all(gates.values()),
        **sweep_range,
        "second_peak_z": z,
        "second_peak_fraction": fraction,
        "second_peak_detuning": float(detuning[i_peak]),
        **outlier_summary(mask),
        "gates": gates,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "03a_qubit_spectroscopy %s: f_01 = %.6f GHz +/- %.1f kHz, FWHM = %.0f kHz",
        status,
        fit["frequency"] / 1e9,
        fit["frequency_error"] / 1e3,
        fit["fwhm"] / 1e3,
    )


NODE_NAME = "03a_qubit_spectroscopy"  # the node file imports it from here

# Panel width in inches. Two panels (full sweep, zoom on the line) at this width keep the Layout C
# page near 12 in wide, so it still reads at README width.
_PANEL_W = 3.55
_ZOOM_FWHM = 3.2  # half width of the zoom panel in fitted FWHMs


def sequence_blocks(state: dict, fit: dict) -> list[style.SeqBlock]:
    """Pulse schematic for one sweep point: a long, weak saturation pulse, then the readout.

    Single source for the static figure and the report. The drive detuning range comes from
    ``fit["detuning_range"]`` (Hz) when present.
    """
    q = state["qubits"][QUBIT]
    sat_us = q["xy"]["operations"]["saturation"]["length"] / 1e3
    ro_us = q["resonator"]["operations"]["readout"]["length"] / 1e3
    swept = None
    sweep = fit.get(sweep_range_key("detuning"))
    if sweep is not None:
        lo, hi = (v / 1e6 for v in sweep)
        swept = f"drive detuning {lo:+.1f} to {hi:+.1f} MHz".replace("-", "−")
    return [
        style.SeqBlock("drive", f"saturation {sat_us:g} {style.US}", swept=swept, width=4.0),
        style.SeqBlock("readout", f"readout {ro_us:g} {style.US}", width=1.6, line="resonator"),
    ]


def _fit_ok(fit: dict) -> bool:
    return bool(fit.get("success")) and fit.get("fit_params") is not None


def _minus(text: str) -> str:
    """U+2212 for the minus signs of formatted numbers (the repo convention for figure text)."""
    return text.replace("-", "−")


def _finite(value) -> bool:
    return value is not None and np.isfinite(value)


def _is_graph_run(context: style.FigureContext | None) -> bool:
    return context is not None and context.in_graph


def _rejection_reason(fit: dict) -> str:
    """Why a fit that produced a curve was rejected (the gates in ``fit_raw_data``)."""
    if new_gate_reason(fit):
        return new_gate_reason(fit)
    err = fit.get("frequency_error", float("nan"))
    if not np.isfinite(err) or err >= MAX_CENTER_ERROR_HZ:
        return f"centre error above the {MAX_CENTER_ERROR_HZ / 1e3:.0f} kHz gate"
    return "line outside the width or range gates"


def _stored_t2star(context: style.FigureContext | None) -> float | None:
    """T2* (ns) in the state before this run, or None when not measured yet."""
    if context is None or context.before is None:
        return None
    value = context.before["qubits"][QUBIT].get("T2ramsey")
    return float(value) if value is not None and np.isfinite(value) else None


def _natural_width_text(t2star_ns: float | None) -> list[str]:
    """Natural linewidth 1/(pi T2*) (FWHM) for scale against the power-broadened line."""
    if t2star_ns is None:
        return ["natural width 1/(πT2*):", "unknown, T2* not", "measured yet (06a)"]
    width_khz = 1 / (np.pi * t2star_ns * 1e-9) / 1e3
    return [f"natural width 1/(πT2*) = {width_khz:.1f} kHz", f"at stored T2* = {t2star_ns / 1e3:.1f} {style.US}"]


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph runs with a reference only) and state change cards."""
    before = None if context is None or context.before is None else context.before["qubits"][QUBIT]["f_01"]
    after = None if context is None or context.after is None else context.after["qubits"][QUBIT]["f_01"]
    cards: list[style.Card] = []
    if _fit_ok(fit):
        lines = [
            f"f_01 = {fit['frequency'] / 1e9:.6f} GHz",
            f"± {fit['frequency_error'] / 1e3:.1f} kHz (fit, 1{style.SIGMA}, stat.)",
            f"FWHM {fit['fwhm'] / 1e3:.0f} kHz, power-broadened",
        ]
        if before is not None:
            lines.append(_minus(f"shift from guess {(fit['frequency'] - before) / 1e6:+.2f} MHz"))
        cards.append(style.Card("Measured", tuple(lines), "ok"))
    else:
        if fit.get("fit_params") is not None:
            # A new gate names its own reason; the older gates keep their two-line text.
            reason = new_gate_reason(fit)
            lines = ("rejected fit, state not updated", reason) if reason else ("fit failed: rejected fit", _rejection_reason(fit))
        else:
            lines = ("fit failed", "no peak above the noise floor")
        cards.append(style.Card("Measured", lines, "fail"))

    reference = context.reference if context is not None else None
    true_f01 = (reference or {}).get("f_01")
    if _is_graph_run(context) and true_f01 is not None and _fit_ok(fit):
        error = fit["frequency"] - true_f01
        lines = [
            f"f_01 = {true_f01 / 1e9:.6f} GHz",
            _minus(f"fit - true = {error / 1e3:+.1f} kHz"),
            _minus(f"= {error / fit['frequency_error']:+.2f}{style.SIGMA}, {error / fit['fwhm']:+.3f} FWHM"),
        ]
        t2_true = reference.get("T2ramsey")
        if t2_true:
            lines.append(f"natural width at true T2*: {1 / (np.pi * t2_true * 1e-9) / 1e3:.1f} kHz")
        cards.append(style.Card("True value", tuple(lines), "info"))

    if _fit_ok(fit) and after is not None:
        change = style.format_change("f_01", before, after, 1e-9, "GHz", "{:.6f}")
    else:
        change = "f_01: unchanged (fit failed)"
    cards.append(style.Card("State change", (change,)))
    return cards


def _peak_marker_text(ax, x: float, text: str, colour: str) -> None:
    """Small label just right of a value marker's triangle on the top edge."""
    ax.annotate(
        text,
        xy=(x, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(6, -2),
        textcoords="offset points",
        color=colour,
        fontsize=8.0,
        ha="left",
        va="top",
        annotation_clip=False,
    )


def _bracket(ax, x0: float, x1: float, y: float, text: str, colour: str) -> None:
    """Double-headed arrow from x0 to x1 at axes height y, with ``text`` centred above it."""
    trans = ax.get_xaxis_transform()
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords=trans,
        textcoords=trans,
        arrowprops={"arrowstyle": "<|-|>", "color": colour, "lw": 0.9, "shrinkA": 0, "shrinkB": 0,
                    "mutation_scale": 7},
    )
    ax.annotate(
        text,
        xy=((x0 + x1) / 2, y),
        xycoords=trans,
        xytext=(0, 3),
        textcoords="offset points",
        color=colour,
        fontsize=8.0,
        fontweight="medium",
        ha="center",
        va="bottom",
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Layout C page: full sweep with the correction from the starting guess, and a zoom on the line.

    The true value is not drawn on the plot (a few kHz from the fit, invisible at this scale); it
    goes in the true-value card. The fitted FWHM is labelled power-broadened, next to the natural
    width 1/(pi T2*) for scale.
    """
    p = style.palette(theme)
    marks = marks or style.Marks()
    f_centre = _drive_frequency_offset(ds)
    x_hz = ds.detuning.values.astype(float)
    x = x_hz / 1e6
    y = ds.signal.sel(qubit=QUBIT).values * 1e3
    ok = _fit_ok(fit)
    has_curve = fit.get("fit_params") is not None

    ncols, resid = 2, has_curve  # no residual row without a fitted curve
    right = style.MARGIN_RIGHT + style.CARDS_GAP + style.CARDS_W
    width = style.MARGIN_LEFT + right + ncols * _PANEL_W + (ncols - 1) * style.GAP_W
    height = style.MARGIN_BOTTOM + style.NODE_MAIN_TOP + style.PANEL_H + style.RESID_H
    lay = style.node_layout(theme, ncols=ncols, residuals=resid, figsize=(width, height))
    fig = lay.fig
    full, zoom = lay.main_axes
    rfull, rzoom = lay.residual_axes

    # figure_context sets this node's status and the downstream "not run" pills; the fit decides
    # only when there is no context.
    statuses = context.statuses if context is not None else {NODE_NAME: "ok" if ok else "fail"}
    order = context.node_order if context is not None else [NODE_NAME]
    style.progress_strip(lay.progress_ax, order, NODE_NAME, statuses, theme)
    if context is not None and context.before is not None:  # no state, no pulse lengths to show
        style.pulse_sequence(lay.sequence_ax, sequence_blocks(context.before, fit), theme)
    style.cards_column(lay.cards_ax, node_cards(fit, context), theme)

    def model_hz(f):
        """Fitted Lorentzian in mV at drive detuning ``f`` (Hz)."""
        return lorentzian_peak(f, **fit["fit_params"]) * 1e3

    model_x = np.linspace(x_hz[0], x_hz[-1], 4000)

    # Full sweep: starting guess, fitted centre and the correction between them.
    for ax in (full, zoom):
        ax.plot(x, y, "o", color=p.data, markersize=3.0, markeredgewidth=0, label="data", zorder=2)
        if has_curve:
            if ok:
                ax.plot(model_x / 1e6, model_hz(model_x), color=p.fit, label="Lorentzian fit", zorder=3)
            else:
                ax.plot(model_x / 1e6, model_hz(model_x), color=p.fit, linestyle=(0, (4, 3)), alpha=0.6,
                        label="rejected Lorentzian", zorder=3)
    # A rejected fit keeps its centre marker, labelled as rejected; the true value is card only.
    fit_label = "fitted f_01" if ok else "rejected fit"
    shown = style.Marks(
        previous=None if marks.previous is None else (marks.previous - f_centre) / 1e6,
        fit=fit["detuning"] / 1e6 if has_curve else None,
    )
    style.mark_values(full, shown, theme, previous_label="starting guess", fit_label=fit_label)
    style.mark_values(zoom, style.Marks(fit=shown.fit), theme, fit_label=fit_label)

    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo if hi > lo else 1.0
    full.set_ylim(lo - 0.08 * span, hi + 0.42 * span)
    full.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    if ok and _finite(shown.previous):
        _bracket(full, shown.previous, shown.fit, 0.88, _minus(f"{shown.fit - shown.previous:+.2f} MHz"), p.fg)
    if not ok:
        reason = "fit failed: rejected by the quality gates" if has_curve else "fit failed: no peak above the noise floor"
        full.text(0.5, 0.9, reason, transform=full.transAxes, color=p.fail, fontsize=9.0,
                  fontweight="medium", ha="center", va="center", zorder=6,
                  bbox={"boxstyle": "round,pad=0.35", "facecolor": p.bg, "edgecolor": p.fail, "linewidth": 0.8})

    # Zoom on the line: FWHM at half maximum, natural width for scale, centre uncertainty.
    if has_curve and np.isfinite(fit["fwhm"]):
        centre_mhz, fwhm_mhz = fit["detuning"] / 1e6, fit["fwhm"] / 1e6
        half = max(_ZOOM_FWHM * fwhm_mhz, 1.0)
    else:
        centre_mhz, fwhm_mhz = float(x[np.argmax(y)]), float("nan")
        half = 2.5
    zoom.set_xlim(centre_mhz - half, centre_mhz + half)
    zoom.set_ylim(lo - 0.08 * span, hi + 0.30 * span)
    if ok and np.isfinite(fwhm_mhz):  # the line-width labels only for an accepted fit
        params = fit["fit_params"]
        half_max = (params["offset"] + params["amplitude"] / 2) * 1e3
        left, right_x = centre_mhz - fwhm_mhz / 2, centre_mhz + fwhm_mhz / 2
        zoom.annotate(
            "", xy=(right_x, half_max), xytext=(left, half_max),
            arrowprops={"arrowstyle": "<|-|>", "color": p.fg, "lw": 0.9, "shrinkA": 0, "shrinkB": 0,
                        "mutation_scale": 7},
            zorder=4,
        )
        zoom.annotate(
            f"FWHM {fit['fwhm'] / 1e3:.0f} kHz\npower-broadened\nby the drive",
            xy=(right_x, half_max), xytext=(14, 5), textcoords="offset points",
            color=p.fg, fontsize=8.0, ha="left", va="bottom", linespacing=1.35,
        )
        zoom.annotate(
            "\n".join(_natural_width_text(_stored_t2star(context))),
            xy=(left, half_max), xytext=(-14, 5), textcoords="offset points",
            color=p.muted, fontsize=7.5, ha="right", va="bottom", linespacing=1.35,
        )
        _peak_marker_text(zoom, centre_mhz, f"±{fit['frequency_error'] / 1e3:.1f} kHz (1{style.SIGMA})", p.fit)

    # Residuals, normalised by their own standard deviation.
    if has_curve:
        r = y - model_hz(x_hz)
        sigma = float(np.std(r))
        r = r / sigma if sigma > 0 else r * np.nan
        for rax in (rfull, rzoom):
            style.plot_residuals(rax, x, r, theme)
            rax.set_ylabel("resid. / std")

    xlabel = f"drive detuning from {f_centre / 1e9:.4f} GHz (MHz)"
    for ax, rax in ((full, rfull), (zoom, rzoom)):
        ax.set_ylabel("projected IQ signal (mV)")
        (rax if has_curve else ax).set_xlabel(xlabel)
    # Each entry once: data, fit and the starting guess over the full sweep, the fitted f_01 over
    # the zoom (the same marker sits on both panels).
    for line in full.get_lines():
        if line.get_label() == fit_label:
            line.set_label("_" + fit_label)
    for line in zoom.get_lines():
        if line.get_label() in ("data", "Lorentzian fit", "rejected Lorentzian"):
            line.set_label("_" + line.get_label())
    # The rejected labels are longer; two rows keep that legend inside the panel.
    style.place_legend(full, theme, ncols=None if ok else 2, title="full sweep")
    if ok:
        zoom_title = "zoom on the line"
    else:
        zoom_title = "zoom on the rejected fit" if has_curve else "zoom on the highest point"
    style.place_legend(zoom, theme, title=zoom_title)

    if ok:
        headline = f"f_01 = {fit['frequency'] / 1e9:.6f} GHz ± {fit['frequency_error'] / 1e3:.1f} kHz"
        style.title_block(fig, NODE_NAME, "Qubit line found", headline, theme, status="ok")
    else:
        headline = "rejected fit" if has_curve else "fit failed"
        style.title_block(fig, NODE_NAME, "Qubit line not found", headline, theme, status="fail")
    return fig
