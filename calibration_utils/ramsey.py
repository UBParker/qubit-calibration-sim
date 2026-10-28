"""Analysis for 06a_ramsey, using QM's own fit routine from qualibration-libs."""

from __future__ import annotations

import logging
import math

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qualibration_libs.analysis import fit_oscillation_decay_exp
from qualibration_libs.analysis.models import oscillation_decay_exp

from calibration_utils import style
from calibration_utils.common import (
    MAX_OUTLIER_FRACTION,
    MAX_REFITS,
    MIN_DURBIN_WATSON,
    MIN_POINTS_IN_DECAY,
    QUBIT,
    iq_dataset,
    iq_projection,
    new_gate_reason,
    outlier_mask_of,
    outlier_summary,
    sweep_range_key,
)
from calibration_utils.robust import durbin_watson, iq_line_outliers, residual_outliers
from simulator.device import Device

log = logging.getLogger(__name__)

NODE_NAME = "06a_ramsey"
FIT_KEYS = ("a", "f", "phi", "offset", "decay")
MAX_RELATIVE_ERROR = 0.2  # fit_raw_data rejects T2ramsey_error / T2ramsey at or above this


def _flat_iq(ds: xr.Dataset) -> xr.Dataset:
    """Both detuning signs as one sweep of points, so they share one IQ axis."""
    n_t, n_s = ds.sizes["idle_time"], ds.sizes["detuning_signs"]
    return iq_dataset(
        ds.I.sel(qubit=QUBIT).values.ravel(), ds.Q.sel(qubit=QUBIT).values.ravel(), "point", np.arange(n_t * n_s)
    )


def iq_axis(ds: xr.Dataset) -> tuple[complex, complex]:
    """The ``(axis, centre)`` that ``process_raw_dataset`` projects on, for the P(e) axis."""
    _, axis, centre = iq_projection(_flat_iq(ds), "point", first_point_excited=True, exclude=outlier_mask_of(ds))
    return axis, centre


def _project(ds: xr.Dataset, exclude: np.ndarray | None) -> np.ndarray:
    """Signal (1, n_t, n_s) on the axis shared by both signs, leaving ``exclude`` out of the axis."""
    n_t, n_s = ds.sizes["idle_time"], ds.sizes["detuning_signs"]
    flat_exclude = None if exclude is None else np.asarray(exclude, dtype=bool).ravel()
    return iq_projection(_flat_iq(ds), "point", first_point_excited=True, exclude=flat_exclude)[0].values.reshape(1, n_t, n_s)


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Project I/Q onto the |g>-|e> axis, one axis shared by both detuning signs.

    At the shortest idle time the two x90 pulses add up to an x180, so the first point is the
    most excited one and the projected signal grows with excited-state population. Samples off
    the g-e line (each idle time and sign is one sample) are flagged (``outlier``) and left out of
    the axis.
    """
    n_t, n_s = ds.sizes["idle_time"], ds.sizes["detuning_signs"]
    outlier = iq_line_outliers(_flat_iq(ds).pipe(lambda f: (f.I + 1j * f.Q).sel(qubit=QUBIT).values)).reshape(n_t, n_s)
    ds = ds.assign(outlier=(("idle_time", "detuning_signs"), outlier))
    return ds.assign(signal=(("qubit", "idle_time", "detuning_signs"), _project(ds, outlier)))


def _fit_one_sign(signal: xr.DataArray) -> dict:
    """QM's a*exp(-decay*t)*cos(2*pi*f*t + phi) + offset, t in ns, so f in 1/ns and decay in 1/ns."""
    try:
        fit = fit_oscillation_decay_exp(signal, "idle_time")
    except (RuntimeError, ValueError, TypeError):
        # QM's fitter swallows curve_fit failures and returns None, which xarray then rejects.
        return {k: float("nan") for k in (*FIT_KEYS, "f_var", "decay_var")}
    p = {k: float(fit.sel(fit_vals=k)) for k in FIT_KEYS}
    p["f_var"] = float(fit.sel(fit_vals="f_f"))
    p["decay_var"] = float(fit.sel(fit_vals="decay_decay"))
    if p["f"] < 0:
        # cos(-x + phi) = cos(x - phi): same curve with a positive frequency, as QM keeps only f > 0.
        p["f"], p["phi"] = -p["f"], -p["phi"]
    return p


def frequency_offset(f_plus: float, f_minus: float, detuning: float) -> float:
    """Signed offset f_drive - f_01 from the fringe frequencies at +detuning and -detuning (all in Hz).

    With offset d, the fringes are |detuning + d| and |detuning - d|. If |d| < detuning both are
    below 2*detuning and d = (f_plus - f_minus) / 2. Otherwise one fringe has folded through zero:
    |d| is their mean and the larger fringe gives the sign. Same logic as QM's
    calibration_utils/ramsey/analysis.py ``calculate_fit_results``.
    """
    if f_plus < 2 * detuning and f_minus < 2 * detuning:
        return 0.5 * (f_plus - f_minus)
    mean = 0.5 * (f_plus + f_minus)
    return mean if f_plus > f_minus else -mean


def fit_raw_data(ds: xr.Dataset) -> dict:
    """Fit both fringes, combine them into the signed frequency error and T2* = 1/decay.

    Off-line samples (``ds.outlier``) and residual outliers of each sign's fit are masked around
    QM's fitter, with at most MAX_REFITS refits. Gates, in order: ``converged``,
    ``relative_error``, ``fringes_resolved``, ``frequency_error``, ``sampled`` (minimum over the
    signs), ``outliers`` (masked samples of all 2 n_t), ``residual_structure`` (minimum over the
    signs of the Durbin-Watson statistic of the kept residuals).
    """
    detuning = float(ds.attrs["detuning_hz"])
    signs = [int(s) for s in ds.detuning_signs.values]
    mask = np.asarray(ds.outlier.values, dtype=bool).copy() if "outlier" in ds else np.zeros((ds.sizes["idle_time"], len(signs)), dtype=bool)
    signal = ds.signal
    t_all = ds.idle_time.values.astype(float)
    for attempt in range(MAX_REFITS + 1):
        fits, residuals = {}, {}
        for col, s in enumerate(signs):
            trace = signal.sel(qubit=QUBIT, detuning_signs=s)
            keep = ~mask[:, col]
            data = trace if keep.all() else trace.isel(idle_time=np.flatnonzero(keep))
            fits[s] = _fit_one_sign(data)
            p = fits[s]
            residuals[s] = data.values - oscillation_decay_exp(t_all[keep], p["a"], p["f"], p["phi"], p["offset"], p["decay"])
        if attempt == MAX_REFITS:
            break
        new_flags = False
        for col, s in enumerate(signs):
            r = residuals[s]
            if not np.all(np.isfinite(r)):
                continue
            new = residual_outliers(r)
            if new.any():
                mask[np.flatnonzero(~mask[:, col])[new], col] = True
                new_flags = True
        if not new_flags:
            break
        signal = xr.DataArray(_project(ds, mask), dims=("qubit", "idle_time", "detuning_signs"), coords=ds.signal.coords)
    if mask.any():
        ds["signal"], ds["outlier"] = signal, (("idle_time", "detuning_signs"), mask)
    dw = {("+" if s == 1 else "-"): durbin_watson(residuals[s]) if np.all(np.isfinite(residuals[s])) else float("nan") for s in signs}
    f_plus, f_minus = fits[1]["f"] * 1e9, fits[-1]["f"] * 1e9
    freq_offset = frequency_offset(f_plus, f_minus, detuning)

    decays = np.array([fits[s]["decay"] for s in (-1, 1)])
    decay_vars = np.array([fits[s]["decay_var"] for s in (-1, 1)])
    T2s = np.where(decays > 0, 1.0 / decays, np.nan)
    T2s_err = np.sqrt(np.abs(decay_vars)) / decays**2
    T2 = float(np.mean(T2s))
    T2_err = float(np.sqrt(np.sum(T2s_err**2)) / 2)
    f_err = float(0.5 * np.sqrt(abs(fits[1]["f_var"]) + abs(fits[-1]["f_var"])) * 1e9)

    t_max = float(ds.idle_time.max())
    converged = bool(np.isfinite(freq_offset) and np.isfinite(T2) and T2 > 0)
    n_sampled = {
        s: int(np.sum(t_all[~mask[:, col]] < T2s[(-1, 1).index(s)])) if np.isfinite(T2s[(-1, 1).index(s)]) else 0
        for col, s in enumerate(signs)
    }
    min_dw = min(dw.values())
    gates = {
        "converged": converged,
        "relative_error": bool(converged and T2_err / T2 < MAX_RELATIVE_ERROR),
        # Each fringe must be resolved: at least one full period in the window.
        "fringes_resolved": bool(min(f_plus, f_minus) * t_max * 1e-9 > 1),
        "frequency_error": bool(f_err < 0.1 * detuning),
        "sampled": min(n_sampled.values()) >= MIN_POINTS_IN_DECAY,
        "outliers": bool(mask.mean() <= MAX_OUTLIER_FRACTION),
        "residual_structure": bool(np.isfinite(min_dw) and min_dw >= MIN_DURBIN_WATSON),
    }
    return {
        "freq_offset": float(freq_offset),
        "freq_offset_error": f_err,
        "fringe_frequencies": {"+": f_plus, "-": f_minus},
        "detuning": detuning,
        "T2ramsey": T2,
        "T2ramsey_error": T2_err,
        "fit_params": {s: {k: fits[s][k] for k in FIT_KEYS} for s in fits},
        # Ruling Y: the sweep range in native units (ns), for the pulse-sequence bracket.
        sweep_range_key("idle_time"): (float(ds.idle_time.min()), float(ds.idle_time.max())),
        "n_sampled": min(n_sampled.values()),
        "durbin_watson": dw,
        **outlier_summary(mask.ravel()),
        "outlier_unit": "samples",
        "gates": gates,
        "success": all(gates.values()),
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "06a_ramsey %s: detuning to correct = %.1f +/- %.1f kHz, T2* = %.1f +/- %.1f us",
        status,
        fit["freq_offset"] / 1e3,
        fit["freq_offset_error"] / 1e3,
        fit["T2ramsey"] / 1e3,
        fit["T2ramsey_error"] / 1e3,
    )


# ---------------------------------------------------------------- figure (Layout C)

SIGNS = ((1, "+"), (-1, "-"))
STRIP_H = 0.95  # inches per row unit: main 3 units, each residual strip 1 unit
FIG_W = 11.3  # inches; one wide main panel plus the cards column
FIG_H = style.NODE_MAIN_TOP + style.MARGIN_BOTTOM + 5 * STRIP_H + 2 * style.GAP_H
MIN_SLIP_ARROW = 0.08  # cycles; smaller slips get the text and drop lines but no arrow
US = style.US


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _n(value: float, spec: str) -> str:
    """``value`` formatted with ``spec``, with U+2212 for its minus sign (the repo convention for
    figure text). Formula text around it (f_drive - f_01, cos(-x)) stays ASCII."""
    return format(value, spec).replace("-", "−")


def _q(state: dict | None, key: str):
    return None if state is None else state["qubits"][QUBIT].get(key)


def _params_ok(p: dict) -> bool:
    return bool(np.all(np.isfinite(list(p.values()))))


def _fit_state(fit: dict) -> str:
    """"ok" (accepted), "rejected" (finite values that failed a gate) or "failed" (no values)."""
    if fit.get("success", False):
        return "ok"
    params_ok = all(_params_ok(fit["fit_params"][s]) for s, _ in SIGNS)
    values_ok = _finite(fit["freq_offset"], *fit["fringe_frequencies"].values())
    return "rejected" if params_ok and values_ok else "failed"


def _graph_reference(context: style.FigureContext | None) -> dict | None:
    """The reference for the true-value card, only in a graph run (ruling V: never standalone)."""
    if context is None or not context.in_graph or not context.reference:
        return None
    return context.reference


def _shown_fringes(fit: dict) -> tuple[float, float]:
    """The fringe frequencies in kHz as printed (2 decimals), so every text uses the same numbers."""
    return tuple(round(fit["fringe_frequencies"][k] / 1e3, 2) for k in ("+", "-"))


def phase_slip_cycles(fit: dict, t_ns: float) -> float:
    """Accumulated phase of the + fringe minus the - fringe at idle time ``t_ns``, in cycles.

    (f+ - f-) t + c, where c = (phi+ - phi-) / 2 pi, plus half a cycle if the two fitted
    amplitudes have opposite signs, wrapped into [-0.5, 0.5). Unfolded in t: it grows past
    0.5 cycle instead of aliasing back, unlike the distance between neighbouring troughs.
    """
    p_plus, p_minus = fit["fit_params"][1], fit["fit_params"][-1]
    constant = (p_plus["phi"] - p_minus["phi"]) / (2 * math.pi)
    if (p_plus["a"] > 0) != (p_minus["a"] > 0):
        constant += 0.5
    # Both traces start together at tau = 0, so the constant part is the phase offset nearest
    # zero: wrap it into [-0.5, 0.5) so a fitted phase 2 pi away gives the same slip.
    constant -= math.floor(constant + 0.5)
    return float((p_plus["f"] - p_minus["f"]) * t_ns + constant)


def arithmetic_lines(fit: dict) -> list[str]:
    """Why two signs, and the sign arithmetic exactly as ``frequency_offset`` and update_state do it.

    ``frequency_offset`` returns d = f_drive - f_01 = (f+ - f-)/2 when both fringes are below
    2 x detuning, otherwise +/-(f+ + f-)/2 with the sign of the larger fringe (one fringe folded
    through zero). calibrations/06a_ramsey.py:update_state then applies f_01 -= freq_offset.
    """
    lines = [
        "a fringe gives only |δ + d|, since cos(-x) = cos(x);",
        "two signs fix the sign of d and cancel errors in δ",
    ]
    state = _fit_state(fit)
    if state == "failed":
        return lines + ["fit failed: d = f_drive - f_01 not determined, f_01 unchanged"]
    f_plus, f_minus = (fit["fringe_frequencies"][k] for k in ("+", "-"))
    offset, detuning = fit["freq_offset"], fit["detuning"]
    fp, fm = _shown_fringes(fit)
    if f_plus < 2 * detuning and f_minus < 2 * detuning:
        lines.append(f"d = f_drive - f_01 = (f+ - f-)/2 = ({fp:.2f} - {fm:.2f})/2 = {_n(offset / 1e3, '+.2f')} kHz")
    else:
        sign, folded = ("+", "f-") if f_plus > f_minus else ("-", "f+")
        lines.append(
            f"d = f_drive - f_01 = {sign}(f+ + f-)/2 = {sign}({fp:.2f} + {fm:.2f})/2 = {_n(offset / 1e3, '+.2f')} kHz,"
            f" {folded} folded through 0"
        )
    verdict = "applied" if state == "ok" else "rejected fit, not applied"
    lines.append(f"f_01 -= (f_drive - f_01): f_01 {_n(-offset / 1e3, '+.2f')} kHz ({verdict})")
    return lines


def sequence_blocks(state: dict, fit: dict) -> list[style.SeqBlock]:
    """Ramsey schematic: x90, swept idle, x90 with the virtual detuning as a frame rotation, readout.

    The idle's sweep text comes from ``fit["idle_time_range"]`` (ns), "swept" when it is absent.
    """
    detuning = fit.get("detuning")
    delta = f" (δ = {detuning / 1e3:.0f} kHz)" if _finite(detuning) else ""
    span = fit.get(sweep_range_key("idle_time"))
    swept = f"{span[0]:.0f} ns to {span[1] / 1e3:.0f} {US}" if span is not None and _finite(*span) else "swept"
    return [
        style.SeqBlock("drive", "x90"),
        style.SeqBlock("idle", "τ", swept=swept, width=2.4),
        style.SeqBlock("drive", f"x90, frame ±δτ{delta}", width=2.3),
        style.SeqBlock("readout", "readout", width=1.4, line="resonator"),
    ]


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph runs only), state change and the model-mismatch caveat."""
    state = _fit_state(fit)
    ok = state == "ok"
    sigma = style.SIGMA
    offset, offset_err = fit["freq_offset"], fit["freq_offset_error"]
    T2, T2_err = fit["T2ramsey"], fit["T2ramsey_error"]
    measured = {"ok": [], "rejected": ["rejected fit, values not applied"], "failed": ["fit failed"]}[state]
    if state == "rejected" and new_gate_reason(fit):
        measured.append(new_gate_reason(fit))
    if _finite(offset, offset_err):
        measured.append(f"d = f_drive - f_01 = {_n(offset / 1e3, '+.2f')} ± {offset_err / 1e3:.2f} kHz")
    if _finite(T2, T2_err):
        measured.append(f"T2* = {T2 / 1e3:.1f} ± {T2_err / 1e3:.1f} {US} ({'exp fit' if ok else 'rejected fit'})")
    if _finite(offset_err) or _finite(T2_err):
        measured.append("± is statistical (fit covariance) only")
    T1 = _q(context.before if context else None, "T1")
    if _finite(T1):
        ratio = f", T2*/2T1 = {T2 / (2 * T1):.2f}" if _finite(T2) else ""
        measured.append(f"2 T1 = {2 * T1 / 1e3:.1f} {US} (stored T1){ratio}")
    else:
        measured.append("2 T1 = uncalculated (no T1 stored yet)")
    measured.append("limit: T2* ≤ 2 T1")
    cards = [style.Card("Measured", tuple(measured), "ok" if ok else "fail")]

    ref = _graph_reference(context)
    if ref is not None:
        lines = []
        f_true, f_after = ref.get("f_01"), _q(context.after, "f_01")
        if _finite(f_true, f_after):
            diff = f_after - f_true
            if ok and _finite(offset_err) and offset_err > 0:
                lines.append(f"new f_01 - true: {_n(diff / 1e3, '+.2f')} kHz ({_n(diff / offset_err, '+.1f')} {sigma})")
            else:
                lines.append(f"f_01 - true: {_n(diff / 1e3, '+.2f')} kHz (not corrected)")
        T2_true = ref.get("T2ramsey")
        if _finite(T2_true):
            if ok and _finite(T2, T2_err) and T2_err > 0:
                pct = 100 * (T2 / T2_true - 1)
                lines.append(
                    f"T2* - true ({T2_true / 1e3:.1f} {US}): {_n(pct, '+.1f')} % ({_n((T2 - T2_true) / T2_err, '+.1f')} {sigma})"
                )
            else:
                lines.append(f"true T2* = {T2_true / 1e3:.1f} {US}")
        if _finite(ref.get("T1")):
            lines.append(f"2 T1 (true) = {2 * ref['T1'] / 1e3:.1f} {US}")
        if lines:
            cards.append(style.Card("True value", tuple(lines), "info"))

    if context is not None and context.before is not None and context.after is not None:
        f_b, f_a = _q(context.before, "f_01"), _q(context.after, "f_01")
        change = [style.format_change("f_01", f_b, f_a, 1e-9, "GHz", "{:.6f}")]
        if _finite(f_b, f_a) and f_a != f_b:
            change.append(f"(f_01 {_n((f_a - f_b) / 1e3, '+.2f')} kHz)")
        T2_b, T2_a = _q(context.before, "T2ramsey"), _q(context.after, "T2ramsey")
        change.append(style.format_change("T2*", T2_b, T2_a, 1e-3, US, "{:.1f}"))
    elif ok:
        change = [f"f_01 {_n(-offset / 1e3, '+.2f')} kHz", f"T2* -> {T2 / 1e3:.1f} {US}"]
    else:
        change = [f"no change ({'rejected fit' if state == 'rejected' else 'fit failed'})"]
    cards.append(style.Card("State change", tuple(change)))
    # The simulated envelope is exp(-t/T2E - (t/Tq)^2) (simulator/device.py); QM's fit is exponential.
    if ref is not None and _finite(ref.get("T2ramsey"), ref.get("T2echo")):
        sigma_f = Device(T2ramsey=ref["T2ramsey"], T2echo=ref["T2echo"]).sigma_f_quasistatic
        first = f"Model mismatch: the envelope is exp(-t/T2E - (t/Tq)²), {sigma_f / 1e3:.1f}{style.NBSP}kHz rms static noise"
    else:
        first = "Model mismatch: the simulated envelope is not exponential (quasi-static noise)"
    caveat = (first, "An exponential fit reads T2* off the 1/e time; the bias depends on the window")
    cards.append(style.Card("Caveat", caveat, "caveat"))
    return cards


def _troughs(p: dict, t_max: float) -> np.ndarray:
    """Idle times (ns) of the minima of a*cos(2 pi f t + phi) in [0, t_max]."""
    f, phi = p["f"], p["phi"]
    start = math.pi if p["a"] > 0 else 0.0
    k = np.arange(-1, int(f * t_max) + 2)
    t = (start + 2 * math.pi * k - phi) / (2 * math.pi * f)
    return t[(t >= 0) & (t <= t_max)]


def _mark_phase_slip(ax: Axes, fit: dict, t_max: float, colours: dict, theme: str) -> None:
    """Arrow from the last + trough to the same trough of the - fringe, with the unfolded slip.

    The - trough is the one with the same accumulated phase (``phase_slip_cycles``), so a slip
    above half a cycle is drawn and written as such rather than aliased to the nearest trough.
    """
    p = style.palette(theme)
    p_plus, p_minus = fit["fit_params"][1], fit["fit_params"][-1]
    for t_p in _troughs(p_plus, t_max)[::-1]:
        t_m = t_p + phase_slip_cycles(fit, t_p) / p_minus["f"]
        if 0 <= t_m <= t_max:
            break
    else:
        return
    slip = phase_slip_cycles(fit, t_p)
    y_p = oscillation_decay_exp(np.array([t_p]), **p_plus)[0] * 1e3
    y_m = oscillation_decay_exp(np.array([t_m]), **p_minus)[0] * 1e3
    lo, hi = ax.get_ylim()
    y = min(y_p, y_m) - 0.07 * (hi - lo)
    for t_x, y_x, s in ((t_p, y_p, 1), (t_m, y_m, -1)):
        ax.plot([t_x / 1e3] * 2, [y_x, y], color=colours[s][1], linewidth=0.8, linestyle=(0, (2, 1.5)), zorder=4)
    if abs(slip) >= MIN_SLIP_ARROW:  # below this the arrow heads overlap
        ax.annotate(
            "", xy=(t_m / 1e3, y), xytext=(t_p / 1e3, y),
            arrowprops={"arrowstyle": "<->", "color": p.fg, "linewidth": 1.0, "mutation_scale": 8,
                        "shrinkA": 0, "shrinkB": 0},
            zorder=5,
        )
    fp, fm = _shown_fringes(fit)
    ax.annotate(
        # The slip includes the fitted phase offset between the fringes, so it is not just
        # (f+ - f-) t; the label says so.
        f"phase slip {abs(slip):.2f} cycle at {t_p / 1e3:.0f} {US}, incl. phase offset\n"
        f"fringes differ by {abs(fp - fm):.2f} kHz",
        xy=((t_p + t_m) / 2e3, y), xytext=(0, -5), textcoords="offset points",
        color=p.fg, fontsize=7.5, ha="right" if max(t_p, t_m) / t_max > 0.85 else "center", va="top",
        linespacing=1.35, zorder=5,
    )


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Layout C figure: both detuning signs overlaid on one axis, a residual strip per sign.

    ``marks`` is accepted for the shared node interface; the x axis here is idle time, so the
    previous, fitted and true f_01 go in the cards (from ``context``) rather than on the axis.
    A rejected fit (finite values, a failed gate) is drawn dashed and labelled "rejected fit",
    with no envelope, T2* label or phase-slip mark.
    """
    p = style.palette(theme)
    colours = {1: (p.data, p.fit), -1: (p.data2, p.fit2)}  # (points, fit curve)
    lay = style.node_layout(theme, figsize=(FIG_W, FIG_H), residual_rows=2)
    fig, ax = lay.fig, lay.main_axes[0]
    res = dict(zip((1, -1), lay.residual_strips[0]))
    state = _fit_state(fit)
    accepted = state == "ok"

    t = ds.idle_time.values.astype(float)
    t_fine = np.linspace(t.min(), t.max(), 4000)
    detuning_khz = fit["detuning"] / 1e3
    fringes = dict(zip(("+", "-"), _shown_fringes(fit))) if state != "failed" else {}
    for s, name in SIGNS:
        dot, line = colours[s]
        y = ds.signal.sel(qubit=QUBIT, detuning_signs=s).values * 1e3
        ax.plot(t / 1e3, y, "o", color=dot, markersize=2.6, markeredgewidth=0, alpha=0.85,
                gid=f"data{name}", label="_data", zorder=2)
        p_s = fit["fit_params"][s]
        if _params_ok(p_s):
            ax.plot(t_fine / 1e3, oscillation_decay_exp(t_fine, **p_s) * 1e3, color=line,
                    linewidth=1.0, alpha=0.95 if accepted else 0.6,
                    linestyle="-" if accepted else (0, (4, 2.5)), gid=f"fit{name}", label="_fit", zorder=3)
        fringe = f" (fringe {fringes[name]:.2f} kHz)" if accepted else ""
        ax.plot([], [], marker="o", color=line, markerfacecolor=dot, markeredgewidth=0, markersize=3.8,
                linewidth=1.0, linestyle="-" if accepted else (0, (4, 2.5)),
                label=f"{name}{detuning_khz:.0f} kHz{fringe}".replace("-", "−"))
        if _params_ok(p_s):
            r = y - oscillation_decay_exp(t, **p_s) * 1e3
            sd = float(np.std(r))
            r = r / sd if sd > 0 else r * np.nan
        else:
            r = np.full_like(y, np.nan)
        style.plot_residuals(res[s], t / 1e3, r, theme)
        res[s].get_lines()[-1].set_color(dot)
        res[s].set_ylabel(f"{name}δ".replace("-", "−"), fontsize=8.0)
        res[s].set_gid(f"resid{name}")
    if state == "rejected":
        ax.plot([], [], color=p.muted, linestyle=(0, (4, 2.5)), linewidth=1.0, label="rejected fit")

    T2, T2_err = fit["T2ramsey"], fit["T2ramsey_error"]
    params = [fit["fit_params"][s] for s, _ in SIGNS]
    if accepted and _finite(T2, T2_err):
        offset = np.mean([q["offset"] for q in params])
        amp = np.mean([abs(q["a"]) for q in params])
        env = amp * np.exp(-t_fine / T2)
        for side in (1, -1):
            label = f"T2* = {T2 / 1e3:.1f} ± {T2_err / 1e3:.1f} {US} (exp fit)" if side > 0 else "_env"
            ax.plot(t_fine / 1e3, (offset + side * env) * 1e3, color=p.muted, linewidth=0.9,
                    linestyle=(0, (4, 2.5)), zorder=2.5, label=label)

    y_all = ds.signal.sel(qubit=QUBIT).values * 1e3
    lo, hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    span = hi - lo
    ax.set_ylim(lo - 0.3 * span, hi + 0.5 * span)  # room for the slip mark below and the chip above
    ax.set_xlim(0, t.max() / 1e3 * 1.01)
    ax.set_ylabel("signal, rotated I (mV)")
    style.value_chip(ax, arithmetic_lines(fit), theme, loc="upper right")
    if accepted:
        _mark_phase_slip(ax, fit, float(t.max()), colours, theme)
    style.place_legend(ax, theme, title="detuning δ, both signs")
    if context is not None and context.population_levels is not None:
        level_g, level_e = context.population_levels
        style.population_axis(ax, level_g * 1e3, level_e * 1e3, theme)

    res[-1].set_xlabel(f"idle time τ ({US})")
    res[-1].annotate(
        f"strips: (data - fit) / {style.SIGMA}, {style.SIGMA} = std of residuals per sign",
        xy=(1.0, 0.0), xycoords="axes fraction", xytext=(0, -24), textcoords="offset points",
        color=p.muted, fontsize=7.0, ha="right", va="top", annotation_clip=False,
    )

    node_order = context.node_order if context else [NODE_NAME]
    statuses = context.statuses if context else {NODE_NAME: "ok" if accepted else "fail"}
    style.progress_strip(lay.progress_ax, node_order, NODE_NAME, statuses, theme)
    style.pulse_sequence(lay.sequence_ax, sequence_blocks({}, fit), theme)
    style.cards_column(lay.cards_ax, node_cards(fit, context), theme)
    if accepted:
        headline = f"f_01 {_n(-fit['freq_offset'] / 1e3, '+.2f')} kHz, T2* {T2 / 1e3:.1f} {US}"
    else:
        headline = "rejected fit" if state == "rejected" else "fit failed"
        if state == "rejected" and new_gate_reason(fit):
            headline = f"rejected fit: {new_gate_reason(fit)}"
    style.title_block(fig, NODE_NAME, "Ramsey fringes correct f_01 and measure T2*", headline, theme,
                      status="ok" if accepted else "fail")
    return fig
