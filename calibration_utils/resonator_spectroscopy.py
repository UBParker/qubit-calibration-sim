"""Analysis for 02a_resonator_spectroscopy: locate the readout resonator dip in |S21|.

Dip finding and the line shape come from QM's qualibration-libs (``peaks_dips`` and the
``lorentzian_dip`` model), and the amplitude/phase processing is QM's ``add_amplitude_and_phase``.
qualibration-libs has no ready-made Lorentzian fitting wrapper (``fitting.py`` only covers
decays and oscillations), and QM's own node runs ``scipy.optimize.curve_fit`` on a Lorentzian
dip, so the least-squares step here does the same.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from qualibration_libs.analysis import lorentzian_dip, peaks_dips
from qualibration_libs.data import add_amplitude_and_phase
from scipy.optimize import curve_fit

from calibration_utils.common import MAX_OUTLIER_FRACTION, QUBIT, new_gate_reason, outlier_summary, sweep_range_key
from calibration_utils.robust import isolated_spikes
from calibration_utils.style import (
    CARDS_GAP,
    CARDS_W,
    CHI,
    GAP_H,
    KAPPA,
    KET_E,
    KET_G,
    MARGIN_BOTTOM,
    MARGIN_LEFT,
    MARGIN_RIGHT,
    NBSP,
    NODE_MAIN_TOP,
    SIGMA,
    US,
    Card,
    FigureContext,
    Marks,
    SeqBlock,
    cards_column,
    format_change,
    mark_values,
    node_layout,
    palette,
    place_legend,
    plot_residuals,
    progress_strip,
    pulse_sequence,
    title_block,
)

log = logging.getLogger(__name__)

# A dip only counts as the resonator if its depth stands this many noise sigmas above the
# residual scatter (QM's v2 analysis uses the same 6-sigma default, ``min_dip_snr``).
MIN_DIP_SNR = 6.0


def notch_amplitude(x, a, depth, center, kappa):
    """|S21| of a notch resonator: ``a * |1 - depth / (1 + 2j (x - center) / kappa)|``.

    Probst et al. 2015 (arXiv:1410.3365) Eq. 1 with phi = 0 and no cable delay: the simulator's
    steady state has no cable delay; the measured spectrum is windowed, see KAPPA_BIAS. ``depth`` is Ql/|Qc| and ``kappa`` is the full resonator linewidth, in the
    units of ``x``.
    """
    return a * np.abs(1 - depth / (1 + 2j * (np.asarray(x) - center) / kappa))


def _fit_notch(x, y, center0: float, fwhm0: float, offset0: float, step: float, span: float):
    """Fit notch_amplitude (x in MHz, y in mV), starting from the Lorentzian dip fit.

    Returns (popt, perr) in the fit units, ordered as notch_amplitude's (a, depth, center,
    kappa), or None if the fit fails.
    """
    depth0 = float(np.clip((offset0 - y.min()) / offset0, 0.01, 0.99)) if offset0 > 0 else 0.5
    kappa0 = float(np.clip(fwhm0 / 0.7, 1.01 * step, 0.99 * span))
    p0 = [offset0, depth0, center0, kappa0]
    bounds = ([0, 0, x[0], step], [np.inf, 1, x[-1], span])
    try:
        popt, pcov = curve_fit(notch_amplitude, x, y, p0=p0, bounds=bounds, maxfev=10_000)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov))
    if not np.all(np.isfinite(perr)):
        return None
    return popt, perr


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Add IQ_abs and the slope-subtracted phase, as QM's node does with the same helper, and flag
    isolated spikes (``outlier``) in IQ_abs exp(i phase): the I/Q trace with the cable-delay phase
    roll removed, a smooth circle. The raw I/Q turn by 2 pi f tau over the sweep (about 4.7 turns
    over 20 MHz at tau = 237 ns), which would hide spikes from the three-point median test."""
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    flat = (ds.IQ_abs * np.exp(1j * ds.phase)).sel(qubit=QUBIT).values
    return ds.assign(outlier=("detuning", isolated_spikes(flat)))


def fit_raw_data(ds: xr.Dataset) -> dict:
    """Fit a Lorentzian dip to IQ_abs, then the notch amplitude model for kappa.

    |S21| of a notch resonator is symmetric about f0, so the fitted centre is unbiased even
    though |S21| is not exactly Lorentzian (|S21|^2 is). The fitted width is the width of the
    |S21| dip, which for a deep notch is narrower than the resonator linewidth kappa, so it is
    reported as ``fwhm`` and not used as a kappa estimate. ``kappa`` (Hz) and ``notch_depth``
    (Ql/|Qc|) come from a second fit of ``notch_amplitude``, run only when the first succeeded.
    ``notch_params`` is in SI units for ``notch_amplitude(ds.detuning.values, **notch_params)``:
    x is the raw detuning in Hz and the result is in V. ``detuning_range`` is the swept
    (min, max) detuning in Hz, for the pulse schematic.
    """
    detuning_all = ds.detuning.values.astype(float)
    mask = np.asarray(ds.outlier.values, dtype=bool) if "outlier" in ds else np.zeros(detuning_all.size, dtype=bool)
    iq_abs = ds.IQ_abs if not mask.any() else ds.IQ_abs.isel(detuning=np.flatnonzero(~mask))
    detuning = detuning_all[~mask]
    amplitude = iq_abs.sel(qubit=QUBIT).values
    step = float(np.median(np.diff(detuning_all)))
    span = float(detuning_all[-1] - detuning_all[0])
    failed = {
        sweep_range_key("detuning"): (float(detuning_all.min()), float(detuning_all.max())),
        **outlier_summary(mask),
        "frequency": float("nan"),
        "frequency_error": float("nan"),
        "fwhm": float("nan"),
        "snr": 0.0,
        "kappa": float("nan"),
        "kappa_error": float("nan"),
        "notch_depth": float("nan"),
        "notch_params": {},
    }

    # Step 1: QM's peak finder gives the starting point (it detects whether the feature is a dip).
    dip = peaks_dips(iq_abs, "detuning").sel(qubit=QUBIT)
    position, width = float(dip.position), float(dip.width)
    if not np.isfinite(position):
        return {**failed, "fit_params": {}, "gates": {"found": False}, "success": False}

    # Step 2: least-squares fit of QM's lorentzian_dip. Its ``width`` argument is the half width
    # at half maximum (the docstring says FWHM, but L(center +/- width) = offset - amplitude / 2).
    # Fit in MHz and mV so the parameters are of order one.
    x, y = detuning / 1e6, amplitude * 1e3
    off_dip = y[np.abs(x - position / 1e6) > width / 1e6]
    offset0 = float(np.median(off_dip)) if off_dip.size else float(y.max())
    hwhm0 = max(width / 2, 2 * step) / 1e6
    p0 = [offset0 - y.min(), position / 1e6, hwhm0, offset0]
    bounds = ([0, x[0], step / 1e6, 0], [np.inf, x[-1], span / 2e6, np.inf])
    try:
        popt, pcov = curve_fit(lorentzian_dip, x, y, p0=p0, bounds=bounds, maxfev=10_000)
    except (RuntimeError, ValueError):
        return {**failed, "fit_params": {}, "gates": {"found": True, "converged": False}, "success": False}
    perr = np.sqrt(np.diag(pcov))
    depth, center, hwhm, offset = popt

    residual_sigma = float(np.std(y - lorentzian_dip(x, *popt)))
    snr = depth / residual_sigma if residual_sigma > 0 else 0.0
    frequency = float(ds.full_freq.values[0] - detuning_all[0] + center * 1e6)
    frequency_error = float(perr[1] * 1e6)
    fwhm = float(2 * hwhm * 1e6)

    # Accept only a significant, well-resolved dip that sits inside the sweep and is located
    # to a small fraction of its own width.
    inside = x[0] + hwhm < center < x[-1] - hwhm
    resolved = 2 * step <= fwhm <= span / 2 and frequency_error < fwhm / 10
    gates = {
        "found": True,
        "converged": bool(np.all(np.isfinite(perr))),
        "inside": bool(inside),
        "snr": bool(snr >= MIN_DIP_SNR),
        "resolved": bool(resolved),
        "outliers": bool(mask.mean() <= MAX_OUTLIER_FRACTION),
    }
    success = all(gates.values())

    # Step 3: kappa from the notch amplitude model, seeded by the accepted Lorentzian fit
    # (_fit_notch starts kappa at fwhm / 0.7: a Lorentzian fitted to |S21| has FWHM of about
    # 0.7 kappa for depth about 0.8; it is only a starting value).
    # Known systematic (KAPPA_BIAS): the 1 us window is not a steady-state measurement. The field
    # rings up with time constant 2/(2 pi kappa) = 227 ns, so the window-averaged spectrum is not a
    # notch line shape and the notch fit returns kappa about 29 % high and depth 0.63 (true 0.8) for
    # |g> alone (noiseless, 1.061 and 0.755 at 4 us); the thermal |e> dip 2 chi below adds about 1 %.
    # The centre stays unbiased to a few kHz. kappa_error is statistical only.
    notch = {
        "kappa": float("nan"),
        "kappa_error": float("nan"),
        "notch_depth": float("nan"),
        "notch_params": {},
    }
    result = _fit_notch(x, y, center, 2 * hwhm, offset, step / 1e6, span / 1e6) if success else None
    if result is not None:
        (a_n, depth_n, center_n, kappa_n), perr_n = result
        notch = {
            "kappa": float(kappa_n) * 1e6,
            "kappa_error": float(perr_n[3]) * 1e6,
            "notch_depth": float(depth_n),
            # In SI units (V, Hz on the detuning axis), for plotting with notch_amplitude.
            "notch_params": {
                "a": float(a_n) * 1e-3,
                "depth": float(depth_n),
                "center": float(center_n) * 1e6,
                "kappa": float(kappa_n) * 1e6,
            },
        }
    return {
        "frequency": frequency,
        "frequency_error": frequency_error,
        "fwhm": fwhm,
        "snr": float(snr),
        **notch,
        # In SI units (Hz on the detuning axis, volts), for plotting with lorentzian_dip.
        "fit_params": {
            "amplitude": float(depth) * 1e-3,
            "center": float(center) * 1e6,
            "width": float(hwhm) * 1e6,
            "offset": float(offset) * 1e-3,
        },
        "success": success,
        sweep_range_key("detuning"): failed[sweep_range_key("detuning")],
        **outlier_summary(mask),
        "gates": gates,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "02a_resonator_spectroscopy %s: f_res = %.6f GHz +/- %.1f kHz, dip FWHM = %.2f MHz, "
        "kappa (notch fit) = %.3f +/- %.3f MHz, notch depth = %.3f, SNR = %.0f",
        status,
        fit["frequency"] / 1e9,
        fit["frequency_error"] / 1e3,
        fit["fwhm"] / 1e6,
        fit["kappa"] / 1e6,
        fit["kappa_error"] / 1e6,
        fit["notch_depth"],
        fit["snr"],
    )




# ---------------------------------------------------------------- figure (Layout C)
NODE_NAME = "02a_resonator_spectroscopy"
GHZ_AXIS_ROOM = 0.6  # extra bottom margin (inches) for the GHz axis under the detuning axis
GHZ_AXIS_OUTWARD = 36  # points between the detuning axis spine and the GHz axis spine
# One column: |IQ| on top, its residuals, then the phase panel, all sharing the detuning axis.
# Stacking keeps the figure about 10 in (1600 px) wide, so it scales like the other node figures.
PANEL_W_IN = 5.6  # column width, inches
ROW_H_IN = (2.9, 0.95, 1.7)  # |IQ|, residual and phase row heights, inches
# Noiseless kappa bias of the notch fit at the default 1 us readout: ring-up of the resonator
# field inside the integration window (+29.4 % for |g> alone) plus the thermal |e> dip 2 chi below
# (simulator/device.py: thermal_excited_population = 0.012). The noiseless check is
# tests/test_figure_02a_resonator_spectroscopy.py::test_02a_kappa_bias_matches_label.
KAPPA_BIAS = f"+30.7{NBSP}%"


def sequence_blocks(state: dict | None, fit: dict) -> list[SeqBlock]:
    """Pulse schematic of one sweep point: no qubit pulse, a readout at the swept frequency, ring-down.

    ``state`` gives the readout length and ``fit["detuning_range"]`` (Hz) the swept detuning. The
    range is written in the readout box rather than as a ``swept`` bracket, which would sit on
    the qubit line above the resonator line.
    """
    readout = "readout"
    try:
        length_ns = state["qubits"][QUBIT]["resonator"]["operations"]["readout"]["length"]
        readout = f"readout {length_ns / 1e3:g}{NBSP}{US}"
    except (KeyError, TypeError):
        pass
    sweep = fit.get(sweep_range_key("detuning"))
    if sweep is not None:
        lo, hi = (v / 1e6 for v in sweep)
        readout += f", detuning {lo:.2f} to {hi:+.2f}{NBSP}MHz".replace("-", "−")
    return [
        SeqBlock("idle", "qubit undriven", width=1.4, line="qubit"),
        SeqBlock("readout", readout, width=3.0, line="resonator"),
        SeqBlock("wait", "ring-down", width=1.1, line="resonator"),
    ]


def _finite(*values) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def _kappa_text(fit: dict, stat: str = " (stat.)") -> str:
    """The one place kappa is formatted with its error, so every label carries the same number.

    The error is statistical only (see KAPPA_BIAS); the plot label drops the "(stat.)"
    suffix to stay inside the axes, and the Measured card says it.
    """
    return f"{KAPPA} (notch fit) {fit['kappa'] / 1e6:.3f} ± {fit['kappa_error'] / 1e6:.3f}{NBSP}MHz{stat}"


def _verdict(fit: dict) -> str:
    gate = f"{MIN_DIP_SNR:g}{SIGMA} gate"
    snr = fit.get("snr", 0.0)
    snr = snr if _finite(snr) else 0.0
    if fit["success"]:
        return f"accepted: dip SNR {snr:.0f} ≥ {gate}"
    word = "rejected fit" if _finite(fit.get("frequency")) else "fit failed"
    if new_gate_reason(fit):
        return f"{word}: {new_gate_reason(fit)}"
    if snr >= MIN_DIP_SNR:
        return f"{word}: dip SNR {snr:.0f} ≥ {gate},\nwidth or position gate failed"
    return f"{word}: dip SNR {snr:.1f} < {gate}"


def node_cards(fit: dict, context: FigureContext | None) -> list[Card]:
    """Measured, true value (graph runs with a reference only) and state change cards."""
    ok = bool(fit["success"])
    if ok:
        measured = [
            f"f_r = {fit['frequency'] / 1e9:.6f}{NBSP}GHz ± {fit['frequency_error'] / 1e3:.1f}{NBSP}kHz (stat.)",
            f"dressed {KET_G}: qubit undriven, low readout power",
            f"dip FWHM {fit['fwhm'] / 1e6:.2f}{NBSP}MHz, not {KAPPA}",
        ]
        if _finite(fit.get("kappa"), fit.get("kappa_error")):
            measured += [
                _kappa_text(fit),
                f"1{NBSP}{US} window ring-up and thermal {KET_E}: {KAPPA} {KAPPA_BIAS}",
            ]
        else:
            measured.append("notch fit failed: no linewidth")
    else:
        measured = [_verdict(fit).replace("\n", " "), "state not updated"]  # the card wraps itself
    cards = [Card("Measured", tuple(measured), "ok" if ok else "fail")]
    if context is None:
        return cards

    true = (context.reference or {}).get("resonator.f_01")
    if context.in_graph and _finite(true):
        lines = [f"f_r = {true / 1e9:.6f}{NBSP}GHz (dressed {KET_G})"]
        if ok:
            diff = fit["frequency"] - true
            # U+2212 for negative signs, as in the pulse-sequence label.
            lines.append(
                f"fit − true = {diff / 1e3:+.1f}{NBSP}kHz = {diff / fit['frequency_error']:+.1f}{NBSP}{SIGMA}".replace("-", "−")
            )
            if _finite(fit.get("kappa")):
                lines.append(
                    f"= {diff / fit['kappa'] * 100:+.2f}{NBSP}% of {KAPPA} {fit['kappa'] / 1e6:.3f}{NBSP}MHz".replace("-", "−")
                )
        else:
            lines.append("no accepted fit to compare")
        cards.append(Card("True value", tuple(lines), "info"))

    def f_res(state: dict | None) -> float | None:
        return None if state is None else state["qubits"][QUBIT]["resonator"]["f_01"]

    change = format_change("resonator.f_01", f_res(context.before), f_res(context.after), 1e-9, "GHz", "{:.6f}")
    word = "rejected fit" if _finite(fit.get("frequency")) else "fit failed"  # as in _verdict
    lines = [change] if ok else [change, f"unchanged: {word}"]
    cards.append(Card("State change", tuple(lines)))
    return cards


def _span_bar(ax, x0: float, x1: float, y: float, colour: str, text: str, text_x: float) -> None:
    """Horizontal width bar with end ticks, labelled right of ``text_x`` or, if that would cross
    the right spine, left of the bar."""
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x1, y),
        arrowprops={"arrowstyle": "|-|", "color": colour, "lw": 1.0, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 3.5},
        zorder=4,
    )
    label = ax.annotate(
        text, xy=(text_x, y), xytext=(6, 0), textcoords="offset points", color=colour,
        fontsize=8.0, ha="left", va="center", zorder=4,
    )
    renderer = ax.figure.canvas.get_renderer()
    if label.get_window_extent(renderer).x1 > ax.get_window_extent(renderer).x1:
        label.xy = (x0, y)
        label.set_position((-6, 0))
        label.set_ha("right")


def _ghz_axis(ax, guess_hz: float, theme: str) -> None:
    """Absolute readout frequency axis under ``ax``'s detuning axis (node_layout bottom_extra)."""
    p = palette(theme)
    sec = ax.secondary_xaxis(
        "bottom",
        functions=(
            lambda mhz: guess_hz / 1e9 + np.asarray(mhz) / 1e3,
            lambda ghz: (np.asarray(ghz) - guess_hz / 1e9) * 1e3,
        ),
    )
    sec.spines["bottom"].set_position(("outward", GHZ_AXIS_OUTWARD))
    sec.xaxis.set_major_locator(MaxNLocator(5))
    sec.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    sec.spines["bottom"].set_color(p.muted)
    sec.spines["bottom"].set_linewidth(0.6)
    sec.tick_params(axis="x", colors=p.muted, labelcolor=p.muted, labelsize=8.0)
    sec.set_xlabel("readout frequency (GHz)", color=p.fg, fontsize=9.0)


def _hide_from_legend(ax) -> None:
    for line in ax.get_lines():
        line.set_label("_")


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: Marks | None = None,
    theme: str = "light",
    context: FigureContext | None = None,
):
    """Layout C figure: |IQ| with both fits and their widths, residuals, phase, and the cards.

    The three panels are stacked in one column and share the detuning axis.

    The x axis is the detuning from the previous readout setpoint in MHz, with the absolute
    frequency in GHz underneath. Both widths are drawn as bars on their own curves: the
    Lorentzian dip FWHM at half dip depth, and kappa at half maximum of 1 - |S21|^2, which is
    exactly Lorentzian with FWHM kappa for the notch model (Probst et al. 2015, Eq. 1, phi = 0).
    A rejected fit (``success`` False with finite values) is drawn faded and labelled as such,
    without widths, shift arrow or fitted marker.
    """
    p = palette(theme)
    right = MARGIN_RIGHT + CARDS_GAP + CARDS_W
    figsize = (
        MARGIN_LEFT + PANEL_W_IN + right,
        NODE_MAIN_TOP + sum(ROW_H_IN) + 2 * GAP_H + MARGIN_BOTTOM + GHZ_AXIS_ROOM,
    )
    lay = node_layout(theme, residuals=True, cards=True, figsize=figsize, bottom_extra=GHZ_AXIS_ROOM, residual_rows=2)
    fig = lay.fig
    ax = lay.main_axes[0]
    res, ax_phase = lay.residual_strips[0]
    # The second strip is the phase panel, taller than a residual strip. Changing the ratios
    # keeps their mean, so the row gaps stay GAP_H.
    ax.get_subplotspec().get_gridspec().set_height_ratios(ROW_H_IN)
    ax.set_gid("main")
    res.set_gid("residuals")

    guess = float(ds.attrs.get("readout_frequency_guess", np.nan))
    x = (ds.full_freq.values - guess) / 1e6
    y = ds.IQ_abs.sel(qubit=QUBIT).values * 1e3
    xs = np.linspace(x[0], x[-1], 1200)
    ok = bool(fit["success"])

    if context is None:
        progress_strip(lay.progress_ax, [NODE_NAME], NODE_NAME, {NODE_NAME: "ok" if ok else "fail"}, theme)
    else:
        progress_strip(lay.progress_ax, context.node_order, NODE_NAME, context.statuses, theme)
    pulse_sequence(lay.sequence_ax, sequence_blocks(context.before if context else None, fit), theme)

    # |IQ| panel: data, the Lorentzian fit (gives f_r) and the notch fit (gives kappa).
    ax.plot(x, y, "o", color=p.data, markersize=2.6, markeredgewidth=0, label="data", zorder=2)
    lorentz = fit.get("fit_params") or {}
    notch = fit.get("notch_params") or {}
    if lorentz:
        model = lorentzian_dip(xs * 1e6, lorentz["amplitude"], lorentz["center"], lorentz["width"], lorentz["offset"])
        if ok:
            ax.plot(xs, model * 1e3, color=p.fit, linewidth=1.8, label="Lorentzian fit", zorder=3)
        else:
            ax.plot(xs, model * 1e3, color=p.fail, linewidth=1.2, alpha=0.7, linestyle=(0, (4, 2)),
                    label="rejected fit", zorder=3)
    if notch:
        ax.plot(
            xs, notch_amplitude(xs * 1e6, **notch) * 1e3, color=p.fg, linewidth=1.1,
            linestyle=(0, (3, 2)), label="notch fit", zorder=3.5,
        )

    if marks is not None:
        def shifted(value):
            return None if value is None else value - guess

        fitted = shifted(marks.fit) if ok else None
        # True value first, so the fitted marker (often a few kHz away) is drawn on top of it.
        for target in (ax, ax_phase):
            mark_values(target, Marks(true=shifted(marks.true)), theme, scale=1e-6, true_label="true f_r")
            mark_values(target, Marks(previous=shifted(marks.previous), fit=fitted), theme, scale=1e-6,
                        previous_label="previous f_r", fit_label="fitted f_r")
        _hide_from_legend(ax_phase)

    y_lo, y_hi = float(np.min(y)), float(np.max(y))
    span = y_hi - y_lo
    ax.set_ylim(y_lo - 0.34 * span, y_hi + 0.06 * span)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel("|IQ| (mV)")

    if ok and lorentz:
        centre, hwhm = lorentz["center"] / 1e6, lorentz["width"] / 1e6
        half = (lorentz["offset"] - lorentz["amplitude"] / 2) * 1e3
        right = centre + max(hwhm, notch.get("kappa", 0.0) / 2e6)
        _span_bar(ax, centre - hwhm, centre + hwhm, half, p.fit, f"dip FWHM {fit['fwhm'] / 1e6:.2f}{NBSP}MHz", right)
        if notch and _finite(fit.get("kappa"), fit.get("kappa_error")):
            k, c = notch["kappa"] / 1e6, notch["center"] / 1e6
            level = notch["a"] * np.sqrt((1 + (1 - notch["depth"]) ** 2) / 2) * 1e3
            _span_bar(ax, c - k / 2, c + k / 2, level, p.fg, _kappa_text(fit, stat=""), right)
        # The correction from the previous setpoint, under the dip.
        shift = (fit["frequency"] - guess) / 1e6
        y_arrow = y_lo - 0.12 * span
        ax.annotate(
            "", xy=(shift, y_arrow), xytext=(0.0, y_arrow),
            arrowprops={"arrowstyle": "-|>", "color": p.guess, "lw": 1.0, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7},
            zorder=4,
        )
        ax.annotate(
            # Beside the start of the arrow, on the side away from it, clear of the marker lines.
            f"shift {shift:+.2f}{NBSP}MHz", xy=(0.0, y_arrow), xytext=(-5 if shift >= 0 else 5, 0),
            textcoords="offset points", color=p.guess, fontsize=8.0,
            ha="right" if shift >= 0 else "left", va="center", zorder=4,
        )

    ax.text(
        0.015, 0.035, _verdict(fit), transform=ax.transAxes, color=p.ok if ok else p.fail,
        fontsize=8.0, fontweight="medium", ha="left", va="bottom", zorder=5,
        # Opaque backing so the marker lines it crosses do not run through the text.
        bbox={"boxstyle": "round,pad=0.25", "facecolor": p.bg, "edgecolor": "none", "alpha": 0.9},
    )
    place_legend(ax, theme, ncols=3, title="|IQ| dip")

    # Residuals against the notch fit (the physical line shape), else the Lorentzian.
    model_at_x, which = None, ""
    if notch:
        model_at_x, which = notch_amplitude(x * 1e6, **notch) * 1e3, "notch"
    elif lorentz and ok:
        model_at_x = lorentzian_dip(x * 1e6, lorentz["amplitude"], lorentz["center"], lorentz["width"], lorentz["offset"]) * 1e3
        which = "Lorentzian"
    if model_at_x is not None:
        r = y - model_at_x
        plot_residuals(res, x, r / np.std(r), theme)
        res.set_ylabel(f"(data − {which}) / {SIGMA}\n{SIGMA}: residual std", fontsize=7.5)
    else:
        res.set_yticks([])
        res.grid(False)
        res.text(0.5, 0.5, "no accepted fit, no residuals", transform=res.transAxes, color=p.muted,
                 fontsize=8.0, ha="center", va="center")

    # Phase: the linear slope (cable delay) is removed, leaving the S-shaped excursion through f_r.
    # It sits under the residuals with no legend row above it, so its title goes inside the axes.
    phase = ds.phase.sel(qubit=QUBIT).values
    ax_phase.plot(x, phase, "o", color=p.data, markersize=2.6, markeredgewidth=0, zorder=2)
    ax_phase.set_xlim(x[0], x[-1])
    p_lo, p_hi = float(np.nanmin(phase)), float(np.nanmax(phase))
    p_span = (p_hi - p_lo) or 1.0
    ax_phase.set_ylim(p_lo - 0.08 * p_span, p_hi + 0.32 * p_span)
    ax_phase.yaxis.set_major_locator(MaxNLocator(4))
    ax_phase.set_ylabel("phase (rad)")
    ax_phase.set_xlabel("detuning from previous f_r (MHz)")
    ax_phase.text(
        0.012, 0.94, "phase, cable delay removed", transform=ax_phase.transAxes, color=p.fg,
        fontsize=8.0, fontweight="semibold", ha="left", va="top", zorder=5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": p.bg, "edgecolor": "none", "alpha": 0.9},
    )
    _ghz_axis(ax_phase, guess, theme)

    cards_column(lay.cards_ax, node_cards(fit, context), theme)
    if ok:
        title_block(fig, NODE_NAME, f"Readout resonator, dressed {KET_G} frequency",
                    f"f_r = {fit['frequency'] / 1e9:.6f}{NBSP}GHz", theme)
    else:
        headline = "rejected fit" if _finite(fit.get("frequency")) else "fit failed"
        title_block(fig, NODE_NAME, "Readout resonator dip", f"{headline}, state not updated", theme, status="fail")
    return fig
