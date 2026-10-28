"""Analysis for 07_iq_blobs, using QM's ``two_state_discriminator`` from py-qua-tools.

Angle convention. ``two_state_discriminator`` returns an angle theta and rotates the shots as

    I_rot = I cos(theta) - Q sin(theta),    Q_rot = I sin(theta) + Q cos(theta)

with the sign chosen so that |e> lies at larger I_rot than |g>, and a threshold on I_rot
(``I_rot > threshold`` means |e>). The ``rotated_*`` integration weights in
:func:`qua.config.build_config` use an angle a as rotated_cos = (cos a, -sin a) and
rotated_sin = (sin a, cos a), so ``dual_demod.full("rotated_cos", "rotated_sin", I)`` yields
I cos(a) + Q sin(a). The hardware rotation equals the discriminator's when a = -theta, which is
why QM's node does ``integration_weights_angle -= iw_angle``. Our shots are taken with the
unrotated weights (a = 0), so the stored angle is simply -theta.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.patches import Arc
from qualang_tools.analysis.discriminator import two_state_discriminator

from calibration_utils import style
from calibration_utils.common import (
    MAX_SHOT_OUTLIER_FRACTION,
    MIN_STATIONARITY_PVALUE,
    QUBIT,
    new_gate_reason,
)
from calibration_utils.robust import block_homogeneity_pvalue, shot_outliers

log = logging.getLogger(__name__)

MIN_FIDELITY = 0.8

# The OPX returns demodulated I/Q as a fixed-point sum over the readout window. QM's
# ``convert_IQ_to_V`` multiplies by 2**12 / length to get volts, and their node divides the
# threshold by the same factor before storing it, because QUA compares it with raw demod output.
DEMOD_UNITS_PER_VOLT_PER_NS = 1 / 2**12


def volts_to_demod(value_v: float, readout_length_ns: float) -> float:
    return value_v * readout_length_ns * DEMOD_UNITS_PER_VOLT_PER_NS


def demod_to_volts(value: float, readout_length_ns: float) -> float:
    return value / (readout_length_ns * DEMOD_UNITS_PER_VOLT_PER_NS)


def rotate(I, Q, theta: float):
    """The discriminator's rotation: puts the g-e axis along I with |e> at larger I."""
    c, s = np.cos(theta), np.sin(theta)
    return I * c - Q * s, I * s + Q * c


def wrap_angle(angle: float) -> float:
    """Map an angle to (-pi, pi]."""
    return float(-((-angle + np.pi) % (2 * np.pi) - np.pi))


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Shots already arrive in volts (QM's ``convert_IQ_to_V`` step is a no-op here)."""
    return ds


def fit_raw_data(ds: xr.Dataset) -> dict:
    """QM's two-state discriminator on the shots, after excluding outlier shots per preparation
    (``shot_outliers``; if the kept counts differ, both are cut to the shorter from the end).

    Gates, in order: ``finite``, ``fidelity``, ``outlier_shots`` (at most MAX_SHOT_OUTLIER_FRACTION
    excluded per preparation), ``stationary`` (the wrong-assignment fraction is the same in 20
    consecutive blocks of each preparation, p at least MIN_STATIONARITY_PVALUE: a quasiparticle
    burst or another transient during the run fails it).
    """
    d = ds.sel(qubit=QUBIT)
    Ig, Qg, Ie, Qe = (d[v].values for v in ("I_g", "Q_g", "I_e", "Q_e"))
    out_g, out_e = shot_outliers(Ig, Qg), shot_outliers(Ie, Qe)
    if out_g.any() or out_e.any():
        n = min(int((~out_g).sum()), int((~out_e).sum()))
        Ig, Qg = Ig[~out_g][:n], Qg[~out_g][:n]
        Ie, Qe = Ie[~out_e][:n], Qe[~out_e][:n]
    theta, threshold, fidelity_pct, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=False, b_plot=False)
    theta = wrap_angle(theta)
    fidelity = float(fidelity_pct) / 100
    # Repeat-until-success threshold, as in QM's node: the peak of the |g> histogram on rotated I.
    Ig_rot, _ = rotate(Ig, Qg, theta)
    counts, edges = np.histogram(Ig_rot, bins=100)
    rus_threshold = float(edges[1:][np.argmax(counts)])
    values = np.array([theta, threshold, fidelity])
    Ie_rot, _ = rotate(Ie, Qe, theta)
    if np.isfinite(threshold):
        pvalue = min(block_homogeneity_pvalue(Ig_rot > threshold), block_homogeneity_pvalue(Ie_rot < threshold))
    else:
        pvalue = float("nan")
    excluded = max(float(out_g.mean()), float(out_e.mean()))
    gates = {
        "finite": bool(np.all(np.isfinite(values))),
        "fidelity": bool(fidelity > MIN_FIDELITY),
        "outlier_shots": excluded <= MAX_SHOT_OUTLIER_FRACTION,
        "stationary": bool(np.isfinite(pvalue) and pvalue >= MIN_STATIONARITY_PVALUE),
    }
    success = all(gates.values())
    # Blob means for the P(e) axis of later nodes. No unit conversion: these shots and the
    # averaged nodes' ds.I/ds.Q both come from SimulatedQPU (measure_shots, measure_averaged) in volts
    # (process_raw_dataset above is a no-op for the same reason). The demodulation-unit factor
    # (demod_to_volts) applies only to the threshold stored in the state for QUA.
    return {
        "g_mean_iq": [float(np.mean(Ig)), float(np.mean(Qg))],
        "e_mean_iq": [float(np.mean(Ie)), float(np.mean(Qe))],
        "rotation_angle": theta,
        "iw_angle": wrap_angle(-theta),
        "ge_threshold": float(threshold),
        "rus_threshold": rus_threshold,
        "fidelity": fidelity,
        "confusion_matrix": [[float(gg), float(ge)], [float(eg), float(ee)]],
        "n_outliers": {"g": int(out_g.sum()), "e": int(out_e.sum())},
        "outlier_shot_fraction": excluded,
        "stationarity_pvalue": pvalue,
        "gates": gates,
        "success": success,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    log.info(
        "07_iq_blobs %s: IW angle = %.1f deg, threshold = %.3f mV, fidelity = %.1f %% (gg %.3f, ge %.3f, eg %.3f, ee %.3f)",
        status,
        np.degrees(fit["iw_angle"]),
        fit["ge_threshold"] * 1e3,
        fit["fidelity"] * 100,
        gg,
        ge,
        eg,
        ee,
    )


# ---------------------------------------------------------------- figure (Layout C)
NODE_NAME = "07_iq_blobs"
# Main row: IQ scatter and rotated-I histogram; the confusion matrix is the Assignment card.
PAGE_W, PANEL_H = 9.8, 5.8  # page width and main-row height (tall enough for the cards at full size), inches
WIDTH_RATIOS = (0.9, 1.1)
G, E = style.KET_G, style.KET_E
NB = style.NBSP


def _finite(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _readout_length(state: dict | None, ds: xr.Dataset) -> float | None:
    """Readout window in ns, from the state if known, else from the dataset attributes."""
    if state is not None:
        return float(state["qubits"][QUBIT]["resonator"]["operations"]["readout"]["length"])
    length = ds.attrs.get("readout_length")
    return float(length) if length is not None else None


def _error_budget(ds: xr.Dataset, fit: dict) -> dict | None:
    """Split the two assignment errors into thermal, decay-in-window and Gaussian-overlap parts.

    Every number comes from the shots, rotated and thresholded as the fit did:
        - blob centres mu_g, mu_e: medians of rotated I (robust to the few thermal and decayed shots);
        - sigma: spread of rotated Q, pooled over both preparations. Thermal shots and the decay
          streak lie along rotated I, so rotated Q holds only the readout noise;
        - overlap: Gaussian tail of each blob past the threshold, Phi(-|thr - mu| / sigma);
        - thermal: |g>-prepared shots in the |e> region, P(e|g), less the |g> overlap;
        - decay in window (+ pi-pulse error): what is left of P(g|e) after the thermal part
          (thermal |e> shots that the x180 sends to |g>) and the |e> overlap. In the simulator this
          is mostly |e> relaxing during the readout window, which smears a shot along the line
          between the blobs as the resonator field relaxes to the |g> response
          (simulator.readout.decay_window_sums); an x180 error lands here too.
    Returns None when the fit has no finite angle or threshold.
    """
    theta, thr = fit.get("rotation_angle"), fit.get("ge_threshold")
    if not _finite(theta, thr):
        return None
    d = ds.sel(qubit=QUBIT)
    Ig, Qg = rotate(d.I_g.values, d.Q_g.values, theta)
    Ie, Qe = rotate(d.I_e.values, d.Q_e.values, theta)
    mu_g, mu_e = float(np.median(Ig)), float(np.median(Ie))
    sigma = float(np.sqrt((np.var(Qg) + np.var(Qe)) / 2))

    def tail(z: float) -> float:  # P(X > mu + z sigma) for a Gaussian
        return 0.5 * math.erfc(z / math.sqrt(2))

    p_eg, p_ge = float(np.mean(Ig > thr)), float(np.mean(Ie <= thr))
    overlap_g, overlap_e = tail((thr - mu_g) / sigma), tail((mu_e - thr) / sigma)
    thermal = p_eg - overlap_g
    separation = mu_e - mu_g
    return {
        "mu_g": mu_g,
        "mu_e": mu_e,
        "q_g": float(np.median(Qg)),
        "q_e": float(np.median(Qe)),
        "sigma": sigma,
        "separation": separation,
        "separation_sigma": separation / sigma,
        "snr": separation / (2 * sigma),
        "p_e_given_g": p_eg,
        "p_g_given_e": p_ge,
        "overlap_g": overlap_g,
        "overlap_e": overlap_e,
        "thermal": thermal,
        "decay": p_ge - thermal - overlap_e,
        "n_g": int(Ig.size),
        "n_e": int(Ie.size),
    }


def _t1_hint(context: style.FigureContext | None) -> tuple[float, str] | None:
    """A T1 to compare the decay estimate with: measured in the state, else the true value."""
    if context is None:
        return None
    for state in (context.before, context.after):
        t1 = (state or {}).get("qubits", {}).get(QUBIT, {}).get("T1")
        if t1 is not None and math.isfinite(t1):
            return float(t1), "measured"
    if context.in_graph and context.reference and _finite(context.reference.get("T1")):
        return float(context.reference["T1"]), "true value"
    return None


def sequence_blocks(state: dict | None, fit: dict) -> list[style.SeqBlock]:
    """The two preparations, one shot each: thermalize then read out, thermalize, x180, read out."""
    qubit = (state or {}).get("qubits", {}).get(QUBIT, {})
    wait_ns = qubit.get("thermalization_time")
    length = qubit.get("resonator", {}).get("operations", {}).get("readout", {}).get("length")
    wait = f"thermalize {wait_ns / 1e3:.0f} {style.US}" if wait_ns else "thermalize"
    readout = f"readout {length / 1e3:.1f} {style.US}" if length else "readout"
    return [
        style.SeqBlock("wait", wait, width=1.7),
        style.SeqBlock("readout", readout, width=1.5, line="resonator"),
        style.SeqBlock("idle", "", width=0.4),
        style.SeqBlock("wait", wait, width=1.7),
        style.SeqBlock("drive", "x180", width=0.8),
        style.SeqBlock("readout", readout, width=1.5, line="resonator"),
    ]


def _fidelities(fit: dict) -> tuple[float, float]:
    """QM's fidelity (gg + ee)/2 and the measurement fidelity 1 - P(e|g) - P(g|e)."""
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    return (gg + ee) / 2, 1 - ge - eg


def _fit_state(fit: dict) -> str:
    """"ok", "rejected" (finite values that failed the fidelity gate) or "failed" (no values)."""
    if fit.get("success"):
        return "ok"
    values = (fit.get("rotation_angle"), fit.get("ge_threshold"), fit.get("fidelity"))
    return "rejected" if _finite(*values) else "failed"


def _minus(text: str) -> str:
    """A formatted number with a typographic minus (U+2212), as the other nodes write them."""
    return text.replace("-", "−")


def _minus_numbers(line: str) -> str:
    """U+2212 for the sign of every number in a ``style.format_change`` line; "->" stays."""
    return re.sub(r"-(?=\d)", "−", line)


def node_cards(
    fit: dict, context: style.FigureContext | None, ds: xr.Dataset | None = None
) -> list[style.Card]:
    """Measured, simulator reference (graph runs only), assignment, error budget, state change.

    ``ds`` (the node's shot dataset) is needed for the error budget card, the blob separation
    and the statistical errors; without it those are left out. The readout reference is the
    "simulator reference" (a high-shot run with the true pi pulse), never a true value. Errors
    are binomial (statistical only, tagged "(stat.)"); x180 error and decay during readout are
    systematic and show in the error budget instead. Lines are kept short enough for the shared
    card text size, so the column does not shrink its font.
    """
    cards: list[style.Card] = []
    state = _fit_state(fit)
    f_assign, f_meas = _fidelities(fit)
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    # The budget and separation describe an accepted discrimination only; a rejected fit gets none.
    budget = _error_budget(ds, fit) if ds is not None and state == "ok" else None
    n_g = n_e = None
    if ds is not None:
        n_g = n_e = int(ds.sizes["n_runs"])
    err_assign = err_meas = None
    if n_g and _finite(gg, ee):
        var_g, var_e = gg * (1 - gg) / n_g, ee * (1 - ee) / n_e
        err_assign, err_meas = 0.5 * math.sqrt(var_g + var_e), math.sqrt(var_g + var_e)

    def pct(value: float, err: float | None = None) -> str:
        if not _finite(value):
            return "uncalculated"
        if _finite(err):
            return f"{100 * value:.1f} ± {100 * err:.1f}{NB}% (stat.)"
        return f"{100 * value:.1f}{NB}%"

    thr, iw = fit.get("ge_threshold"), fit.get("iw_angle")
    lines: list[str] = []
    if state == "rejected" and new_gate_reason(fit):
        lines.append(f"rejected fit: {new_gate_reason(fit).rstrip('?')}; state not updated")
    elif state == "rejected":
        lines.append(f"rejected fit: fidelity below {100 * MIN_FIDELITY:.0f}{NB}%, state not updated")
    elif state == "failed":
        lines.append("fit failed, state not updated")
    lines += [
        f"threshold {_minus(f'{thr * 1e3:.3f}')}{NB}mV on rotated I" if _finite(thr) else "threshold uncalculated",
        f"weight angle {_minus(f'{math.degrees(iw):.1f}')} deg" if _finite(iw) else "weight angle uncalculated",
    ]
    if budget is not None:
        s = style.SIGMA
        lines += [
            f"separation d = {budget['separation'] * 1e3:.3f}{NB}mV = {budget['separation_sigma']:.1f} {s}",
            f"SNR = d/(2{s}) = {budget['snr']:.1f}, {s} = {budget['sigma'] * 1e3:.3f}{NB}mV",
        ]
    cards.append(style.Card("Measured", tuple(lines), "ok" if state == "ok" else "fail"))

    ref = context.reference if context is not None and context.in_graph else None
    if ref and all(k in ref for k in ("fidelity", "ge_threshold", "iw_angle")):
        rf, rthr, riw = ref["fidelity"], ref["ge_threshold"], ref["iw_angle"]
        d_f = f_assign - rf if _finite(f_assign) else float("nan")
        sig = f" = {abs(d_f) / err_assign:.1f} {style.SIGMA} (stat.)" if _finite(d_f, err_assign) and err_assign > 0 else ""
        d_thr = (thr - rthr) * 1e3 if _finite(thr) else float("nan")
        d_iw = math.degrees(wrap_angle(iw - riw)) if _finite(iw) else float("nan")
        thr_line = f"threshold {_minus(f'{rthr * 1e3:.3f}')}{NB}mV"
        iw_line = f"weight angle {_minus(f'{math.degrees(riw):.1f}')} deg"
        cards.append(
            style.Card(
                "Simulator reference",
                (
                    f"(gg + ee)/2 = {100 * rf:.1f}{NB}%",
                    f"measured {_minus(f'{100 * d_f:+.1f}')}{NB}pp{sig}" if _finite(d_f) else "measured: uncalculated",
                    f"{thr_line} ({_minus(f'{d_thr:+.3f}')}{NB}mV)" if _finite(d_thr) else thr_line,
                    f"{iw_line} ({_minus(f'{d_iw:+.1f}')} deg)" if _finite(d_iw) else iw_line,
                    # Fidelity at the g/e midpoint; angle and threshold turn with the tone.
                    "angle, threshold at this run's tone" if "reference_tone" in ref else "high-shot run, true pi pulse",
                ),
                "info",
            )
        )

    def cell(name: str, value: float) -> str:
        return f"{name} {100 * value:.1f}{NB}%" if _finite(value) else f"{name} n/a"

    cards.append(
        style.Card(
            "Assignment",
            (
                "rows prepared, columns measured",
                f"{cell('gg', gg)}    {cell('ge', ge)}",
                f"{cell('eg', eg)}    {cell('ee', ee)}",
                "(gg + ee)/2, QM's fidelity, stored:",
                pct(f_assign, err_assign),
                "1 - P(e|g) - P(g|e), measurement fidelity:",
                pct(f_meas, err_meas),
            ),
        )
    )

    if budget is not None:
        b = {
            k: _minus(f"{100 * budget[k]:.1f}")
            for k in ("thermal", "decay", "overlap_g", "overlap_e", "p_e_given_g", "p_g_given_e")
        }
        # P(g|e) has three parts: broken after the first on purpose, so no term wraps alone.
        lines = [
            f"P(e|g) {b['p_e_given_g']}{NB}% = thermal {b['thermal']} + overlap {b['overlap_g']}",
            f"P(g|e) {b['p_g_given_e']}{NB}% = thermal {b['thermal']}",
            f"{NB * 6}+ decay {b['decay']} + overlap {b['overlap_e']}",
            f"thermal: {G} shots in the {E} region",
            "decay in window (+ pi-pulse error)",
        ]
        t1 = _t1_hint(context)
        length = _readout_length(context.after if context else None, ds)
        if t1 is not None and length is not None:
            lines.append(f"L/(2 T1) = {100 * length / (2 * t1[0]):.1f}{NB}%, T1 {t1[0] / 1e3:.0f}{NB}{style.US} ({t1[1]})")
        lines.append(f"overlap: Gaussian tails, {style.SIGMA} from rotated Q")
        cards.append(style.Card("Error budget", tuple(lines), None))

    if context is not None:

        def resonator(st):
            return (st or {}).get("qubits", {}).get(QUBIT, {}).get("resonator", {})

        before, after = resonator(context.before), resonator(context.after)
        ro_b = before.get("operations", {}).get("readout", {})
        ro_a = after.get("operations", {}).get("readout", {})
        length = ro_a.get("length") or ro_b.get("length")
        to_mv = 1e3 / (length * DEMOD_UNITS_PER_VOLT_PER_NS) if length else float("nan")
        key = "integration_weights_angle"
        changes = (
            style.format_change("weight angle", ro_b.get(key), ro_a.get(key), 180 / math.pi, "deg", "{:.1f}"),
            style.format_change("threshold", ro_b.get("threshold"), ro_a.get("threshold"), to_mv, "mV", "{:.3f}"),
            style.format_change(
                "readout fidelity", before.get("readout_fidelity"), after.get("readout_fidelity"), 100, "%", "{:.1f}"
            ),
        )
        cards.append(style.Card("State change", tuple(_minus_numbers(c) for c in changes)))
    return cards


def _limits(x: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = np.percentile(x, [0.1, 99.9])
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def _plot_iq(ax, Ig, Qg, Ie, Qe, fit, budget, context, ds, theme) -> None:
    """Raw single shots with the threshold line, the weight-angle arc and two callouts (mV)."""
    p = style.palette(theme)
    ax.plot(Ie, Qe, ".", color=p.excited, ms=1.6, alpha=0.28, mew=0, rasterized=True)
    ax.plot(Ig, Qg, ".", color=p.ground, ms=1.6, alpha=0.28, mew=0, rasterized=True)
    ax.plot([], [], "o", color=p.ground, ms=4, label=f"{G} prepared")
    ax.plot([], [], "o", color=p.excited, ms=4, label=f"{E} prepared")
    ax.set_xlim(*_limits(np.concatenate([Ig, Ie])))
    ax.set_ylim(*_limits(np.concatenate([Qg, Qe])))
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("I (mV)")
    ax.set_ylabel("Q (mV)")
    theta, thr = fit.get("rotation_angle"), fit.get("ge_threshold")
    if budget is None or not _finite(theta, thr):
        return
    thr_mv = thr * 1e3
    accepted = _fit_state(fit) == "ok"

    def raw(i_rot: float, q_rot: float) -> np.ndarray:
        """A rotated-frame point (mV) back in the raw IQ plane."""
        return np.array(rotate(i_rot, q_rot, -theta))

    q_mid = 1e3 * (budget["q_g"] + budget["q_e"]) / 2
    g_c, e_c = raw(1e3 * budget["mu_g"], q_mid), raw(1e3 * budget["mu_e"], q_mid)
    sep = float(np.hypot(*(e_c - g_c)))
    # Decision boundary: rotated I = threshold, a straight line in the raw plane.
    along = np.array(rotate(0.0, 1.0, -theta))  # rotated-Q direction in raw I/Q
    centre = raw(thr_mv, q_mid)
    span = 10 * float(np.ptp(np.concatenate([Ig, Ie, Qg, Qe])))  # longer than any view of the shots
    line = centre[:, None] + along[:, None] * np.array([-span, span])
    ax.plot(
        line[0], line[1], color=p.fit if accepted else p.fail, lw=1.1, ls=(0, (4, 3)),
        label="threshold" if accepted else "rejected fit",
    )
    if not accepted:
        return  # no callouts or weight-angle arc for a fit that was not accepted

    # Misassigned shots, ringed so they read on top of the other cloud: |g>-prepared shots in the
    # |e> region (thermal) and |e>-prepared shots in the |g> region (mostly decay in the window).
    Ig_rot, _ = rotate(Ig, Qg, theta)
    Ie_rot, _ = rotate(Ie, Qe, theta)
    hot, cold = Ig_rot > thr_mv, Ie_rot <= thr_mv
    ring = {"ms": 3.0, "mfc": "none", "mew": 0.7, "zorder": 3, "linestyle": "none"}
    ax.plot(Ig[hot], Qg[hot], "o", mec=p.ground, **ring)
    ax.plot(Ie[cold], Qe[cold], "o", mec=p.excited, **ring)
    ax.plot([], [], "o", mec=p.muted, label="misassigned", **ring)
    for c, colour in ((g_c, p.ground), (e_c, p.excited)):
        ax.plot(*c, "o", ms=6, color=colour, mec=p.fg, mew=0.8, zorder=4)

    # Weight angle: from the +I direction to the g -> e direction, around the |g> centre.
    iw_deg = math.degrees(fit["iw_angle"])
    r = 0.42 * sep
    ax.plot([g_c[0], g_c[0] + 0.62 * sep], [g_c[1]] * 2, color=p.muted, lw=0.8, zorder=4)
    arrow = {"arrowstyle": "-|>", "color": p.fg, "lw": 0.9, "shrinkA": 4, "shrinkB": 5}
    ax.annotate("", xy=e_c, xytext=g_c, arrowprops=arrow, zorder=4)
    ax.add_patch(Arc(g_c, 2 * r, 2 * r, theta1=min(0, iw_deg), theta2=max(0, iw_deg), color=p.fit, lw=1.4, zorder=4))
    mid = math.radians(iw_deg / 2)

    # Decay streak target: |e>-prepared shots clear of both blobs (beyond 3 sigma of each), which
    # only decay during the window puts there; else a point on the g-e line.
    lo, hi = 1e3 * (budget["mu_g"] + 3 * budget["sigma"]), 1e3 * (budget["mu_e"] - 3 * budget["sigma"])
    between = (Ie_rot > lo) & (Ie_rot < hi)
    if between.any():
        streak = (float(np.median(Ie[between])), float(np.median(Qe[between])))
    else:
        streak = tuple(raw(1e3 * (budget["mu_g"] + 0.4 * budget["separation"]), q_mid))

    t1 = _t1_hint(context)
    length = _readout_length(context.after if context else None, ds)
    window = f"{length / 1e3:.1f} {style.US} readout" if length else "readout window"
    t1_text = f"\nT1 = {t1[0] / 1e3:.0f} {style.US}\n({t1[1]})" if t1 else ""
    callouts = [
        (f"integration_weights_angle\n= {_minus(f'{iw_deg:.1f}')} deg", (g_c[0] + r * math.cos(mid), g_c[1] + r * math.sin(mid)), p.fit, p.fit),
        (f"decay streak:\n{E} relaxes to {G}\nduring the\n{window}{t1_text}", streak, p.fg, p.excited),
    ]
    if hot.any():
        target = (float(np.median(Ig[hot])), float(np.median(Qg[hot])))
        thermal = _minus(f"{100 * budget['thermal']:.1f}")
        text = f"{G}-prepared shots\nin the {E} blob:\nthermal {E},\n{thermal} % of {G} shots"
        callouts.append((text, target, p.fg, p.ground))
    _place_callouts(ax, callouts, np.concatenate([Ig, Ie]), np.concatenate([Qg, Qe]), line, theme)


_SPOTS = (  # axes-fraction anchor and alignment: the corners first, then the edge midpoints
    ((0.02, 0.98), "left", "top"),
    ((0.98, 0.98), "right", "top"),
    ((0.02, 0.02), "left", "bottom"),
    ((0.98, 0.02), "right", "bottom"),
    ((0.02, 0.5), "left", "center"),
    ((0.98, 0.5), "right", "center"),
    ((0.5, 0.98), "center", "top"),
    ((0.5, 0.02), "center", "bottom"),
)
_CALLOUT = {"fontsize": 7.0, "linespacing": 1.25, "zorder": 6}


def _place_callouts(ax, callouts, x, y, line, theme) -> None:
    """Put each callout at its own spot along the axes edge, with a leader to its target.

    Longest text first, each callout takes the free spot (corners, then edge midpoints) whose
    text box does not cross the threshold ``line`` (2 x N, data units) or a callout already
    placed, and holds the fewest shots, so the text lands where the blobs are not whatever the
    IQ rotation, and the threshold line is never hidden under a text background.
    """
    p = style.palette(theme)
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()
    pts = to_axes.transform(ax.transData.transform(np.column_stack([x, y])))
    t = np.linspace(0.0, 1.0, 2000)[:, None]
    seg = to_axes.transform(ax.transData.transform(line[:, 0] + t * (line[:, 1] - line[:, 0])))
    seg = seg[(seg >= 0).all(axis=1) & (seg <= 1).all(axis=1)]

    def box(text: str, spot) -> tuple[float, float, float, float]:
        """The text's extent in axes fraction at ``spot``, with the background padding."""
        xy, ha, va = spot
        probe = ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, **_CALLOUT)
        (x0, y0), (x1, y1) = to_axes.transform(probe.get_window_extent(renderer).get_points())
        probe.remove()
        pad = 0.02
        return x0 - pad, y0 - pad, x1 + pad, y1 + pad

    def inside(points: np.ndarray, b) -> np.ndarray:
        return (points[:, 0] > b[0]) & (points[:, 0] < b[2]) & (points[:, 1] > b[1]) & (points[:, 1] < b[3])

    def overlaps(a, b) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    placed: list[tuple] = []
    free = list(range(len(_SPOTS)))
    for i in sorted(range(len(callouts)), key=lambda i: -len(style.plain(callouts[i][0]))):
        text, target, colour, leader = callouts[i]

        def cost(k: int, text=text) -> tuple:
            b = box(text, _SPOTS[k])
            clash = bool(inside(seg, b).any()) or any(overlaps(b, o) for o in placed)
            return clash, int(inside(pts, b).sum()), k

        k = min(free, key=cost)
        if cost(k)[0] and "_" in text:
            # Every spot crosses the threshold line or another callout (a near-vertical line through
            # a tall panel): break long names at their underscores and try again.
            narrow = "\n".join(part.replace("_weights", "_\nweights") for part in text.split("\n"))
            k_narrow = min(free, key=lambda j: cost(j, narrow))
            if not cost(k_narrow, narrow)[0]:
                text, k = narrow, k_narrow
        free.remove(k)
        placed.append(box(text, _SPOTS[k]))
        xy, ha, va = _SPOTS[k]
        ax.annotate(
            text,
            xy=target,
            xytext=xy,
            textcoords="axes fraction",
            ha=ha,
            va=va,
            color=colour,
            bbox={"boxstyle": "round,pad=0.2", "fc": p.bg, "ec": "none", "alpha": 0.85},
            arrowprops={"arrowstyle": "-|>", "color": leader, "lw": 0.7, "shrinkA": 2, "shrinkB": 2},
            **_CALLOUT,
        )


def _plot_histogram(ax, Ig_rot, Ie_rot, fit, budget, marks, theme) -> None:
    """Rotated-I histograms on log counts, with the Gaussian blob model and the thresholds."""
    p = style.palette(theme)
    bins = np.linspace(min(Ig_rot.min(), Ie_rot.min()), max(Ig_rot.max(), Ie_rot.max()), 70)
    width = bins[1] - bins[0]
    peak = 1
    for x, colour in ((Ig_rot, p.ground), (Ie_rot, p.excited)):
        counts, _, _ = ax.hist(x, bins=bins, color=colour, alpha=0.18, lw=0)
        ax.hist(x, bins=bins, histtype="step", color=colour, lw=1.2)
        peak = max(peak, counts.max())
    ax.set_yscale("log")
    ax.set_ylim(0.6, 1.6 * peak)
    if budget is not None:
        xs = np.linspace(bins[0], bins[-1], 400)
        s = budget["sigma"] * 1e3
        for i, (mu, n) in enumerate(((budget["mu_g"], budget["n_g"]), (budget["mu_e"], budget["n_e"]))):
            model = n * width * np.exp(-0.5 * ((xs - mu * 1e3) / s) ** 2) / (s * math.sqrt(2 * math.pi))
            ax.plot(xs, model, color=p.muted, lw=0.9, ls=(0, (2, 2)), label=None if i else "Gaussian blobs")
    accepted = _fit_state(fit) == "ok"
    rotated = _finite(fit.get("rotation_angle"))
    if marks is None:
        marks = style.Marks(fit=fit.get("ge_threshold") if accepted else None)
    if not rotated:
        # The simulator-reference threshold is a rotated-I value; on the unrotated axis it marks
        # nothing, so it is left out.
        marks = dataclasses.replace(marks, true=None)
    style.mark_values(
        ax, marks, theme, scale=1e3, previous_label="previous threshold",
        fit_label="threshold (fit)" if accepted else "rejected fit",
    )
    ax.set_xlabel("rotated I (mV)" if rotated else "I (mV), not rotated: fit failed")
    ax.set_ylabel("counts (log)")


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Layout C page: progress strip, pulse sequence, IQ shots and log histogram, cards.

    The confusion matrix and both fidelity definitions are the Assignment card. ``marks`` marks
    thresholds (volts) on the rotated-I histogram; its true value is the simulator reference.
    With ``context=None`` the strip shows this node alone and no reference or state-change card
    is drawn.
    """
    state = _fit_state(fit)
    h = style.MARGIN_BOTTOM + style.NODE_MAIN_TOP + PANEL_H
    layout = style.node_layout(
        theme, ncols=2, residuals=False, cards=True, figsize=(PAGE_W, h), width_ratios=WIDTH_RATIOS
    )
    fig = layout.fig
    ax_iq, ax_hist = layout.main_axes

    status = "ok" if state == "ok" else "fail"
    if context is not None:
        style.progress_strip(layout.progress_ax, context.node_order, NODE_NAME, context.statuses, theme)
    else:
        style.progress_strip(layout.progress_ax, [NODE_NAME], NODE_NAME, {NODE_NAME: status}, theme)
    style.pulse_sequence(layout.sequence_ax, sequence_blocks(context.before if context else None, fit), theme)

    d = ds.sel(qubit=QUBIT)
    Ig, Qg, Ie, Qe = (d[v].values * 1e3 for v in ("I_g", "Q_g", "I_e", "Q_e"))
    theta = fit.get("rotation_angle")
    budget = _error_budget(ds, fit)
    _plot_iq(ax_iq, Ig, Qg, Ie, Qe, fit, budget, context, ds, theme)
    angle = theta if _finite(theta) else 0.0
    Ig_rot, _ = rotate(Ig, Qg, angle)
    Ie_rot, _ = rotate(Ie, Qe, angle)
    _plot_histogram(ax_hist, Ig_rot, Ie_rot, fit, budget if state == "ok" else None, marks, theme)

    style.place_legend(ax_iq, theme, ncols=2, title="raw IQ shots")
    style.place_legend(ax_hist, theme, ncols=2, title="rotated I")
    style.cards_column(layout.cards_ax, node_cards(fit, context, ds), theme)

    f_assign, _ = _fidelities(fit)
    headline = {
        "ok": f"(gg + ee)/2 = {100 * f_assign:.1f} %",
        "rejected": f"rejected fit: (gg + ee)/2 = {100 * f_assign:.1f} %",
        "failed": "fit failed",
    }[state]
    if state == "rejected" and new_gate_reason(fit):
        headline = f"rejected fit: {new_gate_reason(fit)}"
    style.title_block(fig, NODE_NAME, "Readout discrimination", headline, theme, status=status)
    return fig
