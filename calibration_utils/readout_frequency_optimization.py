"""Analysis for 08a_readout_frequency_optimization: QM's g/e separation maximum and chi estimate.

Adapted from Quantum Machines' qua-libs ``calibration_utils/readout_frequency_optimization/analysis.py``
(BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibration_utils/readout_frequency_optimization/analysis.py
[vendor]. D = |(I_g, Q_g) - (I_e, Q_e)| per detuning; the optimum is the argmax of
``D.rolling({"detuning": 5}).mean("detuning")``; chi = (argmin |IQ_e| - argmin |IQ_g|) / 2.

xarray's ``rolling`` is trailing by default (``center=False``: the mean at index i covers i-4..i),
so on a smooth peak QM's pick sits 2 steps (0.2 MHz at the default 0.1 MHz step) above the maximum
of D. Kept as QM wrote it; tests/test_08a_readout_frequency_optimization.py pins it.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from calibration_utils import style
from calibration_utils.common import QUBIT

log = logging.getLogger(__name__)

NODE_NAME = "08a_readout_frequency_optimization"
ROLLING_POINTS = 5  # QM's window
MHZ = 1e-6
MV = 1e3


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """D (g/e separation) and |IQ| of each preparation. The simulator returns volts already, so
    QM's ``convert_IQ_to_V`` is the identity here."""
    D = np.hypot(ds.I_g - ds.I_e, ds.Q_g - ds.Q_e)
    return ds.assign(
        D=D.assign_attrs(long_name="g/e separation", units="V"),
        IQ_abs_g=np.hypot(ds.I_g, ds.Q_g).assign_attrs(units="V"),
        IQ_abs_e=np.hypot(ds.I_e, ds.Q_e).assign_attrs(units="V"),
    )


def fit_raw_data(ds: xr.Dataset) -> dict:
    d = ds.sel(qubit=QUBIT)
    rolling = d.D.rolling({"detuning": ROLLING_POINTS}).mean("detuning")  # trailing, as QM's code
    optimal_index = int(rolling.argmax("detuning"))
    optimal_detuning = float(d.detuning.values[optimal_index])
    optimal_frequency = float(d.full_freq.values[optimal_index])
    chi = float((d.IQ_abs_e.idxmin("detuning") - d.IQ_abs_g.idxmin("detuning")) / 2)
    # QM's frequency check, copied: np.abs of a boolean is 0 or 1, so it is always below 400e6
    # and never rejects (it presumably meant np.abs(optimal_detuning) < 400e6).
    freq_success = bool(np.abs(np.isnan(optimal_detuning)) < 400e6)
    success = bool(np.isfinite(chi) and np.isfinite(optimal_detuning) and np.isfinite(optimal_frequency) and freq_success)
    D = d.D.values
    detunings = d.detuning.values
    return {
        "optimal_frequency": optimal_frequency,
        "optimal_detuning": optimal_detuning,
        "chi": chi,
        "D_max": float(np.max(D)),
        "D_at_pick": float(D[optimal_index]),
        "D_rolling": rolling.values,
        "detuning_range": (float(detunings.min()), float(detunings.max())),
        "min_g": float(d.IQ_abs_g.idxmin("detuning")),
        "min_e": float(d.IQ_abs_e.idxmin("detuning")),
        "success": success,
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "08a_readout_frequency_optimization %s: readout detuning %+.3f MHz, chi %.3f MHz",
        status, fit["optimal_detuning"] * MHZ, fit["chi"] * MHZ,
    )


# ---------------------------------------------------------------- figure
def _minus(text: str) -> str:
    return text.replace("-", "−")


def _finite(*values) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def _f_ro(state: dict | None) -> float | None:
    return None if not state else state["qubits"][QUBIT]["resonator"]["f_01"]


def sequence_blocks(state: dict | None, fit: dict) -> list[style.SeqBlock]:
    """Each detuning: reset, readout (|g>); reset, x180, readout (|e>)."""
    span = fit.get("detuning_range")
    swept = None
    if span is not None and _finite(*span):
        swept = _minus(f"f_ro {span[0] * MHZ:+.0f} to {span[1] * MHZ:+.0f}{style.NBSP}MHz")
    return [
        style.SeqBlock("wait", "reset", width=0.9),
        style.SeqBlock("readout", "readout |g⟩", swept=swept, width=1.8, line="resonator"),
        style.SeqBlock("wait", "reset", width=0.9),
        style.SeqBlock("drive", "x180", width=0.8),
        style.SeqBlock("readout", "readout |e⟩", width=1.8, line="resonator"),
    ]


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    ok = bool(fit.get("success"))
    before = _f_ro(context.before) if context else None
    after = _f_ro(context.after) if context else None
    if ok:
        measured = []
        if _finite(before, after):
            measured.append(style.format_change("f_ro", before, after, 1e-9, "GHz", "{:.6f}"))
        measured.append(_minus(f"shift {fit['optimal_detuning'] * MHZ:+.2f}{style.NBSP}MHz"))
        measured.append(f"D at the pick: {100 * fit['D_at_pick'] / fit['D_max']:.1f}{style.NBSP}% of max")
    else:
        measured = ["fit failed", "state not updated"]
    cards = [style.Card("Measured", tuple(measured), "ok" if ok else "fail")]

    chi_lines = [_minus(f"χ = (f_e − f_g)/2 = {fit['chi'] * MHZ:+.2f}{style.NBSP}MHz")] if _finite(fit.get("chi")) else []
    reference = (context.reference or {}) if context is not None and context.in_graph else {}
    if _finite(reference.get("chi")):
        chi_lines.append(_minus(f"true value {reference['chi'] * MHZ:+.2f}{style.NBSP}MHz"))
    if chi_lines:
        cards.append(style.Card("Dispersive shift", tuple(chi_lines)))
    cards.append(
        style.Card(
            "Method",
            (
                "QM's 5-point mean is trailing: the",
                f"pick sits 2 steps (+0.2{style.NBSP}MHz) above",
                "the D maximum; kept as QM wrote it",
            ),
            "caveat",
        )
    )
    return cards


def plot_raw_data_with_fit(
    ds: xr.Dataset,
    fit: dict,
    marks: style.Marks | None = None,
    theme: str = "light",
    context: style.FigureContext | None = None,
) -> Figure:
    """Layout C, two columns: |IQ| of both preparations (left), D and its trailing mean (right),
    against the readout detuning from the frequency the node started at."""
    d = ds.sel(qubit=QUBIT)
    x = d.detuning.values * MHZ
    ok = bool(fit.get("success"))
    with style.theme_context(theme):
        p = style.palette(theme)
        layout = style.node_layout(theme, ncols=2, residuals=False)
        fig = layout.fig
        left, right = layout.main_axes
        left.set_gid("main")
        left.plot(x, d.IQ_abs_g.values * MV, "o", ms=2.5, color=p.ground, label="|g⟩")
        left.plot(x, d.IQ_abs_e.values * MV, "o", ms=2.5, color=p.excited, label="|e⟩")
        if _finite(fit.get("min_g"), fit.get("min_e")):
            y = float(np.nanmin([d.IQ_abs_g.min(), d.IQ_abs_e.min()])) * MV
            top = float(np.nanmax([d.IQ_abs_g.max(), d.IQ_abs_e.max()])) * MV
            bar_y = y - 0.08 * (top - y)
            x0, x1 = sorted((fit["min_e"] * MHZ, fit["min_g"] * MHZ))
            from calibration_utils.resonator_spectroscopy import _span_bar

            _span_bar(left, x0, x1, bar_y, p.muted, _minus(f"2χ = {2 * fit['chi'] * MHZ:+.2f}{style.NBSP}MHz"), x1)
            left.set_ylim(bar_y - 0.08 * (top - y), None)
        left.set_xlabel("readout detuning (MHz)")
        left.set_ylabel("|IQ| (mV)")
        style.place_legend(left, theme, title="readout of")

        right.plot(x, d.D.values * MV, "o", ms=2.5, color=p.data, label="D")
        rolling = fit.get("D_rolling")
        if rolling is not None:
            right.plot(x, np.asarray(rolling) * MV, color=p.fit if ok else p.fail, lw=1.8, label="trailing mean")
        if marks is not None:
            show_true = context is not None and context.in_graph
            style.mark_values(
                right,
                style.Marks(
                    previous=None if marks.previous is None else marks.previous * MHZ,
                    fit=None if marks.fit is None else marks.fit * MHZ,
                    true=None if (marks.true is None or not show_true) else marks.true * MHZ,
                    true_label="midpoint (true)",
                ),
                theme,
                fit_label="pick",
                previous_label="previous",
            )
        right.set_xlabel("readout detuning (MHz)")
        right.set_ylabel("separation (mV)")
        style.place_legend(right, theme, ncols=3, title="D = |IQ_g − IQ_e|")

        if context is not None:
            order, statuses = context.node_order, context.statuses
        else:
            order, statuses = [NODE_NAME], {NODE_NAME: "ok" if ok else "fail"}
        style.progress_strip(layout.progress_ax, order, NODE_NAME, statuses, theme)
        style.pulse_sequence(layout.sequence_ax, sequence_blocks(context.before if context else None, fit), theme)
        style.cards_column(layout.cards_ax, node_cards(fit, context), theme)
        headline = _minus(f"readout {fit['optimal_detuning'] * MHZ:+.2f}{style.NBSP}MHz, χ {fit['chi'] * MHZ:+.2f}{style.NBSP}MHz") if ok else "fit failed"
        style.title_block(fig, NODE_NAME, "Readout frequency optimization", headline, theme, status="ok" if ok else "fail")
    return fig
