"""Analysis for 01a_time_of_flight: QM's edge detector on the averaged raw ADC trace.

Adapted from Quantum Machines' qua-libs ``calibration_utils/time_of_flight/analysis.py``
(BSD-3-Clause):
https://github.com/qua-platform/qua-libs/blob/main/qualibration_graphs/superconducting/calibration_utils/time_of_flight/analysis.py
[vendor]. Raw ADC values convert to volts as ``-raw / 2**12``; |IQ| of the averaged inputs is
smoothed with a Savitzky-Golay filter of window 11 and order 3 (Savitzky & Golay 1964,
https://doi.org/10.1021/ac60214a047); the threshold is the mean of two means of the smoothed trace
(from sample 100 on, and up to 100 samples before the end); the delay is the first readout time
above it, rounded to the 4 ns clock; the offsets are the means of the averaged traces.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

from calibration_utils import style
from calibration_utils.common import QUBIT

log = logging.getLogger(__name__)

NODE_NAME = "01a_time_of_flight"
ADC_RAW_PER_VOLT = 2**12  # QM: volts = -raw / 2**12
ADC_RANGE_V = 0.5  # OPX+ analog inputs: +/-0.5 V (QM's 01a description)
CLOCK_CYCLE_NS = 4
SAVGOL_WINDOW, SAVGOL_ORDER = 11, 3
EDGE_MARGIN = 100  # samples, QM's threshold windows
TRACES = ("adcI", "adcQ", "adc_single_runI", "adc_single_runQ")
MV = 1e3


def process_raw_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Raw ADC units to volts (QM: -raw / 2**12) and the averaged |IQ|."""
    volts = {k: -ds[k].astype(float) / ADC_RAW_PER_VOLT for k in TRACES}
    for k in TRACES:
        volts[k].attrs = {"long_name": k, "units": "V"}
    ds = ds.assign(volts)
    return ds.assign(IQ_abs=np.sqrt(ds.adcI**2 + ds.adcQ**2))


def fit_raw_data(ds: xr.Dataset) -> dict:
    """QM's 01a analysis on one qubit. Keys: delay (ns, on the 4 ns grid), tof_to_add, the two
    offsets to add (V), threshold (V), filtered (smoothed |IQ|, V), readout_time_range, success."""
    d = ds.sel(qubit=QUBIT)
    t = d.readout_time.values
    filtered = savgol_filter(d.IQ_abs.values, SAVGOL_WINDOW, SAVGOL_ORDER)
    threshold = (filtered[EDGE_MARGIN:].mean() + filtered[:-EDGE_MARGIN].mean()) / 2
    above = filtered > threshold
    edge = float(t[int(np.argmax(above))]) if above.any() else float("nan")
    delay = float(np.round(edge / CLOCK_CYCLE_NS) * CLOCK_CYCLE_NS)
    offset_i, offset_q = float(d.adcI.mean()), float(d.adcQ.mean())
    # QM's checks: fail only when all three are NaN, and require both offsets inside the ADC range.
    nan_success = bool(np.isnan(delay) and np.isnan(offset_i) and np.isnan(offset_q))
    offset_success = bool(abs(offset_i) < ADC_RANGE_V and abs(offset_q) < ADC_RANGE_V)
    peak = float(np.max(np.abs(np.concatenate([d.adc_single_runI.values, d.adc_single_runQ.values]))))
    return {
        "edge": edge,
        "delay": delay,
        "tof_to_add": delay,
        "offset_I_to_add": offset_i,
        "offset_Q_to_add": offset_q,
        "threshold": float(threshold),
        "filtered": filtered,
        "readout_time_range": (float(t[0]), float(t[-1])),
        "single_run_peak": peak,
        "readout_length": ds.attrs.get("readout_length"),
        "readout_amplitude": ds.attrs.get("readout_amplitude"),
        "num_shots": ds.attrs.get("num_shots"),
        "success": bool(not nan_success and offset_success and np.isfinite(delay)),
    }


def log_fitted_results(fit: dict) -> None:
    status = "SUCCESS" if fit["success"] else "FAIL"
    log.info(
        "01a_time_of_flight %s: time of flight to add %.0f ns, offsets to add I %.3f mV, Q %.3f mV",
        status, fit["tof_to_add"], fit["offset_I_to_add"] * MV, fit["offset_Q_to_add"] * MV,
    )


# ---------------------------------------------------------------- figure
def _minus(text: str) -> str:
    return text.replace("-", "−")


def _finite(*values) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def _tof_of(state: dict | None) -> float | None:
    if not state:
        return None
    return state["qubits"][QUBIT]["resonator"].get("time_of_flight")


def sequence_blocks(state: dict | None, fit: dict) -> list[style.SeqBlock]:
    """One averaged shot: reset_if_phase, readout with the raw trace recorded, depletion wait."""
    length, amplitude = fit.get("readout_length"), fit.get("readout_amplitude")
    readout = "readout, raw ADC trace"
    if _finite(length, amplitude):
        readout = f"readout {length / 1e3:g}{style.NBSP}{style.US}, {amplitude:g}{style.NBSP}V, raw trace"
    return [
        style.SeqBlock("idle", "qubit undriven", width=1.4, line="qubit"),
        style.SeqBlock("readout", readout, width=3.2, line="resonator"),
        style.SeqBlock("wait", f"depletion 1{style.NBSP}{style.US}", width=1.4, line="resonator"),
    ]


def node_cards(fit: dict, context: style.FigureContext | None) -> list[style.Card]:
    """Measured, true value (graph or setup runs with a reference), ADC range and method."""
    ok = bool(fit.get("success"))
    before = _tof_of(context.before) if context else None
    after = _tof_of(context.after) if context else None
    if ok:
        measured = [f"edge at {fit['edge']:.0f}{style.NBSP}ns into the window"]
        if _finite(before, after):
            measured.append(style.format_change("time of flight", before, after, 1.0, "ns", "{:.0f}"))
        else:
            measured.append(f"time of flight +{fit['tof_to_add']:.0f}{style.NBSP}ns")
        measured.append(
            _minus(f"offsets added: I {fit['offset_I_to_add'] * MV:+.3f}, Q {fit['offset_Q_to_add'] * MV:+.3f}{style.NBSP}mV")
        )
    else:
        measured = ["fit failed", "state not updated"]
    cards = [style.Card("Measured", tuple(measured), "ok" if ok else "fail")]

    reference = (context.reference or {}) if context is not None and context.in_graph else {}
    true_tof = reference.get("time_of_flight")
    if _finite(true_tof):
        lines = [f"time of flight {true_tof:.0f}{style.NBSP}ns"]
        if ok and _finite(after):
            lines.append(_minus(f"fit - true: {after - true_tof:+.0f}{style.NBSP}ns"))
        true_i, true_q = reference.get("adc_offset_I"), reference.get("adc_offset_Q")
        if _finite(true_i, true_q):
            lines.append(_minus(f"offsets I {true_i * MV:+.3f}, Q {true_q * MV:+.3f}{style.NBSP}mV"))
        cards.append(style.Card("True value", tuple(lines)))

    peak = fit.get("single_run_peak")
    if _finite(peak):
        cards.append(
            style.Card("ADC range", (f"peak |single run| {peak * MV:.0f}{style.NBSP}mV", f"of ±{ADC_RANGE_V * MV:.0f}{style.NBSP}mV"))
        )
    cards.append(
        style.Card(
            "Method",
            (
                "QM's threshold on the smoothed |IQ|,",
                "first sample above it, rounded to",
                f"the 4{style.NBSP}ns clock",
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
    """Layout C, two columns: the last single run (left) and the averaged trace with QM's edge
    detector (right), both in mV against the time since the window opened."""
    d = ds.sel(qubit=QUBIT)
    t = d.readout_time.values
    with style.theme_context(theme):
        p = style.palette(theme)
        layout = style.node_layout(theme, ncols=2, residuals=False)
        fig = layout.fig
        left, right = layout.main_axes
        left.set_gid("main")
        left.plot(t, d.adc_single_runI.values * MV, color=p.data, lw=0.6, alpha=0.85, label="input 1 (I)")
        left.plot(t, d.adc_single_runQ.values * MV, color=p.muted, lw=0.6, alpha=0.7, label="input 2 (Q)")
        left.set_xlabel("readout time (ns)")
        left.set_ylabel("ADC (mV)")
        style.place_legend(left, theme, title="single run")

        right.plot(t, d.adcI.values * MV, color=p.data, lw=0.6, alpha=0.8, label="I")
        right.plot(t, d.adcQ.values * MV, color=p.muted, lw=0.6, alpha=0.7, label="Q")
        ok = bool(fit.get("success"))
        filtered = fit.get("filtered")
        if filtered is not None:
            right.plot(t, np.asarray(filtered) * MV, color=p.fit if ok else p.fail, lw=1.8, label="|IQ| smoothed")
        if _finite(fit.get("threshold")):
            right.axhline(fit["threshold"] * MV, color=p.muted, lw=1.0, ls=(0, (4, 3)), label="threshold")
        if marks is not None:
            show_true = context is not None and context.in_graph
            style.mark_values(
                right, style.Marks(previous=None, fit=marks.fit, true=marks.true if show_true else None), theme,
                fit_label="edge",
            )
        right.set_xlabel("readout time (ns)")
        n = fit.get("num_shots")
        style.place_legend(right, theme, ncols=4, title=f"averaged, {n} shots" if n else "averaged")

        if context is not None:
            order, statuses = context.node_order, context.statuses
        else:
            order, statuses = [NODE_NAME], {NODE_NAME: "ok" if ok else "fail"}
        style.progress_strip(layout.progress_ax, order, NODE_NAME, statuses, theme)
        style.pulse_sequence(layout.sequence_ax, sequence_blocks(context.before if context else None, fit), theme)
        style.cards_column(layout.cards_ax, node_cards(fit, context), theme)
        after = _tof_of(context.after) if context else None
        if ok:
            tof = after if _finite(after) else fit["tof_to_add"]
            headline = f"time of flight {tof:.0f}{style.NBSP}ns"
        else:
            headline = "fit failed"
        style.title_block(fig, NODE_NAME, "Time of flight and ADC offsets", headline, theme, status="ok" if ok else "fail")
    return fig
