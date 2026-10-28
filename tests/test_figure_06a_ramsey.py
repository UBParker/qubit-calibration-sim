"""Layout C figure for 06a_ramsey: overlaid detuning signs, sign arithmetic, T2* card, P(e) axis.

Most tests draw the 06a data of the shared seed-2026 graph run (``graph_run_2026``); the
standalone tests draw the same data with no context.
"""

import copy
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text
from qualibration_libs.analysis.models import oscillation_decay_exp

from calibration_utils import ramsey, style
from calibration_utils.common import NODE_ORDER, QUBIT, figure_context, load_node
from tests.conftest import assert_export_uses_pre_update_state

node = load_node("06a_ramsey.py")
NODE = node.NODE_NAME
US, SIGMA = style.US, style.SIGMA


@pytest.fixture(scope="module")
def graph_run(graph_run_2026):
    """(ctx, ds, fit, context) of 06a in the seed-2026 graph, deep-copied so tests may mutate it."""
    ctx = copy.deepcopy(graph_run_2026)
    entry = ctx.results[NODE]
    return ctx, entry["dataset"], entry["fit"], entry["context"]


@pytest.fixture(scope="module")
def standalone(graph_run):
    ctx, ds, fit, _ = graph_run
    return ctx, ds, fit


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _render(ds, fit, theme="light", marks=None, context=None):
    with style.theme_context(theme):
        fig = ramsey.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context)
        style.fit_layout(fig)
        fig.canvas.draw()
    return fig


def _rejected_context(ctx, ds, rejected: dict):
    """figure_context as the graph would build it for a rejected 06a: state unchanged, node failed."""
    rerun = copy.deepcopy(ctx)
    record = next(r for r in rerun.history if r.name == NODE)
    rerun.state = copy.deepcopy(record.before)  # update_state skips a rejected fit
    axis, centre = ramsey.iq_axis(ds)
    return figure_context(rerun, NODE, record.before, rejected, axis, centre)


def _nan_fit(fit: dict) -> dict:
    nan = float("nan")
    failed = copy.deepcopy(fit)
    failed.update(
        success=False,
        freq_offset=nan,
        freq_offset_error=nan,
        T2ramsey=nan,
        T2ramsey_error=nan,
        fringe_frequencies={"+": nan, "-": nan},
        fit_params={s: {k: nan for k in ramsey.FIT_KEYS} for s in (-1, 1)},
    )
    return failed


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_06a_figure_both_themes(standalone, theme):
    _, ds, fit = standalone
    fig = _render(ds, fit, theme)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
        assert any(style.NODE_SHORT[NODE] in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_06a_figure_without_reference(standalone):
    _ctx, ds, fit = standalone
    marks = style.Marks(previous=5.1237e9, fit=5.1237e9 - fit["freq_offset"], true=None)
    assert marks.true is None
    fig = _render(ds, fit, marks=marks)
    try:
        labels = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
        assert marks.true_label not in labels
        assert not any(t.upper().startswith("TRUE VALUE") for t in _texts(fig))
    finally:
        plt.close(fig)


def test_06a_figure_renders_on_failed_fit(standalone):
    _, ds, fit = standalone
    fig = _render(ds, _nan_fit(fit))
    try:
        texts = _texts(fig)
        assert any("fit failed" in t for t in texts)
        assert not any(re.search(r"\bnan\b|\binf\b", t) for t in texts)
    finally:
        plt.close(fig)


def test_06a_figure_rejected_fit(graph_run):
    """success=False with finite values: dashed curves labelled "rejected fit", nothing applied."""
    ctx, ds, fit, _ = graph_run
    rejected = copy.deepcopy(fit)
    rejected["success"] = False
    context = _rejected_context(ctx, ds, rejected)
    assert context.statuses[NODE] == "fail"
    assert context.after == context.before
    fig = _render(ds, rejected, context=context)
    try:
        texts = _texts(fig)
        assert any("rejected fit" in t for t in texts)
        assert any("(not corrected)" in t for t in texts)
        f_01 = context.before["qubits"][QUBIT]["f_01"]
        assert style.format_change("f_01", f_01, f_01, 1e-9, "GHz", "{:.6f}") in texts
        pill_status = [t for t in texts if t == "failed"]
        assert pill_status, "the 06a pill is marked failed"
        assert not any(re.search(r"\bnan\b|\binf\b", t) for t in texts)
        assert not any("phase slip" in t for t in texts)
        assert not any("(exp fit)" in t for t in texts)
        assert any("not applied" in t for t in texts)
        labels = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
        assert "rejected fit" in labels
        assert not any("fitted" in lab for lab in labels)
    finally:
        plt.close(fig)


def test_06a_figure_layout_c(graph_run):
    ctx, ds, fit, context = graph_run
    assert context.in_graph
    fig = _render(ds, fit, context=context)
    try:
        texts = _texts(fig)
        for name in NODE_ORDER:
            assert style.NODE_SHORT[name] in texts, name
        for block in ramsey.sequence_blocks(ctx.state, fit):
            assert block.label in texts, block.label
        upper = [t.upper() for t in texts]
        for title in ("MEASURED", "STATE CHANGE", "TRUE VALUE", "CAVEAT"):
            assert title in upper, title
    finally:
        plt.close(fig)


def test_06a_sequence_reads_sweep_range(graph_run):
    _, _, fit, _ = graph_run
    assert fit["idle_time_range"] == (16.0, 50_000.0)
    idle = next(b for b in ramsey.sequence_blocks({}, fit) if b.kind == "idle")
    assert idle.swept == f"16 ns to 50 {US}"
    no_range = {k: v for k, v in fit.items() if k != "idle_time_range"}
    assert next(b for b in ramsey.sequence_blocks({}, no_range) if b.kind == "idle").swept == "swept"


def test_06a_physics_labels(graph_run):
    _, ds, fit, context = graph_run
    assert context.population_levels is not None
    fig = _render(ds, fit, context=context)
    try:
        texts = _texts(fig)
        main = [ax for ax in fig.axes if any((ln.get_gid() or "").startswith("data") for ln in ax.get_lines())]
        assert len(main) == 1, "both detuning signs belong on one axes"
        gids = {ln.get_gid() for ln in main[0].get_lines()}
        assert {"data+", "data-", "fit+", "fit-"} <= gids
        assert any("f_drive - f_01" in t for t in texts)
        assert any("cos(-x)" in t for t in texts)
        secondary = [child for ax in fig.axes for child in ax.child_axes]
        assert any("P(e)" in child.get_ylabel() for child in secondary)
        assert any("phase slip" in t for t in texts)
        strips = [ax for ax in fig.axes if ax.get_gid() in ("resid+", "resid-")]
        assert sorted(ax.get_gid() for ax in strips) == ["resid+", "resid-"]
        assert any("std" in t and SIGMA in t for t in texts)
        # The second series uses its own palette roles, not the |e> population colour.
        p = style.palette("light")
        colours = {ln.get_gid(): matplotlib.colors.to_hex(ln.get_color()) for ln in main[0].get_lines()}
        assert colours["data-"] == p.data2.lower() and colours["fit-"] == p.fit2.lower()
    finally:
        plt.close(fig)


def test_06a_no_population_axis_without_levels(standalone):
    _, ds, fit = standalone
    fig = _render(ds, fit)
    try:
        assert not any(ax.child_axes for ax in fig.axes)
    finally:
        plt.close(fig)


def test_06a_arithmetic_text_follows_code(graph_run):
    """The written arithmetic reproduces frequency_offset and the f_01 update in calibrations/06a."""
    _, _, fit, context = graph_run
    joined = "\n".join(ramsey.arithmetic_lines(fit))
    f_plus, f_minus = (fit["fringe_frequencies"][k] for k in ("+", "-"))
    assert fit["freq_offset"] == pytest.approx(ramsey.frequency_offset(f_plus, f_minus, fit["detuning"]))
    assert "(f+ - f-)/2" in joined
    offset = f"{fit['freq_offset'] / 1e3:+.2f}".replace("-", "−")  # U+2212 on numbers, formula text ASCII
    assert f"({f_plus / 1e3:.2f} - {f_minus / 1e3:.2f})/2 = {offset} kHz" in joined
    assert "f_01 -= (f_drive - f_01)" in joined
    shift = context.after["qubits"][QUBIT]["f_01"] - context.before["qubits"][QUBIT]["f_01"]
    assert shift == pytest.approx(-fit["freq_offset"])
    assert f"f_01 {shift / 1e3:+.2f}".replace("-", "−") + " kHz (applied)" in joined


def test_06a_arithmetic_folded_branch():
    """When one fringe has folded through zero the offset is +/- the mean, as frequency_offset does."""
    fit = {"fringe_frequencies": {"+": 700e3, "-": 100e3}, "detuning": 300e3, "freq_offset": 400e3,
           "freq_offset_error": 1e3, "success": True,
           "fit_params": {s: {k: 1.0 for k in ramsey.FIT_KEYS} for s in (-1, 1)}}
    assert ramsey.frequency_offset(700e3, 100e3, 300e3) == pytest.approx(400e3)
    joined = "\n".join(ramsey.arithmetic_lines(fit))
    assert "+(f+ + f-)/2 = +(700.00 + 100.00)/2 = +400.00 kHz" in joined
    assert "f- folded through 0" in joined


def test_06a_slip_is_unfolded_beyond_half_a_cycle(standalone):
    """d = +6 kHz: fringes 306 and 294 kHz, so the slip reaches 0.6 cycle by 50 us, not 0.4."""
    _, ds, fit = standalone
    big = copy.deepcopy(fit)
    params = {1: {"a": 1.6e-4, "f": 306e-6, "phi": 0.0, "offset": 0.0, "decay": 1 / 31e3},
              -1: {"a": 1.6e-4, "f": 294e-6, "phi": 0.0, "offset": 0.0, "decay": 1 / 31e3}}
    big.update(fit_params=params, fringe_frequencies={"+": 306e3, "-": 294e3}, freq_offset=6e3)
    assert ramsey.phase_slip_cycles(big, 50_000.0) == pytest.approx(0.6)
    # A fitted phase 2 pi away is the same curve, so the slip must not change.
    shifted = copy.deepcopy(big)
    shifted["fit_params"][-1]["phi"] += 2 * np.pi
    assert ramsey.phase_slip_cycles(shifted, 50_000.0) == pytest.approx(0.6)
    shifted["fit_params"][1]["phi"] -= 4 * np.pi
    assert ramsey.phase_slip_cycles(shifted, 50_000.0) == pytest.approx(0.6)
    t = ds.idle_time.values.astype(float)
    signal = np.stack([oscillation_decay_exp(t, **params[s]) for s in ds.detuning_signs.values], axis=-1)
    synth = ds.copy()
    synth["signal"] = (("qubit", "idle_time", "detuning_signs"), signal[None])
    fig = _render(synth, big)
    try:
        slips = [float(m.group(1)) for txt in _texts(fig) if (m := re.search(r"phase slip ([\d.]+) cycle", txt))]
        assert len(slips) == 1
        assert slips[0] > 0.5, "the nearest-trough distance would alias this to below half a cycle"
    finally:
        plt.close(fig)


def test_06a_cards_t1_reference(graph_run):
    _, _, fit, context = graph_run
    # 05 runs after 06a in the graph, so no T1 is stored yet.
    assert context.before["qubits"][QUBIT]["T1"] is None
    measured = next(c for c in ramsey.node_cards(fit, context) if c.title == "Measured")
    assert any("2 T1" in ln and "uncalculated" in ln for ln in measured.lines)
    caveat = next(c for c in ramsey.node_cards(fit, context) if c.title == "Caveat")
    assert caveat.status == "caveat"
    assert any("Model mismatch" in ln for ln in caveat.lines)


def test_06a_stored_t1_renders_through_figure_context(graph_run):
    """A rerun after 05 has T1 in the state: the 2 T1 line shows it, built by figure_context."""
    ctx, ds, fit, _ = graph_run
    rerun = copy.deepcopy(ctx)
    T1 = rerun.qubit["T1"]
    assert T1 is not None  # set by 05 in the graph
    before = rerun.snapshot()
    axis, centre = ramsey.iq_axis(ds)
    context = figure_context(rerun, NODE, before, fit, axis, centre)
    fig = _render(ds, fit, context=context)
    try:
        # Card lines may wrap into several text artists, so search the texts joined in order.
        joined = " ".join(_texts(fig))
        assert f"2 T1 = {2 * T1 / 1e3:.1f} {US} (stored T1), T2*/2T1 = {fit['T2ramsey'] / (2 * T1):.2f}" in joined
        assert "uncalculated (no T1" not in joined
    finally:
        plt.close(fig)


def test_06a_cards_standalone_have_no_true_value(standalone):
    _, _, fit = standalone
    titles = [c.title for c in ramsey.node_cards(fit, None)]
    assert "True value" not in titles
    assert {"Measured", "State change", "Caveat"} <= set(titles)


def test_06a_true_value_card_compares_corrected_frequency(graph_run):
    _, _, fit, context = graph_run
    card = next(c for c in ramsey.node_cards(fit, context) if c.title == "True value")
    diff = context.after["qubits"][QUBIT]["f_01"] - context.reference["f_01"]
    sigma = diff / fit["freq_offset_error"]
    num = lambda v, spec: format(v, spec).replace("-", "−")  # noqa: E731
    assert f"new f_01 - true: {num(diff / 1e3, '+.2f')} kHz ({num(sigma, '+.1f')} {SIGMA})" in card.lines
    pct = 100 * (fit["T2ramsey"] / context.reference["T2ramsey"] - 1)
    assert any(f"{num(pct, '+.1f')} %" in ln for ln in card.lines)


def test_06a_export_uses_pre_update_state_and_rejected_marks(calibrated_state, make_ctx, monkeypatch):
    """The QUA export sees the state before update_state; a rejected fit stores Marks.fit None."""
    params = node.Parameters(num_points=60)
    ctx = make_ctx(copy.deepcopy(calibrated_state), 0)
    assert_export_uses_pre_update_state(node, ctx, params)

    real_fit = ramsey.fit_raw_data
    monkeypatch.setattr(node.analysis, "fit_raw_data", lambda ds: {**real_fit(ds), "success": False})
    ctx = make_ctx(copy.deepcopy(calibrated_state), 0)
    f_01 = ctx.qubit["f_01"]
    node.run(ctx, params)
    assert ctx.results[NODE]["marks"].fit is None
    assert ctx.results[NODE]["marks"].previous == f_01
    assert ctx.qubit["f_01"] == f_01
