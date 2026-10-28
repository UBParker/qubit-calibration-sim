"""Layout C figure for 05_T1: themes, reference handling, failed fits, physics labels."""

import copy
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D

from calibration_utils import T1 as analysis
from calibration_utils import style
from calibration_utils.common import NODE_ORDER, load_node
from tests.conftest import assert_export_uses_pre_update_state

node = load_node("05_T1.py")
NODE = node.NODE_NAME


def _texts(fig) -> list[str]:
    """Every text string drawn on the figure, including legend entries and axis labels."""
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]


def _render(ds, fit, marks=None, theme="light", context=None):
    with style.theme_context(theme):
        fig = analysis.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context)
        style.fit_layout(fig)
    return fig


@pytest.fixture
def graph_run(graph_run_2026):
    """The shared seed-2026 bring-up: 05's dataset, fit and the figure inputs its run stored."""
    record = next(r for r in graph_run_2026.history if r.name == NODE)
    assert record.success
    entry = graph_run_2026.results[NODE]
    assert entry["context"].in_graph
    return entry["dataset"], entry["fit"], entry["marks"], entry["context"]


@pytest.fixture
def standalone(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    before = ctx.snapshot()
    fit = node.run(ctx)
    ds = ctx.results[NODE]["dataset"]
    marks, _ = node.figure_inputs(ctx, before, fit, ds)
    return ds, fit, marks


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_05_T1_figure_both_themes(standalone, theme):
    ds, fit, marks = standalone
    fig = _render(ds, fit, marks, theme)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
        assert style.NODE_SHORT[NODE] in _texts(fig)
    finally:
        plt.close(fig)


def test_05_T1_figure_without_reference(standalone):
    ds, fit, marks = standalone
    assert marks.true is None
    fig = _render(ds, fit, marks)
    try:
        labels = [ln.get_label() for ln in fig.findobj(Line2D)]
        assert marks.true_label not in labels
        assert not any("True value" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_05_T1_figure_renders_on_failed_fit(standalone):
    ds, _, _ = standalone
    nan = float("nan")
    failed = {"T1": nan, "T1_error": nan, "fit_params": {"a": nan, "offset": nan, "decay": nan}, "success": False}
    fig = _render(ds, failed, style.Marks())
    try:
        assert any("fit failed" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_05_T1_figure_layout_c(graph_run):
    ds, fit, marks, context = graph_run
    assert context.node_order == NODE_ORDER
    fig = _render(ds, fit, marks, context=context)
    try:
        texts = _texts(fig)
        for name in NODE_ORDER:
            assert style.NODE_SHORT[name] in texts
        for block in analysis.sequence_blocks(context.before, fit):
            if block.label:
                assert block.label in texts
            if block.swept:
                assert block.swept in texts
        for title in ("Measured", "State change", "True value"):
            assert title.upper() in texts
        assert any(t.startswith("T1: uncalculated -> ") for t in texts)
    finally:
        plt.close(fig)


def test_05_T1_cards_compare_with_true_value(graph_run):
    _, fit, _, context = graph_run
    cards = {card.title: card for card in analysis.node_cards(fit, context)}
    assert set(cards) == {"Measured", "True value", "State change"}
    true_t1 = context.reference["T1"]
    diff = fit["T1"] - true_t1
    joined = " ".join(cards["True value"].lines)
    assert f"{true_t1 / 1e3:.1f} µs" in joined
    # U+2212 for negative numbers in figure text.
    assert f"{100 * diff / true_t1:+.1f} %".replace("-", "−") in joined
    assert f"{diff / fit['T1_error']:+.1f} σ".replace("-", "−") in joined
    assert "-" not in joined.replace("fit − true", "")
    # The sweep coverage is in the chip only, not repeated in the Measured card.
    assert not any("sweep reaches" in line for line in cards["Measured"].lines)
    # A standalone run has no true-value card (ruling V).
    assert {c.title for c in analysis.node_cards(fit, None)} == {"Measured", "State change"}


def test_05_T1_physics_labels(graph_run):
    ds, fit, marks, context = graph_run
    fig = _render(ds, fit, marks, context=context)
    try:
        main = next(ax for ax in fig.axes if ax.get_gid() == "main")
        offset_mv = fit["fit_params"]["offset"] * analysis.SIGNAL_SCALE
        flat = [
            ln for ln in main.get_lines()
            if len(set(np.asarray(ln.get_ydata(), dtype=float))) == 1 and len(ln.get_ydata()) >= 2
        ]
        assert any(abs(float(ln.get_ydata()[0]) - offset_mv) < 1e-12 for ln in flat)
        texts = _texts(fig)
        assert any("1/e" in t for t in texts)
        # The 1/e point sits at B + A/e above the asymptote, at t = T1 (in us on the plot).
        p = fit["fit_params"]
        target = ((fit["T1"] / 1e3), (p["offset"] + p["a"] / math.e) * analysis.SIGNAL_SCALE)
        points = [ln for ln in main.get_lines() if ln.get_gid() == "one_over_e"]
        assert len(points) == 1
        assert np.allclose([points[0].get_xdata()[0], points[0].get_ydata()[0]], target, rtol=0, atol=1e-12)
        coverage = ds.idle_time.values.max() / fit["T1"]
        assert any(f"{coverage:.1f} T1" in t for t in texts)
        # No 2 T1 or 3 T1 annotations (a coverage such as "4.2 T1" is fine).
        assert not any(re.search(r"(?<![\d.])[23] ?T1", t) for t in texts)
        # Population axis from the 07 blob means; the first point may read above 1, so no clipping at 1.
        assert context.population_levels is not None
        secondary = [child for child in main.child_axes if "P(e)" in child.get_ylabel()]
        assert len(secondary) == 1
        level_g, level_e = (v * analysis.SIGNAL_SCALE for v in context.population_levels)
        assert secondary[0].get_ylim() == pytest.approx(
            tuple((y - level_g) / (level_e - level_g) for y in main.get_ylim())
        )
        assert main.get_ylim()[1] > ds.signal.values.max() * analysis.SIGNAL_SCALE
    finally:
        plt.close(fig)


def _true_artists(fig, true_ns):
    main = next(ax for ax in fig.axes if ax.get_gid() == "main")
    x = true_ns * analysis.TIME_SCALE
    labelled = [ln for ln in main.get_lines() if ln.get_label() == "true value"]
    full_height = [
        ln for ln in main.get_lines()
        if len(ln.get_xdata()) == 2 and np.allclose(np.asarray(ln.get_xdata(), dtype=float), x, rtol=0, atol=1e-9)
    ]
    return labelled, full_height


def test_05_T1_true_value_is_an_edge_marker_in_graph(graph_run):
    ds, fit, marks, context = graph_run
    fig = _render(ds, fit, marks, context=context)
    try:
        labelled, full_height = _true_artists(fig, marks.true)
        assert len(labelled) == 1 and len(labelled[0].get_xdata()) == 1 and labelled[0].get_marker() == "v"
        assert full_height == []  # no full-height line hides the 1/e drop line
        texts = _texts(fig)
        assert any("vs true value:" in t for t in texts)
        # The chip carries t_max and the comparison only; the T1 value is in the headline and card.
        chip = [t for t in texts if "t_max" in t]
        assert len(chip) == 1 and "±" not in chip[0]
    finally:
        plt.close(fig)


def test_05_T1_true_value_hidden_outside_graph(standalone):
    ds, fit, marks = standalone
    shown = style.Marks(previous=None, fit=marks.fit, true=48_000.0)
    fig = _render(ds, fit, shown)
    try:
        labelled, full_height = _true_artists(fig, shown.true)
        assert labelled == [] and full_height == []
        assert not any("true value" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_05_T1_standalone_run_sets_no_true_value(calibrated_state, make_ctx):
    """Ruling V: a standalone run with a reference attached still gets no true value."""
    from simulator.reference import build_reference

    ctx = make_ctx(calibrated_state)
    ctx.reference = build_reference(ctx.state)
    assert ctx.graph_nodes is None
    before = ctx.snapshot()
    fit = node.run(ctx)
    marks, context = node.figure_inputs(ctx, before, fit, ctx.results[NODE]["dataset"])
    assert marks.true is None
    assert "True value" not in {c.title for c in analysis.node_cards(fit, context)}
    # The same run inside a graph shows it.
    ctx.graph_nodes = list(NODE_ORDER)
    marks, _ = node.figure_inputs(ctx, before, fit, ctx.results[NODE]["dataset"])
    assert marks.true == ctx.reference["T1"]


def test_05_T1_rejected_fit(standalone):
    ds, fit, _ = standalone
    rejected = {**fit, "T1_error": 0.5 * fit["T1"], "success": False}
    fig = _render(ds, rejected, style.Marks(previous=None, fit=None, true=None))
    try:
        texts = _texts(fig)
        assert any("rejected fit" in t for t in texts)
        assert not any(re.search(r"inf|nan", t) for t in texts)
        main = next(ax for ax in fig.axes if ax.get_gid() == "main")
        labels = [ln.get_label() for ln in main.get_lines()]
        assert "rejected fit" in labels
        assert not any("fitted" in lab or "asymptote" in lab or "1/e" in lab for lab in labels)
        assert not any(ln.get_gid() == "one_over_e" for ln in main.get_lines())
    finally:
        plt.close(fig)


def test_05_T1_marks_fit_none_when_rejected(calibrated_state, make_ctx, monkeypatch):
    real_fit = analysis.fit_raw_data
    monkeypatch.setattr(analysis, "fit_raw_data", lambda ds: {**real_fit(ds), "success": False})
    ctx = make_ctx(calibrated_state)
    node.run(ctx)
    assert ctx.results[NODE]["marks"].fit is None
    assert ctx.qubit["T1"] is None


def test_05_T1_export_uses_pre_update_state(calibrated_state, make_ctx):
    fit = assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(calibrated_state), 0))
    assert fit["success"]
