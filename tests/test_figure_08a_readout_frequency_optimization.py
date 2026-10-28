"""Layout C figure for 08a: both themes, the true midpoint only in graph runs, the cards, and a
failed fit."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from calibration_utils import readout_frequency_optimization as analysis
from calibration_utils import style
from calibration_utils.common import NODE_ORDER, load_node
from simulator.device import TRUE_DEVICE

node = load_node("08a_readout_frequency_optimization.py")
NODE = node.NODE_NAME


def _texts(fig) -> list[str]:
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]


@pytest.fixture
def graph_run(graph_run_2026):
    record = next(r for r in graph_run_2026.history if r.name == NODE)
    assert record.success
    entry = graph_run_2026.results[NODE]
    assert entry["context"].in_graph
    return entry["dataset"], entry["fit"], entry["marks"], entry["context"], record


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_08a_figure_layout_c(graph_run, theme):
    ds, fit, marks, context, _ = graph_run
    fig = analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context)
    try:
        texts = _texts(fig)
        joined = " ".join(texts)
        for name in NODE_ORDER:
            assert style.NODE_SHORT[name] in texts, name
        for word in ("pick", "trailing mean", "midpoint (true)", "DISPERSIVE SHIFT", "METHOD"):
            assert word.lower() in joined.lower(), word
        assert "true value −0.55" in joined
    finally:
        plt.close(fig)


def test_08a_true_mark_is_midpoint_minus_start(graph_run):
    _, fit, marks, _, record = graph_run
    start = record.before["qubits"]["q1"]["resonator"]["f_01"]
    assert marks.true == pytest.approx(TRUE_DEVICE.f_res_g - TRUE_DEVICE.chi - start)
    assert marks.previous == 0.0 and marks.fit == fit["optimal_detuning"]


def test_08a_standalone_has_no_true_value(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 3)
    node.run(ctx)
    assert ctx.results[NODE]["marks"].true is None
    ds, fit, context = ctx.results[NODE]["dataset"], ctx.results[NODE]["fit"], ctx.results[NODE]["context"]
    fig = analysis.plot_raw_data_with_fit(ds, fit, ctx.results[NODE]["marks"], "light", context)
    try:
        assert not any("true value" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_08a_failed_fit_renders(graph_run):
    ds, fit, _, context, _ = graph_run
    fig = analysis.plot_raw_data_with_fit(ds, dict(fit, success=False), None, "light", context)
    try:
        assert "fit failed" in _texts(fig)
    finally:
        plt.close(fig)
