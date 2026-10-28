"""Layout C figure for 06b_echo: theme renders, reference handling, failed and rejected fits, physics labels."""

import copy
import math

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from calibration_utils import echo, style
from calibration_utils.common import INITIAL_STATE_PATH, NODE_ORDER, NodeContext, figure_context, iq_projection, load_node, load_state
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE, Device
from simulator.qpu import SimulatedQPU
from tests.conftest import apply_setup, assert_export_uses_pre_update_state

node = load_node("06b_echo.py")
NODE = node.NODE_NAME


@pytest.fixture(scope="module")
def standalone():
    """A bare 06b run on the calibrated state: no graph, no reference, no T1 or T2* in the state."""
    state = copy.deepcopy(load_state(INITIAL_STATE_PATH))
    q = state["qubits"]["q1"]
    q["f_01"] = TRUE_DEVICE.f_01
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    ctx = NodeContext(state=state, qpu=SimulatedQPU(seed=1234), save_outputs=False)
    node.run(ctx)
    return ctx


def _graph(graph_run_2026):
    """(ctx, fit, dataset, context) of 06b in the shared seed-2026 graph run (not mutated)."""
    ctx = graph_run_2026
    entry = ctx.results[NODE]
    return ctx, entry["fit"], entry["dataset"], entry["context"]


def _texts(fig: Figure) -> list[str]:
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _line_labels(fig: Figure) -> list[str]:
    return [line.get_label() for ax in fig.axes for line in ax.get_lines()]


def _render(ds, fit, marks=None, theme="light", context=None) -> Figure:
    with style.theme_context(theme):
        fig = echo.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context)
        style.fit_layout(fig)
    return fig


def _failed_context(graph_run_2026, fit):
    """The graph-run context rebuilt for a 06b fit that did not succeed: 06b reads "fail"."""
    ctx = copy.deepcopy(graph_run_2026)
    record = ctx.history.pop()  # the context of the run itself, before its record existed
    assert record.name == NODE
    ctx.state = copy.deepcopy(record.before)  # a failed fit leaves the state untouched
    _, axis, centre = iq_projection(record.dataset, "idle_time", first_point_excited=True)
    return figure_context(ctx, NODE, record.before, fit, axis, centre)


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_06b_figure_both_themes(standalone, theme):
    ds, fit = standalone.results[NODE]["dataset"], standalone.results[NODE]["fit"]
    fig = _render(ds, fit, style.Marks(fit=fit["T2echo"]), theme)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1
        assert any("06b" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_06b_figure_without_reference(standalone):
    ds, fit = standalone.results[NODE]["dataset"], standalone.results[NODE]["fit"]
    marks = style.Marks(previous=None, fit=fit["T2echo"], true=None)
    assert marks.true is None
    fig = _render(ds, fit, marks)
    try:
        assert marks.true_label not in _line_labels(fig)
        texts = _texts(fig)
        # No T1 or T2* measured in this ctx: the floor and ceiling are uncalculated, not drawn.
        assert any("uncalculated" in t for t in texts)
        assert not any("T2*" in label or "2T1" in label for label in _line_labels(fig))
    finally:
        plt.close(fig)


NAN_FIT = {
    "T2echo": float("nan"),
    "T2echo_error": float("nan"),
    "fit_params": {"a": float("nan"), "offset": float("nan"), "decay": float("nan")},
    "success": False,
}


@pytest.mark.parametrize("with_context", [False, True])
def test_06b_figure_renders_on_failed_fit(standalone, graph_run_2026, with_context):
    ds = standalone.results[NODE]["dataset"]
    context = _failed_context(graph_run_2026, NAN_FIT) if with_context else None
    if context is not None:
        assert context.statuses[NODE] == "fail"
        assert context.after["qubits"]["q1"]["T2echo"] is None
    fig = _render(ds, NAN_FIT, style.Marks(fit=None), context=context)
    try:
        texts = _texts(fig)
        assert any("fit failed" in t for t in texts)
        assert not any("nan" in t.lower() or "inf" in t for t in texts)
    finally:
        plt.close(fig)


def test_06b_figure_rejected_fit(graph_run_2026):
    """success=False with finite values: faded curve labelled "rejected fit", no fitted marker,
    no ±inf or ±nan text, the 06b pill and cards read fail."""
    _, fit, ds, _ = _graph(graph_run_2026)
    rejected = copy.deepcopy(fit)
    rejected["success"] = False
    rejected["T2echo_error"] = 0.3 * rejected["T2echo"]  # over the 20 % gate
    context = _failed_context(graph_run_2026, rejected)
    assert context.statuses[NODE] == "fail"
    fig = _render(ds, rejected, style.Marks(fit=None), context=context)
    try:
        labels = _line_labels(fig)
        texts = _texts(fig)
        assert "rejected fit" in labels
        assert "fitted value" not in labels and "fit" not in labels
        assert any("rejected fit" in t for t in texts)
        assert not any("nan" in t.lower() or "inf" in t for t in texts)
        assert "fail" in texts and "pass" not in texts  # Measured card status tag
    finally:
        plt.close(fig)


def test_06b_figure_layout_c(graph_run_2026):
    ctx, fit, ds, context = _graph(graph_run_2026)
    marks = ctx.results[NODE]["marks"]
    fig = _render(ds, fit, marks, context=context)
    try:
        texts = _texts(fig)
        for name in NODE_ORDER:
            assert style.NODE_SHORT[name] in texts, name
        for block in echo.sequence_blocks(context.before, fit):
            assert block.label in texts, block.label
        upper = [t.upper() for t in texts]
        for title in ("Measured", "State change"):
            assert title.upper() in upper, title
        assert marks.true_label in _line_labels(fig)
    finally:
        plt.close(fig)


def test_06b_sequence_puts_pi_at_midpoint_and_reads_sweep_range():
    fit = {"idle_time_range": (32.0, 200_000.0)}
    blocks = echo.sequence_blocks({}, fit)
    starts = [sum(b.width for b in blocks[:i]) for i in range(len(blocks))]
    first = next(i for i, b in enumerate(blocks) if b.label == "x90")
    last = max(i for i, b in enumerate(blocks) if b.kind == "drive")
    pi = next(i for i, b in enumerate(blocks) if b.label == "x180")
    pi_mid = starts[pi] + blocks[pi].width / 2
    assert pi_mid == pytest.approx((starts[first] + blocks[first].width + starts[last]) / 2)
    idles = [b for b in blocks if b.kind == "idle"]
    assert len(idles) == 2
    # Each half idle spans half the swept total idle time.
    assert all(b.swept == f"0.016 to 100{style.NBSP}{style.US}" for b in idles)


def test_06b_fit_records_idle_time_range(graph_run_2026):
    _, fit, ds, _ = _graph(graph_run_2026)
    assert fit["idle_time_range"] == (float(ds.idle_time.min()), float(ds.idle_time.max()))


def test_06b_physics_labels(graph_run_2026):
    ctx, fit, ds, context = _graph(graph_run_2026)
    assert context.population_levels is not None
    fig = _render(ds, fit, ctx.results[NODE]["marks"], context=context)
    try:
        labels = _line_labels(fig)
        assert any("T2*" in label for label in labels)
        assert any("2T1" in label for label in labels)
        texts = _texts(fig)
        assert any("emergent" in t.lower() for t in texts)
        # Ratios use the measured T2* (06a) and T1 (05) from the state before this node ran.
        before = context.before["qubits"]["q1"]
        assert any(f"{fit['T2echo'] / before['T2ramsey']:.2f}" in t for t in texts)
        assert any(f"{fit['T2echo'] / (2 * before['T1']):.2f}" in t for t in texts)
        # The caveat quotes the quasi-static noise and the white dephasing time, from the reference values.
        ref = ctx.reference
        d = Device(T1=ref["T1"], T2ramsey=ref["T2ramsey"], T2echo=ref["T2echo"])
        assert any(f"{d.sigma_f_quasistatic / 1e3:.1f}{style.NBSP}kHz" in t for t in texts)
        assert any(f"{d.Tphi_white / 1e3:.0f}{style.NBSP}{style.US}" in t for t in texts)
        assert any("(stat.)" in t for t in texts)
        secondary = [child for ax in fig.axes for child in ax.child_axes]
        assert any("P(e)" in sec.get_ylabel() for sec in secondary)
    finally:
        plt.close(fig)


def test_06b_floor_and_ceiling_curves(graph_run_2026):
    """Normalised axis: the T2* and 2T1 curves follow exp(-t/T2*) and exp(-t/2T1)."""
    _, fit, ds, context = _graph(graph_run_2026)
    before = context.before["qubits"]["q1"]
    fig = _render(ds, fit, context=context)
    try:
        lines = {line.get_label(): line for ax in fig.axes for line in ax.get_lines()}
        star = next(line for label, line in lines.items() if "T2*" in label)
        ceil = next(line for label, line in lines.items() if "2T1" in label)
        for line, tau in ((star, before["T2ramsey"]), (ceil, 2 * before["T1"])):
            t_us, y = line.get_xdata(), line.get_ydata()
            assert y[0] == pytest.approx(math.exp(-t_us[0] * 1e3 / tau))
            assert y[-1] == pytest.approx(math.exp(-t_us[-1] * 1e3 / tau))
    finally:
        plt.close(fig)


def test_06b_run_stores_context_and_marks(graph_run_2026):
    """run() keeps the FigureContext and Marks built from the snapshot and the reference."""
    ctx, fit, _, context = _graph(graph_run_2026)
    marks = ctx.results[NODE]["marks"]
    assert marks.fit == fit["T2echo"]
    assert marks.true == ctx.reference["T2echo"]
    assert marks.previous is None
    assert context.in_graph
    assert context.after["qubits"]["q1"]["T2echo"] == fit["T2echo"]
    assert context.before["qubits"]["q1"]["T2echo"] is None


def test_06b_export_uses_pre_update_state(calibrated_state, make_ctx):
    assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(calibrated_state), 0))
