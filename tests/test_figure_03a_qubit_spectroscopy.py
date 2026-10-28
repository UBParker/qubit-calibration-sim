"""Layout C figure for 03a_qubit_spectroscopy: content, labels and failure handling."""

import copy

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

from calibration_utils import qubit_spectroscopy as analysis
from calibration_utils import style
from calibration_utils.common import (
    INITIAL_STATE_PATH,
    NODE_ORDER,
    NodeContext,
    figure_context,
    load_node,
    load_state,
)
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.qpu import SimulatedQPU
from tests.conftest import apply_setup, assert_export_uses_pre_update_state

node = load_node("03a_qubit_spectroscopy.py")
NODE = node.NODE_NAME
ROUGH_F01 = 5.12e9


def _calibrated_state() -> dict:
    """Same values as the ``calibrated_state`` fixture, with the rough f_01 so 03a has work to do."""
    state = load_state(INITIAL_STATE_PATH)
    q = state["qubits"]["q1"]
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    q["f_01"] = ROUGH_F01
    return state


@pytest.fixture(scope="module")
def standalone():
    """A bare node run: no graph, no reference."""
    ctx = NodeContext(state=_calibrated_state(), qpu=SimulatedQPU(seed=1234), save_outputs=False)
    node.run(ctx)
    return ctx


@pytest.fixture
def graph(graph_run_2026):
    """The shared seed-2026 bring-up (with reference), deep-copied so tests may change it."""
    ctx = copy.deepcopy(graph_run_2026)
    assert ctx.results[NODE]["fit"]["success"]
    return ctx


def _before(ctx) -> dict:
    """The state the graph's 03a run started from."""
    return next(record.before for record in ctx.history if record.name == NODE)


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _render(ctx, theme="light", fit=None, marks="from run", context="from run"):
    result = ctx.results[NODE]
    marks = result["marks"] if marks == "from run" else marks
    context = result["context"] if context == "from run" else context
    with style.theme_context(theme):
        fig = analysis.plot_raw_data_with_fit(
            result["dataset"], fit or result["fit"], marks=marks, theme=theme, context=context
        )
        style.fit_layout(fig)
        fig.canvas.draw()
    return fig


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_03a_figure_both_themes(standalone, theme):
    fig = _render(standalone, theme)
    try:
        assert isinstance(fig, plt.Figure)
        assert any(ax.lines for ax in fig.axes)
        assert style.NODE_SHORT[NODE] in _texts(fig)
    finally:
        plt.close(fig)


def test_03a_figure_without_reference(standalone):
    marks = standalone.results[NODE]["marks"]
    assert marks.true is None
    fig = _render(standalone)
    try:
        labels = [ln.get_label() for ln in fig.findobj(Line2D)]
        assert marks.true_label not in labels
        assert not any("true value" in t.lower() for t in _texts(fig))
    finally:
        plt.close(fig)


def test_03a_figure_renders_on_failed_fit(standalone):
    nan = float("nan")
    failed = {
        "frequency": nan,
        "frequency_error": nan,
        "detuning": nan,
        "fwhm": nan,
        "fit_params": None,
        "success": False,
    }
    # A context from the run's own state, so the sample shows the strip and the pulse sequence.
    context = figure_context(standalone, NODE, copy.deepcopy(standalone.state), failed)
    fig = _render(standalone, fit=failed, marks=style.Marks(previous=ROUGH_F01), context=context)
    try:
        texts = _texts(fig)
        assert any("fit failed" in t for t in texts)
        assert not any("inf" in t or "nan" in t for t in texts)
    finally:
        plt.close(fig)


def test_03a_figure_layout_c(graph):
    fig = _render(graph)
    try:
        texts = _texts(fig)
        assert all(style.NODE_SHORT[name] in texts for name in NODE_ORDER)
        blocks = analysis.sequence_blocks(_before(graph), graph.results[NODE]["fit"])
        assert blocks
        for block in blocks:
            assert block.label in texts
        assert any(block.swept and block.swept in texts for block in blocks)
        upper = [t.upper() for t in texts]
        assert "MEASURED" in upper and "STATE CHANGE" in upper
    finally:
        plt.close(fig)


def test_03a_physics_labels(graph):
    fit = graph.results[NODE]["fit"]
    marks = graph.results[NODE]["marks"]
    assert marks.true == pytest.approx(TRUE_DEVICE.f_01)
    assert marks.fit == fit["frequency"]
    fig = _render(graph)
    try:
        texts = _texts(fig)
        assert any("power-broadened" in t for t in texts)
        # No true-value line on the plot; it sits a few kHz from the fit and goes in the card.
        labels = [ln.get_label() for ln in fig.findobj(Line2D)]
        assert marks.true_label not in labels
        # Correction from the starting guess, bracketed, and the centre error next to the fit.
        correction = (fit["frequency"] - marks.previous) / 1e6
        assert any(f"{correction:+.2f} MHz".replace("-", "−") in t for t in texts)
        assert any(f"{fit['frequency_error'] / 1e3:.1f} kHz" in t for t in texts)
        # Natural width for scale: T2* is not measured before 03a in the bring-up.
        assert any("1/(πT2*)" in t and "unknown" in t for t in texts)
        # True-value card: error in kHz and as a fraction of the fitted FWHM. Card lines may wrap
        # into separate text artists, so search the joined text.
        error = fit["frequency"] - marks.true
        joined = " ".join(texts)
        # U+2212 for negative numbers in figure text, "(stat.)" for the statistical error.
        assert f"{error / 1e3:+.1f} kHz".replace("-", "−") in joined
        assert f"{error / fit['fwhm']:+.3f} FWHM".replace("-", "−") in joined
        assert "fit - true" not in joined and "(fit, 1σ, stat.)" in joined
    finally:
        plt.close(fig)


def test_03a_natural_width_from_stored_T2(standalone):
    """With a T2* in the state before the run, the natural width 1/(pi T2*) is written out."""
    context = copy.copy(standalone.results[NODE]["context"])
    context.before = copy.deepcopy(context.before)
    context.before["qubits"]["q1"]["T2ramsey"] = 31_000.0
    fig = _render(standalone, context=context)
    try:
        assert any("1/(πT2*)" in t and "10.3 kHz" in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_03a_standalone_strip_and_no_true_card(standalone):
    fig = _render(standalone)
    try:
        texts = _texts(fig)
        assert style.NODE_SHORT[NODE] in texts
        assert not any(style.NODE_SHORT[name] in texts for name in NODE_ORDER if name != NODE)
        assert "TRUE VALUE" not in [t.upper() for t in texts]
    finally:
        plt.close(fig)


def test_03a_rejected_fit_with_curve(graph):
    """A finite fit that failed its gates: no infinite error, no accepted-looking annotations."""
    result = graph.results[NODE]
    rejected = dict(result["fit"], success=False, frequency_error=float("inf"))
    marks = style.Marks(previous=result["marks"].previous)  # run sets Marks.fit = None on failure
    context = figure_context(graph, NODE, _before(graph), rejected)
    assert context.statuses[NODE] == "fail"
    fig = _render(graph, theme="dark", fit=rejected, marks=marks, context=context)
    try:
        texts = _texts(fig)
        assert "not run" in texts  # the graph stops after a failed node
        assert not any("inf" in t or "nan" in t for t in texts)
        assert not any("power-broadened" in t for t in texts)
        assert not any("fitted" in t for t in texts)
        assert "rejected fit" in [ln.get_label() for ln in fig.findobj(Line2D)]
        correction = (result["fit"]["frequency"] - marks.previous) / 1e6
        assert not any(f"{correction:+.2f} MHz" in t for t in texts)
        # The fail box covers no other text in its panel.
        renderer = fig.canvas.get_renderer()
        box = next(t for t in fig.findobj(Text) if t.get_text().startswith("fit failed:"))
        patch = box.get_bbox_patch().get_window_extent(renderer)
        for other in box.axes.texts:
            if other is not box and other.get_text():
                assert not patch.overlaps(other.get_window_extent(renderer)), other.get_text()
    finally:
        plt.close(fig)


def test_03a_marks_fit_none_on_failure(make_ctx, rough_state, monkeypatch):
    nan = float("nan")
    monkeypatch.setattr(
        node.analysis,
        "fit_raw_data",
        lambda ds: {"frequency": nan, "frequency_error": nan, "detuning": nan, "fwhm": nan,
                    "fit_params": None, "success": False},
    )
    ctx = make_ctx(copy.deepcopy(rough_state))
    node.run(ctx)
    assert ctx.results[NODE]["marks"].fit is None
    assert ctx.results[NODE]["context"].statuses[NODE] == "fail"


def test_03a_export_uses_pre_update_state(make_ctx, rough_state):
    fit = assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(rough_state), 0))
    assert fit["success"]
