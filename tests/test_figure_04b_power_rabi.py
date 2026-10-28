"""Layout C figure for 04b_power_rabi.

Graph-run tests use the shared ``graph_run_2026`` session fixture (seed 2026 from the rough
state, where 04b refines). Standalone tests share one module-scoped calibrated run (no refine
pass, no reference).
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from calibration_utils import power_rabi as analysis
from calibration_utils import style
from calibration_utils.common import INITIAL_STATE_PATH, NODE_ORDER, NodeContext, load_node, load_state
from simulator.device import TRUE_DEVICE, TRUE_PI_AMPLITUDE
from simulator.qpu import SimulatedQPU
from tests.conftest import apply_setup, assert_export_uses_pre_update_state

node = load_node("04b_power_rabi.py")
NODE = node.NODE_NAME
ROUGH_PI_AMPLITUDE = 0.1  # state_initial.json starting guess


def _calibrated_state() -> dict:
    """Same values as the conftest ``calibrated_state`` fixture, built once per module."""
    state = load_state(INITIAL_STATE_PATH)
    q = state["qubits"]["q1"]
    q["f_01"] = TRUE_DEVICE.f_01
    q["anharmonicity"] = TRUE_DEVICE.anharmonicity
    q["xy"]["operations"]["x180"]["amplitude"] = TRUE_PI_AMPLITUDE
    q["resonator"]["f_01"] = TRUE_DEVICE.f_res_g
    apply_setup(q)
    return state


@pytest.fixture(scope="module")
def standalone():
    ctx = NodeContext(state=_calibrated_state(), qpu=SimulatedQPU(seed=1234), save_outputs=False)
    fit = node.run(ctx)
    return ctx, fit


@pytest.fixture
def graph(graph_run_2026):
    """(ctx, fit, dataset, first-pass dataset, context) of 04b in the seed-2026 graph run."""
    ctx = copy.deepcopy(graph_run_2026)
    entry = ctx.results[NODE]
    return ctx, entry["fit"], entry["dataset"], entry["first_pass_dataset"], entry["context"]


def _texts(fig) -> list[str]:
    found = [t.get_text() for t in fig.texts]
    found += [a.get_text() for a in fig.artists if hasattr(a, "get_text")]
    for ax in fig.axes:
        found += [t.get_text() for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            found += [t.get_text() for t in legend.get_texts()]
    return found


def _render(make, theme="light"):
    with style.theme_context(theme):
        fig = make(theme)
        style.fit_layout(fig)
        fig.canvas.draw()
        return fig


def _main(fig):
    for ax in fig.axes:
        if any(line.get_gid() == "data" for line in ax.get_lines()):
            return ax
    raise AssertionError("no axes with the data series")


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_04b_power_rabi_figure_both_themes(standalone, theme):
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t), theme)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    assert any(style.NODE_SHORT[NODE] in text or NODE in text for text in _texts(fig))
    plt.close(fig)


def test_04b_power_rabi_figure_without_reference(standalone):
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    marks = style.Marks(previous=TRUE_PI_AMPLITUDE, fit=fit["opt_amp"], true=None)
    assert marks.true is None
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, marks, theme=t))
    labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
    assert marks.true_label not in labels
    plt.close(fig)


def test_04b_power_rabi_figure_renders_on_failed_fit(standalone):
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    nan = float("nan")
    failed = {
        **fit,
        "success": False,
        "opt_amp": nan,
        "opt_amp_error": nan,
        "opt_amp_prefactor": nan,
        "half_period_prefactor": nan,
        "osc_amp_snr": nan,
        "phase_error": nan,
        "n_periods": nan,
        "gates": {gate: False for gate in fit["gates"]},
        "fit_params": {k: nan for k in fit["fit_params"]},
        "refined_from": None,
    }
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, failed, theme=t))
    assert any("fit failed" in text for text in _texts(fig))
    plt.close(fig)


def test_04b_power_rabi_rejected_fit_with_finite_values(standalone):
    """success False with finite values: "rejected fit", no fitted-value mark, no correction, no nan text."""
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    rejected = {**fit, "success": False, "gates": {**fit["gates"], "phase": False}}
    marks = style.Marks(previous=0.1, fit=fit["opt_amp"], true=TRUE_PI_AMPLITUDE)
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, rejected, marks, theme=t))
    texts = _texts(fig)
    labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
    assert "rejected fit" in labels
    assert "fitted value" not in labels and "fit" not in labels
    assert not any("from previous" in text or text.endswith(" %") and text.startswith("+") for text in texts)
    assert not any("nan" in text.lower() or "inf" in text.lower() for text in texts)
    assert any("rejected fit" in text for text in texts)
    plt.close(fig)


def test_04b_power_rabi_figure_layout_c(graph):
    ctx, fit, ds, first, context = graph
    record = next(r for r in ctx.history if r.name == NODE)
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t, context=context, first_pass=first))
    texts = _texts(fig)
    for name in NODE_ORDER:
        assert style.NODE_SHORT[name] in texts
    for block in analysis.sequence_blocks(record.before, fit):
        assert block.label in texts
        assert block.swept is None or block.swept in texts
    for title in ("MEASURED", "STATE CHANGE", "TRUE VALUE"):
        assert title in texts
    plt.close(fig)


def test_04b_power_rabi_cards_standalone_have_no_true_value(standalone):
    _, fit = standalone
    titles = [card.title for card in analysis.node_cards(fit, None)]
    assert "Measured" in titles
    assert "True value" not in titles


def test_04b_power_rabi_standalone_state_change_uses_stored_amplitude(standalone):
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t))
    want = style.format_change("x180 amplitude", TRUE_PI_AMPLITUDE, fit["opt_amp"], 1e3, "mV", "{:.2f}")
    assert want in _texts(fig)
    plt.close(fig)


def test_04b_power_rabi_error_invariant_under_sign_flip(standalone):
    """A refit started from (-a, f, phi + pi) is the same curve and must give the same error."""
    ctx, fit = standalone
    ds = ctx.results[NODE]["dataset"]
    x = ds.amp_prefactor.values
    y = ds.signal.sel(qubit="q1").values
    p = fit["fit_params"]
    x_pi = fit["opt_amp_prefactor"]
    plain = analysis._prefactor_error(x, y, (p["a"], p["f"], p["phi"], p["offset"]), x_pi)
    flipped = analysis._prefactor_error(x, y, (-p["a"], p["f"], p["phi"] + np.pi, p["offset"]), x_pi)
    assert np.isfinite(plain) and plain > 0
    assert flipped == pytest.approx(plain, rel=1e-3)
    # A QM x_pi far from the refit's own (here half a period off) is rejected, not propagated.
    off = analysis._prefactor_error(x, y, (p["a"], p["f"], p["phi"], p["offset"]), x_pi + 0.5 / p["f"])
    assert np.isnan(off)


def test_04b_power_rabi_error_is_statistical(graph):
    _, fit, _, _, context = graph
    assert np.isfinite(fit["opt_amp_error"]) and 0 < fit["opt_amp_error"] < 0.01 * fit["opt_amp"]
    lines = [line for card in analysis.node_cards(fit, context) for line in card.lines]
    assert any("±" in line and "(stat.)" in line for line in lines)
    assert not any("statistical" in line for line in lines)


def test_04b_power_rabi_physics_labels(graph):
    _, fit, ds, first, context = graph
    assert fit["refined_from"] is not None
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t, context=context, first_pass=first))
    texts = _texts(fig)
    bare = ("|g>", "|e>", style.KET_G, style.KET_E)
    assert not any(text.strip() in bare for text in texts)
    assert any("periods" in text for text in texts)
    assert any(f"prepared {style.KET_G}, simulator reference" in text for text in texts)
    assert any(f"prepared {style.KET_E}, simulator reference" in text for text in texts)
    assert any("thermal" in text for text in texts)
    main = _main(fig)
    series = [line for line in main.get_lines() if line.get_gid() in ("data", "first pass data")]
    assert len(series) == 2
    assert any("prefactor" in ax.get_xlabel() for ax in main.child_axes)
    plt.close(fig)


def test_04b_power_rabi_first_pass_uses_second_series_colours(graph):
    _, fit, ds, first, context = graph
    with style.theme_context("dark") as p:
        fig = analysis.plot_raw_data_with_fit(ds, fit, theme="dark", context=context, first_pass=first)
        lines = {line.get_gid(): line for line in _main(fig).get_lines()}
        assert lines["first pass data"].get_markeredgecolor() == p.data2
        assert lines["first pass fit"].get_color() == p.fit2
        plt.close(fig)


def test_04b_power_rabi_prefactor_axis_uses_stored_amplitude(graph):
    _, fit, ds, first, context = graph
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t, context=context, first_pass=first))
    twin = _main(fig).child_axes[0]
    forward, _ = twin._functions
    assert float(forward(np.array([ROUGH_PI_AMPLITUDE * 1e3]))[0]) == pytest.approx(1.0)
    assert analysis.prefactor_vs_state(fit) == pytest.approx(fit["opt_amp"] / ROUGH_PI_AMPLITUDE)
    plt.close(fig)


def test_04b_power_rabi_reference_levels_bracket_data(graph):
    """The simulator-reference |g> and |e> levels sit near the swing's ends (thermal start above |g>)."""
    _, _, ds, _, context = graph
    level_g, level_e = analysis.reference_levels(ds, context.after)
    signal = ds.signal.sel(qubit="q1").values
    assert level_g < signal[:3].mean() < level_g + 0.1 * (level_e - level_g)
    assert abs(signal.max() - level_e) < 0.1 * (level_e - level_g)


def test_04b_power_rabi_first_pass_needs_its_dataset(graph):
    """Without the first-pass dataset nothing is drawn for it (its own projection has another
    y scale), and the fit-gates card still reports the first pass."""
    _, fit, ds, _, context = graph
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, theme=t, context=context))
    assert not any(str(line.get_gid()).startswith("first pass") for line in _main(fig).get_lines())
    assert any(text.startswith("first pass") for text in _texts(fig))
    plt.close(fig)


def test_04b_power_rabi_export_uses_pre_update_state_without_refine(calibrated_state, make_ctx):
    fit = assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(calibrated_state), 0))
    assert fit["success"] and fit["refined_from"] is None


def test_04b_power_rabi_export_uses_pre_update_state_with_refine(calibrated_state, make_ctx):
    state = copy.deepcopy(calibrated_state)
    state["qubits"]["q1"]["xy"]["operations"]["x180"]["amplitude"] = ROUGH_PI_AMPLITUDE
    fit = assert_export_uses_pre_update_state(node, make_ctx(state, 0))
    assert fit["success"] and fit["refined_from"] is not None


def test_04b_power_rabi_true_value_is_an_edge_marker_in_graph_only(graph):
    """The true value is a top-edge triangle (no full-height line over the fitted one), and with
    the prepared-level lines it is drawn only in a graph run (ruling V)."""
    import dataclasses

    _, fit, ds, first, context = graph
    true = context.reference["x180.amplitude"]
    marks = style.Marks(previous=ROUGH_PI_AMPLITUDE, fit=fit["opt_amp"], true=true)
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, marks, theme=t, context=context, first_pass=first))
    main = _main(fig)
    tri = [line for line in main.get_lines() if line.get_label() == marks.true_label]
    assert len(tri) == 1 and tri[0].get_marker() == "v" and len(tri[0].get_xdata()) == 1
    assert not any(
        len(line.get_xdata()) == 2 and np.allclose(np.asarray(line.get_xdata(), dtype=float), true * 1e3)
        for line in main.get_lines()
    )
    fitted = [line for line in main.get_lines() if line.get_label() == "fitted value"]
    assert len(fitted) == 1
    plt.close(fig)

    outside = dataclasses.replace(context, in_graph=False)
    fig = _render(lambda t: analysis.plot_raw_data_with_fit(ds, fit, marks, theme=t, context=outside, first_pass=first))
    labels = [line.get_label() for line in _main(fig).get_lines()]
    assert marks.true_label not in labels
    assert not any("simulator reference" in text for text in _texts(fig))
    plt.close(fig)
