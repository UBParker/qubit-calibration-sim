"""Layout C figure for 02a_resonator_spectroscopy: renders, labels the linewidths correctly."""

import copy
import math

import pytest
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

import calibration_utils.resonator_spectroscopy as rs
from calibration_utils import style
from calibration_utils.common import figure_context, load_node
from tests.conftest import assert_export_uses_pre_update_state

node = load_node("02a_resonator_spectroscopy.py")
NODE = node.NODE_NAME


def _texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _labels(fig: Figure) -> list[str]:
    return [line.get_label() for line in fig.findobj(Line2D)]


def _render(ds, fit, marks=None, theme="light", context=None) -> Figure:
    with style.theme_context(theme):
        fig = rs.plot_raw_data_with_fit(ds, fit, marks=marks, theme=theme, context=context)
        style.fit_layout(fig)
        fig.canvas.draw()
    return fig


@pytest.fixture
def standalone(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    fit = node.run(ctx)
    return ctx.results[NODE]["dataset"], fit


@pytest.fixture
def graph_run(graph_run_2026):
    """02a's dataset, fit, marks and figure context from the shared seed-2026 graph run."""
    ctx = copy.deepcopy(graph_run_2026)
    record = next(r for r in ctx.history if r.name == NODE)
    context = figure_context(ctx, NODE, record.before, record.fit)
    context.after = record.after  # the state right after 02a, not after the whole graph
    marks = style.Marks(
        previous=record.before["qubits"]["q1"]["resonator"]["f_01"],
        fit=record.fit["frequency"],
        true=ctx.reference["resonator.f_01"],
    )
    return record.dataset, record.fit, marks, context


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_02a_figure_both_themes(standalone, theme):
    ds, fit = standalone
    fig = _render(ds, fit, theme=theme)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    assert style.NODE_SHORT[NODE] in _texts(fig)


def test_02a_figure_without_reference(standalone):
    ds, fit = standalone
    marks = style.Marks(previous=ds.attrs["readout_frequency_guess"], fit=fit["frequency"], true=None)
    labels = _labels(_render(ds, fit, marks=marks))
    assert marks.true_label not in labels
    assert not any("true" in label for label in labels)


def test_02a_figure_renders_on_failed_fit(standalone):
    ds, _ = standalone
    nan = float("nan")
    failed = {
        "frequency": nan,
        "frequency_error": nan,
        "fwhm": nan,
        "snr": 0.0,
        "kappa": nan,
        "kappa_error": nan,
        "notch_depth": nan,
        "notch_params": {},
        "fit_params": {},
        "success": False,
    }
    fig = _render(ds, failed, marks=style.Marks(previous=ds.attrs["readout_frequency_guess"]))
    assert any("fit failed" in t for t in _texts(fig))


def test_02a_figure_rejected_fit(graph_run):
    # A real dataset with a finite-valued fit that the gates rejected.
    ds, fit, marks, context = graph_run
    rejected = {**fit, "success": False, "kappa": float("nan"), "kappa_error": float("nan"), "notch_params": {}}
    fig = _render(ds, rejected, marks=style.Marks(previous=marks.previous, fit=None, true=marks.true), context=context)
    texts, labels = _texts(fig), _labels(fig)
    assert not any("nan" in t.lower() or "inf" in t.lower() for t in texts)
    assert "rejected fit" in labels
    assert "fitted f_r" not in labels and "Lorentzian fit" not in labels
    assert not any(t.startswith("shift") for t in texts)
    assert not any("dip FWHM" in t for t in texts)


def test_02a_figure_layout_c(graph_run):
    ds, fit, marks, context = graph_run
    blocks = rs.sequence_blocks(context.before, fit)
    for theme in style.THEMES:
        texts = _texts(_render(ds, fit, marks=marks, theme=theme, context=context))
        assert len(context.node_order) == 8
        for name in context.node_order:
            assert style.NODE_SHORT[name] in texts
        for block in blocks:
            if block.label:
                assert block.label in texts
        assert "MEASURED" in texts and "STATE CHANGE" in texts and "TRUE VALUE" in texts


def test_02a_sequence_reads_sweep_range(graph_run):
    _, fit, _, context = graph_run
    readout = next(b for b in rs.sequence_blocks(context.before, fit) if b.kind == "readout")
    assert f"1{style.NBSP}{style.US}" in readout.label
    assert "10.00" in readout.label


def test_02a_standalone_has_no_true_value_card(standalone):
    ds, fit = standalone
    assert "TRUE VALUE" not in _texts(_render(ds, fit))


def test_02a_physics_labels(graph_run):
    ds, fit, marks, context = graph_run
    fig = _render(ds, fit, marks=marks, context=context)
    texts = _texts(fig)
    kappa = f"{fit['kappa'] / 1e6:.3f}"
    fwhm = f"{fit['fwhm'] / 1e6:.2f}"
    assert kappa != fwhm and math.isfinite(fit["kappa"])
    assert any("dip FWHM" in t for t in texts)
    # kappa is labelled with the notch-fit value; the dip FWHM is never called kappa (the only
    # text with both says the FWHM is "not κ").
    assert any(f"{style.KAPPA} (notch fit) {kappa}" in t for t in texts)
    for t in texts:
        if style.KAPPA in t and fwhm in t:
            assert f"not {style.KAPPA}" in t, t
    main = next(a for a in fig.axes if a.get_gid() == "main")
    residual = next(a for a in fig.axes if a.get_gid() == "residuals")
    assert residual.get_shared_x_axes().joined(residual, main)
    # The column's shared detuning axis is labelled once, under the lowest panel (the phase).
    shared = [a for a in fig.axes if a is not main and a.get_shared_x_axes().joined(a, main)]
    lowest = min(shared, key=lambda a: a.get_position().y0)
    assert "MHz" in lowest.get_xlabel()
    assert residual.get_xlabel() == ""
    assert any("phase, cable delay removed" in t for t in texts)
    assert any(f"dressed {style.KET_G}" in t for t in texts)
    assert any(f"6{style.SIGMA}" in t for t in texts)
    shift = (fit["frequency"] - ds.attrs["readout_frequency_guess"]) / 1e6
    assert any(f"{shift:+.2f}{style.NBSP}MHz" in t for t in texts)
    labels = _labels(fig)
    assert {"previous f_r", "fitted f_r", "true f_r", "Lorentzian fit", "notch fit"} <= set(labels)
    assert not any("f_01" in label for label in labels)
    # The fitted marker is drawn above the true one, which sits a few kHz away.
    lines = main.get_lines()
    order = [line.get_label() for line in lines]
    assert order.index("fitted f_r") > order.index("true f_r")


def test_02a_labels_stay_inside_main_axes(graph_run):
    ds, fit, marks, context = graph_run
    fig = _render(ds, fit, marks=marks, context=context)
    main = next(a for a in fig.axes if a.get_gid() == "main")
    renderer = fig.canvas.get_renderer()
    box = main.get_window_extent(renderer)
    for t in main.texts:
        if t.get_text():
            extent = t.get_window_extent(renderer)
            assert box.x0 - 1 <= extent.x0 and extent.x1 <= box.x1 + 1, t.get_text()


def test_02a_kappa_bias_matches_label(calibrated_state):
    # Noiseless |S21| over the 02a grid: the |g> and |e> window means mixed at the thermal |e>
    # population, fitted with the single-notch model. The relative kappa bias (ring-up in the 1 us
    # window plus the thermal dip) must round to the percentage the card shows.
    import numpy as np

    from calibration_utils.common import readout_settings
    from simulator.device import TRUE_DEVICE as device
    from simulator.qpu import SimulatedQPU

    dfs = node.detunings_hz(node.Parameters())
    readout, qpu = readout_settings(calibrated_state), SimulatedQPU()
    p = device.thermal_excited_population
    z = np.array([(1 - p) * qpu.expected_iq(0, readout, frequency=device.f_res_g + f) + p * qpu.expected_iq(1, readout, frequency=device.f_res_g + f) for f in dfs])
    y, x = np.abs(z) * 1e3, dfs / 1e6
    (_, _, _, kappa_mhz), _ = rs._fit_notch(x, y, 0.0, 1.0, float(y.max()), float(x[1] - x[0]), float(x[-1] - x[0]))
    bias = (kappa_mhz / (device.kappa / 1e6) - 1) * 100
    assert f"{bias:+.1f}{style.NBSP}%" == rs.KAPPA_BIAS


def test_02a_cards_wording(graph_run):
    _, fit, _, context = graph_run
    # Rejected (finite values) and failed (NaN) fits say so in the State change card.
    rejected = {**fit, "success": False}
    failed = {**fit, "success": False, "frequency": float("nan")}
    for bad, word in ((rejected, "rejected fit"), (failed, "fit failed")):
        change = next(c for c in rs.node_cards(bad, context) if c.title == "State change")
        assert f"unchanged: {word}" in change.lines
    # Negative numbers in the True value card use U+2212, not a hyphen.
    true = context.reference["resonator.f_01"]
    low = {**fit, "frequency": true - 5e3}
    card = next(c for c in rs.node_cards(low, context) if c.title == "True value")
    body = " ".join(card.lines[1:])
    assert "−5.0" in body and "-" not in body


def test_02a_export_uses_pre_update_state(rough_state, make_ctx):
    assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(rough_state), 0))


def test_02a_figure_width_matches_other_nodes(graph_run):
    # One stacked column keeps the saved figure near the other node figures' width (1393 to
    # 1913 px), so text is not shrunk further than theirs when the README scales it to 900 px.
    ds, fit, marks, context = graph_run
    fig = _render(ds, fit, marks=marks, context=context)
    width_px = fig.get_size_inches()[0] * style.DPI
    assert 1400 <= width_px <= 1700, width_px
