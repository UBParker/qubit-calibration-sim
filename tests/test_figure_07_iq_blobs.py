"""Layout C figure for 07_iq_blobs."""

import copy
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from calibration_utils import iq_blobs  # noqa: E402
from calibration_utils import style  # noqa: E402
from calibration_utils.common import NODE_ORDER, load_node  # noqa: E402
from tests.conftest import assert_export_uses_pre_update_state  # noqa: E402

node = load_node("07_iq_blobs.py")
NODE = node.NODE_NAME


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def _render(make, theme="light"):
    with style.theme_context(theme):
        fig = make(theme)
        style.fit_layout(fig)
        fig.canvas.draw()
    return fig


@pytest.fixture
def graph(graph_run_2026):
    """The shared seed-2026 graph run: the 07 dataset, fit, marks and figure context."""
    ctx = copy.deepcopy(graph_run_2026)
    entry = ctx.results[NODE]
    return ctx, entry["fit"], entry["dataset"], entry["marks"], entry["context"]


@pytest.fixture
def standalone(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state)
    fit = node.run(ctx)
    return ctx, fit, ctx.results[NODE]["dataset"]


def _plot(ds, fit, marks=None, context=None, theme="light"):
    return _render(lambda t: iq_blobs.plot_raw_data_with_fit(ds, fit, marks=marks, theme=t, context=context), theme)


@pytest.mark.slow  # smoke render in both themes, no physics or label checks
@pytest.mark.parametrize("theme", style.THEMES)
def test_07_iq_blobs_figure_both_themes(standalone, theme):
    _, fit, ds = standalone
    fig = _plot(ds, fit, theme=theme)
    try:
        assert len(fig.axes) >= 2
        assert any(style.NODE_SHORT[NODE] in t for t in _texts(fig))
    finally:
        plt.close(fig)


def test_07_iq_blobs_figure_without_reference(standalone):
    _, fit, ds = standalone
    marks = style.Marks(previous=None, fit=fit["ge_threshold"], true=None, true_label="simulator reference")
    fig = _plot(ds, fit, marks=marks)
    try:
        labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
        assert marks.true_label not in labels
        assert "SIMULATOR REFERENCE" not in _texts(fig)
    finally:
        plt.close(fig)


def _nan_fit() -> dict:
    nan = float("nan")
    return {
        "g_mean_iq": [nan, nan],
        "e_mean_iq": [nan, nan],
        "rotation_angle": nan,
        "iw_angle": nan,
        "ge_threshold": nan,
        "rus_threshold": nan,
        "fidelity": nan,
        "confusion_matrix": [[nan, nan], [nan, nan]],
        "success": False,
    }


def test_07_iq_blobs_figure_renders_on_failed_fit(standalone):
    _, _, ds = standalone
    fig = _plot(ds, _nan_fit())
    try:
        texts = _texts(fig)
        assert any("fit failed" in t for t in texts)
        assert not any("nan" in t.lower().split() or "inf" in t.lower() for t in texts)
    finally:
        plt.close(fig)


def test_07_failed_fit_histogram_has_no_reference_marker(graph):
    """With the fit failed the histogram axis is unrotated, so the rotated-frame simulator
    reference threshold is not drawn on it, even when the marks carry one."""
    _, _, ds, marks, context = graph
    assert marks.true is not None
    fig = _plot(ds, _nan_fit(), marks=marks, context=context)
    try:
        hist = next(ax for ax in fig.axes if "not rotated: fit failed" in ax.get_xlabel())
        labels = [line.get_label() for line in hist.get_lines()]
        assert marks.true_label not in labels
        x_marks = {round(float(x), 9) for line in hist.get_lines() for x in np.atleast_1d(line.get_xdata())}
        assert round(marks.true * 1e3, 9) not in x_marks
    finally:
        plt.close(fig)


def test_07_iq_blobs_figure_renders_rejected_fit(calibrated_state, make_ctx):
    """A real run with a near-zero x180: finite values, fidelity under the gate, success False."""
    state = copy.deepcopy(calibrated_state)
    state["qubits"]["q1"]["xy"]["operations"]["x180"]["amplitude"] *= 0.05
    ctx = make_ctx(state)
    fit = node.run(ctx)
    assert not fit["success"] and all(math.isfinite(fit[k]) for k in ("fidelity", "ge_threshold", "iw_angle"))
    entry = ctx.results[NODE]
    assert entry["marks"].fit is None
    fig = _plot(entry["dataset"], fit, marks=entry["marks"], context=entry["context"])
    try:
        texts = _texts(fig)
        labels = [line.get_label() for ax in fig.axes for line in ax.get_lines()]
        assert any("rejected fit" in t for t in texts)
        assert "rejected fit" in labels
        assert not {"fitted value", "threshold (fit)", "threshold"} & set(labels)
        assert not any("nan" in t.lower() or "inf" in t.lower() for t in texts)
        assert not any("integration_weights_angle" in t for t in texts)  # no accepted-fit arc
        state_change = next(c for c in iq_blobs.node_cards(fit, entry["context"], entry["dataset"]) if c.title == "State change")
        assert all("uncalculated -> uncalculated" in line or "0.0 -> 0.0" in line for line in state_change.lines)
    finally:
        plt.close(fig)


def test_07_iq_blobs_figure_layout_c(graph):
    ctx, fit, ds, marks, context = graph
    assert context.in_graph
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        texts = _texts(fig)
        for name in NODE_ORDER:
            assert style.NODE_SHORT[name] in texts, name
        for block in [b for b in iq_blobs.sequence_blocks(context.before, fit) if b.label]:
            assert block.label in texts, block.label
        for title in ("Measured", "State change", "Simulator reference", "Assignment", "Error budget"):
            assert title.upper() in texts, title
        titles = [c.title for c in iq_blobs.node_cards(fit, context, ds)]
        assert titles.index("Assignment") + 1 == titles.index("Error budget")
        assert not any("truth" in t.lower() for t in texts)
    finally:
        plt.close(fig)


def test_07_iq_blobs_physics_labels(graph):
    ctx, fit, ds, marks, context = graph
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        hist = [ax for ax in fig.axes if "rotated I" in ax.get_xlabel()]
        assert len(hist) == 1
        assert hist[0].get_yscale() == "log"
        texts = _texts(fig)
        assert any("(gg + ee)/2" in t for t in texts)
        assert any("1 - P(e|g) - P(g|e)" in t for t in texts)
        assert any("thermal" in t for t in texts)
        assert any("decay" in t and "readout" in t for t in texts)  # streak callout
        assert any("weights_angle" in t.replace("_\n", "_") for t in texts)  # rotation arc (may wrap at "_")
        assert any("QM's fidelity" in t for t in texts)
        assert any("decay in window (+ pi-pulse error)" in t for t in texts)
        assert not any("RUS" in t for t in texts)
    finally:
        plt.close(fig)


def test_07_reference_card_matches_measurement(graph):
    """The convergence reference reads out at the true g/e midpoint (Ruling AB); 07 reads out at
    08a's pick, 144 kHz above it on seed 2026, and the blob axis turns with the readout frequency
    (25 deg here). The 07 figure therefore takes the reference angle and threshold at the run's own
    tone (measured within 0.2 deg and 0.16 mV of the fit) and keeps the midpoint fidelity."""
    ctx, fit, ds, marks, context = graph
    ref = context.reference
    assert ref["fidelity"] == ctx.reference["fidelity"]
    assert abs(fit["fidelity"] - ref["fidelity"]) < 0.01
    before = next(r.before for r in ctx.history if r.name == NODE)
    assert ref["reference_tone"] == before["qubits"]["q1"]["resonator"]["f_01"]
    assert abs(math.degrees(iq_blobs.wrap_angle(fit["iw_angle"] - ref["iw_angle"]))) < 3
    assert abs(fit["ge_threshold"] - ref["ge_threshold"]) < 1.0e-3
    assert marks.true == ref["ge_threshold"]
    cards = {c.title: c for c in iq_blobs.node_cards(fit, context, ds=ds)}
    assert "angle, threshold at this run's tone" in cards["Simulator reference"].lines


def test_07_assignment_card_names_both_fidelities(graph):
    ctx, fit, ds, marks, context = graph
    cards = {c.title: c for c in iq_blobs.node_cards(fit, context, ds=ds)}
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    text = "\n".join(cards["Assignment"].lines)
    assert "(gg + ee)/2, QM's fidelity, stored" in text
    assert f"{100 * (gg + ee) / 2:.1f} ±" in text
    assert f"{100 * (1 - ge - eg):.1f} ±" in text
    assert "(stat.)" in text
    for name, value in (("gg", gg), ("ge", ge), ("eg", eg), ("ee", ee)):
        assert f"{name} {100 * value:.1f}" in text
    measured = " ".join(cards["Measured"].lines)
    assert "mV" in measured and "SNR" in measured
    assert "true value" not in " ".join(cards["Simulator reference"].lines)


def test_07_thermal_callout_matches_budget(graph):
    ctx, fit, ds, marks, context = graph
    b = iq_blobs._error_budget(ds, fit)
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        callout = next(t for t in _texts(fig) if t.startswith(f"{style.KET_G}-prepared shots"))
        assert f"{100 * b['thermal']:.1f} %" in callout
    finally:
        plt.close(fig)


def test_07_cards_standalone_have_no_reference_card(standalone):
    _, fit, ds = standalone
    titles = [c.title for c in iq_blobs.node_cards(fit, None, ds=ds)]
    assert "Simulator reference" not in titles
    assert {"Measured", "Assignment", "Error budget"} <= set(titles)


def test_07_error_budget_from_data(graph):
    ctx, fit, ds, _, _ = graph
    b = iq_blobs._error_budget(ds, fit)
    (gg, ge), (eg, ee) = fit["confusion_matrix"]
    # the shot counts are recomputed from the data and agree with the discriminator's matrix
    assert b["p_e_given_g"] == pytest.approx(ge, abs=1e-3)
    assert b["p_g_given_e"] == pytest.approx(eg, abs=1e-3)
    # thermal is |g>-prepared shots in the |e> region minus the Gaussian overlap of the |g> blob
    assert b["thermal"] == pytest.approx(b["p_e_given_g"] - b["overlap_g"], abs=1e-12)
    # decay is the rest of P(g|e) once thermal (flipped by the x180) and overlap are removed
    assert b["decay"] == pytest.approx(b["p_g_given_e"] - b["thermal"] - b["overlap_e"], abs=1e-12)
    # simulator: 1.2 % thermal, L/(2 T1) = 1 us / 96 us = 1.0 % decay; loose bounds for 4000 shots
    assert 0.004 < b["thermal"] < 0.025
    assert 0.002 < b["decay"] < 0.025
    # Separation about 4 sigma with ring-up in the window: Gaussian overlap about Phi(-2) = 2.3 %.
    assert 0 < b["overlap_g"] < 0.04 and 0 < b["overlap_e"] < 0.04
    assert b["separation_sigma"] == pytest.approx(b["separation"] / b["sigma"])
    assert b["snr"] == pytest.approx(b["separation"] / (2 * b["sigma"]))


def test_07_error_budget_tracks_the_data(graph):
    """Removing the |g>-prepared shots in the |e> tail drops the thermal estimate."""
    ctx, fit, ds, _, _ = graph
    d = ds.sel(qubit="q1")
    I_rot, _ = iq_blobs.rotate(d.I_g.values, d.Q_g.values, fit["rotation_angle"])
    b = iq_blobs._error_budget(ds, fit)
    keep = I_rot < b["mu_g"] + 3 * b["sigma"]
    n = int(keep.sum())
    clean = xr.Dataset(
        {
            "I_g": (("qubit", "n_runs"), d.I_g.values[keep][None, :]),
            "Q_g": (("qubit", "n_runs"), d.Q_g.values[keep][None, :]),
            "I_e": (("qubit", "n_runs"), d.I_e.values[:n][None, :]),
            "Q_e": (("qubit", "n_runs"), d.Q_e.values[:n][None, :]),
        },
        coords={"qubit": ["q1"], "n_runs": np.arange(n)},
    )
    assert iq_blobs._error_budget(clean, fit)["thermal"] < b["thermal"] / 4


def test_07_threshold_marks_on_histogram(graph):
    ctx, fit, ds, marks, context = graph
    assert marks.true == context.reference["ge_threshold"]  # the reference at the run's tone
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        hist = next(ax for ax in fig.axes if "rotated I" in ax.get_xlabel())
        labels = [line.get_label() for line in hist.get_lines()]
        assert "simulator reference" in labels
        assert "threshold (fit)" in labels
        assert "true value" not in labels
    finally:
        plt.close(fig)


def test_07_export_uses_pre_update_state(calibrated_state, make_ctx):
    assert_export_uses_pre_update_state(node, make_ctx(copy.deepcopy(calibrated_state), 0))


def _card_body_size() -> float:
    """The shared card body size: one short card in a tall axes, where cards_column does not shrink."""
    fig = plt.figure(figsize=(style.CARDS_W, 4))
    try:
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        artists = style.cards_column(ax, [style.Card("t", ("x",))], "light")
        return next(a.get_fontsize() for a in artists if isinstance(a, Text) and a.get_text() == "x")
    finally:
        plt.close(fig)


def test_07_cards_use_the_shared_text_size(graph):
    """The seed-2026 cards render at the shared card size, every line on one row (no wrapping)."""
    ctx, fit, ds, marks, context = graph
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        cards = iq_blobs.node_cards(fit, context, ds)
        cards_ax = next(ax for ax in fig.axes if any(t.get_text() == "MEASURED" for t in ax.texts))
        heads = {c.title.upper() for c in cards} | {"pass"}
        body = [t for t in cards_ax.texts if t.get_text() not in heads]
        assert [t.get_text() for t in body] == [line for c in cards for line in c.lines]
        assert {t.get_fontsize() for t in body} == {_card_body_size()}
    finally:
        plt.close(fig)


def test_07_error_budget_breaks_p_g_given_e_deliberately(graph):
    ctx, fit, ds, marks, context = graph
    lines = next(c for c in iq_blobs.node_cards(fit, context, ds) if c.title == "Error budget").lines
    assert lines[1].startswith("P(g|e)") and lines[1].split()[-2] == "thermal"
    assert lines[2].strip().startswith("+ decay") and "+ overlap" in lines[2]


def test_07_callouts_leave_the_threshold_line_visible(graph):
    """No callout background covers the dashed threshold line in the IQ panel."""
    ctx, fit, ds, marks, context = graph
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        ax = next(a for a in fig.axes if a.get_xlabel() == "I (mV)")
        line = next(ln for ln in ax.get_lines() if ln.get_label() == "threshold")
        xy = np.column_stack(line.get_data())
        t = np.linspace(0, 1, 4000)[:, None]
        pts = ax.transData.transform(xy[0] + t * (xy[1] - xy[0]))
        panel = ax.patch.get_window_extent()
        pts = [p for p in pts if panel.contains(*p)]
        assert pts
        renderer = fig.canvas.get_renderer()
        callouts = [a for a in ax.texts if a.get_bbox_patch() is not None]
        assert len(callouts) == 3
        for a in callouts:
            box = a.get_bbox_patch().get_window_extent(renderer)
            assert not any(box.contains(*p) for p in pts), a.get_text()
    finally:
        plt.close(fig)


def test_07_negative_numbers_use_unicode_minus(graph):
    """Signed numbers read with U+2212; the formula text "1 - P(e|g) - P(g|e)" keeps its hyphens."""
    ctx, fit, ds, marks, context = graph
    cards = {c.title: c for c in iq_blobs.node_cards(fit, context, ds)}
    measured = cards["Simulator reference"].lines[1]
    assert re.match(r"measured [−+]\d", measured) and "(stat.)" in measured  # seed 2026: +0.0 pp
    fig = _plot(ds, fit, marks=marks, context=context)
    try:
        texts = _texts(fig)
        plain = [t for t in texts if not t.startswith("$")]  # log tick labels are mathtext, minus typeset
        assert not [t for t in plain if re.search(r"(?<![\w)])-\d", t)]
        assert any("1 - P(e|g) - P(g|e)" in t for t in texts)
    finally:
        plt.close(fig)


def test_07_minus_helpers():
    assert iq_blobs._minus(f"{-0.25:+.2f}") == "−0.25"
    assert iq_blobs._minus_numbers("threshold: -0.100 -> -0.262 mV") == "threshold: −0.100 -> −0.262 mV"
