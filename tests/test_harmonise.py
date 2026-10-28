"""Shared style and plumbing that every node figure adopts (Task 13x harmonisation items)."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from calibration_utils import common, style  # noqa: E402
from calibration_utils.common import NODE_ORDER, SETUP_NODES  # noqa: E402

from .test_style import _contrast  # noqa: E402

BASE_ROLES = ("data", "fit", "guess", "truth", "reference", "ground", "excited", "leak", "ok", "fail")


# 2. title_block fail colour


@pytest.mark.parametrize("theme", style.THEMES)
def test_title_block_headline_fail_colour(theme):
    with style.theme_context(theme) as p:
        for status, want in (("ok", p.fit), ("fail", p.fail)):
            fig, _, _ = style.fit_axes(theme)
            style.title_block(fig, "05_T1", "T1", "fit failed", theme, status=status)
            texts = fig.texts + [a for a in fig.artists if hasattr(a, "get_text")]
            headline = next(t for t in texts if t.get_text() == "fit failed")
            assert headline.get_color() == want
            plt.close(fig)
        fig, _, _ = style.fit_axes(theme)
        with pytest.raises(ValueError):
            style.title_block(fig, "05_T1", "T1", "x", theme, status="maybe")
        plt.close(fig)


# 3. public mix


def test_mix_is_public_and_aliased():
    assert style.mix("#000000", "#FFFFFF", 0.0) == "#ffffff"
    assert style.mix("#000000", "#FFFFFF", 1.0) == "#000000"
    assert style.mix("#FF0000", "#0000FF", 0.5) == "#800080"
    assert style._mix is style.mix


# 4, 20, 22. node_layout extras


def _main_box_inches(lay, ax):
    lay.fig.canvas.draw()
    w, h = lay.fig.get_size_inches()
    pos = ax.get_position()
    return pos.x0 * w, (1 - pos.y1) * h, pos.y0 * h  # left, top gap, bottom gap


def test_node_layout_top_and_bottom_extra():
    with style.theme_context("light"):
        base = style.node_layout("light")
        extra = style.node_layout("light", top_extra=0.3, bottom_extra=0.4)
        _, top0, bot0 = _main_box_inches(base, base.main_axes[0])
        _, top1, bot1 = _main_box_inches(extra, extra.residual_axes[0])
        _, top1_main, _ = _main_box_inches(extra, extra.main_axes[0])
        _, _, bot0_res = _main_box_inches(base, base.residual_axes[0])
        assert top1_main == pytest.approx(top0 + 0.3, abs=1e-6)
        assert bot1 == pytest.approx(bot0_res + 0.4, abs=1e-6)
        # Panel sizes stay the same; the figure grows.
        h0, h1 = base.fig.get_size_inches()[1], extra.fig.get_size_inches()[1]
        assert h1 == pytest.approx(h0 + 0.7)
        assert extra.main_axes[0].get_position().height * h1 == pytest.approx(
            base.main_axes[0].get_position().height * h0
        )
        # The cards column ends where the residual row ends.
        renderer = extra.fig.canvas.get_renderer()
        cards = extra.cards_ax.get_window_extent(renderer)
        res = extra.residual_axes[0].get_window_extent(renderer)
        assert cards.y0 == pytest.approx(res.y0, abs=0.5)
        plt.close("all")


def test_node_layout_extras_keep_fit_layout_deterministic(tmp_path):
    def make(theme):
        lay = style.node_layout(theme, top_extra=0.3, bottom_extra=0.4)
        ax = lay.main_axes[0]
        x = np.linspace(0, 1, 20)
        ax.plot(x, 1e5 * x, "o", label="data")
        sec = ax.secondary_xaxis("top", functions=(lambda v: 2 * v, lambda v: v / 2))
        sec.set_xlabel("prefactor")
        style.place_legend(ax, theme, title="panel")
        return lay.fig

    a = style.save_themed(make, "extra", tmp_path / "a")
    b = style.save_themed(make, "extra", tmp_path / "b")
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes()


def test_place_legend_clears_top_twin_axis():
    with style.theme_context("light"):
        lay = style.node_layout("light", top_extra=0.5)
        ax = lay.main_axes[0]
        ax.plot([0, 1], [0, 1], label="data")
        sec = ax.secondary_xaxis("top", functions=(lambda v: 2 * v, lambda v: v / 2))
        sec.set_xlabel("prefactor")
        leg = style.place_legend(ax, "light", title="panel")
        lay.fig.canvas.draw()
        renderer = lay.fig.canvas.get_renderer()
        twin = sec.get_tightbbox(renderer)
        assert leg.get_window_extent(renderer).y0 >= twin.y1 - 0.5
        plt.close(lay.fig)


def test_node_layout_width_ratios():
    with style.theme_context("light"):
        lay = style.node_layout("light", ncols=3, residuals=False, width_ratios=[2.0, 1.0, 1.0])
        widths = [ax.get_window_extent().width for ax in lay.main_axes]
        assert widths[0] == pytest.approx(2 * widths[1], rel=1e-6)
        assert widths[1] == pytest.approx(widths[2], rel=1e-6)
        # Gaps stay GAP_W inches before and after fit_layout widens the figure.
        for ax in lay.main_axes:
            ax.set_ylabel("long label")
            ax.plot([0, 1], [1e5, 2e5])
        style.fit_layout(lay.fig)
        lay.fig.canvas.draw()
        boxes = [ax.get_window_extent() for ax in lay.main_axes]
        widths2 = [b.width for b in boxes]
        assert widths2[0] == pytest.approx(2 * widths2[1], rel=1e-6)
        gaps = [(b.x0 - a.x1) / lay.fig.dpi for a, b in zip(boxes, boxes[1:])]
        assert gaps[0] == pytest.approx(gaps[1], abs=1e-6)
        with pytest.raises(ValueError):
            style.node_layout("light", ncols=2, width_ratios=[1.0])
        plt.close("all")


def test_node_layout_two_residual_rows():
    with style.theme_context("light"):
        lay = style.node_layout("light", ncols=2, residual_rows=2)
        assert len(lay.residual_strips) == 2
        for main, strips, first in zip(lay.main_axes, lay.residual_strips, lay.residual_axes):
            assert len(strips) == 2 and strips[0] is first
            shared = main.get_shared_x_axes()
            assert all(shared.joined(main, s) for s in strips)
            assert strips[0].get_position().y0 > strips[1].get_position().y0
            for s in strips:
                assert s.get_position().height == pytest.approx(strips[0].get_position().height)
        # Only the lowest strip carries x tick labels.
        assert not lay.residual_strips[0][0].xaxis.get_tick_params()["labelbottom"]
        plt.close(lay.fig)
        lay = style.node_layout("light")
        assert lay.residual_strips == [[lay.residual_axes[0]]]
        plt.close(lay.fig)
        lay = style.node_layout("light", residuals=False)
        assert lay.residual_strips == [[]]
        plt.close(lay.fig)


# 18. fit_layout measures subgridspec residual strips


def _labels_inside(fig, axes_list):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes_list:
        for label in ax.get_yticklabels() + [ax.yaxis.label]:
            if label.get_text() and label.get_visible():
                yield ax, label.get_window_extent(renderer)


def test_fit_layout_measures_subgridspec_strips():
    with style.theme_context("light"):
        lay = style.node_layout("light", ncols=2)
        gs = lay.fig._style_layout["gs"]
        strips_by_col = []
        for col, (main, res) in enumerate(zip(lay.main_axes, lay.residual_axes)):
            res.remove()
            sub = gs[1, col].subgridspec(2, 1, hspace=0.1)
            strips = [lay.fig.add_subplot(sub[i, 0], sharex=main) for i in range(2)]
            main.plot([0, 1], [0, 1])  # short tick labels on the main axes
            for s in strips:
                s.plot([0, 1], [-1.0, 1.0])
                s.set_yticks([-1.0, 0.0, 1.0], ["−1234567.89", "0", "1234567.89"])  # long labels
                s.set_ylabel("resid. long label")
            strips_by_col.append(strips)
        style.fit_layout(lay.fig)
        boxes = list(_labels_inside(lay.fig, [s for col in strips_by_col for s in col]))
        assert boxes
        left_col_right_edge = lay.main_axes[0].get_window_extent().x1
        for ax, box in boxes:
            assert box.x0 >= 0
            if ax in strips_by_col[1]:
                assert box.x0 > left_col_right_edge
        plt.close(lay.fig)


# 5. place_legend title on the first row


def test_place_legend_title_on_first_row_when_wrapped():
    with style.theme_context("light"):
        lay = style.node_layout("light")
        ax = lay.main_axes[0]
        for k in range(5):
            ax.plot([0, 1], [k, k], label=f"series {k}")
        leg = style.place_legend(ax, "light", ncols=2, title="panel")
        lay.fig.canvas.draw()
        renderer = lay.fig.canvas.get_renderer()
        title = next(t for t in ax.texts if t.get_text() == "panel")
        tb = title.get_window_extent(renderer)
        texts = leg.get_texts()
        first = texts[0].get_window_extent(renderer)
        last_row = texts[2].get_window_extent(renderer)  # column-major: entry 2 is row 3
        assert first.y0 > last_row.y1  # the legend did wrap
        assert (tb.y0 + tb.y1) / 2 == pytest.approx((first.y0 + first.y1) / 2, abs=2.0)
        assert tb.x1 < leg.get_window_extent(renderer).x0
        plt.close(lay.fig)


# 7. pulse_sequence sweep text on drive blocks


def test_pulse_sequence_sweeps_drive_blocks():
    with style.theme_context("light") as p:
        fig, ax = plt.subplots()
        blocks = [
            style.SeqBlock("drive", "x180", swept="0.2 to 1.8 × amp"),
            style.SeqBlock("readout", "readout", line="resonator"),
        ]
        style.pulse_sequence(ax, blocks, "light")
        sweep = [t for t in ax.texts if t.get_text() == "0.2 to 1.8 × amp"]
        assert len(sweep) == 1 and sweep[0].get_color() == p.fit
        brackets = [ln for ln in ax.lines if len(ln.get_xdata()) == 4]
        assert len(brackets) == 1
        plt.close(fig)


# 9. typography


def test_typography_constants():
    assert style.US == "µs"
    # Kets are mathtext: the bar and letter in the text font, the angle bracket from Computer
    # Modern (Inter has no U+27E9 and the DejaVu Sans one reads as ")" at figure size).
    assert (style.KET_G, style.KET_E, style.KET_F) == tuple(style.ket(s) for s in "gef")
    assert style.KET_G == r"$\mathdefault{|g}\rangle$"
    assert "⟩" not in style.KET_G + style.KET_E + style.KET_F
    for theme in style.THEMES:
        with style.theme_context(theme):
            assert plt.rcParams["mathtext.fontset"] == "cm"
    assert (style.KAPPA, style.CHI, style.SIGMA) == ("κ", "χ", "σ")
    assert "rangle" in style.__doc__ and "µs" in style.__doc__


@pytest.mark.parametrize("theme", style.THEMES)
def test_typography_glyphs_render_without_warnings(theme):
    text = f"T1 = 47.7 {style.US}  {style.KET_G} {style.KET_E} {style.KET_F}  {style.KAPPA} {style.CHI} {style.SIGMA}"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with style.theme_context(theme):
            fig, ax = plt.subplots()
            ax.set_title(text)
            ax.text(0.5, 0.5, text, fontweight="semibold")
            ax.set_xlabel(text)
            fig.canvas.draw()
            plt.close(fig)


def test_fallback_weight_notice_is_filtered_only_for_dejavu():
    import logging

    def record(*args):
        return logging.LogRecord("matplotlib.font_manager", logging.WARNING, "", 0,
                                 "findfont: Failed to find font weight %s for %s, now using %s.", args, None)

    assert not style._quiet_fallback_weight(record("semibold", "DejaVu Sans", 700))
    assert style._quiet_fallback_weight(record("semibold", "Inter", 400))


# 6. format_change no-break space (the exact strings live in test_style.py)


def test_format_change_keeps_value_and_unit_together():
    got = style.format_change("f", 7.04e9, 7.04124e9, 1e-9, "GHz", "{:.5f}")
    assert got == "f: 7.04000 -> 7.04124 GHz"
    assert style.format_change("F", 0.9, None, 1.0, "", "{:.2f}") == "F: 0.90 -> uncalculated"


# 17. second-series palette roles


@pytest.mark.parametrize("theme", style.THEMES)
def test_second_series_roles(theme):
    p = style.palette(theme)
    others = {getattr(p, role).lower() for role in BASE_ROLES}
    assert p.data2.lower() not in others and p.fit2.lower() not in others
    assert p.data2.lower() != p.fit2.lower()
    for colour in (p.data2, p.fit2):
        assert _contrast(colour, p.panel) >= 3.0
        assert _contrast(colour, p.bg) >= 3.0
    # Far enough in RGB from every hue role that a second trace never reads as one of them.
    for role in ("fit", "excited", "leak", "ground", "guess", "truth"):
        for colour in (p.fit2,):
            dist = np.linalg.norm(np.array(to_rgb(colour)) - np.array(to_rgb(getattr(p, role))))
            assert dist > 0.2, (role, colour)


# 21. mark_values custom labels


def test_mark_values_custom_labels():
    fig, ax = plt.subplots()
    style.mark_values(
        ax,
        style.Marks(previous=1.0, fit=2.0, true=3.0),
        "light",
        previous_label="state before",
        fit_label="rejected fit",
        true_label="device value",
    )
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["state before", "rejected fit", "device value"]
    plt.close(fig)
    fig, ax = plt.subplots()
    style.mark_values(ax, style.Marks(previous=1.0, fit=2.0, true=3.0, true_label="simulator reference"), "light")
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["previous value", "fitted value", "simulator reference"]
    plt.close(fig)


# 1. FigureContext.in_graph


def test_figure_context_in_graph_flag(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    fc = common.figure_context(ctx, "05_T1", {}, {"success": True})
    assert fc.in_graph is False
    ctx.graph_nodes = list(NODE_ORDER)
    fc = common.figure_context(ctx, "05_T1", {}, {"success": True})
    assert fc.in_graph is True
    # Positional construction from before this field still works.
    old = style.FigureContext(["05_T1"], {}, None, None, None, None)
    assert old.in_graph is False


# 11. sweep range convention


def test_sweep_range_convention_documented():
    assert common.SWEEP_RANGE_SUFFIX == "_range"
    assert common.sweep_range_key("idle_time") == "idle_time_range"
    assert "_range" in common.__doc__


# 13. store_figure_inputs


def test_store_figure_inputs(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    marks = style.Marks(previous=1.0, fit=2.0)
    context = common.figure_context(ctx, "05_T1", {}, {"success": True})
    common.store_figure_inputs(ctx, "05_T1", marks, context)
    assert ctx.results["05_T1"]["marks"] is marks
    assert ctx.results["05_T1"]["context"] is context
    # Keeps what the node already stored.
    ctx.results["06b_echo"] = {"fit": {"success": True}}
    common.store_figure_inputs(ctx, "06b_echo", marks, None)
    assert ctx.results["06b_echo"]["fit"] == {"success": True}
    assert ctx.results["06b_echo"]["context"] is None


# 8. shared seed-2026 graph fixture


def test_graph_run_2026_fixture(graph_run_2026):
    ctx = graph_run_2026
    assert ctx.save_outputs is False
    assert [r.name for r in ctx.history] == NODE_ORDER
    assert all(r.success for r in ctx.history)
    assert set(ctx.results) == set(NODE_ORDER) | set(SETUP_NODES)  # 01a ran first, outside the history
    assert ctx.reference is not None and {"fidelity", "T1", "f_01"} <= set(ctx.reference)
    assert ctx.graph_nodes == NODE_ORDER
