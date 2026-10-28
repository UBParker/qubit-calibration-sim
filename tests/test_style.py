"""Tests for the shared figure style (calibration_utils/style.py)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from calibration_utils import style  # noqa: E402
from calibration_utils.common import NODE_ORDER  # noqa: E402

ROLES = ("data", "fit", "guess", "truth", "ground", "excited", "leak")


def _luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of an sRGB hex colour."""
    channels = []
    for i in (1, 3, 5):
        c = int(hex_colour[i : i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("theme", style.THEMES)
def test_palette_roles_distinct(theme):
    p = style.palette(theme)
    values = [getattr(p, role).lower() for role in ROLES]
    assert len(set(values)) == len(ROLES)


@pytest.mark.parametrize("theme", style.THEMES)
def test_theme_text_contrast(theme):
    p = style.palette(theme)
    assert _contrast(p.fg, p.bg) >= 7
    assert _contrast(p.muted, p.bg) >= 4.5


def test_themes_are_light_and_dark():
    assert style.THEMES == ("light", "dark")
    with pytest.raises(ValueError):
        style.palette("sepia")


def _trivial(theme: str):
    fig, axes, res = style.fit_axes(theme)
    x = np.linspace(0, 1, 21)
    axes[0].plot(x, x**2, "o")
    axes[0].plot(x, x**2, "-", color=style.palette(theme).fit)
    style.mark_values(axes[0], style.Marks(previous=0.2, fit=0.5, true=float("nan")), theme)
    style.value_chip(axes[0], ["f = 1.0 GHz"], theme)
    style.plot_residuals(res[0], x, np.zeros_like(x), theme)
    style.title_block(fig, "04b_power_rabi", "pi amplitude", "0.1 V", theme)
    return fig


def test_save_themed_writes_both_and_is_deterministic(tmp_path):
    first = style.save_themed(_trivial, "trivial", tmp_path / "a")
    second = style.save_themed(_trivial, "trivial", tmp_path / "b")
    assert [p.name for p in first] == ["trivial-light.png", "trivial-dark.png"]
    for a, b in zip(first, second):
        assert a.exists() and b.exists()
        assert a.read_bytes() == b.read_bytes()
    assert first[0].read_bytes() != first[1].read_bytes()
    assert plt.get_fignums() == []


def test_inter_font_registered():
    assert (style.FONT_DIR / "Inter-Regular.ttf").exists()
    with style.theme_context("light"):
        path = font_manager.findfont(font_manager.FontProperties(family=plt.rcParams["font.family"]))
    assert "Inter" in font_manager.FontProperties(fname=path).get_name()


def test_theme_context_restores_rcparams():
    before = plt.rcParams["axes.facecolor"]
    with style.theme_context("dark"):
        assert plt.rcParams["axes.facecolor"].lower() == style.palette("dark").bg.lower()
        assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.facecolor"] == before


def test_mark_values_skips_missing_and_nonfinite():
    fig, ax = plt.subplots()
    style.mark_values(ax, style.Marks(previous=None, fit=1.0, true=float("nan")), "light")
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["fitted value"]
    plt.close(fig)


def test_fit_mark_reads_as_marker_not_curve():
    """The fitted-value line is thin and translucent with a triangle tick, unlike the fit curve."""
    with style.theme_context("light") as p:
        fig, ax = plt.subplots()
        style.mark_values(ax, style.Marks(fit=1.0), "light")
        fit_lines = [ln for ln in ax.lines if ln.get_color() == p.fit]
        vline = [ln for ln in fit_lines if ln.get_marker() in ("None", None, "")]
        tick = [ln for ln in fit_lines if ln.get_marker() == "v"]
        assert len(vline) == 1 and len(tick) == 1
        assert vline[0].get_linewidth() <= 1.1
        assert vline[0].get_alpha() is not None and vline[0].get_alpha() <= 0.7
        assert vline[0].get_linewidth() < plt.rcParams["lines.linewidth"]
        plt.close(fig)


def _legend_fig(theme: str):
    fig, axes, res = style.fit_axes(theme)
    x = np.linspace(0, 1, 30)
    axes[0].plot(x, x, "o", label="data")
    axes[0].plot(x, x, "-", label="fit")
    style.mark_values(axes[0], style.Marks(previous=0.3, fit=0.5, true=0.52), theme)
    style.place_legend(axes[0], theme)
    style.title_block(fig, "node", "finds", "1.0", theme)
    return fig


def test_place_legend_sits_above_axes_without_frame():
    with style.theme_context("light"):
        fig = _legend_fig("light")
        ax = fig.axes[0]
        leg = ax.get_legend()
        assert leg is not None and not leg.get_frame_on()
        renderer = fig.canvas.get_renderer()
        lb = leg.get_window_extent(renderer)
        ab = ax.get_window_extent(renderer)
        assert lb.y0 >= ab.y1  # entirely above the data area
        assert lb.y1 <= fig.bbox.y1  # still on the canvas
        texts = [t for t in fig.texts if t.get_text() == "finds"]
        assert lb.y1 < texts[0].get_window_extent(renderer).y0  # under the title block
        plt.close(fig)


def _ghz_fig(theme: str, use_offset: bool):
    fig, axes, _ = style.fit_axes(theme, residuals=False)
    x = np.linspace(0, 1, 30)
    y = 5.1234e9 + 3e3 * np.sin(6 * x)  # narrow scan: 7-digit ticks clip a fixed margin
    axes[0].plot(x, y, "o")
    axes[0].set_ylabel("qubit frequency (Hz)")
    if use_offset:
        style.offset_formatter(axes[0], axis="y", unit_scale=1e3, unit="kHz", name="f")
    _ghz_fig.last = fig
    return fig


@pytest.mark.parametrize("use_offset", [False, True])
def test_large_y_values_keep_labels_on_canvas(tmp_path, use_offset):
    style.save_themed(lambda t: _ghz_fig(t, use_offset), "ghz", tmp_path)
    fig = _ghz_fig.last  # the dark render, closed by save_themed; reattach a canvas to measure
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # lay the labels out again at this canvas's dpi before measuring
    renderer = canvas.get_renderer()
    ax = fig.axes[0]
    assert ax.yaxis.label.get_window_extent(renderer).x0 >= 0
    for label in ax.get_yticklabels():
        if label.get_text():
            assert label.get_window_extent(renderer).x0 >= 0


def test_offset_formatter_shows_short_ticks_and_offset_label():
    with style.theme_context("light"):
        fig = _ghz_fig("light", use_offset=True)
        fig.canvas.draw()
        ax = fig.axes[0]
        assert ax.get_ylabel() == "f \u2212 5.1234 GHz (kHz)"
        ticks = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert ticks and all(len(t.replace("\u2212", "-").lstrip("-")) <= 5 for t in ticks)
        plt.close(fig)


def test_fit_axes_custom_figsize_keeps_gaps():
    fig, axes, res = style.fit_axes("light", ncols=2, figsize=(12.0, 5.0))
    dpi = fig.dpi
    a0, a1 = (ax.get_window_extent() for ax in axes)
    assert abs((a1.x0 - a0.x1) / dpi - style.GAP_W) < 0.01
    r0 = res[0].get_window_extent()
    assert abs((a0.y0 - r0.y1) / dpi - style.GAP_H) < 0.01
    plt.close(fig)


def test_fit_axes_shapes():
    fig, axes, res = style.fit_axes("light", ncols=2)
    assert len(axes) == 2 and len(res) == 2 and all(r is not None for r in res)
    plt.close(fig)
    fig, axes, res = style.fit_axes("light", ncols=3, residuals=False)
    assert len(axes) == 3 and res == [None, None, None]
    plt.close(fig)


# Layout C helpers


def _texts(fig) -> list[str]:
    found = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        found += [t.get_text() for t in ax.texts]
    return found


def test_node_layout_axes():
    with style.theme_context("light"):
        lay = style.node_layout("light", ncols=2)
        assert lay.progress_ax is not None and lay.sequence_ax is not None
        assert lay.cards_ax is not None
        assert len(lay.main_axes) == 2 and len(lay.residual_axes) == 2
        assert all(r is not None for r in lay.residual_axes)
        shared = lay.main_axes[0].get_shared_x_axes()
        assert shared.joined(lay.residual_axes[0], lay.main_axes[0])
        plt.close(lay.fig)
        lay = style.node_layout("light", ncols=2, residuals=False)
        assert lay.residual_axes == [None, None]
        plt.close(lay.fig)
        lay = style.node_layout("light", cards=False)
        assert lay.cards_ax is None and len(lay.main_axes) == 1
        plt.close(lay.fig)


def _sequence():
    return [
        style.SeqBlock("drive", "x90"),
        style.SeqBlock("idle", "τ", swept="0 to 50 us", width=2.0),
        style.SeqBlock("drive", "x90"),
        style.SeqBlock("readout", "readout", width=1.5, line="resonator"),
    ]


def test_pulse_sequence_labels():
    with style.theme_context("light"):
        fig, ax = plt.subplots()
        style.pulse_sequence(ax, _sequence(), "light")
        texts = _texts(fig)
        for want in ("x90", "0 to 50 us", "readout", "not to scale"):
            assert want in texts
        assert not ax.axison
        plt.close(fig)
        fig, ax = plt.subplots()
        style.pulse_sequence(ax, _sequence(), "light", to_scale=True)
        assert "not to scale" not in _texts(fig)
        plt.close(fig)


NODES = list(NODE_ORDER)  # the strip node figures and the convergence hero draw: the eight graph nodes


@pytest.mark.parametrize("theme", style.THEMES)
def test_progress_strip_statuses(theme):
    from matplotlib.colors import to_hex

    with style.theme_context(theme) as p:
        fig, ax = plt.subplots(figsize=(11, 0.5))
        n0, n1, n2 = NODES[:3]
        statuses = {n0: "ok", n1: "fail", n2: "not run"}
        style.progress_strip(ax, NODES, n1, statuses, theme)
        texts = _texts(fig)
        assert all(style.NODE_SHORT[n] in texts for n in NODES)
        assert "not run" in texts
        tagged = {pt.get_gid(): pt for pt in ax.patches if pt.get_gid()}
        assert all(n in tagged for n in NODES)  # one status dot per node
        assert to_hex(tagged[n1].get_facecolor()) == p.fail.lower()
        assert to_hex(tagged[n0].get_facecolor()) == p.ok.lower()
        current = tagged[f"{n1}:pill"]
        assert to_hex(current.get_edgecolor()) == p.fg.lower()
        assert current.get_linewidth() > tagged[f"{n0}:pill"].get_linewidth()
        plt.close(fig)


@pytest.mark.parametrize("width", [8.1, 11.0])
def test_progress_strip_pills_stay_inside(width):
    """Pills keep the same inset at both ends, whether labels shrink or the pills stretch.

    Eight labels need about 7.8 in at the smallest text size; narrower strips still overflow.
    """
    with style.theme_context("light"):
        fig = plt.figure(figsize=(width + 0.4, 0.6))
        ax = fig.add_axes((0.2 / (width + 0.4), 0.1, width / (width + 0.4), 0.8))
        style.progress_strip(ax, NODES, None, {n: "ok" for n in NODES}, "light")
        pills = [pt for pt in ax.patches if (pt.get_gid() or "").endswith(":pill")]
        assert len(pills) == len(NODES)
        w = ax.get_xlim()[1]
        assert min(pt.get_x() for pt in pills) >= 0
        assert max(pt.get_x() + pt.get_width() for pt in pills) <= w + 1e-9
        plt.close(fig)


def test_progress_strip_custom_labels():
    with style.theme_context("light"):
        fig, ax = plt.subplots(figsize=(6, 0.5))
        style.progress_strip(ax, ["a_x", "b_y"], None, {}, "light", labels={"a_x": "A"})
        texts = _texts(fig)
        assert "A" in texts and "b_y" in texts
        plt.close(fig)


@pytest.mark.parametrize("theme", style.THEMES)
def test_cards_column_stays_inside(theme):
    with style.theme_context(theme):
        lay = style.node_layout(theme)
        cards = [
            style.Card("measured", ("T1 = 47.7 +/- 1.2 us", "fit: A exp(-t/T1) + B", "chi2 1.03"), "ok"),
            style.Card("true value", ("T1 = 48.0 us", "difference -0.25 sigma", "within 1 sigma"), "info"),
            style.Card("state change", ("T1: uncalculated -> 47.7 us", "f_01 unchanged", "x"), None),
            style.Card("caveat", ("Markovian dephasing only", "exponential envelope", "by design"), "caveat"),
        ]
        style.cards_column(lay.cards_ax, cards, theme)
        lay.fig.canvas.draw()
        renderer = lay.fig.canvas.get_renderer()
        box = lay.cards_ax.get_window_extent(renderer)
        assert lay.cards_ax.texts
        for t in lay.cards_ax.texts:
            e = t.get_window_extent(renderer)
            inside = box.x0 <= e.x0 and e.x1 <= box.x1 and box.y0 <= e.y0 and e.y1 <= box.y1
            assert inside, t.get_text()
        assert not lay.cards_ax.axison
        plt.close(lay.fig)


def test_format_change_uncalculated():
    got = style.format_change("T1", None, 47700.0, 1e-3, "us", "{:.1f}")
    # A no-break space keeps the unit on the same line as its value.
    assert got == "T1: uncalculated -> 47.7\u00a0us"
    assert style.format_change("x", 1.0, 2.0, 1.0, "V", "{:.2f}") == "x: 1.00 -> 2.00\u00a0V"


def test_population_axis_mapping():
    with style.theme_context("light"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [-3.0, 5.0])
        ax.set_ylim(-3.0, 5.0)
        sec = style.population_axis(ax, -3.0, 5.0, "light")
        fig.canvas.draw()
        lo, hi = sec.get_ylim()
        assert abs(lo) < 1e-9 and abs(hi - 1.0) < 1e-9
        assert sec.get_ylabel() == "P(e), from 07 blob means"
        plt.close(fig)


def _ramsey_layout(theme: str):
    lay = style.node_layout(theme, ncols=2)
    x = np.linspace(0, 50, 30)
    for ax in lay.main_axes:
        ax.plot(x, np.cos(x), "o", label="data")
        ax.set_ylabel("rotated I (mV)")
    style.population_axis(lay.main_axes[1], -1.0, 1.0, theme)
    return lay


def test_population_label_clears_cards():
    with style.theme_context("light"):
        lay = _ramsey_layout("light")
        style.fit_layout(lay.fig)
        lay.fig.canvas.draw()
        renderer = lay.fig.canvas.get_renderer()
        sec = lay.main_axes[1].child_axes[0]
        label_right = sec.get_tightbbox(renderer).x1
        cards_left = lay.cards_ax.get_window_extent(renderer).x0
        assert (cards_left - label_right) / lay.fig.dpi >= 0.35 - 1e-6
        plt.close(lay.fig)


def test_fit_layout_makes_room_for_secondary_axis():
    with style.theme_context("light"):
        fig, axes, _ = style.fit_axes("light", residuals=False)
        axes[0].plot([0, 1], [0, 1])
        style.population_axis(axes[0], 0.0, 1.0, "light")
        width = fig.get_size_inches()[0]
        style.fit_layout(fig)
        assert fig.get_size_inches()[0] > width
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        right = axes[0].child_axes[0].get_tightbbox(renderer).x1
        assert right <= fig.bbox.x1 - 0.05 * fig.dpi
        plt.close(fig)


def test_population_axis_follows_set_ylim():
    with style.theme_context("light"):
        fig, ax = plt.subplots()
        sec = style.population_axis(ax, 2.0, 4.0, "light")
        ax.set_ylim(1.0, 5.0)
        fig.canvas.draw()
        lo, hi = sec.get_ylim()
        assert abs(lo + 0.5) < 1e-9 and abs(hi - 1.5) < 1e-9
        ticks = [t for t in sec.get_yticks() if lo <= t <= hi]
        assert ticks == [0.0, 0.25, 0.5, 0.75, 1.0]
        plt.close(fig)


def test_place_legend_title_does_not_overlap():
    with style.theme_context("light"):
        lay = _ramsey_layout("light")
        ax = lay.main_axes[0]
        leg = style.place_legend(ax, "light", title="detuning +300 kHz")
        lay.fig.canvas.draw()
        renderer = lay.fig.canvas.get_renderer()
        title = next(t for t in ax.texts if t.get_text() == "detuning +300 kHz")
        tb = title.get_window_extent(renderer)
        lb = leg.get_window_extent(renderer)
        assert tb.x1 < lb.x0
        ab = ax.get_window_extent(renderer)
        assert tb.y0 >= ab.y1  # in the legend row, above the data
        plt.close(lay.fig)


def test_title_block_kicker_uses_short_label():
    with style.theme_context("light"):
        fig, _, _ = style.fit_axes("light")
        style.title_block(fig, "06a_ramsey", "finds", "", "light")
        assert "06a Ramsey" in [t.get_text() for t in fig.texts]
        plt.close(fig)
