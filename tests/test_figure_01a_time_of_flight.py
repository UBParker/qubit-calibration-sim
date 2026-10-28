"""Layout C figure for the 01a setup step: both themes, the true edge only in a setup run with a
reference, the cards, and a failed fit."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from calibration_utils import style
from calibration_utils import time_of_flight as analysis
from calibration_utils.common import load_node
from simulator.device import TRUE_DEVICE

node = load_node("01a_time_of_flight.py")
NODE = node.NODE_NAME


def _texts(fig) -> list[str]:
    fig.canvas.draw()
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]


@pytest.fixture
def setup_run(graph_run_2026):
    entry = graph_run_2026.results[NODE]
    assert entry["fit"]["success"] and entry["context"].in_graph
    return entry["dataset"], entry["fit"], entry["marks"], entry["context"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_01a_figure_both_themes(setup_run, theme):
    ds, fit, marks, context = setup_run
    fig = analysis.plot_raw_data_with_fit(ds, fit, marks, theme, context)
    texts = " ".join(_texts(fig))
    plt.close(fig)
    assert "edge" in texts and "threshold" in texts and "true value" in texts
    assert f"time of flight {TRUE_DEVICE.time_of_flight:.0f}{style.NBSP}ns" in texts.replace(" ns", f"{style.NBSP}ns")
    assert "ADC RANGE" in texts.upper()


def test_01a_true_edge_is_true_tof_minus_window_start(setup_run):
    ds, fit, marks, _ = setup_run
    assert marks.true == TRUE_DEVICE.time_of_flight - ds.attrs["time_of_flight"]
    assert abs(marks.fit - marks.true) <= 3


def test_01a_standalone_has_no_true_value(initial_state, make_ctx):
    ctx = make_ctx(initial_state, 3)
    node.run(ctx)
    assert ctx.results[NODE]["marks"].true is None


def test_01a_failed_fit_renders(setup_run):
    ds, fit, _, context = setup_run
    failed = dict(fit, success=False)
    fig = analysis.plot_raw_data_with_fit(ds, failed, None, "light", context)
    plt.close(fig)
