"""Population levels from the 07 blob means, node order and the figure context."""

import numpy as np
import pytest

from calibration_utils import common
from calibration_utils.common import NODE_ORDER, blob_levels, figure_context, iq_dataset, iq_projection, load_node, project_on_iq_axis
from calibration_utils.style import FigureContext
from simulator.device import TRUE_DEVICE

graph = load_node("90_calibration_graph_bringup.py")


def _flat_ramsey(ds):
    """06a projects its 2D (idle_time, detuning_signs) data as one flat sweep."""
    I = ds.I.sel(qubit="q1").values.ravel()
    Q = ds.Q.sel(qubit="q1").values.ravel()
    return iq_dataset(I, Q, "point", np.arange(I.size))


@pytest.mark.parametrize(
    ("filename", "sweep", "first_point_excited"),
    [("04b_power_rabi.py", "amp_prefactor", False), ("05_T1.py", "idle_time", True), ("06a_ramsey.py", "point", True)],
)
def test_project_on_iq_axis_unchanged(calibrated_state, make_ctx, filename, sweep, first_point_excited):
    node = load_node(filename)
    ctx = make_ctx(calibrated_state, 0)
    node.run(ctx)
    ds = ctx.results[node.NODE_NAME]["dataset"]
    if sweep == "point":
        ds = _flat_ramsey(ds)
    old = project_on_iq_axis(ds, sweep, first_point_excited=first_point_excited)
    signal, axis, centre = iq_projection(ds, sweep, first_point_excited=first_point_excited)
    assert np.array_equal(old.values, signal.values)
    assert old.dims == signal.dims
    assert abs(axis) == pytest.approx(1.0)
    # The signal is the projection of (z - centre) on the returned axis, sign included.
    z = (ds.I + 1j * ds.Q).sel(qubit="q1").values
    assert np.allclose(((z - centre) * np.conj(axis)).real, signal.values[0])


def test_iq_projection_peak_orientation(calibrated_state, make_ctx):
    node = load_node("03a_qubit_spectroscopy.py")
    ctx = make_ctx(calibrated_state, 0)
    node.run(ctx)
    ds = ctx.results[node.NODE_NAME]["dataset"]
    old = project_on_iq_axis(ds, "detuning", orientation="peak")
    signal, axis, centre = iq_projection(ds, "detuning", orientation="peak")
    assert np.array_equal(old.values, signal.values)
    z = (ds.I + 1j * ds.Q).sel(qubit="q1").values
    assert np.allclose(((z - centre) * np.conj(axis)).real, signal.values[0])


def test_iq_blobs_reports_means(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    fit = load_node("07_iq_blobs.py").run(ctx)
    for key in ("g_mean_iq", "e_mean_iq"):
        assert len(fit[key]) == 2
        assert all(isinstance(v, float) and np.isfinite(v) for v in fit[key])
    g, e = np.array(fit["g_mean_iq"]), np.array(fit["e_mean_iq"])
    assert np.hypot(*(e - g)) > 0


def test_blob_levels_projects_means():
    fit07 = {"g_mean_iq": [1.0, 2.0], "e_mean_iq": [3.0, 2.0]}
    level_g, level_e = blob_levels(fit07, axis=1j, centre=0.5 + 0.5j)
    assert (level_g, level_e) == pytest.approx((1.5, 1.5))
    level_g, level_e = blob_levels(fit07, axis=-1 + 0j, centre=2 + 0j)
    assert (level_g, level_e) == pytest.approx((1.0, -1.0))


def test_population_mapping_t1(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    fit07 = load_node("07_iq_blobs.py").run(ctx)
    t1 = load_node("05_T1.py")
    t1.run(ctx)
    ds = ctx.results[t1.NODE_NAME]["dataset"]
    signal, axis, centre = iq_projection(ds, "idle_time", first_point_excited=True)
    level_g, level_e = blob_levels(fit07, axis, centre)
    p_e = (signal.values[0] - level_g) / (level_e - level_g)
    # Guards the units of the 07 means against the averaged I/Q: a demodulation-unit slip would be
    # off by a large factor. The 07 blob means and the averaged data share one readout model
    # (SimulatedQPU.measure_shots and measure_averaged both let |e> decay in the window), so a fully
    # excited point maps to 1 up to the x180 error and noise.
    expected = 1.0
    assert abs(p_e[0] - expected) < 0.05
    assert -0.03 <= p_e[-1] <= 0.08


def test_figure_context_statuses(rough_state, make_ctx):
    ctx = make_ctx(rough_state, 0)
    graph.run_graph(ctx, ["02a_resonator_spectroscopy.py", "03a_qubit_spectroscopy.py"])
    record = ctx.history[-1]
    fc = figure_context(ctx, "03a_qubit_spectroscopy", record.before, record.fit)
    assert isinstance(fc, FigureContext)
    # A subset run's progress strip shows only the nodes that were asked to run.
    assert fc.node_order == ["02a_resonator_spectroscopy", "03a_qubit_spectroscopy"]
    assert fc.statuses["02a_resonator_spectroscopy"] == "ok"
    assert fc.statuses["03a_qubit_spectroscopy"] == "ok"
    # The graph did not stop, so later nodes are plain (absent), not "not run".
    assert "04b_power_rabi" not in fc.statuses
    assert fc.population_levels is None
    assert fc.before is record.before
    assert fc.after == ctx.state and fc.after is not ctx.state
    assert fc.reference is ctx.reference


def test_figure_context_marks_not_run_on_live_failure(rough_state, make_ctx):
    # Inside node.run the graph has not yet appended the current record, so the failed fit alone
    # must mark the later nodes "not run".
    ctx = make_ctx(rough_state, 0)
    ctx.graph_nodes = list(NODE_ORDER)
    ctx.history.append(common.NodeRecord("02a_resonator_spectroscopy", {}, {}, {"success": True}, None, True))
    fc = figure_context(ctx, "03a_qubit_spectroscopy", {}, {"success": False})
    assert fc.statuses["02a_resonator_spectroscopy"] == "ok"
    assert fc.statuses["03a_qubit_spectroscopy"] == "fail"
    assert all(fc.statuses[name] == "not run" for name in NODE_ORDER[2:])


def test_figure_context_standalone_shows_current_node_only(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    assert ctx.graph_nodes is None
    fc = figure_context(ctx, "05_T1", {}, {"success": True})
    assert fc.node_order == ["05_T1"]
    assert fc.statuses == {"05_T1": "ok"}
    fc = figure_context(ctx, "05_T1", {}, {"success": False})
    assert fc.node_order == ["05_T1"]
    assert fc.statuses == {"05_T1": "fail"}


def test_figure_context_population_levels_from_history(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    ctx.graph_nodes = list(NODE_ORDER)
    fit07 = {"success": True, "g_mean_iq": [1.0, 0.0], "e_mean_iq": [3.0, 0.0]}
    ctx.history.append(common.NodeRecord("07_iq_blobs", {}, {}, fit07, None, True))
    fc = figure_context(ctx, "05_T1", {}, {"success": True}, axis=1 + 0j, centre=2 + 0j)
    assert fc.population_levels == pytest.approx((-1.0, 1.0))
    assert fc.statuses["07_iq_blobs"] == "ok"
    # Without an axis there is nothing to project on.
    assert figure_context(ctx, "05_T1", {}, {"success": True}).population_levels is None
    # A failed 07 gives no levels.
    ctx.history[-1] = common.NodeRecord("07_iq_blobs", {}, {}, {**fit07, "success": False}, None, False)
    assert figure_context(ctx, "05_T1", {}, {"success": True}, axis=1 + 0j, centre=0j).population_levels is None


def test_figure_context_population_levels_from_results(calibrated_state, make_ctx):
    ctx = make_ctx(calibrated_state, 0)
    ctx.results["07_iq_blobs"] = {"fit": {"success": True, "g_mean_iq": [1.0, 0.0], "e_mean_iq": [3.0, 0.0]}}
    fc = figure_context(ctx, "06a_ramsey", {}, {"success": True}, axis=1 + 0j, centre=0j)
    assert fc.population_levels == pytest.approx((1.0, 3.0))
    # Nodes before 07 in graph order keep their signal axis.
    assert figure_context(ctx, "04b_power_rabi", {}, {"success": True}, axis=1 + 0j, centre=0j).population_levels is None


def test_node_order_matches_graph():
    assert [n.removesuffix(".py") for n in graph.NODES] == NODE_ORDER
    assert len(NODE_ORDER) == 8
