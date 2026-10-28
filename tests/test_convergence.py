"""Tests for the bring-up convergence hero figure (viz/convergence.py)."""

import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from calibration_utils.common import INITIAL_STATE_PATH, load_node, load_state
from calibration_utils.style import THEMES
from simulator.device import TRUE_DEVICE
from viz.convergence import PARAMETERS, convergence_rows, plot_convergence, save_convergence

graph = load_node("90_calibration_graph_bringup.py")
NODE_ORDER = [name.removesuffix(".py") for name in graph.NODES]


@pytest.fixture
def run(graph_run_2026):
    """History and reference of the shared seed-2026 graph run, deep-copied so tests may mutate them."""
    return copy.deepcopy((graph_run_2026.history, graph_run_2026.reference))


def _texts(ax) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _strip_and_panels(fig):
    strips = [ax for ax in fig.axes if ax.get_label() == "progress strip"]
    panels = [ax for ax in fig.axes if ax.get_label().startswith("panel:")]
    return strips, panels


def test_convergence_full_history(run):
    history, reference = run
    for theme in THEMES:
        fig = plot_convergence(history, reference, theme, NODE_ORDER)
        strips, panels = _strip_and_panels(fig)
        assert len(strips) == 1
        assert len(panels) == 7
        assert [ax.get_label() for ax in panels] == [f"panel:{p.key}" for p in PARAMETERS]
        strip_texts = _texts(strips[0])
        assert strip_texts.count("passed") == 8
        assert "not run" not in strip_texts
        for ax in panels:
            assert not any("not measured" in t for t in _texts(ax))
            assert any("within" in t or "inside" in t for t in _texts(ax)), ax.get_label()
        fidelity = history[-1].after["qubits"]["q1"]["resonator"]["readout_fidelity"]
        fidelity_texts = _texts(panels[3])
        assert f"F {fidelity:.4f}, inside the 0.95 to 0.995 window" in fidelity_texts
        assert {"F 0.995", "F 0.95"} <= set(fidelity_texts)
        plt.close(fig)


def test_convergence_labels_stay_inside(run):
    """The window edge labels sit inside the fidelity panel, and no panel text runs off the page."""
    history, reference = run
    fig = plot_convergence(history, reference, "light", NODE_ORDER)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        page = fig.bbox
        _, panels = _strip_and_panels(fig)
        for ax in panels:
            for text in ax.texts:
                box = text.get_window_extent(renderer)
                assert page.x0 <= box.x0 and box.x1 <= page.x1, (ax.get_label(), text.get_text())
        fidelity = panels[3]
        inner = fidelity.get_window_extent(renderer)
        edges = [t for t in fidelity.texts if t.get_text() in ("F 0.995", "F 0.95")]
        assert len(edges) == 2
        for text in edges:
            box = text.get_window_extent(renderer)
            assert inner.y0 <= box.y0 and box.y1 <= inner.y1, text.get_text()
    finally:
        plt.close(fig)


def test_convergence_value_labels_avoid_zero_line(run):
    """No fitted-value or rough-guess label is crossed by the dotted zero line (seed 2026 put
    "-18.88" on the f_01 row's zero line and "-0.50 κ" on the readout row's)."""
    history, reference = run
    for theme in THEMES:
        fig = plot_convergence(history, reference, theme, NODE_ORDER)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            _, panels = _strip_and_panels(fig)
            checked = 0
            for ax in panels:
                zero_y = ax.transData.transform((0.0, 0.0))[1]
                for text in ax.texts:
                    if not str(text.get_gid() or "").startswith(("value:", "guess:")):
                        continue
                    box = text.get_window_extent(renderer)
                    assert not (box.y0 < zero_y < box.y1), (theme, ax.get_label(), text.get_text())
                    checked += 1
            assert checked >= 10
        finally:
            plt.close(fig)


def test_convergence_rows_are_computed_from_history(run):
    history, reference = run
    rows = {row.key: row for row in convergence_rows(history, reference, NODE_ORDER)}
    q = history[-1].after["qubits"]["q1"]
    assert rows["f_01"].final == pytest.approx((q["f_01"] - reference["f_01"]) / 1e3)
    assert rows["readout.frequency"].final == pytest.approx(
        (q["resonator"]["f_01"] - reference["readout.frequency"]) / TRUE_DEVICE.kappa
    )
    assert rows["x180.amplitude"].final == pytest.approx(
        100 * (q["xy"]["operations"]["x180"]["amplitude"] / reference["x180.amplitude"] - 1)
    )
    assert rows["fidelity"].final == pytest.approx(
        100 * (q["resonator"]["readout_fidelity"] - reference["fidelity"])
    )
    assert rows["T1"].final == pytest.approx(100 * (q["T1"] / reference["T1"] - 1))
    # Parameters enter at the first node that sets them.
    first = {key: NODE_ORDER[row.first_set] for key, row in rows.items()}
    assert first == {
        "readout.frequency": "02a_resonator_spectroscopy",
        "f_01": "03a_qubit_spectroscopy",
        "x180.amplitude": "04b_power_rabi",
        "fidelity": "07_iq_blobs",
        "T2ramsey": "06a_ramsey",
        "T1": "05_T1",
        "T2echo": "06b_echo",
    }
    # The Ramsey node updates f_01 a second time.
    assert rows["f_01"].changed == [1, 5]
    # 02a moves the readout to the |g> dip, 08a to the midpoint.
    assert rows["readout.frequency"].changed == [0, 3]
    # Rough guesses come from the state before the first node; T1 has none.
    initial_f01 = load_state(INITIAL_STATE_PATH)["qubits"]["q1"]["f_01"]
    assert rows["f_01"].guess == pytest.approx((initial_f01 - reference["f_01"]) / 1e3)
    assert rows["T1"].guess is None
    assert all(row.within for row in rows.values())


def test_convergence_handles_partial_history(run):
    history, reference = run
    cut = [record.name for record in history].index("03a_qubit_spectroscopy") + 1
    partial = history[:cut]
    fig = plot_convergence(partial, reference, "light", NODE_ORDER)
    strips, panels = _strip_and_panels(fig)
    assert len(strips) == 1 and len(panels) == 7
    strip_texts = _texts(strips[0])
    assert strip_texts.count("passed") == 2
    assert strip_texts.count("not run") == 6
    measured = {"panel:readout.frequency", "panel:f_01"}
    for ax in panels:
        count = sum("not measured" in t for t in _texts(ax))
        assert count == (0 if ax.get_label() in measured else 1), ax.get_label()
    # f_01 is set again by 06a, which did not run, so its readout says so.
    f01 = next(ax for ax in panels if ax.get_label() == "panel:f_01")
    assert any("06a Ramsey not run" in t for t in _texts(f01))
    plt.close(fig)


@pytest.mark.slow  # renders both PNGs only to check they are written
def test_convergence_saves_both_themes(run, tmp_path):
    history, reference = run
    paths = save_convergence(history, reference, NODE_ORDER, tmp_path)
    assert [p.name for p in paths] == ["bringup-convergence-light.png", "bringup-convergence-dark.png"]
    assert all(p.stat().st_size > 0 for p in paths)
