"""Smoke tests for viz/build_all.py with the graph and GIF steps stubbed out."""

import hashlib

import matplotlib

matplotlib.use("Agg")

import types

import pytest

import calibration_utils.common as common
from calibration_utils.common import NODE_ORDER, REPO_ROOT, STATE_PATH, NodeRecord
from calibration_utils.style import THEMES
from viz import build_all

GIF_BYTES = 1000
NODE_PNG_BYTES = 100

EXPECTED = (
    [f"{node}-{theme}.png" for node in NODE_ORDER for theme in THEMES]
    + [f"bringup-convergence-{theme}.png" for theme in THEMES]
    + [f"leakage-{theme}.png" for theme in THEMES]
    + ["bloch-rabi.gif", "bloch-ramsey.gif", "bloch-echo.gif"]
)


def _repo_snapshot() -> dict[str, str]:
    """Hashes of every file build_all could write in the repo outside an ``out_dir``."""
    paths = [REPO_ROOT / "state.json", *sorted((REPO_ROOT / "qua" / "programs").glob("*"))]
    media = REPO_ROOT / "docs" / "media"
    if media.exists():
        paths += sorted(media.rglob("*"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


@pytest.fixture
def stubbed(monkeypatch, graph_run_2026):
    """Swap the graph run and the GIFs for stubs that write small placeholder files."""

    def fake_bringup(seed, out_dir):
        paths = []
        for record in graph_run_2026.history:
            for theme in THEMES:
                path = out_dir / f"{record.name}-{theme}.png"
                path.write_bytes(b"\0" * NODE_PNG_BYTES)
                paths.append(path)
        return graph_run_2026, paths

    def fake_gifs(out_dir):
        paths = [out_dir / name for name in build_all.GIF_NAMES]
        for path in paths:
            path.write_bytes(b"\0" * GIF_BYTES)
        return paths

    monkeypatch.setattr(build_all, "run_bringup", fake_bringup)
    monkeypatch.setattr(build_all, "write_gifs", fake_gifs)


def test_build_all_writes_every_media_file(stubbed, tmp_path, capsys):
    before = _repo_snapshot()
    out_dir = tmp_path / "media"

    paths = build_all.build_all(out_dir=out_dir)

    assert [p.name for p in paths] == EXPECTED
    assert all(p.parent == out_dir for p in paths)
    assert sorted(p.name for p in out_dir.rglob("*")) == sorted(EXPECTED)
    assert _repo_snapshot() == before, "build_all wrote outside out_dir"

    printed = capsys.readouterr().out
    for name in EXPECTED:
        assert name in printed
    total = sum(p.stat().st_size for p in paths)
    assert f"{total:,}" in printed


@pytest.fixture
def cheap_figures(monkeypatch):
    """Swap the hero and leakage renders for placeholder files, for tests about the size check."""

    def fake(name):
        def write(*args):
            out_dir = args[-1]
            paths = [out_dir / f"{name}-{theme}.png" for theme in THEMES]
            for path in paths:
                path.write_bytes(b"\0" * NODE_PNG_BYTES)
            return paths

        return write

    monkeypatch.setattr(build_all, "write_hero", fake("bringup-convergence"))
    monkeypatch.setattr(build_all, "write_leakage", fake("leakage"))


def test_build_all_fails_over_budget(stubbed, cheap_figures, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_all, "MEDIA_BUDGET_BYTES", 1_000)
    out_dir = tmp_path / "media"

    with pytest.raises(build_all.MediaOverBudget):
        build_all.build_all(out_dir=out_dir)
    assert "total" in capsys.readouterr().out

    assert build_all.main(["--out-dir", str(out_dir)]) == 1


def test_main_exits_zero_under_budget(stubbed, cheap_figures, tmp_path):
    assert build_all.main(["--out-dir", str(tmp_path / "media")]) == 0


def test_node_media_dir_is_restored(tmp_path):
    saved = common.MEDIA_DIR
    with build_all._node_media_dir(tmp_path):
        assert common.MEDIA_DIR == tmp_path
    assert common.MEDIA_DIR == saved


def test_gif_over_limit_fails(stubbed, cheap_figures, tmp_path, monkeypatch):
    monkeypatch.setattr(build_all, "MAX_GIF_BYTES", GIF_BYTES - 1)
    with pytest.raises(build_all.MediaOverBudget, match="GIF limit"):
        build_all.build_all(out_dir=tmp_path / "media")
    assert build_all.main(["--out-dir", str(tmp_path / "media")]) == 1


def test_failed_bringup_exits_before_saving(tmp_path, monkeypatch, capsys):
    """A node failure stops the build with status 1 before state.json or the hero is written."""
    nodes = [f"{name}.py" for name in NODE_ORDER]

    def run_graph(ctx):
        fit = {"success": False}
        ctx.history.append(NodeRecord(NODE_ORDER[0], ctx.snapshot(), ctx.snapshot(), fit, None, False))
        return {NODE_ORDER[0]: fit}

    fake_graph = types.SimpleNamespace(NODES=nodes, run_setup=lambda ctx: {}, run_graph=run_graph)
    monkeypatch.setattr(build_all, "load_node", lambda filename: fake_graph)
    saved = []
    monkeypatch.setattr(build_all, "save_state", lambda state: saved.append(state))
    rendered = []
    monkeypatch.setattr(build_all, "write_hero", lambda *args: rendered.append("hero") or [])
    state_before = STATE_PATH.read_bytes()

    assert build_all.main(["--out-dir", str(tmp_path / "media")]) == 1

    assert saved == [] and rendered == []
    assert STATE_PATH.read_bytes() == state_before
    assert NODE_ORDER[0] in capsys.readouterr().err


def test_short_graph_run_fails(tmp_path, monkeypatch):
    """A graph that stops early without a failed fit still fails the build."""
    nodes = [f"{name}.py" for name in NODE_ORDER]
    fake_graph = types.SimpleNamespace(NODES=nodes, run_setup=lambda ctx: {}, run_graph=lambda ctx: {})
    monkeypatch.setattr(build_all, "load_node", lambda filename: fake_graph)
    monkeypatch.setattr(build_all, "save_state", lambda state: pytest.fail("state saved"))
    with pytest.raises(build_all.BringupFailed, match="0 of 8"):
        build_all.run_bringup(2026, tmp_path)


def test_failed_setup_step_fails_the_build(tmp_path, monkeypatch):
    """A failed 01a setup step stops the build before the graph runs or state.json is written."""

    def failed_setup(ctx):
        raise RuntimeError("setup step 01a_time_of_flight failed")

    fake_graph = types.SimpleNamespace(NODES=[], run_setup=failed_setup, run_graph=lambda ctx: pytest.fail("graph ran"))
    monkeypatch.setattr(build_all, "load_node", lambda filename: fake_graph)
    monkeypatch.setattr(build_all, "save_state", lambda state: pytest.fail("state saved"))
    with pytest.raises(build_all.BringupFailed, match="01a_time_of_flight failed"):
        build_all.run_bringup(2026, tmp_path)
