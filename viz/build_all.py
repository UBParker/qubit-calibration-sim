"""Rebuild every README media file with one command.

    uv run python -m viz.build_all [--seed 2026]

Steps, in order:

1. Run the bring-up graph exactly as ``calibrations/90_calibration_graph_bringup.py`` does when run
   as a script: rough state from ``state_initial.json``, ``SimulatedQPU(seed)``,
   ``ctx.reference = build_reference(state)``, ``run_graph``, then ``save_state``. The nodes write
   their figures (``<node>-<theme>.png``) and their QUA exports (``qua/programs``), and
   ``state.json`` is rewritten, so with the same seed these files are byte-identical to what the
   90 script writes. A rebuild matches a fresh clone's committed files byte for byte on the
   machine that rendered them (macOS on Apple silicon) with Python 3.11 and the locked
   dependencies; Python 3.12 there changes floats in the last digits, and other platforms are
   not checked.
2. The bring-up convergence hero, ``bringup-convergence-<theme>.png``.
3. The leakage figure, ``leakage-<theme>.png``.
4. The three Bloch sphere GIFs: ``bloch-rabi.gif``, ``bloch-ramsey.gif``, ``bloch-echo.gif``.

The HTML report (``docs/report/index.html``) is not built here yet.

Every media file goes to ``out_dir`` (``docs/media`` by default). The script prints each file
written with its size, then the total size of ``out_dir``, and exits with status 1 if that total
is over ``MEDIA_BUDGET_BYTES`` or any GIF is over 3 MB. If a node fails, it stops with status 1
before writing ``state.json`` or any figure after the graph.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import calibration_utils.common as common
from calibration_utils.common import INITIAL_STATE_PATH, MEDIA_DIR, NODE_ORDER, NodeContext, load_node, load_state, save_state
from calibration_utils.style import THEMES
from simulator.qpu import SimulatedQPU
from simulator.reference import build_reference
from viz.bloch import MAX_GIF_BYTES
from viz.convergence import save_convergence
from viz.dynamics_echo import make_echo_animation
from viz.dynamics_leakage import save_leakage
from viz.dynamics_rabi import make_rabi_animation
from viz.dynamics_ramsey import make_ramsey_animation

DEFAULT_SEED = 2026
MEDIA_BUDGET_BYTES = 15_000_000  # committed media must total under 15 MB
GIF_NAMES = ("bloch-rabi.gif", "bloch-ramsey.gif", "bloch-echo.gif")


class MediaOverBudget(RuntimeError):
    """The media directory is over ``MEDIA_BUDGET_BYTES`` in total, or a GIF is over ``MAX_GIF_BYTES``."""


class BringupFailed(RuntimeError):
    """A node's fit failed or the graph stopped before its last node."""


@contextmanager
def _node_media_dir(out_dir: Path) -> Iterator[None]:
    # Nodes save through common.save_figure, which reads common.MEDIA_DIR when it is called.
    saved = common.MEDIA_DIR
    common.MEDIA_DIR = Path(out_dir)
    try:
        yield
    finally:
        common.MEDIA_DIR = saved


def run_bringup(seed: int, out_dir: Path) -> tuple[NodeContext, list[Path]]:
    """Run the setup step and the graph as the 90 script's ``__main__`` does; return the context and node figures.

    Writes the node figures to ``out_dir``, the QUA exports to ``qua/programs`` and the calibrated
    state to ``state.json``. Raises ``BringupFailed`` without writing ``state.json`` if any node
    fails or the graph does not run every node.
    """
    graph = load_node("90_calibration_graph_bringup.py")
    ctx = NodeContext(state=load_state(INITIAL_STATE_PATH), qpu=SimulatedQPU(seed=seed))
    ctx.reference = build_reference(ctx.state)
    with _node_media_dir(out_dir):
        try:
            graph.run_setup(ctx)
        except RuntimeError as err:
            raise BringupFailed(f"{err}; state.json not written") from err
        outcomes = graph.run_graph(ctx)
    failed = [name for name, fit in outcomes.items() if not fit["success"]]
    if failed or len(ctx.history) != len(graph.NODES):
        detail = f"{', '.join(failed)} failed" if failed else "the graph stopped early"
        raise BringupFailed(f"{detail}: ran {len(ctx.history)} of {len(graph.NODES)} nodes; state.json not written")
    save_state(ctx.state)
    names = list(common.SETUP_NODES) + [record.name for record in ctx.history]
    figures = [Path(out_dir) / f"{name}-{theme}.png" for name in names for theme in THEMES]
    missing = [path.name for path in figures if not path.exists()]
    if missing:
        raise RuntimeError(f"nodes did not write {', '.join(missing)}")
    return ctx, figures


def write_hero(ctx: NodeContext, out_dir: Path) -> list[Path]:
    """Write ``bringup-convergence-<theme>.png`` from the graph run's history."""
    return save_convergence(ctx.history, ctx.reference, ctx.graph_nodes or list(NODE_ORDER), out_dir)


def write_leakage(out_dir: Path) -> list[Path]:
    """Write ``leakage-<theme>.png``."""
    return save_leakage(out_dir)


def write_gifs(out_dir: Path) -> list[Path]:
    """Write the Rabi, Ramsey and echo GIFs."""
    out_dir = Path(out_dir)
    return [
        make_rabi_animation(out_dir / GIF_NAMES[0]),
        make_ramsey_animation(out_dir / GIF_NAMES[1]),
        make_echo_animation(out_dir),
    ]


def directory_bytes(directory: Path) -> int:
    """Total size of the files under ``directory``."""
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


def _shown(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(common.REPO_ROOT))
    except ValueError:
        return str(path)


def size_table(paths: list[Path], out_dir: Path, total: int) -> str:
    """One line per file written with its size, then the total size of ``out_dir``."""
    rows = [(_shown(path), path.stat().st_size) for path in paths]
    total_label = f"{_shown(Path(out_dir))} total"
    width = max([len(name) for name, _ in rows] + [len(total_label)])
    lines = [f"{'file':<{width}}  {'bytes':>12}"]
    lines += [f"{name:<{width}}  {size:>12,}" for name, size in rows]
    lines.append(f"{total_label:<{width}}  {total:>12,}  ({total / 1e6:.2f} MB)")
    return "\n".join(lines)


def build_all(
    seed: int = DEFAULT_SEED, out_dir: Path = MEDIA_DIR, budget: int | None = None
) -> list[Path]:
    """Build every media file into ``out_dir``, print the size table and return the media paths.

    Raises ``MediaOverBudget`` (after printing the table) if ``out_dir`` totals more than
    ``budget`` bytes, which defaults to ``MEDIA_BUDGET_BYTES``, or if a GIF is over
    ``MAX_GIF_BYTES``. Raises ``BringupFailed`` before any other figure if the graph fails.
    """
    budget = MEDIA_BUDGET_BYTES if budget is None else budget
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    ctx, media = run_bringup(seed, out_dir)
    media += write_hero(ctx, out_dir)
    media += write_leakage(out_dir)
    media += write_gifs(out_dir)
    elapsed = time.perf_counter() - start

    total = directory_bytes(out_dir)
    print(size_table(media, out_dir, total))
    print(f"built in {elapsed:.1f} s")
    if total > budget:
        raise MediaOverBudget(f"{out_dir} totals {total:,} bytes, over the {budget:,} byte budget")
    big = [p.name for p in media if p.suffix == ".gif" and p.stat().st_size > MAX_GIF_BYTES]
    if big:
        raise MediaOverBudget(f"{', '.join(big)} over the {MAX_GIF_BYTES:,} byte GIF limit")
    return media


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m viz.build_all", description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="simulator seed (default 2026)")
    parser.add_argument("--out-dir", type=Path, default=MEDIA_DIR, help="media directory (default docs/media)")
    args = parser.parse_args(argv)
    try:
        build_all(seed=args.seed, out_dir=args.out_dir)
    except (MediaOverBudget, BringupFailed) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
