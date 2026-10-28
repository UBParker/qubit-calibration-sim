"""Plumbing shared by every node: state I/O, datasets, QUA export, figures.

QUAlibrate gives each node a ``QualibrationNode`` with ``machine`` (QUAM state),
``parameters``, ``results`` and save logic. ``NodeContext`` is the minimal local stand-in.

Conventions every node follows:

- Sweep range (Ruling Y). Each node's fit dict carries ``"<sweep>_range": (min, max)`` for every
  swept quantity its pulse-sequence strip shows, in the sweep's native units (the units of the
  dataset coordinate: ns for times, Hz for frequencies, a prefactor for amplitudes). The key is
  ``sweep_range_key(<sweep coordinate name>)``, for example ``fit["idle_time_range"]``. The
  figure's ``sequence_blocks`` reads the range from the fit dict, never from ``Parameters`` or
  the dataset directly, so standalone and graph runs label the sweep the same way.
- Figure inputs. After building its ``Marks`` and ``figure_context``, a node stores both with
  ``store_figure_inputs(ctx, NODE_NAME, marks, context)`` so the report reads them the same way
  for every node.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from simulator.qpu import SimulatedQPU
from simulator.transmon import Pulse

if TYPE_CHECKING:
    from calibration_utils.style import FigureContext

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = REPO_ROOT / "state.json"  # latest calibrated state, rewritten by every run
INITIAL_STATE_PATH = REPO_ROOT / "state_initial.json"  # rough pre-bring-up guesses, never rewritten
MEDIA_DIR = REPO_ROOT / "docs" / "media"  # node figures, one PNG per theme
QUA_EXPORT_DIR = REPO_ROOT / "qua" / "programs"
QUBIT = "q1"

# The bring-up graph's nodes in run order (calibrations/90_calibration_graph_bringup.py runs these).
NODE_ORDER = [
    "02a_resonator_spectroscopy",
    "03a_qubit_spectroscopy",
    "04b_power_rabi",
    "08a_readout_frequency_optimization",
    "07_iq_blobs",
    "06a_ramsey",
    "05_T1",
    "06b_echo",
]
IQ_BLOBS_NODE = "07_iq_blobs"
# Setup steps run once before the graph (calibrations/90_calibration_graph_bringup.py run_setup), not
# graph nodes: QM's fixed-frequency graph assumes time of flight and ADC offsets are already set.
SETUP_NODES = ["01a_time_of_flight"]

# Ruling Y: fit[sweep + SWEEP_RANGE_SUFFIX] == (min, max) of the sweep in its native units.
SWEEP_RANGE_SUFFIX = "_range"


def sweep_range_key(sweep: str) -> str:
    """The fit-dict key holding a sweep's (min, max), for example ``"idle_time_range"``."""
    return f"{sweep}{SWEEP_RANGE_SUFFIX}"


def load_state(path: Path = STATE_PATH) -> dict:
    return json.loads(Path(path).read_text())


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    Path(path).write_text(json.dumps(state, indent=2) + "\n")


@dataclass
class NodeRecord:
    """One node's run in a graph: deep copies of the state before and after, its fit and data."""

    name: str
    before: dict
    after: dict
    fit: dict
    dataset: xr.Dataset | None
    success: bool


@dataclass
class NodeContext:
    """What a node gets when it runs: the device state, the (simulated) hardware, a place to put results."""

    state: dict
    qpu: SimulatedQPU
    save_outputs: bool = True
    results: dict = field(default_factory=dict)
    # True device values and simulator readout reference (simulator.reference.build_reference),
    # for figures only. Calibration code must not read it.
    reference: dict | None = None
    # One NodeRecord per node run by the graph, in order; for figures and the report.
    history: list[NodeRecord] = field(default_factory=list)
    # set by run_graph; None for a standalone node run
    graph_nodes: list[str] | None = None

    @property
    def qubit(self) -> dict:
        return self.state["qubits"][QUBIT]

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)


def iq_dataset(I, Q, sweep_name: str, sweep_values, attrs: dict | None = None) -> xr.Dataset:
    """Shape raw data like QM's fetched datasets: variables I and Q over (qubit, sweep)."""
    sweep_values = np.asarray(sweep_values)
    coords = {"qubit": [QUBIT], sweep_name: xr.DataArray(sweep_values, dims=sweep_name, attrs=attrs or {})}
    return xr.Dataset(
        {
            "I": (("qubit", sweep_name), np.asarray(I)[None, :]),
            "Q": (("qubit", sweep_name), np.asarray(Q)[None, :]),
        },
        coords=coords,
    )


def add_iq_derived(ds: xr.Dataset) -> xr.Dataset:
    """Add amplitude and phase, as QM's ``process_raw_dataset`` helpers do."""
    z = ds.I + 1j * ds.Q
    return ds.assign(IQ_abs=np.abs(z), phase=xr.apply_ufunc(np.angle, z))


def iq_projection(
    ds: xr.Dataset,
    sweep_name: str,
    first_point_excited: bool = False,
    orientation: str = "first_point",
    exclude: np.ndarray | None = None,
) -> tuple[xr.DataArray, complex, complex]:
    """Project averaged I/Q onto the line of largest variance, giving one real signal per point.

    Equivalent to rotating the IQ plane so the |g>-|e> axis lies along I. The sign is set so the
    signal grows with excited-state population. With ``orientation="first_point"`` the caller says
    whether the first sweep point is the most excited one (T1) or close to |g> (Rabi, Ramsey).
    With ``orientation="peak"`` the largest excursion from the median is made positive, for sweeps
    where only a few points sit on a narrow line (qubit spectroscopy) and the first point alone
    cannot fix the sign against noise. ``first_point_excited`` is ignored in that mode.

    Returns ``(signal, axis, centre)``: ``signal = Re((I + iQ - centre) * conj(axis))`` with the
    unit vector ``axis`` already carrying the sign, and ``centre`` the mean of the sweep's points.
    Any other I/Q point (such as a 07 blob mean) projects onto the same signal scale this way.

    ``exclude`` (bool per point) leaves masked outliers out of the centre, the axis and the sign
    rule (the first included point for "first_point", the largest included excursion for "peak");
    the signal is still returned for every point. None or all False runs the unmasked code.
    """
    if orientation not in ("first_point", "peak"):
        raise ValueError(f"unknown orientation {orientation!r}")
    z = (ds.I + 1j * ds.Q).sel(qubit=QUBIT).values
    if exclude is not None and np.any(exclude):
        return _masked_projection(ds, sweep_name, z, np.asarray(exclude, dtype=bool), first_point_excited, orientation)
    centre = complex(z.mean())
    centred = z - centre
    _, _, vh = np.linalg.svd(np.stack([centred.real, centred.imag]).T, full_matrices=False)
    axis = complex(vh[0, 0], vh[0, 1])
    signal = (centred * np.conj(axis)).real
    if orientation == "peak":
        excursion = signal - np.median(signal)
        flip = excursion[np.argmax(np.abs(excursion))] < 0
    else:
        flip = (signal[0] > signal.mean()) != first_point_excited
    if flip:
        signal, axis = -signal, -axis
    da = xr.DataArray(signal[None, :], dims=("qubit", sweep_name), coords={"qubit": [QUBIT], sweep_name: ds[sweep_name].values})
    return da, axis, centre


def _masked_projection(ds, sweep_name, z, exclude, first_point_excited, orientation):
    """:func:`iq_projection` with centre, axis and sign from the points not in ``exclude``."""
    kept = z[~exclude]
    centre = complex(kept.mean())
    centred = kept - centre
    _, _, vh = np.linalg.svd(np.stack([centred.real, centred.imag]).T, full_matrices=False)
    axis = complex(vh[0, 0], vh[0, 1])
    signal = ((z - centre) * np.conj(axis)).real
    kept_signal = signal[~exclude]
    if orientation == "peak":
        excursion = kept_signal - np.median(kept_signal)
        flip = excursion[np.argmax(np.abs(excursion))] < 0
    else:
        flip = (kept_signal[0] > kept_signal.mean()) != first_point_excited
    if flip:
        signal, axis = -signal, -axis
    da = xr.DataArray(signal[None, :], dims=("qubit", sweep_name), coords={"qubit": [QUBIT], sweep_name: ds[sweep_name].values})
    return da, axis, centre


def project_on_iq_axis(
    ds: xr.Dataset,
    sweep_name: str,
    first_point_excited: bool = False,
    orientation: str = "first_point",
    exclude: np.ndarray | None = None,
) -> xr.DataArray:
    """The signal from :func:`iq_projection`, for callers that do not need the axis."""
    return iq_projection(ds, sweep_name, first_point_excited, orientation, exclude)[0]


# ---------------------------------------------------------------- gates
# Thresholds of the realistic-noise gates (calibration_utils/robust.py, tests/test_realistic_noise.py).
# Measured on this simulator, each node standalone, seeds 0 to 9 clean and 0 to 5 with a channel.
# Off-line test (iq_line_outliers, k 6): clean max robust z 4.70 (05). Residual test (k 7): clean
# max 5.06 (04b). Spike test (isolated_spikes, k 9): clean max 4.6 over 02a and 03a, seeds 0 to 30.
# Masked fraction: 05 with 50 mV glitches at 3e-4 per shot 1 to 5 of 61 points, accepted within
# 5.8 %; 06b at 3e-4 and 1000 averages 15 to 25 of 81, rejected; 06b at 1e-4 5 to 9, accepted
# within 5.5 %; 06a at 3e-4 22 to 30 of 500 samples, accepted.
MAX_OUTLIER_FRACTION = 0.15  # masked points (06a: samples) above this reject the fit
# 05 with a TLS-limited 5.6 us T1 on the 200 us sweep: 1 or 2 points before 1/e, rejected.
MIN_POINTS_IN_DECAY = 4  # kept sweep points below the fitted 1/e time (05, 06a, 06b)
# 06a Durbin-Watson, minimum over signs: clean 1.52 to 1.79; 800 Hz/s linear drift 1.21 to 1.44
# over seeds 0 to 10 (1.21 at seed 10: accepted on the run mean, but close enough to the limit
# that other seeds are rejected, a clean rejection of a drift-broadened envelope); 10 kHz switch 1.33 to 1.53; 20 kHz switch 0.66 to 0.73 and 50 kHz
# 0.36 to 0.37 (rejected); sweep-outer order 0.36 to 0.53.
MIN_DURBIN_WATSON = 1.2  # 06a residual serial correlation, minimum over the two detuning signs
# 03a: clean z at most 3.7 (fraction at most 0.10); unresolved TLS (g 300 kHz) z 3.6 to 5.7,
# fraction at most 0.16; strong TLS z 7.7 to 25.1, fraction 0.30 to 0.97.
SECOND_PEAK_MIN_Z = 7.0  # 03a residual peak in units of its own noise, and ...
SECOND_PEAK_MIN_FRACTION = 0.2  # ... as a fraction of the fitted line amplitude (both needed to reject)
# 07: 50 mV glitches at 3e-4 per shot exclude at most 0.15 %; at 3e-2, 2.9 to 3.4 % (rejected).
MAX_SHOT_OUTLIER_FRACTION = 0.01  # 07 shots excluded per preparation
# 07: chi-square with 19 dof at 1e-6 is 63.7; a burst adds of order 70 extra errors to one block of
# 200 against about 6 expected. Measured: clean p at least 0.16; a burst 10 ms into the |e> block
# p 2e-71 to 2e-38.
MIN_STATIONARITY_PVALUE = 1e-6


MAX_REFITS = 2  # refits after residual outliers are masked


def failed_gates(fit: dict) -> list[str]:
    """Names of the gates a fit failed, in the order the fit checked them."""
    return [name for name, ok in fit.get("gates", {}).items() if not ok]


def fit_with_masking(
    ds: xr.Dataset,
    sweep_name: str,
    reproject: Callable[[np.ndarray], xr.DataArray],
    fit_once: Callable[[xr.DataArray], object],
    residual_of: Callable[[object, xr.DataArray], np.ndarray | None],
):
    """Fit ``ds.signal``, mask residual outliers and refit, at most MAX_REFITS times.

    ``ds.outlier`` holds the off-line mask from ``process_raw_dataset`` (the signal was already
    projected without those points). ``fit_once`` gets the kept points as a (qubit, sweep)
    DataArray: ``ds.signal`` itself when nothing is masked, so clean data reaches QM's fitter
    unchanged. ``residual_of(result, data)`` returns the residual of the kept points (None when
    the fit has no finite curve); new residual outliers are added to the mask and the I/Q data is
    re-projected with ``reproject(mask)`` before the refit. Returns ``(result, mask, signal)``.
    """
    from calibration_utils.robust import residual_outliers

    mask = np.asarray(ds.outlier.values, dtype=bool).copy() if "outlier" in ds else np.zeros(ds.sizes[sweep_name], dtype=bool)
    signal = ds.signal
    for attempt in range(MAX_REFITS + 1):
        data = signal if not mask.any() else signal.isel({sweep_name: np.flatnonzero(~mask)})
        result = fit_once(data)
        if attempt == MAX_REFITS:
            break
        residual = residual_of(result, data)
        if residual is None or not np.all(np.isfinite(residual)):
            break
        new = residual_outliers(residual)
        if not new.any():
            break
        mask[np.flatnonzero(~mask)[new]] = True
        signal = reproject(mask)
    return result, mask, signal


def decay_residual(fit, data: xr.DataArray) -> np.ndarray | None:
    """Residual of QM's decay_exp fit at the kept points of ``data`` (None without a finite curve)."""
    from qualibration_libs.analysis.models import decay_exp

    params = [float(fit.sel(qubit=QUBIT, fit_vals=k)) for k in ("a", "offset", "decay")]
    if not all(np.isfinite(params)):
        return None
    return data.sel(qubit=QUBIT).values - decay_exp(data[data.dims[-1]].values, *params)


def decay_fit_gates(T: float, T_err: float, times: np.ndarray, mask: np.ndarray, max_relative_error: float):
    """Gates of an exponential decay fit (05, 06b), in order, and the number of kept points below T.

    ``converged``: finite positive T. ``relative_error``: T_err / T below ``max_relative_error``.
    ``sampled``: at least MIN_POINTS_IN_DECAY kept sweep points below the fitted 1/e time.
    ``outliers``: masked fraction at most MAX_OUTLIER_FRACTION.
    """
    converged = bool(np.isfinite(T) and T > 0)
    n_sampled = int(np.sum(np.asarray(times)[~mask] < T)) if converged else 0
    gates = {
        "converged": converged,
        "relative_error": bool(converged and np.isfinite(T_err) and T_err / T < max_relative_error),
        "sampled": n_sampled >= MIN_POINTS_IN_DECAY,
        "outliers": bool(np.mean(mask) <= MAX_OUTLIER_FRACTION),
    }
    return gates, n_sampled


# Gates added for realistic bad data, and the figure text naming each when it is the first to fail.
NEW_GATES = ("outliers", "sampled", "residual_structure", "second_peak", "outlier_shots", "stationary")


def new_gate_reason(fit: dict) -> str | None:
    """Rejection text when the first failed gate is one of NEW_GATES, else None (the node's own text)."""
    failed = failed_gates(fit)
    if not failed or failed[0] not in NEW_GATES:
        return None
    name = failed[0]
    if name == "outliers":
        what = "samples" if fit.get("outlier_unit") == "samples" else "points"
        return f"outliers: {100 * fit['outlier_fraction']:.0f} % of {what} off the g-e line (limit {100 * MAX_OUTLIER_FRACTION:.0f} %)"
    if name == "sampled":
        return f"decay under-sampled: {fit['n_sampled']} points before 1/e (min {MIN_POINTS_IN_DECAY})"
    if name == "residual_structure":
        dw = min(fit["durbin_watson"].values())
        return f"residuals correlated: Durbin-Watson {dw:.2f} (min {MIN_DURBIN_WATSON})"
    if name == "second_peak":
        offset = abs(fit["second_peak_detuning"] - fit["detuning"]) / 1e6
        return f"second peak {offset:.2f} MHz from the line ({fit['second_peak_z']:.0f}σ)"
    if name == "outlier_shots":
        return f"outlier shots: {100 * fit['outlier_shot_fraction']:.1f} % excluded (limit {100 * MAX_SHOT_OUTLIER_FRACTION:.0f} %)"
    p = f"{fit['stationarity_pvalue']:.0e}".replace("-", "−")  # U+2212 in the exponent, as figure text does
    return f"shots not stationary (p = {p}): transient during the run?"


def outlier_mask_of(ds: xr.Dataset) -> np.ndarray | None:
    """The masked points of a processed dataset (``ds.outlier``, flattened), or None."""
    return np.asarray(ds.outlier.values, dtype=bool).ravel() if "outlier" in ds else None


def outlier_summary(mask: np.ndarray) -> dict:
    """The fit-dict keys every masking node reports."""
    mask = np.asarray(mask, dtype=bool)
    return {"n_outliers": int(mask.sum()), "outlier_fraction": float(mask.mean()), "outlier_mask": [bool(m) for m in mask]}


def blob_levels(fit07: dict, axis: complex, centre: complex) -> tuple[float, float]:
    """Project the 07 blob means onto a node's IQ axis: ``(level_g, level_e)`` on its signal scale.

    ``axis`` and ``centre`` come from :func:`iq_projection` on that node's data. Both are in the
    same units (volts) as the 07 means, so P(e) = (signal - level_g) / (level_e - level_g).
    """
    levels = []
    for key in ("g_mean_iq", "e_mean_iq"):
        i, q = fit07[key]
        levels.append(float(((complex(i, q) - centre) * np.conj(axis)).real))
    return levels[0], levels[1]


def figure_context(
    ctx: NodeContext,
    node_name: str,
    before: dict,
    fit: dict,
    axis: complex | None = None,
    centre: complex | None = None,
) -> FigureContext:
    """Collect what a node figure needs from the run: statuses, state change, reference, P(e) levels.

    A standalone node run (``ctx.graph_nodes`` is None) gets a progress strip of the current node
    alone. In a graph run, statuses come from the history records before ``node_name`` plus the
    current fit, and the nodes after it are "not run" when the current fit failed or the history
    holds a failed node (the graph stopped); otherwise they are left out (plain).

    ``population_levels`` are the measured 07 blob means projected on this node's IQ axis
    (``axis``, ``centre`` from :func:`iq_projection`), for nodes after 07 in graph order. They are
    None when no successful 07 ran in this ctx (or no axis is given); plots must then fall back to
    the signal axis. The 07 means and the averaged data of later nodes share one readout model, |e>
    decay during the readout window included, so a fully excited point maps to about P(e) = 1;
    P(e) axes must still not clip at 1 (noise). ``ctx.reference`` is passed through for plotting; no fit reads it.
    ``in_graph`` is True when ``ctx.graph_nodes`` is set (the node runs inside ``run_graph``).
    """
    from calibration_utils.style import FigureContext

    earlier = []
    for record in ctx.history:
        if record.name == node_name:
            break
        earlier.append(record)
    current = "ok" if fit["success"] else "fail"
    if ctx.graph_nodes is None:
        node_order = [node_name]
        statuses = {node_name: current}
    else:
        node_order = list(ctx.graph_nodes)
        statuses = {record.name: "ok" if record.success else "fail" for record in earlier}
        statuses[node_name] = current
        stopped = not fit["success"] or any(not record.success for record in ctx.history)
        if stopped and node_name in node_order:
            for name in node_order[node_order.index(node_name) + 1 :]:
                statuses.setdefault(name, "not run")

    levels = None
    after_blobs = node_name in NODE_ORDER and NODE_ORDER.index(node_name) > NODE_ORDER.index(IQ_BLOBS_NODE)
    if after_blobs and axis is not None and centre is not None:
        if ctx.history:
            blobs = [record.fit for record in earlier if record.name == IQ_BLOBS_NODE and record.success]
            fit07 = blobs[-1] if blobs else None
        else:
            stored = ctx.results.get(IQ_BLOBS_NODE, {}).get("fit")
            fit07 = stored if stored and stored.get("success") else None
        if fit07 is not None:
            levels = blob_levels(fit07, axis, centre)

    return FigureContext(
        node_order=node_order,
        statuses=statuses,
        before=before,
        after=ctx.snapshot(),
        reference=ctx.reference,
        population_levels=levels,
        in_graph=ctx.graph_nodes is not None,
    )


def store_figure_inputs(ctx: NodeContext, node_name: str, marks, context: FigureContext | None) -> None:
    """Keep a node's figure inputs for the report: ``ctx.results[node_name]["marks"]`` and ``["context"]``.

    Merges into whatever the node already stored under ``node_name`` (fit, dataset), so call it
    after ``ctx.results[node_name] = {...}``.
    """
    entry = ctx.results.setdefault(node_name, {})
    entry["marks"] = marks
    entry["context"] = context


# ---------------------------------------------------------------- pulses from state
def x180_pulse(state: dict, amplitude_scale: float = 1.0, phase: float = 0.0, drive_frequency: float | None = None) -> Pulse:
    op = state["qubits"][QUBIT]["xy"]["operations"]["x180"]
    return Pulse(op["amplitude"] * amplitude_scale, op["length"], op["sigma"], phase, drive_frequency)


def x90_pulse(state: dict, phase: float = 0.0, drive_frequency: float | None = None) -> Pulse:
    return x180_pulse(state, 0.5, phase, drive_frequency)


def readout_settings(state: dict, qubit: str = QUBIT):
    """The readout as the OPX would play and demodulate it: ``simulator.readout.ReadoutSettings`` read
    from the config generated from ``state`` (frequency, amplitude, length, time of flight, smearing,
    analog-input offsets and integration weights)."""
    from qua.config import build_config
    from simulator.readout import ReadoutSettings

    return ReadoutSettings.from_config(build_config(state, qubit))


# ---------------------------------------------------------------- outputs
def _sorted_config(value):
    """Return ``value`` with every mapping (dicts and protobuf map containers) as a key-sorted dict."""
    if isinstance(value, Mapping):
        return {k: _sorted_config(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        return [_sorted_config(v) for v in value]
    return value


def export_qua(node_name: str, program, state: dict) -> Path:
    """Write the QUA program, with its full config, as a standalone script (``qm.generate_qua_script``).

    The output is byte-stable across runs. ``generate_qua_script`` stamps a wall-clock time, which
    is dropped, and prints a ``loaded_config`` read back from protobuf maps whose iteration order
    depends on a per-process hash seed, so that block is reprinted with its keys sorted.
    """
    from qm import QuantumMachinesManager, generate_qua_script
    from qm.api.models.capabilities import offline_capabilities
    from qm.program import load_config
    from qm.program._dict_to_pb_converter import DictToQuaConfigConverter
    from qm.serialization.generate_qua_script import _print_config

    from qua.config import build_config

    # No QOP server here, so tell the SDK which capabilities to assume when serializing the config.
    QuantumMachinesManager.set_capabilities_offline()

    config = build_config(state)
    script = generate_qua_script(program, config)
    lines = script.split("\n")
    script = "\n".join(line for line in lines if not line.startswith("# Single QUA script generated at"))
    head, marker, _ = script.partition("\nloaded_config = ")
    if not marker:
        raise RuntimeError("generate_qua_script output has no loaded_config block")
    loaded = DictToQuaConfigConverter(capabilities=offline_capabilities).deconvert(load_config(config))
    script = f"{head}{marker}{_print_config(_sorted_config(loaded))}\n\n"

    QUA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = QUA_EXPORT_DIR / f"{node_name}.py"
    path.write_text(script)
    return path


def save_figure(make_fig: Callable[[str], Figure], node_name: str) -> list[Path]:
    """Render ``make_fig(theme)`` in each theme to ``docs/media/<node_name>-<theme>.png``."""
    from calibration_utils.style import save_themed

    return save_themed(make_fig, node_name, MEDIA_DIR)


def load_node(filename: str):
    """Import a node module by file name (they start with digits, like QM's)."""
    import importlib.util

    path = REPO_ROOT / "calibrations" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
