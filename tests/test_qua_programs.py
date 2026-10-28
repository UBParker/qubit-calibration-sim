"""Every node's QUA program must build, and the generated config must pass QM's schema.

tests/test_qua_single_source.py checks that each program and the node's simulator calls describe the same experiment.
"""

import json
from pathlib import Path

import pytest
from qm import QuantumMachinesManager, generate_qua_script
from qm.program._qua_config_schema import load_config

from calibration_utils.common import REPO_ROOT, load_node
from qua.config import build_config

NODE_FILES = sorted(p.name for p in (REPO_ROOT / "calibrations").glob("[0-8]*.py"))


@pytest.fixture(scope="module", autouse=True)
def offline_capabilities():
    # Without a QOP server, QM's SDK needs its capabilities set explicitly before validating configs.
    QuantumMachinesManager.set_capabilities_offline()


def test_config_passes_qm_schema(rough_state, calibrated_state):
    for state in (rough_state, calibrated_state):
        load_config(build_config(state))


@pytest.mark.parametrize("filename", NODE_FILES)
def test_qua_program_builds_and_exports(filename, calibrated_state):
    node = load_node(filename)
    prog = node.create_qua_program(calibrated_state, node.Parameters())
    script = generate_qua_script(prog, build_config(calibrated_state))
    assert "with program() as prog:" in script
    assert "measure('readout', 'resonator'" in script


def test_state_json_is_valid_json():
    json.loads(Path(REPO_ROOT / "state.json").read_text())


def _export_t1(state: dict) -> Path:
    from calibration_utils.common import export_qua

    node = load_node("05_T1.py")
    return export_qua(node.NODE_NAME, node.create_qua_program(state, node.Parameters()), state)


def test_qua_export_is_deterministic(tmp_path, monkeypatch, calibrated_state):
    import calibration_utils.common as common

    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(common, "QUA_EXPORT_DIR", first)
    a = _export_t1(calibrated_state).read_bytes()
    monkeypatch.setattr(common, "QUA_EXPORT_DIR", second)
    b = _export_t1(calibrated_state).read_bytes()
    assert a == b
    assert b"generated at" not in a


_EXPORT_SCRIPT = """
import sys
from pathlib import Path
import calibration_utils.common as common
from calibration_utils.common import INITIAL_STATE_PATH, export_qua, load_node, load_state
common.QUA_EXPORT_DIR = Path(sys.argv[1])
node = load_node("05_T1.py")
state = load_state(INITIAL_STATE_PATH)
export_qua(node.NODE_NAME, node.create_qua_program(state, node.Parameters()), state)
"""


def test_qua_export_is_stable_across_processes(tmp_path):
    # protobuf's upb maps hash with a per-process seed, so map order in the loaded config can
    # differ between interpreter runs even when it agrees within one.
    import subprocess
    import sys

    outputs = []
    for run in ("a", "b"):
        out_dir = tmp_path / run
        subprocess.run([sys.executable, "-c", _EXPORT_SCRIPT, str(out_dir)], cwd=REPO_ROOT, check=True, capture_output=True)
        outputs.append((out_dir / "05_T1.py").read_bytes())
    assert outputs[0] == outputs[1]


def test_t1_export_uses_pre_update_state(rough_state, make_ctx):
    # 05_T1 exports ctx.state before update_state writes T1, so the helper passes on it.
    from tests.conftest import assert_export_uses_pre_update_state

    node = load_node("05_T1.py")
    ctx = make_ctx(rough_state, 0)
    fit = assert_export_uses_pre_update_state(node, ctx)
    assert fit["success"]
    assert ctx.save_outputs is False
    assert ctx.qubit["T1"] == fit["T1"]  # the run still updated the state afterwards


def test_export_helper_catches_post_update_export(rough_state, make_ctx):
    import types

    from calibration_utils.common import export_qua
    from tests.conftest import assert_export_uses_pre_update_state

    t1 = load_node("05_T1.py")

    def run(ctx, params=None):
        ctx.qubit["T1"] = 12_345.0  # update first, then export: the regression the helper guards
        if ctx.save_outputs:
            module.export_qua(t1.NODE_NAME, t1.create_qua_program(ctx.state, params), ctx.state)
        return {"success": True}

    module = types.SimpleNamespace(
        NODE_NAME=t1.NODE_NAME,
        Parameters=t1.Parameters,
        create_qua_program=t1.create_qua_program,
        export_qua=export_qua,
        run=run,
    )
    with pytest.raises(AssertionError, match="differs from the state before"):
        assert_export_uses_pre_update_state(module, make_ctx(rough_state, 0))
