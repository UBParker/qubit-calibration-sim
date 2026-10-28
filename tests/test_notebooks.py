"""Execute the notebooks in notebooks/ and check the claims their prose makes.

Needs the ``notebook`` dependency group (``uv sync --group notebook``); skipped without it.
"""

from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TOF_NOTEBOOK = NOTEBOOK_DIR / "01a_time_of_flight_success_flag.ipynb"


@pytest.mark.slow
def test_time_of_flight_notebook_runs_and_no_pulse_reports_success():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(TOF_NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(TOF_NOTEBOOK.parent)}}
    )

    def evaluate(expression: str) -> str:
        cell = nbformat.v4.new_code_cell(f"print(repr({expression}))")
        nb.cells.append(cell)  # in memory only; the notebook file is never written
        client.execute_cell(cell, len(nb.cells) - 1)
        return "".join(o.get("text", "") for o in cell.outputs if o.output_type == "stream").strip()

    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            client.execute_cell(cell, index)  # raises CellExecutionError on a failing cell
        # Section 2's claim: QM's analysis reports success on a trace with no pulse in it.
        assert evaluate("NO_PULSE_SUCCESS") == "True"
        assert evaluate("synthetic_fits['pulse with transient']['success']") == "True"

    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors
