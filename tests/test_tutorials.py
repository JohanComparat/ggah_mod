"""The Paper I tutorials in ``tutorials/``.

* **fast**: every notebook is stored executed, and every file the notebooks
  read is in the data registry.
* **slow**: notebooks 00-06, which need no data bundle, execute on reduced
  grids (``GGAH_TUTORIALS_FAST=1``), on a copy: the stored outputs are
  rewritten by ``make tutorials`` and never by the suite.  Each ends on a cell
  that checks its numbers against the paper.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

TUTORIALS = pathlib.Path(__file__).resolve().parents[1] / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.glob("*.ipynb"))
NO_DATA = [p for p in NOTEBOOKS if re.match(r"0[0-6]_", p.name)]


def test_there_is_a_notebook_per_section():
    assert [p.stem[:2] for p in NOTEBOOKS] == [f"{i:02d}" for i in range(12)] + ["A_", "B_"]


def test_every_tutorial_is_stored_executed():
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert all(c.get("execution_count") for c in code), path.name


def test_every_data_file_read_is_registered():
    registry = {line.split()[1] for line in (TUTORIALS / "data_registry.txt").read_text().splitlines() if line}
    names = set()
    for path in [*NOTEBOOKS, TUTORIALS / "_forecast.py", TUTORIALS / "_helpers.py"]:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "code")
        names |= set(re.findall(r'"((?:desi|halo_req|suite|emulators|comparisons)/[^"]+)"', text))
    assert names <= registry, names - registry


@pytest.mark.slow
@pytest.mark.parametrize("path", NO_DATA, ids=[p.stem for p in NO_DATA])
def test_tutorial_executes(path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("GGAH_TUTORIALS_FAST", "1")
    nb = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
    client = nbclient.NotebookClient(nb, timeout=3600, kernel_name="python3",
                                     resources={"metadata": {"path": str(path.parent)}})
    client.execute()
