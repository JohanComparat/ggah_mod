#!/usr/bin/env python3
"""Execute the documentation notebooks in place, so their committed outputs are current.

Read the Docs renders the outputs stored in each ``.ipynb`` and executes nothing,
so what a reader sees is whatever this script last wrote.  Run it with the
interpreter whose results the pages should show -- ``make docs`` passes
``$(PY)``, the environment that also has CLASS, so the notebooks' guarded
``ACCURATE`` cells run rather than skip.

    python docs/tools/run_notebooks.py                   # every notebook
    python docs/tools/run_notebooks.py layer1_cosmology  # one, by stem
"""
from __future__ import annotations

import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
TIMEOUT_S = 1800


def run(path: pathlib.Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=TIMEOUT_S, kernel_name="python3",
                            record_timing=False,
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.perf_counter()
    client.execute()
    nbformat.write(nb, path)
    print(f"{path.name}: executed in {time.perf_counter() - t0:.0f} s")


def main(argv: list[str]) -> int:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if argv:
        paths = [p for p in paths if p.stem in argv]
    for path in paths:
        run(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
