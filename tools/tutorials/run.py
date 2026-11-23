#!/usr/bin/env python3
"""Execute the Paper I tutorials in place, so their stored outputs are current.

    python tools/tutorials/run.py                    # every notebook, FAST grids
    python tools/tutorials/run.py 01 07              # the notebooks whose name starts so
    GGAH_TUTORIALS_FAST=0 python tools/tutorials/run.py   # the paper's grids

Notebooks 07-11, A and B read the data bundle: set ``GGAH_TUTORIALS_DATA`` to a
local copy, or let ``tutorials/_helpers.py`` download it.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

TUTORIALS = pathlib.Path(__file__).resolve().parents[2] / "tutorials"
TIMEOUT_S = 3600


def run(path: pathlib.Path) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=TIMEOUT_S, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.perf_counter()
    client.execute()
    nbformat.write(nb, path)
    print(f"{path.name}: executed in {time.perf_counter() - t0:.0f} s", flush=True)


def main(argv: list[str]) -> int:
    os.environ.setdefault("GGAH_TUTORIALS_FAST", "1")
    paths = sorted(TUTORIALS.glob("*.ipynb"))
    if argv:
        paths = [p for p in paths if any(p.stem.startswith(a) for a in argv)]
    for path in paths:
        run(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
