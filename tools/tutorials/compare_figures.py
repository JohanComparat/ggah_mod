"""Compare the figures the tutorials save with those of Paper I, pixel by pixel.

Run the notebooks with ``GGAH_TUTORIALS_FAST=0`` first (``make tutorials FAST=0``);
each saves its paper figures in ``tutorials/figures/``.  Both PDFs are rendered
at 100 dpi with ``pdftoppm``; a figure is SAME when no pixel differs.

    python tools/tutorials/compare_figures.py PAPER_REPO [OUT_DIR]
"""
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

NB = pathlib.Path(__file__).resolve().parents[2] / "tutorials" / "figures"
#: Figures the paper takes from another repository, relative to the paper's parent.
EXTERNAL = {"fig_suite_errors": "cog-prop-wp1-feasibilty/docs/figures/fig_suite_errors.pdf"}


def render(pdf, stem):
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", str(pdf), str(stem)], check=True)
    return np.asarray(Image.open(f"{stem}.png").convert("L"), float)


def main(paper, out):
    paper = pathlib.Path(paper).resolve()
    os.makedirs(out, exist_ok=True)
    bad = 0
    for pdf in sorted(NB.glob("*.pdf")):
        name = pdf.stem
        ref = paper.parent / EXTERNAL[name] if name in EXTERNAL else paper / "figures" / pdf.name
        a, b = render(pdf, f"{out}/{name}_nb"), render(ref, f"{out}/{name}_paper")
        if a.shape != b.shape:
            print(f"{name:26s} SIZE  {a.shape} against {b.shape}")
            bad += 1
            continue
        d = np.abs(a - b)
        Image.fromarray(np.hstack([a, b, 255 - np.clip(4 * d, 0, 255)]).astype(np.uint8)).save(f"{out}/{name}_diff.png")
        n = int((d > 0).sum())
        print(f"{name:26s} {'SAME' if n == 0 else 'DIFF'}  {n} pixels differ")
        bad += n > 0
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp()))
