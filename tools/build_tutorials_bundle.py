"""Stage the data the Paper I tutorials read, for a Zenodo upload.

Usage::

    python tools/build_tutorials_bundle.py            # stage files + manifest
    python tools/build_tutorials_bundle.py --tar      # also write the archive

The files go to ``~/data/zenodo/ggah_mod_tutorials/staging/<name>/``, with
``MANIFEST.sha256`` and ``ARCHIVE.json`` (sources and git revisions); the manifest is copied to
``tutorials/data_registry.txt``, and the
Zenodo metadata to ``ggah_mod_tutorials.zenodo.json`` beside them.  ``FILES``
lists only what a tutorial reads; it is provisional until every notebook is
written, and the upload waits until then.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

HOME = os.path.expanduser("~")
OUT = os.path.join(HOME, "data", "zenodo", "ggah_mod_tutorials")
NAME = "ggah_mod_tutorials_data"
#: The copy of the manifest the notebooks check each file against.
REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tutorials", "data_registry.txt")

PAPER = os.path.join(HOME, "software", "ggah_mod_technical_paper")
SENS = os.path.join(HOME, "data", "ggah_sens_study_results_v1.8.0dev", "halo_req")
TIER1 = os.path.join(HOME, "data", "ggah_inputs", "halo_req_v1.2.0", "halo_req", "tier1")
WP1 = os.path.join(HOME, "software", "cog-prop-wp1-feasibilty", "results")
EMU = os.path.join(HOME, "data", "xcorr_talk_emupk", "benchmark")
BENCH = os.path.join(HOME, "software", "ggah_mod_benchmark", "results")

#: (path in the bundle, source, notebook that reads it)
FILES = [
    # 07, 08: published DESI posteriors drawn as outlines
    ("desi/desi_dr2_contours.json", f"{PAPER}/desi_dr2_contours.json", "07"),
    ("desi/desi_dr1_fs_contours.json", f"{PAPER}/desi_dr1_fs_contours.json", "08"),
    # 10: accuracy required of the halo layer
    ("halo_req/counts/results.json", f"{SENS}/counts/results.json", "10"),
    ("halo_req/counts/kernels.npz", f"{SENS}/counts/kernels.npz", "10"),
    ("halo_req/tier1/results.json", f"{TIER1}/results.json", "10"),
    ("halo_req/tier1/kernels.npz", f"{TIER1}/kernels.npz", "10"),
    ("halo_req/joint_closure/results_C1+SSC.json", f"{SENS}/joint_closure/results_C1+SSC.json", "10"),
    ("halo_req/joint_closure/kernels_C1+SSC.npz", f"{SENS}/joint_closure/kernels_C1+SSC.npz", "10"),
    # 11: errors of the planned simulation suite
    ("suite/65_suite_tables.json", f"{WP1}/65_suite_tables.json", "11"),
    ("suite/72_suite_redesign.json", f"{WP1}/72_suite_redesign.json", "11"),
    ("suite/33_error_budget.json", f"{WP1}/33_error_budget.json", "11"),
    ("suite/74_comparat17_errors.json", f"{WP1}/74_comparat17_errors.json", "11"),
    ("suite/70_suite_errors.json", f"{WP1}/70_suite_errors.json", "11"),
    # A: emulator benchmark (the Dahu-node CPU run and the laptop-GPU run the paper shows)
    ("emulators/cosmologies.json", f"{EMU}/cosmologies.json", "A"),
    ("emulators/truth.npz", f"{EMU}/truth.npz", "A"),
    ("emulators/ref.npz", f"{EMU}/ref.npz", "A"),
    ("emulators/timing_truth.json", f"{EMU}/timing_truth.json", "A"),
    ("emulators/cpu.json", f"{EMU}/cpu.json", "A"),
    ("emulators/gpu.json", f"{EMU}/gpu.json", "A"),
    ("emulators/bacco_benchmark.json", f"{BENCH}/bacco_benchmark.json", "A"),
    # B (and Fig. parity): code comparisons
    ("comparisons/accuracy.json", f"{PAPER}/accuracy.json", "B"),
]

ZENODO = {
    "title": "ggah_mod tutorials: the data behind the Paper I notebooks",
    "upload_type": "dataset",
    "description": ("Precomputed products read by the ggah_mod tutorial notebooks "
                    "(tutorials/ in the ggah_mod repository) that reproduce the sections "
                    "and figures of Paper I: published DESI posteriors, the halo-layer "
                    "accuracy requirements, the errors of the planned simulation suite, "
                    "the linear-emulator benchmark and the code comparisons. "
                    "MANIFEST.sha256 lists every file; ARCHIVE.json gives each source "
                    "and the git revisions."),
    "creators": [{"name": "Comparat, Johan", "orcid": "0000-0001-9200-1497"}],
    "license": "MIT",
    "keywords": ["cosmology", "halo model", "large-scale structure", "tutorial",
                 "reproducibility"],
    "related_identifiers": [
        {"identifier": "https://github.com/JohanComparat/ggah_mod",
         "relation": "isSupplementTo", "scheme": "url"}],
}


def _rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar", action="store_true", help="also write the .tar.gz")
    args = ap.parse_args()

    stage = os.path.join(OUT, "staging", NAME)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    missing = [src for _, src, _ in FILES if not os.path.isfile(src)]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(missing))

    lines, sources = [], []
    for dst, src, nb in FILES:
        target = os.path.join(stage, dst)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(src, target)
        lines.append(f"{_sha(target)}  {dst}")
        sources.append({"file": dst, "source": src.replace(HOME, "~"), "notebook": nb})
    for path in (os.path.join(stage, "MANIFEST.sha256"), REGISTRY):
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    repos = {r: _rev(os.path.join(HOME, "software", r)) for r in
             ("ggah_mod", "ggah_mod_technical_paper", "cog-prop-wp1-feasibilty",
              "ggah_mod_benchmark")}
    with open(os.path.join(stage, "ARCHIVE.json"), "w") as f:
        json.dump({"files": sources, "git": repos}, f, indent=1)
    with open(os.path.join(OUT, "ggah_mod_tutorials.zenodo.json"), "w") as f:
        json.dump(ZENODO, f, indent=2)
    print(f"staged {len(FILES)} files in {stage}")

    if args.tar:
        tgz = os.path.join(OUT, f"{NAME}-{repos['ggah_mod']}.tar.gz")
        with tarfile.open(tgz, "w:gz") as t:
            t.add(stage, arcname=NAME)
        print(f"{tgz}  sha256 {_sha(tgz)}")


if __name__ == "__main__":
    main()
