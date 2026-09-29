"""A rename here must fail here, not three days later on a cluster.

`ggah_cal` reaches about forty symbols of this package.  When one is renamed,
nothing in this repository notices: the failure arrives downstream as an
`AttributeError`, usually in job 4 of 15 on a compute node, hours after the
commit and in the repository that did not make the change.

`ggah_cal/contract.py` is that list, written as data with no imports precisely
so this test can read it.  Every entry is resolved **by `ast`**, importing
nothing -- the same technique the technical paper's `checksrc.py` uses on its
equation annotations, and for the same reason: importing to check a name means
the check needs the environment, and this one must run wherever the package
does.

Three properties, and the middle one is the point:

* it is **skipped by name** when `ggah_cal` is not on disk, the way
  `test_plan_is_true.ELSEWHERE` skips sibling paths.  A missing sibling is not a
  failure of this package;
* it fails **here**, in the commit doing the renaming, which is the only place
  the person renaming can act on it;
* it is a statement of what `ggah_cal` currently calls, **not a promise this
  package has made**.  Deliberately breaking it is allowed -- the fix is to
  change `contract.py` in the same change, which makes the break a decision
  with a name on it rather than an accident.

The scoped layer-3 renames are what this is for: `alpha_sat` became
`alpha_faint` *in the CLFs only*, `b_sat` became `phi_s_amp`, `alpha_shmr`
became `alpha_shmr_k18` *in `zacharegkas25` only*, and the old spellings still
exist elsewhere meaning other things -- so a blind find-and-replace is wrong and
a downstream caller cannot be checked by grep.
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where `ggah_cal` is, if it is anywhere.  Environment first, then the
#: workstation layout, the same order `ggah_cal/paths.py` resolves its own roots
#: in -- and for the same reason: a hard-coded `/home/<somebody>` is wrong on
#: the cluster, which is where this package's downstream actually runs.
_SIBLINGS = {
    "ggah_cal": pathlib.Path(os.environ.get(
        "GGAH_CAL_REPO", pathlib.Path.home() / "software" / "ggah_cal")
    ) / "ggah_cal" / "contract.py",
    "ggah_bench": pathlib.Path(os.environ.get(
        "GGAH_BENCH_REPO",
        pathlib.Path.home() / "software" / "ggah_mod_benchmark")
    ) / "ggah_bench" / "contract.py",
}
_PRESENT = {k: v for k, v in _SIBLINGS.items() if v.exists()}


def _requirements(contract: pathlib.Path) -> list[tuple[str, str, str]]:
    """`contract.REQUIREMENTS`, read as a literal.

    Parsed rather than imported: `contract.py` promises to be plain data, this
    is where that promise is cashed, and it means the check does not need
    `ggah_cal` installed -- only present.
    """
    tree = ast.parse(contract.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "REQUIREMENTS":
            return [tuple(ast.literal_eval(e)) for e in node.value.elts]
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "REQUIREMENTS" for t in node.targets):
            return [tuple(ast.literal_eval(e)) for e in node.value.elts]
    raise AssertionError("no REQUIREMENTS literal in contract.py")


def _module_path(dotted: str) -> pathlib.Path | None:
    """`ggah_mod.sectors.gas` -> the file, without importing it."""
    rel = pathlib.Path(*dotted.split("."))
    for cand in (ROOT / rel.with_suffix(".py"), ROOT / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _defines(path: pathlib.Path, name: str) -> bool:
    """Is `name` defined, assigned or re-exported by that module?

    Re-exports count: `ggah_mod.sectors` names `HotGasDPM` in its `__init__`
    without defining it, and a caller importing it from there is entitled to it.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == name:
                    return True
                if isinstance(t, ast.Tuple) and any(
                        getattr(e, "id", None) == name for e in t.elts):
                    return True
        elif isinstance(node, ast.AnnAssign):
            if getattr(node.target, "id", None) == name:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if (a.asname or a.name.split(".")[0]) == name:
                    return True
    return False


pytestmark = pytest.mark.skipif(
    not _PRESENT,
    reason="no sibling repository on disk; set GGAH_CAL_REPO or "
           "GGAH_BENCH_REPO to point at one")


@pytest.mark.parametrize("repo", sorted(_PRESENT))
def test_the_contract_is_readable_and_not_empty(repo):
    reqs = _requirements(_PRESENT[repo])
    assert len(reqs) >= 20, (
        f"{repo}'s contract has only {len(reqs)} entries; has it moved?")


@pytest.mark.parametrize(
    "repo,module,symbol",
    [(r, m, s) for r, c in sorted(_PRESENT.items())
     for m, s, _ in _requirements(c)])
def test_every_symbol_a_sibling_calls_still_exists(repo, module, symbol):
    why = {(m, s): w for m, s, w in _requirements(_PRESENT[repo])}[(module, symbol)]
    path = _module_path(module)
    assert path is not None, (
        f"{repo} calls {module}.{symbol} and there is no such module here. "
        f"It needs it for: {why}")
    # `from ggah_mod.cosmology import growth` imports a *submodule*, not a name
    # the package's `__init__` has to define.  Both spellings appear in the
    # siblings' contracts, so both have to resolve.
    if _module_path(f"{module}.{symbol}") is not None:
        return
    assert _defines(path, symbol), (
        f"{repo} calls {module}.{symbol} and this tree no longer defines it. "
        f"It needs it for: {why}\n"
        f"If the rename is intended, change {repo}/contract.py in the same "
        f"commit -- that is what makes this a decision rather than an outage.")


def test_the_resolver_can_fail():
    """A resolver that said yes to everything would make this suite decorative."""
    path = _module_path("ggah_mod.sectors.gas")
    assert path is not None
    assert _defines(path, "HotGasDPM")
    assert not _defines(path, "HotGasDPM_that_does_not_exist")
