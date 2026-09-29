r"""``CROSS_REPO.md``'s rows name symbols that exist.

M3 of the joint plan.  ``PLAN.md``'s *Notes outbound* section was the right idea
with no mechanism: two of its entries had been applied in the target repository
and nothing here knew, while four more obligations accumulated in three other
documents.

A note that names a symbol in another repository is a **claim about that
repository**, and an unchecked claim rots exactly as fast as an unchecked status
word -- which ``test_plan_is_true.py`` learned by carrying six stale ones for two
tiers.  So every row is resolved against its target by ``ast``, importing
nothing, and skipped by name when the sibling is absent.
"""
import ast
import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: The inbox is one of the maintainers' development records, private since
#: 1.0.0; ``_rows`` skips by name when it is absent.
INBOX = pathlib.Path(os.environ.get(
    "GGAH_DEV_REPO",
    pathlib.Path.home() / "software" / "ggah_mod_dev")) / "CROSS_REPO.md"
SIBLINGS = ROOT.parent

#: repository name -> the package directory its symbols live under.
#:
#: The paper is the fourth member of this family and it was not here, so a row
#: naming it failed as an "unknown repository" rather than resolving. It is
#: also the one member with no installable package: its checkers are loose
#: modules at the repository root, so the root *is* the package directory. Its
#: rows are as real as any other's -- it consumes `ggah_mod`'s symbols in
#: `checksrc.py` annotations and `ggah_cal`'s output in `checkcal.py`, and both
#: break the same way when an upstream name moves.
PACKAGE = {
    "ggah_mod": ROOT / "ggah_mod",
    "ggah_cal": SIBLINGS / "ggah_cal" / "ggah_cal",
    "ggah_mod_benchmark": SIBLINGS / "ggah_mod_benchmark" / "ggah_bench",
    "ggah_mod_technical_paper":
        pathlib.Path.home() / "Documents" / "papers" / "ggah_mod_technical_paper",
    # The two emulators are family members too, and a row addressed to one
    # failed as an "unknown repository" rather than resolving.  `emu_pk` was
    # taken at `emu_pk_bench`, a clone held at v1.0.0 while X9 kept the pin.
    # X9's own condition -- a checkpoint retrained *with* `Omega_k` -- was met
    # by v2.0.0, so the pin is lifted and this points at the repository the
    # package actually is.  The clone stayed at v1.0.0 for ten days after the
    # package moved, which is how the paper came to measure an eight-parameter
    # network while describing an eleven-parameter one.
    "emu_pk": pathlib.Path.home() / "software" / "emu_pk" / "emu_pk",
    "emu_hmf": pathlib.Path.home() / "software" / "emu_hmf" / "emu_hmf",
    # The sensitivity studies on the differentiable path: they differentiate
    # every layer in the cosmology, which is how the w_p line-of-sight grid and
    # the beyond-linear bias's kinks were found (X42-X46).
    "ggah_sens_study": SIBLINGS / "ggah_sens_study" / "ggah_sens_study",
}
STATUSES = {"open", "applied", "settled", "refused"}

_ROW = re.compile(r"^\|\s*(X\d+)\s*\|(.+)\|\s*$")


def _rows():
    if not INBOX.exists():
        pytest.skip(f"no development records at {INBOX.parent}; set GGAH_DEV_REPO")
    out = []
    for line in INBOX.read_text().splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        # Split on *unescaped* pipes: a claim may quote `|cos|`, and markdown's
        # escape for that is `\|`.  Splitting naively turned one row into
        # seven cells and looked like a malformed table.
        cells = [c.replace("\\|", "|").strip()
                 for c in re.split(r"(?<!\\)\|", m.group(2))]
        if len(cells) != 5:
            pytest.fail(f"row {m.group(1)} has {len(cells)} cells, expected 5")
        frm, to, _claim, symbol, status = cells
        out.append((m.group(1), frm, to, symbol.strip("`"), status))
    return out


ROWS = _rows() if INBOX.exists() else []


def _resolve(package: pathlib.Path, symbol: str) -> bool:
    """``module.py::Name`` or ``module.py::Class.method``, by ``ast``."""
    rel, _, dotted = symbol.partition("::")
    path = package / rel
    if not path.exists():
        return False
    tree = ast.parse(path.read_text())
    parts = dotted.split(".")

    def walk(node, want):
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            if name != want[0]:
                continue
            if len(want) == 1:
                return True
            if walk(child, want[1:]):
                return True
        return False

    return walk(tree, parts)


def test_the_inbox_exists_and_has_rows():
    if not INBOX.exists():
        pytest.skip(f"no development records at {INBOX.parent}; set GGAH_DEV_REPO")
    assert ROWS, "CROSS_REPO.md has no rows; M3 is not seeded"


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_every_row_names_a_symbol_that_resolves(row):
    """Against the repository that **owns** the symbol, not the one it was sent
    to.  A correction sent to the benchmark is usually about a symbol here, and
    the first version of this test assumed otherwise."""
    rid, frm, to, symbol, _status = row
    assert frm in PACKAGE and to in PACKAGE, (
        f"{rid}: unknown repository in {frm!r} -> {to!r}")
    owner, _, rest = symbol.partition(":")
    assert rest, (
        f"{rid}: symbol {symbol!r} is not qualified.  Write it as "
        f"`repo:path/to/module.py::Name`, because which repository owns a "
        f"symbol is not the same question as which one the note was sent to.")
    assert owner in PACKAGE, f"{rid}: unknown owning repository {owner!r}"
    pkg = PACKAGE[owner]
    if not pkg.exists():
        pytest.skip(f"{owner} is not on disk")
    assert _resolve(pkg, rest), (
        f"{rid} ({frm} -> {to}) names {rest!r} in {owner}, which does not "
        f"resolve.  Either the symbol was renamed -- in which case this row is "
        f"the notice that a cross-repo claim went stale -- or the row is wrong, "
        f"which is how three capability probes came to name symbols that had "
        f"never existed.")


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_every_row_has_a_known_status(row):
    rid, _frm, _to, _symbol, status = row
    assert status in STATUSES, (
        f"{rid} has status {status!r}; expected one of {sorted(STATUSES)}.  A "
        f"vocabulary that grows silently is how 'done' and 'DONE' and 'landed' "
        f"came to mean three things in one document")


def test_the_ids_are_unique():
    ids = [r[0] for r in ROWS]
    assert len(ids) == len(set(ids)), f"duplicate ids in CROSS_REPO.md: {ids}"


def test_it_is_not_a_second_plan():
    """Rows cross a boundary; anything that does not belongs in PLAN.md.

    A file that accumulates single-repository notes becomes a second plan, and
    ``test_plan_is_true.py::TestOnePlanDocument`` exists one document over for
    exactly that reason.
    """
    same = [(r[0], r[1]) for r in ROWS if r[1] == r[2]]
    assert not same, (
        f"these rows have the same `from` and `to`, so they do not cross a "
        f"boundary and belong in PLAN.md: {same}")
