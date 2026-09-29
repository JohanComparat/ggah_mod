r"""Verification: the plan of record says what the package measures.

Three planning documents described this package at once -- ``PLAN.md``,
``TODO_PLAN.md`` and ``plan_complete_baryon_census.md``, 1641 lines between them
-- and they drifted from the code and from each other.  Twelve claims were wrong
when they were finally checked against the source: two retired backend names
used throughout as live ones, the parity budget marked **TODO** on one row and
**DONE** two rows later, a *Next* list naming work the same file's status table
marked done, three different test counts, four registry sizes, and two
capabilities deferred by name that had already shipped.

None of it was carelessness.  Each claim was true when it was written, and
nothing existed that could notice when it stopped being.  A document is not
covered by the suite the way a module is, so the only claims that stay true are
the ones something checks.

This module is that something.  It does **not** check prose -- prose is where
the reasons live and reasons are not testable -- it checks the small number of
statements in ``PLAN.md`` that are facts about the code: how many entries a
registry has, which modules exist, and which names are retired.  The direction
matters: the package is the truth and the document follows it, so a registry
that grows makes this suite red until ``PLAN.md`` is updated, and never the
other way round.
"""
import os
import pathlib
import re

import pytest

from ggah_mod.backend import _RETIRED
from ggah_mod.halos.concentration import CONCENTRATION
from ggah_mod.halos.linear_bias import BIAS
from ggah_mod.halos.mass_function import MULTIPLICITY
from ggah_mod.sectors.clf import CLF
from ggah_mod.sectors.galaxies import GALAXY_MODELS
from ggah_mod.sectors.occupation import OCCUPATION
from ggah_mod.sectors.sham import SHMR

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: The maintainers' development records, private since 1.0.0: ``PLAN.md`` lives
#: there and not in this repository.  Every check that reads it skips by name
#: when they are absent, as the sibling-contract checks do.
DEV = pathlib.Path(os.environ.get(
    "GGAH_DEV_REPO",
    pathlib.Path.home() / "software" / "ggah_mod_dev"))
PLAN = DEV / "PLAN.md"
README = ROOT / "README.md"
#: The layer-2 documentation page, which quotes the same two agreements.
LAYER2_PAGE = ROOT / "docs" / "layer2_halos.md"


def _plan() -> str:
    if not PLAN.is_file():
        pytest.skip(f"no development records at {DEV}; set GGAH_DEV_REPO")
    return PLAN.read_text(encoding="utf-8")


def _status_row(label: str) -> str:
    """The one status-table row naming ``label`` in backticks.

    Raises rather than returning ``None`` on a miss: a registry that has lost
    its row is exactly the drift this module exists to catch, and an assertion
    against an empty string would pass for the wrong reason.
    """
    rows = [ln for ln in _plan().splitlines()
            if ln.startswith("|") and f"`{label}`" in ln]
    assert len(rows) == 1, (
        f"expected exactly one status-table row naming `{label}`, found "
        f"{len(rows)}: {rows}")
    return rows[0]


# ==========================================================================
# The counts
# ==========================================================================
class TestTheCountsAreMeasured:
    """Every number ``PLAN.md`` states about a registry, against the registry.

    Three revisions of that file disagreed with the package *and with each
    other* on all of these -- 16 multiplicity fits in one paragraph and 17 in
    another against a registry holding 18; 8 occupations against 12.  A count
    is the cheapest kind of claim to check and was the most often wrong,
    because it is the kind that changes without anyone editing prose.
    """

    #: ``(status-table label, registry)``.  The label is what appears in
    #: backticks in the row; several registries legitimately share one row,
    #: because ``occupation`` states its HODs, its CLFs and its SHMRs together.
    COUNTED = [
        ("mass_function", MULTIPLICITY),
        ("linear_bias", BIAS),
        ("concentration", CONCENTRATION),
        ("occupation", OCCUPATION),
        ("occupation", CLF),
        ("occupation", SHMR),
        ("GalaxySector", GALAXY_MODELS),
    ]

    @pytest.mark.parametrize("label,registry", COUNTED,
                             ids=[f"{a}-{len(b)}" for a, b in COUNTED])
    def test_the_plan_states_the_measured_size(self, label, registry):
        row = _status_row(label)
        assert f"**{len(registry)}**" in row, (
            f"`{label}`'s registry holds {len(registry)} entries and its "
            f"PLAN.md row does not say so:\n  {row}\n"
            f"The package is the truth here -- update the document.")


# ==========================================================================
# One document
# ==========================================================================
class TestOnePlanDocument:
    """``PLAN.md`` is the only plan, and nothing points at the deleted two.

    The two that were deleted are in git history and that is where they belong;
    a module docstring citing one by filename sends a reader to a file that no
    longer exists, which is worse than citing nothing.  ``sectors/ejecta.py``
    did exactly that until the deletion commit repointed it.
    """

    #: Deleted 2026-09-01, folded into ``PLAN.md``.  Spelled in parts so this
    #: module is not itself a hit when the search below runs over the tree.
    GONE = ["TODO" "_PLAN.md", "plan_complete" "_baryon_census.md"]

    def test_the_plan_of_record_exists(self):
        assert _plan()

    @pytest.mark.parametrize("name", GONE)
    def test_the_superseded_documents_are_gone(self, name):
        assert not (ROOT / name).exists(), (
            f"{name} is back.  It was folded into PLAN.md; two plan documents "
            f"is how the last three came to disagree.")

    @pytest.mark.parametrize("name", GONE)
    def test_no_source_file_cites_a_deleted_document(self, name):
        stem = name[:-3]
        offenders = [
            f"{p.relative_to(ROOT)}:{i}"
            for p in sorted((ROOT / "ggah_mod").rglob("*.py")) + [README]
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if stem in ln
        ]
        assert not offenders, (
            f"these cite the deleted {name}: {offenders}.  Point them at "
            f"PLAN.md's Tier 6 or Tier 7 item instead.")


# ==========================================================================
# The retired names
# ==========================================================================
class TestTheRetiredNamesAreGone:
    """A retired flavour name may be *explained* in the docs, never *used*.

    ``backend.__getattr__`` keeps the old names importable behind a
    ``DeprecationWarning`` so the benchmark and existing notebooks migrate at
    their own pace.  That is a kindness to callers and it is precisely why the
    documentation has to be stricter than the code: an importable name that
    still reads as current in the README is one nobody migrates away from.

    So the rule is not "never appears" -- ``PLAN.md`` has to be able to record
    that these were the old names -- but "appears only while being called old".
    """

    #: A line may name a retired flavour if it also says one of these.  Both
    #: earn their place: the corrections table says ``retired``, and the row
    #: recording the stale ``cm_model`` finding says ``stale``.
    EXPLAINING = ("retired", "stale")

    @pytest.mark.parametrize("name", sorted(_RETIRED))
    def test_the_readme_does_not_use_a_retired_name(self, name):
        hits = [f"{i}: {ln.strip()}"
                for i, ln in enumerate(README.read_text(encoding="utf-8")
                                       .splitlines(), 1)
                if re.search(rf"\b{name}\b", ln)]
        assert not hits, (
            f"README.md still calls the flavour {name!r}; it is "
            f"{_RETIRED[name][0]}.  Lines: {hits}")

    @pytest.mark.parametrize("name", sorted(_RETIRED))
    def test_the_plan_names_a_retired_flavour_only_to_call_it_retired(self, name):
        hits = [f"{i}: {ln.strip()}"
                for i, ln in enumerate(_plan().splitlines(), 1)
                if re.search(rf"\b{name}\b", ln)
                and not any(w in ln for w in self.EXPLAINING)]
        assert not hits, (
            f"PLAN.md uses {name!r} as though it were current. Say "
            f"{_RETIRED[name][0]}, or keep the mention and say it is retired. "
            f"Lines: {hits}")


# ==========================================================================
# The modules the plan names
# ==========================================================================
class TestEveryModuleThePlanNamesExists:
    r"""A path in ``PLAN.md`` either exists, or is declared as not existing yet.

    The old document named ``halos/subhalos.py`` and ``cosmology/snapshot.py``
    as though they were on disk; neither has ever existed.  Nothing
    distinguished those from the forty paths in the same file that did, so a
    reader had to open each one to find out.

    ``PLANNED`` is that distinction, and it is a *shrinking* list: the second
    test below fails when one of its files appears, so the entry has to be
    removed in the commit that creates it.  That is the same mechanism
    ``test_backends.py::UNREAD`` uses, and for the same reason -- an exemption
    that outlives its cause is worse than no exemption, because it reads as a
    considered decision.
    """

    #: Files a Tier 6 item will create, keyed by the path **as PLAN.md writes
    #: it** -- which is sometimes package-relative and sometimes not, because
    #: prose naming a module beside its siblings drops the top level and prose
    #: naming a new subpackage does not.  :meth:`_resolve` accepts both, so the
    #: keys have to match the citation rather than the file system; getting
    #: that backwards was this module's own first failure.  Value is the item.
    PLANNED = {
        # Retired as their files landed, each noticed by the test below rather
        # than by anyone remembering: "spectra/transition.py" (A1),
        # "tests/test_parity_budget.py" (B1),
        # "sectors/occupation_params.py" (C1),
        # "tests/test_priors_match_table.py" (C6),
        # "ggah_mod/interfaces/cobaya.py" (E1).
    }

    #: Paths that belong to a sibling repository.  ``PLAN.md`` cites them
    #: because Tier 6's gates and two of its findings live there; they are not
    #: this repository's to create, and their absence here is not drift.
    #: Shrinks the same way ``PLANNED`` does. ``ggah_bench/adapters/cobaya.py``
    #: and ``ggah_bench/units.py`` left when E1 landed and PLAN.md stopped
    #: citing them as where the mapping lived -- it lives here now.
    ELSEWHERE = {
        "ggah_bench/findings.py", "ggah_bench/features/inventory.py",
        "features/inventory.py", "report/prose.py",
        "tests/test_benchmark_more2015.py",
        # E5's home.  The CCL halo-model comparison this plan asked the
        # benchmark for turned out to exist there already, which is why the
        # path is cited at all.
        "ggah_bench/halomodel.py",
        # T7-1's courier: the generator that produced the Oguri et al. (2026)
        # goldens.  It lives in the predecessor and is named so the arrays can
        # be regenerated from the reference rather than trusted.
        "hod_mod/scripts/cosmology/make_lensing_goldens.py",
        # The halofit row -- a third method against layer 4, and the only
        # comparison that can see an error all three halo-model
        # implementations share.  The benchmark's script, not this repo's.
        "scripts/30_halomodel.py",
        # Its sweep over the neutrino ladder, which is where the control that
        # makes the halofit number readable lives.
        "scripts/32_halofit.py",
        # T7-3's second edge effect.  The guard has to live where the 75
        # directly-integrated band tables are, which is `ggah_cal`; it is cited
        # here because it is what would tell this repository that BandCooling
        # had become good enough to retire them.
        "ggah_cal/tests/test_band_cooling_against_direct.py",
    }

    _PATH = re.compile(r"`([A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)+\.py)")

    @classmethod
    def _cited(cls):
        return sorted(set(cls._PATH.findall(_plan())))

    @staticmethod
    def _resolve(rel: str):
        """``PLAN.md`` writes package paths with and without the top level."""
        for cand in (ROOT / rel, ROOT / "ggah_mod" / rel):
            if cand.exists():
                return cand
        return None

    def test_every_path_is_on_disk_or_declared_absent(self):
        missing = [p for p in self._cited()
                   if p not in self.ELSEWHERE
                   and p not in self.PLANNED
                   and self._resolve(p) is None]
        assert not missing, (
            f"PLAN.md names these and they do not exist: {missing}.  Either "
            f"the path is wrong, or the file is planned -- in which case add "
            f"it to PLANNED with the Tier 6 item that will write it.")

    @pytest.mark.parametrize("rel,item", sorted(PLANNED.items()))
    def test_a_planned_file_loses_its_exemption_when_it_lands(self, rel, item):
        assert self._resolve(rel) is None, (
            f"{rel} exists now, so Tier 6 item {item} has landed.  Remove it "
            f"from PLANNED in the same commit, and mark {item} done in "
            f"PLAN.md.")

    def test_an_empty_planned_list_means_every_named_file_landed(self):
        """``PLANNED`` is empty, and that is a state worth naming.

        It shrank one entry at a time -- A1, B1, C1, C6, E1 -- each retired by
        the test above failing rather than by anyone remembering, which is the
        mechanism ``test_backends.py::UNREAD`` uses and the reason that list is
        also empty. An empty exemption list is the goal, not an oversight; the
        parametrised test above simply has nothing left to check, and pytest
        reports that as a skip.
        """
        assert not self.PLANNED, (
            f"still planned: {sorted(self.PLANNED)} -- which is fine, but this "
            f"test's docstring is then out of date")

    def test_the_exemption_lists_are_not_padded(self):
        cited = set(self._cited())
        stale = (set(self.PLANNED) | self.ELSEWHERE) - cited
        assert not stale, (
            f"these are exempted and PLAN.md no longer cites them: {stale}")


# ==========================================================================
# The one claim the repository makes twice, differently
# ==========================================================================
class TestTheClaimThatWasDisputed:
    """``lensing_profiles`` was documented two ways, and now is not.

    ``README.md`` said it "stopped being a placeholder"; the module,
    ``halos/__init__.py`` and two live assertions in ``test_coherence.py`` said
    it was a placeholder awaiting external goldens.  Both were describing
    something true -- the word *placeholder* was doing the disagreeing.  This
    was an ``xfail(strict=True)`` naming PLAN.md item **F2** so that whichever
    way F2 went, the test would turn red and have to be rewritten.  It went.

    F2 made the precise statement in all four places -- cross-validated
    internally to **3.0e-7 of the peak**, no external reference -- and moved the
    *gap* under it to Tier 7, where obtaining a reference was work rather than
    wording.  Tier 7 did it: the kernels now run against the Oguri et al. (2026)
    implementation at ``mpmath`` 50 digits and agree to **1.2e-13**.

    So this class now pins **two** numbers rather than one status word, which is
    the durable form.  A status word can improve quietly; two measurements
    cannot, and if either check is ever dropped the number for it disappears
    from the docstrings and this turns red.

    The README carried the third copy until 1.0.0, when its layer-5 notes were
    withdrawn until that layer is documented; the layer-2 page of the
    documentation carries it now.  The README is still held to never
    contradicting the claim.
    """

    def test_the_readme_and_the_module_make_the_same_claim(self):
        from ggah_mod.halos import lensing_profiles

        module = (lensing_profiles.__doc__ or "").lower()
        readme = README.read_text(encoding="utf-8").lower()
        page = LAYER2_PAGE.read_text(encoding="utf-8").lower()

        # Both name the external reference, in the same words.
        assert "oguri" in module
        assert "oguri" in page

        # None still says the reference is missing, and none calls the module
        # untested -- the two ways this claim has been wrong before.
        for text in (module, readme, page):
            assert "no external reference" not in text
            assert "not yet validated" not in text

    @pytest.mark.parametrize("number,what", [
        ("3.0e-7", "the internal cross-check"),
        ("1.2e-13", "the external reference"),
    ])
    def test_both_numbers_are_the_same_numbers_everywhere(self, number, what):
        """Each agreement is quoted in the module, in ``halos/__init__.py`` and
        on the layer-2 page.  Two measurements, three copies each, and copies
        drift."""
        import ggah_mod.halos as halos
        from ggah_mod.halos import lensing_profiles

        for text in ((lensing_profiles.__doc__ or ""),
                     (halos.__doc__ or ""),
                     LAYER2_PAGE.read_text(encoding="utf-8")):
            assert number in text, what


# ---------------------------------------------------------------------------
# Status words
# ---------------------------------------------------------------------------
#
# Everything above this line checks a *fact*: a registry's size, a module path, a
# banned name, two measurements.  Nothing checked a *status*, and that is exactly
# and only where this document rots.
#
# Audited on 2026-09-03: six Block-G items were marked open and were implemented
# -- G2, G3's first half, G4, G6, G8, G9, all landed in one commit that touched
# PLAN.md by a single line.  So was "settled decision 8", still describing the
# state before Block C closed.  So was Tier-8 step 5, reading `DAHU RUNNING` for
# a campaign the same document reports finished forty lines later.  So was E5's
# row.  Every drift was a status word, and every one moved in the same direction:
# the code ahead of the document.
#
# The mechanism is a registry mapping each closable item to a predicate on the
# code.  It is asserted in *both* directions, because either alone is half a
# check: an item the code has closed must say DONE, and an item that says DONE
# must be closed.

import inspect  # noqa: E402

#: item -> (predicate on the code, one line saying what closing it means).
#: A predicate raising is a failure, not a skip: these import the package the
#: document describes, and if that import breaks the document is unverifiable.
CLOSED_BY: dict[str, tuple] = {
    "G2": (lambda: ((ROOT / "tests" / "test_cooling.py").exists()
                    and (ROOT / "tests" / "test_clf.py").exists()),
           "both missing test modules exist"),
    "G4": (lambda: _src("ggah_mod.sectors.clf", "L_S_OVER_L_C")
                   and _src("ggah_mod.sectors.occupation", "M_S_OVER_M_C"),
           "both constants are named, each cross-referencing the other"),
    "G6": (lambda: _src("ggah_mod.sectors.energetics", "Muratov"),
           "the wind anchor names Muratov et al. (2015)"),
    "G8": (lambda: all(_src("ggah_mod.sectors.energetics", n)
                       for n in ("Sorini", "Davies")),
           "the energy closure's calibrations are cited"),
    "G9": (lambda: _src("ggah_mod.sectors.occupation_params", "phi_s_amp"),
           "the three colliding names are renamed"),
    # Still open, and the predicate is the deliverable rather than a proxy: G3's
    # second half closes when RETURN_FRACTION stops being a bare literal that a
    # Param merely wraps with an honest `why`, and starts being derived the way
    # `_t_b_coefficient_mk` derives the 21-cm coefficient.
    "G3": (lambda: _src("ggah_mod.sectors.energetics", "_return_fraction_from_imf"),
           "RETURN_FRACTION is derived rather than declared-and-flagged"),
    # Tier 10: the two-halo term, against pyhalomodel.
    "H1": (lambda: _src("ggah_mod.spectra.neutrinos", "def neutrino_leg")
                   and _src("ggah_mod.spectra.pk", "tracer_leg"),
           "the neutrino leg exists and the two-halo amplitude reads it"),
    "H2": (lambda: _src("ggah_mod.spectra.bnl", "add(low_mass_counterterm")
                   and _src("ggah_mod.halos.beyond_linear_bias",
                            "def correction_2h_fused"),
           "the beyond-linear term carries the low-mass point mass"),
    "H3": (lambda: _default_is("bnl", True),
           "the beyond-linear term is on by default, on both flavours"),
    "H4": (lambda: _default_is("one_halo_transition", "none")
                   and "mead15" in _retired_transitions(),
           "the damping is off by default and the hybrid is retired"),
    "H5": (lambda: _src("ggah_mod.halos.linear_bias", "def unresolved")
                   and _src("ggah_mod.sectors.census", "omega_outside_by_phase")
                   and _src("ggah_mod.spectra.counterterm", "w.w_unresolved"),
           "the unresolved matter is one object, read by census and spectra"),
    # 0.9.5 made both lines stricter -- the neutral gas cannot be expelled
    # either, and the budget hands on the hot gas -- so they read differently
    # while saying more than H6 asked for.
    "H6": (lambda: _src("ggah_mod.sectors.energetics",
                        "f_b_cosmic - f_star - f_cold - f_retained_min")
                   and _src("ggah_mod.sectors.gas", "hot = f_ret - (f_cen + f_sat)"),
           "the closure cannot expel stars and hands on gas"),
    "H7": (lambda: _src("ggah_mod.sectors.ejecta", "def split_from_gas")
                   and _src("ggah_mod.spectra.tracers", "def _check_gas_closure"),
           "the electrons composite can be built to close, and says when not"),
    "H8": (lambda: _src("ggah_mod.covariance.multihalo", "self.point_mass"),
           "the multi-leg moments carry the low-mass point mass"),
    "H10": (lambda: _src("ggah_mod.sectors.matter", "def _u_cold")
                    and "coldgas" in _optional_peers().get("matter", ()),
            "the matter field puts the neutral gas on the sector's profile"),
    "H9": (lambda: _src("ggah_mod.halos.beyond_linear_bias", "def _extrapolate")
                   and _src("ggah_mod.halos.beyond_linear_bias",
                            "check=True, extrapolate=True"),
           "the beyond-linear table is carried past the MultiDark sequence"),
}


def _default_is(field: str, value) -> bool:
    from ggah_mod.backend import ACCURATE, DIFFERENTIABLE
    from ggah_mod.spectra.spec import PkOptions
    return (getattr(PkOptions(), field) == value
            and getattr(ACCURATE, field) == value
            and getattr(DIFFERENTIABLE, field) == value)


def _optional_peers():
    from ggah_mod.spectra.tracers import OPTIONAL_PEERS
    return OPTIONAL_PEERS


def _retired_transitions():
    from ggah_mod.spectra.transition import ONE_HALO_TRANSITION, RETIRED
    return set(RETIRED) - set(ONE_HALO_TRANSITION)


def _src(module: str, needle: str) -> bool:
    """Is `needle` in that module's source?  Read, not imported for its value."""
    import importlib

    return needle in inspect.getsource(importlib.import_module(module))


def _bullet(item: str) -> str:
    """The PLAN.md bullet for a Block-G item, as one line."""
    m = re.search(rf"^- \*\*{re.escape(item)}\.\s*(.*?)(?=^- \*\*[A-Z]\d|\Z)",
                  _plan(), re.M | re.S)
    assert m, f"PLAN.md has no bullet for {item}"
    return " ".join(m.group(1).split())


class TestTheStatusWordsAreTrue:
    """The check the document's six stale bullets needed and did not have."""

    @pytest.mark.parametrize("item", sorted(CLOSED_BY))
    def test_an_item_the_code_has_closed_is_marked_done(self, item):
        closed, what = CLOSED_BY[item]
        if not closed():
            pytest.skip(f"{item} is genuinely open ({what} is not true yet)")
        head = _bullet(item)[:120]
        assert "DONE" in head, (
            f"{item} is closed in the code -- {what} -- and PLAN.md still opens "
            f"its bullet with: {head!r}")

    @pytest.mark.parametrize("item", sorted(CLOSED_BY))
    def test_an_item_marked_done_is_actually_closed(self, item):
        closed, what = CLOSED_BY[item]
        head = _bullet(item)[:120]
        if "DONE" not in head:
            pytest.skip(f"{item} is not claimed done")
        # "HALF DONE" is a third state and says so; it is allowed to be open.
        if "HALF DONE" in head:
            pytest.skip(f"{item} is claimed half done, which is not a claim to check")
        assert closed(), (
            f"PLAN.md marks {item} DONE and the code says otherwise: {what} is "
            "not true.")

    def test_no_campaign_is_described_as_running_and_finished_at_once(self):
        """Tier-8 step 5 read `DAHU RUNNING` while steps 6-7 reported it landed.

        A campaign is either in flight or it is not, and the document said both
        on one page.

        The check is for the marker rather than the words, because the first
        version of this test failed on the sentence that *records* the drift --
        a status word and a quotation of one are different things, and a test
        that cannot tell them apart makes the history unwritable.  In backticks
        it is a quotation; bare, it is a claim.
        """
        bare = [ln for ln in _plan().splitlines()
                if re.search(r"(?<!`)\bDAHU RUNNING\b(?!`)", ln)]
        assert not bare, (
            "PLAN.md still carries DAHU RUNNING as a status rather than as a "
            f"quotation: {bare[:2]}")

    def test_the_status_table_carries_exactly_the_open_rows(self):
        """`TODO` in the Status table must match what the tiers carry forward.

        Tier 6's summary names what it carries; the Status table marks it. The
        two disagreed -- the table listed Block G's items with no marker while
        the summary said Tier 6 was complete but for B4.

        B4 closed on 2026-09-04 and the table now carries **no** TODO, so this
        no longer asserts that one exists.  What it asserts instead is the
        property that was always the point: every marker in the table names an
        item this test knows about, and any that does not has to be declared
        here and in the tier summary before it can sit in the document.  An
        empty table is a legitimate state and an unexplained marker is not.
        """
        t = _plan()
        table = t[t.index("## Status"):t.index("## Decisions that are settled")]
        todos = re.findall(r"\*\*TODO\*\*(?:\s*—\s*([A-Z]\d))?", table)
        assert set(filter(None, todos)) <= {"B4"}, (
            f"Status table carries TODOs this test does not know about: {todos}. "
            "Add them here and to the tier summary, or close them.")
        bare = len(todos) - len(list(filter(None, todos)))
        assert not bare, (
            f"{bare} Status row(s) are marked TODO without naming the item that "
            f"carries them -- which is how Block G's rows sat open for two tiers")

    def test_b4_is_closed_in_both_places_or_neither(self):
        """The Status row and the B4 block have to agree about it.

        B4 was the table's only TODO for two tiers, so closing it touches two
        places, and the failure mode this pins is closing one of them.  The
        marker is looked for outside backticks: the block quotes the words it
        is retiring.
        """
        t = _plan()
        table = t[t.index("## Status"):t.index("## Decisions that are settled")]
        row = [ln for ln in table.splitlines() if "FFTLog's small-`r` end" in ln]
        assert len(row) == 1, "the Status table's B4 row moved or was reworded"
        row_done = "**DONE**" in row[0]

        block = [ln for ln in t.splitlines()
                 if re.search(r"(?<!`)\*\*B4\.", ln)]
        assert len(block) == 1, f"expected one B4 block heading, found {block}"
        block_done = "DONE" in block[0]

        assert row_done == block_done, (
            f"the Status row says {'DONE' if row_done else 'open'} and the B4 "
            f"block says {'DONE' if block_done else 'open'}")
