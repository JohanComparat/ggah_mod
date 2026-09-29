r"""The gate that keeps "every parameter has its prior" true a month later.

``ggah_cal``'s campaign fits layer 3 one parameter block at a time and writes a
``priors.yaml``; its ``40_emit_params.py`` rewrites the matching ``Param`` here
and appends the provenance to its ``why``.  Without something on this side
asserting the two agree, a parameter whose fit has landed can quietly stay
``Flat`` and the campaign's headline claim decays without a single test going
red.  That is what this module is: **the code half of the pairing.**

It is useful before that table exists, which is the point of writing it now.
The census below is measured from the package on every run, so the number that
matters -- how many parameters carry no reason for their bound -- is checked
rather than asserted in a README, and Block C's effect on it is visible rather
than claimed.
"""
import pathlib

import pytest

import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod.sectors.params import BOUND_KINDS, Flat, Param, SectorParams

#: Where ``ggah_cal``'s ``30_collect_priors.py`` writes its table, if it has.
PRIORS_TABLE = pathlib.Path(
    __file__).resolve().parents[2] / "ggah_cal" / "results" / "priors.yaml"


def _assert_every_params_name_is_a_container(module):
    """A name ending in ``Params`` must actually *be* a container.

    The audits below find containers by filtering ``__all__`` for
    ``SectorParams`` subclasses, and a filter cannot tell a name it should skip
    from a container that has stopped being one.  ``ObscurationParams`` spent a
    commit as a ``PjitFunction`` -- a stray ``@jax.jit`` landed on the class
    instead of the function below it -- so ``isinstance(obj, type)`` was False
    and all five audits dropped its eight parameters in silence.  Nothing was
    wrong with them; nothing was checking either, and the census reported
    "no parameter without a reason" about the wrong set.

    Same shape as ``GalaxyParams`` escaping for want of an export, one step
    over: exported, and not a type.  So the walk asserts rather than filters.
    """
    bad = []
    for name in module.__all__:
        if not name.endswith("Params") or name == "SectorParams":
            continue
        obj = getattr(module, name, None)
        if not (isinstance(obj, type) and issubclass(obj, SectorParams)):
            bad.append(f"{module.__name__}.{name} is {type(obj).__name__}, "
                       f"not a SectorParams subclass")
    assert not bad, bad


def _containers() -> dict[str, type]:
    """Every declared parameter container, layers 3 and 4."""
    out = {}
    for module in (S, SP):
        _assert_every_params_name_is_a_container(module)
        for name in module.__all__:
            obj = getattr(module, name, None)
            if (isinstance(obj, type) and issubclass(obj, SectorParams)
                    and obj is not SectorParams):
                out[obj.__name__] = obj
    return out


def _distinct_params() -> dict[int, tuple[str, str, Param]]:
    """``id -> (container, name, Param)``, deduplicated by object identity.

    Deduplicated because sharing is deliberate: ``ETA_EJ`` is one parameter in
    two containers and ``K_BOL`` is one in two more, so counting slots would
    report a package with more freedom than it has.
    """
    out = {}
    for cname, cls in sorted(_containers().items()):
        for pname, p in cls._PARAMS.items():
            out.setdefault(id(p), (cname, pname, p))
    return out


class TestTheCensus:
    """Measured on every run, so the claim cannot decay quietly."""

    def test_no_parameter_lacks_a_reason_for_its_bound(self):
        """The headline. ``Param.__post_init__`` refuses a bound with no
        ``why``, so this cannot fail while that holds -- and it is asserted
        anyway, because the check being *structural* is the claim, and a future
        edit that makes ``why`` optional would otherwise pass in silence."""
        naked = [f"{c}.{n}" for c, n, p in _distinct_params().values()
                 if not p.why.strip()]
        assert not naked, f"no recorded reason for: {naked}"

    def test_every_bound_declares_which_kind_it_is(self):
        """A ``prior`` moves when a measurement does and a ``physical`` one
        when the argument does; a bound that does not say which cannot be
        revised by anyone but its author."""
        for cname, pname, p in _distinct_params().values():
            assert p.kind in BOUND_KINDS, f"{cname}.{pname}: kind={p.kind!r}"

    def test_every_default_is_inside_its_own_bounds(self):
        for cname, pname, p in _distinct_params().values():
            lo, hi = p.bounds
            assert lo <= p.default <= hi, f"{cname}.{pname}"

    def test_the_census_is_what_it_was_measured_to_be(self):
        """Recorded so a change is a visible edit rather than a drift.

        ``ggah_cal``'s campaign opened with "68 declared, 18 informative, 50
        Flat" measured from this package.  Block C multiplied the first number
        by more than three: the occupation and CLF registries alone were 153
        numbers with no bound at all.
        """
        params = _distinct_params()
        containers = _containers()
        informative = sum(1 for _, _, p in params.values()
                          if not isinstance(p.prior, Flat))

        assert len(containers) >= 24, (
            f"{len(containers)} containers; it was 24 when this was written, "
            f"and containers only appear")
        assert len(params) >= 193, (
            f"{len(params)} distinct parameters; it was 193. If this fell, a "
            f"container stopped being reachable and its audits went with it -- "
            f"which is not hypothetical: ObscurationParams spent a commit as a "
            f"PjitFunction and this count read 185")
        assert informative >= 17

    def test_sharing_is_real_and_not_two_equal_copies(self):
        """``ETA_EJ`` is one object in two containers, so the count of
        distinct parameters is genuinely below the count of slots. If that
        stops being true it has been duplicated back.  ``K_BOL`` was shared the
        same way until 0.8.8 removed the luminosity channel that read it from
        ``EnergeticsParams``."""
        slots = sum(len(c._PARAMS) for c in _containers().values())
        assert len(_distinct_params()) < slots, (
            "no parameter is shared between containers any more")

        from ggah_mod.sectors.agn import K_BOL, AgnParams
        from ggah_mod.sectors.ejecta import ETA_EJ, EjectaParams
        from ggah_mod.sectors.energetics import EnergeticsParams

        assert EjectaParams._PARAMS["eta_ej"] is ETA_EJ
        assert EnergeticsParams._PARAMS["eta_ej"] is ETA_EJ
        assert AgnParams._PARAMS["k_bol"] is K_BOL
        assert "k_bol" not in EnergeticsParams._PARAMS


class TestTheTableAndTheCodeAgree:
    """The half that runs once ``ggah_cal`` has produced a table."""

    @staticmethod
    def _table():
        if not PRIORS_TABLE.is_file():
            pytest.skip(
                f"no {PRIORS_TABLE} yet: ggah_cal writes it from completed "
                f"fits, and its stage 4 has not run. This test is the gate "
                f"for when it has -- it is skipped, not absent, so it starts "
                f"checking on its own")
        import yaml

        return yaml.safe_load(PRIORS_TABLE.read_text())

    def test_every_fitted_parameter_stopped_being_flat(self):
        """The claim the campaign exists to keep true: a parameter whose fit
        has landed and which is still ``Flat`` here means the fit was not
        written back."""
        rows = self._table()
        params = {(c, n): p for c, n, p in _distinct_params().values()}
        stale = []
        for row in rows:
            cls = row["container"].rsplit(".", 1)[-1]
            key = (cls, row["name"])
            p = params.get(key)
            if p is None:
                stale.append(f"{key} is in the table and not in the package")
            elif row.get("kind") == "prior" and isinstance(p.prior, Flat):
                stale.append(f"{key} was fitted ({row['fit']}) and is still Flat")
        assert not stale, stale

    def test_the_values_match(self):
        rows = self._table()
        params = {(c, n): p for c, n, p in _distinct_params().values()}
        for row in rows:
            cls = row["container"].rsplit(".", 1)[-1]
            p = params[(cls, row["name"])]
            if row.get("bounds"):
                assert list(p.bounds) == pytest.approx(row["bounds"]), row["name"]
