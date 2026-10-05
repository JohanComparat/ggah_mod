"""One mass definition, declared once, reaching everything.

The package has had this defect four times, in four places, and each was
invisible for the same reason: a hard-coded boundary returns a smooth, positive,
correctly-shaped number for a halo nobody asked about.

  * the mass function never received the field's ``Delta``;
  * ``make_field`` never passed ``mdef`` to the peak-height concentrations;
  * the gas sector built its own ``R_200c`` whatever the field said;
  * ``energetics`` did the same for the binding energy the feedback budget
    divides by.

The first two were found by a reader, the third and fourth by asking whether
they had been.  What makes this test worth having is that none of the four
would have been caught by checking a value: every one of them was *plausible*.
So the test is differential -- change the definition, and everything that
depends on a halo boundary must move.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

import ggah_mod.sectors.gas as G
from ggah_mod import ACCURATE, DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import energetics as E

_M = np.logspace(13.0, 15.0, 5)


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


def _field(pk, mdef, hmf="tinker08", cm="duffy08"):
    return make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0, m=_M, mdef=mdef,
                      hmf_model=hmf, bias_model="tinker10", cm_model=cm)


class TestTheDefinitionIsDeclaredInOnePlace:
    def test_the_flavours_are_the_only_declaration(self):
        """Every shipped flavour names one, and they agree with each other.

        Not because they must -- a caller may build a flavour at any definition
        -- but because shipping two defaults that differ would make "the
        package's mass definition" a question rather than a fact.
        """
        mdefs = {b.mdef for b in (ACCURATE, DIFFERENTIABLE)}
        assert len(mdefs) == 1, f"the shipped flavours disagree: {mdefs}"

    #: Modules that may write a definition as a literal, and why.
    #:
    #: Naming them one by one rather than exempting a pattern: each of these is
    #: a place where a definition is *data* rather than a choice, and a new
    #: module wanting on this list should have to argue for it.
    MAY_NAME_ONE = {
        "backend.py": "declares the flavours -- the single source of truth",
        "mass_definitions.py": "knows what the names mean",
        "concentration.py": "CM_CALIBRATION records what each fit was "
                            "calibrated in, and the relations take an mdef "
                            "argument whose default is their own definition",
        "mass_function.py": "CALIBRATION does the same for the abundance",
        "field.py": "HaloField's dataclass default, which make_field always "
                    "overrides from the flavour",
    }

    def test_no_new_module_hard_codes_a_definition(self):
        """A definition written as a literal is either data or a bug.

        The distinction is not syntactic -- ``"200c"`` in a calibration table
        records what a fit was calibrated in, and ``"200c"`` in a radius
        calculation ignores the caller -- so this cannot tell them apart.  What
        it can do is hold the line: the modules where a literal is legitimate
        are listed above with a reason, and a *new* one has to be added
        deliberately.  Every place this defect appeared was a module that had
        no business naming a definition at all.
        """
        import ast
        import pathlib

        import ggah_mod

        root = pathlib.Path(ggah_mod.__file__).parent
        offenders = {}
        for f in sorted(root.rglob("*.py")):
            if f.name in self.MAY_NAME_ONE:
                continue
            for node in ast.walk(ast.parse(f.read_text())):
                if isinstance(node, ast.Constant) and node.value in (
                        "200c", "200m", "vir", "500c"):
                    offenders.setdefault(
                        str(f.relative_to(root)), []).append(node.lineno)
        assert not offenders, (
            f"{sorted(offenders)} name a mass definition as a value.  Either "
            f"take it as an argument -- the gas sector and the binding energy "
            f"both had to -- or add the module to MAY_NAME_ONE with the "
            f"reason it is data rather than a choice.")

    def test_the_allow_list_has_no_dead_entries(self):
        """A module that stopped naming one should leave the list."""
        import ast
        import pathlib

        import ggah_mod

        root = pathlib.Path(ggah_mod.__file__).parent
        by_name = {f.name: f for f in root.rglob("*.py")}
        dead = []
        for name in self.MAY_NAME_ONE:
            f = by_name.get(name)
            if f is None:
                dead.append(f"{name} (no such module)")
                continue
            if not any(isinstance(n, ast.Constant)
                       and n.value in ("200c", "200m", "vir", "500c")
                       for n in ast.walk(ast.parse(f.read_text()))):
                dead.append(name)
        assert not dead, f"these no longer name a definition: {dead}"


class TestChangingItChangesEverythingItShould:
    """The differential check, which a hard-coded boundary cannot pass."""

    def test_the_halo_radius_moves(self, pk):
        a = np.asarray(_field(pk, "200m").r_delta)
        c = np.asarray(_field(pk, "200c").r_delta)
        assert np.all(a > c), "200m encloses more, so its radius is larger"
        assert 1.4 < float(np.mean(a / c)) < 1.6

    def test_the_gas_profile_moves(self, pk):
        """The defect this test was written for.

        ``_r_delta`` returned ``R_200c`` regardless, so these were identical.
        """
        g = G.HotGasDPM()
        p = G.DpmParams()
        r_m = np.asarray(g._r_delta(_M, 0.0, PLANCK18, "200m"))
        r_c = np.asarray(g._r_delta(_M, 0.0, PLANCK18, "200c"))
        assert not np.allclose(r_m, r_c)
        assert np.asarray(_field(pk, "200m").r_delta) == pytest.approx(
            r_m, rel=1e-10), "the gas sector and the field disagree on R_Delta"

    def test_the_binding_energy_moves(self):
        """And this is the one that feeds the feedback budget."""
        v_m = float(E.v_delta_squared(1e14, 0.0, PLANCK18, "200m"))
        v_c = float(E.v_delta_squared(1e14, 0.0, PLANCK18, "200c"))
        assert v_m < v_c, "a larger radius means a shallower potential"
        assert 0.55 < v_m / v_c < 0.75

    def test_the_gas_fraction_moves(self):
        g = G.HotGasDPM()
        p = G.DpmParams()
        # Each definition with its own concentration, as its field carries.
        from ggah_mod.halos.concentration import c_duffy08
        f_m = np.asarray(g.f_gas(_M, 0.0, PLANCK18, p, mdef="200m",
                                 conc=c_duffy08(_M, 0.0, "200m")))
        f_c = np.asarray(g.f_gas(_M, 0.0, PLANCK18, p, mdef="200c",
                                 conc=c_duffy08(_M, 0.0, "200c")))
        assert np.all(f_m > f_c), "a larger aperture encloses more gas"
