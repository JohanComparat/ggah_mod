"""Three neutrinos of unequal mass: the solve, the domains, and the refusals.

The oscillation experiments say the masses are not degenerate, and two measured
splittings plus a declared ordering fix all three from the sum alone.  What is
tested here is that the derivation is right, that each hierarchy's domain is one
interval rather than two, and that the backend which cannot carry a split says
so instead of answering.
"""
import numpy as np
import pytest

from ggah_mod.cosmology import Cosmology
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.parameters import nu_masses

#: The ladder every mass-dependent claim is checked on: zero, both floors, the
#: fiducial, and up through the range where the split still matters.  Above
#: 0.3 eV the three masses are nearly equal anyway -- the shift in Omega_nu is
#: 9.1e-8 there against 2.6e-3 at the fiducial -- so the ladder stops where the
#: effect does.
LADDER = (0.06, 0.10, 0.15, 0.30)

SPLIT = ("normal", "inverted")


class TestTheSolve:
    """The three masses, from the sum and the ordering."""

    def test_the_fiducial_triplet_is_the_published_one(self):
        """(0.00095, 0.00871, 0.05035) eV at 0.06 eV, normal ordering.

        The number the technical paper quotes, and it is a real check rather
        than a restatement: it comes from a Newton solve on
        `m + sqrt(m^2 + dm2_21) + sqrt(m^2 + dm2_31) = sum`, and nothing in the
        package stores it.
        """
        m = np.asarray(nu_masses(0.06, "normal"))
        assert m == pytest.approx([0.00094621, 0.00870605, 0.05034774], rel=1e-4)

    @pytest.mark.parametrize("hierarchy", SPLIT)
    @pytest.mark.parametrize("mnu", LADDER)
    def test_the_masses_sum_to_the_sum(self, hierarchy, mnu):
        """To machine precision, which is what makes `sum_mnu` still the input.

        Everything downstream is parameterised by the sum -- the 93.14
        convention, the emulator's box, every prior anyone has written -- so a
        solve that returned three masses adding to something else would move
        the meaning of the sampled parameter.
        """
        if mnu < C.NU_MASS_FLOOR[hierarchy]:
            pytest.skip(f"{mnu} eV is below the {hierarchy} floor")
        m = np.asarray(nu_masses(mnu, hierarchy))
        assert float(m.sum()) == pytest.approx(mnu, abs=1e-15)

    @pytest.mark.parametrize("hierarchy", list(C.NU_OFFSETS))
    def test_the_masses_come_back_ascending(self, hierarchy):
        """Ascending, so the consumers that care get what they want.

        CLASS's `m_ncdm` list, CAMB's `nu_mass_fractions` and emu_pk 2.0's
        ordered simplex all want them sorted.  In a normal ordering ascending
        is (m1, m2, m3); in an inverted one it is (m3, m1, m2), and that
        relabelling is the whole difference between two spellings of the same
        three numbers.
        """
        mnu = max(0.30, C.NU_MASS_FLOOR[hierarchy])
        m = np.asarray(nu_masses(mnu, hierarchy))
        assert np.all(np.diff(m) >= -1e-18), m

    @pytest.mark.parametrize("hierarchy", list(C.NU_OFFSETS))
    def test_zero_gives_three_massless_states(self, hierarchy):
        """Exactly zero, so the massless control is bit for bit what it was."""
        assert np.array_equal(np.asarray(nu_masses(0.0, hierarchy)), np.zeros(3))

    @pytest.mark.parametrize("hierarchy", SPLIT)
    def test_the_floor_is_where_the_lightest_state_vanishes(self, hierarchy):
        """And it is derived from the splittings, not typed beside them.

        The paper quotes 0.058 and 0.098 eV, which are the rounded literature
        values; these splittings give 0.058993 and 0.099447.  A floor written
        as a literal next to the constants it is a function of is a pair that
        drifts.
        """
        floor = C.NU_MASS_FLOOR[hierarchy]
        m = np.asarray(nu_masses(floor, hierarchy))
        assert m[0] == pytest.approx(0.0, abs=1e-12)
        a, b = C.NU_OFFSETS[hierarchy]
        assert floor == pytest.approx(np.sqrt(a) + np.sqrt(b), rel=1e-15)

    def test_the_offsets_ascend(self):
        """One character from being wrong, and silent if it were.

        Swapping the pair for an inverted ordering returns (m3, m2, m1) --
        the same three numbers in an order that no longer matches what the
        Boltzmann solvers are told the labels mean.
        """
        for h, (a, b) in C.NU_OFFSETS.items():
            assert a <= b, h

    @pytest.mark.parametrize("hierarchy", SPLIT)
    def test_the_lightest_state_absorbs_a_small_increase_at_the_floor(self, hierarchy):
        """dm1/dSum = 1 exactly there, and finite, which is what the corner needed.

        At the floor the two heavy states are fixed by the splittings and the
        lightest is zero, so the whole of a small increase in the sum goes into
        it.  Differentiating through the Newton iteration would give roughly
        this; the implicit derivative gives exactly it.
        """
        import jax
        g = jax.grad(lambda s: nu_masses(s, hierarchy)[0])(
            C.NU_MASS_FLOOR[hierarchy])
        assert float(g) == pytest.approx(1.0, rel=1e-12)


class TestEachHierarchyHasOneInterval:
    """A disconnected domain is what a sampler falls into."""

    @pytest.mark.parametrize("hierarchy", SPLIT)
    def test_below_the_floor_is_refused(self, hierarchy):
        floor = C.NU_MASS_FLOOR[hierarchy]
        with pytest.raises(ValueError, match="no .*-ordering solution"):
            Cosmology.create(sum_mnu=0.5 * floor, nu_hierarchy=hierarchy)

    @pytest.mark.parametrize("hierarchy", SPLIT)
    def test_the_message_names_the_floor_and_the_alternative(self, hierarchy):
        """The house rule: a refusal names the accepted values or the fix."""
        floor = C.NU_MASS_FLOOR[hierarchy]
        with pytest.raises(ValueError) as e:
            Cosmology.create(sum_mnu=0.5 * floor, nu_hierarchy=hierarchy)
        msg = str(e.value)
        assert f"{floor:.6f}" in msg
        assert "degenerate" in msg and "massless" in msg

    def test_the_fiducial_is_normal_only(self):
        """0.06 eV clears the normal floor and does not clear the inverted one.

        Not a quirk of this package: the oscillation data excludes an inverted
        ordering at the fiducial mass, and a refusal is the honest way to say
        so.
        """
        Cosmology.create(sum_mnu=0.06, nu_hierarchy="normal")
        with pytest.raises(ValueError):
            Cosmology.create(sum_mnu=0.06, nu_hierarchy="inverted")

    def test_massless_pins_the_sum(self):
        Cosmology.create(sum_mnu=0.0, nu_hierarchy="massless")
        with pytest.raises(ValueError, match="carries no mass"):
            Cosmology.create(sum_mnu=0.06, nu_hierarchy="massless")

    def test_degenerate_admits_everything(self):
        """Which is why it is the escape hatch the refusals point at."""
        for mnu in (0.0,) + LADDER + (0.6,):
            Cosmology.create(sum_mnu=mnu, nu_hierarchy="degenerate")

    def test_an_unknown_hierarchy_is_refused(self):
        with pytest.raises(ValueError, match="nu_hierarchy must be one of"):
            Cosmology.create(nu_hierarchy="sideways")

    def test_replace_into_the_gap_is_refused(self):
        """Validated on the result, because neither argument alone is wrong."""
        c = Cosmology.create(sum_mnu=0.30, nu_hierarchy="inverted")
        with pytest.raises(ValueError):
            c.replace(sum_mnu=0.06)

    def test_replace_with_a_tracer_is_not_refused(self):
        """The check is skipped under tracing rather than attempted.

        `jax.grad` reaches `replace` with a tracer, and a `float()` on it
        raises `ConcretizationTypeError` -- killing the gradient the
        differentiable flavour exists for.  A value gets into a trace by having
        been put in a `Cosmology` outside one, so what is skipped is the second
        check and not the first.
        """
        import jax
        c = Cosmology.create(sum_mnu=0.15)
        g = jax.grad(lambda s: c.replace(sum_mnu=s).Omega_de)(0.15)
        assert np.isfinite(float(g))


class TestWhatTheSplitMovesAndWhatItCannot:
    """The matter budget is invariant; the expansion is not."""

    @pytest.mark.parametrize("hierarchy", SPLIT)
    @pytest.mark.parametrize("mnu", LADDER)
    def test_the_matter_budget_does_not_move_one_bit(self, hierarchy, mnu):
        """`Omega_nu_matter` is linear in y, so it can only depend on the sum.

        And it is *written* that way -- from `sum_mnu` directly rather than as
        a mean over the three masses -- so it is exact rather than nearly so.
        The two agree mathematically and differ by an ulp in float64, and this
        quantity feeds `Omega_cb`, `Omega_cdm`, `f_nu` and `rho_cold`.  Writing
        it in closed form is what keeps the entire halo layer out of the blast
        radius of the split.

        A failure here means the mass vector leaked into the matter budget.
        """
        if mnu < C.NU_MASS_FLOOR[hierarchy]:
            pytest.skip(f"{mnu} eV is below the {hierarchy} floor")
        d = Cosmology.create(sum_mnu=mnu, nu_hierarchy="degenerate")
        s = Cosmology.create(sum_mnu=mnu, nu_hierarchy=hierarchy)
        for q in ("Omega_nu_matter", "Omega_cb", "Omega_cdm", "f_nu",
                  "rho_cold", "nu_y0", "Omega_nu"):
            assert float(getattr(d, q)) == float(getattr(s, q)), q

    @pytest.mark.parametrize("hierarchy", list(C.NU_OFFSETS))
    @pytest.mark.parametrize("mnu", (0.0,) + LADDER)
    def test_e_of_zero_is_still_one(self, hierarchy, mnu):
        """The identity `Omega_de` exists to preserve.

        `Omega_nu_today` and `nu_density_shape` reduce over the species with
        the same expression, so the cancellation at z = 0 is the one that
        always worked.  It survives to an ulp of 1.0 rather than exactly,
        because XLA is free to reassociate a three-element reduce differently
        at the two call sites -- which is 4000 times under the tolerance this
        package's own flatness test uses.
        """
        from ggah_mod.cosmology.background import hubble_e
        if hierarchy == "massless" and mnu != 0.0:
            pytest.skip("massless carries no mass")
        if mnu < C.NU_MASS_FLOOR[hierarchy] and mnu != 0.0:
            pytest.skip(f"{mnu} eV is below the {hierarchy} floor")
        c = Cosmology.create(sum_mnu=mnu, nu_hierarchy=hierarchy)
        assert float(hubble_e(0.0, c)) == pytest.approx(1.0, abs=1e-15)

    def test_the_expansion_does_move_and_by_the_priced_amount(self):
        """2.70e-3 on `Omega_nu_today` at the fiducial, and it falls away with mass.

        The technical paper prices the degenerate approximation at 2.7e-3 on
        the neutrino density using CLASS's exact relic energy integral, and
        since 0.9.8 this package integrates the same function at the same
        temperature: 2.698e-3.  The Komatsu fit it replaced gave 2.64e-3, 2 per
        cent of the shift low -- the fit's own error, sitting exactly where the
        lightest state is half relativistic.  By 0.30 eV the shift is 5.9e-8,
        because the three masses really are nearly equal there.
        """
        def shift(mnu):
            d = Cosmology.create(sum_mnu=mnu, nu_hierarchy="degenerate")
            n = Cosmology.create(sum_mnu=mnu, nu_hierarchy="normal")
            return float(n.Omega_nu_today) / float(d.Omega_nu_today) - 1.0
        assert shift(0.06) == pytest.approx(2.698e-3, rel=2e-3)
        assert shift(0.30) == pytest.approx(5.87e-8, rel=0.01)
        assert shift(0.06) > shift(0.15) > shift(0.30) > 0.0

    def test_the_massless_case_is_untouched(self):
        """Bit for bit, not to a tolerance.

        Three zero masses give `nu_energy_factor(0) = 1` exactly -- the sum is
        written as one plus terms that each vanish at zero mass -- and a mean
        of three ones is exactly one, so the whole neutrino term is the
        radiation of `Omega_nu_rel` and nothing else.
        """
        from ggah_mod.cosmology.background import nu_density_shape
        c = Cosmology.create(sum_mnu=0.0, nu_hierarchy="massless")
        z = np.linspace(0.0, 100.0, 51)
        assert np.array_equal(np.asarray(nu_density_shape(z, c)), np.ones_like(z))
        assert float(c.Omega_nu_today) == float(c.Omega_nu_rel)


class TestTheDifferentiablePathCarriesTheSplit:
    """emu_pk 1.x was trained on degenerate species and refused a split.

    **2.0.0 carries it.**  Its box gained ``nu_r1``/``nu_r2``, the two lightest
    masses as fractions of the sum, and the shipped checkpoint is trained on
    them -- so the refusal these tests were written for no longer fires and
    what they check instead is that the answer *moves*.  A network handed a
    split it ignored would return the degenerate spectrum and look exact.
    """

    @pytest.mark.parametrize("hierarchy", SPLIT)
    def test_a_split_ordering_changes_the_spectrum(self, hierarchy):
        from conftest import constructible
        pk = constructible("emu_pk")
        k = np.logspace(-3, 0, 12)
        deg = np.asarray(pk.pk(k, 0.0, Cosmology.create(
            sum_mnu=0.30, nu_hierarchy="degenerate")))
        got = np.asarray(pk.pk(k, 0.0, Cosmology.create(
            sum_mnu=0.30, nu_hierarchy=hierarchy)))
        assert np.all(got > 0.0)
        assert np.max(np.abs(got / deg - 1.0)) > 1e-5, (
            f"{hierarchy} returned the degenerate spectrum, which is what an "
            f"ignored input looks like")

    @pytest.mark.parametrize("hierarchy", ["degenerate", "massless"])
    def test_the_parameterisation_it_was_trained_on_still_works(self, hierarchy):
        from conftest import constructible
        pk = constructible("emu_pk")
        mnu = 0.0 if hierarchy == "massless" else 0.30
        c = Cosmology.create(sum_mnu=mnu, nu_hierarchy=hierarchy)
        assert np.all(np.isfinite(np.asarray(pk.pk(np.array([0.1]), 0.0, c))))

    def test_the_capability_is_declared_not_sniffed(self):
        """A backend that ignored the split would answer, and look exact."""
        from ggah_mod.cosmology.power import PK_BACKENDS
        for name, cls in PK_BACKENDS.items():
            assert isinstance(getattr(cls, "supports_nondegenerate_nu"), bool), name
        assert PK_BACKENDS["emu_pk"].supports_nondegenerate_nu is True
        assert PK_BACKENDS["class"].supports_nondegenerate_nu is True
        assert PK_BACKENDS["camb"].supports_nondegenerate_nu is True

    def test_the_refusal_still_exists_for_a_backend_that_cannot(self):
        """No shipped backend exercises it now, so a stub keeps it reachable.
        The message is what a future flat-box emulator would have to say, and a
        branch nothing takes is a branch that rots."""
        from ggah_mod.cosmology.power import _check_nu_split

        class Degenerate:
            name = "stub"
            supports_nondegenerate_nu = False

        c = Cosmology.create(sum_mnu=0.30, nu_hierarchy="normal")
        _check_nu_split(Degenerate(), Cosmology.create(
            sum_mnu=0.30, nu_hierarchy="degenerate"))          # fine
        with pytest.raises(ValueError) as e:
            _check_nu_split(Degenerate(), c)
        msg = str(e.value)
        assert "make_pk('class')" in msg and "degenerate" in msg
