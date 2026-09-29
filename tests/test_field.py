r"""Verification: the halo field owns the grids, and adds no physics.

Two claims, and the second is the one that could go wrong silently.

**It owns the grids.**  Five classes in the predecessor each built their own
``logspace(10, 16, N)``.  Here there is one, it lives on the field, and a sector
that wanted its own would have to be written to ignore the one it was handed.

**It adds no physics.**  ``make_field`` is a *repackaging* of the loose layer-2
functions.  If it quietly did something else -- a different mass definition, a
different density, a growth factor slipped in -- every number downstream would
move and nothing would say so.  So the test is equality with the hand-assembled
chain, element for element, not agreement to a tolerance.
"""
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.halos as H
from ggah_mod import ACCURATE, DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.cosmology.growth import growth_factor
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.calibration import MismatchWarning
from ggah_mod.halos.field import (
    DEFAULT_CM_MODEL, HaloField, make_field, make_fields,
)
from conftest import AnalyticPk


@pytest.fixture(scope="module")
def pk():
    """The default concentration relation needs a redshift-aware backend.

    ``diemer19`` reads the growth *rate* off the spectrum, so it refuses a
    backend that declares ``has_native_z = False`` -- which the cheap analytic
    stub does.  Most of this module drives the *default* field, so the default
    fixture has to be a real one; `analytic_pk` below is kept for the tests
    that are specifically about refusing a backend that cannot answer.
    """
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def analytic_pk():
    """A cheap spectrum with no redshift dependence, for the refusal tests."""
    return AnalyticPk()


@pytest.fixture(scope="module")
def native_z_pk():
    """A backend that carries its own redshift dependence.

    Needed by anything that reads a growth factor off the spectrum, which in
    this package is the only way to get one.
    """
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0)


class TestItIsARepackagingAndNothingMore:
    """Every table, against the chain written out by hand."""

    def test_tables_are_identical_to_the_hand_assembled_chain(self, field, pk):
        m, k, z = field.m, field.k, 0.0
        p = pk.pk_cb(k, z, PLANCK18)
        s = H.sigma_of_mass(m, k, p, PLANCK18.rho_cold)
        d = H.dln_sigma_dln_mass(m, k, p, PLANCK18.rho_cold)
        # The one thing `make_field` supplies that the loose functions do not:
        # the boundary, converted to the mean-density convention the published
        # fits are indexed by.  `tinker08` and `tinker10` both take a Delta and
        # both default to 200, so a hand-assembled chain that omits it silently
        # computes a 200m halo whatever the field says -- which is what
        # `make_field` did until `TestTheMassDefinitionReachesTheAbundance` was
        # written.  Passing it here is not a concession: it is the statement
        # that this is the only step the repackaging adds.
        delta_m = H.MassDef.from_string(field.mdef).delta_mean(z, PLANCK18)

        # What the flavour's mass function actually takes, mirrored rather than
        # imported: `tinker08` is stated as a function of Delta and gets one;
        # `tinker08_csst` is fitted at 200m alone, so it takes no Delta and
        # takes a *cosmology* instead.  Using `field._delta_kw` here would make
        # this step vacuous, since it is the step under test.
        from ggah_mod.halos.field import _DELTA_ARG
        from ggah_mod.halos.mass_function import (
            COSMOLOGY_DEPENDENT_MULTIPLICITY)

        _hmf_kw = {}
        if field.hmf_model in _DELTA_ARG:
            _hmf_kw[_DELTA_ARG[field.hmf_model]] = delta_m
        if field.hmf_model in COSMOLOGY_DEPENDENT_MULTIPLICITY:
            _hmf_kw["cosmo"] = PLANCK18

        ref = {
            "sigma": s,
            "dlns": d,
            "nu": H.DELTA_C / s,
            # The flavour's own mass function, and the cosmology it needs.
            # `tinker08_csst` is recalibrated per cosmology and so takes one;
            # hard-coding `tinker08` here compared the field against a
            # different model and called the difference a repackaging bug.
            "dndm": H.dndm(m, s, d, PLANCK18.rho_cold,
                           model=field.hmf_model, z=z, **_hmf_kw),
            "bias": H.make_bias("tinker10")(s, delta_m),
            # the field's OWN mass definition, not a hard-coded one: DIFFERENTIABLE is
            # 200c and ACCURATE is 200m, and reading the wrong one is a 32%
            # error in r_delta that no other assertion here would catch.
            "r_delta": H.MassDef.from_string(field.mdef).radius(m, z, PLANCK18),
            # Both read from the flavour, never one from it and one from here.
            # This line used to hard-code `c_dutton14` while reading the mass
            # definition from the backend -- the exact mismatch the comment
            # above it warned against -- and it survived until the flavour's
            # relation changed underneath it.  The assertion below pins the
            # name, so the next such move fails loudly here rather than
            # comparing the field against a different model.
            "conc": H.c_bhattacharya13(
                s, growth_factor(z, PLANCK18, pk), field.mdef),
        }
        assert field.hmf_model == "tinker08_csst", (
            f"the flavour now declares {field.hmf_model!r}; check this "
            f"reference chain still passes what that model needs")
        assert field.cm_model == "bhattacharya13", (
            f"the flavour now declares {field.cm_model!r}; this reference "
            f"chain builds bhattacharya13 and would be comparing two models")
        for name, want in ref.items():
            np.testing.assert_array_equal(
                np.asarray(getattr(field, name)), np.asarray(want),
                err_msg=f"make_field changed {name}")

    def test_scale_radius_is_the_radius_over_the_concentration(self, field):
        np.testing.assert_array_equal(
            np.asarray(field.r_s),
            np.asarray(field.r_delta) / np.asarray(field.conc))

    def test_it_takes_the_mass_definition_from_the_backend(self, pk):
        for b in (DIFFERENTIABLE, ACCURATE):
            assert make_field(PLANCK18, b, pk).mdef == b.mdef
        # Both shipped flavours are 200c now -- that is the point of
        # `TestTheCalibrationTablesAreRead` -- so a differing backend has to be
        # built rather than found, or the two assertions above are satisfied by
        # a constant and prove nothing.
        other = DIFFERENTIABLE.with_(mdef="vir", cm_model="dutton14",
                                     hmf_model="tinker08")
        assert other.mdef != DIFFERENTIABLE.mdef
        assert make_field(PLANCK18, other, pk).mdef == "vir"

    def test_the_two_backends_give_different_grids(self, pk):
        assert make_field(PLANCK18, DIFFERENTIABLE, pk).n_m == DIFFERENTIABLE.n_m
        assert make_field(PLANCK18, ACCURATE, pk).n_m == ACCURATE.n_m


class TestTheTablesArePhysical:
    """Each rung against an invariant it must satisfy on its own."""

    def test_sigma_falls_and_peak_height_crosses_unity(self, field):
        s, nu = np.asarray(field.sigma), np.asarray(field.nu)
        assert np.all(np.diff(s) < 0)
        assert np.all(np.diff(nu) > 0)
        assert nu[0] < 1.0 < nu[-1]

    def test_mass_function_is_positive_and_steep(self, field):
        n = np.asarray(field.dndm)
        assert np.all(n > 0) and np.all(np.diff(n) < 0)

    def test_bias_rises_through_unity(self, field):
        b = np.asarray(field.bias)
        assert np.all(np.diff(b) > 0)
        assert b[0] < 1.0 < b[-1]

    def test_concentration_is_physical_and_turns_up_where_it_should(self, field):
        """`diemer19` is not monotone, and that is the model, not a defect.

        Concentration falls with mass through the range fits are usually quoted
        over, then **rises again** at high peak height -- the upturn Diemer &
        Kravtsov (2015) report for the rarest halos, which is why the relation
        is parameterised by nu and n_eff rather than by mass.  The field's grid
        runs to 1e16, so it straddles the turn; asserting a monotone decline
        over it would be asserting the fit away.
        """
        c, nu = np.asarray(field.conc), np.asarray(field.nu)
        assert np.all((c > 1.0) & (c < 30.0))
        turn = int(np.argmin(c))
        assert np.all(np.diff(c[:turn]) < 0)          # falls, then
        assert np.all(np.diff(c[turn:]) > 0)          # rises: one turn, not noise
        assert nu[turn] > 1.5                         # and it is in the rare tail

    def test_radius_scales_as_the_cube_root_of_mass(self, field):
        lm = np.log(np.asarray(field.m))
        lr = np.log(np.asarray(field.r_delta))
        assert np.polyfit(lm, lr, 1)[0] == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_nfw_transform_tends_to_one_at_large_scales(self, field):
        u0 = np.asarray(field.u_nfw(jnp.asarray([1e-5])))
        np.testing.assert_allclose(u0[0], 1.0, atol=1e-9)

    def test_the_two_densities_stay_distinct(self, field):
        """`rho_cold` built sigma; `rho_matter` is what will normalise the
        matter tracer.  They differ by 0.46% at the minimal neutrino mass, and
        the field must not collapse them into one attribute."""
        assert field.rho_cold < field.rho_matter
        assert field.rho_matter / field.rho_cold - 1.0 > 4e-3


class TestMassIntegrals:
    """One implementation of `int f dM`, in `ln M`, on the field's own grid."""

    def test_integrate_matches_an_explicit_trapezoid(self, field):
        f = field.dndm * field.m
        got = float(field.integrate(f))
        want = float(jnp.trapezoid(f * field.m, jnp.log(field.m)))
        assert got == pytest.approx(want, rel=1e-14)

    def test_effective_bias_of_every_halo_is_of_order_unity(self, field):
        occ = jnp.ones_like(field.m)
        assert 0.4 < float(field.effective_bias(occ)) < 3.0

    def test_effective_mass_lies_inside_the_grid(self, field):
        occ = jnp.ones_like(field.m)
        m_eff = float(field.effective_mass(occ))
        assert float(field.m[0]) < m_eff < float(field.m[-1])

    def test_a_constant_occupation_cancels_out_of_the_bias(self, field):
        """b_eff is a ratio, so scaling the occupation must not move it."""
        one, ten = jnp.ones_like(field.m), 10.0 * jnp.ones_like(field.m)
        assert float(field.effective_bias(one)) == pytest.approx(
            float(field.effective_bias(ten)), rel=1e-14)

    def test_number_density_is_linear_in_the_occupation(self, field):
        one = jnp.ones_like(field.m)
        assert float(field.number_density(3.0 * one)) == pytest.approx(
            3.0 * float(field.number_density(one)), rel=1e-14)


class TestItIsAPytree:
    """Leaves are traced; model names are not, and could not be."""

    def test_every_table_is_a_leaf(self, field):
        leaves, aux = field.tree_flatten()
        assert len(leaves) == len(HaloField._LEAVES)
        assert aux == (field.mdef, field.hmf_model, field.bias_model,
                       field.cm_model)

    def test_the_model_names_travel_in_the_treedef(self, field):
        """A string among the leaves would make `jax.grad` see it as a value.
        In the treedef it is a structural impossibility instead."""
        for leaf in jax.tree_util.tree_leaves(field):
            assert not isinstance(leaf, str)

    def test_roundtrip(self, field):
        leaves, aux = field.tree_flatten()
        back = HaloField.tree_unflatten(aux, leaves)
        np.testing.assert_array_equal(np.asarray(back.dndm),
                                      np.asarray(field.dndm))
        assert back.mdef == field.mdef

    def test_unflatten_does_not_validate(self):
        """It runs inside a trace, so it must not branch on values.  A bogus
        model name has to survive unflattening and be caught by `create`."""
        HaloField.tree_unflatten(("200m", "nonsense", "tinker10", "diemer19"),
                                 (jnp.ones(3),) * 13 + (PLANCK18,))
        with pytest.raises(ValueError, match="unknown mass function"):
            HaloField.create(m=jnp.ones(3), k=jnp.ones(3), z=0.0,
                             sigma=jnp.ones(3), dlns=jnp.ones(3),
                             nu=jnp.ones(3), dndm=jnp.ones(3),
                             bias=jnp.ones(3), conc=jnp.ones(3),
                             r_delta=jnp.ones(3), r_s=jnp.ones(3),
                             pk_cb=jnp.ones(3), pk_lin=jnp.ones(3),
                             cosmo=PLANCK18, hmf_model="nonsense")


class TestItIsTraceable:
    def test_jit_over_the_whole_construction(self, pk):
        f = jax.jit(lambda c: make_field(c, DIFFERENTIABLE, pk, z=0.0)
                    .effective_bias(jnp.ones(DIFFERENTIABLE.n_m)))
        assert np.isfinite(float(f(PLANCK18)))

    def test_jit_is_value_identical(self, pk, field):
        jitted = jax.jit(lambda c: make_field(c, DIFFERENTIABLE, pk, z=0.0))(PLANCK18)
        for key in ("sigma", "dndm", "bias", "conc", "r_delta"):
            np.testing.assert_allclose(np.asarray(getattr(jitted, key)),
                                       np.asarray(getattr(field, key)),
                                       rtol=1e-13, err_msg=key)

    @pytest.mark.x64
    @pytest.mark.parametrize("table", ["sigma", "dndm", "bias", "conc",
                                       "r_delta", "nu"])
    def test_gradient_reaches_every_table(self, pk, table):
        """With respect to `Omega_m`, deliberately.

        It is the parameter that reaches the halo *geometry*.  An amplitude
        parameter leaves `r_delta` untouched, which is exactly how the
        predecessor's numpy `einasto_uk` sat on the traced path undetected.
        """
        # No model override any more, and that is the point: the flavour's own
        # set is now cosmology-carrying end to end, so this exercises what a
        # forecast would actually run.  It used to override `cm_model` to
        # `diemer19` because the flavour declared `dutton14`, an empirical
        # power law that takes no cosmology and *could not* respond to one --
        # the conc row was a structural zero for a reason that belonged to the
        # fit rather than to this code.
        #
        # `z = 0.5` survives, for its own reason.  A *critical*-overdensity
        # radius at z = 0 does not depend on Omega_m at all, because
        # E(0) = 1 identically, so the r_delta row would be a true zero and the
        # test would assert the wrong thing about a correct answer.  The
        # default is now a *mean*-density boundary, where that argument does not
        # bite -- but `bhattacharya13` reaches the cosmology through D(z), and
        # D(0) = 1 identically, so z = 0 would flatten the conc row instead.
        # One redshift away from zero and both rows are live.
        def f(om):
            fl = make_field(PLANCK18.replace(Omega_m=om), DIFFERENTIABLE, pk,
                            z=0.5)
            return jnp.log(jnp.sum(jnp.abs(getattr(fl, table))))

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0, table
        assert ad == pytest.approx(fd, rel=1e-4), table

    @pytest.mark.x64
    def test_gradient_with_respect_to_redshift(self, native_z_pk):
        """z is a leaf, not a Python float forced concrete.

        The limitation hides behind the obvious test: a gradient in a
        *cosmological* parameter still works when z is frozen.

        **z = 0.37, and the odd number is load-bearing.**  The distilled
        neutrino-ratio table is interpolated monotone-cubic along its f_nu axis
        but only *linearly* along its z axis
        (`cosmology/nu_ratio.py:128`), whose nodes are
        `[0, 0.25, 0.5, 1, 2, 3, 5]`.  At a node the interpolant is C0, so
        autodiff returns one side's slope while a central difference straddles
        the kink and averages both.  Measured here, step-independently:
        2.6e-4 at z = 0.5 and 1.0e-3 at z = 1.0, against 1e-10 off-node.  That
        is a layer-1 property, not a defect in this module -- and it is the
        same class as the f_nu-axis kink layer 1 already fixed, on the axis it
        did not.  Testing on a node would blame this module for it.
        """
        f = lambda zz: jnp.log(jnp.sum(
            make_field(PLANCK18, DIFFERENTIABLE, native_z_pk, z=zz).dndm))
        ad = float(jax.grad(f)(0.37))
        fd = float((f(0.37 + 1e-6) - f(0.37 - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)


class TestDeclarationsAndRefusals:
    def test_it_refuses_to_invent_a_spectrum(self):
        """Which spectrum produced a number is not left implicit."""
        with pytest.raises(ValueError, match="linear-P.k. backend"):
            make_field(PLANCK18)

    @pytest.mark.parametrize("kw,match", [
        ({"hmf_model": "nope"}, "unknown mass function"),
        ({"bias_model": "nope"}, "unknown bias model"),
        ({"cm_model": "nope"}, "unknown concentration"),
        ({"mdef": "200q"}, "cannot parse mass definition"),
    ])
    def test_unknown_model_names_raise(self, pk, kw, match):
        with pytest.raises(ValueError, match=match):
            make_field(PLANCK18, DIFFERENTIABLE, pk, **kw)

    def test_the_default_concentration_responds_to_cosmology(self, pk):
        """The empirical power laws cannot, by construction.  Defaulting to one
        would make `dc/dtheta` identically zero for an uninteresting reason."""
        assert DEFAULT_CM_MODEL in H.concentration.PEAK_HEIGHT_MODELS
        kw = dict(cm_model=DEFAULT_CM_MODEL)
        a = make_field(PLANCK18, DIFFERENTIABLE, pk, **kw).conc
        b = make_field(PLANCK18.replace(Omega_m=0.35), DIFFERENTIABLE, pk, **kw).conc
        assert not np.allclose(np.asarray(a), np.asarray(b))

    def test_the_default_concentration_is_no_longer_frozen(self, pk):
        """The inverse of what this test used to assert, and the reason matters.

        It read: ``DIFFERENTIABLE.cm_model`` is ``dutton14``, an empirical power
        law fitted at one cosmology with that cosmology baked into its
        coefficients, so a forecast run through this flavour gets a **frozen**
        c(M) -- ``dc/dtheta`` identically zero, and a Fisher matrix built on it
        reporting more confidence than the model has.  That was true, and it
        was asserted rather than left in a reader's memory.

        The default is now ``bhattacharya13``, which is parameterised by peak
        height and the growth factor and therefore does respond.  So the
        assertion inverts: the concentration must *move* when the cosmology
        does.  Keeping the test and flipping it is deliberate -- a deleted test
        would leave no record that the flavour ever had this defect, and the
        parity budget's concentration row is only interpretable against it.
        """
        from ggah_mod.halos.concentration import CM_CALIBRATION

        assert CM_CALIBRATION[DIFFERENTIABLE.cm_model][-1] is True, \
            "the default flavour has gone back to a cosmology-blind relation"
        a = np.asarray(make_field(PLANCK18, DIFFERENTIABLE, pk).conc)
        b = np.asarray(make_field(PLANCK18.replace(Omega_m=0.35),
                                  DIFFERENTIABLE, pk).conc)
        assert not np.allclose(a, b), "c(M) did not respond to Omega_m"
        # And it responds by a physically sensible amount rather than by noise.
        assert 0.001 < np.max(np.abs(a / b - 1.0)) < 0.5

    def test_the_mass_definition_and_the_relation_are_a_matched_pair(self, pk):
        """Each fit returns the concentration for the Delta it was calibrated at.

        Overriding one without the other builds a halo whose radius and
        concentration come from different definitions -- which is not a
        tolerance, it is a different halo.  `dutton14` used to be the only one
        that said so, because it is the only relation that takes an `mdef`
        argument at all; the peak-height relations could not speak for
        themselves.  Now the table speaks for all of them, so the message comes
        from `CM_CALIBRATION` rather than from inside one fit.
        """
        with pytest.raises(ValueError, match="was calibrated in"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, mdef="200m", cm_model="dutton14")
        with pytest.raises(ValueError, match="was calibrated in"):
            # `diemer19` takes no mdef and so could not refuse this itself.
            make_field(PLANCK18, DIFFERENTIABLE, pk, mdef="200m", cm_model="diemer19")
        for b in (DIFFERENTIABLE, ACCURATE):
            make_field(PLANCK18, b, pk)                        # the declared pair

    def test_the_backend_concentration_field_is_honoured(self, pk):
        """`Backend.cm_model` exists to be read, and layer 3 is its first reader.

        That is why it was wrong until this module arrived: ACCURATE asked for
        `diemer19` and TRACEABLE_CM offered `diemer19_jax`, and the registry
        has neither.  Nothing consumed the field, so nothing noticed.  Honour
        it, and assert the names resolve -- a silent fallback here would put
        the same class of defect straight back.
        """
        for b in (DIFFERENTIABLE, ACCURATE):
            assert b.cm_model in H.CONCENTRATION, b.name
            assert make_field(PLANCK18, b, pk).cm_model == b.cm_model

    def test_an_explicit_argument_overrides_the_backend(self, pk):
        assert make_field(PLANCK18, ACCURATE, pk,
                          cm_model="duffy08").cm_model == "duffy08"

    def test_bhattacharya_is_dispatched_with_a_growth_factor(self, native_z_pk):
        """The two peak-height relations take different second arguments --
        a slope (negative) and a growth factor (positive).  Swapping them gives
        NaN, because the growth is raised to a fractional power."""
        c = make_field(PLANCK18, DIFFERENTIABLE, native_z_pk,
                       cm_model="bhattacharya13").conc
        assert np.all(np.isfinite(np.asarray(c)))
        assert np.all(np.asarray(c) > 1.0)

    @pytest.mark.parametrize("name", ["bhattacharya13", "diemer19"])
    def test_a_growth_parameterised_relation_refuses_a_backend_with_no_z(
            self, analytic_pk, name):
        """The dangerous direction is a plausible number, not a raise.

        `bhattacharya13` needs D(z) and `diemer19` needs dlnD/dlna.  A backend
        with no redshift dependence has neither to offer.  Substituting a
        fitting formula there is precisely the shortcut layer 1 removed, so both
        must raise rather than quietly supply one.
        """
        from ggah_mod.halos.concentration import CM_CALIBRATION

        assert analytic_pk.has_native_z is False
        # At a definition the relation covers, so that the *growth* refusal is
        # what fires.  At the flavour's 200m the calibration guard would raise
        # first for `diemer19`, and the test would pass while checking nothing
        # it means to check.
        mdef = CM_CALIBRATION[name][1].replace("SO ", "").split("/")[0].strip()
        with pytest.raises(ValueError, match="growth"):
            make_field(PLANCK18, DIFFERENTIABLE, analytic_pk, mdef=mdef,
                       cm_model=name, hmf_model="tinker08")

    def test_seppi21_needs_no_native_z_backend(self, analytic_pk):
        """The positive counterpart of the refusal above, and why the third
        signature exists.

        `bhattacharya13` and `diemer19` refuse a spectrum with no redshift
        dependence because they need a D(z) or a dlnD/dlna off it.  `seppi21`
        is tabulated in z directly and asks for neither, so refusing it here
        would be a guard copied rather than reasoned -- and it would make the
        one relation that works on a cheap stub unusable on one.
        """
        assert analytic_pk.has_native_z is False
        c = np.asarray(make_field(PLANCK18, DIFFERENTIABLE, analytic_pk,
                                  mdef="vir", cm_model="seppi21",
                                  hmf_model="tinker08").conc)
        assert np.all(np.isfinite(c)) and np.all(c > 1.0)

    @pytest.mark.parametrize("name", sorted(H.CONCENTRATION))
    def test_every_concentration_relation_can_build_a_field(self, native_z_pk, name):
        """If a relation is added to the registry, `_concentration` must know
        which family it is in -- otherwise it is called with the wrong second
        argument and returns NaN rather than raising.

        Each is built at a definition **it was calibrated in**, read from
        ``CM_CALIBRATION`` rather than assumed: the flavour default is 200m and
        three of the five relations do not cover it, so a fixed definition here
        would be testing the calibration guard rather than the dispatch.
        """
        from ggah_mod.halos.concentration import CM_CALIBRATION

        # "200c/vir/200m" -> "200c".  The description is the source of truth
        # and its first entry is always a definition the fit covers.
        mdef = CM_CALIBRATION[name][1].replace("SO ", "").split("/")[0].strip()
        # `tinker08` rather than the flavour default: it is stated as a
        # function of Delta and so covers every definition, which keeps this
        # test about the concentration dispatch instead of about whether the
        # default mass function happens to share a definition with the
        # relation under test.
        c = np.asarray(make_field(PLANCK18, DIFFERENTIABLE, native_z_pk,
                                  mdef=mdef, cm_model=name,
                                  hmf_model="tinker08").conc)
        assert np.all(np.isfinite(c)) and np.all(c > 0.0), f"{name} at {mdef}"


class TestTheMassIntegralServesLayerFour:
    r"""One implementation, two shapes, and the measure it can be written as.

    Layer 4's one-halo integrand is ``(Nk, NM)`` -- the mass integral is taken
    at every wavenumber at once.  That is an ``axis`` argument rather than a
    second function, because a second ``jnp.trapezoid`` written at the call site
    is what ``tests/test_sector_coherence.py`` looks for, and the ``dM`` versus
    ``dlnM`` choice would then be made twice.
    """

    def test_a_two_dimensional_integrand_reduces_over_mass(self, field):
        g = field.dndm * field.bias
        stacked = jnp.stack([g, 2.0 * g, 3.0 * g])
        got = field.integrate(stacked, axis=-1)
        one = float(field.integrate(g))
        assert got.shape == (3,)
        assert np.allclose(np.asarray(got), one * np.array([1.0, 2.0, 3.0]),
                           rtol=1e-13)

    def test_the_quadrature_measure_reproduces_the_integral(self, field):
        r"""``int f dM == sum(f * w)``.

        Derived from ``ln_m`` rather than restated, so the two cannot disagree.
        :func:`~ggah_mod.halos.beyond_linear_bias.correction_2h_gg` contracts
        its weights against a tabulated beta^NL with no measure of its own, so
        its documented ``dndm * N_tot * b / n_bar`` is missing exactly this
        ``dM`` -- which is the adapter layer 4 needs and the docstring denied.
        """
        g = field.dndm * field.bias * field.m
        got = float(jnp.sum(g * field.quadrature_measure()))
        assert got == pytest.approx(float(field.integrate(g)), rel=1e-13)

    def test_the_grid_sizes_are_read_from_the_last_axis(self, field):
        """Layer 5 stacks a field over redshift, which puts Nz on the leading
        axis of every leaf.  Read from the front, `n_m` would return the number
        of redshifts -- silently, since it stays a plausible integer."""
        stacked = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x, x]) if jnp.ndim(x) >= 1 else x, field)
        assert stacked.n_m == field.n_m
        assert stacked.n_k == field.n_k


class TestTheCalibrationTablesAreRead:
    """`CALIBRATION` and `CM_CALIBRATION` were documentation until they weren't.

    Both tables recorded, per fit, the definition it was calibrated in and the
    redshift range it was fitted over -- and no module read either.  A table
    nobody consults cannot prevent the error it describes, and the benchmark
    priced one such pairing at an 81 per cent apparent discrepancy that was not
    a disagreement at all.
    """

    def test_the_shipped_flavours_are_internally_consistent(self, pk):
        """The check's first catch was the package's own default.

        `ACCURATE` declared `mdef="200m"` while asking for `diemer19`, which is
        calibrated in 200c and in nothing else.  That is the whole argument for
        reading the tables: the clash had been recorded, in the repository, for
        as long as the relation had been the default.
        """
        for b in (ACCURATE, DIFFERENTIABLE):
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # no extrapolation either
                make_field(PLANCK18, b, pk, z=0.0)

    def test_a_friends_of_friends_fit_matches_no_spherical_definition(self, pk):
        """FoF is not an overdensity, so *every* mdef is a mismatch for it."""
        with pytest.raises(ValueError, match="friends-of-friends"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, hmf_model="jenkins01")

    def test_a_delta_parameterised_fit_is_unrestricted(self, pk):
        """`tinker08` is published as a function of Delta, so it travels.

        Paired with `duffy08`, which covers all three of these definitions, the
        chain is consistent at each of them and the check stays silent.
        """
        from ggah_mod.halos.calibration import calibrated_mass_defs
        from ggah_mod.halos.mass_function import CALIBRATION
        assert calibrated_mass_defs(CALIBRATION["tinker08"][0]) is None
        for mdef in ("200c", "200m", "vir"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                make_field(PLANCK18, DIFFERENTIABLE, pk, mdef=mdef,
                           hmf_model="tinker08", cm_model="duffy08")

    def test_a_redshift_outside_the_fitted_range_warns_rather_than_raises(self, pk):
        """An extrapolation of a smooth fit is not a category error.

        `tinker08` was fitted to z = 2.5.  Refusing z = 3 would leave the
        package unable to say what it thinks happens there, which is a worse
        failure than saying it loudly.
        """
        with pytest.warns(MismatchWarning, match="extrapolation"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, z=3.0)

    def test_the_policy_is_an_argument_and_not_a_default_to_discover(self, pk):
        """`warn` downgrades the refusal; `off` disables it."""
        with pytest.warns(MismatchWarning, match="was calibrated in"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, mdef="200m",
                       cm_model="diemer19", calibration="warn")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_field(PLANCK18, DIFFERENTIABLE, pk, mdef="200m",
                       cm_model="diemer19", calibration="off")
        with pytest.raises(ValueError, match="calibration must be one of"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, calibration="lenient")

    def test_an_unmatched_bias_warns_and_a_matched_one_does_not(self, pk):
        """The pairing is a warning, not a refusal.

        Most published mass functions have no matching bias at all, and pairing
        one with `tinker10` is the usual thing to do -- so refusing would refuse
        ordinary practice.  But it costs about ten per cent in the closure
        integral, which a fitted galaxy bias absorbs rather than reveals, so it
        should not be silent either.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make_field(PLANCK18, DIFFERENTIABLE, pk,
                       hmf_model="tinker08", bias_model="tinker10")
        with pytest.warns(MismatchWarning, match="peak-background split"):
            make_field(PLANCK18, DIFFERENTIABLE, pk,
                       hmf_model="tinker08", bias_model="sheth01")

    def test_a_fit_with_no_published_partner_is_silent(self, pk):
        """`None` from `matched_bias_for` is an answer, not a failure: there is
        nothing for the choice to be inconsistent with."""
        from ggah_mod.halos.calibration import check_pairing
        from ggah_mod.halos.linear_bias import matched_bias_for
        assert matched_bias_for("rodriguezpuebla16") is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_pairing("rodriguezpuebla16", "tinker10")

    @pytest.mark.parametrize("name", sorted(H.CONCENTRATION))
    def test_every_cm_entry_parses_to_a_definition_set(self, name):
        """A relation added with an unparseable definition string would be
        silently unrestricted, which is the failure mode this replaces."""
        from ggah_mod.halos.calibration import calibrated_mass_defs
        from ggah_mod.halos.concentration import CM_CALIBRATION
        allowed = calibrated_mass_defs(CM_CALIBRATION[name][1])
        assert allowed is None or len(allowed) > 0

    @pytest.mark.parametrize("name", sorted(H.MULTIPLICITY))
    def test_every_hmf_entry_parses(self, name):
        from ggah_mod.halos.calibration import calibrated_mass_defs
        from ggah_mod.halos.mass_function import CALIBRATION
        calibrated_mass_defs(CALIBRATION[name][0])      # must not raise


class TestTheKGridMustSupportTheMasses:
    r"""A narrow `k` grid does not fail, it under-counts.

    `sigma(M)` integrates over exactly the grid it is handed.  Narrow it and
    the integral returns a smaller number; narrow it far enough and it returns
    zero, giving `nu = inf` and a mass function of NaN -- with no exception
    anywhere along the way.
    """

    def test_the_shipped_grids_clear_the_thresholds_with_margin(self):
        from ggah_mod.halos.variance import (K_MAX_R_MIN, K_MIN_R_MAX,
                                             lagrangian_radius)
        for b in (ACCURATE, DIFFERENTIABLE):
            r = np.asarray(lagrangian_radius(
                np.array([b.m_min, b.m_max]), float(PLANCK18.rho_cold)))
            assert b.k_max * r.min() > 3 * K_MAX_R_MIN, b.name
            assert b.k_min * r.max() < K_MIN_R_MAX / 3, b.name

    def test_truncating_the_top_is_refused_by_name(self, pk):
        k = jnp.logspace(-4, 0.0, 256)          # k_max = 1 h/Mpc
        with pytest.raises(ValueError, match="k_max"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, k=k)

    def test_truncating_the_bottom_is_refused_by_name(self, pk):
        k = jnp.logspace(-1, jnp.log10(200.0), 256)     # k_min = 0.1 h/Mpc
        with pytest.raises(ValueError, match="k_min"):
            make_field(PLANCK18, DIFFERENTIABLE, pk, k=k)

    def test_the_number_it_protects_actually_moves(self, pk):
        """The threshold is measured, not asserted: at `k_max R = 10` the
        variance is short by ~1e-3, and it is 20 per cent short by `k_max R = 2`.
        """
        from ggah_mod.halos.variance import lagrangian_radius, sigma_of_mass
        m = jnp.array([1e10])
        r = float(lagrangian_radius(m, float(PLANCK18.rho_cold))[0])
        ref = jnp.logspace(-4, jnp.log10(200.0), 2048)
        s_ref = float(sigma_of_mass(m, ref, pk.pk_cb(ref, 0.0, PLANCK18),
                                    PLANCK18.rho_cold)[0])
        got = {}
        for x in (2.0, 10.0):
            kk = jnp.logspace(-4, jnp.log10(x / r), 2048)
            got[x] = float(sigma_of_mass(m, kk, pk.pk_cb(kk, 0.0, PLANCK18),
                                         PLANCK18.rho_cold)[0]) / s_ref - 1.0
        assert abs(got[2.0]) > 0.02          # unmistakable
        assert abs(got[10.0]) < 0.01         # the threshold is not punitive
        assert abs(got[10.0]) < abs(got[2.0])

    def test_a_traced_grid_is_left_to_the_caller(self, pk):
        """Under `jit` the grid cannot be concretised without forcing it, so
        the check steps aside rather than breaking the trace."""
        from ggah_mod.halos.variance import check_k_support
        f = jax.jit(lambda k: check_k_support(
            jnp.array([1e10, 1e16]), k, float(PLANCK18.rho_cold)) or k.sum())
        assert np.isfinite(float(f(jnp.logspace(-4, 0.0, 16))))


class TestTheMassDefinitionReachesTheAbundance:
    r"""A field declaring 200c must not compute a 200m mass function.

    `make_field` took an `mdef`, built $r_\Delta$ from it, recorded it on the
    field -- and did not pass it to either the multiplicity function or the
    bias, both of which take one.  So a field declaring 200c had 200c radii, a
    200m abundance and a 200m bias.  Nothing showed it: the mass function was
    smooth, positive, correctly shaped, and for a different halo.

    It surfaced because the paper's peak-background-split table did the
    conversion by hand in its own script, and the same pairing then gave two
    different numbers depending on which code computed it.
    """

    @staticmethod
    def _fields(pk, hmf="tinker08"):
        # The mass function travels with the definition: the flavour declares
        # all four choices now and validates them together, so overriding one
        # means restating the set.  `tinker08` is stated as a function of Delta
        # and covers all three; the flavour's own `tinker08_csst` is fitted at
        # 200m alone and is correctly refused at the other two.
        return {d: make_field(PLANCK18,
                              DIFFERENTIABLE.with_(mdef=d, hmf_model=hmf),
                              pk, z=0.0, hmf_model=hmf, cm_model="duffy08")
                for d in ("200m", "200c", "vir")}

    def test_delta_mean_is_not_the_number_in_the_name(self):
        """200c is 645 against the mean density at z = 0, and vir is 331."""
        from ggah_mod.halos.mass_definitions import parse_mass_def
        got = {d: float(parse_mass_def(d).delta_mean(0.0, PLANCK18))
               for d in ("200m", "200c", "vir", "500c")}
        assert got["200m"] == pytest.approx(200.0)
        assert got["200c"] == pytest.approx(645.16, rel=1e-3)
        assert got["vir"] == pytest.approx(330.66, rel=1e-3)
        # 500c is 2.5x 200c by construction, both being critical.
        assert got["500c"] / got["200c"] == pytest.approx(2.5, rel=1e-12)

    @pytest.mark.parametrize("hmf", ["tinker08", "despali16"])
    def test_a_delta_aware_fit_responds_to_the_definition(self, pk, hmf):
        f = self._fields(pk, hmf)
        a, b, c = (np.asarray(f[d].dndm) for d in ("200m", "200c", "vir"))
        # Ratios, not `allclose`: dn/dM is ~1e-11 at the low-mass end and
        # ~1e-28 at the high, so `allclose`'s default atol of 1e-8 calls every
        # pair of these arrays equal -- including two that differ by four
        # orders of magnitude.  That is how this assertion first passed against
        # the defect it was written to catch.
        assert np.max(np.abs(b / a - 1.0)) > 0.01, f"{hmf}: 200c == 200m"
        assert np.max(np.abs(c / a - 1.0)) > 0.01, f"{hmf}: vir == 200m"
        # Denser boundary -> fewer haloes of a given mass, at every mass.
        assert np.all(b < a) and np.all(c < a)

    def test_a_fit_with_no_delta_is_left_alone(self, pk):
        """`press74` has no Delta to take, and must not be handed one."""
        f = self._fields(pk, "press74")
        np.testing.assert_array_equal(np.asarray(f["200m"].dndm),
                                      np.asarray(f["200c"].dndm))

    def test_the_bias_carries_the_same_boundary(self, pk):
        """`bias_tinker10` is a function of log10(Delta), and a
        peak-background-split pair is only a pair at a common Delta."""
        f = self._fields(pk)
        a, b = (np.asarray(f[d].bias) for d in ("200m", "200c"))
        assert np.max(np.abs(b / a - 1.0)) > 0.01, "the bias ignored the mdef"
        assert np.all(b > a), "a denser boundary selects more biased haloes"

    def test_an_explicit_delta_still_wins(self, pk):
        """The caller may override; the field only fills in what was not said."""
        b200c = DIFFERENTIABLE.with_(mdef="200c", hmf_model="tinker08")
        b200m = DIFFERENTIABLE.with_(mdef="200m", hmf_model="tinker08")
        forced = make_field(PLANCK18, b200c, pk, z=0.0, cm_model="duffy08",
                            hmf_model="tinker08", delta=200.0)
        at_200m = make_field(PLANCK18, b200m, pk, z=0.0, cm_model="duffy08",
                             hmf_model="tinker08")
        np.testing.assert_allclose(np.asarray(forced.dndm),
                                   np.asarray(at_200m.dndm), rtol=1e-12)

    def test_the_pair_closes_better_at_a_matched_delta(self, pk):
        r"""The measurement that would have caught it.

        The peak-background split says $\int b\,(M/\bar\rho)\,(dn/dM)\,dM = 1$.
        Over a finite mass range it does not reach 1 -- that deficit is the mass
        below the grid -- but pairing a 645 abundance with a 200 bias moves the
        integral by ten per cent, in the wrong direction, for no physical
        reason.
        """
        from ggah_mod.halos.linear_bias import make_bias, mass_weighted_bias
        from ggah_mod.halos.mass_definitions import parse_mass_def
        f = self._fields(pk)["200c"]
        d_m = parse_mass_def("200c").delta_mean(0.0, PLANCK18)
        matched = float(mass_weighted_bias(f.m, f.dndm, f.bias, f.rho_cold))
        mismatched = float(mass_weighted_bias(
            f.m, f.dndm, make_bias("tinker10")(f.sigma, 200.0), f.rho_cold))
        assert abs(matched / mismatched - 1.0) > 0.05, (matched, mismatched)
        # And the field's own bias is the matched one.
        np.testing.assert_allclose(
            np.asarray(f.bias),
            np.asarray(make_bias("tinker10")(f.sigma, d_m)), rtol=1e-12)


class TestTheRecalibratedMultiplicityIsSelectable:
    r"""The recalibration has to be reachable from the layer it recalibrates.

    ``emu_hmf`` fits a correction to ``tinker08``'s four shape parameters and
    validates it, and for a while nothing here could select it: the registry
    held sixteen fits, none of them that one, and the only mentions of
    ``emu_hmf`` in this package were two comments.  A recalibration nothing can
    choose is a result rather than a tool.
    """

    _M = np.logspace(12.0, 15.0, 6)

    @classmethod
    def _field(cls, pk, model="tinker08_csst", mdef="200m", cm="duffy08",
               cosmo=PLANCK18):
        return make_field(cosmo, z=0.0, pk=pk, m=cls._M, mdef=mdef,
                          hmf_model=model, bias_model="tinker10", cm_model=cm)

    def test_it_is_in_the_registry_and_marked_as_needing_a_cosmology(self):
        from ggah_mod.halos.mass_function import (
            CALIBRATION, COSMOLOGY_DEPENDENT_MULTIPLICITY, MULTIPLICITY)

        assert "tinker08_csst" in MULTIPLICITY
        assert "tinker08_csst" in COSMOLOGY_DEPENDENT_MULTIPLICITY
        assert "tinker08_csst" in CALIBRATION, \
            "an unrecorded calibration is the gap the guard exists to close"

    def test_make_field_passes_the_cosmology_for_it(self, pk):
        """The caller should not have to know which family a name belongs to."""
        d = np.asarray(self._field(pk).dndm)
        assert np.all(d > 0.0) and np.all(np.isfinite(d))

    def test_it_differs_from_tinker08_by_a_few_per_cent(self, pk):
        """The offset the recalibration exists to remove.

        Two-ish per cent at the fiducial: large enough to matter for a cluster
        count, small enough that nobody would have found it as a bug.
        """
        a = np.asarray(self._field(pk, model="tinker08").dndm)
        b = np.asarray(self._field(pk).dndm)
        r = b / a
        assert np.all(r > 1.0), "the correction is positive at Planck18"
        assert 1.005 < r.min() and r.max() < 1.10, r

    def test_the_gradient_reaches_it(self, pk):
        """It is a network, and it has to stay on the differentiable path."""
        def logn(a):
            return jnp.log(self._field(
                pk, cosmo=PLANCK18.replace(ln10A_s=a)).dndm[3])

        g = float(jax.grad(logn)(PLANCK18.ln10A_s))
        assert np.isfinite(g) and g > 0.0

    def test_it_refuses_a_definition_it_was_not_fitted_in(self, pk):
        """Fitted at 200m.

        Whether the *fractional* correction transfers to another overdensity is
        untested -- ``tinker08``'s own Delta dependence is in the parameters
        this correction multiplies -- so the guard refuses rather than assuming
        it does.
        """
        with pytest.raises(ValueError, match="different halo boundaries"):
            self._field(pk, mdef="200c", cm="diemer19")

    def test_calling_it_without_a_cosmology_says_so(self):
        from ggah_mod.halos.mass_function import fsigma_tinker08_csst

        with pytest.raises(TypeError, match="needs a cosmology"):
            fsigma_tinker08_csst(np.array([1.0]), 0.0)


class TestTheFieldHandsItsDefinitionToTheConcentration:
    r"""The same omission as the abundance's, at the same call site.

    Sec. 3.7 of the paper records that ``make_field`` took an ``mdef``, built
    :math:`r_\Delta` from it, recorded it on the field, and never passed it to
    the *mass function*.  It also never passed it to the peak-height
    *concentration* relations, which was invisible for a different reason: the
    default relation there is calibrated in 200c and nothing else, so it never
    had another definition to get wrong.  Select one that does cover several
    and a field declaring 200m came back with 200c concentrations -- half as
    large again at :math:`10^{12}\,\msunh`, smooth, positive and correctly
    shaped.
    """

    _M = np.logspace(12.0, 15.0, 6)

    def _conc(self, pk, mdef):
        return np.asarray(make_field(
            PLANCK18, z=0.0, pk=pk, m=self._M, mdef=mdef,
            hmf_model="tinker08", bias_model="tinker10",
            cm_model="bhattacharya13").conc)

    def test_the_definition_changes_the_answer(self, pk):
        """The plumbing test, and the one that failed.

        200m and 200c are different boundaries, so the concentrations must
        differ.  They were identical, because both were 200c.
        """
        a, c = self._conc(pk, "200m"), self._conc(pk, "200c")
        assert not np.allclose(a, c, rtol=1e-6), \
            "200m and 200c gave the same concentration: the field's mass " \
            "definition is not reaching the relation"
        # 200m encloses more, so c_200m > c_200c at fixed mass.
        assert np.all(a > c)

    @pytest.mark.parametrize("mdef", ["200m", "200c", "vir"])
    def test_it_matches_the_relation_called_directly(self, pk, mdef):
        """Whatever the field reports must be what the fit says at that mdef."""
        from ggah_mod.cosmology import growth
        from ggah_mod.halos.concentration import c_bhattacharya13

        f = make_field(PLANCK18, z=0.0, pk=pk, m=self._M, mdef=mdef,
                       hmf_model="tinker08", bias_model="tinker10",
                       cm_model="bhattacharya13")
        direct = np.asarray(c_bhattacharya13(
            np.asarray(f.sigma), growth.growth_factor(0.0, PLANCK18, pk), mdef))
        assert np.asarray(f.conc) == pytest.approx(direct, rel=1e-10)


class TestTheTwoRecalibrationsAreDifferentModels:
    r"""Two definitions, two corrections, and they are not interchangeable.

    ``emu_hmf`` fits the same architecture on the same designs against the
    emulator's ``RockstarM200m`` and its ``RockstarMvir`` -- a different
    overdensity from the *same* halo finder, so the comparison isolates the
    boundary rather than the catalogue.  Both reduce the residual: 7.0 per cent
    to 0.52 at 200m, 10.9 to 0.54 at virial.

    They are still different functions, and by more than either residual.  That
    is the measurement behind refusing to apply one at the other's definition,
    and behind their being two registry entries instead of one model with a
    ``Delta`` argument.
    """

    _S = np.linspace(0.5, 2.0, 12)

    def test_both_are_registered_and_cosmology_dependent(self):
        from ggah_mod.halos.mass_function import (
            CALIBRATION, COSMOLOGY_DEPENDENT_MULTIPLICITY, MULTIPLICITY)

        for name in ("tinker08_csst", "tinker08_csst_vir"):
            assert name in MULTIPLICITY
            assert name in COSMOLOGY_DEPENDENT_MULTIPLICITY
            assert name in CALIBRATION

    def test_they_disagree_by_more_than_either_residual(self):
        """The number that makes them two models rather than one.

        Each is worth about half a per cent against the emulator it was fitted
        to.  They disagree with *each other* by ten per cent and more, so using
        one where the other belongs is worse than not correcting at all.
        """
        from ggah_mod.halos.mass_function import (
            fsigma_tinker08_csst, fsigma_tinker08_csst_vir)

        a = np.asarray(fsigma_tinker08_csst(self._S, 0.0, cosmo=PLANCK18))
        b = np.asarray(fsigma_tinker08_csst_vir(self._S, 0.0, cosmo=PLANCK18))
        d = np.abs(b / a - 1.0)
        assert d.min() > 0.05, "the two corrections have become the same fit"

    @pytest.mark.parametrize(
        "name,good,bad",
        [("tinker08_csst", "200m", "vir"),
         ("tinker08_csst_vir", "vir", "200m")])
    def test_each_is_refused_outside_its_own_definition(self, name, good, bad):
        from ggah_mod.halos.calibration import check_calibration

        check_calibration(name, "bhattacharya13", good, 0.0, "strict",
                          "tinker10")
        with pytest.raises(ValueError, match="different halo boundaries"):
            check_calibration(name, "bhattacharya13", bad, 0.0, "strict",
                              "tinker10")

    def test_the_virial_set_is_a_second_cosmology_carrying_option(self, pk):
        """What shipping it buys: a whole consistent set at another definition.

        ``bhattacharya13`` covers virial as well as 200m, so the virial
        recalibration completes a second combination in which every rung can
        respond to a cosmology.  200c still cannot have one, because the
        emulator's only 200c is a friends-of-friends mass.
        """
        f = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0, mdef="vir",
                       hmf_model="tinker08_csst_vir", bias_model="tinker10",
                       cm_model="bhattacharya13")
        d = np.asarray(f.dndm)
        assert np.all(np.isfinite(d)) and np.all(d > 0.0)


class TestOneEpochPerField:
    r"""The redshift axis stops at layer 2, and the two ways across it.

    `make_field` builds the tables at one epoch.  That is not an omission to be
    worked around: layer 4's contractions are ``(Nk, NM)`` and tell a ``(NM,)``
    weight from a ``(Nk, NM)`` one by the leading axis, so a third axis on every
    table turns a check that raises into a branch that is silently wrong.  What
    the redshift set needs instead is one *layer-1* call, because a Boltzmann
    solve is memoised on the whole redshift tuple and a loop of scalar requests
    is a loop of solves --- 93.7 s against 6.0 s for twelve, measured with CLASS.

    These tests run on `CountingPk` rather than a real backend precisely so the
    solve count is an assertion rather than a stopwatch.
    """

    # The definition each relation was calibrated in, so this class tests the
    # redshift axis rather than the calibration guard.  Same idiom as
    # `test_every_concentration_relation_can_build_a_field` above.
    @staticmethod
    def _mdef_for(name):
        from ggah_mod.halos.concentration import CM_CALIBRATION
        return CM_CALIBRATION[name][1].replace("SO ", "").split("/")[0].strip()

    def _kw(self, name):
        return dict(mdef=self._mdef_for(name), cm_model=name,
                    hmf_model="tinker08")

    # ------------------------------------------------------------- refusal
    def test_a_vector_redshift_is_refused_naming_both_routes(self, counting_pk):
        """A refusal that does not say what to do instead is a traceback.

        Before this, the vector died eight lines later inside the sigma(M)
        quadrature as `Incompatible shapes for broadcasting: (256,), (3,)`, and
        the clear guard in `_validate` never got to run.
        """
        with pytest.raises(ValueError, match="one.{0,3}redshift|one epoch"):
            make_field(PLANCK18, DIFFERENTIABLE, counting_pk,
                       z=jnp.array([0.0, 0.5, 1.0]), hmf_model="tinker08")
        try:
            make_field(PLANCK18, DIFFERENTIABLE, counting_pk,
                       z=jnp.array([0.0, 0.5]), hmf_model="tinker08")
        except ValueError as exc:
            assert "make_fields" in str(exc)
            assert "jax.vmap" in str(exc)

    def test_the_refusal_costs_no_solve(self, counting_pk):
        """It fires before the two spectrum calls, not after them."""
        with pytest.raises(ValueError):
            make_field(PLANCK18, DIFFERENTIABLE, counting_pk,
                       z=jnp.array([0.0, 0.5]), hmf_model="tinker08")
        assert counting_pk.calls == []

    def test_a_traced_scalar_redshift_is_still_allowed(self, counting_pk):
        """Guards against over-refusing: the check is on rank, and `jit` of a
        scalar leaves the rank at zero.

        The eager call first is a property of the *stub*, not of `make_field`:
        `AnalyticPk` normalises itself to sigma_8 on first use and does that
        arithmetic in numpy, so the very first call must not be the one inside
        the trace.  Warming it here keeps this test about the rank check.
        """
        counting_pk.pk_cb(np.logspace(-3, 1, 8), 0.0, PLANCK18)   # warm the stub
        out = jax.jit(lambda zz: make_field(
            PLANCK18, DIFFERENTIABLE, counting_pk, z=zz,
            hmf_model="tinker08").dndm)(0.5)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_make_fields_refuses_a_two_dimensional_grid(self, counting_pk):
        with pytest.raises(ValueError, match="one-dimensional"):
            make_fields(PLANCK18, DIFFERENTIABLE, counting_pk,
                                jnp.zeros((2, 2)), hmf_model="tinker08")

    def test_a_backend_that_drops_the_redshift_axis_is_refused(self, counting_pk):
        """A spectrum that ignored the axis would return D(z) = 1 and look
        exact, so the returned shape is checked rather than trusted."""
        class Flat(type(counting_pk)):
            def pk_cb(self, k, z, cosmo):
                return self._pk(k, cosmo)            # (Nk,) whatever z is

        with pytest.raises(ValueError, match="one row per entry"):
            make_fields(PLANCK18, DIFFERENTIABLE, Flat(),
                                jnp.array([0.0, 0.5]), hmf_model="tinker08")

    # -------------------------------------------------------------- parity
    @pytest.mark.parametrize("name", sorted(H.CONCENTRATION))
    def test_make_fields_is_the_per_z_loop(self, name):
        """Every table, every relation, exactly equal -- not to a tolerance.

        `make_fields` differs from a loop only in where the spectra came from,
        so anything but equality means the chain was copied rather than shared.
        """
        from conftest import CountingPk
        z = jnp.array([0.0, 0.5, 1.0])
        kw = self._kw(name)
        got = make_fields(PLANCK18, DIFFERENTIABLE, CountingPk(), z, **kw)
        want = [make_field(PLANCK18, DIFFERENTIABLE, CountingPk(),
                           z=float(zz), **kw) for zz in z]
        assert len(got) == len(want)
        for a, b in zip(got, want):
            for leaf in ("sigma", "dlns", "nu", "dndm", "bias", "conc",
                         "r_delta", "r_s", "pk_cb", "pk_lin"):
                np.testing.assert_array_equal(
                    np.asarray(getattr(a, leaf)), np.asarray(getattr(b, leaf)),
                    err_msg=f"{name}: {leaf}")

    @pytest.mark.parametrize("name", sorted(H.CONCENTRATION))
    def test_vmap_over_the_redshift_matches_the_loop(self, name):
        """The second route, and the one whose claim was false.

        `diemer19` raised `TracerArrayConversionError` here until
        `growth.growth_rate` stopped forcing its redshift concrete with numpy;
        it was latent because every shipped flavour declares `bhattacharya13`.
        Round-off only, from XLA reordering under the map: 1.1e-15 measured.
        """
        from conftest import CountingPk
        z = jnp.array([0.0, 0.5, 1.0])
        kw = self._kw(name)
        stacked = jax.vmap(lambda zz: make_field(
            PLANCK18, DIFFERENTIABLE, CountingPk(), z=zz, **kw))(z)
        assert stacked.n_m == make_field(
            PLANCK18, DIFFERENTIABLE, CountingPk(), z=0.0, **kw).n_m
        for i, zz in enumerate(z):
            one = make_field(PLANCK18, DIFFERENTIABLE, CountingPk(),
                             z=float(zz), **kw)
            for leaf in ("sigma", "dndm", "bias", "conc", "r_delta"):
                np.testing.assert_allclose(
                    np.asarray(getattr(stacked, leaf))[i],
                    np.asarray(getattr(one, leaf)),
                    rtol=1e-12, err_msg=f"{name}: {leaf} at z={float(zz)}")

    # ----------------------------------------------------------- the payoff
    @pytest.mark.parametrize("name,solves", [("duffy08", 1),
                                             ("bhattacharya13", 2),
                                             ("diemer19", 2)])
    def test_one_solve_serves_every_redshift(self, name, solves):
        """The test that makes the rule pay off, and it needs no Boltzmann code.

        `CountingPk.solves` records distinct redshift *tuples*, which is what
        the real backends memoise on --- so this is the solve count the same
        sequence would cost against CLASS.  The relations that read a growth off
        the spectrum cost one more than the rest, and *one* more, not N more:
        that is `_growth_for_stack` hoisting the call out of the loop.
        """
        from conftest import CountingPk
        z = jnp.linspace(0.0, 1.5, 8)
        kw = self._kw(name)

        batched = CountingPk()
        make_fields(PLANCK18, DIFFERENTIABLE, batched, z, **kw)
        assert len(batched.solves) == solves, batched.solves

        looped = CountingPk()
        for zz in z:
            make_field(PLANCK18, DIFFERENTIABLE, looped, z=float(zz), **kw)
        assert len(looped.solves) >= len(z)
        assert len(batched.solves) < len(looped.solves)

    def test_the_range_warning_reaches_a_stacked_redshift(self, counting_pk):
        """`_check_one` answered "traced" and "an array" with the same skip, so
        every stacked path had the calibration range check silently off.  Here
        the redshifts are concrete and each one answers for itself."""
        with pytest.warns(MismatchWarning, match="fitted over z"):
            make_fields(PLANCK18, DIFFERENTIABLE, counting_pk,
                                jnp.array([0.0, 9.0]), mdef="200c",
                                cm_model="duffy08", hmf_model="tinker08",
                                calibration="warn")
