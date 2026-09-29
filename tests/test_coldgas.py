r"""Neutral gas: the halo-total relations, and what the census does with them.

The per-galaxy relation, ``catinella18``, has its own file,
``test_coldgas_galaxies.py``; what is here is the two relations that map a halo
mass to the HI of every galaxy in it.

At the shipped parameters -- Padmanabhan, Refregier & Amara (2017), Table 3 --
Omega_HI comes out at 0.94 times Dev et al. (2024).  It was 1.76 at the earlier
fit's alpha = 0.17, and the tests that pinned that story were re-derived rather
than loosened: each says below what it asserted before and why it changed.

The interesting tests are still not the total but *where* it comes from, which
is the question a census is for and which neither relation's own paper had to
answer -- and, now, the fact that the fit to observations and the measurement in
a simulation no longer agree with each other.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import (
    BaryonSplit, ColdGasParams, ColdGasSector, cosmic_baryon_fraction,
)
from ggah_mod.sectors import coldgas as CG
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import sham as SH
from ggah_mod.sectors.census import census
from ggah_mod.sectors.coldgas import (
    HI_CALIBRATION, HI_HALO, Y_HELIUM, helium_correction, make_hi_halo,
    m_hi_padmanabhan17, m_hi_villaescusa18,
)

pytestmark = pytest.mark.slow

ZM15 = (12.10, 10.31, 0.33, 0.42, 1.21)
F_B = cosmic_baryon_fraction(PLANCK18)

#: Dev et al. (2024), z = 0.  Omega_HI = 4.9e-4 (+2.5/-1.2),
#: Omega_neutral = 7.1e-4 (+3.9/-1.8).
DEV_OMEGA_HI = 4.9e-4


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)


def _omega(field, mass):
    return float(field.integrate(field.dndm * mass) / C.RHO_CRIT0)


class TestTheRegistry:
    def test_every_relation_is_registered_with_its_calibration(self):
        assert set(CG.RELATIONS) == set(HI_CALIBRATION)
        assert set(HI_HALO) | set(CG.HI_GALAXY) == set(CG.RELATIONS)
        assert not set(HI_HALO) & set(CG.HI_GALAXY)
        for name, cal in HI_CALIBRATION.items():
            assert cal.fit and cal.notes
            assert isinstance(cal.cosmology_dependent, bool)
            # Read by name since the registries were unified; the range is the
            # field the sector check actually consumes, so it has to be there.
            assert cal.z_range is not None, name

    def test_an_unknown_relation_is_refused(self):
        with pytest.raises(ValueError, match="unknown HI-halo relation"):
            make_hi_halo("bagla10")
        with pytest.raises(ValueError, match="unknown HI relation"):
            ColdGasSector("bagla10")

    def test_the_sector_has_no_default_relation(self):
        """The two disagree, and one is a fit to data while the other is a
        measurement in one simulation.  Choosing silently would put an
        unasked-for model into a census row."""
        with pytest.raises(TypeError):
            ColdGasSector()

    def test_an_unknown_view_is_refused(self, field):
        with pytest.raises(ValueError, match="unknown cold-gas view"):
            ColdGasSector("villaescusa18").weights(
                field, ColdGasParams(), view="co")


class TestTheRelationsThemselves:
    @pytest.mark.parametrize("name", sorted(HI_HALO))
    def test_the_hi_mass_is_positive_and_finite(self, field, name):
        m_hi = np.asarray(ColdGasSector(name).m_hi(field, ColdGasParams()))
        assert np.all(np.isfinite(m_hi)) and np.all(m_hi > 0)

    @pytest.mark.parametrize("name", sorted(HI_HALO))
    def test_the_hi_fraction_never_exceeds_the_baryons(self, field, name):
        """M_HI/M above f_b^cosmic would be a halo whose neutral gas alone
        outweighs every baryon it is entitled to."""
        f_hi = np.asarray(ColdGasSector(name).m_hi(field, ColdGasParams())
                          / field.m)
        assert np.max(f_hi) < F_B

    def test_the_low_mass_cutoff_bites(self, field):
        """The lower cutoff, v_c0 = 36 km/s: haloes below it host
        little HI, which is what keeps the relation from putting neutral gas
        everywhere the mass function does."""
        p = ColdGasParams()
        m_hi = np.asarray(m_hi_padmanabhan17(field.m, 0.0, PLANCK18,
                                             mdef=field.mdef))
        f_hi = m_hi / np.asarray(field.m)
        assert f_hi[0] < f_hi[np.argmax(f_hi)] / 2

    def test_the_upper_cutoff_removes_hi_from_clusters(self, field):
        """A relation without one puts neutral gas in clusters and the census
        then sees it."""
        p = ColdGasParams()
        lo = np.asarray(m_hi_padmanabhan17(field.m, 0.0, PLANCK18,
                                           log10_vc1=2.3, mdef=field.mdef))
        hi = np.asarray(m_hi_padmanabhan17(field.m, 0.0, PLANCK18,
                                           mdef=field.mdef))
        assert lo[-1] < hi[-1]

    def test_only_one_relation_can_respond_to_a_cosmology(self, field):
        """`villaescusa18` is a measurement in one simulation at one
        cosmology, so its gradient is a structural zero -- the same
        distinction `halos.concentration` draws between its two families, and
        a property of the relation rather than of the transcription."""
        def tng(omega_b):
            return jnp.sum(m_hi_villaescusa18(field.m))

        def pr17(omega_b):
            c = PLANCK18.replace(Omega_b=omega_b)
            return jnp.sum(m_hi_padmanabhan17(field.m, 0.0, c,
                                              mdef=field.mdef))

        assert float(jax.grad(tng)(PLANCK18.Omega_b)) == 0.0
        assert float(jax.grad(pr17)(PLANCK18.Omega_b)) != 0.0
        assert HI_CALIBRATION["villaescusa18"].cosmology_dependent is False
        assert HI_CALIBRATION["padmanabhan17"].cosmology_dependent is True


class TestHeliumAndMolecules:
    def test_the_neutral_mass_is_hi_plus_h2_plus_helium(self, field):
        p = ColdGasParams()
        s = ColdGasSector("villaescusa18")
        want = (1.0 + p.r_mol) * helium_correction() * np.asarray(
            s.m_hi(field, p))
        np.testing.assert_allclose(np.asarray(s.m_neutral(field, p)), want,
                                   rtol=1e-14)

    def test_the_helium_correction_is_derived_not_written_twice(self):
        assert helium_correction() == pytest.approx(1.0 / (1.0 - Y_HELIUM))
        assert helium_correction(0.0) == 1.0

    def test_zero_molecular_gas_is_a_real_limit(self, field):
        """r_mol = 0 is atomic gas only, exactly -- not approximately."""
        s = ColdGasSector("villaescusa18")
        dry = s.m_neutral(field, ColdGasParams(r_mol=0.0))
        want = helium_correction() * s.m_hi(field, ColdGasParams(r_mol=0.0))
        np.testing.assert_array_equal(np.asarray(dry), np.asarray(want))


class TestAgainstTheMeasuredNeutralGasDensity:
    """Predicted, never fitted.  The fit to observations matches the
    measurement and the simulation does not, which is the informative part."""

    @pytest.mark.parametrize("name", sorted(HI_HALO))
    def test_omega_hi_is_the_right_order(self, field, name):
        o = _omega(field, ColdGasSector(name).m_hi(field, ColdGasParams()))
        assert 0.3 * DEV_OMEGA_HI < o < 3.0 * DEV_OMEGA_HI

    def test_observation_matches_and_simulation_does_not(self, field):
        """The two relations now disagree by a factor of two, and which one is
        right is not in question.

        This test used to assert the opposite -- that the two agreed to 4 per
        cent -- and used it to argue that a common 1.8x excess over Dev et al.
        (2024) was a property of both relations rather than of this
        integration.  That held at the earlier fit's alpha = 0.17, where both
        sat 1.8x high.  At Padmanabhan, Refregier & Amara's alpha = 0.09 the fit
        to observations comes down to 0.94x and the IllustrisTNG measurement
        stays at 1.83x, so the agreement is gone and the argument with it.  The
        integration is now cross-checked where it belongs, by the mass function
        re-integrating to the direct Omega_HI (below).
        """
        a = _omega(field, ColdGasSector("padmanabhan17").m_hi(
            field, ColdGasParams()))
        b = _omega(field, ColdGasSector("villaescusa18").m_hi(
            field, ColdGasParams()))
        assert 0.7 < a / DEV_OMEGA_HI < 1.4        # measured 0.94
        assert 1.5 < b / DEV_OMEGA_HI < 2.2        # measured 1.83
        assert 1.5 < b / a < 2.5                   # measured 1.95

    def test_half_the_budget_is_where_the_relation_is_least_known(self, field):
        """The finding, and half of it survived the change of fit.

        Roughly half of Omega_HI still comes from below 1e11 Msun/h -- 48 per
        cent, where the HI-halo relation is least constrained by anything.
        What reversed is the rest.  At alpha = 0.17 the *truncated* integral
        reproduced Dev et al. (2024) (0.98x) and the excess was blamed on the
        low-mass end; at 0.09 the *full* integral does (0.94x) and the
        truncated one is half that.  So matching the measurement now depends on
        the low-mass extrapolation being right, which is a caveat on the match
        rather than an explanation of a miss.  A census that quoted one number
        would have hidden either reading.
        """
        m = np.asarray(field.m)
        m_hi = np.asarray(ColdGasSector("padmanabhan17").m_hi(
            field, ColdGasParams()))
        full = _omega(field, m_hi)
        above = _omega(field, m_hi * (m >= 1e11))
        assert 0.4 < above / full < 0.7             # measured 0.52
        assert 0.7 < full / DEV_OMEGA_HI < 1.4      # measured 0.94
        assert above / DEV_OMEGA_HI < 0.7           # measured 0.49


class TestItReachesTheCensus:
    def test_the_budget_still_closes_with_cold_gas_in_it(self, field):
        lm = jnp.log10(field.m)
        f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)) / field.m
        f_cold = ColdGasSector("padmanabhan17").f_cold(field, ColdGasParams())
        split = BaryonSplit.from_hot(F_B, E.f_gas_sigmoid(lm, F_B),
                                     f_star_cen=f_star, f_cold=f_cold)
        assert np.max(np.abs(np.asarray(split.residual()))) < 5e-16
        c = census(field, split)
        assert abs(float(c.closure_residual())) < 1e-12
        assert float(c.omega["cold"]) > 0.0

    def test_cold_gas_comes_out_of_the_hot_phase_not_the_dark_matter(self,
                                                                    field):
        """`from_hot` holds the hot phase fixed, so adding cold gas must reduce
        the ejected component -- the baryons have to come from somewhere, and
        the collisionless share is not a place they can come from."""
        lm = jnp.log10(field.m)
        f_hot = E.f_gas_sigmoid(lm, F_B)
        f_cold = ColdGasSector("padmanabhan17").f_cold(field, ColdGasParams())
        dry = BaryonSplit.from_hot(F_B, f_hot)
        wet = BaryonSplit.from_hot(F_B, f_hot, f_cold=f_cold)
        np.testing.assert_array_equal(np.asarray(dry.f_collisionless),
                                      np.asarray(wet.f_collisionless))
        assert np.all(np.asarray(wet.f_ejected) <= np.asarray(dry.f_ejected))


class TestParameters:
    @pytest.mark.parametrize("key", sorted(ColdGasParams._PARAMS))
    def test_every_parameter_carries_a_bound_and_a_reason(self, key):
        p = ColdGasParams._PARAMS[key]
        assert p.why, key
        lo, hi = p.bounds
        assert lo < hi and lo <= p.default <= hi, key

    @pytest.mark.x64
    @pytest.mark.parametrize("key,val", [("alpha_hi", 0.09),
                                         ("beta_hi", -0.58),
                                         ("log10_vc0", 1.56),
                                         ("r_mol", 0.3)])
    def test_the_parameters_are_differentiable(self, field, key, val):
        s = ColdGasSector("padmanabhan17")

        def f(v):
            return jnp.sum(s.m_neutral(field, ColdGasParams(**{key: v})))

        ad = float(jax.grad(f)(val))
        fd = float((f(val + 1e-6) - f(val - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5), key


#: Jones, Haynes, Giovanelli & Moorman (2018), MNRAS 477, 2, "The ALFALFA HI
#: mass function: A dichotomy in the low-mass slope and a locally suppressed
#: 'knee' mass" (doi:10.1093/mnras/sty521, arXiv:1802.00053), **converted to
#: this package's units**.
#:
#: Verified against the paper's abstract rather than recalled:
#: phi* = (4.5 +/- 0.2 +/- 0.8)e-3 h_70^3 Mpc^-3 dex^-1,
#: log10(M* h_70^2 / Msun) = 9.94 +/- 0.01 +/- 0.05, alpha = -1.25 +/- 0.02
#: +/- 0.1, and Omega_HI = (3.9 +/- 0.1 +/- 0.6)e-4 h_70^-1.
#:
#: Published as phi* = 4.5e-3 h_70^3 Mpc^-3 dex^-1 and
#: log10(M*/h_70^-2 Msun) = 9.94, alpha = -1.25.  A mass in h_70^-2 Msun is
#: M70 (h/0.7)^-2 physical, hence M70 * 0.49/h in Msun/h; a density in
#: h_70^3 Mpc^-3 is phi/0.7^3 in (Mpc/h)^-3.  The conversion is written out
#: because getting it wrong moves phi* by 2.9x and log M* by 0.14 dex, which is
#: larger than the disagreement it would be used to diagnose.
ALFALFA_PHI_STAR = 4.5e-3 / 0.7 ** 3
ALFALFA_LOG_M_STAR = 9.94 + np.log10(0.49 / PLANCK18.h)
ALFALFA_ALPHA = -1.25


def _alfalfa(log10_mhi):
    x = 10.0 ** (np.asarray(log10_mhi) - ALFALFA_LOG_M_STAR)
    return (np.log(10.0) * ALFALFA_PHI_STAR
            * x ** (ALFALFA_ALPHA + 1.0) * np.exp(-x))


class TestTheHiMassFunction:
    """The halo-total relations have no HI mass function, on purpose.

    They map a halo mass to the HI of every galaxy in it, and ALFALFA and FASHI
    count galaxies.  The halo-total function this sector used to compute put a
    group's galaxies into one entry: it had no knee, stood 4.5 times ALFALFA at
    10^10.65, and was read by nothing downstream, so it went, with its scatter
    parameter.  The per-galaxy function is ``catinella18``'s and is tested in
    ``test_coldgas_galaxies.py``; what stays here is the unit conversion the
    comparison needs, and the refusal.
    """

    def test_the_unit_conversion_reproduces_the_published_density(self):
        """A check on the conversion before it is used to judge anything.
        Integrating the converted Schechter must give ALFALFA's own
        Omega_HI ~ 4e-4; if it does not, every comparison below is measuring
        an h."""
        lg = np.linspace(6.0, 11.0, 400)
        rho = np.trapezoid(_alfalfa(lg) * 10.0 ** lg, lg)
        assert 3.3e-4 < rho / 2.77536627e11 < 4.5e-4

    @pytest.mark.parametrize("name", sorted(HI_HALO))
    def test_a_halo_total_relation_has_no_per_galaxy_function(self, field,
                                                              name):
        with pytest.raises(ValueError, match="halo-total"):
            ColdGasSector(name).hi_mass_function(
                field, np.array([9.0]), ColdGasParams())

    def test_the_halo_function_and_its_scatter_are_gone(self):
        """Removed, not renamed: nothing may quietly reintroduce the
        comparison they invited."""
        assert not hasattr(ColdGasSector, "halo_hi_mass_function")
        assert "sigma_hi" not in ColdGasParams._PARAMS


class TestWhatTwentyOneCentimetreExperimentsMeasure:
    @pytest.mark.parametrize("name", sorted(HI_HALO))
    def test_the_hi_bias_matches_the_measured_value(self, field, name):
        """b_HI ~ 0.85 at z = 0 in the literature.  This is *predicted* -- the
        relations were fitted to HI masses, not to clustering -- so it landing
        in range is the one unambiguous success the sector has."""
        b = float(ColdGasSector(name).bias_hi(field, ColdGasParams()))
        assert 0.7 < b < 1.0, name

    def test_the_bias_is_mass_weighted_not_number_weighted(self, field):
        """An intensity map sums flux: every halo contributes in proportion to
        the HI it holds.  Number-weighting would give the bias of the halo
        population instead, which is a different and larger number."""
        s = ColdGasSector("padmanabhan17")
        p = ColdGasParams()
        mass_weighted = float(s.bias_hi(field, p))
        number_weighted = float(field.effective_bias(jnp.ones_like(field.m)))
        assert mass_weighted != pytest.approx(number_weighted, rel=1e-3)

    def test_the_brightness_temperature_scales_with_omega_hi(self, field):
        from ggah_mod.sectors.coldgas import T_B_COEF_MK, brightness_temperature
        a = float(brightness_temperature(0.0, PLANCK18, 4e-4))
        b = float(brightness_temperature(0.0, PLANCK18, 8e-4))
        assert b == pytest.approx(2.0 * a, rel=1e-12)
        assert a == pytest.approx(T_B_COEF_MK * PLANCK18.h * 4e-4, rel=1e-12)

    def test_the_coefficient_is_derived_rather_than_recalled(self):
        r"""``PLAN.md`` item **G3**.

        It was the literal 189.0, with a docstring saying it was "the value
        this author recalls being standard", not checked against a paper and
        not re-derived, and asking to be one or the other before anything
        computed from it was published.  It is derived now, from
        :math:`A_{10}`, :math:`\nu_{21}` and the fundamental constants, and
        comes to **188.78 mK** -- 0.115 per cent from the remembered value.

        So the memory was sound and, more usefully, the *convention* is now a
        consequence rather than an assumption: the single power of :math:`h` and
        the :math:`(1+z)^2` fall out of the derivation instead of being
        inherited from whichever normalisation was recalled.  The literature
        quotes this in at least three, and a stray :math:`h` here is a 33 per
        cent error that reads as a calibration difference.
        """
        from ggah_mod.sectors import coldgas as CG

        assert CG.T_B_COEF_MK == pytest.approx(188.78, abs=0.01)
        assert CG.T_B_COEF_MK == pytest.approx(189.0, rel=2e-3)
        assert CG.T_B_COEF_MK == CG._t_b_coefficient_mk()

        # ...and the shape of the formula, which is the half a value cannot
        # check: one power of h, and (1+z)^2 / E(z).
        from ggah_mod.cosmology.background import hubble_e

        z, om = 1.5, 6e-4
        got = float(CG.brightness_temperature(z, PLANCK18, om))
        want = (CG.T_B_COEF_MK * PLANCK18.h * om * (1.0 + z) ** 2
                / float(hubble_e(z, PLANCK18)))
        assert got == pytest.approx(want, rel=1e-12)

    def test_it_grows_with_redshift(self, field):
        """(1+z)^2/E(z) rises to z ~ 2 before the expansion wins, at fixed
        Omega_HI.  Which is why intensity mapping targets that range."""
        from ggah_mod.sectors.coldgas import brightness_temperature
        t = [float(brightness_temperature(z, PLANCK18, 4e-4))
             for z in (0.0, 0.5, 1.0)]
        assert t[0] < t[1] < t[2]

    def test_the_observable_combination_is_predicted(self, field):
        """T_b b_HI is what a 21-cm power spectrum's amplitude gives, and both
        halves come from the census rather than from a fit."""
        s = ColdGasSector("padmanabhan17")
        p = ColdGasParams()
        amp = float(s.brightness_temperature(field, p) * s.bias_hi(field, p))
        # mK.  Measured 0.045 at the current fit; it was 0.086 at
        # alpha = 0.17, when it was high with Omega_HI.
        assert 0.02 < amp < 0.20


class TestBothHaloFinderRowsAreAvailable:
    r"""``PLAN.md`` Tier 7: the FoF against FoF-SO row, made selectable.

    The module had recorded, correctly, that the shipped halo field is
    spherical-overdensity while the shipped HI parameters are the FoF row, and
    that the two differ by more than either error bar.  It was a caveat in a
    docstring, reachable only by editing three defaults.
    """

    def test_the_rows_are_the_published_table(self):
        """Table 1: FoF-SO is alpha = 0.16, M_0 = 4.1e10, M_min = 2.4e12."""
        r = CG.villaescusa18_row("fof_so")
        assert r["alpha_tng"] == 0.16
        assert 10.0 ** r["log10_m0"] == pytest.approx(4.1e10, rel=1e-4)
        assert 10.0 ** r["log10_m_min"] == pytest.approx(2.4e12, rel=1e-4)

    def test_the_default_row_is_the_shipped_default(self):
        """The FoF row and ``ColdGasParams``' defaults are the same numbers --
        which is what makes ``VILLAESCUSA18_ROWS`` a description of the shipped
        state rather than a second one beside it."""
        p, fof = CG.ColdGasParams(), CG.villaescusa18_row("fof")
        for k, v in fof.items():
            assert getattr(p, k) == v

    def test_a_row_reaches_the_sector(self):
        """The point of putting the rows where the numbers live.

        A second *registry* entry would have looked like a choice and done
        nothing, because :meth:`ColdGasSector.m_hi` reads its numbers from the
        params and not from the registry function's defaults.
        """
        sec = CG.ColdGasSector(relation="villaescusa18", calibration="off")
        from conftest import AnalyticPk, STUB_CM_MODEL
        f = make_field(PLANCK18, DIFFERENTIABLE, AnalyticPk(), z=0.0,
                       cm_model=STUB_CM_MODEL)
        fof = np.asarray(sec.m_hi(f, CG.ColdGasParams()))
        so = np.asarray(sec.m_hi(f, CG.ColdGasParams(
            **CG.villaescusa18_row("fof_so"))))
        assert np.any(np.abs(so / fof - 1.0) > 0.05), \
            "the two rows must actually differ where it matters"
        # A third shallower, so FoF-SO is the flatter of the two in mass.
        lo, hi = f.m.min() * 10.0, f.m.max() / 10.0
        i, j = int(np.argmin(np.abs(f.m - lo))), int(np.argmin(np.abs(f.m - hi)))
        slope_fof = np.log(fof[j] / fof[i]) / np.log(f.m[j] / f.m[i])
        slope_so = np.log(so[j] / so[i]) / np.log(f.m[j] / f.m[i])
        assert slope_so < slope_fof

    def test_an_unknown_row_says_what_the_choice_is(self):
        with pytest.raises(ValueError, match="fof_so"):
            CG.villaescusa18_row("sphericaloverdensity")

    def test_the_calibration_record_names_the_finder(self):
        """So the unsettled choice is visible from the provenance row and not
        only from the module docstring."""
        sel = CG.HI_CALIBRATION["villaescusa18"].selection.lower()
        assert "fof" in sel and "fof-so" in sel
