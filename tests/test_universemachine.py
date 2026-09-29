r"""UniverseMachine's SHMR and TRINITY's black-hole chain, beside the incumbents.

Both are registry entries rather than replacements, and the point of that is the
comparison: two relations on one halo field is a statement about the relations,
where one relation on one halo field is a statement about nothing.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import agn as A
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import sham as SH
from ggah_mod.sectors import BaryonSplit
from ggah_mod.sectors.census import census
from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

H = PLANCK18.h
#: Physical log10 halo masses inside Appendix J's stated validity range.
LG_OK = np.linspace(10.6, 14.9, 40)


def _um(lg_phys, z=0.0, **kw):
    """The relation, called on *physical* log10 M_peak."""
    return np.asarray(SH.mstar_universemachine(
        jnp.asarray(lg_phys) + np.log10(H), z, h=H, **kw))


class TestTheRegistryEntry:
    def test_it_is_registered_everywhere_a_relation_must_be(self):
        assert "universemachine" in SH.SHMR
        assert set(SH.SHMR) == set(SH.SHMR_CALIBRATION)
        assert set(SH.SHMR) == set(SH.SHMR_INCLUDES_REMNANTS)

    def test_the_calibration_row_records_the_mass_range(self):
        """Appendix J is fitted for 10^10.5 < Mpeak < 10^15 Msun, and the
        shipped grid starts below that.  `SHMR_CALIBRATION` exists so applying
        a relation outside its row is visible rather than silent."""
        cal = SH.SHMR_CALIBRATION["universemachine"]
        assert "10^10.5" in cal.fit and "Msun" in cal.fit
        assert cal.z_range == (0.0, 10.0)

    def test_observed_stellar_mass_means_remnants_are_included(self):
        """The chosen Table J1 row is the *observed* SM fit, and observed
        stellar mass is a population-synthesis mass: living stars plus
        remnants."""
        assert SH.SHMR_INCLUDES_REMNANTS["universemachine"] is True


class TestItIsNotBehroozi13:
    """Same author, same purpose, different function -- worth a test because
    the temptation to share a kernel is real and the algebra does not allow
    it."""

    def test_the_two_relations_differ(self):
        um = _um(LG_OK)
        b13 = np.asarray(SH.mstar_behroozi13(
            jnp.asarray(LG_OK) + np.log10(H), 0.0))
        assert np.max(np.abs(um - b13)) > 0.1

    def test_only_one_of_them_has_the_high_redshift_damping(self):
        """B13 damps its evolution by exp(-4a^2); Appendix J has no such term,
        so the two diverge with redshift rather than converging."""
        d0 = np.max(np.abs(_um(LG_OK, 0.0) - np.asarray(
            SH.mstar_behroozi13(jnp.asarray(LG_OK) + np.log10(H), 0.0))))
        d3 = np.max(np.abs(_um(LG_OK, 3.0) - np.asarray(
            SH.mstar_behroozi13(jnp.asarray(LG_OK) + np.log10(H), 3.0))))
        assert d3 > d0


class TestTheRelationItself:
    def test_the_stellar_fraction_peaks_near_1e12(self, ):
        """The one robust feature of every SHMR ever measured."""
        f_star = 10.0 ** (_um(LG_OK) - LG_OK)
        peak = LG_OK[int(np.argmax(f_star))]
        assert 11.8 < peak < 12.6
        assert 0.01 < float(np.max(f_star)) < 0.05

    def test_it_is_monotone_in_halo_mass(self):
        assert np.all(np.diff(_um(LG_OK)) > 0)

    def test_the_slopes_bracket_the_peak(self):
        """alpha ~ 2 below the pivot and beta ~ 0.5 above it, so the relation is
        steep at low mass and shallow at high mass.  If the sign convention of
        Eq. (J1) were inverted this would come out backwards."""
        lo = np.polyfit(LG_OK[:8], _um(LG_OK)[:8], 1)[0]
        hi = np.polyfit(LG_OK[-8:], _um(LG_OK)[-8:], 1)[0]
        assert lo > 1.5 and hi < 0.8 and lo > hi

    def test_h_is_required_and_matters(self):
        """Appendix J is in physical solar masses and every mass here is in
        Msun/h.  There is no safe default, so there is none -- and the size of
        the mistake it prevents is 0.17 dex, five times the fit's tolerance."""
        with pytest.raises(TypeError):
            SH.mstar_universemachine(jnp.asarray([12.0]), 0.0)
        right = float(SH.mstar_universemachine(jnp.asarray([12.0]), 0.0, h=H)[0])
        as_if_physical = float(
            SH.mstar_universemachine(jnp.asarray([12.0]), 0.0, h=1.0)[0])
        assert abs(right - as_if_physical) > 0.15

    def test_it_survives_the_low_mass_end_of_the_grid(self):
        """Below the fit's floor the relation is being extrapolated, which is
        recorded rather than forbidden -- but it must not overflow.  The naive
        `log10(10^u + 10^v)` does, which is why the kernel uses `logaddexp`."""
        out = _um(np.linspace(8.0, 10.5, 20))
        assert np.all(np.isfinite(out))


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("key,val", [("eps0", -1.435), ("m0", 12.081),
                                         ("alpha0", 1.957), ("beta0", 0.474),
                                         ("gamma0", -1.065)])
    def test_every_coefficient_carries_a_gradient(self, key, val):
        def f(v):
            return jnp.sum(SH.mstar_universemachine(
                jnp.asarray(LG_OK) + np.log10(H), 0.0, h=H, **{key: v}))

        ad = float(jax.grad(f)(val))
        fd = float((f(val + 1e-6) - f(val - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5), key

    @pytest.mark.x64
    def test_the_redshift_dependence_is_differentiable(self):
        """The half `zu15` does not have: it is a z = 0 relation, and this one
        is fitted over 0 < z < 10."""
        def f(z):
            return jnp.sum(SH.mstar_universemachine(
                jnp.asarray(LG_OK) + np.log10(H), z, h=H))

        ad = float(jax.grad(f)(0.5))
        fd = float((f(0.5 + 1e-6) - f(0.5 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5)


@pytest.mark.slow
class TestAgainstTheIncumbent:
    """The comparison the registry exists for."""

    @pytest.fixture(scope="class")
    def field(self):
        return make_field(PLANCK18, DIFFERENTIABLE,
                          make_pk(DIFFERENTIABLE.pk), z=0.0)

    @staticmethod
    def _omega_star(field, f_star):
        f_b = PLANCK18.Omega_b / PLANCK18.Omega_m
        split = BaryonSplit.from_hot(
            f_b, E.f_gas_sigmoid(jnp.log10(field.m), f_b), f_star_cen=f_star)
        return float(census(field, split).omega["star_cen"])

    def test_the_two_relations_agree_to_about_ten_per_cent(self, field):
        """And that is the finding.

        Omega_star is 1.30e-3 under `zu15` and 1.45e-3 under
        `universemachine` -- 11% apart, against Dev et al. (2024)'s 2.09e-3
        which both undershoot by ~35%.  So the stellar deficit in the census is
        **not** the choice of relation: two independently calibrated SHMRs put
        it in the same place, and what is missing is the satellites neither of
        them carries.
        """
        lm = jnp.log10(field.m)
        zu = self._omega_star(field, jnp.power(
            10.0, SH.mstar_from_mh_zu15(lm, 12.10, 10.31, 0.33, 0.42, 1.21))
            / field.m)
        um = self._omega_star(field, jnp.power(
            10.0, SH.mstar_universemachine(lm, 0.0, h=H)) * H / field.m)
        assert 0.85 < um / zu < 1.3
        assert um < 2.09e-3 and zu < 2.09e-3

    def test_the_extrapolation_below_the_fit_is_negligible(self, field):
        """0.4% of Omega_star comes from haloes below Appendix J's 10^10.5
        floor, so the recorded extrapolation is a caveat rather than a
        problem."""
        from ggah_mod.cosmology import constants as C
        lm = jnp.log10(field.m)
        ms = np.asarray(jnp.power(10.0, SH.mstar_universemachine(lm, 0.0, h=H)) * H)
        keep = np.asarray(field.m) / H >= 10 ** 10.5
        full = float(field.integrate(field.dndm * ms) / C.RHO_CRIT0)
        inside = float(field.integrate(
            jnp.asarray(np.asarray(field.dndm) * keep) * ms) / C.RHO_CRIT0)
        assert 1.0 - inside / full < 0.02


_GAL = GalaxySector("zumandelbaum15")
_GP = galaxy_defaults("zumandelbaum15")


class TestTheTrinityChain:
    def test_the_chains_are_a_registry_with_powell_the_default(self):
        assert set(A.BH_CHAINS) == {"powell", "trinity"}
        assert A.AgnSector(_GAL).bh_chain == "powell"

    def test_an_unknown_chain_is_refused(self):
        with pytest.raises(ValueError, match="unknown BH chain"):
            A.AgnSector(_GAL, bh_chain="soltan")

    def test_it_refuses_to_run_without_parameters(self):
        """TRINITY has no citable best-fit values: the paper points at an
        Appendix H it does not contain, releases no chains, and the 2023
        erratum re-ran the fit and published figures.  So the model is here and
        refuses to pretend it is calibrated."""
        with pytest.raises(ValueError, match="no published values"):
            A.AgnSector(_GAL, bh_chain="trinity").mean_log10_mbh(
                jnp.asarray([12.0]), A.AgnParams(), _GP, h=0.7)

    def test_the_refusal_names_the_erratum(self):
        with pytest.raises(ValueError, match="erratum"):
            A.mbh_trinity(jnp.asarray([11.0]), A.AgnParams())

    def test_equation_34_reproduces_a_hand_value(self):
        """At M_bulge = 1e11 the log ratio is zero, so log M_BH is beta_BH
        exactly -- and at z = 0, beta_BH is beta_0."""
        p = A.AgnParams()
        for k, v in (("bh_beta0", 8.0), ("bh_beta_a", 0.5), ("bh_beta_z", 0.1),
                     ("bh_gamma0", 1.1), ("bh_gamma_a", 0.0),
                     ("bh_gamma_z", 0.0)):
            object.__setattr__(p, k, v)
        assert float(A.mbh_trinity(jnp.asarray([11.0]), p, z=0.0)[0]) == \
            pytest.approx(8.0, abs=1e-12)
        # one decade up in bulge mass moves it by gamma
        assert float(A.mbh_trinity(jnp.asarray([12.0]), p, z=0.0)[0]) == \
            pytest.approx(9.1, abs=1e-12)

    def test_the_erdf_integrates_to_the_duty_cycle(self):
        """Eq. (49)'s delta function at zero accretion is deliberately absent:
        the inactive fraction is the occupation's duty cycle, and carrying it
        twice is how two numbers for one quantity disagree."""
        lam = jnp.linspace(-4.0, 2.0, 2000)
        for f_duty in (1.0, 0.1):
            e = A.erdf_trinity(lam, -1.0, 0.6, 2.0, f_duty=f_duty)
            assert float(jnp.trapezoid(e, lam)) == pytest.approx(f_duty,
                                                                 rel=1e-6)

    def test_the_erdf_is_a_double_power_law(self):
        """Two slopes, not one -- which is what distinguishes it from the
        Schechter-like `erdf` already in this module."""
        lam = jnp.linspace(-4.0, 2.0, 2000)
        e = np.asarray(A.erdf_trinity(lam, -1.0, 0.6, 2.0))
        lg = np.asarray(lam)
        lo = np.polyfit(lg[:200], np.log10(e[:200]), 1)[0]
        hi = np.polyfit(lg[-200:], np.log10(e[-200:]), 1)[0]
        assert lo == pytest.approx(-0.6, abs=0.05)
        assert hi == pytest.approx(-2.0, abs=0.05)

    @staticmethod
    def _params():
        p = A.AgnParams()
        for k, v in (("bh_beta0", 8.0), ("bh_beta_a", 0.5), ("bh_beta_z", 0.1),
                     ("bh_gamma0", 1.1), ("bh_gamma_a", 0.0),
                     ("bh_gamma_z", 0.0)):
            object.__setattr__(p, k, v)
        return p

    def test_the_redshift_actually_reaches_the_chain(self):
        """The bug this test exists for: an earlier version fetched the
        redshift with `getattr(p, "z", 0.0)`, and `AgnParams` has no `z` field
        -- so the one chain whose entire content is redshift evolution
        evaluated at z = 0 on every field, silently and forever.  It is an
        argument now."""
        t = A.AgnSector(_GAL, bh_chain="trinity")
        p = self._params()
        vals = [float(t.mean_log10_mbh(jnp.asarray([13.0]), p, _GP, h=0.7, z=z)[0])
                for z in (0.0, 1.0, 2.0)]
        assert len(set(np.round(vals, 6))) == 3

    def test_the_powell_chain_ignores_redshift(self):
        """So threading `z` through changed nothing for the incumbent, which is
        what makes the two safe to compare."""
        a = A.AgnSector(_GAL)
        p = A.AgnParams()
        v0 = float(a.mean_log10_mbh(jnp.asarray([13.0]), p, _GP, h=0.7, z=0.0)[0])
        v2 = float(a.mean_log10_mbh(jnp.asarray([13.0]), p, _GP, h=0.7, z=2.0)[0])
        assert v0 == pytest.approx(v2, abs=1e-14)

    def test_the_field_aware_callers_pass_the_field_redshift(self):
        """`omega_bh`, `black_hole_mass_function` and the L_X chain all have a
        field and must use its redshift rather than defaulting to zero."""
        import inspect
        for name in ("omega_bh", "black_hole_mass_function", "occupation"):
            src = inspect.getsource(getattr(A.AgnSector, name))
            assert "field.z" in src or "z=field.z" in src, name
