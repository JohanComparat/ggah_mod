r"""The cosmic baryon budget: :math:`\sum_i \Omega_i \equiv \Omega_b`.

The per-halo budget of :mod:`~ggah_mod.sectors.matter` closes algebraically and
says nothing about the cosmic one.  These tests are about the difference: the
mass function holds only about half the cold matter density on the shipped
grid, so the two closures need separate machinery and separate checks.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import matter as MT
from ggah_mod.sectors import occupation as O
from ggah_mod.sectors import sham as SH
from ggah_mod.sectors.census import (
    PHASES, census, census_over_mass_ranges,
)

pytestmark = pytest.mark.slow          # every case builds a P(k)

ZM15 = (12.10, 10.31, 0.33, 0.42, 1.21)
F_B = MT.cosmic_baryon_fraction(PLANCK18)

#: Devereux et al. (2024), the stellar row every comparison here is against.
DEV2024 = 2.09e-3


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)


def _split(field, f_cold=0.0):
    lm = jnp.log10(field.m)
    f_hot = E.f_gas_sigmoid(lm, F_B)
    # zu15 returns h^-2 Msun; M_*/M_h wants the Msun/h of M_h.
    f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                       - jnp.log10(field.cosmo.h)) / field.m
    return MT.BaryonSplit.from_hot(F_B, f_hot, f_star_cen=f_star,
                                   f_cold=f_cold)


class TestTheCosmicSumCloses:
    """Against the *input* Omega_b, and by two routes rather than one."""

    def test_the_total_is_the_input_omega_b(self, field):
        c = census(field, _split(field))
        assert float(c.omega_total / c.omega_b - 1.0) == pytest.approx(0.0,
                                                                      abs=1e-14)

    def test_the_outside_term_agrees_with_its_closed_form(self, field):
        """The invariant that is not a tautology.

        `omega_outside` is a residual, so the total closes however wrong the
        quadrature is.  `omega_outside_direct` comes from the mass fraction
        alone and knows nothing about the phases, so the two agreeing is a real
        check on the integration.
        """
        c = census(field, _split(field))
        assert abs(float(c.closure_residual())) < 1e-12

    def test_a_missing_phase_breaks_the_closure(self, field):
        """The check has teeth: drop the ejected component -- the exact mistake
        the three-component matter field was making -- and the two routes to
        the outside term stop agreeing."""
        split = _split(field)
        maimed = split.replace(f_ejected=jnp.zeros_like(split.f_ejected))
        c = census(field, maimed)
        assert abs(float(c.closure_residual())) > 0.1

    def test_the_split_between_phases_cannot_change_the_total(self, field):
        """Only `f_baryon` reaches the sum, so moving mass between phases must
        leave `omega_halos` alone.  This is what makes the census a test of the
        *sectors* rather than of itself."""
        a = _split(field)
        moved = a.replace(f_hot=a.f_hot - 0.01,
                          f_cold=a.f_cold + 0.01)
        ca, cb = census(field, a), census(field, moved)
        assert float(cb.omega_halos / ca.omega_halos - 1.0) == pytest.approx(
            0.0, abs=1e-13)
        assert float(cb.omega["cold"]) > float(ca.omega["cold"])



class TestTheOutsideRowIsOneObject:
    """The census's outside row and the two-halo counterterm book the same
    unresolved matter, by mass and by bias-weighted mass."""

    def test_its_composition_sums_to_the_row(self, field):
        c = census(field, _split(field, f_cold=0.003))
        total = sum(c.omega_outside_by_phase.values())
        assert float(total / c.omega_outside_direct) == pytest.approx(1.0,
                                                                     abs=1e-14)

    def test_the_smallest_halo_sets_the_composition(self, field):
        """Stars and neutral gas as the first resolved node holds them, every
        other baryon diffuse; and a raised cut moves the node."""
        s = _split(field, f_cold=0.003)
        for m_min, j in ((None, 0), (1e12, int(np.searchsorted(
                np.asarray(field.m), 1e12)))):
            c = census(field, s, m_min=m_min)
            frac = {k: float(v / c.omega_outside_direct)
                    for k, v in c.omega_outside_by_phase.items()}
            f_b = float(F_B)
            assert frac["star_cen"] == pytest.approx(
                float(s.f_star_cen[j]) / f_b, rel=1e-12)
            assert frac["cold"] == pytest.approx(0.003 / f_b, rel=1e-12)
            assert frac["diffuse"] == pytest.approx(
                float(s.f_hot[j] + s.f_ejected[j]) / f_b, rel=1e-10)

    def test_its_bias_is_the_ratio_of_the_two_deficits(self, field):
        """0.475 of the cold mass is outside the grid, 0.285 of the
        bias-weighted mass, and the matter there has bias 0.60."""
        from ggah_mod.halos import unresolved
        from ggah_mod.spectra.counterterm import mass_deficit
        u = unresolved(field.m, field.dndm, field.bias, field.rho_cold)
        assert float(u.mass) == pytest.approx(0.475, abs=0.01)
        assert float(u.bias_weighted) == pytest.approx(
            float(mass_deficit(field)), rel=1e-10)
        assert float(u.bias) == pytest.approx(0.60, abs=0.02)
        c = census(field, _split(field))
        assert float(c.unresolved_bias) == pytest.approx(float(u.bias), rel=1e-10)
        assert float(1.0 - c.mass_fraction_in_halos) == pytest.approx(
            float(u.mass), rel=1e-10)


class TestTheDifferentialForm:
    def test_it_integrates_to_the_scalar(self, field):
        """`dOmega/dlog10M` is the quantity Dev et al. (2024) measure and the
        one worth plotting; if it did not integrate to the scalar, the figure
        and the table would be two different results."""
        c = census(field, _split(field))
        lg = np.log10(np.asarray(field.m))
        for phase in PHASES:
            got = np.trapezoid(np.asarray(c.d_omega_dlog10m[phase]), lg)
            want = float(c.omega[phase])
            if want == 0.0:
                assert got == 0.0
            else:
                assert got == pytest.approx(want, rel=1e-3), phase


class TestTheGridIsNotAResult:
    """The number a census table has to carry beside every other number."""

    def test_the_outside_fraction_grows_with_the_mass_cut(self, field):
        runs = census_over_mass_ranges(field, _split(field))
        outside = [float(c.omega_outside / c.omega_b)
                   for _, c in sorted(runs.items())]
        assert outside == sorted(outside), (
            "raising m_min must move baryons from 'in halos' to 'outside'")

    def test_the_move_is_large_enough_to_change_a_conclusion(self, field):
        """Between 1e10 and 1e14 the diffuse fraction roughly doubles.  A
        census quoting one row against Fukugita & Peebles (2004) is comparing a
        resolution limit with a baryon phase."""
        runs = census_over_mass_ranges(field, _split(field))
        lo = float(runs[1e10].omega_outside / runs[1e10].omega_b)
        hi = float(runs[1e14].omega_outside / runs[1e14].omega_b)
        assert 0.3 < lo < 0.6
        assert hi > 0.85
        assert hi - lo > 0.3

    def test_the_closure_holds_at_every_cut(self, field):
        for m, c in census_over_mass_ranges(field, _split(field)).items():
            assert abs(float(c.closure_residual())) < 1e-12, m


class TestAgainstThePublishedCensuses:
    """Predicted, never fitted.  Disagreement is a finding, not a failure."""

    def test_the_stellar_density_is_the_right_order(self, field):
        """Dev et al. (2024) measure Omega_star = 2.09e-3.  This is
        centrals-only -- `f_star_sat` is zero here -- so it must come out
        *below* that, and by roughly the satellite share rather than by an
        order of magnitude."""
        c = census(field, _split(field))
        omega_star = float(c.omega["star_cen"] + c.omega["star_sat"])
        assert 0.3e-3 < omega_star < 2.09e-3

    def test_most_of_the_budget_is_not_in_hot_halo_gas(self, field):
        """Every published census agrees on this much, whatever else they
        disagree about: the collapsed hot phase is a minority of Omega_b.
        Fukugita & Peebles (2004) put intracluster plasma at 4%."""
        c = census(field, _split(field))
        assert float(c.omega["hot"] / c.omega_b) < 0.5


class TestDifferentiability:
    @pytest.mark.x64
    def test_the_census_carries_a_gradient_in_the_gas_pivot(self, field):
        """The census is a forward-modelled observable like any other, so it
        has to differentiate -- otherwise it cannot enter a fit."""
        lm = jnp.log10(field.m)
        f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                           - jnp.log10(field.cosmo.h)) / field.m

        def f(pivot):
            f_hot = E.f_gas_sigmoid(lm, F_B, log10_m_pivot=pivot)
            split = MT.BaryonSplit.from_hot(F_B, f_hot, f_star_cen=f_star)
            return census(field, split).omega["hot"]

        ad = float(jax.grad(f)(13.5))
        fd = float((f(13.5 + 1e-6) - f(13.5 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5)


class TestTheStellarMassConvention:
    """Fukugita & Peebles' remnant rows are not a missing component.

    They are already inside Omega_star, because every SHMR here is calibrated
    against population-synthesis masses that report living stars plus remnants.
    Adding a remnant term would count them twice -- and the census is exactly
    where that would have looked like agreement improving.
    """

    def test_every_relation_declares_its_convention(self):
        from ggah_mod.sectors.sham import SHMR, SHMR_INCLUDES_REMNANTS
        assert set(SHMR_INCLUDES_REMNANTS) == set(SHMR)
        assert all(SHMR_INCLUDES_REMNANTS.values())

    def test_splitting_the_stellar_mass_conserves_it(self):
        """The whole point: a census row can be reported in two parts without
        the budget it belongs to changing by anything."""
        from ggah_mod.sectors.sham import split_stellar
        m = jnp.asarray([1e9, 1e10, 1e11, 1e12])
        ms, rem = split_stellar(m)
        np.testing.assert_allclose(np.asarray(ms + rem), np.asarray(m),
                                   rtol=1e-15)

    def test_the_other_convention_adds_mass_instead(self):
        """A relation reporting living stars only needs the remnants added, and
        then the total *does* grow.  Keeping the two cases distinguishable is
        what the flag is for."""
        from ggah_mod.sectors.sham import split_stellar
        m = jnp.asarray([1e11])
        _, rem_in = split_stellar(m, includes_remnants=True)
        ms_out, rem_out = split_stellar(m, includes_remnants=False)
        assert float(ms_out[0]) == pytest.approx(1e11)
        assert float(rem_out[0]) > float(rem_in[0])
        assert float((ms_out + rem_out)[0]) > 1e11

    def test_the_remnant_share_matches_fukugita_and_peebles(self):
        """Their rows 3.5-3.7 over 3.3-3.7: 0.00048/0.00253 = 0.19.

        Row 3.8 (substellar) is excluded on purpose: a brown dwarf is not a
        remnant, and a population-synthesis stellar mass does not contain one
        either -- so it is genuinely absent from Omega_star rather than hidden
        inside it, unlike the three rows above."""
        from ggah_mod.sectors.sham import REMNANT_FRACTION
        fp04_remnants = 0.00036 + 0.00005 + 0.00007
        fp04_stars = 0.0015 + 0.00055 + fp04_remnants
        assert fp04_remnants == pytest.approx(0.00048)
        assert REMNANT_FRACTION == pytest.approx(
            fp04_remnants / fp04_stars, abs=0.01)

    def test_omega_star_compares_against_the_summed_rows(self, field):
        """Not against rows 3.3+3.4 alone.  Our Omega_star contains remnants,
        so the like-for-like FP04 number is rows 3.3-3.7 summed, 2.53e-3 --
        and comparing to rows 3.3+3.4 alone would manufacture a 23%
        disagreement out of a convention."""
        c = census(field, _split(field))
        omega_star = float(c.omega["star_cen"] + c.omega["star_sat"])
        ms_only = 0.0015 + 0.00055
        with_remnants = ms_only + 0.00036 + 0.00005 + 0.00007
        assert with_remnants / ms_only == pytest.approx(1.23, abs=0.02)
        # centrals-only, so still below either
        assert omega_star < ms_only


class TestTheSatelliteSplitAgainstTheStellarRow:
    r"""T7-7 S1, measured 2026-09-04.  Two claims, and the second is a trap.

    The stellar row is the census disagreement the satellite term was expected
    to explain, and it does: *every* halo's central plus the conditional
    stellar-mass function's satellites reaches 93% of Devereux et al. (2024),
    against 62% for centrals alone.

    The trap is that "centrals alone" means two different things here, 28%
    apart, and the disagreement is recorded against the one
    :meth:`GalaxySector.stellar_fraction` does *not* return.

    **Every number here is at Paper I's iHOD** (``ZU15_PUBLISHED``), the
    galaxy default until 0.8.4 and still the census's stellar row
    (``ZM15`` above).  On the LS10 defaults of 0.8.5 the two centrals-only
    fractions differ by 72%, and extending the satellites' range to
    ``lg M_* = 5`` adds 25% rather than 1.1%, because their low-mass slope is
    shallower: the truncation is then not a small correction.
    """

    PUB = dict(O.ZU15_PUBLISHED)

    def _sat(self, field, lo):
        from ggah_mod.sectors import galaxies as G
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        g = GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
        keep = (G.CSMF_LG_MSTAR_RANGE, G.CSMF_N_MSTAR)
        # Fixed dlg M_*, so a wider range cannot be confounded with a finer grid.
        G.CSMF_LG_MSTAR_RANGE = (lo, keep[0][1])
        G.CSMF_N_MSTAR = int(round((keep[0][1] - lo) / 0.0137)) + 1
        try:
            return g.stellar_fraction(field, self.PUB, satellites=True)
        finally:
            G.CSMF_LG_MSTAR_RANGE, G.CSMF_N_MSTAR = keep

    def _omega_star(self, field, f_cen, f_sat):
        split = MT.BaryonSplit.from_hot(F_B, jnp.zeros_like(field.m),
                                        f_star_cen=f_cen, f_star_sat=f_sat)
        c = census(field, split)
        return float(c.omega["star_cen"] + c.omega["star_sat"])

    def _f_cen_unselected(self, field):
        """Every halo hosts a central, whatever its stellar mass."""
        lm = jnp.log10(field.m)
        return jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                         - jnp.log10(field.cosmo.h)) / field.m

    def test_the_csmf_truncation_is_worth_about_one_per_cent(self, field):
        """The docstring calls the result a lower bound.  This says by how much.

        `CSMF_LG_MSTAR_RANGE` starts at the low end of the fit, so satellites
        below it are dropped.  Extending three and a half decades to
        `lg M_* = 5` adds 1.1%, with the increments falling geometrically --
        so the bound is tight and the constant can stay where it is.
        """
        _, near = self._sat(field, 8.5)
        _, far = self._sat(field, 5.0)
        a = self._omega_star(field, jnp.zeros_like(field.m), near)
        b = self._omega_star(field, jnp.zeros_like(field.m), far)
        assert b > a                            # it is a bound, and it is low
        assert (b / a - 1.0) < 0.02             # and it is tight

    def test_the_satellites_close_most_of_the_stellar_gap(self, field):
        """Against Devereux et al. (2024): centrals alone 1.08 low, and with the
        satellites 0.72 -- **high** by 1.38.

        It read 1.60 -> 1.07, "the satellites close the gap", while zu15's
        h^-2 Msun was divided by an Msun/h halo mass: every stellar fraction
        was a factor h low.  With one convention the centrals nearly close it
        alone and the satellites overshoot.  The name is kept so the history
        reads; the numbers are the consistent ones (2026-09-15)."""
        f_cen = self._f_cen_unselected(field)
        alone = self._omega_star(field, f_cen, jnp.zeros_like(field.m))
        _, f_sat = self._sat(field, 8.5)
        both = self._omega_star(field, f_cen, f_sat)
        assert DEV2024 / alone == pytest.approx(1.08, abs=0.05)
        assert DEV2024 / both == pytest.approx(0.72, abs=0.05)

    def test_the_two_centrals_only_fractions_are_not_the_same_quantity(
            self, field):
        r"""And this is why the split cannot simply be wired into the census.

        :meth:`GalaxySector.stellar_fraction` weights by the central occupation,
        so it drops halos whose central is fainter than
        ``log10m_star_thresh`` -- a *selected* sample.  The census's stellar row
        counts every halo's central, because for a cosmic density a faint
        central is still stars.  They differ by 28%, and the recorded
        disagreement (low by 1.60) is the unselected one.

        Passing ``satellites=True`` straight through would therefore add a
        satellite row and silently shrink the central row.  If this test ever
        fails because the two agree, the mismatch has been resolved and T7-7's
        S2 should say which way.
        """
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        g = GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
        selected, _ = g.stellar_fraction(field, self.PUB)
        a = self._omega_star(field, self._f_cen_unselected(field),
                             jnp.zeros_like(field.m))
        b = self._omega_star(field, selected, jnp.zeros_like(field.m))
        assert b < a
        assert (1.0 - b / a) == pytest.approx(0.28, abs=0.03)

    def test_the_central_fraction_is_proportional_to_a_completeness(self, field):
        r"""And this is why it is the wrong quantity for a mass budget.

        ``n_cen`` carries Zu & Mandelbaum's :math:`f_c` as a prefactor, so
        :meth:`stellar_fraction`'s central is proportional to it.  ``fc`` is a
        *completeness normalisation* -- ``occupation_params`` says so, and says
        their own fit allows it above 1 -- so this makes the stellar mass of a
        halo depend on how complete somebody's survey was.  ``ggah_cal`` samples
        it on (0.1, 3.0) and feeds the result to the lensing point mass.

        A halo's stellar mass cannot depend on a survey.  Whatever S2 decides,
        it has to decide this.
        """
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        g = GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
        base = self.PUB
        i = int(np.argmin(np.abs(np.log10(np.asarray(field.m)) - 13.0)))
        f = {}
        for fc in (0.1, 0.86, 3.0):
            cen, _ = g.stellar_fraction(field, dict(base, fc=fc))
            f[fc] = float(cen[i])
        # Strictly proportional: 30x across the box ggah_cal samples.
        assert f[3.0] / f[0.1] == pytest.approx(30.0, rel=1e-6)
        assert f[0.86] / f[0.1] == pytest.approx(8.6, rel=1e-6)
        # And it can exceed what the SHMR says the halo has.
        unselected = float(self._f_cen_unselected(field)[i])
        assert f[3.0] > unselected
