r"""Verification: the feedback budget, the baryon fractions, and mass conservation.

Three things are checked that the physics does not check itself:

1. **The saturation is smooth.**  The predecessor's hard ``minimum`` sits
   exactly on its tie at the fiducial parameters, which is where JAX splits a
   gradient 50/50 -- so the parameter its own documentation calls "a flat
   direction" was not flat, it was being discarded.
2. **The closure reaches the cosmology.**  It contains :math:`v_\Delta^2`,
   through the halo's boundary radius, so a forecast varying :math:`\Omega_m`
   or :math:`h` must see the baryon fraction move.
3. **The matter budget closes exactly**, and keeps closing when the gas is given
   a different profile from the dark matter -- which is the case the whole
   design exists for.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos import nfw_uk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import matter as MT
from ggah_mod.sectors import sham as SH
from conftest import AnalyticPk, STUB_CM_MODEL

F_B = PLANCK18.Omega_b / PLANCK18.Omega_m
ZM15 = (12.10, 10.31, 0.33, 0.42, 1.21)


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, AnalyticPk(), z=0.0,
                      cm_model=STUB_CM_MODEL)


def _split(field):
    """A `BaryonSplit` from the parameterised route, which is a *gas* fraction.

    `from_hot`, not `from_retained`: `f_gas_sigmoid` was calibrated against
    FLAMINGO, which measures gas.  Using the wrong constructor here would put
    `f_star` on the wrong side of the budget, which is the defect the two
    constructors exist to prevent -- so the choice is made once, here.
    """
    lm = jnp.log10(field.m)
    f_gas = E.f_gas_sigmoid(lm, F_B)
    f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                       - jnp.log10(field.cosmo.h)) / field.m
    return MT.BaryonSplit.from_hot(F_B, f_gas, f_star_cen=f_star)


class TestSoftSaturation:
    """The replacement for a `minimum` that the model sits on."""

    def test_it_is_exact_where_nothing_is_saturating(self):
        """x << c must be x, or the unsaturated regime has been distorted to
        buy smoothness in the saturated one.

        The deviation is exactly the documented `O(x/2c)`: asserted as that,
        rather than against a flat tolerance, so a change in the functional
        form fails here instead of being absorbed.
        """
        c = 0.15
        for x in (1e-6, 1e-5, 1e-4, 1e-3):
            got = float(E.soft_saturate(x, c))
            assert got == pytest.approx(x, rel=1.01 * x / (2 * c))
            assert got < x                       # approaches from below

    def test_the_leading_correction_is_the_documented_one(self):
        """`c[1 - e^{-x/c}] = x - x^2/2c + O(x^3)`."""
        c, x = 0.15, 1e-3
        got = float(E.soft_saturate(x, c))
        assert (x - got) == pytest.approx(x ** 2 / (2 * c), rel=1e-2)

    def test_it_approaches_the_ceiling_from_below_and_never_exceeds_it(self):
        c = 0.15
        x = jnp.asarray([0.01, 0.1, 1.0, 10.0, 1e3])
        got = np.asarray(E.soft_saturate(x, c))
        # Never above the ceiling.  `<=` rather than `<`: beyond x/c ~ 36 the
        # gap c*exp(-x/c) drops below float64's relative resolution and the
        # ceiling is attained exactly.  That is the arithmetic, not the model,
        # and it is the safe direction -- what matters is that a baryon
        # fraction never goes negative, which `<=` guarantees.
        assert np.all(got <= c)
        assert np.all(got[:3] < c)               # strictly below, where resolvable
        assert np.all(np.diff(got) >= 0)
        assert got[-1] == pytest.approx(c, rel=1e-9)

    def test_the_approach_is_exponential_where_it_is_resolvable(self):
        """`c - f(x) = c e^{-x/c}`, which is what makes it C-infinity rather
        than a ramp that flattens."""
        c = 0.15
        for x in (0.1, 0.3, 0.6):
            gap = c - float(E.soft_saturate(x, c))
            assert gap == pytest.approx(c * np.exp(-x / c), rel=1e-9)

    def test_its_gradient_never_dies(self):
        """Deep in saturation the hard version gives exactly 0."""
        c = 0.15
        g_soft = float(jax.grad(lambda v: E.soft_saturate(v, c))(5.0))
        g_hard = float(jax.grad(lambda v: jnp.minimum(v, c))(5.0))
        assert g_soft > 0.0
        assert g_hard == 0.0

    def test_the_hard_version_splits_its_gradient_at_the_tie(self):
        """The 50/50 tie, demonstrated rather than cited.

        This is why the replacement is not cosmetic: at the fiducial the AGN
        channel saturates, so the model sits *on* this point.
        """
        c = 0.15
        assert float(jax.grad(lambda v: jnp.minimum(v, c))(c)) == 0.5


class TestTheEnergyClosure:
    @staticmethod
    def _inputs(field):
        """``(m_star, m_bh)``.

        `m_bh` from the same `m_star`, at the published M_BH-M_* relation, so
        both channels -- the stars and the black hole -- are fed by one stellar
        mass rather than two.  That shared quantity is the only genuine coupling
        the closure has.
        """
        lm = jnp.log10(field.m)
        # zu15 is h^-2 Msun: Msun/h for the closure, physical for Powell.
        lg_native = SH.mstar_from_mh_zu15(lm, *ZM15)
        log10h = jnp.log10(field.cosmo.h)
        m_star = jnp.power(10.0, lg_native - log10h)
        m_bh = jnp.power(10.0, 7.76 + 0.67 * (lg_native - 2.0 * log10h - 11.0))
        return m_star, m_bh

    def test_it_stays_inside_the_baryon_budget(self, field):
        m_star, m_bh = self._inputs(field)
        f = np.asarray(E.f_retained_energy(field.m, 0.0, PLANCK18, m_star,
                                           F_B, m_bh=m_bh))
        assert np.all(f > 0.01) and np.all(f <= F_B)

    def test_more_feedback_removes_more_gas(self, field):
        m_star, m_bh = self._inputs(field)
        weak = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                   m_bh=m_bh, log10_eps_agn=-4.0)
        strong = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                     m_bh=m_bh, log10_eps_agn=-1.0)
        assert np.all(np.asarray(strong) < np.asarray(weak))

    def test_massive_halos_hold_on_to_their_gas(self, field):
        """E_bind grows as M^{5/3} while the black-hole mass grows more slowly
        than M, so the budget must lose at the cluster scale.  That direction
        is the physical content of the closure, not a fitted trend."""
        m_star, m_bh = self._inputs(field)
        f = np.asarray(E.f_retained_energy(field.m, 0.0, PLANCK18, m_star,
                                           F_B, m_bh=m_bh))
        assert f[-1] > f[0]
        assert f[-1] == pytest.approx(F_B, rel=1e-2)

    @pytest.mark.x64
    @pytest.mark.parametrize("key,val", [("log10_eps_agn", -2.0),
                                         ("eps_radiative", 0.1),
                                         ("eps_sn", 0.1)])
    def test_every_coupling_parameter_is_differentiable(self, field, key, val):
        m_star, m_bh = self._inputs(field)

        def f(v):
            return jnp.sum(E.f_retained_energy(
                field.m, 0.0, PLANCK18, m_star, F_B, m_bh=m_bh, **{key: v}))
        ad = float(jax.grad(f)(val))
        fd = float((f(val + 1e-6) - f(val - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5), key

    @pytest.mark.x64
    @pytest.mark.parametrize("param", ["Omega_m", "h"])
    def test_the_closure_reaches_the_cosmology(self, field, param):
        """Through v_Delta^2, whose boundary radius carries the background,
        and for h through the black-hole mass, which the closure converts from
        physical Msun to the Msun/h of the halo and stellar masses.  That h was
        missing until 0.9.0: the derivative was exactly zero, and E_AGN 1/h too
        large.

        **At z = 0.5, and the redshift is load-bearing.**  Omega_m enters only
        through E(z), and E(0) = 1 *identically* in this package -- flatness
        plus Omega_m carrying the neutrinos.  So at z = 0 the derivative is an
        exact structural zero for a correct reason, and testing there would
        assert the wrong thing about a right answer.
        """
        m_star, m_bh = self._inputs(field)

        def f(v):
            c = PLANCK18.replace(**{param: v})
            return jnp.sum(E.f_retained_energy(field.m, 0.5, c, m_star, F_B,
                                               m_bh=m_bh))
        x = float(getattr(PLANCK18, param))
        ad = float(jax.grad(f)(x))
        fd = float((f(x + 1e-6) - f(x - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4), param

    def test_at_z_zero_omega_m_enters_only_through_the_boundary(self, field):
        r"""This test used to assert the derivative was exactly zero, and the
        reasoning was sound while the binding energy was anchored to a
        *critical* overdensity: :math:`E(0) = 1` identically, so nothing at
        :math:`z = 0` could depend on :math:`\Omega_{\rm m}`.

        The package now carries one mass definition, and the shipped one is
        :math:`200\rho_{\rm m}`.  A mean-density boundary depends on
        :math:`\Omega_{\rm m}` at every redshift, :math:`z = 0` included ---
        :math:`\bar\rho_{\rm m} = \Omega_{\rm m}\rho_{\rm c}` --- so the
        derivative is no longer zero and should not be.  What is asserted
        instead is that it enters *only* that way: freezing the boundary
        restores the exact zero.
        """
        m_star, m_bh = self._inputs(field)

        def with_mdef(mdef):
            return jax.grad(lambda v: jnp.sum(E.f_retained_energy(
                field.m, 0.0, PLANCK18.replace(Omega_m=v), m_star, F_B,
                m_bh=m_bh, mdef=mdef)))(PLANCK18.Omega_m)

        assert float(with_mdef("200c")) == 0.0, (
            "at a critical-overdensity boundary E(0) = 1 leaves no route for "
            "Omega_m, so this must still be exactly zero")
        assert float(with_mdef("200m")) != 0.0, (
            "a mean-density boundary is proportional to Omega_m, so the "
            "derivative cannot vanish")

    def test_v200_uses_a_proper_radius(self, field):
        """Comoving would understate it by (1+z): zero at z = 0, 100% at z = 1
        -- and z = 0 is where the predecessor checked."""
        v0 = float(E.v200_squared(1e14, 0.0, PLANCK18))
        v1 = float(E.v200_squared(1e14, 1.0, PLANCK18))
        assert v1 > v0
        assert 100.0 < np.sqrt(v0) < 2000.0

    def test_g_is_consistent_with_the_critical_density(self):
        """Derived from RHO_CRIT0, not taken from a second source."""
        assert float(E.G_MPC_KMS2_MSUN) == pytest.approx(4.30091e-9, rel=1e-4)


class TestParameterisedBaryonFractions:
    def test_the_registry_and_its_table_agree(self):
        assert set(E.F_GAS) == set(E.F_GAS_CALIBRATION)

    @pytest.mark.parametrize("name", sorted(E.F_GAS))
    def test_bounded_by_the_cosmic_fraction(self, name, field):
        f = np.asarray(E.make_f_gas(name)(jnp.log10(field.m), F_B))
        assert np.all(f >= 0.0)
        assert np.all(f <= F_B + 1e-9), f"{name}: max = {np.max(f)}"

    @pytest.mark.parametrize("name", sorted(E.F_GAS))
    def test_massive_halos_are_closer_to_the_cosmic_fraction(self, name, field):
        f = np.asarray(E.make_f_gas(name)(jnp.log10(field.m), F_B))
        assert f[-1] > f[0]

    def test_make_f_gas_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unknown baryon-fraction"):
            E.make_f_gas("nope")

    def test_the_gas_concentration_factor_only_puffs_out(self, field):
        eta = np.asarray(E.gas_concentration_factor(jnp.log10(field.m)))
        assert np.all((eta > 0.0) & (eta <= 1.0))
        assert np.all(np.diff(eta) > 0)          # massive halos less affected

    def test_the_wind_loading_is_the_identity_at_its_fiducial(self, field):
        """eta_w_norm = 0 must be *exactly* off, so switching it on is a strict
        extension rather than a different model."""
        w = E.wind_mass_loading(field.m, 0.0, PLANCK18, eta_w_norm=0.0)
        assert float(jnp.max(jnp.abs(w))) == 0.0


class TestMassConservation:
    """The layer's strongest invariant: exact algebra, not a tolerance."""

    def test_the_budget_closes(self, field):
        w = MT.matter_weights(field, _split(field))
        res = np.abs(np.asarray(
            MT.MatterField.mass_conservation_residual(field, w)))
        assert np.max(res) < 1e-6

    def test_it_still_closes_when_the_gas_has_its_own_profile(self, field):
        """The case the design exists for: feedback puffs the gas out, and the
        total must not notice."""
        c_redist = field.conc * E.gas_concentration_factor(jnp.log10(field.m))
        u_gas = nfw_uk(field.k, field.r_delta / c_redist, c_redist)
        assert float(jnp.max(jnp.abs(u_gas - field.u_nfw()))) > 0.01
        w = MT.matter_weights(field, _split(field), u_gas=u_gas)
        res = np.abs(np.asarray(
            MT.MatterField.mass_conservation_residual(field, w)))
        assert np.max(res) < 1e-6

    def test_the_six_fractions_sum_to_one(self, field):
        res = np.abs(np.asarray(_split(field).residual()))
        assert np.max(res) < 5e-16, "the split is algebra, not a tolerance"

    def test_the_ejected_component_carries_what_cdm_used_to_absorb(self, field):
        """The reason the split went from three parts to six.

        `f_ejected` is what the old `f_cdm = 1 - f_gas - f_star` contained
        beyond the actual dark matter, and it rode `u_DM`.  It is not a small
        number: at the parameterised route's fiducial most of the baryon budget
        is outside the halo, which is the whole content of the census.
        """
        split = _split(field)
        assert float(jnp.max(split.f_ejected)) > 0.1 * F_B
        old_f_cdm = 1.0 - split.f_hot - split.f_star_cen
        # `atol` as well as `rtol`: the difference crosses zero near
        # log10 M = 13.4, where a relative tolerance measures round-off against
        # a vanishing denominator rather than measuring the identity.
        np.testing.assert_allclose(
            np.asarray(old_f_cdm - split.f_collisionless),
            np.asarray(split.f_ejected), rtol=1e-12, atol=1e-15)

    def test_the_matter_tracer_is_continuous(self, field):
        """It is a field, not a countable population, so its one-halo
        auto-spectrum has no self-pair to exclude."""
        assert MT.matter_weights(field, _split(field)).discrete is False

    def test_stars_are_a_point_mass(self, field):
        """u_star = 1 at every k, exactly -- a galaxy is unresolved on halo
        scales, so this is not an approximation."""
        split = _split(field)
        w = MT.matter_weights(field, split)
        expected = np.asarray(field.m / field.rho_matter * split.f_star_cen)
        np.testing.assert_allclose(np.asarray(w.w_point), expected, rtol=1e-14)

    def test_it_normalises_against_the_total_matter_density(self, field):
        """Not rho_cold, which is 0.46% smaller and is what halos form from."""
        w = MT.matter_weights(field, _split(field))
        got = float(w.at_large_scales()[0] * field.rho_matter / field.m[0])
        assert got == pytest.approx(1.0, abs=1e-6)
        wrong = float(w.at_large_scales()[0] * field.rho_cold / field.m[0])
        assert abs(wrong - 1.0) > 1e-3          # the confusion is measurable

    def test_f_ejected_is_not_clipped(self):
        """A clip would give a plausible number with a dead gradient.  Giving a
        halo more baryons than exist must produce a negative, which is visible.

        The old version asserted this of `f_cdm`, where an overdraft came out
        of the *dark matter* -- so the symptom was a wrong collisionless
        fraction rather than a wrong baryon one, and nothing named it."""
        over = MT.BaryonSplit.from_hot(0.2, 0.9, f_star_cen=0.3)
        assert float(over.f_ejected[0]) == pytest.approx(-1.0)
        assert float(over.worst_overdraft()) == pytest.approx(1.0)

    def test_the_collisionless_share_is_not_the_cdm_share(self):
        """It carries the neutrinos too, because this module normalises by
        rho_matter.  Naming it `f_cdm` put that 0.46% into a symbol that reads
        as if it had been thought about."""
        f_b = PLANCK18.Omega_b / PLANCK18.Omega_m
        got = float(MT.f_collisionless(f_b))
        cdm_only = float(PLANCK18.Omega_cdm / PLANCK18.Omega_m)
        assert got == pytest.approx(1.0 - f_b)
        assert abs(got - cdm_only) > 1e-3

    @pytest.mark.x64
    def test_the_weights_reach_the_cosmology(self, field):
        pk = AnalyticPk()

        def f(om):
            c = PLANCK18.replace(Omega_m=om)
            fl = make_field(c, DIFFERENTIABLE, pk, z=0.0,
                            cm_model=STUB_CM_MODEL)
            lm = jnp.log10(fl.m)
            fb = c.Omega_b / c.Omega_m
            fg = E.f_gas_sigmoid(lm, fb)
            fs = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                           - jnp.log10(c.h)) / fl.m
            split = MT.BaryonSplit.from_hot(fb, fg, f_star_cen=fs)
            return jnp.log(jnp.sum(MT.matter_weights(fl, split).total()))

        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        fd = float((f(PLANCK18.Omega_m + 1e-6)
                    - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)


class TestTheFourRepairs:
    r"""What the closure's own case requires, measured one repair at a time.

    Its thesis is that an X-ray measurement of the AGN sector constrains the
    lensing suppression.  At the published couplings the AGN channel was 1-3%
    of the supernova one at every mass, so the duty cycle the argument turns on
    had no leverage and the thesis was untestable.  Three of these repairs move
    that; the fourth ties the ejection radius to the energy that made it.

    The first repair, a duty cycle applied once rather than twice, was a
    property of the luminosity channel and went with it in 0.8.8; the
    numbers below that compare with that channel are kept as the record of
    what the Soltan form changed.
    """

    @staticmethod
    def _inputs(field):
        """``(lm, m_star, m_bh)``: the **AGN sector's own** black-hole mass.

        Not the stub relation the tests above use: the whole claim here is
        about the size of the AGN channel at the published parameters, so
        feeding it an invented mass would measure the invention.  It comes from
        the stellar mass through one chain, which is the coupling the closure's
        case rests on.
        """
        from ggah_mod.sectors import agn as A

        lm = jnp.log10(field.m)
        m_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)
                           - jnp.log10(field.cosmo.h))
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        gal, gp = GalaxySector("zumandelbaum15"), galaxy_defaults("zumandelbaum15")
        # The published AGN parameters, which the claim is about; the 0.8.7
        # defaults (the AGN mock MAP) give 0.28 at 1e13 instead of 0.80.
        sec, p = A.AgnSector(gal), A.AgnParams(**A.AGN_PUBLISHED)
        return lm, m_star, sec.mean_mbh(lm, p, gp, h=field.cosmo.h)

    def test_repair_2_the_soltan_channel_dominates_where_the_other_did_not(
            self, field):
        """The substitution the predecessor's own comment describes and its
        code did not make: a coupling fraction of M_BH c^2, not a present
        luminosity extrapolated over a Hubble time.

        Measured at 1e13 Msun/h it moves E_AGN/E_SN from 0.023, through the
        luminosity form at the predecessor's settings, to 0.35 here -- a factor
        15 -- and at 1e11 by a factor of a few hundred.  The AGN channel is not
        intrinsically weak; the luminosity form made it weak.  That form is
        gone since 0.8.8, so what is asserted is the Soltan side.  It read 0.52
        until 0.9.0 put the black-hole mass in Msun/h: E_AGN was 1/h too
        large against the supernovae.
        """
        lm, m_star, m_bh = self._inputs(field)
        e_sn_new = E.e_supernova(m_star, 0.1)
        # m_star is Msun/h and m_bh physical Msun: one unit for both energies.
        new = E.e_agn_soltan(m_bh * field.cosmo.h) / e_sn_new
        at13 = lambda a: float(jnp.interp(13.0, lm, a))
        assert at13(new) > 0.2, "ten times the luminosity form's 0.023"

    def test_repair_3_supernovae_count_the_mass_formed(self, field):
        """M_* is the surviving mass; supernova counts scale with the mass
        formed, which is larger by 1/(1-R) = 1.67.  The correction was simply
        absent."""
        _, m_star, _ = self._inputs(field)
        surviving = E.e_supernova(m_star, 0.1, return_fraction=0.0)
        formed = E.e_supernova(m_star, 0.1)
        r = float(jnp.max(formed / surviving))
        assert r == pytest.approx(1.0 / (1.0 - E.RETURN_FRACTION), rel=1e-12)
        assert r == pytest.approx(1.667, abs=0.01)

    def test_the_return_fraction_is_not_the_remnant_fraction(self):
        """Two numbers about stellar mass, with different denominators, and
        one symbol away from being confused into one."""
        from ggah_mod.sectors.sham import REMNANT_FRACTION
        assert E.RETURN_FRACTION != REMNANT_FRACTION

    def test_repair_4_a_finite_ejection_radius_costs_less(self, field):
        """`Delta f_b M v^2` is the work to unbind to infinity.  Moving the gas
        to eta_ej R_Delta costs (1 - 1/eta_ej) of that in a point-mass
        potential -- half, at the ejecta sector's fiducial radius."""
        to_inf = E.binding_energy(field.m, 0.0, PLANCK18, 1.0)
        to_2r = E.binding_energy(field.m, 0.0, PLANCK18, 1.0, eta_ej=2.0)
        np.testing.assert_allclose(np.asarray(to_2r / to_inf), 0.5, rtol=1e-12)

    def test_a_cheaper_ejection_expels_more_gas(self, field):
        """Which is the point of tying the two: the radius the ejecta sector
        puts the gas at is no longer free of the energy that put it there."""
        _, m_star, m_bh = self._inputs(field)
        # `eta_ej=None` is now explicit: since PLAN.md item C4 the *default*
        # is the ejecta sector's own 2.0, so the unbind-to-infinity case has to
        # be asked for.  That is the point of C4 -- the closure charges for the
        # move the profile actually makes -- and leaving this implicit would
        # have made the contrast below compare the default with itself.
        far = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                  m_bh=m_bh, eta_ej=None)
        near = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                   m_bh=m_bh, eta_ej=2.0)
        assert np.all(np.asarray(near) <= np.asarray(far))
        assert float(jnp.min(far - near)) >= 0.0
        assert float(jnp.max(far - near)) > 1e-3

    def test_the_shipped_default_now_charges_for_the_move_it_makes(self, field):
        r"""PLAN.md item **C4**, and it moves numbers.

        ``f_retained_energy`` defaulted to ``eta_ej=None`` -- the work to unbind
        the gas *to infinity* -- while :class:`~ggah_mod.sectors.ejecta.EjectaParams`
        put that same gas at :math:`2R_\Delta`.  One quantity, two values, and
        the closure was charging for a move the profile did not make.  They are
        one ``Param`` object now, so the default is 2.0 and the bracket is
        :math:`1 - 1/\eta_{\rm ej} = 0.5`: half the energy per unit mass, so
        roughly twice the mass expelled for the same budget.

        Measured, ``f_retained/f_b`` against the old default:

        ==============  ==========  ============  =====
        ``log10 M``     infinity    eta_ej = 2    ratio
        ==============  ==========  ============  =====
        11              0.500       0.267         0.53
        12              0.332       0.140         0.42
        13              0.864       0.748         0.87
        15              0.999       0.999         1.00
        ==============  ==========  ============  =====

        The dip is at :math:`10^{12}`, not at the low-mass end: the stellar mass
        that feeds both channels comes from a real SHMR, which turns over.  An
        earlier version of this table was measured with a stand-in
        ``m_star = 0.01 M`` and reported a monotonic curve -- the numbers were
        of the stand-in, not of the model, and ``_inputs`` above exists exactly
        so that a measurement of this closure is not a measurement of an
        invented input.

        Recorded here rather than only in the commit, because it is a
        shipped-default change and the next reader of a group-scale ``f_gas``
        deserves to find it.
        """
        _, m_star, m_bh = self._inputs(field)
        from ggah_mod.sectors.ejecta import ETA_EJ

        kw = dict(m_bh=m_bh)
        default = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B, **kw)
        explicit = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                       eta_ej=ETA_EJ.default, **kw)
        assert np.allclose(np.asarray(default), np.asarray(explicit))

        infinity = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                       eta_ej=None, **kw)
        assert np.all(np.asarray(default) <= np.asarray(infinity))

    def test_the_container_supplies_what_the_keywords_do_not(self, field):
        """A container nothing reads is the defect ``wind_mass_loading``
        carries; this is the check that ``EnergeticsParams`` is not that."""
        _, m_star, m_bh = self._inputs(field)
        strong = E.EnergeticsParams(log10_eps_agn=-1.0)
        via_container = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star,
                                            F_B, m_bh=m_bh, params=strong)
        via_keyword = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                          m_bh=m_bh, log10_eps_agn=-1.0)
        assert np.allclose(np.asarray(via_container), np.asarray(via_keyword))
        # ...and an explicit keyword still overrides the container.
        override = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                       m_bh=m_bh, params=strong,
                                       log10_eps_agn=-2.0)
        assert not np.allclose(np.asarray(override), np.asarray(via_container))

    def test_the_closure_refuses_a_luminosity(self, field):
        """A rate is not an integrated history, and converting one to the other
        is exactly the substitution the Soltan form undoes.  Since 0.8.8 there
        is no luminosity channel to ask for, so the keywords are unknown."""
        _, m_star, _ = self._inputs(field)
        with pytest.raises(ValueError, match="needs `m_bh`"):
            E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B)
        for kw in ({"l_x_agn": jnp.ones_like(field.m)},
                   {"agn_channel": "luminosity"}):
            with pytest.raises(TypeError, match=next(iter(kw))):
                E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B, **kw)

    def test_the_repaired_closure_expels_gas_where_feedback_should(self, field):
        """Published: f_ret/f_b dipped only to 0.76 near 1e12 and was back at
        cosmic by 1e14, through the luminosity form removed in 0.8.8.
        Repaired: 0.14 at 1e12, which is the order the FLAMINGO-calibrated
        sigmoid asks for -- so the two routes to the baryon fraction stop being
        a factor of ten apart for a reason other than calibration."""
        lm, m_star, m_bh = self._inputs(field)
        new = E.f_retained_energy(field.m, 0.0, PLANCK18, m_star, F_B,
                                  m_bh=m_bh, eta_ej=2.0) / F_B
        at12 = lambda a: float(jnp.interp(12.0, lm, a))
        assert at12(new) < 0.3
        # and both still recover at the cluster scale, where E_bind wins
        assert float(jnp.interp(15.0, lm, new)) > 0.9

    @pytest.mark.x64
    def test_the_soltan_channel_is_differentiable_in_its_efficiency(self,
                                                                    field):
        _, m_star, m_bh = self._inputs(field)

        def f(eps_r):
            return jnp.sum(E.f_retained_energy(
                field.m, 0.0, PLANCK18, m_star, F_B, m_bh=m_bh,
                eps_radiative=eps_r))

        ad = float(jax.grad(f)(0.1))
        fd = float((f(0.1 + 1e-7) - f(0.1 - 1e-7)) / 2e-7)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_ejection_radius_carries_a_gradient_into_the_budget(self,
                                                                    field):
        """Repair 4's real consequence: eta_ej now reaches f_retained, so the
        ejecta profile's radius and the mass it carries are one prediction."""
        _, m_star, m_bh = self._inputs(field)

        def f(eta):
            return jnp.sum(E.f_retained_energy(
                field.m, 0.0, PLANCK18, m_star, F_B, m_bh=m_bh, eta_ej=eta))

        ad = float(jax.grad(f)(2.0))
        fd = float((f(2.0 + 1e-6) - f(2.0 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5)


def test_the_retained_floor_is_in_halo_mass_units_not_a_fraction_of_f_b():
    r"""Which its own ``why`` got wrong, and nothing would have caught.

    ``expelled = soft_saturate(..., f_b_cosmic - f_retained_min)``, so the floor
    is in the same units as :math:`f_b^{\rm cosmic}` --- a fraction of the halo
    mass. At Planck 2018's :math:`f_b = 0.159` the default 0.01 is 6.3% of
    :math:`f_b` and not 1% of it, a factor of 6.3 for a reader who took the
    parameter at its old description.

    Measured in the corner where the floor is what decides: at a coupling far
    past saturation the retained fraction lands on the floor itself, to float
    precision, and nowhere near ``floor * f_b``.
    """
    import jax.numpy as jnp
    from ggah_mod.cosmology import PLANCK18
    from ggah_mod.sectors import energetics as E

    f_b = float(PLANCK18.Omega_b / PLANCK18.Omega_m)
    f_star = 1e-3
    for floor in (0.01, 0.02, 0.05):
        ret = float(E.f_retained_energy(
            jnp.asarray([1e11]), 0.0, PLANCK18, jnp.asarray([f_star * 1e11]),
            f_b, m_bh=jnp.asarray([1e12]), f_retained_min=floor)[0])
        # Stars cannot be expelled, so the floor sits on top of them.
        assert ret == pytest.approx(floor + f_star, rel=1e-9), \
            "the floor is in halo-mass units"
        assert ret != pytest.approx(floor * f_b + f_star, rel=1e-3), \
            "and is not a fraction of f_b, which is what the why used to say"


def test_stars_cannot_be_expelled():
    """Far past saturation the retained baryons are the stars plus the floor,
    at every stellar fraction -- the cap is the non-stellar share."""
    import jax.numpy as jnp
    from ggah_mod.cosmology import PLANCK18
    from ggah_mod.sectors import energetics as E
    from ggah_mod.sectors.matter import cosmic_baryon_fraction

    f_b = float(cosmic_baryon_fraction(PLANCK18))
    m = jnp.full(4, 1e11)
    f_star = jnp.asarray([0.0, 0.01, 0.05, 0.12])
    ret = np.asarray(E.f_retained_energy(m, 0.0, PLANCK18, f_star * m, f_b,
                                         m_bh=jnp.full(4, 1e12),
                                         f_retained_min=0.01))
    assert np.allclose(ret, np.asarray(f_star) + 0.01, rtol=1e-9)
    assert np.all(ret - np.asarray(f_star) > 0.0)
