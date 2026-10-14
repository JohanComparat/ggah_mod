r"""Verification: the AGN chain, and the luminosity function it predicts.

The chain's distinguishing property is that :math:`\Phi(L_X)` is an **output**.
An abundance-matched model reproduces a measured XLF by construction and so
cannot be tested against one; this can, and is.

What that test can and cannot say
---------------------------------

At the *published central* parameters the predicted XLF tracks Aird et al.
(2015) to ~10% at the faint end and runs high by a factor of a few at
:math:`\log L_X \sim 44.3`, with a shallower slope.  That is a statement about
the parameters, not about the transcription -- Powell's own analysis fits them.
So the tests here assert the two things that *are* properties of the code:

* the predicted XLF is a falling, smooth function of the right order over the
  range the model is used on;
* the chain matches the reference implementation's kernel, which is checked in
  ``tests/parity/`` to 1e-9 of peak.
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
from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
from ggah_mod.sectors.occupation import ZU15_PUBLISHED

Z = 0.135

#: The AGN chain starts from a threshold occupation's stellar masses.
GAL = GalaxySector("zumandelbaum15")
GP = galaxy_defaults("zumandelbaum15")
#: Paper I's iHOD, the galaxy default until 0.8.4: where an audit below was
#: derived, it is kept there rather than re-derived against another input.
GP_PUBLISHED = dict(GP, **ZU15_PUBLISHED)

#: Both stellar-mass slopes at zero: the constant duty cycle and the
#: mass-independent Eddington-ratio break.
NEUTRAL = dict(log10_ferdf_mslope=0.0, log10_lstar_mslope=0.0)
#: The AgnParams defaults of 1.1.0.dev2-1.2.0, the eROSITA c030 MAP, at the
#: catalogue's photon index: the configuration a test pins where its number is
#: a property of that configuration rather than of the defaults.
AGN_C030 = dict(mu_bh=7.954, al_bh=0.842, sig_bh=0.280, log10_lstar=-0.798,
                delta1=0.232, delta2=3.749, log10_ferdf=-1.640,
                f_duty_sat=0.296, gamma_x=A.GAMMA_X, **NEUTRAL)
#: The obscured fraction of Comparat et al. (2019) as published, no halo-mass
#: tilt: the ObscurationParams defaults up to 1.2.0.
C19 = A.ObscurationParams(**A.OBSCURATION_PUBLISHED)


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                      make_pk("emu_pk"), z=Z)


@pytest.fixture(scope="module")
def sector():
    return A.AgnSector(GAL)


class TestTheChain:
    def test_black_hole_mass_rises_with_halo_mass(self, sector):
        lm = jnp.asarray(np.linspace(11.0, 15.0, 9))
        mbh = np.asarray(sector.mean_log10_mbh(lm, A.AgnParams(), GP, h=PLANCK18.h))
        assert np.all(np.diff(mbh) > 0)
        assert 5.0 < mbh[0] and mbh[-1] < 11.0

    def test_the_kernel_is_a_normalised_density(self, field):
        """Each row is dP/dlog L_X and must integrate to the active fraction
        -- except where the luminosity runs off the grid, which the low-mass
        end genuinely does.  The grid starts at 10^36 erg/s, at the default
        spacing of 0.02 dex: at the defaults the faint tail of massive
        centrals reaches below the default 10^39, which holds 0.9994 of the
        kernel at 10^16 Msun/h."""
        sector = A.AgnSector(GAL, loglx_min=36.0, n_lx=550)
        p = A.AgnParams()
        prob = np.asarray(sector.p_loglx_given_m(jnp.log10(field.m), p, GP, h=PLANCK18.h))
        norm = np.trapezoid(prob, np.asarray(sector.loglx), axis=1)
        assert norm[len(norm) // 2:].min() > 0.99
        assert norm.max() < 1.0 + 1e-6

    def test_rho_shrinks_the_scatter(self, sector):
        """Powell's 'Model 2': part of the M_* scatter aligning with halo mass
        *narrows* M_BH at fixed halo mass."""
        wide = float(sector._sigma_lm(A.AgnParams(rho=0.0), 0.25))
        narrow = float(sector._sigma_lm(A.AgnParams(rho=0.9), 0.25))
        assert narrow < wide

    def test_one_erdf_is_convolved_with_every_width(self, sector):
        """Shift invariance is what makes the chain cheap: one ERDF, convolved
        with the Gaussian of each halo mass's width in one product.  A row of
        the batched kernel is the kernel of that width alone."""
        p = A.AgnParams()
        sig = jnp.asarray([0.3, 0.6, 0.9])
        t, k = sector._kernel(p, sig)
        assert k.shape[0] == 3 and k.ndim == 2
        for i, s1 in enumerate(np.asarray(sig)):
            t1, k1 = sector._kernel(p, float(s1))
            assert k1.ndim == 1
            np.testing.assert_array_equal(np.asarray(t1), np.asarray(t))
            np.testing.assert_allclose(np.asarray(k[i]), np.asarray(k1),
                                       rtol=1e-12, atol=1e-14 * float(k1.max()))

    def test_the_stellar_mass_scatter_is_the_galaxy_sectors(self, sector, field):
        """The central's sigma_M*(M_h) is the galaxy sector's
        sigma_lnM*(M_h)/ln 10, Zu & Mandelbaum (2015) Eq. 20."""
        from ggah_mod.sectors.sham import scatter_zu15
        lm = jnp.log10(field.m)
        got = np.asarray(sector.sigma_mstar(lm, GP, h=field.cosmo.h))
        want = np.asarray(scatter_zu15(lm, GP["sigma_lnmstar"], GP["eta"],
                                       GP["lg_m1h"])) / np.log(10.0)
        np.testing.assert_allclose(got, want, rtol=1e-14)
        assert got.max() == pytest.approx(GP["sigma_lnmstar"] / np.log(10.0))

    @pytest.mark.x64
    def test_the_gradient_reaches_the_galaxy_scatter(self, sector, field):
        p = A.AgnParams()

        def f(v):
            return jnp.sum(sector.l_x_agn(field, p, dict(GP, sigma_lnmstar=v)))

        x = float(GP["sigma_lnmstar"])
        ad = float(jax.grad(f)(x))
        fd = float((f(x + 1e-6) - f(x - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)


class TestThePredictedLuminosityFunction:
    def test_it_falls_and_is_smooth(self, sector, field):
        grid, phi = sector.xlf(field, A.AgnParams(), GP)
        grid, phi = np.asarray(grid), np.asarray(phi)
        sel = (grid > 42.0) & (grid < 45.5)
        assert np.all(np.diff(phi[sel]) < 0)
        assert np.all(phi[sel] > 0)

    def test_it_is_the_right_order_against_a_measured_xlf(self, sector, field):
        """Order of magnitude, over the range the sample is selected on.

        Not a tight tolerance: `log10_ferdf` is a free normalisation and the
        published central parameters are not a fit to this HMF.  What would
        fail here is a units error or a wrong bolometric correction.
        """
        grid, phi = sector.xlf(field, A.AgnParams(), GP)
        grid, phi = np.asarray(grid), np.asarray(phi)
        meas = np.asarray(A.xlf_in_h_units(A.xlf_aird15, grid, Z, PLANCK18.h))
        sel = (grid > 42.0) & (grid < 43.5) & (phi > 0)
        ratio = phi[sel] / meas[sel]
        assert 0.2 < np.median(ratio) < 5.0

    def test_the_soft_band_is_a_shift(self, sector, field):
        p = A.AgnParams()
        hard, _ = sector.xlf(field, p, GP, band="hard")
        soft, _ = sector.xlf(field, p, GP, band="soft")
        np.testing.assert_allclose(np.asarray(soft - hard),
                                   float(jnp.log10(p.k_h2s)), rtol=1e-12)


class TestXlfFits:
    def test_the_registry(self):
        assert set(A.XLF) == {"aird15", "ueda14"}
        with pytest.raises(ValueError, match="unknown XLF"):
            A.make_xlf("nope")

    @pytest.mark.parametrize("name", sorted(A.XLF))
    def test_falls_with_luminosity_and_is_differentiable(self, name):
        fn = A.make_xlf(name)
        lx = jnp.asarray(np.linspace(42.0, 46.0, 20))
        v = np.asarray(fn(lx, 0.5))
        assert np.all(np.diff(v) < 0) and np.all(v > 0)
        g = float(jax.grad(lambda z: jnp.sum(fn(lx, z)))(0.5))
        assert np.isfinite(g) and g != 0.0

    def test_the_two_fits_agree_in_order_at_the_knee(self):
        """They are fits to overlapping data; disagreeing by decades would
        mean one is transcribed wrong."""
        for z in (0.1, 0.5, 1.0):
            a = float(A.xlf_aird15(44.0, z))
            u = float(A.xlf_ueda14(44.0, z))
            assert 0.1 < a / u < 10.0, z

    def test_the_aird_break_is_at_the_transition_redshift(self):
        """Aird et al. (2015) Eq. 38, against the defect it replaced.

        Both exponents are applied as tabulated and Table 5 gives
        `p2 = -2.12`, so `p1 > 0 > p2` and the two powers of
        `(1+zc)/(1+z)` pull opposite ways -- "a different evolution of L_*
        above and below a transition redshift". `L_*` peaks near `zc`.

        This used to raise the second power to `-p2` on top of that negative
        default. Two same-signed powers of one base sum to something
        monotone, so there was no break and `zc` was a scale factor.
        """
        zf = np.linspace(0.0, 6.0, 2001)
        lg_ls = np.array([44.84 - np.log10(
            ((1.0 + 2.0) / (1.0 + z)) ** 3.87
            + ((1.0 + 2.0) / (1.0 + z)) ** -2.12) for z in zf])

        peak = zf[int(np.argmax(lg_ls))]
        assert 1.8 < peak < 2.6, peak
        assert lg_ls[-1] < lg_ls.max() - 0.3, "L_* must decline past z_c"
        assert not np.all(np.diff(lg_ls) > 0), "the defect was monotonicity"

        # The sector's function agrees with the hand-written Eq. 38: the
        # break enters Phi only through L_*, so a ratio at fixed L pins it.
        for z in (0.1, 1.0, 3.0):
            u = (1.0 + 2.0) / (1.0 + z)
            want = 44.84 - np.log10(u ** 3.87 + u ** -2.12)
            phi = float(A.xlf_aird15(want, z))
            k = 10.0 ** (-4.03 - 0.19 * (1.0 + z))
            assert abs(phi / (k / 2.0) - 1.0) < 1e-6, z

    def test_the_aird_soft_band_column_is_reachable(self):
        """`zc` is fitted (2.00 hard, 2.31 soft), not a constant, and was a
        literal -- which made Table 5's soft-band column unreachable."""
        soft = dict(k0=-4.28, k1=-0.22, l0=44.93, g1=0.44, g2=2.18,
                    p1=3.39, p2=-3.58, zc=2.31)
        a = float(A.xlf_aird15(44.0, 0.5))
        b = float(A.xlf_aird15(44.0, 0.5, **soft))
        assert a > 0 and b > 0 and a != b
        # The soft band breaks later, so its L_* is still rising at z = 2.
        assert float(A.xlf_aird15(44.0, 2.31, **soft)) > 0

    def test_ueda_is_table_4_at_z0(self):
        """At z = 0 the evolution factor is 1 and Eq. 16 is the bare double
        power law with Table 4's normalisation, 2.91e-6."""
        lx = np.linspace(41.0, 46.0, 11)
        r = 10.0 ** (lx - 43.97)
        want = 2.91e-6 / (r ** 0.96 + r ** 2.71)
        np.testing.assert_allclose(np.asarray(A.xlf_ueda14(lx, 0.0)), want,
                                   rtol=1e-12)

    def test_ueda_golden_value(self):
        """By hand from Table 4: at lg L = 43, p1 = 4.78 - 0.84 = 3.94 and
        z_c1 = 1.86 * 10**(0.29 * (43 - 44.61)) = 0.635 > 0.202, so
        Phi = 2.91e-6 / (r**0.96 + r**2.71) * 1.202**3.94, r = 10**-0.97."""
        r = 10.0 ** (43.0 - 43.97)
        want = 2.91e-6 / (r ** 0.96 + r ** 2.71) * 1.202 ** 3.94
        assert abs(np.log10(want) - (-4.2988)) < 5e-4
        assert abs(float(A.xlf_ueda14(43.0, 0.202)) / want - 1.0) < 1e-12

    @pytest.mark.parametrize("lx", [42.0, 43.0, 44.3, 45.0])
    def test_ueda_evolution_is_continuous_at_both_cutoffs(self, lx):
        """Eq. 16's three regimes join: e(z, L) has no step at z_c1 or z_c2."""
        zc1 = 1.86 * 10.0 ** (0.29 * min(lx - 44.61, 0.0))
        zc2 = 3.0 * 10.0 ** (-0.1 * min(lx - 44.0, 0.0))
        for zc in (zc1, zc2):
            lo = float(A.xlf_ueda14(lx, zc - 1e-9))
            hi = float(A.xlf_ueda14(lx, zc + 1e-9))
            assert abs(hi / lo - 1.0) < 1e-7, (lx, zc)

    def test_ueda_p1_depends_on_luminosity(self):
        """Eq. 17: the low-z evolution index is p1* + beta1 (lg L - 44), so the
        z = 0 -> 0.2 growth differs by 0.84 * log10(1.2) per dex of L."""
        g = [np.log10(float(A.xlf_ueda14(l, 0.2) / A.xlf_ueda14(l, 0.0)))
             for l in (43.0, 44.0)]
        assert abs((g[1] - g[0]) - 0.84 * np.log10(1.2)) < 1e-12


class TestXlfInHUnits:
    r"""A published :math:`\Phi` (Mpc^-3 at h = 0.7) in (Mpc/h)^-3 at the
    model's own :math:`L_X`: :math:`\Phi/0.7^3`, evaluated at
    :math:`\log L - 2\log(0.7/h)`."""

    def test_at_h_0p7_it_is_the_volume_alone(self):
        lx = np.linspace(42.0, 45.0, 7)
        np.testing.assert_allclose(
            np.asarray(A.xlf_in_h_units(A.xlf_aird15, lx, 0.2, 0.7)),
            np.asarray(A.xlf_aird15(lx, 0.2)) / 0.7 ** 3, rtol=1e-12)

    def test_the_luminosity_axis_carries_h_squared(self):
        h = float(PLANCK18.h)
        lx = np.linspace(42.0, 45.0, 7)
        shift = 2.0 * np.log10(0.7 / h)
        assert abs(shift - 0.0334) < 1e-4
        np.testing.assert_allclose(
            np.asarray(A.xlf_in_h_units("ueda14", lx, 0.2, h)),
            np.asarray(A.xlf_ueda14(lx - shift, 0.2)) / 0.343, rtol=1e-12)

    def test_the_old_form_fails_loudly(self):
        """`xlf_in_h_units(phi, h)` returned phi * (0.7/h)**3 -- 2.6x low."""
        with pytest.raises(TypeError):
            A.xlf_in_h_units(A.xlf_aird15(43.0, 0.2), PLANCK18.h)


class TestObscuration:
    def test_it_is_a_fraction(self):
        lx = jnp.asarray(np.linspace(41.0, 46.0, 30))
        for z in (0.0, 0.5, 2.0):
            f = np.asarray(A.obscured_fraction(lx, z))
            assert np.all((f >= 0.0) & (f <= 1.0))

    def test_faint_agn_are_more_obscured(self):
        assert float(A.obscured_fraction(42.0, 0.5)) > \
               float(A.obscured_fraction(45.5, 0.5))

    def test_it_is_differentiable(self):
        g = float(jax.grad(lambda l: A.obscured_fraction(l, 0.5))(43.5))
        assert np.isfinite(g) and g != 0.0


class TestOccupationAndTheContract:
    def test_occupation_is_a_probability(self, sector, field):
        """Within [0, 1] at the defaults, and increasing with halo mass at
        zero slopes.  At the defaults it is not monotonic: the duty cycle is one
        above 10^10.9 Msun and the X-ray luminosity at fixed Eddington ratio
        falls with stellar mass (al_bh + log10_lstar_mslope = -1.16), so the
        occupation peaks at 0.30 near 10^12.8 Msun/h and declines above."""
        n = np.asarray(sector.occupation(field, A.AgnParams(), GP)[0])
        assert np.all((n >= 0.0) & (n <= 1.0))
        n0 = np.asarray(sector.occupation(field, A.AgnParams(**NEUTRAL), GP)[0])
        assert np.all((n0 >= 0.0) & (n0 <= 1.0))
        assert np.all(np.diff(n0) > 0)

    def test_a_higher_threshold_selects_fewer(self, sector, field):
        lo, _ = sector.occupation(field, A.AgnParams(log10lx_min=42.0), GP)
        hi, _ = sector.occupation(field, A.AgnParams(log10lx_min=44.0), GP)
        assert np.all(np.asarray(hi) < np.asarray(lo))

    def test_centrals_only_has_no_one_halo_term(self, sector, field):
        """At the centrals-only opt-out, ``f_duty_sat = 0``: a Bernoulli central
        occupation has no self-pairs, so the AGN auto-spectrum's one-halo term
        vanishes identically.  Left as `None` rather than a zero array so the
        contract *says* so."""
        w = sector.weights(field, A.AgnParams(f_duty_sat=0.0), GP)
        assert w.w_extended is None
        assert w.discrete is True

    def test_the_bias_is_that_of_a_group_scale_host(self, sector, field):
        b = float(sector.effective_bias(field, A.AgnParams(), GP))
        assert 0.5 < b < 2.5

    def test_the_agn_luminosity_is_named_apart_from_the_gas(self):
        """The AGN point source and the hot gas's own luminosity are two
        X-ray sources, and neither is called `lx`, so one cannot be passed
        where the other was meant."""
        assert hasattr(A.AgnSector, "l_x_agn")
        assert not hasattr(A.AgnSector, "lx")
        from ggah_mod.sectors.gas import HotGasDPM
        assert hasattr(HotGasDPM, "x_ray_luminosity")
        assert not hasattr(HotGasDPM, "lx")


class TestDifferentiability:
    KEYS = ["mu_bh", "al_bh", "sig_bh", "rho", "log10_lstar",
            "log10_lstar_mslope", "delta1", "delta2", "log10_ferdf",
            "log10_ferdf_mslope", "k_bol", "gamma_x", "log10lx_min"]

    @pytest.mark.x64
    @pytest.mark.parametrize("key", KEYS)
    def test_gradient_reaches_every_parameter(self, sector, field, key):
        p = A.AgnParams()

        def grads(sec):
            def f(v):
                return jnp.sum(sec.l_x_agn(field, p.replace(**{key: v}), GP))

            x = float(getattr(p, key))
            h = 1e-6 * max(abs(x), 1.0)
            return (float(jax.grad(f)(x)),
                    float((f(x + h) - f(x - h)) / (2 * h)))

        ad, fd = grads(sector)
        assert np.isfinite(ad), key
        if key == "gamma_x":
            assert ad == 0.0        # the hard band does not use it
            return
        if key == "k_bol":
            # Read only at kbol="constant": at the default, "duras20", its
            # gradient is identically zero (agn.K_BOL), so it is checked on
            # a constant-mode sector.
            assert ad == 0.0 and fd == 0.0
            ad, fd = grads(A.AgnSector(GAL, kbol="constant"))
            assert np.isfinite(ad), key
        assert ad != 0.0, key
        assert ad == pytest.approx(fd, rel=1e-4), key

    def test_the_selection_is_smooth_not_a_step(self, sector, field):
        """A hard `L_X > L_min` cut has no derivative in the threshold, and the
        threshold is a survey property a joint fit varies."""
        p = A.AgnParams()
        g = float(jax.grad(lambda v: jnp.sum(sector.l_x_agn(
            field, p.replace(log10lx_min=v), GP)))(42.0))
        assert np.isfinite(g) and g != 0.0

    def test_the_occupation_ceiling_is_not_a_clip(self, sector, field):
        """`tanh`, not `clip`: a clip puts a tie at 1 and kills the gradient
        above it, and the occupation genuinely approaches 1."""
        import ast
        import inspect
        # Structural, not textual: the docstring *explains* why a clip is
        # wrong, so grepping the source finds the word and fails.
        tree = ast.parse(inspect.getsource(A.AgnSector.occupation).lstrip())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "tanh" in called
        assert "clip" not in called

    def test_it_jits(self, sector, field):
        v = jax.jit(lambda p: jnp.sum(sector.l_x_agn(field, p, GP)))(A.AgnParams())
        assert np.isfinite(float(v))

    def test_the_sector_declares_itself(self):
        assert A.AgnSector.differentiable is True
        assert A.AgnSector.name == "agn"


class TestTheWeightConventionExcludesTheMassFunction:
    r"""``W`` carries no ``dn/dM``; layer 4's integrand carries it once.

    :mod:`~ggah_mod.sectors.protocol` states
    :math:`P^{1h} = \int dM\,(dn/dM)\,W_a W_b`, so a weight that also carried
    ``dn/dM`` would put :math:`(dn/dM)^2` in every one-halo integrand and
    :math:`dn/dM` in every two-halo one.  This sector did, alone among the
    three, and nothing raised because nothing consumed the weights yet.
    """

    def test_the_point_weight_is_the_occupation_itself(self, sector, field):
        p = A.AgnParams()
        n_cen, _ = sector.occupation(field, p, GP)
        w = sector.weights(field, p, GP)
        assert np.array_equal(np.asarray(w.w_point), np.asarray(n_cen))

    def test_it_is_not_multiplied_by_the_mass_function(self, sector, field):
        w = sector.weights(field, A.AgnParams(), GP)
        n_cen, _ = sector.occupation(field, A.AgnParams(), GP)
        assert not np.allclose(np.asarray(w.w_point),
                               np.asarray(field.dndm * n_cen))

    def test_the_weight_is_dimensionless_like_the_others(self, sector, field):
        r"""With ``norm = n_bar``, ``W = N_cen/n_bar`` has units of volume, the
        same as the gas and matter weights.  Carrying ``dn/dM`` as well left it
        in :math:`(M_\odot/h)^{-1}` -- a mismatch no shape comparison shows."""
        # int dM (dn/dM) W  ==  1 exactly, for any correctly normalised
        # discrete tracer: it is n_bar/n_bar.  Centrals only here, because the
        # satellites' weight carries u_sat(k_min) = 1 - O(1e-10), not 1.
        w = sector.weights(field, A.AgnParams(f_duty_sat=0.0), GP)
        assert float(field.integrate(field.dndm * w.total()[0])) == \
            pytest.approx(1.0, rel=1e-10)
        # With the satellites, the same statement on the occupation the two
        # weights are built from.
        w = sector.weights(field, A.AgnParams(), GP)
        assert float(field.integrate(field.dndm * w.self_pair / w.norm)) == \
            pytest.approx(1.0, rel=1e-10)


class TestTheBlackHolesThemselves:
    """Omega_BH and the mass function behind it, both predicted.

    The chain was never abundance-matched to a black-hole mass function, so
    comparing to one is a test in the same sense the XLF comparison is.
    """

    def test_the_mean_mass_is_not_ten_to_the_mean_log(self, field):
        """A lognormal's mean exceeds the exponential of its mean log, by
        exp[(ln10 sigma)^2/2].  Measured here: 1.40, so taking the naive route
        is a 29% underestimate that looks like a plausible answer."""
        # At the published relation, sigma_lm = 0.356: the claim is the size
        # of the correction there (1.128 at the 0.8.7 defaults, sigma_lm 0.213).
        a, p = A.AgnSector(GAL), A.AgnParams(**A.AGN_PUBLISHED)
        lg = a.mean_log10_mbh(jnp.log10(field.m), p, GP, h=PLANCK18.h)
        naive = jnp.power(10.0, lg)
        # Per central above the stellar-mass cut: `mean_mbh` carries the
        # fraction of them, which is not the lognormal's business.
        frac = a.central_fraction(jnp.log10(field.m), p, GP, h=PLANCK18.h)
        mean = a.mean_mbh(jnp.log10(field.m), p, GP, h=PLANCK18.h) / frac
        r = np.asarray(mean / naive)
        sig = np.asarray(a._sigma_lm(p, a.sigma_mstar(jnp.log10(field.m), GP,
                                                       h=PLANCK18.h)))
        assert np.allclose(r, np.exp(0.5 * (np.log(10.0) * sig) ** 2),
                           rtol=1e-12)
        assert r[0] == pytest.approx(1.51, abs=0.02)

    def test_more_scatter_means_more_mass(self, field):
        a = A.AgnSector(GAL)
        lo = a.mean_mbh(jnp.log10(field.m), A.AgnParams(sig_bh=0.05), GP, h=PLANCK18.h)
        hi = a.mean_mbh(jnp.log10(field.m), A.AgnParams(sig_bh=0.60), GP, h=PLANCK18.h)
        assert np.all(np.asarray(hi) > np.asarray(lo))

    def test_omega_bh_is_the_right_order(self, field):
        """Fukugita & Peebles (2004) row 3.13 is 10^-5.4 = 4.0e-6; integrating
        a local black-hole mass function gives ~3.3e-6.  Centrals and
        satellites together predict 2.08e-6 at z = 0 at the defaults, 0.52 of
        the first and 0.63 of the second.  Asserted as
        an order of magnitude, because tightening it would be asserting the
        disagreement rather than measuring it."""
        o = float(A.AgnSector(GAL).omega_bh(field, A.AgnParams(), GP))
        assert 1e-7 < o < 1e-5

    def test_the_mass_function_re_integrates_to_the_density(self, field):
        """Two routes to one number: the density directly, and the mass
        function integrated.  They must agree, or the convolution has moved
        mass that the direct integral kept."""
        a, p = A.AgnSector(GAL), A.AgnParams()
        lg = np.linspace(4.0, 11.0, 141)
        phi = np.asarray(a.black_hole_mass_function(field, lg, p, GP))
        rho = np.trapezoid(phi * 10.0 ** lg, lg) * field.cosmo.h
        direct = float(a.omega_bh(field, p, GP)) * 2.77536627e11
        assert rho == pytest.approx(direct, rel=2e-2)

    def test_the_mass_function_falls_at_the_high_mass_end(self, field):
        """By more than six decades from 10^6 to 10^10 Msun at the eROSITA c030
        AGN defaults (sig_bh = 0.280), the configuration this was written for.
        At the eROSITA x DESI YR3 defaults (sig_bh = 0.314) it still falls
        monotonically, by 6.5 decades."""
        phi = np.asarray(A.AgnSector(GAL).black_hole_mass_function(
            field, jnp.linspace(6.0, 10.0, 9), A.AgnParams(**AGN_C030), GP))
        assert np.all(np.diff(phi) < 0)
        assert phi[-1] < 1e-6 * phi[0]

    @pytest.mark.x64
    def test_omega_bh_is_differentiable_in_the_scaling_relation(self, field):
        def f(mu):
            return A.AgnSector(GAL).omega_bh(field, A.AgnParams(mu_bh=mu), GP)

        ad = float(jax.grad(f)(7.76))
        fd = float((f(7.76 + 1e-6) - f(7.76 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5)


class TestTheStellarMassIsTheGalaxySectors:
    r"""The chain starts from a stellar mass, and the galaxy sector owns it.

    The AGN sector used to evaluate ``zu15`` at its own copy of the fiducial
    parameters.  At the defaults the two copies agree, which is why nothing
    failed; a joint fit varying ``lg_m1h`` moved the galaxies and left every AGN
    where it was.  Now the sector is built from a threshold occupation and
    evaluated at its parameters, so every test that couples the two moves a
    galaxy parameter far enough that the uncoupled answer is measurably wrong.
    """

    LG_M1H = 12.40          # the default is 12.10

    def test_it_needs_a_galaxy_sector(self):
        with pytest.raises(ValueError, match="needs a galaxy sector"):
            A.AgnSector()

    @pytest.mark.parametrize("model,shmr", [
        ("zheng07", None), ("zheng07", "zu15"), ("more15", None),
        ("guo18", None), ("cacciato09", None)])
    def test_a_simple_occupation_cannot_carry_it(self, model, shmr):
        """A simple HOD assigns no stellar masses; a bolted-on SHMR would give
        the AGN one unrelated to the galaxies it counted.  ``guo18`` carries an
        SHMR but no threshold, so it has no satellite stellar-mass function."""
        with pytest.raises(ValueError, match="threshold occupation"):
            A.AgnSector(GalaxySector(model, shmr=shmr))

    @pytest.mark.parametrize("model", sorted(__import__(
        "ggah_mod.sectors.galaxies", fromlist=["THRESHOLD_SHMR"]).THRESHOLD_SHMR))
    def test_the_stellar_mass_is_the_one_the_occupation_counts_with(self, model):
        r"""Every threshold central is :math:`A(M_h)\,\tfrac12{\rm erfc}` in
        :math:`\log M_*^{\rm t} - \log M_*^{\rm c}(M_h)`, so the occupation is
        symmetric about its midpoint *exactly* at the stellar mass it counts
        with.  Checked at that point, and shown to fail 0.05 dex away, for every
        row of ``THRESHOLD_SHMR`` -- which is what catches a wrong argument
        map."""
        gal = GalaxySector(model)
        gp = galaxy_defaults(model)
        kw = {n: gp[n] for n in gal._cen_req + gal._cen_opt
              if n in gp and n != "log10m_star_thresh"}
        lm = jnp.asarray([11.5, 12.5, 13.5, 14.5])
        ms = gal._log10_mstar_native(lm, gp)

        def asymmetry(centre, d=0.3):
            up = gal._cen(lm, log10m_star_thresh=centre + d, **kw)
            dn = gal._cen(lm, log10m_star_thresh=centre - d, **kw)
            mid = gal._cen(lm, log10m_star_thresh=centre, **kw)
            # A quenched colour can empty a halo exactly (zumandelbaum16_blue
            # above ~10^13 h^-1 Msun with the LS10 quenching of 1.3.0): 0/0
            # there says nothing about the threshold, so only occupied haloes
            # are checked.
            ok = np.asarray(mid) > 0
            assert ok.sum() >= 2, model
            return np.asarray(jnp.abs(0.5 * (up + dn) / mid - 1.0))[ok]

        assert np.max(asymmetry(ms)) < 1e-9, model
        assert np.min(asymmetry(ms + 0.05)) > 1e-4, model

    def test_a_threshold_occupation_refuses_a_second_relation(self):
        with pytest.raises(ValueError, match="second stellar mass"):
            GalaxySector("zumandelbaum15", shmr="moster13")
        assert GalaxySector("zumandelbaum15", shmr="zu15").is_threshold

    def test_a_declared_shmr_with_missing_parameters_raises(self):
        gal = GalaxySector("zheng07", shmr="zu15")
        with pytest.raises(ValueError, match="lg_m1h"):
            gal.log10_mstar(jnp.asarray([12.0]), {"log10mmin": 12.0}, h=0.7)

    def _spectrum_inputs(self):
        gal = GalaxySector("zumandelbaum15")
        gal_p = dict(GP, lg_m1h=self.LG_M1H)
        return ({"galaxies": gal, "agn": A.AgnSector(gal)},
                {"galaxies": gal_p, "agn": A.AgnParams()}, gal, gal_p)

    @pytest.mark.parametrize("view", ["counts", "emission"])
    def test_build_weights_uses_the_galaxy_parameters(self, field, view):
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        sectors, params, gal, gal_p = self._spectrum_inputs()
        got = np.asarray(build_weights(Component("agn", view), field, sectors,
                                       params).w_point)
        a = A.AgnSector(gal)
        n_cen, lx = a.occupation(field, A.AgnParams(), gal_p)
        expected = np.asarray(n_cen if view == "counts" else lx)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)
        # The comparison must be able to fail: at the default galaxy
        # parameters the answer is far from it.
        n0, lx0 = a.occupation(field, A.AgnParams(), GP)
        stale = np.asarray(n0 if view == "counts" else lx0)
        live = expected > 1e-6 * expected.max()
        assert np.max(np.abs(stale[live] / expected[live] - 1.0)) > 0.05

    def test_the_gradient_reaches_the_galaxy_shmr(self, field):
        """``d n_AGN / d lg_m1h``, by autodiff through ``build_weights`` and by a
        central difference."""
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        sectors, params, _, _ = self._spectrum_inputs()

        def n_bar(lg_m1h):
            pp = dict(params, galaxies=dict(params["galaxies"], lg_m1h=lg_m1h))
            return build_weights(Component("agn", "counts"), field, sectors,
                                 pp).norm

        x, h = jnp.asarray(self.LG_M1H), 1e-3
        ad = float(jax.grad(n_bar)(x))
        fd = float((n_bar(x + h) - n_bar(x - h)) / (2 * h))
        assert fd != 0.0
        np.testing.assert_allclose(ad, fd, rtol=1e-4)

    def test_an_agn_component_needs_the_galaxy_parameters(self, field):
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        sectors, params, _, _ = self._spectrum_inputs()
        with pytest.raises(ValueError, match="'galaxies' parameters"):
            build_weights(Component("agn", "counts"), field,
                          {"agn": sectors["agn"]}, {"agn": params["agn"]})

    def test_the_spectrum_refuses_a_second_galaxy_sector(self, field):
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        sectors, params, _, _ = self._spectrum_inputs()
        other = dict(sectors, galaxies=GalaxySector("zumandelbaum15"))
        with pytest.raises(ValueError, match="two\\s+different stellar masses"):
            build_weights(Component("agn", "counts"), field, other, params)


class TestSatellites:
    r"""Every satellite hosts a black hole; ``f_duty_sat`` says whether it shines.

    The black-hole relation is the central's at the satellite's stellar mass,
    and the satellites' stellar masses are the galaxy occupation's own
    conditional stellar-mass function.  The duty cycle relative to a central's
    is the one number the model cannot supply; it defaults to one, because AGN
    live in galaxies and not in haloes, and zero is the centrals-only opt-out.
    """

    @staticmethod
    def _p(f):
        return A.AgnParams(f_duty_sat=f)

    def test_it_is_a_declared_parameter_with_a_reason(self):
        p = A.AgnParams._PARAMS["f_duty_sat"]
        assert p.default == 0.046
        assert A.AgnParams().f_duty_sat == 0.046
        lo, hi = p.bounds
        assert lo == 0.0
        assert hi > 1.0, (
            "the box must admit an enhancement: satellites in groups have a "
            "higher interaction rate, and a ceiling at 1 refuses that by "
            "construction rather than by measurement")
        assert "measurement" not in p.why or "not" in p.why.lower()
        assert "does not settle it" in p.why
        assert "eROSITA" in p.why and str(p.default) in p.why
        assert "opt-out" in p.why

    def test_the_default_has_active_satellites(self, field, sector):
        """Satellite galaxies host active nuclei by default, at 0.046 of a
        central's duty cycle (the eROSITA x DESI YR3 MAP): the satellite weight
        is linear in ``f_duty_sat``, so the default one is 0.046 of the
        ``f_duty_sat = 1`` one."""
        w = sector.weights(field, A.AgnParams(), GP)
        one = sector.weights(field, self._p(1.0), GP)
        assert w.w_extended is not None and w.discrete
        np.testing.assert_allclose(np.asarray(w.w_extended),
                                   0.046 * np.asarray(one.w_extended),
                                   rtol=1e-12, atol=0.0)
        assert sector.has_satellites(A.AgnParams())
        assert not sector.has_satellites(self._p(0.0))

    def test_active_satellites_sit_on_the_galaxy_satellite_profile(
            self, field, sector):
        p = self._p(1.0)
        n_cen, _ = sector.occupation(field, p, GP)
        n_sat = sector.satellite_occupation(field, p, GP)
        w = sector.weights(field, p, GP)
        np.testing.assert_array_equal(np.asarray(w.w_point), np.asarray(n_cen))
        np.testing.assert_allclose(
            np.asarray(w.w_extended),
            np.asarray(n_sat[None, :] * GAL.satellite_uk(field, GP)), rtol=1e-14)
        np.testing.assert_allclose(np.asarray(w.self_pair),
                                   np.asarray(n_cen + n_sat), rtol=1e-14)
        assert float(w.norm) == pytest.approx(
            float(field.number_density(n_cen + n_sat)), rel=1e-14)

    def test_the_duty_cycle_is_a_linear_scale(self, field, sector):
        half = np.asarray(sector.satellite_occupation(field, self._p(0.5), GP))
        one = np.asarray(sector.satellite_occupation(field, self._p(1.0), GP))
        np.testing.assert_allclose(2.0 * half, one, rtol=1e-12)
        assert one.max() > 1.0, "a massive host carries several active satellites"

    def test_the_measured_numbers_the_docstring_quotes(self, field, sector):
        """At the eROSITA x DESI YR3 AGN defaults (``f_duty_sat = 0.046``) and
        the 1.2.0 galaxy defaults satellites are 1.52% of the AGN and b_eff is
        1.016, against 1.002 for centrals alone (module docstring)."""
        n_cen, _ = sector.occupation(field, A.AgnParams(), GP)
        n_sat = sector.satellite_occupation(field, A.AgnParams(), GP)
        nc = float(field.number_density(n_cen))
        ns = float(field.number_density(n_sat))
        assert ns / (nc + ns) == pytest.approx(0.0152, abs=0.0005)
        assert float(sector.effective_bias(field, self._p(0.0), GP)) == \
            pytest.approx(1.002, abs=0.001)
        assert float(sector.effective_bias(field, A.AgnParams(), GP)) == \
            pytest.approx(1.016, abs=0.001)

    def test_the_satellite_xlf_integrates_to_the_satellite_density(
            self, field, sector):
        """Two routes: the luminosity function's increment integrated above the
        selection, and the occupation integrated over the mass function."""
        g0, phi0 = sector.xlf(field, self._p(0.0), GP)
        _, phi1 = sector.xlf(field, self._p(1.0), GP)
        sel = np.asarray(sector._selection(A.AgnParams()))
        from_xlf = np.trapezoid((np.asarray(phi1) - np.asarray(phi0)) * sel,
                                np.asarray(g0))
        n_sat = sector.satellite_occupation(field, self._p(1.0), GP)
        assert from_xlf == pytest.approx(float(field.number_density(n_sat)),
                                         rel=2e-3)

    @pytest.mark.x64
    def test_the_gradient_reaches_the_duty_cycle(self, field, sector):
        def n_bar(f):
            return sector.weights(field, self._p(f), GP).norm

        ad = float(jax.grad(n_bar)(0.5))
        fd = float((n_bar(0.5 + 1e-4) - n_bar(0.5 - 1e-4)) / 2e-4)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-6)

    def test_a_satellite_black_hole_follows_the_central_relation(
            self, field, sector):
        """Shifting ``mu_bh`` by 0.1 dex multiplies the satellites' black-hole
        mass by exactly 10^0.1, as it does the central's."""
        mu = A.AgnParams().mu_bh
        lo = sector.mean_mbh_satellites(field, A.AgnParams(), GP)
        hi = sector.mean_mbh_satellites(field, A.AgnParams(mu_bh=mu + 0.1), GP)
        np.testing.assert_allclose(np.asarray(hi / lo), 10 ** 0.1, rtol=1e-10)

    def test_every_satellite_has_a_black_hole_whatever_the_duty_cycle(
            self, field, sector):
        """``omega_bh`` is the centrals plus the satellites, at any
        ``f_duty_sat``; satellites add 25.6% at the eROSITA x DESI YR3 AGN
        defaults on the 1.2.0 galaxies, down to the 10^8 Msun cut."""
        p = A.AgnParams()
        cen = float(field.integrate(
            field.dndm * sector.mean_mbh(jnp.log10(field.m), p, GP, h=field.cosmo.h, z=field.z)
            * field.cosmo.h) / A.C.RHO_CRIT0)
        tot = float(sector.omega_bh(field, p, GP))
        assert tot == float(sector.omega_bh(field, self._p(0.0), GP))
        assert tot / cen == pytest.approx(1.256, abs=0.003)


class TestTheEmissionWeight:
    r"""The emission weight is the selected luminosity **per halo**.

    It used to be :math:`N_{\rm cen}\langle N_{\rm cen}L_X\rangle`: the second
    factor already carries the duty cycle, so the occupation was counted twice
    and the emission was 0.01--0.03 of the right answer in every halo.
    """

    def test_the_point_weight_is_the_halo_luminosity(self, field, sector):
        """The central galaxy's nucleus at the centre, the satellites' on the
        satellite profile, and nothing extended at the centrals-only opt-out."""
        p = A.AgnParams()
        _, lx = sector.occupation(field, p, GP)
        w = sector.emission_weights(field, p, GP)
        np.testing.assert_array_equal(np.asarray(w.w_point), np.asarray(lx))
        assert float(w.norm) == 1.0
        lx_sat = sector.satellite_occupation(field, p, GP, 1)
        np.testing.assert_allclose(
            np.asarray(w.w_extended),
            np.asarray(lx_sat[None, :] * GAL.satellite_uk(field, GP)), rtol=1e-14)
        assert sector.emission_weights(
            field, A.AgnParams(f_duty_sat=0.0), GP).w_extended is None

    def test_the_self_pair_is_the_second_moment(self, field, sector):
        r""":math:`\langle N L^2\rangle \ge \langle N L\rangle^2/\langle N\rangle`
        (Cauchy-Schwarz), with equality only for a single luminosity."""
        p = A.AgnParams()
        prob = sector.p_active_loglx_given_m(jnp.log10(field.m), p, GP,
                                             h=field.cosmo.h, z=field.z)
        n = np.asarray(sector._moment(prob, p, 0))
        w = sector.emission_weights(field, p, GP)
        l1, l2 = np.asarray(w.w_point), np.asarray(w.self_pair)
        live = n > 1e-8
        assert np.all(l2[live] * n[live] > l1[live] ** 2)


class TestTheEmissionInABand:
    r"""0.9.2: the emitted luminosity carried into another band.

    A photon map in 0.5--2 keV is what needed it -- the LS10 galaxy x eROSITA
    event cross-correlation, where below :math:`M_* \sim 10^{10.5}` the AGN
    dominate the inner 80 kpc.  The pieces were all here; they entered only the
    selection.
    """

    @pytest.fixture(scope="class")
    def split(self):
        return A.AgnSector(GAL, obscuration="split")

    def test_no_band_is_the_hard_band_bit_for_bit(self, field, sector, split):
        """A sector built with the split answers band=None exactly as one
        built without it: the split reaches a prediction only when asked."""
        p = A.AgnParams()
        a = sector.emission_weights(field, p, GP)
        b = split.emission_weights(field, p, GP)
        for f in ("w_point", "w_extended", "self_pair"):
            np.testing.assert_array_equal(np.asarray(getattr(a, f)),
                                          np.asarray(getattr(b, f)))
        assert a.name == b.name == "agn:emission"

    def test_rest_frame_hard_band_at_the_floor_column_is_the_hard_band(
            self, field, split):
        """K = 1 unobscured by construction, and 1 - O(1e-4) obscured at the
        column's 1e20 floor: the band path reproduces band=None there."""
        p = A.AgnParams(log10_nh=20.0)
        hard = split.emission_weights(field, p, GP)
        band = split.emission_weights(field, p, GP, band=(2.0, 10.0),
                                      frame="rest")
        np.testing.assert_allclose(np.asarray(band.w_point),
                                   np.asarray(hard.w_point), rtol=1e-3)
        assert np.all(np.asarray(band.w_point) <= np.asarray(hard.w_point))

    def test_the_soft_band_sits_below_the_unabsorbed_ratio(self, field, split):
        r"""Unabsorbed, the soft emission would be ``k_h2s`` times the hard;
        with the obscured branch absorbed it is less, and by more than the
        few per cent a formality would cost: 0.89 of the AGN at 1e42 erg/s
        are obscured (Comparat et al. 2019's faint branch; 0.59 until
        1.3.0.dev0, when the fraction's ceiling compressed it), and they pass
        4% of their soft energy.  The ratio runs 0.15-0.65 of the unabsorbed
        one, median 0.55, at the 1.2.0 defaults (0.83 in the median before the
        fix).  It cannot fall below the obscured branch's own transmission,
        0.041 at the default column and photon index; at the defaults it runs
        0.16-0.69, median 0.33."""
        p = A.AgnParams()
        hard = np.asarray(split.emission_weights(field, p, GP).w_point)
        soft = np.asarray(split.emission_weights(
            field, p, GP, band=(0.5, 2.0), frame="rest").w_point)
        live = hard > 0
        ratio = soft[live] / hard[live]
        transmission = float(A.absorbed_band_energy(0.5, 2.0, p.gamma_x, p.log10_nh)
                             / A.band_energy(0.5, 2.0, p.gamma_x))
        assert np.all(ratio < float(p.k_h2s))
        assert np.all(ratio > transmission * float(p.k_h2s))
        assert np.median(ratio) < 0.9 * float(p.k_h2s)

    def test_the_observer_frame_carries_the_k_correction(self, split):
        r"""Unobscured, observer over rest is exactly
        :math:`(1+z)^{2-\Gamma}`; the obscured branch gains more, because the
        observed band reaches harder, less absorbed energies."""
        p = A.AgnParams()
        obs = A.ObscurationParams(f_faint_norm=0.0, f_bright_floor=0.0,
                                  f_bright_amp=0.0)
        z = 0.27
        ko = np.asarray(split.band_factor(p, (0.5, 2.0), frame="observer",
                                          z=z, obsc=obs))
        kr = np.asarray(split.band_factor(p, (0.5, 2.0), frame="rest", z=z,
                                          obsc=obs))
        # f_obs is now the Compton-thick fraction alone, which is tiny below
        # 1e43; where it vanishes the ratio is the unobscured one.
        f = np.asarray(A.obscured_fraction(split.loglx, z, obs))
        clean = f < 1e-6
        assert clean.any()
        # rtol 1e-5, not 1e-12: f < 1e-6 still leaves the obscured branch in
        # at the 1e-7 level, which is the branch this test sets aside.
        np.testing.assert_allclose(ko[clean] / kr[clean],
                                   (1 + z) ** (2 - p.gamma_x), rtol=1e-5)
        full = A.ObscurationParams(f_faint_norm=5.0, f_bright_floor=1.0)
        fo = np.asarray(split.band_factor(p, (0.5, 2.0), frame="observer",
                                          z=z, obsc=full))
        fr = np.asarray(split.band_factor(p, (0.5, 2.0), frame="rest", z=z,
                                          obsc=full))
        assert np.all(fo / fr > (1 + z) ** (2 - p.gamma_x))

    def test_the_self_pair_averages_the_square(self, split):
        r"""An AGN is obscured or not, so the self-pair is
        :math:`\langle K^2\rangle \ge \langle K\rangle^2`, strictly where the
        split is not all one way."""
        p = A.AgnParams()
        k1 = np.asarray(split.band_factor(p, (0.5, 2.0), z=0.2, power=1))
        k2 = np.asarray(split.band_factor(p, (0.5, 2.0), z=0.2, power=2))
        f = np.asarray(A.obscured_fraction(split.loglx, 0.2))
        mixed = (f > 1e-3) & (f < 1 - 1e-3)
        assert mixed.any()
        assert np.all(k2[mixed] > k1[mixed] ** 2)

    def test_a_band_without_the_split_is_refused(self, field, sector):
        with pytest.raises(ValueError, match="obscuration='split'"):
            sector.emission_weights(field, A.AgnParams(), GP, band=(0.5, 2.0))

    def test_a_bad_frame_or_band_is_refused(self, split):
        with pytest.raises(ValueError, match="frame"):
            split.band_factor(A.AgnParams(), (0.5, 2.0), frame="source")
        with pytest.raises(ValueError, match="0 < emin < emax"):
            split.band_factor(A.AgnParams(), (2.0, 0.5))

    def test_the_band_moves_with_gamma_and_the_column(self, field, split):
        """The spectrum the band reads is the fitted one, as the flux
        selection's is."""
        base = np.asarray(split.emission_weights(
            field, A.AgnParams(), GP, band=(0.5, 2.0)).w_point)
        thick = np.asarray(split.emission_weights(
            field, A.AgnParams(log10_nh=23.5), GP, band=(0.5, 2.0)).w_point)
        soft = np.asarray(split.emission_weights(
            field, A.AgnParams(gamma_x=2.2), GP, band=(0.5, 2.0)).w_point)
        live = base > 0
        assert np.all(thick[live] < base[live])
        assert np.all(soft[live] > base[live])

    def test_the_band_is_in_the_name(self, field, split):
        w = split.emission_weights(field, A.AgnParams(), GP, band=(0.5, 2.0))
        assert w.name == "agn:emission[0.5-2keV/o]"


class TestTheObscurationBranches:
    r"""0.9.3: each obscuration branch on its own.

    A detector's energy-conversion factor differs between the branches -- by up
    to 1e6 in a 100 eV band at the soft end -- and the branch weight sits
    inside the luminosity integral, so counts need the two luminosities apart.
    Summing back to the total is necessary and not sufficient: the sum is
    invariant under moving weight from one branch to the other, which is the
    error the split exists to prevent, so each branch is also checked against
    its own hand-computed term (ggah-cal-29).
    """

    @pytest.fixture(scope="class")
    def split(self):
        return A.AgnSector(GAL, obscuration="split")

    def test_each_branch_is_its_own_term(self, split):
        """At z = 0.2 the obscured fraction runs 0.39-0.90 across 1e41-1e44
        erg/s: far from 0 and 1, where a swap between branches would show."""
        p, z, band = A.AgnParams(), 0.2, (0.5, 2.0)
        g = p.gamma_x
        hard = float(A.band_energy(2.0, 10.0, g))
        k_u = (1 + z) ** (2 - g) * float(A.band_energy(*band, g)) / hard
        k_o = float(A.absorbed_band_energy(*band, g, p.log10_nh, z)) / hard
        f = np.asarray(A.obscured_fraction(split.loglx, z))
        mid = (f > 0.1) & (f < 0.9)
        assert mid.sum() > 100
        for n in (1, 2):
            u = np.asarray(split.band_factor(p, band, z=z, power=n,
                                             branch="unobscured"))
            o = np.asarray(split.band_factor(p, band, z=z, power=n,
                                             branch="obscured"))
            np.testing.assert_allclose(u, (1 - f) * k_u ** n, rtol=1e-12)
            np.testing.assert_allclose(o, f * k_o ** n, rtol=1e-12)
            both = np.asarray(split.band_factor(p, band, z=z, power=n))
            np.testing.assert_array_equal(u + o, both)

    def test_the_branch_weights_add_to_the_total(self, field, split):
        p = A.AgnParams()
        for band in (None, (0.5, 2.0)):
            kw = {} if band is None else {"band": band}
            tot = split.emission_weights(field, p, GP, **kw)
            u = split.emission_weights(field, p, GP, branch="unobscured", **kw)
            o = split.emission_weights(field, p, GP, branch="obscured", **kw)
            for attr in ("w_point", "w_extended", "self_pair"):
                np.testing.assert_allclose(
                    np.asarray(getattr(u, attr)) + np.asarray(getattr(o, attr)),
                    np.asarray(getattr(tot, attr)), rtol=1e-12, atol=0)
            assert u.name.endswith("/unobscured") and o.name.endswith("/obscured")

    def test_the_hard_band_branch_is_the_fraction_of_each_luminosity(
            self, field, split):
        """Intrinsic, so nothing is absorbed: the obscured branch carries f of
        every luminosity, which puts its share between f's extremes."""
        p = A.AgnParams()
        tot = np.asarray(split.emission_weights(field, p, GP).w_point)
        o = np.asarray(split.emission_weights(field, p, GP,
                                              branch="obscured").w_point)
        f = np.asarray(A.obscured_fraction(split.loglx, field.z))
        live = tot > 0
        share = o[live] / tot[live]
        assert np.all(share >= f.min() - 1e-12) and np.all(share <= f.max() + 1e-12)

    def test_a_branch_is_refused_without_the_split_or_by_name(self, field,
                                                                sector, split):
        with pytest.raises(ValueError, match="obscuration"):
            sector.emission_weights(field, A.AgnParams(), GP, branch="obscured")
        with pytest.raises(ValueError, match="branch"):
            split.band_factor(A.AgnParams(), (0.5, 2.0), branch="partly")


class TestAPhotonMapHasNoSelection:
    r"""``selection="none"``: an event map counts every AGN's photons."""

    def test_every_node_is_selected(self):
        s = A.AgnSector(GAL, selection="none")
        np.testing.assert_array_equal(
            np.asarray(s._selection(A.AgnParams())), 1.0)

    def test_it_counts_more_than_any_threshold(self, field, sector):
        p = A.AgnParams()
        s = A.AgnSector(GAL, calibration="off", selection="none")
        n_all = float(field.number_density(s.occupation(field, p, GP)[0]))
        n_cut = float(field.number_density(sector.occupation(field, p, GP)[0]))
        assert n_all > n_cut > 0.0

    def test_the_threshold_parameters_are_inert(self, field):
        s = A.AgnSector(GAL, calibration="off", selection="none")
        a = s.occupation(field, A.AgnParams(), GP)[0]
        b = s.occupation(field, A.AgnParams(log10lx_min=45.0,
                                            log10_flux_min=-11.0), GP)[0]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestMassesAreInOneConvention:
    r"""The occupation's stellar masses reach the black-hole relation in physical
    solar masses.

    ``zumandelbaum15`` writes stellar masses in :math:`h^{-2}M_\odot` and halo
    masses in :math:`h^{-1}M_\odot` (Zu & Mandelbaum 2015, Sec. 1), and
    :func:`~ggah_mod.sectors.agn.mbh_powell` is a relation in physical
    :math:`M_\odot`.  Handing it the occupation's number unconverted put every
    black hole :math:`2\alpha_{\rm BH}\log h` low.
    """

    def test_zu15_stellar_mass_is_converted_by_two_log_h(self):
        lm = jnp.asarray([11.5, 12.5, 13.5, 14.5])
        h = PLANCK18.h
        native = np.asarray(GAL._log10_mstar_native(lm, GP))
        physical = np.asarray(GAL.log10_mstar_msun(lm, GP, h=h))
        np.testing.assert_allclose(physical, native - 2.0 * np.log10(h),
                                   rtol=0, atol=1e-12)

    def test_the_black_hole_relation_sees_the_physical_mass(self):
        lm = jnp.asarray([12.0, 13.0, 14.0])
        h, p = PLANCK18.h, A.AgnParams()
        got = np.asarray(A.AgnSector(GAL).mean_log10_mbh(lm, p, GP, h=h))
        ms = np.asarray(GAL._log10_mstar_native(lm, GP)) - 2.0 * np.log10(h)
        np.testing.assert_allclose(got, p.mu_bh + p.al_bh * (ms - 11.0),
                                   rtol=0, atol=1e-12)
        # And the unconverted answer is measurably different: by al_bh x
        # 2 log10(1/h), 0.362 dex at the defaults.
        stale = p.mu_bh + p.al_bh * (np.asarray(GAL._log10_mstar_native(lm, GP)) - 11.0)
        np.testing.assert_allclose(stale - got, 2.0 * p.al_bh * np.log10(h),
                                   rtol=0, atol=1e-12)
        assert np.min(np.abs(got - stale)) > 0.1

    def test_the_satellite_axis_is_physical_and_the_density_unchanged(
            self, field):
        lg_n, dn_n = GAL.satellite_csmf(field, GP)
        lg_p, dn_p = GAL.satellite_csmf_msun(field, GP)
        np.testing.assert_allclose(np.asarray(lg_p),
                                   np.asarray(lg_n) - 2.0 * np.log10(field.cosmo.h),
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.asarray(dn_p), np.asarray(dn_n))

    def test_the_package_stellar_mass_is_in_msun_per_h(self):
        """``log10_mstar`` is in the halo mass's convention, so M_*/M_h is a
        fraction: native minus one log h for zu15's h^-2 Msun."""
        lm = jnp.asarray([12.0, 13.0])
        h = PLANCK18.h
        np.testing.assert_allclose(
            np.asarray(GAL.log10_mstar(lm, GP, h=h)),
            np.asarray(GAL._log10_mstar_native(lm, GP)) - np.log10(h),
            rtol=0, atol=1e-12)

    def test_a_physical_occupation_sees_physical_halo_masses(self, field):
        """leauthaud12 is written in physical Msun, so its centrals are counted
        at log M_h - log h, not at the field's Msun/h."""
        from ggah_mod.sectors.occupation import n_cen_leauthaud12
        g = GalaxySector("leauthaud12")
        gp = galaxy_defaults("leauthaud12")
        n_cen, _ = g.occupation(field, gp)
        kw = {n: gp[n] for n in g._cen_req + g._cen_opt if n in gp}
        want = n_cen_leauthaud12(jnp.log10(field.m) - jnp.log10(field.cosmo.h),
                                 **kw)
        np.testing.assert_allclose(np.asarray(n_cen), np.asarray(want),
                                   rtol=1e-12, atol=0)
        stale = n_cen_leauthaud12(jnp.log10(field.m), **kw)
        assert float(jnp.max(jnp.abs(n_cen - stale))) > 1e-2

    def test_every_threshold_occupation_declares_its_units(self):
        from ggah_mod.sectors.galaxies import THRESHOLD_MASS_UNITS, THRESHOLD_SHMR
        assert set(THRESHOLD_MASS_UNITS) == set(THRESHOLD_SHMR)

    def test_the_soft_to_hard_ratio_is_the_gamma_1p8_power_law(self):
        # int_0.5^2 E^-0.8 dE / int_2^10 E^-0.8 dE, written out by hand.
        want = (2 ** 0.2 - 0.5 ** 0.2) / (10 ** 0.2 - 2 ** 0.2)
        assert A.HARD_TO_SOFT == pytest.approx(want, rel=1e-12)
        assert float(A.AgnParams(gamma_x=A.GAMMA_X).k_h2s) == \
            pytest.approx(0.6377, abs=1e-4)
        assert "k_h2s" not in A.AgnParams._PARAMS, \
            "k_h2s is band_energy_ratio(Gamma): one quantity, one number"


class TestTheFittedRangeIsDeclaredAndPriced:
    r"""Where the Powell chain is valid, and what leaving it costs.

    The sector had no calibration registry at all while five others in the
    layer did, so a chain fitted at :math:`z \simeq 0.04` over
    :math:`10 < \lg M_\star < 12` was evaluated at any redshift and on every
    halo of the grid without saying so.

    The extrapolation itself is **not** repaired here, and that is the decision
    these tests record: Eq. (mbh) stays a straight line below its fitted range,
    because the alternative is a second published relation joined to the first
    at an arbitrary mass, and that adds a coefficient nobody fitted.  What
    changes is that the range is declared and the price is measured.
    """

    def test_both_chains_are_in_the_registry(self):
        assert set(A.AGN_CALIBRATION) == set(A.BH_CHAINS)

    def test_powell_carries_a_redshift_and_a_stellar_mass_range(self):
        e = A.AGN_CALIBRATION["powell"]
        assert e.z_range == (0.0, 0.1), "Powell et al. (2022) is a local fit"
        assert e.mstar_range == (10.0, 12.0)
        assert "Powell" in e.fit

    def test_trinity_says_the_question_does_not_apply(self):
        """`None` is deliberately not the same as a wide range."""
        assert A.AGN_CALIBRATION["trinity"].z_range is None
        assert A.AGN_CALIBRATION["trinity"].mstar_range is None

    def test_weights_warns_on_both_ranges_at_once(self, field, sector):
        """z = 0.135 is outside the fit, and the grid reaches below lg M* = 10."""
        import warnings
        from ggah_mod.sectors.calibration import MismatchWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sector.weights(field, A.AgnParams(), GP)
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, MismatchWarning)]
        assert len(msgs) == 2, f"expected a z and an M_* warning, got {msgs}"
        assert any("z in [0, 0.1]" in m for m in msgs)
        assert any("mstar range in [10, 12]" in m for m in msgs)

    def test_the_policy_is_honoured(self, field):
        import warnings
        from ggah_mod.sectors.calibration import MismatchWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            A.AgnSector(GAL, calibration="off").weights(field, A.AgnParams(), GP)
            assert not [x for x in w
                        if issubclass(x.category, MismatchWarning)]
        with pytest.raises(ValueError, match="fitted"):
            A.AgnSector(GAL, calibration="strict").weights(
                field, A.AgnParams(), GP)

    def test_the_warning_points_at_the_measured_price(self, field, sector):
        """A distance outside a box is not by itself a reason to act."""
        import warnings
        from ggah_mod.sectors.calibration import MismatchWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sector.weights(field, A.AgnParams(), GP)
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, MismatchWarning)]
        assert any("validity_cost" in m for m in msgs)

    @pytest.mark.parametrize("quantity", ["omega_bh", "n_agn", "l_x"])
    def test_validity_cost_is_a_fraction_below_one(self, field, sector, quantity):
        """The shipped grid always extrapolates, so the cost is never 1."""
        c = float(sector.validity_cost(field, A.AgnParams(), GP,
                                       quantity=quantity))
        assert 0.0 < c < 1.0, f"{quantity} -> {c}"

    def test_the_count_is_hurt_more_than_the_luminosity(self, field, sector):
        r"""Which is the shape of the extrapolation, not an accident.

        Low-mass halos are many and faint, so they carry more of the AGN
        *count* than of the summed :math:`L_X`.  A cost ordering that came out
        the other way would mean the restriction was not selecting on stellar
        mass at all.  At zero slopes (0.60 against 0.82); at the defaults the
        duty cycle, 0.010 at 10^10 Msun, leaves the low-mass hosts few AGN, and
        the two costs are 0.96 and 0.84.
        """
        p = A.AgnParams(**NEUTRAL)
        n = float(sector.validity_cost(field, p, GP, quantity="n_agn"))
        lx = float(sector.validity_cost(field, p, GP, quantity="l_x"))
        assert n < lx

    def test_validity_cost_refuses_an_unknown_quantity(self, field, sector):
        with pytest.raises(ValueError, match="omega_bh"):
            sector.validity_cost(field, A.AgnParams(), GP, quantity="nonsense")

    #: A sector whose stellar-mass cut sits at 10^8.5 Msun, so that a fitted
    #: range can lie below every satellite and still hold some centrals.
    CUT = 8.5

    @staticmethod
    def _below_every_satellite(monkeypatch):
        """A fitted range below the satellites' stellar masses (the sector's
        satellite grid starts at its cut, 8.5) that still holds the centrals of
        the lightest haloes: 8.03 at 1e10 Msun/h on the LS10 defaults, counted
        with the fraction of them above the cut."""
        entry = A.AGN_CALIBRATION["powell"]
        monkeypatch.setitem(A.AGN_CALIBRATION, "powell",
                            entry._replace(mstar_range=(0.0, 8.45)))

    def _central_inside(self, field, sector):
        lg_c = sector.log10_mstar(jnp.log10(field.m), GP, h=field.cosmo.h,
                                  z=field.z)
        inside = (lg_c >= 0.0) & (lg_c <= 8.45)
        assert np.any(inside) and not np.all(inside), \
            "the range must split the centrals, or the test says nothing"
        return inside

    def test_a_satellite_black_hole_is_masked_by_its_own_stellar_mass(
            self, field, monkeypatch):
        """Every satellite is outside the range, so none counts -- not even
        those whose host's central is inside it.  The satellites enter the
        denominator through the public `mean_mbh_satellites`, a route
        independent of the mask."""
        sector = A.AgnSector(GAL, lg_mstar_min=self.CUT)
        self._below_every_satellite(monkeypatch)
        inside = self._central_inside(field, sector)
        p = A.AgnParams()
        cen = sector.mean_mbh(jnp.log10(field.m), p, GP, h=field.cosmo.h,
                              z=field.z)
        sat = sector.mean_mbh_satellites(field, p, GP)
        expect = (field.integrate(field.dndm * cen * inside)
                  / field.integrate(field.dndm * (cen + sat)))
        assert float(sector.validity_cost(field, p, GP)) == \
            pytest.approx(float(expect), rel=1e-12)

    @pytest.mark.parametrize("quantity,power", [("n_agn", 0), ("l_x", 1)])
    def test_the_satellite_agn_are_counted_and_masked_by_their_own_mass(
            self, field, monkeypatch, quantity, power):
        """The AGN of satellite galaxies are in the denominator, at the duty
        cycle `satellite_occupation` gives them, and out of the numerator
        because their own stellar mass is outside the range."""
        sector = A.AgnSector(GAL, lg_mstar_min=self.CUT)
        self._below_every_satellite(monkeypatch)
        inside = self._central_inside(field, sector)
        p = A.AgnParams(f_duty_sat=1.0)
        prob = sector.p_active_loglx_given_m(jnp.log10(field.m), p, GP,
                                             h=field.cosmo.h, z=field.z)
        cen = sector._moment(prob, p, power, field=field)
        sat = sector.satellite_occupation(field, p, GP, power)
        total = field.integrate(field.dndm * (cen + sat))
        assert float(field.integrate(field.dndm * sat) / total) > 0.05, \
            "the satellites must carry a real share, or leaving them out passes"
        expect = field.integrate(field.dndm * cen * inside) / total
        got = float(sector.validity_cost(field, p, GP, quantity=quantity))
        assert got == pytest.approx(float(expect), rel=1e-12)

    def test_validity_cost_is_a_diagnostic_and_changes_no_answer(
            self, field, sector):
        """It must not clamp, truncate or reweight anything.

        The straight line continues; this method only reports what that is
        worth.  So Omega_BH is what it was before the registry existed.
        """
        p = A.AgnParams()
        before = float(sector.omega_bh(field, p, GP))
        sector.validity_cost(field, p, GP)
        assert float(sector.omega_bh(field, p, GP)) == before


class TestTheBreakEvolves:
    r"""Redshift evolution, in the one place the arithmetic puts it.

    The sector was single-redshift: nothing in ``AgnParams`` carried a
    :math:`z` term, so a chain whose luminosity function is an *output* was
    asked to match a measured one at two redshifts with no freedom between
    them.  :func:`lstar_of_z` gives it that freedom, in the ERDF break and
    nowhere else -- a pure normalisation cannot be wrong in opposite
    directions at two redshifts, and :math:`\mu_{\rm BH}` would drag
    :math:`\Omega_{\rm BH}` and the black-hole mass function with it.
    """

    def test_the_defaults_are_the_shipped_model(self):
        """All coefficients zero is exactly no evolution, at every redshift."""
        p = A.AgnParams()
        assert p.gam_lam == 0.0 and p.gam_lam_hi == 0.0
        for z in (0.0, 0.135, 1.0, 3.0):
            assert float(A.lstar_of_z(z, p.log10_lstar, p.gam_lam,
                                      p.gam_lam_hi, p.z_lam)) == p.log10_lstar

    @pytest.mark.parametrize("gam,gam_hi,z_lam",
                             [(3.5, 0.0, 2.0), (3.9, 1.5, 2.0), (-1.0, 2.0, 0.8)])
    def test_z_zero_is_the_identity_whatever_the_coefficients(self, gam, gam_hi,
                                                              z_lam):
        r"""What keeps the prior on the break (Powell et al. 2022, App. B) honest.

        The normalisation is at :math:`z = 0`, not at :math:`z_\lambda`, so
        ``log10_lstar`` still means *the local break* -- the quantity that
        prior is a measurement of.  Normalising at the pivot instead would
        silently redefine it.
        """
        base = A.AgnParams().log10_lstar
        assert float(A.lstar_of_z(0.0, base, gam, gam_hi, z_lam)) == base

    def test_it_brightens_with_redshift_and_then_turns_over(self):
        """Broken, because both published fits here saturate."""
        base = A.AgnParams().log10_lstar
        sh = [float(A.lstar_of_z(z, base, 3.5, 1.5, 2.0)) - base
              for z in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)]
        assert sh[0] == 0.0
        assert sh[1] < sh[2] < sh[3], "must brighten up to the pivot"
        assert sh[5] < sh[3], "and turn over above it"

    def test_it_is_differentiable_in_all_three(self, field):
        base = A.AgnParams().log10_lstar
        for i, arg in enumerate(("gam_lam", "gam_lam_hi", "z_lam")):
            g = float(jax.grad(
                lambda v, i=i: A.lstar_of_z(
                    1.0, base, *[v if j == i else d
                                 for j, d in enumerate((3.5, 1.5, 2.0))]))(
                [3.5, 1.5, 2.0][i]))
            assert np.isfinite(g) and g != 0.0, f"{arg} has no gradient"

    def test_the_evolution_reaches_the_luminosity_function(self, field, sector):
        """z enters the kernel, which is what `_kernel(z=)` exists for."""
        g0, phi0 = sector.xlf(field, A.AgnParams(), GP)
        g1, phi1 = sector.xlf(field, A.AgnParams(gam_lam=3.5), GP)
        assert np.allclose(np.asarray(g0), np.asarray(g1))
        assert not np.allclose(np.asarray(phi0), np.asarray(phi1))

    def test_the_stellar_mass_is_one_quantity_across_the_two_sectors(self):
        r"""The trap `AgnSector.log10_mstar` carried: it dropped ``z``.

        ``GalaxySector.stellar_fraction`` passes ``z=field.z`` and this sector
        did not, so the AGN chain and the stellar fraction would have read two
        different stellar masses the moment a redshift-dependent threshold
        relation was admitted.  It moves no number today -- every
        ``THRESHOLD_SHMR`` entry is redshift-independent -- which is why it is
        checked by source rather than by value.
        """
        import inspect
        src = inspect.getsource(A.AgnSector.log10_mstar)
        assert "z=z" in src, "z must be forwarded to the galaxy sector"
        assert "z=None" in src, "and be optional, as the galaxy sector has it"


@pytest.mark.slow
class TestOneCoefficientIsNotEnoughAndTwoAre:
    r"""The specification test: what the evolution is *for*.

    Measured against Aird et al. (2015), median ratio over
    :math:`42 < \lg L_X < 43.5` and over :math:`44 < \lg L_X < 44.6`, at
    :math:`z = 0.1` and :math:`z = 1`, with the :math:`h = 0.7` conversion
    Eq. (xlf-h) applies.

    **On Paper I's galaxies** (``GP_PUBLISHED``) and the published AGN
    parameters (``AGN_PUBLISHED``, both stellar-mass slopes zero), where every number below was derived.  The LS10 galaxy defaults of 0.8.5 are another input: against
    them the shipped model is 1.55 dex off and evolution alone
    (``gam_lam = 3.87``) reaches 0.81, while the pair still reaches 0.6.  The
    claim is about the AGN chain, so it is kept on the galaxies it was
    audited on.  The pair's break is 0.5 dex below the eROSITA c030 default
    (``AGN_C030``), at -1.298.
    """

    @staticmethod
    def _worst(sector, fields, **kw):
        p = A.AgnParams(**{**A.AGN_PUBLISHED, **NEUTRAL, **kw})
        worst = 0.0
        for z, f in fields.items():
            g, phi = sector.xlf(f, p, GP_PUBLISHED)
            g, phi = np.asarray(g), np.asarray(phi)
            ref = np.asarray(A.xlf_in_h_units(A.xlf_aird15, g, z, PLANCK18.h))
            ok = phi > 0
            for lo, hi in ((42.0, 43.5), (44.0, 44.6)):
                sel = ok & (g > lo) & (g < hi)
                worst = max(worst,
                            abs(np.log10(np.median(phi[sel] / ref[sel]))))
        return worst

    @pytest.fixture(scope="class")
    def fields(self):
        return {z: make_field(PLANCK18,
                              DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                              make_pk("emu_pk"), z=z)
                for z in (0.1, 1.0)}

    def test_evolution_alone_does_not_fix_it(self, fields):
        """It repairs z = 1 and makes z = 0.1 worse, which is the honest result.

        The break brightens monotonically from its z = 0 value, so raising
        ``gam_lam`` lifts *both* redshifts and the low-z knee was already too
        high.  Anyone reading "redshift evolution" as the whole answer should
        read this test instead.
        """
        s = A.AgnSector(GAL, calibration="off")
        assert self._worst(s, fields, gam_lam=3.87) > \
               self._worst(s, fields, gam_lam=0.0) - 0.3

    def test_the_break_and_its_evolution_together_do(self, fields):
        """From a factor 15 at the z = 0.1 knee to a factor 3, everywhere.

        Neither parameter alone: the base break sets where the knee sits at
        z = 0 and the evolution sets how it moves, and the measured failure
        needs both moved.  This is what Phase 5's refit is being set up to do
        properly, with a likelihood rather than a grid.

        **The three thresholds were re-derived on 2026-09-16 and they were
        loosened, so here is the audit.**  They are not bent to fit; the
        *reference curve moved*.  `xlf_aird15` had negated ``p2`` on top of its
        negative tabulated value, so the Aird comparison had no break in it at
        all.  Corrected against Eq. 38, the shipped model is 1.29 dex off
        rather than 1.45 and the pair reaches 0.67 rather than 0.42 -- the
        comparison is both less wrong to begin with and less completely
        repaired, which moves all three numbers the same way.  Previous
        thresholds: 1.4 / 0.5 / 0.9.  If these ever need loosening *without* a
        change to the reference, that is a real regression and not a rounding.

        **Re-derived again on 2026-09-18, and the reference moved again** --
        this time its units.  :func:`~ggah_mod.sectors.agn.xlf_in_h_units`
        multiplied Aird by (0.7/h)^3 = 1.122 where the conversion to
        (Mpc/h)^-3 divides by 0.7^3 (2.915), and it applied no luminosity
        shift; the reference is now 0.415 dex higher and 0.033 dex to the
        right.  Against it the shipped model is 1.18 dex off (+0.95 at the
        z = 0.1 knee, -1.17 at z = 1), a break shift alone buys nothing
        (1.18 at every shift from 0 to -1.2), evolution alone reaches 0.99 at
        best (gam_lam = 1), and at Aird's own p1 = 3.87 the break moves by
        -0.5 rather than -1.0, reaching 0.48.  The pair -1.0 / 3.87 had been
        chosen against a reference 0.41 dex too low and now overshoots, to
        1.05.  A 2-D grid finds 0.39 at (-0.7, 6.0), so the claim this test
        makes -- one coefficient is not enough and two are -- survives the
        correction; its numbers do not.  Previous thresholds: 1.2 / 0.8 / 0.5,
        and the pair -1.0 / 3.87.
        """
        s = A.AgnSector(GAL, calibration="off")
        base = AGN_C030["log10_lstar"]
        shipped = self._worst(s, fields, gam_lam=0.0)
        both = self._worst(s, fields, log10_lstar=base - 0.5, gam_lam=3.87)
        assert shipped > 1.1, f"the shipped model is off by {shipped:.2f} dex"
        assert both < 0.6, f"the pair should reach 0.6 dex, got {both:.2f}"
        assert shipped - both > 0.5, "the pair must buy a factor of three"


class TestTheObscuredSplitIsAPredictionOrNothing:
    r"""Obscuration reaches a prediction, or the sector says it does not.

    :func:`obscured_fraction` and its eight declared parameters existed and were
    consumed by **nothing**: the split an X-ray selection function depends on
    most reached no observable.  ``obscuration='split'`` is the mode that lets
    it, and ``'none'`` -- the default -- keeps asserting nothing.
    """

    def test_the_default_asserts_nothing_and_refuses_a_branch(self, field,
                                                              sector):
        assert sector.obscuration == "none"
        with pytest.raises(ValueError, match="obscuration='split'"):
            sector.effective_bias(field, A.AgnParams(), GP, branch="obscured")

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="obscuration mode"):
            A.AgnSector(GAL, obscuration="sort-of")

    def test_an_unknown_branch_is_refused(self, field):
        s = A.AgnSector(GAL, calibration="off", obscuration="split")
        with pytest.raises(ValueError, match="branch"):
            s.occupation(field, A.AgnParams(), GP, branch="sort-of")

    def test_the_two_branches_partition_the_population(self, field):
        """Obscured plus unobscured is the whole sample, to the saturation, at
        the eROSITA c030 AGN defaults and the published obscured fraction.  At
        the eROSITA x DESI YR3 defaults the sum is 1.0043 of the sample: the
        central occupation reaches 0.30 near 10^12.8 Msun/h (0.023 at the
        c030 defaults), where each branch saturates on its own."""
        s = A.AgnSector(GAL, calibration="off", obscuration="split")
        p = A.AgnParams(**AGN_C030)
        n = lambda b: float(field.number_density(
            s.occupation(field, p, GP, branch=b, obsc=C19)[0]))
        assert (n("obscured") + n("unobscured")) / n(None) == \
            pytest.approx(1.0, abs=1e-3)

    def test_the_default_mode_reproduces_the_shipped_numbers(self, field,
                                                             sector):
        """No branch means no factor: the mode must not move a shipped answer,
        b_eff = 1.016 at the defaults (module docstring)."""
        p = A.AgnParams()
        assert float(sector.effective_bias(field, p, GP)) == \
            pytest.approx(1.016, abs=5e-3)

    def test_a_weight_could_not_have_produced_a_split_at_all(self, field):
        r"""Why two populations and not one weighted one -- the structural reason.

        :math:`f_{\rm obs}` depends on :math:`L_X` and :math:`z`, never on halo
        mass, and the effective bias is a ratio of two integrals over halo mass
        with the same weight in each.  Any halo-mass-independent factor cancels
        exactly, so one population carrying an obscured weight predicts a ratio
        of precisely one whatever the eight parameters are.  Here that is
        checked by construction: scaling the occupation by a constant leaves the
        bias untouched.
        """
        s = A.AgnSector(GAL, calibration="off", obscuration="split")
        p = A.AgnParams()
        n, _ = s.occupation(field, p, GP)
        assert float(field.effective_bias(0.37 * n)) == \
            pytest.approx(float(field.effective_bias(n)), rel=1e-12)

    @pytest.mark.slow
    def test_the_predicted_split_is_measured_not_asserted(self, field):
        r"""And it comes out with the **wrong sign**, which is the finding.

        Petter et al. (2023) measure :math:`b = 2.96\pm0.07` obscured against
        :math:`2.27\pm0.06` unobscured -- obscured AGN *more* clustered, a ratio
        of 1.30 at 9 sigma.  This chain predicts 0.95 at the 1.2.0 defaults
        (the eROSITA c030 MAP, ``AGN_C030``) with Comparat et al. (2019)'s
        fraction as published (0.98 at the 0.8.7
        defaults with the compressed fraction of before 1.3.0.dev0): obscured
        AGN slightly *less* clustered.  The mechanism is visible in one line of
        :func:`obscured_fraction` -- the faint branch dominates at low
        :math:`L_X`, faint AGN sit in lighter halos, so obscuration correlates
        with *low* bias here.

        The conclusion is therefore not that the eight parameters need
        refitting.  A fraction that depends on :math:`L_X` and :math:`z` alone
        cannot reverse this sign, so reproducing Petter needs a halo-mass or
        environment term -- a new parameter with a reason, which this test
        exists to justify rather than assume.  The eROSITA x DESI YR3 defaults
        carry one, ``l_blend_m_amp`` = -0.873, and give 0.83 (1.04 with the
        published fraction).
        """
        s = A.AgnSector(GAL, calibration="off", obscuration="split")
        p = A.AgnParams(**AGN_C030)
        b_ob = float(s.effective_bias(field, p, GP, branch="obscured", obsc=C19))
        b_un = float(s.effective_bias(field, p, GP, branch="unobscured", obsc=C19))
        assert b_ob / b_un == pytest.approx(0.948, abs=0.02)
        assert b_ob < b_un, "the predicted sign is the opposite of Petter 2023"


class TestTheSpectrumAndTheSelection:
    r"""A photon index that can be fitted, a column that absorbs, and a cut in flux.

    The sector carried one spectral assumption -- a fixed :math:`\Gamma = 1.8`
    hidden inside a scalar band ratio -- and selected in rest-frame
    :math:`L_X` while every X-ray survey selects in flux.  Both are now
    declared, and the same :math:`\Gamma` does both jobs.
    """

    def test_the_cross_section_is_the_published_table(self):
        r"""Morrison & McCammon (1983), transcribed and not derived.

        Checked against ``soxs``'s copy of the same table where it is
        installed, because two transcriptions of one paper agreeing is worth
        more than one transcription looking plausible.
        """
        soxs_fa = pytest.importorskip("soxs.spectra.foreground_absorption")
        e = np.array([0.3, 0.5, 1.0, 2.0, 5.0, 9.0])
        assert np.allclose(np.asarray(A.photoabs_cross_section(e)),
                           soxs_fa.wabs_cross_section(e), rtol=1e-12)

    def test_no_column_is_exactly_the_k_correction(self):
        r"""The K-correction falls out of the limits; there is no second factor.

        :math:`\int_{a(1+z)}^{b(1+z)}E^{1-\Gamma}dE = (1+z)^{2-\Gamma}
        \int_a^b E^{1-\Gamma}dE`, which is exactly ``ggah_cal``'s
        ``kcorr = (1+z)**(gamma_x - 2)`` inverted.  Writing that factor
        separately would be the same quantity twice.
        """
        for z in (0.0, 0.3, 1.0):
            for g in (1.8, 2.0, 2.4):
                got = float(A.absorbed_band_energy(0.5, 2.0, g, -30.0, z))
                want = (1.0 + z) ** (2.0 - g) * float(A.band_energy(0.5, 2.0, g))
                assert got == pytest.approx(want, rel=1e-10)

    def test_ten_to_the_twenty_is_not_zero_column(self):
        r"""Which is why the unobscured branch is fixed at zero and not at 1e20.

        :math:`\tau(0.5\,{\rm keV}) = 0.0736` at
        :math:`N_{\rm H} = 10^{20}\,{\rm cm}^{-2}`, so the soft band transmits
        0.970 -- a three per cent effect that "unobscured" should not quietly
        contain.
        """
        t = (float(A.absorbed_band_energy(0.5, 2.0, 1.8, 20.0, 0.0))
             / float(A.band_energy(0.5, 2.0, 1.8)))
        assert t == pytest.approx(0.9705, abs=1e-3)
        assert float(A.photoabs_tau(0.5, 20.0)) == pytest.approx(0.0736, abs=1e-4)

    @pytest.mark.parametrize("lnh,soft,hard",
                             [(22.0, 0.211, 0.918), (23.0, 0.001, 0.560)])
    def test_absorption_hits_the_soft_band_and_spares_the_hard(self, lnh, soft,
                                                               hard):
        """Which is the whole reason a soft-selected sample is not a hard one."""
        f = lambda a, b: (float(A.absorbed_band_energy(a, b, 1.8, lnh, 0.0))
                          / float(A.band_energy(a, b, 1.8)))
        assert f(0.5, 2.0) == pytest.approx(soft, abs=2e-3)
        assert f(2.0, 10.0) == pytest.approx(hard, abs=2e-3)

    def test_gamma_is_fittable_now(self):
        """The point of rewriting the band integral: a traced, finite gradient.

        The old `band_energy_ratio` returned a Python float and branched on
        `gamma == 2.0`, so the index could not be differentiated at all and had
        a hole at the one value a reader checks by hand.
        """
        g = float(jax.grad(lambda gm: A.band_energy(2.0, 10.0, gm))(1.8))
        assert np.isfinite(g) and g != 0.0
        at_two = float(jax.grad(lambda gm: A.band_energy(2.0, 10.0, gm))(2.0))
        assert np.isfinite(at_two), "Gamma = 2 must not be a hole"
        gn = float(jax.grad(
            lambda nh: A.absorbed_band_energy(0.5, 2.0, 1.8, nh, 0.2))(22.0))
        assert np.isfinite(gn) and gn < 0.0, "more column, less transmitted"

    def test_k_h2s_follows_the_index(self):
        """One quantity, one number -- the duplication this removed."""
        assert float(A.AgnParams(gamma_x=1.8).k_h2s) == \
            pytest.approx(A.HARD_TO_SOFT, rel=1e-12)
        assert float(A.AgnParams(gamma_x=2.0).k_h2s) != \
            pytest.approx(A.HARD_TO_SOFT, rel=1e-6)

    def test_the_selection_mode_is_validated(self):
        with pytest.raises(ValueError, match="selection mode"):
            A.AgnSector(GAL, selection="sort-of")

    def test_flux_selection_uses_the_fitted_spectrum(self, field):
        r"""And this is the collision it closes.

        ``ggah_cal`` converts its flux limit with its *own* fixed
        :math:`\Gamma`, so the spectrum the model fits and the spectrum the
        selection assumes were two numbers a fit could move apart.  Here the
        threshold moves when the index does.
        """
        s = A.AgnSector(GAL, calibration="off", selection="flux")
        l1 = float(s.flux_limit_luminosity(field, A.AgnParams(gamma_x=1.8)))
        l2 = float(s.flux_limit_luminosity(field, A.AgnParams(gamma_x=2.4)))
        assert l1 != l2, "the flux-to-luminosity conversion must see Gamma"
        l3 = float(s.flux_limit_luminosity(field, A.AgnParams(log10_nh=24.0)))
        assert l3 > l1, "a thicker column needs a brighter source to be seen"

    def test_luminosity_mode_is_untouched_by_the_flux_parameters(self, field,
                                                                 sector):
        """The default mode must not have moved, and the new knobs are inert in it."""
        p = A.AgnParams()
        n0 = float(field.number_density(sector.occupation(field, p, GP)[0]))
        n1 = float(field.number_density(sector.occupation(
            field, A.AgnParams(log10_flux_min=-15.0), GP)[0]))
        assert n0 == n1, "log10_flux_min is inert at selection='luminosity'"

    def test_flux_selection_changes_the_sample(self, field):
        s = A.AgnSector(GAL, calibration="off", selection="flux")
        p = A.AgnParams()
        n_flux = float(field.number_density(s.occupation(field, p, GP)[0]))
        assert n_flux > 0.0 and np.isfinite(n_flux)
        brighter = float(field.number_density(s.occupation(
            field, A.AgnParams(log10_flux_min=-12.0), GP)[0]))
        assert brighter < n_flux, "a brighter limit selects fewer AGN"


class TestTheObscuredFractionCanDependOnHaloMass:
    r"""The one term that lets the split be a measurement rather than a sign.

    The obscured fraction adapted from Comparat et al. (2019) is a function of
    :math:`L_X` and :math:`z` alone. Such a fraction **cannot** produce an obscured-against-unobscured
    bias difference at all: it cancels between the numerator and the
    denominator of an effective bias. So Petter et al. (2023)'s 9σ split is not
    a tension the published form fits badly but one it cannot express, at any
    value of its coefficients.

    ``l_blend_m_amp`` tilts the transition luminosity with host halo mass. Zero
    is the published model; the default, -0.873, is the eROSITA x DESI YR3
    MAP's.
    """

    def test_zero_is_the_published_model_exactly(self):
        lx = jnp.linspace(41.0, 45.0, 9)
        flat = np.asarray(A.obscured_fraction(lx, 0.1, C19))
        tilted = np.asarray(A.obscured_fraction(
            lx, 0.1, C19, log10m=jnp.asarray([12.0, 13.0, 14.0])))
        assert np.array_equal(tilted, np.broadcast_to(flat, tilted.shape))

    def test_without_it_no_split_is_expressible(self, field):
        r"""A halo-mass-independent weight cancels, so the ratio is one up to
        the sampling of the two populations' own $L_X(M_h)$: 0.948 at zero
        slopes, where the luminosity rises with host mass and the obscured,
        fainter AGN sit in lighter haloes, and 1.036 at the defaults, where the
        X-ray luminosity at fixed Eddington ratio falls with host mass
        (al_bh + log10_lstar_mslope = -1.16) and they sit in heavier ones.
        Never the 1.30 Petter et al. (2023) measure.
        """
        s = A.AgnSector(GAL, calibration="off", obscuration="split")

        def ratio(p):
            b = lambda br: float(s.effective_bias(field, p, GP, branch=br,
                                                  obsc=C19))
            return b("obscured") / b("unobscured")

        assert ratio(A.AgnParams(**NEUTRAL)) < 1.0
        assert ratio(A.AgnParams()) == pytest.approx(1.036, abs=5e-3)

    def test_the_tilt_reverses_the_sign(self, field):
        """The sign of ``l_blend_m_amp`` is the sign of the split: 0.74 at
        -1.5 and 1.21 at +1.5 at the defaults, where the untilted ratio is 1.09;
        at zero slopes, where it is below one, a positive tilt reverses it."""
        s = A.AgnSector(GAL, calibration="off", obscuration="split")

        def r(t, p):
            o = A.ObscurationParams(l_blend_m_amp=t)
            return (float(s.effective_bias(field, p, GP, branch="obscured",
                                           obsc=o))
                    / float(s.effective_bias(field, p, GP, branch="unobscured",
                                             obsc=o)))

        p = A.AgnParams()
        assert r(-1.5, p) < 1.0 < r(1.5, p)
        p0 = A.AgnParams(**NEUTRAL)
        assert r(0.0, p0) < 1.0 < r(1.5, p0)

    @pytest.mark.slow
    def test_it_reaches_petter_at_petters_own_sample_and_not_at_ours(self):
        r"""Which is why the configuration has to be quoted with the number.

        Petter et al. (2023) measure :math:`b = 2.96\pm0.07` obscured against
        :math:`2.27\pm0.06` unobscured --- a ratio of 1.30 --- for WISE-selected
        quasars at :math:`z \sim 1`--2, far brighter than an
        :math:`\lg L_X > 42` local sample.

        At that configuration, on the eROSITA c030 AGN defaults (``AGN_C030``)
        and the published fraction, this chain gives 1.20 with the tilt at 1.5, with
        absolute biases 1.96 and 1.63 against their 2.96 and 2.27 (1.33, 2.46
        and 1.85 with the compressed obscured fraction of before 1.3.0.dev0).
        At :math:`z = 0.135` and :math:`\lg L_X > 42` the same tilt reaches
        only 1.10, and the mechanism saturates near 1.10 however far the tilt is
        pushed: the AGN there occupy too narrow a range of halo mass for
        sorting them by it to produce more contrast.  A comparison against
        Petter run at the local configuration would have called the model
        wrong for being asked the wrong question.

        **Central AGN only** (``f_duty_sat = 0``), which is the population the
        tilt is defined on: ``satellite_occupation`` integrates the host out at
        each stellar mass before a host-mass factor could enter, so satellite
        AGN carry the untilted fraction.  With them in, the local ratio is 1.30
        -- two populations, one of them unsorted by halo mass -- and that is a
        statement about the missing satellite tilt, not about the mechanism.

        At the eROSITA x DESI YR3 AGN and obscuration defaults, with the tilt
        at 1.5, the local ratio is 1.19 and Petter's sample's 1.00, both below
        the 1.30 they measure.
        """
        s = A.AgnSector(GAL, calibration="off", obscuration="split")
        o = C19.replace(l_blend_m_amp=1.5)
        r = {}
        for z, lmin in ((0.135, 42.0), (1.0, 44.0)):
            f = make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                           make_pk("emu_pk"), z=z)
            p = A.AgnParams(**{**AGN_C030, "log10lx_min": lmin, "f_duty_sat": 0.0})
            r[z] = (float(s.effective_bias(f, p, GP, branch="obscured", obsc=o))
                    / float(s.effective_bias(f, p, GP, branch="unobscured", obsc=o)))
        assert r[1.0] == pytest.approx(1.20, abs=0.05), "Petter's own sample"
        assert r[0.135] < 1.15, "a local sample cannot reach it"
        assert r[1.0] > r[0.135]


class TestTheStellarMassCut:
    r"""No galaxy below :data:`~ggah_mod.sectors.agn.LG_MSTAR_MIN`
    (:math:`10^8\,M_\odot`, physical) hosts an AGN or a black hole: there is no
    evidence for active nuclei below it.  Satellites are cut exactly, on a
    physical stellar-mass grid; centrals through the fraction of them above the
    cut."""

    def test_the_default_cut_is_ten_to_the_eight(self):
        assert A.LG_MSTAR_MIN == 8.0
        assert A.AgnSector(GAL).lg_mstar_min == 8.0

    def test_the_central_fraction_is_the_lognormal_above_the_cut(self, field,
                                                                 sector):
        from scipy.special import erfc
        p = A.AgnParams()
        lgm = jnp.log10(field.m)
        ms = np.asarray(sector.log10_mstar(lgm, GP, h=field.cosmo.h, z=field.z))
        got = np.asarray(sector.central_fraction(lgm, p, GP, h=field.cosmo.h,
                                                 z=field.z))
        sig = np.asarray(sector.sigma_mstar(lgm, GP, h=field.cosmo.h))
        want = 0.5 * erfc((8.0 - ms) / (np.sqrt(2.0) * sig))
        assert np.allclose(got, want, rtol=1e-12, atol=1e-15)
        assert 0.0 < got.min() < 0.9 and got.max() == pytest.approx(1.0)

    def test_the_satellite_grid_starts_at_the_cut(self, field, sector):
        lg_ms, dn = sector._satellite_csmf(field, GP)
        assert float(lg_ms[0]) == 8.0
        assert float(lg_ms[-1]) == A.LG_MSTAR_MAX_SAT
        assert np.all(np.asarray(dn) >= -1e-12)

    def test_a_cut_above_every_galaxy_leaves_no_agn_and_no_black_hole(
            self, field, sector):
        """The guard on the whole: if any term escaped the cut -- a satellite
        read from the galaxy sector's own range, a central left unweighted --
        this would not vanish."""
        p = A.AgnParams()
        high = A.AgnSector(GAL, lg_mstar_min=12.45)
        assert float(high.omega_bh(field, p, GP)) < \
            1e-4 * float(sector.omega_bh(field, p, GP))
        n_cen, _ = high.occupation(field, p, GP)
        n_sat = high.satellite_occupation(field, p, GP)
        n_def = sector.occupation(field, p, GP)[0]
        assert float(field.number_density(n_cen + n_sat)) < \
            1e-4 * float(field.number_density(n_def))

    def test_a_higher_cut_counts_fewer(self, field, sector):
        p = A.AgnParams()
        nine = A.AgnSector(GAL, lg_mstar_min=9.0)
        assert float(nine.omega_bh(field, p, GP)) < \
            float(sector.omega_bh(field, p, GP))
        assert float(field.number_density(
            nine.satellite_occupation(field, p, GP))) < float(
            field.number_density(sector.satellite_occupation(field, p, GP)))


class TestTheCatalogueSelection:
    r"""1.3.0: ``selection="catalogue"``, a cut on the luminosity a soft-band
    catalogue *assigns*.

    The catalogue converts an observed 0.2--2.3 keV flux to rest 2--10 keV
    with an unabsorbed :math:`\Gamma = 1.8` power law, so an absorbed AGN is
    catalogued :math:`|\Delta_b|` dex below its intrinsic luminosity and the
    selected sample is a mixture of three branches, each crossing the
    threshold at its own :math:`L_X`.  What has to hold: with no obscured AGN
    it is the shipped unobscured threshold exactly; the unobscured shift
    vanishes at the catalogue's own index; the branches partition the sample;
    the gradient reaches every obscuration coefficient; and the
    Compton-thick branch is invisible in the soft band.
    """

    @pytest.fixture(scope="class")
    def cat(self):
        """The single-column mode, which these tests were written on; the
        Comparat et al. (2019) columns are TestComparat19Columns'."""
        return A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_nh="single")

    @staticmethod
    def _no_obscured_agn(monkeypatch):
        """``f_obs = 0`` exactly, in the shape the real one has."""
        real = A.obscured_fraction
        monkeypatch.setattr(A, "obscured_fraction",
                            lambda lx, z, p=None, log10m=None:
                            0.0 * real(lx, z, p, log10m=log10m))

    @staticmethod
    def _counts(s, field, p, **kw):
        n_cen, _ = s.occupation(field, p, GP, **kw)
        n_sat = s.satellite_occupation(field, p, GP, **kw)
        return np.asarray(n_cen), np.asarray(n_sat)

    def test_it_needs_the_split(self):
        with pytest.raises(ValueError, match="obscuration='split'"):
            A.AgnSector(GAL, selection="catalogue")
        with pytest.raises(ValueError, match="catalogue band"):
            A.AgnSector(GAL, obscuration="split", selection="catalogue",
                        catalogue_band=(2.3, 0.2))

    def test_with_no_obscured_agn_it_is_the_luminosity_threshold(
            self, field, cat, monkeypatch):
        """``f_obs = 0`` and ``gamma_x = 1.8``: every source is catalogued at
        its intrinsic luminosity, so the selection is the shipped one."""
        self._no_obscured_agn(monkeypatch)
        lum = A.AgnSector(GAL, calibration="off", obscuration="split")
        for lmin in (42.0, 43.0):
            p = A.AgnParams(log10lx_min=lmin, gamma_x=A.GAMMA_X)
            got = self._counts(cat, field, p)
            want = self._counts(lum, field, p)
            unobs = self._counts(lum, field, p, branch="unobscured")
            for g, w, u in zip(got, want, unobs):
                np.testing.assert_allclose(g, w, rtol=1e-10, atol=0.0)
                np.testing.assert_allclose(g, u, rtol=1e-10, atol=0.0)

    def test_with_no_obscured_agn_the_flux_limit_is_the_unabsorbed_one(
            self, field, monkeypatch):
        r"""The flux limit is a threshold on the same catalogue luminosity,
        :math:`\log(4\pi d_L^2F) + \log R(\Gamma) + (\Gamma-2)\log(1+z)` --
        the conversion ``desi_erosita_agn``'s s4/s11 apply to every source --
        and with no obscured AGN and ``gamma_x = 1.8`` it selects what that
        threshold selects."""
        from ggah_mod.cosmology.background import luminosity_distance
        from ggah_mod.cosmology import constants as C
        self._no_obscured_agn(monkeypatch)
        flux = A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_flux_limit=True,
                           catalogue_nh="single")
        p = A.AgnParams(log10lx_min=39.5, log10_flux_min=np.log10(2.5e-14),
                        gamma_x=A.GAMMA_X)
        d_cm = (float(np.atleast_1d(luminosity_distance(field.z, field.cosmo))[0])
                / field.cosmo.h * C.MPC_CM)
        g = 1.8
        ratio = ((10.0 ** (2 - g) - 2.0 ** (2 - g))
                 / (2.3 ** (2 - g) - 0.2 ** (2 - g)))
        want_l = (np.log10(4 * np.pi * d_cm ** 2 * 2.5e-14) + np.log10(ratio)
                  + (g - 2) * np.log10(1 + float(field.z)))
        got_l = float(flux.catalogue_threshold(p, field))
        assert got_l == pytest.approx(want_l, abs=1e-10)
        lum = A.AgnSector(GAL, calibration="off")
        got = self._counts(flux, field, p)
        want = self._counts(lum, field, p.replace(log10lx_min=got_l))
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=0.0)
        # The larger of the two thresholds applies.
        p_hi = p.replace(log10lx_min=got_l + 0.5)
        assert float(flux.catalogue_threshold(p_hi, field)) == got_l + 0.5

    def test_the_unobscured_shift_vanishes_at_the_catalogue_index(self, cat):
        for z in (0.0, 0.135, 0.3, 0.5):
            d = cat.catalogue_shift(A.AgnParams(gamma_x=1.8), z)
            assert float(d["unobscured"]) == 0.0, "exactly, not to round-off"
            d2 = cat.catalogue_shift(A.AgnParams(gamma_x=2.0), z)
            assert float(d2["unobscured"]) != 0.0

    def test_an_absorbed_source_is_catalogued_below_its_luminosity(self, cat):
        """Negative, falling with the column, and the Compton-thick branch far
        below any thin one.  At the catalogue's own index, where the
        unabsorbed shift is zero, so the shift is the absorption alone; at the
        default ``gamma_x`` = 1.837 the softer spectrum puts more of a source
        in the band than the catalogue's 1.8 assumes, and one behind
        10^20.5 cm^-2 is catalogued 0.03-0.05 dex below its luminosity at
        z = 0.3-0.5."""
        p = A.AgnParams(gamma_x=A.GAMMA_X)
        for z in (0.135, 0.3, 0.5):
            d = [float(cat.catalogue_shift(p.replace(log10_nh=nh), z)["thin"])
                 for nh in (20.5, 21.5, 22.5, 23.5)]
            assert all(x < 0.0 for x in d)
            assert np.all(np.diff(d) < 0.0)
            thick = float(cat.catalogue_shift(p, z)["thick"])
            assert thick < d[-1] - 5.0

    def test_the_log_quadrature_is_the_linear_one(self):
        for nh in (-30.0, 21.0, 22.5, 23.5):
            for z in (0.0, 0.3):
                lin = float(A.absorbed_band_energy(0.2, 2.3, 1.8, nh, z))
                log = float(A.log10_absorbed_band_energy(0.2, 2.3, 1.8, nh, z))
                assert log == pytest.approx(np.log10(lin), abs=1e-12)
        # And it stays finite where the linear one is denormal or zero.
        assert np.isfinite(float(A.log10_absorbed_band_energy(
            0.2, 2.3, 1.8, A.LOG10_NH_CT, 0.0)))

    def test_the_three_fractions_partition_the_population(self, cat, field):
        lm = jnp.log10(field.m)
        for o in (A.ObscurationParams(), A.ObscurationParams(f_faint_norm=0.2,
                                                             l_blend_m_amp=1.0)):
            b = A.obscuration_branches(cat.loglx, field.z, o, log10m=lm)
            tot = b["unobscured"] + b["thin"] + b["thick"]
            np.testing.assert_allclose(np.asarray(tot), 1.0, rtol=0, atol=1e-14)
            assert float(jnp.min(b["thin"])) >= -1e-15
            assert float(jnp.min(b["thick"])) >= -1e-15
        # At the defaults the cap is a guard: the thick share is the
        # Compton-thick fraction itself.
        b = A.obscuration_branches(cat.loglx, field.z)
        ct = np.asarray(A.compton_thick_fraction(cat.loglx, field.z))
        live = ct > 1e-30
        np.testing.assert_allclose(np.asarray(b["thick"])[live], ct[live],
                                   rtol=2e-6)
        # And where the obscured fraction falls below it, the cap holds.
        low = A.ObscurationParams(f_faint_norm=0.05, f_bright_floor=0.0,
                                  f_bright_amp=0.0)
        b = A.obscuration_branches(cat.loglx, field.z, low)
        f = np.asarray(A.obscured_fraction(cat.loglx, field.z, low))
        assert np.all(np.asarray(b["thick"]) <= f + 1e-15)

    def test_the_branches_add_to_the_mixed_sample(self, field, cat):
        """Before the ceiling for the centrals -- each branch carries its own
        ``tanh`` -- and exactly for the satellites, which have none."""
        p = A.AgnParams()
        tot = self._counts(cat, field, p)
        u = self._counts(cat, field, p, branch="unobscured")
        o = self._counts(cat, field, p, branch="obscured")
        np.testing.assert_allclose(np.arctanh(u[0]) + np.arctanh(o[0]),
                                   np.arctanh(tot[0]), rtol=1e-10, atol=1e-300)
        np.testing.assert_allclose(u[1] + o[1], tot[1], rtol=1e-12, atol=0.0)
        # Both sides are populated: this is a mixture, not a relabelling.
        assert float(field.number_density(o[0])) > \
            0.05 * float(field.number_density(tot[0]))
        for band in (None, (0.5, 2.0)):
            kw = {} if band is None else {"band": band}
            w = cat.emission_weights(field, p, GP, **kw)
            wu = cat.emission_weights(field, p, GP, branch="unobscured", **kw)
            wo = cat.emission_weights(field, p, GP, branch="obscured", **kw)
            for attr in ("w_point", "w_extended", "self_pair"):
                np.testing.assert_allclose(
                    np.asarray(getattr(wu, attr)) + np.asarray(getattr(wo, attr)),
                    np.asarray(getattr(w, attr)), rtol=1e-12, atol=0)

    def test_compton_thick_sources_are_not_in_a_soft_band_catalogue(
            self, field, cat):
        """The thick branch's share of the selected counts, isolated with a
        per-branch weight: invisible, as :data:`LOG10_NH_CT` says."""
        p = A.AgnParams()
        prob = cat.p_active_loglx_given_m(jnp.log10(field.m), p, GP,
                                          h=field.cosmo.h, z=field.z)
        kw = dict(field=field, z=field.z, log10m=jnp.log10(field.m))
        only = {"unobscured": 0.0, "thin": 0.0, "thick": 1.0}
        thick = field.number_density(cat._moment(prob, p, 0, lweight=only, **kw))
        alls = field.number_density(cat._moment(prob, p, 0, **kw))
        assert 0.0 <= float(thick) < 1e-3 * float(alls)

    @pytest.mark.x64
    def test_the_gradient_reaches_every_obscuration_coefficient(self, field,
                                                                cat):
        r"""Centrals and satellites, by autodiff and by a central difference.
        ``l_blend_m_amp`` is a host-mass tilt, which the satellites do not
        carry (``satellite_occupation``), so its satellite gradient is zero."""
        p0, o0 = A.AgnParams(), A.ObscurationParams()

        def counts(p, o):
            n_cen, _ = cat.occupation(field, p, GP, obsc=o)
            n_sat = cat.satellite_occupation(field, p, GP, obsc=o)
            return (jnp.log(field.number_density(n_cen)),
                    jnp.log(field.number_density(n_sat)))

        for which in (0, 1):
            g = jax.grad(lambda o: counts(p0, o)[which])(o0)
            for name in A.ObscurationParams._PARAMS:
                v = float(getattr(g, name))
                assert np.isfinite(v), (which, name)
                if which == 1 and name == "l_blend_m_amp":
                    assert v == 0.0
                else:
                    assert v != 0.0, (which, name)
            for key in ("gamma_x", "log10_nh"):
                f = lambda x: counts(p0.replace(**{key: x}), o0)[which]
                x = float(getattr(p0, key))
                ad = float(jax.grad(f)(x))
                h = 1e-5 * max(abs(x), 1.0)
                fd = float((f(x + h) - f(x - h)) / (2 * h))
                assert np.isfinite(ad) and ad != 0.0, (which, key)
                assert ad == pytest.approx(fd, rel=1e-4), (which, key)
        name = "f_faint_norm"
        f = lambda x: counts(p0, o0.replace(**{name: x}))[0]
        ad = float(jax.grad(f)(0.9))
        fd = float((f(0.9 + 1e-6) - f(0.9 - 1e-6)) / 2e-6)
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_the_observed_xlf_is_opt_in_and_reduces_to_the_intrinsic_one(
            self, field, cat, sector, monkeypatch):
        """With no obscured AGN and ``gamma_x = 1.8`` the observed XLF is the
        intrinsic one."""
        p = A.AgnParams(gamma_x=A.GAMMA_X)
        g0, phi0 = cat.xlf(field, p, GP)
        gs, phis = sector.xlf(field, p, GP)
        np.testing.assert_array_equal(np.asarray(phi0), np.asarray(phis))
        with pytest.raises(ValueError, match="catalogue"):
            sector.xlf(field, p, GP, observed=True)
        g, phi = cat.xlf(field, p, GP, observed=True)
        # Obscured AGN move down in catalogue luminosity: fewer above 10^43.
        hi = np.asarray(g) > 43.0
        assert np.sum(np.asarray(phi)[hi]) < np.sum(np.asarray(phi0)[hi])
        self._no_obscured_agn(monkeypatch)
        g, phi = cat.xlf(field, p, GP, observed=True)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi0),
                                   rtol=1e-12, atol=0.0)

    def test_the_default_modes_do_not_read_it(self, field, sector):
        """``obsc`` reaches ``weights`` now, and changes nothing where no
        obscured fraction is read."""
        p = A.AgnParams()
        w0 = sector.weights(field, p, GP)
        w1 = sector.weights(field, p, GP,
                            obsc=A.ObscurationParams(f_faint_norm=0.1))
        np.testing.assert_array_equal(np.asarray(w0.w_point),
                                      np.asarray(w1.w_point))


class TestTheObscurationReachesTheSpectrum:
    r"""1.3.0: ``params["obscuration"]`` reaches the AGN adapters.

    Before, ``spectra.tracers`` called the sector without the obscuration
    coefficients, so a band or branch view -- and now the catalogue counts --
    was always built at their defaults, whatever a fit had set them to.
    """

    def _inputs(self, sector, obsc=None):
        sectors = {"agn": sector, "galaxies": GAL}
        params = {"agn": A.AgnParams(), "galaxies": GP}
        if obsc is not None:
            params["obscuration"] = obsc
        return sectors, params

    def test_the_counts_view_carries_it(self, field):
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        cat = A.AgnSector(GAL, calibration="off", obscuration="split",
                          selection="catalogue")
        o = A.ObscurationParams(f_faint_norm=0.3)
        c = Component("agn", "counts")
        got = build_weights(c, field, *self._inputs(cat, o))
        want = cat.weights(field, A.AgnParams(), GP, obsc=o)
        np.testing.assert_array_equal(np.asarray(got.w_point),
                                      np.asarray(want.w_point))
        default = build_weights(c, field, *self._inputs(cat))
        assert not np.allclose(np.asarray(default.w_point),
                               np.asarray(got.w_point), rtol=1e-3)

    def test_a_band_view_carries_it(self, field):
        import ggah_mod.spectra as SP
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        split = A.AgnSector(GAL, calibration="off", obscuration="split")
        c = Component("agn", "emission", population="agn",
                      band=SP.Band(0.5, 2.0))
        o = A.ObscurationParams(f_faint_norm=0.3)
        got = build_weights(c, field, *self._inputs(split, o))
        want = split.emission_weights(field, A.AgnParams(), GP,
                                      band=(0.5, 2.0), obsc=o)
        np.testing.assert_array_equal(np.asarray(got.w_point),
                                      np.asarray(want.w_point))
        default = build_weights(c, field, *self._inputs(split))
        assert not np.allclose(np.asarray(default.w_point),
                               np.asarray(got.w_point), rtol=1e-3)



class TestComparat19Columns:
    r"""``catalogue_nh="comparat19"``, the catalogue mode's default: every AGN
    at a :math:`\log N_{\rm H}` uniform in its class -- [20, 22), [22, 24),
    [24, 26) -- as Comparat et al. (2019) Sec. 3.2 assign them, and the
    selection averaged over each class by Gauss-Legendre."""

    @pytest.fixture(scope="class")
    def c19(self):
        return A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue")

    @staticmethod
    def _counts(s, field, p, **kw):
        n_cen, _ = s.occupation(field, p, GP, **kw)
        n_sat = s.satellite_occupation(field, p, GP, **kw)
        return (float(field.number_density(n_cen)),
                float(field.number_density(n_sat)))

    def test_it_is_the_default_and_validated(self, c19):
        assert c19.catalogue_nh == "comparat19"
        assert c19.catalogue_nh_classes == A.CATALOGUE_NH_CLASSES
        with pytest.raises(ValueError, match="catalogue_nh"):
            A.AgnSector(GAL, obscuration="split", selection="catalogue",
                        catalogue_nh="two")

    def test_a_negligible_column_everywhere_is_the_unobscured_threshold(
            self, field):
        """Every class moved to [-30, -30 + 1e-9) and ``gamma_x`` at the
        catalogue's 1.8: nothing is absorbed, so every branch is catalogued at
        its intrinsic luminosity and the selection is the luminosity
        threshold's, whatever the fractions."""
        zero = A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_scatter=0.0,
                           catalogue_nh_classes={
                               b: (-30.0, -30.0 + 1e-9) for b in A.CATALOGUE_BRANCHES})
        lum = A.AgnSector(GAL, calibration="off", obscuration="split")
        p = A.AgnParams(gamma_x=A.GAMMA_X)
        for a, b in zip(self._counts(zero, field, p), self._counts(lum, field, p)):
            assert a == pytest.approx(b, rel=1e-10)

    def test_the_lowest_unobscured_column_is_a_shifted_threshold(self, field):
        """All classes at 10^20: one shift for every AGN, the threshold moved
        by it -- the unobscured result, dimmed by 10^20 of gas.  At the
        catalogue's 1.8, so the shift is the gas alone: at the default
        ``gamma_x`` = 1.837 the spectrum's own shift, +0.03 dex at z = 0.135,
        nearly cancels the gas's -0.03."""
        at20 = A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_scatter=0.0,
                           catalogue_nh_classes={
                               b: (20.0, 20.0 + 1e-9) for b in A.CATALOGUE_BRANCHES})
        p = A.AgnParams(gamma_x=A.GAMMA_X)
        d = np.asarray(at20.catalogue_shift(p, field.z)["unobscured"])
        assert np.all(d < 0) and np.ptp(d) < 1e-8
        lum = A.AgnSector(GAL, calibration="off", obscuration="split")
        want = self._counts(lum, field, p.replace(log10lx_min=p.log10lx_min - float(d[0])))
        for a, b in zip(self._counts(at20, field, p), want):
            assert a == pytest.approx(b, rel=1e-8)

    @pytest.mark.parametrize("branch", ["unobscured", "thin"])
    def test_more_column_selects_fewer(self, field, branch):
        p = A.AgnParams()
        base = A.CATALOGUE_NH_CLASSES[branch]
        n = []
        for up in (0.0, 0.5, 1.0):
            s = A.AgnSector(GAL, calibration="off", obscuration="split",
                            selection="catalogue", catalogue_nh_classes={
                                branch: (base[0] + up, base[1] + up)})
            n.append(sum(self._counts(s, field, p)))
        assert n[0] > n[1] > n[2]

    def test_the_shift_falls_across_each_class_and_the_thick_class_is_dark(
            self, field):
        """Without the scattered light (catalogue_scatter=0)."""
        dark = A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_scatter=0.0)
        d = dark.catalogue_shift(A.AgnParams(), field.z)
        for b in A.CATALOGUE_BRANCHES:
            assert np.all(np.diff(np.asarray(d[b])) < 0), b
        assert float(np.max(d["thin"])) < float(np.min(d["unobscured"]))
        assert float(np.max(d["thick"])) < -5.0

    def test_scattered_light_puts_compton_thick_agn_1p7_dex_down(self, c19,
                                                                  field):
        r"""The default, F_SCATTER = 0.02 of the intrinsic power law unabsorbed:
        a thick source is catalogued at log10(0.02) plus the unabsorbed shift,
        the thin class tends to that floor from above, and the thick class is
        no longer dark -- it is a few per cent of the selected counts."""
        p = A.AgnParams()
        assert c19.catalogue_scatter == A.F_SCATTER
        d = c19.catalogue_shift(p, field.z)
        zero = A.AgnSector(GAL, calibration="off", obscuration="split",
                           selection="catalogue", catalogue_scatter=0.0,
                           catalogue_nh_classes={b: (-30.0, -30.0 + 1e-9)
                                                 for b in A.CATALOGUE_BRANCHES})
        d0 = float(np.asarray(zero.catalogue_shift(p, field.z)["unobscured"])[0])
        floor = d0 + np.log10(A.F_SCATTER)
        np.testing.assert_allclose(np.asarray(d["thick"]), floor, atol=1e-6)
        thin = np.asarray(d["thin"])
        assert np.all(thin >= floor - 1e-9) and np.all(np.diff(thin) <= 0)
        prob = c19.p_active_loglx_given_m(jnp.log10(field.m), p, GP,
                                          h=field.cosmo.h, z=field.z)
        kw = dict(field=field, z=field.z, log10m=jnp.log10(field.m))
        n = c19.catalogue_nh_nodes
        only = {"unobscured": jnp.zeros(n), "thin": jnp.zeros(n), "thick": jnp.ones(n)}
        share = float(field.number_density(c19._moment(prob, p, 0, lweight=only, **kw))
                      / field.number_density(c19._moment(prob, p, 0, **kw)))
        assert 0.005 < share < 0.2
        single = A.AgnSector(GAL, calibration="off", obscuration="split",
                             selection="catalogue", catalogue_nh="single")
        assert single.catalogue_scatter == 0.0

    def test_log10_nh_is_not_read(self, c19, field):
        p = A.AgnParams()
        a = self._counts(c19, field, p)
        b = self._counts(c19, field, p.replace(log10_nh=24.9))
        assert a == b

    def test_gauss_legendre_has_converged(self, field):
        p = A.AgnParams()
        n = {k: self._counts(A.AgnSector(GAL, calibration="off", obscuration="split",
                                         selection="catalogue",
                                         catalogue_nh_nodes=k), field, p)
             for k in (16, 32, 64)}
        # the defaults' 0.05-dex changeover is a near-step in the column
        # average: 16 nodes (the default) are 2.8e-6 off for the satellites,
        # 32 are 1.2e-6 off
        for i in (0, 1):
            assert n[16][i] == pytest.approx(n[64][i], rel=5e-4)
            assert n[32][i] == pytest.approx(n[64][i], rel=1e-5)

    def test_the_branches_add_and_the_emission_too(self, c19, field):
        p = A.AgnParams()
        tot = np.asarray(c19.occupation(field, p, GP)[0])
        u = np.asarray(c19.occupation(field, p, GP, branch="unobscured")[0])
        o = np.asarray(c19.occupation(field, p, GP, branch="obscured")[0])
        np.testing.assert_allclose(np.arctanh(u) + np.arctanh(o), np.arctanh(tot),
                                   rtol=1e-10, atol=1e-300)
        for band in (None, (0.5, 2.0)):
            kw = {} if band is None else {"band": band}
            w = c19.emission_weights(field, p, GP, **kw)
            wu = c19.emission_weights(field, p, GP, branch="unobscured", **kw)
            wo = c19.emission_weights(field, p, GP, branch="obscured", **kw)
            np.testing.assert_allclose(np.asarray(wu.w_point) + np.asarray(wo.w_point),
                                       np.asarray(w.w_point), rtol=1e-12, atol=0)
        g, phi = c19.xlf(field, p, GP, observed=True)
        assert np.all(np.isfinite(np.asarray(phi)))

    @pytest.mark.x64
    def test_the_gradient_reaches_the_coefficients_and_the_index(self, c19, field):
        p0, o0 = A.AgnParams(), A.ObscurationParams()

        def total(p, o):
            n_cen, _ = c19.occupation(field, p, GP, obsc=o)
            return jnp.log(field.number_density(n_cen))

        g = jax.grad(lambda o: total(p0, o))(o0)
        for name in A.ObscurationParams._PARAMS:
            v = float(getattr(g, name))
            assert np.isfinite(v) and v != 0.0, name
        f = lambda x: total(p0.replace(gamma_x=x), o0)              # noqa: E731
        ad = float(jax.grad(f)(1.8))
        fd = float((f(1.8 + 1e-5) - f(1.8 - 1e-5)) / 2e-5)
        assert np.isfinite(ad) and ad == pytest.approx(fd, rel=1e-4)
        assert float(jax.grad(lambda x: total(p0.replace(log10_nh=x), o0))(22.5)) == 0.0


class TestTheObservedHardBandXlf:
    r"""``xlf(observed="aird15")``, :meth:`AgnSector.xlf_observed_hard`: the XLF
    in Aird et al. (2015)'s luminosity, which neglects absorption (their
    Sec. 4) -- the quantity their LADE fit describes.  Each Comparat et al.
    (2019) class's intrinsic XLF is moved by its columns' shifts in an observed
    2--7 keV band against a Gamma = 1.9, 10^20-21 conversion, with 2 per cent
    scattered light on the obscured classes."""

    @pytest.fixture(scope="class")
    def split(self):
        return A.AgnSector(GAL, calibration="off", obscuration="split")

    def test_it_needs_the_split(self, field, sector):
        with pytest.raises(ValueError, match="obscuration"):
            sector.xlf(field, A.AgnParams(), GP, observed="aird15")

    @pytest.mark.parametrize("gamma", [1.9, 1.8])
    def test_no_column_is_the_intrinsic_xlf_moved_by_the_index_alone(
            self, field, gamma):
        """Every class at a negligible column and no scattering: one shift,
        the photon-index mismatch against Aird's 1.9 (and their 10^20-21
        conversion column), and nothing else."""
        s = A.AgnSector(GAL, calibration="off", obscuration="split",
                        catalogue_nh_classes={b: (-30.0, -30.0 + 1e-9)
                                              for b in A.CATALOGUE_BRANCHES})
        p = A.AgnParams(gamma_x=gamma)
        d = s.observed_hard_shift(p, field.z, f_scatter=0.0)
        allv = np.concatenate([np.asarray(v[0]) for v in d.values()])
        assert np.ptp(allv) < 1e-9
        if gamma == 1.9:
            # Only the survey's own 10^20-21 conversion column: a few 1e-3 dex.
            assert 0.0 < allv[0] < 3e-3
        g, phi0 = s.xlf(field, p, GP)
        _, phi = s.xlf_observed_hard(field, p, GP, f_scatter=0.0)
        want = np.interp(np.asarray(g) - allv[0], np.asarray(g), np.asarray(phi0),
                         left=0.0, right=0.0)
        np.testing.assert_allclose(np.asarray(phi), want, rtol=1e-10, atol=1e-300)

    def test_obscured_agn_move_to_lower_luminosity(self, split, field):
        # Without the scattered light the column alone orders the classes; with
        # it the bottom of the thin class gains 2 per cent over the top of the
        # unobscured one, so it is the class means that order.
        d0 = split.observed_hard_shift(A.AgnParams(), field.z, f_scatter=0.0)
        u, t, k = (np.asarray(d0[b][0]) for b in A.CATALOGUE_BRANCHES)
        assert np.all(np.diff(t) < 0) and t.max() < u.min()
        assert np.all(k <= t.min() + 1e-12)
        d = split.observed_hard_shift(A.AgnParams(), field.z)
        means = [float(np.sum(np.asarray(d[b][0]) * np.asarray(d[b][1])))
                 for b in A.CATALOGUE_BRANCHES]
        assert means[0] > means[1] > means[2]
        g, phi0 = split.xlf(field, A.AgnParams(), GP)
        _, phi = split.xlf(field, A.AgnParams(), GP, observed="aird15")
        hi = np.asarray(g) > 43.0
        assert np.sum(np.asarray(phi)[hi]) < np.sum(np.asarray(phi0)[hi])

    def test_compton_thick_agn_appear_through_their_scattered_light(self, split,
                                                                    field):
        """Above 10^25 the absorbed power law is gone and the class sits at
        log10(0.02) below the unabsorbed one; without scattering it vanishes."""
        p = A.AgnParams()
        d = split.observed_hard_shift(p, field.z)
        d0 = split.observed_hard_shift(p, field.z, f_scatter=0.0)
        nodes = np.asarray(split.catalogue_columns(p)["thick"][0])
        k, k0 = np.asarray(d["thick"][0]), np.asarray(d0["thick"][0])
        floor = float(np.max(np.asarray(d0["unobscured"][0]))) + np.log10(A.F_SCATTER)
        top = nodes > 25.0
        assert np.all(np.abs(k[top] - k[top].mean()) < 1e-3)
        assert np.all(k0[top] < k[top] - 5.0)
        assert float(np.min(k)) > floor - 0.1

    @pytest.mark.x64
    def test_it_differentiates_in_the_coefficients_and_the_index(self, split,
                                                                  field):
        p0, o0 = A.AgnParams(), A.ObscurationParams()
        g = np.asarray(split.loglx)
        w = jnp.asarray(((g > 43.0) & (g < 44.0)).astype(float))

        def n(p, o):
            _, phi = split.xlf_observed_hard(field, p, GP, obsc=o)
            return jnp.log(jnp.sum(w * phi))

        gr = jax.grad(lambda o: n(p0, o))(o0)
        assert all(np.isfinite(float(getattr(gr, k))) for k in A.ObscurationParams._PARAMS)
        assert float(gr.l_blend_0) != 0.0
        ad = float(jax.grad(lambda x: n(p0.replace(gamma_x=x), o0))(1.8))
        assert np.isfinite(ad) and ad != 0.0



class TestTheObscuredFractionIsComparat19:
    r"""1.3.0.dev0: :func:`obscured_fraction` is Comparat et al. (2019) Eqs.
    4-11 as published -- f_thick = 0.3 constant, the bright branch's 0.4, and a
    ceiling that no longer compresses it (``soft_saturate`` turned their
    faint-end 0.889 into 0.589)."""

    @staticmethod
    def published(l, z):
        from scipy.special import erf as serf
        f1 = 0.3 + 0.01 + 0.4 * serf(z / 4.0)
        f2 = 0.9 * np.sqrt(41.0 / l)
        ll = 43.2 + 1.2 * serf(z)
        b = 0.5 + 0.5 * serf((ll - l) / 0.6)
        return f1 + (f2 - f1) * b

    C19 = A.ObscurationParams(**A.OBSCURATION_PUBLISHED)

    def test_it_is_their_eqs_4_to_11(self):
        lx = np.linspace(38.0, 48.0, 401)
        for z in (0.0, 0.19, 0.5, 1.0, 3.0, 6.0):
            np.testing.assert_allclose(np.asarray(A.obscured_fraction(lx, z, self.C19)),
                                       self.published(lx, z), atol=1e-4, rtol=0)
        assert float(A.obscured_fraction(42.0, 0.19, self.C19)) == pytest.approx(0.889, abs=1e-3)

    def test_it_is_their_fig_6(self):
        assert float(A.obscured_fraction(38.0, 0.0, self.C19)) == pytest.approx(0.935, abs=2e-3)
        assert float(A.obscured_fraction(48.0, 0.0, self.C19)) == pytest.approx(0.31, abs=1e-6)
        assert float(A.obscured_fraction(48.0, 6.0, self.C19)) == pytest.approx(0.696, abs=1e-3)
        ct = np.asarray(A.compton_thick_fraction(np.linspace(38, 48, 11), 0.7))
        np.testing.assert_array_equal(ct, 0.3)

    def test_the_ceiling_is_exact_below_it_and_smooth_through_it(self):
        f = jnp.linspace(0.0, 0.95, 50)
        assert float(jnp.max(jnp.abs(A._smooth_min(f, 1.0, A.F_OBS_CEILING_SHARPNESS)
                                     - f))) < 1e-4
        g = jax.vmap(jax.grad(lambda x: A._smooth_min(x, 1.0, A.F_OBS_CEILING_SHARPNESS)))(
            jnp.linspace(0.5, 1.5, 21))
        assert np.all(np.asarray(g) > 0.0) and np.all(np.isfinite(np.asarray(g)))
        far = float(A.obscured_fraction(30.0, 0.0, self.C19))
        assert far < 1.0, "the faint branch exceeds one below 1e33 and is capped"

    def test_the_classes_are_their_shares(self):
        lx = jnp.linspace(38.0, 48.0, 101)
        b = A.obscuration_branches(lx, 0.3, self.C19)
        np.testing.assert_allclose(np.asarray(b["thick"]), 0.3, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(b["thin"]),
                                   self.published(np.asarray(lx), 0.3) - 0.3, atol=1e-4)
        assert A.ObscurationParams().f_bright_amp == 0.4


class TestTheDutyCycleDependsOnStellarMass:
    r"""``log10_ferdf_mslope`` (1.3.0.dev0): the active fraction
    :math:`\log f_{\rm ERDF}(M_*) = {\tt log10\_ferdf} + s(\log M_* - 10.5)`,
    a central's at its halo's mean stellar mass, a satellite's at its own.

    Igo et al. (2026) motivate it: the constant duty cycle over-predicts their
    :math:`p(\lambda_{\rm SAR}|M_*)` by 0.3--0.6 dex at
    :math:`10^{11}`--:math:`10^{12}\,M_\odot`.
    """

    SLOPE = -0.5

    @pytest.fixture(scope="class")
    def sectors(self):
        return dict(
            lum=A.AgnSector(GAL, calibration="off"),
            split=A.AgnSector(GAL, calibration="off", obscuration="split"),
            cat=A.AgnSector(GAL, calibration="off", obscuration="split",
                            selection="catalogue", catalogue_flux_limit=True))

    @staticmethod
    def _outputs(sec, field, p):
        """Every output the duty cycle reaches, as ``{name: array}``."""
        lum, split, cat = sec["lum"], sec["split"], sec["cat"]
        o = A.ObscurationParams(l_blend_m_amp=0.2)
        out = {}
        out["occupation"], out["occupation_lx"] = lum.occupation(field, p, GP)
        out["satellite_occupation"] = lum.satellite_occupation(field, p, GP)
        out["satellite_lx"] = lum.satellite_occupation(field, p, GP, 1)
        out["xlf"] = lum.xlf(field, p, GP)[1]
        out["effective_bias"] = lum.effective_bias(field, p, GP)
        out["bias_obscured"] = split.effective_bias(field, p, GP,
                                                    branch="obscured", obsc=o)
        w = lum.weights(field, p, GP)
        out["n_bar"], out["w_extended"] = w.norm, w.w_extended
        e = lum.emission_weights(field, p, GP)
        out["emission"], out["emission_self"] = e.w_point, e.self_pair
        e = split.emission_weights(field, p, GP, band=(0.5, 2.0), obsc=o)
        out["emission_band"], out["emission_band_ext"] = e.w_point, e.w_extended
        out["xlf_aird15"] = split.xlf(field, p, GP, observed="aird15", obsc=o)[1]
        out["catalogue_occupation"] = cat.occupation(field, p, GP, obsc=o)[0]
        out["catalogue_satellites"] = cat.satellite_occupation(field, p, GP, obsc=o)
        out["catalogue_n_bar"] = cat.weights(field, p, GP, obsc=o).norm
        out["catalogue_xlf"] = cat.xlf(field, p, GP, observed=True, obsc=o)[1]
        out["validity_cost"] = lum.validity_cost(field, p, GP, quantity="n_agn")
        return {k: np.asarray(v) for k, v in out.items()}

    def test_it_is_declared_in_both_copies_with_a_reason(self):
        par = A.AgnParams._PARAMS["log10_ferdf_mslope"]
        assert A.AgnParams().log10_ferdf_mslope == 2.12 == par.default
        assert par.bounds == (-3.0, 3.0)
        assert isinstance(par.prior, A.Flat)
        assert par.unit == "" and par.kind == "physical"
        assert "Igo et al. (2026)" in par.why and "10^10.5" in par.why
        assert "ZERO REPRODUCES THE CONSTANT DUTY CYCLE EXACTLY" in par.why
        assert "YR3" in par.why and str(par.default) in par.why
        assert A.LG_MSTAR_DUTY_PIVOT == 10.5

    def test_the_helper_is_the_formula(self):
        p = A.AgnParams(log10_ferdf=-1.2, log10_ferdf_mslope=self.SLOPE)
        lg = jnp.linspace(8.0, 12.5, 37)
        raw = 10.0 ** (-1.2 + self.SLOPE * (np.asarray(lg) - 10.5))
        k = A.DUTY_CEILING_SHARPNESS
        capped = -np.logaddexp(-k * raw, -k) / k
        assert raw.max() > 1.0          # the grid reaches the cap
        np.testing.assert_allclose(np.asarray(A.duty(lg, p)), capped,
                                   rtol=1e-13)
        np.testing.assert_allclose(np.asarray(A.log10_duty(lg, p)),
                                   np.log10(capped), rtol=0, atol=1e-13)
        below = raw < 0.4
        np.testing.assert_allclose(np.asarray(A.duty(lg, p))[below],
                                   raw[below], rtol=1e-13)
        # The pivot is log10_ferdf whatever the slope, and the methods are the
        # functions themselves.
        assert float(A.duty(10.5, p)) == float(jnp.power(10.0, -1.2))
        assert A.AgnSector.duty is A.duty
        assert A.AgnSector.log10_duty is A.log10_duty

    def test_the_fraction_is_capped_at_one(self):
        """At the shipped slope the uncapped fraction exceeds one above
        :math:`\\log M_* = 11.33`; the smooth minimum keeps it at or below
        one, non-decreasing (one to double precision once saturated), the
        uncapped value to the last bit below 0.469, and differentiable through
        the cap."""
        p = A.AgnParams()
        lg = jnp.linspace(8.0, 12.5, 451)
        raw = (np.power(10.0, p.log10_ferdf)
               * np.power(10.0, p.log10_ferdf_mslope * (np.asarray(lg) - 10.5)))
        d = np.asarray(A.duty(lg, p))
        assert raw.max() > 2.0
        assert np.all(d <= 1.0) and np.all(np.diff(d) >= 0.0)
        assert np.all(np.diff(d)[raw[1:] < 1.0] > 0.0)
        np.testing.assert_array_equal(d[raw < 0.469], raw[raw < 0.469])
        np.testing.assert_allclose(d[raw > 2.0], 1.0, rtol=0, atol=1e-15)
        assert float(A.duty(10.5 + (0.0 - p.log10_ferdf) / p.log10_ferdf_mslope,
                            p)) == pytest.approx(1.0 - np.log(2.0) / 64.0,
                                                 rel=1e-12)

        def f(s, lf):
            q = A.AgnParams(log10_ferdf=lf, log10_ferdf_mslope=s)
            return jnp.sum(A.duty(lg, q))

        g = jax.grad(f, argnums=(0, 1))(p.log10_ferdf_mslope, p.log10_ferdf)
        assert all(np.isfinite(float(x)) and float(x) > 0.0 for x in g)

    def test_slope_zero_is_bit_for_bit_the_constant_duty_cycle(
            self, field, sectors, monkeypatch):
        """Every output at slope zero against the same output with the
        helper replaced by the constant ``10**log10_ferdf`` it superseded:
        equal to the last bit, not to round-off."""
        p = A.AgnParams(log10_ferdf_mslope=0.0)
        new = self._outputs(sectors, field, p)

        def constant(log10_mstar, q):
            return jnp.broadcast_to(jnp.power(10.0, q.log10_ferdf),
                                    jnp.shape(log10_mstar))

        monkeypatch.setattr(A, "duty", constant)
        old = self._outputs(sectors, field, p)
        assert set(new) == set(old)
        for k in new:
            np.testing.assert_array_equal(new[k], old[k], err_msg=k)

    def test_a_negative_slope_moves_agn_from_massive_hosts_to_light_ones(
            self, field, sectors):
        lum = sectors["lum"]
        # Far below the cap of one at every stellar mass of the grids, so the
        # ratios are the uncapped ones.
        p0 = A.AgnParams(log10_ferdf=-2.0, log10_ferdf_mslope=0.0)
        p1 = A.AgnParams(log10_ferdf=-2.0, log10_ferdf_mslope=self.SLOPE)
        lm = jnp.log10(field.m)
        ms = np.asarray(lum.log10_mstar(lm, GP, h=field.cosmo.h, z=field.z))
        # Centrals: the raw moment scales by the duty ratio exactly, and the
        # tanh-saturated occupation follows its sign.
        ratio = np.asarray(lum.central_duty(lm, p1, GP, h=field.cosmo.h, z=field.z)
                           / lum.central_duty(lm, p0, GP, h=field.cosmo.h, z=field.z))
        np.testing.assert_allclose(ratio, 10.0 ** (self.SLOPE * (ms - 10.5)),
                                   rtol=1e-12)
        n0 = np.asarray(lum.occupation(field, p0, GP)[0])
        n1 = np.asarray(lum.occupation(field, p1, GP)[0])
        live = n0 > 1e-12
        heavy, light = live & (ms > 10.6), live & (ms < 10.4)
        assert heavy.any() and light.any()
        assert np.all(n1[heavy] < n0[heavy]) and np.all(n1[light] > n0[light])
        # Satellites: each stellar mass by its own duty ratio.
        _, _, s0 = lum._satellite_kernel(field, p0, GP)
        lg_s, _, s1 = lum._satellite_kernel(field, p1, GP)
        on = np.asarray(s0) > 0
        np.testing.assert_allclose(
            np.asarray(s1)[on],
            (np.asarray(s0) * 10.0 ** (self.SLOPE * (np.asarray(lg_s)[:, None]
                                                     - 10.5)))[on], rtol=1e-12)
        # The AGN number density: down in hosts above the pivot, up below it.
        dens = {}
        for name, p in (("p0", p0), ("p1", p1)):
            n = lum.occupation(field, p, GP)[0]
            dens[name] = (float(field.number_density(n * (ms > 11.0))),
                          float(field.number_density(n * (ms < 10.0))))
        assert dens["p1"][0] < 0.7 * dens["p0"][0]
        assert dens["p1"][1] > 1.3 * dens["p0"][1]

    @pytest.mark.x64
    @pytest.mark.parametrize("slope", [0.0, -0.5])
    def test_the_gradient_reaches_the_slope(self, field, sectors, slope):
        cat = sectors["cat"]
        o = A.ObscurationParams()

        def f(s):
            p = A.AgnParams(log10_ferdf_mslope=s)
            return (jnp.log(cat.weights(field, p, GP, obsc=o).norm)
                    + jnp.log(jnp.sum(cat.xlf(field, p, GP, observed=True,
                                              obsc=o)[1])))

        ad = float(jax.grad(f)(slope))
        fd = float((f(slope + 1e-5) - f(slope - 1e-5)) / 2e-5)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    def test_every_duty_cycle_is_the_helper(self):
        """Structural: ``log10_ferdf`` is read by :func:`duty` and nowhere
        else in the module (:func:`log10_duty` is its logarithm), so a new path
        cannot apply the constant and forget the slope or the cap."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(A))
        readers = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Attribute) and n.attr == "log10_ferdf":
                        readers.add(fn.name)
        assert readers == {"duty"}, readers


@pytest.fixture(scope="module")
def sectors_three():
    """The three selections the duty cycle's outputs are taken with."""
    return dict(
        lum=A.AgnSector(GAL, calibration="off"),
        split=A.AgnSector(GAL, calibration="off", obscuration="split"),
        cat=A.AgnSector(GAL, calibration="off", obscuration="split",
                        selection="catalogue", catalogue_flux_limit=True))


@pytest.fixture(scope="module")
def fields_by_z(field):
    out = {Z: field}
    for z in (0.4, 0.8):
        out[z] = make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                            make_pk("emu_pk"), z=z)
    return out


class TestTheEddingtonRatioBreakDependsOnStellarMass:
    r"""``log10_lstar_mslope`` (1.3.0.dev0): the Eddington-ratio break
    :math:`\log\lambda_*(M_*) = \log\lambda_*(z) + s(\log M_* - 10.5)`, a
    central's at its halo's mean stellar mass, a satellite's at its own, as a
    shift of the luminosity kernel; the black-hole masses do not read it.

    Every luminosity reads ``al_bh`` and the slope only through
    :math:`\alpha_{\rm BH} + s` and :math:`\mu_{\rm BH} - 11\alpha_{\rm BH} -
    10.5\,s`, so ``al_bh`` + :math:`d`, slope :math:`-d` and ``mu_bh`` +
    :math:`d/2` leave every one of them unchanged, and only the quantities
    built on :math:`M_{\rm BH}` tell the two points apart.
    """

    D = 0.3
    SLOPE = -0.5

    @staticmethod
    def _luminosities(sec, field, p):
        """Every luminosity output: the duty cycle's list, and the
        distributions themselves."""
        out = TestTheDutyCycleDependsOnStellarMass._outputs(sec, field, p)
        lum, lm, h = sec["lum"], jnp.log10(field.m), field.cosmo.h
        more = dict(
            p_loglx_given_m=lum.p_loglx_given_m(lm, p, GP, h=h, z=field.z),
            p_active=lum.p_active_loglx_given_m(lm, p, GP, h=h, z=field.z),
            satellite_kernel=lum._satellite_kernel(field, p, GP)[2],
            xlf_soft=lum.xlf(field, p, GP, band="soft")[1],
            validity_cost_lx=lum.validity_cost(field, p, GP, quantity="l_x"))
        out.update({k: np.asarray(v) for k, v in more.items()})
        return out

    @staticmethod
    def _black_holes(sec, field, p):
        """The quantities built on the black-hole mass itself."""
        lum, lm, h = sec["lum"], jnp.log10(field.m), field.cosmo.h
        out = dict(
            mean_log10_mbh=lum.mean_log10_mbh(lm, p, GP, h=h, z=field.z),
            mean_mbh=lum.mean_mbh(lm, p, GP, h=h, z=field.z),
            mean_mbh_satellites=lum.mean_mbh_satellites(field, p, GP),
            bhmf=lum.black_hole_mass_function(
                field, jnp.linspace(5.0, 10.5, 23), p, GP),
            omega_bh=lum.omega_bh(field, p, GP),
            validity_cost_bh=lum.validity_cost(field, p, GP, quantity="omega_bh"))
        return {k: np.asarray(v) for k, v in out.items()}

    @staticmethod
    def _without_the_break(monkeypatch):
        """The sector before the slope: the kernel placed at the black-hole
        mass alone, with the black-hole mass's width, at the sector's own
        bolometric correction (log L_bol(L_X) and its Jacobian at
        ``kbol="duras20"``, ``k_bol`` at ``"constant"``)."""
        def shift(self, t_grid, k, p, mean_log10_mbh, log10_mstar):
            if self.kbol == "duras20":
                y = self._log10_lbol[None, :] - A._LOG10_LBOL_COEF
                jac = self._dlog10_lbol_dlog10_lx[None, :]
            else:
                y = self.loglx[None, :] - jnp.log10(A.LBOL_COEF / p.k_bol)
                jac = None
            t = y - jnp.atleast_1d(mean_log10_mbh)[:, None]
            if k.ndim == 1:
                out = jnp.interp(t.ravel(), t_grid, k, left=0.0,
                                 right=0.0).reshape(t.shape)
            else:
                out = jax.vmap(lambda tt, kk: jnp.interp(tt, t_grid, kk, left=0.0,
                                                         right=0.0))(t, k)
            return out if jac is None else out * jac

        monkeypatch.setattr(A.AgnSector, "_shift", shift)
        monkeypatch.setattr(A.AgnSector, "_sigma_lx", A.AgnSector._sigma_lm)

    def test_it_is_declared_in_both_copies_with_a_reason(self):
        import dataclasses
        par = A.AgnParams._PARAMS["log10_lstar_mslope"]
        assert "log10_lstar_mslope" in {f.name for f in dataclasses.fields(A.AgnParams)}
        assert A.AgnParams().log10_lstar_mslope == -2.219 == par.default
        assert par.bounds == (-3.0, 3.0)
        assert isinstance(par.prior, A.Flat)
        assert par.unit == "" and par.kind == "physical"
        assert "Igo et al. (2026)" in par.why and "10^10.5" in par.why
        assert "al_bh" in par.why
        assert "ZERO REPRODUCES THE MASS-INDEPENDENT BREAK EXACTLY" in par.why
        assert "YR3" in par.why and str(par.default) in par.why
        assert A.LG_MSTAR_LSTAR_PIVOT == A.LG_MSTAR_DUTY_PIVOT == 10.5

    def test_the_break_is_the_formula(self):
        p = A.AgnParams(log10_lstar=-1.1, log10_lstar_mslope=self.SLOPE,
                        gam_lam=2.0, gam_lam_hi=0.5)
        lg = jnp.linspace(8.0, 12.5, 37)
        for z in (0.0, 0.4, 1.5):
            at_z = float(A.lstar_of_z(z, p.log10_lstar, p.gam_lam,
                                      p.gam_lam_hi, p.z_lam))
            np.testing.assert_allclose(
                np.asarray(A.lstar_of_mstar(lg, p, z)),
                at_z + self.SLOPE * (np.asarray(lg) - 10.5), rtol=0, atol=1e-14)
            # At the pivot it is the redshift-dependent break, whatever the slope.
            assert float(A.lstar_of_mstar(10.5, p, z)) == at_z
        assert A.AgnSector.lstar_of_mstar is A.lstar_of_mstar
        # The luminosity width carries the slope beside al_bh; the black-hole
        # mass's width does not.
        sec = A.AgnSector(GAL, calibration="off")
        q = A.AgnParams(rho=0.2, log10_lstar_mslope=self.SLOPE)
        lx = (q.al_bh + self.SLOPE) ** 2 * 0.3 ** 2 * 0.8 + q.sig_bh ** 2
        bh = q.al_bh ** 2 * 0.3 ** 2 * 0.8 + q.sig_bh ** 2
        assert float(sec._sigma_lx(q, 0.3)) == pytest.approx(np.sqrt(lx), rel=1e-14)
        assert float(sec._sigma_lm(q, 0.3)) == pytest.approx(np.sqrt(bh), rel=1e-14)

    @pytest.mark.parametrize("kw", [{}, dict(rho=0.2, log10_ferdf_mslope=-0.4)])
    def test_slope_zero_is_bit_for_bit_the_mass_independent_break(
            self, field, sectors_three, monkeypatch, kw):
        """Every output at slope zero, eager and jitted, against the
        sector with the kernel placed at the black-hole mass alone and the
        black-hole mass's width: equal to the last bit, not to round-off."""
        p, sectors = A.AgnParams(**{"log10_lstar_mslope": 0.0, **kw}), sectors_three
        cat, lum = sectors["cat"], sectors["lum"]
        o = A.ObscurationParams()

        def jitted(q):
            return (cat.weights(field, q, GP, obsc=o).norm,
                    cat.xlf(field, q, GP, observed=True, obsc=o)[1],
                    lum.occupation(field, q, GP)[0], lum.xlf(field, q, GP)[1])

        def everything():
            out = {**self._luminosities(sectors, field, p),
                   **self._black_holes(sectors, field, p)}
            for i, v in enumerate(jax.jit(lambda q: jitted(q))(p)):
                out[f"jit{i}"] = np.asarray(v)
            return out

        new = everything()
        self._without_the_break(monkeypatch)
        old = everything()
        assert set(new) == set(old)
        for k in new:
            np.testing.assert_array_equal(new[k], old[k], err_msg=k)

    @pytest.mark.parametrize("z", [Z, 0.4, 0.8])
    def test_al_bh_and_the_slope_are_exactly_degenerate_in_every_luminosity(
            self, fields_by_z, sectors_three, z):
        """al_bh + d, slope -d and mu_bh + d/2 (the pivots 10.5 and 11 apart by
        half a dex), with the stellar-mass scatter and its correlation on: no
        luminosity output moves beyond round-off, and every black-hole
        quantity does."""
        field, sectors = fields_by_z[z], sectors_three
        p = A.AgnParams(rho=0.2, gam_lam=1.5, log10_ferdf_mslope=-0.4)
        q = p.replace(al_bh=p.al_bh + self.D,
                      log10_lstar_mslope=p.log10_lstar_mslope - self.D,
                      mu_bh=p.mu_bh + 0.5 * self.D)
        a, b = self._luminosities(sectors, field, p), self._luminosities(sectors, field, q)
        for k in a:
            assert np.abs(a[k]).max() > 0, k
            np.testing.assert_allclose(b[k], a[k], rtol=1e-10,
                                       atol=1e-13 * np.abs(a[k]).max(), err_msg=k)
        a, b = self._black_holes(sectors, field, p), self._black_holes(sectors, field, q)
        for k in a:
            assert np.abs(b[k] - a[k]).max() > 1e-3 * np.abs(a[k]).max(), k

    def test_the_break_moves_the_luminosities_and_not_the_black_holes(
            self, field, sectors_three):
        """With the stellar-mass scatter at its floor the two widths are equal and the
        slope is a pure shift: each central's distribution moves by
        s (log M*_c - 10.5), each satellite's by s (log M* - 10.5).  At the
        c030 black-hole relation, whose 0.28 dex scatter keeps the whole
        distribution on the L_X grid; at the defaults' 0.31 dex its faint tail
        leaves it, and the mean is then not a pure shift.  At
        ``kbol="constant"``: the slope shifts log L_bol, which is log L_X
        there up to a constant; at ``"duras20"`` log L_X(L_bol) is not
        linear, and a shift in one is a stretch in the other."""
        sectors = dict(sectors_three,
                       lum=A.AgnSector(GAL, calibration="off", kbol="constant"))
        lum = sectors["lum"]
        p0 = A.AgnParams(**AGN_C030)
        p1 = p0.replace(log10_lstar_mslope=self.SLOPE)
        lm, h = jnp.log10(field.m), field.cosmo.h
        # the galaxy sector's scatter at its floor, constant in halo mass
        gp = dict(GP, sigma_lnmstar=0.01, eta=0.0)
        lx = np.asarray(lum.loglx)

        def moments(prob):
            prob = np.asarray(prob)
            n = prob.sum(axis=1) * lum.dlx
            return n, (prob * lx).sum(axis=1) * lum.dlx / np.maximum(n, 1e-300)

        ms = np.asarray(lum.log10_mstar(lm, GP, h=h, z=field.z))
        frac = np.asarray(lum.central_fraction(lm, p0, gp, h=h, z=field.z))
        n0, m0 = moments(lum.p_loglx_given_m(lm, p0, gp, h=h, z=field.z))
        n1, m1 = moments(lum.p_loglx_given_m(lm, p1, gp, h=h, z=field.z))
        f = np.maximum(frac, 1e-300)
        whole = (frac > 0.5) & (np.abs(n0 / f - 1) < 1e-6) & (np.abs(n1 / f - 1) < 1e-6)
        assert whole.sum() > 20
        np.testing.assert_allclose(m1[whole] - m0[whole],
                                   self.SLOPE * (ms[whole] - 10.5), atol=1e-5)
        lg_s, _, s0 = lum._satellite_kernel(field, p0, GP)
        _, _, s1 = lum._satellite_kernel(field, p1, GP)
        duty = np.asarray(A.duty(lg_s, p0))
        n0, m0 = moments(s0)
        n1, m1 = moments(s1)
        whole = (np.abs(n0 / duty - 1) < 1e-6) & (np.abs(n1 / duty - 1) < 1e-6)
        assert whole.sum() > 20
        np.testing.assert_allclose(m1[whole] - m0[whole],
                                   self.SLOPE * (np.asarray(lg_s)[whole] - 10.5), atol=1e-5)
        # The black holes do not read the slope at all.
        a, b = self._black_holes(sectors, field, p0), self._black_holes(sectors, field, p1)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    @pytest.mark.x64
    @pytest.mark.parametrize("slope", [0.0, -0.5])
    def test_the_gradient_reaches_the_slope(self, field, sectors_three, slope):
        cat = sectors_three["cat"]
        o = A.ObscurationParams()

        def f(s):
            p = A.AgnParams(log10_lstar_mslope=s)
            return (jnp.log(cat.weights(field, p, GP, obsc=o).norm)
                    + jnp.log(jnp.sum(cat.xlf(field, p, GP, observed=True,
                                              obsc=o)[1])))

        ad = float(jax.grad(f)(slope))
        fd = float((f(slope + 1e-6) - f(slope - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_the_slope_is_read_by_the_helper_alone(self):
        """Structural: ``log10_lstar_mslope`` and its pivot are read by
        ``_lstar_tilt`` and nowhere else in the module; every luminosity kernel
        on the luminosity grid is placed by ``_shift`` (at a stellar mass, or
        at a black-hole mass in ``joint_mbh_lx``) and has the luminosity width
        by default; the black-hole mass's width serves the quantities built on
        M_BH alone.  ``_selected_nodes`` and ``active_bhmf`` sum the
        Eddington-ratio distribution on its own nodes rather than placing it."""
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(A))
        readers, calls = set(), {}
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if ((isinstance(n, ast.Attribute) and n.attr == "log10_lstar_mslope")
                        or (isinstance(n, ast.Constant)
                            and n.value == "log10_lstar_mslope")
                        or (isinstance(n, ast.Name) and n.id == "LG_MSTAR_LSTAR_PIVOT")):
                    readers.add(fn.name)
                if isinstance(n, ast.Call):
                    f = n.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                    calls.setdefault(name, set()).add(fn.name)
        assert readers == {"_lstar_tilt"}, readers
        assert calls["_lstar_tilt"] == {"_shift", "_sigma_lx", "lstar_of_mstar",
                                        "_selected_nodes",
                                        "_central_bh_conditional", "active_bhmf"}
        assert calls["_shift"] == {"p_loglx_given_m", "p_loglx_given_mstar",
                                   "joint_mbh_lx"}
        assert calls["_kernel"] == {"p_loglx_given_m", "p_loglx_given_mstar",
                                    "active_bhmf"}
        assert calls["_sigma_lx"] == {"p_loglx_given_m"}
        assert calls["_sigma_lm"] == {"mean_mbh", "black_hole_mass_function",
                                      "_central_bh_conditional", "active_bhmf"}
