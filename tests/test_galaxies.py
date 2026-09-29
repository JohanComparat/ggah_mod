r"""Verification: the galaxy tracer, and the placeholder it refuses to ship.

Two things are worth reading here.

**It adds no occupation physics.**  ``GalaxySector`` exists to turn a registry
entry into :class:`~ggah_mod.sectors.protocol.TracerWeights`, and
:class:`TestItAddsNoOccupationPhysics` asserts the occupations it returns are
*identical* -- not close -- to calling the registry function by hand.  That is
the same claim ``tests/test_field.py`` makes about ``make_field``, and it is what
makes the wrapper safe to insert under twelve published models at once.

**The satellite stellar mass is a recorded failure, not a tuned one.**  The
obvious placeholder -- give every satellite its host's central stellar mass --
puts *half a cluster's mass* into satellite stars, because
:math:`M_\star(M_h)` saturates while :math:`N_{\rm sat}` keeps growing.
:class:`TestTheNaivePlaceholderIsWrong` measures that, so the number stays in the
suite rather than in a commit message, and the sector ships the proxy switched
**off** instead.  It would have passed every :math:`k\to0` check: mass
conservation closes by construction whichever component the stars sit in, which
is exactly why it needed a test of its own.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import matter as MT
from ggah_mod.sectors.clf import CLF
from ggah_mod.sectors.galaxies import (
    GALAXY_MODELS, GalaxySector, galaxy_defaults,
)
from ggah_mod.sectors.occupation import OCCUPATION
from ggah_mod.sectors.sham import SHMR, SHMR_CALIBRATION, make_shmr

pytestmark = pytest.mark.slow          # every case builds a P(k)

#: The SHMR the `zumandelbaum15` occupation is actually built on, and its
#: published parameters -- so a stellar mass asked of it is the one its own
#: paper defines rather than an unrelated abundance-matching fit.
ZU15 = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)

#: The cosmic baryon fraction, which a `BaryonSplit` needs and the old
#: two-argument `matter_weights` did not: without it there is nothing to
#: subtract the retained baryons from, so the ejected component cannot exist.
F_B = PLANCK18.Omega_b / PLANCK18.Omega_m


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0)


class TestTheRegistryCoversBothFamilies:
    """One name, one model, over the occupation *and* CLF registries."""

    def test_the_two_registries_do_not_share_a_name(self):
        """A collision would make a lookup order decide which fit ran."""
        assert not set(OCCUPATION) & set(CLF)

    def test_every_model_is_reachable_and_has_defaults(self):
        assert set(GALAXY_MODELS) == set(OCCUPATION) | set(CLF)
        for name in GALAXY_MODELS:
            assert galaxy_defaults(name)

    @pytest.mark.parametrize("name", sorted(GALAXY_MODELS))
    def test_every_model_produces_finite_weights(self, name, field):
        w = GalaxySector(name, backend=DIFFERENTIABLE).weights(field, galaxy_defaults(name))
        assert np.all(np.isfinite(np.asarray(w.w_extended)))
        assert np.all(np.isfinite(np.asarray(w.w_point)))
        assert float(w.norm) > 0.0

    def test_an_unknown_model_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown galaxy model"):
            GalaxySector("not_a_model")


class TestItAddsNoOccupationPhysics:
    """The wrapper is wiring.  Identical, not close."""

    @pytest.mark.parametrize("name", sorted(OCCUPATION))
    def test_the_occupation_is_the_registry_function(self, name, field):
        sector = GalaxySector(name, backend=DIFFERENTIABLE)
        p = galaxy_defaults(name)
        n_cen, n_sat = sector.occupation(field, p)

        fn_cen, fn_sat = OCCUPATION[name]
        # In the occupation's own halo-mass units: the field's Msun/h for most,
        # physical for leauthaud12 and zacharegkas25 (THRESHOLD_MASS_UNITS).
        # The conversion is the one thing the wrapper adds, and it is units.
        log10m = sector._native_log10m(field)
        want_cen = fn_cen(log10m, **{k: v for k, v in p.items()
                                     if k in sector._cen_req + sector._cen_opt})
        want_sat = fn_sat(log10m, **{k: v for k, v in p.items()
                                     if k in sector._sat_req + sector._sat_opt})
        assert np.array_equal(np.asarray(n_cen), np.asarray(want_cen))
        assert np.array_equal(np.asarray(n_sat), np.asarray(want_sat))


class TestSignaturesRatherThanDictKeys:
    """Which parameters a model takes is read off the model, once."""

    def test_a_defaults_superset_is_not_an_error(self, field):
        """`zheng07`'s DEFAULTS carry five keys; `n_cen_zheng07` takes two."""
        p = galaxy_defaults("zheng07")
        assert {"log10m0", "log10m1", "alpha"} <= set(p)
        GalaxySector("zheng07", backend=DIFFERENTIABLE).occupation(field, p)

    @pytest.mark.parametrize("name", ["vanuitert16", "vandenbosch13"])
    def test_a_signature_default_need_not_be_in_defaults(self, name, field):
        """`f_s_star = 0.562` is Cacciato's L_s*/L_c and is in neither
        DEFAULTS table.  Demanding every named parameter refused two of the
        twelve models over a number they already carry."""
        assert "f_s_star" not in galaxy_defaults(name)
        GalaxySector(name, backend=DIFFERENTIABLE).occupation(field, galaxy_defaults(name))

    def test_a_genuinely_missing_parameter_still_raises(self, field):
        p = galaxy_defaults("zheng07")
        del p["sigma_logm"]
        with pytest.raises(ValueError, match="sigma_logm"):
            GalaxySector("zheng07", backend=DIFFERENTIABLE).occupation(field, p)


class TestTheContract:
    """What layer 4 will read off the weights."""

    @pytest.mark.parametrize("view", ["total", "cen", "sat"])
    def test_galaxies_are_discrete_in_every_view(self, view, field):
        w = GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(
            field, galaxy_defaults("zheng07"), view=view)
        assert w.discrete is True
        assert w.name == f"galaxies:{view}"

    def test_the_central_view_has_nothing_on_a_profile(self, field):
        """Same statement `AgnSector` makes: with no extended part the one-halo
        auto-spectrum vanishes identically, and `None` says so."""
        w = GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(
            field, galaxy_defaults("zheng07"), view="cen")
        assert w.w_extended is None

    def test_the_satellite_view_has_nothing_at_the_centre(self, field):
        w = GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(
            field, galaxy_defaults("zheng07"), view="sat")
        assert w.w_point is None

    def test_each_view_is_normalised_by_its_own_number_density(self, field):
        g, p = GalaxySector("zheng07", backend=DIFFERENTIABLE), galaxy_defaults("zheng07")
        n_cen, n_sat = g.occupation(field, p)
        for view, occ in (("total", n_cen + n_sat), ("cen", n_cen), ("sat", n_sat)):
            got = float(g.weights(field, p, view=view).norm)
            assert got == pytest.approx(float(field.number_density(occ)), rel=1e-12)

    def test_the_satellite_profile_tends_to_one_at_large_scales(self, field):
        u = GalaxySector("zheng07", backend=DIFFERENTIABLE).satellite_uk(
            field, galaxy_defaults("zheng07"))
        assert u.shape == (field.n_k, field.n_m)
        assert np.allclose(np.asarray(u[0]), 1.0, atol=2e-3)

    def test_an_unknown_view_is_refused(self, field):
        with pytest.raises(ValueError, match="unknown galaxy view"):
            GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(
                field, galaxy_defaults("zheng07"), view="nope")


class TestStellarMassIsNeverSubstituted:
    """Four occupations carry no stellar mass; none is invented for them."""

    def test_it_refuses_without_an_shmr(self, field):
        with pytest.raises(ValueError, match="without an `shmr="):
            GalaxySector("zheng07", backend=DIFFERENTIABLE).stellar_fraction(
                field, galaxy_defaults("zheng07"))

    def test_an_unknown_shmr_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown SHMR"):
            GalaxySector("zheng07", shmr="not_a_relation")

    def test_the_three_inverted_relations_are_registered(self):
        """They existed and were exported while the registry listed none of
        them -- and SHMR_CALIBRATION was incomplete the same way, so the
        key-parity test compared one absence against another."""
        assert {"zu15", "leauthaud12", "kravtsov18"} <= set(SHMR)
        assert set(SHMR) == set(SHMR_CALIBRATION)

    def test_satellites_are_off_and_exactly_zero(self, field):
        f_cen, f_sat = GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE) \
            .stellar_fraction(field, dict(galaxy_defaults("zheng07"), **ZU15))
        assert np.all(np.asarray(f_sat) == 0.0)
        assert np.all(np.asarray(f_cen) > 0.0)

    def test_turning_them_on_needs_the_subhalo_fraction(self, field):
        with pytest.raises(ValueError, match="f_sub"):
            GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE).stellar_fraction(
                field, dict(galaxy_defaults("zheng07"), **ZU15), satellites=True)


class TestTheNaivePlaceholderIsWrong:
    r"""The measurement that keeps the placeholder out.

    ``N_sat(M) * M_*(M_h) / M`` -- every satellite given its host's central
    stellar mass -- reaches 0.53 at :math:`10^{15}M_\odot/h`.  It diverges
    upward because :math:`M_\star(M_h)` saturates while :math:`N_{\rm sat}`
    grows linearly, so it fails hardest exactly where satellite stars matter.
    """

    def test_it_puts_half_a_cluster_into_satellite_stars(self, field):
        g = GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE)
        p = dict(galaxy_defaults("zheng07"), **ZU15)
        _, n_sat = g.occupation(field, p)
        m_star = jnp.power(10.0, make_shmr("zu15")(jnp.log10(field.m), **ZU15))
        naive = np.asarray(n_sat * m_star / field.m)
        i = int(np.argmin(np.abs(np.log10(np.asarray(field.m)) - 15.0)))
        assert naive[i] > 0.5

    def test_the_subhalo_proxy_does_not(self, field):
        g = GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE)
        p = dict(galaxy_defaults("zheng07"), f_sub=0.01, **ZU15)
        _, f_sat = g.stellar_fraction(field, p, satellites=True)
        i = int(np.argmin(np.abs(np.log10(np.asarray(field.m)) - 15.0)))
        assert float(f_sat[i]) < 0.1


class TestSatelliteStarsInTheMatterField:
    """They join `w_extended`, and the budget still closes identically."""

    def _fractions(self, field, satellites):
        g = GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE)
        p = dict(galaxy_defaults("zheng07"), f_sub=0.01, **ZU15)
        f_cen, f_sat = g.stellar_fraction(field, p, satellites=satellites)
        return g, p, f_cen, f_sat

    def test_mass_conservation_closes_with_satellites_in(self, field):
        r"""The budget closes to the accuracy of the transform, not to a constant.

        The residual here is not a failure of the invariant -- the algebra
        closes to :math:`4\times10^{-16}` -- it is
        :math:`u_{\rm NFW}(k_{\min}) - 1`, the profile transform's own
        departure from unity at the coarsest mode on the grid.  So the bound is
        that quantity, measured on the same field, rather than a number chosen
        once.

        A fixed threshold would have been wrong twice over.  It read
        :math:`10^{-8}`, which passed at 200c and failed at 200m for a reason
        that is entirely geometric: a 200m halo is larger, so :math:`r_{\rm s}`
        is larger, so :math:`k_{\min}r_{\rm s}` is larger and the transform
        departs from 1 by more.  Loosening the constant would have hidden that
        the test was measuring the grid all along.
        """
        from ggah_mod.halos.profiles import nfw_uk

        g, p, f_cen, f_sat = self._fractions(field, True)
        split = MT.BaryonSplit.from_hot(F_B, jnp.full(field.n_m, 0.1),
                                        f_star_cen=f_cen, f_star_sat=f_sat)
        w = MT.matter_weights(field, split,
                              u_sat=g.satellite_uk(field, p))
        res = np.max(np.abs(np.asarray(
            MT.MatterField.mass_conservation_residual(field, w))))

        r_s = np.asarray(field.r_delta) / np.asarray(field.conc)
        floor = float(np.max(np.abs(np.asarray(nfw_uk(
            field.k[:1], r_s, np.asarray(field.conc))) - 1.0)))
        assert res <= 1.5 * floor, (
            f"residual {res:.3e} exceeds the transform's own {floor:.3e} at "
            f"k_min: the budget is not closing for a reason the grid explains")
        assert floor < 1e-7, "the grid has become too coarse to say anything"

    def test_zero_satellite_stars_is_bit_identical_to_omitting_them(self, field):
        """The centrals-only limit is a real limit, and it is the one every
        existing test is stated in."""
        g, p, f_cen, _ = self._fractions(field, False)
        f_gas = jnp.full(field.n_m, 0.1)
        base = MT.BaryonSplit.from_hot(F_B, f_gas, f_star_cen=f_cen)
        with_zero = MT.matter_weights(field, base,
                                      u_sat=g.satellite_uk(field, p))
        without = MT.matter_weights(field, base)
        assert np.array_equal(np.asarray(with_zero.w_extended),
                              np.asarray(without.w_extended))

    def test_satellites_move_power_from_the_point_to_the_profile(self, field):
        """The signature of the fix: same budget, different k dependence."""
        g, p, f_cen, f_sat = self._fractions(field, True)
        f_gas = jnp.full(field.n_m, 0.1)
        on = MT.matter_weights(
            field, MT.BaryonSplit.from_hot(F_B, f_gas, f_star_cen=f_cen,
                                           f_star_sat=f_sat),
            u_sat=g.satellite_uk(field, p))
        off = MT.matter_weights(
            field, MT.BaryonSplit.from_hot(F_B, f_gas, f_star_cen=f_cen))
        assert np.allclose(np.asarray(on.at_large_scales()),
                           np.asarray(off.at_large_scales()), rtol=1e-12)
        assert not np.allclose(np.asarray(on.total()[-1]),
                               np.asarray(off.total()[-1]), rtol=1e-6)


class TestDifferentiability:
    """A cosmological parameter, never an amplitude."""

    def _model(self, pk, name):
        def f(omega_m):
            fl = make_field(PLANCK18.replace(Omega_m=omega_m), DIFFERENTIABLE, pk, z=0.0)
            w = GalaxySector(name, backend=DIFFERENTIABLE).weights(fl, galaxy_defaults(name))
            return jnp.log(jnp.sum(w.total()))
        return f

    @pytest.mark.x64
    @pytest.mark.parametrize("name", ["zheng07", "zumandelbaum15", "cacciato09"])
    def test_the_gradient_reaches_the_cosmology(self, name, pk):
        f = self._model(pk, name)
        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_gradient_reaches_an_occupation_parameter(self, field):
        def f(log10mmin):
            p = dict(galaxy_defaults("zheng07"), log10mmin=log10mmin)
            w = GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(field, p)
            return jnp.log(jnp.sum(w.total()))
        x, h = 11.35, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_gradient_reaches_the_satellite_profile(self, field):
        """`b_sat_conc` was `float()`-ed in the predecessor, so only two of the
        three satellite freedoms carried a derivative."""
        def f(b_sat_conc):
            p = dict(galaxy_defaults("zheng07"), b_sat_conc=b_sat_conc)
            w = GalaxySector("zheng07", backend=DIFFERENTIABLE).weights(field, p)
            return jnp.log(jnp.sum(jnp.abs(w.total())))
        x, h = 1.0, 1e-5
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)

    def test_jit_is_value_identical(self, pk):
        f = self._model(pk, "zheng07")
        assert float(jax.jit(f)(0.31)) == pytest.approx(float(f(0.31)), rel=1e-12)


class TestTheSatelliteStellarMassNeedsNoSubhaloes:
    r"""The claim this replaces was wrong, and wrong in an instructive way.

    Three places said the satellite stellar fraction needs a subhalo mass
    function "which this package does not have" -- here, in
    :mod:`~ggah_mod.sectors.agn`, and in the paper.  That describes the SHAM
    route, where a satellite's stellar mass comes from its own subhalo at
    infall.  An HOD parameterises satellites *per host* and never resolves one:
    a threshold occupation is already the conditional stellar mass function of
    its satellites, so differentiating it in its own threshold gives the
    distribution, and the first moment gives the mass.
    """

    _M = np.logspace(12.0, 15.0, 12)

    @staticmethod
    def _sector():
        from ggah_mod.sectors.galaxies import GalaxySector
        return GalaxySector("zumandelbaum15", shmr="zu15")

    def test_it_computes_without_a_subhalo_parameter(self, pk):
        """No ``f_sub`` anywhere, and a finite answer."""
        from ggah_mod.sectors.galaxies import GalaxyParams

        field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.1, m=self._M)
        _, f_sat = self._sector().stellar_fraction(
            field, GalaxyParams(), satellites=True)
        f_sat = np.asarray(f_sat)
        assert np.all(np.isfinite(f_sat)) and np.all(f_sat > 0.0)
        # A few parts in a thousand of the halo, rising with host mass and
        # saturating: satellites are a small but growing share of the stars.
        assert 1e-4 < f_sat.min() and f_sat.max() < 5e-2
        assert f_sat[-1] > f_sat[0]

    def test_it_is_far_below_the_placeholder_it_replaces(self, pk):
        """The placeholder gave every satellite the host's central mass.

        It fails upward, because ``M_star(M_h)`` saturates while ``N_sat``
        keeps growing: at :math:`10^{15}` it put half a cluster's mass into
        satellite stars.  The honest integral is smaller by more than an order
        of magnitude there, which is the size of the error that placeholder
        would have carried into the matter budget.
        """
        from ggah_mod.sectors.galaxies import GalaxyParams
        from ggah_mod.sectors.occupation import ZU15_PUBLISHED
        from ggah_mod.sectors.sham import mstar_from_mh_zu15

        # At Paper I's iHOD, where the placeholder was measured.  On the LS10
        # defaults of 0.8.5 it is 2.2x the integral at 1e15, still upward.
        field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.1, m=self._M)
        g, p = self._sector(), GalaxyParams(**ZU15_PUBLISHED)
        _, f_sat = g.stellar_fraction(field, p, satellites=True)
        n_cen, n_sat = g.occupation(field, p)
        shmr = dict(lg_m1h=p.lg_m1h, lg_m0star=p.lg_m0star, beta=p.beta,
                    delta=p.delta, gamma=p.gamma)
        placeholder = np.asarray(n_sat) * np.power(
            10.0, np.asarray(mstar_from_mh_zu15(
                jnp.log10(field.m), **shmr))) / np.asarray(field.m)
        ratio = placeholder / np.asarray(f_sat)
        assert ratio[-1] > 10.0, "the placeholder no longer fails upward"

    def test_the_gradient_reaches_the_occupation_parameters(self, pk):
        """It is an integral of a derivative, and both must stay traced."""
        from ggah_mod.sectors.galaxies import GalaxyParams

        field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.1, m=self._M)
        g = self._sector()

        def total(beta):
            _, fs = g.stellar_fraction(field, GalaxyParams(beta=beta),
                                       satellites=True)
            return jnp.sum(fs)

        d = float(jax.grad(total)(0.33))
        assert np.isfinite(d) and d != 0.0

    def test_an_occupation_without_a_threshold_still_asks_for_f_sub(self, pk):
        """The proxy survives for the occupations that cannot do better.

        ``zheng07`` is a luminosity-threshold HOD with no stellar mass in it at
        all, so there is no conditional stellar mass function to integrate and
        the one free number is still the honest answer.
        """
        from ggah_mod.sectors.galaxies import GalaxySector

        field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.1, m=self._M)
        from ggah_mod.sectors.galaxies import GalaxyParams

        g = GalaxySector("zheng07", shmr="zu15")
        # zheng07's own defaults carry no stellar mass at all, so the SHMR's
        # parameters have to come from somewhere -- which is itself the reason
        # this occupation cannot take the conditional-stellar-mass-function
        # route: it has no stellar mass of its own to integrate.
        gp = GalaxyParams()
        p = dict(galaxy_defaults("zheng07"),
                 lg_m1h=gp.lg_m1h, lg_m0star=gp.lg_m0star, beta=gp.beta,
                 delta=gp.delta, gamma=gp.gamma)
        with pytest.raises(ValueError, match="f_sub"):
            g.stellar_fraction(field, p, satellites=True)


class TestTheStellarMassFunction:
    r""":meth:`GalaxySector.stellar_mass_function` and the central CSMF it adds.

    The centrals of a threshold occupation are its own conditional
    stellar-mass function once differentiated in the threshold, exactly as the
    satellites are; for ``zumandelbaum15`` that is a log-normal in closed form,
    so the derivative is checked against it rather than against itself.
    """

    _M = np.logspace(12.0, 15.0, 12)

    def test_the_central_csmf_is_the_published_lognormal(self, pk):
        from ggah_mod.sectors.galaxies import GalaxyParams
        from ggah_mod.sectors.sham import mstar_from_mh_zu15, scatter_zu15

        field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.1, m=self._M)
        g, p = GalaxySector("zumandelbaum15"), GalaxyParams()
        lg_ms, dn = g.central_csmf(field, p)
        lgm = jnp.log10(field.m)
        mu = np.asarray(mstar_from_mh_zu15(lgm, p.lg_m1h, p.lg_m0star, p.beta,
                                           p.delta, p.gamma))
        s = np.asarray(scatter_zu15(lgm, p.sigma_lnmstar, p.eta,
                                    p.lg_m1h)) / np.log(10.0)
        x = np.asarray(lg_ms)[:, None]
        ref = p.fc * np.exp(-0.5 * ((x - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        assert np.allclose(np.asarray(dn), ref, rtol=1e-10,
                           atol=1e-12 * ref.max())

    @pytest.mark.parametrize("which", ["cen", "sat"])
    def test_it_integrates_back_to_the_number_above_a_threshold(self, field,
                                                                which):
        r""":math:`\int_t^{12}\Phi\,\dd\lg M_* = \bar n(>t) - \bar n(>12)`."""
        from ggah_mod.sectors.galaxies import GalaxyParams

        g, p = GalaxySector("zumandelbaum15"), GalaxyParams()
        lg_ms, phi_cen, phi_sat = g.stellar_mass_function(field, p)
        phi = np.asarray(phi_cen if which == "cen" else phi_sat)
        lg_ms = np.asarray(lg_ms)
        k = int(np.searchsorted(lg_ms, 10.2))
        integral = np.trapezoid(phi[k:], lg_ms[k:])

        def n_above(t):
            n_cen, n_sat = g.occupation(
                field, GalaxyParams(log10m_star_thresh=float(t)))
            n = n_cen if which == "cen" else n_sat
            return float(field.integrate(field.dndm * n))

        expect = n_above(lg_ms[k]) - n_above(lg_ms[-1])
        assert integral == pytest.approx(expect, rel=1e-3)

    def test_a_model_without_a_threshold_is_refused(self, field):
        g = GalaxySector("zheng07")
        with pytest.raises(ValueError, match="no stellar-mass threshold"):
            g.stellar_mass_function(field, galaxy_defaults("zheng07"))
