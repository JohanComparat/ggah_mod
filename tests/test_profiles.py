"""Density profiles, mass definitions and concentration relations.

Validated against external references throughout -- ``scipy.special`` for the
sine/cosine integrals, COLOSSUS for profiles, mass definitions and every
concentration fit, and closed form for the normalisations.
"""
import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.halos import profiles as P
from ggah_mod.halos import concentration as CM
from ggah_mod.halos.mass_definitions import (
    MassDef, delta_vir, translate_mass, nfw_params_from_mass,
)
from ggah_mod.halos.variance import DELTA_C

_K = np.logspace(-4, 2, 40)
_RS = np.array([0.2, 0.5])
_C = np.array([5.0, 8.0])
_M = np.logspace(11, 14.5, 8)


@pytest.fixture(scope="module")
def colossus():
    col = pytest.importorskip("colossus")
    from colossus.cosmology import cosmology as cc
    cc.setCosmology("planck18")
    return cc.getCurrent()


class TestSineCosineIntegrals:
    """A JAX reimplementation of scipy.special.sici, so grad flows."""

    def test_against_scipy(self):
        from scipy.special import sici
        x = np.logspace(-3, 3, 600)
        s_ref, c_ref = sici(x)
        assert np.max(np.abs(np.asarray(P.si(x)) - s_ref)) < 1e-11
        assert np.max(np.abs(np.asarray(P.ci(x)) - c_ref)) < 1e-11

    def test_branches_agree_at_the_switch(self):
        """Both branches evaluated at the *same* x.

        Comparing si(switch - h) with si(switch + h) instead measures the slope
        of Si, which at x = 40 is sin(40)/40 ~ 0.019 and swamps the thing being
        tested -- the same trap as the top-hat window in test_amplitude.
        """
        from scipy.special import sici
        e = P._SICI_SWITCH
        s_ref, c_ref = sici(e)
        # Just below the switch: quadrature branch.  Just above: asymptotic.
        # Measured: each lands on scipy to 1.9e-11, so they agree to 3.7e-11.
        # Both must land on the true value, so they agree with each other.
        assert float(P.si(e - 1e-9)) == pytest.approx(s_ref, abs=1e-10)
        assert float(P.si(e + 1e-9)) == pytest.approx(s_ref, abs=1e-10)
        assert float(P.ci(e - 1e-9)) == pytest.approx(c_ref, abs=1e-10)
        assert float(P.ci(e + 1e-9)) == pytest.approx(c_ref, abs=1e-10)

    def test_ci_is_nan_for_nonpositive(self):
        """The predecessor's guard compared the *guarded copy*, so every
        x <= 0 silently returned Ci(1) instead of an undefined value."""
        assert np.isnan(float(P.ci(-1.0)))
        assert np.isnan(float(P.ci(0.0)))

    def test_si_is_odd(self):
        assert float(P.si(-3.0)) == pytest.approx(-float(P.si(3.0)), rel=1e-12)

    @pytest.mark.x64
    def test_differentiable(self):
        from scipy.special import sici
        for fn, ref in ((P.si, lambda x: np.sin(x) / x),
                        (P.ci, lambda x: np.cos(x) / x)):
            for x0 in (0.5, 5.0, 50.0):
                g = float(jax.grad(lambda x: fn(x))(x0))
                assert g == pytest.approx(ref(x0), rel=1e-6), (fn.__name__, x0)


class TestFourierNormalisation:
    """Every profile must satisfy u(k->0) = 1 exactly.

    Not decoration: it is what makes mass conservation checkable, and the
    matter tracer built on these depends on it.
    """

    @pytest.mark.parametrize("name,fn", [
        ("nfw", lambda k: P.nfw_uk(k, _RS, _C)),
        ("einasto", lambda k: P.einasto_uk(k, _RS, _C)),
        ("gnfw", lambda k: P.gnfw_uk(k, _RS, _C * _RS)),
        ("satellite", lambda k: P.satellite_uk(k, _C * _RS, _C)),
        ("ejected", lambda k: P.ejected_uk(k, _C * _RS)),
    ])
    def test_unity_at_large_scales(self, name, fn):
        u = np.asarray(fn(np.array([1e-6])))
        assert np.max(np.abs(u - 1.0)) < 1e-6, name

    @pytest.mark.parametrize("name,fn", [
        ("nfw", lambda k: P.nfw_uk(k, _RS, _C)),
        ("einasto", lambda k: P.einasto_uk(k, _RS, _C)),
        ("gnfw", lambda k: P.gnfw_uk(k, _RS, _C * _RS)),
    ])
    def test_decreasing(self, name, fn):
        u = np.asarray(fn(_K))
        assert np.all(np.diff(u, axis=0) < 0), name


class TestTheEjectedProfile:
    """A Gaussian shell, and the three properties it was chosen for."""

    def test_it_is_exactly_one_at_k_zero(self):
        """Not `< 1e-6`, like the quadrature-based profiles above: this one is
        closed form, so it is unity to the last bit.  That matters because the
        matter budget's residual is currently set by the NFW transform's
        departure from unity at k_min, and an ejected term with its own floor
        would raise it."""
        u = np.asarray(P.ejected_uk(np.array([0.0]), _C * _RS))
        assert np.all(u == 1.0)

    def test_it_decreases_until_it_underflows(self):
        """`<=`, not `<`, and the difference is the physics rather than a
        loosened tolerance.  A Gaussian in k reaches float64 zero well inside
        the wavenumber range a halo model uses -- exp(-k^2 r_ej^2/2) with
        r_ej of order a megaparsec is below 1e-300 by k ~ 40 h/Mpc -- so
        strict monotonicity is false for a correct profile.  The strict
        statement is made where the profile is still resolvable."""
        u = np.asarray(P.ejected_uk(_K, _C * _RS))
        assert np.all(np.diff(u, axis=0) <= 0)
        alive = u > 1e-300
        cols = [u[alive[:, j], j] for j in range(u.shape[1])]
        for col in cols:
            assert np.all(np.diff(col) < 0)

    def test_a_larger_ejection_radius_suppresses_more_power(self):
        r_delta = _C * _RS
        near = np.asarray(P.ejected_uk(_K, r_delta, eta_ej=1.0))
        far = np.asarray(P.ejected_uk(_K, r_delta, eta_ej=4.0))
        assert np.all(far <= near)
        assert np.max(near - far) > 0.1

    def test_it_is_differentiable_in_the_ejection_radius(self):
        r_delta = jnp.asarray(_C * _RS)

        def f(eta):
            return jnp.sum(P.ejected_uk(_K, r_delta, eta_ej=eta))

        ad = float(jax.grad(f)(2.0))
        fd = float((f(2.0 + 1e-6) - f(2.0 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)


class TestTheEjectedMassWithin:
    """The radius at which the ejected gas -- and so the budget -- closes."""

    @pytest.mark.parametrize("x,want", [(1.0, 0.199), (2.50, 0.900),
                                        (3.0, 0.971), (3.37, 0.990)])
    def test_the_quoted_fractions(self, x, want):
        got = float(P.ejected_mass_within(x * 2.0, 1.0, eta_ej=2.0))
        assert got == pytest.approx(want, abs=1e-3)

    def test_r_delta_holds_three_per_cent_at_the_default(self):
        assert float(P.ejected_mass_within(1.0, 1.0, eta_ej=2.0)) == \
            pytest.approx(0.0309, abs=2e-4)

    def test_it_is_the_enclosed_mass_of_the_transformed_density(self):
        """Integrate the Gaussian that ``ejected_uk`` transforms, shell by
        shell, and compare; and the whole of it is the transform's k -> 0
        value, one."""
        r_ej = 2.0 * 0.7
        r = np.linspace(0.0, 12.0 * r_ej, 40001)
        rho = np.exp(-0.5 * (r / r_ej) ** 2) / (2 * np.pi * r_ej ** 2) ** 1.5
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (4 * np.pi * r[1:] ** 2 * rho[1:]
                   + 4 * np.pi * r[:-1] ** 2 * rho[:-1]) * np.diff(r))])
        for rr in (0.7, 1.4, 3.5, 7.0):
            got = float(P.ejected_mass_within(rr, 0.7, eta_ej=2.0))
            assert got == pytest.approx(np.interp(rr, r, cum), abs=1e-7)
        uk0 = float(P.ejected_uk(jnp.asarray([1e-8]), jnp.asarray([0.7]))[0, 0])
        assert float(P.ejected_mass_within(60.0, 0.7)) == pytest.approx(uk0,
                                                                        abs=1e-12)

    def test_it_differentiates_in_eta(self):
        g = float(jax.grad(lambda e: P.ejected_mass_within(2.0, 1.0, eta_ej=e))(2.0))
        h = 1e-6
        fd = float((P.ejected_mass_within(2.0, 1.0, eta_ej=2.0 + h)
                    - P.ejected_mass_within(2.0, 1.0, eta_ej=2.0 - h)) / (2 * h))
        assert g < 0.0 and g == pytest.approx(fd, rel=1e-5)


class TestProfilesAgree:
    """Independent methods on the same profile must give the same answer."""

    def test_gnfw_quadrature_reproduces_the_analytic_nfw_transform(self):
        """gNFW(1,1,3) *is* NFW, and its u(k) is computed by radial
        quadrature while nfw_uk uses the closed form with Si/Ci.  Two
        completely different routes."""
        analytic = np.asarray(P.nfw_uk(_K, _RS, _C))
        numeric = np.asarray(P.gnfw_uk(_K, _RS, _C * _RS, 1.0, 1.0, 3.0, n_gl=400))
        assert np.max(np.abs(numeric / analytic - 1.0)) < 1e-8

    def test_satellite_defaults_recover_nfw(self):
        """b_sat_conc=1, f_cut_inv=0, gamma_inner=0 must be exactly NFW."""
        analytic = np.asarray(P.nfw_uk(_K, _RS, _C))
        sat = np.asarray(P.satellite_uk(_K, _C * _RS, _C, n_gl=400))
        assert np.max(np.abs(sat / analytic - 1.0)) < 1e-8

    @pytest.mark.parametrize("kw", [
        {"b_sat_conc": 0.7}, {"f_cut_inv": 0.5}, {"gamma_inner": 0.3}])
    def test_each_satellite_freedom_does_something(self, kw):
        base = np.asarray(P.satellite_uk(_K, _C * _RS, _C))
        got = np.asarray(P.satellite_uk(_K, _C * _RS, _C, **kw))
        assert np.max(np.abs(got / base - 1.0)) > 1e-3

    def test_the_truncation_is_zero_neutral_and_continuous_there(self):
        """The defect the inverse parameterisation exists to remove.

        ``f_cut`` was a truncation radius with 0 special-cased to mean "none",
        so zero was isolated: NFW at exactly 0, a point mass in the limit from
        above, and a nan gradient at the default.  ``f_cut_inv`` makes 0 the
        interior limit, so the approach is continuous and differentiable.
        """
        at_zero = np.asarray(P.satellite_uk(_K, _C * _RS, _C))
        just_above = np.asarray(P.satellite_uk(_K, _C * _RS, _C, f_cut_inv=1e-9))
        # Relative, because u(k) spans 1 to ~3e-4 here and the claim is about
        # the *shape* not jumping.  The defect this replaced moved u(k_max)
        # from 0.157 to 1.000 -- a factor 6 -- so any tolerance that admits
        # quadrature noise still excludes it by four orders of magnitude.
        assert np.allclose(just_above, at_zero, rtol=1e-6, atol=0.0), (
            np.max(np.abs(just_above / at_zero - 1.0)))

        g = jax.grad(lambda f: jnp.sum(
            P.satellite_uk(_K, _C * _RS, _C, f_cut_inv=f)))
        for f in (0.0, 1e-8, 0.5, 5.0):
            assert np.isfinite(float(g(f))), f"gradient is not finite at {f}"

    def test_tighter_truncation_moves_power_to_larger_k(self):
        """Confining satellites inward makes u(k) fall off later, monotonically.

        Checked at every halo mass, not just one: `satellite_uk` returns
        ``(Nk, NM)`` and a monotonicity that held for one mass and not another
        would be a bug this test exists to catch.
        """
        u = np.stack([np.asarray(P.satellite_uk(_K, _C * _RS, _C, f_cut_inv=f))[-1]
                      for f in (0.0, 1.0, 5.0, 10.0)])          # (4, NM)
        assert np.all(np.diff(u, axis=0) > 0.0), u

    def test_the_shape_freedoms_are_keyword_only(self):
        """So a positional call written against the old signature fails rather
        than reinterpreting a truncation radius as its reciprocal."""
        with pytest.raises(TypeError):
            P.satellite_uk(_K, _C * _RS, _C, 1.0, 0.5, 0.0)


class TestAgainstColossusProfiles:
    def test_nfw_real_space(self, colossus):
        from colossus.halo import profile_nfw
        p = profile_nfw.NFWProfile(M=1e14, c=6.0, z=0.0, mdef="200m")
        rho_s, r_s = p.par["rhos"], p.par["rs"]
        R = np.logspace(-1.3, 0.6, 8) * 1000.0            # kpc/h, colossus units
        for name, ours, theirs in (
            ("rho", P.nfw_rho(R, rho_s, r_s), p.density(R)),
            ("M(<r)", P.nfw_mass(R, rho_s, r_s), p.enclosedMass(R)),
            ("Sigma", P.nfw_sigma(R, rho_s, r_s), p.surfaceDensity(R)),
            ("DeltaSigma", P.nfw_delta_sigma(R, rho_s, r_s), p.deltaSigma(R)),
        ):
            assert np.max(np.abs(np.asarray(ours) / theirs - 1.0)) < 1e-6, name


class TestMassDefinitions:
    def test_parse(self):
        assert MassDef.from_string("200m") == MassDef(200.0, "matter")
        assert MassDef.from_string("500c") == MassDef(500.0, "critical")
        assert MassDef.from_string("vir").delta == "vir"
        with pytest.raises(ValueError, match="cannot parse"):
            MassDef.from_string("200x")

    def test_arbitrary_overdensity(self):
        """The predecessor had a general class and a narrow three-case
        function, and routed the profile path through the narrow one -- so
        500c was unreachable from it."""
        r = MassDef.from_string("2500c").radius(1e14, 0.0, PLANCK18)
        assert np.isfinite(float(r)) and float(r) > 0

    def test_radius_mass_round_trip(self):
        md = MassDef.from_string("500c")
        r = md.radius(1e14, 0.3, PLANCK18)
        assert float(md.mass(r, 0.3, PLANCK18)) == pytest.approx(1e14, rel=1e-12)

    def test_matter_definition_uses_total_not_cold(self):
        """A halo boundary is set against the mean density of the universe,
        which neutrinos contribute to."""
        _, rho = MassDef.from_string("200m").delta_rho(0.0, PLANCK18)
        assert float(rho) == pytest.approx(PLANCK18.rho_matter, rel=1e-12)
        assert float(rho) != pytest.approx(PLANCK18.rho_cold, rel=1e-12)

    def test_delta_vir_limits(self):
        """Bryan & Norman: ~100 at z=0 for Planck, tending to 18 pi^2."""
        assert 95.0 < float(delta_vir(0.0, PLANCK18)) < 110.0
        assert float(delta_vir(10.0, PLANCK18)) == pytest.approx(18 * np.pi ** 2, rel=0.02)

    def test_delta_vir_against_colossus(self, colossus):
        from colossus.halo import mass_so
        # ~0.15%: colossus's planck18 has Omega_m = 0.3111 against our 0.31.
        assert float(delta_vir(0.0, PLANCK18)) == pytest.approx(
            mass_so.deltaVir(0.0), rel=3e-3)

    @pytest.mark.parametrize("mdef", ["200m", "200c", "500c", "vir"])
    def test_radius_against_colossus(self, colossus, mdef):
        from colossus.halo import mass_so
        z = 0.0
        ours = float(MassDef.from_string(mdef).radius(1e14, z, PLANCK18)) * 1000.0
        theirs = mass_so.M_to_R(1e14, z, mdef) * (1.0 + z)   # -> comoving
        assert ours == pytest.approx(theirs, rel=3e-3), mdef

    def test_mass_ordering(self):
        """m2500c < m500c < m200c < m200m for any real halo."""
        got = [float(translate_mass(1e14, 6.0, "200m", d, 0.0, PLANCK18)[0])
               for d in ("2500c", "500c", "200c", "200m")]
        assert got == sorted(got)

    def test_identity_is_exact(self):
        m, r, c = translate_mass(1e14, 6.0, "200m", "200m", 0.0, PLANCK18)
        assert float(m) == 1e14 and float(c) == 6.0

    def test_round_trip(self):
        m2, _, c2 = translate_mass(1e14, 6.0, "200m", "200c", 0.0, PLANCK18)
        back, _, _ = translate_mass(m2, c2, "200c", "200m", 0.0, PLANCK18)
        assert float(back) == pytest.approx(1e14, rel=1e-8)

    @pytest.mark.x64
    def test_gradient_survives_the_bisection(self):
        """The defect this guards is silent, not loud.

        `translate_mass` brackets its root by bisection, which depends on its
        inputs only through the sign of F at each midpoint -- a boolean carries
        no derivative, so `jax.grad` through a bare bisection returns *exactly
        zero* and nothing complains.  The Newton polish from a stop_gradient-ed
        bracket is what restores it.
        """
        f = lambda om: jnp.log(translate_mass(
            1e14, 6.0, "200m", "500c", 0.0, PLANCK18.replace(Omega_m=om))[0])
        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        fd = float((f(PLANCK18.Omega_m + 1e-6) - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    def test_nfw_params_from_mass(self):
        rho_s, r_s, r_d = nfw_params_from_mass(1e14, 6.0, 0.0, PLANCK18, "200c")
        assert float(r_d) / float(r_s) == pytest.approx(6.0, rel=1e-12)
        m_back = 4 * np.pi * float(rho_s) * float(r_s) ** 3 * float(P.g_nfw(6.0))
        assert m_back == pytest.approx(1e14, rel=1e-10)


class TestConcentration:
    @pytest.mark.parametrize("name,colname,cosmo_name", [
        ("duffy08", "duffy08", "planck18"),
        ("dutton14", "dutton14", "planck18"),
        ("klypin16", "klypin16_m", "planck13"),
    ])
    def test_empirical_fits_match_colossus_exactly(self, name, colname, cosmo_name):
        pytest.importorskip("colossus")
        from colossus.cosmology import cosmology as cc
        from colossus.halo import concentration as ccon
        cc.setCosmology(cosmo_name)
        ours = np.asarray(CM.make_concentration(name)(_M, 0.0, "200c"))
        theirs = ccon.concentration(_M, "200c", 0.0, model=colname)
        cc.setCosmology("planck18")
        assert np.max(np.abs(ours / theirs - 1.0)) < 1e-10, name

    def test_klypin_is_planck13_only_in_colossus_too(self):
        """Corroborates CM_CALIBRATION: colossus refuses this fit outside the
        cosmology it was calibrated on."""
        assert CM.CM_CALIBRATION["klypin16"][0] == "Planck13/MultiDark"
        assert CM.CM_CALIBRATION["klypin16"][3] is False   # no cosmology response

    def test_diemer19_against_colossus(self, colossus):
        """The whole chain, at a *matched* cosmology.

        Matching matters and used to be skipped: colossus' own ``planck18`` is
        Om = 0.3111, h = 0.6766, n_s = 0.9665 against this package's 0.3100,
        0.6736, 0.9649, so an unmatched comparison measures the cosmologies as
        much as the code -- it reads 1.2% where the matched one reads 0.6%.
        """
        # CLASS is the ACCURATE flavour's default and is an *extra*
        # (`pip install ggah_mod[reference]`).  Skipped rather than failed
        # when it is absent, so the suite passes in the environment a plain
        # `pip install ggah_mod` creates -- which is what CI checks, and
        # which was not true until this guard existed.
        pytest.importorskip("classy")
        from colossus.cosmology import cosmology as ccosmo
        from colossus.halo import concentration as ccon
        from ggah_mod.cosmology.amplitude import sigma8
        from ggah_mod.cosmology.power import make_pk
        from ggah_mod.halos.field import make_field
        from ggah_mod import ACCURATE

        pk = make_pk("class")
        kk = np.logspace(-4, 1.5, 1024)
        s8 = float(sigma8(np.asarray(pk.pk(kk, 0.0, PLANCK18)), kk))
        ccosmo.setCosmology("ggah_matched", **{
            "flat": True, "H0": float(PLANCK18.h) * 100,
            "Om0": float(PLANCK18.Omega_m), "Ob0": float(PLANCK18.Omega_b),
            "sigma8": s8, "ns": float(PLANCK18.n_s), "persistence": ""})
        # 200c with `diemer19`, which is the only definition it covers, and
        # `tinker08` because the flavour's own mass function is fitted at 200m
        # alone -- the set has to be restated together, not one field at a time.
        f = make_field(PLANCK18, ACCURATE, pk, z=0.0, mdef="200c",
                       hmf_model="tinker08", cm_model="diemer19")
        m, c = np.asarray(f.m), np.asarray(f.conc)
        for mm in np.logspace(11, 15, 9):
            ours = float(np.interp(np.log(mm), np.log(m), c))
            theirs = ccon.concentration(mm, "200c", 0.0, model="diemer19")
            # The residual is sigma(M): colossus builds it from its own EH98
            # spectrum and the *total* matter density, this package from CLASS
            # and the *cold* one.  Both deliberate; neither is the relation.
            assert abs(ours / theirs - 1.0) < 0.01, f"M = {mm:.2e}"

    def test_the_fitted_constants_are_colossus_own(self, colossus):
        """Six numbers, read out of colossus rather than retyped from the paper.

        This is the cheap half of "is the relation right": if the constants
        match and the equation is solved to round-off (the next test), the only
        thing left that can differ is the input, which is sigma(M) -- and that
        difference is a convention, not an error.  Retyping them from the PDF
        is exactly how a transcription slip gets in, so they are compared
        against the reference implementation's own module constants.
        """
        import colossus.halo.concentration as CC
        import inspect
        src = inspect.getsource(CC.modelDiemer19)
        ours = CM.DIEMER19["median"]
        for name, val in (("kappa", ours["kappa"]), ("a_0", ours["a0"]),
                          ("a_1", ours["a1"]), ("b_0", ours["b0"]),
                          ("b_1", ours["b1"]), ("c_alpha", ours["c_alpha"])):
            assert f"{name}" in src and f"{val}" in src, (
                f"{name} = {val} not found in colossus' modelDiemer19")

    def test_diemer19_inversion_is_exact_and_differentiable(self):
        """G(c) has no closed-form inverse.  Bracket, then one Newton step.

        The residual proves the value; the gradient is the reason the Newton
        step is there at all -- a bisection alone returns the same number with a
        derivative of exactly zero.
        """
        import jax
        from ggah_mod.halos.concentration import DIEMER19, _nfw_mu
        q = DIEMER19["median"]
        for sig, n, al in ((0.5, -2.3, 0.52), (2.0, -1.9, 0.75), (3.0, -1.5, 0.9)):
            c = float(CM.c_diemer19(sig, n, al))
            raw = c / (1.0 - q["c_alpha"] * (1.0 - al))
            nu = DELTA_C / sig
            A = q["a0"] * (1 + q["a1"] * (n + 3))
            B = q["b0"] * (1 + q["b1"] * (n + 3))
            rhs = np.log10(A / nu * (1 + nu ** 2 / B))
            lhs = np.log10(raw) - (5.0 + n) / 6.0 * np.log10(float(_nfw_mu(raw)))
            assert abs(lhs - rhs) < 1e-12

        g = float(jax.grad(lambda s: CM.c_diemer19(s, -2.0, 0.52))(1.0))
        assert abs(g) > 1e-3, "the inversion lost its gradient"
        fd = (float(CM.c_diemer19(1.0 + 1e-6, -2.0, 0.52))
              - float(CM.c_diemer19(1.0 - 1e-6, -2.0, 0.52))) / 2e-6
        assert g == pytest.approx(fd, rel=1e-6)

    def test_n_eff_comes_from_sigma_not_from_a_fitting_function(self):
        """The reason the analytic transfer function could be deleted.

        DK15's slope is of P(k) and oscillates with the baryon feature, so it
        had to be evaluated on a smooth analytic fit rather than on the
        spectrum in use.  DJ19's is of sigma(M), which is an integral over P(k)
        and smooth whatever the spectrum does -- so it is taken from the real
        backend, and is monotonic in mass without any help.
        """
        from ggah_mod.cosmology.power import make_pk
        from ggah_mod.halos.variance import dln_sigma_dln_mass
        k = np.logspace(-4, 2, 1024)
        pytest.importorskip("classy")  # an extra; see test_coherence.py
        pk_cb = np.asarray(make_pk("class").pk_cb(k, 0.0, PLANCK18))
        mm = np.logspace(10, 15.5, 30)
        n = np.asarray(CM.n_eff_from_sigma(
            mm, PLANCK18,
            lambda x: dln_sigma_dln_mass(x, k, pk_cb, PLANCK18.rho_cold)))
        assert np.all(np.diff(n) > 0), "n_eff should rise with mass"
        assert -3.0 < n.min() and n.max() < 0.0

    def test_bhattacharya_takes_growth_explicitly(self):
        """Passed in, not computed internally, so the relation cannot acquire
        its own growth model -- the predecessor hard-wired Carroll+1992 here
        and a w0waCDM fit silently got LambdaCDM concentrations."""
        import inspect
        assert "growth" in inspect.signature(CM.c_bhattacharya13).parameters

    @pytest.mark.parametrize("name", ["duffy08", "dutton14", "klypin16"])
    def test_empirical_fits_take_no_cosmology(self, name):
        """They cannot respond to one -- the cosmology is baked into the
        coefficients -- so they do not accept one and silently ignore it."""
        import inspect
        params = set(inspect.signature(CM.make_concentration(name)).parameters)
        assert "cosmo" not in params and "cosmology" not in params
        assert CM.CM_CALIBRATION[name][3] is False

    def test_every_model_declares_its_calibration(self):
        assert set(CM.CM_CALIBRATION) == set(CM.CONCENTRATION)

    def test_unknown_model_refused(self):
        with pytest.raises(ValueError, match="unknown concentration"):
            CM.make_concentration("nfw15")

    def test_uncalibrated_mass_definition_refused(self):
        with pytest.raises(ValueError, match="calibrated"):
            CM.c_dutton14(_M, 0.0, "200m")

    def test_seppi21_takes_a_redshift_not_a_growth(self):
        """A third signature inside the peak-height family, and the one whose
        second argument is the *least* safe to get wrong.

        Handing `bhattacharya13` the effective slope where the growth belongs
        gives NaN, because the growth is raised to a fractional power.  Handing
        it a redshift does not: z and D are both positive and both order unity
        over the range these fits cover, so the answer comes back smooth and
        wrong.  Hence `field._concentration` names all three rather than
        letting one fall through to the other's branch.
        """
        import inspect
        params = inspect.signature(CM.c_seppi21).parameters
        assert "z" in params and "growth" not in params
        assert "seppi21" in CM.PEAK_HEIGHT_MODELS

    def test_seppi21_refuses_another_mass_definition(self):
        """The paper published no per-definition coefficients, so unlike
        `duffy08` there is nothing here to select."""
        with pytest.raises(ValueError, match="M_vir alone"):
            CM.c_seppi21(1.0, 0.0, "200c")

    def test_seppi21_has_the_high_mass_upturn(self):
        """The physical content of the second bracket, and what distinguishes
        it from the monotone power laws.

        `klypin16` has the upturn too, but as an interpolation in (M, z) that
        cannot respond to a cosmology; this one reaches it through sigma and so
        does.  So: c(nu) must have an interior minimum, not fall monotonically.
        """
        nu = np.linspace(0.9, 6.0, 400)
        c = np.asarray(CM.c_seppi21(DELTA_C / nu, 0.0))
        assert nu[int(np.argmin(c))] == pytest.approx(4.0, abs=0.2)
        assert c[-1] / c.min() > 1.05
        # and the fits that miss it, do miss it
        m = np.logspace(12.5, 15.5, 400)
        assert np.all(np.diff(np.asarray(CM.c_duffy08(m, 0.0, "vir"))) < 0.0)

    def test_seppi21_and_bhattacharya_agree_in_vir(self):
        """Two independent fits to the same boundary, 5-14 per cent apart.

        Recorded rather than tuned.  It is also the size of the disagreement
        that makes `seppi21_scale` worth having: the distribution's shape is
        imported and its scale is not, so which mean relation is underneath
        stays the caller's choice.
        """
        sig = np.array([1.3, 1.0, 0.8, 0.6])
        for z, growth in ((0.0, 1.0), (0.5, 0.774), (1.0, 0.610)):
            r = np.abs(np.asarray(CM.c_seppi21(sig, z)
                                  / CM.c_bhattacharya13(sig, growth, "vir")) - 1.0)
            assert r.max() < 0.15, (z, r)


#: The sigma grid Table A.1 was fitted over: the six mass slices span
#: 2e13 to 2e14 Msun/h at each snapshot, which is sigma ~ 1.3 down to ~0.8 at
#: z = 0 and ~0.66 down to ~0.40 at z = 1.43.
_S21_SIGMA = np.array([0.40, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6])
_S21_Z = tuple(float(z) for z in CM.SEPPI21_A1[:, 0])


class TestSeppi21ConcentrationDistribution:
    """Seppi et al. (2021) Eq. 10: P(c) around any mean relation.

    There is no COLOSSUS implementation to read the constants out of, the way
    `test_the_fitted_constants_are_colossus_own` does for `diemer19`.  What
    replaces it is three independent things: the closed forms (checked against
    `scipy.integrate.quad` and `scipy.special.gammaincinv`), the paper's own
    internal consistency (Eq. 10's mean against Eq. 9), and the invariances the
    model is *used* through -- that a width in ln c cannot see the anchor, and
    barely sees the mass definition.
    """

    # -- transcription, via the paper's own internal consistency -----------

    def test_the_pdf_mean_reproduces_the_mean_relation(self):
        """The transcription check, and the reason the scale is anchored.

        Eqs. 9 and 10 were fitted separately -- 9 to all three simulations at
        every redshift, 10 to HMD alone in six mass slices at four snapshots --
        so the mean of one reproducing the other is a real constraint on
        alpha, beta, x0, e0, e1 and e2 together.  A typo in any of them breaks
        it.

        The tolerance is 8 per cent where the slices are (0.6 < sigma < 1.0)
        and 25 per cent over the wider grid; measured, 7.7 and 23.6.  **That
        residual is why this module anchors the scale instead of importing
        it**: outside 2e13-2e14 Msun/h the exponents e0, e1 and e2 are weakly
        constrained and degenerate, so Eq. 10's own scale is the part of the
        fit least worth carrying.  Read as a failure being tolerated it would
        be alarming; read as a measurement it is the design.

        **What this does not pin.**  A mutation sweep over the twenty-four
        transcribed constants puts ten of them -- e0, e1 and e2 away from
        z = 0 -- outside the reach of this test and of every other in this
        file at a 5 per cent perturbation, and seven of them at 20 per cent.
        They are the exponents of sigma, and sigma spans barely more than a
        decade over the fitted range, so a few per cent on an exponent moves
        p, q and a by two or three and the mean by less still.
        They are ``e0`` at all four redshifts and ``e1``, ``e2`` at z = 0.52,
        1.03 and 1.43; `test_the_table_reads_as_printed` is the only thing
        standing behind them, and says so.
        """
        core = (_S21_SIGMA >= 0.6) & (_S21_SIGMA <= 1.0)
        for z in _S21_Z:
            p, q = CM.seppi21_shape(_S21_SIGMA, z)
            mean = (CM.seppi21_table_scale(_S21_SIGMA, z)
                    * jnp.exp(CM._gg_log_moment((p + 1.0) / q, q, 1)))
            r = np.abs(np.asarray(mean / CM.c_seppi21(_S21_SIGMA, z)) - 1.0)
            assert r[core].max() < 0.08, (z, r)
            assert r.max() < 0.25, (z, r)

    def test_the_paper_prints_some_of_these_numbers_twice(self):
        """Section 4.1.2's prose against Table A.1 and Eq. 9's coefficients.

        There is no COLOSSUS to read the constants out of, so the strongest
        transcription check available is where the authors themselves quote a
        value in the discussion as well as in the appendix.  Four such places,
        and they are independent printings rather than a second reading of the
        same table:

        * "The average value of concentration ... goes from 4 at z = 1.43 to
          5.8 at z = 0" for haloes at peak height nu = 2 -- which pins a0 and
          b0 of Eq. 9, and nothing else in this file does.
        * "the slope of the power-law is smaller, going from ~4.5 at z = 1.43
          to ~1.4 in the present day" -- the ``alpha`` column at both ends.
        * "the mass trend of the exponential decay at z = 0 is negative with
          sigma (e2 = -0.959)" -- one of the six exponents, exactly.

        See `test_the_pdf_mean_reproduces_the_mean_relation` for what is
        *not* pinned this way, and why.
        """
        sig = DELTA_C / 2.0
        assert float(CM.c_seppi21(sig, 0.0)) == pytest.approx(5.8, abs=0.1)
        assert float(CM.c_seppi21(sig, 1.43)) == pytest.approx(4.0, abs=0.2)
        alpha = CM.SEPPI21_A1[:, CM.SEPPI21_A1_COLUMNS.index("alpha")]
        assert alpha[0] == pytest.approx(1.4, abs=0.05)
        assert alpha[-1] == pytest.approx(4.5, abs=0.06)
        e2 = CM.SEPPI21_A1[:, CM.SEPPI21_A1_COLUMNS.index("e2")]
        assert e2[0] == -0.959, "quoted verbatim in Sect. 4.1.2"

    def test_the_table_reads_as_printed(self):
        """Table A.1, typed a second time in a different layout.

        A weaker device than the two tests above and it is worth saying why it
        is here anyway.  It cannot catch a *misreading* of the paper -- both
        copies come from the same eyes -- so it is not validation.  What it
        catches is a slipped digit in one of the two copies and, because the
        layout is transposed, a column or row swapped in the array without the
        tuple.

        **Ten of the twenty-four entries have no stronger check than this**:
        ``e0`` at all four redshifts, and ``e1`` and ``e2`` at z = 0.52, 1.03
        and 1.43.  A mutation sweep perturbing each constant in turn and
        re-running every other assertion in this file catches sixteen of the
        twenty-six (the twenty-four here plus Eq. 9's a0 and b0) at 5 per cent
        and nineteen at 20 per cent; those ten survive both.  They are the
        exponents of sigma, sigma spans barely more than a decade over the
        fitted range, and their effects on p, q and a partly cancel in the
        mean -- which is the same degeneracy that makes the paper's own fit
        weak in them, and the reason this module anchors the scale.
        """
        printed = {
            0.00: dict(A=4.10e-2, alpha=1.397, beta=2.604, x0=7.225,
                       e0=0.089, e1=0.776, e2=-0.959),
            0.52: dict(A=4.40e-2, alpha=2.501, beta=1.283, x0=2.394,
                       e0=0.579, e1=-0.325, e2=0.0334),
            1.03: dict(A=3.37e-3, alpha=4.14, beta=0.927, x0=0.688,
                       e0=0.188, e1=0.081, e2=0.0813),
            1.43: dict(A=2.54e-3, alpha=4.45, beta=0.924, x0=0.623,
                       e0=0.198, e1=0.095, e2=0.0813),
        }
        cols = CM.SEPPI21_A1_COLUMNS
        assert [row[0] for row in CM.SEPPI21_A1] == list(printed)
        for row in CM.SEPPI21_A1:
            want = printed[row[0]]
            for name, value in want.items():
                assert row[cols.index(name)] == value, (row[0], name)
            assert set(want) | {"z"} == set(cols)

    def test_the_fitted_amplitude_is_not_the_normalisation(self):
        """Column A of Table A.1 cannot be used, and the reason is measurable.

        Against the analytic unit-integral amplitude q/(a Gamma(k)) it is low
        by a factor that is itself different at every redshift, so there is no
        constant convention to divide out.  This test exists so that a later
        reader does not "fix" the module by putting A back.
        """
        from scipy.special import gammaln as sgammaln
        ratios = []
        for row in CM.SEPPI21_A1:
            z, amp = float(row[0]), float(row[1])
            p, q = CM.seppi21_shape(1.0, z)
            a = float(CM.seppi21_table_scale(1.0, z))
            ratios.append(float(q) / (a * np.exp(sgammaln(float((p + 1.0) / q)))) / amp)
        assert np.allclose(ratios, [8.351, 7.705, 7.105, 5.783], rtol=2e-3)
        assert max(ratios) / min(ratios) > 1.4, \
            "a constant factor would be reconstructable"

    # -- the closed forms, against scipy -----------------------------------

    @staticmethod
    def _pdf_np(c_ref, s, z, anchor="mean"):
        """A scalar, jitted `seppi21_pdf` for `scipy.integrate.quad`.

        Compiled once per (c_ref, sigma, z, anchor): `quad` calls its integrand
        hundreds of times, and the ``median`` anchor runs a 60-step bisection
        on every one of them.
        """
        f = jax.jit(lambda c: CM.seppi21_pdf(c, c_ref, s, z, anchor))
        return lambda c: float(f(c))

    def test_the_density_integrates_to_one(self):
        """Analytically, not by quadrature: the amplitude is q/(a Gamma(k))
        because int y^p exp(-y^q) dy = Gamma((p+1)/q)/q."""
        from scipy.integrate import quad
        for z in (0.0, 0.7, 1.43):
            for s in (0.45, 0.8, 1.3):
                v = quad(self._pdf_np(6.0, s, z), 0.0, 400.0, limit=400)[0]
                assert abs(v - 1.0) < 1e-9, (z, s, v)

    def test_the_moments_are_closed_form(self):
        from scipy.integrate import quad
        for z in (0.0, 0.7, 1.43):
            for s in (0.45, 0.8, 1.3):
                p, q = CM.seppi21_shape(s, z)
                a = CM.seppi21_scale(6.0, s, z)
                pdf = self._pdf_np(6.0, s, z)
                for n in (1, 2, 3):
                    got = quad(lambda c, n=n: c ** n * pdf(c), 0.0, 400.0,
                               limit=400)[0]
                    ref = float(a ** n * jnp.exp(
                        CM._gg_log_moment((p + 1.0) / q, q, n)))
                    assert got == pytest.approx(ref, rel=1e-8), (z, s, n)

    def test_quantile_matches_scipy(self):
        """Against `gammaincinv`/`gammainccinv` over twelve decades of u."""
        from scipy.special import gammaincinv, gammainccinv
        u = np.array([1e-12, 1e-8, 1e-4, 0.01, 0.2, 0.5, 0.8, 0.99,
                      1 - 1e-4, 1 - 1e-8, 1 - 1e-12])
        for z in _S21_Z:
            for s in (0.45, 0.8, 1.3):
                p, q = CM.seppi21_shape(s, z)
                k, qq = float((p + 1.0) / q), float(q)
                a = float(CM.seppi21_scale(6.0, s, z))
                ref = a * np.where(u < 0.5, gammaincinv(k, u),
                                   gammainccinv(k, 1.0 - u)) ** (1.0 / qq)
                got = np.asarray(CM.seppi21_quantile(u, 6.0, s, z))
                assert np.max(np.abs(got / ref - 1.0)) < 1e-12, (z, s)

    def test_the_upper_branch_is_why_the_split_exists(self):
        """The one-branch inversion is not slightly worse, it is unusable.

        Solving P(k, x) = 1 - v for the upper nodes means forming 1 - v with
        v ~ 3e-15, which rounds to exactly 1.0 -- a value P never attains --
        so the bisection runs to the top of its bracket and stays there.  The
        shipped split solves Q(k, x) = v instead.  Measured here, not asserted
        from the docstring.
        """
        from scipy.special import gammainccinv
        from jax.scipy.special import gammainc
        from ggah_mod.numerics import invert_monotone
        worst_split = worst_single = 0.0
        for z in _S21_Z:
            for s in (0.45, 0.8, 1.3):
                p, q = CM.seppi21_shape(s, z)
                k = (p + 1.0) / q
                ref = gammainccinv(
                    float(k), np.asarray(CM._DE_V_HI)) ** (1.0 / float(q))
                split = np.asarray(CM._gg_quantile_unit(k, q, CM._DE_V_HI, False))
                single = np.asarray(jnp.exp(invert_monotone(
                    lambda t: gammainc(k, jnp.exp(t)), 1.0 - CM._DE_V_HI,
                    *CM._ln_x_bracket()) / q))
                worst_split = max(worst_split, np.max(np.abs(split / ref - 1.0)))
                worst_single = max(worst_single, np.max(np.abs(single / ref - 1.0)))
        assert worst_split < 1e-13
        assert worst_single > 1e-5, "the split would then be buying nothing"

    # -- anchoring ---------------------------------------------------------

    def test_each_anchor_is_exact(self):
        """`c_ref` really is the statistic `anchor` names, to machine precision."""
        from scipy.integrate import quad
        s, z, c_ref = 0.9, 0.3, 6.0
        mean_pdf = self._pdf_np(c_ref, s, z, "mean")
        got = quad(lambda c: c * mean_pdf(c), 0.0, 400.0, limit=400)[0]
        assert got == pytest.approx(c_ref, rel=1e-9)
        below = quad(self._pdf_np(c_ref, s, z, "median"), 0.0, c_ref,
                     limit=400)[0]
        assert below == pytest.approx(0.5, abs=1e-9)
        h = 1e-4
        d = (CM.seppi21_logpdf(c_ref + h, c_ref, s, z, "mode")
             - CM.seppi21_logpdf(c_ref - h, c_ref, s, z, "mode")) / (2 * h)
        assert abs(float(d)) < 1e-6

    def test_the_width_cannot_see_the_anchor(self):
        """A width in ln c is invariant under a rescaling of c, and an anchor
        is exactly a rescaling.

        Which is why `sigma_ln_c_seppi21` takes neither `c_ref` nor `anchor`,
        and why the same scatter can be attached to any mean relation.  Checked
        the hard way rather than by reading the signature: the *distribution*
        itself is built at three anchors and three reference concentrations,
        its sd of ln c is taken from the nodes, and all nine reproduce the
        closed form sqrt(psi_1(k))/q to 2e-14.
        """
        ref = np.asarray(CM.sigma_ln_c_seppi21(_S21_SIGMA, 0.3))
        for c_ref in (3.0, 6.0, 12.0):
            for anchor in CM.ANCHORS:
                c, w = CM.seppi21_nodes(c_ref, _S21_SIGMA, 0.3, anchor)
                lnc = jnp.log(c)
                m1 = jnp.sum(w * lnc, -1)
                sd = np.asarray(jnp.sqrt(jnp.sum(w * (lnc - m1[:, None]) ** 2, -1)))
                assert np.allclose(sd, ref, rtol=1e-12), (c_ref, anchor)

    def test_it_attaches_to_any_relation(self):
        """The portability claim, as an assertion.

        Anchored on `bhattacharya13` and on `seppi21`'s own Eq. 9, the two
        distributions have means that differ by the fits' own 5-14 per cent and
        fractional widths that are identical to the last bit.
        """
        s, z = _S21_SIGMA, 0.52
        c_b = CM.c_bhattacharya13(s, 0.774, "vir")
        c_s = CM.c_seppi21(s, z)
        assert np.max(np.abs(np.asarray(c_b / c_s) - 1.0)) > 0.02
        def frac(c, w):
            m1 = jnp.sum(w * c, -1)
            return np.asarray(jnp.sqrt(jnp.sum(w * (c - m1[:, None]) ** 2, -1)) / m1)

        cb, w = CM.seppi21_nodes(c_b, s, z)
        cs, _ = CM.seppi21_nodes(c_s, s, z)
        assert np.allclose(frac(cb, w), frac(cs, w), rtol=1e-12)
        # and the two anchored means really are the two relations
        assert np.allclose(np.asarray(jnp.sum(w * cb, -1)), np.asarray(c_b), rtol=1e-12)
        assert np.allclose(np.asarray(jnp.sum(w * cs, -1)), np.asarray(c_s), rtol=1e-12)

    def test_a_definition_change_barely_moves_the_width(self):
        """d ln c_out / d ln c_in is 0.96-1.06, and does not depend on M.

        The M-independence is structural rather than lucky -- `translate_mass`
        returns c_out = c_in * x with x from a solve that never sees the mass --
        and it is asserted here rather than trusted, because it is what lets
        `seppi21_definition_jacobian` take a concentration and no mass.
        """
        from ggah_mod.halos.mass_definitions import translate_mass
        for out in ("200m", "200c"):
            for z in (0.0, 0.5, 1.0):
                j = np.asarray(CM.seppi21_definition_jacobian(
                    np.array([3.0, 5.0, 7.0, 10.0]), out, z, PLANCK18))
                assert 0.96 < j.min() and j.max() < 1.06, (out, z, j)
                # ...and the same number comes out of `translate_mass` at any
                # mass, which is what lets the helper take no mass at all.
                for c in (3.0, 5.0, 10.0):
                    g = [float(jax.grad(lambda lc, mm=mm: jnp.log(translate_mass(
                        mm, jnp.exp(lc), "vir", out, z, PLANCK18)[2]))(np.log(c)))
                        for mm in (1e12, 1e13, 1e14, 1e15, 1e16)]
                    assert max(g) - min(g) < 1e-12, (out, z, c, g)
                    assert g[0] == pytest.approx(
                        float(CM.seppi21_definition_jacobian(c, out, z, PLANCK18)),
                        rel=1e-12)

    def test_the_translated_shape_is_exact_not_first_order(self):
        """`seppi21_shape_translated` is an identity, not an approximation.

        If :math:`c' = A c^J` then :math:`(c/a)^q = (c'/a')^{q/J}`, so the
        transformed variable is the *same* generalised gamma with
        :math:`q \to q/J` and :math:`p \to (p+1)/J - 1`.  Checking that
        :math:`k` is invariant is necessary and nowhere near sufficient -- it
        holds for any map that scales p and q together.  What is checked here
        is the statement itself: every moment of :math:`c'^n` under the
        original shape equals the n-th moment under the translated one, for
        J on both sides of 1 and well away from it.
        """
        p, q = CM.seppi21_shape(0.9, 0.3)
        k = (p + 1.0) / q
        c, w = CM.seppi21_nodes(1.0, 0.9, 0.3)          # unit mean
        y = c * jnp.exp(CM._gg_log_moment(k, q, 1))     # y = c/a
        for J in (0.6, 0.977, 1.0, 1.058, 1.7):
            pt, qt = CM.seppi21_shape_translated(p, q, J)
            assert float((pt + 1.0) / qt) == pytest.approx(float(k), rel=1e-14)
            for n in (1, 2, 3):
                closed = float(CM._gg_log_moment(k, qt, n))
                assert closed == pytest.approx(
                    float(CM._gg_log_moment(k, q, J * n)), rel=1e-13), (J, n)
                # and the quadrature agrees with it, so the map really does
                # describe the distribution of y**J and not merely its algebra
                assert float(jnp.sum(w * y ** (J * n))) == pytest.approx(
                    np.exp(closed), rel=1e-10), (J, n)

    def test_the_table_columns_are_where_they_are_declared(self):
        """`SEPPI21_A1_COLUMNS` is the only thing naming the layout.

        The private jnp views slice by integer, so a column reordered in the
        array and not in the tuple would swap two fitted parameters silently.
        """
        i = {c: n for n, c in enumerate(CM.SEPPI21_A1_COLUMNS)}
        assert CM.SEPPI21_A1.shape[1] == len(CM.SEPPI21_A1_COLUMNS)
        for name, view in (("z", CM._S21_Z), ("alpha", CM._S21_ALPHA),
                           ("beta", CM._S21_BETA), ("x0", CM._S21_X0),
                           ("e0", CM._S21_E0), ("e1", CM._S21_E1),
                           ("e2", CM._S21_E2)):
            assert np.array_equal(np.asarray(view), CM.SEPPI21_A1[:, i[name]]), name

    def test_it_is_finite_where_the_shipped_mass_grids_reach(self):
        """The calibration stops at sigma ~ [0.4, 1.6]; the mass grids do not.

        `ACCURATE` spans 1e8 to 1e16 Msun/h, which at z = 0 is sigma from about
        5 down to 0.3, and the shape parameter k -- the thing the quantile
        bracket has to reach -- runs from 0.04 to 14 across that.  k = 0.04 is
        exactly where the lowest node's true quantile falls below the smallest
        representable float and the bracket saturates.  Whether the model
        *should* be used there is the calibration table's business; that it
        must not answer NaN is this test's.
        """
        sig = np.geomspace(0.08, 6.0, 40)
        k_lo, k_hi = np.inf, -np.inf
        for z in np.linspace(0.0, 1.43, 9):
            p, q = CM.seppi21_shape(sig, z)
            k = np.asarray((p + 1.0) / q)
            k_lo, k_hi = min(k_lo, k.min()), max(k_hi, k.max())
            c, w = CM.seppi21_nodes(1.0, sig, z)
            assert np.all(np.isfinite(np.asarray(c))) and np.all(np.asarray(c) > 0)
            for n in (1, 2, 3):
                got = jnp.sum(w * (c * jnp.exp(CM._gg_log_moment(
                    (p + 1.0) / q, q, 1))[:, None]) ** n, -1)
                ref = jnp.exp(CM._gg_log_moment((p + 1.0) / q, q, n))
                assert np.max(np.abs(np.asarray(got / ref) - 1.0)) < 1e-8, (z, n)
            assert np.all(np.isfinite(np.asarray(CM.sigma_ln_c_seppi21(sig, z))))
        assert k_lo < 0.05 and k_hi > 13.0, (k_lo, k_hi)   # the quoted range

    def test_a_stack_of_redshifts_is_vmap_not_a_broadcast(self):
        """`seppi21_shape` documents a scalar z and `jax.vmap` for a stack.

        Checked because "use vmap" is only advice until something proves the
        traced path works: `lin_weights` does a `searchsorted` on z, and an
        index is the one thing that must not become a tracer by accident.
        """
        zz = jnp.asarray([0.0, 0.3, 0.52, 1.2])
        p_v, q_v = jax.vmap(lambda z: CM._seppi21_shape(0.9, z))(zz)
        for i, z in enumerate(np.asarray(zz)):
            p, q = CM.seppi21_shape(0.9, float(z))
            assert float(p_v[i]) == pytest.approx(float(p), rel=1e-13)
            assert float(q_v[i]) == pytest.approx(float(q), rel=1e-13)

    # -- the quadrature ----------------------------------------------------

    def test_the_weights_sum_to_one(self):
        """A property of the truncated tanh-sinh rule, not a renormalisation."""
        _, w = CM.seppi21_nodes(6.0, 1.0, 0.0)
        assert w.shape == (35,)
        assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-13)

    def test_nodes_reproduce_the_closed_form_moments(self):
        """n = 1, 2, 3 to 1e-12; n = -1 only to 1e-5, and that is honest.

        <1/c> has an algebraic endpoint singularity in the probability
        variable -- c goes like u^(1/kq) at the bottom, so 1/c goes like
        u^(-1/kq) -- which the rule damps rather than removes.
        """
        worst = {n: 0.0 for n in (1, 2, 3, -1)}
        for z in (0.0, 0.3, 0.52, 0.9, 1.03, 1.43):
            for s in _S21_SIGMA:
                p, q = CM.seppi21_shape(s, z)
                k = (p + 1.0) / q
                c, w = CM.seppi21_nodes(1.0, s, z)          # mean 1 -> y = c
                y = c * jnp.exp(CM._gg_log_moment(k, q, 1))
                for n in worst:
                    got = float(jnp.sum(w * y ** n))
                    ref = float(jnp.exp(CM._gg_log_moment(k, q, n)))
                    worst[n] = max(worst[n], abs(got / ref - 1.0))
        assert worst[1] < 1e-12 and worst[2] < 1e-12 and worst[3] < 1e-11
        assert worst[-1] < 1e-5

    def test_the_double_exponential_rule_beats_gauss_legendre(self):
        """The record of why the rule is what it is: same node count, eleven
        orders of magnitude.  Gauss-Legendre in u converges at n^-2 here
        because the integrand is analytic at neither endpoint."""
        from numpy.polynomial.legendre import leggauss
        n = CM._DE_W.size
        x, wg = leggauss(n)
        u, w_gl = jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * wg)
        e_gl = e_de = 0.0
        for z in _S21_Z:
            for s in _S21_SIGMA:
                p, q = CM.seppi21_shape(s, z)
                y = CM._gg_quantile((p + 1.0) / q, q, u)
                ref = float(jnp.exp(CM._gg_log_moment((p + 1.0) / q, q, 1)))
                e_gl = max(e_gl, abs(float(jnp.sum(w_gl * y)) / ref - 1.0))
                c, w = CM.seppi21_nodes(1.0, s, z)
                e_de = max(e_de, abs(float(jnp.sum(w * c)) - 1.0))
        assert e_gl > 1e-5
        assert e_de < 1e-13
        assert e_gl / e_de > 1e8

    def test_marginalising_u_nfw_and_the_r_s_trap(self):
        """The documented recipe, and the mistake it is written to prevent.

        `r_delta` is fixed by the mass and the definition; `r_s = r_delta / c`
        is not.  Freezing `r_s` and varying only the `c` argument of `nfw_uk`
        is wrong by +21 per cent at k = 10 h/Mpc -- against a real
        marginalisation effect of -3.3 per cent.  Wrong sign, wrong size.
        """
        k = jnp.asarray([1.0, 3.0, 10.0])
        r_delta, c_bar, sig = jnp.asarray([1.0]), jnp.asarray([6.0]), jnp.asarray([1.0])
        c_i, w_i = CM.seppi21_nodes(c_bar, sig, 0.0)
        u_ref = P.nfw_uk(k, r_delta / c_bar, c_bar)
        u_bar = jnp.einsum("n,nkm->km", w_i, jax.vmap(
            lambda c: P.nfw_uk(k, r_delta / c, c), in_axes=1)(c_i))
        u_bad = jnp.einsum("n,nkm->km", w_i, jax.vmap(
            lambda c: P.nfw_uk(k, r_delta / c_bar, c), in_axes=1)(c_i))
        good = np.asarray(u_bar / u_ref - 1.0)[:, 0]
        bad = np.asarray(u_bad / u_ref - 1.0)[:, 0]
        assert good == pytest.approx([-0.0015, -0.0144, -0.0332], abs=2e-3)
        assert bad[2] > 0.15, bad
        assert np.sign(bad[2]) != np.sign(good[2])

    # -- the redshift axis -------------------------------------------------

    def test_p_and_q_are_monotone_in_z_where_the_table_is_not(self):
        """Why the derived pair is interpolated and the table columns are not.

        x0 runs 7.225, 2.394, 0.688, 0.623 and e1 changes sign, so a linear
        interpolation of the raw parameters between two snapshots means
        nothing.  p and q at fixed sigma are each monotone across all four.
        """
        e1 = CM.SEPPI21_A1[:, 6]
        assert e1.min() < 0.0 < e1.max(), \
            "e1 changes sign, so the raw table does not interpolate"
        for s in (0.6, 0.9, 1.2):
            p = np.array([float(CM.seppi21_shape(s, z)[0]) for z in _S21_Z])
            q = np.array([float(CM.seppi21_shape(s, z)[1]) for z in _S21_Z])
            assert np.all(np.diff(p) > 0.0), (s, p)
            assert np.all(np.diff(q) < 0.0), (s, q)

    def test_derived_statistics_are_smooth_across_z(self):
        """The interpolation is in (ln p, ln q); what has to come out sane is
        the distribution, not the parameters."""
        zz = np.arange(0.0, 1.431, 0.05)
        for s in (0.6, 0.9, 1.2):
            p, q = (np.array([float(v) for v in
                              (CM.seppi21_shape(s, z)[i] for z in zz)])
                    for i in (0, 1))
            k = (p + 1.0) / q
            cv = np.sqrt(np.exp(np.asarray(CM._gg_log_moment(k, q, 2)
                                           - 2 * CM._gg_log_moment(k, q, 1))) - 1.0)
            assert np.all((cv > 0.40) & (cv < 0.50)), (s, cv.min(), cv.max())
            med = np.asarray(CM._gg_quantile_unit(k, q, jnp.asarray(0.5), True)
                             / jnp.exp(CM._gg_log_moment(k, q, 1)))
            assert np.all((med > 0.90) & (med < 1.01)), (s, med.min(), med.max())

    def test_out_of_range_z_clamps_and_says_so(self):
        """A frozen shape is not an extrapolated fit, so it warns separately.

        And the warning is a *check*: the clamp itself lives in `lin_weights`
        and runs identically either way, so a jitted call gives the same
        numbers and simply cannot see a concrete redshift to warn about.
        """
        from ggah_mod.halos.calibration import MismatchWarning
        with pytest.warns(MismatchWarning, match="clamped"):
            eager = CM.seppi21_shape(0.9, 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jitted = jax.jit(lambda z: CM._seppi21_shape(0.9, z))(2.0)
        assert float(jitted[0]) == pytest.approx(float(eager[0]), rel=1e-13)
        assert float(jitted[1]) == pytest.approx(float(eager[1]), rel=1e-13)
        # clamped, not extrapolated: the endpoint snapshot's shape exactly
        end = CM.seppi21_shape(0.9, _S21_Z[-1])
        assert float(eager[0]) == pytest.approx(float(end[0]), rel=1e-13)

    # -- the two scatters --------------------------------------------------

    def test_the_two_scatters_differ_because_the_distribution_is_skewed(self):
        """0.195-0.268 dex against 0.177-0.204 dex, up to a factor 1.48 apart.

        They would coincide for a lognormal.  Eq. 10 is not one: its skewness
        runs from -0.09 to 1.05 over the calibrated range.  Anyone who takes
        one number and builds a lognormal from it gets the other one's variance,
        which is the substitution this pair of names exists to make visible.
        """
        ex, ln = [], []
        for z in _S21_Z:
            ex.append(np.asarray(CM.sigma_ln_c_seppi21(_S21_SIGMA, z)) / np.log(10))
            ln.append(np.asarray(CM.sigma_ln_c_seppi21(_S21_SIGMA, z, "lognormal"))
                      / np.log(10))
        ex, ln = np.array(ex), np.array(ln)
        assert 0.19 < ex.min() and ex.max() < 0.27
        assert 0.17 < ln.min() and ln.max() < 0.21
        assert (ex / ln).max() == pytest.approx(1.478, rel=1e-2)
        skew = []
        for z in _S21_Z:
            p, q = CM.seppi21_shape(_S21_SIGMA, z)
            k = (p + 1.0) / q
            m = [jnp.exp(CM._gg_log_moment(k, q, n)) for n in (1, 2, 3)]
            sd = jnp.sqrt(m[1] - m[0] ** 2)
            skew.append(np.asarray((m[2] - 3 * m[0] * m[1] + 2 * m[0] ** 3) / sd ** 3))
        skew = np.array(skew)
        assert skew.max() > 1.0 and skew.min() < 0.1

    def test_an_unknown_anchor_or_kind_is_refused(self):
        with pytest.raises(ValueError, match="anchor must be"):
            CM.seppi21_scale(6.0, 1.0, 0.0, "mode_of_log")
        with pytest.raises(ValueError, match="exact.*lognormal"):
            CM.sigma_ln_c_seppi21(1.0, 0.0, "dex")

    # -- gradients ---------------------------------------------------------

    @pytest.mark.x64
    def test_the_mean_relation_is_differentiable(self):
        for f, x0 in ((lambda s: CM.c_seppi21(s, 0.3), 0.9),
                      (lambda z: CM.c_seppi21(0.9, z), 0.3)):
            ad = float(jax.grad(lambda x: jnp.sum(f(x)))(x0))
            fd = float((f(x0 + 1e-6) - f(x0 - 1e-6)) / 2e-6)
            assert ad != 0.0
            assert ad == pytest.approx(fd, rel=1e-5)

    @pytest.mark.x64
    def test_the_nodes_keep_their_gradient_through_the_inversion(self):
        """`invert_monotone`'s Newton polish is what makes this non-zero.

        A bare bisection depends on its inputs only through the sign of the
        residual at each midpoint, and a boolean carries no derivative -- so
        `jax.grad` through one returns exactly zero, silently.
        """
        f = lambda s: jnp.sum(CM.seppi21_nodes(6.0, s, 0.3)[0])
        ad = float(jax.grad(f)(0.9))
        fd = float((f(0.9 + 1e-6) - f(0.9 - 1e-6)) / 2e-6)
        assert ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)
        g = lambda c: jnp.sum(CM.seppi21_nodes(c, 0.9, 0.3)[0])
        ad = float(jax.grad(g)(6.0))
        assert ad == pytest.approx(float((g(6.0 + 1e-6) - g(6.0 - 1e-6)) / 2e-6),
                                   rel=1e-6)

    @pytest.mark.x64
    def test_the_scatter_is_differentiable_through_polygamma(self):
        f = lambda s: jnp.sum(CM.sigma_ln_c_seppi21(s, 0.3))
        ad = float(jax.grad(f)(0.9))
        fd = float((f(0.9 + 1e-6) - f(0.9 - 1e-6)) / 2e-6)
        assert ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    @pytest.mark.x64
    def test_the_redshift_derivative_is_not_halved_at_a_table_node(self):
        """`lin_weights`, not `jnp.interp` -- and z = 0 is the fiducial.

        Table A.1 has a node at z = 0, so essentially every call sits exactly
        on the clamp boundary.  A clamp written with `clip` is
        `minimum(maximum(...))`, and JAX splits the gradient of a `minimum` tie
        50/50 between its arguments: the derivative comes back at exactly half
        its true value, with nothing else wrong. `lin_weights` clamps the query
        with `where`, which selects a branch cleanly.
        """
        f = lambda z: jnp.sum(CM.seppi21_shape(0.9, z)[0])
        h = 1e-7
        # z = 0 is both a table node and the clamp boundary, so the left side
        # is flat and a 50/50 tie would return exactly half the true slope.
        # Stepping left of it is deliberately outside the table, hence the
        # warning filter -- that is the check under test elsewhere, not here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad0, right0 = float(jax.grad(f)(0.0)), float((f(h) - f(0.0)) / h)
            left0 = float((f(0.0) - f(-h)) / h)
        assert ad0 != 0.0
        assert ad0 == pytest.approx(right0, rel=1e-4)
        assert left0 == pytest.approx(0.0, abs=1e-6 * abs(right0)), "not clamped"
        assert ad0 != pytest.approx(0.5 * right0, rel=1e-3), "the clip tie is back"
        # z = 0.52 is an interior node: the two sides differ, and the answer
        # must be one of them rather than their average.
        ad, right = float(jax.grad(f)(0.52)), float((f(0.52 + h) - f(0.52)) / h)
        left = float((f(0.52) - f(0.52 - h)) / h)
        assert abs(left - right) > 1e-3 * abs(right), "the node is not a kink"
        assert (ad == pytest.approx(right, rel=1e-4)
                or ad == pytest.approx(left, rel=1e-4)), (ad, left, right)
        assert ad != pytest.approx(0.5 * (left + right), rel=1e-4)
