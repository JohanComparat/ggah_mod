"""One neutrino convention, CLASS's, and the guards that it stays one.

Until 0.9.8 the package carried three pressureless neutrino densities: the
typed :math:`\\Sigma m_\\nu/93.14` it reported, the Komatsu fit's
:math:`\\Sigma m_\\nu/92.717` it subtracted from :math:`\\Omega_m`, and the
:math:`\\Sigma m_\\nu/93.143` CLASS added back.  The massive states now sit at
CLASS's temperature with a degeneracy of one, the massless remainder of
:math:`N_{\\rm eff}` is its own term, and the relic energy integral is
integrated rather than fitted.  Each test here states one consequence, and
each was made to fail once against a deliberately broken tree before it was
kept.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import Cosmology, PLANCK18
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.parameters import nu_energy_factor
from ggah_mod.cosmology.background import hubble_e, nu_density_shape


def _fd_exact(y):
    """The relic energy integral by adaptive quadrature: the definition,
    evaluated by a rule that shares nothing with the package's."""
    from scipy.integrate import quad
    norm = 120.0 / (7.0 * np.pi ** 4)
    g = lambda x: x * x * np.sqrt(x * x + y * y) / (np.exp(x) + 1.0)
    return norm * sum(quad(g, a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                      for a, b in ((0, 1), (1, 5), (5, 20), (20, 60), (60, 200)))


class TestTheIntegral:
    @pytest.mark.parametrize("y", [0.0, 1e-4, 3e-3, 0.03, 0.3, 3.0, 30.0,
                                   300.0, 3e3, 3e4])
    def test_the_fixed_rule_is_the_integral(self, y):
        """4.2e-11 is the rule's stated accuracy over :math:`[0, 10^6]`."""
        assert float(nu_energy_factor(y)) == pytest.approx(_fd_exact(y), rel=1e-10)

    def test_massless_is_one_to_the_bit(self):
        assert float(nu_energy_factor(0.0)) == 1.0

    def test_the_cold_limit_is_kappa_y_with_the_known_correction(self):
        """:math:`F(y) = \\kappa y\\,[1 + c_2y^{-2} - c_4y^{-4} + O(y^{-6})]`, with
        :math:`c_2 = 15\\zeta(5)/2\\zeta(3)` and :math:`c_4 = 59.0625\\,\\zeta(7)/
        \\zeta(3)`: the expansion of :math:`\\sqrt{x^2+y^2}` under the
        Fermi-Dirac moments.  A fit exact only in its leading term fails the
        first correction by tens of per cent."""
        from scipy.special import zeta
        c2 = 15.0 * zeta(5) / (2.0 * zeta(3))
        c4 = 59.0625 * zeta(7) / zeta(3)
        for y in (1e3, 1e4):
            got = float(nu_energy_factor(y)) / (C.NU_KAPPA * y) - 1.0
            # The rule's cold asymptote is kappa to 1e-14: 1.5e-7 of the
            # correction at y = 1e4, and the O(y^-6) term is 1e-11 at 1e3.
            assert got == pytest.approx(c2 / y ** 2 - c4 / y ** 4, rel=5e-7)

    def test_it_differentiates_through_zero(self):
        g0 = jax.grad(lambda y: nu_energy_factor(y))(0.0)
        assert float(g0) == 0.0
        g = jax.grad(lambda y: nu_energy_factor(y))(5.0)
        h = 1e-5
        fd = (float(nu_energy_factor(5.0 + h)) - float(nu_energy_factor(5.0 - h))) / (2 * h)
        assert float(g) == pytest.approx(fd, rel=1e-8)


class TestOneConvention:
    def test_one_pressureless_density(self):
        """`Omega_nu` *is* `Omega_nu_matter`, bit for bit, at any mass."""
        for mnu, hier in ((0.06, "degenerate"), (0.06, "normal"), (0.3, "inverted"),
                          (0.0, "massless")):
            c = Cosmology.create(sum_mnu=mnu, nu_hierarchy=hier)
            assert float(c.Omega_nu) == float(c.Omega_nu_matter)

    def test_the_denominator_is_derived_and_is_classs(self):
        """93.1434, from CODATA and CLASS's ``T_ncdm``; CLASS says 93.14."""
        c = PLANCK18
        assert C.NU_DENOM_EV == pytest.approx(93.1434, abs=5e-5)
        assert float(c.Omega_nu_matter) == pytest.approx(
            0.06 / (C.NU_DENOM_EV * c.h ** 2), rel=1e-14)

    def test_neff_is_3044_when_relativistic(self):
        """Massive states plus the remainder carry :math:`N_{\\rm eff}` exactly
        at early times -- the other limit the temperature choice must get right."""
        c = Cosmology.create(sum_mnu=0.3)
        # Far enough back that the heaviest state's F - 1 = 0.072 y^2 is below
        # 1e-15: at z = 1e8 it is still 2.5e-12, which is physics, not error.
        z = 1e10
        rho = (float(c.Omega_nu_massive_rel) * float(nu_density_shape(z, c))
               + float(c.Omega_ur))
        n_eff = rho / (C.NU_REL_COEF * float(c.Omega_gamma))
        assert n_eff == pytest.approx(C.N_EFF, rel=1e-13)
        assert C.N_MASSIVE_EFF + C.N_UR_REMAINDER == pytest.approx(C.N_EFF, rel=1e-15)

    def test_the_rest_mass_redshifts_as_matter(self):
        """Once cold, the massive states' density over :math:`\\Omega_\\nu^{\\rm
        nr}(1+z)^3` is their mean energy per unit rest mass, :math:`\\to 1`."""
        c = Cosmology.create(sum_mnu=1.5, nu_hierarchy="degenerate")
        for z in (0.0, 0.5):
            one_z = 1.0 + z
            g = (float(c.Omega_nu_massive_rel) * one_z * float(nu_density_shape(z, c))
                 / float(c.Omega_nu_matter))
            y = float(c.nu_y0) / one_z
            assert g - 1.0 == pytest.approx(_fd_exact(y) / (C.NU_KAPPA * y) - 1.0,
                                            rel=1e-6)
            assert 0.0 < g - 1.0 < 2e-6

    def test_the_relativistic_part_is_kinetic_plus_remainder(self):
        c = Cosmology.create(sum_mnu=0.06, nu_hierarchy="degenerate")
        y0 = float(c.nu_y0)
        kinetic = (_fd_exact(y0) / (C.NU_KAPPA * y0) - 1.0) * float(c.Omega_nu_matter)
        assert float(c.Omega_nu_r) == pytest.approx(kinetic + float(c.Omega_ur), rel=1e-7)

    def test_e0_is_one_to_the_bit_everywhere(self):
        for kw in (dict(sum_mnu=0.06), dict(sum_mnu=0.0, nu_hierarchy="massless"),
                   dict(sum_mnu=0.3, Omega_k=0.05, w0=-0.9, wa=0.2,
                        nu_hierarchy="degenerate"),
                   dict(sum_mnu=0.12, nu_hierarchy="inverted", Omega_k=-0.03)):
            c = Cosmology.create(**kw)
            assert float(hubble_e(jnp.array(0.0), c)) == 1.0


class TestTheSolversIntegrateTheSameNeutrinos:
    """Against CLASS and CAMB themselves.  Skipped, with this sentence, where a
    solver is absent: **a skip here is not a pass** -- nothing then checks that
    the convention is CLASS's rather than merely self-consistent."""

    CASES = [dict(sum_mnu=0.06, nu_hierarchy="degenerate"), dict(sum_mnu=0.06),
             dict(sum_mnu=0.12, nu_hierarchy="inverted"), dict(sum_mnu=0.3),
             dict(sum_mnu=0.0, nu_hierarchy="massless")]

    @pytest.mark.slow
    @pytest.mark.parametrize("kw", CASES)
    def test_the_neutrino_density_is_classs(self, kw):
        """:math:`\\rho_\\nu/\\rho_\\gamma` against CLASS's background table on
        :math:`z \\in [0, 10^4]`.  Taken over the photons so CLASS's own
        constants, 1.6e-6 from CODATA in :math:`\\Omega_\\gamma h^2`, cancel;
        what remains is 1.1e-6, CLASS's rounding of the same constants in the
        ncdm prefactor, and 0 when massless."""
        classy = pytest.importorskip(
            "classy", reason="CLASS absent: the convention is not checked "
                             "against CLASS here, and a skip is not a pass")
        from ggah_mod.cosmology.power import class_input
        c = Cosmology.create(**kw)
        cl = classy.Class()
        params = class_input(c, output="", precision={"tol_ncdm_bg": 1e-10})
        # CLASS's *own* temperature, not ours: handed ours, a wrong constant
        # here would be integrated by both sides alike and this could not fail
        # -- which it did not, until this line, when the constant was broken.
        params.pop("T_ncdm", None)
        cl.set(params)
        cl.compute()
        try:
            bg = cl.get_background()
        finally:
            cl.struct_cleanup(); cl.empty()
        zc = bg["z"][::-1]
        nu = np.array(bg["(.)rho_ur"][::-1])
        for key in bg:
            if key.startswith("(.)rho_ncdm"):
                nu = nu + bg[key][::-1]
        z = np.concatenate([[0.0], np.geomspace(1e-3, 1e4, 80)])
        theirs = np.exp(np.interp(np.log1p(z), np.log1p(zc),
                                  np.log(nu / bg["(.)rho_g"][::-1])))
        ours = np.asarray((c.Omega_nu_massive_rel * nu_density_shape(jnp.asarray(z), c)
                           + c.Omega_ur) / c.Omega_gamma)
        assert np.max(np.abs(ours / theirs - 1.0)) < 3e-6

    @pytest.mark.slow
    @pytest.mark.parametrize("kw", CASES)
    def test_camb_and_class_expand_alike(self, kw):
        """:math:`H(z)` of CAMB against CLASS on :math:`z \\in [0, 10]`, both
        handed this package's inputs.  Until 0.9.8 CAMB carried a third
        convention and, split, masses 20 per cent off; the pass is 3e-7."""
        classy = pytest.importorskip("classy", reason="CLASS absent: CAMB is "
                                     "not checked against it, and a skip is not a pass")
        camb = pytest.importorskip("camb", reason="CAMB absent: its neutrino "
                                   "spelling is not checked, and a skip is not a pass")
        from ggah_mod.cosmology.power import class_input, camb_input
        c = Cosmology.create(**kw)
        z = np.concatenate([[0.0], np.geomspace(1e-3, 10.0, 60)])
        cl = classy.Class()
        cl.set(class_input(c, output="", precision={"tol_ncdm_bg": 1e-10}))
        cl.compute()
        try:
            e_class = np.array([cl.Hubble(float(zz)) for zz in z]) / cl.Hubble(0.0)
        finally:
            cl.struct_cleanup(); cl.empty()
        res = camb.get_background(camb_input(c, precision={}))
        e_camb = res.hubble_parameter(z) / (100.0 * float(c.h))
        assert np.max(np.abs(e_camb / e_class - 1.0)) < 3e-7
        e_ours = np.asarray(hubble_e(jnp.asarray(z), c))
        assert np.max(np.abs(e_ours / e_class - 1.0)) < 3e-8

    @pytest.mark.slow
    def test_camb_is_handed_densities_not_masses(self):
        """``nu_mass_fractions`` are CAMB's shares of the *density*: handed the
        three states' densities, CAMB's own neutrino density today returns
        them.  The mass shares 0.9.7 passed made the lightest state 20 per cent
        light, and this is the assertion that would have seen it."""
        camb = pytest.importorskip("camb", reason="CAMB absent: its neutrino "
                                   "spelling is not checked, and a skip is not a pass")
        from ggah_mod.cosmology.power import camb_input
        c = Cosmology.create(sum_mnu=0.06)            # normal ordering
        pars = camb_input(c, precision={})
        res = camb.get_background(pars)
        rho = (float(c.Omega_nu_massive_rel) / 3.0
               * np.asarray(nu_energy_factor(c.nu_y)))
        np.testing.assert_allclose(np.asarray(pars.nu_mass_fractions[:3]),
                                   rho / rho.sum(), rtol=1e-12)
        assert res.get_Omega("nu") == pytest.approx(rho.sum(), rel=3e-5)


class TestPrecisionReachesTheSolver:
    def test_a_stated_precision_reaches_cl_set(self, monkeypatch):
        """Patched by name, and the patch asserted to have hit exactly once, per
        the rule that a guard satisfied by an entry that was already correct is
        not a guard."""
        classy = pytest.importorskip("classy", reason="CLASS absent: the "
                                     "precision routing is not checked")
        from ggah_mod.cosmology.power import make_pk
        seen = []

        class Stop(Exception):
            pass

        class FakeClass:
            """`classy.Class` is an immutable C type, so the name the backend
            imports is replaced rather than the method."""
            def set(self, params):
                seen.append(dict(params))
                raise Stop

        monkeypatch.setattr(classy, "Class", FakeClass)
        pk = make_pk("class", precision={"tol_ncdm_bg": 1e-10, "l_max_ncdm": 50})
        with pytest.raises(Stop):
            pk.pk(np.array([0.1]), 0.0, PLANCK18)
        assert len(seen) == 1
        assert seen[0]["tol_ncdm_bg"] == 1e-10 and seen[0]["l_max_ncdm"] == 50
        assert seen[0]["T_ncdm"] == C.T_NCDM_OVER_T_GAMMA

    def test_the_default_is_the_declared_one(self):
        from ggah_mod.cosmology.power import make_pk, CLASS_PRECISION, CAMB_PRECISION
        assert make_pk("class").precision == CLASS_PRECISION
        assert make_pk("camb").precision == CAMB_PRECISION
        assert make_pk("class", precision={}).precision == {}

    def test_the_emulator_refuses_a_precision(self):
        from ggah_mod.cosmology.power import make_pk
        with pytest.raises(ValueError, match="no solver"):
            make_pk("emu_pk", precision={"tol_ncdm_bg": 1e-10})
