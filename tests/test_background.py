"""E(z), the neutrino transition, and the distances built on them."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggah_mod.cosmology import (
    Cosmology, hubble_e, nu_density_shape, comoving_distance,
    angular_diameter_distance, luminosity_distance, comoving_volume_element,
)
from ggah_mod.cosmology import constants as C

_Z = np.logspace(-3, np.log10(5.0), 40)


class TestHubbleE:
    def test_e0_is_exactly_one(self, cosmo):
        """Not approximately: Omega_de is defined by flatness so it cancels."""
        assert float(hubble_e(jnp.array(0.0), cosmo)) == 1.0

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_e0_is_one_at_every_neutrino_mass(self, mnu):
        """To one eps, which is as exact as a sum of four terms gets.

        `Omega_de` is defined to make the sum 1, so the only thing between this
        and a bit-exact 1.0 is the order the four densities are added in --
        `Omega_de` subtracts them in one order and `hubble_e` adds them back in
        another.
        """
        c = Cosmology.create(sum_mnu=mnu)
        assert float(hubble_e(jnp.array(0.0), c)) == pytest.approx(1.0, abs=2.3e-16)

    def test_monotonic(self, cosmo):
        e = np.asarray(hubble_e(jnp.asarray(_Z), cosmo))
        assert np.all(np.diff(e) > 0)

    def test_radiation_is_present(self, cosmo):
        """Removing the photons must change E(z) at high z -- the predecessor
        had no radiation term at all and its budget row hid it by comparing
        against astropy with Tcmb0 = 0."""
        no_gamma = cosmo.replace(T_cmb=1e-6)
        z = jnp.array(1000.0)
        assert abs(float(hubble_e(z, cosmo) / hubble_e(z, no_gamma)) - 1.0) > 1e-3

    def test_cpl_reduces_to_lcdm(self, cosmo):
        a = hubble_e(jnp.asarray(_Z), cosmo.replace(w0=-1.0, wa=0.0))
        np.testing.assert_allclose(np.asarray(a), np.asarray(hubble_e(jnp.asarray(_Z), cosmo)),
                                   rtol=1e-14)

    def test_responds_to_dark_energy(self, cosmo):
        a = np.asarray(hubble_e(jnp.asarray(_Z), cosmo))
        b = np.asarray(hubble_e(jnp.asarray(_Z), cosmo.replace(w0=-0.7)))
        assert np.max(np.abs(b / a - 1.0)) > 1e-3


class TestNeutrinoTransition:
    def test_massless_shape_is_exactly_one(self, massless):
        """With no mass the species is relativistic at every z, so the shape
        factor is identically 1 and rho_nu joins the radiation."""
        s = np.asarray(nu_density_shape(jnp.asarray(_Z), massless))
        np.testing.assert_array_equal(s, np.ones_like(s))

    def test_pressureless_when_cold(self, cosmo):
        """rho_nu/(1+z)^3 must be flat at low z: at 0.02 eV per species the
        non-relativistic transition is at z ~ 38.

        Taken as a ratio to its own z=0 value, because `nu_density_shape` is
        normalised relativistically rather than today -- the physics under test
        is the *shape*, and it is unchanged.
        """
        s0 = float(nu_density_shape(jnp.array(0.0), cosmo))
        for z in (0.0, 0.5, 1.0, 2.0):
            rho = (1 + z) ** 4 * float(nu_density_shape(jnp.array(z), cosmo)) / s0
            assert abs(rho / (1 + z) ** 3 - 1.0) < 0.02, f"z={z}"

    def test_radiation_like_when_hot(self, cosmo):
        """Deep in the relativistic regime rho_nu must scale as (1+z)^4."""
        z1, z2 = 1e6, 1e7
        r1 = (1 + z1) ** 4 * float(nu_density_shape(jnp.array(z1), cosmo))
        r2 = (1 + z2) ** 4 * float(nu_density_shape(jnp.array(z2), cosmo))
        assert abs((r2 / r1) / ((1 + z2) / (1 + z1)) ** 4 - 1.0) < 1e-3

    def test_transition_is_where_physics_says(self, cosmo):
        """z_nr ~ 1890 (m/eV); at 0.02 eV per species that is z ~ 38, so the
        departure from pressureless must be small below it and large above."""
        s0 = float(nu_density_shape(jnp.array(0.0), cosmo))
        below = (1 + 5.0) ** 4 * float(nu_density_shape(jnp.array(5.0), cosmo)) / s0 / (1 + 5.0) ** 3
        above = (1 + 500.) ** 4 * float(nu_density_shape(jnp.array(500.), cosmo)) / s0 / (1 + 500.) ** 3
        assert below < 1.15 and above > 3.0

    def test_mass_matters(self, cosmo, massless):
        """A massive species carries *more* energy than a massless one.

        The inequality runs this way round now, and did not before: the factor
        is normalised to 1 relativistically, so a massless species sits at 1 at
        every z and a massive one is above it wherever the mass matters.  Under
        the old normalisation both were ratios to their own z=0 value and the
        comparison meant something else.
        """
        z = jnp.array(100.0)
        assert float(nu_density_shape(z, massless)) == pytest.approx(1.0, rel=1e-15)
        assert float(nu_density_shape(z, cosmo)) > 1.0


class TestDistances:
    def test_chi_zero_at_zero(self, cosmo):
        assert float(comoving_distance(jnp.array(0.0), cosmo)[0]) == pytest.approx(0.0, abs=1e-12)

    def test_chi_monotonic(self, cosmo):
        chi = np.asarray(comoving_distance(jnp.asarray(_Z), cosmo))
        assert np.all(np.diff(chi) > 0)

    def test_hogg_relations(self, cosmo):
        z = jnp.asarray(_Z)
        chi = np.asarray(comoving_distance(z, cosmo))
        np.testing.assert_allclose(np.asarray(angular_diameter_distance(z, cosmo)),
                                   chi / (1 + _Z), rtol=1e-13)
        np.testing.assert_allclose(np.asarray(luminosity_distance(z, cosmo)),
                                   chi * (1 + _Z), rtol=1e-13)

    def test_volume_element_matches_its_definition(self, cosmo):
        z = jnp.asarray(_Z)
        chi = np.asarray(comoving_distance(z, cosmo))
        expect = cosmo.hubble_distance * chi ** 2 / np.asarray(hubble_e(z, cosmo))
        np.testing.assert_allclose(np.asarray(comoving_volume_element(z, cosmo)),
                                   expect, rtol=1e-13)

    def test_quadrature_is_converged(self, cosmo):
        """256-point GL against a dense trapezoid on the same integrand."""
        zt = np.linspace(0.0, 3.0, 200_001)
        ref = cosmo.hubble_distance * np.trapezoid(
            1.0 / np.asarray(hubble_e(jnp.asarray(zt), cosmo)), zt)
        got = float(comoving_distance(jnp.array(3.0), cosmo)[0])
        assert abs(got / ref - 1.0) < 1e-9


@pytest.mark.x64
class TestDifferentiability:
    @pytest.mark.parametrize("name", ["Omega_m", "Omega_b", "h", "sum_mnu", "w0", "wa"])
    def test_grad_of_e_matches_finite_difference(self, cosmo, name):
        z = jnp.array(1.0)
        f = lambda v: hubble_e(z, cosmo.replace(**{name: v}))
        x = float(getattr(cosmo, name))
        ad = float(jax.grad(f)(x))
        # 1e-4 relative, not 1e-6.  dE/dsum_mnu at z = 1 is 9.4e-5, so a step of
        # 6e-8 eV moves E by 5.6e-12 and the difference is roundoff to 2e-5 --
        # the test's own tolerance.  It passed at 1e-6 under the Komatsu fit by
        # the luck of the rounding, and stopped when the integral became a sum.
        # A central difference at 1e-4 is truncated at ~1e-9 for every name.
        step = 1e-4 * max(abs(x), 1e-3)
        fd = float((f(x + step) - f(x - step)) / (2 * step))
        assert ad == pytest.approx(fd, rel=2e-5, abs=1e-12), f"{name}: {ad} vs {fd}"

    def test_grad_of_chi_matches_finite_difference(self, cosmo):
        f = lambda v: comoving_distance(jnp.array(1.0), cosmo.replace(Omega_m=v))[0]
        ad = float(jax.grad(f)(cosmo.Omega_m))
        fd = float((f(cosmo.Omega_m + 1e-7) - f(cosmo.Omega_m - 1e-7)) / 2e-7)
        assert ad == pytest.approx(fd, rel=1e-5)

    def test_massless_gradient_is_finite(self, massless):
        """sum_mnu = 0 is a corner of the Komatsu form: (k y)^1.83 at y = 0.
        A NaN here would poison every massless forecast."""
        g = float(jax.grad(lambda m: hubble_e(jnp.array(1.0),
                                              massless.replace(sum_mnu=m)))(0.0))
        assert np.isfinite(g)


class TestMasslessNeutrinosCarryRadiation:
    """The defect the cross-code benchmark found, as an assertion.

    `Omega_nu = Sum m_nu/(93.14 h^2)` is proportional to the mass, so using it
    as the amplitude of the neutrino term left a massless model with no
    neutrino energy density at all -- not a small one, none.  `E(z=100)` came
    out 0.6 per cent low against CCL and astropy, which agree with each other
    to 7e-6 there.
    """

    def test_the_relativistic_amplitude_is_the_closed_form(self):
        c = Cosmology.create(sum_mnu=0.0)
        want = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * C.N_EFF * c.Omega_gamma
        assert float(c.Omega_nu_rel) == pytest.approx(want, rel=1e-14)
        assert float(c.Omega_nu_rel) > 0.0

    def test_a_massless_species_still_carries_energy(self):
        """The whole defect, in one line: this used to be exactly zero."""
        c = Cosmology.create(sum_mnu=0.0)
        assert float(c.Omega_nu) == 0.0                     # the mass budget
        assert float(c.Omega_nu_today) > 3.7e-5             # the expansion

    def test_a_massless_model_scales_as_radiation(self):
        """Deep in the radiation era, well past equality at z ~ 3400."""
        c = Cosmology.create(sum_mnu=0.0)
        z1, z2 = 1e6, 1e7
        r = (float(hubble_e(jnp.array(z2), c)) / float(hubble_e(jnp.array(z1), c))) ** 2
        assert r == pytest.approx(((1 + z2) / (1 + z1)) ** 4, rel=1e-2)

    def test_the_neutrinos_are_a_third_of_that_radiation(self):
        """rho_nu/rho_gamma -> (7/8)(4/11)^(4/3) N_eff = 0.6912, exactly.

        The sharper statement: it is not merely non-zero, it is the right
        fraction.  A term that had simply been forgotten would fail this as
        loudly as one that was zero.
        """
        c = Cosmology.create(sum_mnu=0.0)
        assert float(c.Omega_nu_rel / c.Omega_gamma) == pytest.approx(
            7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * C.N_EFF, rel=1e-14)

    @pytest.mark.parametrize("hierarchy", ["degenerate", "normal", "inverted"])
    @pytest.mark.parametrize("mnu", [0.15, 0.30])
    def test_the_z0_density_matches_the_exact_integral(self, hierarchy, mnu):
        """`Omega_nu_today` against the relic energy integral, done independently.

        Each of the three masses integrated by ``scipy.integrate.quad`` at the
        massive states' temperature, plus the massless remainder: the
        definition, evaluated by a different rule.  The package's fixed
        50-node quadrature has to reproduce it to its stated 4.4e-11.

        This compared against astropy until 0.9.8, and could not do better
        than 6.3e-5: astropy uses the Komatsu fit, and so did this package.
        Since 0.9.8 astropy is also a different *convention* -- it puts the
        neutrinos' heating into their degeneracy at :math:`(4/11)^{1/3}T` where
        this package and CLASS put it into their temperature -- 0.46 per cent
        in the rest mass, which is a statement about astropy and not a test of
        this code.
        """
        from scipy.integrate import quad
        c = Cosmology.create(sum_mnu=mnu, nu_hierarchy=hierarchy)
        norm = 120.0 / (7.0 * np.pi ** 4)

        def f_exact(y):
            g = lambda x: x * x * np.sqrt(x * x + y * y) / (np.exp(x) + 1.0)
            return norm * sum(quad(g, a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                              for a, b in ((0, 1), (1, 5), (5, 20), (20, 60), (60, 200)))

        per_state = float(c.Omega_nu_massive_rel) / C.N_NU_MASSIVE
        want = sum(per_state * f_exact(y) for y in np.asarray(c.nu_y)) + float(c.Omega_ur)
        assert float(c.Omega_nu_today) == pytest.approx(want, rel=1e-10)

    def test_the_rest_mass_and_the_integral_differ_by_physics_alone(self):
        r"""One pressureless density, and today's density above it by exactly
        the kinetic energy plus the massless remainder.

        Until 0.9.8 there were two pressureless densities -- the typed
        :math:`\Sigma m_\nu/93.14` the package reported and the
        :math:`\Sigma m_\nu/92.717` it subtracted -- and today's density sat
        0.53 per cent above the first, a ratio of conventions.  Now `Omega_nu`
        *is* `Omega_nu_matter`, and what separates it from `Omega_nu_today` at
        the fiducial is 4.57e-4 of kinetic energy, :math:`\tfrac{15\zeta(5)}
        {2\zeta(3)}y_0^{-2}` to leading order, and 3.8e-5 of massless
        remainder.
        """
        from scipy.integrate import quad
        c = Cosmology.create(sum_mnu=0.06, nu_hierarchy="degenerate")
        assert float(c.Omega_nu) == float(c.Omega_nu_matter)
        y0 = float(c.nu_y0)
        norm = 120.0 / (7.0 * np.pi ** 4)
        g = lambda x: x * x * np.sqrt(x * x + y0 * y0) / (np.exp(x) + 1.0)
        f = norm * sum(quad(g, a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
                       for a, b in ((0, 1), (1, 5), (5, 20), (20, 60), (60, 200)))
        kinetic = f / (C.NU_KAPPA * y0) - 1.0
        assert kinetic == pytest.approx(4.573e-4, rel=1e-3)
        want = kinetic + float(c.Omega_ur) / float(c.Omega_nu_matter)
        got = float(c.Omega_nu_today) / float(c.Omega_nu_matter) - 1.0
        assert got == pytest.approx(want, rel=1e-6)
        assert got == pytest.approx(float(c.Omega_nu_r) / float(c.Omega_nu), rel=1e-9)
