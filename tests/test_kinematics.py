"""Satellite kinematics: the NFW Jeans factor, and the stacked dispersions."""

import numpy as np
import pytest
from scipy import integrate
from scipy.special import spence

from ggah_mod.sectors import kinematics as K


def _m(y):
    return np.log1p(y) - y / (1.0 + y)


def _sigma_r2_jeans(s, c):
    """sigma_r^2 / V^2 of isotropic NFW tracers, the Jeans integral by quad."""
    x = c * s
    nu = 1.0 / (x * (1 + x) ** 2)
    val, _ = integrate.quad(lambda y: _m(y) / (y ** 3 * (1 + y) ** 2), x, np.inf, limit=200)
    return c / _m(c) * val / nu


def _sigma_r2_lokas_mamon(s, c):
    """Lokas & Mamon (2001) Eq. 14, isotropic, with Li2(-x) = spence(1 + x)."""
    x = c * s
    g = 1.0 / _m(c)
    br = (np.pi ** 2 - np.log(x) - 1 / x - 1 / (1 + x) ** 2 - 6 / (1 + x)
          + (1 + 1 / x ** 2 - 4 / x - 2 / (1 + x)) * np.log1p(x)
          + 3 * np.log1p(x) ** 2 + 6 * spence(1 + x))
    return 0.5 * c ** 2 * g * s * (1 + x) ** 2 * br


@pytest.mark.parametrize("s,c", [(0.1, 4.0), (0.5, 7.0), (1.0, 12.0)])
def test_the_jeans_integral_is_lokas_and_mamon(s, c):
    assert _sigma_r2_jeans(s, c) == pytest.approx(_sigma_r2_lokas_mamon(s, c), rel=1e-6)


@pytest.mark.parametrize("c", [2.0, 5.0, 10.0, 20.0])
def test_the_dispersion_factor_is_the_tracer_weighted_mean(c):
    num, _ = integrate.quad(lambda s: s ** 2 / (c * s * (1 + c * s) ** 2) * _sigma_r2_jeans(s, c),
                            1e-6, 1.0, limit=200)
    den, _ = integrate.quad(lambda s: s ** 2 / (c * s * (1 + c * s) ** 2), 1e-6, 1.0)
    assert float(K.dispersion_factor(c)) == pytest.approx(num / den, rel=1e-5)


def test_the_dispersion_factor_is_a_fraction_of_the_circular_velocity():
    k = np.asarray(K.dispersion_factor(np.array([3.0, 6.0, 12.0])))
    # sigma_1D / V_Delta of 0.6-0.75 for cluster-to-galaxy concentrations
    assert np.all((np.sqrt(k) > 0.55) & (np.sqrt(k) < 0.8))


@pytest.mark.x64
def test_the_stacked_dispersions_rise_with_central_mass_and_weight_as_stated():
    import jax.numpy as jnp
    from ggah_mod import ACCURATE
    from ggah_mod.cosmology import PLANCK18
    from ggah_mod.cosmology.power import make_pk
    from ggah_mod.halos.field import make_field
    from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

    field = make_field(PLANCK18, ACCURATE, make_pk("emu_pk"), z=0.1)
    gal = GalaxySector("zumandelbaum15")
    gp = galaxy_defaults("zumandelbaum15")
    edges = np.array([10.0, 10.5, 11.0, 11.5])
    sw = np.asarray(K.satellite_velocity_dispersion(field, gal, gp, edges, lg_mstar_sat_min=9.5))
    hw = np.asarray(K.satellite_velocity_dispersion(field, gal, gp, edges, lg_mstar_sat_min=9.5,
                                                    weighting="host"))
    assert np.all(np.diff(sw) > 0) and np.all(np.diff(hw) > 0)
    # satellites live in the more massive haloes of a bin: sw > hw (More+11)
    assert np.all(sw > hw)
    assert 80.0 < hw[0] < 300.0 and 200.0 < sw[-1] < 900.0
    # b_v scales both exactly
    sw2 = np.asarray(K.satellite_velocity_dispersion(field, gal, gp, edges, lg_mstar_sat_min=9.5,
                                                     p=K.KinematicsParams(b_v=1.2)))
    np.testing.assert_allclose(sw2, 1.2 * sw, rtol=1e-12)
    with pytest.raises(ValueError, match="weighting"):
        K.satellite_velocity_dispersion(field, gal, gp, edges, lg_mstar_sat_min=9.5,
                                        weighting="mass")
