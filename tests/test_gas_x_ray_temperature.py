r"""``HotGasDPM.x_ray_temperature`` and the explicit apertures of 0.9.6.

The band-luminosity-weighted temperature is a ratio of two volume integrals,
so each test compares it with something computed another way: a fine
trapezoid in radius, the pointwise temperatures it must lie between, and the
default aperture it must reproduce when handed that aperture explicitly.
"""

import numpy as np
import pytest

import jax

from ggah_mod.cosmology import PLANCK18
from ggah_mod.sectors import gas as G

M = np.logspace(12.0, 15.0, 7)
Z = 0.1
CONC = 5.0 * (M / 1e14) ** -0.1


@pytest.fixture(scope="module")
def gas():
    jax.config.update("jax_enable_x64", True)
    return G.HotGasDPM(), G.DpmParams()


def _r_delta(g, p):
    return np.asarray(g._r_delta(np.asarray(M), Z, PLANCK18, None))


def test_explicit_default_aperture_is_the_default(gas):
    g, p = gas
    r_max = p.r_max_over_rdelta * _r_delta(g, p)
    for f in (g.y_amplitude, g.x_ray_luminosity, g.x_ray_temperature):
        a = np.asarray(f(M, Z, PLANCK18, p, conc=CONC))
        b = np.asarray(f(M, Z, PLANCK18, p, conc=CONC, r_max=r_max))
        np.testing.assert_array_equal(a, b)


def _trapezoid_tx(g, p, r_lo, r_hi, n=20001):
    """The same ratio on a fine log grid, per halo, with numpy's trapezoid."""
    out = []
    for i in range(M.size):
        r = np.geomspace(max(r_lo[i], 1e-5 * r_hi[i]), r_hi[i], n)[None, :]
        m = M[i:i + 1]
        c = CONC[i:i + 1]
        eps = np.asarray(g.emissivity(r, m, Z, PLANCK18, p, conc=c))[0]
        kt = np.asarray(g.temperature(r, m, Z, PLANCK18, p, conc=c))[0]
        rr = r[0]
        out.append(np.trapezoid(eps * kt * rr ** 2, rr)
                   / np.trapezoid(eps * rr ** 2, rr))
    return np.array(out)


def test_temperature_matches_an_independent_quadrature(gas):
    # A core-excised aperture, 0.15-1 R_Delta, where both quadratures are
    # smooth; the fine trapezoid is the reference.
    g, p = gas
    rd = _r_delta(g, p)
    tx = np.asarray(g.x_ray_temperature(M, Z, PLANCK18, p, conc=CONC,
                                        r_min=0.15 * rd, r_max=rd))
    ref = _trapezoid_tx(g, p, 0.15 * rd, rd)
    np.testing.assert_allclose(tx, ref, rtol=2e-4)


def test_temperature_lies_between_the_pointwise_extremes(gas):
    g, p = gas
    rd = _r_delta(g, p)
    tx = np.asarray(g.x_ray_temperature(M, Z, PLANCK18, p, conc=CONC,
                                        r_max=rd))
    r = np.geomspace(1e-3, 1.0, 400)[None, :] * rd[:, None]
    kt = np.asarray(g.temperature(r, M, Z, PLANCK18, p, conc=CONC))
    assert np.all(tx >= kt.min(axis=1)) and np.all(tx <= kt.max(axis=1))
    # and it rises with mass, as any kT-M relation must
    assert np.all(np.diff(tx) > 0)


def test_the_amplitude_cancels_in_the_temperature_not_in_the_luminosity(gas):
    g, p = gas
    t1 = np.asarray(g.x_ray_temperature(M, Z, PLANCK18, p, conc=CONC))
    t3 = np.asarray(g.x_ray_temperature(M, Z, PLANCK18, p, conc=CONC,
                                        amplitude=3.0))
    np.testing.assert_allclose(t3, t1, rtol=1e-12)
    l1 = np.asarray(g.x_ray_luminosity(M, Z, PLANCK18, p, conc=CONC))
    l3 = np.asarray(g.x_ray_luminosity(M, Z, PLANCK18, p, conc=CONC,
                                       amplitude=3.0))
    # n_e and P_e both scale, so kT does not and L_X = n_e^2 Lambda goes as 9
    np.testing.assert_allclose(l3 / l1, 9.0, rtol=1e-10)


def test_a_smaller_aperture_holds_less(gas):
    g, p = gas
    rd = _r_delta(g, p)
    for f in (g.y_amplitude, g.x_ray_luminosity):
        small = np.asarray(f(M, Z, PLANCK18, p, conc=CONC, r_max=0.5 * rd))
        full = np.asarray(f(M, Z, PLANCK18, p, conc=CONC, r_max=rd))
        assert np.all(small < full) and np.all(small > 0)
    # excising a core removes the brightest shells, whatever their temperature
    l_ex = np.asarray(g.x_ray_luminosity(M, Z, PLANCK18, p, conc=CONC,
                                         r_min=0.15 * rd, r_max=rd))
    l_in = np.asarray(g.x_ray_luminosity(M, Z, PLANCK18, p, conc=CONC,
                                         r_max=rd))
    assert np.all(l_ex < l_in)
