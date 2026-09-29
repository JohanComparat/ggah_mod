"""Internal consistency for the truncated lensing profiles.

**The external reference lives in ``tests/test_lensing_goldens.py``**, against
the Oguri et al. (2026) implementation at 50 digits.  This module is the other
half, and is not redundant with it: what is checked here is what a profile has
to satisfy on its own terms -- that the functions run, stay finite, and meet
their own analytic limits, mass conservation, the truncation, the ``k -> 0``
normalisation.

Those limits catch a *different* class of error than goldens do.  A golden
compares one function to one array on one grid; a mass-conservation check ties
two functions together, so it still bites at parameters the goldens never
sampled -- which is most of them, since the goldens are three concentrations
and one truncation.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.halos import lensing_profiles as L
from ggah_mod.halos import profiles as P

_R = np.logspace(-2.5, 0.8, 12)
_RHO_S, _R_S, _C_T, _TAU = 1e15, 0.2, 6.0, 15.0
_K = np.logspace(-3, 2, 25)


class TestFiniteAndSane:
    @pytest.mark.parametrize("name,fn", [
        ("tnfw_sigma", lambda: L.tnfw_sigma(_R, _RHO_S, _R_S, _C_T)),
        ("tnfw_mean_sigma", lambda: L.tnfw_mean_sigma(_R, _RHO_S, _R_S, _C_T)),
        ("tnfw_delta_sigma", lambda: L.tnfw_delta_sigma(_R, _RHO_S, _R_S, _C_T)),
        ("bmo_sigma", lambda: L.bmo_sigma(_R, _RHO_S, _R_S, _TAU)),
        ("bmo_mean_sigma", lambda: L.bmo_mean_sigma(_R, _RHO_S, _R_S, _TAU)),
        ("bmo_delta_sigma", lambda: L.bmo_delta_sigma(_R, _RHO_S, _R_S, _TAU)),
        ("hernquist_sigma", lambda: L.hernquist_sigma(_R, 1e12, 0.01)),
        ("hernquist_delta_sigma", lambda: L.hernquist_delta_sigma(_R, 1e12, 0.01)),
    ])
    def test_finite(self, name, fn):
        a = np.asarray(fn())
        assert np.all(np.isfinite(a)), name
        assert np.all(a >= 0.0), name


class TestAnalyticLimits:
    """The limits each profile must satisfy by construction."""

    def test_tnfw_mass_conservation(self):
        """Beyond the truncation all the mass is enclosed, so
        pi R^2 Sigma-bar -> M_t exactly."""
        m_t = 4 * np.pi * _RHO_S * _R_S ** 3 * float(P.g_nfw(_C_T))
        big = np.array([50.0])
        got = float(np.pi * big[0] ** 2
                    * np.asarray(L.tnfw_mean_sigma(big, _RHO_S, _R_S, _C_T))[0])
        assert got == pytest.approx(m_t, rel=1e-10)

    def test_tnfw_sigma_vanishes_beyond_truncation(self):
        outside = np.array([_C_T * _R_S * 1.001, _C_T * _R_S * 5.0])
        assert np.all(np.asarray(L.tnfw_sigma(outside, _RHO_S, _R_S, _C_T)) == 0.0)

    def test_tnfw_mass_saturates(self):
        m_t = 4 * np.pi * _RHO_S * _R_S ** 3 * float(P.g_nfw(_C_T))
        assert float(L.tnfw_mass(1e3, _RHO_S, _R_S, _C_T)) == pytest.approx(m_t, rel=1e-12)

    def test_bmo_total_mass_is_the_limit_of_the_enclosed_mass(self):
        ratio = float(L.bmo_mass(1e6, _RHO_S, _R_S, _TAU)
                      / L.bmo_mass_total(_RHO_S, _R_S, _TAU))
        assert ratio == pytest.approx(1.0, rel=1e-9)

    def test_hernquist_mass_conservation(self):
        big = np.array([50.0])
        got = float(np.pi * big[0] ** 2
                    * np.asarray(L.hernquist_mean_sigma(big, 1e12, 0.01))[0])
        assert got == pytest.approx(1e12, rel=1e-3)

    def test_hernquist_enclosed_mass_closed_form(self):
        for r in (0.005, 0.01, 0.1):
            x = r / 0.01
            assert float(L.hernquist_mass(r, 1e12, 0.01)) == pytest.approx(
                1e12 * x ** 2 / (1 + x) ** 2, rel=1e-12)


class TestFourierNormalisation:
    def test_bmo_uk_unity_at_large_scales(self):
        """Normalised by the *total* BMO mass, deliberately -- the reference
        normalises by M_Delta, a ~34% different convention."""
        u = np.asarray(L.bmo_uk(np.array([1e-6]), np.array([_R_S]), np.array([_TAU])))
        assert float(u[0, 0]) == pytest.approx(1.0, abs=1e-9)

    def test_bmo_uk_decreasing_and_positive(self):
        u = np.asarray(L.bmo_uk(_K, np.array([_R_S]), np.array([_TAU])))[:, 0]
        assert np.all(u > 0) and np.all(np.diff(u) < 0)

    def test_bmo_falls_faster_than_nfw(self):
        """A smoothly truncated profile has less power on small scales."""
        u_b = np.asarray(L.bmo_uk(_K, np.array([_R_S]), np.array([_TAU])))[:, 0]
        u_n = np.asarray(P.nfw_uk(_K, np.array([_R_S]), np.array([_C_T])))[:, 0]
        assert np.all(u_b[3:] < u_n[3:])

    def test_tnfw_uk_is_the_nfw_transform(self):
        """An alias, not a reimplementation: the analytic NFW transform is
        already the integral cut at the truncation radius."""
        assert L.tnfw_uk is P.nfw_uk


@pytest.mark.x64
class TestDifferentiable:
    @pytest.mark.parametrize("name,fn,x0", [
        ("bmo tau", lambda t: jnp.sum(L.bmo_delta_sigma(jnp.asarray(_R), _RHO_S, _R_S, t)), 15.0),
        ("tnfw c_t", lambda c: jnp.sum(L.tnfw_delta_sigma(jnp.asarray(_R), _RHO_S, _R_S, c)), 6.0),
        ("hern r_b", lambda rb: jnp.sum(L.hernquist_delta_sigma(jnp.asarray(_R), 1e12, rb)), 0.01),
    ])
    def test_gradient_matches_finite_difference(self, name, fn, x0):
        ad = float(jax.grad(fn)(x0))
        step = 1e-6 * abs(x0)
        fd = float((fn(x0 + step) - fn(x0 - step)) / (2 * step))
        assert np.isfinite(ad)
        assert ad == pytest.approx(fd, rel=1e-4), name

    def test_bmo_uk_differentiable(self):
        g = float(jax.grad(lambda t: jnp.sum(
            L.bmo_uk(jnp.asarray(_K), jnp.array([_R_S]), jnp.array([t]))))(_TAU))
        assert np.isfinite(g) and g != 0.0
