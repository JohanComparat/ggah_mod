r"""Verification: the joint distribution of black-hole mass and X-ray luminosity.

:meth:`~ggah_mod.sectors.agn.AgnSector.joint_mbh_lx` integrated over
:math:`\log M_{\rm BH}` must be the sector's own luminosity distribution at
fixed stellar mass, to round-off, at both bolometric corrections;
:meth:`~ggah_mod.sectors.agn.AgnSector.active_bhmf` without a sample or a duty
cycle must be the black-hole mass function; the type-1 weight is a fraction
that falls as the obscured fraction rises; and every new output has the
derivative of a central difference in every AGN and obscuration parameter.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.background import luminosity_distance
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import agn as A
from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
from test_agn_kbol import AGN_140

Z = 0.135
GAL = GalaxySector("zumandelbaum15")
GP = galaxy_defaults("zumandelbaum15")
MSTARS = jnp.array([8.5, 9.5, 10.5, 11.5, 12.0])
KBOL = ("constant", "duras20")


def _sector(kbol="constant", **kw):
    return A.AgnSector(GAL, calibration="off", kbol=kbol, obscuration="split",
                       **kw)


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE.with_(n_m=64, m_min=1e10,
                                                     m_max=1e16),
                      make_pk("emu_pk"), z=Z)


class TestTheMarginalIsTheLuminosityDistribution:
    @pytest.mark.parametrize("kbol", KBOL)
    def test_the_integral_over_mbh_is_the_fixed_mstar_distribution(self, kbol):
        sec, p = _sector(kbol), A.AgnParams()
        _, joint = sec.joint_mbh_lx(MSTARS, p, z=0.3)
        marg = np.asarray(jnp.sum(joint, axis=1) * sec.dlam)
        ref = np.asarray(sec.p_active_loglx_given_mstar(MSTARS, p, z=0.3))
        big = ref > 1e-8 * ref.max(axis=1, keepdims=True)
        assert big.sum() > 1000
        np.testing.assert_allclose(marg[big], ref[big], rtol=1e-10)
        assert np.max(np.abs(marg - ref)) < 1e-12 * ref.max()

    def test_it_is_the_satellites_kernel(self, field):
        sec, p = _sector(), A.AgnParams()
        lg_ms, _, prob = sec._satellite_kernel(field, p, GP)
        np.testing.assert_array_equal(
            np.asarray(prob),
            np.asarray(sec.p_active_loglx_given_mstar(lg_ms, p, z=field.z)))

    def test_a_caller_grid_on_the_nodes_is_the_native_grid(self):
        sec, p = _sector(), A.AgnParams()
        mb, joint = sec.joint_mbh_lx(10.5, p, z=0.3)
        mb2, joint2 = sec.joint_mbh_lx(10.5, p, z=0.3, log10_mbh_grid=mb)
        np.testing.assert_array_equal(np.asarray(mb2), np.asarray(mb))
        np.testing.assert_allclose(np.asarray(joint2), np.asarray(joint),
                                   rtol=0, atol=1e-13 * float(joint.max()))

    def test_the_nodes_are_the_relation_and_sig_bh(self):
        """With no sample the moments are the relation and the kernel's
        Gaussian of width sig_bh, truncated at 2.26 dex: at the default
        sig_bh = 0.657 the truncation narrows it by 3.6e-3 (1.1e-3 at 1.4.0's
        0.597)."""
        sec, p = _sector(), A.AgnParams()
        mean, std = sec.selected_mbh_moments(MSTARS, None, None, p, z=0.3,
                                             type1=False)
        np.testing.assert_allclose(np.asarray(mean),
                                   np.asarray(A.mbh_powell(MSTARS, p)),
                                   rtol=0, atol=1e-12)
        x, g = sec._gauss_nodes(p.sig_bh)
        truncated = float(jnp.sqrt(jnp.sum(g * x ** 2)))
        assert float(x[-1]) == pytest.approx(2.26, abs=5e-3)
        np.testing.assert_allclose(np.asarray(std), truncated, rtol=1e-12)
        assert truncated / p.sig_bh - 1.0 == pytest.approx(-3.6e-3, abs=1e-4)

    def test_no_galaxy_below_the_cut_hosts_one(self):
        sec, p = _sector(), A.AgnParams()
        _, joint = sec.joint_mbh_lx(jnp.array([7.9, 8.0]), p, z=0.3)
        assert float(jnp.max(joint[0])) == 0.0 and float(jnp.max(joint[1])) > 0.0
        assert float(jnp.max(sec.p_active_loglx_given_mstar(7.9, p))) == 0.0
        _, pdf = sec.type1_erdf(7.9, p, z=0.3, normalise=False)
        assert float(jnp.max(pdf)) == 0.0


class TestTheSample:
    def test_type1_is_a_fraction_of_the_active(self):
        sec, p = _sector(), A.AgnParams()
        _, every = sec.joint_mbh_lx(MSTARS, p, z=0.3)
        _, t1 = sec.joint_mbh_lx(MSTARS, p, z=0.3, type1=True)
        live = np.asarray(every) > 0.0
        r = np.asarray(t1)[live] / np.asarray(every)[live]
        assert r.min() >= 0.0 and r.max() <= 1.0 and r.max() > 0.3

    @pytest.mark.parametrize("name, values", [
        ("f_bright_floor", (0.0, 0.1, 0.2, 0.3)),
        ("f_faint_norm", (0.4, 0.6, 0.8, 1.0)),
        ("l_blend_0", (42.5, 43.0, 43.5, 44.0)),
    ])
    def test_more_obscuration_fewer_type1(self, name, values):
        """Each sequence raises the obscured fraction at every luminosity."""
        sec, p = _sector(), A.AgnParams()
        n, f_obs = [], []
        for v in values:
            o = A.ObscurationParams().replace(**{name: v})
            _, j = sec.joint_mbh_lx(10.5, p, z=0.3, type1=True, obsc=o)
            n.append(float(jnp.sum(j)))
            f_obs.append(np.asarray(A.obscured_fraction(sec.loglx, 0.3, o)))
        assert all(np.all(b >= a - 1e-15) for a, b in zip(f_obs, f_obs[1:]))
        assert all(b < a for a, b in zip(n, n[1:])), n

    def test_type1_needs_the_split(self):
        sec = A.AgnSector(GAL, calibration="off")
        with pytest.raises(ValueError, match="obscuration='split'"):
            sec.selected_mbh_moments(10.5, 43.0, 44.0, A.AgnParams())
        mb, j = sec.joint_mbh_lx(10.5, A.AgnParams())
        assert j.shape == (mb.shape[0], sec.loglx.size)

    def test_a_flux_limit_is_a_threshold_at_the_unabsorbed_luminosity(self):
        sec, p = _sector(), A.AgnParams()
        z, band = 0.5, (0.5, 2.0)
        d_cm = (float(jnp.atleast_1d(luminosity_distance(z, PLANCK18))[0])
                / PLANCK18.h * C.MPC_CM)
        g = p.gamma_x
        k = (1.0 + z) ** (2.0 - g) * ((2.0 ** (2.0 - g) - 0.5 ** (2.0 - g))
                                      / (10.0 ** (2.0 - g) - 2.0 ** (2.0 - g)))
        ref = np.log10(4.0 * np.pi * d_cm ** 2) + p.log10_flux_min - np.log10(k)
        lmin = sec.unabsorbed_flux_luminosity(z, p, PLANCK18, band=band)
        assert float(lmin) == pytest.approx(ref, abs=1e-10)
        _, a = sec.joint_mbh_lx(10.5, p, z=z, type1=True, flux_limit=True,
                                cosmo=PLANCK18, band=band)
        _, b = sec.joint_mbh_lx(10.5, p, z=z, type1=True, lx_min=lmin)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        with pytest.raises(ValueError, match="cosmo"):
            sec.joint_mbh_lx(10.5, p, z=z, flux_limit=True)

    def test_a_luminosity_selected_sample_holds_heavier_black_holes(self):
        sec, p = _sector(), A.AgnParams()
        mean, std = sec.selected_mbh_moments(10.5, 44.0, None, p, z=0.3)
        assert float(mean) > float(A.mbh_powell(10.5, p)) + 0.2
        assert float(std) < p.sig_bh

    @pytest.mark.parametrize("kbol", KBOL)
    def test_the_moments_are_those_of_the_joint_on_the_luminosity_grid(self, kbol):
        """Two quadratures of one distribution: on the nodes of the kernel,
        and on the luminosity grid, interpolated (measured: 6e-5 dex)."""
        sec, p, o = _sector(kbol), A.AgnParams(), A.ObscurationParams()
        mean, std = sec.selected_mbh_moments(10.5, 43.0, 44.0, p, z=0.3, obsc=o)
        mb, j = sec.joint_mbh_lx(10.5, p, z=0.3, type1=True, obsc=o)
        w = jnp.sum(j * sec._sample_weight(sec.loglx, p, z=0.3, lx_lo=43.0,
                                           lx_hi=44.0), axis=-1)
        m = jnp.sum(w * mb) / jnp.sum(w)
        s = jnp.sqrt(jnp.sum(w * (mb - m) ** 2) / jnp.sum(w))
        assert float(mean) == pytest.approx(float(m), abs=2e-4)
        assert float(std) == pytest.approx(float(s), abs=2e-4)

    def test_a_measurement_error_adds_in_quadrature(self):
        sec, p = _sector(), A.AgnParams()
        m0, s0 = sec.selected_mbh_moments(10.5, 43.0, 44.0, p, z=0.3)
        m1, s1 = sec.selected_mbh_moments(10.5, 43.0, 44.0, p, z=0.3,
                                          log10_mbh_error=0.35)
        assert float(m1) == float(m0)
        assert float(s1) ** 2 == pytest.approx(float(s0) ** 2 + 0.35 ** 2,
                                               rel=1e-12)


class TestTheType1Erdf:
    def test_it_is_a_distribution_and_the_sample_raises_lambda(self):
        sec, p = _sector(), A.AgnParams()
        lam, pdf = sec.type1_erdf(10.5, p, z=0.3, type1=False)
        assert float(jnp.sum(pdf) * sec.dlam) == pytest.approx(1.0, abs=1e-12)
        e = sec._erdf_pivot(p, 0.3)
        np.testing.assert_allclose(np.asarray(pdf), np.asarray(e), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(lam),
            np.asarray(sec.loglam + A._lstar_tilt(10.5, p)), rtol=0, atol=1e-14)
        _, bright = sec.type1_erdf(10.5, p, z=0.3, lx_min=44.0)
        mean_all = float(jnp.sum(pdf * lam) * sec.dlam)
        mean_bright = float(jnp.sum(bright * lam) * sec.dlam)
        assert mean_bright > mean_all + 0.3

    def test_per_galaxy_it_integrates_to_the_sample_fraction(self):
        sec, p = _sector(), A.AgnParams()
        _, pdf = sec.type1_erdf(10.5, p, z=0.3, lx_min=43.0, normalise=False)
        _, j = sec.joint_mbh_lx(10.5, p, z=0.3, type1=True, lx_min=43.0)
        n_grid = float(jnp.sum(j) * sec.dlam * sec.dlx)
        assert float(jnp.sum(pdf) * sec.dlam) == pytest.approx(n_grid, rel=1e-3)

    def test_a_mass_error_widens_it(self):
        sec, p = _sector(), A.AgnParams()
        lam, pdf = sec.type1_erdf(10.5, p, z=0.3, lx_min=42.0)
        _, wide = sec.type1_erdf(10.5, p, z=0.3, lx_min=42.0,
                                 log10_mbh_error=0.35)
        assert float(jnp.sum(wide) * sec.dlam) == pytest.approx(1.0, abs=1e-12)

        def var(f):
            m = jnp.sum(f * lam) * sec.dlam
            return float(jnp.sum(f * (lam - m) ** 2) * sec.dlam)

        assert var(wide) > var(pdf) + 0.05


class TestTheActiveMassFunction:
    LGMBH = jnp.linspace(5.0, 10.5, 23)

    @pytest.mark.parametrize("kbol", KBOL)
    def test_without_a_sample_or_a_duty_cycle_it_is_the_bhmf(self, field, kbol):
        sec, p = _sector(kbol), A.AgnParams()
        a = sec.active_bhmf(field, self.LGMBH, p, GP, type1=False,
                            with_duty=False)
        b = sec.black_hole_mass_function(field, self.LGMBH, p, GP)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)

    def test_each_cut_removes_black_holes(self, field):
        sec, p = _sector(), A.AgnParams()
        every = sec.black_hole_mass_function(field, self.LGMBH, p, GP)
        active = sec.active_bhmf(field, self.LGMBH, p, GP, type1=False)
        bright = sec.active_bhmf(field, self.LGMBH, p, GP, type1=False,
                                 lx_min=42.0)
        t1 = sec.active_bhmf(field, self.LGMBH, p, GP, lx_min=42.0)
        assert np.all(np.asarray(active) < np.asarray(every))
        assert np.all(np.asarray(bright) < np.asarray(active))
        assert np.all(np.asarray(t1) < np.asarray(bright))

    @pytest.mark.parametrize("kbol, params", [
        pytest.param("constant", AGN_140, id="constant"),
        pytest.param("duras20", {}, id="duras20")])
    def test_its_integral_is_the_number_of_agn_the_xlf_counts(self, field, kbol,
                                                              params):
        """Over M_BH and over L_X, centrals at fixed halo mass and satellites:
        the central's conditional (beta, tau) must rebuild the kernel of width
        sigma_lx, or the two counts part.  Each correction at the parameters
        fitted with it: k_bol = 20 at those of 1.4.0, Duras et al. (2020) at
        the defaults.  What remains is the Gaussian the kernel truncates at
        2.26 dex and the mass function does not: 3.2e-4 at 1.4.0's sig_bh =
        0.597, 1.2e-3 at the defaults' 0.657, 5e-5 with the kernel's grid
        widened to 3.76 dex."""
        sec, p = _sector(kbol), A.AgnParams(**params)
        lg = jnp.linspace(2.0, 12.5, 841)
        n_bh = float(jnp.trapezoid(sec.active_bhmf(field, lg, p, GP,
                                                   type1=False, lx_min=42.5),
                                   lg))
        grid, phi = sec.xlf(field, p, GP)
        sel = sec._sample_weight(grid, p, z=field.z, lx_min=42.5)
        n_xlf = float(jnp.sum(phi * sel) * sec.dlx)
        assert n_bh == pytest.approx(n_xlf, rel=2e-3)

    def test_the_central_conditional_rebuilds_the_central_kernel(self, field):
        sec, p = _sector(), A.AgnParams()
        sig_ms = sec.sigma_mstar(jnp.log10(field.m), GP, h=field.cosmo.h)
        beta, tau = sec._central_bh_conditional(p, sig_ms)
        sig_lm = sec._sigma_lm(p, sig_ms)
        sig_lx = sec._sigma_lx(p, sig_ms)
        np.testing.assert_allclose(np.asarray(tau ** 2 + (1 + beta) ** 2 * sig_lm ** 2),
                                   np.asarray(sig_lx ** 2), rtol=1e-12)
        q = p.replace(log10_lstar_mslope=0.0)
        b0, t0 = sec._central_bh_conditional(q, sig_ms)
        assert float(jnp.max(jnp.abs(b0))) == 0.0
        assert float(jnp.max(t0)) == pytest.approx(1e-3)


# ---------------------------------------------------------------- derivatives
def _theta():
    return ravel_pytree((A.AgnParams(), A.ObscurationParams()))


def _central_differences(f, theta, h=1e-5):
    """``(central, f(theta))``: central differences of ``f`` along every axis
    of ``theta``, the step ``h`` times the value (or ``h`` below one)."""
    cols = []
    for i in range(theta.size):
        step = h * max(1.0, abs(float(theta[i])))
        e = jnp.zeros_like(theta).at[i].set(step)
        cols.append((f(theta + e) - f(theta - e)) / (2 * step))
    return jnp.stack(cols, axis=-1), f(theta)


def _scale(fd, f0):
    """Per parameter, the largest derivative, floored at 1e-4 of the largest
    output: a derivative that is zero exactly (``log10_ferdf`` in a normalised
    distribution) is compared against that, not against round-off."""
    fd = np.asarray(fd)
    col = np.max(np.abs(fd), axis=tuple(range(fd.ndim - 1)), keepdims=True)
    return np.maximum(col, 1e-4 * np.max(np.abs(np.asarray(f0))))


def _assert_matches(ad, fd, f0, *, mask=None, rtol=1e-6):
    ad, fd = np.asarray(ad), np.asarray(fd)
    scale = _scale(fd, f0)
    bad = np.abs(ad - fd) > rtol * (np.abs(fd) + scale)
    if mask is not None:
        bad &= mask
    assert not bad.any(), (f"{bad.sum()} of {bad.size} entries differ; worst "
                           f"{np.max(np.abs(ad - fd) / (np.abs(fd) + scale))}")


@pytest.mark.x64
class TestTheDerivativesAreCentralDifferences:
    """On a coarse sector, at the defaults, in all 29 AGN and obscuration
    parameters."""

    SEC = _sector(n_lam=96, n_lx=160)

    def _check(self, out):
        theta, unravel = _theta()
        f = jax.jit(lambda t: out(*unravel(t)))
        ad = jax.jacfwd(f)(theta)
        fd, f0 = _central_differences(f, theta)
        _assert_matches(ad, fd, f0)

    def test_the_moments(self):
        self._check(lambda p, o: jnp.stack(self.SEC.selected_mbh_moments(
            jnp.array([9.5, 10.5, 11.5]), 43.0, 44.5, p, z=0.3, obsc=o,
            flux_limit=True, cosmo=PLANCK18, band=(0.5, 2.0))))

    def test_the_type1_erdf(self):
        self._check(lambda p, o: self.SEC.type1_erdf(
            10.5, p, z=0.3, obsc=o, lx_min=42.5, log10_mbh_error=0.3)[1])

    def test_the_active_mass_function(self, field):
        lg = jnp.linspace(6.0, 10.0, 9)
        self._check(lambda p, o: self.SEC.active_bhmf(
            field, lg, p, GP, obsc=o, lx_min=42.0, flux_limit=True,
            band=(0.5, 2.0)))

    def test_the_joint(self):
        """Linear in Phi_lambda between its nodes, as the kernel is, so an
        entry whose argument t - x_k is within 1e-4 dex of a node, or whose t
        is within 1e-4 dex of the kernel's ends, has two one-sided slopes and
        is left out; the steps move t by at most 8e-6 dex."""
        sec, p, theta, unravel = self.SEC, A.AgnParams(), *_theta()
        t = (sec.loglx - jnp.log10(A.LBOL_COEF / p.k_bol)
             - A.mbh_powell(10.5, p) - A._lstar_tilt(10.5, p))
        x, _ = sec._gauss_nodes(p.sig_bh)
        nodes = jnp.concatenate([sec.loglam[:1] - sec.dlam, sec.loglam,
                                 sec.loglam[-1:] + sec.dlam])
        arg = t[None, :] - x[:, None]
        near = jnp.min(jnp.abs(arg[..., None] - nodes), axis=-1)
        half = sec.loglam.size // 2
        t_lo = sec.loglam[0] - half * sec.dlam
        t_hi = t_lo + (sec.loglam.size + 2 * half - 1) * sec.dlam
        edge = jnp.minimum(jnp.abs(t - t_lo), jnp.abs(t - t_hi))
        smooth = np.asarray((near > 1e-4) & (edge[None, :] > 1e-4))
        assert smooth.mean() > 0.99

        def joint(t):
            p, o = unravel(t)
            return self.SEC.joint_mbh_lx(10.5, p, z=0.3, type1=True, obsc=o,
                                         lx_min=42.0)[1]

        f = jax.jit(joint)
        ad = jax.jacfwd(f)(theta)
        fd, f0 = _central_differences(f, theta, h=1e-6)
        _assert_matches(ad, fd, f0, mask=smooth[..., None])
