r"""Galaxy-galaxy lensing: :math:`\Sigma_{\rm crit}`, the source sample, and the
Gaussian covariance of :math:`\Delta\Sigma` and of :math:`w_p\times\Delta\Sigma`.

The covariance follows Singh et al. (2017, Eq. 10 and appendix A) for the
estimator with the signal around random points subtracted, the one with the
smaller variance:

.. math::

    {\rm Cov}\left[\Delta\Sigma_i, \Delta\Sigma_j\right] = \frac{1}{V_W}
    \int\frac{k\,dk}{2\pi}\,\bar J_2(k;i)\,\bar J_2(k;j)
    \Big[(P_{gg} + N_g)(P_{\Sigma\Sigma} + N_\Sigma)
         + \Delta\Pi_2\,\bar\rho_m^2 P_{gm}^2\Big],

and, for the same lenses,

.. math::

    {\rm Cov}\left[w_p(r_i), \Delta\Sigma_j\right] = \frac{2\,\Delta\Pi_\times}{V_W}
    \int\frac{k\,dk}{2\pi}\,\bar J_0(k;i)\,\bar J_2(k;j)\,
    (P_{gg} + N_g)\,\bar\rho_m P_{gm},

with :math:`V_W = A\,L_W` the lens volume *inside the lens-source overlap*,
:math:`P_{\Sigma\Sigma}(k) = \Sigma_{\rm crit}^2\chi_l^2 C_{\kappa\kappa}(k\chi_l)`
the convergence of everything along the source line of sight mapped onto the
lens plane, :math:`N_\Sigma = \Sigma_{\rm crit}^2\sigma_e^2/n_s` the shape noise
per comoving area at the lens, :math:`\Delta\Pi_2 = \int d\Pi\,W^2(\Pi)` the
lensing window's length (about 700 Mpc/h in Singh et al.), and
:math:`\Delta\Pi_\times = \int_{-\Pi}^{\Pi}W(\Pi)\,d\Pi \le 2\Pi` its overlap with
the :math:`w_p` window.  The line-of-sight integrals are taken in the
long-window limit, which is the approximation Singh et al. make; the
:math:`w_p` block alone keeps its exact :math:`k_\parallel` treatment
(:mod:`~ggah_mod.covariance.gaussian`).

Everything here is comoving, in the units of
:func:`~ggah_mod.observables.real_space.delta_sigma`:
:math:`(M_\odot/h)(\mathrm{pc}/h)^{-2}`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ..cosmology import background
from ..cosmology.constants import C_KM_S
from ..observables.real_space import SIGMA_UNIT
from .gaussian import annulus_j0, annulus_j2, annulus_overlap, _log_interp

__all__ = ["G_MPC_KMS2_MSUN", "sigma_crit_comoving", "effective_sigma_crit",
           "SourceSample", "LensingQuadrature", "make_lensing_quadrature",
           "delta_sigma_gaussian", "wp_delta_sigma_gaussian", "ARCMIN2_PER_SR"]

#: Newton's constant in Mpc (km/s)^2 / Msun (IAU 2015 nominal solar mass
#: parameter over the 2012 astronomical unit and parsec).
G_MPC_KMS2_MSUN = 4.30091727003628e-9
#: Square arcminutes per steradian.
ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2


def sigma_crit_comoving(z_l, z_s, cosmo):
    r"""Comoving :math:`\Sigma_{\rm crit}` [(Msun/h)(Mpc/h)^-2].

    .. math::

        \Sigma_{\rm crit} = \frac{c^2}{4\pi G}\,
            \frac{f_K(\chi_s)}{f_K(\chi_l)\,f_K(\chi_s-\chi_l)\,(1+z_l)},

    with transverse distances in Mpc/h, so a curved model needs no branch.  The
    :math:`h` units cancel into the ones :math:`\Delta\Sigma` is reported in.
    Infinite (no lensing) for a source at or in front of the lens.
    """
    z_l = jnp.asarray(z_l)
    z_s = jnp.asarray(z_s)
    prefactor = C_KM_S ** 2 / (4.0 * np.pi * G_MPC_KMS2_MSUN)
    chi_l = background.transverse_distance(background.comoving_distance(z_l, cosmo), cosmo)
    chi_s = background.transverse_distance(background.comoving_distance(z_s, cosmo), cosmo)
    chi_ls = background.comoving_distance_z1z2(z_l, z_s, cosmo)
    behind = z_s > z_l
    safe = jnp.where(behind, chi_ls, 1.0)
    out = jnp.where(behind, prefactor * chi_s / (chi_l * safe * (1.0 + z_l)),
                    jnp.inf)
    # Layer 1 returns distances at least 1-d; hand back the inputs' own shape.
    return jnp.reshape(out, jnp.broadcast_shapes(z_l.shape, z_s.shape))


def effective_sigma_crit(z_l, z_s, n_s, cosmo):
    r""":math:`\langle\Sigma_{\rm crit}^{-1}\rangle^{-1}` over a source
    distribution ``n_s(z_s)`` for a lens at ``z_l``, trapezoid in :math:`z_s`.

    The inverse is what is averaged, because the shear is linear in
    :math:`\Sigma_{\rm crit}^{-1}`; sources in front of the lens contribute zero
    rather than a negative weight.
    """
    z_s = jnp.asarray(z_s)
    n_s = jnp.asarray(n_s)
    inv = 1.0 / sigma_crit_comoving(z_l, z_s, cosmo)
    return jnp.trapezoid(n_s, z_s) / jnp.trapezoid(n_s * inv, z_s)


@dataclass(frozen=True)
class SourceSample:
    r"""What a :math:`\Delta\Sigma` covariance needs from the sources.

    Parameters
    ----------
    name : str
    sigma_crit : float
        Effective comoving :math:`\Sigma_{\rm crit}` for this lens-source pair
        [(Msun/h)(Mpc/h)^-2], e.g. from :func:`effective_sigma_crit`.
    sigma_e : float
        Shape noise **per component**.
    n_eff_arcmin2 : float
        Effective source density [arcmin^-2].
    delta_pi2 : float
        :math:`\int d\Pi\,W^2(\Pi)` of the lensing window [Mpc/h].
    delta_pi_cross : float
        :math:`\int_{-\Pi}^{\Pi} W(\Pi)\,d\Pi` for the :math:`w_p` window it is
        crossed with [Mpc/h]; at most :math:`2\Pi`.
    kappa_kernel : str or None
        Name of the lensing kernel whose :math:`C_{\kappa\kappa}` supplies
        :math:`P_{\Sigma\Sigma}`.  ``None`` drops the line-of-sight lensing term
        -- stated, not defaulted: Singh et al. find shape noise dominates at
        current source densities, and a caller relying on that says so here.
    provenance : str
    """

    name: str
    sigma_crit: float
    sigma_e: float
    n_eff_arcmin2: float
    delta_pi2: float
    delta_pi_cross: float
    kappa_kernel: str | None
    provenance: str

    def shape_noise(self, chi_l: float) -> float:
        r""":math:`N_\Sigma = \Sigma_{\rm crit}^2\sigma_e^2/n_s` per comoving
        (Mpc/h)^2 at the lens plane."""
        n_sr = self.n_eff_arcmin2 * ARCMIN2_PER_SR
        n_lens_plane = n_sr / chi_l ** 2
        return self.sigma_crit ** 2 * self.sigma_e ** 2 / n_lens_plane


@dataclass(frozen=True)
class LensingQuadrature:
    r"""Static :math:`k` nodes and bin-averaged :math:`J_0`, :math:`J_2` for one
    binning.  ``weight`` already holds :math:`k\,dk/2\pi`."""

    k: np.ndarray
    weight: np.ndarray
    j0: np.ndarray
    j2: np.ndarray
    edges: np.ndarray


def make_lensing_quadrature(rp_edges, *, k_min=1e-4, k_max=200.0,
                            n=768) -> LensingQuadrature:
    """Gauss-Legendre in ln k; kernels averaged over annulus area."""
    e = np.asarray(rp_edges, dtype=float)
    if e.size and e[0] <= 0:
        raise ValueError("rp_edges must be positive")
    if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0) or e[0] <= 0:
        raise ValueError("rp_edges must be positive and increasing")
    x, w = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (np.log(k_max) - np.log(k_min))
    k = np.exp(half * (x + 1.0) + np.log(k_min))
    weight = half * w * k * k / (2.0 * np.pi)
    return LensingQuadrature(k, weight, annulus_j0(k, e[:-1], e[1:]),
                             annulus_j2(k, e[:-1], e[1:]), e)


def delta_sigma_gaussian(quad_i: LensingQuadrature, quad_j: LensingQuadrature,
                         volume, k, p_g1g2, n_g1g2, p_g1m, p_g2m, rho_m,
                         p_sigma_sigma, n_sigma, delta_pi2):
    r"""Gaussian :math:`{\rm Cov}[\Delta\Sigma^{g_1}_i, \Delta\Sigma^{g_2}_j]`
    for lens samples :math:`g_1, g_2` and one source sample, in
    :math:`(M_\odot/h)^2(\mathrm{pc}/h)^{-4}`.

    ``p_sigma_sigma`` is on the quadrature nodes already -- a projected,
    lens-plane quantity, not a 3D spectrum on ``k``.  The constant
    :math:`N_{g_1g_2}N_\Sigma` is added in closed form by
    :func:`~ggah_mod.covariance.gaussian.annulus_overlap`: it is the pair-count
    shape noise, :math:`\Sigma_{\rm crit}^2\sigma_e^2/N_{\rm pairs}`, and its
    Bessel integral lies past every grid.
    """
    kn = jnp.asarray(quad_i.k)
    pgg = _log_interp(kn, k, p_g1g2)
    pgm_i = _log_interp(kn, k, p_g1m)
    pgm_j = _log_interp(kn, k, p_g2m)
    pss = jnp.asarray(p_sigma_sigma)
    g = pgg * pss + pgg * n_sigma + n_g1g2 * pss \
        + delta_pi2 * rho_m ** 2 * pgm_i * pgm_j
    cov = jnp.einsum("in,n,jn->ij", jnp.asarray(quad_i.j2),
                     jnp.asarray(quad_i.weight) * g, jnp.asarray(quad_j.j2))
    noise = n_g1g2 * n_sigma * jnp.asarray(annulus_overlap(quad_i.edges,
                                                           quad_j.edges))
    return (cov + noise) / volume * SIGMA_UNIT ** 2


def wp_delta_sigma_gaussian(quad_wp: LensingQuadrature,
                            quad_ds: LensingQuadrature, volume, k,
                            p_ac, n_ac, p_bm, p_bc, n_bc, p_am, rho_m,
                            delta_pi_cross):
    r"""Gaussian :math:`{\rm Cov}[w_p^{ab}(r_i), \Delta\Sigma^{c}(R_j)]`, in
    :math:`(\mathrm{Mpc}/h)(M_\odot/h)(\mathrm{pc}/h)^{-2}`.

    Wick's theorem pairs the lens of :math:`\Delta\Sigma` with either leg of
    :math:`w_p`: :math:`\tilde P_{ac}P_{bm} + \tilde P_{bc}P_{am}`, which is
    :math:`2\tilde P_{gg}P_{gm}` for one sample.  No constant term survives
    (every product carries a :math:`P_{gm}`), so there is no closed-form part.
    """
    kn = jnp.asarray(quad_wp.k)
    g = delta_pi_cross * rho_m * (
        (_log_interp(kn, k, p_ac) + n_ac) * _log_interp(kn, k, p_bm)
        + (_log_interp(kn, k, p_bc) + n_bc) * _log_interp(kn, k, p_am))
    cov = jnp.einsum("in,n,jn->ij", jnp.asarray(quad_wp.j0),
                     jnp.asarray(quad_wp.weight) * g, jnp.asarray(quad_ds.j2))
    return cov / volume * SIGMA_UNIT
