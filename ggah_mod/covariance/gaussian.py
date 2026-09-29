r"""The Gaussian (disconnected) covariance: products of two spectra per mode.

For four fields :math:`a, b, c, d` with measured spectra :math:`\tilde P = P + N`,
the covariance of :math:`\hat P_{ab}` and :math:`\hat P_{cd}` in a Gaussian field is
Wick's theorem and nothing else:

.. math::

    {\rm Cov}\left[\hat P_{ab}, \hat P_{cd}\right] =
    \frac{\tilde P_{ac}\tilde P_{bd} + \tilde P_{ad}\tilde P_{bc}}{N_{\rm modes}} .

What differs between statistics is only how modes are counted and how a bin
averages them.  This module holds those two things, as pure functions of arrays,
so that the spectra -- the traced part -- are computed by layers 4 and 5 and
nowhere here.

**Every kernel is static.**  Bin-averaged Bessel functions depend on the bin
edges and the wavenumber nodes, never on a parameter, so they are built once in
numpy with :mod:`scipy.special` and the traced path is a matrix product.  That is
layer 5's rule for its transforms, kept for the same reason: no Bessel function
on the differentiable path.

Harmonic space
--------------

A band :math:`[\ell_{\rm lo}, \ell_{\rm hi})` of integer multipoles holds
:math:`\sum_\ell (2\ell+1) = \ell_{\rm hi}^2 - \ell_{\rm lo}^2` modes on the full
sky, and :math:`f_{\rm sky}` of them on a footprint.  That count is exact for
integer edges, so it is used rather than :math:`(2\ell_c+1)\Delta\ell`; the two
differ by :math:`1/(2\ell_c+1)` of the band, which is a per cent at
:math:`\ell_c = 50`.  The :math:`f_{\rm sky}` scaling is the approximation the
DES Y3 covariance validation found to matter most (Friedrich et al. 2021), and
it is the one stated here rather than hidden.

Projected clustering in a volume
--------------------------------

:math:`w_p(r_p) = \int_{-\Pi}^{\Pi}\xi(r_p,\pi)\,d\pi` averaged over an annulus
is the field seen through the kernel
:math:`S_\Pi(k_\parallel)\,\bar J_0(k_\perp)`, with
:math:`S_\Pi = 2\sin(k_\parallel\Pi)/k_\parallel`, so for a survey volume
:math:`V`

.. math::

    {\rm Cov}\left[w_p^{ab}(r_i), w_p^{cd}(r_j)\right] = \frac{1}{V}
    \int\frac{k_\perp dk_\perp}{2\pi}\,\bar J_0(k_\perp; i)\,\bar J_0(k_\perp; j)
    \int\frac{dk_\parallel}{2\pi}\,S_\Pi^2(k_\parallel)\,
    \left[\tilde P_{ac}\tilde P_{bd} + \tilde P_{ad}\tilde P_{bc}\right]\!(k).

The part that is constant in :math:`k` -- noise times noise -- is done in closed
form, because its :math:`k_\perp` integral converges only past
:math:`1/\Delta r_p`, which is :math:`3\times10^{2}\,h`/Mpc for the smallest LS10
bin and beyond every spectrum grid.  With
:math:`\int dk_\parallel S_\Pi^2/2\pi = 2\Pi` and
:math:`\int k\,dk\,\bar J_0(k;i)\bar J_0(k;j)/2\pi = \delta_{ij}/A_i`, it is

.. math::

    (N_{ac}N_{bd} + N_{ad}N_{bc})\,\frac{2\Pi}{V A_i}\,\delta_{ij},
    \qquad A_i = \pi\left(r_{{\rm hi},i}^2 - r_{{\rm lo},i}^2\right),

which is the Poisson pair-count variance.  Redshift-space distortions are not
in this: :math:`w_p` integrates them out to the extent :math:`\Pi` allows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from scipy import special

__all__ = ["band_modes", "cl_gaussian", "annulus_j0", "annulus_j2",
           "annulus_overlap", "WpKernel", "make_wp_kernel", "wp_gaussian",
           "wp_noise_term"]


# --------------------------------------------------------------------------
# harmonic space
# --------------------------------------------------------------------------

def band_modes(ell_edges, f_sky: float) -> np.ndarray:
    r"""Number of modes per band, :math:`f_{\rm sky}(\ell_{\rm hi}^2 -
    \ell_{\rm lo}^2)` for integer edges :math:`[\ell_{\rm lo}, \ell_{\rm hi})`.

    Edges must be integers: a fractional edge splits a multipole, and a mode
    cannot be split.
    """
    e = np.asarray(ell_edges, dtype=float)
    if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("ell_edges must be increasing, with at least two")
    if not np.allclose(e, np.round(e)) or e[0] < 0:
        raise ValueError(f"ell_edges must be non-negative integers, got {e}")
    return f_sky * (e[1:] ** 2 - e[:-1] ** 2)


def cl_gaussian(cl_ac, cl_bd, cl_ad, cl_bc, n_modes):
    r"""Diagonal of :math:`{\rm Cov}[C^{ab}_\ell, C^{cd}_\ell]` per band.

    The inputs are the *measured* spectra, noise included on autos.  Bands do
    not couple: the Gaussian term of a full-sky field is diagonal in
    :math:`\ell`, and the :math:`f_{\rm sky}` approximation keeps it so.
    """
    return (jnp.asarray(cl_ac) * cl_bd + jnp.asarray(cl_ad) * cl_bc) \
        / jnp.asarray(n_modes)


# --------------------------------------------------------------------------
# real space: bin-averaged kernels
# --------------------------------------------------------------------------

def _annulus_area(r_lo, r_hi):
    return np.pi * (np.asarray(r_hi) ** 2 - np.asarray(r_lo) ** 2)


def annulus_j0(k, r_lo, r_hi) -> np.ndarray:
    r""":math:`\bar J_0` averaged over the annulus area,
    :math:`2[rJ_1(kr)]_{r_{\rm lo}}^{r_{\rm hi}}/[k(r_{\rm hi}^2-r_{\rm lo}^2)]`.

    Shape ``(n_bins, n_k)``.  Tends to 1 as :math:`k \to 0`.
    """
    k = np.asarray(k, dtype=float)[None, :]
    lo, hi = np.asarray(r_lo)[:, None], np.asarray(r_hi)[:, None]
    num = hi * special.j1(k * hi) - lo * special.j1(k * lo)
    return 2.0 * num / (k * (hi ** 2 - lo ** 2))


def annulus_j2(k, r_lo, r_hi) -> np.ndarray:
    r""":math:`\bar J_2` averaged over the annulus area.

    From :math:`\int xJ_2(x)\,dx = -xJ_1(x) - 2J_0(x)`.  The kernel of
    :math:`\Delta\Sigma` and :math:`\gamma_t`; tends to 0 as :math:`k \to 0`.
    """
    k = np.asarray(k, dtype=float)[None, :]
    lo, hi = np.asarray(r_lo)[:, None], np.asarray(r_hi)[:, None]

    def prim(r):
        x = k * r
        return -x * special.j1(x) - 2.0 * special.j0(x)

    return 2.0 * (prim(hi) - prim(lo)) / (k ** 2 * (hi ** 2 - lo ** 2))


@dataclass(frozen=True)
class WpKernel:
    r"""Static quadrature for the :math:`w_p` Gaussian covariance.

    ``weight[i, j, n]`` multiplies :math:`G(k_n)` on a flattened
    :math:`(k_\perp, k_\parallel)` grid, and :math:`k_n` is ``k_eval``.  Built
    once for one binning, one :math:`\Pi` and one wavenumber reach.
    """

    k_eval: np.ndarray
    weight: np.ndarray
    area: np.ndarray
    pi_max: float
    edges: np.ndarray
    k_par: np.ndarray
    sigma_los: float = 0.0


def make_wp_kernel(rp_edges, pi_max: float, *, k_min: float = 1e-4,
                   k_max: float = 200.0, n_perp: int = 384,
                   n_panels: int | None = None, nodes_per_panel: int = 6,
                   n_tail: int = 48, sigma_los: float = 0.0) -> WpKernel:
    r"""Build the :math:`(k_\perp, k_\parallel)` quadrature for :math:`w_p`.

    :math:`k_\perp` is Gauss-Legendre in :math:`\ln k` on
    ``[k_min, k_max]``.  :math:`k_\parallel` is exact where :math:`S_\Pi^2`
    oscillates slowly enough to matter -- Gauss-Legendre panels between its
    zeros :math:`n\pi/\Pi` -- and replaced by its average
    :math:`2/k_\parallel^2` beyond, where the spectrum varies over many
    oscillations.  ``n_panels`` defaults to the zeros below 1 h/Mpc.

    ``sigma_los`` [Mpc/h] is a Gaussian line-of-sight dispersion of each
    galaxy's position -- a photometric redshift error
    :math:`\sigma_\chi = c\,\sigma_z/H(z)`.  Each signal spectrum is then
    damped by :math:`e^{-k_\parallel^2\sigma_{\rm los}^2}` (the product of the
    two galaxies' :math:`e^{-k_\parallel^2\sigma^2/2}`), noise is not, and the
    closed-form noise term is unchanged.  **The** :math:`w_p` **model must be
    smeared the same way** for the pair to be consistent; layer 5 does not do
    that, so a caller using this states that the signal is handled elsewhere.
    """
    if sigma_los < 0:
        raise ValueError("sigma_los is a dispersion and cannot be negative")
    e = np.asarray(rp_edges, dtype=float)
    if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0) or e[0] <= 0:
        raise ValueError("rp_edges must be positive and increasing")
    if pi_max <= 0:
        raise ValueError("pi_max must be positive")
    lo, hi = e[:-1], e[1:]

    x, w = np.polynomial.legendre.leggauss(n_perp)
    ln_k = 0.5 * (np.log(k_max) - np.log(k_min)) * (x + 1.0) + np.log(k_min)
    k_perp = np.exp(ln_k)
    w_perp = 0.5 * (np.log(k_max) - np.log(k_min)) * w * k_perp  # dk
    jb = annulus_j0(k_perp, lo, hi)                              # (nb, np)
    radial = (w_perp * k_perp / (2.0 * np.pi))[None, None, :] \
        * jb[:, None, :] * jb[None, :, :]                        # (nb, nb, np)

    # k_parallel on [0, inf), doubled for the even integrand.
    step = np.pi / pi_max
    if n_panels is None:
        n_panels = max(8, int(np.ceil(1.0 / step)))
    xg, wg = np.polynomial.legendre.leggauss(nodes_per_panel)
    kp, wkp = [], []
    for n in range(n_panels):
        a, b = n * step, (n + 1) * step
        kp.append(0.5 * (b - a) * (xg + 1.0) + a)
        wkp.append(0.5 * (b - a) * wg)
    kp, wkp = np.concatenate(kp), np.concatenate(wkp)
    s2 = np.where(kp > 0, (2.0 * np.sin(kp * pi_max) / np.where(kp > 0, kp, 1.0)) ** 2,
                  (2.0 * pi_max) ** 2)
    k_cut = n_panels * step
    xt, wt = np.polynomial.legendre.leggauss(n_tail)
    ln_t = 0.5 * (np.log(k_max) - np.log(k_cut)) * (xt + 1.0) + np.log(k_cut)
    kt = np.exp(ln_t)
    wkt = 0.5 * (np.log(k_max) - np.log(k_cut)) * wt * kt
    k_par = np.concatenate([kp, kt])
    w_par = 2.0 * np.concatenate([wkp * s2, wkt * 2.0 / kt ** 2]) / (2.0 * np.pi)

    k_eval = np.sqrt(k_perp[:, None] ** 2 + k_par[None, :] ** 2)  # (np, npar)
    kpar_grid = np.broadcast_to(k_par[None, :], k_eval.shape)
    weight = radial[:, :, :, None] * w_par[None, None, None, :]
    nb = lo.size
    return WpKernel(k_eval.reshape(-1),
                    weight.reshape(nb, nb, -1),
                    _annulus_area(lo, hi), float(pi_max), e,
                    np.ascontiguousarray(kpar_grid).reshape(-1),
                    float(sigma_los))


def _log_interp(kq, k, p):
    r"""Spectrum at ``kq`` by log-log interpolation, zero outside ``k``.

    Zero rather than extrapolated: past the grid a spectrum has no value here,
    and the kernel's reach is chosen to lie inside it.
    """
    k = jnp.asarray(k)
    p = jnp.asarray(p)
    inside = (kq >= k[0]) & (kq <= k[-1])
    lq = jnp.log(jnp.clip(kq, k[0], k[-1]))
    # Signed spectra (cross terms can be negative) are interpolated linearly
    # in log k; the log of |P| would lose the sign.
    return jnp.where(inside, jnp.interp(lq, jnp.log(k), p), 0.0)


def wp_gaussian(kernel: WpKernel, volume, k, p_ac, p_bd, p_ad, p_bc,
                n_ac=0.0, n_bd=0.0, n_ad=0.0, n_bc=0.0, w_overlap=0.0):
    r"""The :math:`w_p` Gaussian covariance, signal and noise together.

    ``p_*`` are **signal** spectra on ``k`` and ``n_*`` constant noise levels
    [(Mpc/h)\ :sup:`3`], non-zero only for an auto of one sample.  The integrand
    :math:`\tilde P\tilde P` is split: the part that varies with :math:`k` --
    signal times signal and signal times noise -- is integrated on the kernel's
    grid, and noise times noise, which is constant and whose integral lies past
    every grid, is added in closed form by :func:`wp_noise_term`.  Passing the
    noise inside ``p_*`` instead would count that constant twice and truncate
    it once.

    ``w_overlap`` adds the clustering of the pair itself to that constant; see
    :func:`wp_noise_term`.
    """
    kq = jnp.asarray(kernel.k_eval)
    damp = jnp.exp(-(jnp.asarray(kernel.k_par) * kernel.sigma_los) ** 2) \
        if kernel.sigma_los > 0 else 1.0
    pac, pbd = _log_interp(kq, k, p_ac) * damp, _log_interp(kq, k, p_bd) * damp
    pad, pbc = _log_interp(kq, k, p_ad) * damp, _log_interp(kq, k, p_bc) * damp
    g = (pac * pbd + pac * n_bd + n_ac * pbd
         + pad * pbc + pad * n_bc + n_ad * pbc)
    varying = jnp.einsum("ijn,n->ij", jnp.asarray(kernel.weight), g) / volume
    return varying + wp_noise_term(kernel, volume, n_ac, n_bd, n_ad, n_bc,
                                   w_overlap)


def annulus_overlap(edges_i, edges_j) -> np.ndarray:
    r""":math:`\int\frac{k\,dk}{2\pi}\bar J_\nu(k;i)\bar J_\nu(k;j) =
    |A_i\cap A_j|/(A_iA_j)`, for any order :math:`\nu`.

    From the closure relation :math:`\int_0^\infty k\,dk\,J_\nu(kr)J_\nu(kr')
    = \delta(r-r')/r`, which holds for every :math:`\nu > -1` -- so the same
    matrix closes the noise term of :math:`w_p` (:math:`J_0`) and of
    :math:`\Delta\Sigma` (:math:`J_2`), and two blocks on different binnings
    need no special case: the intersection of their annuli is the answer.
    Shape ``(n_i, n_j)``.
    """
    ei, ej = np.asarray(edges_i, float), np.asarray(edges_j, float)
    lo = np.maximum(ei[:-1, None], ej[None, :-1])
    hi = np.minimum(ei[1:, None], ej[None, 1:])
    inter = np.pi * np.clip(hi ** 2 - lo ** 2, 0.0, None)
    ai = np.pi * (ei[1:] ** 2 - ei[:-1] ** 2)
    aj = np.pi * (ej[1:] ** 2 - ej[:-1] ** 2)
    return inter / (ai[:, None] * aj[None, :])


def annulus_intersection_nodes(edges_i, edges_j, n: int = 8):
    r"""Nodes and weights for :math:`\int_{A_i\cap A_j} f(r)\,d^2r/(A_iA_j)`.

    Returns ``(r, weight)`` with ``r`` of shape ``(m,)`` and ``weight`` of
    shape ``(n_i, n_j, m)``: Gauss-Legendre in :math:`r` with the :math:`2\pi r`
    measure, ``n`` nodes on every non-empty intersection.  An empty
    intersection has no nodes, so two disjoint binnings give ``m = 0``.
    """
    ei, ej = np.asarray(edges_i, float), np.asarray(edges_j, float)
    lo = np.maximum(ei[:-1, None], ej[None, :-1])
    hi = np.minimum(ei[1:, None], ej[None, 1:])
    ai = np.pi * (ei[1:] ** 2 - ei[:-1] ** 2)
    aj = np.pi * (ej[1:] ** 2 - ej[:-1] ** 2)
    x, w = np.polynomial.legendre.leggauss(n)
    cells = [(i, j) for i in range(ai.size) for j in range(aj.size)
             if hi[i, j] > lo[i, j]]
    r = np.zeros(len(cells) * n)
    weight = np.zeros((ai.size, aj.size, r.size))
    for c, (i, j) in enumerate(cells):
        half = 0.5 * (hi[i, j] - lo[i, j])
        rr = half * (x + 1.0) + lo[i, j]
        r[c * n:(c + 1) * n] = rr
        weight[i, j, c * n:(c + 1) * n] = half * w * 2.0 * np.pi * rr \
            / (ai[i] * aj[j])
    return r, weight


def wp_noise_term(kernel: WpKernel, volume, n_ac, n_bd, n_ad, n_bc,
                  w_overlap=0.0):
    r"""The part that is constant in :math:`k`, in closed form.

    Two pair counts share a pair only if the bins overlap, and a shared pair is
    counted wherever its two members are, which is
    :math:`2\Pi_{\max} + w_p(r)` inside the window rather than
    :math:`2\Pi_{\max}`:

    .. math::

        (N_{ac}N_{bd} + N_{ad}N_{bc})\,\frac{1}{V}\,
        \frac{1}{A_iA_j}\int_{A_i\cap A_j} d^2r\,
        \big[2\Pi_{\max} + w^{ab}_p(r)\big] .

    The first part is Poisson noise times Poisson noise, the only part a
    Gaussian field has.  The second is the clustering of the shared pair
    (the :math:`1+\xi` of a pair-count variance), and it is several times the
    first below the one-halo scale, where :math:`w_p \gg 2\Pi_{\max}`.
    ``w_overlap`` is that integral, ``(n_i, n_j)``, from
    :func:`annulus_intersection_nodes`; zero leaves the Poisson part alone.
    """
    return (n_ac * n_bd + n_ad * n_bc) / volume * (
        2.0 * kernel.pi_max * jnp.asarray(annulus_overlap(kernel.edges,
                                                          kernel.edges))
        + w_overlap)
