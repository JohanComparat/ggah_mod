r"""Truncated halo profiles for cluster lensing: tNFW, BMO, Hernquist.

Three profiles that an NFW is not:

* **tNFW** (Takada & Jain 2003) -- NFW cut sharply at :math:`c_t r_s`.  Finite
  mass, discontinuous density.
* **BMO** (Baltz, Marshall & Oguri 2009, :math:`n=2`) -- NFW multiplied by
  :math:`[\tau^2/(\tau^2+x^2)]^2`.  Finite mass, smooth, and the standard
  choice for cluster weak lensing.
* **Hernquist** (1990) -- a stellar profile, for the central galaxy's own
  contribution to :math:`\Delta\Sigma`.

All pure JAX and differentiable.

Status: **validated against the Oguri et al. (2026) reference**
---------------------------------------------------------------

This section has said three different things, and the sequence is worth keeping.
``README.md`` once said these had "stopped being a placeholder" while this
docstring said they were untested; both were describing something true and the
word *placeholder* was doing the disagreeing.  ``PLAN.md`` item **F2** settled
that into one precise statement -- cross-validated internally, no external
reference -- and moved the *gap* to Tier 7, where obtaining a reference was work
rather than wording.  That work is done, and this is what it found.

**What is checked.**  The BMO profile is implemented *twice*, from two different
papers' algebra: as :func:`bmo_uk` in Fourier space and as :func:`bmo_sigma` /
:func:`bmo_delta_sigma` in real space.  Pushing the first through
``pk_to_delta_sigma`` reproduces the second to **3.0e-7 of the peak** on the four
radii ``tests/test_real_space.py`` pins (0.05 to 0.5 Mpc/h), and to 3.4e-7 over
forty from 0.05 to 2 -- through either transform engine, which are themselves
independent implementations.  The grid is named because the figure in the
technical paper measures the same thing on its own radii and gets its own number
in the same decade; a bare "3.0e-7" beside a bare "4.1e-7" reads as a
disagreement, and one unqualified number is how the status of this module came
to be argued about in the first place.  And
``tests/test_real_space.py`` demonstrates the check is *sharp*: the same
comparison against a profile scaled by 1.01 fails, so a wrong coefficient in
either representation shows rather than cancelling.  That is a good deal more
than a smoke test, which would pass any constant multiple.

(It was 1.2e-4 until the parity budget found that the Ogata rule's reach was
truncated; see :attr:`~ggah_mod.backend.Backend.hankel_h`.  The agreement was
never the profiles' error -- it was the transform's.)

**The external reference, and what it found.**  The dimensionless kernels are
now checked against the **Oguri et al. (2026)** implementation
(``github.com/massarin/halo_lensing``, arXiv:2512.13954), evaluated with
``mpmath`` at 50 decimal digits -- an independent code, which is what the
internal cross-check above structurally could not be.  First run, no tuning:

===============================  ================
quantity                         ``max err/peak``
===============================  ================
:math:`\Sigma` (TJ, BMO, Hern.)   **1.2e-13**
:math:`\bar\Sigma`                4.0e-10
:math:`M_{\rm BMO}(<x)`           2.1e-13
:math:`M^{\rm tot}_{\rm BMO}`     12 digits
:math:`P(x)`, :math:`Q(x)`       2.8e-9
:math:`\tilde u_{\rm BMO}(k)`     5.9e-11 relative
===============================  ================

The surface densities are at float64 round-off for expressions this long, so
there is nothing left to find in them.  The *mean* surface densities are three
decades worse, and locatably so: :math:`\bar\Sigma` carries the :math:`P`/
:math:`Q` hyperbolic combination, which is itself the loosest row -- the error
budget is one term deep, and it is the term the reference computes at 50 digits
for the same reason.

What the transcription above rules out is precisely what the internal check
could not: a misreading shared by both representations.  ``tests/test_lensing_goldens.py``
carries the arrays, their provenance, and the one caveat -- they are transported
through the predecessor's generator rather than regenerated here, so an error in
*the courier* would be invisible.  The generator is named there so anyone with
the clone can redo it.

Two normalisation conventions, both deliberate
----------------------------------------------

Neither is a bug, and both differ from the reference implementation:

1. :math:`\Sigma_{\rm tNFW} = 4\rho_s r_s f(x, c_t)` is the **exact projection
   of the truncated density**.  The reference's real-space ``tj_*`` carry an
   extra factor :math:`m_{\rm NFW}(c)`, which is inconsistent with its own
   Fourier window -- that window integrates to the halo mass.  The
   *dimensionless kernels* are identical.
2. :func:`bmo_uk` is normalised by the **total** BMO mass so
   :math:`u(k\to0) = 1`, matching every other transform in
   :mod:`ggah_mod.halos.profiles`.  The reference normalises by
   :math:`M_\Delta` instead; the two differ by
   :math:`M_{\rm tot}/M_\Delta \approx 1.34` at :math:`\tau = 2.5c`,
   :math:`c = 6`.

Numerical care
--------------

The closed forms cancel catastrophically at :math:`x = 1` (values like
:math:`1/\delta`, gradients like :math:`1/\delta^2`).  Each kernel is bridged
across :math:`|x-1| < \epsilon` by a series or a quadratic through anchors in
the well-conditioned region.  **Use x64**: several kernels lose relative
accuracy in float32 below :math:`x \approx 0.05`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import exp1, expi

from ..numerics import arctan
from .profiles import ci, g_nfw, nfw_uk, si

__all__ = [
    "tnfw_rho", "tnfw_mass", "tnfw_sigma", "tnfw_mean_sigma",
    "tnfw_delta_sigma", "tnfw_uk",
    "bmo_rho", "bmo_mass", "bmo_mass_total", "bmo_sigma", "bmo_mean_sigma",
    "bmo_delta_sigma", "bmo_uk",
    "hernquist_rho", "hernquist_mass", "hernquist_sigma",
    "hernquist_mean_sigma", "hernquist_delta_sigma", "HERNQUIST_RB_RE",
]

# Half-width of the x = 1 bridging window, and the anchor offset for the
# quadratic used by the tNFW kernels.  The series inside |x-1| < _X1_EPS were
# derived symbolically:
#   F(1+d)                = 1 - 2d/3 + 7d^2/15 - 12d^3/35
#   [(2+x^2)F-3]/(x^2-1)^2 = 4/15 - 16d/35 + 8d^2/15    (Hernquist Sigma)
#   2(1-F)/(x^2-1)         = 2/3 - 4d/5 + 26d^2/35      (Hernquist Sigma-bar)
#   (1-F)/(x^2-1)          = 1/3 - 2d/5 + 13d^2/35      (BMO f1)
_TJ_W = 1.0e-2
_TJ_D = 2.0e-2
_X1_EPS = 1.0e-2


def _quad_window(x, f_m, f_0, f_p):
    """Quadratic through (1-D, f_m), (1, f_0), (1+D, f_p), evaluated at x.

    The anchors sit at +-D in the well-conditioned region, so the interpolant
    *and its gradient* stay accurate where the closed forms cannot.
    """
    d = x - 1.0
    return (f_0 + (f_p - f_m) / (2.0 * _TJ_D) * d
            + (f_p - 2.0 * f_0 + f_m) / (2.0 * _TJ_D ** 2) * d ** 2)


# =========================================================================
# Takada & Jain (2003): sharply truncated NFW
# =========================================================================

def _tj_sigma_lo(x, c):
    """Sigma kernel, x < 1.  arccosh from d = u-1 as an exact product, so
    there is no cancellation approaching x = 1."""
    s = jnp.sqrt(c * c - x ** 2)
    om = 1.0 - x ** 2
    d = (1.0 - x) * (c - x) / (x * (1.0 + c))
    acosh_u = jnp.log1p(d + jnp.sqrt(d * (d + 2.0)))
    return -s / (om * (1.0 + c)) + acosh_u / (om * jnp.sqrt(om))


def _tj_sigma_hi(x, c):
    """Sigma kernel, 1 < x < c.  arccos(u) = 2 arcsin(sqrt(d/2)), stable at
    both ends."""
    s = jnp.sqrt(jnp.maximum(c * c - x ** 2, 1e-30))
    xm = x ** 2 - 1.0
    d = (x - 1.0) * (c - x) / (x * (1.0 + c))
    acos_u = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(0.5 * d, 0.0, 1.0)))
    return s / (xm * (1.0 + c)) - acos_u / (xm * jnp.sqrt(xm))


def _tj_sigma_dl(x, c):
    r"""Dimensionless :math:`f(x, c)`; :math:`\Sigma = 4\rho_s r_s f`.

    Requires :math:`c > 1 + 0.02` -- the truncation must lie outside the scale
    radius.
    """
    lo = x < 1.0 - _TJ_W
    hi = (x > 1.0 + _TJ_W) & (x < c)
    inside = x < c
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 0.5 * (1.0 + c))
    f_m = _tj_sigma_lo(1.0 - _TJ_D, c)
    f_p = _tj_sigma_hi(1.0 + _TJ_D, c)
    f_0 = jnp.sqrt(c * c - 1.0) * (1.0 + 1.0 / (1.0 + c)) / (3.0 * (1.0 + c))
    f = jnp.where(lo, _tj_sigma_lo(x_lo, c),
                  jnp.where(hi, _tj_sigma_hi(x_hi, c),
                            _quad_window(x, f_m, f_0, f_p)))
    return 0.5 * jnp.where(inside, f, 0.0)


def _tj_bsigma_lo(x, c):
    """Sigma-bar kernel, x < 1.

    A cancellation-free rewrite: the ~ln(x)/x^2 pieces are paired into a
    ``log1p`` of an O(x^2) argument, so small projected radii stay accurate.
    The textbook form loses all float32 precision below x ~ 3e-3.
    """
    s = jnp.sqrt(c * c - x ** 2)
    q = jnp.sqrt(1.0 - x ** 2)
    d = (1.0 - x) * (c - x) / (x * (1.0 + c))
    acosh_u = jnp.log1p(d + jnp.sqrt(d * (d + 2.0)))
    return (-1.0 / ((s + c) * (1.0 + c))
            + jnp.log1p(x ** 2 * (1.0 - s / (1.0 + q)) / (c + s)) / x ** 2
            + acosh_u / (q * (1.0 + q)))


def _tj_bsigma_hi(x, c):
    """Sigma-bar kernel, 1 < x < c."""
    s = jnp.sqrt(jnp.maximum(c * c - x ** 2, 1e-30))
    q = jnp.sqrt(x ** 2 - 1.0)
    d = (x - 1.0) * (c - x) / (x * (1.0 + c))
    acos_u = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(0.5 * d, 0.0, 1.0)))
    return (-1.0 / ((s + c) * (1.0 + c))
            + jnp.log(x * (1.0 + c) / (c + s)) / x ** 2
            + acos_u / (x ** 2 * q))


def _tj_bsigma_dl(x, c):
    r"""Dimensionless :math:`g(x, c)`; :math:`\bar\Sigma(<R) = 4\rho_s r_s g`."""
    lo = x < 1.0 - _TJ_W
    hi = (x > 1.0 + _TJ_W) & (x < c)
    out = x >= c
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 0.5 * (1.0 + c))
    x_out = jnp.where(x > 0, x, 1.0)
    g_m = _tj_bsigma_lo(1.0 - _TJ_D, c)
    g_p = _tj_bsigma_hi(1.0 + _TJ_D, c)
    sc1 = jnp.sqrt(c * c - 1.0)
    g_0 = (2.0 * sc1 - c) / (1.0 + c) + jnp.log((1.0 + c) / (c + sc1))
    # Beyond the truncation all the mass is enclosed: Sigma-bar = M_t/(pi R^2).
    g_out = g_nfw(c) / x_out ** 2
    return jnp.where(out, g_out,
                     jnp.where(lo, _tj_bsigma_lo(x_lo, c),
                               jnp.where(hi, _tj_bsigma_hi(x_hi, c),
                                         _quad_window(x, g_m, g_0, g_p))))


@jax.jit
def tnfw_rho(r, rho_s, r_s, c_t):
    r"""Truncated NFW density [Msun h^2/Mpc^3]; zero beyond :math:`c_t r_s`."""
    x = jnp.asarray(r) / r_s
    xs = jnp.where(x > 0, x, 1.0)
    return jnp.where(x < c_t, rho_s / (xs * (1.0 + xs) ** 2), 0.0)


@jax.jit
def tnfw_mass(r, rho_s, r_s, c_t):
    r"""Enclosed mass [Msun/h]; saturates at :math:`4\pi\rho_s r_s^3 g(c_t)`."""
    x = jnp.minimum(jnp.asarray(r) / r_s, c_t)
    return 4.0 * jnp.pi * rho_s * r_s ** 3 * g_nfw(x)


@jax.jit
def tnfw_sigma(R, rho_s, r_s, c_t):
    r""":math:`\Sigma(R)` [Msun h/Mpc^2]; identically zero beyond the truncation."""
    return 4.0 * rho_s * r_s * _tj_sigma_dl(jnp.asarray(R) / r_s, c_t)


@jax.jit
def tnfw_mean_sigma(R, rho_s, r_s, c_t):
    r""":math:`\bar\Sigma(<R)` [Msun h/Mpc^2]."""
    return 4.0 * rho_s * r_s * _tj_bsigma_dl(jnp.asarray(R) / r_s, c_t)


@jax.jit
def tnfw_delta_sigma(R, rho_s, r_s, c_t):
    r""":math:`\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)` [Msun h/Mpc^2]."""
    return tnfw_mean_sigma(R, rho_s, r_s, c_t) - tnfw_sigma(R, rho_s, r_s, c_t)


#: Fourier window of the sharply truncated NFW.
#:
#: An **alias**, not a reimplementation: the analytic NFW transform is already
#: the integral cut at :math:`r = c r_s`, so it *is* the Takada-Jain window with
#: the truncation concentration in place of the halo concentration.
tnfw_uk = nfw_uk


# =========================================================================
# Baltz, Marshall & Oguri (2009), n = 2
# =========================================================================

def _bmo_ff(x):
    """``(f1, F)`` with F the NFW projection kernel and ``f1 = (F-1)/(1-x^2)``."""
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)
    q_lo = jnp.sqrt(1.0 - x_lo ** 2)
    f1_lo = (2.0 * jnp.arctanh(jnp.sqrt((1.0 - x_lo) / (1.0 + x_lo))) / q_lo
             - 1.0) / (1.0 - x_lo ** 2)
    q_hi = jnp.sqrt(x_hi ** 2 - 1.0)
    f1_hi = (1.0 - 2.0 * arctan(jnp.sqrt((x_hi - 1.0) / (x_hi + 1.0))) / q_hi
             ) / (x_hi ** 2 - 1.0)
    d = x - 1.0
    f1_mid = 1.0 / 3.0 - 2.0 * d / 5.0 + 13.0 * d ** 2 / 35.0
    f1 = jnp.where(lo, f1_lo, jnp.where(hi, f1_hi, f1_mid))
    return f1, f1 * (1.0 - x ** 2) + 1.0


def _bmo_L(x, tau):
    r""":math:`L(x,\tau) = \ln[x/(\sqrt{x^2+\tau^2}+\tau)]`."""
    xs = jnp.where(x > 0, x, 1.0)
    return jnp.log(xs / (jnp.sqrt(xs ** 2 + tau ** 2) + tau))


def _bmo_sigma_dl(x, tau):
    """Dimensionless BMO surface density (BMO09 Eq. A.28)."""
    ff1, ff2 = _bmo_ff(x)
    t2 = tau * tau
    tx2 = t2 + x ** 2
    pre = t2 ** 2 / (4.0 * (t2 + 1.0) ** 3)
    return pre * (2.0 * (t2 + 1.0) * ff1 + 8.0 * ff2
                  + (t2 ** 2 - 1.0) / (t2 * tx2)
                  - jnp.pi * (4.0 * tx2 + t2 + 1.0) / (tx2 * jnp.sqrt(tx2))
                  + (t2 * (t2 ** 2 - 1.0) + tx2 * (3.0 * t2 ** 2 - 6.0 * t2 - 1.0))
                  * _bmo_L(x, tau) / (tau ** 3 * tx2 * jnp.sqrt(tx2)))


def _bmo_bsigma_dl(x, tau):
    """Dimensionless BMO mean surface density.

    Cancels ~ln(x)/x^2 terms; accuracy degrades below x ~ 0.05 in float32.
    """
    _, ff2 = _bmo_ff(x)
    t2 = tau * tau
    xs = jnp.where(x > 0, x, 1.0)
    tx2 = t2 + xs ** 2
    pre = t2 ** 2 / (2.0 * (t2 + 1.0) ** 3 * xs ** 2)
    return pre * (2.0 * (t2 + 4.0 * xs ** 2 - 3.0) * ff2
                  + (jnp.pi * (3.0 * t2 - 1.0)
                     + 2.0 * tau * (t2 - 3.0) * jnp.log(tau)) / tau
                  + (-(tau ** 3) * jnp.pi * (4.0 * xs ** 2 + 3.0 * t2 - 1.0)
                     + (2.0 * t2 ** 2 * (t2 - 3.0)
                        + xs ** 2 * (3.0 * t2 ** 2 - 6.0 * t2 - 1.0))
                     * _bmo_L(xs, tau)) / (tau ** 3 * jnp.sqrt(tx2)))


def _m_bmo_dl(x, tau):
    """Dimensionless BMO enclosed mass."""
    t2 = tau * tau
    xs = jnp.where(x > 0, x, 1.0)
    pre = t2 / (2.0 * (t2 + 1.0) ** 3 * (1.0 + x) * (t2 + x ** 2))
    return pre * ((t2 + 1.0) * x * (x * (x + 1.0)
                                    - t2 * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * t2 ** 2)
                  + tau * (x + 1.0) * (t2 + x ** 2)
                  * (2.0 * (3.0 * t2 - 1.0) * arctan(xs / tau)
                     + tau * (t2 - 3.0)
                     * jnp.log(t2 * (1.0 + xs) ** 2 / (t2 + xs ** 2))))


def _m_bmo_tot_dl(tau):
    """Dimensionless BMO total mass -- finite, unlike NFW."""
    t2 = tau * tau
    return t2 / (2.0 * (t2 + 1.0) ** 3) * (
        (3.0 * t2 - 1.0) * (jnp.pi * tau - t2 - 1.0)
        + 2.0 * t2 * (t2 - 3.0) * jnp.log(tau))


def _pq_hyperbolic(x):
    r""":math:`P = \sinh x\,\mathrm{Chi}(x) - \cosh x\,\mathrm{Shi}(x)` and
    :math:`Q = \cosh x\,\mathrm{Chi}(x) - \sinh x\,\mathrm{Shi}(x)`.

    Computed cancellation-free through :math:`E_1` and :math:`\mathrm{Ei}`,
    with a scaled asymptotic series above :math:`x = 30`.  Valid for
    :math:`x > 0`.
    """
    xs = jnp.where(x > 0, x, 1.0)
    small = xs < 30.0
    x_sm = jnp.where(small, xs, 1.0)
    e1s = jnp.exp(x_sm) * exp1(x_sm)
    eis = jnp.exp(-x_sm) * expi(x_sm)
    p_sm = -0.5 * (e1s + eis)
    q_sm = 0.5 * (eis - e1s)
    xi = 1.0 / xs
    xi2 = xi * xi
    p_lg = -xi * (1.0 + xi2 * (2.0 + xi2 * (24.0 + xi2 * 720.0)))
    q_lg = xi2 * (1.0 + xi2 * (6.0 + xi2 * (120.0 + xi2 * 5040.0)))
    return jnp.where(small, p_sm, p_lg), jnp.where(small, q_sm, q_lg)


@jax.jit
def bmo_rho(r, rho_s, r_s, tau):
    r""":math:`\rho_{\rm NFW}(r)\,[\tau^2/(\tau^2+x^2)]^2` [Msun h^2/Mpc^3]."""
    x = jnp.asarray(r) / r_s
    xs = jnp.where(x > 0, x, 1.0)
    return (rho_s / (xs * (1.0 + xs) ** 2)) * (tau ** 2 / (tau ** 2 + xs ** 2)) ** 2


@jax.jit
def bmo_mass(r, rho_s, r_s, tau):
    r"""Enclosed mass [Msun/h] (BMO09 Eq. A.2)."""
    return 4.0 * jnp.pi * rho_s * r_s ** 3 * _m_bmo_dl(jnp.asarray(r) / r_s, tau)


@jax.jit
def bmo_mass_total(rho_s, r_s, tau):
    r"""Total BMO mass [Msun/h] -- finite.

    For :math:`(\rho_s, r_s)` matched to an NFW halo of mass :math:`M_\Delta`,
    :math:`M_{\rm tot}/M_\Delta = m_{\rm tot}(\tau)/g(c) \approx 1.34` at
    :math:`\tau = 2.5c`, :math:`c = 6`.  That is why :func:`bmo_uk` must choose
    which mass it normalises by, and says so.
    """
    return 4.0 * jnp.pi * rho_s * r_s ** 3 * _m_bmo_tot_dl(tau)


@jax.jit
def bmo_sigma(R, rho_s, r_s, tau):
    r""":math:`\Sigma(R)` [Msun h/Mpc^2]."""
    return 4.0 * rho_s * r_s * _bmo_sigma_dl(jnp.asarray(R) / r_s, tau)


@jax.jit
def bmo_mean_sigma(R, rho_s, r_s, tau):
    r""":math:`\bar\Sigma(<R)` [Msun h/Mpc^2]."""
    return 4.0 * rho_s * r_s * _bmo_bsigma_dl(jnp.asarray(R) / r_s, tau)


@jax.jit
def bmo_delta_sigma(R, rho_s, r_s, tau):
    r""":math:`\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)` [Msun h/Mpc^2]."""
    return bmo_mean_sigma(R, rho_s, r_s, tau) - bmo_sigma(R, rho_s, r_s, tau)


@jax.jit
def bmo_uk(k, r_s, tau):
    r"""Analytic BMO Fourier window, normalised to the **total** BMO mass.

    So :math:`u(k\to0) = 1`, matching every transform in
    :mod:`ggah_mod.halos.profiles`.  The reference implementation normalises by
    :math:`M_\Delta` instead -- a ~34% difference at typical truncations.

    Returns (Nk, NM).  Guarded below :math:`kr_s = 10^{-4}`, where the log terms
    cancel; use x64 below :math:`kr_s = 0.01`.
    """
    k = jnp.atleast_1d(jnp.asarray(k)).reshape(-1, 1)
    r_s = jnp.atleast_1d(jnp.asarray(r_s)).reshape(1, -1)
    tau = jnp.atleast_1d(jnp.asarray(tau)).reshape(1, -1)
    K = k * r_s
    # The safe threshold matches the output guard, so no gradient ever flows
    # through the cancellation-prone small-K expression.
    Ks = jnp.where(K >= 1e-4, K, 1.0)
    t2 = tau * tau
    s_i, c_i = si(Ks), ci(Ks)
    p, q = _pq_hyperbolic(tau * Ks)
    sK, cK = jnp.sin(Ks), jnp.cos(Ks)
    f2 = (2.0 * (3.0 * t2 ** 2 - 6.0 * t2 - 1.0) * p
          - 2.0 * tau * (t2 ** 2 - 1.0) * Ks * q
          - 2.0 * t2 * jnp.pi * jnp.exp(-tau * Ks) * ((t2 + 1.0) * Ks + 4.0 * tau)
          + 2.0 * tau ** 3 * (jnp.pi - 2.0 * s_i) * (4.0 * cK + (t2 + 1.0) * Ks * sK)
          + 4.0 * tau ** 3 * c_i * (4.0 * sK - (t2 + 1.0) * Ks * cK))
    uk = tau / (4.0 * _m_bmo_tot_dl(tau) * (1.0 + t2) ** 3 * Ks) * f2
    return jnp.where(K < 1e-4, 1.0, uk)


# =========================================================================
# Hernquist (1990)
# =========================================================================

#: :math:`r_b/R_e` for a Hernquist profile matched to a de Vaucouleurs
#: effective radius.  Pass ``r_b = HERNQUIST_RB_RE * r_e`` when starting from a
#: measured size.
HERNQUIST_RB_RE = 0.551


def _hern_sigma_dl(x):
    """Dimensionless Hernquist Sigma kernel (Hernquist 1990 Eq. 32)."""
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)
    a_lo = jnp.sqrt(1.0 - x_lo ** 2)
    f_lo = ((2.0 + x_lo ** 2) * jnp.arctanh(a_lo) / a_lo - 3.0) / (x_lo ** 2 - 1.0) ** 2
    a_hi = jnp.sqrt(x_hi ** 2 - 1.0)
    f_hi = ((2.0 + x_hi ** 2) * arctan(a_hi) / a_hi - 3.0) / (x_hi ** 2 - 1.0) ** 2
    d = x - 1.0
    f_mid = 4.0 / 15.0 - 16.0 * d / 35.0 + 8.0 * d ** 2 / 15.0
    return jnp.where(lo, f_lo, jnp.where(hi, f_hi, f_mid))


def _hern_bsigma_dl(x):
    """Dimensionless Hernquist Sigma-bar kernel; ``-> 2/x^2`` as x -> inf."""
    lo = x < 1.0 - _X1_EPS
    hi = x > 1.0 + _X1_EPS
    x_lo = jnp.where(lo, x, 0.5)
    x_hi = jnp.where(hi, x, 2.0)
    a_lo = jnp.sqrt(1.0 - x_lo ** 2)
    g_lo = 2.0 * (1.0 - jnp.arctanh(a_lo) / a_lo) / (x_lo ** 2 - 1.0)
    a_hi = jnp.sqrt(x_hi ** 2 - 1.0)
    g_hi = 2.0 * (1.0 - arctan(a_hi) / a_hi) / (x_hi ** 2 - 1.0)
    d = x - 1.0
    g_mid = 2.0 / 3.0 - 4.0 * d / 5.0 + 26.0 * d ** 2 / 35.0
    return jnp.where(lo, g_lo, jnp.where(hi, g_hi, g_mid))


@jax.jit
def hernquist_rho(r, m_tot, r_b):
    r""":math:`\rho = M r_b/[2\pi r(r+r_b)^3]` [Msun h^2/Mpc^3]."""
    rs = jnp.where(jnp.asarray(r) > 0, jnp.asarray(r), r_b)
    return m_tot * r_b / (2.0 * jnp.pi * rs * (rs + r_b) ** 3)


@jax.jit
def hernquist_mass(r, m_tot, r_b):
    r""":math:`M(<r) = M x^2/(1+x)^2`, :math:`x = r/r_b` [Msun/h]."""
    x = jnp.asarray(r) / r_b
    return m_tot * x ** 2 / (1.0 + x) ** 2


@jax.jit
def hernquist_sigma(R, m_tot, r_b):
    r""":math:`\Sigma(R)` [Msun h/Mpc^2]."""
    return m_tot / (2.0 * jnp.pi * r_b ** 2) * _hern_sigma_dl(jnp.asarray(R) / r_b)


@jax.jit
def hernquist_mean_sigma(R, m_tot, r_b):
    r""":math:`\bar\Sigma(<R)` [Msun h/Mpc^2]; :math:`\pi R^2\bar\Sigma \to M`."""
    return m_tot / (2.0 * jnp.pi * r_b ** 2) * _hern_bsigma_dl(jnp.asarray(R) / r_b)


@jax.jit
def hernquist_delta_sigma(R, m_tot, r_b):
    r""":math:`\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)` [Msun h/Mpc^2]."""
    return hernquist_mean_sigma(R, m_tot, r_b) - hernquist_sigma(R, m_tot, r_b)
