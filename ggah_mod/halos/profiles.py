r"""Halo density profiles: :math:`\rho(r)`, :math:`u(k|M)`, :math:`\Sigma(R)`.

The halo model needs the normalised Fourier transform :math:`u(k|M)` of the
radial profile.  This module provides it, with the density and the projected
quantities where they are analytic, for three families:

* *NFW* (Navarro, Frenk & White 1997), the default, with an analytic Fourier
  transform and analytic projected quantities;
* *Einasto* (Einasto 1965), with a shape parameter :math:`\alpha` instead of a
  fixed inner slope, Fourier-transformed numerically;
* *gNFW*, generalised NFW with free inner, transition and outer slopes, the one
  definition of the shape the hot-gas sector needs.

:func:`satellite_uk` (NFW with three shape freedoms) and :func:`ejected_uk` (a
Gaussian for the ejected gas) build on them, and :func:`profile_uk_gl` is the
Gauss-Legendre transform any other profile takes its :math:`u(k|M)` from.

Conventions for a profile
-------------------------

Radii and :math:`r_s` are *comoving* Mpc/h, masses M_sun/h, wavenumbers h/Mpc,
and :math:`\rho_s` M_sun h^2/Mpc^3.  Every Fourier transform is normalised so
that :math:`u(k \to 0) = 1`, which makes mass conservation checkable; the matter
tracer built on these profiles depends on it exactly.

Every profile has one implementation, pure JAX and differentiable, and
``tests/test_paths.py`` differentiates all of them with respect to
:math:`\Omega_m`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

from ..numerics import arctan

__all__ = [
    "g_nfw", "si", "ci",
    "nfw_rho", "nfw_mass", "nfw_uk", "nfw_sigma", "nfw_mean_sigma",
    "nfw_delta_sigma", "nfw_scale_density",
    "einasto_rho", "einasto_uk",
    "gnfw_shape", "gnfw_rho", "gnfw_uk",
    "satellite_uk", "profile_uk_gl", "ejected_uk", "ejected_mass_within",
]


# =========================================================================
# Shared helpers
# =========================================================================

@jax.jit
def g_nfw(y):
    r""":math:`g(y) = \ln(1+y) - y/(1+y)`.

    The NFW mass profile shape: :math:`M(<y\,r_s) = 4\pi\rho_s r_s^3 g(y)`, so
    :math:`g(c)` is the normalisation of every NFW quantity.  ``log1p`` rather
    than ``log(1+y)`` because the two differ in the last digits at small ``y``,
    which is where the difference with the second term is a cancellation.
    """
    y = jnp.asarray(y)
    return jnp.log1p(y) - y / (1.0 + y)


# Gauss-Legendre nodes for the sine/cosine integrals, built once with numpy so
# importing this module does not initialise a JAX backend.
_N_GL_SICI = 64
_GL_T_SICI_NP, _GL_W_SICI_NP = np.polynomial.legendre.leggauss(_N_GL_SICI)
_GL_T_SICI = jnp.asarray(0.5 * (_GL_T_SICI_NP + 1.0))
_GL_W_SICI = jnp.asarray(0.5 * _GL_W_SICI_NP)
_EULER_GAMMA = 0.5772156649015328606

#: Where :func:`si`/:func:`ci` switch from quadrature to the asymptotic series.
#:
#: Chosen by measurement, not habit.  The 64-node Gauss-Legendre branch is good
#: to 3e-14 all the way out to x = 40 -- the integrand's oscillation is not the
#: limiting factor -- while the asymptotic series is only marginally converged
#: at small x, because it is asymptotic and its optimal truncation is around
#: n ~ x terms.  Switching at 12 (the predecessor's choice, with four terms) put
#: the worst error at x ~ 12.7 and cost **1.9e-4** in Ci.  Switching at 40 with
#: six terms costs 1e-12.
_SICI_SWITCH = 40.0

#: (2n)! and (2n+1)! for the asymptotic auxiliary functions f and g.
_ASYM_F_FAC = (1.0, 2.0, 24.0, 720.0, 40320.0, 3628800.0)
_ASYM_G_FAC = (1.0, 6.0, 120.0, 5040.0, 362880.0, 39916800.0)


def _asym_fg(u):
    """Auxiliary functions f(u), g(u) of the Si/Ci asymptotic expansion."""
    inv2 = 1.0 / u ** 2
    f = sum((-1) ** i * _ASYM_F_FAC[i] * inv2 ** i for i in range(len(_ASYM_F_FAC))) / u
    g = sum((-1) ** i * _ASYM_G_FAC[i] * inv2 ** i for i in range(len(_ASYM_G_FAC))) * inv2
    return f, g


@jax.jit
def si(x):
    r"""Sine integral :math:`\mathrm{Si}(x) = \int_0^x \sin t/t\,dt`.

    64-point Gauss-Legendre below :data:`_SICI_SWITCH`, six-term asymptotic
    above.  A JAX reimplementation of ``scipy.special.sici`` so the NFW Fourier
    transform can be differentiated; scipy's is a compiled kernel with no JVP.

    Agrees with ``scipy.special.sici`` to ~1e-12 absolute over
    :math:`x \in [10^{-3}, 10^3]`.
    """
    x = jnp.asarray(x)
    ax = jnp.abs(x)
    # Quadrature branch: integrand sin(t)/t, series-safe at t = 0.
    t = ax[..., None] * _GL_T_SICI
    integ = jnp.where(t < 1e-8, 1.0 - t ** 2 / 6.0, jnp.sin(t) / jnp.where(t == 0, 1.0, t))
    quad = ax * jnp.sum(integ * _GL_W_SICI, axis=-1)
    # Asymptotic branch, on a guarded copy so the unused side stays finite.
    u = jnp.where(ax < 1e-30, 1.0, ax)
    f, g = _asym_fg(u)
    asym = 0.5 * jnp.pi - f * jnp.cos(u) - g * jnp.sin(u)
    return jnp.sign(x) * jnp.where(ax < _SICI_SWITCH, quad, asym)


@jax.jit
def ci(x):
    r"""Cosine integral :math:`\mathrm{Ci}(x) = \gamma + \ln x + \int_0^x (\cos t - 1)/t\,dt`.

    Defined only for :math:`x > 0`, and NaN for :math:`x \le 0`: the branch
    selector tests ``x``, not its guarded copy, which would return
    :math:`\mathrm{Ci}(1)` there.

    Agrees with ``scipy.special.sici`` to ~1e-12 absolute for :math:`x > 0`.
    Note the *relative* error is unbounded near :math:`x \approx 0.616`, where
    :math:`\mathrm{Ci}` crosses zero -- compare absolute error there.
    """
    x = jnp.asarray(x)
    pos = x > 0.0
    xs = jnp.where(pos, x, 1.0)
    t = xs[..., None] * _GL_T_SICI
    integ = jnp.where(t < 1e-8, -t / 2.0, (jnp.cos(t) - 1.0) / jnp.where(t == 0, 1.0, t))
    quad = _EULER_GAMMA + jnp.log(xs) + xs * jnp.sum(integ * _GL_W_SICI, axis=-1)
    f, g = _asym_fg(xs)
    asym = f * jnp.sin(xs) - g * jnp.cos(xs)
    out = jnp.where(xs < _SICI_SWITCH, quad, asym)
    return jnp.where(pos, out, jnp.nan)


def profile_uk_gl(k, r_max, integrand_fn, n_gl: int = 200):
    r"""Normalised Fourier transform of a radial profile, by Gauss-Legendre.

    .. math::

        u(k|M) = \frac{\int_0^{r_{\max}} f(r|M)\,j_0(kr)\,r^2\,dr}
                      {\int_0^{r_{\max}} f(r|M)\,r^2\,dr}

    so :math:`u(k\to0) = 1` by construction rather than by a guard.

    Parameters
    ----------
    k : array, shape (Nk,) [h/Mpc]
    r_max : array, shape (NM,) [Mpc/h]
        Outer truncation radius per halo.  May be traced.
    integrand_fn : callable
        ``f(r_nodes) -> (NM, n_gl)``; the unnormalised radial profile evaluated
        on the quadrature grid.  Must itself be traceable.
    n_gl : int
        Node count.  Static -- it sets an array shape.

    Returns
    -------
    array, shape (Nk, NM)

    Notes
    -----
    ``jnp.sinc(x) = sin(pi x)/(pi x)``, so :math:`j_0(kr)` is ``sinc(kr/pi)``,
    which is branchless and exact at :math:`kr = 0` -- no special case for the
    large-scale limit.
    """
    x_gl, w_gl = _leggauss_cached(int(n_gl))
    k = jnp.asarray(k)
    r_max = jnp.atleast_1d(jnp.asarray(r_max))
    half = 0.5 * r_max                                   # (NM,)
    r_nodes = half[:, None] * (jnp.asarray(x_gl) + 1.0)[None, :]      # (NM, n_gl)
    f = integrand_fn(r_nodes)                                          # (NM, n_gl)
    w = half[:, None] * jnp.asarray(w_gl)[None, :] * f * r_nodes ** 2  # (NM, n_gl)
    j0 = jnp.sinc(k[:, None, None] * r_nodes[None, :, :] / jnp.pi)     # (Nk,NM,n_gl)
    num = jnp.einsum("kmg,mg->km", j0, w)
    den = jnp.sum(w, axis=-1)[None, :]
    return num / den


@functools.lru_cache(maxsize=8)
def _leggauss_cached(n: int):
    """Gauss-Legendre nodes on [-1, 1], memoised. numpy, built once."""
    return np.polynomial.legendre.leggauss(n)


# =========================================================================
# NFW
# =========================================================================

@jax.jit
def nfw_scale_density(m, c, r_delta):
    r""":math:`(\rho_s, r_s)` for an NFW halo of mass ``m`` within ``r_delta``.

    :math:`r_s = r_\Delta/c` and :math:`\rho_s = M/(4\pi r_s^3 g(c))`.
    """
    r_s = jnp.asarray(r_delta) / jnp.asarray(c)
    rho_s = jnp.asarray(m) / (4.0 * jnp.pi * r_s ** 3 * g_nfw(c))
    return rho_s, r_s


@jax.jit
def nfw_rho(r, rho_s, r_s):
    r""":math:`\rho(r) = \rho_s/[(r/r_s)(1+r/r_s)^2]` [M_sun h^2/Mpc^3]."""
    x = jnp.asarray(r) / r_s
    return rho_s / (x * (1.0 + x) ** 2)


@jax.jit
def nfw_mass(r, rho_s, r_s):
    r""":math:`M(<r) = 4\pi\rho_s r_s^3 g(r/r_s)` [M_sun/h]."""
    return 4.0 * jnp.pi * rho_s * r_s ** 3 * g_nfw(jnp.asarray(r) / r_s)


@jax.jit
def nfw_uk(k, r_s, c):
    r"""Analytic :math:`u(k|M)` for an NFW halo truncated at :math:`c\,r_s`.

    .. math::

        u(k) = \frac{1}{g(c)}\Big\{
            \sin(\kappa)[\mathrm{Si}((1+c)\kappa) - \mathrm{Si}(\kappa)]
          + \cos(\kappa)[\mathrm{Ci}((1+c)\kappa) - \mathrm{Ci}(\kappa)]
          - \frac{\sin(c\kappa)}{(1+c)\kappa}\Big\},\quad \kappa = k r_s

    Parameters
    ----------
    k : array (Nk,) [h/Mpc]
    r_s : array (NM,) [Mpc/h]
    c : array (NM,)

    Returns
    -------
    array (Nk, NM), dimensionless, ``u(k->0) = 1``.
    """
    k = jnp.atleast_1d(jnp.asarray(k))[:, None]
    r_s = jnp.atleast_1d(jnp.asarray(r_s))[None, :]
    c = jnp.atleast_1d(jnp.asarray(c))[None, :]
    kappa = k * r_s
    # kappa -> 0 is a removable 0/0; evaluate on a safe copy and select after.
    ks = jnp.where(kappa < 1e-8, 1.0, kappa)
    s_lo, s_hi = si(ks), si((1.0 + c) * ks)
    c_lo, c_hi = ci(ks), ci((1.0 + c) * ks)
    uk = (jnp.sin(ks) * (s_hi - s_lo)
          + jnp.cos(ks) * (c_hi - c_lo)
          - jnp.sin(c * ks) / ((1.0 + c) * ks)) / g_nfw(c)
    return jnp.where(kappa < 1e-8, 1.0, uk)


def _nfw_f_of_x(x):
    r"""The :math:`x`-dependent factor of the NFW projected profile.

    :math:`\Sigma(R) = 2\rho_s r_s f(x)`, :math:`x = R/r_s`, with the three
    branches :math:`x<1`, :math:`x=1`, :math:`x>1` written so that both unused
    branches stay finite: ``jnp.where`` evaluates all of them, and a NaN in a
    discarded branch still poisons the gradient.
    """
    x = jnp.asarray(x)
    lo = jnp.where(x < 1.0, x, 0.5)
    hi = jnp.where(x > 1.0, x, 2.0)
    f_lo = (1.0 - 2.0 / jnp.sqrt(1.0 - lo ** 2)
            * jnp.arctanh(jnp.sqrt((1.0 - lo) / (1.0 + lo)))) / (lo ** 2 - 1.0)
    f_hi = (1.0 - 2.0 / jnp.sqrt(hi ** 2 - 1.0)
            * arctan(jnp.sqrt((hi - 1.0) / (1.0 + hi)))) / (hi ** 2 - 1.0)
    return jnp.where(x < 1.0, f_lo, jnp.where(x > 1.0, f_hi, 1.0 / 3.0))


@jax.jit
def nfw_sigma(R, rho_s, r_s):
    r"""Projected surface density :math:`\Sigma(R)` [M_sun h/Mpc^2]."""
    return 2.0 * rho_s * r_s * _nfw_f_of_x(jnp.asarray(R) / r_s)


def _nfw_h_of_x(x):
    r"""Mean-interior factor: :math:`\bar\Sigma(<R) = 4\rho_s r_s h(x)/x^2`."""
    x = jnp.asarray(x)
    lo = jnp.where(x < 1.0, x, 0.5)
    hi = jnp.where(x > 1.0, x, 2.0)
    h_lo = jnp.log(lo / 2.0) + jnp.arccosh(1.0 / lo) / jnp.sqrt(1.0 - lo ** 2)
    h_hi = jnp.log(hi / 2.0) + jnp.arccos(1.0 / hi) / jnp.sqrt(hi ** 2 - 1.0)
    return jnp.where(x < 1.0, h_lo, jnp.where(x > 1.0, h_hi, 1.0 + jnp.log(0.5)))


@jax.jit
def nfw_mean_sigma(R, rho_s, r_s):
    r"""Mean interior surface density :math:`\bar\Sigma(<R)` [M_sun h/Mpc^2]."""
    x = jnp.asarray(R) / r_s
    return 4.0 * rho_s * r_s * _nfw_h_of_x(x) / x ** 2


@jax.jit
def nfw_delta_sigma(R, rho_s, r_s):
    r""":math:`\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)` [M_sun h/Mpc^2]."""
    return nfw_mean_sigma(R, rho_s, r_s) - nfw_sigma(R, rho_s, r_s)


# =========================================================================
# Einasto
# =========================================================================

@jax.jit
def einasto_rho(r, rho_s, r_s, alpha=0.18):
    r""":math:`\rho(r) = \rho_s\exp\{-\tfrac{2}{\alpha}[(r/r_s)^\alpha - 1]\}`."""
    x = jnp.asarray(r) / r_s
    return rho_s * jnp.exp(-2.0 / alpha * (x ** alpha - 1.0))


def einasto_uk(k, r_s, c, alpha=0.18, n_gl: int = 200):
    r""":math:`u(k|M)` for an Einasto halo truncated at :math:`c\,r_s`.

    No closed form, so Gauss-Legendre in radius (:func:`profile_uk_gl`), in pure
    JAX, so the gradient reaches the halo radius and the concentration.  ``k``
    [h/Mpc]; ``r_s`` [comoving Mpc/h] and ``c`` per halo; ``alpha`` the Einasto
    shape (default 0.18).

    Returns (Nk, NM), ``u(k->0) = 1``.
    """
    r_s = jnp.atleast_1d(jnp.asarray(r_s))
    c = jnp.atleast_1d(jnp.asarray(c))
    r_max = c * r_s
    shape = lambda r: einasto_rho(r, 1.0, r_s[:, None], alpha)
    return profile_uk_gl(k, r_max, shape, n_gl=n_gl)


# =========================================================================
# Generalised NFW
# =========================================================================

@jax.jit
def gnfw_shape(x, alpha_in, alpha_tr, alpha_out):
    r"""Dimensionless gNFW shape.

    .. math::

        f(x) = x^{-\alpha_{\rm in}}\,
               \left(1 + x^{\alpha_{\rm tr}}\right)^{
                   (\alpha_{\rm in}-\alpha_{\rm out})/\alpha_{\rm tr}}

    Inner slope :math:`-\alpha_{\rm in}`, outer slope :math:`-\alpha_{\rm out}`,
    transition sharpness :math:`\alpha_{\rm tr}`.  NFW is
    :math:`(1, 1, 3)`.
    """
    x = jnp.maximum(jnp.asarray(x), 1e-10)
    return x ** (-alpha_in) * (1.0 + x ** alpha_tr) ** ((alpha_in - alpha_out) / alpha_tr)


@jax.jit
def gnfw_rho(r, rho_s, r_s, alpha_in=1.0, alpha_tr=1.0, alpha_out=3.0):
    r""":math:`\rho(r) = \rho_s f(r/r_s)` with the gNFW shape."""
    return rho_s * gnfw_shape(jnp.asarray(r) / r_s, alpha_in, alpha_tr, alpha_out)


def gnfw_uk(k, r_s, r_max, alpha_in=1.0, alpha_tr=1.0, alpha_out=3.0,
            n_gl: int = 200):
    r""":math:`u(k|M)` for a gNFW profile truncated at ``r_max``.

    ``r_max`` is given directly rather than as :math:`c\,r_s`, because a gas
    profile's truncation radius is not tied to a concentration.  Returns
    (Nk, NM), ``u(k->0) = 1``.
    """
    r_s = jnp.atleast_1d(jnp.asarray(r_s))
    shape = lambda r: gnfw_shape(r / r_s[:, None], alpha_in, alpha_tr, alpha_out)
    return profile_uk_gl(k, jnp.atleast_1d(jnp.asarray(r_max)), shape, n_gl=n_gl)


# =========================================================================
# Satellites
# =========================================================================

def satellite_uk(k, r_delta, c, *, b_sat_conc=1.0, f_cut_inv=0.0,
                 gamma_inner=0.0, n_gl: int = 200):
    r""":math:`u(k|M)` for satellites, with three shape freedoms.

    Satellites need not trace the dark matter exactly.  Three deviations, each
    zero-neutral: at the defaults this is the NFW result exactly, and the
    approach to it is continuous and differentiable rather than a special case.
    ``k`` is in h/Mpc, ``r_delta`` in comoving Mpc/h, and ``c`` is the halo
    concentration.

    ``b_sat_conc``
        multiplies the concentration, :math:`c_{\rm sat} = b\,c`.  Values below
        1 make the satellite distribution shallower than the mass.
    ``f_cut_inv``
        *inverse* outer truncation scale, in units of :math:`r_\Delta^{-1}`:
        the profile is multiplied by
        :math:`\exp(-f_{\rm cut}^{-1}\,r/r_\Delta)`, so the truncation radius
        is :math:`r_{\rm cut} = r_\Delta / f_{\rm cut}^{-1}` and 0 means no
        truncation.
    ``gamma_inner``
        extra inner power :math:`(r/r_s)^{-\gamma}` steepening or flattening the
        core.

    .. rubric:: The inverted truncation parameter

    The truncation is carried as its inverse so that the default, no
    truncation, is an interior point of the parameter.  In the truncation
    radius ``f_cut``, a fraction of :math:`r_\Delta`, no truncation is the
    limit :math:`f_{\rm cut}\to\infty`, while :math:`f_{\rm cut}\to0^+` kills
    everything but the origin, so :math:`u(k)\to1` at every :math:`k`, a point
    mass: at :math:`c=5`, :math:`u(k{=}10)` is :math:`0.157` without truncation
    and :math:`1.000` at ``f_cut = 1e-6``.  With ``f_cut_inv`` there is no
    branch, and the derivative at the default is an ordinary finite number.
    Convert a value of the earlier parameter with
    :math:`f_{\rm cut}^{-1} = 1/f_{\rm cut}`; ``f_cut = 0``, which meant no
    truncation, becomes ``f_cut_inv = 0``.

    The three shape parameters are keyword-only, so a positional call written
    against the earlier signature fails instead of reading a truncation radius
    as its reciprocal.  All three are traceable.

    Returns (Nk, NM), ``u(k->0) = 1``.
    """
    r_delta = jnp.atleast_1d(jnp.asarray(r_delta))
    c_sat = jnp.atleast_1d(jnp.asarray(c)) * b_sat_conc
    r_s = (r_delta / c_sat)[:, None]

    def shape(r):
        x = r / r_s
        base = 1.0 / (x * (1.0 + x) ** 2)
        # Branchless and zero-neutral: gamma_inner = 0 gives x**0 = 1 and
        # f_cut_inv = 0 gives exp(0) = 1, so the NFW case is recovered exactly
        # by arithmetic rather than by a `where` whose unselected branch would
        # still be differentiated.
        inner = x ** (-gamma_inner)
        cut = jnp.exp(-f_cut_inv * r / r_delta[:, None])
        return base * inner * cut

    return profile_uk_gl(k, r_delta, shape, n_gl=n_gl)


# =========================================================================
# The gas that left
# =========================================================================

@jax.jit
def ejected_uk(k, r_delta, eta_ej=2.0):
    r"""Ejected gas: a Gaussian shell, :math:`\tilde u = e^{-k^2r_{\rm ej}^2/2}`.

    .. math::

        \rho_{\rm ej}(r) \propto \frac{e^{-r^2/2r_{\rm ej}^2}}
                                     {(2\pi r_{\rm ej}^2)^{3/2}},
        \qquad r_{\rm ej} = \eta_{\rm ej}\,R_\Delta

    Schneider & Teyssier (2015), the baryonification ejected component.  The
    baryons a halo lost are neither in it nor uniformly spread: they sit in a
    puffed-out envelope at a few times the halo radius, and giving them the
    dark-matter profile in :mod:`~ggah_mod.sectors.matter` would put them where
    the lensing signal is largest instead of where they are.

    We chose it over the alternatives for three properties, in this order:

    * :math:`u(k\to0) = 1` exactly, in closed form.  Every other component of
      the matter budget has that by construction, and an ejected term that only
      approached it numerically would put a floor under the module's
      mass-conservation residual, currently :math:`8.5\times10^{-9}` and set by
      the NFW transform rather than by anything here.
    * no special function on the traced path, so no quadrature and no table:
      it is an ``exp`` of a product, differentiable in :math:`\eta_{\rm ej}`
      to machine precision.
    * one parameter.  A profile with a shape freedom would be fitted, and there
      is nothing at these radii to fit it to yet.

    Parameters
    ----------
    k : array (Nk,)
        Wavenumber [h/Mpc].
    r_delta : array (NM,)
        Halo radius [comoving Mpc/h], from
        :attr:`~ggah_mod.halos.field.HaloField.r_delta`.
    eta_ej : float
        :math:`r_{\rm ej}/R_\Delta`.  Traced.

    Returns
    -------
    array (Nk, NM)
        With ``u(k->0) = 1``.
    """
    k = jnp.atleast_1d(jnp.asarray(k))[:, None]
    r_ej = eta_ej * jnp.atleast_1d(jnp.asarray(r_delta))[None, :]
    return jnp.exp(-0.5 * (k * r_ej) ** 2)


def ejected_mass_within(r, r_delta, eta_ej=2.0):
    r"""The fraction of the ejected gas inside radius ``r``.

    .. math::

        \frac{M_{\rm ej}(<r)}{M_{\rm ej}} = {\rm erf}\!\left(\frac{x}{\sqrt2}\right)
            - \sqrt{\frac{2}{\pi}}\,x\,e^{-x^2/2},
        \qquad x = \frac{r}{\eta_{\rm ej}R_\Delta}

    The enclosed mass of the Gaussian whose transform is :func:`ejected_uk`,
    so it tends to that transform's :math:`k\to0` value, one, at large ``r``.
    It sets the radius at which the baryon budget closes: the census integrates
    masses over :math:`M` and has no radius, and every other component closes
    within :math:`R_\Delta` by its own normalisation.  90 per cent of the
    ejected gas lies within :math:`2.50\,\eta_{\rm ej}R_\Delta` and 99 per cent
    within
    :math:`3.37\,\eta_{\rm ej}R_\Delta`; at the default
    :math:`\eta_{\rm ej} = 2`, :math:`R_\Delta` itself holds 3 per cent.

    Parameters
    ----------
    r : array
        Radius [comoving Mpc/h].  Broadcast against ``r_delta``.
    r_delta : array
        Halo radius [comoving Mpc/h].
    eta_ej : float
        :math:`r_{\rm ej}/R_\Delta`.  Traced.
    """
    x = jnp.asarray(r) / (eta_ej * jnp.asarray(r_delta))
    return (jax.scipy.special.erf(x / jnp.sqrt(2.0))
            - jnp.sqrt(2.0 / jnp.pi) * x * jnp.exp(-0.5 * x ** 2))
