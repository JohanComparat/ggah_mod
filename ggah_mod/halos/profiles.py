r"""Halo density profiles: :math:`\rho(r)`, :math:`u(k|M)`, :math:`\Sigma(R)`.

Three families, all pure JAX and all differentiable:

* **NFW** (Navarro, Frenk & White 1997) -- the default, with an analytic Fourier
  transform and analytic projected quantities;
* **Einasto** (Einasto 1965) -- a shape parameter :math:`\alpha` instead of a
  fixed inner slope, Fourier-transformed numerically;
* **gNFW** -- generalised NFW with free inner, transition and outer slopes.  The
  shape the hot-gas sector needs, and a first-class profile here rather than a
  private helper repeated inside three gas modules.

Conventions for a profile
-------------------------

Radii and :math:`r_s` are **comoving** Mpc/h, masses M_sun/h, wavenumbers
h/Mpc, and :math:`\rho_s` M_sun h^2/Mpc^3.  Every Fourier transform is
**normalised so that** :math:`u(k \to 0) = 1`.  That is not cosmetic: it is what
makes mass conservation checkable, and the matter tracer built on these profiles
depends on it exactly.

Everything is traceable
-----------------------

The predecessor had `nfw_uk` (numpy, via ``scipy.special.sici``) *and*
`nfw_uk_jax` as a hand-maintained twin pair, and its `einasto_uk` and
`satellite_nfw_uk` were numpy while being called from the traced path -- they
never failed only because the one gradient test differentiated with respect to
an amplitude parameter, which does not reach the halo radius or concentration.
Here there is one implementation of each, in ``jnp``, and
``tests/test_paths.py`` differentiates all of them with respect to
:math:`\Omega_m`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np

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

    One definition.  The predecessor wrote this expression eight times across
    five files, one copy of which was dead code.
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

    Defined only for :math:`x > 0`.  The branch selector tests **x**, not the
    guarded copy of it: the predecessor compared ``x_safe < 12`` after replacing
    non-positive ``x`` by 1.0, so every :math:`x \le 0` silently returned
    :math:`\mathrm{Ci}(1)` instead of a NaN.  Here it returns NaN, which is what
    an undefined value should be.

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
    branches :math:`x<1`, :math:`x=1`, :math:`x>1` written so that **both**
    unused branches stay finite -- ``jnp.where`` evaluates all of them, and a
    NaN in a discarded branch still poisons the gradient.
    """
    x = jnp.asarray(x)
    lo = jnp.where(x < 1.0, x, 0.5)
    hi = jnp.where(x > 1.0, x, 2.0)
    f_lo = (1.0 - 2.0 / jnp.sqrt(1.0 - lo ** 2)
            * jnp.arctanh(jnp.sqrt((1.0 - lo) / (1.0 + lo)))) / (lo ** 2 - 1.0)
    f_hi = (1.0 - 2.0 / jnp.sqrt(hi ** 2 - 1.0)
            * jnp.arctan(jnp.sqrt((hi - 1.0) / (1.0 + hi)))) / (hi ** 2 - 1.0)
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

    No closed form, so Gauss-Legendre in radius.  **Pure JAX** -- the
    predecessor's version was numpy and would raise on any gradient that reached
    the halo radius or concentration, which no test exercised.

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
    **zero-neutral**: at the defaults this is the NFW result exactly, and the
    approach to it is continuous and differentiable rather than a special case.

    ``b_sat_conc``
        multiplies the concentration, :math:`c_{\rm sat} = b\,c`.  Values below
        1 make the satellite distribution shallower than the mass.
    ``f_cut_inv``
        **inverse** outer truncation scale, in units of :math:`r_\Delta^{-1}`:
        the profile is multiplied by
        :math:`\exp(-f_{\rm cut}^{-1}\,r/r_\Delta)`, so the truncation radius
        is :math:`r_{\rm cut} = r_\Delta / f_{\rm cut}^{-1}` and **0 means no
        truncation**.
    ``gamma_inner``
        extra inner power :math:`(r/r_s)^{-\gamma}` steepening or flattening the
        core.

    .. rubric:: Why the truncation is carried inverted

    It used to be ``f_cut``, the truncation radius *as a fraction of*
    :math:`r_\Delta`, with ``f_cut = 0`` special-cased to mean "none".  That
    made zero an **isolated point**: "no truncation" is
    :math:`f_{\rm cut}\to\infty`, and the limit from *below* is the opposite --
    as :math:`f_{\rm cut}\to0^+` the exponential kills everything but the
    origin, so :math:`u(k)\to1` at every :math:`k`, a point mass.  Measured at
    :math:`c=5`: :math:`u(k{=}10)` was :math:`0.157` at ``f_cut = 0`` and
    :math:`1.000` at ``f_cut = 1e-6``.  The parameter was discontinuous at its
    own default, ``jax.grad`` returned **nan** there, and a sampler given the
    box ``(0, 0.3)`` would have stepped straight across the gap and fitted a
    delta function.

    Inverting it removes all of that rather than guarding it: :math:`0` is now
    the *interior* limit of "no truncation", the branch is gone, and the
    derivative at the default is an ordinary finite number.  Convert an old
    value with :math:`f_{\rm cut}^{-1} = 1/f_{\rm cut}`; ``f_cut = 0`` becomes
    ``f_cut_inv = 0``, which is the one value that meant the same thing in both.

    The three shape parameters are **keyword-only**, so a positional call
    written against the old signature fails instead of silently reinterpreting
    a truncation radius as its reciprocal.

    **Pure JAX**, and all three parameters are traceable.  The predecessor's
    version forced ``b_sat_conc`` through ``float()`` and took ``k.min()/k.max()``
    to build a numpy grid, so only the truncation and ``gamma_inner`` carried
    gradients; it also took an ``r_s`` argument it ignored and recomputed.

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
    dark-matter profile -- which is what
    :mod:`~ggah_mod.sectors.matter` did before it had this term -- puts them
    where the lensing signal is largest instead of where they are.

    Chosen over the alternatives for three properties, in this order:

    * :math:`u(k\to0) = 1` **exactly**, in closed form.  Every other component
      of the matter budget has that by construction, and an ejected term that
      only approached it numerically would put a floor under the module's
      mass-conservation residual -- currently :math:`8.5\times10^{-9}`, and set
      by the NFW transform rather than by anything here.
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
    It answers *at what radius the baryon budget closes*: the census integrates
    masses over :math:`M` and has no radius, and every other component closes
    within :math:`R_\Delta` by its own normalisation, so this is the one that
    sets it.  90 per cent of the ejected gas lies within
    :math:`2.50\,\eta_{\rm ej}R_\Delta` and 99 per cent within
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
