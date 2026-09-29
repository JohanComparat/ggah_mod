r"""Stellar mass and halo mass, and the inverse that has to keep its gradient.

Every occupation model in layer 3 needs a stellar-mass--halo-mass relation, and
several of the published ones are written **backwards** -- Zu & Mandelbaum
(2015) Eq. 19 and Leauthaud et al. (2012) Eq. 3 both give :math:`M_h(M_*)`,
while what an occupation wants is :math:`M_*(M_h)` on a halo mass grid.  So they
have to be inverted numerically.

The inversion is where the predecessor lost a gradient
------------------------------------------------------

Three separate bisections -- ``_mstar_from_mh_zu15``, ``_inverse_shmr_z25``,
``_mstar_from_mh_leauthaud12`` -- each a bare ``jax.lax.fori_loop`` of 60
iterations.  A bisection depends on its inputs only through the *sign* of the
residual at each midpoint, and a boolean carries no derivative, so
``jax.grad`` through one returns **exactly zero, and returns it silently**.

That is not a hypothetical.  It is the predecessor's referee item A7, measured:
:math:`d\log_{10}M_*/d\beta` came back as autodiff ``0.0`` against a central
difference of ``0.105``.  Every Fisher forecast that varied a SHMR shape
parameter inherited it, and the symptom -- a column of zeros in the information
matrix, reported as infinite confidence -- looks like a well-constrained
parameter rather than a broken one.

:func:`invert_monotone` fixes it once, with the idiom layer 2 already uses in
:func:`~ggah_mod.halos.mass_definitions.translate_mass`: bisect to bracket the
root under ``stop_gradient``, then take **one Newton step**.  The value is
unchanged to machine precision -- the residual at the bracketed point is already
~1e-16 -- while the derivative becomes the implicit-function result
:math:`dx/d\theta = -(\partial_\theta F)/(\partial_x F)`, which is exact.

Units
-----

Each relation here is written in **its paper's units**, which are not one
convention: :data:`SHMR_MASS_UNITS` records them.  Zu & Mandelbaum (2015) use
:math:`h^{-1}M_\odot` for haloes and :math:`h^{-2}M_\odot` for stars; Girelli et
al. (2020) the same powers of :math:`h_{67} = H_0/67`; Leauthaud et al. (2012),
Kravtsov et al. (2018), Moster et al. (2013) and Behroozi et al. (2013) physical
:math:`M_\odot`; ``mstar_universemachine`` takes :math:`M_\odot/h` and converts
it itself.  The package's halo masses are :math:`M_\odot/h`, and
:class:`~ggah_mod.sectors.galaxies.GalaxySector` converts into and out of these
units at its boundary.  This paragraph used to say every relation was already
in :math:`M_\odot/h`, which put the ``zu15`` stellar fraction a factor
:math:`h` low.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ..numerics import N_BISECT, invert_monotone
from .calibration import Calibration

__all__ = [
    "SHMR_INCLUDES_REMNANTS", "REMNANT_FRACTION", "split_stellar",
    "invert_monotone",
    "mh_from_mstar_zu15", "mstar_from_mh_zu15",
    "mh_from_mstar_leauthaud12", "mstar_from_mh_leauthaud12",
    "f_kravtsov18", "mstar_from_mh_kravtsov18", "mh_from_mstar_kravtsov18",
    "mstar_moster13", "mstar_behroozi13", "mstar_girelli20",
    "mstar_universemachine",
    "SHMR", "make_shmr", "SHMR_CALIBRATION", "SHMR_MASS_UNITS",
    "halo_mass_to_relation", "stellar_mass_to_msun",
    "f_early_type", "scatter_zu15", "SIGMA_LNMSTAR_FLOOR",
]

_LN10 = jnp.log(10.0)

# =========================================================================
# The inverse
# =========================================================================

# `invert_monotone` moved to :mod:`ggah_mod.numerics`.  Layer 2's concentration
# relations need the same bracket-then-Newton inversion -- Diemer & Joyce (2019)
# solves G(c) = rhs -- and layer 2 cannot import from layer 3.  A helper that
# two layers need belongs at the root, which is where `lin_weights` already
# lives for the same reason.


# =========================================================================
# Zu & Mandelbaum (2015) -- the workhorse
# =========================================================================

@jax.jit
def mh_from_mstar_zu15(log10m_star, lg_m1h, lg_m0star, beta, delta, gamma):
    r"""Zu & Mandelbaum (2015) Eq. 19: :math:`\log_{10}M_h(\log_{10}M_*)`.

    .. math::

        \log_{10}M_h = \log_{10}M_1 + \beta x
        + \frac{1}{\ln 10}\left[\frac{t^\delta}{1+t^{-\gamma}}
                                - \frac12\right],
        \quad x = \log_{10}\frac{M_*}{M_{*0}},\; t = 10^{x}

    Note the :math:`1/\ln 10` on the non-linear term.  Leauthaud et al. (2012)
    write the same functional form **without** it -- see
    :func:`mh_from_mstar_leauthaud12`.  The two are not interchangeable and the
    difference is a factor 2.3 in the high-mass turnover.
    """
    x = jnp.asarray(log10m_star) - lg_m0star
    t = jnp.power(10.0, x)
    t_neg_gamma = jnp.power(jnp.maximum(t, 1e-30), -gamma)
    return lg_m1h + beta * x + (jnp.power(t, delta) / (1.0 + t_neg_gamma)
                                - 0.5) / _LN10


def mstar_from_mh_zu15(log10m_h, lg_m1h, lg_m0star, beta, delta, gamma,
                       lo=4.0, hi=13.0):
    r"""Inverse of :func:`mh_from_mstar_zu15`, **with its gradient**."""
    return invert_monotone(
        lambda ms: mh_from_mstar_zu15(ms, lg_m1h, lg_m0star, beta, delta, gamma),
        log10m_h, lo, hi)


#: Lower bound on :func:`scatter_zu15`, equal to the lower bound of the
#: ``sigma_lnmstar`` prior in ``GalaxyParams``.
SIGMA_LNMSTAR_FLOOR = 0.01


@jax.jit
def scatter_zu15(log10m_h, sigma_lnmstar, eta, lg_m1h):
    r"""Zu & Mandelbaum (2015) Eq. 20: scatter in :math:`\ln M_*` at fixed halo mass.

    .. math::

        \sigma_{\ln M_*}(M_h) = \max\!\left\{\sigma_{\rm floor},\
        \sigma_0 + \eta\,\max\!\left[0,\ \log_{10}(M_h/M_1)\right]\right\}

    Written with ``maximum`` rather than a ``where`` on the mass, so it is
    :math:`C^0` at the pivot exactly as published.  ``eta`` is negative in the
    fiducial fit, so the scatter *falls* in massive halos.

    **The floor is not in the paper.**  With :math:`\eta < 0` the published line
    reaches zero at :math:`\log_{10}M_h = \log_{10}M_1 - \sigma_0/\eta` and is
    negative above it, and a negative width flips the sign of the ``erfc``
    argument in :func:`~ggah_mod.sectors.occupation.n_cen_zu15`: the central
    occupation drops from :math:`f_{\rm c}` to zero and the satellites, which
    scale with it, go too.  At the published values that point is
    :math:`10^{24.6}`, far off any grid; the LS10 mass-bin MAP of ggah_cal's
    v0.8.0 campaign (:math:`\sigma_0 = 0.848`, :math:`\eta = -0.344`,
    :math:`\log_{10}M_1 = 12.491`) put it at :math:`10^{14.955}`, emptying every
    cluster.  :data:`SIGMA_LNMSTAR_FLOOR` is the prior's own lower bound, so the
    floor never binds at a width the prior admits as a constant; above it the
    function is the published one bit for bit.
    """
    return jnp.maximum(SIGMA_LNMSTAR_FLOOR, sigma_lnmstar + eta * jnp.maximum(
        jnp.asarray(log10m_h) - lg_m1h, 0.0))


# =========================================================================
# Leauthaud et al. (2012)
# =========================================================================

@jax.jit
def mh_from_mstar_leauthaud12(log10m_star, log10m1, log10m_star0, beta, delta,
                              gamma):
    r"""Leauthaud et al. (2012) Eq. 3.

    The Zu & Mandelbaum form **without** the :math:`1/\ln 10` prefactor on the
    non-linear term.  Kept as a separate function rather than a flag, because a
    flag is how the two conventions get mixed.
    """
    x = jnp.asarray(log10m_star) - log10m_star0
    t = jnp.power(10.0, x)
    t_neg_gamma = jnp.power(jnp.maximum(t, 1e-30), -gamma)
    return log10m1 + beta * x + jnp.power(t, delta) / (1.0 + t_neg_gamma) - 0.5


def mstar_from_mh_leauthaud12(log10m_h, log10m1, log10m_star0, beta, delta,
                              gamma, lo=4.0, hi=13.0):
    """Inverse of :func:`mh_from_mstar_leauthaud12`, **with its gradient**."""
    return invert_monotone(
        lambda ms: mh_from_mstar_leauthaud12(ms, log10m1, log10m_star0, beta,
                                             delta, gamma),
        log10m_h, lo, hi)


# =========================================================================
# Kravtsov et al. (2018), as used by Zacharegkas & Chang (2025)
# =========================================================================

@jax.jit
def f_kravtsov18(x, alpha, gamma, delta):
    r"""Kravtsov et al. (2018) auxiliary function.

    .. math::

        f(x) = -\log_{10}(10^{\alpha x}+1)
             + \delta\,\frac{[\log_{10}(1+e^x)]^\gamma}{1+e^{10^{-x}}}

    Both logs go through ``logaddexp``, which is the difference between this
    evaluating and it overflowing: at :math:`x = 3` and :math:`\alpha = -1.6`
    the naive form computes :math:`10^{-4.8}+1` fine, but at the other end
    :math:`e^{10^{-x}}` overflows for :math:`x \lesssim -2.7`, so the exponent
    is capped.  The cap sits far inside the region where the term it guards has
    already saturated to zero, so it changes no value and no derivative on the
    domain the relation is used over.
    """
    x = jnp.asarray(x)
    term1 = -jnp.logaddexp(alpha * x * _LN10, 0.0) / _LN10
    log10_1pex = jnp.logaddexp(x, 0.0) / _LN10
    pow10_neg_x = jnp.minimum(jnp.power(10.0, -x), 500.0)
    term2 = (delta * jnp.power(jnp.maximum(log10_1pex, 0.0), gamma)
             / (1.0 + jnp.exp(pow10_neg_x)))
    return term1 + term2


@jax.jit
def mstar_from_mh_kravtsov18(log10m_h, log10m1, log10eps, alpha, gamma, delta):
    r"""Kravtsov et al. (2018) SHMR -- **forward**, so no inversion is needed.

    .. math::

        \log_{10}M_* = (\log_{10}\epsilon + \log_{10}M_1)
                     + f(\log_{10}(M_h/M_1)) - f(0)
    """
    x = jnp.asarray(log10m_h) - log10m1
    return (log10eps + log10m1 + f_kravtsov18(x, alpha, gamma, delta)
            - f_kravtsov18(0.0, alpha, gamma, delta))


def mh_from_mstar_kravtsov18(log10m_star, log10m1, log10eps, alpha, gamma,
                             delta, lo=10.0, hi=16.0):
    """Inverse of :func:`mstar_from_mh_kravtsov18`, **with its gradient**."""
    return invert_monotone(
        lambda mh: mstar_from_mh_kravtsov18(mh, log10m1, log10eps, alpha,
                                            gamma, delta),
        log10m_star, lo, hi)


# =========================================================================
# Forward abundance-matching relations
# =========================================================================

@jax.jit
def mstar_moster13(log10m_h, z=0.0, m10=11.590, m11=1.195, n10=0.0351,
                   n11=-0.0247, b10=1.376, b11=-0.826, g10=0.608, g11=0.329):
    r"""Moster et al. (2013) double power law, Eqs. 2 and 11--14.

    .. math::

        \frac{M_*}{M_h} = 2N\left[
            \left(\frac{M_h}{M_1}\right)^{-\beta}
          + \left(\frac{M_h}{M_1}\right)^{\gamma}\right]^{-1}

    with every coefficient linear in :math:`z/(1+z)`.
    """
    a = jnp.asarray(z) / (1.0 + jnp.asarray(z))
    log10m1 = m10 + m11 * a
    n = n10 + n11 * a
    b = b10 + b11 * a
    g = g10 + g11 * a
    x = jnp.power(10.0, jnp.asarray(log10m_h) - log10m1)
    ratio = 2.0 * n / (jnp.power(x, -b) + jnp.power(x, g))
    return jnp.asarray(log10m_h) + jnp.log10(ratio)


@jax.jit
def mstar_behroozi13(log10m_h, z=0.0, eps0=-1.777, eps_a=-0.006, eps_z=0.0,
                     eps_a2=-0.119, m0=11.514, m_a=-1.793, m_z=-0.251,
                     alpha0=-1.412, alpha_a=0.731, delta0=3.508, delta_a=2.608,
                     delta_z=-0.043, gamma0=0.316, gamma_a=1.319,
                     gamma_z=0.279):
    r"""Behroozi, Wechsler & Conroy (2013) Eqs. 3--4, with the :math:`\nu(a)` damping.

    .. math::

        \log_{10}M_* = \log_{10}(\epsilon M_1)
          + f\!\left(\log_{10}\frac{M_h}{M_1}\right) - f(0),
        \quad
        f(x) = -\log_{10}(10^{\alpha x}+1)
             + \delta\frac{[\log_{10}(1+e^{x})]^{\gamma}}{1+e^{10^{-x}}}

    The redshift terms are damped by :math:`\nu(a) = e^{-4a^2}`, which is what
    keeps the evolution from running away at high redshift.

    Note :math:`f(0)`: the denominator there is :math:`1 + e^{10^0} = 1 + e`,
    **not** 2.  Using 2 puts the whole relation low by
    :math:`\delta(\log_{10}2)^{\gamma}\,[\tfrac12 - 1/(1+e)]` -- the same at
    every mass, but not at every redshift, because :math:`\delta` and
    :math:`\gamma` evolve: at the defaults here 0.555 dex at :math:`z = 0`,
    0.563 at 1, 0.473 at 2 and 0.235 at 4.  (The same slip in
    :func:`mstar_from_mh_kravtsov18`, which has no redshift terms, is 0.43 dex
    at the ``zacharegkas25`` defaults.)  An offset flat in mass is exactly the
    kind that looks like a calibration choice rather than an error.  Evaluating
    ``f`` at 0 rather than writing the special case out is how that stays
    impossible here.
    """
    z = jnp.asarray(z)
    a = 1.0 / (1.0 + z)
    nu = jnp.exp(-4.0 * a * a)
    log10_eps = eps0 + (eps_a * (a - 1.0) + eps_z * z) * nu + eps_a2 * (a - 1.0)
    log10m1 = m0 + (m_a * (a - 1.0) + m_z * z) * nu
    alpha = alpha0 + (alpha_a * (a - 1.0)) * nu
    delta = delta0 + (delta_a * (a - 1.0) + delta_z * z) * nu
    gamma = gamma0 + (gamma_a * (a - 1.0) + gamma_z * z) * nu

    def f(x):
        t1 = -jnp.logaddexp(alpha * x * _LN10, 0.0) / _LN10
        l = jnp.logaddexp(x, 0.0) / _LN10
        cap = jnp.minimum(jnp.power(10.0, -x), 500.0)
        return t1 + delta * jnp.power(jnp.maximum(l, 0.0), gamma) / (1.0 + jnp.exp(cap))

    x = jnp.asarray(log10m_h) - log10m1
    return log10_eps + log10m1 + f(x) - f(0.0)


@jax.jit
def mstar_girelli20(log10m_h, z=0.0, B=11.79, mu=0.20, C=0.046, nu=-0.38,
                    D=0.709, eta=-0.18, F=0.043, E=0.96):
    r"""Girelli et al. (2020) A&A 634 A135 Eq. 6, Table 3 (no-scatter fit).

    .. math::

        \frac{M_*}{M_h}(z) = \frac{2A(z)}
             {(M_h/M_A)^{-\beta} + (M_h/M_A)^{\gamma}}

    with :math:`\log_{10}M_A = B + z\mu`, :math:`A = C(1+z)^\nu`,
    :math:`\gamma = D(1+z)^\eta` and :math:`\beta = Fz + E`.

    The two exponents are easy to swap, and swapping them is not obviously
    wrong from the output -- :math:`M_*` still rises monotonically.  What it
    does is put the steeper index on the *high*-mass side (:math:`\beta = 0.96`
    against :math:`\gamma = 0.709` at :math:`z = 0`), and move both ends: at
    :math:`z = 0` and :math:`M_h = 10^{15}\,h_{67}^{-1}M_\odot`, in the
    relation's own units, the swap gives :math:`M_*/M_h = 7.6\times10^{-5}`
    where the relation gives :math:`4.9\times10^{-4}`, a factor 6.4 low, and at
    :math:`10^{11}` it gives 0.024 against 0.015.  A stellar fraction wrong by
    that much at the cluster scale would go straight into the matter field's
    budget.
    """
    z = jnp.asarray(z)
    log10_ma = B + z * mu
    a = C * jnp.power(1.0 + z, nu)
    gamma = D * jnp.power(1.0 + z, eta)
    beta = F * z + E
    ratio = jnp.power(10.0, jnp.asarray(log10m_h) - log10_ma)
    return jnp.asarray(log10m_h) + jnp.log10(
        2.0 * a / (jnp.power(ratio, -beta) + jnp.power(ratio, gamma)))


# =========================================================================
# Morphology -- the early-type fraction
# =========================================================================

def mstar_universemachine(log10m_h, z=0.0, *, h,
                          eps0=-1.435, eps_a=1.813, eps_lna=1.353,
                          eps_z=-0.214,
                          m0=12.081, m_a=4.696, m_lna=4.485, m_z=-0.740,
                          alpha0=1.957, alpha_a=-2.650, alpha_lna=-1.953,
                          alpha_z=0.204,
                          beta0=0.474, beta_a=-0.903, beta_z=-0.492,
                          delta0=0.386,
                          gamma0=-1.065, gamma_a=-3.243, gamma_z=-1.107):
    r"""UniverseMachine's median stellar mass--halo mass fit, Appendix J.

    Behroozi, Wechsler, Hearin & Conroy (2019), MNRAS 488, 3143
    (arXiv:1806.07893), Eqs. (J1)--(J8):

    .. math::

        \log_{10}\frac{M_*}{M_1} &= \epsilon
            - \log_{10}\left(10^{-\alpha x} + 10^{-\beta x}\right)
            + \gamma \exp\left[-\tfrac12\left(\frac{x}{\delta}\right)^2\right] \\
        x &= \log_{10}\frac{M_{\rm peak}}{M_1}

    with :math:`\log_{10}(M_1/M_\odot)`, :math:`\epsilon` and :math:`\alpha`
    evolving as :math:`p_0 + p_a(a-1) - p_{\ln a}\ln a + p_z z`, :math:`\beta`
    without the :math:`\ln a` term, :math:`\delta` constant, and
    :math:`\log_{10}\gamma` as :math:`\gamma_0+\gamma_a(a-1)+\gamma_z z`.

    **Not the same function as** :func:`mstar_behroozi13`, which is worth saying
    because the two share an author and a purpose.  B13 is a single power law
    plus a :math:`\delta/\gamma` term with an :math:`e^{-4a^2}` damping; this is
    a genuine double power law plus a Gaussian, with no damping.  They cannot
    share a kernel, and an attempt to make them would have to distort one.

    Coefficients: **Table J1, row** ``SM = Obs., Q/SF = All, Cen/Sat = Cen.,
    IHL = Excl.`` -- read from the paper, not recalled.  Four consequences of
    that row choice, each of which changes a number:

    * **It is the *observed* stellar-mass fit**, so UM's own systematic offsets
      (:math:`\mu`, :math:`\kappa`, :math:`\sigma_{\rm SM}`) are already inside
      it.  A forward model that applies its own offsets on top double-counts
      them.  It is also why
      :data:`SHMR_INCLUDES_REMNANTS`\ ``["universemachine"]`` is ``True``:
      observed stellar mass is what population synthesis reports, which is
      living stars plus remnants.
    * **Centrals**, and intra-halo light **excluded** -- so this is a galaxy,
      not a galaxy plus its ICL, which is what ``f_star_cen`` wants.
    * **The mass is** :math:`M_{\rm peak}`, not :math:`M_{200m}`.  The shipped
      :class:`~ggah_mod.halos.field.HaloField` carries current mass at
      :math:`200m`.  For centrals the two are close and the error is bounded;
      for stripped satellites they are not, which is a second reason this entry
      is the centrals row.  Named rather than absorbed.
    * **Validity is** :math:`10^{10.5} < M_{\rm peak}/M_\odot < 10^{15}`,
      :math:`0 < z < 10`, fitted to 0.03 dex.  The shipped mass grid starts at
      :math:`10^{10}\,M_\odot/h = 1.5\times10^{10}\,M_\odot`, so its bottom two
      thirds of a decade are **outside the fit**.  Recorded in
      :data:`SHMR_CALIBRATION`.

    Parameters
    ----------
    log10m_h : array
        :math:`\log_{10}(M_{\rm peak}/(M_\odot/h))` -- this package's units.
    z : float
    h : float
        **Required, no default.**  Appendix J is in physical solar masses and
        every mass on a :class:`HaloField` is in :math:`M_\odot/h`, so the
        conversion has to happen and there is no value of ``h`` that is safe to
        assume.  A default of 1 would silently mean "already physical" and be
        wrong by 0.17 dex on this package's grids -- five times the fit's own
        tolerance.  The other relations in this registry duck the question; this
        one asks it.

    Returns
    -------
    array
        :math:`\log_{10}(M_*/M_\odot)`, physical solar masses.
    """
    z = jnp.asarray(z)
    a = 1.0 / (1.0 + z)
    am1 = a - 1.0
    ln_a = jnp.log(a)

    log10m1 = m0 + m_a * am1 - m_lna * ln_a + m_z * z
    eps = eps0 + eps_a * am1 - eps_lna * ln_a + eps_z * z
    alpha = alpha0 + alpha_a * am1 - alpha_lna * ln_a + alpha_z * z
    beta = beta0 + beta_a * am1 + beta_z * z
    delta = delta0
    gamma = jnp.power(10.0, gamma0 + gamma_a * am1 + gamma_z * z)

    # Msun/h -> Msun.  One line, and the reason it is not a default is in the
    # `h` parameter's docstring.
    x = (jnp.asarray(log10m_h) - jnp.log10(h)) - log10m1

    # `logaddexp` rather than `log10(10^u + 10^v)`: at the low-mass end
    # -alpha*x reaches +20 on the shipped grid and the naive form overflows to
    # inf before the log takes it back down.
    lse = jnp.logaddexp(-alpha * x * _LN10, -beta * x * _LN10) / _LN10
    bump = gamma * jnp.exp(-0.5 * jnp.square(x / delta))
    return eps - lse + bump + log10m1


@jax.jit
def f_early_type(log10m_h, log10m_morph=12.5, beta_morph=1.0):
    r"""Early-type fraction, :math:`1 - \exp[-(M_h/M_{\rm morph})^{\beta}]`.

    A Weibull CDF, the same shape the red fraction takes in Zu & Mandelbaum
    (2016).  Rises from 0 to 1 with halo mass; used as a bulge-to-total proxy
    where the AGN sector wants one.
    """
    x = jnp.power(10.0, jnp.asarray(log10m_h) - log10m_morph)
    return -jnp.expm1(-jnp.power(x, beta_morph))


# =========================================================================
# Registry
# =========================================================================

#: Forward relations: ``log10m_h -> log10 M_*``.  The two that are published
#: backwards, ``zu15`` and ``leauthaud12``, are here through their
#: differentiable inverses, so a caller never has to know which way round its
#: source paper wrote it; ``kravtsov18`` is published forward and is here as
#: written.
#:
#: Those three were **missing** until layer 4 asked for one.  The sentence above
#: was already written, ``mstar_from_mh_zu15``, ``mstar_from_mh_leauthaud12``
#: and ``mstar_from_mh_kravtsov18`` already existed and were already exported,
#: and the registry listed none of them -- so the docstring described
#: entries that were not there.  It went unnoticed because
#: :data:`SHMR_CALIBRATION` was incomplete in exactly the same way, which is
#: what the key-parity test compares against: two dicts agreeing about an
#: absence look identical to two dicts agreeing about a presence.
#:
#: It matters beyond tidiness.  ``zumandelbaum15`` and ``leauthaud12`` are
#: *occupations* in :mod:`~ggah_mod.sectors.occupation` built on precisely these
#: relations, so asking for the stellar mass that goes with one of those
#: occupations used to mean picking an unrelated abundance-matching fit.
#: ``relation -> (n_halo, n_star, h_ref)``: its masses are in
#: :math:`(h/h_{\rm ref})^{-n}M_\odot`, so ``n = 0`` is physical.  The input halo
#: mass and the output stellar mass each have their own power.
#:
#: * ``zu15``: Zu & Mandelbaum (2015), Sec. 1, "the stellar mass and halo mass are
#:   in units of :math:`h^{-2}M_\odot` and :math:`h^{-1}M_\odot`".
#: * ``girelli20``: Girelli et al. (2020), Sec. 1, halo masses in
#:   :math:`h_{67}^{-1}M_\odot` and stellar masses in :math:`h_{67}^{-2}M_\odot`.
#: * ``leauthaud12``, ``kravtsov18`` (as used by Zacharegkas et al. 2025),
#:   ``moster13`` (Table 1 notes: "All masses are in units of M⊙") and
#:   ``behroozi13`` (:math:`h = 0.7`, masses in :math:`M_\odot`): physical.
#: * ``universemachine``: takes :math:`M_\odot/h` and returns physical
#:   :math:`M_\odot`, converting with its required ``h``.
SHMR_MASS_UNITS: dict[str, tuple[int, int, float]] = {
    "zu15": (1, 2, 1.0),
    "girelli20": (1, 2, 0.67),
    "leauthaud12": (0, 0, 1.0),
    "kravtsov18": (0, 0, 1.0),
    "moster13": (0, 0, 1.0),
    "behroozi13": (0, 0, 1.0),
    "universemachine": (1, 0, 1.0),
}


def halo_mass_to_relation(log10m, units, h):
    r""":math:`\log_{10}` of a halo mass in :math:`M_\odot/h`, in a relation's units.

    A mass :math:`Y\,M_\odot/h` is :math:`Y h^{n-1}h_{\rm ref}^{-n}` in
    :math:`(h/h_{\rm ref})^{-n}M_\odot`.
    """
    n, _, h_ref = units
    return jnp.asarray(log10m) + (n - 1) * jnp.log10(h) - n * jnp.log10(h_ref)


def stellar_mass_to_msun(log10m_star, units, h):
    r""":math:`\log_{10}(M_\star/M_\odot)`, physical, from a relation's units."""
    _, n, h_ref = units
    return jnp.asarray(log10m_star) - n * (jnp.log10(h) - jnp.log10(h_ref))


SHMR = {
    "moster13": mstar_moster13,
    "behroozi13": mstar_behroozi13,
    "girelli20": mstar_girelli20,
    "zu15": mstar_from_mh_zu15,
    "leauthaud12": mstar_from_mh_leauthaud12,
    "kravtsov18": mstar_from_mh_kravtsov18,
    "universemachine": mstar_universemachine,
}

#: What each relation was fitted to, and over what range it means anything.
#: Applying one outside its row is a systematic error, not a tolerance.
SHMR_CALIBRATION = {
    "moster13": Calibration("abundance matching, SDSS+high-z", (0.0, 4.0)),
    "behroozi13": Calibration("abundance matching + SFR, compiled", (0.0, 8.0)),
    "girelli20": Calibration("abundance matching, COSMOS/UltraVISTA", (0.0, 4.0)),
    "zu15": Calibration("SDSS locus, iHOD; inverted from M_h(M_*)", (0.0, 0.3),
                        notes="the published relation; the zumandelbaum "
                              "occupations default to an LS10 fit of it, "
                              "recorded in occupation.OCC_CALIBRATION"),
    "leauthaud12": Calibration("COSMOS lensing+clustering+SMF; inverted",
                               (0.22, 1.0)),
    "kravtsov18": Calibration("BCG+ICL abundance matching; forward", (0.0, 0.1)),
    # The only row here fitted over the whole redshift range it claims, and the
    # only one whose *mass* range excludes part of the shipped grid: Appendix J
    # is fitted for 10^10.5 < Mpeak/Msun < 10^15, and the grid starts at
    # 10^10 Msun/h = 1.5e10 Msun.  Applying it below that is extrapolating a
    # fit, which this table exists to make visible.
    "universemachine": Calibration(
        "UniverseMachine mock catalogues, Table J1 Obs./All/Cen./Excl.; "
        "Mpeak in Msun, 10^10.5-10^15", (0.0, 10.0)),
}


#: Whether each relation's :math:`M_\star` already contains stellar remnants.
#:
#: **All seven are `True`, and that is the point.**  Fukugita & Peebles (2004)
#: list main-sequence stars (rows 3.3, 3.4) separately from white dwarfs,
#: neutron stars and stellar black holes (rows 3.5-3.7), which together are
#: 1.1% of :math:`\Omega_b` -- above the one per cent the
#: census is scoped to, and so apparently a missing component.  It is not
#: missing.  Every relation here is calibrated against stellar masses from
#: population-synthesis fits, whose reported mass is the *surviving* mass:
#: living stars **plus** remnants.  Adding a remnant term on top would count
#: them twice.
#:
#: The declaration exists rather than the assumption because the assumption is
#: not safe.  Some synthesis codes report living-star mass instead, the
#: difference is 19% of :math:`M_\star`, and nothing in a relation's published
#: form says which convention its calibration used.  A relation added later
#: with the other convention sets this to ``False`` and
#: :func:`remnant_fraction` becomes live for it alone.
SHMR_INCLUDES_REMNANTS = {
    "moster13": True,
    "behroozi13": True,
    "girelli20": True,
    "zu15": True,
    "leauthaud12": True,
    "kravtsov18": True,
    # The Table J1 row chosen is the *observed* stellar-mass fit, and observed
    # stellar mass is a population-synthesis mass: living stars plus remnants.
    "universemachine": True,
}

#: Remnants as a fraction of the surviving stellar mass.
#:
#: Anchored on Fukugita & Peebles (2004) themselves: rows 3.5-3.7 over rows
#: 3.3-3.7 is :math:`0.00048/0.00253 = 0.19`, consistent with a Chabrier IMF
#: integrated over a Hubble time.  Row 3.8, substellar objects, is deliberately
#: **not** in that sum: a brown dwarf is not a remnant, and it is not in a
#: population-synthesis stellar mass either -- so it is a genuinely absent 0.31%
#: of :math:`\Omega_b`, below this census's scope and named here so the
#: exclusion is a decision rather than an oversight.  This does **not** enter
#: the mass budget --
#: see :data:`SHMR_INCLUDES_REMNANTS` -- and exists so a census table can
#: report a main-sequence row and a remnant row that sum to
#: :math:`\Omega_\star`, which is what makes the comparison with FP04 row by
#: row rather than lumped.
REMNANT_FRACTION = 0.19


def split_stellar(m_star, remnant_fraction: float = REMNANT_FRACTION,
                  includes_remnants: bool = True):
    r"""``(main sequence, remnants)`` from a stellar mass, without changing it.

    .. math::

        M_\star^{\rm MS} = (1-f_{\rm rem})M_\star, \qquad
        M_\star^{\rm rem} = f_{\rm rem}M_\star

    when the relation already includes remnants -- the sum is
    :math:`M_\star` **identically**, so splitting a census row cannot change
    the budget it belongs to.  When it does not, the remnants are added:
    :math:`M_\star^{\rm MS} = M_\star` and
    :math:`M_\star^{\rm rem} = f_{\rm rem}M_\star/(1-f_{\rm rem})`, so the
    total grows, which is the case the flag exists to keep distinguishable.

    Parameters
    ----------
    m_star : array
    remnant_fraction : float
        Of the surviving stellar mass.  See :data:`REMNANT_FRACTION`.
    includes_remnants : bool
        From :data:`SHMR_INCLUDES_REMNANTS`, never guessed.
    """
    m_star = jnp.asarray(m_star)
    f = jnp.asarray(remnant_fraction)
    if includes_remnants:
        return (1.0 - f) * m_star, f * m_star
    return m_star, f * m_star / (1.0 - f)


def make_shmr(name: str):
    """Look up a forward stellar-mass--halo-mass relation by name."""
    key = str(name).lower()
    if key not in SHMR:
        raise ValueError(f"unknown SHMR {name!r}; expected one of {sorted(SHMR)}")
    return SHMR[key]
