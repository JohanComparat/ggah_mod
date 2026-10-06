r"""Conditional luminosity and stellar-mass functions.

An HOD answers "how many galaxies above a threshold?".  A **conditional
luminosity function** answers the finer question -- *how many, at each
luminosity* -- and integrates it:

.. math::

    \Phi_c(L|M)\,dL &: \text{one central, log-normal about } L_c(M) \\
    \Phi_s(L|M)\,dL &= \frac{\phi_s^*(M)}{L_c}
        \left(\frac{L}{L_c}\right)^{\alpha_s}
        e^{-(L/L_c)^2}\,dL

The measure is :math:`dL`, as in Cacciato et al. (2009) Eq. 36, whose cut-off
is :math:`L_s^* = 0.562\,L_c` rather than the :math:`L_c` of the default
``cacciato09`` entry (see :data:`CLF_DEFAULTS`).  The satellite integral is
what makes this family worth having in closed form:

.. math::

    \langle N_{\rm sat}(>L)\rangle
      = \frac{\phi_s^*(M)}{2}\,
        \Gamma\!\left(\frac{\alpha_s+1}{2},\;(L/L_c)^2\right)

with :math:`\Gamma` the **unregularised** upper incomplete gamma function.  No
quadrature, and differentiable in :math:`\alpha_s` -- which matters, because
:math:`\alpha_s` is negative in every published fit and a numerical integral
over :math:`L^{\alpha_s}` near zero is where precision goes.

Why this is a different signature from :mod:`~ggah_mod.sectors.occupation`
--------------------------------------------------------------------------

These take a *luminosity* (or stellar-mass) threshold and a set of CLF shape
parameters; the HODs take a mass threshold and occupation parameters.  Forcing
one signature on both would mean the HODs accepting a luminosity they cannot
use -- the mistake layer 2 documents for its two families of concentration
relation, and the one that leaves a fit reporting a constraint on a parameter
nothing read.

The gamma function below :math:`\alpha_s = -1`
-----------------------------------------------

:math:`(\alpha_s+1)/2` is negative for :math:`\alpha_s < -1`, which every
published fit has, and ``gammaincc`` is defined only for a positive first
argument.  :func:`upper_gamma` therefore uses the recurrence
:math:`\Gamma(a,x) = [\Gamma(a{+}1,x) - x^a e^{-x}]/a`, which moves the argument
into the supported range.  Written branchlessly with ``where``, so both
branches are traced and neither can produce a NaN that poisons the gradient of
the other -- the standard JAX hazard with a ``where`` over a singular
expression.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..numerics import arctan
from .calibration import Calibration

__all__ = [
    "log10_lc", "phi_s_star", "upper_gamma",
    "clf_central_mean", "clf_satellite_mean",
    "phi_s_star_cacciato13", "phi_sat_cacciato09", "alpha_faint_cacciato09",
    "clf_central_vdb13", "clf_satellite_vdb13",
    "CLF", "CLF_CALIBRATION", "CLF_DEFAULTS", "make_clf", "clf_defaults",
    "L_S_OVER_L_C",
]

_SQRT2 = jnp.sqrt(2.0)

#: The CLF faint-end slope is ``alpha_faint`` and its normalisation is
#: ``phi_s_amp``.
#:
#: Both were renamed by ``PLAN.md`` item **G9**, and both for the same reason.
#: ``alpha_sat`` is a *positive* satellite power-law index in the occupations
#: -- published at (0.5, 1.5) -- and was a *negative* Schechter faint-end slope
#: here, at -1.15 and -1.3: same name, opposite sign, different quantity, and
#: :data:`~ggah_mod.sectors.galaxies.GALAXY_MODELS` merges the two registries so
#: one lookup reaches both.  ``b_sat`` differed from the occupations' ``bsat``
#: by a single underscore, which is the trap ``width_logmstar`` was named to
#: avoid one module over.
#:
#: :math:`L_s^*/L_c`, the offset between the satellite Schechter cut-off and
#: the central luminosity.  Fitted, and the same value in every paper of the
#: family, so it is a default rather than a constant.
L_S_OVER_L_C = 0.562


@jax.jit
def log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen):
    r"""Central luminosity, Cacciato et al. (2009) Eq. 37.

    .. math::

        \log_{10}L_c = \log_{10}L_0 + \alpha x
            + (\beta - \alpha)\log_{10}(1 + 10^{x}),
        \quad x = \log_{10}(M/M_1)

    A broken power law: slope :math:`\alpha` below :math:`M_1`, :math:`\beta`
    above -- their :math:`\gamma_1` and :math:`\gamma_2`.
    """
    x = jnp.asarray(log10m) - log10m1
    return (log10l0 + alpha_cen * x
            + (beta_cen - alpha_cen) * jnp.logaddexp(x * jnp.log(10.0), 0.0)
            / jnp.log(10.0))


@jax.jit
def clf_central_mean(log10m, log10l_lim, log10l0, log10m1, alpha_cen,
                     beta_cen, sigma_c):
    r""":math:`\langle N_{\rm cen}(>L)\rangle = \tfrac12{\rm erfc}
    [(\log L_{\rm lim} - \log L_c)/(\sqrt2\sigma_c)]`.

    :math:`\sigma_c` is a genuine log-normal width, so the :math:`\sqrt2` is
    the one that always accompanies a Gaussian CDF written with ``erfc`` -- not
    the same symbol as Zheng's :math:`\sigma_{\log M}`.
    """
    from jax.scipy.special import erfc
    lc = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    return 0.5 * erfc((log10l_lim - lc) / (_SQRT2 * sigma_c))


@jax.jit
def phi_s_star(log10m, log10m1, sigma_c, phi_s_amp):
    r""":math:`\phi_s^* = A_\phi\,(M/M_1)/(\sqrt{2\pi}\,\sigma_c)`, with
    :math:`A_\phi` = ``phi_s_amp``.

    A simplification, linear in :math:`M`, and not Cacciato et al. (2009)'s
    normalisation, which is the quadratic in :math:`\log(M/10^{12})` of their
    Eq. 40 (:func:`phi_sat_cacciato09`)."""
    return (phi_s_amp / (jnp.sqrt(2.0 * jnp.pi) * sigma_c)
            * jnp.power(10.0, jnp.asarray(log10m) - log10m1))


@jax.jit
def upper_gamma(a, x):
    r"""Unregularised :math:`\Gamma(a,x)`, valid for :math:`a > -1`.

    ``jax.scipy.special.gammaincc`` needs :math:`a > 0`.  Every published CLF
    has :math:`\alpha_s < -1`, hence :math:`a = (\alpha_s+1)/2 < 0`, so the
    recurrence

    .. math::  \Gamma(a,x) = \frac{\Gamma(a+1,x) - x^{a}e^{-x}}{a}

    is used there.  Both branches are evaluated -- that is how ``where`` works
    -- so each is guarded independently: without the guards the unused branch
    produces a NaN, and a NaN in an unused ``where`` branch **still poisons the
    gradient** of the used one.  That is the single most common way a JAX
    special function silently returns ``nan`` for a valid input.
    """
    from jax.scipy.special import gamma, gammaincc
    a = jnp.asarray(a)
    x = jnp.maximum(jnp.asarray(x), 0.0)

    a_pos = jnp.maximum(a, 1e-10)
    direct = gamma(a_pos) * gammaincc(a_pos, x)

    a1 = jnp.maximum(a + 1.0, 1e-10)
    g_a1 = gamma(a1) * gammaincc(a1, x)
    x_safe = jnp.maximum(x, 1e-30)
    denom = jnp.where(jnp.abs(a) > 1e-10, a, 1.0)
    recur = (g_a1 - jnp.power(x_safe, a) * jnp.exp(-x)) / denom

    return jnp.where(a > 0.0, direct, recur)


@jax.jit
def clf_satellite_mean(log10m, log10l_lim, log10l0, log10m1, alpha_cen,
                       beta_cen, sigma_c, alpha_faint, phi_s_amp):
    r""":math:`\langle N_{\rm sat}(>L)\rangle = \tfrac{\phi_s^*}{2}
    \Gamma[(\alpha_s{+}1)/2,\ (L/L_c)^2]`.

    Closed form, from substituting :math:`t = (L/L_c)^2` in the modified
    Schechter.  The ``maximum`` at the end is a floor on a *count*: the
    incomplete gamma can return a tiny negative from cancellation deep in its
    tail, and a negative occupation is not a small error, it is a different
    quantity.
    """
    lc = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    x = jnp.power(10.0, jnp.asarray(log10l_lim) - lc)
    g = upper_gamma((alpha_faint + 1.0) / 2.0, x * x) / 2.0
    return phi_s_star(log10m, log10m1, sigma_c, phi_s_amp) * jnp.maximum(g, 0.0)


# =========================================================================
# The Cacciato 2013 / van den Bosch 2013 normalisation
# =========================================================================

@jax.jit
def phi_s_star_cacciato13(log10m, log10m1, b0, b1, b2):
    r""":math:`\log_{10}\phi_s^* = b_0 + b_1 x + b_2 x^2`, :math:`x = \log(M/M_1)`."""
    x = jnp.asarray(log10m) - log10m1
    return jnp.power(10.0, b0 + b1 * x + b2 * x * x)


@jax.jit
def phi_sat_cacciato09(log10m, b_0, b_1, b_2):
    r"""The same quadratic, pivoted at :math:`10^{12}M_\odot/h` instead of
    :math:`M_1`: Cacciato et al. (2009) Eq. 40, van den Bosch et al. (2013)
    Eq. 79.

    Kept separate from :func:`phi_s_star_cacciato13` rather than given a pivot
    argument: the two pivots go with different fitted coefficients, and a
    single function with a default pivot is how they get mixed.
    """
    x = jnp.asarray(log10m) - 12.0
    return jnp.power(10.0, b_0 + b_1 * x + b_2 * x ** 2)


@jax.jit
def alpha_faint_cacciato09(log10m, a_1, a_2, log_m_2):
    r"""Mass-dependent faint-end slope, Cacciato et al. (2009) Eq. 39,
    :math:`-2 + a_1[1 - \tfrac{2}{\pi}\arctan(a_2(\log M - \log M_2))]`.

    Present in the paper and **not wired into any class** in the predecessor --
    defined, exported, and called by nothing.  Kept here because it is a
    published option, and named so that its absence from the defaults is
    visible.
    """
    return -2.0 + a_1 * (1.0 - 2.0 / jnp.pi
                         * arctan(a_2 * (jnp.asarray(log10m) - log_m_2)))


@jax.jit
def clf_central_vdb13(log10m, log10l_lim, log10l0, log10m1, alpha_cen,
                      beta_cen, sigma_c):
    """van den Bosch et al. (2013) centrals -- the Cacciato form."""
    return clf_central_mean(log10m, log10l_lim, log10l0, log10m1, alpha_cen,
                            beta_cen, sigma_c)


@jax.jit
def clf_satellite_vdb13(log10m, log10l_lim, log10l0, log10m1, alpha_cen,
                        beta_cen, alpha_faint, b_0, b_1, b_2,
                        f_s_star=L_S_OVER_L_C):
    r"""van den Bosch et al. (2013) satellites, Eqs. 74 and 77--79.

    Differs from :func:`clf_satellite_mean` in two ways that go together: the
    normalisation is the quadratic :func:`phi_sat_cacciato09` rather than
    :math:`\propto M`, and the Schechter cut-off sits at
    :math:`L_s^* = 0.562\,L_c` rather than at :math:`L_c`.  Both are
    Cacciato et al. (2009)'s own (their Eqs. 38 and 40), so this is the
    published form and ``clf_satellite_mean`` the simplification; van den
    Bosch et al. differ from Cacciato et al. only in holding
    :math:`\alpha_s` constant (their Eq. 78) where Cacciato et al. let it run
    with mass (:func:`alpha_faint_cacciato09`).
    """
    lc = log10_lc(log10m, log10l0, log10m1, alpha_cen, beta_cen)
    ls = lc + jnp.log10(f_s_star)
    x = jnp.power(10.0, jnp.asarray(log10l_lim) - ls)
    g = upper_gamma((alpha_faint + 1.0) / 2.0, x * x) / 2.0
    return phi_sat_cacciato09(log10m, b_0, b_1, b_2) * jnp.maximum(g, 0.0)


# =========================================================================
# Registry
# =========================================================================

CLF = {
    "cacciato09": (clf_central_mean, clf_satellite_mean),
    "vandenbosch13": (clf_central_vdb13, clf_satellite_vdb13),
}

#: What each was fitted to, and in what variable.
CLF_CALIBRATION = {
    "cacciato09": Calibration("SDSS group catalogue", (0.0, 0.2), "luminosity"),
    "vandenbosch13": Calibration("SDSS, CLF + lensing + clustering", (0.0, 0.2),
                                 "luminosity"),
}

#: Neither entry is a published fit.  ``cacciato09`` is illustrative, close to
#: but not Cacciato et al. (2009) Table 3 (WMAP3: log L0 9.935, log M1 11.07,
#: gamma_1 3.273, gamma_2 0.255, sigma_c 0.143), on the simplified form above.
#: ``vandenbosch13`` is the CLF van den Bosch et al. (2013) populate their mocks
#: with (their Sec. 4.1), not a fit to data.
CLF_DEFAULTS = {
    "cacciato09": dict(log10l_lim=9.5, log10l0=9.94, log10m1=11.0,
                       alpha_cen=2.95, beta_cen=0.18, sigma_c=0.15,
                       alpha_faint=-1.15, phi_s_amp=9.0),
    "vandenbosch13": dict(log10l_lim=9.5, log10l0=9.9, log10m1=10.9,
                          alpha_cen=5.0, beta_cen=0.24, sigma_c=0.16,
                          alpha_faint=-1.3, b_0=-1.2, b_1=1.4, b_2=-0.17),
}


def make_clf(name: str):
    """Look up a CLF by name; returns ``(central, satellite)``."""
    key = str(name).lower()
    if key not in CLF:
        raise ValueError(f"unknown CLF {name!r}; expected one of {sorted(CLF)}")
    return CLF[key]


def clf_defaults(name: str) -> dict:
    """The default parameters for one CLF (see ``CLF_DEFAULTS``)."""
    key = str(name).lower()
    if key not in CLF_DEFAULTS:
        raise ValueError(f"unknown CLF {name!r}; expected one of "
                         f"{sorted(CLF_DEFAULTS)}")
    return dict(CLF_DEFAULTS[key])
