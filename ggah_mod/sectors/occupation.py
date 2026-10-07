r"""Galaxy occupation: how many galaxies a halo of mass :math:`M` hosts.

Thirteen published parameterisations and variants (:data:`OCCUPATION`), on one
registry, in the shape layer 2 uses for its eighteen multiplicity functions and
six bias fits: jitted free functions, a ``dict``, a ``make_*`` factory and a
parallel calibration table saying what each was fitted to.

Every one returns the pair :math:`(\langle N_{\rm cen}\rangle,
\langle N_{\rm sat}\rangle)` on a :math:`\log_{10}M` grid, because the two enter
the one-halo term differently: a central sits at the halo centre and a satellite
follows a profile, and a central cannot pair with itself.  Returning only the
total would make that distinction unrecoverable downstream -- which is the same
reason :class:`~ggah_mod.sectors.protocol.TracerWeights` keeps its point and
extended parts apart.

The three sigmas are three different quantities
-----------------------------------------------

It is tempting to read ``base.py``'s :math:`\tfrac12\,{\rm erfc}[(\log M_{\min}
- \log M)/\sigma]` next to Zu & Mandelbaum's :math:`\tfrac12\,{\rm erfc}[\ldots
/(\sqrt2\,\sigma)]` and conclude that one of them has lost a :math:`\sqrt2`.
They have not, and **unifying them would be the bug**:

* **Zheng et al. (2007) Eq. 2** defines :math:`\sigma_{\log M}` as the width
  parameter of the error function itself.  There is no :math:`\sqrt2`, by
  definition of the symbol.
* **Zu & Mandelbaum (2015) Eq. 21** has a genuine log-normal scatter in
  :math:`\ln M_*`, so the :math:`\sqrt2` is the one that always accompanies a
  Gaussian CDF written with ``erfc``, and the :math:`\ln 10` converts the
  threshold from :math:`\log_{10}` into :math:`\ln`.
* **Leauthaud et al. (2012) Eq. 8** is the same log-normal, in
  :math:`\log_{10}M_*`, so it carries the :math:`\sqrt2` and no :math:`\ln 10`.

And a **fourth** convention, which is a width rather than a scatter:

* **Zacharegkas & Chang (2025) Eq. 2** writes the same log-normal in
  :math:`\log_{10}M_*` with the :math:`\sqrt2` *absorbed into the symbol*, so
  its width is :math:`\sqrt2` times Leauthaud's :math:`\sigma`.  Numerically
  identical models; the number you fit is not the same number.  (Guo et al.
  2018, 2019 do not: their :math:`\sigma_c = 0.173` is a Gaussian
  :math:`\sigma`, Leauthaud's convention.  ``n_cen_guo18`` takes the
  ``width_logmstar`` convention all the same, so its width is
  :math:`\sqrt2\times` theirs.)

Four symbols spelled two ways in the literature and meaning four things.  So the
parameters are *named* differently here:

=========================  ==================  =========================
name                       variable            convention
=========================  ==================  =========================
``sigma_logm``             :math:`\log_{10}M_h`  erf width, no :math:`\sqrt2`
``sigma_logmstar``         :math:`\log_{10}M_*`  Gaussian :math:`\sigma`
``sigma_lnmstar``          :math:`\ln M_*`       Gaussian :math:`\sigma`
``width_logmstar``         :math:`\log_{10}M_*`  erf width, :math:`= \sqrt2\sigma`
=========================  ==================  =========================

so a fit that swaps two of them gets a ``TypeError`` rather than a 40% error in
the central occupation.

**Two of these were the same name until PLAN.md item G5.**  ``leauthaud12``
spelled its stellar-mass scatter ``sigma_logm`` -- the *halo*-mass name, shared
with ``zheng07``, ``kravtsov04`` and the two ``more15`` variants -- so those were
the one pair the module's own rule did not cover, and they could be swapped
silently.  The fourth row was called ``sigma_logm_star``, which differs from
``sigma_logmstar`` by one underscore and would have replaced a collision with a
typo.  ``width_`` says what it is instead.

Differentiable, including through the inverses
----------------------------------------------

Four of these models need :math:`M_*(M_h)` from a relation published as
:math:`M_h(M_*)`.  The predecessor inverted each with a bare bisection, whose
gradient is **exactly zero** -- see :mod:`ggah_mod.sectors.sham`.  Every
inversion here goes through :func:`~ggah_mod.sectors.sham.invert_monotone`, so
the shape parameters of the stellar-mass relation are differentiable, which is
what a forecast that varies them requires.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from .calibration import Calibration
from .sham import (
    invert_monotone, mh_from_mstar_kravtsov18, mh_from_mstar_leauthaud12,
    mh_from_mstar_zu15, mstar_from_mh_kravtsov18, mstar_from_mh_leauthaud12,
    mstar_from_mh_zu15, scatter_zu15,
)

__all__ = [
    "n_cen_zheng07", "n_sat_zheng07", "n_sat_kravtsov04",
    "n_cen_lange25", "n_sat_lange25",
    "f_inc_more15", "n_cen_more15", "n_sat_more15",
    "n_cen_more15_const", "n_sat_more15_const",
    "shmr_guo18", "completeness_guo", "n_cen_guo18", "n_sat_guo18",
    "quenched_fraction_guo19", "n_cen_guo19", "n_sat_guo19",
    "quenched_fraction_zu16", "n_cen_zu16_red", "n_cen_zu16_blue",
    "n_sat_zu16_red", "n_sat_zu16_blue",
    "n_cen_zu15", "n_sat_zu15",
    "n_cen_leauthaud12", "n_sat_leauthaud12",
    "n_cen_zacharegkas25", "n_sat_zacharegkas25",
    "shmr_vanuitert16", "n_cen_vanuitert16", "n_sat_vanuitert16",
    "OCCUPATION", "OCC_CALIBRATION", "make_occupation", "occupation_defaults",
    "DEFAULTS", "ZU15_PUBLISHED", "M_S_OVER_M_C",
]

_LN10 = jnp.log(10.0)
_SQRT2 = jnp.sqrt(2.0)


# =========================================================================
# Zheng et al. (2005, 2007) -- the kernel everything else decorates
# =========================================================================

@jax.jit
def n_cen_zheng07(log10m, log10mmin, sigma_logm):
    r"""Zheng et al. (2007) Eq. 2.

    .. math::

        \langle N_{\rm cen}\rangle = \tfrac12\left[1 +
            {\rm erf}\frac{\log M - \log M_{\min}}{\sigma_{\log M}}\right]
        = \tfrac12\,{\rm erfc}\frac{\log M_{\min} - \log M}{\sigma_{\log M}}

    **No** :math:`\sqrt2`: :math:`\sigma_{\log M}` is defined as the error
    function's own width here, not as a log-normal scatter.  See the module
    docstring.
    """
    return 0.5 * erfc((log10mmin - jnp.asarray(log10m)) / sigma_logm)


@jax.jit
def n_sat_zheng07(log10m, log10mmin, sigma_logm, log10m0, log10m1, alpha):
    r"""Zheng et al. (2007) Eq. 5, gated by the central occupation.

    .. math::

        \langle N_{\rm sat}\rangle = \langle N_{\rm cen}\rangle
            \left(\frac{M - M_0}{M_1}\right)^{\alpha},\quad M > M_0

    The ``where`` on the mass, rather than a ``clip`` on the ratio, is
    deliberate: below :math:`M_0` the bracket is negative and raising it to a
    fractional power gives NaN, while a ``clip`` at zero would put a tie exactly
    at :math:`M_0` and split the gradient there.
    """
    nc = n_cen_zheng07(log10m, log10mmin, sigma_logm)
    m = jnp.power(10.0, jnp.asarray(log10m))
    m0, m1 = jnp.power(10.0, log10m0), jnp.power(10.0, log10m1)
    ratio = jnp.where(m > m0, (m - m0) / m1, 0.0)
    return nc * jnp.power(ratio, alpha)


@jax.jit
def n_sat_kravtsov04(log10m, log10mmin, sigma_logm, log10m0, log10m1, alpha):
    r"""A power law with an **exponential** cutoff; the name is historical.

    .. math::

        \langle N_{\rm sat}\rangle = \langle N_{\rm cen}\rangle
            \left(\frac{M}{M_1}\right)^{\alpha} e^{-M_0/M}

    Not Kravtsov et al. (2004), whose Eq. 18 is
    :math:`(M/M_1 - C)^{\beta}`, with no exponential.  The exponential roll-off
    is Conroy, Wechsler & Kravtsov (2006) Eq. 7, after Tinker et al. (2005),
    at :math:`\alpha = 1` and without the central gate; with both it is
    Zu & Mandelbaum (2015) Eq. 22.

    Smooth at the cutoff where :func:`n_sat_zheng07` has a kink, which is the
    whole difference between them.  ``M_0`` is ``Mcut`` and ``M_1`` is ``Msat``
    in the ``aum`` naming.
    """
    nc = n_cen_zheng07(log10m, log10mmin, sigma_logm)
    log10m = jnp.asarray(log10m)
    ratio = jnp.power(10.0, alpha * (log10m - log10m1))
    cutoff = jnp.exp(-jnp.power(10.0, log10m0 - log10m))
    return nc * ratio * cutoff


# =========================================================================
# More et al. (2015) -- incompleteness
# =========================================================================

@jax.jit
def f_inc_more15(log10m, alpha_inc, log10m_inc):
    r"""More et al. (2015) Eq. 5: a linear incompleteness ramp, clipped to [0,1].

    .. math::  f_{\rm inc}(M) = \min\{1, \max[0, 1 + \alpha_{\rm inc}
                                (\log M - \log M_{\rm inc})]\}

    The clip is a genuine saturation of a physical fraction, not an
    interpolation guard, and the model is flat on both sides of it -- so the
    zero derivative there is the right answer rather than a lost one.
    """
    return jnp.clip(1.0 + alpha_inc * (jnp.asarray(log10m) - log10m_inc),
                    0.0, 1.0)


@jax.jit
def n_cen_more15(log10m, log10mmin, sigma_logm, alpha_inc, log10m_inc):
    """More et al. (2015): Zheng centrals scaled by the incompleteness."""
    return (f_inc_more15(log10m, alpha_inc, log10m_inc)
            * n_cen_zheng07(log10m, log10mmin, sigma_logm))


@jax.jit
def n_sat_more15(log10m, log10mmin, sigma_logm, log10m1, alpha, kappa,
                 alpha_inc, log10m_inc):
    r"""More et al. (2015): the satellite threshold is :math:`\kappa M_{\min}`.

    Replaces Zheng's independent :math:`M_0` with a multiple of the central
    cut-off mass, which removes one parameter and ties the two scales together.
    """
    nc = n_cen_more15(log10m, log10mmin, sigma_logm, alpha_inc, log10m_inc)
    m = jnp.power(10.0, jnp.asarray(log10m))
    m_thresh = kappa * jnp.power(10.0, log10mmin)
    m1 = jnp.power(10.0, log10m1)
    ratio = jnp.where(m > m_thresh, (m - m_thresh) / m1, 0.0)
    return nc * jnp.power(ratio, alpha)


@jax.jit
def n_cen_more15_const(log10m, log10mmin, sigma_logm, f_inc):
    r"""More et al. (2015) with a **mass-independent** :math:`f_{\rm inc}`.

    A scalar duty cycle rather than a ramp: a fixed fraction of hosts is
    active at any time.
    """
    return f_inc * n_cen_zheng07(log10m, log10mmin, sigma_logm)


@jax.jit
def n_sat_more15_const(log10m, log10mmin, sigma_logm, log10m1, alpha, kappa,
                       f_inc):
    """Satellites for :func:`n_cen_more15_const`; the same duty cycle applies."""
    nc = n_cen_more15_const(log10m, log10mmin, sigma_logm, f_inc)
    m = jnp.power(10.0, jnp.asarray(log10m))
    m_thresh = kappa * jnp.power(10.0, log10mmin)
    ratio = jnp.where(m > m_thresh,
                      (m - m_thresh) / jnp.power(10.0, log10m1), 0.0)
    return nc * jnp.power(ratio, alpha)


# =========================================================================
# Guo et al. (2018, 2019) -- the incomplete conditional stellar mass function
# =========================================================================

@jax.jit
def shmr_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr):
    r"""Guo et al. (2018) broken power-law SHMR, **forward**.

    .. math::

        \log_{10}\langle M_*\rangle = \log_{10}M_{*0}
          + (\alpha+\beta)x - \beta\log_{10}(1 + 10^{x}),
        \quad x = \log_{10}(M/M_1)
    """
    x = jnp.asarray(log10m) - log10m1_shmr
    return (log10m_star0 + (alpha_shmr + beta_shmr) * x
            - beta_shmr * jnp.logaddexp(x * _LN10, 0.0) / _LN10)


@jax.jit
def completeness_guo(log10m_star, f_comp, log10m_star_min, sigma_c):
    r"""Guo et al. (2018) completeness:
    :math:`\tfrac{f}{2}[1 + {\rm erf}((\log M_* - \log M_{*,\min})/\sigma_c)]`."""
    return 0.5 * f_comp * (
        1.0 + erf((jnp.asarray(log10m_star) - log10m_star_min) / sigma_c))


@jax.jit
def n_cen_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
                width_logmstar, f_cen, log10m_star_min_cen, sigma_c_cen):
    r"""Guo et al. (2018) centrals.

    The completeness is evaluated at the *mean* SHMR rather than integrated
    over its scatter -- an analytic approximation, and this package's, not the
    source's: Guo et al. integrate the log-normal times the completeness over
    :math:`M_*` (their Eqs. 13--14).  The two agree where the completeness
    varies slowly across the scatter.
    """
    ms = shmr_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr)
    c = completeness_guo(ms, f_cen, log10m_star_min_cen, sigma_c_cen)
    return c * 0.5 * erfc((log10m_star_min_cen - ms) / width_logmstar)


@jax.jit
def n_sat_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
                f_sat, log10m_star_min_sat, sigma_c_sat, log10m1_sat,
                alpha_sat):
    """Guo et al. (2018) satellites: a power law times the completeness."""
    ms = shmr_guo18(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr)
    c = completeness_guo(ms, f_sat, log10m_star_min_sat, sigma_c_sat)
    return c * jnp.power(10.0, alpha_sat * (jnp.asarray(log10m) - log10m1_sat))


@jax.jit
def quenched_fraction_guo19(log10m, log10m_q):
    r"""Guo et al. (2019): :math:`f_q = 1/(1 + M_q/M)`, so
    :math:`f_{\rm sf} = 1 - f_q`.

    Note the direction: :math:`f_q \to 1` for :math:`M \gg M_q`, so the
    star-forming fraction is suppressed in massive halos -- which is what makes
    this the ELG-flavoured occupation.  Their printed Eq. 10 has
    :math:`M/M_q` in place of :math:`M_q/M`, which would make :math:`f_q`
    fall with mass; their Fig. 5 has it rise, as here.
    """
    return 1.0 / (1.0 + jnp.power(10.0, log10m_q - jnp.asarray(log10m)))


@jax.jit
def n_cen_guo19(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
                width_logmstar, f_cen, log10m_star_min_cen, sigma_c_cen,
                log10m_q):
    """Guo et al. (2019) ELG centrals: Guo18 times the star-forming fraction."""
    return (1.0 - quenched_fraction_guo19(log10m, log10m_q)) * n_cen_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
        width_logmstar, f_cen, log10m_star_min_cen, sigma_c_cen)


@jax.jit
def n_sat_guo19(log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr,
                f_sat, log10m_star_min_sat, sigma_c_sat, log10m1_sat,
                alpha_sat, log10m_q):
    """Guo et al. (2019) ELG satellites."""
    return (1.0 - quenched_fraction_guo19(log10m, log10m_q)) * n_sat_guo18(
        log10m, log10m_star0, log10m1_shmr, alpha_shmr, beta_shmr, f_sat,
        log10m_star_min_sat, sigma_c_sat, log10m1_sat, alpha_sat)


# =========================================================================
# Zu & Mandelbaum (2015) -- the workhorse
# =========================================================================

def n_cen_lange25(log10m, log10mmin, sigma_logm, f_gamma):
    r"""Lange et al. (2025): a Zheng07 central with a completeness factor.

    .. math::

        \langle N_{\rm cen}\rangle = f_\Gamma \cdot
            \tfrac12\,{\rm erfc}\frac{\log M_{\min} - \log M}{\sigma_{\log M}}

    :math:`f_\Gamma` is the central *completeness* of the DESI DR1 sample --
    the fraction of haloes above threshold whose central actually enters the
    catalogue.  It multiplies the occupation and therefore the number density;
    it is not a statement about the halo.

    Lange et al. (2025), arXiv:2512.15962, DESI DR1 clustering + lensing.
    """
    return f_gamma * n_cen_zheng07(log10m, log10mmin, sigma_logm)


def n_sat_lange25(log10m, log10mmin, sigma_logm, log10m0, log10m1, alpha,
                  f_gamma):
    r"""Satellites for :func:`n_cen_lange25`: :func:`n_sat_kravtsov04`,
    **undecorated**.

    **Not Lange et al.'s satellites.**  Their Eq. 7 is
    :math:`((M - M_0)/M_1)^{\alpha}`, Zheng's form without the central gate
    (Table 1: :math:`M_0` is the mass below which
    :math:`\langle N_{\rm sat}\rangle = 0`).  This package uses the
    exponential cut-off and the gate of :func:`n_sat_kravtsov04` instead.

    ``f_gamma`` is accepted and deliberately unused, so that the pair of
    functions takes one parameter set and the registry's signature check has
    something to bind.  The completeness applies to centrals only: it is a
    selection property of the central galaxy, and applying it twice would put
    :math:`f_\Gamma^2` into the satellite term of every pair count.

    **The assembly-bias half of Lange et al. (2025) is not here.**  They fit
    the Hearin et al. (2016) decorated HOD, which splits haloes of one mass by
    concentration and gives the two halves different occupations, evaluated on
    AbacusSummit.  This package does not implement it, so this model is its
    occupation and nothing more.
    """
    del f_gamma
    return n_sat_kravtsov04(log10m, log10mmin, sigma_logm, log10m0, log10m1,
                            alpha)


def n_cen_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
               gamma, sigma_lnmstar, eta, fc):
    r"""Zu & Mandelbaum (2015) Eq. 21: centrals above a stellar-mass threshold.

    .. math::

        \langle N_{\rm cen}^{>M_*}\rangle = \frac{f_c}{2}\,{\rm erfc}\!
        \left[\frac{(\log M_*^{\rm t} - \log M_*^c(M_h))\ln 10}
                   {\sqrt2\,\sigma_{\ln M_*}(M_h)}\right]

    :math:`\sigma_{\ln M_*}` is a genuine log-normal width in :math:`\ln M_*`,
    hence both the :math:`\sqrt2` and the :math:`\ln 10`.  It is **not** the
    same symbol as Zheng's :math:`\sigma_{\log M}`.
    """
    ms = mstar_from_mh_zu15(log10m, lg_m1h, lg_m0star, beta, delta, gamma)
    sigma = scatter_zu15(log10m, sigma_lnmstar, eta, lg_m1h)
    return 0.5 * fc * erfc(
        (log10m_star_thresh - ms) * _LN10 / (_SQRT2 * sigma))


def n_sat_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
               gamma, sigma_lnmstar, eta, fc, bsat, beta_sat, bcut, beta_cut,
               alpha_sat):
    r"""Zu & Mandelbaum (2015) Eq. 22.

    .. math::

        \langle N_{\rm sat}^{>M_*}\rangle
        = \langle N_{\rm cen}^{>M_*}\rangle
          \left(\frac{M_h}{M_{\rm sat}}\right)^{\alpha_{\rm sat}}
          e^{-M_{\rm cut}/M_h}

    with :math:`M_{\rm sat}` and :math:`M_{\rm cut}` set by
    :math:`M_{\min} = f_{\rm SHMR}^{-1}(M_*^{\rm t})` -- a **forward**
    evaluation of Eq. 19, so no inversion is needed for this leg.
    """
    log10m_min = mh_from_mstar_zu15(log10m_star_thresh, lg_m1h, lg_m0star,
                                    beta, delta, gamma)
    m_min_norm = jnp.power(10.0, log10m_min - 12.0)
    msat = bsat * jnp.power(m_min_norm, beta_sat) * 1e12
    mcut = bcut * jnp.power(m_min_norm, beta_cut) * 1e12
    nc = n_cen_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
                    gamma, sigma_lnmstar, eta, fc)
    m_h = jnp.power(10.0, jnp.asarray(log10m))
    return nc * jnp.power(m_h / msat, alpha_sat) * jnp.exp(-mcut / m_h)


# =========================================================================
# Zu & Mandelbaum (2016) -- halo quenching on top of the 2015 iHOD
# =========================================================================
#
# Paper III of the series.  The 2015 model says how many galaxies a halo holds
# above a stellar-mass threshold; this says what colour they are, and it does so
# as a function of **halo mass alone** -- which is the paper's result rather
# than its assumption.  It compares that against a "hybrid" model in which
# centrals quench on stellar mass and finds halo quenching significantly
# preferred for the blue population above 1e11 h^-2 Msun, and correctly
# predicting the lensing-measured halo masses of massive blue centrals where
# stellar-mass quenching does not.
#
# Note the definition: Zu & Mandelbaum take M_h = M_200m throughout (their
# Sec. 1), which is the definition the default flavour now declares.  Pairing
# this with a 200c field would be the same category error the halo layer
# refuses one rung down.


def quenched_fraction_zu16(log10m, lg_mh_q, mu):
    r"""Zu \& Mandelbaum (2016) Eqs. 12 and 13: the red fraction.

    .. math::

        f_{\rm red}(M_h) = 1 - \exp\left[-\left(M_h/M_h^{q}\right)^{\mu}\right]

    One functional form, used twice with different parameters -- once for
    centrals (:math:`M_h^{qc}, \mu_c`) and once for satellites
    (:math:`M_h^{qs}, \mu_s`).  The paper's point is that the two critical
    masses come out nearly equal (:math:`\lg M_h^{q} \simeq 12.2` for both)
    while the indices do not (:math:`\mu_c = 0.38` against
    :math:`\mu_s = 0.15`): the same threshold, a more gradual transition for
    satellites.  Keeping them as four free parameters rather than two is what
    lets a fit say so.
    """
    x = jnp.power(10.0, jnp.asarray(log10m) - jnp.asarray(lg_mh_q))
    return -jnp.expm1(-jnp.power(x, jnp.asarray(mu)))


def n_cen_zu16_red(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
                   gamma, sigma_lnmstar, eta, fc, lg_mh_qc, mu_c):
    r"""Red centrals: the 2015 occupation times the 2016 red fraction."""
    return n_cen_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta,
                      delta, gamma, sigma_lnmstar, eta, fc) * \
        quenched_fraction_zu16(log10m, lg_mh_qc, mu_c)


def n_cen_zu16_blue(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
                    gamma, sigma_lnmstar, eta, fc, lg_mh_qc, mu_c):
    r"""Blue centrals: the complement, so red and blue sum to the 2015 total."""
    return n_cen_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta,
                      delta, gamma, sigma_lnmstar, eta, fc) * \
        (1.0 - quenched_fraction_zu16(log10m, lg_mh_qc, mu_c))


def n_sat_zu16_red(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
                   gamma, sigma_lnmstar, eta, fc, bsat, beta_sat, bcut,
                   beta_cut, alpha_sat, lg_mh_qs, mu_s):
    r"""Red satellites, with their own critical mass and index."""
    return n_sat_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta,
                      delta, gamma, sigma_lnmstar, eta, fc, bsat, beta_sat,
                      bcut, beta_cut, alpha_sat) * \
        quenched_fraction_zu16(log10m, lg_mh_qs, mu_s)


def n_sat_zu16_blue(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta, delta,
                    gamma, sigma_lnmstar, eta, fc, bsat, beta_sat, bcut,
                    beta_cut, alpha_sat, lg_mh_qs, mu_s):
    r"""Blue satellites."""
    return n_sat_zu15(log10m, log10m_star_thresh, lg_m1h, lg_m0star, beta,
                      delta, gamma, sigma_lnmstar, eta, fc, bsat, beta_sat,
                      bcut, beta_cut, alpha_sat) * \
        (1.0 - quenched_fraction_zu16(log10m, lg_mh_qs, mu_s))


# =========================================================================
# Leauthaud et al. (2012)
# =========================================================================

def n_cen_leauthaud12(log10m, log10m_star_thresh, log10m1, log10m_star0, beta,
                      delta, gamma, sigma_logmstar):
    r"""Leauthaud et al. (2012) Eq. 8.

    A log-normal in :math:`\log_{10}M_*`, so the :math:`\sqrt2` is present and
    there is no :math:`\ln 10`.
    """
    ms = mstar_from_mh_leauthaud12(log10m, log10m1, log10m_star0, beta, delta,
                                   gamma)
    return 0.5 * erfc((log10m_star_thresh - ms) / (_SQRT2 * sigma_logmstar))


def n_sat_leauthaud12(log10m, log10m_star_thresh, log10m1, log10m_star0, beta,
                      delta, gamma, sigma_logmstar, log10m_sat, log10m_cut,
                      alpha_sat):
    r"""Leauthaud et al. (2012) Eq. 12: the Zu & Mandelbaum satellite form."""
    nc = n_cen_leauthaud12(log10m, log10m_star_thresh, log10m1, log10m_star0,
                           beta, delta, gamma, sigma_logmstar)
    m_h = jnp.power(10.0, jnp.asarray(log10m))
    return (nc * jnp.power(m_h / jnp.power(10.0, log10m_sat), alpha_sat)
            * jnp.exp(-jnp.power(10.0, log10m_cut) / m_h))


# =========================================================================
# Zacharegkas & Chang (2025), on the Kravtsov et al. (2018) SHMR
# =========================================================================

def n_cen_zacharegkas25(log10m, log10m_star_thresh, log10m1_shmr, log10eps,
                        alpha_shmr_k18, gamma_shmr, delta_shmr, width_logmstar,
                        f_cen):
    r"""Zacharegkas & Chang (2025) Eq. 2: centrals above a threshold."""
    ms = mstar_from_mh_kravtsov18(log10m, log10m1_shmr, log10eps, alpha_shmr_k18,
                                  gamma_shmr, delta_shmr)
    return 0.5 * f_cen * (
        1.0 + erf((ms - log10m_star_thresh) / width_logmstar))


def n_sat_zacharegkas25(log10m, log10m_star_thresh, log10m1_shmr, log10eps,
                        alpha_shmr_k18, gamma_shmr, delta_shmr, alpha_sat, kappa,
                        B_sat, beta_sat, B_cut, beta_cut, f_sat):
    r"""Zacharegkas & Chang (2025) Eqs. 5--7.

    Note what is **absent**: unlike Zheng, More and Zu & Mandelbaum, this
    satellite term is *not* gated by the central occupation, so it takes
    neither ``width_logmstar`` nor ``f_cen``.  They are left out of the
    signature rather than accepted and ignored -- a parameter a function
    quietly does nothing with is how a fit ends up reporting a constraint on
    something it never used.
    """
    log10m_min = mh_from_mstar_kravtsov18(log10m_star_thresh, log10m1_shmr,
                                          log10eps, alpha_shmr_k18, gamma_shmr,
                                          delta_shmr)
    m_min = jnp.power(10.0, log10m_min)
    m_h = jnp.power(10.0, jnp.asarray(log10m))
    m_min_norm = m_min / 1e12
    m_sat = B_sat * jnp.power(m_min_norm, beta_sat) * 1e12
    m_cut = B_cut * jnp.power(m_min_norm, beta_cut) * 1e12
    ratio = jnp.where(m_h > kappa * m_min, (m_h - kappa * m_min) / m_sat, 0.0)
    return (f_sat * jnp.power(ratio, alpha_sat) * jnp.exp(-m_cut / m_h))


# =========================================================================
# van Uitert et al. (2016) -- a conditional stellar mass function in a bin
# =========================================================================

#: Nodes for the satellite modified-Schechter integral.  Static, and built with
#: `jnp` rather than `np` so nothing numpy-shaped appears inside a traced
#: function -- see `tests/test_sectors_paths.py`.
_N_CSMF_NODES = 128


@jax.jit
def shmr_vanuitert16(log10m, log10m_h1, log10m_star0, beta1, log10_beta2):
    r"""van Uitert et al. (2016) Eq. 16, a double power law.

    .. math::

        M_*^c = M_{*0}\,\frac{(M_h/M_{h,1})^{\beta_1}}
                             {\left[1 + M_h/M_{h,1}\right]^{\beta_1-\beta_2}},
        \quad\text{i.e.}\quad
        \log_{10}M_*^c = \log_{10}M_{*0} + \beta_1 x
            - (\beta_1-\beta_2)\log_{10}\!\left(1 + 10^{x}\right),
        \quad x = \log_{10}(M_h/M_{h,1})

    Slope :math:`\beta_1` below :math:`M_{h,1}` and :math:`\beta_2` above.
    The exponent :math:`\beta_1-\beta_2` multiplies the logarithm: it is
    their Eq. B1 at :math:`\beta_3 = 1`.  Up to 1.0.0 it sat *inside* it, as
    :math:`\log_{10}[1 + 10^{(\beta_1-\beta_2)x}]` -- the same two asymptotic
    slopes and a different turnover, :math:`(\beta_1-\beta_2-1)\log_{10}2`
    high at :math:`M_{h,1}`, 1.1 dex at the defaults.

    :math:`\beta_2` is sampled as :math:`\log_{10}\beta_2` so it cannot go
    negative, which would turn the high-mass end over.
    """
    x = jnp.asarray(log10m) - log10m_h1
    beta2 = jnp.power(10.0, log10_beta2)
    return (log10m_star0 + beta1 * x
            - (beta1 - beta2) * jnp.logaddexp(x * _LN10, 0.0) / _LN10)


@jax.jit
def n_cen_vanuitert16(log10m, log10m_star_lo, log10m_star_hi, log10m_h1,
                      log10m_star0, beta1, log10_beta2, sigma_c):
    r"""van Uitert et al. (2016) Eqs. 11 and 15: centrals in a stellar-mass **bin**.

    The difference of two Gaussian CDFs, clipped into :math:`[0,1]`: a central
    is either in the bin or not, so the occupation cannot exceed one.
    """
    mu = shmr_vanuitert16(log10m, log10m_h1, log10m_star0, beta1, log10_beta2)
    hi = erf((log10m_star_hi - mu) / (_SQRT2 * sigma_c))
    lo = erf((log10m_star_lo - mu) / (_SQRT2 * sigma_c))
    return jnp.clip(0.5 * (hi - lo), 0.0, 1.0)


#: :math:`M_*^s/M_*^c`, van Uitert et al. (2016) Eq. 18.
#:
#: **The same fitted offset as** :data:`~ggah_mod.sectors.clf.L_S_OVER_L_C`,
#: which is 0.562: one paper writes it in luminosity to three figures and the
#: other in stellar mass to two.  Both figures are kept, each in the module
#: transcribing its own source, because a transcription that adopts another
#: paper's precision is no longer a transcription -- the same rule
#: :func:`~ggah_mod.sectors.agn.xlf_in_h_units` follows about units.
#:
#: The difference is 0.4 per cent, far below anything either fit constrains, so
#: nothing here turns on which is used.  What turned on it was a reader being
#: unable to tell a typo from two quantities, which is what naming both fixes.
#: ``PLAN.md`` item **G4**.
M_S_OVER_M_C = 0.56


@jax.jit
def n_sat_vanuitert16(log10m, log10m_star_lo, log10m_star_hi, log10m_h1,
                      log10m_star0, beta1, log10_beta2, alpha_s, b0, b1,
                      f_s_star=M_S_OVER_M_C):
    r"""van Uitert et al. (2016) Eqs. 17--18: a modified Schechter, integrated.

    .. math::

        \Phi_s(M_*|M_h)\,dM_* = \frac{\phi_s}{M_*^s}
            \left(\frac{M_*}{M_*^s}\right)^{\alpha_s}
            e^{-(M_*/M_*^s)^2}dM_*,
        \quad M_*^s = 0.56\,M_*^c(M_h)

    **The measure is** :math:`dM_*`, **not** :math:`dM_*/M_*`.  Substituting
    :math:`u = M_*/M_*^s` and integrating in :math:`\log_{10}M_*` gives
    :math:`\int \phi_s\,u^{\alpha_s+1}e^{-u^2}\ln 10\;d\log_{10}M_*` -- note the
    :math:`\alpha_s+1`, which is the Jacobian, not a typo.  Reading the
    exponent off a "``dM_*/M_*``" statement of the same equation drops it and
    leaves the satellite occupation wrong by a factor of order :math:`u`, which
    varies across the bin and so cannot be absorbed into :math:`\phi_s`.

    The integral has no closed form for general :math:`\alpha_s`, so it is a
    fixed 128-node trapezoid in :math:`\log_{10}M_*`.  Fixed rather than
    adaptive because the node count sets an array shape, and a shape that
    depended on the parameters would stop the function jitting.
    """
    mu = shmr_vanuitert16(log10m, log10m_h1, log10m_star0, beta1, log10_beta2)
    log10_ms_star = mu + jnp.log10(f_s_star)
    phi_s = jnp.power(10.0, b0 + b1 * (jnp.asarray(log10m) - 13.0))

    t = jnp.linspace(0.0, 1.0, _N_CSMF_NODES)
    grid = log10m_star_lo + t * (log10m_star_hi - log10m_star_lo)   # (N,)
    u = jnp.power(10.0, grid[..., :] - log10_ms_star[..., None])    # (..., N)
    u = jnp.maximum(u, 1e-30)                # alpha_s + 1 may be negative
    integrand = jnp.power(u, alpha_s + 1.0) * jnp.exp(-u ** 2)
    return phi_s * jnp.trapezoid(integrand, grid, axis=-1) * _LN10


# =========================================================================
# The registry
# =========================================================================

#: ``name -> (central, satellite)``.  Kept as the *pair* rather than a fused
#: total: the one-halo term needs them apart, because a central cannot pair with
#: itself and a satellite follows a profile the central does not.
OCCUPATION: dict[str, tuple[Callable, Callable]] = {
    "zheng07": (n_cen_zheng07, n_sat_zheng07),
    "kravtsov04": (n_cen_zheng07, n_sat_kravtsov04),
    "lange25": (n_cen_lange25, n_sat_lange25),
    "more15": (n_cen_more15, n_sat_more15),
    "more15_const": (n_cen_more15_const, n_sat_more15_const),
    "guo18": (n_cen_guo18, n_sat_guo18),
    "guo19": (n_cen_guo19, n_sat_guo19),
    "zumandelbaum15": (n_cen_zu15, n_sat_zu15),
    # The same iHOD split by colour with Paper III's halo quenching.  Two
    # entries rather than a `colour` argument because a model name is what the
    # calibration tables are keyed on, and red and blue are fitted together but
    # describe different selections.
    "zumandelbaum16_red": (n_cen_zu16_red, n_sat_zu16_red),
    "zumandelbaum16_blue": (n_cen_zu16_blue, n_sat_zu16_blue),
    "leauthaud12": (n_cen_leauthaud12, n_sat_leauthaud12),
    "zacharegkas25": (n_cen_zacharegkas25, n_sat_zacharegkas25),
    "vanuitert16": (n_cen_vanuitert16, n_sat_vanuitert16),
}

#: What the ``zumandelbaum`` iHOD defaults are the fit of.
_LS10_FIT = ("MAP of ggah_cal massbins_zu15_gt10.5_nbar-wp on ggah_mod 1.1.0.dev0 "
             "and sum_stat 0.6.0's products: five stellar-mass bins 10.6-12.0 "
             "log10 Msun, chi2 = 152.91 for 105 dof")


#: What each was fitted to, and the selection it describes.  Not decoration:
#: applying one outside its row is a systematic error, not a tolerance.
#: ``selection`` matters as much as the redshift range -- an occupation fitted
#: to a *threshold* sample and evaluated as a *bin* one is a different number.
OCC_CALIBRATION: dict[str, Calibration] = {
    "zheng07": Calibration("SDSS Main", (0.0, 0.25), "luminosity threshold"),
    "lange25": Calibration("DESI DR1 BGS and LRG", (0.1, 1.1),
                           "magnitude- or colour-selected tracer sample"),
    "kravtsov04": Calibration("N-body subhalos / aum", (0.0, 1.0),
                              "mass threshold"),
    "more15": Calibration("BOSS CMASS", (0.4, 0.7),
                          "stellar-mass threshold, incomplete"),
    "more15_const": Calibration("generic", (0.0, 2.0),
                                "threshold with a constant duty cycle"),
    "guo18": Calibration("BOSS LOWZ", (0.15, 0.43),
                         "stellar-mass, incomplete CSMF"),
    "guo19": Calibration("eBOSS ELG", (0.6, 1.1),
                         "star-forming, incomplete CSMF"),
    "zumandelbaum15": Calibration(
        "LS10 x DESI-BGS volume-limited sample, n_bar + w_p", (0.05, 0.18),
        "stellar-mass threshold; fitted as five bins, each N(>lo) - N(>hi)",
        notes=_LS10_FIT, mstar_range=(10.6, 12.0)),
    "zumandelbaum16_red": Calibration(
        "LS10 x DESI-BGS volume-limited sample, n_bar + w_p", (0.05, 0.18),
        "stellar-mass threshold, red",
        notes=_LS10_FIT + "; the red fraction is Paper III's SDSS fit",
        mstar_range=(10.6, 12.0)),
    "zumandelbaum16_blue": Calibration(
        "LS10 x DESI-BGS volume-limited sample, n_bar + w_p", (0.05, 0.18),
        "stellar-mass threshold, blue",
        notes=_LS10_FIT + "; the blue fraction is Paper III's SDSS fit",
        mstar_range=(10.6, 12.0)),
    "leauthaud12": Calibration("COSMOS", (0.22, 1.0), "stellar-mass threshold"),
    "zacharegkas25": Calibration("DES Y3", (0.2, 1.0),
                                 "stellar-mass bin or threshold"),
    "vanuitert16": Calibration("GAMA", (0.0, 0.5), "stellar-mass bin (CSMF)"),
}

#: Zu & Mandelbaum (2015) Paper I's own iHOD values, which were the
#: ``zumandelbaum`` defaults up to 0.8.4.
ZU15_PUBLISHED = dict(log10m_star_thresh=10.2, lg_m1h=12.10, lg_m0star=10.31,
                      beta=0.33, delta=0.42, gamma=1.21, sigma_lnmstar=0.50,
                      eta=-0.04, fc=0.86, bsat=8.98, beta_sat=0.90, bcut=0.86,
                      beta_cut=0.41, alpha_sat=1.00)

#: The iHOD the three ``zumandelbaum`` models default to: the LS10 fit, rounded
#: to three decimals, and the threshold of its sample, 10^10.5 Msun at
#: h = 0.6736 (``GalaxyParams``).  Refitted in 1.1.0 on sum_stat 0.6.0, whose
#: survey area raised every n_bar by 11-12 per cent; the 0.8.5 values, fitted
#: to the products it replaced, are ``ZU15_LS10_085``.
_ZU15_LS10 = dict(log10m_star_thresh=10.157, lg_m1h=12.289, lg_m0star=10.331,
                  beta=0.765, delta=0.781, gamma=0.496, sigma_lnmstar=0.538,
                  eta=-0.125, fc=0.874, bsat=12.53, beta_sat=0.856, bcut=0.750,
                  beta_cut=0.669, alpha_sat=1.083)

#: The ``zumandelbaum`` defaults of 0.8.5 to 1.1.0.dev0: the same fit on
#: sum_stat 0.5's products (chi2 = 17.60 for 103 dof), superseded by 0.6.0.
ZU15_LS10_085 = dict(log10m_star_thresh=10.157, lg_m1h=12.307, lg_m0star=10.325,
                     beta=0.792, delta=0.781, gamma=0.534, sigma_lnmstar=0.609,
                     eta=-0.175, fc=0.796, bsat=11.42, beta_sat=0.815,
                     bcut=1.747, beta_cut=0.711, alpha_sat=1.051)

#: Default parameters, one dict per model.  The ``zumandelbaum`` iHOD is the
#: LS10 fit.  The other ten are **not** published fits: ``zheng07`` is Zheng et
#: al. (2007)'s M_r < -18 fit with alpha = 1 for their 0.83; ``leauthaud12``
#: and ``zacharegkas25`` carry their papers' stellar-mass relations (Leauthaud
#: Table 5 z1, Zacharegkas Table 3) with illustrative occupation parameters;
#: ``vanuitert16`` mixes their Table 2 prior means (beta1, alpha_s, b0, b1)
#: with illustrative values; the rest are illustrative throughout.
DEFAULTS: dict[str, dict] = {
    "zheng07": dict(log10mmin=11.35, sigma_logm=0.25, log10m0=11.20,
                    log10m1=12.40, alpha=1.0),
    "kravtsov04": dict(log10mmin=13.0, sigma_logm=0.5, log10m0=13.5,
                       log10m1=14.0, alpha=1.0),
    # Illustrative, inside the prior ranges of Lange et al. (2025) Table 1;
    # the paper publishes posteriors, not a point fit.  `f_gamma = 1` is
    # the complete-sample limit, so the model reduces exactly to the
    # `kravtsov04` form -- which is the property `test_occupation.py` asserts
    # rather than a coincidence to notice later.
    "lange25": dict(log10mmin=13.0, sigma_logm=0.3, log10m0=13.5,
                    log10m1=14.0, alpha=1.0, f_gamma=1.0),
    "more15": dict(log10mmin=13.03, sigma_logm=0.38, log10m1=14.00, alpha=1.0,
                   kappa=1.0, alpha_inc=1.0, log10m_inc=13.0),
    "more15_const": dict(log10mmin=12.5, sigma_logm=0.8, log10m1=14.0,
                         alpha=0.8, kappa=0.3, f_inc=0.1),
    "guo18": dict(log10m_star0=10.7, log10m1_shmr=11.9, alpha_shmr=0.3,
                  beta_shmr=1.5, width_logmstar=0.15, f_cen=1.0,
                  log10m_star_min_cen=10.5, sigma_c_cen=0.1, f_sat=1.0,
                  log10m_star_min_sat=10.2, sigma_c_sat=0.2, log10m1_sat=13.0,
                  alpha_sat=1.0),
    "guo19": dict(log10m_star0=10.0, log10m1_shmr=11.5, alpha_shmr=0.3,
                  beta_shmr=1.5, width_logmstar=0.15, f_cen=0.5,
                  log10m_star_min_cen=9.8, sigma_c_cen=0.1, f_sat=0.3,
                  log10m_star_min_sat=9.5, sigma_c_sat=0.2, log10m1_sat=12.5,
                  alpha_sat=1.0, log10m_q=12.0),
    # The LS10 fit (`OCC_CALIBRATION`), not Paper I's; that is `ZU15_PUBLISHED`.
    "zumandelbaum15": dict(_ZU15_LS10),
    # The same iHOD, plus Paper III Table 2 (halo quenching, prior case):
    # lg M_h^qc = 12.20 +0.07/-0.08, mu_c = 0.38 +0.04/-0.03,
    # lg M_h^qs = 12.17 +0.12/-0.10, mu_s = 0.15 +0.03/-0.02.
    # The two critical masses agree and the two indices do not, which is the
    # paper's physical result and the reason all four stay free.
    "zumandelbaum16_red": dict(_ZU15_LS10, lg_mh_qc=12.20, mu_c=0.38,
                               lg_mh_qs=12.17, mu_s=0.15),
    "zumandelbaum16_blue": dict(_ZU15_LS10, lg_mh_qc=12.20, mu_c=0.38,
                                lg_mh_qs=12.17, mu_s=0.15),
    "leauthaud12": dict(log10m_star_thresh=10.0, log10m1=12.520,
                        log10m_star0=10.916, beta=0.457, delta=0.566,
                        gamma=1.53, sigma_logmstar=0.206, log10m_sat=12.500,
                        log10m_cut=11.500, alpha_sat=1.0),
    "zacharegkas25": dict(log10m_star_thresh=10.0, log10m1_shmr=11.506,
                          log10eps=-1.632, alpha_shmr_k18=-1.638,
                          gamma_shmr=0.596, delta_shmr=3.810,
                          width_logmstar=0.3, f_cen=1.0, alpha_sat=1.0,
                          kappa=1.0, B_sat=10.0, beta_sat=1.0, B_cut=5.0,
                          beta_cut=1.0, f_sat=1.0),
    "vanuitert16": dict(log10m_star_lo=9.8, log10m_star_hi=10.3,
                        log10m_h1=11.5, log10m_star0=10.5, beta1=5.0,
                        log10_beta2=-0.5, sigma_c=0.15, alpha_s=-1.1, b0=0.0,
                        b1=1.5),
}


def make_occupation(name: str):
    """Look up an occupation model by name; returns ``(n_cen, n_sat)``."""
    key = str(name).lower()
    if key not in OCCUPATION:
        raise ValueError(f"unknown occupation model {name!r}; expected one of "
                         f"{sorted(OCCUPATION)}")
    return OCCUPATION[key]


def occupation_defaults(name: str) -> dict:
    """The default parameters for one model (see ``DEFAULTS``)."""
    key = str(name).lower()
    if key not in DEFAULTS:
        raise ValueError(f"unknown occupation model {name!r}; expected one of "
                         f"{sorted(DEFAULTS)}")
    return dict(DEFAULTS[key])
