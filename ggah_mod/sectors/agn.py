r"""AGN: a chain from halo mass to X-ray luminosity, predicting its own XLF.

.. math::

    M_h \xrightarrow{\rm SHMR} M_* \xrightarrow{M_{\rm BH}-M_*} M_{\rm BH}
       \xrightarrow{\rm ERDF} \lambda_{\rm Edd}
       \xrightarrow{L=1.26\times10^{38}M_{\rm BH}\lambda} L_{\rm bol}
       \xrightarrow{k_{\rm bol}} L_X

Every step is closed form.  The only integral is over the Eddington-ratio
distribution, and it is *the same integral for every halo*: the chain is
**shift-invariant** in :math:`\log L_X`, because

.. math::  \log L_X - \log k - \langle\log M_{\rm BH}\rangle(M_h)
           = \underbrace{(\log M_{\rm BH} - \langle\log M_{\rm BH}\rangle)}_{
               \mathcal{N}(0,\sigma_{\rm lm})} + \log\lambda

and :math:`\sigma_{\rm lm}` does not depend on halo mass.  So the kernel
:math:`K = {\rm ERDF} \ast \mathcal{N}` is computed **once** and shifted per
halo -- :math:`O(N_M N_{L})` rather than an :math:`(N_M, N_L, N_\lambda)` cube.

Predicting the XLF rather than fitting one
------------------------------------------

.. math::  \Phi(L_X) = f_{\rm ERDF}\Big[\int d\log_{10}M_h\;\frac{dn}{d\log_{10}M_h}\,
                       F_{\rm c}(M_h)\,P(\log L_X | M_h)
                       + f_{\rm duty}^{\rm sat}\int_{\lg M_{*,\min}} d\lg M_*\,
                       \phi_{\rm s}(M_*)\,P(\log L_X | M_*)\Big]

with :math:`F_{\rm c}` the fraction of centrals above the stellar-mass cut and
:math:`\phi_{\rm s}` the satellites' stellar-mass function above it (below).

is a *prediction*, and comparing it to a measured luminosity function is a real
test of the chain.  The alternative -- taking a published XLF and abundance
matching to it -- cannot be tested that way, because it reproduces the XLF by
construction.  Both the Aird et al. (2015) LADE and the Ueda et al. (2014) LDDE
fits are here, as the things to be predicted.

What is excluded, and why it is named
--------------------------------------

The predecessor's **halo abundance-matching** AGN model is not ported.  Its
inversion is a ``scipy`` ``RegularGridInterpolator`` over a table built by
``interp1d`` on cumulative number densities -- differentiable in principle
through the monotone-interpolation-plus-Newton idiom this package uses
elsewhere, but not as written.  Layer 3 admits no non-differentiable module, so
it is excluded rather than admitted behind a flag.  Named here so the exclusion
is visible rather than an absence.

Its parametric :math:`L_X(M_*)` sibling is also excluded, for a different
reason: it duplicates this chain's output with a *different* hard-to-soft ratio
(0.35 against 0.638), so keeping both would mean two answers to one question.

Where the stellar mass comes from
---------------------------------

From the galaxy sector, and nowhere else.  :class:`AgnSector` is built with a
:class:`~ggah_mod.sectors.galaxies.GalaxySector` whose occupation is a
**threshold** model (:data:`~ggah_mod.sectors.galaxies.THRESHOLD_SHMR`), and
every method takes that sector's parameters.  The central's stellar mass is the
relation the occupation counts its centrals with; the satellites' is the
occupation's own conditional stellar-mass function, :math:`N_{\rm sat}(>M_*|M_h)`
differentiated in its threshold.  A simple HOD assigns no stellar masses, so it
cannot carry this chain, and the sector refuses one at construction.

It used to carry ``zu15`` at a private copy of the fiducial parameters, which
agreed with the galaxy sector at the defaults and nowhere else.

Satellites
----------

AGN live in galaxies, not in haloes: a halo enters only through the galaxies it
holds.  **Every galaxy above** :data:`LG_MSTAR_MIN` **hosts a black hole**, the
satellites from the central's :math:`M_{\rm BH}`-:math:`M_*` relation at the
satellite's own stellar mass, so :meth:`AgnSector.black_hole_mass_function` and
:meth:`AgnSector.omega_bh` count them whatever ``f_duty_sat`` is.  At fixed :math:`M_*` the width is ``sig_bh`` alone: the
conditional stellar-mass function already distributes the stellar mass, so
there is no :math:`\sigma_{M_*}` to fold in.

**Whether a satellite's black hole is shining** is ``f_duty_sat``, its duty cycle
at fixed :math:`M_*` relative to a central's:

.. math::

    N^{\rm AGN}_{\rm sat}(>L_{\min}|M_h) = f_{\rm duty}^{\rm sat}\int \dd\lg M_*\,
        \frac{\dd N_{\rm sat}}{\dd \lg M_*}(M_*|M_h)\,
        P(>L_{\min}|M_*).

It defaults to one: a satellite galaxy is as likely to host an active nucleus
as a central of the same stellar mass.  That is an assumption, not a
measurement -- this package does not have one -- and zero is the centrals-only
opt-out, under which an AGN auto-spectrum has **no one-halo term at all**,
because a Bernoulli central occupation has no self-pairs.  On the
``zumandelbaum15`` defaults at :math:`z = 0.135` and :math:`L_X > 10^{42}`,
satellites are 7.2% of the AGN at the default ``f_duty_sat = 0.296`` (the
eROSITA c030 MAP) on the 1.2.0 galaxy defaults, and :math:`b_{\rm eff}` is
0.895 against 0.822 for centrals alone (13.4%, 0.918 and 0.789 on the
1.1.0.dev1 galaxies; 2.2%, 0.925 and 0.904 at the 0.8.7 mock-fit defaults,
``f_duty_sat = 0.052``; 1.8%, 0.919 and 0.901 on the 0.8.5 galaxy
defaults).  At
``f_duty_sat = 1`` and the published AGN parameters they were 30%, 1.051 and
0.764 (0.8.6) -- the largest clustering choice in this sector, not a small
correction.

**No galaxy below** :data:`LG_MSTAR_MIN` **hosts an AGN or a black hole**,
:math:`M_* = 10^8\,M_\odot` physical, central or satellite: there is no
evidence for active nuclei in galaxies below it.  It is a cut, not a truncation
waiting to be extended, and it reaches everything the sector returns -- the
occupations, the XLF, the weights, :meth:`AgnSector.mean_mbh`,
:meth:`AgnSector.omega_bh` and the black-hole mass function.

* A **satellite** has its own stellar mass, so the cut is exact: the
  conditional stellar-mass function is integrated on a physical grid from
  :data:`LG_MSTAR_MIN` to :data:`LG_MSTAR_MAX_SAT` (:meth:`AgnSector._satellite_csmf`).
* A **central** is not integrated over its stellar mass: its scatter at fixed
  halo mass, ``sigma_ms``, is folded into the width of :math:`\log M_\bullet`.
  So a halo's central term is weighted by the fraction of its centrals above
  the cut, :math:`\tfrac12{\rm erfc}[(\lg M_{*,\min} - \lg M_*^{\rm c}(M_h))
  /(\sqrt2\,\sigma_{M_*})]` (:meth:`AgnSector.central_fraction`), and the
  black-hole distribution of the centrals that remain is **not** re-conditioned
  on the cut: that would need a kernel per halo instead of one shifted kernel.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from ..cosmology import constants as C
from ..numerics import arctan, require_x64, soft_saturate
from .calibration import (Calibration, check_sector_calibration,
                          check_sector_range)
from .params import Flat, Gaussian, Param, SectorParams, sector_params
from .galaxies import THRESHOLD_SHMR, _is_active
from .protocol import TracerWeights

__all__ = ["AgnParams", "AgnSector", "LG_MSTAR_MIN", "AGN_PUBLISHED", "erdf", "xlf_aird15", "xlf_ueda14",
           "XLF", "make_xlf", "obscured_fraction", "compton_thick_fraction",
           "photoabs_cross_section", "photoabs_tau", "absorbed_band_energy",
           "band_energy",
           "lstar_of_z",
           "AGN_CALIBRATION",
           "LBOL_COEF", "K_BOL_HARD", "K_BOL", "HARD_TO_SOFT", "GAMMA_X",
           "band_energy_ratio",
           "xlf_in_h_units", "ObscurationParams",
           "BH_CHAINS", "mbh_powell", "mbh_trinity", "erdf_trinity"]

#: The published values of the eight parameters the L3-8 AGN fits set, and
#: the ``AgnParams`` defaults up to 0.8.6: Powell et al. (2022) Model 1 for the
#: black-hole relation, Powell et al. (2022) App. B for the ERDF (their
#: Fig. 12: lambda* = 0.13, so log10 = -0.8861, delta1 = 0.29 +/- 0.14, and
#: delta2 = 3.70, which they report unconstrained; the functional form is
#: Ananna et al. 2022's), f_ERDF = 10^-1.5 and satellites as active as
#: centrals.  The Gaussian priors are still centred here.
AGN_PUBLISHED = dict(mu_bh=7.76, al_bh=0.67, sig_bh=0.33, log10_lstar=-0.8861,
                     delta1=0.30, delta2=3.70, log10_ferdf=-1.5, f_duty_sat=1.0)

#: No galaxy below this stellar mass hosts an AGN or a black hole
#: [log10 Msun, physical].  See the module docstring.
LG_MSTAR_MIN = 8.0
#: Upper end of the satellite stellar-mass grid [log10 Msun, physical], above
#: the conditional stellar-mass function's own range (12 in h^-2 Msun).
LG_MSTAR_MAX_SAT = 12.5
#: Nodes per dex of that grid: the galaxy sector's own spacing.
N_MSTAR_PER_DEX = 73

#: :math:`L_{\rm bol} = 1.26\times10^{38}\,(M_{\rm BH}/M_\odot)\,\lambda` [erg/s].
#: The Eddington luminosity for solar-abundance ionised gas.
LBOL_COEF = 1.26e38
#: Bolometric correction to the hard (2--10 keV) band.
#:
#: Duras et al. (2020) A&A 636, A73 and Marconi et al. (2004) -- traced through
#: the predecessor rather than re-derived, and recorded because the value stood
#: here with no source at all.  It is luminosity-dependent in both, which is
#: why :data:`K_BOL` gives it a Gaussian rather than a point value.
#: ``PLAN.md`` item **G8**.
K_BOL_HARD = 20.0
#: Photon index of the unobscured power law the band ratio below assumes.
GAMMA_X = 1.8


def _sinhc(x):
    r""":math:`\sinh(x)/x`, and 1 at the origin, with a gradient at both.

    The double-``where`` is the standard device and is not decoration: a bare
    ``jnp.where(small, 1.0, jnp.sinh(x)/x)`` still *evaluates* the 0/0 branch,
    and ``jax.grad`` propagates the ``nan`` from the arm it did not take.
    Substituting a safe argument inside the unsafe arm is what keeps the
    derivative finite at :math:`\Gamma = 2`.
    """
    small = jnp.abs(x) < 1e-6
    safe = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 + x * x / 6.0, jnp.sinh(safe) / safe)


@jax.jit
def band_energy(e_lo, e_hi, gamma):
    r""":math:`\int_{e_{\rm lo}}^{e_{\rm hi}} E^{1-\Gamma}\,dE`, in
    keV\ :math:`^{2-\Gamma}`.

    Written so that :math:`\Gamma = 2` is **not a branch**.  With
    :math:`k = 2-\Gamma`,

    .. math::

        \frac{b^{k}-a^{k}}{k} = \sqrt{ab}^{\,k}\,\ln\frac{b}{a}\,
            {\rm sinhc}\!\left(\frac{k}{2}\ln\frac{b}{a}\right),

    and :math:`{\rm sinhc}` is entire, so the flat-:math:`\nu F_\nu` case falls
    out as :math:`\ln(b/a)` rather than being tested for.

    The predecessor of this function returned a Python ``float`` from ``math``
    and branched on ``gamma == 2.0``.  Both mattered: :math:`\Gamma` could not
    be traced at all, so it could not be a parameter, and its gradient had a
    hole at the one value every reader checks the formula against by hand.
    """
    a, b = jnp.asarray(e_lo), jnp.asarray(e_hi)
    k = 2.0 - jnp.asarray(gamma)
    ln_ratio = jnp.log(b / a)
    return (jnp.power(jnp.sqrt(a * b), k) * ln_ratio
            * _sinhc(0.5 * k * ln_ratio))


def band_energy_ratio(e1, e2, e3, e4, gamma=GAMMA_X):
    r""":math:`L_{[e_1,e_2]}/L_{[e_3,e_4]}` for a photon spectrum
    :math:`N(E)\propto E^{-\Gamma}`.

    Traced in :math:`\Gamma`, which is what lets ``gamma_x`` be a parameter and
    ``k_h2s`` stop being one: the two were always the same number, and the
    sector's own ``why`` for ``k_h2s`` said so while carrying it separately.
    """
    return band_energy(e1, e2, gamma) / band_energy(e3, e4, gamma)


# -- photoelectric absorption ---------------------------------------------
#: Morrison & McCammon (1983), ApJ 270, 119, Table 2: the interstellar
#: photoelectric cross-section per hydrogen atom, as a piecewise quadratic
#:
#:     sigma(E) = (c0 + c1 E + c2 E^2) * 1e-24 / E^3   cm^2,   E in keV,
#:
#: over fourteen energy intervals spanning 0.03-10 keV.  Transcribed rather
#: than derived, and taken from `soxs`'s `wabs_cross_section` (which is the
#: same table) rather than from memory, for the reason this package transcribes
#: every published relation the same way: a coefficient nobody can point at is
#: a fit of our own wearing someone else's citation.
#:
#: This is `wabs` -- solar abundances, no molecules, no grains.  `tbabs`
#: (Wilms, Allen & McCray 2000) is the modern replacement and differs by tens
#: of per cent below 1 keV; swapping it is a table change and not a code one,
#: which is why the coefficients are module data rather than literals inside
#: the function.
WABS_EMAX = (0.0, 0.1, 0.284, 0.4, 0.532, 0.707, 0.867, 1.303, 1.84, 2.471,
             3.21, 4.038, 7.111, 8.331, 10.0)
WABS_C0 = (17.3, 34.6, 78.1, 71.4, 95.5, 308.9, 120.6, 141.3, 202.7, 342.7,
           352.2, 433.9, 629.0, 701.2)
WABS_C1 = (608.1, 267.9, 18.8, 66.8, 145.8, -380.6, 169.3, 146.8, 104.7, 18.7,
           18.7, -2.4, 30.9, 25.2)
WABS_C2 = (-2150.0, -476.1, 4.3, -51.4, -61.1, 294.0, -47.7, -31.5, -17.0, 0.0,
           0.0, 0.75, 0.0, 0.0)


@jax.jit
def photoabs_cross_section(e_kev):
    r"""Photoelectric cross-section per hydrogen atom, cm\ :math:`^2`.

    Differentiable in :math:`E` inside each interval, and piecewise there: the
    bin index is a ``searchsorted``, which has zero derivative, so the gradient
    is the quadratic's own.  The edges are absorption edges -- real
    discontinuities in the physics, not an artefact of the fit -- so smoothing
    across them would be inventing a spectrum.
    """
    e = jnp.asarray(e_kev)
    idx = jnp.clip(jnp.searchsorted(jnp.asarray(WABS_EMAX), e) - 1, 0, 13)
    c0 = jnp.asarray(WABS_C0)[idx]
    c1 = jnp.asarray(WABS_C1)[idx]
    c2 = jnp.asarray(WABS_C2)[idx]
    return (c0 + c1 * e + c2 * e * e) * 1.0e-24 / jnp.power(e, 3)


@jax.jit
def photoabs_tau(e_kev, log10_nh):
    r""":math:`\tau(E) = N_{\rm H}\,\sigma(E)`, dimensionless."""
    return jnp.power(10.0, jnp.asarray(log10_nh)) * photoabs_cross_section(e_kev)


def absorbed_band_energy(e_lo, e_hi, gamma, log10_nh, z=0.0, *, n_e: int = 64):
    r""":math:`\int_{E_{\rm lo}(1+z)}^{E_{\rm hi}(1+z)} E^{1-\Gamma}
    e^{-\tau(E)}\,dE`, the emitted energy that arrives in an **observed** band.

    **The K-correction falls out of the limits**, exactly as
    :class:`~ggah_mod.sectors.cooling.BandCooling` does it.  There is no
    separate :math:`(1+z)^{\Gamma-2}` factor anywhere here, and writing one
    would be the same quantity twice -- the failure mode this package spends
    most of its docstrings on.  As :math:`N_{\rm H} \to 0` this reduces
    *exactly* to :math:`(1+z)^{2-\Gamma}\,` :func:`band_energy`, which is the
    test, and that limit is also the reason the unobscured branch is fixed at
    **zero** column rather than at :math:`10^{20}\,{\rm cm}^{-2}`: measured
    here, :math:`10^{20}` already transmits only 0.970 of the 0.5--2 keV band,
    because :math:`\tau(0.5\,{\rm keV}) = 0.0736` there.  "Unobscured" and "a
    small column" are not the same statement and the difference is 3 per cent
    in the soft band.

    Gauss-Legendre in :math:`\log E`, because the integrand spans decades and
    the absorbed edge is steep.  ``n_e`` is static: it sets an array shape.
    """
    from ..halos.profiles import _leggauss_cached
    x, w = _leggauss_cached(n_e)
    a = jnp.log(jnp.asarray(e_lo) * (1.0 + jnp.asarray(z)))
    b = jnp.log(jnp.asarray(e_hi) * (1.0 + jnp.asarray(z)))
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    le = mid + half * jnp.asarray(x)
    e = jnp.exp(le)
    # dE = E dlnE, and the integrand is E^{1-Gamma} e^{-tau}.
    f = jnp.power(e, 2.0 - jnp.asarray(gamma)) * jnp.exp(-photoabs_tau(e, log10_nh))
    return half * jnp.sum(jnp.asarray(w) * f, axis=-1)


#: Soft (0.5--2 keV) over hard (2--10 keV) energy, unobscured,
#: :math:`\Gamma = 1.8`: 0.6377.  It stood here as 0.607, which is no
#: :math:`\Gamma = 1.8` ratio -- it is :math:`\Gamma \simeq 1.77`.
HARD_TO_SOFT = float(band_energy_ratio(0.5, 2.0, 2.0, 10.0))
#: Both published XLFs assume :math:`h = 0.7`.
#:
#: Read by :func:`xlf_in_h_units`, which is the only place the conversion
#: happens.  It was defined here and read *nowhere* for as long as the sector
#: existed, while the conversion was done by hand in a test and in the paper's
#: figure -- a constant recording an assumption that nothing acted on, which is
#: the same shape as the calibration registries before
#: :mod:`~ggah_mod.sectors.calibration` gave them a consumer.
H_XLF = 0.70

_LN10 = jnp.log(10.0)


# =========================================================================
# X-ray luminosity functions -- what the chain has to predict
# =========================================================================

@jax.jit
def xlf_aird15(log10lx, z, k0=-4.03, k1=-0.19, l0=44.84, p1=3.87, p2=-2.12,
               g1=0.48, g2=2.27, zc=2.00):
    r"""Aird et al. (2015) LADE, hard band [Mpc^-3 dex^-1] at :math:`h = 0.7`.

    Transcribed from `arXiv:1503.01120 <https://arxiv.org/abs/1503.01120>`_,
    Eqs. 30, 38 and 39, with the hard-band column of Table 5:

    .. math::

        \Phi(L, z) = \frac{K(z)}
                          {(L/L_*)^{\gamma_1} + (L/L_*)^{\gamma_2}},
        \qquad \log K(z) = \log K_0 + d\,(1+z),

    .. math::

        \log L_*(z) = \log L_0 - \log\left[
            \left(\tfrac{1+z_c}{1+z}\right)^{p_1}
          + \left(\tfrac{1+z_c}{1+z}\right)^{p_2}\right]

    **Both exponents are applied as tabulated, and the sign lives in the
    value.**  Table 5's hard band gives :math:`p_2 = -2.12 \pm 0.39`, so
    :math:`p_1 > 0 > p_2` and the two powers of :math:`(1+z_c)/(1+z)` pull in
    opposite directions -- which is the whole point, since the paper's own
    words are that "``p1`` and ``p2`` allow for a different evolution of
    :math:`L_*` above and below a transition redshift, :math:`z_c`".
    :math:`L_*` therefore peaks near :math:`z_c` and declines above it.

    .. note::

       Until 2026-09-16 this function raised the second power to ``-p2`` on
       top of the negative tabulated default, making the exponent
       :math:`+2.12`.  A sum of two same-signed powers of one base is
       monotone, so :math:`L_*` had no break at all: it climbed to
       :math:`\lg L_* = 45.53` at :math:`z = 6` and was still rising, leaving
       :math:`z_c` as a scale factor rather than a transition.  Corrected
       against Eq. 38.  The published comparisons in the technical paper are
       quoted at :math:`z = 0.1` and :math:`z = 1`, where the two forms differ
       by :math:`0.07` and :math:`0.14` dex in :math:`\lg L_*`; the error
       reached :math:`0.81` dex by :math:`z = 4`.

    ``zc`` is a fitted parameter rather than a constant of the model -- Table 5
    gives :math:`2.00 \pm 0.13` hard and :math:`2.31 \pm 0.07` soft -- so it is
    a keyword and not the literal it used to be, which made the soft-band
    column unreachable.  The other soft-band values are
    ``k0=-4.28, l0=44.93, g1=0.44, g2=2.18, p1=3.39, p2=-3.58, zc=2.31,
    k1=-0.22``.

    :func:`lstar_of_z`, this sector's own Eddington-ratio evolution, is a
    separate parameterisation and shares nothing with this one.
    """
    z = jnp.asarray(z)
    kz = jnp.power(10.0, k0 + k1 * (1.0 + z))            # Eq. 39
    u = (1.0 + zc) / (1.0 + z)
    log_ls = l0 - jnp.log10(jnp.power(u, p1) + jnp.power(u, p2))  # Eq. 38
    ratio = jnp.power(10.0, jnp.asarray(log10lx) - log_ls)
    return kz / (jnp.power(ratio, g1) + jnp.power(ratio, g2))


@jax.jit
def xlf_ueda14(log10lx, z, A=2.91e-6, log_ls=43.97, g1=0.96, g2=2.71,
               p1s=4.78, beta1=0.84, log_lp=44.0, p2=-1.5, p3=-6.2,
               zc1s=1.86, log_la1=44.61, alpha1=0.29,
               zc2s=3.0, log_la2=44.0, alpha2=-0.1):
    r"""Ueda et al. (2014) LDDE, hard band [Mpc^-3 dex^-1] at :math:`h = 0.7`.

    Transcribed from `arXiv:1402.1836 <https://arxiv.org/abs/1402.1836>`_,
    Eqs. 16--19, with the 2--10 keV row of Table 4 and the parameters its note
    fixes (:math:`p_2`, :math:`p_3`, :math:`z^*_{c2}`, :math:`L_p = L_{a2}`,
    :math:`\alpha_2`):

    .. math::

        \Phi(L, z) = \frac{A}{(L/L_*)^{\gamma_1} + (L/L_*)^{\gamma_2}}\;
                     e(z, L),

    .. math::

        e(z, L) = \begin{cases}
            (1+z)^{p_1} & z \le z_{c1} \\
            (1+z_{c1})^{p_1}\left(\frac{1+z}{1+z_{c1}}\right)^{p_2}
                & z_{c1} < z \le z_{c2} \\
            (1+z_{c1})^{p_1}\left(\frac{1+z_{c2}}{1+z_{c1}}\right)^{p_2}
                \left(\frac{1+z}{1+z_{c2}}\right)^{p_3} & z > z_{c2}
        \end{cases}

    with :math:`p_1(L) = p_1^* + \beta_1(\log L - \log L_p)` (Eq. 17) and each
    cut-off redshift a power law in :math:`L` below its threshold,
    :math:`z_{ci}(L) = z^*_{ci}(L/L_{ai})^{\alpha_i}` (Eqs. 18--19) -- which is
    what makes it a *luminosity-dependent* density evolution.

    **This is the XLF of Compton-thin AGN**, :math:`\log N_H < 24`,
    de-absorbed and rest-frame (Ueda et al. 2014, Section 6.1 and Fig. 10).
    Compton-thick AGN enter that paper through the absorption function, not
    through this XLF.  The *h* in ``A`` is :math:`h_{70}^3`; convert with
    :func:`xlf_in_h_units`.

    .. note::

       Until 2026-09-18 this function carried a different parameter set under
       the same name: a single :math:`p_1 = 5.54` with no luminosity
       dependence, :math:`p_2 = -0.36`, :math:`z_{c,0} = 1.84`,
       :math:`\alpha = 0.335`, :math:`A = 3.31\times10^{-6}` and no third
       regime.  Only :math:`L_*`, :math:`\gamma_{1,2}` and :math:`L_{a1}` were
       Table 4's.  At :math:`z = 0.2` it sat 0.3--0.5 dex above the table
       over :math:`42 < \log L_X < 44.4`.  The same set is in ``hod_mod``,
       from which it was ported.
    """
    z = jnp.asarray(z)
    lg = jnp.asarray(log10lx)
    ratio = jnp.power(10.0, lg - log_ls)
    phi0 = A / (jnp.power(ratio, g1) + jnp.power(ratio, g2))
    p1 = p1s + beta1 * (lg - log_lp)                                   # Eq. 17
    zc1 = jnp.where(lg <= log_la1,
                    zc1s * jnp.power(10.0, alpha1 * (lg - log_la1)), zc1s)
    zc2 = jnp.where(lg <= log_la2,
                    zc2s * jnp.power(10.0, alpha2 * (lg - log_la2)), zc2s)
    e1 = jnp.power(1.0 + z, p1)
    e2 = (jnp.power(1.0 + zc1, p1)
          * jnp.power((1.0 + z) / (1.0 + zc1), p2))
    e3 = (jnp.power(1.0 + zc1, p1)
          * jnp.power((1.0 + zc2) / (1.0 + zc1), p2)
          * jnp.power((1.0 + z) / (1.0 + zc2), p3))
    e = jnp.where(z <= zc1, e1, jnp.where(z <= zc2, e2, e3))          # Eq. 16
    return phi0 * e



def xlf_in_h_units(fit, log10lx, z, h):
    r"""A published XLF, evaluated in this package's units at its own :math:`L_X`.

    ``fit`` is :func:`xlf_aird15`, :func:`xlf_ueda14`, a name in :data:`XLF`,
    or any callable ``fit(log10lx, z)`` returning
    :math:`\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}` at :math:`h = 0.7`.  Returns
    :math:`\Phi` in :math:`(\mathrm{Mpc}/h)^{-3}\,\mathrm{dex}^{-1}` at
    ``log10lx``, where ``log10lx`` is an :math:`L_X` [erg/s] in a universe with
    Hubble parameter ``h`` -- what :meth:`AgnSector.xlf` produces.

    The transcriptions return what their papers print, because a transcription
    that silently rescales is no longer one.  The conversion is therefore this
    separate, named step, and it has two halves.

    **The volume.**  A survey counts :math:`N` sources in a comoving volume
    computed from redshifts.  In :math:`(\mathrm{Mpc}/h)^3` that volume does
    not depend on :math:`h`; in :math:`\mathrm{Mpc}^3` it is
    :math:`V_h/h^3`.  A published :math:`\Phi = N/V_{\rm Mpc}` quoted at
    :math:`h_{\rm XLF} = 0.7` is therefore

    .. math::  \Phi_{(\mathrm{Mpc}/h)^{-3}} = \Phi_{\mathrm{Mpc}^{-3}}\,
               h_{\rm XLF}^{-3} = 2.915\,\Phi_{\mathrm{Mpc}^{-3}},

    whatever the model's :math:`h` -- which is the point of counting in
    :math:`h`-units.

    **The luminosity.**  :math:`L_X = 4\pi d_L^2 F` and
    :math:`d_L \propto h^{-1}`, so the same flux is a luminosity larger by
    :math:`(h_{\rm XLF}/h)^2` in the model's universe:
    :math:`\log L = \log L_{\rm pub} + 2\log(h_{\rm XLF}/h)`, +0.0334 dex at
    :data:`~ggah_mod.cosmology.PLANCK18`'s :math:`h = 0.6736`.  The fit is
    evaluated at the published luminosity that corresponds to the model's.

    Not converted: the published fits' :math:`\Omega_m = 0.3` against the
    model's.  It changes the comoving volume by about a per cent at
    :math:`z \lesssim 1`, far inside any error these fits carry.

    .. note::

       Until 2026-09-18 this was ``xlf_in_h_units(phi, h)`` and returned
       :math:`\Phi\,(h_{\rm XLF}/h)^3` -- 1.122 at Planck 2018 instead of
       2.915, a target 2.60 times (0.415 dex) too low -- with no luminosity
       shift, the docstring arguing that erg/s on both sides made one
       unnecessary.  The signature changed with the fix so that a caller of the
       old form fails rather than receiving the new meaning.
    """
    fn = make_xlf(fit) if isinstance(fit, str) else fit
    h = jnp.asarray(h)
    lg_pub = jnp.asarray(log10lx) - 2.0 * jnp.log10(H_XLF / h)
    return fn(lg_pub, z) / H_XLF ** 3


XLF = {"aird15": xlf_aird15, "ueda14": xlf_ueda14}


def make_xlf(name: str):
    """Look up an X-ray luminosity function by name."""
    key = str(name).lower()
    if key not in XLF:
        raise ValueError(f"unknown XLF {name!r}; expected one of {sorted(XLF)}")
    return XLF[key]


# =========================================================================
# Obscuration
# =========================================================================

@jax.jit
def compton_thick_fraction(log10lx, z):
    r"""Adapted from Comparat et al. (2019) Eq. 4, whose Compton-thick fraction
    is a constant 0.3: the luminosity and redshift dependence is not theirs.
    Branchless, so it differentiates."""
    ll = 41.5 + arctan(5.0 * jnp.asarray(z)) * 1.5
    return 0.30 * (0.5 + 0.5 * erf((ll - jnp.asarray(log10lx)) / 0.25))


@sector_params
class ObscurationParams(SectorParams):
    r"""Coefficients of the obscured fraction adapted from Comparat et al.
    (2019) Eqs. 4-10, with bounds and reasons.

    Eight numbers that decide which AGN a soft-band survey sees.  They were
    literals in :func:`obscured_fraction`, so the obscured/unobscured split --
    the thing an X-ray selection function is most sensitive to -- was the one
    part of this chain a campaign could not vary.

    All eight are **reasoned Flat** rather than Gaussian.  Seven are one fit's
    published coefficients (``f_bright_amp`` = 0.3 replaces that paper's 0.4)
    and that paper quotes no covariance for them, so a
    Gaussian here would invent a precision; the bounds are what each coefficient
    can mean rather than how well it is known.
    """

    f_faint_norm: float = 0.9
    l_faint_pivot: float = 41.0
    f_bright_floor: float = 0.01
    z_bright_scale: float = 4.0
    f_bright_amp: float = 0.3
    l_blend_0: float = 43.2
    l_blend_z_amp: float = 1.2
    l_blend_width: float = 0.6
    l_blend_m_amp: float = 0.0

    _STATIC = ()

    _PARAMS = {
        "f_faint_norm": Param(
            0.9, (0.0, 1.0), Flat(), "",
            "the obscured fraction of the faint branch at the pivot "
            "luminosity. A fraction there, so [0, 1] is definitional",
            "definitional"),
        "l_faint_pivot": Param(
            41.0, (39.0, 43.0), Flat(), "log10(erg/s)",
            "the luminosity the faint branch is normalised at. Bounded by "
            "where the faint branch is the one describing the population: "
            "below 1e39 an AGN is not separable from X-ray binaries, and above "
            "1e43 the bright branch has taken over", "physical"),
        "f_bright_floor": Param(
            0.01, (0.0, 0.5), Flat(), "",
            "the obscured fraction the bright branch keeps on top of the "
            "Compton-thick population, at z = 0. Non-negative definitionally; "
            "the ceiling is where it stops being a floor and becomes the "
            "branch", "physical"),
        "z_bright_scale": Param(
            4.0, (0.5, 20.0), Flat(), "",
            "the redshift scale over which the bright branch's obscuration "
            "grows. Below ~0.5 the erf saturates within the range any survey "
            "reaches, so the evolution is a step; above ~20 it is linear "
            "across it and the parameter is degenerate with the amplitude",
            "physical"),
        "f_bright_amp": Param(
            0.3, (0.0, 1.0), Flat(), "",
            "how much obscuration the bright branch gains by high redshift. A "
            "fraction added to a fraction, so [0, 1]", "definitional"),
        "l_blend_0": Param(
            43.2, (41.0, 46.0), Flat(), "log10(erg/s)",
            "the luminosity at which the faint and bright branches change "
            "over, at z = 0. Bounded by the range the X-ray luminosity "
            "functions in this module are fitted over", "physical"),
        "l_blend_z_amp": Param(
            1.2, (0.0, 3.0), Flat(), "log10(erg/s)",
            "how far that changeover moves by high redshift. Non-negative "
            "because the crossover tracks L*, which brightens with redshift in "
            "every published XLF", "physical"),
        "l_blend_m_amp": Param(
            0.0, (-2.0, 2.0), Flat(), "dex per dex",
            "how the obscured/unobscured transition luminosity moves with host "
            "halo mass, about 1e13 Msun/h. ZERO IS THE DEFAULT AND IS THE "
            "FORM ADAPTED FROM Comparat et al. (2019), whose obscured fraction "
            "(their Eqs. 5-10) is a function of L_X and z alone, and this term "
            "is not theirs. It exists because a "
            "fraction with no halo-mass dependence CANNOT produce an "
            "obscured-against-unobscured bias difference -- it cancels between "
            "the numerator and denominator of an effective bias -- so the 9 "
            "sigma split Petter et al. (2023) measure (2.96 vs 2.27) is not a "
            "tension the published form fits badly but one it cannot express "
            "at any value of its eight coefficients. Positive means obscured "
            "AGN sit in more massive halos, which is the sign the measurement "
            "asks for. The box is +/- 2 dex per dex, far wider than any "
            "plausible fit, because nothing has fitted it", "physical"),
        "l_blend_width": Param(
            0.6, (0.05, 3.0), Flat(), "log10(erg/s)",
            "the width of the changeover in dex. Strictly positive by "
            "definition -- zero is a discontinuity, not a blend -- and above "
            "~3 dex the two branches overlap across the whole luminosity "
            "range and neither describes anything on its own",
            "definitional"),
    }


@jax.jit
def obscured_fraction(log10lx, z, p: "ObscurationParams" = None,
                      log10m=None):
    r"""The obscured fraction of AGN, adapted from Comparat et al. (2019)
    Eqs. 4-10.

    An error-function blend (their Eqs. 5-6) between a faint branch (Eq. 9) and
    a bright one (Eq. 8), the second carrying the Compton-thick population.
    Their constant f_thick = 0.3 (Eq. 4) is replaced by
    :func:`compton_thick_fraction` and their 0.4 by ``f_bright_amp`` = 0.3.
    Every piece is an ``erf`` or an ``arctan``, so the whole thing is smooth --
    the predecessor wrote it that way too, and it is the one part of its AGN
    sector that was already differentiable.

    **Its eight coefficients are declared now** (``PLAN.md`` item **C3**).  They
    were literals in this function body, which made the obscured/unobscured
    split the one part of the AGN chain a campaign could not vary -- and that
    split is exactly what an X-ray sample's selection function depends on.

    **And it no longer ends in a clip.**  ``jnp.clip(f, 0, 1)`` splits its
    gradient at the tie, the idiom
    :func:`~ggah_mod.numerics.lin_weights` documents and which this same module
    already avoided once, choosing ``tanh`` over ``clip`` for the central-AGN
    count "because a clip puts a tie at 1 and kills the gradient above it".
    The ceiling here is a physical saturation -- a fraction cannot exceed one --
    so it is :func:`~ggah_mod.numerics.soft_saturate`, exact where nothing is
    saturating and :math:`C^\infty` where something is.  The floor was never
    reachable: both branches are non-negative and ``blend`` is in [0, 1], so the
    result is a convex combination of non-negative numbers.
    """
    p = ObscurationParams() if p is None else p
    log10lx, z = jnp.broadcast_arrays(jnp.asarray(log10lx), jnp.asarray(z))
    f_ct = compton_thick_fraction(log10lx, z)
    f_faint = p.f_faint_norm * jnp.sqrt(
        p.l_faint_pivot / jnp.maximum(log10lx, 1.0))
    f_bright = f_ct + p.f_bright_floor + erf(z / p.z_bright_scale) * p.f_bright_amp
    ll = p.l_blend_0 + erf(z) * p.l_blend_z_amp
    if log10m is not None:
        # The transition luminosity, tilted with host halo mass.  A trailing
        # luminosity axis is added to every L_X-shaped term so the result is
        # (NM, N_lx): the fraction is now a property of the pair, not of L_X
        # alone, which is the whole point of the term.
        lm = (jnp.asarray(log10m) - 13.0)[..., None]
        ll = ll[None, :] + p.l_blend_m_amp * lm
        log10lx = log10lx[None, :]
        f_faint = f_faint[None, :]
        f_bright = f_bright[None, :]
    blend = 0.5 + 0.5 * erf((ll - log10lx) / p.l_blend_width)
    return soft_saturate(f_bright + (f_faint - f_bright) * blend, 1.0)


# =========================================================================
# The Eddington-ratio distribution
# =========================================================================

@jax.jit
def erdf(loglam, log10_lstar, delta1, delta2):
    r"""Ananna et al. (2022) broken power law, :math:`1/(x^{\delta_1}+x^{\delta_2})`.

    Unnormalised; :class:`AgnSector` normalises it on its own grid.
    """
    x = jnp.power(10.0, jnp.asarray(loglam) - log10_lstar)
    return 1.0 / (jnp.power(x, delta1) + jnp.power(x, delta2))


# =========================================================================
# Parameters
# =========================================================================

@jax.jit
def lstar_of_z(z, log10_lstar, gam_lam, gam_lam_hi, z_lam):
    r"""The ERDF break, evolving: Aird's LADE shape applied to :math:`\lambda`.

    .. math::

        \log\lambda_*(z) = \log\lambda_*
            - \log_{10}\frac{f(x)}{f(x_0)},\qquad
        f(u) = u^{\gamma_\lambda} + u^{-\gamma_\lambda^{\rm hi}},

    with :math:`x = (1+z_\lambda)/(1+z)` and :math:`x_0 = 1+z_\lambda`.

    **Why the break evolves and nothing else does.**  Aird et al. (2015)'s LADE
    fit (:func:`xlf_aird15`) brightens :math:`\log L_\star` by 0.91 dex between
    :math:`z = 0.1` and :math:`z = 1` and lowers its normalisation by
    :math:`10^{-0.19\times0.9} = 0.67`.  Feed 0.91 dex of brightening into that
    shape at fixed :math:`L`: at the bright-end slope :math:`\gamma_2 = 2.27` it
    is a factor :math:`10^{0.91\times2.27} = 116`, and at the faint-end slope
    :math:`\gamma_1 = 0.48` it is :math:`10^{0.91\times0.48} = 2.7`, times 0.67,
    so 1.8.  The measured failure of this sector is that it must gain a factor
    2 at the faint end and 100 at the knee across the same interval.  Those are
    the same two numbers, so the whole discrepancy is a brightening of the
    characteristic luminosity and the active fraction is not the knob: a pure
    normalisation cannot be wrong in opposite directions at two redshifts.

    :math:`\mu_{\rm BH}` is not the knob either.  The growth between
    :math:`z = 0` and 2 is accretion-rate evolution rather than black-hole mass
    at fixed :math:`M_*`, :func:`mbh_powell` records a decision against adding a
    redshift term with no fitted coefficient, and moving it would drag
    :math:`\Omega_{\rm BH}` and the black-hole mass function, which are
    separately measured and already low.

    **Broken, not a single power law**, because both published fits in this
    module saturate: LADE turns over at a fitted :math:`z_c = 2.00` and Ueda et
    al. (2014)'s :math:`z^*_{c1}` is 1.86, which is the same statement twice.  A
    power law fitted below :math:`z = 1` and used at :math:`z = 3` is exactly
    the extrapolation :mod:`~ggah_mod.sectors.calibration` exists to report.

    **Normalised at** :math:`z = 0`, and that is load-bearing twice.  All
    coefficients at zero reproduces the shipped model bit for bit, so this
    lands without moving a golden; and ``log10_lstar`` keeps meaning *the local
    break*, which is what its Gaussian prior, Powell et al. (2022) App. B's
    BASS value, is a constraint on.  Normalising at :math:`z_\lambda` instead
    -- the obvious choice -- would silently redefine the quantity that prior
    constrains.

    :math:`\phi_\star` is deliberately not evolved, and here is the number:
    Aird's :math:`k_1` is worth 0.67 over :math:`0 < z < 1` against the 116 the
    break supplies, below both factors the failure is measured at.
    """
    z = jnp.asarray(z)
    x = (1.0 + z_lam) / (1.0 + z)
    x0 = 1.0 + z_lam

    def f(u):
        return jnp.power(u, gam_lam) + jnp.power(u, -gam_lam_hi)

    return log10_lstar - jnp.log10(f(x) / f(x0))



#: The bolometric correction to the hard (2--10 keV) band.
#:
#: **One object, two containers.**  :mod:`~ggah_mod.sectors.energetics` took the
#: same quantity as a bare ``k_bol=20.0`` keyword, so the correction was sampled
#: in one module and frozen in the other -- half of ``PLAN.md`` item **G6**,
#: closed the way ``C4`` closes the other duplication: by sharing the object
#: rather than by keeping two that agree today.
K_BOL = Param(K_BOL_HARD, (5.0, 100.0), Gaussian(20.0, 5.0), "",
              "bolometric correction to the hard band; varies with "
              "luminosity, so a fixed value carries a real spread",
              "prior")


@sector_params
class AgnParams(SectorParams):
    """The AGN sector's free parameters.

    **The eight fitted defaults are the MAP of ggah_cal's eROSITA c030 AGN
    fit** (1.1.0.dev2): ``mu_bh``, ``al_bh``, ``sig_bh``, ``log10_lstar``,
    ``delta1``, ``delta2``, ``log10_ferdf`` and ``f_duty_sat``, fitted on
    ggah_mod 1.1.0.dev1 -- the zumandelbaum15 refit on sum_stat 0.6.0 -- to
    eROSITA AGN in DESI Legacy Survey hosts at 0.2 < z < 0.5: six flux-limited
    samples (F(0.2-2.3 keV) > 2.5e-14 erg/s/cm^2) in three bins of intrinsic
    2-10 keV luminosity, their n_bar, w(theta), w_p and Delta Sigma, with the
    luminosity function of all AGN in the same bins; 70 points.  chi2 = 103.2
    for 62 dof, rounded here to three decimals (ggah_cal campaign v1.1.0.dev1,
    ``erosita_c030_agn``).  The data constrain ``al_bh``, ``log10_ferdf`` and
    ``f_duty_sat``; the other five stay within one standard deviation of their
    Gaussian priors, which keep the published centres -- Powell et al. (2022)
    Model 1 for the black-hole relation and their App. B (Fig. 12) for the
    ERDF, whose delta2 they report unconstrained -- i.e. :data:`AGN_PUBLISHED`.
    The defaults of 0.8.7-1.1.0.dev1 were the MAP of a fit to a synthetic
    vector (``agn_mock_z0p202_g086``), and up to 0.8.6 the published values.
    """

    mu_bh: float = 7.954
    al_bh: float = 0.842
    sig_bh: float = 0.280
    sigma_ms: float = 0.20
    rho: float = 0.0
    log10_lstar: float = -0.798
    delta1: float = 0.232
    delta2: float = 3.749
    gam_lam: float = 0.0
    gam_lam_hi: float = 0.0
    z_lam: float = 2.0
    log10_ferdf: float = -1.640
    log10lx_min: float = 42.0
    k_bol: float = K_BOL_HARD
    gamma_x: float = GAMMA_X
    log10_nh: float = 22.5
    lx_sel_width: float = 0.05
    log10_flux_min: float = -13.3
    f_duty_sat: float = 0.296

    # No annotation: an annotated assignment in a dataclass body declares a
    # *field*, so `_STATIC: tuple = ()` would make the classification list
    # itself a traced leaf.  Caught by `test_every_declared_field_is_classified`.
    _STATIC = ()

    _PARAMS = {
        "mu_bh": Param(
            7.954, (6.5, 9.0), Gaussian(7.76, 0.30), "log10 Msun",
            "M_BH-M_* normalisation at log10 M_* = 11; the prior is the "
            "published measurement, the default the eROSITA c030 MAP (class "
            "docstring)", "prior"),
        "al_bh": Param(0.842, (0.0, 2.0), Gaussian(0.67, 0.24), "",
                       "M_BH-M_* slope; negative would invert the relation",
                       "physical"),
        "sig_bh": Param(0.280, (0.02, 1.5), Gaussian(0.33, 0.18), "dex",
                        "intrinsic scatter; the floor keeps the convolution "
                        "kernel wider than the grid spacing", "definitional"),
        "sigma_ms": Param(0.20, (0.0, 1.0), Flat(), "dex",
                          "scatter of M_* at fixed halo mass, entering the "
                          "M_BH width through the SHMR slope", "physical"),
        "rho": Param(
            0.0, (-1.0, 0.99), Flat(), "",
            "correlation between M_BH and halo mass at fixed M_* -- Powell's "
            "'Model 2'. Bounded below 1 because the M_BH|M_h variance would "
            "otherwise collapse to zero and the kernel become a delta",
            "definitional"),
        "log10_lstar": Param(-0.798, (-2.8, 0.5), Gaussian(-0.8861, 0.20),
                             "log10 lambda_Edd",
                             "ERDF break; above 0.5 the typical AGN would be "
                             "super-Eddington", "physical"),
        "delta1": Param(0.232, (0.0, 1.5), Gaussian(0.30, 0.15), "",
                        "faint-end ERDF slope; must be below delta2 for the "
                        "break to be a break", "physical"),
        "delta2": Param(3.749, (1.5, 6.0), Gaussian(3.70, 0.66), "",
                        "bright-end ERDF slope; > 1 for a finite integral",
                        "definitional"),
        "gam_lam": Param(
            0.0, (-2.0, 8.0), Flat(), "",
            "how fast the ERDF break brightens with redshift, the low-z "
            "exponent of `lstar_of_z`. THE DEFAULT IS 0, WHICH IS THE MODEL "
            "THIS SECTOR SHIPPED: no evolution, the one value that asserts "
            "nothing, and the value at which every existing golden is "
            "reproduced exactly. It is not the fiducial -- at 0 the predicted "
            "XLF is high at the z = 0.1 knee and low at the z = 1 knee, and "
            "the 0.9 dex of brightening that removes both is near 3.9, which "
            "is Aird et al. (2015)'s own p1 = 3.87 for L_X. The box reaches 8 "
            "because a chain evolving lambda rather than L_X may need more "
            "than a fit to L_X did; negative is admitted because a dimming "
            "population is a claim the data may make, and refusing it by "
            "construction would be a prior wearing a bound's clothes",
            "physical"),
        "gam_lam_hi": Param(
            0.0, (-4.0, 8.0), Flat(), "",
            "the high-redshift branch of the same break: above z_lam the "
            "brightening reverses, which is what both published XLFs here do "
            "-- LADE through its second exponent, LDDE through z_c(L). Zero is "
            "again the no-evolution value, and the one to hold fixed when "
            "fitting below z_lam, where it is unconstrained and degenerate "
            "with z_lam", "physical"),
        "z_lam": Param(
            2.0, (0.5, 5.0), Gaussian(2.0, 0.5), "",
            "the redshift the break stops brightening at. Aird et al. (2015) "
            "turn over at z_c = 2.00 and Ueda et al. (2014)'s z*_c1 is "
            "1.86, so this prior is those two agreeing rather than a fit of "
            "ours. Bounded below at 0.5, under which the turnover sits inside "
            "every low-z sample and the parameter stops being a pivot, and "
            "above at 5, over which nothing in this package has data",
            "prior"),
        "log10_ferdf": Param(
            -1.640, (-5.0, 0.0), Flat(), "",
            "active fraction. The upper bound is definitional: a fraction "
            "cannot exceed 1", "definitional"),
        "log10lx_min": Param(
            42.0, (39.0, 46.0), Flat(), "log10 erg/s",
            "selection threshold of the sample being modelled -- a property "
            "of the survey, not of the AGN", "validity"),
        "lx_sel_width": Param(
            0.05, (0.005, 0.5), Flat(), "dex",
            "width of the survey's selection in log10 L_X. Not a physical "
            "scatter but a smoothing: a step has no derivative in the "
            "threshold, and the threshold is a survey property a joint fit "
            "varies. Bounded below by the L_X grid spacing -- 0.0201 dex at "
            "the shipped 400 nodes, so the default is 2.5 cells -- under which "
            "the sigmoid is a step the quadrature cannot resolve; and above by "
            "the published 0.33 dex black-hole scatter it has to stay inside, or it "
            "stops being a selection and becomes a second scatter",
            "definitional"),
        "log10_flux_min": Param(
            -13.3, (-17.0, -10.0), Flat(), "log10 erg/s/cm^2",
            "the survey's flux limit, used when the sector selects in flux "
            "rather than in luminosity. 1e-17 is the Chandra Deep Field and "
            "1e-10 is brighter than any AGN a wide survey sees, so the box is "
            "the range of real X-ray surveys. Ignored at selection='luminosity'"
            ", where it has an identically zero gradient and that is a "
            "property of the mode rather than a defect", "validity"),
        "k_bol": K_BOL,
        "gamma_x": Param(
            GAMMA_X, (1.0, 3.0), Gaussian(GAMMA_X, 0.2), "",
            "photon index of the AGN power law, N(E) ~ E^-Gamma. It sets every "
            "band ratio this sector forms -- `k_h2s` is nothing but its 0.5-2 "
            "over 2-10 keV value, and was carried separately as a second free "
            "number for the same quantity until `band_energy` could be traced "
            "in Gamma. A stacked X-ray spectrum measures it to 0.1-0.2, which "
            "is the prior. The box is where the source is still a "
            "power-law-dominated AGN: below 1.0 is harder than any unbeamed "
            "Seyfert measured, above 3.0 is a soft excess rather than a corona. "
            "Gamma = 2 is inside the box and is not a special case -- "
            "`band_energy` is smooth through it", "prior"),
        "log10_nh": Param(
            22.5, (20.0, 25.0), Flat(), "log10 cm^-2",
            "the hydrogen column of the obscured branch -- the one number that "
            "turns `obscured_fraction` from a bookkeeping split into a "
            "spectrum. Bounded below at 1e20, under which absorption is a few "
            "per cent across 0.5-2 keV and the branch stops being obscured in "
            "any measurable sense; above at 1e25, where the source is "
            "Compton-thick, a photoelectric cross-section is the wrong model, "
            "and `compton_thick_fraction` already counts that population "
            "separately. Flat because the column distribution of an X-ray "
            "sample is what a survey measures, not something a prior should "
            "assert. The UNOBSCURED branch is fixed at zero column by "
            "definition rather than fitted: a column below the band's "
            "sensitivity is a parameter the data cannot constrain",
            "physical"),
        "f_duty_sat": Param(
            0.296, (0.0, 2.0), Flat(), "",
            "the duty cycle of a satellite AGN at fixed M_*, RELATIVE to a "
            "central's, so 0 means no satellite is active and 1 says one is as "
            "likely to be active as a central of the same stellar mass. The "
            "default, 0.296, is the eROSITA c030 MAP's (class docstring), "
            "where the posterior is 0.33 +- 0.16: the first value of it this "
            "package has from data, though not a measurement of it alone, and "
            "it was 0.052 when it followed the 1-2% satellites of the AGN "
            "mock (0.8.7-1.1.0.dev1); 1, the default of 0.8.2-0.8.6, says a black hole "
            "sees its own galaxy whether that galaxy is a central or a "
            "satellite.  Whether the two "
            "are equal is a physical question with observational claims on "
            "both sides -- ram-pressure stripping and starvation argue for "
            "suppression, while enhanced interaction rates in groups argue for "
            "the opposite -- and this package does not settle it. Zero is the "
            "centrals-only opt-out. The box reaches 2 rather than 1 "
            "because an enhancement is a claim the data may yet make and a "
            "ceiling at 1 would refuse it by construction. PLAN.md item T7-7, "
            "step S3", "physical"),
    }

    @property
    def k_h2s(self):
        r"""Soft (0.5--2 keV) over hard (2--10 keV) energy, from :attr:`gamma_x`.

        **Derived, not declared.**  It was a free ``Param`` beside ``gamma_x``'s
        ancestor constant, which made two numbers for one quantity -- exactly
        the duplication ``K_BOL`` was created to remove, and its own ``why``
        admitted it ("depends on the photon index, which is fitted in a full
        X-ray analysis").  A fit could move the ratio and the index apart, and
        nothing would have noticed.

        A property rather than a field, so it stays out of the dataclass
        census: ``_PARAMS`` and the declared fields must agree, and this is not
        a parameter any more.  Downstream reads -- ``ggah_cal`` takes
        ``float(ap.k_h2s)`` to convert a soft flux limit to a hard luminosity --
        keep working unchanged.
        """
        return band_energy_ratio(0.5, 2.0, 2.0, 10.0, self.gamma_x)


# =========================================================================
# The sector
# =========================================================================

def mbh_powell(log10_mstar, p, z=0.0):
    r""":math:`\langle\log M_\bullet\rangle = \mu + \alpha(\log M_* - 11)`.

    The relation this sector has always used, now behind a name so a second one
    can exist beside it.  No redshift dependence: the relation is a local one
    and pretending otherwise by adding a term with no fitted coefficient would
    be worse than not having it.
    """
    return p.mu_bh + p.al_bh * (jnp.asarray(log10_mstar) - 11.0)


def mbh_trinity(log10_mbulge, p, z=0.0):
    r"""TRINITY's median SMBH mass, Eqs. (34)--(36).

    Zhang, Behroozi, Volonteri et al. (2023), MNRAS 518, 2123
    (arXiv:2105.10474):

    .. math::

        \log_{10}\tilde M_\bullet &= \beta_{\rm BH}
            + \gamma_{\rm BH}\log_{10}\frac{M_{\rm bulge}}{10^{11}M_\odot} \\
        \beta_{\rm BH}  &= \beta_0 + \beta_a(a-1) + \beta_z z \\
        \gamma_{\rm BH} &= \gamma_0 + \gamma_a(a-1) + \gamma_z z

    **Two things this needs and does not have, both deliberate.**

    First, ``p`` must carry ``bh_beta0``, ``bh_beta_a``, ``bh_beta_z``,
    ``bh_gamma0``, ``bh_gamma_a``, ``bh_gamma_z``, and
    :class:`AgnParams` does not define them, so this raises until a caller
    supplies them.  That is not an oversight.  TRINITY's published best-fit
    values are **not usable**: the paper's own text points at an "Appendix H"
    that the paper does not contain, it gives no code or data release, and the
    erratum -- Zhang et al. (2023), MNRAS 522, 3627,
    ``doi:10.1093/mnras/stad1137`` -- re-ran the entire fit after three bugs,
    fixed :math:`\rho_{\rm BH}\equiv1`, changed the Eddington-ratio
    parameterisation, and published *figures* rather than a table.  There is no
    citable central value to default to, and defaults digitised off a plot would
    be the only numbers in this package not traceable to a printed one.  So the
    model is here, fittable, and refuses to pretend it is calibrated -- the same
    refusal :class:`~ggah_mod.sectors.galaxies.GalaxySector` makes about its
    ``shmr=``.

    Second, the argument is :math:`M_{\rm bulge}`, **not** :math:`M_*`.
    :func:`mbh_powell` takes total stellar mass; this one does not, and nothing
    in this package supplies a bulge fraction.  Converting one to the other with
    an invented number would put a morphology model into an AGN sector without
    saying so, so the caller passes bulge mass or does not use this chain.
    """
    z = jnp.asarray(z)
    a = 1.0 / (1.0 + z)
    missing = [k for k in ("bh_beta0", "bh_beta_a", "bh_beta_z",
                           "bh_gamma0", "bh_gamma_a", "bh_gamma_z")
               if getattr(p, k, None) is None]
    if missing:
        raise ValueError(
            f"the trinity chain needs {missing}, and there are no published "
            f"values to default to.  TRINITY's paper refers to an Appendix H "
            f"it does not contain, releases no code or chains, and its 2023 "
            f"erratum (MNRAS 522, 3627) re-ran the fit after three bugs and "
            f"reported figures rather than a table.  Supply them explicitly, or "
            f"use bh_chain='powell'.")
    beta = p.bh_beta0 + p.bh_beta_a * (a - 1.0) + p.bh_beta_z * z
    gamma = p.bh_gamma0 + p.bh_gamma_a * (a - 1.0) + p.bh_gamma_z * z
    return beta + gamma * (jnp.asarray(log10_mbulge) - 11.0)


def erdf_trinity(loglam, log10_eta0, c1, c2, f_duty=1.0):
    r"""TRINITY's Eddington-ratio distribution, Eq. (49), without its delta.

    .. math::

        P(\eta) = \frac{f_{\rm duty}P_0}
                       {(\eta/\eta_0)^{c_1} + (\eta/\eta_0)^{c_2}}

    A double power law, which is a different shape from
    :func:`erdf`'s Schechter-like form -- the two are alternatives, not
    refinements of one another.

    The :math:`(1-f_{\rm duty})\delta(\eta)` term of Eq. (49) is **not** here:
    a delta function at zero accretion is a statement about how many black holes
    are off, which this package already carries as the duty cycle in the
    occupation.  Representing it twice is how two numbers for one quantity get
    to disagree.  The returned distribution integrates to ``f_duty``, and the
    remainder is the caller's inactive population.

    ``c1`` and ``c2`` evolve as Eqs. (50)--(51),
    :math:`c_i = c_{i,0} + c_{i,a}(a-1)`; that evolution is the caller's, so
    this takes the evaluated slopes.
    """
    x = jnp.asarray(loglam) - log10_eta0
    shape = 1.0 / (jnp.power(10.0, c1 * x) + jnp.power(10.0, c2 * x))
    norm = jnp.trapezoid(shape, jnp.asarray(loglam))
    return f_duty * shape / norm


#: How the sector resolves the obscured/unobscured split.
OBSCURATION_MODES = ("none", "split")

#: Whether the survey cut is a luminosity threshold or a flux one -- or, for a
#: photon map, no cut at all: a cross-correlation with an event map counts
#: every AGN's photons, detected or not, and a threshold there would remove the
#: faint population the map is made of.
SELECTION_MODES = ("luminosity", "flux", "none")

#: ``name -> M_BH(log10 mass, params, z)``.  The mass is stellar for
#: ``powell`` and **bulge** for ``trinity``; see each function.
BH_CHAINS = {
    "powell": mbh_powell,
    "trinity": mbh_trinity,
}


#: What each black-hole chain was fitted to, and where it means anything.
#:
#: The sixth registry in this layer, and the first to carry a **mass** range as
#: well as a redshift one.  Both are needed here and for different reasons.
#: The redshift range is compared the way the other five are: Powell's fit is a
#: local one, so the forecast's 0.1 < z < 0.5 and any z = 1 comparison are
#: outside it and say so.  The stellar-mass range is compared by
#: :func:`~ggah_mod.sectors.calibration.check_sector_range`, because this sector
#: evaluates its relation on every halo in ``field.m`` and integrates satellites
#: from :data:`LG_MSTAR_MIN` to :data:`LG_MSTAR_MAX_SAT`, so unlike the
#: five registries the module docstring was written for there *is* a requested
#: range to compare against.
#:
#: **The extrapolation below the fitted range is a decision, not an oversight.**
#: Eq. (mbh) is linear in log M_* and it stays linear all the way down: no
#: clamp, no truncation, no second relation joined on at an arbitrary mass.
#: That adds no coefficient nobody fitted, and its price is measured rather
#: than argued -- see :meth:`AgnSector.validity_cost`.  Two consequences follow
#: and are carried openly: Omega_BH sits at 0.80 of Fukugita & Peebles (2004)
#: as a share of Omega_b, and E_AGN/E_SN rises towards the low-mass end where it
#: should fall -- 2.42 at 1e11 Msun/h against 0.99 at 1e15 at the 1.1.0.dev2
#: defaults (0.49 and 0.42 at 0.9.0's), measured by the paper's
#: `sector_invariants`.  It read 4.7 at 1e11 when this was written,
#: at earlier defaults and with the black-hole mass 1/h too large in the
#: closure.  Both are properties of a straight line extended past its data.
AGN_CALIBRATION = {
    "powell": Calibration(
        fit="Powell et al. (2022) Model 1: BASS DR2 hard-band XLF and the "
            "AGN-galaxy cross-correlation at z ~ 0.04",
        z_range=(0.0, 0.1),
        selection="hard X-ray selected AGN; stellar masses from Behroozi et "
                  "al. (2010) on peak (sub)halo mass, scatter 0.2 dex",
        mstar_range=(10.0, 12.0),
        notes="the M_BH-M_star relation and the Eddington-ratio distribution "
              "of this sector. Evaluated below lg M_star = 10 on every field "
              "this package builds, deliberately and linearly; "
              "AgnSector.validity_cost measures what that is worth.",
        cosmology_dependent=False),
    "trinity": Calibration(
        fit="TRINITY, uncalibrated here",
        z_range=None,
        notes="no citable coefficients: the paper points at an appendix it "
              "does not contain and releases no chains, so `mbh_trinity` "
              "refuses rather than defaulting. z_range=None says the question "
              "does not apply, which is deliberately not the same as a wide "
              "range."),
}


class AgnSector:
    """The AGN sector: the Powell chain, once.

    The predecessor had it twice -- numpy in ``agn/powell.py`` and JAX in the
    forecast module -- kept in agreement by a parity test.  Two implementations
    of one model is the failure mode this package exists to remove, so there is
    one here and it is the differentiable one.
    """

    name = "agn"
    differentiable = True

    def __init__(self, galaxies=None, *, n_lam: int = 240,
                 loglam_min: float = -3.0, loglam_max: float = 1.5,
                 n_lx: int = 400, loglx_min: float = 39.0,
                 loglx_max: float = 47.0, bh_chain: str = "powell",
                 calibration: str = "warn",
                 obscuration: str = "none",
                 selection: str = "luminosity",
                 lg_mstar_min: float = LG_MSTAR_MIN):
        r"""
        Parameters
        ----------
        galaxies : GalaxySector
            **Required**, and a threshold occupation
            (:data:`~ggah_mod.sectors.galaxies.THRESHOLD_SHMR`).  The chain
            starts from a stellar mass, and the galaxy sector is where one
            lives: its stellar-mass relation gives the central's, and its
            conditional stellar-mass function gives the satellites'.  Every
            method then takes that sector's parameters as ``galaxy_params``.

            It used to be optional, with ``zu15`` at a private copy of the
            fiducial parameters behind it.  That copy agreed with the galaxy
            sector's at the defaults and nowhere else, so a joint fit moved the
            galaxies and not the AGN.  A simple HOD counts galaxies without
            ever assigning one a stellar mass, and a bolted-on ``shmr=`` would
            give the AGN a stellar mass unrelated to the galaxies that were
            counted, so both are refused.
        n_lam, loglam_min, loglam_max, n_lx, loglx_min, loglx_max
            The Eddington-ratio and luminosity grids.  Static: they set array
            shapes.
        bh_chain : str
            A key of :data:`BH_CHAINS`.
        lg_mstar_min : float
            The stellar-mass cut, :data:`LG_MSTAR_MIN` by default, in physical
            :math:`\log_{10}(M_*/M_\odot)`.  Static: it sets the satellite grid.
        obscuration : {"none", "split"}
            Whether the sector resolves the obscured/unobscured split.
            ``"none"``, the default, asserts nothing and is what shipped:
            :func:`obscured_fraction` is computed by the module and consumed by
            nothing, so the split reaches no prediction.  ``"split"`` makes it
            two populations, which is the only arrangement that can produce a
            *different halo bias* for the two.

            That is a structural statement rather than a preference.
            ``obscured_fraction`` depends on :math:`L_X` and :math:`z`, not on
            halo mass, and :meth:`effective_bias` is
            :math:`\int bN\,dn/dM \div \int N\,dn/dM` -- any factor
            independent of halo mass cancels **exactly**.  So one population
            carrying an obscured *weight* predicts a zero bias difference by
            construction, and Petter et al. (2023)'s 9-sigma split
            (2.96 +/- 0.07 obscured against 2.27 +/- 0.06) is not a tension it
            fits badly but one it cannot express.  Two populations make the
            split a prediction, through :math:`L_X(M_h)`.
        selection : {"luminosity", "flux", "none"}
            The survey cut every moment applies.  ``"luminosity"``, the
            default, is a threshold at ``log10lx_min``; ``"flux"`` converts
            ``log10_flux_min`` at the field's redshift with the fitted spectrum;
            ``"none"`` is a photon map, where every AGN's photons count and a
            threshold would remove the population the map is made of.
        calibration : {"warn", "strict", "off"}
            What to do when the field asks for a redshift, or the halo grid for
            a stellar mass, outside :data:`AGN_CALIBRATION`'s fitted ranges.
            Checked in ``weights``, where a :class:`~ggah_mod.halos.field.HaloField`
            supplies both.  Not a formality for this sector: Powell's fit is
            local, so any field above z ~ 0.1 extrapolates it, and the shipped
            mass grid reaches stellar masses well below its floor on every call.
        """
        if galaxies is None:
            raise ValueError(
                "AgnSector needs a galaxy sector: the chain starts from a "
                "stellar mass, and there is no stellar mass without a "
                "stellar-mass relation.  Build it as "
                "AgnSector(GalaxySector('zumandelbaum15')) -- or any of "
                f"{sorted(THRESHOLD_SHMR)} -- and pass that sector's parameters "
                "as `galaxy_params`.")
        if not getattr(galaxies, "is_threshold", False):
            raise ValueError(
                f"the AGN chain needs central stellar masses and a satellite "
                f"stellar-mass function, which only a threshold occupation "
                f"carries: one of {sorted(THRESHOLD_SHMR)}.  "
                f"{getattr(galaxies, 'model', galaxies)!r} counts galaxies "
                f"without assigning them stellar masses, and an `shmr=` added "
                f"to it would give the AGN a stellar mass unrelated to the "
                f"galaxies it counted.")
        self.galaxies = galaxies
        self.lg_mstar_min = float(lg_mstar_min)
        n_ms = int(round((LG_MSTAR_MAX_SAT - self.lg_mstar_min)
                         * N_MSTAR_PER_DEX)) + 1
        self.lg_ms_sat = jnp.linspace(self.lg_mstar_min, LG_MSTAR_MAX_SAT, n_ms)
        self.loglam = jnp.linspace(loglam_min, loglam_max, n_lam)
        self.dlam = float((loglam_max - loglam_min) / (n_lam - 1))
        self.loglx = jnp.linspace(loglx_min, loglx_max, n_lx)
        self.dlx = float((loglx_max - loglx_min) / (n_lx - 1))
        key = str(bh_chain).lower()
        if key not in BH_CHAINS:
            raise ValueError(f"unknown BH chain {bh_chain!r}; expected one of "
                             f"{sorted(BH_CHAINS)}")
        self.bh_chain = key
        self._mbh = BH_CHAINS[key]
        self.calibration = str(calibration)
        if str(obscuration) not in OBSCURATION_MODES:
            raise ValueError(
                f"unknown obscuration mode {obscuration!r}; expected one "
                f"of {sorted(OBSCURATION_MODES)}")
        self.obscuration = str(obscuration)
        if str(selection) not in SELECTION_MODES:
            raise ValueError(
                f"unknown selection mode {selection!r}; expected one of "
                f"{sorted(SELECTION_MODES)}")
        self.selection = str(selection)

    # -- the chain -----------------------------------------------------------
    def log10_mstar(self, log10m, galaxy_params, *, h, z=None):
        r""":math:`\log_{10}(M_*^{\rm c}/M_\odot)`, the galaxy sector's, **physical**.

        :meth:`~ggah_mod.sectors.galaxies.GalaxySector.log10_mstar_msun`: the
        relation the occupation counts its centrals with, at the galaxy
        parameters, converted from the occupation's units into physical solar
        masses, because :func:`mbh_powell` is a relation in physical
        :math:`M_\odot` for both masses.  ``log10m`` is in :math:`M_\odot/h`
        and ``h`` is the field's.  Feeding it ``zumandelbaum15``'s
        :math:`h^{-2}M_\odot` unconverted put every black hole
        :math:`2\alpha_{\rm BH}\log h = -0.23` dex low at Planck 2018.
        Differentiable **through the inversion** -- see
        :func:`ggah_mod.sectors.sham.invert_monotone`.

        ``sigma_ms`` is **not** taken from the galaxy sector.  Replacing 0.20
        dex by ``zu15``'s :math:`\sigma_{\ln M_*}/\ln 10 = 0.217` (Paper I's
        :math:`\sigma_0`, the default then; 0.264 on the LS10 defaults) moved the
        central AGN's :math:`\bar n_{\rm AGN}` by +0.23% and :math:`b_{\rm AGN}`
        by -0.05% at the defaults (it does not reach the satellites, whose
        stellar mass the conditional stellar-mass function distributes), and letting it run with halo mass through ``eta`` adds
        less than 0.01% more, because ``sig_bh`` dominates :meth:`_sigma_lm`.
        Measured 2026-09-15 on the ``emu_pk`` field at :math:`z = 0.135`.

        ``z`` is forwarded rather than dropped.  It moved no number when this
        line was written -- every :data:`~ggah_mod.sectors.galaxies.THRESHOLD_SHMR`
        entry is redshift-independent -- which is exactly why it should be right
        before the first one is not.  ``GalaxySector.stellar_fraction`` already
        passes ``z=field.z``, so leaving it out here meant the AGN chain and the
        stellar fraction would read two different stellar masses the moment a
        redshift-dependent threshold relation was admitted: the failure commit
        c29304c removed by making this sector read the galaxy sector's masses in
        the first place.
        """
        return self.galaxies.log10_mstar_msun(log10m, galaxy_params, h=h, z=z)

    def mean_log10_mbh(self, log10m, p: AgnParams, galaxy_params, *, h, z=0.0):
        r""":math:`\langle\log M_\bullet\rangle` from the selected chain.

        ``powell`` takes the stellar mass this sector computes.  ``trinity``
        wants a *bulge* mass and this sector has none, so it is handed the
        stellar mass and will be wrong by the bulge fraction -- which is why
        :func:`mbh_trinity` says so and why selecting it also requires
        parameters that have no published defaults.  The chain is a model
        choice; supplying it what it needs is the caller's.

        ``z`` defaults to 0 and ``powell`` ignores it, so the incumbent chain is
        unaffected.  It is an **argument** rather than something read off
        ``p``: :class:`AgnParams` has no redshift field, and an earlier version
        of this method fetched one with ``getattr(p, "z", 0.0)`` -- which meant
        the one chain whose entire content is redshift evolution silently
        evaluated at :math:`z = 0` on every field.  The callers that have a
        field pass its redshift; the ones that do not take it explicitly.
        """
        ms = self.log10_mstar(log10m, galaxy_params, h=h, z=z)
        return self._mbh(ms, p, z)

    def _sigma_lm(self, p: AgnParams):
        r"""Width of :math:`\log M_{\rm BH}` at fixed halo mass.

        :math:`\rho > 0` means part of the :math:`M_*` scatter aligns with halo
        mass, which *shrinks* this width -- the whole content of Powell's
        "Model 2".
        """
        var = p.al_bh ** 2 * p.sigma_ms ** 2 * (1.0 - p.rho) + p.sig_bh ** 2
        return jnp.sqrt(jnp.maximum(var, 1e-6))

    def _sigma_bh_at_fixed_mstar(self, p: AgnParams):
        r"""Width of :math:`\log M_{\rm BH}` at fixed **stellar** mass.

        What a satellite gets.  Its stellar mass is not drawn about a mean at
        fixed halo mass -- the conditional stellar-mass function already
        distributes it -- so there is no :math:`\sigma_{M_*}` to fold in, and
        :math:`\rho`, a correlation with the *host* halo at fixed :math:`M_*`,
        does not apply.  The black-hole relation itself is the central's.
        """
        return jnp.sqrt(jnp.maximum(p.sig_bh ** 2, 1e-6))

    def _kernel(self, p: AgnParams, sig=None, *, z=0.0):
        r""":math:`K = {\rm ERDF} \ast \mathcal{N}(0,\sigma)`, on the
        :math:`\log\lambda` grid, with :math:`\sigma = \sigma_{\rm lm}` unless
        given.

        Computed once per parameter set, not once per halo: the chain is
        shift-invariant, which is the observation that turns an
        :math:`(N_M,N_L,N_\lambda)` cube into a 1-D convolution.

        The Gaussian is evaluated on a **fixed-width** grid rather than a
        :math:`\pm6\sigma` one.  A width that tracked :math:`\sigma` would make
        an array shape depend on a traced value, which does not jit -- and the
        fixed grid spans the whole prior range of :math:`\sigma_{\rm bh}` with
        room to spare.
        """
        e = erdf(self.loglam,
                 lstar_of_z(z, p.log10_lstar, p.gam_lam,
                            p.gam_lam_hi, p.z_lam),
                 p.delta1, p.delta2)
        e = e / (jnp.sum(e) * self.dlam)
        if sig is None:
            sig = self._sigma_lm(p)
        half = self.loglam.size // 2
        x = (jnp.arange(-half, half + 1) * self.dlam)
        g = jnp.exp(-0.5 * (x / sig) ** 2)
        g = g / jnp.sum(g)
        k = jnp.convolve(e * self.dlam, g, mode="full") / self.dlam
        t0 = self.loglam[0] - half * self.dlam
        t = t0 + jnp.arange(k.size) * self.dlam
        return t, k

    def p_loglx_given_m(self, log10m, p: AgnParams, galaxy_params, *, h,
                        z=0.0):
        r""":math:`dP(\log L_X|M_h)/d\log L_X` of the **central**, shape ``(NM, N_lx)``,
        times :meth:`central_fraction`: a central below the stellar-mass cut
        hosts no AGN."""
        t_grid, k = self._kernel(p, z=z)
        mean_bh = self.mean_log10_mbh(log10m, p, galaxy_params, h=h, z=z)
        frac = self.central_fraction(log10m, p, galaxy_params, h=h, z=z)
        return self._shift(t_grid, k, p, mean_bh) * frac[:, None]

    def central_fraction(self, log10m, p: AgnParams, galaxy_params, *, h,
                         z=0.0):
        r"""The fraction of a halo's centrals above the stellar-mass cut.

        .. math::

            f_{\rm c}^{\rm cut}(M_h) = \tfrac12\,{\rm erfc}\!\left[
                \frac{\lg M_{*,\min} - \lg M_*^{\rm c}(M_h)}
                     {\sqrt2\,\sigma_{M_*}}\right]

        with :math:`\sigma_{M_*}` = ``sigma_ms``, the central's stellar-mass
        scatter at fixed halo mass that :meth:`_sigma_lm` already carries.  It
        multiplies the central term everywhere; the black-hole distribution of
        the centrals that remain is not re-conditioned on the cut.
        """
        ms = self.log10_mstar(log10m, galaxy_params, h=h, z=z)
        sig = jnp.maximum(p.sigma_ms, 1e-6)
        return 0.5 * erfc((self.lg_mstar_min - ms) / (jnp.sqrt(2.0) * sig))

    def _satellite_csmf(self, field, galaxy_params):
        r"""``(lg_ms, dn)``: the satellites' conditional stellar-mass function
        on this sector's physical grid, from the cut to
        :data:`LG_MSTAR_MAX_SAT`.  Every satellite term reads it, so none
        counts a satellite below the cut."""
        return self.galaxies.satellite_csmf_msun(field, galaxy_params,
                                                 lg_ms_msun=self.lg_ms_sat)

    def _shift(self, t_grid, k, p: AgnParams, mean_log10_mbh):
        r"""The kernel placed at each mean black-hole mass, on the
        :math:`\log L_X` grid: shape ``(len(mean_log10_mbh), N_lx)``."""
        logk = jnp.log10(LBOL_COEF / p.k_bol)
        t = (self.loglx[None, :] - logk
             - jnp.atleast_1d(mean_log10_mbh)[:, None])
        return jnp.interp(t.ravel(), t_grid, k, left=0.0,
                          right=0.0).reshape(t.shape)

    def _selection(self, p: AgnParams, *, field=None, band=None):
        r"""The survey cut, a **smooth** sigmoid in :math:`\log L_X`.

        Not a hard step.  A step has no useful derivative in the threshold, and
        the threshold is a survey property a joint fit legitimately varies; the
        predecessor's own gas prior makes the same substitution for the same
        reason.  The width is a fraction of a dex, far below the scatter it
        sits inside, and is now declared (``lx_sel_width``) rather than a
        literal.

        At ``selection="flux"`` the threshold is a **flux** limit converted to
        the luminosity it corresponds to at the field's redshift,

        .. math::

            \log L_{\min}(z) = \log\!\left(4\pi d_L^2 F_{\min}\right)
                               - \log K,\qquad
            K = \frac{\int_{E_1(1+z)}^{E_2(1+z)}E^{1-\Gamma}e^{-\tau}dE}
                     {\int_{E_1}^{E_2}E^{1-\Gamma}dE},

        with :math:`K` from :func:`absorbed_band_energy` -- the *fitted*
        :math:`\Gamma` and :math:`N_{\rm H}`, not an assumed pair.  That is the
        point of the mode: ``ggah_cal`` has been doing this conversion in numpy
        with its own fixed :math:`\Gamma`, so the spectrum the model fits and
        the spectrum the selection assumes were two numbers that a fit could
        move apart.  Here they are one.

        At ``selection="none"`` there is no cut, which is what a photon map
        is: its cross-correlation with galaxies counts every AGN down to the
        bottom of the luminosity grid, and those below any catalogue's
        threshold are most of them by number.
        """
        if self.selection == "none":
            return jnp.ones_like(self.loglx)
        if self.selection == "luminosity" or field is None:
            lmin = p.log10lx_min
        else:
            lmin = self.flux_limit_luminosity(field, p, band=band)
        return 0.5 * (1.0 + erf((self.loglx - lmin) / p.lx_sel_width))

    def flux_limit_luminosity(self, field, p: AgnParams, *, band=None):
        r"""The rest-frame :math:`\log L_X` a flux limit corresponds to.

        ``band`` is the observed band in keV, defaulting to this sector's hard
        band.  Returns a scalar; the whole grid is selected against it.
        """
        from ..cosmology.background import luminosity_distance
        e1, e2 = (2.0, 10.0) if band is None else (float(band[0]), float(band[1]))
        d_l = jnp.atleast_1d(
            luminosity_distance(field.z, field.cosmo))[0] / field.cosmo.h
        d_cm = d_l * C.MPC_CM
        k = (absorbed_band_energy(e1, e2, p.gamma_x, p.log10_nh, field.z)
             / band_energy(e1, e2, p.gamma_x))
        return (jnp.log10(4.0 * jnp.pi * d_cm ** 2)
                + p.log10_flux_min - jnp.log10(k))

    def _moment(self, prob, p: AgnParams, power: int, *, branch=None,
                z=0.0, obsc: "ObscurationParams" = None, field=None,
                band=None, log10m=None, lweight=None):
        r""":math:`\int d\log L_X\, P\, S\, L_X^{\rm power}` along the last axis:
        the selected count (0), luminosity (1) or squared luminosity (2).

        Powers of :math:`L_X` are formed only when asked for, because
        :math:`L_X^2` reaches :math:`10^{94}` and is ``inf`` in single
        precision, where the counts view must still be finite.

        ``branch`` selects one side of the obscured/unobscured split, and the
        factor goes **inside** this integral rather than outside it.  That is
        not a detail: :math:`f_{\rm obs}` is a function of :math:`L_X`, which is
        the variable being integrated over, so pulling it out would evaluate it
        at the mean luminosity of the selected population instead of weighting
        each luminosity by its own obscured fraction.  The two differ most
        exactly where the fraction varies fastest.

        ``lweight`` is a further factor on the luminosity grid, and goes inside
        for the same reason: :meth:`band_factor` is a function of
        :math:`L_X` through the obscured fraction.
        """
        w = prob * self._selection(p, field=field, band=band)
        if lweight is not None:
            w = w * lweight
        if branch is not None:
            f = obscured_fraction(self.loglx, z, obsc, log10m=log10m)
            w = w * (f if branch == "obscured" else 1.0 - f)
        if power:
            w = w * jnp.power(10.0, power * self.loglx)
        return jnp.sum(w, axis=-1) * self.dlx

    def band_factor(self, p: AgnParams, band, *, frame: str = "observer",
                    z=0.0, obsc: "ObscurationParams" = None, log10m=None,
                    power: int = 1, branch=None):
        r""":math:`\langle K^{n}\rangle(L_X)`, ``n = power``: what carries a
        hard-band luminosity into ``band``, on the luminosity grid.

        .. math::

            K_{\rm b} = \frac{\int_{E_1(1+z)}^{E_2(1+z)} E^{1-\Gamma}\,
                              e^{-\tau_{\rm b}(E)}\,dE}
                             {\int_2^{10} E^{1-\Gamma}\,dE},
            \qquad
            \langle K^n\rangle = (1-f_{\rm obs})\,K_{\rm u}^n
                                 + f_{\rm obs}\,K_{\rm o}^n ,

        the energy an AGN of hard-band luminosity :math:`L_X` sends into the
        band, averaged over whether it is obscured.  ``frame="observer"`` puts
        the limits at :math:`E(1+z)`, so the K-correction falls out of them as
        it does in :func:`absorbed_band_energy`; ``"rest"`` puts them at
        :math:`E`, with the obscured branch still absorbed.  The unobscured
        branch is at zero column, where :func:`absorbed_band_energy` is exactly
        :math:`(1+z)^{2-\Gamma}` :func:`band_energy`, and is written so.

        **The power is inside the average.**  An AGN is obscured or it is not,
        so the self-pair carries :math:`\langle K^2\rangle` and not
        :math:`\langle K\rangle^2`.  In the soft band the two branches differ
        by an order of magnitude: at the default column, :math:`10^{22.5}`, an
        obscured source passes 4.2% of its 0.5--2 keV energy at
        :math:`z = 0` and 11% in the observed band at :math:`z = 0.27`.

        ``z`` is the source redshift, which the obscured fraction reads in both
        frames.  ``log10m`` tilts the split with halo mass, as in
        :meth:`occupation`.

        Needs ``obscuration="split"``: in a soft band the split *is* the
        spectrum, and a sector built without it would count every AGN as
        unabsorbed -- 0.59 of them are obscured at :math:`10^{42}` erg/s and
        :math:`z = 0.2`.

        ``branch`` returns one term of the sum instead: ``"unobscured"`` is
        :math:`(1-f_{\rm obs})K_{\rm u}^n` and ``"obscured"`` is
        :math:`f_{\rm obs}K_{\rm o}^n`, and the two add to ``branch=None``
        exactly.  The terms are what a detector needs apart: counts are an
        energy-conversion factor times each branch's luminosity, the two
        factors differ -- by up to :math:`10^6` in a 100 eV band at the soft
        end, where an absorbed source's counts are photons redistributed down
        from harder energies -- and the branch weight sits inside the
        luminosity integral, so once summed the two cannot be separated.
        (0.9.3; ggah-cal-29's requirement for the 15-band fit.)
        """
        if self.obscuration != "split":
            raise ValueError(
                "an AGN luminosity carried into another band needs the "
                "obscured/unobscured split, and this sector was built with "
                "obscuration='none'.  In the soft band the split is the "
                "spectrum: an obscured source at the default column passes a "
                "few per cent of its 0.5-2 keV energy.  Build it with "
                "AgnSector(..., obscuration='split').")
        if frame not in ("observer", "rest"):
            raise ValueError(f"unknown band frame {frame!r}; expected "
                             f"'observer' or 'rest'")
        e1, e2 = float(band[0]), float(band[1])
        if not 0.0 < e1 < e2:
            raise ValueError(f"a band needs 0 < emin < emax, got ({e1}, {e2})")
        zb = jnp.asarray(z) if frame == "observer" else jnp.asarray(0.0)
        hard = band_energy(2.0, 10.0, p.gamma_x)
        k_u = (jnp.power(1.0 + zb, 2.0 - p.gamma_x)
               * band_energy(e1, e2, p.gamma_x) / hard)
        k_o = absorbed_band_energy(e1, e2, p.gamma_x, p.log10_nh, zb) / hard
        f = obscured_fraction(self.loglx, z, obsc, log10m=log10m)
        unobscured = (1.0 - f) * k_u ** power
        obscured = f * k_o ** power
        if branch is None:
            return unobscured + obscured
        return {"unobscured": unobscured,
                "obscured": obscured}[self._branch(branch)]

    def _branch(self, branch):
        """Validate a requested branch against the sector's declared mode."""
        if branch is None:
            return None
        if branch not in ("obscured", "unobscured"):
            raise ValueError(
                f"unknown obscuration branch {branch!r}; expected 'obscured' "
                f"or 'unobscured'")
        if self.obscuration == "none":
            raise ValueError(
                f"this sector was built with obscuration='none', which asserts "
                f"no split, so it cannot answer for the {branch!r} population. "
                f"Build it with AgnSector(..., obscuration='split') to make the "
                f"obscured fraction reach a prediction -- and note that is a "
                f"model choice, not a refinement: the two branches sample "
                f"nearly the same halo masses, so the bias difference it "
                f"predicts is a measurement to report, not an assertion.")
        return branch

    # -- satellites ----------------------------------------------------------
    def _satellite_duty(self, p: AgnParams):
        # The dataclass default, not a literal: a second copy of it here stayed
        # at 0 once already.
        return getattr(p, "f_duty_sat", AgnParams.f_duty_sat)

    def has_satellites(self, p: AgnParams) -> bool:
        r"""Whether ``f_duty_sat`` switches satellite AGN on.  ``True`` under
        tracing, where a zero cannot be told from a value (see
        :func:`~ggah_mod.sectors.galaxies._is_active`)."""
        return _is_active(self._satellite_duty(p))

    def _satellite_kernel(self, field, p: AgnParams, galaxy_params):
        r"""``(lg_ms, dn, prob)``: the satellites' stellar masses and the
        luminosity distribution of an active one at each.

        ``dn`` is :meth:`~ggah_mod.sectors.galaxies.GalaxySector.satellite_csmf`,
        shape ``(N_*, NM)``.  ``prob`` is
        :math:`f_{\rm ERDF}\,dP(\log L_X|M_*)/d\log L_X`, shape ``(N_*, N_lx)``:
        the central's black-hole relation and Eddington-ratio distribution at a
        given stellar mass, with the width at fixed :math:`M_*`
        (:meth:`_sigma_bh_at_fixed_mstar`).  Centrals and satellites of equal
        stellar mass have the same black hole.
        """
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        t_grid, k = self._kernel(p, self._sigma_bh_at_fixed_mstar(p),
                                 z=field.z)
        mean_bh = self._mbh(lg_ms, p, field.z)
        prob = self._shift(t_grid, k, p, mean_bh) * jnp.power(10.0, p.log10_ferdf)
        return lg_ms, dn, prob

    def satellite_occupation(self, field, p: AgnParams, galaxy_params,
                             power: int = 0, *, branch=None,
                             obsc: "ObscurationParams" = None, lweight=None):
        r""":math:`N_{\rm sat}^{\rm AGN}(>L_{\min}|M_h)` (``power=0``), or its
        selected luminosity (1) or squared luminosity (2), per host.

        .. math::

            N^{\rm AGN}_{\rm sat}(M_h) = f_{\rm duty}^{\rm sat}\int \dd\lg M_*\,
                \frac{\dd N_{\rm sat}}{\dd\lg M_*}(M_*|M_h)\,
                P(>L_{\min}|M_*)

        No ceiling: a host may carry several active satellites, and they are
        Poisson about its central, which is the rule the one-halo pair term
        already applies.  Computed whatever ``f_duty_sat`` is; :meth:`weights`
        decides whether a satellite component exists at all.

        ``branch`` applies the obscured fraction **without** its halo-mass
        tilt, ``l_blend_m_amp``: the moment is taken per stellar mass, before
        the host is integrated in, so no host mass is there to tilt by.  At the
        default tilt of zero that changes nothing; at a non-zero one only the
        central AGN are sorted by halo mass.
        """
        lg_ms, dn, prob = self._satellite_kernel(field, p, galaxy_params)
        per_mstar = self._moment(prob, p, power, branch=self._branch(branch),
                                 z=field.z, obsc=obsc, field=field,
                                 lweight=lweight)  # (N_*,)
        return self._satellite_duty(p) * jnp.trapezoid(
            dn * per_mstar[:, None], lg_ms, axis=0)

    # -- predictions ---------------------------------------------------------
    def xlf(self, field, p: AgnParams, galaxy_params, band: str = "hard"):
        r"""The **predicted** :math:`\Phi(L_X)` [(Mpc/h)^-3 dex^-1].

        Returns ``(log10_lx_grid, phi)``.  Comparing this to a published XLF is
        a genuine test; an abundance-matched model reproduces one by
        construction and so cannot be tested the same way.

        Satellites are included when :meth:`has_satellites`.  Their halo mass
        is integrated out first, into the satellite stellar-mass function, so
        the :math:`(N_*, N_M, N_{\rm lx})` cube is never formed.
        """
        log10m = jnp.log10(field.m)
        prob = (self.p_loglx_given_m(log10m, p, galaxy_params,
                                     h=field.cosmo.h, z=field.z)
                * jnp.power(10.0, p.log10_ferdf))
        dn_dlog10m = field.dndm * field.m * _LN10
        phi = jnp.trapezoid(dn_dlog10m[:, None] * prob, log10m, axis=0)
        if self.has_satellites(p):
            lg_ms, dn, prob_s = self._satellite_kernel(field, p, galaxy_params)
            smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)   # (N_*,)
            phi = phi + self._satellite_duty(p) * jnp.trapezoid(
                smf_sat[:, None] * prob_s, lg_ms, axis=0)
        grid = self.loglx + (jnp.log10(p.k_h2s) if band == "soft" else 0.0)
        return grid, phi

    def occupation(self, field, p: AgnParams, galaxy_params, *, branch=None,
                   obsc: "ObscurationParams" = None):
        r""":math:`N_{\rm cen}^{\rm AGN}(>L_{\min}|M_h)`, and the central's selected
        luminosity :math:`\langle N_{\rm cen} L_X\rangle` [erg/s].

        The second is **per halo**, not per AGN: it carries the duty cycle.
        It is what :mod:`ggah_mod.sectors.energetics` wants, and it is already
        the emission weight -- the emission view once multiplied it by the
        occupation a second time.
        """
        branch = self._branch(branch)
        prob = (self.p_loglx_given_m(jnp.log10(field.m), p, galaxy_params,
                                     h=field.cosmo.h, z=field.z)
                * jnp.power(10.0, p.log10_ferdf))
        kw = dict(branch=branch, z=field.z, obsc=obsc, field=field,
                  log10m=jnp.log10(field.m))
        n_cen = self._moment(prob, p, 0, **kw)
        lx_cen = self._moment(prob, p, 1, **kw)
        # A halo hosts at most one central AGN.  `tanh` rather than `clip`:
        # a clip puts a tie at 1 and kills the gradient above it, and the
        # occupation genuinely approaches 1 in the most massive halos.
        return jnp.tanh(n_cen), lx_cen

    def l_x_agn(self, field, p: AgnParams, galaxy_params):
        r""":math:`\langle L_X\rangle(M_h)` [erg/s], hard band, **central** AGN.

        **The name is deliberate.**  This is the AGN *point source*; the hot
        gas has its own luminosity,
        :meth:`~ggah_mod.sectors.gas.HotGasDPM.x_ray_luminosity`, and neither
        is ever called ``lx``, so the two X-ray sources cannot be confused.

        Satellite AGN are not in it: this is the central's luminosity alone.
        The feedback budget does not read it --
        :meth:`~ggah_mod.sectors.gas.HotGasDPM.feedback_budget` takes the
        Soltan channel, whose input is the black-hole mass of the central
        *and* the satellites, ``mean_mbh + mean_mbh_satellites``.  The
        luminosity channel that read this array was removed in 0.8.8.

        Refused in single precision rather than returned as ``inf``.  The guard
        is here and **not** in :meth:`occupation`, which forms the same number:
        ``weights`` calls that method and throws the luminosity away, so the
        counts view must keep working in float32 and a guard there would break
        it.  Refuse where the bad number reaches a caller, not where it is
        formed and dropped.
        """
        require_x64("l_x_agn", order="1e44 erg/s",
                    in_range="The counts view -- `weights`, and `occupation`'s "
                             "first return -- is in range and does not need this.")
        return self.occupation(field, p, galaxy_params)[1]

    def effective_bias(self, field, p: AgnParams, galaxy_params, *,
                       branch=None, obsc: "ObscurationParams" = None):
        r""":math:`b_{\rm AGN} = \int b\,N_{\rm AGN}\,dn/dM / \int N_{\rm AGN}\,dn/dM`,
        satellites included when :meth:`has_satellites`.

        ``branch`` asks for one side of the obscured/unobscured split, and is
        the method Petter et al. (2023)'s 9-sigma result is a test of.  It is
        only answerable at ``obscuration='split'``, and the reason is the
        arithmetic of this very expression: an obscured *weight* that depends on
        :math:`L_X` and not on :math:`M_h` cancels between the numerator and the
        denominator, so a single weighted population predicts a ratio of exactly
        one however the fraction is parameterised.
        """
        branch = self._branch(branch)
        kw = dict(branch=branch, obsc=obsc)
        n, _ = self.occupation(field, p, galaxy_params, **kw)
        if self.has_satellites(p):
            n = n + self.satellite_occupation(field, p, galaxy_params, **kw)
        return field.effective_bias(n)

    # -- the sector contract -------------------------------------------------

    def _check_calibration(self, field, galaxy_params):
        """Both of this sector's fitted ranges, against what the field asks for.

        The redshift check is the one the other five registries get.  The
        stellar-mass check is this sector's own: it evaluates Eq. (mbh) on every
        halo in ``field.m``, so the requested range is computable and comparing
        it is not inventing an API for the checker's benefit.

        The price is **not** computed here.  :meth:`validity_cost` costs two
        halo integrals over the black-hole chain, and this runs inside
        ``weights`` on every call of a fit; the message points at it instead.
        """
        check_sector_calibration("AGN chain", self.bh_chain, AGN_CALIBRATION,
                                 field.z, policy=self.calibration)
        check_sector_range(
            "the M_BH-M_star relation of AGN chain", self.bh_chain,
            AGN_CALIBRATION,
            self.log10_mstar(jnp.log10(field.m), galaxy_params,
                             h=field.cosmo.h, z=field.z),
            policy=self.calibration,
            hint="`AgnSector.validity_cost` measures what it is worth; the "
                 "extrapolation below the fitted range is a declared choice, "
                 "not an oversight.")

    def weights(self, field, params: AgnParams, galaxy_params) -> TracerWeights:
        r"""AGN as a **discrete** tracer: the nuclei of central galaxies at the
        halo centre, and of satellite galaxies on the **galaxy** sector's
        satellite profile, because an active satellite is a satellite galaxy.

        ``f_duty_sat`` is the one number the model cannot supply: whether a
        satellite's duty cycle at fixed :math:`M_*` matches a central's.  It
        defaults to one, which says it does.

        With ``f_duty_sat = 0``, the centrals-only opt-out, ``w_extended`` is
        ``None``: with nothing on a profile the one-halo auto-spectrum vanishes
        identically -- a Bernoulli central occupation has no self-pairs.
        Leaving the field ``None`` rather than zero is what makes that visible
        in the contract.
        """
        n_cen, _ = self.occupation(field, params, galaxy_params)
        # `n_cen`, NOT `field.dndm * n_cen`.  The contract is
        # `P^1h = int dM (dn/dM) W_a W_b`, so `W` cannot carry `dn/dM` as well:
        # this sector used to, which put `dndm^2` in every AGN one-halo
        # integrand and `dndm` in every two-halo one, and left `W_agn` with
        # units of (Msun/h)^-1 while `W_gas` and `W_m` are dimensionless.  No
        # test caught it because nothing consumed the weights yet.
        # `self_pair = <N> l^2` with l = 1: these are number counts, so the
        # shot noise is 1/n_bar.  Declared rather than inferred -- the same
        # sector's *emission* view weights each AGN by its luminosity, where
        # <N> l^2 is not <N> l and no algebra on the fused weight recovers it.
        self._check_calibration(field, galaxy_params)
        occ, w_ext = n_cen, None
        if self.has_satellites(params):
            n_sat = self.satellite_occupation(field, params, galaxy_params)
            w_ext = n_sat[None, :] * self.galaxies.satellite_uk(field,
                                                                galaxy_params)
            occ = n_cen + n_sat
        return TracerWeights(w_point=n_cen, w_extended=w_ext, self_pair=occ,
                             norm=field.number_density(occ), discrete=True,
                             bias_weight=None, name="agn")

    def emission_weights(self, field, params: AgnParams, galaxy_params, *,
                         band=None, frame: str = "observer",
                         obsc: "ObscurationParams" = None,
                         branch=None) -> TracerWeights:
        r"""AGN X-ray emission, hard band: a **discrete** population weighted by
        luminosity.

        ``w_point`` is the central's selected luminosity per halo,
        :math:`\int d\log L_X\,P\,S\,L_X = \langle N_{\rm cen}L_X\rangle`, and
        ``self_pair`` is :math:`\langle N_{\rm cen}L_X^2\rangle` -- each AGN
        paired with itself carries its own luminosity squared, not the mean's.
        Satellites, when present, add the same two moments on the galaxy
        satellite profile.  ``norm = 1``: this is an emissivity per unit
        volume, not a per-object quantity.

        The weight used to be :math:`N_{\rm cen}\langle N_{\rm cen}L_X\rangle`
        and the self-pair :math:`N_{\rm cen}\langle N_{\rm cen}L_X\rangle^2`,
        which counted the occupation twice: at the defaults :math:`N_{\rm cen}`
        is 0.01--0.03, so the emission was that factor low in every halo.

        ``band=(E1, E2)`` in keV puts it in another band instead: each AGN's
        energy arriving there, through :meth:`band_factor`, with ``frame`` and
        the obscuration parameters ``obsc`` passed on.  The factor goes inside
        both luminosity moments -- :math:`\langle K\rangle` in the weight,
        :math:`\langle K^2\rangle` in the self-pair -- because it depends on
        :math:`L_X` through the obscured fraction.  ``band=None`` is the hard
        band and the arithmetic of 0.9.1 unchanged.

        ``branch="unobscured"`` or ``"obscured"`` keeps one side of the split:
        in a band through :meth:`band_factor`'s branch terms, and in the hard
        band -- intrinsic, so unabsorbed on both sides -- as the fraction
        :math:`1-f_{\rm obs}` or :math:`f_{\rm obs}` inside the moments.  The
        two branches' weights and self-pairs add to ``branch=None``'s.  An AGN
        is in one branch or the other, never both, so two branch components of
        one tracer pair as one object per halo, not as independent samples;
        :func:`~ggah_mod.spectra.tracers.spectrum` applies that rule.
        """
        require_x64("emission_weights", order="1e44 erg/s, and 1e88 for the "
                                               "self-pair it squares",
                    in_range="The counts view, `weights`, is in range and does "
                             "not need this.")
        prob = (self.p_loglx_given_m(jnp.log10(field.m), params, galaxy_params,
                                     h=field.cosmo.h, z=field.z)
                * jnp.power(10.0, params.log10_ferdf))
        if branch is not None:
            branch = self._branch(branch)
            if self.obscuration != "split":
                raise ValueError(
                    "an obscuration branch needs a sector built with "
                    "obscuration='split'")
        if band is None and branch is not None:
            # The hard band is intrinsic: each branch carries its fraction of
            # every luminosity, and nothing is absorbed on either side.
            lm = jnp.log10(field.m)
            f_c = obscured_fraction(self.loglx, field.z, obsc, log10m=lm)
            f_s = obscured_fraction(self.loglx, field.z, obsc)
            k1 = k2 = f_c if branch == "obscured" else 1.0 - f_c
            k1_sat = k2_sat = f_s if branch == "obscured" else 1.0 - f_s
            name = f"agn:emission/{branch}"
        elif band is None:
            k1 = k2 = k1_sat = k2_sat = None
            name = "agn:emission"
        else:
            kw = dict(frame=frame, z=field.z, obsc=obsc)
            # Centrals tilt with their host's mass; satellites are taken per
            # stellar mass, before a host is integrated in, so they do not --
            # the rule satellite_occupation already states for its branches.
            lm = jnp.log10(field.m)
            kw["branch"] = branch
            k1 = self.band_factor(params, band, log10m=lm, power=1, **kw)
            k2 = self.band_factor(params, band, log10m=lm, power=2, **kw)
            k1_sat = self.band_factor(params, band, power=1, **kw)
            k2_sat = self.band_factor(params, band, power=2, **kw)
            name = (f"agn:emission[{band[0]:g}-{band[1]:g}keV/{frame[0]}]"
                    + (f"/{branch}" if branch else ""))
        lx = self._moment(prob, params, 1, field=field, lweight=k1)
        lx2 = self._moment(prob, params, 2, field=field, lweight=k2)
        w_ext = None
        if self.has_satellites(params):
            lx_sat = self.satellite_occupation(field, params, galaxy_params, 1,
                                               lweight=k1_sat)
            lx2 = lx2 + self.satellite_occupation(field, params, galaxy_params,
                                                  2, lweight=k2_sat)
            w_ext = lx_sat[None, :] * self.galaxies.satellite_uk(field,
                                                                 galaxy_params)
        return TracerWeights(w_point=lx, w_extended=w_ext,
                             norm=jnp.asarray(1.0), discrete=True,
                             bias_weight=None, name=name,
                             self_pair=lx2)


    # ==================================================================
    # The black holes themselves, rather than the ones that are shining
    # ==================================================================
    def mean_mbh(self, log10m, p: AgnParams, galaxy_params, *, h, z=0.0):
        r""":math:`\langle M_{\rm BH}\rangle(M_h)` of the **central** [Msun], **not**
        :math:`10^{\langle\log M_{\rm BH}\rangle}`.

        .. math::

            \langle M_{\rm BH}\rangle
                = 10^{\langle\log_{10}M_{\rm BH}\rangle}\,
                  \exp\!\left[\tfrac12(\ln 10\,\sigma_{\rm lm})^2\right]

        The lognormal correction is not decoration.  Measured at the published
        parameters :math:`\sigma_{\rm lm} = 0.356` dex, so it is a factor
        **1.40**: a mass density built from :math:`10^{\langle\log M\rangle}` is
        **0.714** of the right answer, and low in a way that looks plausible
        because both numbers are the right order.  Note the width is
        :meth:`_sigma_lm` and not ``sig_bh``: the scatter that matters here is
        the one at fixed *halo* mass, which the stellar-mass scatter widens.  The same scatter is already carried by :meth:`_sigma_lm`, so
        this reuses it rather than declaring a second width.

        Every central above the stellar-mass cut hosts one, which is the
        modelling choice worth naming, and the result carries
        :meth:`central_fraction`: the
        :meth:`occupation` above counts *active* nuclei above a luminosity
        threshold, and a mass density wants all of them.  The satellites' black
        holes are :meth:`mean_mbh_satellites`; this is the central's alone.
        The feedback budget wants both:
        :meth:`~ggah_mod.sectors.gas.HotGasDPM.feedback_budget` hands
        :mod:`~ggah_mod.sectors.energetics` the sum
        ``mean_mbh + mean_mbh_satellites``, every black hole in the halo, as
        :meth:`omega_bh` counts them.
        """
        lg = self.mean_log10_mbh(log10m, p, galaxy_params, h=h, z=z)
        sig = self._sigma_lm(p)
        frac = self.central_fraction(log10m, p, galaxy_params, h=h, z=z)
        return (jnp.power(10.0, lg) * jnp.exp(0.5 * (jnp.log(10.0) * sig) ** 2)
                * frac)

    def mean_mbh_satellites(self, field, p: AgnParams, galaxy_params):
        r"""Black-hole mass in a host's satellites [Msun per host].

        .. math::

            \int \dd\lg M_*\,\frac{\dd N_{\rm sat}}{\dd\lg M_*}(M_*|M_h)\,
                10^{\langle\log M_{\rm BH}\rangle(M_*)}\,
                \exp\!\left[\tfrac12(\ln 10\,\sigma_{\rm BH})^2\right]

        The central's relation at the satellite's stellar mass, and every
        satellite hosts one: ``f_duty_sat`` is a duty cycle, which says whether
        a black hole is shining, not whether it is there.  The width is
        :meth:`_sigma_bh_at_fixed_mstar`, because the stellar mass is already
        distributed by the conditional stellar-mass function.
        """
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        sig = self._sigma_bh_at_fixed_mstar(p)
        mbh = (jnp.power(10.0, self._mbh(lg_ms, p, field.z))
               * jnp.exp(0.5 * (jnp.log(10.0) * sig) ** 2))
        return jnp.trapezoid(dn * mbh[:, None], lg_ms, axis=0)

    def black_hole_mass_function(self, field, log10_mbh, p: AgnParams,
                                 galaxy_params):
        r""":math:`dn/d\log_{10}M_{\rm BH}` [(Mpc/h)^-3 dex^-1].

        .. math::

            \frac{dn}{d\log M_{\rm BH}} = \int dM\,\frac{dn}{dM}\,
                F_{\rm c}(M)\,\mathcal{N}\!\left(\log M_{\rm BH};
                \langle\log M_{\rm BH}\rangle(M),\ \sigma_{\rm lm}\right)
                + (\text{satellites above the cut})

        A convolution, not a change of variables: with scatter,
        :math:`|dM/dM_{\rm BH}|` is not the answer and the two differ most where
        the relation is steepest, which is where the knee of the mass function
        is.  This is the local BHMF, and it is a **prediction** -- the chain was
        never abundance-matched to one -- so comparing it to a measured BHMF is
        a test in the same sense :meth:`xlf` is.

        Centrals and satellites both, the satellites through the satellite
        stellar-mass function with their halo mass integrated out first.

        Parameters
        ----------
        log10_mbh : array (Nb,)
            Where to evaluate, in :math:`\log_{10}(M_{\rm BH}/M_\odot)`.
        """
        def gauss(mean, sig):
            x = (jnp.asarray(log10_mbh)[:, None] - mean[None, :]) / sig
            return jnp.exp(-0.5 * x ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))

        lg_mean = self.mean_log10_mbh(jnp.log10(field.m), p, galaxy_params,
                                      h=field.cosmo.h, z=field.z)
        frac = self.central_fraction(jnp.log10(field.m), p, galaxy_params,
                                     h=field.cosmo.h, z=field.z)
        phi_cen = field.integrate(
            (field.dndm * frac)[None, :] * gauss(lg_mean, self._sigma_lm(p)),
            axis=-1)
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)   # (N_*,)
        g_sat = gauss(self._mbh(lg_ms, p, field.z),
                      self._sigma_bh_at_fixed_mstar(p))              # (Nb, N_*)
        phi_sat = jnp.trapezoid(g_sat * smf_sat[None, :], lg_ms, axis=-1)
        return phi_cen + phi_sat


    def validity_cost(self, field, p: AgnParams, galaxy_params, *,
                      quantity: str = "omega_bh"):
        r"""What the :math:`M_\bullet`-:math:`M_\star` extrapolation is worth, as a number.

        The quantity computed with every **galaxy** -- central or satellite --
        counted only when its stellar mass lies inside the fitted range of
        :data:`AGN_CALIBRATION`, divided by the same quantity unrestricted: a
        satellite by its own stellar mass, a central by the mean
        :math:`M_*^{\rm c}(M_h)` of its halo, both after the stellar-mass cut.
        ``1.0`` means the extrapolation contributes nothing; ``0.7`` means three
        tenths of the answer comes from outside the data.

        The mask is on each galaxy, not on its host: a black hole sees its own
        galaxy's stellar mass, so a satellite inside the range counts even when
        its central is above it, and one below the range does not count merely
        because its central is inside.  It used to mask the satellites by their
        central's stellar mass, which put the whole of a massive host's
        satellite population inside the fit.

        This is the house rule in one method.  A fitted range is not a boolean:
        evaluating a relation 0.1 dex below its floor and 3 dex below it are not
        the same act, and a warning that fires identically for both says
        nothing.  So the sector measures the price and
        :func:`~ggah_mod.sectors.calibration.check_sector_range` prints it,
        rather than either refusing at any excursion or staying silent.

        It is a *diagnostic* and deliberately not a correction.  Nothing here
        clamps, truncates or reweights the relation: Eq. (mbh) stays a straight
        line below its fitted range by decision, and this number is what that
        decision costs.

        Parameters
        ----------
        quantity : {"omega_bh", "n_agn", "l_x"}
            ``omega_bh`` is the mass density -- the census row, and the one the
            paper quotes against Fukugita & Peebles; it counts every black
            hole, whatever ``f_duty_sat`` is.  ``n_agn`` is the selected AGN
            number density and ``l_x`` their summed luminosity, centrals and
            satellites as :meth:`xlf` counts them, which are what a fit to a
            luminosity function sees.

        Returns
        -------
        float-like
            The ratio, a traced scalar.
        """
        entry = AGN_CALIBRATION.get(self.bh_chain)
        rng = None if entry is None else entry.mstar_range
        log10m = jnp.log10(field.m)
        if rng is None:
            return jnp.asarray(1.0)
        lo, hi = rng
        lg_ms = self.log10_mstar(log10m, galaxy_params, h=field.cosmo.h, z=field.z)
        inside = ((lg_ms >= lo) & (lg_ms <= hi)).astype(field.dndm.dtype)

        if quantity == "omega_bh":
            per_cen = self.mean_mbh(log10m, p, galaxy_params,
                                    h=field.cosmo.h, z=field.z)
            # `mean_mbh_satellites` per stellar mass, before the M_* integral.
            lg_s, dn = self._satellite_csmf(field, galaxy_params)
            sig = self._sigma_bh_at_fixed_mstar(p)
            per_sat = (jnp.power(10.0, self._mbh(lg_s, p, field.z))
                       * jnp.exp(0.5 * (jnp.log(10.0) * sig) ** 2))
            duty = 1.0
        elif quantity in ("n_agn", "l_x"):
            power = 0 if quantity == "n_agn" else 1
            prob = (self.p_loglx_given_m(log10m, p, galaxy_params,
                                         h=field.cosmo.h, z=field.z)
                    * jnp.power(10.0, p.log10_ferdf))
            per_cen = self._moment(prob, p, power, field=field)
            # `satellite_occupation` per stellar mass, before the M_* integral.
            lg_s, dn, prob_s = self._satellite_kernel(field, p, galaxy_params)
            per_sat = self._moment(prob_s, p, power, z=field.z, field=field)
            duty = self._satellite_duty(p)
        else:
            raise ValueError(
                f"unknown quantity {quantity!r}; expected one of "
                f"'omega_bh', 'n_agn', 'l_x'")
        inside_s = ((lg_s >= lo) & (lg_s <= hi)).astype(per_sat.dtype)

        def per_host(sat_weight):
            return jnp.trapezoid(dn * (per_sat * sat_weight)[:, None], lg_s, axis=0)

        full = field.integrate(field.dndm * (per_cen + duty * per_host(1.0)))
        restricted = field.integrate(
            field.dndm * (per_cen * inside + duty * per_host(inside_s)))
        return restricted / full

    def omega_bh(self, field, p: AgnParams, galaxy_params):
        r""":math:`\Omega_{\rm BH} = \rho_{\rm BH}/\rho_{c,0}`.

        .. math::

            \Omega_{\rm BH} = \frac{h}{\rho_{c,0}}\int dM\,\frac{dn}{dM}\,
                              \left[\langle M_{\rm BH}\rangle(M)
                              + M_{\rm BH}^{\rm sat}(M)\right]

        with :math:`\langle M_{\rm BH}\rangle` of :meth:`mean_mbh` (it carries
        :meth:`central_fraction`) and :math:`M_{\rm BH}^{\rm sat}` of
        :meth:`mean_mbh_satellites`, both above the stellar-mass cut.

        Fukugita & Peebles (2004) row 3.13 put this at :math:`10^{-5.4}`, i.e.
        :math:`4\times10^{-6}` -- below the one per cent of :math:`\Omega_b`
        that the rest of the census is scoped to, and here anyway because it
        costs one integral over a chain this package already has.

        Centrals and satellites, each with a black hole from the same relation
        (:meth:`mean_mbh`, :meth:`mean_mbh_satellites`).

        **Measured, it comes out at** :math:`3.57\times10^{-6}` at :math:`z = 0`
        on the 1.1.0.dev2 defaults (the eROSITA c030 MAP), 0.89 of it, and 0.80 of
        their share of :math:`\Omega_b` at this package's :math:`\Omega_b`.  The
        centrals alone give :math:`2.37\times10^{-6}`; the satellites add 50%,
        counted down to the :math:`10^8\,M_\odot` cut and not below (module
        docstring).  At the 0.8.7-1.1.0.dev1 defaults, the MAP of a mock fit,
        it was :math:`9.22\times10^{-7}` (8.64e-7 on the 0.8.5 galaxies), low
        by ~4.3 (4.8 as a share of :math:`\Omega_b`): that fit's lower ``mu_bh`` and ``sig_bh``, which the eROSITA
        fit moves back near their published values.  (At Paper I's iHOD and the published AGN parameters,
        before the cut: 2.42e-6, 1.62e-6 and 49%.)  Both rose by 1.7
        when the stellar mass reached :func:`mbh_powell` in physical solar
        masses rather than in the occupation's :math:`h^{-2}M_\odot`.  What remains is
        the :math:`M_\bullet`-:math:`M_\star` relation being extrapolated well
        below where it was fitted, which also makes
        :math:`M_\bullet/M_\star` *rise* toward low mass -- the opposite of
        what is observed below :math:`M_\star \sim 10^{10}`, and the reason the
        Soltan channel in :mod:`~ggah_mod.sectors.energetics` dominates at
        :math:`10^{11}` when it should not.

        **What would supply a relation fitted down there**: TRINITY (Zhang, Behroozi, Volonteri et al.
        2023, MNRAS 518, 2123, arXiv:2105.10474), which infers the
        halo-galaxy-SMBH connection from :math:`z = 0` to 10 against quasar
        luminosity functions, and gives black-hole mass functions and an
        :math:`M_\bullet`-bulge relation with evolving slope and normalisation
        rather than one power law extrapolated.

        **The `h` is real.**  :math:`\langle M_{\rm BH}\rangle` is in
        :math:`M_\odot` -- the :math:`M_\bullet`-:math:`M_*` relation is
        published in physical solar masses -- while ``dn/dM`` and
        :data:`~ggah_mod.cosmology.constants.RHO_CRIT0` are in
        :math:`M_\odot/h` per :math:`({\rm Mpc}/h)^3`.  Multiplying by
        :math:`h` converts the numerator, and omitting it inflates the answer by
        :math:`1/h = 1.48` at the fiducial -- almost exactly cancelling the
        1.40 of the lognormal correction above if *both* are dropped, which is
        how a pair of errors can hide inside a number that looks right.
        """
        mbh = (self.mean_mbh(jnp.log10(field.m), p, galaxy_params,
                             h=field.cosmo.h, z=field.z)
               + self.mean_mbh_satellites(field, p, galaxy_params)) * field.cosmo.h
        return field.integrate(field.dndm * mbh) / C.RHO_CRIT0
