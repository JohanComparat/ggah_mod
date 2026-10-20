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
               - s\,(\log M_*^{\rm c} - 10.5)
           = \underbrace{(\log M_{\rm BH} - \langle\log M_{\rm BH}\rangle)
               + s\,\delta_*}_{\mathcal{N}(0,\sigma_{\rm lx})} + \log\tilde\lambda

with :math:`M_*^{\rm c}(M_h)` the central's mean stellar mass and
:math:`\delta_*` its scatter about it, :math:`s` = ``log10_lstar_mslope`` the
slope of the Eddington-ratio break with stellar mass (:func:`lstar_of_mstar`),
and :math:`\tilde\lambda = \lambda\,\lambda_*(10^{10.5}\,M_\odot)/\lambda_*(M_*)`,
distributed at every stellar mass as the ERDF of the pivot.  Neither
:math:`\sigma_{\rm lx}` (:meth:`AgnSector._sigma_lx`) nor that distribution
depends on halo mass.  So the kernel :math:`K = {\rm ERDF} \ast \mathcal{N}` is
computed **once** and shifted per halo -- :math:`O(N_M N_{L})` rather than an
:math:`(N_M, N_L, N_\lambda)` cube.  At the default :math:`s = 0` the shift is
:math:`\langle\log M_{\rm BH}\rangle` alone and :math:`\sigma_{\rm lx}` the
width of :math:`\log M_{\rm BH}` at fixed halo mass, :math:`\sigma_{\rm lm}`.

The bolometric correction
-------------------------

The shift is in :math:`\log L_{\rm bol}`, and the bolometric correction
:math:`K_X = L_{\rm bol}/L_X` carries the kernel to the 2--10 keV grid
(:meth:`AgnSector._shift`).
``AgnSector(kbol="duras20")``, the default, takes the general
:math:`K_X(L_{\rm bol})` of Duras et al. (2020, Eq. 2 and Table 1),
:func:`log10_kx_duras20`, which rises from 11.1 at :math:`L_X = 10^{42}` erg/s to
18.5 at :math:`10^{44}` and 99 at :math:`10^{45}`, and places the kernel at
:math:`\log L_{\rm bol}(L_X)` times :math:`d\log L_{\rm bol}/d\log L_X`; it does
not read ``k_bol``.
``kbol="constant"`` takes :math:`K_X` = ``k_bol``, so that
:math:`\log L_X = \log L_{\rm bol} - \log k_{\rm bol}` and the kernel is
shifted without a Jacobian.

Predicting the XLF rather than fitting one
------------------------------------------

.. math::  \Phi(L_X) = \int d\log_{10}M_h\;\frac{dn}{d\log_{10}M_h}\,
                       f_{\rm ERDF}(M_*^{\rm c})\,F_{\rm c}(M_h)\,P(\log L_X | M_h)
                       + f_{\rm duty}^{\rm sat}\int_{\lg M_{*,\min}} d\lg M_*\,
                       f_{\rm ERDF}(M_*)\,\phi_{\rm s}(M_*)\,P(\log L_X | M_*)

with :math:`F_{\rm c}` the fraction of centrals above the stellar-mass cut and
:math:`\phi_{\rm s}` the satellites' stellar-mass function above it (below).
:math:`f_{\rm ERDF}(M_*)` is the active fraction, :func:`duty`:
``10**log10_ferdf`` at the pivot :math:`10^{10.5}\,M_\odot`, tilted by
``log10_ferdf_mslope`` per dex of stellar mass, a central's at its halo's mean
stellar mass :math:`M_*^{\rm c}(M_h)`; at a slope of zero it is the
constant it was until 1.3.0.dev0.

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

At one, a satellite galaxy is as likely to host an active nucleus as a central
of the same stellar mass; zero is the centrals-only opt-out, under which an AGN
auto-spectrum has **no one-halo term at all**, because a Bernoulli central
occupation has no self-pairs.  The default is the eROSITA x DESI YR3 MAP,
which the clustering of the AGN samples sets (:class:`AgnParams`).  On the
``zumandelbaum15`` defaults at :math:`z = 0.135` and :math:`L_X > 10^{42}`,
satellites are 1.52% of the AGN at the default ``f_duty_sat = 0.046`` (the
eROSITA x DESI YR3 MAP, :class:`AgnParams`), and :math:`b_{\rm eff}` is
1.016 against 1.002 for centrals alone.
At ``f_duty_sat = 1`` and the published AGN parameters they were 30%, 1.051
and 0.764 (0.8.6): the largest clustering choice in this sector.

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
  halo mass, :math:`\sigma_{M_*}(M_h)` of the galaxy sector
  (:meth:`AgnSector.sigma_mstar`), is folded into the width of
  :math:`\log M_\bullet`.
  So a halo's central term is weighted by the fraction of its centrals above
  the cut, :math:`\tfrac12{\rm erfc}[(\lg M_{*,\min} - \lg M_*^{\rm c}(M_h))
  /(\sqrt2\,\sigma_{M_*})]` (:meth:`AgnSector.central_fraction`), and the
  black-hole distribution of the centrals that remain is **not** re-conditioned
  on the cut.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from ..cosmology import constants as C
from ..numerics import invert_monotone, require_x64
from .calibration import (Calibration, check_sector_calibration,
                          check_sector_range)
from .params import Flat, Gaussian, Param, SectorParams, sector_params
from .galaxies import THRESHOLD_SHMR, _is_active
from .protocol import TracerWeights

__all__ = ["AgnParams", "AgnSector", "LG_MSTAR_MIN", "LG_MSTAR_DUTY_PIVOT",
           "DUTY_CEILING_SHARPNESS",
           "LG_MSTAR_LSTAR_PIVOT",
           "log10_duty", "duty", "AGN_PUBLISHED", "OBSCURATION_PUBLISHED", "erdf", "xlf_aird15", "xlf_ueda14",
           "XLF", "make_xlf", "obscured_fraction", "compton_thick_fraction",
           "photoabs_cross_section", "photoabs_tau", "absorbed_band_energy",
           "band_energy",
           "lstar_of_z", "lstar_of_mstar",
           "AGN_CALIBRATION",
           "LBOL_COEF", "K_BOL_HARD", "K_BOL", "HARD_TO_SOFT", "GAMMA_X",
           "band_energy_ratio",
           "xlf_in_h_units", "ObscurationParams",
           "BH_CHAINS", "mbh_powell", "mbh_trinity", "erdf_trinity",
           "LOG10_NH_CT", "CATALOGUE_BRANCHES", "obscuration_branches",
           "log10_absorbed_band_energy", "CATALOGUE_NH_CLASSES", "F_THICK",
           "CATALOGUE_NH_MODES", "F_SCATTER", "AIRD15_BAND", "AIRD15_GAMMA",
           "AIRD15_LOG_NH", "KBOL_MODES", "DURAS20_KX", "DURAS20_LOG_LBOL_RANGE",
           "DURAS20_SPREAD", "log10_kx_duras20", "dlog10_kx_duras20",
           "log10_lbol_of_lx_duras20"]

#: The published values of the eight parameters the L3-8 AGN fits set, and
#: the ``AgnParams`` defaults up to 0.8.6: Powell et al. (2022) Model 1 for the
#: black-hole relation, Powell et al. (2022) App. B for the ERDF (their
#: Fig. 12: lambda* = 0.13, so log10 = -0.8861, delta1 = 0.29 +/- 0.14, and
#: delta2 = 3.70, which they report unconstrained; the functional form is
#: Ananna et al. 2022's), f_ERDF = 10^-1.5 and satellites as active as
#: centrals.  The Gaussian priors are still centred here.
AGN_PUBLISHED = dict(mu_bh=7.76, al_bh=0.67, sig_bh=0.33, log10_lstar=-0.8861,
                     delta1=0.30, delta2=3.70, log10_ferdf=-1.5, f_duty_sat=1.0,
                     log10_ferdf_mslope=0.0, log10_lstar_mslope=0.0,
                     gamma_x=1.8)       # GAMMA_X, the catalogue convention

#: The obscured fraction of Comparat et al. (2019) Eqs. 5-10 as published, as
#: :class:`ObscurationParams` keywords: ``ObscurationParams(**OBSCURATION_PUBLISHED)``
#: is their Fig. 6.  ``l_blend_m_amp`` = 0 is their form (no halo-mass term).
OBSCURATION_PUBLISHED = dict(f_faint_norm=0.9, l_faint_pivot=41.0, f_bright_floor=0.01,
                             z_bright_scale=4.0, f_bright_amp=0.4, l_blend_0=43.2,
                             l_blend_z_amp=1.2, l_blend_width=0.6, l_blend_m_amp=0.0)

#: No galaxy below this stellar mass hosts an AGN or a black hole
#: [log10 Msun, physical].  See the module docstring.
LG_MSTAR_MIN = 8.0
#: Upper end of the satellite stellar-mass grid [log10 Msun, physical], above
#: the conditional stellar-mass function's own range (12 in h^-2 Msun).
LG_MSTAR_MAX_SAT = 12.5
#: Nodes per dex of that grid: the galaxy sector's own spacing.
N_MSTAR_PER_DEX = 73
#: The pivot of the active fraction's stellar-mass slope, ``log10_ferdf_mslope``
#: [log10 Msun, physical]: ``log10_ferdf`` is the active fraction here
#: (:func:`log10_duty`).
LG_MSTAR_DUTY_PIVOT = 10.5
#: Sharpness :math:`k` of the smooth minimum that keeps the active fraction
#: (:func:`duty`) at or below one.  A power of two, so the products by
#: :math:`k` are exact: the capped fraction equals the uncapped one to the last
#: bit wherever that is below 0.469, is 0.989 where it is one, and is one to
#: double precision where it is two.
DUTY_CEILING_SHARPNESS = 64.0
#: The pivot of the Eddington-ratio break's stellar-mass slope,
#: ``log10_lstar_mslope`` [log10 Msun, physical]: ``log10_lstar`` is the break
#: here (:func:`lstar_of_mstar`).  The duty cycle's pivot, so both constants of
#: the Eddington-ratio distribution are defined at one stellar mass.
LG_MSTAR_LSTAR_PIVOT = LG_MSTAR_DUTY_PIVOT

#: :math:`L_{\rm bol} = 1.26\times10^{38}\,(M_{\rm BH}/M_\odot)\,\lambda` [erg/s].
#: The Eddington luminosity for solar-abundance ionised gas.
LBOL_COEF = 1.26e38
#: Bolometric correction to the hard (2--10 keV) band at
#: ``AgnSector(kbol="constant")``.
#:
#: Duras et al. (2020) A&A 636, A73 and Marconi et al. (2004) -- traced through
#: the predecessor rather than re-derived, and recorded because the value stood
#: here with no source at all.  It is luminosity-dependent in both, which is
#: why :data:`K_BOL` gives it a Gaussian rather than a point value, and why
#: ``kbol="duras20"`` replaces it by :func:`log10_kx_duras20`.
#: ``PLAN.md`` item **G8**.
K_BOL_HARD = 20.0
#: Photon index of the unobscured power law the band ratio below assumes.
GAMMA_X = 1.8

#: How :class:`AgnSector` converts a bolometric luminosity to the hard band,
#: its ``kbol`` argument: ``k_bol`` at every luminosity, or the general
#: :math:`K_X(L_{\rm bol})` of Duras et al. (2020), :func:`log10_kx_duras20`.
KBOL_MODES = ("constant", "duras20")
#: Duras et al. (2020) A&A 636, A73, Eq. 2 and Table 1, general sample:
#: :math:`K_X = a[1 + (\log_{10}(L_{\rm bol}/L_\odot)/b)^c]`.
DURAS20_KX = {"a": 10.96, "b": 11.93, "c": 17.79}
#: The range of :math:`\log_{10}(L_{\rm bol}/L_\odot)` they fitted it over.
DURAS20_LOG_LBOL_RANGE = (7.5, 14.5)
#: The spread of their relation at fixed :math:`L_{\rm bol}` [dex], their
#: Table 1.  Recorded and not applied (:func:`log10_kx_duras20`).
DURAS20_SPREAD = 0.27
_LOG10_LSUN = math.log10(C.L_SUN_ERG_S)
_LOG10_LBOL_COEF = math.log10(LBOL_COEF)


def _duras20_inside(u):
    r""":math:`\log_{10}K_X` and :math:`d\log_{10}K_X/d\log_{10}L_{\rm bol}` at
    :math:`u = \log_{10}(L_{\rm bol}/L_\odot)` inside
    :data:`DURAS20_LOG_LBOL_RANGE`, where :math:`u > 0`."""
    a, b, c = DURAS20_KX["a"], DURAS20_KX["b"], DURAS20_KX["c"]
    xc = jnp.power(u / b, c)
    log_k = math.log10(a) + jnp.log1p(xc) / math.log(10.0)
    slope = c * xc / (u * (1.0 + xc) * math.log(10.0))
    return log_k, slope


def log10_kx_duras20(log10_lbol):
    r""":math:`\log_{10}K_X(L_{\rm bol})`, with :math:`K_X = L_{\rm bol}/L_X`
    the bolometric correction to the 2--10 keV band, at ``log10_lbol``
    [erg/s].

    .. math::

        K_X(L_{\rm bol}) = a\left[1 + \left(\frac{\log_{10}(L_{\rm bol}/L_\odot)}
                                           {b}\right)^{c}\right],

    with :math:`a = 10.96`, :math:`b = 11.93` and :math:`c = 17.79`, the
    general sample of Duras et al. (2020, Eq. 2 and Table 1), and
    :math:`L_\odot` = :data:`~ggah_mod.cosmology.constants.L_SUN_ERG_S`.
    :math:`K_X` is 10.96 at the faint end and rises to 363 at the top of their
    range, :math:`\log_{10}(L_{\rm bol}/L_\odot) = 14.5`.

    **Monotonic.**  The slope :math:`d\log K_X/d\log L_{\rm bol}`
    (:func:`dlog10_kx_duras20`) is at most 0.522, at
    :math:`\log_{10}(L_{\rm bol}/L_\odot) = 13.98`, so
    :math:`\log L_X = \log L_{\rm bol} - \log K_X` increases with
    :math:`L_{\rm bol}` at a rate between 0.478 and 1, and has an inverse
    (:func:`log10_lbol_of_lx_duras20`).

    **Outside** :data:`DURAS20_LOG_LBOL_RANGE`, 7.5--14.5, :math:`\log K_X`
    continues linearly at its slope at the nearer edge: :math:`2.7\times10^{-4}`
    below, where the formula is already flat, and 0.517 above.
    The continuation is :math:`C^1`, keeps the inverse, and stops the formula's
    own growth, which no data constrain, from setting the bright end; in
    :math:`L_X` the range is :math:`10^{40.04}`--:math:`10^{45.52}` erg/s.

    The spread of the relation, :data:`DURAS20_SPREAD` = 0.27 dex, is not
    added: at fixed stellar mass it would widen the luminosity kernel as
    ``sig_bh`` does, and the fit sets that width.
    """
    u = jnp.asarray(log10_lbol) - _LOG10_LSUN
    uc = jnp.clip(u, *DURAS20_LOG_LBOL_RANGE)
    log_k, slope = _duras20_inside(uc)
    return log_k + slope * (u - uc)


def dlog10_kx_duras20(log10_lbol):
    r""":math:`d\log_{10}K_X/d\log_{10}L_{\rm bol}` of :func:`log10_kx_duras20`,
    at ``log10_lbol`` [erg/s]: in [0, 0.522], constant outside
    :data:`DURAS20_LOG_LBOL_RANGE`."""
    u = jnp.asarray(log10_lbol) - _LOG10_LSUN
    return _duras20_inside(jnp.clip(u, *DURAS20_LOG_LBOL_RANGE))[1]


def log10_lbol_of_lx_duras20(log10_lx):
    r""":math:`\log_{10}L_{\rm bol}` [erg/s] of an AGN of 2--10 keV luminosity
    ``log10_lx`` [erg/s]: the root of
    :math:`\log L_{\rm bol} - \log K_X(L_{\rm bol}) = \log L_X`, with
    :math:`K_X` of :func:`log10_kx_duras20`.

    The left side increases with slope 0.478--1, so the root is bracketed in
    :math:`[\log L_X, \log L_X + 12]`; we bisect under a ``stop_gradient`` and
    take one Newton step (:func:`~ggah_mod.numerics.invert_monotone`), so the
    derivative is :math:`d\log L_{\rm bol}/d\log L_X = 1/(1 - d\log K_X/d\log
    L_{\rm bol})`.
    """
    x = jnp.asarray(log10_lx, dtype=float)
    return invert_monotone(lambda y: y - log10_kx_duras20(y), x, x, x + 12.0)


def _convolve_full(a, b):
    r"""The full discrete convolution of a 1-D array with a 1-D array, or with
    each row of a 2-D one: ``jnp.convolve(a, b, mode="full")``, as a matrix
    product.

    ``jnp.convolve`` lowers to a cuDNN convolution, and in float64 cuDNN fails
    to execute it on some GPUs (status 5003 on a V100 with cuDNN 9.24); a
    product of the shifted copies of ``a`` with ``b`` runs on every backend and
    agrees with ``numpy.convolve`` to 6e-16.  The shapes are static, so the
    index matrix is a constant of the trace.
    """
    n, m = a.shape[0], b.shape[-1]
    i = jnp.arange(n + m - 1)[:, None] - jnp.arange(m)[None, :]
    inside = (i >= 0) & (i < n)
    shifted = jnp.where(inside, a[jnp.clip(i, 0, n - 1)], 0.0)
    return shifted @ b if b.ndim == 1 else (shifted @ b.T).T


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

#: Sharpness :math:`k` of the smooth minimum that keeps :func:`obscured_fraction`
#: at or below one: within :math:`7\times10^{-5}` of the unsaturated value
#: wherever that is at most 0.95.
F_OBS_CEILING_SHARPNESS = 100.0
#: Sharpness of the smooth minimum that caps the Compton-thick share at the
#: obscured fraction (:func:`obscuration_branches`): at Comparat et al.'s
#: coefficients the obscured fraction is at least 0.31 against a constant 0.3,
#: and at a gap of 0.01 a sharpness of 1000 leaves the share
#: :math:`\ln(1+e^{-10})/1000 = 4.5\times10^{-8}` below 0.3.
CT_SOFTMIN_SHARPNESS = 1000.0


def _smooth_min(a, b, k):
    r""":math:`-\ln(e^{-ka} + e^{-kb})/k`: below both arguments by at most
    :math:`\ln 2/k`, by :math:`\ln(1 + e^{-k|a-b|})/k` in general, so exact to
    exponentially small terms away from the crossing; :math:`C^\infty`, and
    increasing in both, where ``jnp.minimum`` splits its gradient at the tie
    and gives the larger argument none.  The log-sum-exp form does not
    overflow."""
    return -jnp.logaddexp(-k * jnp.asarray(a), -k * jnp.asarray(b)) / k


#: Comparat et al. (2019)'s Compton-thick fraction, their Eq. 4: a constant
#: 0.3 of all AGN, at every luminosity and redshift.
#:
#: Until 1.3.0.dev0 :func:`compton_thick_fraction` replaced it by a form of
#: this package's own -- 0.3 below an :math:`L_X` that rose with redshift and
#: zero above it, through an erf of 0.25 dex -- which no paper fits and which
#: removed every Compton-thick AGN above :math:`10^{43}` at :math:`z = 0.3`.
#: The catalogue selection reads the Compton-thick share directly, so the
#: adaptation was no longer a detail of a fraction nobody consumed; it is
#: Comparat et al.'s constant again.  Its bright-branch partner is the
#: published one too (``f_bright_amp`` = 0.4, :class:`ObscurationParams`).
F_THICK = 0.3


@jax.jit
def compton_thick_fraction(log10lx, z):
    r"""Comparat et al. (2019) Eq. 4: :data:`F_THICK` = 0.3, the same at every
    :math:`L_X` and :math:`z`, broadcast to their shape.

    A function rather than the bare constant because it is what
    :func:`obscured_fraction` and :func:`obscuration_branches` call, and a
    luminosity- or redshift-dependent Compton-thick fraction -- which is what
    this was until 1.3.0.dev0 -- is a change of this one function.
    """
    log10lx, z = jnp.broadcast_arrays(jnp.asarray(log10lx, dtype=float),
                                      jnp.asarray(z, dtype=float))
    return jnp.full_like(log10lx, F_THICK)


@sector_params
class ObscurationParams(SectorParams):
    r"""Coefficients of the obscured fraction adapted from Comparat et al.
    (2019) Eqs. 4-10, with bounds and reasons.

    Nine numbers that decide which AGN a soft-band survey sees: the eight
    coefficients of :func:`obscured_fraction` and ``l_blend_m_amp``, this
    package's tilt of the transition luminosity with host halo mass.

    All nine are **reasoned Flat** rather than Gaussian: Comparat et al.
    (2019) quote no covariance for their coefficients, so a Gaussian here would
    invent a precision, and the bounds are what each coefficient can mean.
    The defaults of ``f_faint_norm``, ``l_blend_0``, ``l_blend_z_amp``,
    ``l_blend_width`` and ``l_blend_m_amp`` are the eROSITA x DESI YR3 MAP
    (:class:`AgnParams`); the other four are the published coefficients
    (their Eqs. 5-10), which
    :data:`OBSCURATION_PUBLISHED` holds in full, ``l_blend_m_amp`` = 0
    included.
    """

    f_faint_norm: float = 1.0
    l_faint_pivot: float = 41.0
    f_bright_floor: float = 0.01
    z_bright_scale: float = 4.0
    f_bright_amp: float = 0.4
    l_blend_0: float = 43.084
    l_blend_z_amp: float = 0.0
    l_blend_width: float = 0.05
    l_blend_m_amp: float = -0.873

    _STATIC = ()

    _PARAMS = {
        "f_faint_norm": Param(
            1.0, (0.0, 1.0), Flat(), "",
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
            0.4, (0.0, 1.0), Flat(), "",
            "how much obscuration the bright branch gains by high redshift: "
            "Comparat et al. (2019) Eq. 8's 0.4 (0.3 until 1.3.0.dev0). A "
            "fraction added to a fraction, so [0, 1]", "definitional"),
        "l_blend_0": Param(
            43.084, (41.0, 46.0), Flat(), "log10(erg/s)",
            "the luminosity at which the faint and bright branches change "
            "over, at z = 0. Bounded by the range the X-ray luminosity "
            "functions in this module are fitted over", "physical"),
        "l_blend_z_amp": Param(
            0.0, (0.0, 3.0), Flat(), "log10(erg/s)",
            "how far that changeover moves by high redshift. Non-negative "
            "because the crossover tracks L*, which brightens with redshift in "
            "every published XLF", "physical"),
        "l_blend_m_amp": Param(
            -0.873, (-2.0, 2.0), Flat(), "dex per dex",
            "how the obscured/unobscured transition luminosity moves with host "
            "halo mass, about 1e13 Msun/h. ZERO IS THE "
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
            "asks for. The default, -0.873 +- 0.241, is the eROSITA x DESI YR3 MAP's "
            "(AgnParams docstring) and has the other sign; the photometric "
            "fit gives -1.57 +- 0.71, within 0.9 sigma of it. The box is "
            "+/- 2 dex per dex, far wider than either fit", "physical"),
        "l_blend_width": Param(
            0.05, (0.05, 3.0), Flat(), "log10(erg/s)",
            "the width of the changeover in dex. Strictly positive by "
            "definition -- zero is a discontinuity, not a blend -- and above "
            "~3 dex the two branches overlap across the whole luminosity "
            "range and neither describes anything on its own",
            "definitional"),
    }


@jax.jit
def obscured_fraction(log10lx, z, p: "ObscurationParams" = None,
                      log10m=None):
    r"""The obscured fraction of AGN (everything above
    :math:`10^{22}\,{\rm cm}^{-2}`, Compton-thick included), Comparat et al.
    (2019) Eqs. 4-11.

    An error-function blend (their Eqs. 5-6) between a faint branch (Eq. 9),
    :math:`0.9\sqrt{41/\log L_X}`, and a bright one (Eq. 8),
    :math:`f_{\rm thick} + 0.01 + 0.4\,{\rm erf}(z/4)`, which carries the
    Compton-thick population; the changeover at :math:`43.2 + 1.2\,{\rm
    erf}(z)` (Eq. 10), 0.6 dex wide, with :data:`F_THICK` = 0.3.
    With their coefficients, :data:`OBSCURATION_PUBLISHED`, the curve is their
    Fig. 6: 0.94 at :math:`\log L_X = 38`, falling to 0.31 at :math:`z = 0` and
    towards 0.70 at high redshift at the bright end.
    The defaults of :class:`ObscurationParams` move five of the coefficients to
    the eROSITA x DESI YR3 MAP: faint-branch normalisation 1.000, its upper
    bound; changeover at 43.084, with no rise with redshift (``l_blend_z_amp``
    = 0, its lower bound) and 0.050 dex wide, its lower bound; tilted by
    -0.873 dex per dex of host halo mass.
    Every piece is an ``erf``, so the whole thing is smooth.

    **The ceiling is a smooth minimum with one, exact below it.**  A fraction
    cannot exceed one, and the faint branch does at low luminosity.
    The minimum is :math:`-\ln(e^{-kf} + e^{-k})/k` with :math:`k` =
    :data:`F_OBS_CEILING_SHARPNESS`: :math:`C^\infty`, increasing, and within
    :math:`\ln(1 + e^{-k(1-f)})/k < 7\times10^{-5}` of :math:`f` for every
    :math:`f \le 0.95`, where ``jnp.clip(f, 0, 1)`` would split its gradient
    at the tie.
    The floor is never reached: both branches are non-negative and the blend
    is in [0, 1].
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
    return _smooth_min(f_bright + (f_faint - f_bright) * blend, 1.0,
                       F_OBS_CEILING_SHARPNESS)


# -- the catalogue's three branches -----------------------------------------
#: Hydrogen column of the **Compton-thick** branch of ``selection="catalogue"``
#: at ``catalogue_nh="single"`` [log10 cm^-2]; ``"comparat19"`` spreads the
#: branch over :data:`CATALOGUE_NH_CLASSES` instead.
#:
#: Comparat et al. (2019) draw a Compton-thick source's column uniformly in
#: :math:`24 < \log N_{\rm H} < 26`; 24.5 sits inside that range at the end a
#: photoelectric model is least wrong about.  The value is not load-bearing in
#: a soft band and that is the reason one number is enough: at
#: :math:`10^{24.5}\,{\rm cm}^{-2}` the optical depth at the top of an observed
#: 0.2--2.3 keV band is about 90 at :math:`z = 0`, 60 at :math:`z = 0.2` and 30
#: at :math:`z = 0.5`, so the transmitted fraction is :math:`e^{-30}` or less
#: and the absorbed power law is invisible in the band for any column in that
#: range.  Above
#: :math:`10^{24}` Compton scattering, not photoelectric absorption, sets the
#: spectrum; the few per cent of scattered light a real Compton-thick nucleus
#: shows in the soft band is the one way this branch reaches a soft-band
#: catalogue, and it is ``catalogue_scatter`` (:data:`F_SCATTER`), zero by
#: default in this single-column mode and 0.02 under Comparat et al.'s classes.
#: Reflection is not modelled.
LOG10_NH_CT = 24.5

#: Comparat et al. (2019)'s column classes, Sec. 3.2 [log10 cm^-2]: every AGN is
#: given a :math:`\log N_{\rm H}` **uniform** in its class's interval --
#: unobscured in [20, 22), Compton-thin in [22, 24), Compton-thick in [24, 26)
#: -- with the class fractions of their Eqs. 4--15
#: (:func:`obscuration_branches`).  ``selection="catalogue"`` at
#: ``catalogue_nh="comparat19"`` (its default) averages the catalogue
#: selection over these distributions.
CATALOGUE_NH_CLASSES = {"unobscured": (20.0, 22.0), "thin": (22.0, 24.0),
                        "thick": (24.0, 26.0)}
#: How ``selection="catalogue"`` assigns a column to each branch: Comparat et
#: al. (2019)'s uniform classes, or one column per branch (zero, ``log10_nh``,
#: :data:`LOG10_NH_CT`), which is what 1.3.0.dev0 shipped first.
CATALOGUE_NH_MODES = ("comparat19", "single")

#: The fraction of an obscured nucleus's intrinsic power-law flux scattered
#: around the absorber into the line of sight, unabsorbed, in
#: :meth:`AgnSector.xlf_observed_hard`: Aird et al. (2015)'s spectral model
#: (their Sec. 3.1, Fig. 4) carries a scattered component of about 2 per cent
#: beside the absorbed power law and a ``pexrav`` reflection.  Applied to the
#: Compton-thin and Compton-thick classes, not the unobscured one, whose
#: absorber is too thin for the distinction to mean anything -- which leaves a
#: 2 per cent step at :math:`10^{22}`, where the classes change.  It is what lets
#: a Compton-thick AGN reach a hard-band sample at all -- at
#: :math:`10^{24.5}` the absorbed 2--7 keV power law is a few per cent of the
#: scattered one or less.  The soft-band catalogue selection uses the same
#: spectrum (``AgnSector(catalogue_scatter=...)``, this value by default at
#: ``catalogue_nh="comparat19"``), so a Compton-thick AGN is catalogued about
#: :math:`\log_{10} 0.02 = -1.7` dex below its intrinsic luminosity there too.
F_SCATTER = 0.02
#: Aird et al. (2015)'s hard-band luminosity convention (their Sec. 4): an
#: observed 2--7 keV band (Chandra; ASCA's 2--10 keV at the bright end),
#: converted to rest-frame 2--10 keV for a :math:`\Gamma = 1.9` power law with
#: :math:`\log N_{\rm H}` uniform in [20, 21) -- "tracing the 'observed'
#: luminosity", not corrected for absorption.  Their LADE fit (Table 5,
#: :func:`xlf_aird15`) is a function of **this** luminosity.
AIRD15_BAND = (2.0, 7.0)
AIRD15_GAMMA = 1.9
AIRD15_LOG_NH = (20.0, 21.0)


def log10_absorbed_band_energy(e_lo, e_hi, gamma, log10_nh, z=0.0, *,
                               n_e: int = 64):
    r""":math:`\log_{10}` of :func:`absorbed_band_energy`, the same quadrature
    taken as a log-sum-exp.

    Not a convenience.  At a Compton-thick column the integrand is
    :math:`e^{-\tau}` with :math:`\tau` in the hundreds over most of a soft
    band, so the integral is :math:`10^{-30}` or smaller: representable in
    double precision, zero in single, and its logarithm -- which is what a
    catalogue luminosity shift is -- is then :math:`-\infty` with a ``nan``
    gradient in :math:`\Gamma`.  Summing in log space keeps both finite in any
    precision; it agrees with the linear form to round-off wherever that one is
    representable.  The Gauss-Legendre weights are positive, so they enter as
    the log-sum-exp's ``b`` without a sign to track.
    """
    from jax.scipy.special import logsumexp

    from ..halos.profiles import _leggauss_cached
    x, w = _leggauss_cached(n_e)
    a = jnp.log(jnp.asarray(e_lo) * (1.0 + jnp.asarray(z)))
    b = jnp.log(jnp.asarray(e_hi) * (1.0 + jnp.asarray(z)))
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    le = mid + half * jnp.asarray(x)
    log_f = ((2.0 - jnp.asarray(gamma)) * le
             - photoabs_tau(jnp.exp(le), log10_nh))
    return (jnp.log(half) + logsumexp(log_f, b=jnp.asarray(w), axis=-1)) / _LN10


def obscuration_branches(log10lx, z, p: "ObscurationParams" = None,
                         log10m=None) -> dict:
    r"""The three populations a soft-band catalogue is a mixture of.

    .. math::

        f_{\rm CT} = \min{}_{\rm s}\!\left(f_{\rm CT}^{0}(L_X, z),\,
                      f_{\rm obs}\right),\qquad
        f_{\rm thin} = f_{\rm obs} - f_{\rm CT},\qquad
        f_{\rm unobs} = 1 - f_{\rm obs},

    with :math:`f_{\rm obs}` :func:`obscured_fraction`,
    :math:`f_{\rm CT}^0` :func:`compton_thick_fraction` (Comparat et al.'s
    constant 0.3) and :math:`\min_{\rm s}` the smooth minimum of
    :data:`CT_SOFTMIN_SHARPNESS`.  Returned as ``{"unobscured", "thin",
    "thick"}``, each broadcast to :func:`obscured_fraction`'s shape; they add
    to one to round-off.  At the published coefficients this is their Eqs.
    4-15 exactly: :math:`f_{\rm CT} = 0.3`, :math:`f_{\rm thin} = f_{\rm obs} -
    0.3`, and :math:`f_{\rm thin}/(f_{\rm thin} + f_{\rm unobs})` their
    Eq. 15 absorbed fraction among the Compton-thin.

    **Why the Compton-thick share is taken out of the obscured fraction and not
    added beside it.**  Comparat et al. (2019) assign three column ranges --
    :math:`20 < \log N_{\rm H} < 22` unobscured, 22--24 Compton-thin obscured,
    24--26 Compton-thick -- and their obscured fraction (their Eqs. 5--10,
    adapted in :func:`obscured_fraction`) is of **everything above**
    :math:`10^{22}`: the bright branch is :math:`f_{\rm CT} + f_0 +
    a_{\rm b}\,{\rm erf}(z/z_{\rm b})`, so it already carries the Compton-thick
    population, and the faint branch, :math:`0.9\sqrt{41/\log L_X}`, sits above
    their constant 0.3 everywhere it applies.  So the Compton-thick sources are
    a part of :math:`f_{\rm obs}`, the thin ones are the rest, and adding the
    Compton-thick fraction on top would count it twice.

    **Why the minimum.**  At the published coefficients
    :math:`f_{\rm obs} \ge 0.31 > f_{\rm thick}` holds identically and the
    cap is a guard: the bright branch is 0.31 at :math:`z = 0`, the faint one
    above 0.85.  But the coefficients are free, so a fit may push
    :math:`f_{\rm obs}` under 0.3; the smooth minimum keeps
    :math:`f_{\rm thin} \ge 0` wherever they go, without the zero gradient a
    hard minimum leaves in the larger argument, and at the defaults it is 0.3
    to :math:`4.5\times10^{-8}`.

    ``log10m`` tilts :math:`f_{\rm obs}` with halo mass as in
    :func:`obscured_fraction`; the Compton-thick fraction does not tilt, so the
    halo-mass dependence of the split lives in the thin branch and the cap.
    """
    f_obs = obscured_fraction(log10lx, z, p, log10m=log10m)
    f_ct = _smooth_min(compton_thick_fraction(log10lx, z), f_obs,
                       CT_SOFTMIN_SHARPNESS)
    return {"unobscured": 1.0 - f_obs, "thin": f_obs - f_ct, "thick": f_ct}


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



#: The bolometric correction to the hard (2--10 keV) band at
#: ``AgnSector(kbol="constant")``; ``kbol="duras20"`` does not read it, and its
#: gradient there is identically zero.
#:
#: **One object, two containers.**  :mod:`~ggah_mod.sectors.energetics` took the
#: same quantity as a bare ``k_bol=20.0`` keyword, so the correction was sampled
#: in one module and frozen in the other -- half of ``PLAN.md`` item **G6**,
#: closed the way ``C4`` closes the other duplication: by sharing the object
#: rather than by keeping two that agree today.
K_BOL = Param(K_BOL_HARD, (5.0, 100.0), Gaussian(20.0, 5.0), "",
              "bolometric correction to the hard band, read only at "
              "AgnSector(kbol='constant'); kbol='duras20' uses the "
              "luminosity-dependent K_X(L_bol) of Duras et al. (2020) "
              "instead. Varies with luminosity, so a fixed value carries a "
              "real spread",
              "prior")


@sector_params
class AgnParams(SectorParams):
    """The AGN sector's free parameters.

    **The fitted defaults are the MAP of ggah_cal's eROSITA x DESI YR3 AGN
    fit** (option B of ggah_cal's ``docs/agn_defaults.rst``) with the
    bolometric correction of Duras et al. (2020) (``AgnSector(kbol="duras20")``),
    on the galaxy defaults of ggah_mod 1.2.0, with the duty cycle capped at one.
    The data are the spectroscopic samples (DESI YR3 redshifts) of eROSITA AGN
    in three bins of 2-10 keV luminosity (the XLF of Aird et al. 2015 in
    their observed luminosity, n_bar, w_p and the lensing Delta Sigma of HSC
    and KiDS), their cross-correlations with the Legacy Survey galaxies of
    M_* > 10^10 Msun, and p(lambda_SAR | M_*, z) of Igo et al. (2026) with a
    0.30 dex floor, forward-modelled with the black-hole scatter at fixed
    M_* beyond the 0.3 dex they deconvolve and at their h = 0.7: chi2 =
    127.8 for 108 degrees of freedom (Fit 2 of option B).
    Sixteen parameters are free: ``mu_bh``, ``al_bh``, ``sig_bh``,
    ``log10_lstar``, ``log10_lstar_mslope``, ``delta1``, ``delta2``,
    ``log10_ferdf``, ``log10_ferdf_mslope``, ``gamma_x``, ``f_duty_sat`` and
    five coefficients of :class:`ObscurationParams`; the values are rounded
    to three decimals.  The same fit to the photometric samples (XLF, n_bar,
    w(theta) and their cross-correlations, with Igo et al. 2026) agrees with
    it on every shared parameter within 1.4 sigma, ``f_duty_sat`` the
    farthest (0.158 +- 0.066), but at chi2 = 156.0 for 120 it falls 0.25 and
    0.47 dex below the measured densities of its two brightest samples.
    The Gaussian priors keep the published centres of :data:`AGN_PUBLISHED`,
    Powell et al. (2022) Model 1 for ``mu_bh`` and ``sig_bh`` and their App. B
    (Fig. 12) for the ERDF, except that of ``al_bh``, which is Reines &
    Volonteri (2015)'s, 1.05 +- 0.11: the luminosities read ``al_bh`` and
    ``log10_lstar_mslope`` only through their sum, so the prior sets the
    black-hole relation's slope and the fit the break's.

    ``log10_ferdf_mslope`` and ``log10_lstar_mslope`` tilt the active fraction
    and the Eddington-ratio break with host stellar mass about 10^10.5 Msun
    (:func:`log10_duty`, :func:`lstar_of_mstar`).
    Every luminosity observable reads ``al_bh`` and ``log10_lstar_mslope``
    only through their sum, so ``al_bh`` is set by its prior and the
    quantities built on M_BH itself (:meth:`AgnSector.mean_mbh`, the
    black-hole mass function, :meth:`AgnSector.omega_bh`) by ``al_bh`` alone.
    Both slopes at zero give the mass-independent forms exactly.
    """

    mu_bh: float = 7.893
    al_bh: float = 1.055
    sig_bh: float = 0.314
    rho: float = 0.0
    log10_lstar: float = -0.967
    log10_lstar_mslope: float = -2.219
    delta1: float = 0.319
    delta2: float = 2.365
    gam_lam: float = 0.0
    gam_lam_hi: float = 0.0
    z_lam: float = 2.0
    log10_ferdf: float = -0.937
    log10_ferdf_mslope: float = 2.12
    log10lx_min: float = 42.0
    k_bol: float = K_BOL_HARD
    gamma_x: float = 1.837
    log10_nh: float = 22.5
    lx_sel_width: float = 0.05
    log10_flux_min: float = -13.3
    f_duty_sat: float = 0.046

    # No annotation: an annotated assignment in a dataclass body declares a
    # *field*, so `_STATIC: tuple = ()` would make the classification list
    # itself a traced leaf.  Caught by `test_every_declared_field_is_classified`.
    _STATIC = ()

    _PARAMS = {
        "mu_bh": Param(
            7.893, (6.5, 9.0), Gaussian(7.76, 0.30), "log10 Msun",
            "M_BH-M_* normalisation at log10 M_* = 11; the prior is the "
            "published measurement, the default the eROSITA x DESI YR3 MAP "
            "(class docstring)", "prior"),
        "al_bh": Param(1.055, (0.0, 2.0), Gaussian(1.05, 0.11), "",
                       "M_BH-M_* slope. The prior is Reines & Volonteri (2015)'s "
                       "for AGN hosts and their total stellar mass, 1.05 +- 0.11; "
                       "it sets the slope, because every luminosity observable "
                       "reads al_bh only through al_bh + log10_lstar_mslope. "
                       "Negative would invert the relation",
                       "physical"),
        "sig_bh": Param(0.314, (0.02, 1.5), Gaussian(0.33, 0.18), "dex",
                        "intrinsic scatter; the floor keeps the convolution "
                        "kernel wider than the grid spacing", "definitional"),
        "rho": Param(
            0.0, (-1.0, 0.99), Flat(), "",
            "correlation between M_BH and halo mass at fixed M_* -- Powell's "
            "'Model 2'. Bounded below 1 because the M_BH|M_h variance would "
            "otherwise collapse to zero and the kernel become a delta",
            "definitional"),
        "log10_lstar": Param(-0.967, (-2.8, 0.5), Gaussian(-0.8861, 0.20),
                             "log10 lambda_Edd",
                             "ERDF break; above 0.5 the typical AGN would be "
                             "super-Eddington", "physical"),
        "log10_lstar_mslope": Param(
            -2.219, (-3.0, 3.0), Flat(), "",
            "the slope of the Eddington-ratio break with host stellar mass, "
            "log10 lambda*(M*) = log10_lstar(z) + log10_lstar_mslope (log10 M* "
            "- 10.5), M* physical in Msun and the pivot 10^10.5 Msun, the duty "
            "cycle's: a central's at the mean stellar mass of its halo, a "
            "satellite's at its own. It lets the break of Igo et al. (2026)'s "
            "p(lambda_SAR|M*) move with stellar mass without bending the "
            "M_BH-M* relation. Every luminosity observable reads the two only "
            "through al_bh + log10_lstar_mslope and mu_bh - 11 al_bh - 10.5 "
            "log10_lstar_mslope, so they are exactly degenerate there: "
            "al_bh is set by its prior, and the quantities built on M_BH "
            "itself (the black-hole mass function, omega_bh, the Soltan E_AGN) "
            "by al_bh alone. ZERO REPRODUCES THE MASS-INDEPENDENT BREAK "
            "EXACTLY, bit for bit; the default, -2.219, is the eROSITA x DESI "
            "YR3 MAP's (class docstring), with al_bh set by its prior. Flat: no "
            "measurement of it is adopted "
            "here. The box, +-3 dex per dex, moves the break by up to 7.5 dex "
            "at the ends of the 10^8-10^12.5 Msun grid, far wider than any "
            "plausible fit",
            "physical"),
        "delta1": Param(0.319, (0.0, 1.5), Gaussian(0.30, 0.15), "",
                        "faint-end ERDF slope; must be below delta2 for the "
                        "break to be a break", "physical"),
        "delta2": Param(2.365, (1.5, 6.0), Gaussian(3.70, 0.66), "",
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
            -0.937, (-5.0, 0.0), Flat(), "",
            "active fraction at the pivot stellar mass, 10^10.5 Msun "
            "(`log10_duty`; the whole of it while `log10_ferdf_mslope` is "
            "0). The upper bound is definitional: a fraction cannot exceed 1",
            "definitional"),
        "log10_ferdf_mslope": Param(
            2.12, (-3.0, 3.0), Flat(), "",
            "the slope of the active fraction with host stellar mass, "
            "log10 f_ERDF(M*) = log10_ferdf + log10_ferdf_mslope (log10 M* - "
            "10.5), M* physical in Msun and the pivot 10^10.5 Msun: a "
            "central's at the mean stellar mass of its halo, a satellite's at "
            "its own. Motivated by Igo et al. (2026), whose p(lambda_SAR|M*) "
            "the constant duty cycle over-predicts by 0.3-0.6 dex at "
            "10^11-10^12 Msun. ZERO REPRODUCES THE CONSTANT DUTY CYCLE "
            "EXACTLY, bit for bit; the default, 2.12, is the eROSITA x DESI "
            "YR3 MAP's (class docstring). Flat: no measurement of it is "
            "adopted here. The box, +-3 dex per dex, moves the fraction by "
            "up to 7.5 dex at the ends of the 10^8-10^12.5 Msun grid, far "
            "wider than any plausible fit; the fraction is capped smoothly "
            "at 1 (DUTY_CEILING_SHARPNESS)",
            "physical"),
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
            1.837, (1.0, 3.0), Gaussian(GAMMA_X, 0.2), "",
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
            0.046, (0.0, 2.0), Flat(), "",
            "the duty cycle of a satellite AGN at fixed M_*, RELATIVE to a "
            "central's, so 0 means no satellite is active and 1 says one is as "
            "likely to be active as a central of the same stellar mass. The "
            "default, 0.046, is the eROSITA x DESI YR3 MAP's (class "
            "docstring); 1 says a black hole sees its own galaxy whether that "
            "galaxy is a central or a satellite.  Whether the two "
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

def _duty_tilt(log10_mstar, p):
    r""":math:`s\,(\log M_* - 10.5)`, the stellar-mass part of
    :func:`log10_duty`: the one place the slope and the pivot are read."""
    slope = getattr(p, "log10_ferdf_mslope", AgnParams.log10_ferdf_mslope)
    return slope * (jnp.asarray(log10_mstar) - LG_MSTAR_DUTY_PIVOT)


def log10_duty(log10_mstar, p):
    r""":math:`\log_{10}f_{\rm ERDF}(M_*)`, the active fraction of black holes
    in galaxies of stellar mass ``log10_mstar`` (physical
    :math:`\log_{10}M_\odot`): the logarithm of :func:`duty`,

    .. math::

        \log_{10}f_{\rm ERDF}(M_*) = \log_{10}\,{\rm smin}_k\!\left(
            10^{\,{\tt log10\_ferdf} + {\tt log10\_ferdf\_mslope}\,
            (\log_{10}M_* - 10.5)},\ 1\right),

    with :math:`{\rm smin}_k(a, b) = -\ln(e^{-ka} + e^{-kb})/k` and
    :math:`k` = :data:`DUTY_CEILING_SHARPNESS`.

    **Every** duty cycle the sector applies is :func:`duty`, the same number
    in linear form: the central's at :math:`M_*^{\rm c}(M_h)`
    (:meth:`AgnSector.central_duty`), the satellites' at their own stellar
    mass (:meth:`AgnSector._satellite_kernel`), and ``ggah_cal``'s
    :math:`p(\lambda_{\rm SAR}|M_*)` on its own grid.

    The slope's motivation is Igo et al. (2026): the constant duty cycle
    over-predicts their :math:`p(\lambda_{\rm SAR}|M_*)` by 0.3--0.6 dex at
    :math:`10^{11}`--:math:`10^{12}\,M_\odot`.  At slope zero and
    ``log10_ferdf`` below :math:`\log_{10}0.469 = -0.33` the fraction is the
    constant, bit for bit.
    """
    return jnp.log10(duty(log10_mstar, p))


def duty(log10_mstar, p):
    r""":math:`f_{\rm ERDF}(M_*)`, the active fraction of :func:`log10_duty`
    in linear form, broadcast to ``log10_mstar``'s shape.

    The uncapped fraction is formed as :math:`10^{\tt log10\_ferdf}\times
    10^{s(\log M_* - 10.5)}` rather than as one power, so that at :math:`s = 0`
    it multiplies the shipped constant by an exact 1.0.  The smooth minimum
    with one (:data:`DUTY_CEILING_SHARPNESS`) then leaves it unchanged to the
    last bit below 0.469, so every output of the sector at :math:`s = 0` is the
    constant-duty one bit for bit, not to round-off.

    **Capped at one.**  A fraction of black holes cannot exceed one, and with a
    slope the uncapped fraction does: at the defaults (``log10_ferdf`` =
    -0.937, :math:`s` = 2.120) above :math:`\log M_* = 10.94`.  The cap is
    smooth so that the gradient reaches both parameters through it.
    """
    raw = (jnp.power(10.0, p.log10_ferdf)
           * jnp.power(10.0, _duty_tilt(log10_mstar, p)))
    return _smooth_min(raw, 1.0, DUTY_CEILING_SHARPNESS)


def _lstar_tilt(log10_mstar, p):
    r""":math:`s\,(\log M_* - 10.5)` [dex], the stellar-mass part of
    :func:`lstar_of_mstar`, or the slope :math:`s` itself at
    ``log10_mstar=None``: the one place ``log10_lstar_mslope`` and the pivot are
    read."""
    slope = getattr(p, "log10_lstar_mslope", AgnParams.log10_lstar_mslope)
    if log10_mstar is None:
        return slope
    return slope * (jnp.asarray(log10_mstar) - LG_MSTAR_LSTAR_PIVOT)


def lstar_of_mstar(log10_mstar, p, z=0.0):
    r""":math:`\log_{10}\lambda_*(M_*, z)`, the Eddington-ratio break in galaxies
    of stellar mass ``log10_mstar`` (physical :math:`\log_{10}M_\odot`) at
    redshift ``z``:

    .. math::

        \log_{10}\lambda_*(M_*, z) = \log_{10}\lambda_*(z)
            + {\tt log10\_lstar\_mslope}\,(\log_{10}M_* - 10.5),

    with :math:`\log_{10}\lambda_*(z)` of :func:`lstar_of_z`, ``log10_lstar``
    at :math:`z = 0`, and the pivot :data:`LG_MSTAR_LSTAR_PIVOT`.  A central's
    break is at the mean stellar mass of its halo, a satellite's at its own, as
    for :func:`duty`.  At a slope of zero the break is the same at
    every stellar mass.

    The ERDF is a function of :math:`\lambda/\lambda_*`, normalised on the
    sector's :math:`\log\lambda` grid at the pivot, so the sector applies the
    slope as a shift of its luminosity kernel at each stellar mass
    (:meth:`AgnSector._shift`), the grid moving with the break.  The
    luminosity then reads the black-hole relation and the break only through
    :math:`\mu_{\rm BH} + \alpha_{\rm BH}(\log M_* - 11) + s(\log M_* -
    10.5)` and the kernel width :meth:`AgnSector._sigma_lx`, through
    :math:`\alpha_{\rm BH} + s`: ``al_bh`` and ``log10_lstar_mslope`` are
    exactly degenerate in every luminosity observable, and only the quantities
    built on :math:`M_{\rm BH}` itself (:meth:`AgnSector.mean_mbh`,
    :meth:`AgnSector.black_hole_mass_function`, :meth:`AgnSector.omega_bh`)
    tell them apart.
    """
    return (lstar_of_z(z, p.log10_lstar, p.gam_lam, p.gam_lam_hi, p.z_lam)
            + _lstar_tilt(log10_mstar, p))


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
#: faint population the map is made of.  ``"catalogue"`` (1.3.0) is a threshold
#: on the luminosity a catalogue *assigns*, which is not the intrinsic one for
#: an absorbed source: see :meth:`AgnSector.catalogue_selection`.
SELECTION_MODES = ("luminosity", "flux", "none", "catalogue")

#: The branches ``selection="catalogue"`` sums, in the order it sums them.
CATALOGUE_BRANCHES = ("unobscured", "thin", "thick")

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
#: as a share of Omega_b, and E_AGN/E_SN rose towards the low-mass end where it
#: should fall -- 2.42 at 1e11 Msun/h against 0.99 at 1e15 at the 1.1.0.dev2
#: defaults (0.49 and 0.42 at 0.9.0's), measured by the paper's
#: `sector_invariants`.  It read 4.7 at 1e11 when this was written,
#: at earlier defaults and with the black-hole mass 1/h too large in the
#: closure.  Both are properties of a straight line extended past its data.
#: Since the 1.5.0 defaults the ratio no longer rises: 0.88 at 1e11 against
#: 1.07 at 1e15, z = 0.2, with epsilon_AGN = 0.010 since 1.6.0 (3.5 and 4.3
#: at 1.5.0's 0.040).
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
                 lg_mstar_min: float = LG_MSTAR_MIN,
                 catalogue_band=(0.2, 2.3),
                 catalogue_gamma: float = GAMMA_X,
                 catalogue_flux_limit: bool = False,
                 catalogue_nh: str = "comparat19",
                 catalogue_nh_nodes: int = 16,
                 catalogue_nh_classes=None,
                 catalogue_scatter=None,
                 kbol: str = "duras20"):
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
            shapes.  At the 1.5.0 defaults the default luminosity grid, from
            :math:`10^{39}` erg/s, holds 0.9999 of the kernel of the central of
            a :math:`10^{15}\,h^{-1}M_\odot` halo and 0.9994 of that of a
            :math:`10^{16}\,h^{-1}M_\odot` one, because
            ``log10_lstar_mslope`` = -2.219 puts the mean :math:`\log L_X` of
            massive centrals near 40.6.
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

            ``"catalogue"`` selects on the luminosity a **catalogue assigns**:
            the observed flux in ``catalogue_band`` converted to rest-frame
            2--10 keV with an *unabsorbed* power law of index
            ``catalogue_gamma``.  An absorbed source is assigned less than its
            intrinsic luminosity, so the selected sample is a mixture of
            unobscured, Compton-thin and Compton-thick AGN, each crossing the
            threshold at its own intrinsic luminosity
            (:meth:`catalogue_selection`).  Needs ``obscuration="split"``.
        catalogue_band : (float, float)
            The observed band of the catalogue's flux, keV; eROSITA's
            0.2--2.3 keV by default.  Read at ``selection="catalogue"`` only.
        catalogue_gamma : float
            The photon index the catalogue converted its fluxes with: a
            property of the catalogue, not of the AGN, so a plain number and
            not ``gamma_x``.  :data:`GAMMA_X` by default.
        catalogue_flux_limit : bool
            Whether the catalogue is also cut at ``log10_flux_min`` in
            ``catalogue_band``.  The flux cut is then a threshold on the same
            catalogue luminosity -- a flux and the luminosity the catalogue
            assigns it are one number at fixed redshift -- and the larger of
            the two thresholds applies.  ``False`` by default: the threshold
            is ``log10lx_min`` alone.
        catalogue_nh : {"comparat19", "single"}
            The columns of the three branches.  ``"comparat19"``, the default,
            is Comparat et al. (2019)'s: :math:`\log N_{\rm H}` uniform in
            [20, 22), [22, 24) and [24, 26) for the unobscured, Compton-thin
            and Compton-thick AGN (:data:`CATALOGUE_NH_CLASSES`), and the
            selection is averaged over each class by Gauss-Legendre in
            :math:`\log N_{\rm H}`; ``log10_nh`` is **not read**.
            ``"single"`` puts each branch at one column -- zero, ``log10_nh``
            and :data:`LOG10_NH_CT` -- and is kept for reproducibility: with
            the thin column free, a fit can push every obscured AGN out of a
            soft-band catalogue (the first YR3 MAPs did, at
            :math:`10^{23.7}`-:math:`10^{24.2}`), an edge the uniform classes
            do not admit, and its "unobscured" AGN carry no column at all
            where Comparat et al.'s carry up to :math:`10^{22}`.
        catalogue_nh_nodes : int
            Gauss-Legendre nodes per class at ``"comparat19"``; static, it sets
            an array shape.  At the defaults, whose obscuration changeover
            is 0.05 dex wide, 16 matches 64 to :math:`3\times10^{-6}` in the
            selected satellite counts and to :math:`5\times10^{-7}` in the
            central ones; 8, enough at the 1.4.0 width of 0.691 dex, is
            :math:`2\times10^{-3}` off for the satellites
            (``tests/test_agn.py``).
        catalogue_nh_classes : dict, optional
            ``{branch: (lo, hi)}`` in :math:`\log N_{\rm H}`, replacing
            :data:`CATALOGUE_NH_CLASSES`: what a test or a different
            column model needs.
        catalogue_scatter : float, optional
            The unabsorbed scattered fraction of the Compton-thin and
            Compton-thick classes' intrinsic power law in the catalogue band
            (and in :meth:`emission_weights`' band factors).  ``None`` is
            :data:`F_SCATTER` (0.02) at ``catalogue_nh="comparat19"`` -- the
            X-ray spectrum :meth:`xlf_observed_hard` forward-models Aird et
            al. (2015) with, so the catalogue and the hard-band XLF see one
            spectrum, and a Compton-thick AGN enters the soft-band catalogue
            about 1.7 dex below its intrinsic luminosity instead of never --
            and zero at ``"single"``, which reproduces the first YR3 runs;
            ``0.0`` turns it off.
        kbol : {"constant", "duras20"}
            The bolometric correction :math:`K_X = L_{\rm bol}/L_X` to the
            2--10 keV band (:data:`KBOL_MODES`, :meth:`_shift`).
            ``"duras20"``, the default, is the general
            :math:`K_X(L_{\rm bol})` of Duras et al. (2020),
            :func:`log10_kx_duras20`, with the Jacobian of the change of
            variable; ``k_bol`` is not read.  The defaults of
            :class:`AgnParams` and :class:`ObscurationParams` are fitted with
            it.
            ``"constant"`` is ``k_bol`` at every luminosity; with the 1.4.0
            parameters it reproduces 1.4.0 bit for bit.
            Static: :math:`\log L_{\rm bol}(L_X)` on the luminosity grid and
            the Jacobian are computed once, here.
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
        if self.selection == "catalogue" and self.obscuration != "split":
            raise ValueError(
                "selection='catalogue' selects on the luminosity a catalogue "
                "assigns, which is the intrinsic one for an unobscured source "
                "and less for an absorbed one, so it needs the obscured "
                "fraction to know which sources are which.  Build it with "
                "AgnSector(..., obscuration='split', selection='catalogue').")
        e1, e2 = float(catalogue_band[0]), float(catalogue_band[1])
        if not 0.0 < e1 < e2:
            raise ValueError(f"a catalogue band needs 0 < emin < emax, got "
                             f"({e1}, {e2})")
        self.catalogue_band = (e1, e2)
        self.catalogue_gamma = float(catalogue_gamma)
        self.catalogue_flux_limit = bool(catalogue_flux_limit)
        if str(catalogue_nh) not in CATALOGUE_NH_MODES:
            raise ValueError(f"unknown catalogue_nh {catalogue_nh!r}; expected "
                             f"one of {CATALOGUE_NH_MODES}")
        self.catalogue_nh = str(catalogue_nh)
        self.catalogue_nh_nodes = int(catalogue_nh_nodes)
        classes = dict(CATALOGUE_NH_CLASSES)
        classes.update(catalogue_nh_classes or {})
        if set(classes) != set(CATALOGUE_BRANCHES):
            raise ValueError(f"catalogue_nh_classes must name {CATALOGUE_BRANCHES}")
        self.catalogue_nh_classes = {b: (float(classes[b][0]), float(classes[b][1]))
                                     for b in CATALOGUE_BRANCHES}
        if catalogue_scatter is None:
            catalogue_scatter = F_SCATTER if self.catalogue_nh == "comparat19" else 0.0
        if not 0.0 <= float(catalogue_scatter) < 1.0:
            raise ValueError(f"catalogue_scatter is a fraction in [0, 1), got "
                             f"{catalogue_scatter}")
        self.catalogue_scatter = float(catalogue_scatter)
        if str(kbol) not in KBOL_MODES:
            raise ValueError(f"unknown kbol {kbol!r}; expected one of "
                             f"{KBOL_MODES}")
        self.kbol = str(kbol)
        if self.kbol == "duras20":
            # Constants of the sector: Duras et al.'s coefficients are not
            # parameters, so the inverse is taken once, on the static grid.
            self._log10_lbol = log10_lbol_of_lx_duras20(self.loglx)
            self._dlog10_lbol_dlog10_lx = 1.0 / (
                1.0 - dlog10_kx_duras20(self._log10_lbol))

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

        The scatter about it is the galaxy sector's too (:meth:`sigma_mstar`).

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

    def sigma_mstar(self, log10m, galaxy_params, *, h):
        r""":math:`\sigma_{M_*}(M_h)`, the scatter of a central's
        :math:`\log_{10}M_*` at fixed halo mass [dex], for halo masses in
        :math:`M_\odot/h`: the galaxy sector's
        (:meth:`~ggah_mod.sectors.galaxies.GalaxySector.log10_mstar_scatter`),
        as the mean relation is (:meth:`log10_mstar`), so the AGN and the
        galaxies of one spectrum share both.  It sets
        :meth:`central_fraction`, :meth:`_sigma_lm` and :meth:`_sigma_lx`.
        """
        return self.galaxies.log10_mstar_scatter(log10m, galaxy_params, h=h)

    def _sigma_lm(self, p: AgnParams, sigma_mstar):
        r"""Width of :math:`\log M_{\rm BH}` at fixed halo mass, with
        ``sigma_mstar`` the central's stellar-mass scatter there
        (:meth:`sigma_mstar`, broadcast).

        :math:`\rho > 0` means part of the :math:`M_*` scatter aligns with halo
        mass, which *shrinks* this width -- the whole content of Powell's
        "Model 2".
        """
        var = p.al_bh ** 2 * sigma_mstar ** 2 * (1.0 - p.rho) + p.sig_bh ** 2
        return jnp.sqrt(jnp.maximum(var, 1e-6))

    def _sigma_lx(self, p: AgnParams, sigma_mstar):
        r"""Width of the central's luminosity kernel at fixed halo mass [dex].

        .. math::

            \sigma_{\rm lx}^2 = (\alpha_{\rm BH} + s)^2\,\sigma_{M_*}^2\,
                (1 - \rho) + \sigma_{\rm BH}^2,

        with :math:`s` = ``log10_lstar_mslope``: the central's stellar-mass
        scatter at fixed halo mass moves its black-hole mass by
        :math:`\alpha_{\rm BH}` and its Eddington-ratio break by :math:`s` per
        dex (:func:`lstar_of_mstar`), so the luminosity carries both.
        :meth:`_sigma_lm`, the width of :math:`\log M_{\rm BH}` itself, is the
        :math:`s = 0` case and stays the width of every quantity built on the
        black-hole mass; at a slope of zero the two are equal bit for bit.
        """
        a = p.al_bh + _lstar_tilt(None, p)
        var = a ** 2 * sigma_mstar ** 2 * (1.0 - p.rho) + p.sig_bh ** 2
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

    def _kernel(self, p: AgnParams, sig, *, z=0.0):
        r""":math:`K = {\rm ERDF} \ast \mathcal{N}(0,\sigma)`, on the
        :math:`\log\lambda` grid.  ``sig`` is a scalar (the satellites'
        width, :meth:`_sigma_bh_at_fixed_mstar`), giving ``k`` of shape
        ``(N_t,)``, or one width per halo mass (the centrals'
        :meth:`_sigma_lx`, which carries the galaxy sector's
        :math:`\sigma_{M_*}(M_h)`), giving ``(N_M, N_t)``.  The ERDF's break is
        the one at the pivot stellar mass,
        :math:`\lambda_*(10^{10.5}\,M_\odot, z)`; :meth:`_shift` moves it to
        each stellar mass.

        The chain is shift-invariant, which is the observation that turns an
        :math:`(N_M,N_L,N_\lambda)` cube into convolutions of one ERDF: one
        matrix product of its shifted copies with the Gaussian of every width
        (:func:`_convolve_full`).

        The Gaussian is evaluated on a **fixed-width** grid rather than a
        :math:`\pm6\sigma` one.  A width that tracked :math:`\sigma` would make
        an array shape depend on a traced value, which does not jit -- and the
        fixed grid spans the whole prior range of :math:`\sigma_{\rm bh}` with
        room to spare.
        """
        e = self._erdf_pivot(p, z)
        _, g = self._gauss_nodes(sig)
        k = _convolve_full(e * self.dlam, g) / self.dlam
        half = self.loglam.size // 2
        t0 = self.loglam[0] - half * self.dlam
        t = t0 + jnp.arange(k.shape[-1]) * self.dlam
        return t, k

    def _erdf_pivot(self, p: AgnParams, z=0.0):
        r"""The Eddington-ratio distribution at the pivot stellar mass on the
        :math:`\log\lambda` grid, per dex, normalised so that its sum times the
        grid spacing is one: the first factor of :meth:`_kernel`."""
        e = erdf(self.loglam,
                 lstar_of_z(z, p.log10_lstar, p.gam_lam,
                            p.gam_lam_hi, p.z_lam),
                 p.delta1, p.delta2)
        return e / (jnp.sum(e) * self.dlam)

    def _gauss_nodes(self, sig):
        r"""``(x, g)``: the offsets :math:`x_k = k\,\Delta\log\lambda`,
        :math:`|k| \le N_\lambda/2`, and the Gaussian of width ``sig`` [dex] at
        them, normalised to unit sum along the last axis: the second factor of
        :meth:`_kernel`.  ``sig`` is a scalar or one width per row."""
        half = self.loglam.size // 2
        x = (jnp.arange(-half, half + 1) * self.dlam)
        sig = jnp.asarray(sig)
        g = jnp.exp(-0.5 * (x / sig[..., None]) ** 2)
        return x, g / jnp.sum(g, axis=-1, keepdims=True)

    def p_loglx_given_m(self, log10m, p: AgnParams, galaxy_params, *, h,
                        z=0.0):
        r""":math:`dP(\log L_X|M_h)/d\log L_X` of the **central**, shape ``(NM, N_lx)``,
        times :meth:`central_fraction`: a central below the stellar-mass cut
        hosts no AGN.  The kernel is placed at :meth:`mean_log10_mbh` and at the
        Eddington-ratio break of the mean stellar mass :math:`M_*^{\rm c}(M_h)`
        that relation reads (:meth:`_shift`)."""
        sig_ms = self.sigma_mstar(log10m, galaxy_params, h=h)
        t_grid, k = self._kernel(p, self._sigma_lx(p, sig_ms), z=z)
        # mean_log10_mbh, with the stellar mass it reads kept for the break:
        # the inversion behind log10_mstar is the costly step of this method.
        ms = self.log10_mstar(log10m, galaxy_params, h=h, z=z)
        mean_bh = self._mbh(ms, p, z)
        frac = self.central_fraction(log10m, p, galaxy_params, h=h, z=z)
        return self._shift(t_grid, k, p, mean_bh, ms) * frac[:, None]

    #: :func:`log10_duty` and :func:`duty`, the active fraction at a stellar
    #: mass, and :func:`lstar_of_mstar`, the Eddington-ratio break at one, as
    #: methods: the same functions, not copies.
    log10_duty = staticmethod(log10_duty)
    duty = staticmethod(duty)
    lstar_of_mstar = staticmethod(lstar_of_mstar)

    def central_duty(self, log10m, p: AgnParams, galaxy_params, *, h, z=0.0):
        r"""The central's active fraction, :func:`duty` at the mean stellar mass
        :math:`\log M_*^{\rm c}(M_h)` of :meth:`log10_mstar` -- the stellar
        mass :meth:`mean_log10_mbh` reads -- shape ``(NM,)``.

        At the mean, not averaged over :meth:`sigma_mstar`: the central's stellar-mass
        scatter is folded into the width of the luminosity kernel
        (:meth:`_sigma_lx`), not integrated over, so there is no distribution
        of :math:`M_*` at fixed :math:`M_h` here to average the fraction on.
        """
        return duty(self.log10_mstar(log10m, galaxy_params, h=h, z=z), p)

    def p_active_loglx_given_m(self, log10m, p: AgnParams, galaxy_params, *,
                               h, z=0.0):
        r""":math:`f_{\rm ERDF}(M_*^{\rm c})\,dP(\log L_X|M_h)/d\log L_X`: the
        **active** central's luminosity distribution, :meth:`p_loglx_given_m`
        times :meth:`central_duty`, shape ``(NM, N_lx)``.

        What every central term of the sector integrates -- the occupation,
        the XLF in all its forms, the emission weights and
        :meth:`validity_cost` -- so the duty cycle is applied in one place.
        :meth:`p_loglx_given_m` stays the distribution of a black hole that is
        shining, integrating to :meth:`central_fraction`.
        """
        return (self.p_loglx_given_m(log10m, p, galaxy_params, h=h, z=z)
                * self.central_duty(log10m, p, galaxy_params, h=h, z=z)[:, None])

    def p_loglx_given_mstar(self, log10_mstar, p: AgnParams, *, z=0.0):
        r""":math:`dP(\log L_X|M_*)/d\log L_X` of a shining black hole in a
        galaxy of stellar mass ``log10_mstar`` (physical
        :math:`\log_{10}M_\odot`, an array of shape ``(N_*,)``), shape
        ``(N_*, N_lx)`` on :attr:`loglx`, per dex.

        The kernel of width ``sig_bh`` (:meth:`_sigma_bh_at_fixed_mstar`)
        placed at :math:`\langle\log M_{\rm BH}\rangle(M_*)` and at the
        Eddington-ratio break of :math:`M_*` (:meth:`_shift`): at fixed stellar
        mass the scatter of :math:`M_*` about the stellar-mass relation does not
        enter.  The satellites' luminosity distribution
        (:meth:`_satellite_kernel`); a central of the same stellar mass has the
        same one.
        """
        ms = jnp.atleast_1d(jnp.asarray(log10_mstar, dtype=float))
        t_grid, k = self._kernel(p, self._sigma_bh_at_fixed_mstar(p), z=z)
        mean_bh = self._mbh(ms, p, z)
        return self._shift(t_grid, k, p, mean_bh, ms)

    def p_active_loglx_given_mstar(self, log10_mstar, p: AgnParams, *, z=0.0):
        r""":math:`f_{\rm ERDF}(M_*)\,dP(\log L_X|M_*)/d\log L_X`, shape
        ``(N_*, N_lx)``: :meth:`p_loglx_given_mstar` times the active fraction
        :func:`duty` at ``log10_mstar`` (physical :math:`\log_{10}M_\odot`),
        and zero below the stellar-mass cut ``lg_mstar_min``.
        Its integral over :math:`\log L_X` is the fraction of the galaxies'
        black holes that are shining (on the luminosity grid)."""
        ms = jnp.atleast_1d(jnp.asarray(log10_mstar, dtype=float))
        return (self.p_loglx_given_mstar(ms, p, z=z)
                * self._duty_above_cut(ms, p)[:, None])

    def _duty_above_cut(self, log10_mstar, p: AgnParams):
        r""":func:`duty` at ``log10_mstar``, and zero below ``lg_mstar_min``."""
        return jnp.where(log10_mstar >= self.lg_mstar_min,
                         duty(log10_mstar, p), 0.0)

    def central_fraction(self, log10m, p: AgnParams, galaxy_params, *, h,
                         z=0.0):
        r"""The fraction of a halo's centrals above the stellar-mass cut.

        .. math::

            f_{\rm c}^{\rm cut}(M_h) = \tfrac12\,{\rm erfc}\!\left[
                \frac{\lg M_{*,\min} - \lg M_*^{\rm c}(M_h)}
                     {\sqrt2\,\sigma_{M_*}}\right]

        with :math:`\sigma_{M_*}(M_h)` of :meth:`sigma_mstar`, the central's stellar-mass
        scatter at fixed halo mass that :meth:`_sigma_lm` and
        :meth:`_sigma_lx` already carry.  It
        multiplies the central term everywhere; the black-hole distribution of
        the centrals that remain is not re-conditioned on the cut.
        """
        ms = self.log10_mstar(log10m, galaxy_params, h=h, z=z)
        sig = jnp.maximum(self.sigma_mstar(log10m, galaxy_params, h=h), 1e-6)
        return 0.5 * erfc((self.lg_mstar_min - ms) / (jnp.sqrt(2.0) * sig))

    def _satellite_csmf(self, field, galaxy_params):
        r"""``(lg_ms, dn)``: the satellites' conditional stellar-mass function
        on this sector's physical grid, from the cut to
        :data:`LG_MSTAR_MAX_SAT`.  Every satellite term reads it, so none
        counts a satellite below the cut."""
        return self.galaxies.satellite_csmf_msun(field, galaxy_params,
                                                 lg_ms_msun=self.lg_ms_sat)

    def _shift(self, t_grid, k, p: AgnParams, mean_log10_mbh, log10_mstar):
        r"""The luminosity kernel placed at each mean black-hole mass and at the
        Eddington-ratio break of each stellar mass, on the :math:`\log L_X`
        grid: shape ``(len(mean_log10_mbh), N_lx)``.

        .. math::

            P(\log L_X) = K\big(\log L_{\rm bol}(L_X) - \log(1.26\times10^{38})
                - \langle\log M_{\rm BH}\rangle - s\,(\log M_* - 10.5)\big)\,
                \frac{d\log L_{\rm bol}}{d\log L_X},

        with :math:`K` of :meth:`_kernel`, ``log10_mstar`` (physical
        :math:`\log_{10}M_\odot`, broadcast against ``mean_log10_mbh``) the
        stellar mass the black-hole mass was read at, and :math:`s` =
        ``log10_lstar_mslope`` (:func:`lstar_of_mstar`).
        At ``kbol="constant"``, :math:`\log L_{\rm bol} = \log L_X +
        \log k_{\rm bol}` and the Jacobian is one, so it is not formed; at
        ``"duras20"``, :math:`L_{\rm bol}(L_X)` is
        :func:`log10_lbol_of_lx_duras20` and the Jacobian
        :math:`1/(1 - d\log K_X/d\log L_{\rm bol})`, between 1 and 2.09, both
        on the sector's grid.  Every luminosity distribution of the sector is
        placed here, the joint one with :math:`M_{\rm BH}` of
        :meth:`joint_mbh_lx` included (one row per black-hole mass, with the
        black-hole mass in ``mean_log10_mbh``).
        """
        # y = log L_bol - log 1.26e38 on the L_X grid.
        if self.kbol == "duras20":
            y = self._log10_lbol[None, :] - _LOG10_LBOL_COEF
            jac = self._dlog10_lbol_dlog10_lx[None, :]
        else:
            y = self.loglx[None, :] - jnp.log10(LBOL_COEF / p.k_bol)
            jac = None
        t = (y
             - jnp.atleast_1d(mean_log10_mbh)[:, None]
             - jnp.atleast_1d(_lstar_tilt(log10_mstar, p))[:, None])
        if k.ndim == 1:
            out = jnp.interp(t.ravel(), t_grid, k, left=0.0,
                             right=0.0).reshape(t.shape)
        else:
            out = jax.vmap(lambda tt, kk: jnp.interp(tt, t_grid, kk, left=0.0,
                                                     right=0.0))(t, k)
        return out if jac is None else out * jac

    def log10_kbol(self, log10_lbol, p: AgnParams):
        r""":math:`\log_{10}K_X`, the bolometric correction
        :math:`L_{\rm bol}/L_X` this sector applies at ``log10_lbol`` [erg/s]:
        :math:`\log_{10}` ``k_bol`` at ``kbol="constant"`` (broadcast to
        ``log10_lbol``), :func:`log10_kx_duras20` at ``"duras20"``."""
        if self.kbol == "duras20":
            return log10_kx_duras20(log10_lbol)
        return jnp.zeros_like(jnp.asarray(log10_lbol, dtype=float)) + jnp.log10(p.k_bol)

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

        At ``selection="catalogue"`` it is :meth:`catalogue_selection` summed
        over the three branches at the default obscuration coefficients, on
        the field's redshift (zero without a field).  :meth:`_moment` does not
        come through here in that mode: it calls :meth:`catalogue_selection`
        with the coefficients and the halo masses it was given.
        """
        if self.selection == "none":
            return jnp.ones_like(self.loglx)
        if self.selection == "catalogue":
            return self.catalogue_selection(p, field=field)
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

    # -- the catalogue's luminosity -------------------------------------------
    def _log10_k_catalogue(self, z):
        r""":math:`\log_{10}K_{\rm cat}(z)`, the catalogue's own conversion:
        an unabsorbed power law of ``catalogue_gamma`` from ``catalogue_band``
        observed to 2--10 keV rest.  The unobscured branch's
        :math:`\log_{10}K` in :meth:`catalogue_shift` is the same expression in
        ``gamma_x``, term for term, so the two cancel **exactly** -- not to
        round-off -- when the indices agree."""
        e1, e2 = self.catalogue_band
        g = self.catalogue_gamma
        return ((2.0 - g) * jnp.log10(1.0 + jnp.asarray(z))
                + jnp.log10(band_energy(e1, e2, g))
                - jnp.log10(band_energy(2.0, 10.0, g)))

    def catalogue_columns(self, p: AgnParams) -> dict:
        r"""``{branch: (log10_nh nodes, weights)}``: the columns each branch is
        averaged over, weights summing to one.

        At ``catalogue_nh="comparat19"`` the Gauss-Legendre nodes of
        ``catalogue_nh_nodes`` points over the branch's class interval, a
        uniform distribution in :math:`\log N_{\rm H}`; at ``"single"`` one
        node per branch -- ``None`` for the unobscured branch's zero column,
        ``log10_nh`` for the thin one, :data:`LOG10_NH_CT` for the thick one.
        """
        if self.catalogue_nh == "single":
            one = jnp.ones(1)
            return {"unobscured": (None, one),
                    "thin": (jnp.atleast_1d(jnp.asarray(p.log10_nh, dtype=float)), one),
                    "thick": (jnp.atleast_1d(jnp.asarray(LOG10_NH_CT)), one)}
        return self._class_nodes()

    def _class_nodes(self) -> dict:
        """``{branch: (log10_nh nodes, weights)}`` over ``catalogue_nh_classes``,
        Gauss-Legendre, uniform in log N_H, whatever ``catalogue_nh`` is."""
        from ..halos.profiles import _leggauss_cached
        x, w = _leggauss_cached(self.catalogue_nh_nodes)
        out = {}
        for b in CATALOGUE_BRANCHES:
            lo, hi = self.catalogue_nh_classes[b]
            out[b] = (jnp.asarray(0.5 * (lo + hi) + 0.5 * (hi - lo) * x),
                      jnp.asarray(0.5 * w))
        return out

    def _shift_nodes(self, p: AgnParams, z) -> dict:
        r"""``{branch: (Delta at each column node, weights)}``, the nodes of
        :meth:`catalogue_columns`."""
        e1, e2 = self.catalogue_band
        g = p.gamma_x
        log_hard = jnp.log10(band_energy(2.0, 10.0, g))
        log_cat = self._log10_k_catalogue(z)
        # Zero column, closed form, so Delta = 0 exactly at gamma_x =
        # catalogue_gamma; also the scattered component's spectrum.
        log_unabs = ((2.0 - g) * jnp.log10(1.0 + jnp.asarray(z))
                     + jnp.log10(band_energy(e1, e2, g)))
        f_sc = self.catalogue_scatter
        out = {}
        for b, (nodes, w) in self.catalogue_columns(p).items():
            if nodes is None:
                d = jnp.atleast_1d(log_unabs - log_hard - log_cat)
            else:
                la = jnp.reshape(log10_absorbed_band_energy(
                    e1, e2, g, nodes[:, None], z), (-1,))
                if b != "unobscured" and f_sc > 0.0:
                    la = jnp.logaddexp(_LN10 * la,
                                       _LN10 * (log_unabs + math.log10(f_sc))) / _LN10
                d = la - log_hard - log_cat
            out[b] = (d, w)
        return out

    def catalogue_shift(self, p: AgnParams, z) -> dict:
        r""":math:`\Delta_b(z) = \log L_{\rm cat} - \log L_X` per branch: how far
        below its intrinsic 2--10 keV luminosity the catalogue puts a source.

        .. math::

            \Delta_b = \log_{10}\frac{K(z;\Gamma_x, N_{{\rm H},b})}
                                     {K(z;\Gamma_{\rm cat}, 0)},\qquad
            K(z;\Gamma, N_{\rm H}) = \frac{\int_{E_1(1+z)}^{E_2(1+z)}
                E^{1-\Gamma}e^{-\tau}\,dE}{\int_2^{10}E^{1-\Gamma}\,dE},

        with :math:`[E_1, E_2]` the catalogue band.  The numerator is the
        energy the source sends into the observed band per unit intrinsic hard
        luminosity, so :math:`4\pi d_L^2 F = K L_X`; the denominator is what
        the catalogue *assumed* it was when it divided the same flux back out.
        The K-correction is in both through the limits, as in
        :func:`absorbed_band_energy`.

        Returned as ``{"unobscured", "thin", "thick"}``.  At
        ``catalogue_nh="single"`` one number each: the unobscured branch at
        **zero** column, written in closed form as :meth:`band_factor` writes
        it (so :math:`\Delta = 0` exactly at :math:`\Gamma_x =
        \Gamma_{\rm cat}`), the thin one at ``log10_nh`` and the thick one at
        :data:`LOG10_NH_CT`, both through :func:`log10_absorbed_band_energy`;
        at the defaults, in the 0.2--2.3 keV band, :math:`-1.03` thin and
        :math:`-28` thick at :math:`z = 0.2`, and :math:`-0.79` and :math:`-16`
        at :math:`z = 0.5`.  At ``"comparat19"`` an array per branch, the shift
        at each Gauss-Legendre column node of its class
        (:meth:`catalogue_columns`); the selection averages the sigmoid over
        them, not the shift.  Not rising with the column either way, and
        negative for the absorbed classes; at zero column the unobscured shift
        is positive, :math:`+0.03` to :math:`+0.02` for
        :math:`0 \le z \le 0.5`, because the default
        :math:`\Gamma_x = 1.837` is softer than the catalogue's 1.8.  With
        ``catalogue_scatter`` the absorbed classes level off at the
        scattered light's :math:`\log_{10}f_{\rm sc}` below the unabsorbed
        shift, 1.7 dex at the default 0.02.
        """
        nodes = self._shift_nodes(p, z)
        if self.catalogue_nh == "single":
            return {b: d[0] for b, (d, _) in nodes.items()}
        return {b: d for b, (d, _) in nodes.items()}

    def catalogue_flux_luminosity(self, field, p: AgnParams):
        r"""The catalogue luminosity of a source at the flux limit
        ``log10_flux_min`` in ``catalogue_band``:
        :math:`\log(4\pi d_L^2F_{\min}) - \log K_{\rm cat}(z)`.

        Unlike :meth:`flux_limit_luminosity` it reads no fitted spectrum: it is
        the catalogue's own conversion, so the flux limit and the luminosity
        threshold are cuts on one variable and combine by taking the larger.
        """
        from ..cosmology.background import luminosity_distance
        d_l = jnp.atleast_1d(
            luminosity_distance(field.z, field.cosmo))[0] / field.cosmo.h
        d_cm = d_l * C.MPC_CM
        return (jnp.log10(4.0 * jnp.pi * d_cm ** 2) + p.log10_flux_min
                - self._log10_k_catalogue(field.z))

    def catalogue_threshold(self, p: AgnParams, field=None):
        r"""The threshold on the catalogue luminosity: ``log10lx_min``, or the
        larger of it and :meth:`catalogue_flux_luminosity` when the sector was
        built with ``catalogue_flux_limit=True``.

        The larger and not a smooth blend of the two, because the data apply
        both cuts and the sample is their intersection; ``ggah_cal``'s
        flux-binned samples combine the two the same way.  The tie is where
        the flux limit crosses the luminosity threshold in redshift, and the
        gradient there goes to one of them, which is what the data do too.
        """
        if not self.catalogue_flux_limit:
            return p.log10lx_min
        if field is None:
            raise ValueError(
                "a flux-limited catalogue's threshold depends on redshift, "
                "and no field was given to read one from")
        return jnp.maximum(p.log10lx_min,
                           self.catalogue_flux_luminosity(field, p))

    def catalogue_selection(self, p: AgnParams, *, field=None, z=None,
                            obsc: "ObscurationParams" = None, log10m=None,
                            branch=None, kweight=None):
        r"""The selection of a soft-band catalogue, on the **intrinsic**
        :math:`\log L_X` grid, with the branch fractions inside it.

        .. math::

            S(L_X) = \sum_{b} f_b(L_X, z)\;
                \tfrac12\left[1 + {\rm erf}\,\frac{\log L_X + \Delta_b(z)
                               - \log L_{\min}}{w_{\rm sel}}\right]

        over :data:`CATALOGUE_BRANCHES`, with :math:`f_b` from
        :func:`obscuration_branches` (``obsc``, tilted by ``log10m`` when it is
        given), :math:`\Delta_b` from :meth:`catalogue_shift`,
        :math:`L_{\min}` from :meth:`catalogue_threshold` -- a threshold on
        the *catalogue's* luminosity -- and :math:`w_{\rm sel}` =
        ``lx_sel_width``.

        **Why the selection and the split are one factor here.**  The
        catalogue cut is on :math:`\log L_X + \Delta_b`, and :math:`\Delta_b`
        differs between branches by a dex or more in a soft band, so an
        obscured AGN enters a catalogue bin at an intrinsic luminosity a dex
        above an unobscured one.  The selected count is then a sum of three
        integrals, each with its own threshold, and the fractions cannot be
        applied after the sum.  Like every factor in :meth:`_moment` it sits
        inside the :math:`L_X` integral.

        ``branch=None`` is the observed, mixed sample; ``"unobscured"`` keeps
        the first term and ``"obscured"`` the other two, so the two add to the
        mixed one exactly.  ``z`` defaults to the field's redshift, or zero
        without a field.  ``kweight`` is a further factor -- a dict keyed like
        :data:`CATALOGUE_BRANCHES` with one value per column node of
        :meth:`catalogue_columns`, or one array on the luminosity grid for all
        three -- which is how :meth:`emission_weights` carries each column's
        own band factor: the same branch fraction must not be applied twice,
        and a column's band factor goes with that column's selection.

        At ``catalogue_nh="comparat19"`` each branch's sigmoid is the average
        over its class's columns, :math:`\log N_{\rm H}` uniform in
        :data:`CATALOGUE_NH_CLASSES`, by Gauss-Legendre; at ``"single"`` it is
        one sigmoid at one column.

        **Reduces to the shipped unobscured selection** where it should: at
        ``catalogue_nh="single"``, :math:`f_{\rm obs} = 0` and
        :math:`\Gamma_x = \Gamma_{\rm cat}` it is
        ``selection="luminosity"``'s sigmoid, bit for bit; at
        ``"comparat19"`` with every class moved to a negligible column, to
        :math:`10^{-10}`.
        """
        zz = field.z if field is not None else (0.0 if z is None else z)
        shift = self._shift_nodes(p, zz)
        frac = obscuration_branches(self.loglx, zz, obsc, log10m=log10m)
        lmin = self.catalogue_threshold(p, field)
        keep = {None: CATALOGUE_BRANCHES, "unobscured": ("unobscured",),
                "obscured": ("thin", "thick")}[self._branch(branch)]
        out = 0.0
        for b in keep:
            d, wq = shift[b]
            if isinstance(kweight, dict):
                # Per column node: a band factor depends on the column, so it
                # is averaged with the selection, not after it.
                wq = wq * kweight[b]
            sel = jnp.sum(wq[:, None] * (0.5 * (1.0 + erf(
                (self.loglx[None, :] + d[:, None] - lmin) / p.lx_sel_width))),
                axis=0)
            term = frac[b] * sel
            if kweight is not None and not isinstance(kweight, dict):
                term = term * kweight
            out = out + term
        return out

    def _band_factor_by_column(self, p: AgnParams, band, *, frame="observer",
                               z=0.0, power: int = 1) -> dict:
        r""":math:`K^n` at each column node of each catalogue branch
        (:meth:`catalogue_columns`): the energy a source of that column sends
        into ``band`` per unit hard luminosity, **without** its fraction.

        :meth:`band_factor` averages two columns with the obscured fraction
        folded in; here the fraction is already inside
        :meth:`catalogue_selection`, and every column has its own factor,
        averaged there together with that column's selection.
        """
        e1, e2 = float(band[0]), float(band[1])
        if not 0.0 < e1 < e2:
            raise ValueError(f"a band needs 0 < emin < emax, got ({e1}, {e2})")
        zb = jnp.asarray(z) if frame == "observer" else jnp.asarray(0.0)
        hard = band_energy(2.0, 10.0, p.gamma_x)
        unabs = (jnp.power(1.0 + zb, 2.0 - p.gamma_x)
                 * band_energy(e1, e2, p.gamma_x) / hard)
        out = {}
        for b, (nodes, _) in self.catalogue_columns(p).items():
            if nodes is None:
                k = jnp.atleast_1d(unabs)
            else:
                k = jnp.reshape(absorbed_band_energy(e1, e2, p.gamma_x,
                                                     nodes[:, None], zb) / hard, (-1,))
                if b != "unobscured":
                    k = k + self.catalogue_scatter * unabs
            out[b] = k ** power
        return out

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

        At ``selection="catalogue"`` the branch fractions are part of the
        selection itself (:meth:`catalogue_selection`, at the field's
        redshift), so they are applied there and not a second time here, and
        ``lweight`` may be a dict per branch.  Every other mode takes the
        arithmetic below unchanged.
        """
        if self.selection == "catalogue":
            w = prob * self.catalogue_selection(
                p, field=field, z=z, obsc=obsc, log10m=log10m, branch=branch,
                kweight=lweight)
            if power:
                w = w * jnp.power(10.0, power * self.loglx)
            return jnp.sum(w, axis=-1) * self.dlx
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
        obscured source passes 4.1% of its 0.5--2 keV energy at
        :math:`z = 0` and 11% in the observed band at :math:`z = 0.27`.

        ``z`` is the source redshift, which the obscured fraction reads in both
        frames.  ``log10m`` tilts the split with halo mass, as in
        :meth:`occupation`.

        Needs ``obscuration="split"``: in a soft band the split *is* the
        spectrum, and a sector built without it would count every AGN as
        unabsorbed -- at the defaults 0.99 of them are obscured at
        :math:`10^{42}` erg/s and :math:`z = 0.2`.

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
        :math:`f_{\rm ERDF}(M_*)\,dP(\log L_X|M_*)/d\log L_X`, shape
        ``(N_*, N_lx)``, the active fraction :func:`duty` at each satellite's
        own stellar mass:
        the central's black-hole relation and Eddington-ratio distribution at a
        given stellar mass, the break :func:`lstar_of_mstar` at the
        satellite's own stellar mass, with the width at fixed :math:`M_*`
        (:meth:`_sigma_bh_at_fixed_mstar`), :meth:`p_active_loglx_given_mstar`.
        Centrals and satellites of equal stellar mass have the same black hole.
        """
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        return lg_ms, dn, self.p_active_loglx_given_mstar(lg_ms, p, z=field.z)

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

        At ``selection="catalogue"`` the moment is the catalogue's mixed
        sample (``branch=None``) or one side of it, with the untilted branch
        fractions inside the selection for the same reason, and ``obsc``
        reaches them.
        """
        lg_ms, dn, prob = self._satellite_kernel(field, p, galaxy_params)
        per_mstar = self._moment(prob, p, power, branch=self._branch(branch),
                                 z=field.z, obsc=obsc, field=field,
                                 lweight=lweight)  # (N_*,)
        return self._satellite_duty(p) * jnp.trapezoid(
            dn * per_mstar[:, None], lg_ms, axis=0)

    # -- predictions ---------------------------------------------------------
    def xlf(self, field, p: AgnParams, galaxy_params, band: str = "hard", *,
            observed: bool = False, obsc: "ObscurationParams" = None,
            frame: str = "rest"):
        r"""The **predicted** :math:`\Phi(L_X)` [(Mpc/h)^-3 dex^-1].

        Returns ``(log10_lx_grid, phi)``.  Comparing this to a published XLF is
        a genuine test; an abundance-matched model reproduces one by
        construction and so cannot be tested the same way.

        Satellites are included when :meth:`has_satellites`.  Their halo mass
        is integrated out first, into the satellite stellar-mass function, so
        the :math:`(N_*, N_M, N_{\rm lx})` cube is never formed.

        ``band="soft"`` is the 0.5--2 keV function.  At
        ``obscuration="split"`` it is the sum over the three classes of
        Comparat et al. (2019), each class's intrinsic function
        (:meth:`_branch_xlfs`) moved by its own absorbed band ratio at each of
        its column nodes (:meth:`band_shift`, in the ``frame`` given, rest by
        default):

        .. math::

            \Phi_{0.5-2}(\log L) = \sum_b\sum_i w_i\,
                \Phi_b(\log L - \Delta_{b,i}).

        At ``obscuration="none"`` the sector asserts no split, every AGN is
        unabsorbed, and the function is the hard one on a grid moved by the
        unabsorbed ratio, ``log10(k_h2s)``.

        **This is the intrinsic XLF, of every AGN, and stays the default.**
        ``observed="aird15"`` is the hard-band XLF in Aird et al. (2015)'s
        observed luminosity, :meth:`xlf_observed_hard`.
        ``observed=True`` (or ``"catalogue"``; ``selection="catalogue"`` only,
        hard band) is instead the luminosity function of the luminosities the
        *catalogue* assigns, which a :math:`1/V_{\max}` estimate on those
        luminosities measures:

        .. math::

            \Phi_{\rm cat}(\log L) = \sum_b \Phi_b(\log L - \Delta_b),\qquad
            \Phi_b(\log L_X) = \int\! d\log M\,\frac{dn}{d\log M}
                \,f_{\rm ERDF}(M_*^{\rm c})\,P(\log L_X|M)\,f_b(L_X, M)
                + (\text{satellites}),

        each branch's intrinsic function, its fraction inside the halo
        integral (tilted for centrals, as in :meth:`occupation`), moved by its
        own :meth:`catalogue_shift`.  A shift in :math:`\log L` has no
        Jacobian, and the Compton-thick branch moves off the grid.  No
        threshold is applied: a :math:`1/V_{\max}` estimate undoes it.
        """
        log10m = jnp.log10(field.m)
        prob = self.p_active_loglx_given_m(log10m, p, galaxy_params,
                                           h=field.cosmo.h, z=field.z)
        dn_dlog10m = field.dndm * field.m * _LN10
        if observed == "aird15":
            if band != "hard":
                raise ValueError("the observed hard-band XLF is on Aird et al.'s "
                                 "2-10 keV luminosity; no soft-band shift")
            return self.xlf_observed_hard(field, p, galaxy_params, obsc=obsc)
        if observed:
            return self._observed_xlf(field, p, galaxy_params, band, obsc,
                                      log10m, prob, dn_dlog10m)
        if band == "soft" and self.obscuration == "split":
            phi_b = self._branch_xlfs(field, p, galaxy_params, obsc, log10m,
                                      prob, dn_dlog10m)
            return self.loglx, self._shifted(
                phi_b, self.band_shift(p, field.z, (0.5, 2.0), frame=frame))
        phi = jnp.trapezoid(dn_dlog10m[:, None] * prob, log10m, axis=0)
        if self.has_satellites(p):
            lg_ms, dn, prob_s = self._satellite_kernel(field, p, galaxy_params)
            smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)   # (N_*,)
            phi = phi + self._satellite_duty(p) * jnp.trapezoid(
                smf_sat[:, None] * prob_s, lg_ms, axis=0)
        grid = self.loglx + (jnp.log10(p.k_h2s) if band == "soft" else 0.0)
        return grid, phi

    def _branch_xlfs(self, field, p, galaxy_params, obsc, log10m, prob,
                     dn_dlog10m) -> dict:
        r"""``{branch: Phi_b(log L_X)}``: the intrinsic XLF of each obscuration
        class, its fraction inside the halo integral (tilted for centrals, as
        in :meth:`occupation`) and the satellites' untilted."""
        f_c = obscuration_branches(self.loglx, field.z, obsc, log10m=log10m)
        phi_b = {b: jnp.trapezoid(dn_dlog10m[:, None] * prob * f_c[b], log10m,
                                  axis=0) for b in CATALOGUE_BRANCHES}
        if self.has_satellites(p):
            lg_ms, dn, prob_s = self._satellite_kernel(field, p, galaxy_params)
            smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)
            f_s = obscuration_branches(self.loglx, field.z, obsc)
            for b in CATALOGUE_BRANCHES:
                phi_b[b] = phi_b[b] + self._satellite_duty(p) * jnp.trapezoid(
                    smf_sat[:, None] * prob_s * f_s[b], lg_ms, axis=0)
        return phi_b

    def _shifted(self, phi_b, shift) -> jnp.ndarray:
        r""":math:`\sum_b\sum_i w_i\,\Phi_b(\log L - \Delta_{b,i})` on the grid:
        each class moved by each of its columns' shifts, no Jacobian."""
        phi = 0.0
        for b in CATALOGUE_BRANCHES:
            d, wq = shift[b]
            for j in range(d.shape[0]):
                phi = phi + wq[j] * jnp.interp(self.loglx - d[j], self.loglx,
                                               phi_b[b], left=0.0, right=0.0)
        return phi

    def _observed_xlf(self, field, p, galaxy_params, band, obsc, log10m, prob,
                      dn_dlog10m):
        """:meth:`xlf` at ``observed=True``: the catalogue-luminosity XLF."""
        if self.selection != "catalogue":
            raise ValueError(
                "an observed XLF is the luminosity function of the luminosities "
                "a catalogue assigns, and the catalogue's conversion is declared "
                "by selection='catalogue'.  Build the sector with "
                "AgnSector(..., obscuration='split', selection='catalogue').")
        if band != "hard":
            raise ValueError(
                "the observed XLF is on the catalogue's own 2-10 keV "
                "luminosity; a soft-band grid shift applies to the intrinsic "
                "XLF only")
        phi_b = self._branch_xlfs(field, p, galaxy_params, obsc, log10m, prob,
                                  dn_dlog10m)
        return self.loglx, self._shifted(phi_b, self._shift_nodes(p, field.z))

    def band_shift(self, p: AgnParams, z, band=(0.5, 2.0), *,
                   frame: str = "rest", f_scatter=None) -> dict:
        r"""``{branch: (Delta at each column node, weights)}``: the logarithm of
        the luminosity an AGN of each Comparat et al. (2019) class emits in
        ``band`` [keV], per unit intrinsic 2--10 keV luminosity.

        .. math::

            \Delta_b(N_{\rm H}) = \log_{10}\frac{B(E_1, E_2; \Gamma_x, N_{\rm H})
                + f_{\rm sc}\,B(E_1, E_2; \Gamma_x, 0)}{B(2, 10; \Gamma_x, 0)},

        with :math:`B` of :func:`absorbed_band_energy` at the limits
        :math:`E` (``frame="rest"``, the luminosity a soft-band XLF is quoted
        in) or :math:`E(1+z)` (``"observer"``, the energy that arrives in the
        observed band, the K-correction in the limits);
        :math:`N_{\rm H}` at the Gauss-Legendre nodes of the classes
        (``catalogue_nh_classes``), all three, whatever ``catalogue_nh`` is;
        :math:`f_{\rm sc}` = :data:`F_SCATTER` on the Compton-thin and
        Compton-thick classes, zero on the unobscured one, as in
        :meth:`observed_hard_shift`.
        :meth:`xlf` moves each class's intrinsic function by these shifts and
        weights them as Gauss-Legendre weights, so the column distribution
        is averaged in :math:`\Phi`, not in :math:`\Delta`.

        At :math:`\Gamma_x = 1.723`, in the rest-frame 0.5--2 keV band, the
        unabsorbed ratio is :math:`-0.25`; across its column range the
        unobscured class runs from :math:`-0.26` to :math:`-0.91` and the
        Compton-thin class from :math:`-0.87` to :math:`-1.95`, the scattered
        light dominating above :math:`10^{22.6}\,{\rm cm^{-2}}`, and the
        Compton-thick class sits at the scattered light,
        :math:`\log_{10}0.02 - 0.25 = -1.95`.
        """
        if frame not in ("observer", "rest"):
            raise ValueError(f"unknown band frame {frame!r}; expected "
                             f"'observer' or 'rest'")
        e1, e2 = float(band[0]), float(band[1])
        if not 0.0 < e1 < e2:
            raise ValueError(f"a band needs 0 < emin < emax, got ({e1}, {e2})")
        f_sc = F_SCATTER if f_scatter is None else float(f_scatter)
        g = p.gamma_x
        zb = jnp.asarray(z) if frame == "observer" else jnp.asarray(0.0)
        log_hard = jnp.log10(band_energy(2.0, 10.0, g))
        log_unabs = ((2.0 - g) * jnp.log10(1.0 + zb)
                     + jnp.log10(band_energy(e1, e2, g)))
        out = {}
        for b, (nodes, wq) in self._class_nodes().items():
            la = log10_absorbed_band_energy(e1, e2, g, nodes[:, None], zb).reshape(-1)
            if b != "unobscured" and f_sc > 0.0:
                la = jnp.logaddexp(_LN10 * la,
                                   _LN10 * (log_unabs + math.log10(f_sc))) / _LN10
            out[b] = (la - log_hard, wq)
        return out

    def observed_hard_shift(self, p: AgnParams, z, *, band=AIRD15_BAND,
                            gamma=AIRD15_GAMMA, log_nh=AIRD15_LOG_NH,
                            f_scatter=None) -> dict:
        r"""``{branch: (Delta at each column node, weights)}``: how far below its
        intrinsic 2--10 keV luminosity a hard-band survey with Aird et al.
        (2015)'s conversion puts an AGN.

        .. math::

            \Delta_{b}(N_{\rm H}) = \log_{10}\frac{K(z;\Gamma_x, N_{\rm H})
                + f_{\rm sc}\,K(z;\Gamma_x, 0)}
              {\langle K(z; 1.9, N_{\rm H}')\rangle_{\log N_{\rm H}'\in[20, 21)}},

        :math:`K` the energy in the observed ``band`` (2--7 keV) per unit
        intrinsic 2--10 keV luminosity, the K-correction in the limits as in
        :meth:`catalogue_shift`; :math:`N_{\rm H}` at the Gauss-Legendre nodes
        of Comparat et al. (2019)'s classes (``catalogue_nh_classes``), all
        three, whatever ``catalogue_nh`` is; :math:`f_{\rm sc}` =
        :data:`F_SCATTER` on the thin and thick classes, zero on the unobscured
        one.  The denominator is the survey's own conversion: :data:`AIRD15_GAMMA`
        and a column uniform in :data:`AIRD15_LOG_NH`, averaged in :math:`K`.

        **Reflection is not modelled.**  Aird et al.'s spectral model carries a
        ``pexrav`` component too; in an observed 2--7 keV band at
        :math:`z < 0.6` it adds of order ten per cent to a Compton-thin source
        below :math:`10^{23.5}`, and its hump sits above 10 keV rest.  Nor is
        Galactic absorption: it dims the data and enters their conversion
        alike.
        """
        e1, e2 = float(band[0]), float(band[1])
        f_sc = F_SCATTER if f_scatter is None else float(f_scatter)
        g = p.gamma_x
        z = jnp.asarray(z)
        log_hard = jnp.log10(band_energy(2.0, 10.0, g))
        from jax.scipy.special import logsumexp

        from ..halos.profiles import _leggauss_cached
        x, w = _leggauss_cached(self.catalogue_nh_nodes)
        ref_nodes = jnp.asarray(0.5 * (log_nh[0] + log_nh[1])
                                + 0.5 * (log_nh[1] - log_nh[0]) * x)
        lk_ref = (log10_absorbed_band_energy(e1, e2, gamma, ref_nodes[:, None], z)
                  - jnp.log10(band_energy(2.0, 10.0, gamma))).reshape(-1)
        log_ref = logsumexp(_LN10 * lk_ref, b=jnp.asarray(0.5 * w)) / _LN10
        log_unabs = (2.0 - g) * jnp.log10(1.0 + z) + jnp.log10(band_energy(e1, e2, g))
        out = {}
        for b, (nodes, wq) in self._class_nodes().items():
            la = log10_absorbed_band_energy(e1, e2, g, nodes[:, None], z).reshape(-1)
            if b != "unobscured" and f_sc > 0.0:
                la = jnp.logaddexp(_LN10 * la, _LN10 * (log_unabs + math.log10(f_sc))) / _LN10
            out[b] = (la - log_hard - log_ref, wq)
        return out

    def xlf_observed_hard(self, field, p: AgnParams, galaxy_params, *,
                          obsc: "ObscurationParams" = None, band=AIRD15_BAND,
                          gamma=AIRD15_GAMMA, log_nh=AIRD15_LOG_NH,
                          f_scatter=None):
        r"""The luminosity function of the luminosities a hard-band survey with
        Aird et al. (2015)'s conversion assigns: ``(log10 L grid, Phi)``.

        .. math::

            \Phi_{\rm obs}(\log L) = \sum_b \sum_i w_i\,
                \Phi_b(\log L - \Delta_{b,i}),

        each Comparat et al. (2019) class's intrinsic XLF (:meth:`xlf`'s, the
        fraction inside the halo integral) moved by its columns' shifts
        (:meth:`observed_hard_shift`).  This, not the intrinsic XLF, is what
        Aird et al. (2015)'s LADE fit describes: their hard-band XLF "can be
        described as tracing the 'observed' luminosity, corrected to 2-10 keV
        rest-frame values" (their Sec. 4), with absorption neglected -- so an
        obscured AGN sits in it below its intrinsic luminosity and a
        Compton-thick one, through its scattered light, about 1.7 dex below.
        ``band``, ``gamma`` and ``log_nh`` are the survey's convention,
        Aird et al.'s by default.

        Needs ``obscuration="split"``: the classes are the split.
        """
        if self.obscuration != "split":
            raise ValueError(
                "an observed hard-band XLF is a sum over the obscuration classes, "
                "and this sector was built with obscuration='none'.  Build it "
                "with AgnSector(..., obscuration='split').")
        log10m = jnp.log10(field.m)
        prob = self.p_active_loglx_given_m(log10m, p, galaxy_params,
                                           h=field.cosmo.h, z=field.z)
        dn_dlog10m = field.dndm * field.m * _LN10
        phi_b = self._branch_xlfs(field, p, galaxy_params, obsc, log10m, prob,
                                  dn_dlog10m)
        shift = self.observed_hard_shift(p, field.z, band=band, gamma=gamma,
                                         log_nh=log_nh, f_scatter=f_scatter)
        return self.loglx, self._shifted(phi_b, shift)

    def occupation(self, field, p: AgnParams, galaxy_params, *, branch=None,
                   obsc: "ObscurationParams" = None):
        r""":math:`N_{\rm cen}^{\rm AGN}(>L_{\min}|M_h)`, and the central's selected
        luminosity :math:`\langle N_{\rm cen} L_X\rangle` [erg/s].

        The second is **per halo**, not per AGN: it carries the duty cycle.
        It is what :mod:`ggah_mod.sectors.energetics` wants, and it is already
        the emission weight -- the emission view once multiplied it by the
        occupation a second time.

        At ``selection="catalogue"`` the moment inside the ``tanh`` is the
        catalogue's: the three branches, each at its own threshold in
        intrinsic :math:`L_X`, summed before the ceiling for the mixed sample
        (``branch=None``), so a halo still hosts at most one central AGN
        whichever branch it is in; ``branch="unobscured"`` or ``"obscured"``
        takes one side, with its own ``tanh`` as in the split.
        """
        branch = self._branch(branch)
        prob = self.p_active_loglx_given_m(jnp.log10(field.m), p, galaxy_params,
                                           h=field.cosmo.h, z=field.z)
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

        At ``selection="catalogue"``, ``branch=None`` is the bias of the
        catalogue's mixed sample, and ``obsc`` sets its split.
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

    def weights(self, field, params: AgnParams, galaxy_params, *,
                obsc: "ObscurationParams" = None) -> TracerWeights:
        r"""AGN as a **discrete** tracer: the nuclei of central galaxies at the
        halo centre, and of satellite galaxies on the **galaxy** sector's
        satellite profile, because an active satellite is a satellite galaxy.

        ``f_duty_sat`` is the one number the model cannot supply: a
        satellite's duty cycle at fixed :math:`M_*`, relative to a central's.
        One says they match; the default, 0.046, is the eROSITA x DESI YR3
        MAP (:class:`AgnParams`).

        With ``f_duty_sat = 0``, the centrals-only opt-out, ``w_extended`` is
        ``None``: with nothing on a profile the one-halo auto-spectrum vanishes
        identically -- a Bernoulli central occupation has no self-pairs.
        Leaving the field ``None`` rather than zero is what makes that visible
        in the contract.

        ``obsc`` reaches both occupations.  It changes nothing at the default
        ``selection="luminosity"``, which reads no obscured fraction; at
        ``selection="catalogue"`` it is the split of the sample being counted.
        It was not an argument before 1.3.0, so a spectrum could only ever be
        built at the default coefficients.
        """
        n_cen, _ = self.occupation(field, params, galaxy_params, obsc=obsc)
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
            n_sat = self.satellite_occupation(field, params, galaxy_params,
                                              obsc=obsc)
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

        At ``selection="catalogue"`` the emission is that of the catalogue's
        sample, each branch selected at its own threshold
        (:meth:`catalogue_selection`), with ``obsc`` reaching the fractions.
        In the hard band every branch carries its intrinsic luminosity; in a
        band each carries its own :math:`K_b^n` -- unabsorbed, thin at
        ``log10_nh`` and thick at :data:`LOG10_NH_CT`, so three columns where
        :meth:`band_factor` averages two -- and the fraction is applied once,
        inside the selection.  ``"obscured"`` is the thin and thick branches
        together.
        """
        require_x64("emission_weights", order="1e44 erg/s, and 1e88 for the "
                                               "self-pair it squares",
                    in_range="The counts view, `weights`, is in range and does "
                             "not need this.")
        prob = self.p_active_loglx_given_m(jnp.log10(field.m), params,
                                           galaxy_params, h=field.cosmo.h,
                                           z=field.z)
        if branch is not None:
            branch = self._branch(branch)
            if self.obscuration != "split":
                raise ValueError(
                    "an obscuration branch needs a sector built with "
                    "obscuration='split'")
        if self.selection == "catalogue":
            return self._catalogue_emission(field, params, galaxy_params, prob,
                                            band=band, frame=frame, obsc=obsc,
                                            branch=branch)
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

    def _catalogue_emission(self, field, params, galaxy_params, prob, *,
                            band, frame, obsc, branch) -> TracerWeights:
        """:meth:`emission_weights` at ``selection="catalogue"``."""
        if band is None:
            k1 = k2 = None
            name = "agn:emission"
        else:
            if frame not in ("observer", "rest"):
                raise ValueError(f"unknown band frame {frame!r}; expected "
                                 f"'observer' or 'rest'")
            k1 = self._band_factor_by_column(params, band, frame=frame,
                                             z=field.z, power=1)
            k2 = self._band_factor_by_column(params, band, frame=frame,
                                             z=field.z, power=2)
            name = f"agn:emission[{band[0]:g}-{band[1]:g}keV/{frame[0]}]"
        name = name + "/catalogue" + (f"/{branch}" if branch else "")
        kw = dict(field=field, z=field.z, obsc=obsc, branch=branch)
        lm = jnp.log10(field.m)
        lx = self._moment(prob, params, 1, log10m=lm, lweight=k1, **kw)
        lx2 = self._moment(prob, params, 2, log10m=lm, lweight=k2, **kw)
        w_ext = None
        if self.has_satellites(params):
            kws = dict(branch=branch, obsc=obsc)
            lx_sat = self.satellite_occupation(field, params, galaxy_params, 1,
                                               lweight=k1, **kws)
            lx2 = lx2 + self.satellite_occupation(field, params, galaxy_params,
                                                  2, lweight=k2, **kws)
            w_ext = lx_sat[None, :] * self.galaxies.satellite_uk(field,
                                                                 galaxy_params)
        return TracerWeights(w_point=lx, w_extended=w_ext,
                             norm=jnp.asarray(1.0), discrete=True,
                             bias_weight=None, name=name, self_pair=lx2)


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
        sig = self._sigma_lm(p, self.sigma_mstar(log10m, galaxy_params, h=h))
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
            sig = jnp.broadcast_to(sig, mean.shape)[None, :]
            x = (jnp.asarray(log10_mbh)[:, None] - mean[None, :]) / sig
            return jnp.exp(-0.5 * x ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))

        lg_mean = self.mean_log10_mbh(jnp.log10(field.m), p, galaxy_params,
                                      h=field.cosmo.h, z=field.z)
        frac = self.central_fraction(jnp.log10(field.m), p, galaxy_params,
                                     h=field.cosmo.h, z=field.z)
        phi_cen = field.integrate(
            (field.dndm * frac)[None, :] * gauss(lg_mean, self._sigma_lm(
                p, self.sigma_mstar(jnp.log10(field.m), galaxy_params,
                                    h=field.cosmo.h))),
            axis=-1)
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)   # (N_*,)
        g_sat = gauss(self._mbh(lg_ms, p, field.z),
                      self._sigma_bh_at_fixed_mstar(p))              # (Nb, N_*)
        phi_sat = jnp.trapezoid(g_sat * smf_sat[None, :], lg_ms, axis=-1)
        return phi_cen + phi_sat

    # ==================================================================
    # Black-hole mass and luminosity together
    # ==================================================================
    def _require_split(self, what: str):
        if self.obscuration != "split":
            raise ValueError(
                f"{what} weights each AGN by its unobscured fraction, and this "
                f"sector was built with obscuration='none', which asserts no "
                f"split.  Build it with AgnSector(..., obscuration='split').")

    def _unobscured_fraction(self, log10_lx, z, obsc, log10m=None):
        r""":math:`1 - f_{\rm obs}` (:func:`obscured_fraction`) at ``log10_lx``
        of any shape; ``log10m`` is one host halo mass [log10 Msun/h], which
        tilts the fraction as for a central in :meth:`occupation`."""
        if log10m is None:
            return 1.0 - obscured_fraction(log10_lx, z, obsc)
        lx = jnp.asarray(log10_lx)
        f = obscured_fraction(lx.reshape(-1), z, obsc,
                              log10m=jnp.reshape(log10m, (1,)))
        return 1.0 - f[0].reshape(lx.shape)

    def unabsorbed_flux_luminosity(self, z, p: AgnParams, cosmo, *,
                                   band=(2.0, 10.0)):
        r"""The rest-frame 2--10 keV :math:`\log_{10}L_X` [erg/s] of an
        unabsorbed AGN at redshift ``z`` whose flux in the observed ``band``
        [keV] is ``10**p.log10_flux_min`` [erg/s/cm^2].

        .. math::

            \log L_{\min} = \log\!\left(4\pi d_L^2 F_{\min}\right)
                - \log\!\left[(1+z)^{2-\Gamma}\,\frac{B(E_1, E_2; \Gamma)}
                                                     {B(2, 10; \Gamma)}\right],

        with :math:`B` of :func:`band_energy`, :math:`\Gamma` = ``gamma_x`` and
        :math:`d_L` of ``cosmo``.  The spectrum of a broad-line AGN, so the
        threshold of a type-1 sample (:meth:`joint_mbh_lx`);
        :meth:`flux_limit_luminosity` is the obscured branch's.
        """
        from ..cosmology.background import luminosity_distance
        e1, e2 = float(band[0]), float(band[1])
        if not 0.0 < e1 < e2:
            raise ValueError(f"a band needs 0 < emin < emax, got ({e1}, {e2})")
        d_cm = (jnp.atleast_1d(luminosity_distance(z, cosmo))[0] / cosmo.h
                * C.MPC_CM)
        k = (jnp.power(1.0 + jnp.asarray(z), 2.0 - p.gamma_x)
             * band_energy(e1, e2, p.gamma_x) / band_energy(2.0, 10.0, p.gamma_x))
        return (jnp.log10(4.0 * jnp.pi * d_cm ** 2) + p.log10_flux_min
                - jnp.log10(k))

    def _sample_weight(self, log10_lx, p: AgnParams, *, z, type1=False,
                       obsc=None, log10m=None, lx_min=None, flux_limit=False,
                       cosmo=None, band=(2.0, 10.0), lx_lo=None, lx_hi=None):
        r"""The probability that an AGN of 2--10 keV luminosity ``log10_lx``
        (any shape) is in the sample: the unobscured fraction if ``type1``,
        times one sigmoid :math:`\tfrac12[1 + {\rm erf}((\log L_X - \log
        L_{\rm lim})/\Delta_{\rm sel})]` per lower limit (``lx_min``, the flux
        limit, ``lx_lo``) and one complement per upper limit (``lx_hi``), with
        :math:`\Delta_{\rm sel}` = ``lx_sel_width`` as in :meth:`_selection`."""
        w = jnp.ones_like(jnp.asarray(log10_lx, dtype=float))
        if type1:
            w = w * self._unobscured_fraction(log10_lx, z, obsc, log10m)
        lows = [lx_min, lx_lo]
        if flux_limit:
            if cosmo is None:
                raise ValueError("a flux limit needs the cosmology for the "
                                 "luminosity distance: pass cosmo=")
            lows.append(self.unabsorbed_flux_luminosity(z, p, cosmo, band=band))
        for lo in lows:
            if lo is not None:
                w = w * 0.5 * (1.0 + erf((log10_lx - lo) / p.lx_sel_width))
        if lx_hi is not None:
            w = w * 0.5 * erfc((log10_lx - lx_hi) / p.lx_sel_width)
        return w

    def _log10_lx_at(self, y, p: AgnParams):
        r""":math:`\log_{10}L_X` [erg/s] of an AGN with :math:`y = \log_{10}
        (M_{\rm BH}\lambda/M_\odot)`, i.e. :math:`L_{\rm bol} = 1.26\times10^{38}
        \times 10^y` erg/s, through the sector's bolometric correction
        (:meth:`log10_kbol`)."""
        lbol = y + _LOG10_LBOL_COEF
        return lbol - self.log10_kbol(lbol, p)

    def joint_mbh_lx(self, log10_mstar, p: AgnParams, *, z=0.0,
                     log10_mbh_grid=None, type1=False,
                     obsc: "ObscurationParams" = None, lx_min=None,
                     flux_limit=False, cosmo=None, band=(2.0, 10.0)):
        r"""``(log10_mbh, joint)``: the joint distribution of black-hole mass and
        2--10 keV luminosity of the active black holes in galaxies of stellar
        mass ``log10_mstar``, per dex of :math:`M_{\rm BH}` per dex of
        :math:`L_X`, on :attr:`loglx`; zero below the stellar-mass cut
        ``lg_mstar_min``.

        .. math::

            p(\log M_{\rm BH}, \log L_X | M_*) = f_{\rm ERDF}(M_*)\,
                \mathcal{N}\!\left(u;\, 0,\, \sigma_{\rm BH}\right)\,
                \Phi_\lambda\!\left(t - u\right)
                \frac{d\log L_{\rm bol}}{d\log L_X}\,S(L_X),

        with :math:`u = \log M_{\rm BH} - \langle\log M_{\rm BH}\rangle(M_*)`,
        :math:`t = \log L_{\rm bol}(L_X) - \log(1.26\times10^{38}) - \langle
        \log M_{\rm BH}\rangle - s\,(\log M_* - 10.5)`, :math:`\Phi_\lambda` the
        Eddington-ratio distribution at the pivot (:meth:`_kernel`), so that
        :math:`t - u` is :math:`\log\lambda` minus the stellar-mass tilt of the
        break (:func:`lstar_of_mstar`), :math:`f_{\rm ERDF}` = :func:`duty`,
        the Jacobian of :meth:`_shift` (one at ``kbol="constant"``), and
        :math:`S` the sample (below).  At fixed :math:`M_*` the black-hole mass
        scatters by ``sig_bh`` alone; the scatter of :math:`M_*` at fixed halo
        mass does not enter.

        **Its integral over** :math:`\log M_{\rm BH}` **is**
        :meth:`p_active_loglx_given_mstar` **at** :math:`S = 1`, to round-off.
        The Gaussian and :math:`\Phi_\lambda` are the discrete ones the
        sector's kernel convolves, on the kernel's nodes :math:`u = x_k`
        (:meth:`_gauss_nodes`, spacing :math:`\Delta\log\lambda`, truncated at
        :math:`\pm N_\lambda\Delta\log\lambda/2`), and :math:`\Phi_\lambda` is
        placed by :meth:`_shift` at each black-hole mass, linear in
        :math:`t - u` as the kernel is in :math:`t`, so the sum over the nodes
        times their spacing is the interpolated kernel exactly.  Grid-limited, as every luminosity distribution of the
        sector: the discretisation of the convolution and the truncation of the
        Gaussian (a fraction :math:`6\times10^{-13}` of it at the default
        ``sig_bh`` = 0.314, the grid's half-width of 2.26 dex fixed by
        ``loglam_min`` and ``loglam_max``).

        Parameters
        ----------
        log10_mstar : float or array (N_*,)
            Physical :math:`\log_{10}(M_*/M_\odot)`.
        p : AgnParams
        z : float
            Redshift: it moves the Eddington-ratio break (:func:`lstar_of_z`)
            and the obscured fraction.
        log10_mbh_grid : array (N_b,), optional
            :math:`\log_{10}(M_{\rm BH}/M_\odot)` at which to evaluate.
            ``None`` uses the kernel's nodes,
            :math:`\langle\log M_{\rm BH}\rangle(M_*) + x_k`, one row per
            stellar mass, on which the identity above is exact; on a grid of
            the caller's the Gaussian keeps the nodes' normalisation and
            truncation, and the integral over it is a quadrature.
        type1 : bool
            Weight by the unobscured fraction :math:`1 - f_{\rm obs}(L_X, z)`
            (:func:`obscured_fraction`, Comparat et al. 2019, without the
            halo-mass tilt ``l_blend_m_amp``: there is no host halo at fixed
            :math:`M_*`, as in :meth:`satellite_occupation`).  Needs
            ``obscuration="split"``.
        obsc : ObscurationParams, optional
            The obscured fraction's coefficients; their defaults if ``None``.
        lx_min : float, optional
            A threshold in :math:`\log_{10}L_X` [erg/s], an erf of width
            ``lx_sel_width`` as in :meth:`occupation`.
        flux_limit : bool
            Also select above the flux ``log10_flux_min`` in the observed
            ``band`` [keV], converted with the unabsorbed spectrum
            (:meth:`unabsorbed_flux_luminosity`); needs ``cosmo``.  With
            ``type1`` this is a sample of broad-line AGN above a flux at
            :math:`z`.
        cosmo : Cosmology, optional
            For the luminosity distance of the flux limit.

        Returns
        -------
        log10_mbh : array (N_*, N_x) or (N_*, N_b)
            The black-hole masses of each row [log10 Msun].
        joint : array (N_*, N_x or N_b, N_lx)
            The joint distribution, per dex per dex.  Without the leading axis
            for a scalar ``log10_mstar``.
        """
        if type1:
            self._require_split("a type-1 selection")
        scalar = jnp.ndim(log10_mstar) == 0
        ms = jnp.atleast_1d(jnp.asarray(log10_mstar, dtype=float))
        mean = self._mbh(ms, p, z)
        sig = self._sigma_bh_at_fixed_mstar(p)
        x, g = self._gauss_nodes(sig)
        e = self._erdf_pivot(p, z)
        # Where the kernel itself is non-zero: `_shift` interpolates it on its
        # own t-grid with zeros outside, so the joint is zero there too.
        half = self.loglam.size // 2
        t_grid = (self.loglam[0] - half * self.dlam
                  + jnp.arange(self.loglam.size + 2 * half) * self.dlam)
        inside = self._shift(t_grid, jnp.ones_like(t_grid), p, mean, ms) > 0.0
        if log10_mbh_grid is None:
            u = jnp.broadcast_to(x, (ms.size, x.size))
            dens = jnp.broadcast_to(g / self.dlam, u.shape)
            log10_mbh = mean[:, None] + u
        else:
            mb = jnp.asarray(log10_mbh_grid, dtype=float)
            u = mb[None, :] - mean[:, None]
            norm = jnp.sum(jnp.exp(-0.5 * (x / sig) ** 2)) * self.dlam
            # (mean + x_k) - mean can land one ulp beyond x_k: allow for it,
            # or the outermost node of a grid on the nodes is dropped.
            dens = jnp.where(jnp.abs(u) <= x[-1] * (1.0 + 1e-12),
                             jnp.exp(-0.5 * (u / sig) ** 2) / norm, 0.0)
            log10_mbh = jnp.broadcast_to(mb, u.shape)
        # Phi_lambda, linear between its nodes and zero one node beyond them
        # as the shifted copies the kernel sums are, placed by `_shift` at
        # each black-hole mass: one row per (M_*, M_BH), the Jacobian included.
        lam = jnp.concatenate([self.loglam[:1] - self.dlam, self.loglam,
                               self.loglam[-1:] + self.dlam])
        e_pad = jnp.concatenate([jnp.zeros(1), e, jnp.zeros(1)])
        n_u = u.shape[1]
        phi = self._shift(lam, e_pad, p, (mean[:, None] + u).reshape(-1),
                          jnp.repeat(ms, n_u)).reshape(ms.size, n_u, -1)
        out = jnp.where(inside[:, None, :], dens[:, :, None] * phi, 0.0)
        out = out * self._duty_above_cut(ms, p)[:, None, None]
        out = out * self._sample_weight(
            self.loglx, p, z=z, type1=type1, obsc=obsc, lx_min=lx_min,
            flux_limit=flux_limit, cosmo=cosmo, band=band)
        if scalar:
            return log10_mbh[0], out[0]
        return log10_mbh, out

    def _selected_nodes(self, log10_mstar, p: AgnParams, *, z, **sample):
        r"""``(ms, mean, tilt, x, w)``: the discrete joint of
        :math:`u = \log M_{\rm BH} - \langle\log M_{\rm BH}\rangle` and
        :math:`\log\tilde\lambda` that the kernel convolves, at fixed
        :math:`M_*`, weighted by the sample.

        ``w[a, k, j]`` is :math:`f_{\rm ERDF}(M_{*,a})\,g_k\,
        \Phi_\lambda(\log\lambda_j)\,\Delta\log\lambda\;S(L_X)`, shape
        ``(N_*, N_x, N_lambda)``, with :math:`g_k` and :math:`\Phi_\lambda` of
        :meth:`_gauss_nodes` and :meth:`_erdf_pivot` (each summing to one) and
        :math:`L_X` the luminosity of that node,
        :math:`\log L_{\rm bol} = \log(1.26\times10^{38}) + \langle\log M_{\rm
        BH}\rangle + x_k + s(\log M_* - 10.5) + \log\lambda_j`
        (:meth:`_log10_lx_at`).  The sample :math:`S` is evaluated at each
        node's own luminosity, so every moment taken here is a sum over the
        nodes, exactly normalised, and smooth in the parameters.
        """
        ms = jnp.atleast_1d(jnp.asarray(log10_mstar, dtype=float))
        mean = self._mbh(ms, p, z)
        tilt = _lstar_tilt(ms, p)
        x, g = self._gauss_nodes(self._sigma_bh_at_fixed_mstar(p))
        e = self._erdf_pivot(p, z)
        y = ((mean + tilt)[:, None, None] + x[None, :, None]
             + self.loglam[None, None, :])
        lx = self._log10_lx_at(y, p)
        w = (g[None, :, None] * (e * self.dlam)[None, None, :]
             * self._sample_weight(lx, p, z=z, **sample))
        return ms, mean, tilt, x, w * duty(ms, p)[:, None, None]

    def selected_mbh_moments(self, log10_mstar, lx_lo, lx_hi, p: AgnParams, *,
                             z=0.0, type1=True, obsc: "ObscurationParams" = None,
                             lx_min=None, flux_limit=False, cosmo=None,
                             band=(2.0, 10.0), log10_mbh_error=None):
        r"""``(mean, std)`` of :math:`\log_{10}(M_{\rm BH}/M_\odot)` of the
        AGN of a sample in galaxies of stellar mass ``log10_mstar``, in the
        luminosity bin ``lx_lo`` to ``lx_hi``: what a reverberation-mapping or
        virial-mass sample measures.

        .. math::

            \langle\log M_{\rm BH}\rangle_S = \frac{\int\!\!\int
                \log M_{\rm BH}\; p(\log M_{\rm BH}, \log L_X|M_*)\,S\,
                d\log M_{\rm BH}\,d\log L_X}{\int\!\!\int p\,S\,
                d\log M_{\rm BH}\,d\log L_X},

        and the standard deviation likewise, with :math:`p` the joint of
        :meth:`joint_mbh_lx` and :math:`S` the sample (type 1, flux limit,
        threshold) times the bin, whose edges are erfs of width
        ``lx_sel_width``.  Taken on the discrete joint of
        :meth:`_selected_nodes`, with the sample at each node's own luminosity;
        :meth:`joint_mbh_lx` on the luminosity grid gives the same moments to
        the grid's interpolation.  A sample selected in :math:`L_X` holds
        black holes above the mean relation at its stellar mass (the bias of
        Lauer et al. 2007).  Moments of the sample, so neither the duty cycle
        nor the stellar-mass cut enters.

        Parameters
        ----------
        log10_mstar : float or array (N_*,)
            Physical :math:`\log_{10}(M_*/M_\odot)`.
        lx_lo, lx_hi : float or None
            The bin in :math:`\log_{10}L_X` [erg/s, 2--10 keV, intrinsic];
            ``None`` leaves that side open.
        type1, obsc, lx_min, flux_limit, cosmo, band
            The sample, as in :meth:`joint_mbh_lx`; ``type1`` is ``True`` by
            default.
        log10_mbh_error : float, optional
            The measurement error of :math:`\log M_{\rm BH}` [dex], added in
            quadrature to the standard deviation (0.3--0.4 dex for single-epoch
            virial masses); ``None`` is the true masses.

        Returns
        -------
        mean, std : float or array (N_*,)
            [log10 Msun] and [dex].
        """
        if type1:
            self._require_split("a type-1 selection")
        scalar = jnp.ndim(log10_mstar) == 0
        _, mean, _, x, w = self._selected_nodes(
            log10_mstar, p, z=z, type1=type1, obsc=obsc, lx_min=lx_min,
            flux_limit=flux_limit, cosmo=cosmo, band=band, lx_lo=lx_lo,
            lx_hi=lx_hi)
        wu = jnp.sum(w, axis=-1)                                  # (N_*, N_x)
        n = jnp.sum(wu, axis=-1)
        mu = jnp.sum(wu * x, axis=-1) / n
        var = jnp.sum(wu * (x[None, :] - mu[:, None]) ** 2, axis=-1) / n
        if log10_mbh_error is not None:
            var = var + jnp.asarray(log10_mbh_error) ** 2
        out_mean, out_std = mean + mu, jnp.sqrt(var)
        if scalar:
            return out_mean[0], out_std[0]
        return out_mean, out_std

    def type1_erdf(self, log10_mstar, p: AgnParams, *, z=0.0, type1=True,
                   obsc: "ObscurationParams" = None, lx_min=None,
                   flux_limit=False, cosmo=None, band=(2.0, 10.0), lx_lo=None,
                   lx_hi=None, log10_mbh_error=None, normalise=True):
        r"""``(log10_lambda, pdf)``: the distribution of the Eddington ratio
        :math:`\lambda = L_{\rm bol}/(1.26\times10^{38}M_{\rm BH})` of the AGN
        of a sample in galaxies of stellar mass ``log10_mstar``, per dex.

        .. math::

            p_S(\log\lambda|M_*) \propto \Phi_\lambda(\log\lambda - s\,(\log M_*
                - 10.5)) \int d\log M_{\rm BH}\,
                \mathcal{N}(\log M_{\rm BH};\langle\log M_{\rm BH}\rangle,
                \sigma_{\rm BH})\,S\big(L_X(M_{\rm BH}\lambda)\big),

        on the discrete joint of :meth:`_selected_nodes`.  The Eddington-ratio
        distribution of every active black hole is :math:`\Phi_\lambda`; a
        sample selected in luminosity favours the high :math:`\lambda`, and a
        type-1 sample the luminosities at which the unobscured fraction is
        high.

        Parameters
        ----------
        log10_mstar : float or array (N_*,)
            Physical :math:`\log_{10}(M_*/M_\odot)`.
        type1, obsc, lx_min, flux_limit, cosmo, band
            The sample, as in :meth:`joint_mbh_lx`; ``type1`` is ``True`` by
            default.
        lx_lo, lx_hi : float, optional
            A bin in :math:`\log_{10}L_X` [erg/s], as in
            :meth:`selected_mbh_moments`.
        log10_mbh_error : float, optional
            The measurement error of :math:`\log M_{\rm BH}` [dex]: the measured
            :math:`\log\lambda` is the true one minus that error, so the
            distribution is convolved with a Gaussian of that width (the
            discrete one of :meth:`_gauss_nodes`), on the same grid; the part
            moved off the grid is lost.  ``None`` is the true masses.
        normalise : bool
            ``True``: the distribution of the sample, integrating to one over
            the grid.  ``False``: per galaxy, integrating to the fraction of the
            galaxies that host an AGN of the sample (the duty cycle included,
            zero below ``lg_mstar_min``).

        Returns
        -------
        log10_lambda : array (N_*, N_lambda)
            :math:`\log_{10}\lambda`, the sector's grid moved by the stellar-mass
            tilt of the break.  Without the leading axis for a scalar
            ``log10_mstar``.
        pdf : array (N_*, N_lambda)
            Per dex of :math:`\lambda`.
        """
        if type1:
            self._require_split("a type-1 selection")
        scalar = jnp.ndim(log10_mstar) == 0
        ms, _, tilt, _, w = self._selected_nodes(
            log10_mstar, p, z=z, type1=type1, obsc=obsc, lx_min=lx_min,
            flux_limit=flux_limit, cosmo=cosmo, band=band, lx_lo=lx_lo,
            lx_hi=lx_hi)
        pdf = jnp.sum(w, axis=1) / self.dlam                  # (N_*, N_lam)
        if log10_mbh_error is not None:
            _, g = self._gauss_nodes(jnp.asarray(log10_mbh_error))
            half = self.loglam.size // 2
            pdf = _convolve_full(g, pdf)[:, half:half + self.loglam.size]
        if normalise:
            pdf = pdf / (jnp.sum(pdf, axis=-1, keepdims=True) * self.dlam)
        else:
            pdf = jnp.where((ms >= self.lg_mstar_min)[:, None], pdf, 0.0)
        log10_lambda = self.loglam[None, :] + tilt[:, None]
        if scalar:
            return log10_lambda[0], pdf[0]
        return log10_lambda, pdf

    def _central_bh_conditional(self, p: AgnParams, sigma_mstar):
        r"""``(beta, tau)`` of :meth:`active_bhmf`: the mean :math:`\beta u`
        and the width :math:`\tau` [dex] of the shift :math:`s\,\delta_*` of a
        central's Eddington-ratio break at fixed :math:`u = \log M_{\rm BH} -
        \langle\log M_{\rm BH}\rangle(M_h)`, with ``sigma_mstar`` its
        stellar-mass scatter at fixed halo mass (:meth:`sigma_mstar`).
        :math:`\tau` is floored at :math:`10^{-3}` dex, as :meth:`_sigma_lx` is;
        at :math:`s = 0` both vanish."""
        s = _lstar_tilt(None, p)
        var_d = sigma_mstar ** 2 * (1.0 - p.rho)
        sig_lm = self._sigma_lm(p, sigma_mstar)
        beta = p.al_bh * s * var_d / sig_lm ** 2
        tau = jnp.sqrt(jnp.maximum(s ** 2 * var_d * p.sig_bh ** 2 / sig_lm ** 2,
                                   1e-6))
        return beta, tau

    def active_bhmf(self, field, log10_mbh, p: AgnParams, galaxy_params, *,
                    type1=True, obsc: "ObscurationParams" = None, lx_min=None,
                    flux_limit=False, band=(2.0, 10.0), with_duty=True,
                    batch_size: int = 16):
        r""":math:`dn/d\log_{10}M_{\rm BH}` [(Mpc/h)^-3 dex^-1] of the active
        black holes of a sample: shining (``with_duty``), type 1
        (``type1``), above a luminosity or a flux (``lx_min``,
        ``flux_limit``).

        .. math::

            \frac{dn_S}{d\log M_{\rm BH}} = \int dM\,\frac{dn}{dM}\,
                F_{\rm c}\,f_{\rm ERDF}(M_*^{\rm c})\,
                \mathcal{N}(\log M_{\rm BH};\langle\log M_{\rm BH}\rangle,
                \sigma_{\rm lm})\,A_{\rm c}
            + f_{\rm duty}^{\rm sat}\int d\lg M_*\,\phi_{\rm s}(M_*)\,
                f_{\rm ERDF}(M_*)\,\mathcal{N}(\log M_{\rm BH};
                \langle\log M_{\rm BH}\rangle(M_*), \sigma_{\rm BH})\,
                A_{\rm s},

        the two terms of :meth:`black_hole_mass_function` (centrals at fixed
        halo mass above the cut, satellites through their stellar-mass
        function) times the duty cycles and :math:`A`, the fraction of the
        shining black holes of that mass that are in the sample.  For a
        satellite, :math:`A_{\rm s} = \int d\log\lambda\,\Phi_\lambda\,S(L_X)`
        with :math:`\log L_{\rm bol} = \log(1.26\times10^{38}) + \log M_{\rm BH}
        + s(\log M_* - 10.5) + \log\lambda`.  A central's stellar mass scatters
        at fixed halo mass by :math:`\delta_*`, of variance
        :math:`\sigma_{M_*}^2(1 - \rho)` (:meth:`_sigma_lm`), which moves
        :math:`\log M_{\rm BH}` by :math:`\alpha_{\rm BH}\delta_*` and the
        break by :math:`s\,\delta_*`; at fixed :math:`u = \log M_{\rm BH} -
        \langle\log M_{\rm BH}\rangle`, :math:`s\,\delta_*` is Gaussian with
        mean :math:`\beta u` and width :math:`\tau`,

        .. math::

            \beta = \frac{\alpha_{\rm BH}\,s\,\sigma_{M_*}^2(1-\rho)}
                         {\sigma_{\rm lm}^2},\qquad
            \tau^2 = \frac{s^2\sigma_{M_*}^2(1-\rho)\,\sigma_{\rm BH}^2}
                          {\sigma_{\rm lm}^2},

        so :math:`A_{\rm c} = \int dt\,K_\tau(t)\,S(L_X)` with
        :math:`\log L_{\rm bol} = \log(1.26\times10^{38}) + \langle\log M_{\rm
        BH}\rangle + s(\log M_*^{\rm c} - 10.5) + (1 + \beta)u + t` and
        :math:`K_\tau` :meth:`_kernel` of width :math:`\tau`.  Integrated over
        :math:`u` it gives the central's kernel of width :math:`\sigma_{\rm lx}`
        (:meth:`_sigma_lx`), and at :math:`s = 0` the satellites' form.  The
        type-1 fraction of a central is tilted with its halo mass, as in
        :meth:`occupation`; a satellite's is not.

        **At no selection and** ``with_duty=False`` **it is**
        :meth:`black_hole_mass_function` to round-off: both :math:`A` are
        sums of a distribution normalised to one on its own nodes.

        Parameters
        ----------
        field : HaloField
        log10_mbh : array (N_b,)
            :math:`\log_{10}(M_{\rm BH}/M_\odot)`.
        p : AgnParams
        galaxy_params
            The galaxy sector's parameters.
        type1, obsc, lx_min, flux_limit, band
            The sample, as in :meth:`joint_mbh_lx`, at the field's redshift
            and cosmology; ``type1`` is ``True`` by default.
        with_duty : bool
            Apply the duty cycles :func:`duty` and ``f_duty_sat``; ``False``
            sets both to one.
        batch_size : int
            Stellar or halo masses per step of the ``lax.map`` over them, which
            bounds the memory of the :math:`(N_b, N_\lambda)` slices.
        """
        if type1:
            self._require_split("a type-1 selection")
        z, h = field.z, field.cosmo.h
        mb = jnp.asarray(log10_mbh, dtype=float)
        sample = dict(type1=type1, obsc=obsc, lx_min=lx_min,
                      flux_limit=flux_limit, cosmo=field.cosmo, band=band)

        def gauss(mean, sig):
            sig = jnp.broadcast_to(sig, mean.shape)[None, :]
            x = (mb[:, None] - mean[None, :]) / sig
            return jnp.exp(-0.5 * x ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))

        # -- centrals, at fixed halo mass ------------------------------------
        log10m = jnp.log10(field.m)
        ms_c = self.log10_mstar(log10m, galaxy_params, h=h, z=z)
        mean_c = self._mbh(ms_c, p, z)
        sig_ms = self.sigma_mstar(log10m, galaxy_params, h=h)
        sig_lm = self._sigma_lm(p, sig_ms)
        frac = self.central_fraction(log10m, p, galaxy_params, h=h, z=z)
        beta, tau = self._central_bh_conditional(p, sig_ms)
        t_grid, k_tau = self._kernel(p, tau, z=z)                # (NM, N_t)

        def a_cen(args):
            mean, tilt, b, k, lm = args
            y = (mean + tilt + (1.0 + b) * (mb - mean))[:, None] + t_grid[None, :]
            lx = self._log10_lx_at(y, p)
            return jnp.sum(k * self._sample_weight(lx, p, z=z, log10m=lm,
                                                   **sample), axis=-1) * self.dlam

        a_c = jax.lax.map(a_cen, (mean_c, _lstar_tilt(ms_c, p), beta, k_tau,
                                  log10m), batch_size=batch_size)  # (NM, N_b)
        weight_c = field.dndm * frac
        if with_duty:
            weight_c = weight_c * duty(ms_c, p)
        phi_cen = field.integrate(
            weight_c[None, :] * gauss(mean_c, sig_lm) * a_c.T, axis=-1)

        # -- satellites, at fixed stellar mass -------------------------------
        lg_ms, dn = self._satellite_csmf(field, galaxy_params)
        smf_sat = field.integrate(field.dndm[None, :] * dn, axis=-1)   # (N_*,)
        e = self._erdf_pivot(p, z)

        def a_sat(tilt):
            lx = self._log10_lx_at(mb[:, None] + tilt + self.loglam[None, :], p)
            return jnp.sum(e * self._sample_weight(lx, p, z=z, **sample),
                           axis=-1) * self.dlam

        a_s = jax.lax.map(a_sat, _lstar_tilt(lg_ms, p),
                          batch_size=batch_size)                  # (N_*, N_b)
        weight_s = smf_sat
        if with_duty:
            weight_s = weight_s * duty(lg_ms, p) * self._satellite_duty(p)
        g_sat = gauss(self._mbh(lg_ms, p, z), self._sigma_bh_at_fixed_mstar(p))
        phi_sat = jnp.trapezoid(g_sat * a_s.T * weight_s[None, :], lg_ms,
                                axis=-1)
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
            prob = self.p_active_loglx_given_m(log10m, p, galaxy_params,
                                               h=field.cosmo.h, z=field.z)
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

        At the defaults and the ``zumandelbaum15`` galaxy defaults, on a halo
        field from :math:`10^{10}` to :math:`10^{16}\,h^{-1}M_\odot` at
        :math:`z = 0`, it is :math:`2.08\times10^{-6}`, 0.52 of Fukugita &
        Peebles (2004), counted down to the :math:`10^8\,M_\odot` cut (module
        docstring).  What remains is
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
