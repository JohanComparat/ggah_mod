r"""Neutral gas: HI, H2, and the helium that comes with them.

The smallest phase in the census that still clears one per cent of
:math:`\Omega_b` -- Fukugita & Peebles (2004) put HI + He I at 1.4% and
molecular gas at 0.36% -- and the one with the most direct measurements
attached to it: an HI mass function from ALFALFA, :math:`\Omega_{\rm HI}(z)`
from 21-cm emission and damped Lyman-:math:`\alpha` systems, and
:math:`\Omega_{\rm H_2}(z)` from CO.  It is here because a census row nobody can
check is not worth having, and this row has three independent checks.

Three published relations, two questions
----------------------------------------

**HI lives in galaxies.**  ``catinella18`` gives every galaxy -- central or
satellite -- an HI mass set by its own stellar mass, a split normal in
:math:`\lg M_{\rm HI}` whose median is the xGASS gas-fraction scaling
(Catinella et al. 2018, MNRAS 476, 875, Table 1), whose gas-rich width is the
star-forming sequence's (Janowiecki et al. 2020) and whose gas-poor width keeps
xGASS's interquartile range, with satellites suppressed in massive hosts as
Brown et al. (2017) measure.  It reaches a halo only through the galaxies the
galaxy sector puts in it: the central and satellite conditional stellar-mass
functions, the chain :class:`~ggah_mod.sectors.agn.AgnSector` already runs.
The two widths are not a refinement: xGASS's interquartile range as one
symmetric width puts 50 times FASHI's galaxies at :math:`10^{10.7}\,M_\odot`
and :math:`\Omega_{\rm HI}` at 1.9 times Dev et al. (2024), because it hands the
gas-rich side the quenched galaxies' tail.  It is the relation the
per-galaxy HI mass function, the census at :math:`z \simeq 0` and HI-selected
galaxies need, and the only one that can be compared with ALFALFA or FASHI at
all.  See :meth:`ColdGasSector.hi_mass_function`.

**HI per halo, for intensity mapping.**  The other two map a *halo* mass to the
HI summed over every galaxy it holds.  That is what a 21-cm intensity map sums,
and ``padmanabhan17`` is the only relation here calibrated above
:math:`z \simeq 0`.  Same shape as :mod:`~ggah_mod.sectors.sham`'s: a name maps
to a function of halo mass, and the *published* central parameters are the
defaults, so a disagreement with data is a statement about the relation rather
than about the transcription.

* ``padmanabhan17`` -- the form of Padmanabhan & Refregier (2017), MNRAS 464,
  4008, Eq. (1): a power law in mass with virial-velocity cutoffs.  **The
  numbers are not that paper's.**  They are Padmanabhan, Refregier & Amara
  (2017), MNRAS 469, 2323, Table 3 -- the same group's joint fit to HI galaxy
  clustering and column densities, intensity mapping and DLAs over
  :math:`0 < z < 5`, which is also where the profile and its concentration come
  from, and which Padmanabhan, Maartens, Umeh & Camera (2023) restate as their
  Eqs. (16), (19) and (21).

  The first paper's Table 1 gives :math:`\alpha = 0.17`; the second gives
  :math:`0.09`.  That factor of two is the neutral-gas budget: at 0.17,
  :math:`\Omega_{\rm HI}` is 1.76 times Dev et al. (2024), and at 0.09 it is
  0.94.  The second fit also drops the upper cutoff, which is kept here -- it
  is worth nothing at these parameters, measured, and removing it would remove
  a tested branch for no number.
* ``villaescusa18`` -- Villaescusa-Navarro et al. (2018), ApJ 866, 135
  (arXiv:1804.09180), Eq. (13) and Table 1, measured in IllustrisTNG
  (TNG100-1).  A power law with one low-mass exponential cutoff.

  The defaults are the **FoF** row at :math:`z = 0`.  Table 1 also gives an
  FoF-SO row -- :math:`\alpha = 0.16`, :math:`M_0 = 4.1\times10^{10}`,
  :math:`M_{\min} = 2.4\times10^{12}` -- and the shipped halo field uses a
  spherical-overdensity definition, so **FoF-SO is arguably the row this
  package should use**.

  Still not switched, and now **selectable**: :data:`VILLAESCUSA18_ROWS` carries
  both, and ``ColdGasParams(**villaescusa18_row("fof_so"))`` is the one call
  that changes it.  Choosing between them is a decision about halo finders that
  nobody here has made -- the slope differs by a third, larger than either error
  bar -- so this package still declines to make it.  What it no longer does is
  make the other row reachable only by editing three defaults, which is the
  difference between a recorded caveat and an available option.

They are not the same function and are not meant to agree: one is a fit to
observations across redshift, the other a measurement in one simulation at one
redshift.  Which is why both are here and neither is the default in the sense of
being right.

Both are *halo-total*, so neither has a per-galaxy HI mass function, and the
halo-total one they used to be compared through measured a different object
from ALFALFA's -- HI per halo against HI per galaxy -- and had no knee.  It
was read by nothing downstream and was removed with its scatter parameter;
``catinella18`` is what answers the per-galaxy question now.

The profile: an exponential, with one parameter
-----------------------------------------------

:meth:`ColdGasSector.u_k` puts the neutral gas on the exponential of
Padmanabhan, Refregier & Amara (2017), Eq. (2) -- restated by Padmanabhan,
Maartens, Umeh & Camera (2023) as their Eq. (18) --
:math:`\rho_{\rm HI} \propto e^{-r/r_s}` with :math:`r_s = R_\Delta/c_{\rm HI}`,
whose transform is exact and closed:

.. math::  \tilde u_{\rm HI}(k|M) = \left[1 + (k r_s)^2\right]^{-2}.

One parameter, :attr:`ColdGasParams.c_hi_0`, and no quadrature.  HI profiles
are not measured well enough to fit a shape, so this is the first-order choice
made for parameter economy.

**Why not the hot gas's shape.**  That was the first design, and measuring it
is what ruled it out.  The concentration comes from a fit to *this* profile,
and it does not transplant: at the same :math:`c = 114.6`, the gas's
generalised NFW (1, 1.9, 2.7) puts the half-mass radius at
:math:`0.206\,R_\Delta` against :math:`0.023` for the exponential -- nine times
further out, because an outer slope below three keeps the enclosed mass
growing all the way to the truncation.  The borrowed shape gave HI only about
twice the compactness of the dark matter it replaced.  Using the paper's own
profile also removes three borrowed exponents and any coupling to the hot-gas
sector, so it is the *cheaper* option as well as the consistent one.

**The amplitude is not a parameter here, and adding one would be degenerate.**
:meth:`ColdGasSector.weights` forms ``amp * u`` with ``amp`` a *mass* and
:math:`\tilde u(k \to 0) = 1` exactly, so the profile's own normalisation
cancels identically.  What sets it is Padmanabhan & Refregier (2017)'s Eq. (4),
:math:`\int 4\pi r^2\rho_{\rm HI}\,dr = M_{\rm HI}(M)`, which for this profile
is Padmanabhan, Refregier & Amara (2017)'s Eq. (4) -- the 2023 paper's Eq. (20)
-- :math:`\rho_0 = M_{\rm HI}/8\pi r_s^3`, and
:math:`M_{\rm HI}`'s own normalisation :attr:`ColdGasParams.alpha_hi` is
already fitted.

**Untruncated, as the paper has it.**  The closed form integrates to infinity,
and the paper says where that is allowed: its Eq. (4) "can be well
approximated" for :math:`c_{\rm HI} > 10`.  A measurement here agrees: against
the profile cut at the halo, the closed form is within
:math:`8.7\times10^{-4}` at :math:`c_{\rm HI,0} = 10` -- where the smallest
concentration on the grid is 11.4 -- and off by per cent once the concentration
falls below ten.  Across the published prior it is good to
:math:`9\times10^{-10}`, so the prior's floor is not a numerical limit.

Nothing on the mass side reads the profile -- :math:`\Omega_{\rm HI}` and
:math:`b_{\rm HI}` are integrals over :math:`M` with no :math:`\tilde u` in
them -- which a test holds to bit-for-bit.

``catinella18`` has no profile parameter at all.  A central's HI is a point
mass and a satellite's sits on the galaxy sector's
:meth:`~ggah_mod.sectors.galaxies.GalaxySector.satellite_uk`, like the stars;
:meth:`ColdGasSector.u_k` is their mass-weighted mixture.  Measured at
:math:`z = 0`, on :math:`P_{\rm mm}/P_{\rm DMO}` at
:math:`k = 10\,h\,{\rm Mpc}^{-1}`: removing the neutral gas altogether moves it
by :math:`3\times10^{-3}`; replacing PRA17's exponential by a point mass, which
is the same compactness, by :math:`2\times10^{-4}`; and replacing
``padmanabhan17`` by ``catinella18`` by :math:`-2.1\times10^{-3}`
(:math:`-6.6\times10^{-3}` at :math:`k = 30`) with 0.9 of the mass -- the
satellites' share, 70 per cent of a :math:`10^{13}` halo's HI, is on an
extended profile, which no single exponential could say.  So the matter field
does not need a profile *parameter*, but it does need the central/satellite
split.

What is deliberately not here
------------------------------

**A molecular fraction that depends on anything.**  :attr:`ColdGasParams.r_mol`
is a constant :math:`M_{\rm H_2}/M_{\rm HI}`, anchored on the local ratio.  The
pressure-based prescription of Blitz & Rosolowsky (2006) makes it a function of
the disc, which this package does not model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..cosmology import constants as C
from .calibration import (Calibration, check_sector_calibration,
                          check_sector_range)
from .energetics import v_delta_squared
from .params import Flat, Gaussian, Param, SectorParams, sector_params
from .protocol import TracerWeights

__all__ = ["ColdGasParams", "ColdGasSector", "HI_HALO", "HI_GALAXY",
           "HI_CALIBRATION", "RELATIONS",
           "make_hi_halo", "m_hi_padmanabhan17", "m_hi_villaescusa18",
           "lg_fhi_catinella18", "lg_mhi_catinella18", "fit_xgass_medians",
           "XGASS_TABLE1", "XGASS_H", "XGASS_FIT_BINS", "XGASS_IQR_DEX",
           "SIGMA_FHI_UP_J20", "SIGMA_FHI_DOWN_XGASS", "LG_MH_SAT_PIVOT",
           "split_normal_quantile", "split_normal_iqr", "split_normal_pdf",
           "split_normal_mode_minus_median",
           "split_normal_mean_over_median", "satellite_hi_offset",
           "LG_MSTAR_MIN_HI", "LG_MSTAR_MAX_HI",
           "N_MSTAR_PER_DEX_HI",
           "VILLAESCUSA18_ROWS", "villaescusa18_row", "c_hi_padmanabhan",
           "C_HI_0", "R200M_OVER_RVIR_Z0",
           "Y_HELIUM", "helium_correction", "brightness_temperature",
           "T_B_COEF_MK", "A_10_HZ", "NU_21_HZ"]


#: Primordial helium mass fraction.  Planck-consistent BBN.
Y_HELIUM = 0.2454

#: Neutral gas per unit neutral hydrogen: :math:`1/(1-Y)`.
#:
#: Written as a function of :data:`Y_HELIUM` rather than as the literal 1.325,
#: because the two would otherwise be free to disagree -- and because the
#: quantity the literature reports flips between the two conventions without
#: always saying which.  Fukugita & Peebles (2004) row 3.9 is "HI + He I", i.e.
#: with this factor already in; Dev et al. (2024) quote
#: :math:`\Omega_{\rm HI}` without it and
#: :math:`\Omega_{\rm neutral\,gas}` with it.
def helium_correction(y_helium: float = Y_HELIUM):
    r""":math:`1/(1-Y)`, the neutral-gas mass per unit HI mass."""
    return 1.0 / (1.0 - y_helium)


# =========================================================================
# The HI-halo mass relations
# =========================================================================

def m_hi_padmanabhan17(m, z, cosmo, alpha=0.09, beta=-0.58,
                       log10_vc0=1.56, log10_vc1=4.39, mdef=None):
    r""":math:`M_{\rm HI}(M)` -- Padmanabhan & Refregier (2017), Eq. (1).

    The form is that paper's; the default numbers are Padmanabhan, Refregier &
    Amara (2017), Table 3.  See the module docstring for why they differ by a
    factor of two in :math:`\alpha`, and why that is the whole neutral-gas
    budget.

    .. math::

        M_{\rm HI}(M) = \alpha f_{\rm H,c} M
            \left(\frac{M}{10^{11}h^{-1}M_\odot}\right)^{\beta}
            \exp\!\left[-\left(\frac{v_{c0}}{v_c(M)}\right)^{3}\right]
            \exp\!\left[-\left(\frac{v_c(M)}{v_{c1}}\right)^{3}\right]

    with :math:`f_{\rm H,c} = (1-Y)\,\Omega_b/\Omega_m` the cosmic hydrogen
    fraction, so the relation carries the cosmology rather than a frozen
    baryon fraction -- which the published form leaves implicit and which a
    forecast that varies :math:`\Omega_b` needs.

    The two cutoffs are the reason to prefer this one over a bare power law:
    haloes below :math:`v_c \sim 30` km/s and above :math:`\sim 200` km/s
    preferentially host no HI, so a relation without them puts neutral gas in
    clusters, where the census then sees it.

    Parameters
    ----------
    m : array (NM,) [Msun/h]
    z : float
    cosmo : Cosmology
    alpha : float
        HI fraction relative to cosmic.  0.09 +/- 0.01 (Padmanabhan, Refregier
        & Amara 2017, Table 3); Padmanabhan & Refregier (2017), Table 1, gave
        0.17 +/- 0.02.
    beta : float
        Excess logarithmic slope, pivoted at 1e11 Msun/h.  -0.58 +/- 0.06
        (was -0.55 +/- 0.12); negative, so the relation is sub-linear.
    log10_vc0 : float
        Lower virial-velocity cutoff [log10 km/s].  1.56 +/- 0.04 (was
        1.57 +/- 0.03).
    log10_vc1 : float
        Upper cutoff [log10 km/s].  Only the earlier fit has one, at
        4.39 +/- 0.75 -- above every halo's virial velocity, so it changes
        nothing at the current parameters.  Kept so that it can be switched
        on.

    Returns
    -------
    array (NM,) [Msun/h]
    """
    m = jnp.atleast_1d(jnp.asarray(m))
    f_h_cosmic = (1.0 - Y_HELIUM) * cosmo.Omega_b / cosmo.Omega_m
    v_c = jnp.sqrt(v_delta_squared(m, z, cosmo, mdef))            # km/s
    vc0 = jnp.power(10.0, log10_vc0)
    vc1 = jnp.power(10.0, log10_vc1)
    return (alpha * f_h_cosmic * m
            * jnp.power(m / 1e11, beta)
            * jnp.exp(-jnp.power(vc0 / v_c, 3.0))
            * jnp.exp(-jnp.power(v_c / vc1, 3.0)))


@jax.jit
def m_hi_villaescusa18(m, log10_m0=10.6335, alpha=0.24, log10_m_min=12.3010):
    r""":math:`M_{\rm HI}(M)` -- Villaescusa-Navarro et al. (2018), IllustrisTNG.

    .. math::

        M_{\rm HI}(M) = M_0\left(\frac{M}{M_{\min}}\right)^{\alpha}
            \exp\!\left[-\left(\frac{M_{\min}}{M}\right)^{0.35}\right]

    Table 1, FoF halos at :math:`z = 0`: :math:`\alpha = 0.24 \pm 0.05`,
    :math:`M_0 = (4.3 \pm 1.1)\times10^{10}\,h^{-1}M_\odot`,
    :math:`M_{\min} = (2.0 \pm 0.6)\times10^{12}\,h^{-1}M_\odot`.  The defaults
    here are those, as :math:`\log_{10}` -- a mass parameter that a sampler
    walks linearly spends most of its proposals in the wrong decade.  See the
    module docstring for the FoF against FoF-SO choice, which is not settled.

    Takes no cosmology, and cannot: it is a measurement in one simulation at one
    cosmology, so a forecast that varies :math:`\Omega_b` gets
    :math:`\partial M_{\rm HI}/\partial\Omega_b \equiv 0` from it.  That is a
    property of the relation rather than of this transcription, and it is the
    same distinction :mod:`~ggah_mod.halos.concentration` draws between fits
    that accept a cosmology and fits that cannot.
    """
    m = jnp.atleast_1d(jnp.asarray(m))
    m0 = jnp.power(10.0, log10_m0)
    m_min = jnp.power(10.0, log10_m_min)
    return (m0 * jnp.power(m / m_min, alpha)
            * jnp.exp(-jnp.power(m_min / m, 0.35)))


#: Villaescusa-Navarro et al. (2018) Table 1, both halo-finder rows.
#:
#: **A row and not a registry entry, deliberately.**  The two are the same
#: functional form with different fitted values, and ``HI_HALO`` is a registry
#: of *forms* -- ``padmanabhan17`` is there because it is a different equation
#: with a different signature.  Adding ``villaescusa18_fofso`` beside it would
#: have looked like a choice and done nothing, because
#: :meth:`ColdGasSector.m_hi` reads its numbers from :class:`ColdGasParams` and
#: not from the registry function's defaults.  Putting the rows where the
#: numbers live is the only place selecting one of them can have an effect.
#:
#: **Which is right is still not settled, and this makes it selectable rather
#: than settling it.**  The shipped halo field is spherical-overdensity, so
#: FoF-SO is arguably the row this package should use; the slope differs by a
#: third, which is larger than either error bar, and choosing between them is a
#: decision about halo finders that nobody here has made.  What changes is that
#: it is now a decision a caller can *make* in one call rather than by editing
#: three defaults, and that the calibration record names the finder.
VILLAESCUSA18_ROWS = {
    # alpha = 0.24 +/- 0.05, M_0 = (4.3 +/- 1.1)e10, M_min = (2.0 +/- 0.6)e12
    "fof": dict(log10_m0=10.6335, alpha_tng=0.24, log10_m_min=12.3010),
    # alpha = 0.16, M_0 = 4.1e10, M_min = 2.4e12
    "fof_so": dict(log10_m0=10.6128, alpha_tng=0.16, log10_m_min=12.3802),
}


def villaescusa18_row(name: str) -> dict:
    """The Table 1 row for one halo finder, as keyword overrides."""
    key = str(name).lower().replace("-", "_")
    if key not in VILLAESCUSA18_ROWS:
        raise ValueError(
            f"unknown Villaescusa-Navarro row {name!r}; expected one of "
            f"{sorted(VILLAESCUSA18_ROWS)}.  'fof' is the shipped default; "
            f"'fof_so' is the row matching a spherical-overdensity halo "
            f"definition, which is what this package's field carries.")
    return dict(VILLAESCUSA18_ROWS[key])


# =========================================================================
# HI per galaxy: the xGASS gas-fraction scaling
# =========================================================================

#: Catinella et al. (2018), MNRAS 476, 875 (arXiv:1802.02373), **Table 1**,
#: transcribed: bin centre :math:`\lg M_*`, the weighted average of
#: :math:`\lg(M_{\rm HI}/M_*)` with its error, the weighted median, and the
#: number of galaxies.  Non-detections are at their upper limits in both
#: statistics.  xGASS is selected on stellar mass and redshift alone
#: (:math:`9 < \lg M_*/M_\odot < 11.5`, :math:`0.01 < z < 0.05`), so its
#: medians are :math:`P(M_{\rm HI}|M_*)` for centrals and satellites together,
#: which is the conditional this sector needs; stellar masses are MPA-JHU DR7,
#: Chabrier IMF, :math:`H_0 = 70`.
XGASS_TABLE1 = (
    # lg M*   <lg f>   err     median  N
    (9.14, -0.242, 0.053, -0.092, 113),
    (9.44, -0.459, 0.067, -0.320, 92),
    (9.74, -0.748, 0.069, -0.656, 96),
    (10.07, -0.869, 0.042, -0.854, 214),
    (10.34, -1.175, 0.037, -1.278, 191),
    (10.65, -1.231, 0.036, -1.223, 189),
    (10.95, -1.475, 0.033, -1.707, 196),
    (11.20, -1.589, 0.044, -1.785, 86),
)

#: The Hubble parameter xGASS's masses assume.  Both masses scale as
#: :math:`h^{-2}`, so the gas *fraction* is independent of it and only the
#: stellar-mass axis moves: :math:`2\lg(h/0.7) = -0.033` dex at Planck 2018.
XGASS_H = 0.7

#: How many of :data:`XGASS_TABLE1`'s bins the median relation is fitted to.
#: The last two medians, :math:`-1.707` and :math:`-1.785`, sit at the
#: survey's gas-fraction limit (:math:`M_{\rm HI}/M_* > 0.02` above
#: :math:`\lg M_* = 9.7`), so more than half of each bin is an upper limit and
#: its median is a limit rather than a measurement.
XGASS_FIT_BINS = 6

#: The mean interquartile width of :math:`\lg(M_{\rm HI}/M_*)` over the xGASS
#: bins, :math:`\bar\Delta = 0.96` dex (Catinella et al. 2018, Sec. 4, with
#: non-detections at their upper limits).
XGASS_IQR_DEX = 0.96

#: The gas-rich side's width, :math:`\sigma_\uparrow`: the standard deviation
#: of :math:`\lg(M_{\rm HI}/M_*)` about the HI gas-fraction sequence of
#: star-forming main-sequence galaxies, 0.36 dex (Janowiecki et al. 2020, MNRAS
#: 493, 1982, Table 1, 349 detections).  Above the median a galaxy is
#: star-forming, and that is the population whose scatter this measures.
SIGMA_FHI_UP_J20 = 0.36

#: The gas-poor side's width, :math:`\sigma_\downarrow`: whatever makes the
#: split normal's interquartile range equal xGASS's :data:`XGASS_IQR_DEX`
#: given :data:`SIGMA_FHI_UP_J20` -- solved once and held to that by a test
#: (see :func:`split_normal_iqr`).  Wide because the gas-poor side is where the
#: quenched galaxies are, and xGASS's non-detections at their upper limits
#: make it a lower bound on the true width.
SIGMA_FHI_DOWN_XGASS = 0.976444


def split_normal_quantile(p, s_up, s_dn):
    r"""Quantile of a split normal with mode zero.

    Width ``s_dn`` below the mode and ``s_up`` above, normalised so that the
    density is continuous: the mass below the mode is
    :math:`F_0 = \sigma_\downarrow/(\sigma_\uparrow + \sigma_\downarrow)`.
    """
    from jax.scipy.special import ndtri

    p = jnp.asarray(p)
    f0 = s_dn / (s_up + s_dn)
    lo = s_dn * ndtri(jnp.clip(p * (s_up + s_dn) / (2.0 * s_dn), 1e-300, 1.0))
    hi = s_up * ndtri(jnp.clip(0.5 + (p - f0) * (s_up + s_dn) / (2.0 * s_up),
                               0.0, 1.0 - 1e-16))
    return jnp.where(p <= f0, lo, hi)


def split_normal_iqr(s_up, s_dn):
    """The 75th minus the 25th percentile [dex]."""
    return (split_normal_quantile(0.75, s_up, s_dn)
            - split_normal_quantile(0.25, s_up, s_dn))


def split_normal_mode_minus_median(s_up, s_dn):
    """Mode minus median [dex]: positive when the gas-poor side is the wider."""
    return -split_normal_quantile(0.5, s_up, s_dn)


def split_normal_pdf(x, s_up, s_dn):
    """Density of a split normal with mode zero, per unit ``x``."""
    x = jnp.asarray(x)
    amp = 2.0 / (jnp.sqrt(2.0 * jnp.pi) * (s_up + s_dn))
    sig = jnp.where(x > 0.0, s_up, s_dn)
    return amp * jnp.exp(-0.5 * jnp.square(x / sig))


def split_normal_mean_over_median(s_up, s_dn):
    r""":math:`\langle 10^X\rangle / 10^{{\rm median}\,X}` for a split normal.

    With :math:`c = \ln 10` and the mode at zero,

    .. math::  \langle 10^X\rangle = \frac{2}{\sigma_\uparrow+\sigma_\downarrow}
               \left[\sigma_\downarrow e^{c^2\sigma_\downarrow^2/2}
               \Phi(-c\sigma_\downarrow) + \sigma_\uparrow
               e^{c^2\sigma_\uparrow^2/2}\Phi(c\sigma_\uparrow)\right],

    written through ``log_ndtr`` so the wide side's :math:`e^{c^2\sigma^2/2}`
    and its vanishing :math:`\Phi` never meet as a product of extremes.  2.07 at
    the defaults, against 3.8 for a symmetric normal of the same interquartile
    range: most of the difference between an HI budget that matches and one
    that is twice too high.
    """
    from jax.scipy.special import log_ndtr

    c = jnp.log(10.0)
    lower = s_dn * jnp.exp(0.5 * jnp.square(c * s_dn) + log_ndtr(-c * s_dn))
    upper = s_up * jnp.exp(0.5 * jnp.square(c * s_up) + log_ndtr(c * s_up))
    mean_over_mode = 2.0 / (s_up + s_dn) * (lower + upper)
    return mean_over_mode * jnp.power(
        10.0, split_normal_mode_minus_median(s_up, s_dn))


#: Where satellites' HI suppression starts, :math:`\lg M_h` [:math:`M_\odot/h`]:
#: the edge of Brown et al. (2017)'s pairs-and-small-groups bin, the
#: environment every other bin is compared with.
LG_MH_SAT_PIVOT = 12.0


def satellite_hi_offset(log10m_host, gamma):
    r"""Satellites' :math:`\Delta\lg M_{\rm HI}` in a host of mass
    :math:`M_h`: :math:`-\gamma\max(0, \lg M_h - 12)`.

    Brown et al. (2017) stack ALFALFA on SDSS satellites in Yang et al. group
    haloes and find, at :math:`M_* = 10^{10}`, a satellite in a pair or small
    group (:math:`\lg M_h < 12`) is 0.2 to 0.5 dex more HI-rich than in a
    medium or large group (12 to 14) and 0.8 dex more than in a cluster
    (above 14).  A line from the pivot through those bins' centres gives
    :math:`\gamma \simeq 0.32`, which also returns the groups' 0.16 and 0.48.
    """
    return -gamma * jnp.maximum(jnp.asarray(log10m_host) - LG_MH_SAT_PIVOT,
                                0.0)


def fit_xgass_medians(n_bins: int = XGASS_FIT_BINS):
    r"""Weighted least-squares line through the xGASS medians.

    :math:`{\rm median}\,\lg(M_{\rm HI}/M_*) = a + b\,(\lg M_* - 10)` on the
    first ``n_bins`` rows of :data:`XGASS_TABLE1`, weighted by the error of the
    average (the table gives none for the median).  The errors are scaled by
    :math:`\sqrt{\chi^2/\nu}` when that exceeds one -- it is 9.4 on the default
    six bins, because the medians scatter about a line by more than the
    averages' errors, and an unscaled width would be a prior claiming a
    precision the table does not have.

    Returns ``(a, b, sigma_a, sigma_b)``, with the stellar mass on xGASS's own
    :math:`h = 0.7` axis.  :class:`ColdGasParams`' defaults and Gaussians are
    this fit, rounded; a test holds them to it.
    """
    import numpy as np

    rows = np.asarray(XGASS_TABLE1[:n_bins], dtype=float)
    x = rows[:, 0] - 10.0
    y = rows[:, 3]
    w = 1.0 / rows[:, 2] ** 2
    a_mat = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(a_mat.T @ (a_mat * w[:, None]))
    p = cov @ (a_mat.T @ (w * y))
    chi2 = float(np.sum(w * (y - a_mat @ p) ** 2))
    scale = np.sqrt(max(chi2 / (len(x) - 2), 1.0))
    err = np.sqrt(np.diag(cov)) * scale
    return float(p[0]), float(p[1]), float(err[0]), float(err[1])


def lg_fhi_catinella18(lg_ms_msun, h, a, b):
    r"""Median :math:`\lg(M_{\rm HI}/M_*)` at a physical stellar mass.

    .. math::  \lg\frac{M_{\rm HI}}{M_*} = a + b\left[\lg M_*
               + 2\lg\frac{h}{0.7} - 10\right]

    The bracket puts a stellar mass on the cosmology's own :math:`h` onto the
    :math:`h = 0.7` axis the xGASS fit was made on.
    """
    lg70 = jnp.asarray(lg_ms_msun) + 2.0 * jnp.log10(h / XGASS_H)
    return a + b * (lg70 - 10.0)


def lg_mhi_catinella18(lg_ms_msun, h, a, b, dsat=0.0):
    r"""Median :math:`\lg M_{\rm HI}` [:math:`M_\odot`, physical] of a galaxy.

    .. math::  \lg M_{\rm HI} = \lg M_* + a + b\,(\lg M_* - 10)
               + \Delta_{\rm sat}

    xGASS's gas-fraction scaling (Catinella et al. 2018, Table 1) plus an
    additive offset, which :class:`ColdGasSector` uses for satellites'
    suppression in massive hosts (:func:`satellite_hi_offset`).  A median,
    because the relation is a
    regression in the logarithm -- the convention of
    :meth:`~ggah_mod.sectors.agn.AgnSector.black_hole_mass_function`, so the
    two mass functions are built the same way.
    """
    return (jnp.asarray(lg_ms_msun) + lg_fhi_catinella18(lg_ms_msun, h, a, b)
            + dsat)


#: The stellar-mass window ``catinella18`` integrates over [physical
#: :math:`\lg M_\odot`].  The floor is xGASS's own: below :math:`10^9` neither
#: the gas fraction nor -- measured -- the galaxy sector's stellar-mass function
#: is calibrated (the LS10 defaults are 4.55, 3.59 and 2.51 times GAMA's --
#: Baldry et al. 2012 -- at :math:`\lg M_* = 8.5`, 9 and 9.5, Paper I's 0.46,
#: 0.55 and 0.64 times).  The ceiling
#: is the AGN sector's, above xGASS's :math:`11.5` because the galaxies there
#: are few and gas-poor rather than absent.
LG_MSTAR_MIN_HI = 9.0
LG_MSTAR_MAX_HI = 12.5
#: Grid density, the AGN sector's (``agn.N_MSTAR_PER_DEX``).
N_MSTAR_PER_DEX_HI = 73

#: ``name -> median lg M_HI(lg M_*)``: the relations of a *galaxy's* stellar
#: mass.  Kept apart from :data:`HI_HALO`, whose relations take a halo mass,
#: because the two are not interchangeable arguments to one sector method.
HI_GALAXY = {
    "catinella18": lg_mhi_catinella18,
}


#: :math:`R_{200m}/R_{\rm vir}` at :math:`z = 0` for Planck 2018, the factor by
#: which :data:`C_HI_0` differs from the published 28.65.  The paper defines
#: :math:`r_s` on the virial radius; this package's field carries
#: :math:`R_{200m}`, and rather than evaluate a second radius the difference is
#: absorbed into the normalisation.  **That is exact at z = 0 only**: the ratio
#: is 1.1825 there, 1.0485 at z = 0.5, 1.0032 at z = 1 and 0.963 at z = 5, so
#: above z ~ 0.5 the absorbed value makes the profile more compact than the
#: paper intended -- by up to 18 per cent at z = 1.  Mass-independent at every
#: redshift (spread 2e-16), so a constant is the right *form*; only its value
#: is pinned to z = 0.
R200M_OVER_RVIR_Z0 = 1.1825

#: The published :math:`c_{\rm HI,0} = 28.65 \pm 1.76` (Padmanabhan, Refregier &
#: Amara 2017, Table 3), re-expressed on :math:`R_{200m}`.  See
#: :data:`R200M_OVER_RVIR_Z0`.
C_HI_0 = 28.65 * R200M_OVER_RVIR_Z0


def c_hi_padmanabhan(m, z, c_hi_0=C_HI_0):
    r""":math:`c_{\rm HI}(M, z)` -- Padmanabhan, Refregier & Amara (2017), Eq. (3).

    Restated by Padmanabhan, Maartens, Umeh & Camera (2023) as their Eq. (19).

    .. math::

        c_{\rm HI}(M, z) = c_{\rm HI,0}
            \left(\frac{M}{10^{11}h^{-1}M_\odot}\right)^{-0.109}
            \frac{4}{(1+z)^{\gamma}}, \qquad \gamma = 1.45

    **The factor 4 is part of the form, so** ``c_hi_0`` **is not the
    concentration.**  It is Macciò et al. (2007)'s dark-matter relation with a
    free normalisation, and at :math:`z = 0` the concentration at the pivot is
    :math:`4 c_{\rm HI,0}`.

    **The default is not the published 28.65.**  It is that times
    :data:`R200M_OVER_RVIR_Z0`, because the paper puts :math:`r_s` on the virial
    radius and this is evaluated on the field's :math:`R_{200m}`.

    :math:`-0.109` and :math:`\gamma = 1.45 \pm 0.04` are literals and not
    parameters, for the reason ``HARD_TO_SOFT`` is one: they are coefficients of
    one published fit, and freeing them separately would let a sampler build a
    relation nobody fitted.

    The pivot is taken in :math:`h^{-1}M_\odot`, the unit of this package's mass
    grid and of the same paper's :math:`M_{\rm HI}` relation (its Eq. 1).  Its
    Eq. (3) prints the pivot without the :math:`h`, and so does the 2023
    restatement; the two readings differ by :math:`h^{-0.109} = 1.044` at
    Planck 2018, inside ``c_hi_0``'s own six per cent.
    """
    m = jnp.atleast_1d(jnp.asarray(m))
    return (c_hi_0 * jnp.power(m / 1e11, -0.109)
            * 4.0 / jnp.power(1.0 + jnp.asarray(z), 1.45))


#: ``name -> M_HI(M_h)``.  Signatures differ: ``padmanabhan17`` needs a
#: cosmology and a redshift because its cutoffs are in virial velocity, and
#: ``villaescusa18`` cannot use one.  :class:`ColdGasSector` adapts.
HI_HALO = {
    "padmanabhan17": m_hi_padmanabhan17,
    "villaescusa18": m_hi_villaescusa18,
}

#: What each was fitted to, over what redshift range, and whether it can
#: respond to a cosmology at all.
#:
#: The ranges were prose until the sector check was written, and lifting them
#: out settled one of them.  ``villaescusa18`` is a fit to *one snapshot* -- the
#: notes column already said "one simulation, one redshift, one cosmology" --
#: so its range is the single point (0, 0), and every z above zero is an
#: extrapolation this now reports.  Recording it as a range would have been the
#: same error the whole registry exists to prevent.
HI_CALIBRATION = {
    "padmanabhan17": Calibration(
        fit="HI galaxy clustering and column densities, 21-cm intensity "
            "mapping, and DLAs, jointly",
        z_range=(0.0, 5.0),
        notes="form: Padmanabhan & Refregier (2017), MNRAS 464, 4008, Eq. (1); "
              "numbers, profile and concentration: Padmanabhan, Refregier & "
              "Amara (2017), MNRAS 469, 2323, Eqs. (1)-(4) and Table 3",
        cosmology_dependent=True),
    "villaescusa18": Calibration(
        fit="IllustrisTNG TNG100-1 at z = 0",
        z_range=(0.0, 0.0),
        selection="FoF haloes by default; see VILLAESCUSA18_ROWS for the "
                  "FoF-SO row, which is the one matching this package's "
                  "spherical-overdensity field",
        notes="one simulation, one redshift, one cosmology",
        cosmology_dependent=False),
    "catinella18": Calibration(
        fit="xGASS median HI gas fractions at fixed stellar mass, "
            "non-detections at their upper limits",
        z_range=(0.0, 0.05),
        selection="stellar-mass selected, 0.01 < z < 0.05, centrals and "
                  "satellites together",
        notes="Catinella et al. (2018), MNRAS 476, 875, Table 1; the gas "
              "fraction takes no cosmology, the halo-total HI it sums to "
              "inherits the galaxy sector's through the stellar-mass "
              "functions",
        cosmology_dependent=False,
        mstar_range=(9.0, 11.5)),
}


def make_hi_halo(name: str):
    """Look up an HI-halo mass relation by name."""
    key = str(name).lower()
    if key not in HI_HALO:
        raise ValueError(f"unknown HI-halo relation {name!r}; expected one of "
                         f"{sorted(HI_HALO)}")
    return HI_HALO[key]


#: Every relation :class:`ColdGasSector` accepts: the halo-total ones and the
#: per-galaxy one.
RELATIONS = {**HI_HALO, **HI_GALAXY}


# =========================================================================
# The sector
# =========================================================================

@sector_params
class ColdGasParams(SectorParams):
    """The neutral-gas sector's free parameters, with bounds and reasons.

    Named for ``padmanabhan17``, which is the relation that has free parameters
    worth varying; ``villaescusa18``'s three are carried too, so switching
    relation does not change which object holds the parameters.
    """

    # -- padmanabhan17 --------------------------------------------------------
    alpha_hi: float = 0.09
    beta_hi: float = -0.58
    log10_vc0: float = 1.56
    log10_vc1: float = 4.39
    # -- villaescusa18 --------------------------------------------------------
    log10_m0: float = 10.6335
    alpha_tng: float = 0.24
    log10_m_min: float = 12.3010
    # -- catinella18 ----------------------------------------------------------
    lg_fhi_10: float = -0.826
    dlg_fhi_dlgms: float = -0.802
    sigma_fhi_up: float = SIGMA_FHI_UP_J20
    sigma_fhi_down: float = SIGMA_FHI_DOWN_XGASS
    gamma_fhi_sat: float = 0.32
    # -- shared ---------------------------------------------------------------
    r_mol: float = 0.3
    # -- the profile ----------------------------------------------------------
    c_hi_0: float = C_HI_0

    _STATIC = ()

    _PARAMS = {
        "alpha_hi": Param(
            0.09, (0.05, 0.5), Gaussian(0.09, 0.01), "",
            "The HI fraction relative to cosmic, and the one number that sets "
            "the neutral-gas budget: Omega_HI is exactly linear in it. The "
            "value and Gaussian are Padmanabhan, Refregier & Amara (2017), "
            "Table 3; the bounds are their flat prior, the same as "
            "Padmanabhan & Refregier (2017)'s. The earlier paper's 0.17 +/- "
            "0.02 puts Omega_HI at 1.76 times Dev et al. (2024); this puts it "
            "at 0.94. It is also the profile's amplitude: the profile adds no "
            "normalisation of its own, which would be degenerate with this.",
            kind="prior"),
        "beta_hi": Param(
            -0.58, (-1.0, 3.0), Gaussian(-0.58, 0.06), "",
            "Excess logarithmic slope at the 1e11 Msun/h pivot. Padmanabhan, "
            "Refregier & Amara (2017), Table 3, with their flat prior as the "
            "bounds; abundance matching independently favours beta < 0, so a "
            "positive value is allowed but is a claim.",
            kind="prior"),
        "log10_vc0": Param(
            1.56, (1.30, 1.90), Gaussian(1.56, 0.04), "log10 km/s",
            "Lower virial-velocity cutoff, Padmanabhan, Refregier & Amara "
            "(2017), Table 3. Simulations find haloes below ~30 km/s host no "
            "HI (Pontzen et al. 2008), which is the middle of this range; the "
            "bounds are the published flat prior.",
            kind="prior"),
        "log10_vc1": Param(
            4.39, (2.1, 6.5), Flat(), "log10 km/s",
            "Upper cutoff, from Padmanabhan & Refregier (2017) alone: the "
            "later fit that supplies the other numbers has none. Kept because "
            "it is inert rather than wrong -- 10^4.39 = 24500 km/s against "
            "virial velocities of 29 to 2900 km/s on the shipped grid, so it "
            "suppresses M_HI by at most 1.6e-3 at the top of the grid and "
            "Omega_HI by under 3e-8 -- and removing it would remove a tested "
            "branch for no number. Flat rather than Gaussian: a 0.75-wide "
            "Gaussian on an inactive parameter is a prior pretending to be a "
            "measurement.",
            kind="prior"),
        "log10_m0": Param(
            10.6335, (9.0, 12.0), Gaussian(10.6335, 0.11), "log10 Msun/h",
            "TNG100-1 amplitude, 4.3e10 +/- 1.1e10 Msun/h.  The bounds span "
            "the range over which the relation was measured.",
            kind="prior"),
        "alpha_tng": Param(
            0.24, (0.0, 1.0), Gaussian(0.24, 0.05), "",
            "TNG100-1 slope.  Zero is a flat relation and one is linear; "
            "outside that the fit is extrapolated beyond anything measured.",
            kind="prior"),
        "log10_m_min": Param(
            12.3010, (11.0, 13.5), Gaussian(12.3010, 0.13), "log10 Msun/h",
            "TNG100-1 low-mass cutoff, 2e12 +/- 0.6e12 Msun/h.",
            kind="prior"),
        "lg_fhi_10": Param(
            -0.826, (-2.0, 0.5), Gaussian(-0.826, 0.059), "dex",
            "Median log10(M_HI/M*) of a galaxy of 10^10 Msun (on xGASS's "
            "h = 0.7 axis): the weighted least-squares line through the xGASS "
            "medians (Catinella et al. 2018, Table 1) in the six bins below "
            "lg M* = 10.9, where the median is a detection. The width is the "
            "fit's error scaled by sqrt(chi2/dof) = 3.1, because the medians "
            "scatter about the line by more than the table's errors; see "
            "fit_xgass_medians, which a test holds these numbers to. The "
            "bounds span every bin's median and average with room either way.",
            kind="prior"),
        "dlg_fhi_dlgms": Param(
            -0.802, (-2.0, 0.5), Gaussian(-0.802, 0.114), "",
            "Slope of the median log gas fraction in log M*, the same fit. "
            "Negative: bigger galaxies are gas-poorer. The weighted averages of "
            "the log give -0.67, inside this width, and the medians are used "
            "because the average carries every non-detection at its upper "
            "limit at full weight.",
            kind="prior"),
        "sigma_fhi_up": Param(
            SIGMA_FHI_UP_J20, (0.1, 1.0), Flat(), "dex",
            "Width of the gas-rich side of the split normal in log M_HI at "
            "fixed M*: the standard deviation of the star-forming main "
            "sequence's HI gas fractions about their own sequence, 0.36 dex "
            "(Janowiecki et al. 2020, Table 1). Measured on detections only, "
            "which is right for this side -- the gas-rich galaxies are the "
            "detected ones. It is what sets the knee of the HI mass function: "
            "a symmetric normal of xGASS's full interquartile range put 25 to "
            "50 times too many galaxies at 10^10.7. Flat, because the paper "
            "gives the width no error.",
            kind="prior"),
        "sigma_fhi_down": Param(
            SIGMA_FHI_DOWN_XGASS, (0.2, 2.0), Flat(), "dex",
            "Width of the gas-poor side: the value that makes the split "
            "normal's interquartile range xGASS's 0.96 dex (Catinella et al. "
            "2018, Sec. 4) given sigma_fhi_up, solved once and held to it by a "
            "test. Wide because this is where the quenched galaxies are; a "
            "lower bound, since xGASS's interquartile range carries "
            "non-detections at their upper limits. The median stays the "
            "xGASS median whatever the two widths are -- the mode moves.",
            kind="prior"),
        "gamma_fhi_sat": Param(
            0.32, (0.0, 1.0), Flat(), "dex/dex",
            "Satellites' HI suppression per dex of host halo mass above "
            "10^12 Msun/h, on top of the xGASS median (which averages "
            "centrals and satellites). Brown et al. (2017): a 10^10 Msun "
            "satellite in a pair or small group is 0.2 to 0.5 dex richer than "
            "in a medium or large group and 0.8 dex richer than in a cluster "
            "above 10^14; 0.32 is the line through those bins from the pivot, "
            "read here rather than fitted there, hence Flat. Zero restores no "
            "suppression, which drives b_HI to 1.08 -- satellites in massive "
            "haloes would then hold 35 per cent of the HI.",
            kind="prior"),
        "r_mol": Param(
            0.3, (0.0, 2.0), Flat(), "",
            "M_H2/M_HI, held constant.  Anchored on the local ratio: "
            "Fukugita & Peebles (2004) give molecular / (HI + He I) = 0.26, "
            "and the 21-cm and CO compilations in Peroux & Howk (2020) put "
            "Omega_H2/Omega_HI near 0.3 at z = 0.  Zero is a real limit -- "
            "atomic gas only -- and the upper bound is where the molecular "
            "phase would dominate, which is observed at high redshift and not "
            "here.  It is a constant because the pressure-based prescription "
            "that makes it vary (Blitz & Rosolowsky 2006) needs a disc model "
            "this package does not have.",
            kind="physical"),
        "c_hi_0": Param(
            C_HI_0, (20.0 * R200M_OVER_RVIR_Z0, 400.0 * R200M_OVER_RVIR_Z0),
            Gaussian(C_HI_0, 1.76 * R200M_OVER_RVIR_Z0), "",
            "Normalisation of the HI concentration of the exponential profile, "
            "Padmanabhan, Refregier & Amara (2017), Eqs. (2)-(3) and Table 3. "
            "NOT the concentration: the relation carries a factor 4. NOT the "
            "published 28.65 +/- 1.76 either: that is on the virial radius, "
            "and this is on the field's R_200m, so the value, the width and "
            "the bounds are all scaled by R200M_OVER_RVIR_Z0 -- exact at z = 0 "
            "and drifting above it; see that constant. The bounds are the "
            "paper's flat prior, [20, 400], so scaled. The only profile "
            "parameter the neutral gas has. The prior's floor is not a "
            "numerical limit: the closed-form transform is the untruncated "
            "exponential, and against the profile cut at the halo it is good "
            "to 9e-10 at the floor and 4e-15 at the ceiling. It fails only "
            "well below the floor -- 8.7e-4 at 10, 2.8e-2 at 6, where the HI is "
            "as extended as the dark matter -- which is the paper's own "
            "statement that its closed form holds for c_HI > 10.",
            kind="prior"),
    }


class ColdGasSector:
    r"""Neutral gas as a :class:`~ggah_mod.sectors.protocol.Sector`.

    Parameters
    ----------
    relation : str
        One of :data:`RELATIONS`.  **No default**: the three answer different
        questions -- HI per galaxy at :math:`z \simeq 0`, HI per halo for
        intensity mapping, HI per halo in one simulation -- and choosing one
        silently would put an unasked-for model into a census row.  The same
        rule :class:`~ggah_mod.sectors.galaxies.GalaxySector` applies to its
        SHMR.
    galaxies : GalaxySector, optional
        Required by ``catinella18`` and refused by the other two.  Held rather
        than borrowed, for the reason
        :class:`~ggah_mod.sectors.agn.AgnSector` holds its galaxies: the stellar
        masses the HI is set by must be the ones the galaxy tracer uses, and
        layer 4 refuses a spectrum whose galaxy sector is a different instance.
        It must be a threshold occupation, the only kind with a conditional
        stellar-mass function.
    lg_mstar_min : float
        ``catinella18`` only: the lowest physical :math:`\lg M_*` that holds HI,
        :data:`LG_MSTAR_MIN_HI` by default.  Static, because it sets the grid's
        shape.  The HI of galaxies below it is in neither
        :math:`\Omega_{\rm HI}` nor :math:`f_{\rm cold}`.
    calibration : {"strict", "warn", "off"}
    """

    name = "coldgas"
    differentiable = True

    #: ``hi`` is atomic hydrogen alone; ``h2`` is molecular; ``mass`` is the
    #: neutral gas the census counts -- both, with helium.  Three names because
    #: the literature reports all three and they differ by factors of order
    #: one, which is exactly the size of the disagreements being tested.
    VIEWS = ("hi", "h2", "mass")

    def __init__(self, relation: str, *, galaxies=None,
                 lg_mstar_min: float = LG_MSTAR_MIN_HI,
                 calibration: str = "warn"):
        key = str(relation).lower()
        if key not in RELATIONS:
            raise ValueError(f"unknown HI relation {relation!r}; expected "
                             f"one of {sorted(RELATIONS)}")
        if key in HI_GALAXY:
            if galaxies is None:
                raise ValueError(
                    f"relation {key!r} sets each galaxy's HI by its own stellar "
                    f"mass, so it needs the galaxy sector those masses come "
                    f"from: ColdGasSector({key!r}, galaxies=GalaxySector(...)).")
            if not getattr(galaxies, "is_threshold", False):
                raise ValueError(
                    f"relation {key!r} integrates over the galaxy sector's "
                    f"conditional stellar-mass functions, which only a "
                    f"threshold occupation has; "
                    f"{getattr(galaxies, 'model', galaxies)!r} is not one.")
        elif galaxies is not None:
            raise ValueError(
                f"relation {key!r} is halo-total: it maps a halo mass to the "
                f"HI of every galaxy in it and reads no stellar mass, so a "
                f"galaxy sector here would be carried and never used.  Drop "
                f"`galaxies=`, or use 'catinella18'.")
        self.relation = key
        self._fn = RELATIONS[key]
        self.galaxies = galaxies
        self.lg_mstar_min = float(lg_mstar_min)
        #: The physical stellar-mass grid ``catinella18`` integrates on, shared
        #: by centrals and satellites so that its HI mass function is the
        #: stellar-mass function convolved and nothing else.
        self.lg_ms = None
        if key in HI_GALAXY:
            n = int(round((LG_MSTAR_MAX_HI - self.lg_mstar_min)
                          * N_MSTAR_PER_DEX_HI)) + 1
            self.lg_ms = jnp.linspace(self.lg_mstar_min, LG_MSTAR_MAX_HI, n)
        #: Checked in `weights`, where a `HaloField` supplies the redshift.
        #: `villaescusa18` is fitted at z = 0 alone and `catinella18` at
        #: z < 0.05, so this is not a formality for either.
        self.calibration = str(calibration)

    def __repr__(self) -> str:                           # pragma: no cover
        return f"ColdGasSector(relation={self.relation!r})"

    @property
    def per_galaxy(self) -> bool:
        """Whether the HI is set galaxy by galaxy (``catinella18``) rather
        than halo by halo."""
        return self.relation in HI_GALAXY

    # ------------------------------------------------------ HI in galaxies
    def _galaxy_params(self, galaxies_params):
        if galaxies_params is None:
            raise ValueError(
                f"relation {self.relation!r} needs the galaxy sector's "
                f"parameters as well as its own: each galaxy's HI is set by its "
                f"stellar mass, which the galaxy sector computes at those "
                f"parameters.  Pass `galaxies_params=`, or put a 'galaxies' "
                f"block in the spectrum's parameters.")
        return galaxies_params

    def _csmfs(self, field, galaxies_params):
        """Centrals and satellites per host per dex, ``(N_*, N_M)`` each, on
        :attr:`lg_ms`."""
        gp = self._galaxy_params(galaxies_params)
        _, dn_cen = self.galaxies.central_csmf_msun(field, gp,
                                                    lg_ms_msun=self.lg_ms)
        _, dn_sat = self.galaxies.satellite_csmf_msun(field, gp,
                                                      lg_ms_msun=self.lg_ms)
        return dn_cen, dn_sat

    def lg_mhi_median(self, field, params: ColdGasParams):
        r"""Median :math:`\lg M_{\rm HI}` [:math:`M_\odot/h`] of a galaxy at
        each node of :attr:`lg_ms`, from :func:`lg_mhi_catinella18`.  A
        satellite's is this plus :func:`satellite_hi_offset` of its host."""
        p = params if params is not None else ColdGasParams()
        h = field.cosmo.h
        lg = lg_mhi_catinella18(self.lg_ms, h, p.lg_fhi_10, p.dlg_fhi_dlgms)
        return lg + jnp.log10(h)

    @staticmethod
    def mean_over_median(params: ColdGasParams):
        r"""The split normal's mean over its median,
        :func:`split_normal_mean_over_median` -- 2.07 at the defaults."""
        p = params if params is not None else ColdGasParams()
        return split_normal_mean_over_median(p.sigma_fhi_up, p.sigma_fhi_down)

    def satellite_offset(self, field, params: ColdGasParams):
        r""":func:`satellite_hi_offset` on the field's host masses, ``(N_M,)``."""
        p = params if params is not None else ColdGasParams()
        return satellite_hi_offset(jnp.log10(field.m), p.gamma_fhi_sat)

    def m_hi_split(self, field, params: ColdGasParams, galaxies_params=None):
        r"""``(M_HI of the central, M_HI of the satellites)`` per halo
        [:math:`M_\odot/h`], ``catinella18`` only.

        .. math::  \langle M_{\rm HI}\rangle_x(M_h) = \int \dd\lg M_*\,
                   \frac{\dd N_x}{\dd\lg M_*}(M_*|M_h)\,
                   \langle M_{\rm HI}|M_*\rangle_x

        with the mean, not the median, because a halo holds the HI of all its
        galaxies and what they sum to is the mean.  The satellites' mean
        carries their host's suppression, :meth:`satellite_offset`.
        """
        if not self.per_galaxy:
            raise ValueError(
                f"relation {self.relation!r} is halo-total and does not "
                f"divide a halo's HI between its galaxies")
        p = params if params is not None else ColdGasParams()
        dn_cen, dn_sat = self._csmfs(field, galaxies_params)
        mean = (jnp.power(10.0, self.lg_mhi_median(field, p))
                * self.mean_over_median(p))
        m_cen = jnp.trapezoid(dn_cen * mean[:, None], self.lg_ms, axis=0)
        m_sat = (jnp.trapezoid(dn_sat * mean[:, None], self.lg_ms, axis=0)
                 * jnp.power(10.0, self.satellite_offset(field, p)))
        return m_cen, m_sat

    # ------------------------------------------------------------------ masses
    def m_hi(self, field, params: ColdGasParams, galaxies_params=None):
        """:math:`M_{\\rm HI}(M)` on the field's mass grid [Msun/h].

        Per halo for every relation: ``catinella18`` sums its central and
        satellites, and the halo-total relations never read
        ``galaxies_params``.
        """
        p = params if params is not None else ColdGasParams()
        if self.per_galaxy:
            m_cen, m_sat = self.m_hi_split(field, p, galaxies_params)
            return m_cen + m_sat
        if self.relation == "padmanabhan17":
            return m_hi_padmanabhan17(
                field.m, field.z, field.cosmo, alpha=p.alpha_hi,
                beta=p.beta_hi, log10_vc0=p.log10_vc0,
                log10_vc1=p.log10_vc1, mdef=field.mdef)
        return m_hi_villaescusa18(
            field.m, log10_m0=p.log10_m0, alpha=p.alpha_tng,
            log10_m_min=p.log10_m_min)

    def m_h2(self, field, params: ColdGasParams, galaxies_params=None):
        r""":math:`M_{\rm H_2} = r_{\rm mol} M_{\rm HI}`."""
        p = params if params is not None else ColdGasParams()
        return p.r_mol * self.m_hi(field, params, galaxies_params)

    def m_neutral(self, field, params: ColdGasParams, galaxies_params=None):
        r""":math:`(1-Y)^{-1}(M_{\rm HI} + M_{\rm H_2})` -- what the census counts.

        The helium goes on **both** phases, which is the convention Dev et al.
        (2024) use for ``Omega_neutral gas`` and which Fukugita & Peebles
        (2004) row 3.9 already contains for the atomic part.
        """
        p = params if params is not None else ColdGasParams()
        return (helium_correction() * (1.0 + p.r_mol)
                * self.m_hi(field, params, galaxies_params))

    def f_cold(self, field, params: ColdGasParams, galaxies_params=None):
        r""":math:`M_{\rm neutral}/M`, ready for
        :class:`~ggah_mod.sectors.matter.BaryonSplit`."""
        return self.m_neutral(field, params, galaxies_params) / field.m

    # ------------------------------------------------------- what 21-cm sees
    def omega_hi(self, field, params: ColdGasParams, galaxies_params=None):
        r""":math:`\Omega_{\rm HI} = \rho_{\rm HI}/\rho_{c,0}`.

        The same integral :func:`~ggah_mod.sectors.census.census` does for the
        neutral phase, restricted to atomic hydrogen -- so it is comparable
        with the 21-cm and damped-Lyman-:math:`\alpha` compilations directly,
        without the helium and molecular corrections that make
        :math:`\Omega_{\rm neutral}` a different number by a factor 1.7.
        """
        from ..cosmology import constants as C

        return field.integrate(
            field.dndm * self.m_hi(field, params, galaxies_params)) \
            / C.RHO_CRIT0

    def bias_hi(self, field, params: ColdGasParams, galaxies_params=None):
        r""":math:`b_{\rm HI} = \int b\,M_{\rm HI}\,dn/dM / \int M_{\rm HI}\,dn/dM`.

        Mass-weighted, not number-weighted, because 21-cm emission is: an
        intensity map sums flux, and every halo contributes in proportion to
        how much HI it holds.  :meth:`~ggah_mod.halos.field.HaloField.effective_bias`
        takes the weight, so the choice is visible at the call site rather than
        buried in a second implementation.
        """
        return field.effective_bias(self.m_hi(field, params, galaxies_params))

    def brightness_temperature(self, field, params: ColdGasParams,
                               galaxies_params=None):
        r""":math:`\bar T_b` [mK], from this sector's own :math:`\Omega_{\rm HI}`."""
        return brightness_temperature(
            field.z, field.cosmo,
            self.omega_hi(field, params, galaxies_params))

    def hi_mass_function(self, field, log10_mhi, params: ColdGasParams,
                         galaxies_params=None):
        r""":math:`\dd n/\dd\lg M_{\rm HI}` per **galaxy**, centrals and
        satellites apart [:math:`(h^{-1}{\rm Mpc})^{-3}\,{\rm dex}^{-1}`].

        .. math::

            \Phi_x(M_{\rm HI}) = \int \dd M\,\frac{\dd n}{\dd M}
                \int \dd\lg M_*\,\frac{\dd N_x}{\dd\lg M_*}(M_*|M)\,
                \mathcal{S}\!\left(\lg M_{\rm HI} - \mu(M_*)
                - \Delta_x(M)\right),

        with :math:`\mu` the median of :func:`lg_mhi_catinella18`,
        :math:`\mathcal{S}` the split normal of widths
        :attr:`ColdGasParams.sigma_fhi_up` and ``sigma_fhi_down`` centred on
        that median, :math:`\Delta_{\rm cen} = 0` and
        :math:`\Delta_{\rm sat}` the host's :meth:`satellite_offset`.  For the
        centrals the halo integral is done first, which makes theirs exactly the
        galaxy sector's stellar-mass function convolved.  ``log10_mhi`` is in :math:`M_\odot/h`, the package's
        mass unit.  Returns ``(phi_cen, phi_sat)``.

        This is the object ALFALFA and FASHI measure, which the halo-total
        relations cannot supply: they sum a group's galaxies before any
        comparison can be made, and have no knee.  Here the knee is the
        stellar-mass function's, carried through the gas fraction.  Its first
        moment is :meth:`omega_hi` times :math:`\rho_{c,0}`, to quadrature.

        Two things set it that are not this sector's.  Below the floor
        :attr:`lg_mstar_min` there are no galaxies, so the function falls
        away below about :math:`10^9\,M_\odot` of HI for want of dwarfs, not
        for want of gas.  And above the floor it is only as good as the galaxy
        sector's stellar-mass function, which is not fitted below
        :math:`10^{10}\,M_\odot`.
        """
        if not self.per_galaxy:
            raise ValueError(
                f"relation {self.relation!r} is halo-total: it gives the HI of "
                f"every galaxy in a halo summed, which is not the per-galaxy "
                f"HI mass function ALFALFA or FASHI measure.  Use "
                f"ColdGasSector('catinella18', galaxies=...).")
        p = params if params is not None else ColdGasParams()
        dn_cen, dn_sat = self._csmfs(field, galaxies_params)
        s_up, s_dn = p.sigma_fhi_up, p.sigma_fhi_down
        mode = (self.lg_mhi_median(field, p)
                + split_normal_mode_minus_median(s_up, s_dn))          # (N_*,)
        lg = jnp.asarray(log10_mhi)
        # Centrals: the halo integral first, so the stellar-mass function is
        # the galaxy sector's own.
        phi_cen_ms = field.integrate(field.dndm[None, :] * dn_cen, axis=-1)
        kern = split_normal_pdf(lg[:, None] - mode[None, :], s_up, s_dn)
        phi_cen = jnp.trapezoid(kern * phi_cen_ms[None, :], self.lg_ms, axis=1)
        # Satellites: the host's suppression shifts each host's column, so the
        # kernel carries the host axis too -- (N_HI, N_*, N_M).
        off = self.satellite_offset(field, p)
        x = lg[:, None, None] - mode[None, :, None] - off[None, None, :]
        per_host = jnp.trapezoid(split_normal_pdf(x, s_up, s_dn)
                                 * dn_sat[None, :, :], self.lg_ms, axis=1)
        phi_sat = field.integrate(field.dndm[None, :] * per_host, axis=-1)
        return phi_cen, phi_sat

    # ----------------------------------------------------------------- profile
    def c_hi(self, field, params: ColdGasParams):
        """:math:`c_{\\rm HI}(M, z)` on the field's mass grid."""
        if self.per_galaxy:
            raise ValueError(
                f"relation {self.relation!r} has no HI concentration: a "
                f"central's HI is a point mass and a satellite's follows the "
                f"satellites")
        p = params if params is not None else ColdGasParams()
        return c_hi_padmanabhan(field.m, field.z, c_hi_0=p.c_hi_0)

    def u_k(self, field, params: ColdGasParams, galaxies_params=None):
        r"""The neutral gas's normalised transform, :math:`(N_k, N_M)`.

        **Halo-total relations**: the exponential of Padmanabhan, Refregier &
        Amara (2017), Eq. (2) (Padmanabhan, Maartens, Umeh & Camera 2023,
        Eq. 18), in closed form:

        .. math::  \tilde u(k|M) = \left[1 + (k\,r_s)^2\right]^{-2},
                   \qquad r_s = R_\Delta / c_{\rm HI}(M, z).

        **catinella18**: the central's HI at the centre and the satellites' on
        the galaxy sector's satellite profile, mass-weighted,

        .. math::  \tilde u(k|M) = \frac{M_{\rm HI}^{\rm cen}
                   + M_{\rm HI}^{\rm sat}\,\tilde u_{\rm sat}(k|M)}
                   {M_{\rm HI}^{\rm cen} + M_{\rm HI}^{\rm sat}},

        so that ``amp * u`` is the central as a point plus the satellites as a
        profile, exactly the stars' two terms, and the matter field's single
        ``u_cold`` hook carries it without a second neutral-gas fraction.  A
        halo holding no galaxy above :attr:`lg_mstar_min` gets
        :math:`\tilde u = 1` through a guarded denominator, so its gradient is
        finite rather than :math:`0/0`.

        :math:`\tilde u(0) = 1` exactly in both cases, which is why no amplitude
        appears: see the module docstring.
        """
        if self.per_galaxy:
            m_cen, m_sat = self.m_hi_split(field, params, galaxies_params)
            u_sat = self.galaxies.satellite_uk(
                field, self._galaxy_params(galaxies_params))
            total = m_cen + m_sat
            held = total > 0.0
            safe = jnp.where(held, total, 1.0)
            mix = (m_cen[None, :] + m_sat[None, :] * u_sat) / safe[None, :]
            return jnp.where(held[None, :], mix, 1.0)
        r_s = (jnp.atleast_1d(jnp.asarray(field.r_delta))
               / self.c_hi(field, params))
        x = jnp.asarray(field.k)[:, None] * r_s[None, :]
        return 1.0 / jnp.square(1.0 + jnp.square(x))

    # ----------------------------------------------------------------- weights
    def weights(self, field, params: ColdGasParams,
                view: str = "mass", galaxies_params=None) -> TracerWeights:
        r"""One of :attr:`VIEWS`, as :class:`TracerWeights`.

        Continuous, on :meth:`u_k`.  The amplitude is the view's mass and the
        transform is normalised, so :math:`\Omega` -- which reads
        :math:`k\to0` -- is untouched by the profile, and clustering is what it
        changes.
        """
        if view not in self.VIEWS:
            raise ValueError(f"unknown cold-gas view {view!r}; expected one of "
                             f"{sorted(self.VIEWS)}")
        check_sector_calibration(
            "HI-galaxy relation" if self.per_galaxy else "HI-halo relation",
            self.relation, HI_CALIBRATION, field.z, policy=self.calibration)
        if self.per_galaxy:
            check_sector_range("HI-galaxy relation", self.relation,
                               HI_CALIBRATION,
                               jnp.asarray([self.lg_mstar_min]),
                               policy=self.calibration)
        amp = {"hi": self.m_hi, "h2": self.m_h2,
               "mass": self.m_neutral}[view](field, params, galaxies_params)
        u = self.u_k(field, params, galaxies_params)
        return TracerWeights(
            w_point=None, w_extended=amp[None, :] * u,
            norm=jnp.asarray(1.0), discrete=False, bias_weight=None,
            name=f"coldgas:{view}")


# =========================================================================
# What 21-cm experiments measure
# =========================================================================

#: :math:`\bar T_b = 189\,h\,\Omega_{\rm HI}(1+z)^2/E(z)` mK.
#:
#: The standard collapse of the 21-cm emission coefficients, as quoted
#: throughout the intensity-mapping literature.
#:
#: Einstein A of the 21-cm hyperfine transition [s^-1].
A_10_HZ = 2.85e-15

#: Rest frequency of the 21-cm line [Hz].
NU_21_HZ = 1420.405751768e6


def _t_b_coefficient_mk() -> float:
    r"""Derive :data:`T_B_COEF_MK` from :math:`A_{10}`, :math:`\nu_{21}` and
    the fundamental constants.

    In the optically thin, high-spin-temperature limit the mean 21-cm
    brightness temperature of a homogeneous neutral-hydrogen distribution is

    .. math::

        \bar T_b(z) = \frac{3}{32\pi}\,
            \frac{h_{\rm P}c^3A_{10}}{k_{\rm B}\nu_{21}^2}\,
            \frac{n_{\rm HI}(z)}{(1+z)\,H(z)}

    with :math:`n_{\rm HI}` the **proper** number density.  Writing that as
    :math:`\Omega_{\rm HI}\rho_{c,0}(1+z)^3/m_{\rm H}` -- so
    :math:`\Omega_{\rm HI}` is the *comoving* density parameter, which is the
    convention the census reports it in -- and using
    :math:`\rho_{c,0} = 3H_0^2/8\pi G` leaves

    .. math::

        \bar T_b(z) = C\,h\,\Omega_{\rm HI}(z)\,\frac{(1+z)^2}{E(z)},
        \qquad C = \frac{3}{32\pi}
        \frac{h_{\rm P}c^3A_{10}}{k_{\rm B}\nu_{21}^2}
        \frac{\rho_{c,0}(h{=}1)}{m_{\rm H}H_0(h{=}1)}

    which is where the single power of :math:`h` and the :math:`(1+z)^2` come
    from: one factor of :math:`(1+z)^3` from the proper density, one divided out
    by the velocity gradient, and :math:`\rho_{c,0}/H_0 \propto H_0 \propto h`.

    **This replaces a number that was recalled rather than read.**  The constant
    was 189.0 with a docstring saying so and asking to be derived or verified
    before anything computed from it was published.  Derived, it is
    **188.78 mK** -- 0.115 per cent from the remembered value, so the memory was
    sound and the convention with it: the :math:`h` and the :math:`(1+z)^2` are
    now consequences rather than assumptions.  The literature quotes this in at
    least three normalisations, and a stray :math:`h` here is a 33 per cent
    error that reads as a calibration difference, which is why deriving it was
    worth more than citing it.  ``PLAN.md`` item **G3**.
    """
    import math

    h_p = 6.62607015e-34                    # J s, exact since 2019
    c_m_s = C.C_KM_S * 1e3                  # m/s, exact
    k_b = 1.380649e-23                      # J/K, exact since 2019
    g_si = 6.67430e-11                      # m^3 / kg / s^2
    m_h_kg = 1.6735575e-27                  # hydrogen atom
    h0_h1 = 1e5 / (C.MPC_CM * 1e-2)         # s^-1 at h = 1

    pref = 3.0 / (32.0 * math.pi) * h_p * c_m_s ** 3 * A_10_HZ / (
        k_b * NU_21_HZ ** 2)
    rho_c0 = 3.0 * h0_h1 ** 2 / (8.0 * math.pi * g_si)
    return 1e3 * pref * rho_c0 / (m_h_kg * h0_h1)


#: :math:`\bar T_b = C\,h\,\Omega_{\rm HI}(1+z)^2/E(z)` [mK], with
#: :math:`\Omega_{\rm HI}` comoving.  Derived, not quoted -- see
#: :func:`_t_b_coefficient_mk` for the derivation and for the 0.115 per cent it
#: sits from the value this constant used to hold.
T_B_COEF_MK = _t_b_coefficient_mk()


def brightness_temperature(z, cosmo, omega_hi):
    r"""Mean 21-cm brightness temperature :math:`\bar T_b(z)` [mK].

    .. math::  \bar T_b = C\,h\,\frac{(1+z)^2}{E(z)}\,\Omega_{\rm HI}(z)

    with :math:`C` = :data:`T_B_COEF_MK`, derived rather than quoted.

    What an intensity-mapping experiment measures, up to the bias: the
    observable is :math:`\bar T_b b_{\rm HI}`, and the two are degenerate in the
    amplitude of a 21-cm power spectrum.  :meth:`ColdGasSector.bias_hi` supplies
    the other half, and the census supplies :math:`\Omega_{\rm HI}`, so the
    combination is predicted rather than fitted.
    """
    from ..cosmology.background import hubble_e

    z = jnp.asarray(z)
    return (T_B_COEF_MK * cosmo.h * jnp.asarray(omega_hi)
            * (1.0 + z) ** 2 / hubble_e(z, cosmo))
