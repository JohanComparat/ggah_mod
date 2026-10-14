r"""Galaxy X-ray binaries, and the stacked X-ray luminosity of galaxies at fixed stellar mass.

This module takes a stellar mass, a star-formation rate and a redshift, and
returns the 2--10 keV luminosity of a galaxy's X-ray binaries (XRBs); with the
AGN sector it returns the mean X-ray luminosity of the galaxies of one stellar
mass, binaries and nuclei together, which is what a stack of the X-ray events
around a galaxy sample measures.

The binaries follow the global relation of Lehmer et al. (2016, Eq. 15,
Table 3, the fit to their local sample plus the CDF-S stacks in the 2--10 keV
band):

.. math::

    L_X^{\rm XRB} = \alpha_0\,(1+z)^{\gamma}\,M_*
                  + \beta_0\,(1+z)^{\delta}\,{\rm SFR},

with :math:`\log\alpha_0 = 29.37 \pm 0.15` [erg/s/Msun] for the low-mass
binaries, :math:`\log\beta_0 = 39.28 \pm 0.05` [erg/s/(Msun/yr)] for the
high-mass ones, :math:`\gamma = 2.03 \pm 0.60` and :math:`\delta = 1.31 \pm
0.13` (:class:`XrbParams`).  Their stacks are means, so the relation is one
for the mean luminosity at fixed :math:`M_*` and SFR.  Stellar masses and
star-formation rates are for a Kroupa (2001) initial mass function, in
physical :math:`M_\odot` and :math:`M_\odot\,{\rm yr}^{-1}`.

The star-formation rate of a galaxy of stellar mass :math:`M_*` is the main
sequence of Speagle et al. (2014, Eq. 28) for star-forming galaxies,

.. math::

    \log{\rm SFR} = (0.84 - 0.026\,t)\log M_* - (6.51 - 0.11\,t),

with :math:`t` the age of the Universe in Gyr in their cosmology
(:func:`age_speagle14`), and the same sequence moved by ``dlog10_sfr_red``
(-1 dex by default) for quenched galaxies (:func:`mean_sfr`).  The red
fraction at fixed stellar mass is an input: the galaxy sector's
``zumandelbaum16`` split quenches by halo mass, so its red fraction at fixed
:math:`M_*` is a ratio of two stellar-mass functions integrated over a halo
field (:func:`mean_sfr` says how to form it).
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from ..cosmology import constants as C
from .params import Flat, Gaussian, Param, SectorParams, sector_params

__all__ = ["XrbParams", "LEHMER16", "SPEAGLE14_MS", "SPEAGLE14_COSMOLOGY",
           "GAMMA_XRB", "age_speagle14", "log10_sfr_main_sequence",
           "mean_sfr", "xrb_band_ratio", "lx_xrb", "p_log10_lx_xrb",
           "stacked_lx"]

#: Lehmer et al. (2016) Eq. 15, Table 3, "L10 plus CDF-S", 2--10 keV:
#: ``name -> (value, 1 sigma error)``.  :math:`\chi^2/\nu = 1.05` for
#: :math:`\nu = 63`, and a residual scatter of the stacks of 0.17 dex.
LEHMER16 = {"log10_alpha0": (29.37, 0.15), "gamma": (2.03, 0.60),
            "log10_beta0": (39.28, 0.05), "delta": (1.31, 0.13)}
#: Speagle et al. (2014) Eq. 28, their best fit to the star-forming main
#: sequence, :math:`\log{\rm SFR} = (a_0 - a_1 t)\log M_* - (b_0 - b_1 t)`, with
#: :math:`t` in Gyr; the resampling errors are 0.02, 0.003, 0.24 and 0.03.
#: Fitted over :math:`\log M_* = 9.7`--11.1 and :math:`z = 0.25`--2.75.
SPEAGLE14_MS = {"a0": 0.84, "a1": 0.026, "b0": 6.51, "b1": 0.11}
#: The cosmology Speagle et al. (2014) convert to, :math:`(h, \Omega_{\rm m},
#: \Omega_\Lambda) = (0.7, 0.3, 0.7)`: the age :math:`t(z)` of their Eq. 28 is
#: theirs, not the field's.
SPEAGLE14_COSMOLOGY = (0.7, 0.3, 0.7)
#: Photon index of the binaries' spectrum for a change of band, :math:`N(E)
#: \propto E^{-\Gamma}`: the power law Lehmer et al. (2016, Sect. 3) convert
#: their bands with.
GAMMA_XRB = 2.0
#: Seconds in a Julian gigayear.
_GYR_S = 3.15576e16
_LN10 = math.log(10.0)


@sector_params
class XrbParams(SectorParams):
    r"""The X-ray binaries of a galaxy and the star-formation rate that sets
    the high-mass ones.

    ``log10_alpha0``, ``gamma``, ``log10_beta0`` and ``delta`` are the four
    coefficients of Lehmer et al. (2016) Eq. 15, with their fit to the local
    sample plus the CDF-S stacks in the 2--10 keV band (Table 3) as the
    defaults and their errors as Gaussian priors.  ``sigma_xrb`` is the scatter
    of a single galaxy about that mean; ``sigma_sfr`` and ``dlog10_sfr_red`` set
    the mean star-formation rate at fixed stellar mass (:func:`mean_sfr`).
    """

    log10_alpha0: float = 29.37
    gamma: float = 2.03
    log10_beta0: float = 39.28
    delta: float = 1.31
    sigma_xrb: float = 0.3
    sigma_sfr: float = 0.2
    dlog10_sfr_red: float = -1.0

    _STATIC = ()

    _PARAMS = {
        "log10_alpha0": Param(
            29.37, (28.37, 30.37), Gaussian(29.37, 0.15), "log10 erg/s/Msun",
            "the 2-10 keV luminosity of the low-mass binaries per unit stellar "
            "mass at z = 0, Lehmer et al. (2016) Eq. 15 and Table 3, whose "
            "error is the prior; the box is 1 dex either side, about seven "
            "times the error", "prior"),
        "gamma": Param(
            2.03, (-1.0, 6.0), Gaussian(2.03, 0.60), "",
            "the evolution of the low-mass binaries' luminosity per unit "
            "stellar mass, (1+z)^gamma, Lehmer et al. (2016) Table 3; the box "
            "holds no evolution and their 0.5-2 keV value, 3.78 +- 0.82",
            "prior"),
        "log10_beta0": Param(
            39.28, (38.28, 40.28), Gaussian(39.28, 0.05),
            "log10 erg/s/(Msun/yr)",
            "the 2-10 keV luminosity of the high-mass binaries per unit "
            "star-formation rate at z = 0, Lehmer et al. (2016) Eq. 15 and "
            "Table 3; the box is 1 dex either side", "prior"),
        "delta": Param(
            1.31, (-1.0, 4.0), Gaussian(1.31, 0.13), "",
            "the evolution of the high-mass binaries' luminosity per unit "
            "star-formation rate, (1+z)^delta, Lehmer et al. (2016) Table 3; "
            "the box holds no evolution and three times their value",
            "prior"),
        "sigma_xrb": Param(
            0.3, (0.0, 1.0), Flat(), "dex",
            "the log-normal scatter of one galaxy's binary luminosity about "
            "the mean at fixed M* and SFR: Lehmer et al. (2016) Sect. 6.3.2 "
            "expect 0.2-0.4 dex from galaxy to galaxy. Zero is every galaxy "
            "on the mean; it does not enter the mean (stacked_lx)", "physical"),
        "sigma_sfr": Param(
            0.2, (0.0, 0.6), Flat(), "dex",
            "the scatter of log SFR about the main sequence, which sets the "
            "mean SFR of a population, 10^<log SFR> exp((ln10 sigma)^2 / 2): "
            "Speagle et al. (2014) Eq. 27 find 0.2 dex of true scatter. Zero "
            "is every galaxy on the sequence; the box is three times that",
            "physical"),
        "dlog10_sfr_red": Param(
            -1.0, (-3.0, 0.0), Flat(), "dex",
            "the offset of quenched galaxies below the main sequence, at "
            "fixed stellar mass. Zero is no quenching; below -3 their "
            "high-mass binaries are negligible against the low-mass ones at "
            "every stellar mass the main sequence was fitted at",
            "physical"),
    }


def age_speagle14(z):
    r"""The age of the Universe [Gyr] at redshift ``z`` in the flat
    cosmology of Speagle et al. (2014), :data:`SPEAGLE14_COSMOLOGY`:

    .. math::

        t(z) = \frac{2}{3H_0\sqrt{\Omega_\Lambda}}\,
            {\rm arsinh}\!\left[\sqrt{\Omega_\Lambda/\Omega_{\rm m}}\,
            (1+z)^{-3/2}\right],

    13.47 Gyr at :math:`z = 0` and 3.23 Gyr at :math:`z = 2`.
    """
    h, om, ol = SPEAGLE14_COSMOLOGY
    hubble_gyr = C.MPC_CM / 1e5 / (100.0 * h) / _GYR_S
    return (2.0 * hubble_gyr / (3.0 * math.sqrt(ol))
            * jnp.arcsinh(math.sqrt(ol / om)
                          * jnp.power(1.0 + jnp.asarray(z), -1.5)))


def log10_sfr_main_sequence(log10_mstar, z):
    r""":math:`\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})` on the star-forming
    main sequence at stellar mass ``log10_mstar`` (physical
    :math:`\log_{10}M_\odot`, Kroupa IMF) and redshift ``z``: Speagle et al.
    (2014) Eq. 28, :data:`SPEAGLE14_MS`, with :math:`t` of
    :func:`age_speagle14`.  It is the mean of :math:`\log{\rm SFR}`; at
    :math:`M_* = 10^{10}\,M_\odot` it is 0.07 at :math:`z = 0.1` and 1.41 at
    :math:`z = 2`.
    """
    c = SPEAGLE14_MS
    t = age_speagle14(z)
    return ((c["a0"] - c["a1"] * t) * jnp.asarray(log10_mstar)
            - (c["b0"] - c["b1"] * t))


def mean_sfr(log10_mstar, z, p: XrbParams = XrbParams(), red_fraction=None):
    r"""The mean star-formation rate [Msun/yr] of the galaxies of stellar mass
    ``log10_mstar`` (physical :math:`\log_{10}M_\odot`) at redshift ``z``.

    .. math::

        \langle{\rm SFR}\rangle = 10^{\log{\rm SFR}_{\rm MS}(M_*, z)}\,
            e^{(\ln 10\,\sigma_{\rm SFR})^2/2}\,
            \left[1 - f_{\rm red} + f_{\rm red}\,10^{\Delta_{\rm red}}\right],

    with :math:`\log{\rm SFR}_{\rm MS}` of :func:`log10_sfr_main_sequence`,
    :math:`\sigma_{\rm SFR}` = ``sigma_sfr``, the log-normal scatter about it,
    and :math:`\Delta_{\rm red}` = ``dlog10_sfr_red``, the offset of the
    quenched galaxies, which scatter by the same width.

    ``red_fraction`` is :math:`f_{\rm red}(M_*)`, broadcast against
    ``log10_mstar``; ``None`` is zero, every galaxy on the main sequence.  The
    galaxy sector has no red fraction at fixed stellar mass: ``zumandelbaum16``
    quenches by halo mass.  On a halo field it is the ratio of the stellar-mass
    functions of ``GalaxySector("zumandelbaum16_red")`` and
    ``GalaxySector("zumandelbaum15")``,
    :meth:`~ggah_mod.sectors.galaxies.GalaxySector.stellar_mass_function` with
    ``msun=True``, centrals plus satellites, interpolated to ``log10_mstar``.
    """
    f_red = 0.0 if red_fraction is None else jnp.asarray(red_fraction)
    sfr_ms = (jnp.power(10.0, log10_sfr_main_sequence(log10_mstar, z))
              * jnp.exp(0.5 * (_LN10 * p.sigma_sfr) ** 2))
    return sfr_ms * (1.0 - f_red + f_red * jnp.power(10.0, p.dlog10_sfr_red))


def xrb_band_ratio(band=(2.0, 10.0)):
    r"""The luminosity of the binaries in the rest-frame ``band`` [keV] over
    that in 2--10 keV, for the power law of :data:`GAMMA_XRB`:
    :math:`\ln(E_2/E_1)/\ln 5` at :math:`\Gamma = 2`, 0.861 for 0.5--2 keV.
    Lehmer et al. (2016, Table 3) also fit the 0.5--2 keV band directly, with
    other coefficients; the conversion keeps one relation for every band.
    """
    e1, e2 = float(band[0]), float(band[1])
    if not 0.0 < e1 < e2:
        raise ValueError(f"a band needs 0 < emin < emax, got ({e1}, {e2})")
    a = 2.0 - GAMMA_XRB
    if a == 0.0:
        return math.log(e2 / e1) / math.log(5.0)
    return (e2 ** a - e1 ** a) / (10.0 ** a - 2.0 ** a)


def lx_xrb(log10_mstar, sfr, z, p: XrbParams = XrbParams(), *,
           band=(2.0, 10.0)):
    r"""The mean X-ray luminosity [erg/s] of the binaries of galaxies of
    stellar mass ``log10_mstar`` (physical :math:`\log_{10}M_\odot`) and
    star-formation rate ``sfr`` [Msun/yr] at redshift ``z``, in the rest-frame
    ``band`` [keV].

    .. math::

        L_X^{\rm XRB} = \left[10^{\log\alpha_0}(1+z)^{\gamma}\,M_*
                       + 10^{\log\beta_0}(1+z)^{\delta}\,{\rm SFR}\right]
                       \frac{L_{\rm band}}{L_{2-10}},

    Lehmer et al. (2016) Eq. 15, with the band ratio of :func:`xrb_band_ratio`.
    At the defaults, :math:`M_* = 10^{10}\,M_\odot`, SFR = 1 and :math:`z = 0`
    it is :math:`2.34\times10^{39} + 1.91\times10^{39}` erg/s.
    """
    z1 = 1.0 + jnp.asarray(z)
    lmxb = (jnp.power(10.0, p.log10_alpha0) * jnp.power(z1, p.gamma)
            * jnp.power(10.0, jnp.asarray(log10_mstar)))
    hmxb = (jnp.power(10.0, p.log10_beta0) * jnp.power(z1, p.delta)
            * jnp.asarray(sfr))
    return (lmxb + hmxb) * xrb_band_ratio(band)


def p_log10_lx_xrb(log10_lx, log10_mstar, sfr, z, p: XrbParams = XrbParams(),
                   *, band=(2.0, 10.0)):
    r""":math:`dP/d\log_{10}L_X` of the binary luminosity of one galaxy of
    stellar mass ``log10_mstar``, star-formation rate ``sfr`` [Msun/yr] and
    redshift ``z``, at ``log10_lx`` [erg/s, rest-frame ``band``].

    A log-normal of width ``sigma_xrb`` whose mean is :func:`lx_xrb`, the
    stacked mean of Lehmer et al. (2016): its median is
    :math:`\log L_X^{\rm XRB} - \ln 10\,\sigma_{\rm XRB}^2/2`.  The arguments
    broadcast together.
    """
    sig = p.sigma_xrb
    med = (jnp.log10(lx_xrb(log10_mstar, sfr, z, p, band=band))
           - 0.5 * _LN10 * sig ** 2)
    x = (jnp.asarray(log10_lx) - med) / sig
    return jnp.exp(-0.5 * x ** 2) / (sig * math.sqrt(2.0 * math.pi))


def stacked_lx(log10_mstar, z, p_xrb: XrbParams = XrbParams(), *, agn=None,
               agn_params=None, red_fraction=None, band=(2.0, 10.0),
               obsc=None, satellite_fraction=None, components=False):
    r"""The mean X-ray luminosity [erg/s] of the galaxies of stellar mass
    ``log10_mstar`` at redshift ``z``, in the rest-frame ``band`` [keV]: their
    binaries and, with an AGN sector, their active nuclei.

    .. math::

        \langle L_X\rangle(M_*) = L_X^{\rm XRB}\big(M_*,
                \langle{\rm SFR}\rangle(M_*), z\big)
            + \left[1 - f_{\rm s} + f_{\rm s}\,f_{\rm duty}^{\rm sat}\right]
              \int d\log L_X\;L_X\,f_{\rm ERDF}(M_*)\,
              \frac{dP(\log L_X|M_*)}{d\log L_X}\,K(L_X),

    with :math:`L_X^{\rm XRB}` of :func:`lx_xrb` at the mean star-formation
    rate of :func:`mean_sfr` (the relation is linear in SFR, so the mean of
    the binaries is the relation at the mean), and the second term the mean
    over every AGN, without a luminosity cut, of the sector's distribution at
    fixed stellar mass,
    :meth:`~ggah_mod.sectors.agn.AgnSector.p_active_loglx_given_mstar`, which
    carries the duty cycle and is zero below the sector's stellar-mass cut,
    :math:`10^8\,M_\odot` by default.  The AGN luminosity is intrinsic;
    :math:`K` = 1 in
    2--10 keV, and in another band the sector's
    :meth:`~ggah_mod.sectors.agn.AgnSector.band_factor` in the rest frame
    (absorbed and unabsorbed, weighted by the obscured fraction), which needs
    ``obscuration="split"``.

    Parameters
    ----------
    log10_mstar : float or array (N_*,)
        Physical :math:`\log_{10}(M_*/M_\odot)`.
    z : float
        Redshift.
    p_xrb : XrbParams
    agn : AgnSector, optional
        ``None`` returns the binaries alone.
    agn_params : AgnParams, optional
        The sector's parameters; their defaults if ``None``.
    red_fraction : float or array (N_*,), optional
        The quenched fraction at ``log10_mstar`` (:func:`mean_sfr`); ``None``
        is zero.
    band : (float, float)
        The rest-frame band [keV]; the binaries convert with :data:`GAMMA_XRB`.
    obsc : ObscurationParams, optional
        For the AGN band factor outside 2--10 keV.
    satellite_fraction : float or array (N_*,), optional
        :math:`f_{\rm s}`, the fraction of the galaxies at ``log10_mstar`` that
        are satellites, whose nuclei shine ``f_duty_sat`` times as often;
        ``None`` is zero, the central's duty cycle for every galaxy.
    components : bool
        Return ``{"lmxb", "hmxb", "xrb", "agn", "total"}`` instead of the total.
    """
    ms = jnp.asarray(log10_mstar, dtype=float)
    sfr = mean_sfr(ms, z, p_xrb, red_fraction)
    ratio = xrb_band_ratio(band)
    z1 = 1.0 + jnp.asarray(z)
    lmxb = (jnp.power(10.0, p_xrb.log10_alpha0) * jnp.power(z1, p_xrb.gamma)
            * jnp.power(10.0, ms)) * ratio
    hmxb = (jnp.power(10.0, p_xrb.log10_beta0) * jnp.power(z1, p_xrb.delta)
            * sfr) * ratio
    l_agn = jnp.zeros_like(lmxb)
    if agn is not None:
        from .agn import AgnParams
        ap = AgnParams() if agn_params is None else agn_params
        prob = agn.p_active_loglx_given_mstar(jnp.ravel(ms), ap, z=z)
        weight = jnp.power(10.0, agn.loglx)
        if (float(band[0]), float(band[1])) != (2.0, 10.0):
            weight = weight * agn.band_factor(ap, band, frame="rest", z=z,
                                              obsc=obsc)
        l_agn = (jnp.sum(prob * weight, axis=-1) * agn.dlx).reshape(ms.shape)
        if satellite_fraction is not None:
            f_s = jnp.asarray(satellite_fraction)
            l_agn = l_agn * (1.0 - f_s + f_s * agn._satellite_duty(ap))
    xrb = lmxb + hmxb
    if components:
        return {"lmxb": lmxb, "hmxb": hmxb, "xrb": xrb, "agn": l_agn,
                "total": xrb + l_agn}
    return xrb + l_agn
