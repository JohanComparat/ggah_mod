r"""Where the baryons went, and what pushed them.

A halo with feedback holds less gas than its share of the cosmic baryon budget.
This module gives two ways to say how much less:

* **Parametrised fractions.** :math:`f_{\rm gas}(M)` is written as a sigmoid
  and the pivot mass is fitted; this says nothing about why.
* **Energy closure.** The missing baryons were pushed out against the halo's
  binding energy by accreting black holes and supernovae.
  Requiring the two energies to balance turns the retained fraction from a
  fitted function into a prediction of the AGN and stellar sectors.

The black-hole masses the AGN sector builds its X-ray luminosities on set the
feedback energy, so X-ray data constrain the budget and thereby the lensing
suppression: one chain ties two observables that are otherwise fitted apart.

.. math::

    E_{\rm bind}(M) &= \Delta f_b\,M\,v_\Delta^2\,(1 - \eta_{\rm ej}^{-1}) \\
    E_{\rm AGN}(M)  &= \epsilon_{\rm AGN}\,\epsilon_r\,M_{\rm BH}(M)\,c^2 \\
    E_{\rm SN}(M)   &= \epsilon_{\rm SN}\,\frac{M_*(M)}{1-R}\,E_{\rm SN}/M_\odot \\
    f_{\rm ret}(M)  &= f_b^{\rm cosmic} - \Delta f_b

The AGN term is the energy the black hole released while it grew
(:func:`e_agn_soltan`).
The predecessor's form, a present X-ray luminosity extrapolated over a Hubble
time, :math:`\epsilon_{\rm AGN}f_{\rm duty}k_{\rm bol}L_X t_H`, balanced an
instantaneous rate against a cumulative supernova budget; this package removed
it in 0.8.8.

Retained baryons and gas fractions
----------------------------------

The two routes compute different quantities.
The closure balances the energy to displace baryons, so
:func:`f_retained_energy` returns every baryon still in the halo, stars
included.
The parametrised forms were calibrated against FLAMINGO, which measures a gas
fraction.
The difference is :math:`f_\star`, up to 0.028 of the halo mass; a caller who
adds :math:`f_\star` to the retained fraction counts it twice, and at the
default parameters (z = 0, the 256 mass nodes of ``DIFFERENTIABLE``) puts
:math:`f_\star + f_{\rm gas}` above :math:`f_b^{\rm cosmic}` on 56 of them,
worst by a factor 1.041 at :math:`10^{16}\,M_\odot/h`, with the excess taken
out of the dark matter.

:func:`f_retained_energy` therefore feeds
:meth:`~ggah_mod.sectors.matter.BaryonSplit.from_retained`, and the
``f_gas_*`` family feeds :meth:`~ggah_mod.sectors.matter.BaryonSplit.from_hot`.
At the default parameters and z = 0.202 the closure's retained fraction is 3.6
times the sigmoid's gas fraction at :math:`10^{12}\,M_\odot/h`, mostly this
difference of definition: the closure's hot gas, the stars subtracted, is 0.93
of the sigmoid's there and 1.7 times it at :math:`10^{13}`.

Gas mass fraction only
----------------------

The closure predicts the gas mass fraction and nothing else; the X-ray and tSZ
amplitudes stay free parameters of :mod:`ggah_mod.sectors.gas`.
The DPM temperature is :math:`T = P_e/n_e`, and :math:`P_e` does not depend on
the electron-density normalisation.
Rescaling :math:`n_{e,0.3}` by :math:`A` to match an energy-predicted
:math:`f_{\rm gas}` would move every halo's temperature as :math:`1/A` and
de-calibrate the :math:`kT`--:math:`M` relation, and the X-ray emissivity would
scale as :math:`A^{2-p}`, with :math:`p` the local slope of :math:`\Lambda(T)`,
rather than as :math:`A^2`.

So :math:`f_{\rm gas}` has one definition, this one, and the gas sector's own
profile integral is a diagnostic: the ratio of the two says whether the X-ray
amplitude and the feedback budget agree.
It is reported, not imposed.

Soft saturation
---------------

The predecessor wrote the saturation as ``jnp.minimum(f_b - f_floor, x)`` and
``jnp.maximum(x, 0)``.
Both are ties, and at the fiducial parameters the AGN channel saturates, so the
model sits on the clamp, where JAX splits a ``minimum`` gradient 50/50 between
its arguments and the derivative on the far side is zero.
The predecessor's documentation called ``eps_sn`` "a flat direction at the
saturated fiducial"; its gradient was being discarded.

:func:`soft_saturate` replaces both with an exponential approach that is exact
in the unsaturated limit, tends to the ceiling from below, and is
:math:`C^\infty` with a derivative that never vanishes at finite argument.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..cosmology import constants as C
from ..numerics import soft_saturate
from .calibration import Calibration, check_sector_calibration
from .ejecta import ETA_EJ
from .params import (
    Flat, Gaussian, Param, SectorParams, sector_params,
)

#: Sentinel: "this keyword was not passed", so a container can supply it.
#: ``None`` cannot serve, because ``eta_ej=None`` already means something
#: -- no finite-radius correction -- and would be indistinguishable from
#: absent.
_UNSET = object()

__all__ = [
    "G_MPC_KMS2_MSUN", "ERG_PER_MSUN_KMS2", "E_SN_PER_MSUN_KMS2",
    "soft_saturate", "v_delta_squared", "v200_squared",
    "f_gas_sigmoid", "f_gas_powerlaw", "f_gas_upturn", "f_retained_energy",
    "gas_concentration_factor", "wind_mass_loading",
    "F_GAS", "make_f_gas", "F_GAS_CALIBRATION",
    "C2_KMS2", "RETURN_FRACTION", "EnergeticsParams",
    "e_agn_soltan", "e_supernova", "binding_energy",
]


# =========================================================================
# Constants
# =========================================================================

#: Newton's constant as :math:`{\rm Mpc}\,({\rm km/s})^2/M_\odot`.
#:
#: **Derived from** :data:`~ggah_mod.cosmology.constants.RHO_CRIT0` rather than
#: written as a literal, via :math:`\rho_{c,0} = 3H_0^2/8\pi G` with
#: :math:`H_0 = 100h`.  Two reasons: the package keeps one definition per
#: physical constant (``tests/test_coherence.py`` enforces it for the ones the
#: predecessor duplicated), and a literal ``G`` taken from a different source
#: than ``RHO_CRIT0`` makes :math:`v_{200}^2` and :math:`r_{200}` mutually
#: inconsistent at the 1e-4 level -- small, and exactly the kind of thing that
#: never gets found.
#:
#: :math:`h` cancels: ``M`` is in :math:`M_\odot/h` and ``r`` in
#: :math:`{\rm Mpc}/h`, so :math:`GM/r` is an :math:`h`-free velocity squared.
#: Re-exported from :mod:`~ggah_mod.cosmology.constants`, where it moved when
#: the halo field needed it too.  Kept as a name here because this module's
#: readers and its docstrings refer to it.
G_MPC_KMS2_MSUN = C.G_MPC_KMS2_MSUN

#: One :math:`M_\odot\,({\rm km/s})^2` in erg: an energy in erg into the units
#: the binding energy is written in.
ERG_PER_MSUN_KMS2 = 1.989e43

#: Supernova energy per solar mass of stars formed, as :math:`({\rm km/s})^2`.
#: About :math:`10^{49}` erg per :math:`M_\odot` -- one :math:`10^{51}` erg
#: event (Janka 2012, ARNPS 62, 407) per :math:`\sim100\,M_\odot` of stars.
#:
#: For a Chabrier (2003) IMF the number of core-collapse progenitors is
#: :math:`1.18\times10^{-2}\,M_\odot^{-1}` for 8--100 :math:`M_\odot` and
#: :math:`1.74\times10^{-2}` for 6--100 (Dalla Vecchia & Schaye 2012, MNRAS
#: 426, 140; the TNG and EAGLE conventions, Pillepich et al. 2018, MNRAS 473,
#: 4077, and Schaye et al. 2015, MNRAS 446, 521), i.e. 1.2--1.7e49 erg.
#: Failed explosions and a mean kinetic energy of 0.65--0.8e51 erg lower it
#: (Sukhbold et al. 2016, ApJ 821, 38), and older models adopt 4e48 erg
#: (Springel & Hernquist 2003, MNRAS 339, 289; Croton et al. 2006, MNRAS 365,
#: 11, :math:`\tfrac12 V_{\rm SN}^2`).  The constant is uncertain by a factor
#: of about two, and 9.9e48 erg sits inside that range.
E_SN_PER_MSUN_KMS2 = 5.0e5

#: :math:`c^2` as :math:`({\rm km/s})^2`.
#:
#: The predecessor defined this constant for the Soltan-form AGN channel its
#: own comment specifies -- "a coupling fraction of the BH rest mass
#: (M_BH c^2)" -- and then **never used it**: one hit in the package, the
#: definition.  What went in instead was a luminosity times a Hubble time.  It
#: is defined here because the channel it was meant for now exists.
C2_KMS2 = 8.98755178737e10

#: Mass returned to the ISM per unit mass of stars formed, for a Chabrier
#: (2003) IMF.
#:
#: 0.41 is the instantaneous-recycling value for Chabrier (Madau & Dickinson
#: 2014, ARA&A 52, 415, Sect. 3.1; 0.44 with a 60 Msun black-hole cut).  It
#: depends on the population-synthesis code, the yields and the age assumed:
#: 0.40--0.47 with metallicity and yield set (Vincenzo et al. 2016, MNRAS 455,
#: 4183) and 0.48--0.50 for a population 10 Gyr old (Leitner & Kravtsov 2011,
#: ApJ 734, 48; Moster, Naab & White 2013, MNRAS 428, 3121, fitted to Bruzual
#: & Charlot 2003).  0.4 is therefore the low end for the old stars of a
#: low-redshift halo; 0.5 would raise every supernova energy by 20 per cent.
#:
#: Distinct from :data:`~ggah_mod.sectors.sham.REMNANT_FRACTION`, and the two
#: are easy to confuse into one number.  A stellar-population mass is the
#: *surviving* mass, :math:`M_\star = (1-R)M_{\rm formed}`; the remnant fraction
#: is a fraction of that surviving mass, not of the mass formed.  Supernova
#: counts scale with :math:`M_{\rm formed}`, so an energy budget written against
#: :math:`M_\star` is low by :math:`1/(1-R) = 1.67`.
RETURN_FRACTION = 0.4


# =========================================================================
# The pieces
# =========================================================================



def v_delta_squared(m, z, cosmo, mdef=None):
    r""":math:`v_\Delta^2 = GM/r_\Delta` [(km/s)^2], with a physical radius.

    A binding energy is :math:`GM/r` with :math:`r` the proper radius: the
    comoving :math:`r_\Delta` would understate :math:`v^2` by :math:`(1+z)`,
    6 per cent at :math:`z = 0.135`, 100 per cent at :math:`z = 1`, and zero
    at :math:`z = 0`.

    The radius is that of ``mdef``, the halo field's mass definition (the
    backend's by default), so the binding energy is that of the halo the rest of
    the package describes.
    A hard-coded :math:`200\rho_{\rm c}` would be too large at the shipped 200m
    by :math:`R_{200{\rm m}}/R_{200{\rm c}} \simeq 1.66`.
    """
    from ..halos.mass_definitions import parse_mass_def

    m = jnp.asarray(m)
    z = jnp.asarray(z)
    if mdef is None:
        from ..backend import ACCURATE
        mdef = ACCURATE.mdef
    r_comoving = parse_mass_def(mdef).radius(m, z, cosmo)
    return G_MPC_KMS2_MSUN * m / (r_comoving / (1.0 + z))


def v200_squared(m, z, cosmo, mdef=None):
    """Deprecated name for :func:`v_delta_squared`.

    The quantity is at the halo field's mass definition, not necessarily 200;
    a caller who wrote ``v200`` meant the halo's boundary, and gets it.
    """
    return v_delta_squared(m, z, cosmo, mdef)


# =========================================================================
# Parameterised baryon fractions
# =========================================================================

@jax.jit
def f_gas_sigmoid(log10m, f_b_cosmic, log10_m_pivot=13.5, beta_b=1.5,
                  f_gas_min=0.01):
    r""":math:`f_{\rm gas} = f_b^{\rm cosmic}/[1 + (M_{\rm piv}/M)^{\beta_b}]`.

    The production form.
    The pivot is calibrated against FLAMINGO; the floor is a CGM-census lower
    bound rather than a numerical guard.
    """
    x = jnp.power(10.0, log10_m_pivot - jnp.asarray(log10m))
    return f_gas_min + (f_b_cosmic - f_gas_min) / (1.0 + jnp.power(x, beta_b))


@jax.jit
def f_gas_powerlaw(log10m, f_b_cosmic, log10_m_ref=14.0, alpha_b=0.3,
                   f_gas_min=0.0):
    r""":math:`f_{\rm gas} = f_b^{\rm cosmic}(M/M_{\rm ref})^{\alpha_b}`, saturating.

    The ceiling uses :func:`soft_saturate` rather than the predecessor's
    ``clip``, which puts a tie exactly where massive haloes sit.
    """
    ratio = jnp.power(10.0, alpha_b * (jnp.asarray(log10m) - log10_m_ref))
    return f_gas_min + soft_saturate(f_b_cosmic * ratio,
                                     f_b_cosmic - f_gas_min)


@jax.jit
def f_gas_upturn(log10m, f_b_cosmic, log10_m_hi=13.5, beta_hi=1.5,
                 f_gas_min=0.01, f_lo_amp=0.05, log10_m_lo=11.5, beta_lo=2.0):
    r"""Group-scale sigmoid plus a low-mass upturn (a CGM census).

    Two sigmoids: the high-mass one rises to the cosmic value, the low-mass one
    adds a bump where a census finds more cold gas than a single sigmoid allows.
    """
    log10m = jnp.asarray(log10m)
    hi = 1.0 / (1.0 + jnp.power(jnp.power(10.0, log10_m_hi - log10m), beta_hi))
    lo = 1.0 / (1.0 + jnp.power(jnp.power(10.0, log10m - log10_m_lo), beta_lo))
    return f_gas_min + (f_b_cosmic - f_gas_min) * hi + f_lo_amp * lo


# =========================================================================
# The energy closure
# =========================================================================

# =========================================================================
# The two feedback channels, and the energy they are balanced against
# =========================================================================

@jax.jit
def e_agn_soltan(m_bh, log10_eps_agn=-1.252, eps_radiative=0.1):
    r"""AGN energy from the black-hole rest mass [(mass unit of ``m_bh``) (km/s)^2].

    .. math::

        E_{\rm AGN} = \epsilon_{\rm AGN}\,\epsilon_r\,M_{\rm BH}\,c^2

    The Soltan form, which the predecessor's own comment specifies ("a coupling
    fraction of the BH rest mass (M_BH c^2)").

    The predecessor's luminosity form,
    :math:`\epsilon_{\rm AGN}f_{\rm duty}k_{\rm bol}L_X t_H`, is a present
    luminosity extrapolated over a Hubble time; this is the energy the black
    hole released while it grew.
    The supernova channel is cumulative, so it is balanced against a cumulative
    quantity, and the duty cycle and the bolometric correction disappear, as
    neither is a property of the integrated history.
    The luminosity form was removed in 0.8.8.

    At :math:`\epsilon_r = 0.1` this is :math:`1.8\times10^{53}` erg per solar
    mass of black hole against :math:`10^{49}` erg per solar mass of stars, so
    at :math:`M_{\rm BH}/M_\star \sim 10^{-3}` the AGN channel releases ~20
    times the supernova energy.
    At the default couplings, :math:`\epsilon_{\rm AGN} = 0.056` and
    :math:`\epsilon_{\rm SN} = 0.1`, and counting every black hole and every
    star of a halo as :meth:`~ggah_mod.sectors.gas.HotGasDPM.feedback_budget`
    does, :math:`E_{\rm AGN}/E_{\rm SN}` is 4.9 at :math:`10^{11}`, 6.1 at
    :math:`10^{13}` and 6.0 at :math:`10^{15}\,M_\odot/h`, at z = 0.202.

    Parameters
    ----------
    m_bh : array (NM,)
        The mean black-hole mass, not ten to the mean log, which differs by a
        factor 1.40 at the published scatter.
        The energy carries its mass unit, so set against
        :func:`binding_energy` and :func:`e_supernova`, whose masses are in
        :math:`M_\odot/h`, it has to be in :math:`M_\odot/h` too.
        :meth:`~ggah_mod.sectors.agn.AgnSector.mean_mbh` returns physical
        :math:`M_\odot`, and :func:`f_retained_energy` converts.
    log10_eps_agn : float
        :math:`\log_{10}` of the mechanical coupling to the gas; -1.252
        (:math:`\epsilon_{\rm AGN} = 0.056`) by default, see
        :class:`EnergeticsParams`.
    eps_radiative : float
        Accretion radiative efficiency.
        0.1 is the standard thin-disc value and the one the Soltan argument is
        usually quoted at.
    """
    return (jnp.power(10.0, log10_eps_agn) * eps_radiative
            * jnp.asarray(m_bh) * C2_KMS2)


@jax.jit
def e_supernova(m_star, eps_sn=0.1, return_fraction=RETURN_FRACTION):
    r"""Supernova energy [Msun (km/s)^2], from the mass formed.

    .. math::

        E_{\rm SN} = \epsilon_{\rm SN}\,\frac{M_\star}{1-R}\,
                     \frac{E_{\rm SN}}{M_\odot}

    Cumulative: this is the energy every star ever formed has already
    released, so quiescent galaxies need no separate treatment; a red/blue
    split would be needed for a rate, not here.

    :math:`M_\star` is the surviving mass (living stars plus remnants), while
    supernova counts scale with the mass formed, hence the factor
    :math:`1/(1-R) = 1.67`.
    :data:`RETURN_FRACTION` gives the sources of :math:`R`, which is not
    :data:`~ggah_mod.sectors.sham.REMNANT_FRACTION` (a fraction of a different
    denominator).

    :math:`e_{\rm SN} = 5\times10^5\,({\rm km/s})^2 \approx 10^{49}` erg per
    solar mass formed is one supernova per ~100 M_sun at :math:`10^{51}` erg;
    :data:`E_SN_PER_MSUN_KMS2` gives the IMF counts behind it.
    The budget's use follows Dekel & Silk (1986) and Somerville & Dave (2015)
    Sect. 3; :class:`EnergeticsParams` describes :math:`\epsilon_{\rm SN}`.

    The energy was injected while the stars were forming, into progenitors far
    less bound than the halo it is balanced against.
    Accounting for that needs a star-formation history, which this package does
    not have, and is where a quiescent/star-forming distinction would enter.
    """
    return (eps_sn * jnp.asarray(m_star) / (1.0 - return_fraction)
            * E_SN_PER_MSUN_KMS2)


def binding_energy(m, z, cosmo, delta_f_b=1.0, eta_ej=None, mdef=None):
    r"""Energy to move :math:`\Delta f_b M` out to :math:`r_{\rm ej}` [Msun (km/s)^2].

    .. math::

        E_{\rm bind} = \Delta f_b\,M\,v_\Delta^2
                       \left(1 - \frac{1}{\eta_{\rm ej}}\right)

    The form :math:`E_{\rm bind} = \Delta f_b M v_\Delta^2` comes from Silk &
    Rees (1998); Wu, Fabian & Nulsen (2000); and Bower, McCarthy & Benson
    (2008), whose present-day ejection works against the z = 0 binding energy,
    as here.
    McCarthy et al. (2011) find instead the gas ejected earlier, at 2 < z < 4,
    from the progenitors' shallower wells.
    Davies, Crain, McCarthy et al. (2019), MNRAS 485, 3783, show that
    :math:`f_{\rm CGM}` is set by the integrated black-hole feedback energy
    rather than by its instantaneous rate, the argument the Soltan channel of
    :func:`e_agn_soltan` implements.
    These sources were traced (``PLAN.md`` item G8), not re-derived.

    With ``eta_ej=None`` the bracket is 1, the work to unbind the gas to
    infinity, an overestimate whenever the gas ends up at a finite radius.
    In a point-mass potential the work to move mass from :math:`R_\Delta` to
    :math:`\eta_{\rm ej}R_\Delta` is smaller by exactly that bracket: at
    :math:`\eta_{\rm ej} = 2` it is half.

    Supplying it ties this module to
    :class:`~ggah_mod.sectors.ejecta.EjectaSector`'s radius, so
    :math:`f_{\rm ejected}` and :math:`r_{\rm ej}` become one prediction.
    The point mass is an approximation: the real potential is the halo's, and
    its logarithmic tail makes the true bracket approach 1 more slowly.
    """
    v2 = v_delta_squared(m, z, cosmo, mdef)
    reach = 1.0 if eta_ej is None else (1.0 - 1.0 / jnp.asarray(eta_ej))
    return jnp.asarray(delta_f_b) * jnp.asarray(m) * v2 * reach


def f_retained_energy(m, z, cosmo, m_star, f_b_cosmic, *,
                      m_bh=None,
                      params: "EnergeticsParams" = None,
                      log10_eps_agn=_UNSET, eps_radiative=_UNSET,
                      eps_sn=_UNSET,
                      return_fraction=_UNSET, eta_ej=_UNSET,
                      f_retained_min=_UNSET, mdef=None,
                      return_expelled=False, m_cold=None):
    r"""The baryon fraction a halo kept, predicted from the feedback budget.

    This balances the energy to displace baryons against the binding energy,
    so it returns :math:`f_b^{\rm cosmic} - \Delta f_b`: every baryon still in
    the halo, stars included.
    It is not a gas fraction (the name ``f_gas_energy`` was dropped for that
    reason); adding :math:`f_\star` on top puts :math:`f_\star + f_{\rm gas}`
    above :math:`f_b^{\rm cosmic}` on 56 of the 256 mass nodes of
    ``DIFFERENTIABLE`` at the default parameters and z = 0, worst by a factor
    1.041 at :math:`10^{16}\,M_\odot/h`.
    :meth:`~ggah_mod.sectors.matter.BaryonSplit.from_retained` takes it.

    The coefficients come from :class:`EnergeticsParams`, with the keywords
    kept as per-call overrides (``PLAN.md`` item G6).
    ``eta_ej`` keeps its own meaning for ``None`` (no finite-radius
    correction), so absence is marked by a sentinel rather than by ``None``.

    The parametrised forms above calibrate a gas fraction (FLAMINGO's) and feed
    :meth:`~ggah_mod.sectors.matter.BaryonSplit.from_hot`, so the two routes
    are not two estimates of one quantity, and the factor 3.6 between them at
    :math:`10^{12}\,M_\odot/h` (the default parameters, z = 0.202) is mostly a
    difference of definition.

    Parameters
    ----------
    m : array (NM,) [Msun/h]
    z : float
    cosmo : Cosmology
    m_star : array (NM,) [Msun/h]
        Stellar mass per halo, from the occupation sector's SHMR: the
        surviving mass, which :func:`e_supernova` converts.
    f_b_cosmic : float
        :math:`\Omega_b/\Omega_{cb}`, from
        :func:`~ggah_mod.sectors.matter.cosmic_baryon_fraction`, because the
        halo mass is cold mass.
    m_bh : array (NM,) [Msun], physical
        Mean black-hole mass per halo, as
        :meth:`~ggah_mod.sectors.agn.AgnSector.mean_mbh` returns it.
        Required: the AGN term is :func:`e_agn_soltan`, the energy released
        while the hole grew.
        Keyword-only and ``None`` by default, so that a call without it is
        refused by name rather than by a missing positional.

        Converted here to :math:`M_\odot/h`, the unit of ``m`` and ``m_star``
        and so of the binding energy and the supernova budget it is balanced
        against; unconverted, :math:`E_{\rm AGN}` would be :math:`1/h = 1.48`
        too large against both at Planck 2018 (as it was before 0.9.0).
    log10_eps_agn : float
        :math:`\log_{10}` of the AGN mechanical coupling efficiency, the
        fraction of the released energy that couples to the gas.
        It is a parameter of its own, not the baryon sigmoid's pivot-mass slot
        that the predecessor reused for it.
    eps_radiative : float
        Accretion radiative efficiency.
    eps_sn : float
        Supernova coupling efficiency.
    return_fraction : float
        Mass returned to the ISM per unit mass formed; see
        :data:`RETURN_FRACTION`.
    eta_ej : float, optional
        :math:`r_{\rm ej}/R_\Delta`.
        ``None`` means ejection to infinity; supplying it makes the binding
        energy the work actually done.
        See :func:`binding_energy`.
    f_retained_min : float
        Floor on the retained fraction, in halo-mass units like ``f_b_cosmic``
        and not as a fraction of it.
        Approached asymptotically, never clipped to.
    m_cold : array (NM,) [Msun/h], optional
        Neutral gas per halo, HI + H2 with helium (``ColdGasSector.m_neutral``).
        It cannot be expelled either (from 0.9.5): the budget unbinds the hot
        atmosphere, and gas cold enough to be atomic or molecular sits in the
        galaxies' discs.
        Counted as expellable, it makes the implied hot gas negative where the
        closure keeps less than the stars and the neutral gas together: -0.017
        of :math:`f_b` at 1e10.9-1e11.4 Msun/h with ``catinella18`` at the
        default parameters, z = 0.
        ``None`` reproduces 0.9.4 bit for bit.
    return_expelled : bool
        When true, return ``(f_retained, f_expelled)`` instead of just the
        first.
        ``f_expelled`` is the ejected component the matter budget needs;
        recomputing it as ``f_b - f_retained`` at the call site would be a
        second subtraction that could drift from this one.

    Notes
    -----
    Returns a value in
    :math:`(f_\star + f_{\rm cold} + f_{\rm retained,min},\
    f_b^{\rm cosmic}]`, strictly (:math:`f_{\rm cold} = 0` without
    ``m_cold``): the saturation is exponential, so neither end is ever exactly
    attained and the gradient survives at both.
    Stars cannot be expelled, so the floor sits above the stellar fraction: the
    retained gas is at least :math:`f_{\rm retained,min}`.
    """
    p = EnergeticsParams() if params is None else params
    log10_eps_agn = p.log10_eps_agn if log10_eps_agn is _UNSET else log10_eps_agn
    eps_radiative = p.eps_radiative if eps_radiative is _UNSET else eps_radiative
    eps_sn = p.eps_sn if eps_sn is _UNSET else eps_sn
    return_fraction = (p.return_fraction if return_fraction is _UNSET
                       else return_fraction)
    eta_ej = p.eta_ej if eta_ej is _UNSET else eta_ej
    f_retained_min = (p.f_retained_min if f_retained_min is _UNSET
                      else f_retained_min)
    m = jnp.asarray(m)

    if m_bh is None:
        raise ValueError(
            "f_retained_energy needs `m_bh`, the mean black-hole mass per halo "
            "-- `AgnSector.mean_mbh`.  The AGN term is the energy the black "
            "hole released while it grew; it is not derived from an X-ray "
            "luminosity, because a luminosity is a rate and this is an "
            "integrated history.")
    # m_bh is physical Msun; m, m_star and so E_bind and E_SN are Msun/h.
    e_agn = e_agn_soltan(jnp.asarray(m_bh) * cosmo.h, log10_eps_agn,
                         eps_radiative)

    e_sn = e_supernova(m_star, eps_sn, return_fraction)

    # `binding_energy` at unit `delta_f_b`: the ratio below is the fraction it
    # can displace, so the quantity wanted is the energy *per unit* expelled
    # fraction rather than the energy for a given one.
    e_per_fraction = binding_energy(m, z, cosmo, 1.0, eta_ej, mdef)
    # Stars cannot be expelled: the budget displaces gas, so what it can remove
    # is capped by the baryons that are not already stars.  With the cap at
    # f_b - f_min alone, a small halo's retained fraction fell to the floor,
    # below its own stellar fraction, and the gas it implied was negative on 14
    # of 256 nodes.  The guard keeps the cap positive for a stellar fraction
    # that is itself unphysical (above f_b - f_min); it binds nowhere else.
    f_star = jnp.asarray(m_star) / m
    # Nor can the neutral gas (0.9.5): it is in the discs, not the atmosphere.
    f_cold = 0.0 if m_cold is None else jnp.asarray(m_cold) / m
    cap = jnp.maximum(f_b_cosmic - f_star - f_cold - f_retained_min, 1e-12)
    expelled = soft_saturate((e_agn + e_sn) / e_per_fraction, cap)
    retained = f_b_cosmic - expelled
    return (retained, expelled) if return_expelled else retained


# =========================================================================
# Displacement of the gas that stays
# =========================================================================

@jax.jit
def gas_concentration_factor(log10m, eta_min=0.6, log10_m_eta=13.0,
                             beta_eta=1.5):
    r"""Gas puffing-out: :math:`c_{\rm gas} = \eta(M)\,c_{\rm DM}`, :math:`\eta \le 1`.

    .. math::

        \eta(M) = 1 - \frac{1-\eta_{\min}}{1 + (M/M_\eta)^{\beta_\eta}}

    Feedback does not only remove gas, it redistributes what remains outward.
    Calibrated on the hydrodynamic-to-dark-matter-only concentration ratio of
    Sorini, Bose, Pakmor, Hernquist, Springel, Hadzhiyska, Hernandez-Aguayo &
    Kannan (2025), MNRAS 536, 728: MillenniumTNG's :math:`c_{\rm hydro}/
    c_{\rm DMO}` from dwarfs to superclusters, the source of ``eta_min``,
    ``log10_m_eta`` and ``beta_eta``.
    The source was traced through the predecessor's test suite (``PLAN.md``
    item G8), not checked against the paper.

    Distinct from :math:`f_{\rm gas}`: one changes how much gas there is, the
    other where it sits.
    """
    x = jnp.power(10.0, jnp.asarray(log10m) - log10_m_eta)
    return 1.0 - (1.0 - eta_min) / (1.0 + jnp.power(x, beta_eta))


def wind_mass_loading(m, z, cosmo, eta_w_norm=0.0, alpha_w=1.0):
    r"""Supernova wind loading :math:`\eta_w = \eta_0 (v_c/200)^{-\alpha_w}`.

    Applied as :math:`\eta \to \eta/(1+\eta_w)`, puffing low-mass haloes
    further.
    :math:`\alpha_w = 1` is momentum-driven and 2 is energy-driven.
    At the fiducial :math:`\eta_0 = 0` this is exactly the identity, so
    switching it on is a strict extension of the model.
    """
    v_c = jnp.sqrt(v200_squared(m, z, cosmo))
    return eta_w_norm * jnp.power(v_c / 200.0, -alpha_w)


# =========================================================================
# Registry
# =========================================================================

#: Parameterised baryon fractions.  The energy closure is **not** here: it takes
#: the AGN and stellar sectors as arguments, so it has a different signature,
#: and giving it a matching one would mean the others accepting a luminosity
#: they cannot use -- the mistake layer 2 documents for its concentration
#: relations.
F_GAS = {
    "sigmoid": f_gas_sigmoid,
    "powerlaw": f_gas_powerlaw,
    "upturn": f_gas_upturn,
}

#: What each was calibrated against, and what it is for.
#:
#: Every entry carries ``z_range=None``, and that is a statement rather than an
#: omission.  This registry never recorded a redshift range, and there are two
#: different reasons for it: ``powerlaw`` is a simple scaling used as a null
#: test, so there is no fitted range to leave at all; ``sigmoid`` and
#: ``upturn`` are shape fits whose published range was not carried here when
#: the table was written and has not been looked up since.  Both come out as
#: "no range to check", which is the truth, and the second reason is recorded
#: in the notes so it can be closed by reading the papers rather than by
#: guessing.  Inventing a plausible range would have made the check *look*
#: complete while asserting a calibration nobody verified -- the precise error
#: this registry exists to prevent.
F_GAS_CALIBRATION = {
    "sigmoid": Calibration(
        fit="FLAMINGO group-scale suppression", z_range=None,
        notes="the production default; FLAMINGO's fitted redshift range is "
              "not recorded here yet"),
    "powerlaw": Calibration(
        fit="simple scaling", z_range=None,
        notes="a null test, not a fit -- there is no calibrated range to leave"),
    "upturn": Calibration(
        fit="CGM census at low mass", z_range=None,
        notes="when a single sigmoid is too steep; the census redshift range "
              "is not recorded here yet"),
}


def make_f_gas(name: str, z=None, calibration: str = "warn"):
    """Look up a parametrised baryon fraction by name.

    ``z`` is optional.
    Every other sector registry is read from a ``weights()`` that receives a
    :class:`~ggah_mod.halos.field.HaloField` and therefore a redshift; nothing
    in this package consumes :data:`F_GAS`, so this factory is the only place
    one of these names is resolved, with no field within reach.
    A caller who knows the redshift can pass it and get the calibration check;
    without it, no check is made.

    The check has nothing to report yet: every entry in
    :data:`F_GAS_CALIBRATION` carries ``z_range=None``, because the table never
    recorded a range.
    Filling those ranges in is the only step left to make it active.
    """
    key = str(name).lower()
    if key in F_GAS_CALIBRATION and z is not None:
        check_sector_calibration("baryon-fraction shape", key,
                                 F_GAS_CALIBRATION, z, policy=calibration)
    if key not in F_GAS:
        raise ValueError(f"unknown baryon-fraction model {name!r}; expected "
                         f"one of {sorted(F_GAS)}")
    return F_GAS[key]


# =========================================================================
# The container this module did not have
# =========================================================================

@sector_params
class EnergeticsParams(SectorParams):
    r"""The feedback budget's freedoms, with bounds and reasons.

    The numbers that decide how many baryons a halo keeps, including the two
    couplings the closure makes measurable, each with a bound, a prior and a
    reason (``PLAN.md`` item C2).
    The paper's reason: "epsilon_AGN is the sharpest case, because
    Sec. energetics' entire argument is that it is measurable."

    One of these is another module's parameter, shared by object identity:
    :data:`~ggah_mod.sectors.ejecta.ETA_EJ` is the ejection radius, which the
    ejected-baryon profile also uses, so the radius the gas is put at and the
    radius the closure charges for moving it are one number (item C4).
    Two containers holding equal copies would drift the moment one is edited;
    :data:`~ggah_mod.sectors.miscentering.P_OFF` follows the same idiom.
    :data:`~ggah_mod.sectors.agn.K_BOL` and ``f_duty`` were shared the same way
    until 0.8.8 removed the luminosity channel that read both.
    """

    # -- the two couplings the section exists to make measurable -------------
    log10_eps_agn: float = -1.252
    eps_sn: float = 0.1
    # -- the AGN channel's conversion factor ---------------------------------
    eps_radiative: float = 0.1
    # -- the supernova channel ------------------------------------------------
    return_fraction: float = RETURN_FRACTION
    # -- where the expelled gas goes, and the floor on what is kept ----------
    eta_ej: float = 2.0
    f_retained_min: float = 0.01
    # -- what happens to the gas that stays ----------------------------------
    eta_min: float = 0.6
    log10_m_eta: float = 13.0
    beta_eta: float = 1.5
    eta_w_norm: float = 0.0
    alpha_w: float = 1.0

    _STATIC = ()

    _PARAMS = {
        "log10_eps_agn": Param(
            -1.252, (-2.0, 0.0), Flat(), "",
            "log10 of the fraction of the AGN's radiated energy that couples "
            "to the halo gas. Zero is total coupling and is the definitional "
            "ceiling. The default, 0.056 (0.045-0.071), is the joint fit of the "
            "closure to X-ray gas fractions and the ACT kSZ profiles of DESI "
            "LRGs and BOSS CMASS (ggah_cal docs/closure_calibration.rst); the "
            "gas fractions alone want the bottom of the box. CAMELS bounds it "
            "from above (docs/agn_energy_simulations.rst): the simulations "
            "inject gross energy, 0.15 for EAGLE and the epsilon_f of OWLS, "
            "EAGLE and FLAMINGO. The box, 0.01 to 1, holds both. A reasoned "
            "Flat and not a prior: this is the number the sector exists to "
            "make measurable", "physical"),
        "eps_sn": Param(
            0.1, (0.0, 1.0), Flat(), "",
            "the fraction of the supernova energy budget that couples to the "
            "halo gas. A fraction, so [0, 1] is definitional at both ends. "
            "About 10 per cent of the explosion energy survives a remnant's "
            "radiative phase (Thornton et al. 1998); hot outflows carry 10-20 "
            "per cent at launch in resolved ISM simulations (Kim et al. 2020) "
            "and 1-20 per cent is observed in outflow kinetic energy (Chisholm "
            "et al. 2017). The coupling falls with halo mass (Pandya et al. "
            "2021; Li & Bryan 2020), which a constant cannot follow. The range "
            "is deliberately not imposed, for the same reason as "
            "log10_eps_agn", "definitional"),
        "eps_radiative": Param(
            0.1, (0.01, 0.4), Flat(), "",
            "the radiative efficiency of accretion, L = eps_r * Mdot c^2. "
            "Bounded by the black hole's spin: 0.057 for a Schwarzschild hole "
            "and 0.42 for a maximally spinning Kerr one, so the box is the "
            "physics of the accretion disc rather than a fit. The Soltan "
            "argument's own value is near 0.1", "physical"),
        "return_fraction": Param(
            RETURN_FRACTION, (0.2, 0.6), Flat(), "",
            "the fraction of stellar mass returned to the ISM, so supernovae "
            "count M_*/(1-R) rather than M_*. 0.41 for a Chabrier (2003) IMF "
            "under instantaneous recycling (Madau & Dickinson 2014), 0.40-0.47 "
            "with yields and metallicity (Vincenzo et al. 2016) and 0.48-0.50 "
            "at 10 Gyr (Leitner & Kravtsov 2011; Moster et al. 2013). The box "
            "also spans Salpeter (0.27-0.31) and top-heavy IMFs", "physical"),
        "eta_ej": ETA_EJ,
        "f_retained_min": Param(
            0.01, (0.0, 0.5), Flat(), "",
            "the floor on the retained baryon fraction, in the same units as "
            "f_b^cosmic itself -- a fraction of the HALO mass, not of f_b. The "
            "saturation is soft_saturate(..., f_b_cosmic - f_retained_min), so "
            "retained approaches this value and never reaches it. At Planck "
            "2018's f_b = 0.159 the default 0.01 is 6.3% of f_b, not 1% of it. "
            "It is what stops the saturation driving a halo to exactly zero "
            "baryons, which no halo does and which would make the gas "
            "profile's normalisation singular. Zero is allowed and is the "
            "unregularised model", "physical"),
        "eta_min": Param(
            0.6, (0.1, 1.0), Flat(), "",
            "the floor of the gas-to-dark-matter concentration ratio at low "
            "halo mass. One is definitional at the top -- above it feedback "
            "would be *contracting* the gas, which is the opposite of what the "
            "term models. **Calibrated on MillenniumTNG's c_hydro/c_DMO** "
            "(Sorini et al. 2025, MNRAS 536, 728); the source was recovered in "
            "PLAN.md item G8 and the module had recorded the calibration "
            "without the name", "physical"),
        "log10_m_eta": Param(
            13.0, (11.0, 15.0), Flat(), "log10(Msun/h)",
            "the halo mass at which the gas concentration is halfway between "
            "its floor and unity. Bounded by the range over which the "
            "MillenniumTNG measurement resolves a trend", "physical"),
        "beta_eta": Param(
            1.5, (0.5, 5.0), Flat(), "",
            "the sharpness of that transition. Below ~0.5 it is flat across "
            "the whole grid and the pivot stops meaning anything; above ~5 it "
            "is a step and the two regimes stop overlapping", "physical"),
        "eta_w_norm": Param(
            0.0, (0.0, 10.0), Flat(), "",
            "the supernova wind loading at v_c = 200 km/s, applied as "
            "eta -> eta/(1+eta_w). **Zero is the fiducial and is exactly the "
            "identity**, so the term ships off and switching it on is a strict "
            "extension. What is missing is a value to switch it on to: "
            "Muratov et al. (2015) is named in the predecessor's notes as the "
            "intended anchor and was never carried across. PLAN.md item G6",
            "physical"),
        "alpha_w": Param(
            1.0, (0.0, 3.0), Flat(), "",
            "the wind loading's velocity index, eta_w ~ v_c^-alpha_w. One is "
            "momentum-driven and two is energy-driven, which are the two "
            "regimes the literature describes; the box admits both and the "
            "range between them", "physical"),
    }
