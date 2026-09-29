r"""Where the baryons went, and what pushed them.

A halo with feedback holds less gas than its share of the cosmic baryon budget.
Two ways to say how much less:

* **Parameterise it.**  Write :math:`f_{\rm gas}(M)` as a sigmoid and fit the
  pivot mass.  Honest, and says nothing about why.
* **Close it energetically.**  The missing baryons were pushed out against the
  halo's binding energy, and the energy came from somewhere -- accreting black
  holes and supernovae.  Requiring the two to balance turns the retained
  fraction from a fitted function into a *prediction* of the AGN and stellar
  sectors.

Both are here.  The second is the point of the layer: the black-hole masses
the AGN sector builds its X-ray luminosities on are the **same quantity** that
sets the feedback energy, so X-ray data constrains the budget and thereby the
lensing suppression.  One chain ties two observables that are otherwise fitted
apart.

.. math::

    E_{\rm bind}(M) &= \Delta f_b\,M\,v_\Delta^2\,(1 - \eta_{\rm ej}^{-1}) \\
    E_{\rm AGN}(M)  &= \epsilon_{\rm AGN}\,\epsilon_r\,M_{\rm BH}(M)\,c^2 \\
    E_{\rm SN}(M)   &= \epsilon_{\rm SN}\,\frac{M_*(M)}{1-R}\,E_{\rm SN}/M_\odot \\
    f_{\rm ret}(M)  &= f_b^{\rm cosmic} - \Delta f_b

The AGN term is the energy the black hole released while it grew
(:func:`e_agn_soltan`).  The predecessor's form, a present X-ray luminosity
extrapolated over a Hubble time, :math:`\epsilon_{\rm AGN}f_{\rm duty}
k_{\rm bol}L_X t_H`, balanced an instantaneous rate against a cumulative
supernova budget; it was kept as a second channel for comparison and removed in
0.8.8, when nothing called it.

The two routes do not compute the same quantity
------------------------------------------------

They were treated as interchangeable, and are not.  The closure balances the
energy to displace **baryons**, so :func:`f_retained_energy` returns every
baryon still in the halo -- stars included.  The parameterised forms were
calibrated against FLAMINGO, which measures a **gas** fraction.  The difference
is :math:`f_\star`, up to 0.028 of the halo mass, and it was being added twice:
the caller took ``f_gas_energy`` and then added :math:`f_\star` on top of a
number that already contained it.  Measured at the fiducial parameters that put
:math:`f_\star + f_{\rm gas}` above :math:`f_b^{\rm cosmic}` on 143 of 256 mass
nodes, worst by a factor 1.0405 at :math:`10^{13.3}\,M_\odot/h` -- a halo holding
four per cent more baryons than exist, with the excess taken out of the dark
matter, where nothing was looking.

So the names differ now: :func:`f_retained_energy` against the ``f_gas_*``
family, feeding
:meth:`~ggah_mod.sectors.matter.BaryonSplit.from_retained` and
:meth:`~ggah_mod.sectors.matter.BaryonSplit.from_hot` respectively.  This also
withdraws a comparison the layer used to invite: the factor of ten between the
two routes at the group scale is partly a difference of definition, and only the
remainder is a disagreement about physics.

What this sets, and what it deliberately does not
--------------------------------------------------

The closure predicts the gas **mass** fraction, and nothing else.  The X-ray and
tSZ amplitudes stay free parameters of :mod:`ggah_mod.sectors.gas`.

That separation is load-bearing, not caution.  The DPM temperature is
:math:`T = P_e/n_e`, and :math:`P_e` does not depend on the electron-density
normalisation.  Rescaling :math:`n_{e,0.3}` to match an energy-predicted
:math:`f_{\rm gas}` would move **every halo's temperature** as :math:`1/A` and
silently de-calibrate the :math:`kT`--:math:`M` relation -- and the X-ray
emissivity would then scale as :math:`A^{2-p}`, with :math:`p` the local slope
of :math:`\Lambda(T)`, rather than as :math:`A^2`.  A closure aimed at lensing
would have quietly rewritten the X-ray calibration.

So :math:`f_{\rm gas}` has one definition -- this one -- and the gas sector's own
profile integral is a *diagnostic*: the ratio of the two is a measured
statement about whether the X-ray amplitude and the feedback budget agree.  It
is reported, not imposed.

The clamps, and why they had to go
-----------------------------------

The predecessor wrote the saturation as ``jnp.minimum(f_b - f_floor, x)`` and
``jnp.maximum(x, 0)``.  Both are ties, and at the fiducial parameters the AGN
channel *saturates* -- so the model sits exactly on the clamp, where JAX splits
a ``minimum`` gradient 50/50 between its arguments and the derivative on the far
side is zero.  The symptom is recorded in the predecessor's own documentation:
``eps_sn`` is "a flat direction at the saturated fiducial".  It is not flat; its
gradient was being discarded.

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
#: event per :math:`\sim100\,M_\odot` of stars, for a standard IMF.
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
#: (2003) IMF integrated over a Hubble time.
#:
#: **The IMF is cited; the 0.4 is not.**  Chabrier (2003), PASP 115, 763, is
#: the initial mass function.  The return fraction that follows from it depends
#: on the population-synthesis code and the age assumed, is commonly quoted
#: between 0.35 and 0.45, and the value here has not been traced to a
#: measurement that was read.  It is a `Param`-worthy quantity sitting as a
#: module constant, and it should be pinned to a source before any number
#: computed from it is published.
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
    r""":math:`v_\Delta^2 = GM/r_\Delta` [(km/s)^2], with a **physical** radius.

    Two corrections live in this one line, and they were found four months
    apart.

    The first is the redshift.  An earlier version built :math:`r_\Delta` from
    the comoving critical density and used it directly, which understates
    :math:`v^2` by :math:`(1+z)` -- 6 per cent at :math:`z = 0.135`, 100 per
    cent at :math:`z = 1`, and exactly zero at :math:`z = 0`, which is where it
    was checked.  A binding energy is :math:`GM/r` with :math:`r` the *proper*
    radius.

    The second is the mass definition.  This hard-coded :math:`200\rho_{\rm c}`
    whatever the halo field declared -- the same omission the gas sector
    carried, and the mass function before it.  A binding energy computed at a
    boundary the rest of the package does not use is not the binding energy of
    the halo anyone else is describing, and at the shipped 200m it is too large
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

    Kept because the old name says ``200`` and the quantity no longer does;
    a caller who wrote ``v200`` meant the halo's boundary, and now gets it.
    """
    return v_delta_squared(m, z, cosmo, mdef)


# =========================================================================
# Parameterised baryon fractions
# =========================================================================

@jax.jit
def f_gas_sigmoid(log10m, f_b_cosmic, log10_m_pivot=13.5, beta_b=1.5,
                  f_gas_min=0.01):
    r""":math:`f_{\rm gas} = f_b^{\rm cosmic}/[1 + (M_{\rm piv}/M)^{\beta_b}]`.

    The production form.  Pivot calibrated against FLAMINGO; the floor is a
    CGM-census lower bound rather than a numerical guard.
    """
    x = jnp.power(10.0, log10_m_pivot - jnp.asarray(log10m))
    return f_gas_min + (f_b_cosmic - f_gas_min) / (1.0 + jnp.power(x, beta_b))


@jax.jit
def f_gas_powerlaw(log10m, f_b_cosmic, log10_m_ref=14.0, alpha_b=0.3,
                   f_gas_min=0.0):
    r""":math:`f_{\rm gas} = f_b^{\rm cosmic}(M/M_{\rm ref})^{\alpha_b}`, saturating.

    The ceiling uses :func:`soft_saturate` rather than a ``clip``: the
    predecessor clipped, which puts a tie exactly where massive halos sit.
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
def e_agn_soltan(m_bh, log10_eps_agn=-2.0, eps_radiative=0.1):
    r"""AGN energy from the black-hole rest mass [(mass unit of ``m_bh``) (km/s)^2].

    .. math::

        E_{\rm AGN} = \epsilon_{\rm AGN}\,\epsilon_r\,M_{\rm BH}\,c^2

    The Soltan form, and the one the predecessor's own comment specifies -- "a
    coupling fraction of the BH rest mass (M_BH c^2)" -- before allocating
    :data:`C2_KMS2` for it and never using it.

    It differs from the predecessor's luminosity form,
    :math:`\epsilon_{\rm AGN}f_{\rm duty}k_{\rm bol}L_X t_H`, in a way that is
    not a refinement.  That one is a *present* luminosity extrapolated over a
    Hubble time; this is the energy the black hole released while it grew.  The
    supernova channel is cumulative, so balancing it against an instantaneous
    quantity is not a budget -- two clocks in one equation.  Which is also why
    the duty cycle and the bolometric correction disappear here: neither is a
    property of the integrated history.  The luminosity form was kept as a
    second channel until 0.8.8 and removed then, when nothing called it.

    Order of magnitude, and why the substitution mattered: at
    :math:`\epsilon_r = 0.1` this is :math:`1.8\times10^{53}` erg per solar mass
    of black hole against :math:`10^{49}` erg per solar mass of stars, so at
    :math:`M_{\rm BH}/M_\star \sim 10^{-3}` the AGN channel *releases* ~20 times
    the supernova energy.  The measured
    :math:`E_{\rm AGN}/E_{\rm SN} \sim 0.01\text{-}0.03` at the published
    couplings is a property of the luminosity form, not of the physics.

    Parameters
    ----------
    m_bh : array (NM,)
        The *mean* black-hole mass, not ten to the mean log, which is a factor
        1.40 at the published scatter.  The energy carries its mass unit, so
        set against :func:`binding_energy` and :func:`e_supernova`, whose
        masses are in :math:`M_\odot/h`, it has to be in :math:`M_\odot/h`
        too.  :meth:`~ggah_mod.sectors.agn.AgnSector.mean_mbh` returns
        physical :math:`M_\odot`, and :func:`f_retained_energy` converts.
    log10_eps_agn : float
        :math:`\log_{10}` of the mechanical coupling to the gas.
    eps_radiative : float
        Accretion radiative efficiency.  0.1 is the standard thin-disc value
        and the one the Soltan argument is usually quoted at.
    """
    return (jnp.power(10.0, log10_eps_agn) * eps_radiative
            * jnp.asarray(m_bh) * C2_KMS2)


@jax.jit
def e_supernova(m_star, eps_sn=0.1, return_fraction=RETURN_FRACTION):
    r"""Supernova energy [Msun (km/s)^2], from the mass **formed**.

    .. math::

        E_{\rm SN} = \epsilon_{\rm SN}\,\frac{M_\star}{1-R}\,
                     \frac{E_{\rm SN}}{M_\odot}

    Cumulative, which is the answer to the objection that quiescent galaxies
    stop having supernovae: they do, and it does not matter, because this is
    the energy every star ever formed has already released.  A red/blue split
    would be needed for a *rate* and is not needed here.

    What does matter is that :math:`M_\star` is the **surviving** mass -- living
    stars plus remnants -- while supernova counts scale with the mass formed.
    Dividing by :math:`1-R` is a factor 1.67 that was simply missing.  See
    :math:`e_{\rm SN} = 5\times10^5\,({\rm km/s})^2 \approx 10^{49}` erg per
    solar mass formed is one supernova per ~100 M_sun at :math:`10^{51}` erg:
    Chabrier (2003) for the IMF, with Dekel & Silk (1986) and Somerville & Dave
    (2015) Sect. 3 for the budget's use.  Traced through the predecessor rather
    than re-derived here -- ``PLAN.md`` item **G8**.

    :data:`RETURN_FRACTION`, and note it is *not*
    :data:`~ggah_mod.sectors.sham.REMNANT_FRACTION`, which is a fraction of a
    different denominator.

    What is still wrong, and is not repaired here: the energy was injected while
    the stars were forming, into progenitors far less bound than the halo it is
    balanced against.  That needs a star-formation history, which this package
    does not have, and it is the one place a quiescent/star-forming distinction
    would legitimately enter.
    """
    return (eps_sn * jnp.asarray(m_star) / (1.0 - return_fraction)
            * E_SN_PER_MSUN_KMS2)


def binding_energy(m, z, cosmo, delta_f_b=1.0, eta_ej=None, mdef=None):
    r"""Energy to move :math:`\Delta f_b M` out to :math:`r_{\rm ej}` [Msun (km/s)^2].

    .. math::

        E_{\rm bind} = \Delta f_b\,M\,v_\Delta^2
                       \left(1 - \frac{1}{\eta_{\rm ej}}\right)

    :math:`E_{\rm bind} = \Delta f_b M v_\Delta^2` carries no citation in this
    package or its predecessor.  Traced: Silk & Rees (1998); Wu, Fabian & Nulsen
    (2000); Bower, McCarthy & Benson (2008); McCarthy et al. (2011) -- and for
    the modern statement that :math:`f_{\rm CGM}` is set by *integrated* black
    hole feedback energy rather than by its instantaneous rate, **Davies, Crain,
    McCarthy et al. (2019), MNRAS 485, 3783**, which is the argument the Soltan
    channel of :func:`e_agn_soltan` implements.  ``PLAN.md`` item **G8**; these
    are the sources the form came from, traced rather than re-derived.

    With ``eta_ej=None`` the bracket is 1, which is the work to unbind the gas
    **to infinity** -- what the closure assumed, and an overestimate whenever
    the gas ends up at a finite radius.  In a point-mass potential the work to
    move mass from :math:`R_\Delta` to :math:`\eta_{\rm ej}R_\Delta` is smaller
    by exactly that bracket: at :math:`\eta_{\rm ej} = 2` it is half.

    Supplying it ties this module to
    :class:`~ggah_mod.sectors.ejecta.EjectaSector`'s radius, so
    :math:`f_{\rm ejected}` and :math:`r_{\rm ej}` stop being two free
    quantities and become one prediction.  A point mass is the crude part: the
    real potential is the halo's, and its logarithmic tail makes the true
    bracket approach 1 more slowly.  Named rather than hidden.
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
    r"""The **baryon** fraction a halo kept, predicted from the feedback budget.

    Renamed from ``f_gas_energy``, and the rename is the point.  What this
    balances is the energy to displace *baryons* against the binding energy, so
    what it returns is :math:`f_b^{\rm cosmic} - \Delta f_b` -- every baryon
    still in the halo, stars included.  It is not a gas fraction, and calling it
    one meant the caller added :math:`f_\star` on top of a quantity that already
    contained it.  Measured at the fiducial parameters, that put
    :math:`f_\star + f_{\rm gas}` above :math:`f_b^{\rm cosmic}` on 143 of 256
    mass nodes, worst by a factor 1.0405 at :math:`10^{13.3}\,M_\odot/h`.
    :meth:`~ggah_mod.sectors.matter.BaryonSplit.from_retained` is the consumer
    that takes it by its right name.

    **The coefficients come from** :class:`EnergeticsParams` **now**, with the
    keywords kept as per-call overrides.  Declaring a container and leaving
    nothing to read it is the defect :func:`wind_mass_loading` already carries
    and ``PLAN.md`` item **G6** records: a container is a claim about what the
    freedoms are, and the claim is only true if the function agrees.  ``eta_ej``
    keeps its own meaning for ``None`` -- no finite-radius correction -- which
    is why absence is a sentinel here rather than ``None``.

    The parameterised forms above are the other case: FLAMINGO measures a *gas*
    fraction, so they feed
    :meth:`~ggah_mod.sectors.matter.BaryonSplit.from_hot`.  The two routes are
    therefore not two estimates of one quantity, which is worth knowing before
    reading anything into the factor of ten between them at the group scale.

    Parameters
    ----------
    m : array (NM,) [Msun/h]
    z : float
    cosmo : Cosmology
    m_star : array (NM,) [Msun/h]
        Stellar mass per halo, from the occupation sector's SHMR.  The
        **surviving** mass; :func:`e_supernova` converts.
    f_b_cosmic : float
        :math:`\Omega_b/\Omega_{cb}`, from
        :func:`~ggah_mod.sectors.matter.cosmic_baryon_fraction`: the halo mass
        is cold mass.
    m_bh : array (NM,) [Msun], **physical**
        Mean black-hole mass per halo, as
        :meth:`~ggah_mod.sectors.agn.AgnSector.mean_mbh` returns it.  Required:
        the AGN term is :func:`e_agn_soltan`, the energy released while the hole
        grew.  Keyword-only and ``None`` by default so that a call without it is
        refused by name rather than by a missing positional.

        **Converted here to** :math:`M_\odot/h`, the unit of ``m`` and
        ``m_star`` and so of the binding energy and the supernova budget it is
        balanced against.  Until 0.9.0 it went in as it came, so
        :math:`E_{\rm AGN}` was :math:`1/h = 1.48` too large against both at
        Planck 2018.  The luminosity channel's Hubble time carried an
        :math:`h` through the closure and hid it; with the Soltan form alone,
        :math:`\partial f_{\rm ret}/\partial h` at fixed inputs was exactly
        zero, which is how it was found.
    log10_eps_agn : float
        :math:`\log_{10}` of the AGN mechanical coupling efficiency -- the
        fraction of the released energy that couples to the gas.  **Its own
        parameter.**  The predecessor reused the baryon sigmoid's pivot-mass
        slot for it, so the two meanings shared one number and one prior.
    eps_radiative : float
        Accretion radiative efficiency.
    eps_sn : float
        Supernova coupling efficiency.
    return_fraction : float
        Mass returned to the ISM per unit mass formed; see
        :data:`RETURN_FRACTION`.
    eta_ej : float, optional
        :math:`r_{\rm ej}/R_\Delta`.  ``None`` means ejection to infinity,
        which is what the closure used to assume; supplying it makes the
        binding energy the work actually done.  See :func:`binding_energy`.
    f_retained_min : float
        Floor on the retained fraction, in halo-mass units like ``f_b_cosmic``
        and **not** as a fraction of it.  Approached asymptotically, never
        clipped to.
    m_cold : array (NM,) [Msun/h], optional
        Neutral gas per halo, HI + H2 with helium -- ``ColdGasSector.m_neutral``.
        **It cannot be expelled either** (0.9.5): the budget unbinds the hot
        atmosphere, and gas cold enough to be atomic or molecular sits in the
        galaxies' discs.  Without it the cap below counted the neutral gas as
        expellable, so where the closure keeps less than the stars and the
        neutral gas together the hot gas implied was negative -- -0.06 of
        :math:`f_b` at 1e10.6-1e11.3 Msun/h with ``catinella18`` at the LS10
        defaults, z = 0.  ``None`` is 0.9.4 bit for bit.
    return_expelled : bool
        When true, return ``(f_retained, f_expelled)`` instead of just the
        first.  ``f_expelled`` was computed here and thrown away by the
        ``return`` -- and it is exactly the ejected component the matter budget
        needs, so recomputing it as ``f_b - f_retained`` at the call site would
        be one subtraction that could drift from this one.

    Notes
    -----
    Returns a value in
    :math:`(f_\star + f_{\rm cold} + f_{\rm retained,min},\
    f_b^{\rm cosmic}]`, strictly (:math:`f_{\rm cold} = 0` without
    ``m_cold``) --
    the saturation is exponential, so neither end is ever exactly attained and
    the gradient survives at both.  **Stars cannot be expelled**, so the floor
    sits above the stellar fraction: the retained gas is at least
    :math:`f_{\rm retained,min}`.
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

    Feedback does not only *remove* gas, it redistributes what remains outward.
    Calibrated on the hydrodynamic-to-dark-matter-only concentration ratio of
    **Sorini, Bose, Pakmor, Hernquist, Springel, Hadzhiyska, Hernandez-Aguayo &
    Kannan (2025), MNRAS 536, 728** -- MillenniumTNG's :math:`c_{\rm hydro}/
    c_{\rm DMO}` from dwarfs to superclusters, which is where ``eta_min``,
    ``log10_m_eta`` and ``beta_eta`` come from.

    **The name was missing and the sentence was not.**  This docstring said
    "calibrated on the ..." with no source, so three shipped constants rested on
    an unnamed measurement -- the same shape as ``T_B_COEF_MK``, which said it
    was recalled rather than read.  Recovered from the predecessor's own test
    suite while the census thread's record was being folded into ``PLAN.md``,
    and nearly lost with it; ``PLAN.md`` item **G8**.

    Recorded as *the source the numbers came from*, traced through the
    predecessor, rather than as a citation checked against the paper -- which is
    the distinction ``T_B_COEF_MK`` was flagged for not making.
    Distinct from :math:`f_{\rm gas}`, and the two are not interchangeable: one
    changes how much gas there is, the other where it sits.
    """
    x = jnp.power(10.0, jnp.asarray(log10m) - log10_m_eta)
    return 1.0 - (1.0 - eta_min) / (1.0 + jnp.power(x, beta_eta))


def wind_mass_loading(m, z, cosmo, eta_w_norm=0.0, alpha_w=1.0):
    r"""Supernova wind loading :math:`\eta_w = \eta_0 (v_c/200)^{-\alpha_w}`.

    Applied as :math:`\eta \to \eta/(1+\eta_w)`, puffing low-mass halos further.
    :math:`\alpha_w = 1` is momentum-driven and 2 is energy-driven.  At the
    fiducial :math:`\eta_0 = 0` this is exactly the identity, so switching it on
    is a strict extension rather than a different model.
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
    """Look up a parameterised baryon fraction by name.

    ``z`` is optional and the reason is worth stating.  Every other sector
    registry is read from a ``weights()`` that receives a
    :class:`~ggah_mod.halos.field.HaloField` and therefore a redshift; nothing
    in this package consumes :data:`F_GAS`, so this factory is the only place
    one of these names is ever resolved and there is no field within reach.  A
    caller who knows the redshift can pass it and get the check; one who does
    not gets what this function always did.

    That check currently has nothing to say --- every entry in
    :data:`F_GAS_CALIBRATION` carries ``z_range=None``, because the table never
    recorded a range.  The hook is here so that filling those ranges in is the
    only remaining step, rather than filling them in *and* finding somewhere to
    read them from.
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

    This module had no container at all, so thirteen numbers that decide how
    many baryons a halo keeps were keyword defaults -- including the two the
    section's whole argument is about.  ``PLAN.md`` item **C2**, and the paper's
    reason for wanting it: *"epsilon_AGN is the sharpest case, because
    Sec. energetics' entire argument is that it is measurable."*

    **One of these is another module's parameter, shared by object identity.**
    :data:`~ggah_mod.sectors.ejecta.ETA_EJ` is the ejection radius, which the
    ejected-baryon profile also uses -- so the radius the gas is put at and the
    radius the closure charges for moving it are one number rather than two
    that happen to agree (item **C4**).  :data:`~ggah_mod.sectors.agn.K_BOL`
    was shared the same way, and ``f_duty`` sat beside it, until 0.8.8 removed
    the luminosity channel that read both.

    Sharing the object rather than the number is the point: two containers
    holding equal copies drift the moment one is edited, and
    :data:`~ggah_mod.sectors.miscentering.P_OFF` is the idiom.
    """

    # -- the two couplings the section exists to make measurable -------------
    log10_eps_agn: float = -2.0
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
            -2.0, (-4.0, 0.0), Flat(), "",
            "log10 of the fraction of the AGN's radiated energy that couples "
            "to the halo gas. Zero is total coupling and is the definitional "
            "ceiling; 1e-4 is two orders below any value that produces a "
            "visible effect on f_gas, so below it the channel is off. A "
            "reasoned Flat and not a prior: this is the number the sector "
            "exists to make measurable, and giving it a Gaussian here would "
            "answer the question the campaign is meant to ask", "physical"),
        "eps_sn": Param(
            0.1, (0.0, 1.0), Flat(), "",
            "the fraction of the supernova energy budget that couples to the "
            "halo gas. A fraction, so [0, 1] is definitional at both ends; the "
            "commonly quoted range is 0.05-0.3 and it is deliberately not "
            "imposed, for the same reason as log10_eps_agn", "definitional"),
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
            "count M_*/(1-R) rather than M_*. **The value is untraced**: "
            "Chabrier (2003) is cited correctly for the IMF and is in the "
            "bibliography, and the 0.4 does not come from it. Commonly quoted "
            "between 0.35 and 0.45; the box is wider than that because a "
            "constant whose source has not been read should not carry a tight "
            "bound. PLAN.md item G3", "physical"),
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
