"""Physical constants and unit conventions, in one place.

Every number here is a *definition* or a CODATA/Planck-collaboration value, not
a fitted parameter.  Nothing in this module depends on anything else in the
package, so it can be read first.

Unit conventions for the whole package
--------------------------------------

=============================  ==========================  ==============================
Quantity                       Symbol                      Unit
=============================  ==========================  ==============================
Comoving separation            :math:`r,\\ r_p`             Mpc/h
Halo mass                      :math:`M`                   :math:`M_\\odot/h`
Power spectrum                 :math:`P(k)`                :math:`(\\mathrm{Mpc}/h)^3`
Wavenumber                     :math:`k`                   :math:`h\\,\\mathrm{Mpc}^{-1}`
Number density                 :math:`n`                   :math:`(\\mathrm{Mpc}/h)^{-3}`
Distances from ``background``  :math:`\\chi`                Mpc/h
=============================  ==========================  ==============================

The one exception is :attr:`Cosmology.hubble_distance <ggah_mod.cosmology.parameters.Cosmology.hubble_distance>`, which
is quoted in Mpc as well as Mpc/h because both spellings are needed and getting
them confused is a factor of ``h``.
"""

from __future__ import annotations

import math

import numpy as _np

# -- exact / defined --------------------------------------------------------
#: Speed of light [km/s]. Exact by definition of the metre.
C_KM_S = 299_792.458

#: Boltzmann constant [keV/K]. CODATA 2018.
K_B_KEV_PER_K = 8.617333262e-8

#: Boltzmann constant [eV/K].
K_B_EV_PER_K = 8.617333262e-5

#: Megaparsec in cm.
MPC_CM = 3.0856775814913673e24

#: Stefan-Boltzmann constant [W m^-2 K^-4].  Exact in the 2019 SI.
SIGMA_SB = 5.670374419e-8

#: Speed of light [m/s].  Exact.
C_M_S = 299_792_458.0

#: Newton's constant [m^3 kg^-1 s^-2].  CODATA 2018.
G_SI = 6.67430e-11

# -- critical density -------------------------------------------------------
#: Critical density today in h-units: :math:`\rho_{c,0} = 3H_0^2/8\pi G`
#: expressed as :math:`(M_\odot/h)/(\mathrm{Mpc}/h)^3`, so it carries no
#: residual ``h``.  2.77536627e11.
RHO_CRIT0 = 2.77536627e11

# -- the radiation background ----------------------------------------------
#: CMB temperature today [K]. Fixsen (2009).
T_CMB = 2.7255

#: :math:`\rho_{\rm crit}` at :math:`H_0 = 100` km/s/Mpc in kg m^-3,
#: :math:`3H_{100}^2/8\pi G`: the denominator of :data:`OMEGA_GAMMA_H2`.
_RHO_CRIT_100_SI = 3.0 * (1.0e5 / (MPC_CM / 100.0)) ** 2 / (8.0 * math.pi * G_SI)

#: :math:`\Omega_\gamma h^2` for a blackbody at :math:`T_{\rm CMB}`:
#: :math:`4\sigma_{\rm SB}T^4/c^3` over :math:`3H_{100}^2/8\pi G`,
#: 2.472975e-5.
#:
#: **Computed, not typed.**  It was the literal ``2.47282e-5`` with a comment
#: saying it matched astropy's ``Ogamma0`` to the digit; it sat 6.3e-5 below
#: this expression, which is what astropy, CLASS and CAMB all evaluate, and that
#: offset was the whole of the "6e-5, the fit's own accuracy" residual the
#: neutrino density used to be quoted against astropy with.  It also set the
#: neutrino mass convention through :data:`NU_DENOM_EV`, which is derived from
#: it: 93.1492 on the literal against CLASS's 93.1434.
OMEGA_GAMMA_H2 = 4.0 * SIGMA_SB * T_CMB ** 4 / C_M_S ** 3 / _RHO_CRIT_100_SI

#: Effective number of neutrino species at early times, the standard-model
#: value with non-instantaneous decoupling.  It is split between the three
#: massive states and a massless remainder as CLASS splits it: see
#: :data:`T_NCDM_OVER_T_GAMMA` and :data:`N_UR_REMAINDER`.
N_EFF = 3.044

#: Number of massive species the mass sum is split over.
N_NU_MASSIVE = 3

#: Neutrino temperature ratio :math:`T_\nu/T_\gamma = (4/11)^{1/3}` after
#: *instantaneous* electron-positron annihilation.  The reference the massive
#: states' temperature is quoted against; not their temperature.
T_NU_OVER_T_GAMMA = (4.0 / 11.0) ** (1.0 / 3.0)

#: Temperature of the three **massive** states in units of :math:`T_{\rm CMB}`:
#: CLASS's default ``T_ncdm``, whose ``input.c`` gives it as "the value that
#: gives m/omega = 93.14 eV".
#:
#: This is the neutrino convention, and choosing it here is what makes the
#: package agree with itself.  :math:`N_{\rm eff} = 3.044` exceeds three
#: because the neutrinos were heated a little by electron-positron
#: annihilation, and that excess has to go somewhere.  Until 0.9.8 it went into
#: the *degeneracy* -- each species carried :math:`N_{\rm eff}/3 = 1.0147` of
#: a species at :math:`(4/11)^{1/3}T_{\rm CMB}`, which is astropy's and
#: Komatsu et al.'s choice -- and that put the pressureless limit at
#: :math:`\Sigma m_\nu/(92.717\,{\rm eV}\,h^2)`, while the package quoted
#: :math:`\Sigma m_\nu/93.14` beside it and handed CLASS masses it integrates
#: at 93.143.  Three densities for one quantity, and the matter budget closed
#: on none of them for the Boltzmann codes: CLASS integrated
#: :math:`\Omega_m - 6.5\times10^{-6}` at the fiducial mass.
#:
#: CLASS puts the excess into the *temperature* instead: the massive states sit
#: at :math:`0.71611\,T_{\rm CMB}` with a degeneracy of one, which fixes their
#: number density -- and so the pressureless limit, :math:`\Sigma
#: m_\nu/(93.143\,{\rm eV}\,h^2)` -- and the relativistic density still short of
#: 3.044 is carried by a massless remainder, :data:`N_UR_REMAINDER`.  Both
#: limits are then right at once, and they are CLASS's own, and ``emu_pk``'s,
#: which was trained on CLASS called this way.
T_NCDM_OVER_T_GAMMA = 0.71611

#: :math:`\tfrac78(4/11)^{4/3}` -- one massless neutrino species' energy
#: density in units of the photons'.
#:
#: The 7/8 is Fermi-Dirac against Bose-Einstein; the :math:`(4/11)^{4/3}` is
#: :math:`(T_\nu/T_\gamma)^4` after electron-positron annihilation.  With
#: :data:`N_EFF` this gives the density the neutrinos would carry were all three
#: massless, :math:`\Omega_\nu^{\rm rel} = \tfrac78(4/11)^{4/3}N_{\rm
#: eff}\Omega_\gamma`, which is what they carry at early times whatever their
#: masses.  It does not vanish with the mass, so the massless limit is the
#: massless answer rather than zero.
NU_REL_COEF = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)

#: The massive states' share of :data:`N_EFF`,
#: :math:`3(T_{\rm ncdm}/T_\nu)^4 = 3.039605`: three species at
#: :data:`T_NCDM_OVER_T_GAMMA`, each counted relativistically.
N_MASSIVE_EFF = N_NU_MASSIVE * (T_NCDM_OVER_T_GAMMA / T_NU_OVER_T_GAMMA) ** 4

#: The massless remainder, :math:`N_{\rm eff} - 3(T_{\rm ncdm}/T_\nu)^4 =
#: 0.004395`: CLASS's ``N_ur`` when three massive states are declared.  It is
#: radiation at every redshift, and it is neutrino density, so it belongs to
#: :math:`\Omega_\nu^{\rm r}` and not to the photons.
N_UR_REMAINDER = N_EFF - N_MASSIVE_EFF

#: :math:`m c^2/(k_B T_{\rm ncdm,0})` for a 1 eV neutrino, at the massive
#: states' own temperature.  Scales as :math:`1/T_{\rm CMB}`.
NU_Y_PER_EV = 1.0 / (K_B_EV_PER_K * T_NCDM_OVER_T_GAMMA * T_CMB)

#: Apery's constant :math:`\zeta(3)`.
ZETA3 = 1.2020569031595942

#: The pressureless asymptote of the relic energy integral,
#: :math:`F(y) \to \kappa y` with :math:`\kappa = 180\zeta(3)/7\pi^4 =
#: 0.3173219`: the ratio of the number-density moment
#: :math:`\int x^2/(e^x+1) = \tfrac32\zeta(3)` to the energy moment
#: :math:`\int x^3/(e^x+1) = 7\pi^4/120`.
#:
#: Komatsu et al. (2011) wrote it to four figures, 0.3173, as the constant of a
#: fit :math:`[1+(\kappa y)^p]^{1/p}`; the package used that fit until 0.9.8.
#: The fit is exact only in the two limits, and at the fiducial mass it put the
#: neutrinos' kinetic energy today at 7.1e-4 of their rest mass where the
#: integral gives 4.6e-4.  :func:`~ggah_mod.cosmology.parameters.nu_energy_factor`
#: integrates instead, and this constant survives as what it is -- the
#: pressureless limit, exact.
NU_KAPPA = 180.0 * ZETA3 / (7.0 * math.pi ** 4)

#: :math:`\Sigma m_\nu/(\Omega_\nu^{\rm nr}h^2)` [eV]: the mass convention,
#: **derived** -- 93.1434 at :data:`T_CMB`.
#:
#: It was typed, ``93.14``, and a typed denominator beside a computed density
#: is two conventions: the package subtracted the computed one from
#: :math:`\Omega_m` and reported the typed one, 0.46 per cent apart, and
#: :math:`\Omega_\nu(0)/\Omega_\nu = 1.0053` was the ratio between them rather
#: than anything a neutrino does.  Here it is the pressureless limit of the
#: density :func:`~ggah_mod.cosmology.background.hubble_e` integrates, per unit
#: mass, so there is one number.  It equals CLASS's own, since the temperature
#: is CLASS's: m/omega = 93.14 eV in CLASS's words.
#:
#: Kept as a name because ``emu_pk`` restates it (``emu_pk.cosmo``), and the
#: seam test there compares the two floats exactly.
NU_DENOM_EV = N_NU_MASSIVE / (NU_REL_COEF * N_MASSIVE_EFF * OMEGA_GAMMA_H2
                              * NU_KAPPA * NU_Y_PER_EV)

# -- Neutrino mass splittings ------------------------------------------------
#
# The global fit of Esteban, Gonzalez-Garcia, Maltoni et al. (2024, JHEP 12,
# 216).  These are *measurements*, like every other number in this module, and
# what they buy is the three masses from the sum alone: with an ordering
# declared there is nothing left to fit, so `nu_masses` is a root-find and not
# a calibration.  The derivation lives in `parameters.py`; only the measured
# inputs are here.

#: Solar splitting :math:`\Delta m^2_{21} = m_2^2 - m_1^2` [eV^2].
#: :math:`7.49^{+0.19}_{-0.19}\times10^{-5}`.
DELTA_M2_21 = 7.49e-5

#: Atmospheric splitting :math:`\Delta m^2_{31} = m_3^2 - m_1^2` [eV^2] in a
#: **normal** ordering, :math:`+2.534^{+0.025}_{-0.023}\times10^{-3}`.
DELTA_M2_31 = 2.534e-3

#: Atmospheric splitting :math:`\Delta m^2_{32} = m_3^2 - m_2^2` [eV^2] in an
#: **inverted** ordering, :math:`-2.510^{+0.024}_{-0.025}\times10^{-3}`.  Kept
#: signed, because the sign *is* the ordering.
DELTA_M2_32 = -2.510e-3

#: The two squared offsets of the heavier states above the lightest, per
#: hierarchy, **sorted so the pair ascends** -- which is what makes the mass
#: triplet ascend in both orderings, and every consumer of it wants that:
#: CLASS's ``m_ncdm`` list, CAMB's ``nu_mass_fractions``, and ``emu_pk`` 2.0's
#: ordered simplex :math:`r_1 \le r_2 \le r_3`.
#:
#: Both orderings are one function of ``(A, B)``:
#: :math:`m = (x, \sqrt{x^2+A}, \sqrt{x^2+B})` with
#: :math:`x + \sqrt{x^2+A} + \sqrt{x^2+B} = \Sigma m_\nu`.  For an inverted
#: ordering the lightest state is :math:`m_3`, so ``A`` is
#: :math:`|\Delta m^2_{32}| - \Delta m^2_{21}` (that is :math:`m_1`) and ``B``
#: is :math:`|\Delta m^2_{32}|` (that is :math:`m_2`); swapping the pair would
#: silently return :math:`(m_3, m_2, m_1)`.
#:
#: ``degenerate`` and ``massless`` are ``(0, 0)``, so they are *values* of the
#: same expression rather than branches beside it: the split collapses to
#: :math:`\Sigma m_\nu/3` with no special case anywhere downstream.
NU_OFFSETS = {
    "normal":     (DELTA_M2_21, DELTA_M2_31),
    "inverted":   (abs(DELTA_M2_32) - DELTA_M2_21, abs(DELTA_M2_32)),
    "degenerate": (0.0, 0.0),
    "massless":   (0.0, 0.0),
}

#: The smallest :math:`\Sigma m_\nu` each hierarchy admits [eV], at zero
#: lightest mass: :math:`\sqrt A + \sqrt B`.
#:
#: **Derived, never written down.**  A floor typed as a literal beside the
#: splittings it is a function of is a pair that drifts.  These come out at
#: ``0.058993`` (normal) and ``0.099447`` (inverted); the technical paper's
#: Sec. 2.2.1 quotes ``0.058`` and ``0.098``, which are the rounded literature
#: values, and the paper is what is wrong.
#:
#: The inverted floor is **above the fiducial** :math:`\Sigma m_\nu = 0.06` eV,
#: so an inverted ordering at the fiducial mass is not a configuration to be
#: checked -- it is excluded by the oscillation data.
NU_MASS_FLOOR = {k: float(_np.sqrt(a) + _np.sqrt(b))
                 for k, (a, b) in NU_OFFSETS.items()}

#: Pivot in :math:`S_8 = \sigma_8\sqrt{\Omega_m/0.3}`.
S8_PIVOT_OMEGA_M = 0.3

#: Top-hat radius defining :math:`\sigma_8` [Mpc/h].
R8 = 8.0



#: Newton's constant in :math:`{\rm Mpc}\,({\rm km/s})^2/M_\odot`, derived
#: from :data:`RHO_CRIT0` rather than written down.
#:
#: :math:`\rho_{c,0} = 3H_0^2/8\pi G` with :math:`H_0 = 100\,h`, so
#: :math:`G = 3\times100^2/(8\pi\rho_{c,0})`.  Taking it from the same constant
#: the mass definitions use keeps :math:`v_\Delta^2` and :math:`r_\Delta`
#: mutually consistent; an independently rounded :math:`G` makes them disagree
#: at the 1e-4 level, which is small and exactly the kind of thing that never
#: gets found.
#:
#: :math:`h` cancels: a mass in :math:`M_\odot/h` over a radius in
#: :math:`{\rm Mpc}/h` is an :math:`h`-free velocity squared.
#:
#: It lived in :mod:`~ggah_mod.sectors.energetics` until the halo field needed
#: it for :attr:`~ggah_mod.halos.field.HaloField.v_delta_squared`, and layer 2
#: may not import layer 3.  A relocation and not a redefinition -- the same
#: expression, one layer down, the move :func:`~ggah_mod.numerics.soft_saturate`
#: already made.
G_MPC_KMS2_MSUN = 3.0 * 100.0 ** 2 / (8.0 * math.pi * RHO_CRIT0)

__all__ = [
    "C_KM_S", "K_B_KEV_PER_K", "K_B_EV_PER_K", "MPC_CM", "RHO_CRIT0",
    "G_MPC_KMS2_MSUN",
    "SIGMA_SB", "C_M_S", "G_SI",
    "T_CMB", "OMEGA_GAMMA_H2", "N_EFF", "N_NU_MASSIVE", "T_NU_OVER_T_GAMMA",
    "T_NCDM_OVER_T_GAMMA", "N_MASSIVE_EFF", "N_UR_REMAINDER",
    "NU_DENOM_EV", "NU_REL_COEF", "NU_Y_PER_EV", "ZETA3", "NU_KAPPA",
    "DELTA_M2_21", "DELTA_M2_31", "DELTA_M2_32", "NU_OFFSETS", "NU_MASS_FLOOR",
    "S8_PIVOT_OMEGA_M", "R8",
]
