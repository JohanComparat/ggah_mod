r"""Intrinsic alignments, as a sector: a peer, like everything else here.

A weak-lensing likelihood cannot be written without an intrinsic-alignment
model (``PLAN.md`` item E3), and layer 5 ships :math:`\Delta\Sigma` and
:math:`w(\theta)`.
This sector is the consumer of :math:`D(z)` that
:mod:`~ggah_mod.cosmology.growth`'s docstring names as "an
intrinsic-alignment amplitude".

The non-linear alignment model
------------------------------

The non-linear alignment model (Hirata & Seljak 2004; Bridle & King 2007).
A galaxy's shape responds to the tidal field it formed in, and the tidal field
traces matter, so to linear order the intrinsic-shear field is the matter field
times a number:

.. math::

    b_I(z) = -A_{\rm IA}\,f_{\rm red}
             \left(\frac{\langle M_h\rangle}{M_{h,{\rm piv}}}\right)^{\beta_M}
             \frac{C_1\rho_{\rm crit}\,\Omega_m}{D(z)}
             \left(\frac{1+z}{1+z_0}\right)^{\eta}
             \left(\frac{L}{L_0}\right)^{\beta}

The sign is physical, not a convention.
Galaxies align *along* the stretching axis of the tidal field, where lensing
shears *across* it, so the intrinsic and lensing shears anticorrelate:
:math:`P_{gI}` and the galaxy--galaxy-lensing signal have opposite signs, and an
alignment model that came out positive would add to the signal it exists to
subtract.
:func:`alignment_bias` returns a negative number at positive
:math:`A_{\rm IA}`, and a test asserts it.

Which galaxies align
--------------------

By default only red galaxies align (``population="red"``), the NLA-M model of
Wright et al. (2025, KiDS-Legacy, Eq. 11): the red fraction :math:`f_{\rm red}`
of the shape sample aligns with an amplitude that scales with the mean host
halo mass of its red galaxies, and the blue galaxies do not align.
Both trends are measured: only early-type galaxies give significant detections
on large scales, and their amplitude follows a power law in halo mass (Fortuna
et al. 2025).
The halo model supplies both numbers on the field: :math:`f_{\rm red}` is
:math:`\bar n` of ``zumandelbaum16_red`` over :math:`\bar n` of
``zumandelbaum15`` at the shape sample's galaxy parameters, and
:math:`\log_{10}\langle M_h\rangle` is the number-weighted mean of
:math:`\log_{10}M` over the red occupation, centrals and satellites, as
Wright et al. average logarithms (their App. B).
Either can be given instead (``f_red``, ``log10_mh`` in the parameters), for
instance a survey's own measurement.
The defaults of :class:`IaParams`, :math:`A_{\rm IA} = 5.74` and
:math:`\beta_M = 0.44` at :math:`M_{h,{\rm piv}} = 10^{13.5}\,h^{-1}M_\odot`, are
the fit of Fortuna et al. (2025) to early-type alignments that Wright et al.
adopt as their prior (correlation coefficient -0.59, which one-parameter priors
cannot carry).
``population="all"`` is the plain NLA: every galaxy aligns, with
:math:`f_{\rm red} = 1` and no mass term, and :math:`A_{\rm IA}` is then the
sample's amplitude, of order one in cosmic-shear fits, not the red galaxies'.

A tracer, not a correction
--------------------------

The intrinsic-shear field is the matter field scaled, so
:meth:`IntrinsicAlignmentSector.weights` returns
:func:`~ggah_mod.sectors.matter.matter_weights` multiplied by :math:`b_I`, and
layer 4's one integral gives :math:`P_{II}` from the auto pair and
:math:`P_{gI}` from the cross with a galaxy sample, with no new code in layer 4.
The predecessor's generation of codes wrote it as a correction applied to
:math:`C_\ell^{\kappa\kappa}`, so an IA term there could not be crossed with
anything else.

**Which matter field.** ``rho_matter``, the total: it is the field that lenses,
and intrinsic alignments are a contaminant *of* lensing.
:class:`~ggah_mod.sectors.matter.MatterField` normalises on it for the same
reason.

Scope
-----

It is NLA, which is linear alignment evaluated on the non-linear matter power.
It is not TATT and it is not a halo-model alignment: there is no one-halo
alignment term here, so :math:`P_{II}` inherits whatever one-halo term the
matter field has rather than a satellite-alignment model of its own.
On the scales an alignment systematic is fitted over that is the standard
choice; on small scales this sector makes no statement, and a spectrum that
includes :attr:`IntrinsicAlignmentSector.name` makes none either.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .matter import matter_weights
from .params import Flat, Gaussian, Param, SectorParams, sector_params

__all__ = ["C1_RHO_CRIT", "IaParams", "IntrinsicAlignmentSector", "POPULATIONS",
           "alignment_bias"]

#: :math:`C_1\rho_{\rm crit}`, dimensionless, the NLA normalisation.
#:
#: :math:`C_1 = 5\times10^{-14}\,h^{-2}M_\odot^{-1}{\rm Mpc}^3` from the SuperCOSMOS
#: measurement (Brown et al. 2002), and multiplying by :math:`\rho_{\rm crit}`
#: gives this.  It is fixed by convention rather than fitted: every published
#: :math:`A_{\rm IA}` is quoted *relative* to it, so changing this number would
#: silently rescale every prior anyone carries in.  Traced through the
#: predecessor and through Bridle & King (2007), where the convention is set,
#: rather than re-derived -- the distinction ``PLAN.md`` item **G8** is about.
#: The rounded literal 0.0134 is Joachimi et al. (2011), A&A 527, A26.
#:
#: **CCL does not use this number, and the difference is 3.6 per cent.**
#: ``pyccl.nl_pt.translate_IA_norm`` computes :math:`C_1\rho_{\rm crit}` from
#: :math:`\rho_{\rm crit}` rather than taking the rounded literal:
#: :math:`5\times10^{-14}\times2.77536627\times10^{11} = 0.0138768`, against the
#: 0.0134 here.  Neither is wrong; they are two readings of one convention, and
#: the trap is that an :math:`A_{\rm IA}` moved between the two codes changes
#: amplitude by 3.6 per cent without changing name.  This module keeps the
#: literal deliberately: surveys quote :math:`A_{\rm IA}` against the *published*
#: constant, so matching the publication is what makes a carried-in prior mean
#: what it says.  Measured by ``ggah_mod_benchmark``, which found it -- and which
#: is also where the second half of the same difference lives, that CCL scales by
#: the sampled :math:`\Omega_m` where :attr:`IaParams.omega_m_ref` is the
#: published one.
C1_RHO_CRIT = 0.0134


def alignment_bias(z, growth, p: "IaParams", f_red=1.0, log10_mh=None):
    r""":math:`b_I(z)`, the number the matter field is multiplied by.

    Negative for positive :math:`A_{\rm IA}`; see the module docstring for why
    that is the physics rather than a sign convention.

    ``f_red`` is the fraction of the shape sample that aligns, one for the plain
    NLA.  ``log10_mh`` is :math:`\log_{10}\langle M_h\rangle` of the aligning
    galaxies in :math:`h^{-1}M_\odot`; with it, the amplitude carries
    :math:`(\langle M_h\rangle/M_{h,{\rm piv}})^{\beta_M}` (Wright et al. 2025,
    Eq. 11), and without it, no mass term.

    ``growth`` is :math:`D(z)`, normalised to one today, and is required for
    the reason :func:`ggah_mod.sectors.png.png_bias_shift` gives about its own:
    a :class:`~ggah_mod.halos.field.HaloField` carries a spectrum rather than
    the backend that made it, so there is nothing here to evaluate at a second
    redshift, and the two conventions in the literature differ by about 25 per
    cent.
    """
    one_z = (1.0 + jnp.asarray(z)) / (1.0 + p.z_pivot)
    mass = (1.0 if log10_mh is None
            else jnp.power(10.0, p.beta_m * (jnp.asarray(log10_mh) - p.log10_mh_pivot)))
    return -(p.a_ia * f_red * mass * C1_RHO_CRIT * p.omega_m_ref / growth
             * one_z ** p.eta_ia * p.l_over_l0 ** p.beta_ia)


@sector_params
class IaParams(SectorParams):
    r"""The NLA amplitude and its three power laws, with bounds and reasons.

    The defaults are the red-galaxy alignment of Fortuna et al. (2025), the
    prior of Wright et al. (2025, Sect. 2.4.4).
    """

    a_ia: float = 5.74
    eta_ia: float = 0.0
    beta_ia: float = 0.0
    beta_m: float = 0.44
    z_pivot: float = 0.62
    l_over_l0: float = 1.0
    log10_mh_pivot: float = 13.5
    omega_m_ref: float = 0.3

    _STATIC = ()

    _PARAMS = {
        "a_ia": Param(
            5.74, (-15.0, 15.0), Flat(), "",
            "the NLA amplitude of the aligning galaxies at the pivot halo mass, "
            "relative to the C1 rho_crit convention. The default is Fortuna et "
            "al. (2025)'s 5.74 +- 0.29 for early types; under population='all' "
            "it is the whole sample's, of order one. Both "
            "signs are physical and the box admits both: a *negative* A_IA "
            "means galaxies align across the stretching axis rather than along "
            "it, which no measurement supports but which a fit must be free to "
            "find, because a prior that excluded it would make a null result "
            "unrepresentable. The box is the early-type amplitude widened 2.5 "
            "times either side rather than tightened -- this package does not "
            "carry another survey's posterior in as a bound", "physical"),
        "eta_ia": Param(
            0.0, (-6.0, 6.0), Flat(), "",
            "redshift power law, ((1+z)/(1+z_pivot))^eta. Zero is no evolution "
            "and is the default, so a model that says nothing about it is the "
            "plain NLA. The box is where the factor stays within an order of "
            "magnitude across 0 < z < 3, beyond which the pivot stops meaning "
            "anything", "physical"),
        "beta_ia": Param(
            0.0, (-4.0, 4.0), Flat(), "",
            "luminosity power law, (L/L0)^beta. Zero is the default for the "
            "same reason as eta_ia. Measurements find it positive -- brighter, "
            "redder galaxies align more strongly -- and the box admits negative "
            "for the same reason a_ia's does", "physical"),
        "beta_m": Param(
            0.44, (-2.0, 2.0), Flat(), "",
            "halo-mass power law of the red galaxies' amplitude, "
            "(<M_h>/M_piv)^beta_m (Wright et al. 2025, Eq. 11). The default is "
            "Fortuna et al. (2025)'s 0.44 +- 0.03; zero is no mass dependence. "
            "Unused under population='all'", "physical"),
        "z_pivot": Param(
            0.62, (0.0, 3.0), Flat(), "",
            "the redshift the evolution is measured relative to. 0.62 is the "
            "value the joint-analysis convention uses, so an A_IA carried in "
            "from one of those means what it says. It is a *convention* rather "
            "than a fitted quantity, and moving it rescales A_IA -- which is "
            "why it is declared rather than hard-coded", "validity"),
        "l_over_l0": Param(
            1.0, (0.01, 100.0), Flat(), "",
            "the sample's mean luminosity over the pivot. One means the pivot "
            "sample, and with beta_ia = 0 it does nothing at all. A property "
            "of the selection rather than of the physics, like a stellar-mass "
            "threshold", "validity"),
        "log10_mh_pivot": Param(
            13.5, (10.0, 16.0), Flat(), "log10(Msun/h)",
            "the halo mass the amplitude is quoted at. 13.5 is Fortuna et al. "
            "(2025)'s, so their A_IA and beta_m mean what they say. A "
            "convention, like z_pivot: moving it rescales A_IA", "validity"),
        "omega_m_ref": Param(
            0.3, (0.05, 1.0), Flat(), "",
            "the Omega_m appearing in the NLA normalisation. It is *not* the "
            "cosmology's: the published convention fixes it, so an A_IA "
            "carried in from a survey means what that survey meant by it. "
            "Setting it to the sampled Omega_m instead is a defensible "
            "different convention and a silent change of what A_IA means, "
            "which is why this is a declared parameter rather than a read of "
            "`cosmo.Omega_m`", "validity"),
    }


#: The populations an alignment field can be built on (module docstring).
POPULATIONS = ("red", "all")


class IntrinsicAlignmentSector:
    """The intrinsic-shear field as a tracer: a peer, holding no grid.

    Parameters
    ----------
    coldgas : ColdGasSector, optional
        The neutral gas of the matter field it scales.
    population : {"red", "all"}
        Which galaxies of the shape sample align: the red ones, with the
        amplitude scaled by the red fraction and their mean host halo mass (the
        default), or all of them, the plain NLA.
    calibration : {"warn", "strict", "off"}
        The policy of the two galaxy sectors that give the red fraction, when
        the field lies outside the redshifts their occupation was fitted over.
    """

    name = "alignments"
    differentiable = True

    def __init__(self, coldgas=None, *, population: str = "red",
                 calibration: str = "warn"):
        if population not in POPULATIONS:
            raise ValueError(f"unknown population {population!r}; expected one "
                             f"of {POPULATIONS}")
        self.coldgas = coldgas
        self.population = population
        self.calibration = calibration
        self._red = self._every = None

    @property
    def galaxies(self):
        """The galaxy sector the neutral gas reads, or None; see
        :attr:`~ggah_mod.sectors.matter.MatterField.galaxies`."""
        return getattr(self.coldgas, "galaxies", None)

    def red_sample(self, field, sample=None):
        r"""``(f_red, log10_mh)`` of the shape sample on ``field``.

        :math:`f_{\rm red} = \bar n_{\rm red}/\bar n`, with ``zumandelbaum16_red``
        and ``zumandelbaum15`` at the galaxy parameters ``sample`` (a mapping or
        :class:`~ggah_mod.sectors.galaxies.GalaxyParams`; its stellar-mass
        threshold is the shape sample's selection), over
        ``galaxy_defaults("zumandelbaum16_red")`` for what it does not set.
        :math:`\log_{10}\langle M_h\rangle` is the mean of
        :math:`\log_{10}M` [:math:`h^{-1}M_\odot`, the field's mass
        definition] over the red occupation, weighted by
        :math:`(dn/dM)\,\langle N_{\rm red}\rangle`.
        """
        from .galaxies import GalaxySector, galaxy_defaults

        if self._red is None:
            self._red = GalaxySector("zumandelbaum16_red", calibration=self.calibration)
            self._every = GalaxySector("zumandelbaum15", calibration=self.calibration)
        p = dict(galaxy_defaults("zumandelbaum16_red"))
        if sample is not None:
            p.update(sample if isinstance(sample, dict) else sample.as_dict())
        n_red = sum(self._red.occupation(field, p))
        nbar_red = field.number_density(n_red)
        f_red = nbar_red / self._every.number_density(field, p)
        log10_mh = field.integrate(field.dndm * n_red * jnp.log10(field.m)) / nbar_red
        return f_red, log10_mh

    def weights(self, field, params, coldgas_params=None,
                galaxies_params=None):
        r"""Matter's weights, scaled by :math:`b_I(z)`.

        ``params`` wants ``split`` (the same
        :class:`~ggah_mod.sectors.matter.BaryonSplit` the matter field uses, so
        the two describe one matter distribution rather than two), plus
        ``params`` (an :class:`IaParams`) and ``growth``.
        Under ``population="red"`` it also reads, when present, ``sample`` (the
        shape sample's galaxy parameters, :meth:`red_sample`), or ``f_red`` and
        ``log10_mh`` directly, which take precedence.
        The shape sample is not the spectrum's galaxy sector: in
        :math:`P_{gI}` the galaxies are the lenses, so ``galaxies_params`` is
        not read here.
        """
        if "split" not in params:
            raise ValueError(
                "the alignment sector needs `split`, the same BaryonSplit the "
                "matter field is built from. Intrinsic alignments trace the "
                "matter that lenses, so building them on a second, separately "
                "specified matter field would let P_gI and P_gm describe "
                "different universes.")
        if "growth" not in params:
            raise ValueError(
                "the alignment sector needs `growth`, D(z) normalised to one "
                "today. It cannot be taken from the field, which carries a "
                "spectrum rather than the backend that made it, and the two "
                "conventions in the literature differ by about 25 per cent -- "
                "so a default here would be a systematic wearing a plausible "
                "number.")

        p = params.get("params", IaParams())
        from .matter import MatterField
        w = matter_weights(field, params["split"],
                           u_gas=params.get("u_gas"), u_dm=params.get("u_dm"),
                           u_sat=params.get("u_sat"), u_ej=params.get("u_ej"),
                           u_cold=MatterField(self.coldgas)._u_cold(
                               field, params, coldgas_params,
                               galaxies_params))
        if self.population == "red":
            f_red, log10_mh = params.get("f_red"), params.get("log10_mh")
            if f_red is None or log10_mh is None:
                f_sam, m_sam = self.red_sample(field, params.get("sample"))
                f_red = f_sam if f_red is None else f_red
                log10_mh = m_sam if log10_mh is None else log10_mh
            b_i = alignment_bias(field.z, params["growth"], p, f_red, log10_mh)
        else:
            b_i = alignment_bias(field.z, params["growth"], p)
        return dataclasses.replace(
            w,
            w_point=None if w.w_point is None else w.w_point * b_i,
            w_extended=None if w.w_extended is None else w.w_extended * b_i,
            # The intrinsic-shear field is the *matter* field times b_I, and
            # the matter field includes the neutrinos' linear leg -- so the
            # leg is scaled with the rest, or P_gI and P_II would lose it.
            neutrino_weight=(None if w.neutrino_weight is None
                             else w.neutrino_weight * b_i),
            name="alignments")
