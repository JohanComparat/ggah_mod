r"""Intrinsic alignments, as a sector -- a peer, like everything else here.

``PLAN.md`` item **E3**, and the benchmark's reason for it: *"A weak-lensing
likelihood cannot be written without one"*.  Now that layer 5 ships
:math:`\Delta\Sigma` and :math:`w(\theta)`, this is the next binding constraint
on writing one, and :mod:`~ggah_mod.cosmology.growth`'s docstring has named
"an intrinsic-alignment amplitude" as a consumer of :math:`D(z)` since before
there was anything to consume it.

The model, and what it is a model *of*
--------------------------------------

The non-linear alignment model (Hirata & Seljak 2004; Bridle & King 2007).  A
galaxy's shape responds to the tidal field it formed in, and the tidal field
traces matter, so to linear order the intrinsic-shear field is the matter field
times a number:

.. math::

    b_I(z) = -A_{\rm IA}\,\frac{C_1\rho_{\rm crit}\,\Omega_m}{D(z)}
             \left(\frac{1+z}{1+z_0}\right)^{\eta}
             \left(\frac{L}{L_0}\right)^{\beta}

**The sign is the physics and not a convention.**  Galaxies align *along* the
stretching axis of the tidal field, where lensing shears *across* it, so the
intrinsic and lensing shears anticorrelate: :math:`P_{gI}` and the
galaxy--galaxy-lensing signal have opposite signs, and an alignment model that
came out positive would be adding to the very signal it exists to subtract.
:func:`alignment_bias` returns a negative number at positive
:math:`A_{\rm IA}`, and a test asserts it.

Why it is a *tracer* and not a correction
-----------------------------------------

Because that is what it is, and the layer already has the machinery.  The
intrinsic-shear field is the matter field scaled, so
:meth:`IntrinsicAlignmentSector.weights` returns
:func:`~ggah_mod.sectors.matter.matter_weights` multiplied by :math:`b_I` -- and
then layer 4's one integral gives :math:`P_{II}` from the auto pair and
:math:`P_{gI}` from the cross with a galaxy sample, with no new code in layer 4
at all.  Writing it as a *correction applied to* :math:`C_\ell^{\kappa\kappa}`
is the shape the predecessor's generation of codes used, and it is why an
IA term there could not be crossed with anything else.

**Which matter field.**  ``rho_matter``, the total, because that is what lenses
and intrinsic alignments are a contaminant *of* lensing.  The same reason
:class:`~ggah_mod.sectors.matter.MatterField` normalises on it.

What this is not
----------------

It is NLA, which is linear alignment evaluated on the non-linear matter power.
It is **not** TATT and it is not a halo-model alignment: there is no
one-halo alignment term here, so :math:`P_{II}` inherits whatever one-halo term
the matter field has rather than a satellite-alignment model of its own.  On the
scales an alignment systematic is fitted over that is the standard choice; on
small scales it is a statement this sector does not make, and
:attr:`IntrinsicAlignmentSector.name` being in a spectrum is not a claim that it
does.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from .matter import matter_weights
from .params import Flat, Gaussian, Param, SectorParams, sector_params

__all__ = ["C1_RHO_CRIT", "IaParams", "IntrinsicAlignmentSector",
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


def alignment_bias(z, growth, p: "IaParams"):
    r""":math:`b_I(z)`, the number the matter field is multiplied by.

    Negative for positive :math:`A_{\rm IA}`; see the module docstring for why
    that is the physics rather than a sign convention.

    ``growth`` is :math:`D(z)`, **normalised to one today**, and is required for
    the reason :func:`ggah_mod.sectors.png.png_bias_shift` gives about its own:
    a :class:`~ggah_mod.halos.field.HaloField` carries a spectrum rather than
    the backend that made it, so there is nothing here to evaluate at a second
    redshift, and the two conventions in the literature differ by about 25 per
    cent.
    """
    one_z = (1.0 + jnp.asarray(z)) / (1.0 + p.z_pivot)
    return -(p.a_ia * C1_RHO_CRIT * p.omega_m_ref / growth
             * one_z ** p.eta_ia * p.l_over_l0 ** p.beta_ia)


@sector_params
class IaParams(SectorParams):
    r"""The NLA amplitude and its two power laws, with bounds and reasons."""

    a_ia: float = 1.0
    eta_ia: float = 0.0
    beta_ia: float = 0.0
    z_pivot: float = 0.62
    l_over_l0: float = 1.0
    omega_m_ref: float = 0.3

    _STATIC = ()

    _PARAMS = {
        "a_ia": Param(
            1.0, (-6.0, 6.0), Flat(), "",
            "the NLA amplitude, relative to the C1 rho_crit convention. Both "
            "signs are physical and the box admits both: a *negative* A_IA "
            "means galaxies align across the stretching axis rather than along "
            "it, which no measurement supports but which a fit must be free to "
            "find, because a prior that excluded it would make a null result "
            "unrepresentable. The magnitude is the range weak-lensing surveys "
            "report, roughly -2 to 2, widened threefold rather than tightened "
            "-- this package does not carry another survey's posterior in as a "
            "bound", "physical"),
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


class IntrinsicAlignmentSector:
    """The intrinsic-shear field as a tracer.  A peer, holding no grid."""

    name = "alignments"
    differentiable = True

    def __init__(self, coldgas=None):
        self.coldgas = coldgas

    @property
    def galaxies(self):
        """The galaxy sector the neutral gas reads, or None; see
        :attr:`~ggah_mod.sectors.matter.MatterField.galaxies`."""
        return getattr(self.coldgas, "galaxies", None)

    def weights(self, field, params, coldgas_params=None,
                galaxies_params=None):
        r"""Matter's weights, scaled by :math:`b_I(z)`.

        ``params`` wants ``split`` -- the same
        :class:`~ggah_mod.sectors.matter.BaryonSplit` the matter field uses, so
        the two describe one matter distribution rather than two -- plus
        ``params`` (an :class:`IaParams`) and ``growth``.
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
