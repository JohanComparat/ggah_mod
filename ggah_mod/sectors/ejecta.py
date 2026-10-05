r"""The baryons a halo lost, as a tracer in its own right.

:mod:`~ggah_mod.sectors.matter` names :math:`f_{\rm ejected}` and gives it a
profile; this module makes it a :class:`~ggah_mod.sectors.protocol.Sector`, so
the ejected gas can be crossed with anything through the same layer-4 integral
as the hot gas, the galaxies and the AGN.

That is not bookkeeping.  The ejected component is the *only* one of the six
whose defining measurements -- stacked kSZ, a fast radio burst's dispersion
measure, the baryonic suppression of :math:`P_{mm}` -- are measurements of where
it is rather than of how much of it there is.  A census row cannot be compared
with any of them.  A tracer can.

Why it is not a view of the gas sector
--------------------------------------

Because it is not the same gas.  :class:`~ggah_mod.sectors.gas.HotGasDPM` is a
gNFW fitted inside :math:`R_\Delta` with an X-ray-calibrated normalisation, and
its profile is meaningless outside the aperture the fit was made in -- the
module already records that an outer density slope below 3 makes its mass
integral diverge, and that integrated past :math:`R_\Delta` it counts gas that
is not bound to the halo.  The ejected gas
lives *beyond* that radius by construction.  Giving the two one sector would
mean one set of shape parameters describing both, which is the assumption the
whole ejected component exists to avoid.

They are peers, and a caller who wants all the free electrons asks for both --
which is what the ``electrons`` tracer in :mod:`~ggah_mod.spectra.tracers` is.

What sets the ejection radius
-----------------------------

One parameter, :math:`\eta_{\rm ej} = r_{\rm ej}/R_\Delta`, and it is free here.
It should not stay free: the energy that moved the gas to :math:`r_{\rm ej}` is
the same energy the closure in :mod:`~ggah_mod.sectors.energetics` balances
against :math:`\Delta f_b M v_\Delta^2` -- which is the work to unbind the gas
*to infinity*, and therefore an overestimate by :math:`(1 - 1/\eta_{\rm ej})`
for a finite radius.  Tying the two removes this parameter and makes
:math:`f_{\rm ejected}` and :math:`r_{\rm ej}` one prediction instead of two.
That is ``PLAN.md``'s Tier 6 item C4, and it is done by sharing the object
rather than reconciling two: :data:`ETA_EJ` is the one ``Param`` that both this
sector and :class:`~ggah_mod.sectors.energetics.EnergeticsParams` carry, so the
radius the gas is put at and the radius the closure charges for are one number.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..halos.profiles import ejected_uk
from .gas import MU_E
from .params import Flat, Param, SectorParams, sector_params
from .protocol import TracerWeights

__all__ = ["ETA_EJ", "EjectaParams", "EjectaSector", "MU_E"]


#: The ejection radius, in units of :math:`R_\Delta`.
#:
#: **One object, two containers.**  :mod:`~ggah_mod.sectors.energetics` took the
#: same quantity as a bare ``eta_ej=None`` keyword, so the radius the profile put
#: the gas at and the radius the closure charged for moving it were free to be
#: different numbers -- a sampler could put the gas at :math:`4R_\Delta` while
#: paying for a move to :math:`2R_\Delta`.  Shared by identity, the way
#: :data:`~ggah_mod.sectors.miscentering.P_OFF` is, so the two cannot drift and
#: ``PLAN.md``'s item **C4** is a unification rather than a reconciliation.
ETA_EJ = Param(
    2.0, (1.0, 8.0), Flat(), "R_Delta",
    "Below 1 the gas is inside the halo radius and is not ejected -- "
    "it is the hot phase, which has its own sector and its own "
    "profile.  Above ~8 the Gaussian is flat across every wavenumber a "
    "halo model is used at, so the component stops being "
    "distinguishable from the smooth field and the parameter stops "
    "being constrained by anything; Schneider & Teyssier (2015) place "
    "it near 2-4 for the haloes this matters for.  Their own parameter is "
    "not this one: r_ej = eta_a r_esc ~ 7 eta_a R_200 (their Eqs. 2.13, "
    "2.22-2.23), so eta_ej = 2 is their eta_a ~ 0.3.",
    kind="physical")


@sector_params
class EjectaParams(SectorParams):
    """Where the expelled baryons went.  One parameter, and its reason."""

    eta_ej: float = 2.0

    _STATIC = ()

    _PARAMS = {"eta_ej": ETA_EJ}


class EjectaSector:
    r"""The ejected baryons: :math:`f_{\rm ejected}` in a Gaussian of radius :math:`\eta_{\rm ej}R_\Delta`.

    Holds no grid and no fractions of its own -- it reads
    :attr:`~ggah_mod.sectors.matter.BaryonSplit.f_ejected` from the split the
    matter sector was built with, so the two cannot disagree about how much gas
    left.  A sector that recomputed it would be a second definition of the same
    number, which is the defect this package spends most of its docstrings on.
    """

    name = "ejecta"
    differentiable = True

    #: ``view -> what the amplitude means``.  ``density`` is an alias of
    #: ``mass``, exactly as in :class:`~ggah_mod.sectors.gas.HotGasDPM`: the
    #: mass and electron-density profiles have the same *shape*, and for a
    #: fully ionised plasma the amplitudes differ by the constant
    #: :math:`\mu_e m_p`, which cancels in every normalised statistic.  One
    #: function under two names, so a caller asking for either gets the same
    #: array rather than two that can drift.
    VIEWS = ("mass", "density")

    def mass(self, field, split):
        r""":math:`M_{\rm ej}(M) = f_{\rm ejected}\,M` [Msun/h].

        Signed, and deliberately so: a negative value means the halo was given
        more baryons than exist, which :class:`~ggah_mod.sectors.matter.BaryonSplit`
        reports rather than clips.  Clipping here would hide it one layer further
        from where it can be fixed.
        """
        return jnp.atleast_1d(split.f_ejected) * field.m

    def electron_count(self, field, split):
        r""":math:`N_e = M_{\rm ej}/(\mu_e m_p)`, in solar masses per proton mass.

        Not used by :meth:`weights` -- the normalised transform makes the
        constant cancel -- but it is what a dispersion measure or a kSZ optical
        depth needs in absolute terms, and it exists here rather than at the
        call site so there is one place the ionisation assumption is written
        down.  The assumption: the ejected gas is fully ionised, which is why
        :data:`~ggah_mod.sectors.gas.MU_E` is the hot-phase value.
        """
        return self.mass(field, split) / MU_E

    def weights(self, field, params, view: str = "mass") -> TracerWeights:
        r""":math:`W_{\rm ej}(k|M) = M_{\rm ej}(M)\,\tilde u_{\rm ej}(k|M)`.

        Continuous: expelled gas is a field, not a countable population, so its
        one-halo auto-spectrum has no self-pair to exclude.

        ``params`` carries ``split`` (a
        :class:`~ggah_mod.sectors.matter.BaryonSplit`) and optionally
        ``eta_ej`` or an :class:`EjectaParams`.
        """
        if view not in self.VIEWS:
            raise ValueError(f"unknown ejecta view {view!r}; expected one of "
                             f"{sorted(self.VIEWS)}")
        if isinstance(params, EjectaParams):
            split, eta = None, params.eta_ej
        else:
            split = params.get("split")
            p = params.get("params")
            eta = params.get("eta_ej",
                             p.eta_ej if p is not None else EjectaParams().eta_ej)
        if split is None:
            raise ValueError(
                "the ejecta sector needs `split`, a BaryonSplit, so that the "
                "ejected fraction it draws is the same one the matter field "
                "subtracted.  Recomputing it here would be a second definition "
                "of one number.")

        amp = self.mass(field, split)
        uk = ejected_uk(field.k, field.r_delta, eta_ej=eta)
        return TracerWeights(w_point=None, w_extended=amp[None, :] * uk,
                             norm=jnp.asarray(1.0), discrete=False,
                             bias_weight=None, name=f"ejecta:{view}",
                             w_unresolved=self.unresolved_weight(field, split, uk))

    @staticmethod
    def split_from_gas(field, gas, gas_params, *, f_star_cen=0.0,
                       f_star_sat=0.0, f_cold=0.0, agn_params=None,
                       galaxies_params=None):
        r"""The :class:`~ggah_mod.sectors.matter.BaryonSplit` whose hot gas **is**
        the DPM's.

        ``electrons`` is ``gas:density`` plus ``ejecta:mass``: the first
        amplitude is the DPM's gas mass, the second the split's
        :math:`f_{\rm ejected}M`.  Built from two independent models the sum is
        not :math:`f_b - f_\star - f_{\rm cold}`, and the difference is gas
        counted twice or not at all.  This builds the split from the DPM's own
        gas fraction -- ``from_hot`` with :math:`M_{\rm gas}/M` -- so the
        ejected gas is exactly what the hot atmosphere does not hold, and the
        composite's :math:`k\to0` amplitude closes on the budget.

        The feedback peers are forwarded so a closure-coupled gas sector is
        read with the amplitude its spectrum uses.
        """
        from .matter import BaryonSplit, cosmic_baryon_fraction
        a = gas._feedback_amplitude(field, gas_params, agn_params,
                                    galaxies_params)
        f_hot = gas.gas_mass(field.m, field.z, field.cosmo, gas_params,
                             mdef=field.mdef, conc=field.conc,
                             amplitude=a) / field.m
        return BaryonSplit.from_hot(cosmic_baryon_fraction(field.cosmo), f_hot,
                                    f_star_cen=f_star_cen,
                                    f_star_sat=f_star_sat, f_cold=f_cold)

    @staticmethod
    def diffuse_fraction(split):
        r"""The baryons of the smallest resolved halo that are neither stars nor
        neutral gas, as a fraction of its cold mass.

        :math:`f_b - f_{\star,\rm cen} - f_{\star,\rm sat} - f_{\rm cold}`
        at the first mass node, which is :math:`f_{\rm hot} + f_{\rm ejected}`
        there.  This is the composition of the matter below the grid that is
        booked as diffuse, ionised gas -- the same number
        :func:`~ggah_mod.sectors.census.census` uses for its outside row.
        """
        f_hot = jnp.atleast_1d(split.f_hot)
        f_ej = jnp.atleast_1d(split.f_ejected)
        return f_hot[0] + f_ej[0]

    def unresolved_weight(self, field, split, uk):
        r"""What the matter below the grid carries of this tracer, shape ``(Nk,)``.

        Every baryon there that is not a star or neutral gas, on the ejected
        profile of the smallest resolved halo:
        :math:`f_{\rm diffuse}\,\bar\rho_{cb}\,\tilde u_{\rm ej}(k|M_{\min})`.
        The hot-gas views declare zero, so the ``electrons`` composite counts
        these baryons once, here.
        """
        return self.diffuse_fraction(split) * field.rho_cold * uk[:, 0]
