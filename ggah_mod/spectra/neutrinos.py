r"""The neutrinos in the two-halo term: a linear leg, not a halo.

Massive neutrinos free-stream and are in no halo -- the clustered fraction is
:math:`F_h \sim 10^{-3}` even at 0.3 eV (Massara, Villaescusa-Navarro & Viel
2014) -- so they have no weight per halo and no one-halo term.  They are not a
constant either, which would be white noise: above the free-streaming scale
they carry the linear fluctuation of the field they fell into.  The total
matter spectrum is

.. math::

    P_{mm} = (1-f_\nu)^2 P_{cb}^{\rm HM} + 2f_\nu(1-f_\nu)P_{cb\nu}
             + f_\nu^2 P_{\nu\nu},

and the halo model supplies only the first term.

The leg
-------

Linear modes are fully correlated, :math:`\delta_\nu = (T_\nu/T_{cb})\,
\delta_{cb}`, so the neutrino amplitude comes out of the two spectra a
:class:`~ggah_mod.halos.field.HaloField` already carries:

.. math::

    L(k) = f_\nu\frac{T_\nu}{T_{cb}}
         = \sqrt{\frac{P_m(k)}{P_{cb}(k)}} - \frac{\bar\rho_{cb}}{\bar\rho_m},

which tends to :math:`f_\nu` on large scales and to zero below free streaming.
Added to the two-halo amplitude of a tracer that carries neutrinos,

.. math::

    \mathcal I_a(k) = I_a(k) + \frac{\nu_a}{n_a}\,L(k), \qquad
    P^{2h}_{ab} = P_{cb}\,\mathcal I_a\,\mathcal I_b ,

it gives :math:`P^{2h}_{mm} \to P_m` on large scales -- **exactly**, because the
counterterm already puts :math:`I_m(k\to0)` at
:math:`\bar\rho_{cb}/\bar\rho_m`, which is the constant subtracted in
:math:`L`.  :math:`\nu_a` is
:attr:`~ggah_mod.sectors.protocol.TracerWeights.neutrino_weight`: 1 for
matter, :math:`b_I` for intrinsic alignments, absent for everything built on
the cold field.  :math:`P_{gm}` and :math:`P_{II}` receive their cross terms
from the same single integral, and nothing here names a sector.

What it is not
--------------

* **Not a profile.**  :math:`L` has no mass dependence, so it cannot enter
  through :math:`W_m(k|M)` without inventing one.
* **Not beyond-linear.**  The neutrinos are not haloes, so
  :mod:`~ggah_mod.spectra.bnl` never sees the leg.
* **Not on the total spectrum.**  With ``two_halo_spectrum="total"`` the halo
  term is already built on :math:`P_m`, and adding the leg would count the
  neutrinos twice; the pair is refused.

The guard at :math:`\Sigma m_\nu = 0` is an exact zero with a finite (zero)
gradient: the two ``emu_pk`` heads disagree there by :math:`3\times10^{-4}`,
which is not a neutrino.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["NEUTRINO_TWO_HALO", "neutrino_leg", "tracer_leg"]

#: How the neutrinos enter the two-halo term.  ``"linear"`` adds the leg;
#: ``"none"`` leaves the cold-only halo model, kept so the size of the choice is
#: measurable.  Kept in step with :data:`ggah_mod.backend.NEUTRINO_TWO_HALO` by
#: a test rather than by an import, as the transition registry is.
NEUTRINO_TWO_HALO = ("linear", "none")


def neutrino_leg(field):
    r""":math:`L(k) = \sqrt{P_m/P_{cb}} - \bar\rho_{cb}/\bar\rho_m`, shape ``(Nk,)``."""
    cold = field.rho_cold / field.rho_matter
    return jnp.where(cold < 1.0,
                     jnp.sqrt(field.pk_lin / field.pk_cb) - cold, 0.0)


def tracer_leg(field, w, options):
    r"""``w``'s share of the leg, :math:`(\nu_a/n_a)L(k)`, or ``None``.

    ``None`` rather than zeros when the tracer carries no neutrinos or the leg
    is off, so a caller that adds it can skip the addition and stay bit for bit
    on the cold-only result.
    """
    choice = options.neutrino_two_halo
    if choice not in NEUTRINO_TWO_HALO:
        raise ValueError(
            f"unknown neutrino two-halo treatment {choice!r}; expected one of "
            f"{NEUTRINO_TWO_HALO}")
    if choice == "none" or w.neutrino_weight is None:
        return None
    if options.two_halo_spectrum != "cb":
        raise ValueError(
            "the neutrino leg is added to a two-halo term built on the cold "
            "spectrum; with two_halo_spectrum='total' the neutrinos are "
            "already in it.  Set neutrino_two_halo='none' for that pairing.")
    return (w.neutrino_weight / w.norm) * neutrino_leg(field)
