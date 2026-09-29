r"""Mis-centering: the halo's nominal centre is not where the central sits.

A modifier, not a sector.  It does not change *how many* objects a halo hosts,
only where one of them sits, so it multiplies :math:`w_{\rm point}` and leaves
``norm`` alone -- which is the whole reason
:class:`~ggah_mod.sectors.protocol.ProfileModifier` exists as a separate
protocol.

.. math::

    h(k|M) = (1 - p_{\rm off}) + p_{\rm off}\,\tilde W_{\rm off}(k|M)

A fraction :math:`p_{\rm off}` of centrals is displaced from the halo centre by
a 2-D Gaussian of width :math:`R_{\rm off}`, whose Fourier transform is
:math:`\exp(-k^2R_{\rm off}^2/2)`; the rest sit exactly at it.  So
:math:`h(k\to0) = 1` identically, whatever the parameters -- the modifier moves
power between scales and creates none.

Two conventions, and why the choice is static
----------------------------------------------

They differ in what sets the offset width, and the difference is a factor of the
concentration:

``"more15"``
    :math:`R_{\rm off} = R_{\rm off}^{\rm rel}\,r_s(M)`, so the offset scales
    with the halo -- More et al. (2015) Eq. 9.
``"johnston07"``
    :math:`R_{\rm off} = \sigma_{\rm off}`, one length in Mpc/h for every halo
    -- Johnston et al. (2007).

The predecessor chose between them by **dict-key membership**::

    if p_off > 0.0 and R_off > 0.0:        # More+2015
        ...
    elif p_off > 0.0 and sigma_off > 0.0:  # Johnston+2007

so which model ran depended on which key a caller happened to have set, and
setting both silently selected the first.  Two published models behind one
unnamed branch, resolved by ``dict.get``.  Here the convention is a **static**
field: it lives in the treedef, it is named at construction, and asking for one
model while supplying the other's parameter raises instead of switching.

It is also why this is not a sigmoid or a clip anywhere: :math:`p_{\rm off}` is
a genuine fraction, and its bound belongs in the sampler rather than in the
model, as :class:`~ggah_mod.sectors.params.Param` documents.
"""

from __future__ import annotations

import jax.numpy as jnp

from .params import Flat, Gaussian, Param, SectorParams, sector_params

__all__ = ["MisCenteringParams", "MisCentering", "MISCENTERING_CONVENTIONS",
           "P_OFF", "R_OFF"]

#: The two published conventions.  A name, never an inferred branch.
MISCENTERING_CONVENTIONS = ("more15", "johnston07")


#: The two mis-centring parameters, **defined once**.
#:
#: :class:`~ggah_mod.sectors.galaxies.GalaxyParams` imports these objects rather
#: than restating them, and the reason is not brevity.  They were declared twice,
#: and the two copies had drifted: ``p_off`` defaulted to 0.0 here and 0.25 there
#: while :meth:`~ggah_mod.sectors.galaxies.GalaxySector.weights` builds *this*
#: container from *that* default -- so the effective default was 0.25 and this
#: module's "a mis-centred model must be asked for" was false in practice.
#: ``r_off`` carried the identical ``Gaussian(0.25, 0.10)`` in both, which is one
#: published CMASS measurement counted twice if a fit ever frees both containers.
#:
#: One object cannot drift from itself, and a calibration that writes a measured
#: prior back into the package writes it in one place.
P_OFF = Param(
    0.0, (0.0, 1.0), Flat(), "",
    "the fraction of centrals that are displaced, so bounded by being a "
    "fraction. Zero is the default because a mis-centred model must be asked "
    "for: the correction only ever suppresses small-scale power, so leaving it "
    "on by accident is absorbed into the concentration. **The default is not a "
    "measurement** -- More et al. (2015) fit this parameter and report 'very "
    "weak constraints' on it, so no published value exists to adopt, and it is "
    "meant to be fitted. Zero rather than a nominal 0.25 so that the fiducial "
    "is the undecorated model, which is also what "
    "GalaxySector.weights already documents as the neutral value",
    "definitional")

R_OFF = Param(
    0.25, (0.0, 2.0), Gaussian(0.25, 0.10), "r_s",
    "More et al. (2015) Eq. 9, the offset width in units of the NFW scale "
    "radius. The prior is the published CMASS value; above ~2 the offset "
    "exceeds the halo and the 2-D Gaussian stops being a description of "
    "anything. This Param object is shared with GalaxyParams so the one "
    "measurement is stated once", "prior")


@sector_params
class MisCenteringParams(SectorParams):
    """Mis-centering parameters, with bounds and reasons."""

    p_off: float = 0.0
    r_off: float = 0.25
    sigma_off: float = 0.4

    _PARAMS = {
        "p_off": P_OFF,
        "r_off": R_OFF,
        "sigma_off": Param(
            0.4, (0.0, 5.0), Flat(), "Mpc/h",
            "Johnston et al. (2007), the same width as one length for every "
            "halo. Separate from r_off rather than shared, because they are "
            "different quantities in different units and the predecessor's "
            "single slot let one be read as the other", "physical"),
    }
    _STATIC = ()


class MisCentering:
    """A :class:`~ggah_mod.sectors.protocol.ProfileModifier`.

    Parameters
    ----------
    convention : str
        One of :data:`MISCENTERING_CONVENTIONS`.  **Static**, and required to be
        named: the two differ by a factor of :math:`c(M)`, and inferring the
        choice from which parameter is set is how the predecessor ran two
        published models under one branch.
    """

    name = "miscentering"
    differentiable = True

    def __init__(self, convention: str = "more15"):
        key = str(convention).lower()
        if key not in MISCENTERING_CONVENTIONS:
            raise ValueError(
                f"unknown mis-centering convention {convention!r}; expected one "
                f"of {list(MISCENTERING_CONVENTIONS)}.  The two differ by a "
                f"factor of the concentration, so there is no default that is "
                f"right for both.")
        self.convention = key

    def __repr__(self) -> str:                           # pragma: no cover
        return f"MisCentering(convention={self.convention!r})"

    def offset_scale(self, field, params: MisCenteringParams):
        r""":math:`R_{\rm off}(M)` in comoving Mpc/h, shape ``(NM,)``."""
        if self.convention == "more15":
            return params.r_off * field.r_s
        return jnp.broadcast_to(jnp.asarray(params.sigma_off), field.r_s.shape)

    def h_k(self, field, params: MisCenteringParams):
        r""":math:`h(k|M)`, shape ``(Nk, NM)``, with :math:`h(k\to0) = 1`.

        The large-scale limit is exact rather than asymptotic: at :math:`k = 0`
        the exponential is 1 and the two branches sum to
        :math:`(1-p) + p = 1` identically, for every :math:`p_{\rm off}` and
        every offset scale.  So a mis-centred tracer has the same mean density
        and the same linear bias as a centred one, which is what makes this a
        modifier rather than a sector.
        """
        r_off = self.offset_scale(field, params)                   # (NM,)
        k = jnp.atleast_1d(field.k)[:, None]                       # (Nk, 1)
        w_off = jnp.exp(-0.5 * (k * r_off[None, :]) ** 2)          # (Nk, NM)
        p = params.p_off
        return (1.0 - p) + p * w_off

    def apply(self, weights, field, params: MisCenteringParams):
        r"""``weights`` with :math:`w_{\rm point}` multiplied by :math:`h(k|M)`.

        ``norm`` is untouched, deliberately: displacing a central does not
        create or destroy one, so :math:`\bar n` is unchanged and only the
        k dependence moves.
        """
        if weights.w_point is None:
            raise ValueError(
                f"tracer {weights.name!r} has no point component, so there is "
                f"nothing to mis-centre.  Mis-centering displaces the object "
                f"at the halo centre; a tracer that is entirely on a profile "
                f"has none, and applying it would silently do nothing.")
        point = jnp.atleast_2d(jnp.asarray(weights.w_point))
        return weights.with_point(point * self.h_k(field, params))
