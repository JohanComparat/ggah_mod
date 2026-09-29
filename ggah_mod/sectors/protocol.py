r"""What a sector is, and what it hands to layer 4.

One integral serves every pair of tracers:

.. math::

    P_{ab}^{1h}(k) = \int dM\,\frac{dn}{dM}\,W_a(k|M)\,W_b(k|M),
    \qquad
    P_{ab}^{2h}(k) = P_{\rm lin}(k)\,I_a(k)\,I_b(k)

with :math:`I_a(k) = \int dM\,(dn/dM)\,b(M)\,W_a(k|M)`.  The predecessor wrote
that integral out **eight times** and ``w_p``/``\Delta\Sigma`` four times each,
and the copies disagreed: ``P_{gg}^{2h}`` forced :math:`\tilde u \to 1` at all
k while ``P_{my}^{2h}`` used the real :math:`I_m(k)`, so two spectra in one
analysis made different approximations with nothing recording it.

:class:`TracerWeights` is what makes one integral possible.  Every sector
answers the same question -- *what is your weight per halo, as a function of
k and M* -- and layer 4 never learns which sector it is talking to.

The decomposition, and why it is not just ``W``
-----------------------------------------------

.. math::  W(k|M) = \frac{w_{\rm point} + w_{\rm extended}}{n}

The split is **not** cosmetic.  A one-halo auto-spectrum of a *discrete* tracer
must not count a central galaxy paired with itself: the correct term is
:math:`\langle N(N-1)\rangle`, not :math:`\langle N\rangle^2`.  With the point
and extended parts separated, layer 4 can form

.. math::  2\,w_{\rm point}w_{\rm extended} + w_{\rm extended}^2

for a discrete tracer and :math:`(w_{\rm point}+w_{\rm extended})^2` for a
continuous field, from the same object.  Fused into one array the distinction is
unrecoverable, and getting it wrong changes :math:`P_{gg}^{1h}` at the
:math:`1/N_c` level with nothing raising.

:attr:`TracerWeights.discrete` is therefore the **only** physics branch in the
layer-4 integrand, and it is a static flag rather than an inference from
whether some parameter happens to be present.

Conventions, fixed here so no sector has to decide
--------------------------------------------------

* shapes: ``(NM,)`` or ``(Nk, NM)`` -- ``(Nk, NM)`` after broadcasting, always;
* ``w_extended`` is **already multiplied by** its :math:`u(k|M)`, so layer 4
  never has to know which profile a sector chose;
* ``norm`` is :math:`\bar n` for a discrete tracer and ``1.0`` for a field, so
  ``W`` is per-object in the first case and per-unit-volume in the second;
* the weights **exclude** :math:`dn/dM`.  It appears once, in layer 4's
  integrand, exactly as the two formulas at the top of this docstring write it.

That last rule used to read the other way round -- that the weights "include the
mass-function factor's partner", in the convention
:func:`~ggah_mod.halos.beyond_linear_bias.correction_2h_gg` documents as
``dndm * N_tot * b / n_bar``, "so the beyond-linear-bias hook needs no adapter".
Both halves of that were wrong, and the sentence is recorded here rather than
deleted because it is what produced the defect:

* it described a convention **two of the three sectors did not follow**.
  :class:`~ggah_mod.sectors.gas.HotGasDPM` and
  :func:`~ggah_mod.sectors.matter.matter_weights` exclude ``dn/dM``;
  :class:`~ggah_mod.sectors.agn.AgnSector` included it, which put
  :math:`(dn/dM)^2` in every AGN one-halo integrand, and left :math:`W_{\rm agn}`
  carrying units of :math:`(M_\odot/h)^{-1}` while the other two are
  dimensionless.  Nothing raised, because nothing consumed the weights yet.
* the "no adapter" claim is false under **either** convention.
  ``correction_2h_gg`` takes ``weights`` ``(NM,)`` and ``uk`` ``(Nk, NM)`` as
  *separate* arguments and multiplies them itself, while this class deliberately
  fuses them -- ``w_extended`` is already :math:`\times u(k|M)` and
  ``w_point`` has no :math:`u` at all -- so a tracer with both components cannot
  be factorised back into that pair.  It also contracts with **no quadrature
  measure**, so its ``weights`` are missing a ``dM``.  The adapter that does work
  is two lines and is written down in :mod:`ggah_mod.spectra.bnl`:
  :meth:`~ggah_mod.halos.field.HaloField.quadrature_measure` supplies the
  ``dM``, and the fused ``W`` goes in as ``uk`` against unit ``weights``.

Differentiability
-----------------

Every sector in this layer declares ``differentiable = True``, and
``tests/test_sectors_paths.py`` checks the declaration by trying it rather than
believing it.  Layer 2 permits exactly one exception; layer 3 permits **none**,
and that is asserted as an empty set rather than left as prose.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = ["TracerWeights", "Sector", "ProfileModifier", "combine"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TracerWeights:
    r"""One tracer's weight per halo, ready for the layer-4 integral.

    **Registered by hand, and not a ``NamedTuple``.**  A ``NamedTuple`` is a
    pytree already, which makes it the obvious choice and the wrong one: it
    flattens *every* field into the children, so :attr:`discrete` and
    :attr:`name` would become leaves.  A Python ``bool`` among the leaves is a
    value ``jax.grad`` will happily try to differentiate, and a ``str`` is not a
    JAX type at all -- the object would fail the moment it crossed a ``jit``
    boundary rather than when it was built, which is a long way from the
    mistake.  So the split is explicit: arrays are children, the two choices are
    aux data.

    Attributes
    ----------
    w_point : array or None, shape (NM,) or (Nk, NM)
        The part that sits at the halo centre -- a central galaxy, a point-like
        AGN, the stellar mass.  ``None`` when the tracer has none.  May be
        k-dependent: mis-centering makes it so.
    w_extended : array or None, shape (Nk, NM)
        The part that follows a profile, **already multiplied by** its
        :math:`u(k|M)`.
    norm : array
        :math:`\bar n` for a discrete tracer, ``1.0`` for a continuous field.
    discrete : bool
        Static.  Selects the self-pair rule in the one-halo auto-spectrum.
    bias_weight : array or None, shape (NM,) **or** (Nk, NM)
        Overrides :math:`b(M)` in the two-halo integrand.  For a tracer whose
        clustering is not the host halo's.  ``None`` means "use the field's
        ``bias``".

        **The two shapes are both legal and they are not interchangeable.**
        :math:`(N_M,)` is a re-weighting at fixed halo mass, which is what an
        assembly-bias decoration would be (none is shipped).  :math:`(N_k, N_M)` is a *scale-dependent* bias, which
        is what primordial non-Gaussianity would give -- an extension kept in
        :mod:`ggah_mod.sectors.png` and not exported, because the two-point
        function alone does not constrain it -- and the second was admitted only after
        :func:`~ggah_mod.spectra.pk.i_of_k` was checked to broadcast over it
        rather than assumed to -- it multiplies by a :math:`(N_k, N_M)` weight
        before integrating over mass, so both work.

        That check was not a formality.  A decoration once reduced
        the extended part of a weight to its :math:`k\to0` row and not the
        point part, and a :math:`(N_k, N_M)` result **broadcast rather than
        raised**, giving the two-halo term a k dependence belonging to the
        one-halo profile.  The same silence would hide a decoration applied at
        the wrong shape here.
    self_pair : array or None, shape (NM,)
        :math:`\langle N(M)\rangle\,\ell(M)^2` -- the per-halo weight of an
        object paired with **itself**, where :math:`\ell` is the per-object
        weight.  ``None`` for a continuous field, which has no objects.

        It is carried rather than derived because it cannot be derived.  The
        shot-noise term is
        :math:`\int dM\,(dn/dM)\,\langle N\rangle\ell^2/n^2`, and
        ``w_point`` is the *product* :math:`N\ell`: for number counts
        (:math:`\ell = 1`) that recovers :math:`1/\bar n`, but for a
        luminosity-weighted point population -- AGN X-ray emission is one --
        :math:`N\ell^2 \ne N\ell`, and no amount of algebra on the fused
        weight separates them.  Inferring "number counts" from ``norm == 1`` or
        from the tracer's name is exactly the kind of guess this contract
        exists to remove.
    name : str
        For error messages and for labelling a parity-budget row.
    neutrino_weight : array or None, scalar
        How much of the **neutrino** field this tracer carries, per unit of
        total matter: ``1`` for the matter field, :math:`b_I` for the
        intrinsic-alignment field that is matter times a number, and ``None``
        for everything built on the cold field alone -- galaxies, gas, AGN.

        The neutrinos are in no halo, so they have no weight *per halo* and
        cannot ride on ``w_point`` or ``w_extended``.  What they do have is a
        linear fluctuation above the free-streaming scale, fully correlated
        with the cold one, and layer 4 adds it to the two-halo amplitude
        (:mod:`~ggah_mod.spectra.neutrinos`).  This field is the only thing a
        sector says about it: *whether*, and with what coefficient.

        Refused for a discrete tracer, which counts objects and has no
        neutrinos to count.
    w_unresolved : array or None, shape (Nk,)
        What the matter **below the mass grid** carries of this tracer, per
        unit of the cold mean density: the two-halo counterterm multiplies it
        by the bias-weighted deficit (:mod:`~ggah_mod.spectra.counterterm`).
        In the units of the normalised weight, :math:`W/n`.

        ``None`` means the smallest resolved halo stands in:
        :math:`W(k|M_{\min})\,\bar\rho_{cb}/M_{\min}`, the tracer's weight
        per unit mass extrapolated below the grid.  A sector declares a value
        when that extrapolation is the wrong physics -- the hot-gas views
        declare zero, because the baryons of haloes below
        :math:`10^{10}\,M_\odot/h` are not in a hot atmosphere, and the ejecta
        sector books them as diffuse gas instead.  The census reports the same
        composition (:attr:`~ggah_mod.sectors.census.BaryonCensus.omega_outside_by_phase`).

        Refused for a discrete tracer, whose counterterm is zero.
    """

    w_point: Optional[jnp.ndarray]
    w_extended: Optional[jnp.ndarray]
    norm: jnp.ndarray
    discrete: bool
    bias_weight: Optional[jnp.ndarray]
    name: str
    self_pair: Optional[jnp.ndarray] = None
    neutrino_weight: Optional[jnp.ndarray] = None
    w_unresolved: Optional[jnp.ndarray] = None

    def __post_init__(self):
        # A statement about *presence*, so it is safe on a traced instance: a
        # tracer is not None either.
        if self.discrete and self.neutrino_weight is not None:
            raise ValueError(
                f"tracer {self.name!r} is discrete and carries a neutrino "
                f"weight.  A discrete tracer counts objects, and the neutrinos "
                f"are in no object: only a continuous field -- matter, or a "
                f"field that is matter times a number -- can carry them.")
        if self.discrete and self.w_unresolved is not None:
            raise ValueError(
                f"tracer {self.name!r} is discrete and declares an unresolved "
                f"weight.  The low-mass counterterm is zero for a discrete "
                f"tracer -- a threshold sample does not continue below its "
                f"threshold -- so there is nothing for it to weight.")

    # ----------------------------------------------------------------- pytree
    def tree_flatten(self):
        """Arrays travel as children; the two static choices as aux data."""
        return ((self.w_point, self.w_extended, self.norm, self.bias_weight,
                 self.self_pair, self.neutrino_weight, self.w_unresolved),
                (self.discrete, self.name))

    @classmethod
    def tree_unflatten(cls, aux, children):
        w_point, w_extended, norm, bias_weight, self_pair, nu_w, unres = children
        discrete, name = aux
        return cls(w_point, w_extended, norm, discrete, bias_weight, name,
                   self_pair, nu_w, unres)

    def replace(self, **kw) -> "TracerWeights":
        """A copy with some fields changed."""
        return replace(self, **kw)

    # ------------------------------------------------------------------ total
    def total(self, n_k: int | None = None):
        r""":math:`W(k|M) = (w_{\rm point} + w_{\rm extended})/n`, shape (Nk, NM).

        Only for the checks that want the summed weight -- the k->0 limit, the
        mass-conservation assertion.  Layer 4 uses the parts, because it needs
        the pair rule.
        """
        parts = [p for p in (self.w_point, self.w_extended) if p is not None]
        if not parts:
            raise ValueError(
                f"tracer {self.name!r} has neither a point nor an extended "
                f"component, so it has no weight at all")
        for p in parts:
            # Rank, not shape: `(NM,)` and `(Nk, NM)` are both legal and are
            # told apart by it, so a third axis is not a variant to absorb.  It
            # arrives when a sector is handed a redshift-stacked HaloField --
            # `jax.vmap(make_field)` builds one, and it is a legal pytree and an
            # illegal argument here.  Absorbed, `at_large_scales` below would
            # return a *redshift slice* under the name of the k -> 0 row.
            if jnp.asarray(p).ndim > 2:
                raise ValueError(
                    f"tracer {self.name!r} has a weight of shape "
                    f"{jnp.asarray(p).shape}; a weight is (NM,) or (Nk, NM).  A "
                    f"leading redshift axis is how this happens: layers 3 and "
                    f"4 are one epoch per object, so `jax.vmap` the whole "
                    f"z -> spectrum closure rather than the field alone.")
        out = sum(jnp.atleast_2d(jnp.asarray(p)) for p in parts) / self.norm
        if n_k is not None and out.shape[0] == 1:
            out = jnp.broadcast_to(out, (n_k, out.shape[-1]))
        return out

    def at_large_scales(self):
        r""":math:`W(k\to0|M)`: every :math:`u\to1`, so the profiles drop out.

        The quantity mass conservation is stated in, and the one number a
        sector can be checked on before layer 4 exists.
        """
        return self.total()[0]

    def with_point(self, w_point) -> "TracerWeights":
        """A copy with the point part replaced -- how a modifier applies."""
        return replace(self, w_point=w_point)


@runtime_checkable
class Sector(Protocol):
    """A tracer of the halo field.  A **peer**, never a host.

    A sector owns its parameters and nothing else.  In particular it does not
    own a mass grid: it is handed a
    :class:`~ggah_mod.halos.field.HaloField` and reads one.  That is the whole
    point of the layer -- it is what lets a pure hot-gas or AGN spectrum be
    computed without inventing galaxy parameters for it, which the predecessor
    could not do because the mass grid lived inside the galaxy occupation.
    """

    #: Short label, unique across the layer.
    name: str
    #: Always ``True`` here, and checked by trying it.
    differentiable: bool

    def weights(self, field, params) -> TracerWeights:
        """The tracer's weights on ``field``'s grids."""
        ...


@runtime_checkable
class ProfileModifier(Protocol):
    r"""A multiplicative :math:`h(k|M)` applied to a sector's point component.

    Mis-centering is the archetype: it does not change *how many* objects there
    are, only where they sit, so it multiplies :math:`w_{\rm point}` and leaves
    ``norm`` alone.  Written as a modifier rather than inlined because the
    predecessor inlined it -- twenty lines, duplicated across its numpy and JAX
    paths, plus a second differently-named implementation elsewhere -- and the
    two paths could take different branches for the same input.
    """

    name: str
    differentiable: bool

    def h_k(self, field, params) -> jnp.ndarray:
        """Shape ``(Nk, NM)``, tending to 1 as ``k -> 0``."""
        ...


def combine(*tracers: TracerWeights, name: str = "combined") -> TracerWeights:
    r"""Add several tracers' weights into one.

    For a composite field -- ``MatterField`` is CDM plus gas plus stars -- where
    the sum is the physical object and the parts are not separately observable.
    Continuous only: adding two discrete tracers would need a pair rule between
    them, which is layer 4's cross-spectrum, not this.
    """
    if any(t.discrete for t in tracers):
        raise ValueError(
            "combine() is for continuous fields.  Adding discrete tracers "
            "needs a cross-pair rule, which is what layer 4's cross-spectrum "
            "is; summing their weights would silently assume they are the "
            "same population.")
    point = [t.w_point for t in tracers if t.w_point is not None]
    ext = [t.w_extended for t in tracers if t.w_extended is not None]
    # Summed like the halo parts: a composite carries the neutrinos its parts
    # carry, and none if none of them does.
    nu = [t.neutrino_weight for t in tracers if t.neutrino_weight is not None]
    # All declared, or none: a sum of one declared share and one stand-in
    # would need the stand-in's M_min column, which is layer 4's to read.
    unres = [t.w_unresolved for t in tracers]
    if any(u is None for u in unres) and any(u is not None for u in unres):
        raise ValueError(
            "combine() got tracers of which some declare w_unresolved and some "
            "do not; declare it on all of them, or on none")
    norms = jnp.asarray([jnp.asarray(t.norm) for t in tracers])
    if not jnp.all(norms == norms[0]):
        raise ValueError(
            "combine() requires one normalisation; these tracers disagree, so "
            "their weights are not in the same units")
    return TracerWeights(
        w_point=sum(point) if point else None,
        w_extended=sum(ext) if ext else None,
        norm=norms[0], discrete=False, bias_weight=None, name=name,
        self_pair=None, neutrino_weight=sum(nu) if nu else None,
        w_unresolved=None if unres[0] is None else sum(unres))
