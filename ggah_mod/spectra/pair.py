r"""The pair rule: the **only** physics branch in layer 4's integrand.

:mod:`~ggah_mod.sectors.protocol` claims that ``discrete`` is the one branch in
the one-halo integrand.  This module is that branch, alone, so
:mod:`~ggah_mod.spectra.pk` can contain none and a test can say so.

.. math::

    \text{pair}_{ab}(k|M) = \begin{cases}
      2\,w_{\rm point}w_{\rm ext} + w_{\rm ext}^2 & a \equiv b,\ \text{discrete}\\
      (w^a_{\rm point} + w^a_{\rm ext})(w^b_{\rm point} + w^b_{\rm ext}) &
        \text{otherwise}
    \end{cases}

The first line is :math:`\langle N(N-1)\rangle` rather than :math:`\langle
N\rangle^2`: a central galaxy cannot be paired with itself, and satellites are
Poisson about the central.  Getting it wrong changes :math:`P_{gg}^{1h}` at the
:math:`1/N_c` level with nothing raising -- which is why
:class:`~ggah_mod.sectors.protocol.TracerWeights` keeps the point and extended
parts apart in the first place, and why fusing them into one array makes the
distinction unrecoverable.

The second line is right **only for disjoint samples**, which is what
:func:`~ggah_mod.spectra.spec.overlap_of` is for.

``None`` is not zero
--------------------

:attr:`~ggah_mod.sectors.protocol.TracerWeights.w_extended` is ``None`` for a
centrals-only tracer -- the galaxies' ``"cen"`` view, or AGN at
``f_duty_sat = 0`` -- and that is a physical statement -- "a Bernoulli central
occupation has no self-pairs, so the one-halo auto-spectrum vanishes
identically" -- not a small number.  It is branched on in **Python**, which is static and jit-safe, so the
statement survives into the arithmetic instead of becoming a multiply by zero.
"""

from __future__ import annotations

import jax.numpy as jnp

from .spec import Overlap

__all__ = ["normalised_parts", "pair_1h", "shot_noise"]


def normalised_parts(w, n_k: int):
    r""":math:`(w_{\rm point}/n,\ w_{\rm ext}/n)`, each ``(Nk, NM)`` or ``None``.

    Broadcasting happens once, here.  ``w_point`` may arrive as ``(NM,)`` or as
    ``(Nk, NM)`` -- mis-centering makes it the second -- and every consumer
    downstream would otherwise have to know which.

    ``None`` is preserved rather than turned into zeros: see the module
    docstring.

    The rank is what tells the two cases apart, so a **third** axis is refused
    rather than absorbed.  A weight built on a redshift-stacked
    :class:`~ggah_mod.halos.field.HaloField` -- which ``jax.vmap(make_field)``
    legitimately produces -- arrives as ``(Nz, Nk, NM)``, whose leading axis is
    ``Nz``: the test below would then read ``Nz != 1``, decline to broadcast,
    and hand layer 4 an array it treats as ``(Nk, NM)`` with the redshift in
    the wavenumber's place.  Nothing about the result would look wrong.  Layers
    2 to 4 are one epoch per object; map the whole ``z -> spectrum`` closure,
    not the field alone.
    """
    def prep(x):
        if x is None:
            return None
        x = jnp.asarray(x) / w.norm
        if x.ndim > 2:
            raise ValueError(
                f"a tracer weight is (NM,) or (Nk, NM); got {x.shape}.  A "
                f"leading redshift axis is the way this happens, and it is not "
                f"a supported input to layer 4 -- `jax.vmap` the whole "
                f"z -> spectrum closure rather than a HaloField on its own.")
        x = jnp.atleast_2d(x)
        if x.shape[0] == 1 and n_k != 1:
            x = jnp.broadcast_to(x, (n_k, x.shape[-1]))
        return x

    return prep(w.w_point), prep(w.w_extended)


def _total(point, ext):
    """``w_point + w_extended``, with either or both absent."""
    if point is None:
        return ext
    if ext is None:
        return point
    return point + ext


def pair_1h(wa, wb, n_k: int, *, overlap):
    r"""The one-halo integrand's mass-dependent factor, shape ``(Nk, NM)``.

    Parameters
    ----------
    wa, wb : TracerWeights
    n_k : int
        Static; the wavenumber-grid size to broadcast to.
    overlap : str
        From :func:`~ggah_mod.spectra.spec.overlap_of`.

    Returns
    -------
    array (Nk, NM), or ``None`` when the pair contributes nothing at all --
    which happens for the auto-spectrum of a discrete tracer with no extended
    component, and is a prediction rather than a rounding.
    """
    pa, ea = normalised_parts(wa, n_k)

    if isinstance(overlap, Overlap):
        pb, eb = normalised_parts(wb, n_k)
        ta, tb = _total(pa, ea), _total(pb, eb)
        if ta is None or tb is None:
            return None
        if overlap.point == "disjoint" or pa is None or pb is None:
            # Nothing declared shared, or one side has no point part to share.
            return ta * tb
        # One central per halo: the point-point product is not a pair at all.
        return ta * tb - pa * pb

    if overlap == "identical":
        if not wa.discrete:
            t = _total(pa, ea)
            return None if t is None else t * t
        # <N(N-1)>: the point component never pairs with itself.
        if ea is None:
            return None                  # centrals only -> no one-halo term
        if pa is None:
            return ea * ea               # satellites only
        return 2.0 * pa * ea + ea * ea

    if overlap == "none":
        pb, eb = normalised_parts(wb, n_k)
        ta, tb = _total(pa, ea), _total(pb, eb)
        return None if (ta is None or tb is None) else ta * tb

    if overlap == "nested":
        raise NotImplementedError(
            "a nested pair -- every object of one tracer also an object of the "
            "other -- needs to know *which components* are shared, and the "
            "contract does not carry that: `population` says the samples "
            "overlap, not that (say) the AGN centrals are a subset of the "
            "galaxy centrals while the satellites are unrelated.  The rule "
            "differs per component pair, so guessing one here would put a "
            "specific and unstated model into every AGN-galaxy cross-spectrum. "
            "Declare it: pass `spectra.spec.Overlap(point=..., shared=..., "
            "why=...)` as the value in `spectrum(..., overlaps={(pop_a, pop_b): "
            "...})`, or declare the samples disjoint ('none') if that is what "
            "they are.  Splitting the tracer into components whose populations "
            "are each identical or disjoint does NOT work, and the advice to do "
            "so was wrong: `GalaxySector.weights` normalises each view to that "
            "view's own n_bar, so the sum of the parts is not the sample.")

    raise ValueError(f"unknown overlap {overlap!r}")


def _shared_per_halo(wa, wb, kind: str):
    r"""Objects in **both** samples, per halo, for a declared cross shot noise.

    ``"point"``: :math:`N^p_aN^p_b`.  A halo has one central; it is in sample a
    with probability :math:`N^p_a` and in b with :math:`N^p_b`, and taking the
    product asserts those selections are independent at fixed halo mass.  That
    is the model :class:`~ggah_mod.spectra.spec.Overlap` exists to make the
    caller state.  It refuses a k-dependent ``w_point`` -- mis-centring makes
    one -- because a count of shared objects is not a function of wavenumber.

    ``"subset"``: :math:`\min(\langle N_a\rangle, \langle N_b\rangle)`, the
    nested-threshold case, where every object of the smaller sample is in the
    larger.  Exact only for number counts, where ``self_pair`` *is* the
    occupation; for a weighted population the shared term is
    :math:`\langle N\ell_a\ell_b\rangle` and neither self-pair carries it.
    """
    if kind == "point":
        for w in (wa, wb):
            if w.w_point is None:
                raise ValueError(
                    f"shared='point' needs both tracers to have a point "
                    f"component; {w.name!r} has none.")
            if jnp.asarray(w.w_point).ndim > 1:
                raise ValueError(
                    f"shared='point' needs a wavenumber-independent point "
                    f"weight, and {w.name!r}'s is (Nk, NM) -- mis-centring is "
                    f"how that happens.  How many objects two samples share is "
                    f"not a function of k, so there is no honest reduction "
                    f"here; declare the pair some other way.")
        return jnp.asarray(wa.w_point) * jnp.asarray(wb.w_point)

    if kind == "subset":
        for w in (wa, wb):
            if w.self_pair is None:
                raise ValueError(
                    f"shared='subset' reads each tracer's occupation from "
                    f"`self_pair`, and {w.name!r} declares none.")
        return jnp.minimum(jnp.asarray(wa.self_pair), jnp.asarray(wb.self_pair))

    raise ValueError(f"unknown shared rule {kind!r}")


def shot_noise(field, wa, wb, *, overlap):
    r"""The self-pair term,
    :math:`\int dM\,(dn/dM)\,\langle N\rangle\ell^2/\bar n^2`.

    Reduces to :math:`1/\bar n` for number counts, where
    :attr:`~ggah_mod.sectors.protocol.TracerWeights.self_pair` is the occupation
    itself -- and does **not** for a luminosity-weighted point population, which
    is why the contract carries the quantity rather than this function assuming
    it.

    **Carried, never added.**  A pair-counted :math:`w_p` or
    :math:`\Delta\Sigma` has it removed by construction; a :math:`C_\ell`
    measured from two maps needs it, unless the data was already debiased.
    Which applies is a property of the *estimator*, so it is layer 5's to
    decide.

    It carries a gradient, and should: :math:`\bar n` is predicted by the model,
    so :math:`\partial(1/\bar n)/\partial\theta` is a real term in a Fisher
    matrix and a ``stop_gradient`` here would drop it.
    """
    zero = jnp.zeros_like(jnp.asarray(wa.norm) * 1.0)

    if isinstance(overlap, Overlap):
        if overlap.shared == "none" or not (wa.discrete and wb.discrete):
            return zero
        return field.integrate(
            field.dndm * _shared_per_halo(wa, wb, overlap.shared)
        ) / (wa.norm * wb.norm)

    if overlap != "identical" or not wa.discrete:
        return zero
    if wa.self_pair is None:
        raise ValueError(
            f"tracer {wa.name!r} is discrete but declares no `self_pair`, so "
            f"its shot noise cannot be computed.  It is <N> * l^2 per halo, "
            f"with l the per-object weight: the occupation itself for number "
            f"counts, and something else for any weighted population.  It is "
            f"not derivable from `w_point`, which is the product N*l.")
    return field.integrate(field.dndm * wa.self_pair) / (wa.norm ** 2)
