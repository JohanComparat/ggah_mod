r"""One 1-halo + 2-halo integral, for any pair of tracers.

.. math::

    P_{ab}^{1h}(k) = \int dM\,\frac{dn}{dM}\,\mathrm{pair}_{ab}(k|M),
    \qquad
    P_{ab}^{2h}(k) = P_{\rm lin}(k)\,I_a(k)\,I_b(k)

.. math::

    I_a(k) = \int dM\,\frac{dn}{dM}\,b_a(M)\,W_a(k|M)\;+\;\Delta I_a(k)

The predecessor wrote that out **eight times** -- ``_pk_tables_full``,
``_pk_tables_gy``, ``_pk_tables_yy``, ``_pk_tables_gX``, ``_pk_tables_XX``,
``_pk_table_cg``, ``halo_model.py`` and ``forward_jax._pk_tracer_field`` -- and
the copies disagree: :math:`P_{gg}^{2h}` forces :math:`\tilde u\to1` at all k
while :math:`P_{my}^{2h}` uses the real :math:`I_m(k)`, so two spectra in one
analysis make different approximations with nothing recording it.  Here there is
one.

**This module contains no physics branch.**  The self-pair rule is
:mod:`~ggah_mod.spectra.pair`, the low-mass deficit is
:mod:`~ggah_mod.spectra.counterterm`, and a tracer's bias override needs no
branch at all -- it arrives as
:attr:`~ggah_mod.sectors.protocol.TracerWeights.bias_weight` and is read where
:math:`b(M)` would be.  ``tests/test_spectra.py`` asserts this file names no
sector.

Which linear spectrum
---------------------

``two_halo_spectrum="cb"``, the default.  :math:`\sigma(M)`,
:math:`d\ln\sigma/d\ln M`, :math:`dn/dM` and :math:`b(M)` are **all** built from
:attr:`~ggah_mod.halos.field.HaloField.pk_cb`, and the two-halo term is
:math:`\langle\delta_h\delta_h\rangle = b^2P`: the bias is defined against the
field whose variance set it.  Pairing :math:`b(\sigma_{cb})` with the total
spectrum is a category error with no compensating consistency condition, and it
does not become the total-matter answer either -- that needs the additive
neutrino terms :math:`2f_\nu(1-f_\nu)P_{cb\nu} + f_\nu^2P_{\nu\nu}`, which are
linear and belong outside the halo *integral*.  They are supplied by
:mod:`~ggah_mod.spectra.neutrinos` as a leg on the two-halo amplitude,
:math:`\mathcal I_a = I_a + (\nu_a/n_a)L`, so :math:`P^{2h}_{mm} \to P_m`
exactly on large scales -- see :func:`two_halo_amplitude`.

``"total"`` stays selectable so the parity budget can *measure* the difference
rather than assert it.  The ratio tracks :math:`(1-f_\nu)^2`: 0.90% at
:math:`\Sigma m_\nu = 0.06` eV and 4.45% at 0.30 eV -- larger than the
CAMB-versus-CLASS row already in the budget.

Shot noise
----------

Carried on the result, never added to :attr:`PowerSpectrum.total`.  Whether it
belongs is a property of the estimator, not of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .counterterm import low_mass_counterterm
from .neutrinos import tracer_leg
from .pair import normalised_parts, pair_1h, shot_noise, _total
from .spec import PkOptions
from .transition import one_halo_transition

__all__ = ["PowerSpectrum", "i_of_k", "linear_spectrum", "pk_1h",
           "pk_2h", "pk_cross", "two_halo_amplitude"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PowerSpectrum:
    r"""One tracer pair's spectrum, with its terms kept apart.

    The split is not decoration: a 1-halo/2-halo crossover is the first thing
    anyone looks at when a spectrum is wrong, and re-deriving it by subtraction
    loses the shot-noise term entirely.

    Attributes
    ----------
    k : array (Nk,)
    one_halo, two_halo : array (Nk,)
    shot : array, scalar
        :math:`1/\bar n` for a discrete auto-spectrum, else 0.  **Not** in
        :attr:`total`.
    a, b : str
        Static tracer labels.
    """

    k: jnp.ndarray
    one_halo: jnp.ndarray
    two_halo: jnp.ndarray
    shot: jnp.ndarray
    a: str = "a"
    b: str = "b"

    @property
    def total(self):
        """:math:`P^{1h} + P^{2h}`.  Shot noise is deliberately not included."""
        return self.one_halo + self.two_halo

    def with_shot(self):
        """:attr:`total` plus the self-pair term, for a map-based estimator."""
        return self.total + self.shot

    def tree_flatten(self):
        return ((self.k, self.one_halo, self.two_halo, self.shot),
                (self.a, self.b))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def i_of_k(field, w, *, consistency: str = "linear_deficit"):
    r""":math:`I_a(k)`, shape ``(Nk,)``: the two-halo bias integral.

    ``w.bias_weight`` replaces :math:`b(M)` when it is set, which is how a bias
    override such as a scale-dependent shift reaches the two-halo term **and only** the
    two-halo term -- no branch here, and the one-halo term untouched.
    """
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    point, ext = normalised_parts(w, n_k)
    total = _total(point, ext)
    if total is None:
        return jnp.zeros((n_k,))

    bias = field.bias if w.bias_weight is None else w.bias_weight
    integral = field.integrate(field.dndm * bias * total, axis=-1)

    if consistency == "none":
        return integral
    if consistency == "linear_deficit":
        return integral + low_mass_counterterm(field, w, n_k)
    raise ValueError(
        f"unknown two-halo consistency {consistency!r}; expected "
        f"'linear_deficit' or 'none'")


def pk_1h(field, wa, wb, *, overlap: str):
    r""":math:`P^{1h}_{ab}(k)`, shape ``(Nk,)``."""
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    pair = pair_1h(wa, wb, n_k, overlap=overlap)
    if pair is None:
        # Not a small number: a discrete tracer with nothing on a profile has
        # no self-pairs at all, so the term is identically zero.
        return jnp.zeros((n_k,))
    return field.integrate(field.dndm * pair, axis=-1)


def linear_spectrum(field, options: PkOptions = PkOptions()):
    r"""The linear :math:`P(k)` the options select, shape ``(Nk,)``.

    One reader for one choice.  The two-halo term multiplies it and the
    transition of :mod:`~ggah_mod.spectra.transition` measures its dispersion,
    and those two have to be the *same* spectrum: a damping scale derived from
    the total-matter spectrum applied to a two-halo term built on the cold one
    is two conventions in one expression, and the difference tracks
    :math:`(1-f_\nu)^2` rather than cancelling.
    """
    if options.two_halo_spectrum == "cb":
        return field.pk_cb
    if options.two_halo_spectrum == "total":
        return field.pk_lin
    raise ValueError(
        f"unknown two-halo spectrum {options.two_halo_spectrum!r}; "
        f"expected 'cb' or 'total'")


def two_halo_amplitude(field, w, options: PkOptions = PkOptions()):
    r""":math:`\mathcal I_a(k) = I_a(k) + (\nu_a/n_a)L(k)`, shape ``(Nk,)``.

    The halo integral of :func:`i_of_k` plus the tracer's share of the linear
    neutrino leg (:func:`~ggah_mod.spectra.neutrinos.tracer_leg`).  When the
    tracer carries no neutrinos, or the leg is off, this *is*
    :func:`i_of_k` -- the same array, not a sum with zero -- so the cold-only
    result is kept bit for bit.  :func:`i_of_k` itself is unchanged: it is the
    halo integral, and three tests pin it as that.
    """
    halo = i_of_k(field, w, consistency=options.two_halo_consistency)
    leg = tracer_leg(field, w, options)
    return halo if leg is None else halo + leg


def pk_2h(field, wa, wb, *, options: PkOptions = PkOptions()):
    r""":math:`P^{2h}_{ab}(k) = P_{\rm lin}(k)\mathcal I_a(k)\mathcal I_b(k)`, shape ``(Nk,)``."""
    p_lin = linear_spectrum(field, options)
    return (p_lin * two_halo_amplitude(field, wa, options)
            * two_halo_amplitude(field, wb, options))


def pk_cross(field, wa, wb, *, overlap: str, options: PkOptions = PkOptions(),
             bnl_table=None, transition_params=None, a: str | None = None,
             b: str | None = None) -> PowerSpectrum:
    r"""The full spectrum for one tracer pair.

    Parameters
    ----------
    field : HaloField
    wa, wb : TracerWeights
    overlap : str
        From :func:`~ggah_mod.spectra.spec.overlap_of`.  Required, and there is
        no default: for two discrete tracers the answer differs by
        :math:`1/\bar n` at every k, and a default would pick one silently.
    options : PkOptions
    bnl_table : BetaNLTable, optional
        Read when ``options.bnl``; built from ``field`` when not passed
        (:func:`~ggah_mod.spectra.bnl.table_for`).  The rescaling that builds it
        reads the target through its cold spectrum alone, which the field
        carries at its own redshift, so this reaches for nothing the caller did
        not give.  Pass one to share it across many pairs on one field.
    transition_params : TransitionParams, optional
        The one-halo transition's value, as
        :class:`~ggah_mod.spectra.transition.TransitionParams`.  ``None`` takes
        the shipped default.
    """
    one = pk_1h(field, wa, wb, overlap=overlap)
    one = one * one_halo_transition(
        linear_spectrum(field, options), field.k,
        transition=options.one_halo_transition, params=transition_params)
    two = pk_2h(field, wa, wb, options=options)
    if options.bnl:
        from .bnl import table_for, two_halo_correction
        table = bnl_table if bnl_table is not None else table_for(field, options)
        two = two + two_halo_correction(field, wa, wb, table, options=options)
    return PowerSpectrum(
        k=field.k, one_halo=one, two_halo=two,
        shot=shot_noise(field, wa, wb, overlap=overlap),
        a=a or wa.name, b=b or wb.name)
