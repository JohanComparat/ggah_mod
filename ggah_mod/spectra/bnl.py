r"""Beyond-linear bias, as an explicit hook.

:mod:`ggah_mod.halos.beyond_linear_bias` has the kernel -- 35 MDR1 snapshots, an
Angulo-White rescaling, and the two correction integrals.  ``PLAN.md`` deferred
the *wiring* with a precise reason: it is a two-halo correction, not a profile,
so it does not fit the tracer abstraction and belongs behind an explicit hook.
This is that hook, and it is **on by default**: the two-halo term of
:math:`P_{\rm lin} I_a I_b` is 10-50 per cent short across the transition
(Mead & Verde 2021), and against CLASS's HMcode-2020 the corrected matter
spectrum is within 3.2 per cent (rms, :math:`0.05 < k < 1`) at
:math:`z = 0.5` where the linear-bias one is 11.8 per cent off.

The table is built from the field when the caller passes none
(:func:`table_for`).  That is not layer 4 reaching for a cosmology: the
Angulo-White rescaling reads the target through its cold spectrum alone, and
:attr:`~ggah_mod.halos.field.HaloField.pk_cb` *is* that spectrum at the
field's redshift.  A caller that evaluates many spectra on one field builds it
once and passes it, as :func:`~ggah_mod.spectra.tracers.spectrum` does.

The low-mass completion
-----------------------

The counterterm (:mod:`~ggah_mod.spectra.counterterm`) puts the bias-weighted
mass missing below :math:`M_{\min}` back at :math:`M_{\min}` with
:math:`b = 1`.  That mass is in haloes too, so it has a beyond-linear term:
Mead & Verde (2021, Appendix A, Eqs. A7-A10) substitute the same point mass
into both mass integrals and get

.. math::

    I^{\rm NL}_{ab} = I^{22} + A_a\hat W_a \sum_j \beta(\nu_{\min},\nu_j) w_{b,j}
                     + A_b\hat W_b \sum_i \beta(\nu_i,\nu_{\min}) w_{a,i}
                     + A_aA_b\,\beta(\nu_{\min},\nu_{\min})\,\hat W_a\hat W_b ,

with :math:`A\hat W = \Delta I` the counterterm.  Both reference codes carry
the :math:`\beta` in the last term (pyhalomodel ``_I_beta``, HMx ``Inl_11``);
the v2 text of the paper omits it.  Here the four terms are one line: the
counterterm is added to the first mass column of the fused weight before it is
projected onto the table, because a point mass at a grid node *is* an addition
to that node.  It is zero for a discrete tracer and with
``two_halo_consistency="none"``, exactly as the counterterm is.  It matters:
the completion roughly triples the correction for matter, which is what the
paper found (at least half of the matter-halo improvement, and about five
times the matter-matter one, comes from below the simulation's mass limit).

The fused weight, and the measure it needs
------------------------------------------

:func:`~ggah_mod.halos.beyond_linear_bias.correction_2h_gm` wants
``weights`` ``(NM,)`` and ``uk`` ``(Nk, NM)`` as **separate** arguments and
forms ``weights[None, :] * uk`` itself.  :class:`TracerWeights` deliberately
fuses them -- ``w_extended`` is already multiplied by its :math:`u(k|M)` and
``w_point`` has no :math:`u` at all -- and a scale-dependent ``bias_weight``
cannot be factorised that way either.  So this module hands the kernel the
**fused** integrand, :func:`completed_weight`, through
:func:`~ggah_mod.halos.beyond_linear_bias.correction_2h_fused`.

The kernel contracts with **no quadrature measure**: its sum runs over the mass
index with nothing standing in for :math:`dM`.  The fused weight carries it:
:math:`(dn/dM)\,b\,W\,\Delta M`, whose sum over the mass axis is
:math:`\int dM\,(dn/dM)\,b\,W = I(k)`, exactly the quantity the correction is
a correction *to*.  :meth:`~ggah_mod.halos.field.HaloField.quadrature_measure`
supplies :math:`\Delta M`, and is pinned against
:meth:`~ggah_mod.halos.field.HaloField.integrate` so the two can never drift.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..halos.beyond_linear_bias import correction_2h_fused, table_at
from .counterterm import low_mass_counterterm
from .pair import normalised_parts, _total
from .spec import PkOptions

__all__ = ["effective_weight", "completed_weight", "table_for",
           "two_halo_correction"]


def effective_weight(field, w):
    r""":math:`W_a(k|M)`, shape ``(Nk, NM)`` -- the fused weight the table sees."""
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    point, ext = normalised_parts(w, n_k)
    total = _total(point, ext)
    if total is None:
        return jnp.zeros((n_k, field.n_m))
    return total


def completed_weight(field, w, options: PkOptions = PkOptions()):
    r"""The whole integrand of :math:`I_a(k)`, node by node, shape ``(Nk, NM)``.

    :math:`(dn/dM)\,b_a\,W_a\,\Delta M`, with the tracer's ``bias_weight`` in
    place of :math:`b` when it has one -- either shape broadcasts -- and the
    counterterm added to the first column, which is the low-mass completion.
    Its sum over the mass axis is :func:`~ggah_mod.spectra.pk.i_of_k` exactly.
    """
    c = options.two_halo_consistency
    if c not in ("linear_deficit", "none"):
        raise ValueError(
            f"unknown two-halo consistency {c!r}; expected "
            f"'linear_deficit' or 'none'")
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    bias = field.bias if w.bias_weight is None else w.bias_weight
    out = (field.dndm * field.quadrature_measure()) * bias \
        * effective_weight(field, w)
    out = jnp.broadcast_to(out, (n_k, field.n_m))
    if c == "linear_deficit":
        out = out.at[:, 0].add(low_mass_counterterm(field, w, n_k))
    return out


def table_for(field, options: PkOptions = PkOptions()):
    r"""The :math:`\beta^{\rm NL}` table for ``field``, or ``None`` when off.

    Built from the field's own cold spectrum.  The boundary diagnostics read
    concrete floats, so :func:`~ggah_mod.halos.beyond_linear_bias.table_at`
    skips them whenever the solution is traced -- under ``jit`` that is always,
    even for a concrete field -- which is why a model builder should call this
    once outside the trace if it wants them.
    """
    if not options.bnl:
        return None
    return table_at(field.k, k=field.k, pk_cb=field.pk_cb, check=True)


def two_halo_correction(field, wa, wb, table, *,
                        options: PkOptions = PkOptions()):
    r"""The additive :math:`\beta^{\rm NL}` term on :math:`P^{2h}(k)`.

    Returns a spectrum-shaped array, not a ratio: the kernel's correction is to
    :math:`P^{2h}/P_{\rm lin}`, so it is multiplied back by the same linear
    spectrum the two-halo term used -- read through the one function that
    chooses it.  The neutrinos' linear leg is not here: they are in no halo.
    """
    from .pk import linear_spectrum
    p_lin = linear_spectrum(field, options)
    w_a = completed_weight(field, wa, options)
    w_b = w_a if wb is wa else completed_weight(field, wb, options)
    return p_lin * correction_2h_fused(field.nu, w_a, w_b, table)
