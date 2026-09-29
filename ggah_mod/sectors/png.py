r"""Scale-dependent bias from primordial non-Gaussianity.

``PLAN.md`` item **E4**, and the benchmark's cost estimate for it: *"low -- a
k-dependent additive term on b(M), given the existing b(M) and transfer
function."*  That was right about the size and slightly wrong about the shape:
the transfer function is not stored anywhere, because this package never needed
one -- it works from :math:`P(k)` throughout -- so it is recovered from the
spectrum here rather than read off.

The term
--------

Dalal et al. (2008).  A local :math:`f_{\rm NL}` couples long- and
short-wavelength modes, so a halo's abundance responds to the large-scale
potential and its bias acquires a :math:`1/k^2` tail:

.. math::

    \Delta b(k, z) = 3 f_{\rm NL}\,(b - p)\,\delta_c\,
        \frac{\Omega_m}{T(k)\,D(z)}\left(\frac{H_0}{c\,k}\right)^2

with :math:`T` normalised to one at large scales and :math:`D` to one today.

:math:`p` is the one modelling choice and it is **not** given a default that
hides it.  It is 1 for a population that occupies every halo -- which is what
an HOD describes -- and 1.6 for objects selected by a recent merger, quasars
being the case the literature quotes.  The difference is a 60 per cent change in
the amplitude of the signal, so a default of 1 taken silently by a quasar
analysis would be a large error wearing a sensible number.

Why it needs no new plumbing
----------------------------

:attr:`~ggah_mod.sectors.protocol.TracerWeights.bias_weight` already overrides
:math:`b(M)` in the two-halo integrand.  This override is :math:`(N_k, N_M)`,
and :func:`~ggah_mod.spectra.pk.i_of_k` broadcasts correctly over it, because
it multiplies by a :math:`(N_k, N_M)` weight before integrating over mass.
That was verified rather than assumed.

**It does not reach the counterterm, and must not.**
:func:`~ggah_mod.spectra.counterterm.bias_consistency` is the statement *the
mass-weighted bias of all matter is one*, a property of the (mass function,
bias) pairing.  A PNG shift re-weights one tracer's bias at fixed halo mass; it
does not change how much bias-weighted mass the grid is missing, and a
:math:`1/k^2` divergence entering a normalisation condition would be a
:math:`k \to 0` limit that does not exist.  ``PLAN.md`` item **A3** settled that
asymmetry before this arrived, which is why this module adds nothing there.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from ..cosmology import constants as C
from ..halos.variance import DELTA_C
from .protocol import TracerWeights

__all__ = ["P_UNIVERSAL", "P_MERGER", "transfer_from_spectrum",
           "png_bias_shift", "apply_png_bias"]

#: :math:`p` for a population occupying every halo -- what an HOD describes.
P_UNIVERSAL = 1.0

#: :math:`p` for objects selected by a recent merger; quasars, in Slosar et al.
#: (2008).  Sixty per cent more signal than :data:`P_UNIVERSAL` at the same
#: :math:`f_{\rm NL}`, which is why neither is a default.
P_MERGER = 1.6


def transfer_from_spectrum(k, pk, n_s):
    r""":math:`T(k)`, normalised to one at the grid's largest scale.

    :math:`P(k) \propto k^{n_s}T^2(k)`, so :math:`T \propto \sqrt{P/k^{n_s}}`
    and the constant is fixed by :math:`T \to 1` as :math:`k \to 0`.

    **The normalisation is the grid's first node, not zero**, and the difference
    is the one approximation in this module.  On the shipped grids that node is
    :math:`k = 10^{-4}` h/Mpc, where :math:`T` differs from one by less than
    :math:`10^{-4}` -- far inside anything a :math:`f_{\rm NL}` measurement
    resolves.  On a grid that started at :math:`10^{-2}` it would not be, and
    the resulting :math:`\Delta b` would be low by whatever :math:`T` had
    already fallen to, uniformly and invisibly.  Stated because this function
    cannot see the grid it was handed.
    """
    k = jnp.asarray(k)
    pk = jnp.asarray(pk)
    t = jnp.sqrt(pk / jnp.power(k, n_s))
    return t / t[0]


def png_bias_shift(k, bias, f_nl, cosmo, pk, *, p: float, growth: float,
                   transfer=None):
    r""":math:`\Delta b(k)`, shape ``(Nk,)`` or ``(Nk, NM)``.

    Parameters
    ----------
    k : array (Nk,)
        Wavenumbers [h/Mpc].
    bias : float or array (NM,)
        The Gaussian bias the shift is added to.  Broadcasts, so a per-mass
        :math:`b(M)` gives a per-mass shift.
    f_nl : float
        Local-type :math:`f_{\rm NL}`.
    cosmo : Cosmology
        For :math:`\Omega_m` and :math:`n_s`.
    pk : array (Nk,)
        A linear spectrum on ``k``, for :math:`T(k)`.  Any redshift will do:
        :func:`transfer_from_spectrum` normalises at the first node, and the
        growth factor is k-independent, so it divides out exactly.
    p : float
        Required; see the module docstring.
    growth : float
        :math:`D(z)`, **normalised to one today**, and required for the same
        reason ``p`` is.  It cannot be taken from ``pk``: the normalisation
        that recovers :math:`T` from a spectrum divides :math:`D` out, and a
        :class:`~ggah_mod.halos.field.HaloField` carries a spectrum rather than
        the backend that made it, so there is nothing to evaluate at a second
        redshift.  Guessing a convention here is the specific error this
        argument exists to prevent -- the literature normalises :math:`D` at
        :math:`z=0` and in matter domination, and the two differ by about 25
        per cent at Planck parameters, which would read as a shift in
        :math:`f_{\rm NL}`.
    transfer : array (Nk,), optional
        A precomputed :math:`T(k)`; otherwise taken from ``pk``.
    """
    k = jnp.asarray(k)
    t = (transfer_from_spectrum(k, pk, cosmo.n_s) if transfer is None
         else jnp.asarray(transfer))

    # (H0/c)^2 in (h/Mpc)^2: H0 = 100 h km/s/Mpc, so H0/c = 1/2997.92458 h/Mpc
    # and the h cancels against k's.  Written this way the whole expression
    # stays in the package's units with no conversion at all -- which is the
    # half of a PNG term that is easy to get wrong by a factor of h^2.
    h0_over_c2 = (100.0 / C.C_KM_S) ** 2
    b_minus_p = jnp.asarray(bias) - p
    shift = (3.0 * f_nl * DELTA_C * cosmo.Omega_m * h0_over_c2
             / (t * k ** 2 * growth))
    if jnp.ndim(b_minus_p) == 0:
        return shift * b_minus_p
    return shift[:, None] * b_minus_p[None, :]


def apply_png_bias(weights: TracerWeights, field, f_nl, *, p: float,
                   growth: float) -> TracerWeights:
    r"""Add the PNG shift to a tracer's two-halo bias.

    Returns a new :class:`~ggah_mod.sectors.protocol.TracerWeights` whose
    ``bias_weight`` is :math:`(N_k, N_M)`.  The one-halo term is untouched -- a
    within-halo pair count knows nothing about the long-wavelength potential --
    which is why it lives on this attribute rather than on the profile.

    ``growth`` is required; see :func:`png_bias_shift`.
    """
    base = field.bias if weights.bias_weight is None else weights.bias_weight
    if jnp.ndim(base) == 2:
        raise ValueError(
            "this tracer's bias_weight is already k-dependent, so a PNG shift "
            "would be added to a decoration rather than to b(M). Apply the "
            "PNG term first, or decide which of the two the other should see "
            "-- there is no general answer, and guessing one would put an "
            "unstated model into every spectrum.")

    shift = png_bias_shift(field.k, base, f_nl, field.cosmo, field.pk_cb,
                           p=p, growth=growth)
    return dataclasses.replace(
        weights, bias_weight=jnp.asarray(base)[None, :] + shift,
        name=f"{weights.name}+png")
