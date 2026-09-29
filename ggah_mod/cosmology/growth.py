r"""The linear growth factor -- one route, read off the spectrum.

:math:`D(z)` here is defined as

.. math::

    D(z) \equiv \frac{\sigma_8(z)}{\sigma_8(0)}

that is, the ratio of the top-hat variance of the **actual spectrum the rest of
the package uses** at two redshifts.  No fitting formula, no separate ODE, no
mode selection.

Why this definition and not an ODE
----------------------------------

The classic approach computes :math:`\sigma(M, z) = \sigma(M, 0)\,D(z)` with
:math:`D` from a growth ODE or a fitting function.  That is exactly correct only
while growth is scale-independent, and it stops being so as soon as neutrinos
have mass: free-streaming suppresses small scales more than large ones, so
:math:`\sqrt{P(k,z)/P(k,0)}` genuinely depends on :math:`k`.  At
:math:`\Sigma m_\nu = 0.06` eV that spread is a few times :math:`10^{-3}` --
larger than the error of any decent ODE integrator, so choosing a better ODE
does not help.  A *scalar* :math:`D(z)` is simply ambiguous at that point.

The package resolves this by not needing one where it matters:
:math:`\sigma(M,z)` is built from :math:`P_{cb}(k,z)` **directly**, at the
redshift wanted, so the scale dependence is carried exactly and no growth factor
appears in the halo sector at all.

What remains here is for the places that genuinely want a single number -- an
intrinsic-alignment amplitude, a concentration relation calibrated against
:math:`D(z)`, a redshift-weighting kernel.  For those, :math:`\sigma_8(z)/
\sigma_8(0)` is the honest scalar: it is the growth of the amplitude the rest of
the package normalises to, weighted the way :math:`\sigma_8` weights it.

There is deliberately no ``mode=`` argument.  Its predecessor had three routes,
only one of which any call site reached; the other two were selected by silent
fallback when a backend object was not threaded through, and nothing in the
output recorded which had run.

One backend call per array, not per redshift
--------------------------------------------

Every function here takes an array of redshifts and makes **one** call into the
spectrum backend for it.  That is not a micro-optimisation: a Boltzmann solve is
memoised on the whole redshift tuple
(:class:`~ggah_mod.cosmology.power._BoltzmannBase`), so a loop over the entries
is a loop over *solves*.  :func:`growth_rate` was written this way from the
start -- it assembles its three stencil points into one request -- and
:func:`growth_factor` and :func:`f_sigma8` were not, which cost 57.1 s for six
redshifts where 6.6 s was available.  The row-by-row loop that remains is over
the *returned table*, and is there because
:func:`~ggah_mod.cosmology.amplitude.sigma2_tophat` integrates one spectrum at a
time and refuses a stack rather than quietly integrating over redshift.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .amplitude import sigma_tophat
from . import constants as C

__all__ = ["growth_factor", "growth_rate", "f_sigma8",
           "growth_scale_spread", "SPREAD_K_RANGE"]

#: Wavenumber window for :func:`growth_scale_spread` [h/Mpc].
#:
#: Deliberately narrower than the spectrum's full support, because outside it
#: the ratio is dominated by the *solver*, not by physics.  Measured with CLASS
#: at zero neutrino mass, where the true scale-dependence is nil: 2e-5 inside
#: this window, against 1.5e-3 above k = 10 (linear-sampling and interpolation
#: artefacts) and 6e-4 below k = 1e-3 (super-horizon and residual radiation).
#: Reporting those as "scale-dependent growth" would attribute solver noise to
#: neutrinos.
SPREAD_K_RANGE = (1e-3, 10.0)


def growth_factor(z, cosmo, pk, variant: str = "total", k=None):
    r""":math:`D(z) = \sigma_8(z)/\sigma_8(0)`, normalised to 1 today.

    Parameters
    ----------
    z : float or array
    cosmo : :class:`~ggah_mod.cosmology.parameters.Cosmology`
    pk : linear-P(k) backend
        Must declare ``has_native_z``; a backend without its own redshift
        dependence cannot supply a growth factor, and this raises rather than
        quietly substituting an approximation for it.
    variant : ``"total"`` or ``"cold"``
        Which field to grow.  ``"cold"`` uses :math:`P_{cb}` and is the right
        choice for anything about halos; ``"total"`` for anything about all the
        matter.  They differ once neutrinos have mass.
    """
    if not getattr(pk, "has_native_z", False):
        raise ValueError(
            f"{type(pk).__name__} declares has_native_z = False, so it cannot "
            "supply a growth factor.  Growth in this package is read off the "
            "spectrum; a backend with no redshift dependence has none to read. "
            "Use a backend that does, rather than substituting a fitting "
            "formula -- that substitution is what this package removed.")
    if variant not in ("total", "cold"):
        raise ValueError(f"variant must be 'total' or 'cold', got {variant!r}")

    k = np.logspace(-4, 2, 512) if k is None else np.asarray(k)
    getter = pk.pk if variant == "total" else pk.pk_cb

    # jnp throughout, and no float(): this function is differentiable exactly
    # when its backend is, which is the same rule that governs every other
    # function in the cosmology and halo layers.  Writing it in numpy would
    # have made it a silent hole in the differentiable flavour -- it is called by the
    # intrinsic-alignment amplitude and by concentration relations, both of
    # which a forecast differentiates.
    z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=float))

    # **One** backend call for every requested redshift, the way
    # :func:`growth_rate` requests its stencil -- not one call per z.  A
    # Boltzmann solve is per *cosmology*: it integrates the perturbations once
    # and writes out every redshift it was asked for, so the memoisation in
    # :class:`~ggah_mod.cosmology.power._BoltzmannBase` keys on the whole
    # redshift tuple and a loop of scalar calls pays a full solve per entry.
    # Measured with CLASS at six redshifts: 57.1 s that way against 6.6 s
    # batched.  The normalisation stays a separate scalar call because its key
    # ``(cosmo, (0.0,))`` is shared by every growth quantity at this cosmology,
    # so it is one solve amortised rather than one per call.
    s_0 = sigma_tophat(getter(k, 0.0, cosmo), k, C.R8)
    p_z = getter(k, z_arr, cosmo)                      # (Nz, Nk)
    # Row by row, because `sigma_tophat` takes a 1-D spectrum: handed the whole
    # table it would integrate over the redshift axis.  See
    # :func:`~ggah_mod.cosmology.amplitude.sigma2_tophat`, which now refuses it.
    s_z = jnp.stack([sigma_tophat(p_z[i], k, C.R8) for i in range(len(z_arr))])
    out = s_z / s_0
    return out[0] if jnp.ndim(z) == 0 else out


#: Step in :math:`\ln(1+z)` for the growth-rate stencil.
#:
#: Chosen, not tuned: at float64 the central-difference error falls as
#: :math:`h^2` while round-off grows as :math:`\epsilon/h`, and the two cross
#: near :math:`10^{-5}`.  ``1e-3`` sits two decades above that crossing, where
#: the truncation error is ~1e-6 relative and round-off is invisible.
DLN1PZ = 1e-3


def growth_rate(z, cosmo, pk, variant: str = "total", k=None, h: float = DLN1PZ):
    r""":math:`f(z) = \dd\ln D/\dd\ln a`, from the same :math:`D(z)` as everything else.

    .. math::  f(z) = -\frac{\dd\ln D}{\dd\ln(1+z)}

    by a central difference in :math:`\ln(1+z)`, taken on the growth factor this
    package already defines rather than from a separate ODE or a fitting form.
    One route, as :func:`growth_factor` has one route.

    **Why a difference and not** ``jax.grad``.  The redshift derivative would be
    exact under automatic differentiation -- but only for a backend that is
    differentiable *in the redshift*, which a Boltzmann solver is not.  Taking
    it that way would give the two flavours two different definitions of
    :math:`f`, which is the substitution this package exists to remove.  A
    difference is one definition for both, and it costs nothing that matters:
    the derivative with respect to the *cosmology* is still exact, because a
    difference of two smooth functions of :math:`\theta` is a smooth function of
    :math:`\theta`.  That is the derivative a forecast needs.
    ``tests/test_growth.py`` measures the difference against ``jax.grad`` on the
    differentiable backend, where both exist.

    The three stencil points are requested in a **single** backend call, so a
    Boltzmann solver pays one solve rather than three -- see the memoisation
    note in Sec. on the linear power spectrum.
    """
    if not getattr(pk, "has_native_z", False):
        raise ValueError(
            f"{type(pk).__name__} declares has_native_z = False, so it cannot "
            "supply a growth rate any more than it can supply a growth factor.")
    if variant not in ("total", "cold"):
        raise ValueError(f"variant must be 'total' or 'cold', got {variant!r}")

    kk = np.logspace(-4, 2, 512) if k is None else np.asarray(k)
    getter = pk.pk if variant == "total" else pk.pk_cb

    # Central in ln(1+z) where there is room, forward where there is not: z = 0
    # is the value everyone asks for and z < 0 is not a redshift.  Both stencils
    # are second order, both are *computed*, and the choice between them is made
    # by value with `jnp.where` below rather than by a Python branch.
    #
    # This used to be written in numpy, which made the choice static -- and made
    # `z` concrete with it.  That cost more than it bought: `growth_rate`,
    # `f_sigma8` and the `diemer19` concentration relation could not be traced
    # or differentiated in the redshift at all, so `jax.vmap(make_field)` over
    # z raised `TracerArrayConversionError` for `diemer19` while working for
    # every other relation.  The stencil is three cheap arithmetic lines; there
    # was never anything to save here.
    one_p_z = 1.0 + jnp.atleast_1d(jnp.asarray(z, dtype=float))
    lo = jnp.where(one_p_z * jnp.exp(-h) >= 1.0, one_p_z * jnp.exp(-h), one_p_z)
    hi = jnp.where(one_p_z * jnp.exp(-h) >= 1.0, one_p_z * jnp.exp(h),
                   one_p_z * jnp.exp(2.0 * h))
    mid = jnp.where(one_p_z * jnp.exp(-h) >= 1.0, one_p_z, one_p_z * jnp.exp(h))
    central = one_p_z * jnp.exp(-h) >= 1.0

    z_stencil = jnp.concatenate([lo - 1.0, mid - 1.0, hi - 1.0])
    p_all = getter(kk, z_stencil, cosmo)
    s8 = jnp.stack([sigma_tophat(p_all[i], kk, C.R8)
                    for i in range(len(z_stencil))])
    n = len(one_p_z)
    s_lo, s_mid, s_hi = s8[:n], s8[n:2 * n], s8[2 * n:]

    # d ln D / d ln(1+z) -- D's normalisation cancels in the difference.
    d_central = (jnp.log(s_hi) - jnp.log(s_lo)) / (2.0 * h)
    d_forward = (-3.0 * jnp.log(s_lo) + 4.0 * jnp.log(s_mid)
                 - jnp.log(s_hi)) / (2.0 * h)
    d = jnp.where(central, d_central, d_forward)
    out = -d
    return out[0] if jnp.ndim(z) == 0 else out


def f_sigma8(z, cosmo, pk, variant: str = "total", k=None):
    r""":math:`f\sigma_8(z)`, the combination redshift-space distortions measure.

    :math:`f(z)\,\sigma_8(z)` with both factors read off the same spectrum, so
    the pair is consistent by construction.  This is the quantity a full-shape
    analysis constrains, and its absence is why every ``fsigma8`` row of the
    cross-code comparison was a recorded gap rather than a number.
    """
    kk = np.logspace(-4, 2, 512) if k is None else np.asarray(k)
    getter = pk.pk if variant == "total" else pk.pk_cb
    z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    # One call, not one per redshift -- see the note in :func:`growth_factor`.
    p_z = getter(kk, z_arr, cosmo)                     # (Nz, Nk)
    s8_z = jnp.stack([sigma_tophat(p_z[i], kk, C.R8) for i in range(len(z_arr))])
    out = growth_rate(z_arr, cosmo, pk, variant=variant, k=k) * s8_z
    return out[0] if jnp.ndim(z) == 0 else out


def growth_scale_spread(z, cosmo, pk, variant: str = "total", k=None):
    r"""How badly a *scalar* :math:`D(z)` fails, measured.

    Returns a scalar (as a JAX array, so it stays traceable):
    :math:`\max_k|\sqrt{P(k,z)/P(k,0)}/D(z) - 1|` over
    :data:`SPREAD_K_RANGE`: the spread of the true, scale-dependent growth about
    the single number :func:`growth_factor` reports.

    Measured with CLASS, in that window: **2e-5** at zero neutrino mass -- i.e.
    numerically zero, which is the right answer, since without mass the growth
    really is scale-independent -- rising to **1.4e-3 at 0.06 eV** and more at
    larger masses, concentrated between :math:`k \sim 10^{-3}` and
    :math:`10^{-1}` h/Mpc, which is the free-streaming scale.  So the number
    this returns is the neutrino signal, not noise.

    This exists so the ambiguity is a *reported quantity* rather than a caveat
    in a docstring.  A caller who needs better than this must use
    :math:`P(k,z)` directly, which is what the halo sector does.

    **One redshift.**  The return value is a single ``max`` over :math:`k`, so
    there is no shape a stack of redshifts could come back in; asking for one
    used to divide a ``(Nz, Nk)`` ratio by a ``(Nz,)`` growth factor, which
    raises unless ``Nz == Nk`` and, at ``Nz == Nk``, divides along the
    wavenumber axis and returns a plausible wrong number.  Loop, or
    :func:`jax.vmap`.
    """
    if jnp.asarray(z).ndim != 0:
        raise ValueError(
            f"growth_scale_spread reports one number -- the max over k at one "
            f"redshift -- so it takes one redshift; got shape "
            f"{jnp.asarray(z).shape}.  Loop over the redshifts, or `jax.vmap` this "
            f"function over them.")
    if k is None:
        lo, hi = SPREAD_K_RANGE
        k = np.logspace(np.log10(lo), np.log10(hi), 512)
    else:
        k = np.asarray(k)
    getter = pk.pk if variant == "total" else pk.pk_cb
    ratio = jnp.sqrt(jnp.asarray(getter(k, z, cosmo))
                     / jnp.asarray(getter(k, 0.0, cosmo)))
    d = growth_factor(z, cosmo, pk, variant=variant, k=k)
    return jnp.max(jnp.abs(ratio / d - 1.0))
