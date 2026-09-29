r"""Beams and PSFs, as a multiplicative :math:`B_\ell`.

.. math::  C_\ell^{ab,\rm obs} = B_\ell^a\,B_\ell^b\,C_\ell^{ab}

**Once per leg.**  So an auto-spectrum carries :math:`B_\ell^2` and a
cross-spectrum carries one factor from each field.  The predecessor states that
rule and then breaks it: its :math:`C_\ell^{yy}` squares the beam and its
:math:`C_\ell^{XX}` -- also an auto-spectrum, in the same file -- applies it
once.  Here the rule is in :func:`~ggah_mod.observables.limber.c_ell`, which
takes one beam per kernel and cannot apply it any other way.

The King PSF, and the parameter it frees
-----------------------------------------

An eROSITA PSF is better described by a King profile,
:math:`(1 + (\theta/\theta_c)^2)^{-\alpha}`, than by a Gaussian.  Its transform
is closed-form,

.. math::

    B_\ell = \frac{2^{2-\alpha}}{\Gamma(\alpha-1)}
             (\ell\theta_c)^{\alpha-1}K_{\alpha-1}(\ell\theta_c)

and needs :math:`K_\nu` at a **traced** argument, because :math:`\theta_c` is
fitted.  The predecessor reached for ``scipy.special.kv``, which has no JAX
path, and hard-coded the one exponent where the answer is elementary
(:math:`\alpha = 3/2`, where :math:`K_{1/2}` gives a plain
:math:`e^{-\ell\theta_c}`).  With
:func:`~ggah_mod.observables.transforms.bessel_k` the general case is
differentiable in :math:`\theta_c` *and* :math:`\alpha`, so the King slope stops
being a static choice.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln

from .transforms import bessel_k

__all__ = ["gaussian", "king", "EROSITA_FWHM_ARCSEC", "EROSITA_KING",
           "ARCSEC", "ARCMIN"]

#: One arcsecond and one arcminute, in radians.
ARCSEC = jnp.pi / (180.0 * 3600.0)
ARCMIN = jnp.pi / (180.0 * 60.0)

#: eROSITA's soft-band on-axis PSF, as a Gaussian FWHM [arcsec].
EROSITA_FWHM_ARCSEC = 30.0
#: ...and as a King profile, ``(theta_c [arcsec], alpha)``: the TM CalDB
#: on-axis fit.  A survey-averaged 30" effective PSF is nearer
#: ``theta_c = 19.6``.
EROSITA_KING = (8.64, 1.5)


def gaussian(ell, fwhm_arcsec: float):
    r""":math:`B_\ell = e^{-\ell^2\sigma^2/2}`, :math:`\sigma = \mathrm{FWHM}/2.355`."""
    sigma = jnp.asarray(fwhm_arcsec) * ARCSEC / 2.3548200450309493
    return jnp.exp(-0.5 * jnp.asarray(ell) ** 2 * sigma ** 2)


def king(ell, theta_c_arcsec: float = EROSITA_KING[0],
         alpha: float = EROSITA_KING[1]):
    r""":math:`B_\ell` for a King profile.  Differentiable in both parameters.

    At :math:`\alpha = 3/2` this is exactly :math:`e^{-\ell\theta_c}`, which is
    the closed-form check -- and is the only case the predecessor could compute.

    ``alpha`` must exceed 1: below it the King profile has infinite total flux
    and :math:`\Gamma(\alpha-1)` has a pole, so a "beam" there does not
    normalise.  Not clipped -- a clip would give a plausible number for an
    unnormalisable profile.
    """
    x = jnp.asarray(ell) * jnp.asarray(theta_c_arcsec) * ARCSEC
    nu = jnp.asarray(alpha) - 1.0
    log_amp = (2.0 - jnp.asarray(alpha)) * jnp.log(2.0) - gammaln(nu)
    # `x -> 0` is `B_ell -> 1`, but `x^(nu) K_nu(x)` gets there as `0 * inf`.
    # The floor is on the *argument*, far below any ell that matters, so the
    # gradient at every real ell is untouched -- the `lin_weights` rule: clamp
    # the query, never the result.
    x_safe = jnp.maximum(x, 1e-12)
    out = jnp.exp(log_amp + nu * jnp.log(x_safe)) * bessel_k(nu, x_safe)
    return jnp.where(x > 1e-12, out, 1.0)
