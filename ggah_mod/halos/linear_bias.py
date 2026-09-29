r"""Linear halo bias :math:`b(M, z)`.

The peak-background split ties bias to the mass function: if a model's
:math:`f(\sigma)` and its :math:`b(\nu)` are consistent, then

.. math::

    \int b(M)\,\frac{M}{\bar\rho_{cb}}\,\frac{dn}{dM}\,dM = 1,

because all the mass is in halos and the mean field is unbiased with respect to
itself.  That integral is a *check*, not an input, and
:func:`mass_weighted_bias` computes it so the check can be run.  A pairing that
misses it by tens of percent -- which several literature combinations do -- is
using a bias fit calibrated against a different mass function.

**Linear** bias only: every model here returns a single number per halo mass,
independent of scale.  The scale- and mass-pair-dependent correction to the
assumption :math:`P_{hh} = b_1 b_2 P_{\rm lin}` lives in
:mod:`ggah_mod.halos.beyond_linear_bias`, which is a different kind of object --
a tabulated 2-halo *correction* rather than a fit for :math:`b`.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .variance import DELTA_C

__all__ = ["bias_tinker10", "bias_sheth99", "bias_press74",
           "bias_bhattacharya11", "bias_sheth01", "bias_comparat17",
           "BIAS", "make_bias", "mass_weighted_bias", "mass_fraction",
           "matched_bias_for", "MATCHED_BIAS",
           "Unresolved", "unresolved"]

# Every model below is ``jax.jit``-compiled and differentiable in ``sigma``:
# they sit inside the halo integrals that the differentiable flavour differentiates, so a
# numpy one would break the gradient at the last step.  ``delta`` is traced too
# rather than static, so a forecast may vary the overdensity.


@jax.jit
def bias_tinker10(sigma, delta: float = 200.0):
    r"""Tinker et al. (2010), Eq. 6.  Calibrated with their own mass function."""
    y = jnp.log10(jnp.asarray(delta))
    exp_term = jnp.exp(-((4.0 / y) ** 4))
    A = 1.0 + 0.24 * y * exp_term
    a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * exp_term
    c = 2.4
    nu = DELTA_C / jnp.asarray(sigma)
    return (1.0 - A * nu ** a / (nu ** a + DELTA_C ** a)
            + B * nu ** b + C * nu ** c)


@jax.jit
def bias_sheth99(sigma, delta: float = 200.0):
    r"""Sheth & Tormen (1999) bias, the peak-background-split partner of
    :func:`~ggah_mod.halos.mass_function.fsigma_sheth99`."""
    a, p = 0.707, 0.3
    nu = DELTA_C / jnp.asarray(sigma)
    an2 = a * nu ** 2
    return 1.0 + (an2 - 1.0) / DELTA_C + 2.0 * p / (DELTA_C * (1.0 + an2 ** p))


@jax.jit
def bias_press74(sigma, delta: float = 200.0):
    r"""Press-Schechter bias, :math:`b = 1 + (\nu^2-1)/\delta_c`.

    The analytic case: exactly consistent with
    :func:`~ggah_mod.halos.mass_function.fsigma_press74` under the
    peak-background split, so the pair is what validates
    :func:`mass_weighted_bias`.
    """
    nu = DELTA_C / jnp.asarray(sigma)
    return 1.0 + (nu ** 2 - 1.0) / DELTA_C


@jax.jit
def bias_sheth01(sigma, delta: float = 200.0):
    r"""Sheth, Mo & Tormen (2001) Eq. 8 -- an empirical recalibration of
    :func:`bias_sheth99` against simulations.

    Not the peak-background-split partner of any mass function here: it is a
    direct fit to measured bias, which is why it is not in :data:`MATCHED_BIAS`.
    """
    a, b, c = 0.707, 0.5, 0.6
    sa = jnp.sqrt(a)
    nu = DELTA_C / jnp.asarray(sigma)
    an2 = a * nu ** 2
    return 1.0 + (sa * an2 + sa * b * an2 ** (1.0 - c)
                  - an2 ** c / (an2 ** c + b * (1.0 - c) * (1.0 - c / 2.0))
                  ) / (sa * DELTA_C)


@jax.jit
def bias_bhattacharya11(sigma, delta: float = 200.0):
    r"""Bhattacharya et al. (2011) bias, the partner of their mass function.

    Derived by the peak-background split from the same fit, so the pair is
    self-consistent in a way that mixing fits is not.
    """
    a, p, q = 0.788, 0.807, 1.795
    nu = DELTA_C / jnp.asarray(sigma)
    an2 = a * nu ** 2
    return (1.0 + (an2 - q) / DELTA_C
            + 2.0 * p / (DELTA_C * (1.0 + an2 ** p)))


@jax.jit
def bias_comparat17(sigma, delta: float = 200.0):
    r"""Comparat et al. (2017) bias, by peak-background split from their own
    multiplicity function.

    :func:`~ggah_mod.halos.mass_function.fsigma_comparat17` is a
    Bhattacharya-type form,

    .. math::

        f(\sigma) \propto \left[1 + (a\nu^2)^{-p}\right](\nu\sqrt a)^q
                          e^{-a\nu^2/2},

    and for that family the peak-background split gives

    .. math::

        b(\nu) = 1 + \frac{a\nu^2 - q}{\delta_c}
                 + \frac{2p/\delta_c}{1 + (a\nu^2)^p}

    which is the same expression :func:`bias_bhattacharya11` uses, evaluated at
    the Comparat+2017 coefficients :math:`a = 0.897`, :math:`p = 0.624`,
    :math:`q = 1.589`.

    **Derived here, not published.**  Comparat et al. (2017) fit the mass
    function and do not quote a companion bias, so this is the split of *their*
    fit rather than a measurement of bias in their simulation.  That makes it
    self-consistent with ``comparat17`` by construction -- which is the point,
    and why it is in :data:`MATCHED_BIAS` -- but it is not independently
    calibrated against measured clustering, and it inherits the z = 0 -only
    validity of the mass function it comes from.
    """
    a, p, q = 0.897, 0.624, 1.589
    nu = DELTA_C / jnp.asarray(sigma)
    an2 = a * nu ** 2
    return (1.0 + (an2 - q) / DELTA_C
            + 2.0 * p / (DELTA_C * (1.0 + an2 ** p)))


BIAS: dict[str, Callable] = {
    "tinker10": bias_tinker10,
    "sheth99": bias_sheth99,
    "press74": bias_press74,
    "bhattacharya11": bias_bhattacharya11,
    "sheth01": bias_sheth01,
    "comparat17": bias_comparat17,
}

#: The peak-background-split partner of each multiplicity function, where one
#: was derived alongside it.
#:
#: ``None`` -- or absence -- means no partner exists in the literature, so any
#: bias choice is an *ad hoc* pairing.  That is allowed, and often unavoidable,
#: but it should be a decision rather than an accident:
#: :func:`mass_weighted_bias` measures what the pairing costs.
MATCHED_BIAS: dict[str, str] = {
    "press74": "press74",
    "sheth99": "sheth99",
    "tinker08": "tinker10",
    "bhattacharya11": "bhattacharya11",
    "despali16": "sheth99",       # same Sheth-Tormen form, refitted
    "comparat17": "comparat17",   # split of their own fit; see the docstring
}


def matched_bias_for(mass_function: str) -> str | None:
    """The bias fit derived alongside ``mass_function``, or ``None``.

    ``None`` is an answer, not a failure: most mass-function fits were
    published without a matching bias, and pairing one with Tinker10 is the
    usual choice -- just not a consistent one.  Check it with
    :func:`mass_weighted_bias` rather than assuming.
    """
    return MATCHED_BIAS.get(str(mass_function).lower())


def make_bias(name: str) -> Callable:
    key = str(name).lower()
    if key not in BIAS:
        raise ValueError(f"unknown bias model {name!r}; "
                         f"expected one of {sorted(BIAS)}")
    return BIAS[key]


def mass_weighted_bias(m, dndm_arr, bias_arr, rho_cold):
    r""":math:`\int b\,(M/\bar\rho_{cb})\,(dn/dM)\,dM` -- should be 1.

    Integrated in :math:`\ln M`.  How close it gets is a property of the
    (mass function, bias) *pairing* and of the mass range integrated over; both
    matter, and both are the caller's choice, so this reports rather than
    asserts.
    """
    m = jnp.asarray(m)
    integrand = bias_arr * (m / rho_cold) * dndm_arr * m      # * m for dlnM
    return jnp.trapezoid(integrand, jnp.log(m))


def mass_fraction(m, dndm_arr, rho_cold):
    r""":math:`\int (M/\bar\rho_{cb})(dn/dM)\,dM` -- the fraction of mass in halos.

    Unity only if every particle is in a halo *and* the integration covers the
    whole mass range; over any finite range it is less, and how much less is
    worth knowing before reading anything into a bias normalisation.
    """
    m = jnp.asarray(m)
    return jnp.trapezoid((m / rho_cold) * dndm_arr * m, jnp.log(m))


class Unresolved(NamedTuple):
    r"""What the mass grid does not hold, by mass and by bias-weighted mass.

    One object with two weights.  ``mass`` is :math:`1 - F`, the fraction of the
    cold matter density below the grid, which is what the baryon census books
    as outside the resolved haloes.  ``bias_weighted`` is :math:`1 - B`, the
    bias-weighted fraction, which is what the two-halo counterterm restores.
    They differ because the missing matter is not unbiased: it lies where
    :math:`b < 1`, and :attr:`bias` is its effective linear bias.  Read either
    alone and the other looks like a contradiction; the pair is AMH23's smooth
    component (Asgari, Mead & Heymans 2023, Sec. 5.1; Smith & Markovic 2011) --
    a linearly biased field with no one-halo term -- written in the two numbers
    that fix it.
    """

    mass: jnp.ndarray
    bias_weighted: jnp.ndarray

    @property
    def bias(self):
        r""":math:`b_u = (1 - B)/(1 - F)`, the bias of the unresolved matter."""
        return self.bias_weighted / self.mass


def unresolved(m, dndm_arr, bias_arr, rho_cold) -> Unresolved:
    r""":class:`Unresolved` for one (mass function, bias, mass range) triple."""
    return Unresolved(
        mass=1.0 - mass_fraction(m, dndm_arr, rho_cold),
        bias_weighted=1.0 - mass_weighted_bias(m, dndm_arr, bias_arr, rho_cold))
