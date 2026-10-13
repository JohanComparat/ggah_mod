r"""Halo mass functions: :math:`dn/dM` from :math:`\sigma(M)`.

Every fit here has the same shape,

.. math::

    \frac{dn}{dM} = f(\sigma)\,\frac{\bar\rho_{cb}}{M}
                    \left|\frac{d\ln\sigma^{-1}}{dM}\right|,

so the models differ only in the multiplicity function :math:`f(\sigma)`, and
a new fit is one function.  The prefactor is the *cold* density
:math:`\bar\rho_{cb}`, for the reason the variance uses the cold spectrum: it is
the matter that collapses.

:data:`MULTIPLICITY` holds eighteen fits, sixteen published and two
recalibrations of ``tinker08`` (:func:`fsigma_tinker08_csst` and
:func:`fsigma_tinker08_csst_vir`).  Each takes ``sigma``, :math:`\sigma(M, z)`
at the halo masses, and ``z``, and returns the dimensionless :math:`f`.  All are
pure JAX and differentiable in :math:`\sigma` and :math:`z`, so any of them can
sit inside the ``DIFFERENTIABLE`` flavour.  :class:`CsstHMF`, reached through
:func:`make_hmf`, is the emulator they are compared with, and is not
differentiable.

Conventions of the fits
-----------------------

The fits differ less in accuracy than in convention:

* *Mass definition.*  A fit calibrated on friends-of-friends haloes does not
  describe a spherical-overdensity catalogue, and 200m, 200c and virial are
  three different masses.  Each model records the definition it was calibrated
  for in :data:`CALIBRATION`; using one outside it is a systematic, not a
  tolerance.
* *Redshift range.*  ``yung25`` is fitted above :math:`z=6` and is not a
  low-redshift model; ``comparat17`` is a :math:`z=0` fit with no evolution.
* *The bias partner.*  A multiplicity function and a bias fit are consistent
  only if they were calibrated together
  (:func:`~ggah_mod.halos.linear_bias.matched_bias_for`).  Mixing them breaks
  the peak-background split by tens of per cent, which
  :func:`~ggah_mod.halos.linear_bias.mass_weighted_bias` shows.
"""

from __future__ import annotations

import functools
import math
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .variance import DELTA_C

__all__ = [
    "fsigma_press74", "fsigma_sheth99", "fsigma_jenkins01", "fsigma_warren06",
    "fsigma_tinker08", "fsigma_crocce10",
    "fsigma_bhattacharya11", "fsigma_watson13", "fsigma_angulo12",
    "fsigma_bocquet16", "fsigma_despali16", "fsigma_rodriguezpuebla16",
    "fsigma_comparat17", "fsigma_seppi20", "fsigma_yung24", "fsigma_yung25",
    "dndm", "multiplicity_kwargs", "MULTIPLICITY", "make_multiplicity",
    "CALIBRATION",
    "COSMOLOGY_DEPENDENT_MULTIPLICITY", "CurvatureResponse",
    "FitCapability",
    "MULTIPLICITY_CAPABILITY", "VIRIAL_REFERENCED_MULTIPLICITY",
    "fsigma_tinker08_csst",
    "fsigma_tinker08_csst_vir",
    "FittingFunctionHMF", "CsstHMF", "make_hmf", "HMF_BACKENDS",
]


# =========================================================================
# Multiplicity functions
# =========================================================================

@jax.jit
def fsigma_press74(sigma, z=0.0):
    r"""Press & Schechter (1974).

    .. math::  f(\sigma) = \sqrt{2/\pi}\,\nu\,e^{-\nu^2/2},\quad \nu=\delta_c/\sigma

    The analytic case.  It satisfies :math:`\int f\,d\ln\sigma^{-1} = 1` exactly,
    so it validates the algebra around it as well as the fit
    (``tests/test_halos.py::TestPressSchechterIsExact``).
    """
    nu = DELTA_C / jnp.asarray(sigma)
    return jnp.sqrt(2.0 / jnp.pi) * nu * jnp.exp(-0.5 * nu ** 2)


@jax.jit
def fsigma_sheth99(sigma, z=0.0):
    r"""Sheth & Tormen (1999), with the ellipsoidal-collapse correction.

    :math:`A=0.3222`, :math:`a=0.707`, :math:`p=0.3`.  Has a matching bias.
    """
    A, a, p = 0.3222, 0.707, 0.3
    nu = DELTA_C / jnp.asarray(sigma)
    nu2a = a * nu ** 2
    return A * jnp.sqrt(2.0 * a / jnp.pi) * nu * jnp.exp(-0.5 * nu2a) * (1.0 + nu2a ** -p)


@jax.jit
def fsigma_jenkins01(sigma, z=0.0):
    r"""Jenkins et al. (2001), friends-of-friends :math:`b=0.2`.

    .. math::  f(\sigma) = 0.315\,e^{-|\ln\sigma^{-1} + 0.61|^{3.8}}
    """
    return 0.315 * jnp.exp(-jnp.abs(jnp.log(1.0 / jnp.asarray(sigma)) + 0.61) ** 3.8)


@jax.jit
def fsigma_warren06(sigma, z=0.0):
    r"""Warren et al. (2006), friends-of-friends."""
    A, a, b, c = 0.7234, 1.625, 0.2538, 1.1982
    sigma = jnp.asarray(sigma)
    return A * (sigma ** -a + b) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_angulo12(sigma, z=0.0):
    r"""Angulo et al. (2012), from the Millennium-XXL friends-of-friends catalogue."""
    sigma = jnp.asarray(sigma)
    return 0.201 * ((2.08 / sigma) ** 1.7 + 1.0) * jnp.exp(-1.172 / sigma ** 2)


@jax.jit
def fsigma_crocce10(sigma, z=0.0):
    r"""Crocce et al. (2010), friends-of-friends, with power-law z-evolution."""
    sigma = jnp.asarray(sigma)
    zp1 = 1.0 + jnp.asarray(z)
    A = 0.58 * zp1 ** -0.13
    a = 1.37 * zp1 ** -0.15
    b = 0.30 * zp1 ** -0.084
    c = 1.036 * zp1 ** -0.024
    return A * (sigma ** -a + b) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_watson13(sigma, z=0.0):
    r"""Watson et al. (2013), friends-of-friends :math:`b=0.2`."""
    A, a, b, c = 0.282, 2.163, 1.406, 1.210
    sigma = jnp.asarray(sigma)
    return A * ((b / sigma) ** a + 1.0) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_bhattacharya11(sigma, z=0.0):
    r"""Bhattacharya et al. (2011), friends-of-friends, z-dependent.

    Has a matching bias, so the pair satisfies the peak-background split.
    """
    zp1 = 1.0 + jnp.asarray(z)
    A = 0.333 * zp1 ** -0.11
    a = 0.788 * zp1 ** -0.01
    p, q = 0.807, 1.795
    nu = DELTA_C / jnp.asarray(sigma)
    nu2a = a * nu ** 2
    return (A * jnp.sqrt(2.0 / jnp.pi) * jnp.exp(-0.5 * nu2a)
            * (1.0 + nu2a ** -p) * (nu * jnp.sqrt(a)) ** q)


# Tinker et al. (2008) Table 2: parameters against overdensity w.r.t. the mean.
# numpy, so importing this module does not initialise a JAX backend.
_T08_DELTA = np.array([200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
_T08_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_T08_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
_T08_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
_T08_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])


@jax.jit
def fsigma_tinker08(sigma, z=0.0, delta=200.0):
    r"""Tinker et al. (2008), spherical overdensity, any :math:`\Delta`.

    .. math::

        f(\sigma) = A(z)\left[(\sigma/b(z))^{-a(z)} + 1\right]e^{-c/\sigma^2}

    The :math:`z=0` parameters are interpolated in log from their Table 2 across
    :math:`\Delta \in [200, 3200]` (``delta``, w.r.t. mean), and evolve as
    :math:`A\propto(1+z)^{-0.14}`, :math:`a\propto(1+z)^{-0.06}`,
    :math:`b\propto(1+z)^{-\alpha}` with
    :math:`\alpha = 10^{-(0.75/\log_{10}(\Delta/75))^{1.2}}`; :math:`c` does not
    evolve.

    The parameters move by a factor of two across the table, so they are
    interpolated rather than pinned at :math:`\Delta=200`: a 200m fit used at
    :math:`\Delta=800` is not a small error.  Outside the tabulated range
    :func:`jnp.interp` clamps, and :data:`CALIBRATION` records the range.
    """
    sigma = jnp.asarray(sigma)
    log_d = jnp.log(jnp.asarray(delta))
    log_tab = jnp.log(_T08_DELTA)
    A0 = jnp.exp(jnp.interp(log_d, log_tab, jnp.log(_T08_A0)))
    a0 = jnp.exp(jnp.interp(log_d, log_tab, jnp.log(_T08_a0)))
    b0 = jnp.exp(jnp.interp(log_d, log_tab, jnp.log(_T08_b0)))
    c0 = jnp.exp(jnp.interp(log_d, log_tab, jnp.log(_T08_c0)))

    alpha = 10.0 ** (-((0.75 / jnp.log10(jnp.asarray(delta) / 75.0)) ** 1.2))
    zp1 = 1.0 + jnp.asarray(z)
    A = A0 * zp1 ** -0.14
    a = a0 * zp1 ** -0.06
    b = b0 * zp1 ** -alpha
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c0 / sigma ** 2)


@partial(jax.jit, static_argnames=("hydro",))
def fsigma_bocquet16(sigma, z=0.0, hydro: bool = False):
    r"""Bocquet et al. (2016), :math:`\Delta = 200{\rm m}`, z-dependent.

    ``hydro`` selects the fit calibrated on hydrodynamic rather than
    dark-matter-only simulations.  It is a static argument: it selects a set of
    published coefficients, not a value to differentiate.
    """
    if hydro:
        A0, a0, b0, c0 = 0.228, 2.15, 1.69, 1.30
        Az, az, bz, cz = 0.285, -0.058, -0.366, -0.045
    else:
        A0, a0, b0, c0 = 0.175, 1.53, 2.55, 1.19
        Az, az, bz, cz = -0.012, -0.040, -0.194, -0.021
    sigma = jnp.asarray(sigma)
    zp1 = 1.0 + jnp.asarray(z)
    A, a = A0 * zp1 ** Az, a0 * zp1 ** az
    b, c = b0 * zp1 ** bz, c0 * zp1 ** cz
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_despali16(sigma, z=0.0, delta_ratio=1.0):
    r"""Despali et al. (2016), spherical overdensity, any :math:`\Delta`.

    A Sheth-Tormen form whose coefficients are polynomials in
    :math:`x = \log_{10}(\Delta/\Delta_{\rm vir})`, so one fit covers every SO
    definition.  Pass ``delta_ratio = 1`` for virial.

    Despali et al. define :math:`\nu` with the redshift-dependent
    :math:`\delta_c(z)` of Kitayama & Suto (1996); it is evaluated here at the
    constant :data:`~ggah_mod.halos.variance.DELTA_C`.
    """
    x = jnp.log10(jnp.asarray(delta_ratio))
    A = -0.1362 * x + 0.3292
    a = 0.4332 * x ** 2 + 0.2263 * x + 0.7665
    p = -0.1151 * x ** 2 + 0.2554 * x + 0.2488
    nu_p = a * DELTA_C ** 2 / jnp.asarray(sigma) ** 2
    return 2.0 * A * jnp.sqrt(nu_p / (2.0 * jnp.pi)) * jnp.exp(-0.5 * nu_p) * (1.0 + nu_p ** -p)


@jax.jit
def fsigma_rodriguezpuebla16(sigma, z=0.0):
    r"""Rodriguez-Puebla et al. (2016), virial mass, calibrated :math:`0\le z\le7`."""
    z = jnp.asarray(z)
    sigma = jnp.asarray(sigma)
    A = 0.144 - 0.011 * z + 0.003 * z ** 2
    a = 1.351 + 0.068 * z + 0.006 * z ** 2
    b = 3.113 - 0.077 * z - 0.013 * z ** 2
    c = 1.187 + 0.009 * z
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_comparat17(sigma, z=0.0):
    r"""Comparat et al. (2017), virial mass, MultiDark-Planck.

    A :math:`z=0` fit with no redshift evolution: ``z`` is accepted for a
    uniform interface and ignored.  Using it at high redshift is an
    extrapolation the fit does not support.
    """
    A, a, p, q = 0.324, 0.897, 0.624, 1.589
    nu = DELTA_C / jnp.asarray(sigma)
    nu2a = a * nu ** 2
    return (A * jnp.sqrt(2.0 / jnp.pi) * jnp.exp(-0.5 * nu2a)
            * (1.0 + nu2a ** -p) * (nu * jnp.sqrt(a)) ** q)


@jax.jit
def fsigma_seppi20(sigma, z=0.0):
    r"""Seppi et al. (2020), virial mass, marginalised over offset and spin.

    The published model is a joint distribution over
    :math:`(\sigma, x_{\rm off}, \lambda)`; this is its 1-D marginal, obtained
    by integrating over both on a 50x50 log grid.  Calibrated for
    :math:`M > 4\times10^{13}\,M_\odot/h`.

    Costlier than the others by the two extra quadratures, but still traced and
    differentiable.

    ``z`` must be a scalar, and a vector raises.  The two internal quadrature
    axes are 50 points each, and a vector ``z`` would broadcast onto the last of
    them: it would raise for any length but 50, and at ``Nz == 50`` alias the
    redshift axis onto the spin axis and return a smooth, positive, wrong
    marginal.  The check is on the static shape, so it holds under ``jit`` too;
    use :func:`jax.vmap` over redshifts.
    """
    if jnp.asarray(z).ndim != 0:
        raise ValueError(
            f"fsigma_seppi20 takes one redshift; got shape {jnp.asarray(z).shape}.  "
            f"Its xoff and spin quadratures are 50 points each and a redshift "
            f"axis would alias onto them.  Use `jax.vmap` over the redshifts.")
    sigma = jnp.atleast_1d(jnp.asarray(sigma))
    n_xoff = n_spin = 50
    xoff = jnp.logspace(-3.5, -0.3, n_xoff)
    spin = jnp.logspace(-3.5, -0.3, n_spin)

    zp1 = 1.0 + jnp.asarray(z)
    A = -22.004 * zp1 ** -0.0441
    a = 0.886 * zp1 ** -0.1611
    q = 2.285 * zp1 ** 0.0409
    mu = -3.326 * zp1 ** -0.1286
    alpha = 5.623 * zp1 ** 0.1081
    beta = -0.391 * zp1 ** -0.3114
    gamma = 3.024 * zp1 ** 0.0902
    delta = 1.209 * zp1 ** -0.0768
    e = -1.105 * zp1 ** 0.6123

    ln10 = jnp.log(10.0)
    sig_ = sigma[:, None, None]
    xoff_ = xoff[None, :, None]
    spin_ = spin[None, None, :]

    nu_ = DELTA_C / sig_
    t1_ = xoff_ / 10.0 ** (1.83 * mu)
    h_log = (A
             + jnp.log10(jnp.sqrt(2.0 / jnp.pi))
             + q * jnp.log10(jnp.sqrt(a) * nu_)
             - a / 2.0 / ln10 * nu_ ** 2
             + alpha * jnp.log10(t1_)
             - 1.0 / ln10 * t1_ ** (0.05 * alpha)
             + gamma * jnp.log10(spin_ / 10.0 ** mu)
             - 1.0 / ln10 * (t1_ / sig_ ** e) ** beta * (spin_ / 10.0 ** mu) ** delta)
    h = 10.0 ** h_log
    g = jnp.trapezoid(h, jnp.log10(spin), axis=-1)
    return jnp.trapezoid(g, jnp.log10(xoff), axis=-1)


@jax.jit
def fsigma_yung24(sigma, z=0.0):
    r"""Yung et al. (2024), virial mass, GUREFT, calibrated :math:`0\le z\le20`."""
    z = jnp.asarray(z)
    sigma = jnp.asarray(sigma)
    A = 0.11416632 - 0.01486746 * z + 0.00137191 * z ** 2
    a = 1.05274399 + 0.02803087 * z - 0.00306126 * z ** 2
    b = 8.62813020 + 0.00384969 * z - 0.02349983 * z ** 2
    c = 1.13138924 + 0.01713172 * z - 0.00113630 * z ** 2
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c / sigma ** 2)


@jax.jit
def fsigma_yung25(sigma, z=0.0):
    r"""Yung et al. (2025), virial mass, GUREFT, calibrated :math:`6\le z\le30`.

    Not recommended below :math:`z=6`: unlike ``yung24`` this fit is
    optimised for the high-redshift regime and is not a low-redshift model.
    """
    z = jnp.asarray(z)
    sigma = jnp.asarray(sigma)
    A = 2.97165630e-01 - 2.76808434e-03 * z - 1.27528336e-04 * z ** 2
    a = 1.65590338 - 5.50399410e-02 * z - 1.63819807e-06 * z ** 2
    b = 1.69700438 - 0.08628012 * z + 0.01080824 * z ** 2
    c = 1.16098576 + 4.83463488e-03 * z - 3.76272478e-04 * z ** 2
    return A * ((sigma / b) ** -a + 1.0) * jnp.exp(-c / sigma ** 2)


@functools.lru_cache(maxsize=4)
def _hmf_correction(mdef: str = "200m", check_box: bool = True):
    """The shipped ``emu_hmf`` network, loaded once and cached.

    The weights are read from disk and the object is stateless once built; the
    cache is keyed on ``mdef`` and ``check_box``, so a deliberately unchecked
    instance does not evict the checked one.

    It is built under ``jax.ensure_compile_time_eval``, which materialises the
    weights as constants whatever is being traced around it.
    ``HmfCorrection.__init__`` calls ``jnp.asarray`` on the weight arrays, and
    the first call can come from inside a ``jit``, when a traced forward model
    first asks for the default mass function.  A value cached under a trace
    leaks into the next trace (``UnexpectedTracerError``) once a trace has been
    abandoned, which this package does whenever it refuses inside one.  The
    alternative, keeping the weights numpy in ``emu_hmf``, would need a release
    there.
    """
    try:
        from emu_hmf.model import WEIGHTS, HmfCorrection
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "the 'tinker08_csst' mass function is the recalibration carried "
            "in emu_hmf, which is a requirement of this package -- so reaching "
            "this means the install is incomplete: `pip install emu_hmf`.  The "
            "other sixteen multiplicity functions do not need it."
        ) from exc

    with jax.ensure_compile_time_eval():
        return HmfCorrection(WEIGHTS[mdef], check_box=check_box)


def fsigma_tinker08_csst(sigma, z=0.0, cosmo=None, check_box: bool = True):
    r"""``tinker08`` with its four shape parameters recalibrated per cosmology.

    The :math:`(A, a, b, c)` of :func:`fsigma_tinker08` multiplied by
    :math:`1 + g(\theta, z)`, with :math:`g` a small network trained in
    ``emu_hmf`` against the CSST simulation emulator over that emulator's
    eight-parameter box, at 200m.  At :math:`g = 0` this is ``tinker08``
    exactly, so the two are points in one parametrisation rather than two
    models.

    With :func:`fsigma_tinker08_csst_vir`, it is one of the two multiplicity
    functions here that take a cosmology
    (:data:`COSMOLOGY_DEPENDENT_MULTIPLICITY`); ``cosmo`` is required.  The
    other sixteen are fits to :math:`\sigma` and :math:`z`, calibrated once, at
    one place in the parameter space, so they cannot respond to a cosmology and
    take no cosmology argument, for the reason
    :data:`~ggah_mod.halos.concentration.PEAK_HEIGHT_MODELS` separates the
    concentration relations one rung down.

    It refuses outside the emulator's box rather than extrapolating, because a
    fitted correction outside its training region returns a smooth, plausible
    surface with nothing to mark it; ``check_box=False`` skips that check, and
    it steps aside under tracing, where the values are not available.

    Measured against the emulator on 200 cosmologies held out entirely:
    ``tinker08`` is 7.00 per cent rms in :math:`\ln f` and this is 0.52.
    """
    if cosmo is None:
        raise TypeError(
            "fsigma_tinker08_csst needs a cosmology: its recalibration is a "
            "function of the eight cosmological parameters, which is the whole "
            "difference between it and 'tinker08'.  Pass cosmo=... , or use "
            "make_field, which passes it for every model in "
            "COSMOLOGY_DEPENDENT_MULTIPLICITY.")
    from emu_hmf.target import theta_from_cosmology

    return _hmf_correction("200m", check_box).fsigma(
        sigma, theta_from_cosmology(cosmo), z)


def fsigma_tinker08_csst_vir(sigma, z=0.0, cosmo=None, check_box: bool = True):
    r"""The same recalibration, fitted against the emulator's *virial* masses.

    A separate entry rather than an ``mdef`` argument on
    :func:`fsigma_tinker08_csst`, because the two are different functions.
    Fitted with the same architecture on the same designs, the two corrections
    reduce the residual equally well (10.9 per cent to 0.54 at virial, 7.0 to
    0.52 at :math:`200{\rm m}`) and disagree with each other in
    :math:`f(\sigma)` by 4.4 per cent in the median and 10.8 rms, more than the
    residual either achieves and comparable to the offset both correct.  So the
    fractional correction does not transfer between overdensities, and the
    calibration guard refuses each outside the definition it was fitted in.

    There is no :math:`200{\rm c}` version: the emulator's only
    :math:`200{\rm c}` is a friends-of-friends mass, and fitting an SO
    ``tinker08`` against a FoF catalogue would fold a halo-finder change into a
    boundary change.  ``RockstarMvir`` is a spherical overdensity from the same
    finder as ``RockstarM200m``.
    """
    if cosmo is None:
        raise TypeError(
            "fsigma_tinker08_csst_vir needs a cosmology, for the same reason "
            "its 200m sibling does: the recalibration is a function of the "
            "eight cosmological parameters.  Pass cosmo=..., or use make_field.")
    from emu_hmf.target import theta_from_cosmology

    return _hmf_correction("vir", check_box).fsigma(
        sigma, theta_from_cosmology(cosmo), z)


MULTIPLICITY: dict[str, Callable] = {
    "press74": fsigma_press74,
    "sheth99": fsigma_sheth99,
    "jenkins01": fsigma_jenkins01,
    "warren06": fsigma_warren06,
    "tinker08": fsigma_tinker08,
    "crocce10": fsigma_crocce10,
    "bhattacharya11": fsigma_bhattacharya11,
    "watson13": fsigma_watson13,
    "angulo12": fsigma_angulo12,
    "bocquet16": fsigma_bocquet16,
    "despali16": fsigma_despali16,
    "rodriguezpuebla16": fsigma_rodriguezpuebla16,
    "comparat17": fsigma_comparat17,
    "seppi20": fsigma_seppi20,
    "yung24": fsigma_yung24,
    "yung25": fsigma_yung25,
    "tinker08_csst": fsigma_tinker08_csst,
    "tinker08_csst_vir": fsigma_tinker08_csst_vir,
}

#: What each fit was calibrated on: ``(halo definition, redshift range)``.
#:
#: Not decoration.  A friends-of-friends fit does not describe a spherical
#: -overdensity catalogue, and 200m / 200c / virial are three different masses;
#: applying a model outside its row is a systematic error, not a tolerance.
#: ``"SO-any"`` means the fit carries an explicit overdensity argument.
CALIBRATION: dict[str, tuple[str, tuple[float, float]]] = {
    "press74": ("analytic", (0.0, np.inf)),
    "sheth99": ("analytic/FoF", (0.0, np.inf)),
    "jenkins01": ("FoF b=0.2", (0.0, 5.0)),
    "warren06": ("FoF b=0.2", (0.0, 5.0)),
    "tinker08": ("SO-any (200-3200 mean)", (0.0, 2.5)),
    "crocce10": ("FoF b=0.2", (0.0, 1.0)),
    "bhattacharya11": ("FoF b=0.2", (0.0, 2.0)),
    "watson13": ("FoF b=0.2", (0.0, 30.0)),
    "angulo12": ("FoF b=0.2", (0.0, 2.0)),
    "bocquet16": ("SO 200m", (0.0, 2.0)),
    "despali16": ("SO-any (via delta_ratio)", (0.0, 1.25)),
    "rodriguezpuebla16": ("SO virial", (0.0, 7.0)),
    "comparat17": ("SO virial", (0.0, 0.0)),
    "seppi20": ("SO virial", (0.0, 1.5)),
    "yung24": ("SO virial", (0.0, 20.0)),
    "yung25": ("SO virial", (6.0, 30.0)),
    # Rockstar 200m in the Kun suite, which is the definition tinker08 was
    # itself calibrated in -- deliberately, so that reaching 200c goes through
    # the same Delta interpolation as everything else here rather than through
    # a change of halo finder.  z <= 3 is where the emulator was trained.
    "tinker08_csst": ("SO 200m (Rockstar, Kun suite)", (0.0, 3.0)),
    # The same recalibration at a different overdensity, and a *different*
    # function -- see `fsigma_tinker08_csst_vir`.  Same finder, so the guard
    # is separating boundaries and not catalogues.
    "tinker08_csst_vir": ("SO vir (Rockstar, Kun suite)", (0.0, 3.0)),
}

#: The multiplicity functions that take a **cosmology**, not only ``sigma``
#: and ``z``.
#:
#: The same distinction :data:`~ggah_mod.halos.concentration.PEAK_HEIGHT_MODELS`
#: draws one rung down, and for the same reason: a fit calibrated once cannot
#: respond to a cosmology, and giving it a uniform signature would let it
#: accept one and ignore it.  :func:`~ggah_mod.halos.field.make_field` reads
#: this set to decide what to pass.
COSMOLOGY_DEPENDENT_MULTIPLICITY = frozenset({"tinker08_csst",
                                              "tinker08_csst_vir"})


class CurvatureResponse(NamedTuple):
    r"""What a flat-box correction costs when it is used on a curved cosmology.

    A boolean says a fit has no curvature axis, not how much that matters, and
    here the two differ by a factor of 150: refusing at
    :math:`\Omega_k \neq 0` would send a curved run to the uncorrected fit,
    which at the *Planck* + BAO bound is thirteen times worse than the
    correction.  These three measured numbers replace the boolean.

    Attributes
    ----------
    coefficient : float
        Max :math:`|\mathrm{d}\ln f|` per unit :math:`|\Omega_k|`.  Linear in
        :math:`|\Omega_k|` and symmetric in its sign.

        It is the small-curvature coefficient, the mean over
        :math:`|\Omega_k| \le 0.05`, not over the whole range
        :attr:`measured_to` reaches.  The coefficient falls out to 0.30, so the
        mean over the full sweep would dilute it with a tail no realistic prior
        visits, giving 0.2112 instead of 0.2242 at 200m.  This package restates
        these values from ``emu_hmf``, so a redefinition upstream would reach
        it as a bare disagreement.
    val_rms : float
        The correction's own held-out residual in :math:`\ln f`, flat.
    baseline_rms : float
        The residual of the uncorrected fit it replaces, the fallback a refusal
        sends the caller to; with the other two it sets :attr:`crossover`.

    Notes
    -----
    What is measured is the carrier's response.  CSSTemu's ``get_Ez`` already
    carries ``Omegak*(1+z)**2`` and ``set_cosmos`` hard-sets it to zero;
    restoring it and re-evaluating ``get_dndlnM_Castro23`` at fixed sigma gives
    the response of the Castro+23 baseline through :math:`\Omega_m(z)` and
    :math:`\delta_c(\Omega_m(z))`.  The curvature response of the
    Gaussian-process residual is not measured, because the suite has no curved
    simulations.  That little is left there is an argument, not a measurement:
    curvature reaches a multiplicity function at fixed sigma through the growth
    history, and the target's own model is parametrised in exactly that.
    Quoting the coefficient without this caveat overstates it.

    The law is fitted over :math:`|\Omega_k| \le 0.05`, where it holds to 3.4
    per cent, and both crossovers lie beyond that.  The sweep reaches 0.45
    (:attr:`measured_to`), and both crossings are bracketed by measurement:
    virial between 0.10 and 0.15 at 0.120979, and 200m between 0.35 and 0.40 at
    0.371517.  The coefficient falls monotonically (by 16 per cent at 0.30, at
    200m), so the linear law overestimates the cost past the fitted range and
    its crossover lands early, by 4 per cent at virial and 19 at 200m, where the
    law gives 0.3009 against a measured crossing near 0.37.  The refusal
    therefore over-refuses a band, about 0.30 to 0.37 at 200m, where the
    correction is still marginally the better answer.  We keep that band: it is
    twice ``emu_pk``'s box edge, nobody samples a curvature there, and the
    margin inside it is a few per cent on a quantity that has already degraded
    sixfold.

    Over the sweep to 0.45 the coefficient spreads by 23 per cent at 200m and 15
    at virial; the quoted coefficient, the mean below 0.05, does not depend on
    how far the sweep reaches.  :attr:`crossover_is_measured` is per entry
    rather than a constant, because a newly measured fit has no measured
    crossing until its sweep reaches it; the refusal message branches on it.
    """

    coefficient: float
    val_rms: float
    baseline_rms: float
    #: Largest :math:`|\Omega_k|` the linear law has been *checked* at.  Not the
    #: range it was fitted over, which is 0.05 for both -- see the note below on
    #: why the two are different numbers and why only one of them is reassuring.
    measured_to: float = 0.05
    #: The crossing as *measured*, where the checked range brackets it, else
    #: ``None``.  A number and not a flag, for the same reason the coefficient
    #: replaced a boolean: it says how far :attr:`crossover` is off, rather than
    #: only that it might be.
    measured_crossing: "float | None" = None

    def induced(self, omega_k):
        r"""The error the flat box adds at this curvature, in :math:`\ln f`."""
        return self.coefficient * abs(float(omega_k))

    def total(self, omega_k):
        """Induced and intrinsic errors in quadrature, the total in ln f."""
        return math.hypot(self.val_rms, self.induced(omega_k))

    @property
    def crossover_is_measured(self):
        """Whether the checked range brackets the crossing.

        Derived from :attr:`measured_crossing`, so the two cannot disagree.
        """
        return self.measured_crossing is not None

    @property
    def crossover(self):
        r""":math:`|\Omega_k|` at which the correction stops being an improvement.

        Derived rather than stored, so it cannot drift from the three numbers
        it comes from: :math:`\sqrt{b^2 - v^2}/c`.
        """
        return math.sqrt(self.baseline_rms ** 2 - self.val_rms ** 2) / self.coefficient


class FitCapability(NamedTuple):
    """Which axes of the cosmology a fitted correction has.

    The two field names are
    :class:`~ggah_mod.cosmology.power.LinearPowerSpectrum`'s, because they make
    the same claim about a different fit: it was trained on a box with an axis
    for that parameter, so it can answer for it.
    """

    supports_curvature: bool
    supports_nondegenerate_nu: bool
    #: What the missing curvature axis costs, when someone has measured it.
    #: ``None`` means nobody has, and an unmeasured missing axis is refused at
    #: any curvature -- which is where both of these started.
    curvature_response: "CurvatureResponse | None" = None


#: What each cosmology-dependent multiplicity function's training box has an
#: axis for.  The other sixteen take ``sigma`` and ``z`` alone and are not here:
#: they make no claim about a cosmology, so there is none to violate.
#:
#: **Both entries are False twice, and neither is a choice made here.**
#: ``emu_hmf``'s box is CSSTemu's box, ``("Omegab", "Omegam", "H0", "ns", "A",
#: "w", "wa", "mnu")`` -- eight axes, no curvature, and a neutrino *sum* against
#: a suite run with three degenerate masses.  ``check_box`` cannot catch either
#: violation, because a parameter that is not an axis has no bound to be outside
#: of: the correction would return a smooth, plausible surface computed for the
#: flat cosmology carrying the same eight numbers.
#:
#: **The two fields are not both guards, and the asymmetry is measured.**
#: ``supports_curvature`` drives the refusal in
#: :func:`~ggah_mod.halos.calibration.check_cosmology_support`.
#: ``supports_nondegenerate_nu`` drives nothing and is a *record*: every entry
#: of ``theta_from_cosmology``'s vector is bit-for-bit invariant under how the
#: neutrino sum is divided, so the correction cannot move with the ordering and
#: there is nothing to refuse.  The ordering acts on the spectrum, and
#: :func:`~ggah_mod.cosmology.power._check_nu_split` is where it is refused.
#:
#: **This table will not flip itself, and that is now known rather than
#: assumed.**  ``tests/test_curvature_halos.py`` asserts it against
#: ``emu_hmf.box.PARAMS``, the way ``emu_hmf`` asserts its own copy against
#: CSSTemu's ``param_limits`` -- but that assertion is a guard against drift
#: rather than a promise of an upgrade.  ``emu_hmf``'s box *is* CSSTemu's
#: ``param_names``, name for name and order for order, and a ninth column would
#: break that equality correctly; there are also no curved simulations to train
#: one on.  ``emu_pk`` could widen its box because CLASS solves a curved model
#: on request, and this target cannot.  So the route out is a published
#: coefficient rather than an axis, and a coefficient needs a threshold here
#: rather than a boolean.  ``CROSS_REPO.md`` X15 carries that.
MULTIPLICITY_CAPABILITY: dict[str, FitCapability] = {
    # emu_hmf, 40-point design (box.sample(40, seed=11)), all twelve trained
    # redshifts, 1e12 < M < 1e14.  Crossovers derived: 0.3009 and 0.1162.
    "tinker08_csst": FitCapability(
        supports_curvature=False, supports_nondegenerate_nu=False,
        curvature_response=CurvatureResponse(coefficient=0.2242,
                                             val_rms=0.00518,
                                             baseline_rms=0.06766,
                                             measured_to=0.45,
                                             measured_crossing=0.371517)),
    # **Measured, and unreachable anyway.**  This fit is calibrated at vir
    # alone, and `check_cosmology_support` refuses vir at any curvature for an
    # unrelated reason -- `delta_vir` is the flat Bryan & Norman fit.  The
    # numbers are recorded because they were measured and because the two
    # refusals are independent: if the virial overdensity ever carries
    # curvature, this row is what decides the rest.
    "tinker08_csst_vir": FitCapability(
        supports_curvature=False, supports_nondegenerate_nu=False,
        curvature_response=CurvatureResponse(coefficient=0.8911,
                                             val_rms=0.00542,
                                             baseline_rms=0.10368,
                                             measured_to=0.45,
                                             measured_crossing=0.120979)),
}

assert all(c.curvature_response is None
           or c.curvature_response.crossover > 0.0
           for c in MULTIPLICITY_CAPABILITY.values()), (
    "a baseline no worse than the correction makes the crossover imaginary, "
    "which would mean the correction is not one")

assert all(c.curvature_response is None
           or not c.curvature_response.crossover_is_measured
           or c.curvature_response.crossover
           <= c.curvature_response.measured_crossing
           for c in MULTIPLICITY_CAPABILITY.values()), (
    "wherever the crossing has been measured, the linear law has to sit at or "
    "below it: the coefficient falls monotonically, so the law overestimates "
    "the cost and the threshold errs early.  A linear crossover *above* a "
    "measured one would mean that reasoning had inverted, and the threshold "
    "would then refuse late rather than early")

#: Multiplicity functions whose :math:`\Delta` argument is taken **against the
#: virial overdensity**, so they reach
#: :func:`~ggah_mod.halos.mass_definitions.delta_vir` at *every* mass
#: definition rather than only at ``mdef='vir'``.
#:
#: ``despali16`` is indexed by :math:`\Delta/\Delta_{\rm vir}`, and
#: :func:`~ggah_mod.halos.field._delta_kw` forms that ratio with
#: ``MassDef("vir").delta_mean(z, cosmo)`` in the denominator.  Nothing cancels:
#: at ``mdef='200m'`` the numerator is a plain 200, so the flat Bryan & Norman
#: coefficients survive whole into the argument.  Measured at the fiducial and
#: :math:`z = 1`, the ratio moves 0.990 to 0.967 between flat and
#: :math:`\Omega_k = 0.05`.
#:
#: Declared here rather than read off ``field._DELTA_ARG`` because it is a
#: property of the *fit* -- what its argument is measured against -- while that
#: table records which keyword carries it.  ``tests/test_curvature_halos.py``
#: asserts the two agree, so neither can drift.
VIRIAL_REFERENCED_MULTIPLICITY = frozenset({"despali16"})

assert set(MULTIPLICITY_CAPABILITY) == COSMOLOGY_DEPENDENT_MULTIPLICITY, (
    "every multiplicity function that takes a cosmology has to declare what it "
    "can see: an undeclared one is exactly the silent answer this table exists "
    "to prevent")


def make_multiplicity(name: str) -> Callable:
    """Look up a multiplicity function by name."""
    key = str(name).lower()
    if key not in MULTIPLICITY:
        raise ValueError(f"unknown mass function {name!r}; "
                         f"expected one of {sorted(MULTIPLICITY)}")
    return MULTIPLICITY[key]


def dndm(m, sigma, dln_sigma_dln_m, rho_cold, model="tinker08", z=0.0, **kw):
    r""":math:`dn/dM` [(Mpc/h)^-3 (Msun/h)^-1], from a multiplicity function.

    Parameters
    ----------
    m : array [Msun/h]
    sigma : array
        :math:`\sigma(M)` at the same masses and redshift.
    dln_sigma_dln_m : array
        From :func:`~ggah_mod.halos.variance.dln_sigma_dln_mass`.  Negative.
    rho_cold : float
        ``cosmo.rho_cold``.
    model : str
        Any key of :data:`MULTIPLICITY`.
    **kw
        Passed to the multiplicity function -- ``delta`` for ``tinker08``,
        ``delta_ratio`` for ``despali16``, ``hydro`` for ``bocquet16``.
    """
    f = make_multiplicity(model)(sigma, z, **kw)
    m = jnp.asarray(m)
    return f * (rho_cold / m ** 2) * jnp.abs(dln_sigma_dln_m)


#: The keyword through which a multiplicity function takes the mass definition.
#: The other fits are calibrated at one definition and take none.
_DELTA_ARG = {"tinker08": "delta", "despali16": "delta_ratio"}


def multiplicity_kwargs(model, mdef, z, cosmo, kw=None) -> dict:
    r"""The keyword arguments ``model``'s multiplicity function takes for ``mdef``.

    ``tinker08`` receives :math:`\Delta_{\rm m}` as ``delta`` and ``despali16``
    receives :math:`\Delta_{\rm m}/\Delta_{\rm vir}` as ``delta_ratio``; both are
    taken against the mean density.  The fits of
    :data:`COSMOLOGY_DEPENDENT_MULTIPLICITY` receive ``cosmo``.  Every other fit
    receives nothing.  An argument already in ``kw`` is kept.

    Parameters
    ----------
    model : str
        Any key of :data:`MULTIPLICITY`.
    mdef : str or MassDef
    z : float
    cosmo : Cosmology
    kw : dict, optional
        Arguments supplied by the caller.
    """
    from .mass_definitions import VIRIAL, parse_mass_def
    key, kw = str(model).lower(), dict(kw or {})
    if key in COSMOLOGY_DEPENDENT_MULTIPLICITY:
        kw.setdefault("cosmo", cosmo)
    name = _DELTA_ARG.get(key)
    if name is None or name in kw:
        return kw
    delta_m = parse_mass_def(mdef).delta_mean(z, cosmo)
    kw[name] = (delta_m if name == "delta"
                else delta_m / VIRIAL.delta_mean(z, cosmo))
    return kw


# =========================================================================
# Backends: a fitting function, or a simulation emulator
# =========================================================================
#
# The functions above are *multiplicity functions*: fits of f(sigma) to
# simulations, evaluated through the sigma(M) this package computes.  An
# emulator is a different kind of object -- it was trained on the mass function
# directly and produces dn/dlnM from the cosmology, never touching our
# sigma(M).  Both are legitimate and they are not interchangeable internally,
# so they share an interface rather than a mechanism.


class FittingFunctionHMF:
    r"""``dn/dM`` from a multiplicity function and this package's sigma(M).

    Differentiable when the P(k) backend is.  ``pk`` is the linear-P(k)
    backend, ``model`` a key of :data:`MULTIPLICITY`, ``delta`` the overdensity
    against the mean density (default 200), ``k`` the wavenumber grid
    [h/Mpc] for :math:`\sigma(M)` (default 512 points log-spaced over
    :math:`10^{-4}`-:math:`10^{2}`), and ``calibration`` the policy of
    :func:`~ggah_mod.halos.calibration.check_calibration` (default
    ``"strict"``, as in :func:`~ggah_mod.halos.field.make_field`).

    The mass definition reaches the multiplicity function through
    :func:`multiplicity_kwargs`, the dispatch ``make_field`` uses, and
    :meth:`dndm` runs the calibration checks ``make_field`` runs.
    """

    differentiable = True

    def __init__(self, pk, model: str = "tinker08", delta: float = 200.0,
                 k=None, calibration: str = "strict"):
        import numpy as _np
        from .calibration import POLICIES
        from .mass_definitions import MassDef
        if calibration not in POLICIES:
            raise ValueError(f"calibration must be one of {POLICIES}, "
                             f"got {calibration!r}")
        self._pk = pk
        self.model = str(model)
        self.delta = float(delta)
        self.mdef = MassDef(self.delta)
        self.calibration = calibration
        self.name = f"fit:{self.model}"
        self._k = _np.logspace(-4, 2, 512) if k is None else _np.asarray(k)

    def dndm(self, m, z, cosmo):
        from .calibration import check_calibration, check_cosmology_support
        from .variance import sigma_of_mass, dln_sigma_dln_mass
        check_cosmology_support(self.model, self.mdef, cosmo)
        check_calibration(self.model, None, self.mdef, z,
                          policy=self.calibration)
        p_cb = self._pk.pk_cb(self._k, z, cosmo)
        s = sigma_of_mass(m, self._k, p_cb, cosmo.rho_cold)
        d = dln_sigma_dln_mass(m, self._k, p_cb, cosmo.rho_cold)
        return dndm(m, s, d, cosmo.rho_cold, model=self.model, z=z,
                    **multiplicity_kwargs(self.model, self.mdef, z, cosmo))

    def sigma(self, m, z, cosmo):
        from .variance import sigma_of_mass
        return sigma_of_mass(m, self._k, self._pk.pk_cb(self._k, z, cosmo),
                             cosmo.rho_cold)


#: Which recalibration's carrier each CSSTemu mass definition shares, so the
#: measured curvature coefficient can be quoted for it.  ``FoFM200c`` is absent
#: because nothing measured it: a friends-of-friends mass is not one of the two
#: spherical-overdensity files ``emu_hmf`` fitted.
_CSST_LIKE = {"RockstarM200m": "tinker08_csst",
              "RockstarMvir": "tinker08_csst_vir"}


class CsstHMF:
    r"""``dn/dM`` from CSSTemu (Chen et al.).

    A Gaussian-process emulator trained on the CSST simulation suite, so it
    carries the simulations' calibration directly instead of a fit to them; the
    ``tinker08_csst`` recalibrations are trained against it.  Both flavours use
    ``tinker08_csst`` through :func:`~ggah_mod.halos.field.make_field`, and this
    class is reached through :func:`make_hmf` for comparison.

    Not differentiable: the emulator is a scikit-learn GP behind a numpy
    interface, so it cannot appear inside a traced assembly.  A parity-budget
    row records what the differentiable path costs in accuracy.  It refuses any
    nonzero :math:`\Omega_k`, since CSSTemu takes eight parameters and
    curvature is not one of them, and any redshift outside :data:`TRAINED_Z`.

    It does not use this package's :math:`\sigma(M)`: it was trained on
    :math:`dn/d\ln M` directly.  So pairing it with a bias fit calibrated
    against a *different* mass function breaks the peak-background split by
    more than either error alone; check with
    :func:`~ggah_mod.halos.linear_bias.mass_weighted_bias` before trusting a
    pairing.

    Parameters
    ----------
    massdef : {"RockstarM200m", "FoFM200c", "RockstarMvir"}
        Halo finder and overdensity the emulator was trained on.  These are
        different mass definitions, not conventions: a mass converted between
        them is not the mass the emulator was trained on.
    """

    differentiable = False
    name = "csst"

    #: Redshifts the emulator was trained at.  It interpolates between them.
    TRAINED_Z = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

    def __init__(self, massdef: str = "RockstarM200m"):
        # The shim **first**, and the order is the whole point.  Importing it
        # restores `scipy.integrate.simps`, which CEmulator imports at its own
        # module scope -- so it has to be in place before `import CEmulator`
        # runs, not after.  This used to sit below that import, and the result
        # was a class that worked only when something else in the process had
        # already imported the shim: `CsstHMF()` in a fresh interpreter raised
        # `cannot import name 'simps'`, which the handler below then reported
        # as "CSSTemu is not installed" about a package that was installed.
        from ._cemulator_compat import ensure_cemulator_works
        try:
            from CEmulator.Emulator import HMF_CEmulator
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "CSSTemu could not be imported.  If it is not installed: "
                "pip install git+https://github.com/czymh/csstemu.  If it is, "
                "this is an incompatibility the shim in "
                "`halos/_cemulator_compat` did not cover -- the original error "
                f"is attached: {e}"
            ) from e
        valid = {"RockstarM200m", "FoFM200c", "RockstarMvir"}
        if massdef not in valid:
            raise ValueError(f"massdef must be one of {sorted(valid)}, "
                             f"got {massdef!r}")
        self.massdef = massdef
        self._emu = HMF_CEmulator()
        self._set_cosmology(_probe_cosmology())
        self.patched = ensure_cemulator_works(self._emu)

    def _set_cosmology(self, cosmo):
        import numpy as _np
        # The same eight parameters `emu_hmf` carries, and the same two it does
        # not.  This backend is reached through `make_hmf` rather than through
        # `make_field`, so `check_cosmology_support` never sees it and the
        # refusal has to stand here.  Every value is concrete by construction --
        # `set_cosmos` takes floats -- so a plain `if` is safe where layer 2's
        # other guard needs a tracing escape.
        ok = float(getattr(cosmo, "Omega_k", 0.0))
        if ok != 0.0:
            # The size is known even though the verdict is not.  `emu_hmf`
            # measured the full emulated dn/dlnM against the Castro+23 carrier
            # alone and they agree to one part in 1e4 of the signal, so the
            # carrier's coefficient *is* this backend's whole curvature
            # response -- the Gaussian-process ratio has no curvature input to
            # respond with.  Quoting it costs nothing and lets a caller weigh
            # the refusal against whatever they would do instead.
            cap = MULTIPLICITY_CAPABILITY.get(_CSST_LIKE.get(self.massdef, ""))
            r = None if cap is None else cap.curvature_response
            size = ("" if r is None else
                    f"  The induced error here is about {r.induced(ok):.2g} in "
                    f"ln f, from the {r.coefficient:g} per unit |Omega_k| "
                    f"measured for this definition's carrier.")
            raise ValueError(
                f"CsstHMF cannot carry Omega_k = {ok:g}: `set_cosmos` takes "
                f"eight parameters and curvature is not one of them, so the "
                f"emulator would return the mass function of a flat universe "
                f"carrying the same eight numbers.  CSSTemu's suite is flat; "
                f"this is a property of the simulations rather than of the "
                f"interface, so there is no argument to pass instead.{size}  "
                f"Unlike the `tinker08_csst` recalibrations this is refused at "
                f"*any* curvature rather than past a crossover, because a "
                f"crossover is a comparison against a fallback that degrades "
                f"differently and this backend has no fallback.  Use a "
                f"fitting-function HMF -- `dndm(..., model='tinker08')` claims "
                f"no cosmology at all.")
        self._emu.set_cosmos(
            Omegab=float(cosmo.Omega_b),
            Omegac=float(cosmo.Omega_cdm),      # true CDM, neutrinos excluded
            H0=float(cosmo.h) * 100.0,
            As=float(_np.exp(cosmo.ln10A_s) * 1e-10),
            ns=float(cosmo.n_s),
            w=float(cosmo.w0),
            wa=float(cosmo.wa),
            mnu=float(cosmo.sum_mnu),
        )

    def dndm(self, m, z, cosmo):
        r""":math:`dn/dM` [(Mpc/h)^-3 (Msun/h)^-1] at one redshift ``z``.

        The emulator returns :math:`dn/d\ln M`; this divides by :math:`M`.
        """
        import numpy as _np
        z_val = float(_np.asarray(z))
        if not (min(self.TRAINED_Z) <= z_val <= max(self.TRAINED_Z)):
            raise ValueError(
                f"z = {z_val} is outside the CSST emulator's trained range "
                f"[{min(self.TRAINED_Z)}, {max(self.TRAINED_Z)}]; it would "
                f"extrapolate with no accuracy guarantee.")
        self._set_cosmology(cosmo)
        m_np = _np.asarray(m, dtype=float)
        dndlnm = self._emu.get_dndlnM(z=z_val, M=m_np, massdef=self.massdef)
        return jnp.asarray(_np.asarray(dndlnm).reshape(-1)) / jnp.asarray(m_np)


#: A concrete cosmology used only to probe the CSSTemu numpy-2 failure at
#: construction time, so the error surfaces there rather than deep in a fit.
_PROBE_COSMOLOGY = None      # filled in below, after the import cycle is safe


#: Mass-function backends, by name.
HMF_BACKENDS = {"csst": CsstHMF}


def _probe_cosmology():
    from ..cosmology.parameters import PLANCK18
    return PLANCK18


def make_hmf(name: str, pk=None, **kw):
    """Construct a mass-function backend.

    ``"csst"`` gives the emulator; any multiplicity-function name
    (``"tinker08"``, ...) gives :class:`FittingFunctionHMF` around ``pk``.
    """
    key = str(name).lower()
    if key in HMF_BACKENDS:
        return HMF_BACKENDS[key](**kw)
    if key in MULTIPLICITY:
        if pk is None:
            raise ValueError(
                f"the {key!r} mass function is a fit to sigma(M), so it needs a "
                f"linear-P(k) backend; pass pk=...")
        return FittingFunctionHMF(pk, model=key, **kw)
    raise ValueError(f"unknown mass function {name!r}; expected one of "
                     f"{sorted(set(MULTIPLICITY) | set(HMF_BACKENDS))}")
