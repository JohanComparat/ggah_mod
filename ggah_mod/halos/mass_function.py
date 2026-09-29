r"""Halo mass functions: :math:`dn/dM` from :math:`\sigma(M)`.

Every fit here has the same shape,

.. math::

    \frac{dn}{dM} = f(\sigma)\,\frac{\bar\rho_{cb}}{M}
                    \left|\frac{d\ln\sigma^{-1}}{dM}\right|,

so the models differ only in :math:`f(\sigma)`, the multiplicity function.  That
is the whole content of a mass-function fit, and keeping the surrounding algebra
in one place means a new fit is one function.

:math:`\bar\rho_{cb}` -- the **cold** density -- is the right prefactor for the
same reason the variance uses the cold spectrum: it is the matter that collapses.

Choosing a model
----------------

The seventeen fits below are not interchangeable, and the differences that
matter are not accuracy but *convention*:

* **Mass definition.**  A fit calibrated on friends-of-friends halos does not
  describe a spherical-overdensity catalogue, and 200m, 200c and virial are
  three different masses.  Each model records the definition it was calibrated
  for in :data:`CALIBRATION`; using one outside it is a systematic, not a
  tolerance.
* **Redshift range.**  ``yung25`` is fitted above :math:`z=6` and is wrong at
  low redshift; ``comparat17`` is a :math:`z=0` fit with no evolution at all.
* **The bias partner.**  A multiplicity function and a bias fit are consistent
  only if they were derived together -- see
  :func:`~ggah_mod.halos.linear_bias.matched_bias_for`.  Mixing them breaks the
  peak-background split by tens of percent, which
  :func:`~ggah_mod.halos.linear_bias.mass_weighted_bias` will show.

All are pure JAX and differentiable in :math:`\sigma` and :math:`z`, so any of
them can sit inside the ``DIFFERENTIABLE`` flavour.
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
    "dndm", "MULTIPLICITY", "make_multiplicity", "CALIBRATION",
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
    which is what makes it able to validate the algebra around it rather than
    only the fit -- see ``tests/test_halos.py::TestPressSchechterIsExact``.
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

    The :math:`z=0` parameters are interpolated **in log** from Table 2 across
    :math:`\Delta \in [200, 3200]` (w.r.t. mean), and evolve as
    :math:`A\propto(1+z)^{-0.14}`, :math:`a\propto(1+z)^{-0.06}`,
    :math:`b\propto(1+z)^{-\alpha}` with
    :math:`\alpha = 10^{-(0.75/\log_{10}(\Delta/75))^{1.2}}`; :math:`c` does not
    evolve.

    Interpolating rather than pinning :math:`\Delta=200` matters: the parameters
    move by a factor of two across the table, so a 200m fit used at
    :math:`\Delta=800` is not a small error.  Outside the tabulated range
    :func:`jnp.interp` clamps, which is why :data:`CALIBRATION` records the
    range.
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
    dark-matter-only simulations.  It is a *static* argument -- it selects a
    different set of published coefficients, not a value to differentiate.
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

    A :math:`z=0` fit with **no** redshift evolution: ``z`` is accepted for a
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

    **One redshift.**  The two internal quadrature axes are 50 points each, and
    a vector ``z`` broadcasts onto the *last* of them: it raises for any length
    but 50, and at ``Nz == 50`` it aliases the redshift axis onto the spin axis
    and returns a smooth, positive, wrong marginal.  The check is on the static
    shape, so it holds under ``jit`` too.
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

    **Not recommended below** :math:`z=6`: unlike ``yung24`` this fit is
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
    """The shipped ``emu_hmf`` network, loaded once.

    Cached because the weights are read from disk and the object is stateless
    once built; keyed on ``check_box`` so a deliberately unchecked instance
    does not evict the checked one.

    **Built outside whatever trace is running.**  ``HmfCorrection.__init__``
    does ``jnp.asarray`` over the weight arrays, and this function is reached
    from inside a ``jit`` the first time a traced forward model asks for the
    default mass function.  Cache a value made under a trace and the next trace
    leaks it -- ``UnexpectedTracerError``, pointing here, from a line that only
    reads weights off disk.

    It needs an *abandoned* trace to bite, which is why it was invisible: one,
    two or three ordinary traces in a fresh process are fine, measured.  This
    package abandons traces routinely, because it refuses rather than degrades
    and a refusal inside a trace is an exception inside a trace.  The symptom
    was `pytest tests/test_field.py -n auto` failing where the same file passes
    serially -- serially, some earlier test warms this cache from outside a
    trace first.

    ``ensure_compile_time_eval`` is the context that says "materialise these as
    constants, now, regardless of what is being traced around me", which is
    exactly the claim the cache makes about them.  The alternative fix is in
    ``emu_hmf``, keeping the weights numpy and letting each call site convert;
    that is a release there against three lines here.
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
    eight-parameter box.  At :math:`g = 0` this *is* ``tinker08``, exactly, so
    the two are points in one parameterisation rather than two models.

    **This is the only multiplicity function here that takes a cosmology, and
    the signature difference is the point.**  The other sixteen are fits to
    :math:`\sigma` and :math:`z`; they were calibrated once, at one place in
    the parameter space, and they cannot respond to a cosmology and should not
    be able to accept one.  It is the same argument
    :data:`~ggah_mod.halos.concentration.PEAK_HEIGHT_MODELS` makes one rung
    down, and it is why this name is in
    :data:`COSMOLOGY_DEPENDENT_MULTIPLICITY` rather than being made to look
    like the others.

    Refuses outside the emulator's box rather than extrapolating -- a fitted
    correction's failure mode outside its training region is a smooth,
    plausible surface with nothing to mark it -- and steps aside under tracing,
    where the values are not available to check.

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

    A separate entry rather than a ``mdef`` argument on
    :func:`fsigma_tinker08_csst`, because the two are **not the same function**
    and a shared name would invite the assumption that they are.  Fitted with
    the same architecture on the same designs, the two corrections reduce the
    residual equally well --- 10.9 per cent to 0.54 at virial, 7.0 to 0.52 at
    :math:`200{\rm m}` --- and disagree with each other in :math:`f(\sigma)`
    by 4.4 per cent in the median and 10.8 rms, which is larger than the
    residual either achieves and comparable to the offset both correct.

    So the fractional correction does not transfer between overdensities, and
    the calibration guard refuses each of these outside the definition it was
    fitted in.  That refusal is a measurement, not a caution.

    Virial is available and :math:`200{\rm c}` is not, for a reason worth
    keeping in view: the emulator's only :math:`200{\rm c}` is a
    friends-of-friends mass, and fitting an SO ``tinker08`` against a FoF
    catalogue would fold a halo-finder change into what looks like a boundary
    change.  ``RockstarMvir`` is a true spherical overdensity from the same
    finder as ``RockstarM200m``, which is what makes it a clean second option.
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

    A boolean says a fit has no curvature axis.  It cannot say *how much that
    matters*, and the difference turned out to be a factor of 150: refusing at
    :math:`\Omega_k \neq 0` sent a curved run to the uncorrected fit, which at
    the *Planck* + BAO bound is thirteen times worse than the correction it was
    avoiding.  These three measured numbers are what replaces the boolean.

    Attributes
    ----------
    coefficient : float
        Max :math:`|\mathrm{d}\ln f|` per unit :math:`|\Omega_k|`.  Linear in
        :math:`|\Omega_k|` and symmetric in its sign.

        **It is the small-curvature coefficient, meaned over**
        :math:`|\Omega_k| \le 0.05`, and not over the whole range
        :attr:`measured_to` reaches.  The distinction is not pedantry: the
        coefficient *falls* out to 0.30, so meaning over the full sweep dilutes
        it with a tail that no realistic prior visits, giving 0.2112 instead of
        0.2242 at 200m -- a number wrong where it is used and right only on
        average.  ``emu_hmf``'s regeneration script drifted onto exactly that
        definition when its sweep was extended to answer the extrapolation
        question, and its constants were right while its script was wrong.
        Recorded here because this package *restates* these values, so a
        redefinition upstream would reach it as a bare disagreement.
    val_rms : float
        The correction's own held-out residual in :math:`\ln f`, flat.
    baseline_rms : float
        The residual of the uncorrected fit it replaces, which is what a
        refusal would send the caller to.  It is what makes
        :attr:`crossover` a number rather than a taste.

    Notes
    -----
    **What is measured and what is argued.**  This paragraph travels with the
    coefficient; quoting one without the other overstates it.

        What is measured is the carrier's response.  CSSTemu's ``get_Ez``
        already carries ``Omegak*(1+z)**2`` and ``set_cosmos`` hard-sets it to
        zero; restoring it and re-evaluating ``get_dndlnM_Castro23`` at fixed
        sigma gives the response of the Castro+23 baseline through
        :math:`\Omega_m(z)` and :math:`\delta_c(\Omega_m(z))`.  What is NOT
        measured is the curvature response of the Gaussian-process residual,
        because no curved simulations exist in this suite to measure it
        against.  The claim that little is left is an argument, not a
        measurement: it rests on the channel by which curvature reaches a
        multiplicity function at fixed sigma being the growth history, and on
        the target's own model being parameterised in exactly that.

    **The crossover is an extrapolation, and the extrapolation is
    conservative.**  The law is *fitted* over :math:`|\Omega_k| \le 0.05`,
    where it holds to 3.4 per cent, and both crossovers sit beyond that.  At
    200m it has since been *checked* out to 0.30 (:attr:`measured_to`), and the
    coefficient is not constant: it falls monotonically, by 16 per cent at 0.30.

    Every departure is in the same direction, and it is the safe one.  A falling
    coefficient means the linear law **overestimates** the cost everywhere past
    the fitted range, so a crossover computed from it lands *early*.  The
    measured response does not actually reach the 200m baseline until roughly
    0.37, against the 0.3009 the law gives.  So the refusal above the crossover
    over-refuses a band from about 0.30 to 0.37 where the correction is still
    marginally the better answer.

    That band is kept deliberately.  It is twice ``emu_pk``'s box edge, nobody
    samples a curvature there, and the margin inside it is a few per cent on a
    quantity that has already degraded sixfold.  Refusing slightly early on a
    conservative extrapolation is the right failure direction.

    **Neither threshold rests on an extrapolation any more.**  The sweep was
    pushed to 0.45 and both crossings are now bracketed by measurement: virial
    between 0.10 and 0.15 at 0.120979, and 200m between 0.35 and 0.40 at
    0.371517.  The linear law is early on both, by 4 per cent at virial and 19
    at 200m -- more where it reaches further, which is what a falling
    coefficient has to do and is now observed rather than argued.  So the
    over-refused bands are real and measured rather than inferred, and the
    conservative direction is a fact about both files.

    :attr:`crossover_is_measured` remains per-file rather than a constant,
    because it was ``False`` for 200m for most of a day and would be ``False``
    again for any newly measured fit before its sweep reaches its crossing.
    The refusal message still branches on it.

    **The coefficients did not move when the sweep grew, and that is the
    definition above earning its keep.**  Extending to 0.45 changes the tail
    and the spread over it -- now 23 per cent at 200m and 15 at virial -- and
    leaves the quoted number alone, because the quoted number is the mean below
    0.05.  A coefficient defined over the whole sweep would have moved twice in
    two days without anything being remeasured.
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
        """Induced and intrinsic in quadrature -- the correction's real error."""
        return math.hypot(self.val_rms, self.induced(omega_k))

    @property
    def crossover_is_measured(self):
        """Whether the checked range brackets the crossing, rather than
        stopping short of it.  Derived, so it cannot become a third place for
        the same fact to disagree with itself."""
        return self.measured_crossing is not None

    @property
    def crossover(self):
        r""":math:`|\Omega_k|` at which the correction stops being an improvement.

        Derived rather than stored, so it cannot drift from the three numbers
        it comes from: :math:`\sqrt{b^2 - v^2}/c`.
        """
        return math.sqrt(self.baseline_rms ** 2 - self.val_rms ** 2) / self.coefficient


class FitCapability(NamedTuple):
    """Which axes of the cosmology a fitted correction actually has.

    The two field names are
    :class:`~ggah_mod.cosmology.power.LinearPowerSpectrum`'s, deliberately.  A
    reader who has met ``supports_curvature`` on a spectrum backend meets the
    same word here, because it is the same claim about a different fit: *this
    was trained on a box with an axis for that parameter, so it can answer for
    it.*
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
    r""":math:`dn/dM` [(Mpc/h)^-3 (Msun/h)^-1].

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

    Differentiable when the P(k) backend is, which makes it the ``DIFFERENTIABLE``
    flavour's mass function.
    """

    differentiable = True

    def __init__(self, pk, model: str = "tinker08", delta: float = 200.0,
                 k=None):
        import numpy as _np
        self._pk = pk
        self.model = str(model)
        self.delta = float(delta)
        self.name = f"fit:{self.model}"
        self._k = _np.logspace(-4, 2, 512) if k is None else _np.asarray(k)

    def dndm(self, m, z, cosmo):
        from .variance import sigma_of_mass, dln_sigma_dln_mass
        p_cb = self._pk.pk_cb(self._k, z, cosmo)
        s = sigma_of_mass(m, self._k, p_cb, cosmo.rho_cold)
        d = dln_sigma_dln_mass(m, self._k, p_cb, cosmo.rho_cold)
        return dndm(m, s, d, cosmo.rho_cold, model=self.model, z=z,
                    delta=self.delta)

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

    The most accurate mass function available here: a Gaussian-process emulator
    trained on the CSST simulation suite, so it carries the simulations'
    calibration directly instead of a fit to them.  It is the right choice for
    the ``ACCURATE`` flavour.

    **Not differentiable.**  The emulator is a scikit-learn GP behind a numpy
    interface, so it cannot appear inside a traced assembly; ``DIFFERENTIABLE`` uses
    :class:`FittingFunctionHMF` instead.  How much accuracy that costs is a
    parity-budget row, not a guess.

    It also does not use this package's :math:`\sigma(M)` at all -- it was
    trained on :math:`dn/d\ln M` directly.  So pairing it with a bias fit
    calibrated against a *different* mass function breaks the peak-background
    split by more than either error alone; check with
    :func:`~ggah_mod.halos.linear_bias.mass_weighted_bias` before trusting a pairing.

    Parameters
    ----------
    massdef : {"RockstarM200m", "FoFM200c", "RockstarMvir"}
        Halo finder and overdensity the emulator was trained on.  These are
        genuinely different mass definitions, not conventions: a mass converted
        between them is not the mass the emulator was trained on.
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
        r""":math:`dn/dM` [(Mpc/h)^-3 (Msun/h)^-1].

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
