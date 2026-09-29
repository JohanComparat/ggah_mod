r"""Concentration--mass relations :math:`c(M, z)`.

A concentration turns a mass into a *shape*: with a mass definition fixing
:math:`r_\Delta`, and :math:`c`, an NFW halo is fully specified.  Nothing in the
profile layer can be evaluated without one.

Two families, and the difference matters more than the scatter:

**Empirical power laws** -- ``duffy08``, ``dutton14``, ``klypin16``.  Fitted to
one simulation suite at one cosmology, with the cosmology baked into the
coefficients.  They take no :class:`~ggah_mod.cosmology.parameters.Cosmology`
and **cannot respond to one**: a free-cosmology chain gets a frozen c(M).  That
is a property of the fits, not an oversight, and it is why they take no cosmology
argument here rather than accepting and ignoring one.

**Physical relations** -- ``bhattacharya13``, ``diemer19``.  Parameterised by
peak height :math:`\nu = \delta_c/\sigma(M,z)` and, for ``diemer19``, by the
effective slope :math:`n_{\rm eff}` of :math:`\sigma(M)` and the growth rate
:math:`\alpha_{\rm eff} = \dd\ln D/\dd\ln a`.  These carry cosmology through
:math:`\sigma` and transfer between cosmologies as well as the parameterisation
does.

**A distribution, not only a relation** -- ``seppi21``.  The other five answer
"what is :math:`c` at this :math:`(M, z)`".  Seppi et al. (2021), A&A 652,
A155 (arXiv:2008.03179) measured :math:`P(c\,|\,\sigma, z)` as well, on ROCKSTAR
haloes in MultiDark, and this module carries both.  Their Eq. 9 is a mean
relation and enters :data:`CONCENTRATION` like any other.  Their Eq. 10 is used
**only for its dimensionless shape**; the scale is anchored to whatever
concentration the caller already has, through :func:`seppi21_scale`.

That is a measurement rather than a preference.  The PDF was fitted on six mass
slices spanning only :math:`2\times10^{13}` to :math:`2\times10^{14}\,M_\odot/h`
at each of four snapshots, so its :math:`\sigma`-dependence is weakly
constrained: at the paper's own scale, the mean of Eq. 10 reproduces its own
Eq. 9 to 2 per cent near :math:`\sigma = 1`, to 8 per cent across
:math:`0.6 < \sigma < 1.0`, and to 24 per cent by :math:`\sigma = 1.6`.  Anchoring keeps the well-measured thing and discards the
poorly-measured one.  It is also what makes the model portable --- across
relations, since a width in :math:`\ln c` cannot see a rescaling of
:math:`c`, and across mass definitions, since
:func:`seppi21_definition_jacobian` puts
:math:`\dd\ln c_{\rm out}/\dd\ln c_{\rm in}` at 0.965-1.058 for the boundaries
this package uses.

Nothing downstream is changed by any of it.  :func:`seppi21_nodes` hands back
quadrature nodes and weights, so a caller who wants
:math:`\langle u(k|M)\rangle` rather than :math:`u(k|M, \bar c)` can take it,
and one who does not is unaffected.

``diemer19`` replaced Diemer & Kravtsov (2015) here, and the reason is
structural rather than a preference for the newer fit.  The 2015 relation is
parameterised by the slope of :math:`P(k)`, which oscillates with the baryon
acoustic feature and therefore has to be evaluated on a *smooth analytic fit*
rather than on the spectrum in use -- an approximation in the middle of an
otherwise exact chain, and the last thing in this package that needed an
Eisenstein & Hu transfer function.  The 2019 relation's slope is of
:math:`\sigma(M)`, which is an integral over :math:`P(k)` and smooth whatever
the spectrum does, so it is driven by the real backend and the analytic fit is
gone.

Mass definitions
----------------

Each fit was calibrated for a specific :math:`\Delta` and returns *that*
concentration.  ``diemer19`` always returns :math:`c_{200c}`.  Converting
between definitions is :func:`~ggah_mod.halos.mass_definitions.translate_mass`,
which changes :math:`c` as well as :math:`M` -- they are not independent.

All relations are pure JAX and differentiable.
"""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaincc, gammaln, polygamma

from ..numerics import invert_monotone, lin_weights
import numpy as np

from ..cosmology import constants as C
from .mass_definitions import translate_mass
from .variance import DELTA_C

__all__ = [
    "c_duffy08", "c_dutton14", "c_klypin16", "c_bhattacharya13", "c_diemer19",
    "c_seppi21",
    "n_eff_from_sigma", "DIEMER19", "CONCENTRATION", "make_concentration",
    "CM_CALIBRATION",
    # Seppi et al. (2021): the distribution of c around the relation
    "SEPPI21", "SEPPI21_A1", "SEPPI21_A1_COLUMNS", "ANCHORS",
    "seppi21_shape", "seppi21_table_scale", "seppi21_scale",
    "seppi21_logpdf", "seppi21_pdf", "seppi21_quantile",
    "sigma_ln_c_seppi21", "seppi21_nodes",
    "seppi21_shape_translated", "seppi21_definition_jacobian",
]


# =========================================================================
# Empirical power laws -- no cosmology dependence, by construction
# =========================================================================

@partial(jax.jit, static_argnames=("mdef",))
def c_duffy08(m_h, z=0.0, mdef: str = "200m"):
    r"""Duffy et al. (2008): :math:`c = A(M/2\times10^{12})^B(1+z)^C`.

    WMAP5, full sample.  Valid :math:`10^{11} < M < 10^{15}\,M_\odot/h`,
    :math:`0 < z < 2`.
    """
    coeffs = {"200c": (5.71, -0.084, -0.47),
              "vir": (7.85, -0.081, -0.71),
              "200m": (10.14, -0.081, -1.01)}
    if mdef not in coeffs:
        raise ValueError(f"duffy08 is calibrated for {sorted(coeffs)}, got {mdef!r}")
    A, B, Cz = coeffs[mdef]
    return A * (jnp.asarray(m_h) / 2.0e12) ** B * (1.0 + jnp.asarray(z)) ** Cz


@partial(jax.jit, static_argnames=("mdef",))
def c_dutton14(m_h, z=0.0, mdef: str = "200c"):
    r"""Dutton & Maccio (2014): :math:`\log_{10}c = a(z) + b(z)\log_{10}(M/10^{12})`.

    Planck13.  Valid :math:`M > 10^{10}\,M_\odot/h`, :math:`0 < z < 5`.
    """
    z = jnp.asarray(z)
    if mdef == "200c":
        a = 0.520 + (0.905 - 0.520) * jnp.exp(-0.617 * z ** 1.21)
        b = -0.101 + 0.026 * z
    elif mdef == "vir":
        a = 0.537 + (1.025 - 0.537) * jnp.exp(-0.718 * z ** 1.08)
        b = -0.097 + 0.024 * z
    else:
        raise ValueError(f"dutton14 is calibrated for '200c' or 'vir', got {mdef!r}")
    return jnp.power(10.0, a + b * jnp.log10(jnp.asarray(m_h) / 1.0e12))


# Klypin et al. (2016) Table 2, Planck13/MultiDark: (z, C0, gamma, M0/1e12).
_KLYPIN16_200C = np.array([
    [0.00, 7.40, 0.120, 5.5e5], [0.35, 6.25, 0.117, 1e5],
    [0.50, 5.65, 0.115, 2e4], [1.00, 4.30, 0.110, 900.0],
    [1.44, 3.53, 0.095, 300.0], [2.15, 2.70, 0.085, 42.0],
    [2.50, 2.42, 0.080, 17.0], [2.90, 2.20, 0.080, 8.5],
    [4.10, 1.92, 0.080, 2.0], [5.40, 1.65, 0.080, 0.3],
])
_KLYPIN16_VIR = np.array([
    [0.00, 9.75, 0.110, 5e5], [0.35, 7.25, 0.107, 2.2e4],
    [0.50, 6.50, 0.105, 1e4], [1.00, 4.75, 0.100, 1000.0],
    [1.44, 3.80, 0.095, 210.0], [2.15, 3.00, 0.085, 43.0],
    [2.50, 2.65, 0.080, 18.0], [2.90, 2.42, 0.080, 9.0],
    [4.10, 2.10, 0.080, 1.9], [5.40, 1.86, 0.080, 0.42],
])


@partial(jax.jit, static_argnames=("mdef",))
def c_klypin16(m_h, z=0.0, mdef: str = "200c"):
    r"""Klypin et al. (2016): :math:`c = C_0(M/10^{12})^{-\gamma}[1+(M/M_0)^{0.4}]`.

    Planck13/MultiDark.  The upturn at high mass is the physical content -- the
    other power-law fits are monotonic and miss it.  Parameters are linearly
    interpolated in redshift between the tabulated bins, so ``c`` is
    differentiable in ``z`` as well as ``m_h``.
    """
    if mdef == "200c":
        tab = _KLYPIN16_200C
    elif mdef == "vir":
        tab = _KLYPIN16_VIR
    else:
        raise ValueError(f"klypin16 is calibrated for '200c' or 'vir', got {mdef!r}")
    z_tab = jnp.asarray(tab[:, 0])
    c0 = jnp.interp(jnp.asarray(z), z_tab, jnp.asarray(tab[:, 1]))
    gamma = jnp.interp(jnp.asarray(z), z_tab, jnp.asarray(tab[:, 2]))
    m0 = jnp.interp(jnp.asarray(z), z_tab, jnp.asarray(tab[:, 3])) * 1.0e12
    m_h = jnp.asarray(m_h)
    return c0 * (m_h / 1.0e12) ** (-gamma) * (1.0 + (m_h / m0) ** 0.4)


# =========================================================================
# Peak-height relations -- cosmology enters through sigma
# =========================================================================

@partial(jax.jit, static_argnames=("mdef",))
def c_bhattacharya13(sigma, growth, mdef: str = "200c"):
    r"""Bhattacharya et al. (2013): :math:`c = K D^\alpha \nu^\beta`.

    WMAP7.  Valid :math:`2\times10^{12} < M < 2\times10^{15}\,M_\odot/h`,
    :math:`0 < z < 2`.

    Parameters
    ----------
    sigma : array
        :math:`\sigma(M, z)` from
        :func:`~ggah_mod.halos.variance.sigma_of_mass`.
    growth : float
        :math:`D(z)/D(0)` from
        :func:`~ggah_mod.cosmology.growth.growth_factor`.

    Notes
    -----
    ``growth`` is passed in rather than computed here, so the caller decides
    which spectrum it is read off and the relation cannot silently acquire its
    own growth model.  The predecessor hard-wired Carroll+1992 here, so a
    w0waCDM fit got LambdaCDM concentrations with nothing in the output saying
    so.  Note ``sigma`` already carries the redshift, so ``D`` enters only
    through the explicit :math:`D^\alpha` factor of the fit.
    """
    coeffs = {"200c": (5.9, 0.54, -0.35),
              "vir": (7.7, 0.90, -0.29),
              "200m": (9.0, 1.15, -0.29)}
    if mdef not in coeffs:
        raise ValueError(f"bhattacharya13 is calibrated for {sorted(coeffs)}, "
                         f"got {mdef!r}")
    K, alpha, beta = coeffs[mdef]
    nu = DELTA_C / jnp.asarray(sigma)
    return K * jnp.asarray(growth) ** alpha * nu ** beta


#: Diemer & Joyce (2019) fitted parameters.  Six, against Diemer & Kravtsov
#: (2015)'s seven: the low-mass behaviour is derived rather than fitted, which
#: is what removes the seventh.
DIEMER19 = {
    "median": dict(kappa=0.41, a0=2.45, a1=1.82, b0=3.20, b1=2.30, c_alpha=0.21),
    "mean":   dict(kappa=0.42, a0=2.37, a1=1.74, b0=3.39, b1=1.82, c_alpha=0.20),
}

#: Bracket for the G(c) inversion, in log10 c.
#:
#: G(c) is **not** monotonic over all c: it has a minimum whose position depends
#: on ``n_eff``, and below that minimum the inversion is ill-posed.  The minimum
#: sits at c < 1 for every ``n_eff`` this model is defined over -- at n = -1.4,
#: the shallowest slope reached at cluster masses, it is near c = 0.2 -- so a
#: bracket starting at c = 1 is on the ascending branch by construction rather
#: than by luck.  Haloes with c < 1 are not haloes.
_LOG10C_BRACKET = (0.0, 3.0)


def _nfw_mu(c):
    r""":math:`\mu(c) = \ln(1+c) - c/(1+c)`, the NFW mass within :math:`r_s`."""
    return jnp.log1p(c) - c / (1.0 + c)


@partial(jax.jit, static_argnames=("statistic",))
def c_diemer19(sigma, n_eff, alpha_eff, statistic: str = "median"):
    r"""Diemer & Joyce (2019), :math:`c_{200c}` from peak height and two slopes.

    .. math::

        \frac{c}{\mu(c)^{(5+n_{\rm eff})/6}}
          = \frac{A(n_{\rm eff})}{\nu}\left(1 + \frac{\nu^2}{B(n_{\rm eff})}\right),
        \qquad \mu(c) = \ln(1+c) - \frac{c}{1+c},

    with :math:`A = a_0[1 + a_1(n+3)]`, :math:`B = b_0[1 + b_1(n+3)]`, and the
    result scaled by :math:`C = 1 - c_\alpha(1 - \alpha_{\rm eff})`.

    Three things make this the right relation for a differentiable package, and
    the first is the one that matters most here.

    **Its slope is of** :math:`\sigma(M)`, **not of** :math:`P(k)`.  Diemer &
    Kravtsov (2015) needs :math:`\dd\ln P/\dd\ln k`, whose value oscillates with
    the baryon acoustic feature unless it is taken from a *smooth* spectrum ---
    which is why both this package and COLOSSUS fed it an analytic
    zero-baryon fit rather than the spectrum actually in use.  This model needs
    :math:`n_{\rm eff} = -2\,\dd\ln\sigma/\dd\ln R - 3`, and
    :math:`\sigma(M)` is smooth by construction, being an integral over
    :math:`P(k)`.  So the relation can be driven by the *real* spectrum, and the
    analytic fitting function is gone from the package entirely --- see
    :func:`n_eff_from_sigma`.

    **It responds to the expansion history.**  :math:`\alpha_{\rm eff} =
    \dd\ln D/\dd\ln a` is the growth rate, so a cosmology that changes the
    growth changes the concentration, which the power-law fits in :math:`(M,z)`
    cannot do at all.

    **The inversion keeps its gradient.**  :math:`G(c)` has no closed-form
    inverse.  COLOSSUS tabulates it and interpolates; here it is bracketed under
    ``stop_gradient`` and polished with one Newton step
    (:func:`~ggah_mod.numerics.invert_monotone`), so the value is exact to
    machine precision *and* :math:`\partial c/\partial\theta` is the
    implicit-function derivative rather than zero.

    Parameters
    ----------
    sigma : array
        :math:`\sigma(M)` at the halo mass, from this package's variance.
    n_eff : array
        :math:`-2\,\dd\ln\sigma/\dd\ln R - 3` at the :math:`\kappa`-scaled
        Lagrangian radius.  See :func:`n_eff_from_sigma`.
    alpha_eff : array or float
        :math:`\dd\ln D/\dd\ln a` at the redshift wanted, from
        :func:`~ggah_mod.cosmology.growth.growth_rate`.
    statistic : ``"median"`` or ``"mean"``
    """
    if statistic not in DIEMER19:
        raise ValueError(
            f"statistic must be one of {sorted(DIEMER19)}, got {statistic!r}")
    q = DIEMER19[statistic]

    n = jnp.asarray(n_eff)
    nu = DELTA_C / jnp.asarray(sigma)
    a_n = q["a0"] * (1.0 + q["a1"] * (n + 3.0))
    b_n = q["b0"] * (1.0 + q["b1"] * (n + 3.0))
    c_alpha = 1.0 - q["c_alpha"] * (1.0 - jnp.asarray(alpha_eff))
    rhs = jnp.log10(a_n / nu * (1.0 + nu ** 2 / b_n))

    # Solve log10 G(c) = rhs for x = log10 c.  Written as a function of x so the
    # bracket is fixed and data-independent, which is what lets the bisection
    # jit with a static trip count.
    w = (5.0 + n) / 6.0

    def log10_g(x):
        c = 10.0 ** x
        return x - w * jnp.log10(_nfw_mu(c))

    x = invert_monotone(log10_g, rhs, *_LOG10C_BRACKET)
    return 10.0 ** x * c_alpha


#: Radius at which the effective slope is measured, as a fraction of the
#: Lagrangian radius :math:`R(M)`.
#:
#: **kappa is part of the calibration, not a free choice.**  Diemer & Joyce
#: (2019) fitted it jointly with the other five parameters, and it differs
#: between the median and mean statistics (0.41 against 0.42).  It is read from
#: :data:`DIEMER19` rather than written here, so the two cannot drift apart.


def n_eff_from_sigma(m_h, cosmo, sigma_of_m, statistic: str = "median"):
    r"""Effective slope :math:`n_{\rm eff} = -2\,\dd\ln\sigma/\dd\ln R - 3`.

    The Diemer & Joyce (2019) definition, and the reason this package no longer
    carries an analytic transfer function.

    Diemer & Kravtsov (2015) defined the slope on :math:`P(k)`, which forced a
    choice nobody wanted: :math:`\dd\ln P/\dd\ln k` of a spectrum carrying
    baryon acoustic oscillations *oscillates with them*, coming out
    non-monotonic in mass, so both this package and COLOSSUS evaluated it on a
    smooth analytic fit instead of on the spectrum actually being used.  That
    is an approximation sitting in the middle of an otherwise exact chain.

    The 2019 definition is a slope of :math:`\sigma(M)`, which is an integral
    over :math:`P(k)` against a top-hat window and is therefore smooth whatever
    the spectrum does.  It can be taken from the real backend.  Since
    :math:`R \propto M^{1/3}`,

    .. math::  \frac{\dd\ln\sigma}{\dd\ln R} = 3\,\frac{\dd\ln\sigma}{\dd\ln M},

    and the slope is wanted at :math:`\kappa R(M)`, which is the Lagrangian
    radius of :math:`\kappa^3 M`.

    Parameters
    ----------
    m_h : array [Msun/h]
    cosmo : :class:`~ggah_mod.cosmology.parameters.Cosmology`
    sigma_of_m : callable
        ``m -> dln sigma / dln M`` at that mass, from
        :func:`~ggah_mod.halos.variance.dln_sigma_dln_mass` with the spectrum
        and grid already bound.  Passed in rather than rebuilt here so the
        caller's :math:`P(k)` and :math:`k` grid are used, not a second set.
    statistic : ``"median"`` or ``"mean"``
        Selects :math:`\kappa`, which is part of the fit.
    """
    kappa = DIEMER19[statistic]["kappa"]
    m_scaled = jnp.asarray(m_h) * kappa ** 3
    return -6.0 * jnp.asarray(sigma_of_m(m_scaled)) - 3.0


# =========================================================================
# Seppi et al. (2021) -- a distribution, not only a relation
# =========================================================================

#: Seppi et al. (2021) Eq. 9: the two *fitted* parameters of the mean relation.
#:
#: The other constants in that equation -- 7.37, 3/4, 0.14, -2, 1/5 and 1/2 --
#: are fixed by the parameterisation, which is a generalisation of Klypin et
#: al. (2016)'s, and are written into :func:`c_seppi21` rather than here so a
#: reader cannot mistake them for something the fit determined.  The quoted
#: uncertainties are 4e-6 and 2e-6, which is why no bound is carried with them.
SEPPI21 = {"a0": 0.754091, "b0": 0.574413}

#: Seppi et al. (2021) Table A.1: the concentration *distribution*, Eq. 10,
#: fitted independently at four HMD snapshots.  Columns are
#: :data:`SEPPI21_A1_COLUMNS`.
#:
#: **Column ``A`` is not the normalisation, and nothing in this module reads
#: it.**  The paper fits :math:`\log_{10}P` with a per-slice offset, so ``A``
#: carries an amplitude convention that cannot be reconstructed from the paper:
#: against the analytic unit-integral amplitude :math:`q/(a\,\Gamma(k))` it is
#: low by 8.35, 7.71, 7.11 and 5.78 at the four redshifts.  A *varying* factor
#: is what rules out a constant convention.  The column stays for provenance
#: and the density is renormalised in closed form instead --- see
#: :func:`seppi21_logpdf`.
SEPPI21_A1 = np.array([
    [0.00, 4.10e-2, 1.397, 2.604, 7.225, 0.089,  0.776, -0.9590],
    [0.52, 4.40e-2, 2.501, 1.283, 2.394, 0.579, -0.325,  0.0334],
    [1.03, 3.37e-3, 4.140, 0.927, 0.688, 0.188,  0.081,  0.0813],
    [1.43, 2.54e-3, 4.450, 0.924, 0.623, 0.198,  0.095,  0.0813],
])

#: Column names of :data:`SEPPI21_A1`.
SEPPI21_A1_COLUMNS = ("z", "A", "alpha", "beta", "x0", "e0", "e1", "e2")

_S21_Z = jnp.asarray(SEPPI21_A1[:, 0])
_S21_ALPHA = jnp.asarray(SEPPI21_A1[:, 2])
_S21_BETA = jnp.asarray(SEPPI21_A1[:, 3])
_S21_X0 = jnp.asarray(SEPPI21_A1[:, 4])
_S21_E0 = jnp.asarray(SEPPI21_A1[:, 5])
_S21_E1 = jnp.asarray(SEPPI21_A1[:, 6])
_S21_E2 = jnp.asarray(SEPPI21_A1[:, 7])

#: Statistics of :math:`P(c)` that :func:`seppi21_scale` can pin to ``c_ref``.
ANCHORS = ("mean", "median", "mode")

#: Bracket for the quantile inversion, in :math:`\ln x` where
#: :math:`x = (c/a)^q` is the Gamma variate.
#:
#: Wide on purpose.  The node table below reaches a tail probability of 3e-15,
#: and the smallest shape ``k`` this model produces depends on how far outside
#: its calibration it is pushed: k = 0.27 at the edge of the fitted range needs
#: ln x = -125, and k = 0.04 -- sigma = 0.08, which the shipped mass grids do
#: reach -- needs -700.  The width costs nothing, because the bisection has a
#: fixed 60-step trip count and it is the Newton polish that sets the precision.
#:
#: The floor is narrowed to the working precision by :func:`_ln_x_bracket`; see
#: there for why a constant alone would give NaN in single precision.
_LN_X_BRACKET = (-700.0, 20.0)


def _ln_x_bracket():
    r""":data:`_LN_X_BRACKET`, with its floor raised to what the dtype can hold.

    The bisection evaluates :math:`P(k, e^\tau)`, and in float32
    :math:`e^{-700}` is not a small number, it is **zero** --- so
    :math:`P` and its derivative both vanish, and the Newton polish divides by
    that derivative.  The result is a NaN in one node of one mass bin, which is
    exactly the shape of failure that is easiest to miss.

    Raising the floor to just inside the smallest positive normal of the
    working dtype (-707 in float64, -86 in float32) keeps the derivative
    finite.  A node whose true quantile lies below that floor is not
    representable at that precision at all; it saturates at the floor, and
    since its weight is 3e-15 nothing downstream can tell.  The value returned
    in float64 is the constant above, unchanged.
    """
    # `math` and `jnp.finfo`, not numpy: this runs inside a traced call, and
    # `tests/test_paths.py` audits numpy there.  An exemption would be the
    # wrong answer when the rewrite is one line -- what is read is the dtype
    # config, and what comes back is a pair of Python floats, because
    # `invert_monotone` needs its bracket concrete.
    tiny = math.log(float(jnp.finfo(jnp.result_type(float)).tiny)) + 1.0
    return max(_LN_X_BRACKET[0], tiny), _LN_X_BRACKET[1]


# Double-exponential (tanh-sinh) nodes in the *probability* variable u, for
# integrals against P(c) written as int_0^1 f(c(u)) du.  Built once with numpy,
# so importing this module does not initialise a JAX backend and the jaxpr
# carries them as constants.
#
# Gauss-Legendre in u is the obvious rule and it is the wrong one: c(u) goes
# like u^(1/kq) at the bottom and has a heavy tail at the top, so the integrand
# is analytic at neither end and the error falls only as n^-2.  Measured on <c>
# against the closed form, over the four snapshots and sigma in [0.4, 1.6]:
# 1.5e-4 at the same 35 Gauss-Legendre nodes, against 6e-15 at the 35 built
# here.
#
# The nodes are stored as the *small* tail probability on each side -- u below
# the median, 1-u above it -- because that is the only representation in which
# both ends survive.  For the outermost node 1-u is 3e-15, and forming it by
# subtracting u from 1.0 would round it away entirely.
_S21_DE_H = 0.18
_S21_DE_N = 36
_de_t = _S21_DE_H * np.arange(-_S21_DE_N, _S21_DE_N + 1)
_de_s = 0.5 * np.pi * np.sinh(_de_t)
_de_e = np.exp(-2.0 * np.abs(_de_s))
_de_v = _de_e / (1.0 + _de_e)
_de_w = (0.25 * np.pi * _S21_DE_H) * np.cosh(_de_t) * 4.0 * _de_e / (1.0 + _de_e) ** 2
_de_keep = (_de_v > 1e-16) & (_de_w > 0.0) & np.isfinite(_de_w)
_de_v, _de_w = _de_v[_de_keep], _de_w[_de_keep]
_de_low = (_de_s < 0.0)[_de_keep]
_de_order = np.argsort(np.where(_de_low, _de_v, 2.0 - _de_v))    # ascending in c
_de_v, _de_w, _de_low = _de_v[_de_order], _de_w[_de_order], _de_low[_de_order]

_DE_V_LO = jnp.asarray(_de_v[_de_low])
_DE_W_LO = jnp.asarray(_de_w[_de_low])
_DE_V_HI = jnp.asarray(_de_v[~_de_low][::-1])
_DE_W_HI = jnp.asarray(_de_w[~_de_low][::-1])

#: The 35 quadrature weights, in node order.  They sum to 1 as a property of
#: the truncated rule rather than by renormalisation.
_DE_W = jnp.concatenate([_DE_W_LO, _DE_W_HI])

del _de_t, _de_s, _de_e, _de_v, _de_w, _de_keep, _de_low, _de_order


# -------------------------------------------------------------------------
# The generalised gamma, which is what Eq. 10 is
# -------------------------------------------------------------------------
#
# Write Eq. 10 as P(c) proportional to (c/a)^p exp[-(c/a)^q] and substitute
# x = (c/a)^q: then x is a Gamma(k, 1) variate with k = (p+1)/q.  That one
# sentence is why the normalisation, every moment, the mode, the variance of
# ln c and the quantiles below are closed forms and not quadratures.
#
# These helpers are private because there is one caller.  The moment a second
# module needs a generalised gamma -- a mass-observable scatter, an
# Eddington-ratio distribution -- they move to `ggah_mod.numerics` unchanged,
# and `_gg_quantile_unit` moves first, because it is `gammaincinv` and not a
# fact about concentrations.  `sectors.clf.upper_gamma` is the precedent for
# keeping a more general function next to its only consumer.


def _gg_log_moment(k, q, n):
    r""":math:`\ln\langle y^n\rangle` for :math:`y = c/a`, any real ``n``."""
    return gammaln(k + n / q) - gammaln(k)


@partial(jax.jit, static_argnames=("lower",))
def _gg_quantile_unit(k, q, v, lower: bool):
    r""":math:`y = c/a` at tail probability ``v``, on the tail ``lower`` names.

    ``lower=True`` solves :math:`P(k, x) = v` and ``lower=False`` solves
    :math:`Q(k, x) = v`, so ``v`` is in both cases the *small* probability in
    the tail being asked about.

    Notes
    -----
    **JAX has no ``gammaincinv``, and the obvious replacement is unusable in
    the upper tail.**  Bracketing :math:`P(k, e^\tau) = v` under
    ``stop_gradient`` and polishing with one Newton step
    (:func:`~ggah_mod.numerics.invert_monotone`) reproduces
    ``scipy.special.gammaincinv`` to 2e-15 and keeps a live derivative in
    :math:`k`, :math:`q` and :math:`v` --- but only while ``v`` is the *lower*
    probability.  Used on the upper half it would have to solve
    :math:`P = 1 - v`, and at :math:`v \sim 10^{-15}` that target has no
    significant digits left --- it rounds to exactly 1.0, which :math:`P(k, x)`
    never reaches, so the bisection simply runs to the top of its bracket.
    Measured against ``scipy`` at the outermost node, that is wrong by **9e-4**
    relative.  Solving :math:`Q(k, e^\tau) = v` there instead gives **2e-15**.

    Which branch a node is on is fixed by the node table, so the choice is a
    Python ``bool`` here and never a traced one -- which is also what lets this
    be jitted on it.  That matters more than it looks:
    :func:`~ggah_mod.numerics.invert_monotone` builds a
    :func:`jax.lax.fori_loop`, and eagerly that is an XLA compilation *per
    call*, with no cache, because the closure handed to it is a new object
    every time.  Jitted here it is one compilation per array shape.

    :func:`jax.scipy.special.gammaincc` decreases in :math:`x` while
    :func:`~ggah_mod.numerics.invert_monotone` wants an increasing function, so
    both it and the target are negated rather than the bracket reversed.
    """
    # `invert_monotone` sizes its bracket from the target, so the probability
    # has to be broadcast to the full (..., n_node) shape before it goes in.
    v = jnp.broadcast_to(v, jnp.broadcast_shapes(jnp.shape(k), jnp.shape(v)))
    if lower:
        f, y = (lambda t: gammainc(k, jnp.exp(t))), v
    else:
        f, y = (lambda t: -gammaincc(k, jnp.exp(t))), -v
    return jnp.exp(invert_monotone(f, y, *_ln_x_bracket()) / q)


@jax.jit
def _gg_quantile(k, q, u):
    r""":math:`y = c/a` at cumulative probability ``u``, for a *traced* ``u``.

    Both branches of :func:`_gg_quantile_unit` are evaluated and selected with
    :func:`jax.numpy.where`, because ``u`` is a value here rather than a fixed
    node.  ``where`` selects cleanly and carries the full gradient; the 50/50
    tie that :func:`~ggah_mod.numerics.lin_weights` warns about belongs to
    ``minimum``/``maximum``, not to this.
    """
    u = jnp.asarray(u, dtype=float)
    lo = _gg_quantile_unit(k, q, u, True)
    hi = _gg_quantile_unit(k, q, 1.0 - u, False)
    return jnp.where(u < 0.5, lo, hi)


@partial(jax.jit, static_argnames=("anchor",))
def _gg_scale_for(c_ref, p, q, anchor: str):
    r"""The scale :math:`a` that puts ``anchor`` of the distribution on ``c_ref``."""
    if anchor not in ANCHORS:
        raise ValueError(
            f"anchor must be one of {list(ANCHORS)}, got {anchor!r}.  It names "
            f"which statistic of P(c) the concentration you passed *is*, and "
            f"the three differ by several per cent -- at z = 0 and sigma = 1 "
            f"the mean, median and mode of Eq. 10 are 6.16, 6.00 and 5.69.  "
            f"Guessing would put a systematic in the scale.")
    k = (p + 1.0) / q
    if anchor == "mean":
        return c_ref * jnp.exp(-_gg_log_moment(k, q, 1))
    if anchor == "median":
        return c_ref / _gg_quantile_unit(k, q, jnp.asarray(0.5), True)
    return c_ref * (q / p) ** (1.0 / q)          # mode: y_max = (p/q)^(1/q)


# -------------------------------------------------------------------------
# The redshift axis of Table A.1
# -------------------------------------------------------------------------

def _s21_z_weights(z):
    r"""Weights on the four Table A.1 rows: linear in ``z``, clamped outside.

    Uses :func:`~ggah_mod.numerics.lin_weights` rather than
    :func:`jax.numpy.interp`, and the difference is load-bearing here in a way
    it is not for :func:`c_klypin16`.  This table has a node at **z = 0**,
    which is the fiducial redshift of essentially every call, and a clamp
    written with ``clip`` splits the gradient of the tie 50/50 -- returning
    :math:`\partial/\partial z` at exactly half its true value, silently.
    ``lin_weights`` clamps the query with ``where`` instead.  Its docstring
    records what the same geometry cost on the neutrino-mass axis.
    """
    i, t = lin_weights(jnp.asarray(z, dtype=float), _S21_Z)
    j = jnp.arange(_S21_Z.size)
    if jnp.ndim(i):                       # a stack of redshifts: add a row axis
        i, t = i[..., None], t[..., None]
    return jnp.where(j == i, 1.0 - t, 0.0) + jnp.where(j == i + 1, t, 0.0)


def _require_scalar_z(z, what: str):
    r"""Refuse a stack of redshifts where the shape would silently be wrong.

    :func:`_s21_z_weights` returns ``(4,)`` for a scalar and ``(Nz, 4)`` for a
    vector, while ``ln_s`` is ``(NM, 1)``.  The two then meet in a product that
    raises unless ``Nz == NM`` --- and at ``Nz == NM`` it does **not** raise: it
    pairs ``sigma[i]`` with ``z[i]`` and returns the diagonal of the table the
    caller wanted, smooth and correctly shaped and wrong.

    The docstrings of this family have always said "scalar, a stack of
    redshifts is `jax.vmap`, not a broadcast".  Nothing enforced it.  The check
    is on the static shape, so it costs nothing under ``jit`` and -- unlike
    :func:`_warn_clamped_z` -- it is *not* skipped under trace, because a
    tracer's rank is known even when its value is not.
    """
    if jnp.asarray(z).ndim != 0:
        raise ValueError(
            f"{what} takes one redshift; got shape {jnp.asarray(z).shape}.  `sigma` "
            f"and `z` index different axes here, so a broadcast between them "
            f"is ambiguous and silently returns the diagonal when the two "
            f"lengths happen to match.  Use `jax.vmap` over the redshifts.")


def _warn_clamped_z(z):
    r"""Warn that the Table A.1 shape is being *frozen*, not extrapolated.

    Eq. 9 extrapolates: it is a smooth analytic function of :math:`z`, and its
    range is already declared in :data:`CM_CALIBRATION`, so
    :func:`~ggah_mod.halos.calibration.check_calibration` reports it from
    :func:`~ggah_mod.halos.field.make_field` like every other fit's.  Eq. 10
    does not extrapolate.  :func:`_s21_z_weights` clamps, so outside
    :math:`[0, 1.43]` the shape is the endpoint snapshot's while the anchor
    ``c_ref`` keeps moving with redshift --- a frozen distribution around a
    moving centre, and nothing in the number says so.  That is a different
    thing from an extrapolated fit and it is warned about separately.

    Steps aside under tracing, the way
    :func:`~ggah_mod.halos.calibration._check_one` and
    :func:`~ggah_mod.halos.variance.check_k_support` do: ``float()`` on a
    tracer raises, and a check that cannot run must not be a check that fails.
    It only ever *checks* -- the clamp itself is in ``lin_weights`` and runs
    identically either way -- so a jitted call and an eager one agree bit for
    bit and differ only in whether this warning is emitted.
    """
    # Deferred: `calibration` imports CM_CALIBRATION from this module at module
    # scope, so importing it back at module scope here would be a cycle.
    from .calibration import MismatchWarning
    try:
        z_f = float(z)
    except (TypeError, ValueError):       # a tracer; the rank is gated above
        return
    z_lo, z_hi = float(SEPPI21_A1[0, 0]), float(SEPPI21_A1[-1, 0])
    if z_lo <= z_f <= z_hi:
        return
    warnings.warn(
        f"the Seppi et al. (2021) concentration distribution is tabulated at "
        f"z = {list(SEPPI21_A1[:, 0])} and you asked for z = {z_f:g}.  Its "
        f"shape is *clamped* to the nearest snapshot rather than extrapolated, "
        f"because the four fits are independent and their parameters are not "
        f"monotone in z.  The anchor c(M, z) is not clamped, so what comes "
        f"back is a frozen distribution around a centre that has kept moving. "
        f"The scatter it reports is the endpoint's, not this redshift's.",
        MismatchWarning, stacklevel=3)


# -------------------------------------------------------------------------
# Eq. 9 -- the mean relation
# -------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("mdef",))
def c_seppi21(sigma, z, mdef: str = "vir"):
    r"""Seppi et al. (2021) Eq. 9: :math:`c_{\rm vir}` from :math:`\sigma` and z.

    .. math::

        c(\sigma, z) = \frac{b_0}{(1+z)^{1/5}}
          \left[1 + 7.37\left(\frac{\sigma}{a_0\sqrt{1+z}}\right)^{3/4}\right]
          \left[1 + 0.14\left(\frac{\sigma}{a_0\sqrt{1+z}}\right)^{-2}\right]

    with :math:`a_0, b_0` from :data:`SEPPI21`.  MultiDark (HMD, BigMD, MDPL2)
    at the Planck 2014 cosmology :math:`(h, \Omega_{\rm m}, \sigma_8) =
    (0.6777, 0.307115, 0.8228)`, ROCKSTAR haloes, fitted for
    :math:`M > 10^{12.5}\,M_\odot/h` (:math:`\nu \approx 0.95` at z = 0) over
    :math:`0 \le z \le 1.43`.

    **The high-mass upturn is the physical content.**  The second bracket turns
    the relation up again above :math:`\nu \sim 3`, which the monotone power
    laws cannot do; :func:`c_klypin16` has it too, but as an interpolation in
    :math:`(M, z)` that cannot respond to a cosmology, whereas this one reaches
    it through :math:`\sigma` and therefore does.  Against
    :func:`c_bhattacharya13` at ``mdef="vir"`` it agrees to 5-11 per cent for
    :math:`z \le 0.5` and to about 14 per cent at z = 1.

    Parameters
    ----------
    sigma : array
        :math:`\sigma(M, z)` from
        :func:`~ggah_mod.halos.variance.sigma_of_mass`.
    z : float or array
        The redshift itself --- **not** a growth factor and not a growth rate.
        The fit is tabulated in :math:`z`, and :math:`\sigma` already carries
        the redshift, so the explicit :math:`(1+z)` factors are what is left.

    Notes
    -----
    The mass definition is :math:`M_{\rm vir}` with the Bryan & Norman (1998)
    :math:`\Delta_{\rm vir}`, which the paper quotes against the **mean matter
    density** (332.5 at z = 0, tending to 178 at high redshift).  That is the
    same physical boundary as :class:`~ggah_mod.halos.mass_definitions.MassDef`
    ``("vir")`` here, which carries :math:`\Delta` against the critical density
    instead: ``MassDef('vir').delta_mean(0, PLANCK18)`` is 330.7, and the 0.5
    per cent is :math:`\Omega_{\rm m}` alone.  So nothing is converted on the
    way in.

    The paper's :math:`\sigma` is MultiDark's total-matter one and this
    package's is built from :math:`P_{\rm cb}`; the difference is deliberate on
    both sides and is the same one recorded for ``diemer19``.
    """
    if mdef != "vir":
        raise ValueError(
            f"seppi21 was fitted to M_vir alone -- ROCKSTAR haloes at the "
            f"Bryan & Norman (1998) Delta_vir -- and you asked for "
            f"mdef={mdef!r}.  Unlike duffy08 and bhattacharya13 the paper "
            f"published no per-definition coefficients, so there is nothing "
            f"here to select: another Delta is a different halo boundary and "
            f"the number would be a systematic, not a tolerance.  Use "
            f"mdef='vir', or convert afterwards with `translate_mass`, which "
            f"moves M and c together.")
    zp = 1.0 + jnp.asarray(z)
    u = jnp.asarray(sigma) / (SEPPI21["a0"] * jnp.sqrt(zp))
    return (SEPPI21["b0"] / zp ** 0.2
            * (1.0 + 7.37 * u ** 0.75) * (1.0 + 0.14 * u ** -2.0))


# -------------------------------------------------------------------------
# Eq. 10 -- the distribution
# -------------------------------------------------------------------------

@jax.jit
def _seppi21_shape(sigma, z):
    ln_s = jnp.log(jnp.asarray(sigma))[..., None]
    w = _s21_z_weights(z)
    p = jnp.exp(jnp.sum(w * (jnp.log(_S21_ALPHA) + _S21_E1 * ln_s), axis=-1))
    q = jnp.exp(jnp.sum(w * (jnp.log(_S21_BETA) + _S21_E2 * ln_s), axis=-1))
    return p, q


def seppi21_shape(sigma, z):
    r"""The dimensionless shape :math:`(p, q)` of Eq. 10 at :math:`(\sigma, z)`.

    .. math::

        P(c) \propto \left(\frac{c}{a}\right)^{p}
                     \exp\left[-\left(\frac{c}{a}\right)^{q}\right],
        \qquad p = \alpha\sigma^{e_1},\quad q = \beta\sigma^{e_2}

    with :math:`(\alpha, \beta, e_1, e_2)` from :data:`SEPPI21_A1`.  The scale
    :math:`a` is deliberately absent: it is chosen by :func:`seppi21_scale`,
    for the reason in the module docstring.

    **Why the derived pair and not the table columns.**  The four snapshots
    were fitted independently and their parameters are not monotone in
    :math:`z` --- :math:`x_0` runs 7.225, 2.394, 0.688, 0.623 and :math:`e_1`
    changes sign --- so interpolating them is meaningless.  At fixed
    :math:`\sigma`, :math:`p` and :math:`q` are each monotone across all four
    rows, so it is :math:`(\ln p, \ln q)` that is interpolated in :math:`z`.

    Not ``jax.jit``-ed, so that :func:`_warn_clamped_z` can see a concrete
    redshift; the kernel underneath is.  Jit the caller.

    Parameters
    ----------
    sigma : array
    z : float
        Scalar.  A stack of redshifts is :func:`jax.vmap`, not a broadcast ---
        ``sigma`` and ``z`` index different axes and an implicit broadcast
        between them would be ambiguous.
    """
    _require_scalar_z(z, "seppi21_shape")
    _warn_clamped_z(z)
    return _seppi21_shape(sigma, z)


@jax.jit
def _seppi21_table_scale(sigma, z):
    ln_s = jnp.log(jnp.asarray(sigma))[..., None]
    return jnp.exp(jnp.sum(_s21_z_weights(z)
                           * (jnp.log(_S21_X0) + _S21_E0 * ln_s), axis=-1))


def seppi21_table_scale(sigma, z):
    r"""The paper's own scale :math:`a = x_0\,\sigma^{e_0}` of Eq. 10.

    Here so the paper's own figure can be reproduced and so that anchoring can
    be *measured against* rather than merely asserted --- it is what
    ``test_the_pdf_mean_reproduces_the_mean_relation`` uses, and it is why
    :math:`x_0` and :math:`e_0` are carried in :data:`SEPPI21_A1` at all.
    Nothing else in this module reads it.

    At this scale the mean of Eq. 10 reproduces :func:`c_seppi21` to 2 per cent
    near :math:`\sigma = 1`, to 8 per cent across
    :math:`0.6 < \sigma < 1.0`, and to 24 per cent by
    :math:`\sigma = 1.6` --- the PDF was fitted on six mass slices spanning
    only :math:`2\times10^{13}` to :math:`2\times10^{14}\,M_\odot/h`, so
    :math:`e_0`, :math:`e_1` and :math:`e_2` are weakly constrained outside
    that window.  **That residual is the reason this module anchors the scale
    instead of importing it.**
    """
    _require_scalar_z(z, "seppi21_table_scale")
    _warn_clamped_z(z)
    return _seppi21_table_scale(sigma, z)


def seppi21_scale(c_ref, sigma, z, anchor: str = "mean"):
    r"""The scale :math:`a` putting the chosen statistic of Eq. 10 on ``c_ref``.

    This is the whole of how the model attaches to another relation.  ``c_ref``
    is a concentration from anywhere --- :func:`c_bhattacharya13`,
    :func:`c_diemer19`, a field's ``conc`` leaf --- and ``anchor`` says which
    statistic of :math:`P(c)` that concentration *is*:

    ==========  ================================================
    ``mean``    :math:`a = c_{\rm ref}\,\Gamma(k)/\Gamma(k+1/q)`
    ``median``  :math:`a = c_{\rm ref} / Q(k, q, 1/2)`
    ``mode``    :math:`a = c_{\rm ref}\,(q/p)^{1/q}`
    ==========  ================================================

    with :math:`k = (p+1)/q`.  The argument is called ``anchor`` and not
    ``statistic`` because :func:`c_diemer19` already uses ``statistic`` for a
    different thing --- which of two published fits --- and one word for two
    meanings is how the two come apart.
    """
    p, q = seppi21_shape(sigma, z)
    return _gg_scale_for(jnp.asarray(c_ref), p, q, anchor)


def seppi21_logpdf(c, c_ref, sigma, z, anchor: str = "mean"):
    r""":math:`\ln P(c)`, normalised in closed form.

    .. math::

        P(c) = \frac{q}{a\,\Gamma(k)}\left(\frac{c}{a}\right)^{p}
               \exp\left[-\left(\frac{c}{a}\right)^{q}\right],
        \qquad k = \frac{p+1}{q}

    The :math:`\int P\,\dd c = 1` is analytic, not a quadrature: the amplitude
    is :math:`q/(a\Gamma(k))` because :math:`\int_0^\infty y^p e^{-y^q}\dd y =
    \Gamma(k)/q`.  Column ``A`` of :data:`SEPPI21_A1` does not appear --- see
    that table's note for why it cannot.

    The log form is the primary one because the density spans many decades and
    :func:`jax.scipy.special.gammaln` is the only safe route to
    :math:`\Gamma(k)` over the range of :math:`k` this model reaches
    (0.04 to 14 across the shipped mass grids).
    """
    p, q = seppi21_shape(sigma, z)
    a = _gg_scale_for(jnp.asarray(c_ref), p, q, anchor)
    y = jnp.asarray(c) / a
    return (jnp.log(q) - jnp.log(a) - gammaln((p + 1.0) / q)
            + p * jnp.log(y) - y ** q)


def seppi21_pdf(c, c_ref, sigma, z, anchor: str = "mean"):
    r""":math:`P(c)`, the exponential of :func:`seppi21_logpdf`."""
    return jnp.exp(seppi21_logpdf(c, c_ref, sigma, z, anchor))


def seppi21_quantile(u, c_ref, sigma, z, anchor: str = "mean"):
    r"""The concentration below which a fraction ``u`` of haloes of this mass lie.

    :math:`c(u) = a\,[P^{-1}(k, u)]^{1/q}`, evaluated on whichever tail ``u``
    is nearer.  See :func:`_gg_quantile_unit` for why the branch matters: in
    the far upper tail the other one is not slightly worse but unusable.
    """
    p, q = seppi21_shape(sigma, z)
    a = _gg_scale_for(jnp.asarray(c_ref), p, q, anchor)
    return a * _gg_quantile((p + 1.0) / q, q, u)


def sigma_ln_c_seppi21(sigma, z, kind: str = "exact"):
    r"""Width of :math:`\ln c` at fixed mass.  **Natural log, not dex.**

    ===============  ==========================================================
    ``"exact"``      :math:`\sqrt{\psi_1(k)}/q`, the standard deviation of
                     :math:`\ln c` under Eq. 10 itself.
    ``"lognormal"``  :math:`\sqrt{\ln(1+{\rm cv}^2)}`, the width of the
                     lognormal with the same mean and variance in *linear* c.
    ===============  ==========================================================

    **These are two different numbers and the difference is not small.**  Over
    the calibrated range ``"exact"`` gives 0.195-0.268 dex and ``"lognormal"``
    gives 0.177-0.204 dex, a factor 1.48 apart at worst.  They would coincide
    for a lognormal, and Eq. 10 is not one: its skewness runs from -0.09 to
    1.05 across the calibrated range, and is above 0.9 everywhere at z > 1.
    Which one is wanted depends on the use: comparing against a *measured*
    scatter in :math:`\ln c` is ``"exact"``;
    replacing this distribution by a lognormal of the same mean and variance is
    ``"lognormal"``, and that substitution is the thing this package makes
    visible rather than convenient.  Divide by :math:`\ln 10` for dex.

    Notes
    -----
    Two properties make this portable, and both are asserted in the tests
    rather than assumed here.

    **It does not depend on the anchor, or on** ``c_ref``.  An anchor rescales
    :math:`c`, and a rescaling cannot move a width in :math:`\ln c`; so the
    scatter attached to :func:`c_bhattacharya13` is bit-identical to the one
    attached to :func:`c_seppi21`, even though the two mean relations differ by
    5-14 per cent.

    **It barely depends on the mass definition.**
    :func:`seppi21_definition_jacobian` measures
    :math:`\dd\ln c_{\rm out}/\dd\ln c_{\rm in}` at 0.965-0.984 for
    vir to 200m and 1.026-1.058 for vir to 200c, so the fractional width
    survives a change of boundary to under 6 per cent, and
    :func:`seppi21_shape_translated` carries the residual exactly when it is
    wanted.
    """
    if kind not in ("exact", "lognormal"):
        raise ValueError(
            f"kind must be 'exact' or 'lognormal', got {kind!r}.  They differ "
            f"by up to a factor 1.48 here because Eq. 10 is skewed, so there "
            f"is no safe default to fall back on: 'exact' is the standard "
            f"deviation of ln c under the distribution itself, 'lognormal' is "
            f"the width of the lognormal matching its mean and variance in "
            f"linear c.")
    p, q = seppi21_shape(sigma, z)
    k = (p + 1.0) / q
    if kind == "exact":
        return jnp.sqrt(polygamma(1, k)) / q
    cv2 = jnp.exp(_gg_log_moment(k, q, 2) - 2.0 * _gg_log_moment(k, q, 1)) - 1.0
    return jnp.sqrt(jnp.log1p(cv2))


def seppi21_nodes(c_ref, sigma, z, anchor: str = "mean"):
    r"""Quadrature nodes and weights for :math:`\int f(c)P(c)\,\dd c`.

    Returns ``(c, w)`` with ``c`` of shape ``broadcast(c_ref, sigma) + (35,)``
    --- **node axis last** --- and ``w`` of shape ``(35,)``.  The sum
    :math:`\sum_i w_i` is 1 to machine precision as a property of the truncated
    tanh-sinh rule, not by renormalisation, and
    :math:`\sum_i w_i c_i^{\,n}` reproduces the closed-form moments to 6e-15,
    5e-14 and 2e-13 for n = 1, 2, 3 over the calibrated range.  The one
    quantity it does *not* nail is :math:`\langle 1/c\rangle`, at 3e-7: the
    integrand has an algebraic endpoint singularity in :math:`u` that the rule
    only damps.

    The intended use, and the trap in it::

        c_i, w_i = seppi21_nodes(field.conc, field.sigma, field.z)
        u = jax.vmap(lambda c: nfw_uk(field.k, field.r_delta / c, c),
                     in_axes=1)(c_i)                     # (35, Nk, NM)
        u_bar = jnp.einsum("n,nkm->km", w_i, u)

    **The scale radius moves with the concentration.**  :math:`r_\Delta` is
    fixed by the mass and the definition; :math:`r_s = r_\Delta/c` is not.
    Holding ``field.r_s`` fixed and varying only the ``c`` argument of
    :func:`~ggah_mod.halos.profiles.nfw_uk` is a different calculation, and at
    :math:`\bar c = 6` it is wrong by **+21 per cent at k = 10 h/Mpc** and +31
    per cent at k = 30 --- against a true marginalisation effect of **-3.3 per
    cent**.  The sign is wrong as well as the size.

    Marginalising properly moves :math:`\langle u\rangle` by -0.15 per cent at
    k = 1, -1.4 per cent at k = 3 and -3.3 per cent at k = 10 h/Mpc.  Nothing
    in the halo field is changed by this function: it takes arrays and returns
    arrays, which is the point of returning nodes rather than a marginalised
    profile.
    """
    p, q = seppi21_shape(sigma, z)
    return _seppi21_nodes(jnp.asarray(c_ref), p, q, anchor), _DE_W


@partial(jax.jit, static_argnames=("anchor",))
def _seppi21_nodes(c_ref, p, q, anchor: str):
    a = _gg_scale_for(c_ref, p, q, anchor)
    k = jnp.atleast_1d((p + 1.0) / q)[..., None]
    q_n = jnp.atleast_1d(q)[..., None]
    nodes = jnp.concatenate([_gg_quantile_unit(k, q_n, _DE_V_LO, True),
                             _gg_quantile_unit(k, q_n, _DE_V_HI, False)], axis=-1)
    return (jnp.reshape(a, jnp.shape(a) + (1,))
            * jnp.reshape(nodes, jnp.shape(a) + (_DE_W.size,)))


# -------------------------------------------------------------------------
# Carrying the shape to another mass definition
# -------------------------------------------------------------------------

def seppi21_shape_translated(p, q, dlnc):
    r"""The shape of the same distribution after :math:`\ln c' = J\ln c + {\rm const}`.

    If :math:`c' \propto c^{J}` then :math:`(c/a)^q = (c'/a')^{q/J}`, so the
    transformed variable is the *same* generalised gamma with

    .. math::  q \to q/J, \qquad p \to (p+1)/J - 1,

    and :math:`k = (p+1)/q` unchanged --- exactly, not to first order.  ``J``
    is :func:`seppi21_definition_jacobian`.  Offered rather than applied by
    default, because over the definitions this package uses it is at most a 6
    per cent correction on a width.
    """
    dlnc = jnp.asarray(dlnc)
    return (jnp.asarray(p) + 1.0) / dlnc - 1.0, jnp.asarray(q) / dlnc


def seppi21_definition_jacobian(c_in, mdef_out, z, cosmo, mdef_in="vir"):
    r""":math:`\dd\ln c_{\rm out}/\dd\ln c_{\rm in}` for an NFW halo.

    How much of a fractional scatter in concentration survives a change of
    mass definition.  Measured over :math:`10^{13} < M < 10^{15}\,M_\odot/h`,
    :math:`0 < z < 1` and :math:`3 < c < 10`: 0.965 to 0.984 for vir to 200m,
    1.026 to 1.058 for vir to 200c.

    **It does not depend on the mass, and that is structural rather than
    lucky.**  :func:`~ggah_mod.halos.mass_definitions.translate_mass` returns
    :math:`c_{\rm out} = c_{\rm in}x` with :math:`x` from a solve that sees
    only :math:`c_{\rm in}` and the ratio of the two overdensity thresholds;
    the mass enters the radius, which is discarded here.  So the mass passed
    below is arbitrary.  A test asserts the independence rather than trusting
    this paragraph.
    """
    c_in = jnp.asarray(c_in, dtype=float)

    def ln_c_out(ln_c):
        return jnp.log(translate_mass(jnp.ones_like(ln_c), jnp.exp(ln_c),
                                      mdef_in, mdef_out, z, cosmo)[2])

    return jax.grad(lambda x: jnp.sum(ln_c_out(x)))(jnp.log(c_in))

#: What each relation was calibrated on, and whether it can respond to cosmology.
#:
#: **The last column is also the curvature column.**  All six were fitted in
#: flat boxes, and the three marked ``False`` are pure power laws in mass and
#: redshift -- so :math:`\partial c/\partial\Omega_k` is structurally zero for
#: them, not small.  The three marked ``True`` respond through
#: :math:`\sigma(M)`, :math:`D(z)` and :math:`f(z)`, each of which carries
#: curvature already: the first through :math:`P_{cb}(k)` and the other two
#: through the spectrum they are read off.  So a curved cosmology reaches
#: concentration by the same route it reaches everything else in this layer,
#: and nothing here needs a guard.  What it does *not* do is make any of these
#: relations calibrated under curvature, which is why this table records the
#: suite rather than asserting the fit travels.
CM_CALIBRATION = {
    "duffy08": ("WMAP5", "200c/vir/200m", (0.0, 2.0), False),
    "dutton14": ("Planck13", "200c/vir", (0.0, 5.0), False),
    "klypin16": ("Planck13/MultiDark", "200c/vir", (0.0, 5.4), False),
    "bhattacharya13": ("WMAP7", "200c/vir/200m", (0.0, 2.0), True),
    "diemer19": ("universal in (nu, n_eff, alpha_eff)", "200c", (0.0, 5.0), True),
    "seppi21": ("Planck14/MultiDark (HMD, BigMD, MDPL2; Rockstar)", "vir",
                (0.0, 1.43), True),
}

CONCENTRATION: dict[str, Callable] = {
    "duffy08": c_duffy08,
    "dutton14": c_dutton14,
    "klypin16": c_klypin16,
    "bhattacharya13": c_bhattacharya13,
    "diemer19": c_diemer19,
    "seppi21": c_seppi21,
}

#: Relations parameterised by peak height; they take ``sigma``, not ``m_h``.
#:
#: Three of them now, and **three different second arguments**: the growth
#: factor for ``bhattacharya13``, the pair (effective slope, growth rate) for
#: ``diemer19``, and the redshift itself for ``seppi21``.  Membership here says
#: only that the first argument is sigma; :func:`~ggah_mod.halos.field._concentration`
#: is where the rest is dispatched, and it is the one place that knows.
PEAK_HEIGHT_MODELS = frozenset({"bhattacharya13", "diemer19", "seppi21"})


def make_concentration(name: str) -> Callable:
    """Look up a concentration relation by name.

    Note the two families have **different signatures** -- the empirical fits
    take ``(m_h, z, mdef)`` and the peak-height relations take ``sigma`` and a
    second physical argument.  That is deliberate: a uniform signature would
    mean the power-law fits accepting a cosmology they cannot use, which is how
    the predecessor ended up with relations that silently ignored it.
    """
    key = str(name).lower()
    if key not in CONCENTRATION:
        raise ValueError(f"unknown concentration relation {name!r}; "
                         f"expected one of {sorted(CONCENTRATION)}")
    return CONCENTRATION[key]
