r"""Hankel transforms: two engines, one interface, no special function traced.

Everything layer 5 projects is a Hankel transform of a spectrum:

.. math::

    \xi(r) = \frac{1}{2\pi^2}\int dk\,k^2 P(k)\,j_0(kr),
    \qquad
    \Sigma(R) = \frac{\bar\rho_m}{2\pi}\int dk\,k\,P(k)\,J_0(kR),

.. math::

    \Delta\Sigma(R) = \frac{\bar\rho_m}{2\pi}\int dk\,k\,P(k)\,J_2(kR),
    \qquad
    w(\theta) = \frac{1}{2\pi}\int d\ell\,\ell\,C_\ell\,J_0(\ell\theta)

Four orders -- :math:`\nu = \tfrac12, 0, 2, 0` -- of one integral, and one
substitution :math:`t = kr` turns each into

.. math::  \frac{1}{r^{p+1}}\int_0^\infty dt\,t^{p}\,F(t/r)\,J_\nu(t)

so a single quadrature rule in :math:`t` serves all of them.

The observation that makes this cheap
--------------------------------------

**Every node, weight and coefficient depends only on static grid choices.**
Ogata's rule places its abscissae at the zeros of :math:`J_\nu` and weights them
by :math:`Y_\nu/J_{\nu+1}` there; FFTLog's coefficients are ratios of
:math:`\Gamma` at complex argument.  All of them are functions of
:math:`(\nu, N, h)` and the grid's endpoints -- **nothing traced enters them**.
So they are built once, in numpy, with ``scipy.special``, and the traced path is
an ``einsum`` (Ogata) or an ``rfft``, a multiply by a constant complex array and
an ``irfft`` (FFTLog).

That is why there is no ``bessel.py``: no Bessel function is ever evaluated at a
traced argument by either engine.  The one place a traced argument does reach a
special function is the King PSF, whose core radius is fitted -- see
:func:`bessel_k`.

The tie at the high-k continuation
-----------------------------------

Ogata's rule reaches :math:`x_{\max} = (\pi/h)\,\psi(hN) \approx \pi N` once
:math:`hN` is past about 2 (:func:`_ogata_nodes`) -- :math:`1.3\times10^4` at
``ACCURATE``'s ``n_hankel = 4096`` and ``hankel_h = 0.001`` -- so at
:math:`r = 0.01` it samples :math:`k \approx 1.3\times10^6\,h/`\ Mpc, far past
any grid.  The spectrum is continued as a power law with the slope capped below
:math:`-3` for convergence.  The predecessor writes that cap as ``jnp.minimum(slope, -3.0)``
-- and :math:`-3` is the *generic* high-k slope of a :math:`\Lambda`\ CDM
spectrum, so the tie is where the fiducial sits, not a corner case.  JAX splits a
``minimum`` tie 50/50, which is the same defect
:mod:`~ggah_mod.numerics` documents and which has already cost this package a
factor-of-two neutrino gradient and a discarded ``eps_sn``.  Here the cap is a
``where``, and the slope is measured on the log-log grid, as the secant
across the grid's last seven intervals (nodes ``-8`` to ``-1``).

Truncating instead of continuing is not an option either: the predecessor did
that until 2026-07, and fixing it moved :math:`w_p` and :math:`\Delta\Sigma` by
19% and 20%.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from ..numerics import interp_cubic

__all__ = ["HankelRule", "FFTLogRule", "make_hankel", "make_fftlog",
           "LOG_SPACING_ULPS",
           "hankel", "fftlog", "ENGINES", "KERNELS",
           "pk_to_xi", "pk_to_sigma", "pk_to_delta_sigma", "cl_to_wtheta",
           "bessel_k"]

#: The two engines.  Both are pure JAX on the traced path; they differ in the
#: algorithm, not in the flavour, and their disagreement is a measured row.
ENGINES = ("quadrature", "fftlog")

#: Ogata's step, when no backend says otherwise.  See
#: :attr:`~ggah_mod.backend.Backend.hankel_h` for why this is the accuracy knob
#: and ``n_hankel`` is not.
DEFAULT_OGATA_H = 0.001


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HankelRule:
    r"""A quadrature rule for :math:`\int_0^\infty f(t)J_\nu(t)\,dt`.

    Attributes
    ----------
    x : array (N,)
        Abscissae in :math:`t`.  Built in numpy; a constant thereafter.
    w : array (N,)
        Weights, with :math:`J_\nu(x_n)` and :math:`\psi'` already folded in, so
        the sum is :math:`\sum_n w_n f(x_n)` and no Bessel function appears on
        the traced path.
    order : float
        :math:`\nu`.  Static.
    engine : str
        Static.
    """

    x: jnp.ndarray
    w: jnp.ndarray
    order: float = 0.5
    engine: str = "quadrature"

    def tree_flatten(self):
        return ((self.x, self.w), (self.order, self.engine))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


@functools.lru_cache(maxsize=32)
def _ogata_nodes(order: float, n: int, h: float):
    r"""Ogata (2005) Type-II nodes and weights.  numpy, built once.

    .. math::

        \int_0^\infty f(t)J_\nu(t)\,dt \approx \pi\sum_n
            w_{\nu n}\,f(x_n)\,J_\nu(x_n)\,\psi'(h\xi_{\nu n})

    with :math:`\xi_{\nu n} = j_{\nu n}/\pi` the scaled zeros of :math:`J_\nu`,
    :math:`x_n = (\pi/h)\psi(h\xi_{\nu n})`,
    :math:`\psi(t) = t\tanh(\tfrac{\pi}{2}\sinh t)` and
    :math:`w_{\nu n} = Y_\nu(\pi\xi_{\nu n})/J_{\nu+1}(\pi\xi_{\nu n})`.

    The double-exponential :math:`\psi` makes the integrand decay like
    :math:`\exp(-\exp(t))`, which is why 512 nodes suffice for an oscillatory
    integral with no envelope.

    Memoised numpy, exactly like
    :func:`~ggah_mod.halos.profiles._leggauss_cached`: this runs before anything
    is traced and produces a constant.
    """
    from scipy.special import jn_zeros, jv, yv

    if float(order) == 0.5:
        # j_{1/2,n} = n*pi exactly, and scipy's jn_zeros takes integer orders
        # only.  Special-cased rather than approximated: the spherical case is
        # the one every real-space projection goes through.
        zeros = np.pi * np.arange(1, n + 1, dtype=np.float64)
    else:
        zeros = jn_zeros(int(round(order)), n)
    xi = zeros / np.pi
    t = h * xi
    sinh_t = np.sinh(t)
    psi = t * np.tanh(0.5 * np.pi * sinh_t)
    # `sinh(pi sinh t)` overflows float64 at t ~ 5.7 while `psi` has already
    # converged to `t` and `psi'` to 1 -- the double exponential saturates long
    # before the arithmetic does.  Clamped rather than caught after the fact, so
    # the construction runs without warnings and the limit is stated once.
    arg = np.minimum(np.pi * sinh_t, 700.0)
    dpsi = np.where(
        np.pi * sinh_t < 700.0,
        (np.pi * t * np.cosh(np.minimum(t, 700.0)) + np.sinh(arg))
        / (1.0 + np.cosh(arg)),
        1.0)
    x = np.pi * psi / h
    w = yv(order, np.pi * xi) / jv(order + 1.0, np.pi * xi)
    weight = np.pi * w * jv(order, x) * dpsi
    keep = np.isfinite(x) & np.isfinite(weight)
    return np.ascontiguousarray(x[keep]), np.ascontiguousarray(weight[keep])


#: Mellin transforms of the two kernels, :math:`U_K(z) = \int t^{z-1}K(t)dt`.
#:
#: Written as ``exp(loggamma(...) - loggamma(...))`` rather than as a ratio of
#: ``gamma``: the arguments run to :math:`\pm i\eta` with
#: :math:`\eta \sim \pi/\Delta \sim 10^3`, where ``gamma`` itself overflows
#: long before the *ratio* does.
def _mellin_bessel_j(nu, z):
    from scipy.special import loggamma
    return np.exp(np.log(2.0) * (z - 1.0) + loggamma(0.5 * (nu + z))
                  - loggamma(0.5 * (2.0 + nu - z)))


def _mellin_spherical_bessel_j(nu, z):
    r""":math:`\int_0^\infty t^{z-1}j_\nu(t)\,dt`.

    **The** :math:`\sqrt{\pi/2}` **is not optional.**  From
    :math:`j_\nu(t) = \sqrt{\pi/2t}\,J_{\nu+1/2}(t)`,

    .. math::

        \int t^{z-1}j_\nu\,dt
          = \sqrt{\tfrac{\pi}{2}}\;2^{z-3/2}\,
            \frac{\Gamma\big(\tfrac{\nu+z}{2}\big)}
                 {\Gamma\big(\tfrac{3+\nu-z}{2}\big)}

    and the prefactor is a *constant*, so leaving it out multiplies every
    :math:`\xi(r)` by 0.798 and changes no shape at all -- there is no plot on
    which it is visible.  ``mcfit`` omits it here and restores it through the
    per-transform ``prefac``, which is a legitimate bookkeeping choice and a
    trap for anyone reading the kernel as what its name says.  Here the name is
    the thing: this function is the Mellin transform of :math:`j_\nu`, and the
    caller supplies no compensating constant.
    """
    from scipy.special import loggamma
    return np.sqrt(0.5 * np.pi) * np.exp(
        np.log(2.0) * (z - 1.5) + loggamma(0.5 * (nu + z))
        - loggamma(0.5 * (3.0 + nu - z)))


KERNELS = {"bessel": _mellin_bessel_j, "spherical": _mellin_spherical_bessel_j}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FFTLogRule:
    r"""A log-spaced convolution rule for
    :math:`G(y) = \int_0^\infty F(x)K(xy)\,dx/x`.

    The output grid ``y`` is **not** chosen: FFTLog produces a grid reciprocal
    to the input's, ``y = e^{\ln xy - \Delta}/x[::-1]``, and a caller wanting
    other radii interpolates.  That constraint is the price of the
    :math:`O(N\log N)`, and it is why this rule needs the input grid at
    construction while the Ogata rule does not.

    With ``n_pad > 0`` the reciprocal partner is the **padded** grid, so ``y``
    has ``n + 2 n_pad`` points and reaches further at both ends than the grid
    the caller passed.  That is not incidental -- reaching further at small
    :math:`y` is half of what the padding buys, because it moves the caller's
    smallest radii off the end of the reciprocal grid, where FFTLog degrades,
    and into its middle, where it is exact to 1e-9.
    """

    y: jnp.ndarray
    u: jnp.ndarray
    x_fac: jnp.ndarray
    y_fac: jnp.ndarray
    order: float = 0.0
    engine: str = "fftlog"
    #: Nodes appended at *each* end of the input grid, and therefore the number
    #: :func:`fftlog` must continue ``f`` by.  Static: it is a grid property.
    n_pad: int = 0
    #: The grid's ln-spacing.  Carried because the continuation's slope cap is
    #: stated per unit ``ln x`` -- the same units ``hankel`` uses -- while the
    #: padding steps one node at a time, and the two differ by exactly this.
    delta: float = 0.0

    def tree_flatten(self):
        return ((self.y, self.u, self.x_fac, self.y_fac),
                (self.order, self.engine, self.n_pad, self.delta))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


@functools.lru_cache(maxsize=64)
def _fftlog_setup(order: float, kind: str, power: float, q: float,
                  n: int, delta: float, ln_x0: float):
    r"""FFTLog's grid and coefficients.  numpy, built once.

    .. math::

        u_m = U_K\Big(q + \frac{2\pi i m}{N\Delta}\Big)\,
              \exp\Big(-\frac{2\pi i m\ln(xy)}{N\Delta}\Big)

    A function of :math:`(\nu, K, q, N, \Delta, x_0)` and **nothing traced**,
    so ``scipy.special.loggamma`` is called here and the traced path is an
    ``rfft``, a complex multiply and an ``hfft``.  Same exemption category as
    :func:`~ggah_mod.halos.profiles._leggauss_cached`.

    ``lowring`` fixes the reciprocal offset so the kernel's phase is continuous
    across the periodic wrap.  Without it the transform rings at both ends of
    the output grid, and the ringing looks like physics.
    """
    mk = KERNELS[kind]
    ln_xy = delta / np.pi * np.angle(mk(order, q + 1j * np.pi / delta))

    m = np.arange(n // 2 + 1, dtype=np.float64)
    z = q + 2j * np.pi * m / (n * delta)
    u = mk(order, z) * np.exp(-2j * np.pi * ln_xy * m / (n * delta))

    x = np.exp(ln_x0 + delta * np.arange(n))
    y = np.exp(ln_xy - delta) / x[::-1]
    return (np.ascontiguousarray(y), np.ascontiguousarray(u),
            np.ascontiguousarray(x ** (power - q)),
            np.ascontiguousarray(y ** (-q)))


#: The log-spacing check's floor, in units of ``eps * span``.
#:
#: A log grid built as ``10**linspace`` carries a *relative* value error of
#: about ``eps*|ln x|/2``, which lands in ``ln x`` additively -- so the
#: deviation a **correct** grid shows is absolute in ``ln x`` and scales with
#: the grid's ln-range, not with ``delta``.  Measured, ``max|diff(ln x) -
#: delta|`` in units of the grid's own eps, for ``jnp.logspace`` in float32:
#:
#: ========================  ======  =============
#: range                     span    deviation
#: ========================  ======  =============
#: 1e-4 .. 200 (shipped k)   9.2     4.5 - 8.0
#: 1e-5 .. 1e4 (fixture)     11.5    10.6 - 13.6
#: 1e-8 .. 1e8               18.4    16.7 - 19.0
#: 1e-13 .. 1e13             29.9    18.3 - 33.5
#: ========================  ======  =============
#:
#: i.e. ~``0.5*span*eps``.  8 is ~10x that at every span, where a flat multiple
#: calibrated on the shipped grid falls to 1.9x by 26 decades -- and it is still
#: five orders below any grid that is not log-spaced at all.
LOG_SPACING_ULPS = 8.0


def _log_spacing_tol(delta: float, span: float, dtype) -> float:
    r"""How far ``diff(ln x)`` may stray from ``delta``, **absolutely**.

    The tolerance this replaced was ``np.allclose(spacing, delta, rtol=1e-6)``,
    i.e. relative to ``delta`` -- so its threshold *shrank* as the grid refined,
    2.8e-8 at ``n = 512`` against 3.5e-9 at ``n = 4096``, chasing a deviation
    that does not move with ``n`` at all.  Exactly the wrong direction.  Under
    JAX's default float32 that made every grid this package builds fail, and
    with it ``w_p``, ``xi``, ``Sigma``, ``Delta Sigma`` and ``w(theta)`` on the
    one flavour that runs FFTLog -- the one a forecast can differentiate.
    ``tests/conftest.py`` enables x64 before the first import, so the whole
    suite was blind to it.

    The floor keeps the old behaviour where the old behaviour was right, and it
    is not merely that float64 happens to pass: the absolute term is
    ``8*eps64*span <= 1.2e-13`` for any span under 70, while the relative term
    is ``1e-6*delta``, so the absolute term could only win below
    ``delta = 1.2e-7`` -- more than 6e8 points.  **In float64 this function
    returns exactly what the old check enforced.**

    **The ``1e-8`` is part of that and must not be dropped**: ``np.allclose``
    carries a default ``atol=1e-8``, so what was actually in force was
    ``1e-8 + 1e-6*delta``.  Writing the floor as ``1e-6*delta`` alone would
    tighten the float64 path rather than leave it alone.

    Why a tolerance is the honest answer rather than a silent resampling:
    :func:`_fftlog_setup` already *regenerates* the ideal grid from
    ``(ln x0, delta, n)`` and builds ``x_fac`` on it.  The rule never uses the
    caller's values, so what this check actually asks is "is your ``f_grid``
    sampled close enough to the grid I am about to substitute" -- and "to
    within what your dtype can express" is the correct reading of close enough.
    """
    return max(1e-8 + 1e-6 * delta,
               LOG_SPACING_ULPS * float(np.finfo(dtype).eps) * span)


def make_fftlog(order: float, x_grid, *, kind: str = "spherical",
                power: float = 3.0, q: float | None = None,
                n_pad: int | None = None, backend=None) -> FFTLogRule:
    """Build an FFTLog rule for one transform, on one input grid.

    ``n_pad`` extends the grid by that many nodes at *each* end.  Left unset it
    comes from :attr:`~ggah_mod.backend.Backend.fftlog_pad_decades`, converted
    against this grid's own spacing -- the knob is in decades so that it means
    the same thing on any grid.  See the note in the body: it exists for
    periodicity, not for resolution.
    """
    from ..backend import resolve_backend

    b = resolve_backend(backend)
    pad_decades = b.fftlog_pad_decades if n_pad is None else None
    # The dtype is read *before* widening, because widening cannot recover bits
    # already lost and the tolerance below is a statement about the grid's own
    # precision.  `np.asarray` rather than `np.result_type`, which reads a
    # length-2 tuple as a `(kind, size)` dtype spec and raises on any other.
    x_in = np.asarray(x_grid)
    dtype = (x_in.dtype if np.issubdtype(x_in.dtype, np.inexact)
             else np.dtype(np.float64))
    x = np.asarray(x_in, dtype=np.float64)
    n = x.size
    if n < 4:
        raise ValueError("FFTLog needs at least 4 log-spaced points")
    ln_x = np.log(x)
    delta = float((ln_x[-1] - ln_x[0]) / (n - 1))
    span = max(1.0, abs(float(ln_x[0])), abs(float(ln_x[-1])))
    deviation = float(np.max(np.abs(np.diff(ln_x) - delta)))
    tol = _log_spacing_tol(delta, span, dtype)
    if deviation > tol:
        raise ValueError(
            f"FFTLog requires a log-spaced input grid; this one deviates from "
            f"uniform ln-spacing by {deviation:.3g}, against a tolerance of "
            f"{tol:.3g} (delta = {delta:.6g}, n = {n}, dtype = {dtype}).  The "
            f"tolerance is absolute in ln x and floored at eps*span for the "
            f"grid's *own* dtype, read before the widening above: a grid built "
            f"under JAX's default float32 misplaces its nodes by a few eps32 in "
            f"ln x, and casting it to float64 here recovers none of that.  A "
            f"deviation orders above the floor is a genuinely non-log grid.  "
            f"The quadrature engine has no such requirement -- it interpolates "
            f"-- so either use it, or resample onto a log grid deliberately "
            f"rather than letting a transform do it silently.")
    if kind not in KERNELS:
        raise ValueError(f"unknown kernel {kind!r}; expected one of "
                         f"{sorted(KERNELS)}")
    q = (1.5 if kind == "spherical" else 1.0) if q is None else float(q)

    # FFTLog treats its input as periodic in ln x, so a grid whose integrand
    # has not decayed at the edge has a step at the wrap, and it rings.  The
    # padding buys the decay: `fftlog` continues `f` across it by the same
    # capped power law `hankel` uses past its own grid, so the two engines
    # extrapolate identically and the parity row measures quadrature.
    #
    # It is not a convergence fix.  FFTLog is stable to four digits across
    # n_k = 512..4096 on this input; widening the grid without continuing the
    # function changes nothing, which is why "unconverged" was the wrong
    # diagnosis for eight months.
    if n_pad is None:
        n_pad = int(round(float(pad_decades) * np.log(10.0) / delta))
    n_pad = int(n_pad)
    if n_pad < 0:
        raise ValueError(f"n_pad must be non-negative; got {n_pad}")
    if n_pad:
        j = np.arange(1, n_pad + 1)
        ln_x = np.concatenate([ln_x[0] - delta * j[::-1], ln_x,
                               ln_x[-1] + delta * j])
        n = ln_x.size

    y, u, x_fac, y_fac = _fftlog_setup(float(order), kind, float(power), q,
                                       n, delta, float(ln_x[0]))
    return FFTLogRule(jnp.asarray(y), jnp.asarray(u), jnp.asarray(x_fac),
                      jnp.asarray(y_fac), float(order), "fftlog", n_pad,
                      delta)


def _continue_for_pad(f_grid, n_pad: int, delta: float, *, slope_cap: float):
    r"""``f``, extended by ``n_pad`` nodes at each end as a power law.

    The same continuation :func:`hankel` applies past *its* grid, and
    deliberately so: two engines extrapolating differently do not measure each
    other.  The high-side slope is read over the last eight nodes and capped at
    ``slope_cap``, because the raw end slope is a property of whatever the
    caller passed -- a galaxy auto-spectrum's one-halo term leaves it at
    :math:`-1.79`, where :math:`k^3P` still *grows* and the padding would buy
    no decay at all.
    """
    log_f = _safe_log(f_grid)
    j = jnp.arange(1, n_pad + 1)

    # Per unit ln x, so the cap means the same thing it means in `hankel`.
    lo_slope = (log_f[1] - log_f[0]) / delta
    lo_slope = jnp.where(jnp.isfinite(lo_slope), lo_slope, 0.0)
    hi_slope = (log_f[-1] - log_f[-8]) / (7.0 * delta)
    hi_slope = jnp.where(jnp.isfinite(hi_slope) & (hi_slope < slope_cap),
                         hi_slope, slope_cap)

    # ... and back to per node for the walk out.
    below = log_f[0] - lo_slope * delta * j[::-1]
    above = log_f[-1] + hi_slope * delta * j
    return jnp.exp(jnp.concatenate([below, log_f, above]))


def fftlog(rule: FFTLogRule, f_grid, *, slope_cap: float = -3.0):
    r""":math:`G(y)` on ``rule.y``.  The whole traced path, in three lines."""
    f = jnp.asarray(f_grid)
    if rule.n_pad:
        f = _continue_for_pad(f, rule.n_pad, rule.delta, slope_cap=slope_cap)
    n = rule.x_fac.shape[-1]
    f = f * rule.x_fac
    g = jnp.fft.hfft(jnp.fft.rfft(f) * rule.u, n=n) / n
    return rule.y_fac * g


def _safe_log(f):
    r"""``log f``, with a floor that keeps the *slope* finite.

    A spectrum that underflows to exactly zero at high k -- which an analytic
    Gaussian does, and which is exactly the input the closed-form validation
    uses -- gives ``log f = -inf``, and then the tail slope below is
    ``-inf - (-inf) = nan``.  One NaN in the continuation poisons every radius.

    The floor is **relative to the spectrum's own peak**, not absolute: a
    :math:`P(k)` in :math:`(\mathrm{Mpc}/h)^3` and a :math:`C_\ell` differ by
    thirty orders of magnitude, and an absolute floor right for one silently
    truncates the other.  The predecessor used four different absolute floors --
    ``1e-20``, ``1e-30``, ``1e-40`` -- one per spectrum, chosen by hand.

    **And it is clamped to the working dtype's smallest normal, because
    ``1e-300`` is not a small number in float32 -- it is zero.**  The smallest
    normal float32 is 1.2e-38, so ``peak * 1e-300`` and the bare ``1e-300``
    both underflow, ``jnp.maximum(f, 0.0)`` is a no-op, and the ``-inf`` this
    function exists to prevent comes straight back.  Measured before the clamp:
    ``pk_to_xi`` returned ``nan`` at every radius in float32, on the very
    closed-form Gaussian the docstring above describes.  A floor written as a
    literal is a floor that assumes a dtype.
    """
    f = jnp.asarray(f)
    peak = jnp.max(jnp.abs(f))
    floor = jnp.where(peak > 0.0, peak * 1e-300, 1e-300)
    return jnp.log(jnp.maximum(f, jnp.maximum(floor, jnp.finfo(f.dtype).tiny)))


def _log_interp_with_tail(log_x_query, log_x, log_f, *, slope_cap: float):
    r"""``log F`` at ``log_x_query``, continued as a power law past the grid.

    Inside the grid the interpolation is :math:`C^1` cubic, not linear: the
    error falls as :math:`(\Delta\ln k)^4` instead of
    :math:`(\Delta\ln k)^2`, which on the shipped k grid is the difference
    between 9e-5 and 1e-8 against a closed-form transform pair -- at the same
    cost, since the ``searchsorted`` is shared.  Linear interpolation of a
    tabulated function is the defect class ``PLAN.md`` records twice already.

    The tail slope is the secant across the grid's last seven intervals --
    0.043 dex of ``ACCURATE``'s 1024-point log k grid, 0.086 dex of
    ``DIFFERENTIABLE``'s 512 -- and is capped **with a ``where``**:
    ``jnp.minimum(slope, -3.0)`` puts a tie at exactly -3, which is the generic
    high-k slope of a LambdaCDM spectrum, so the tie is where the fiducial sits
    and JAX splits it 50/50.

    A non-finite slope falls back to the cap rather than propagating: it means
    the grid ran into the floor, where the spectrum is negligible anyway.
    """
    lo_slope = (log_f[1] - log_f[0]) / (log_x[1] - log_x[0])
    hi_slope = (log_f[-1] - log_f[-8]) / (log_x[-1] - log_x[-8])
    lo_slope = jnp.where(jnp.isfinite(lo_slope), lo_slope, 0.0)
    hi_slope = jnp.where(jnp.isfinite(hi_slope) & (hi_slope < slope_cap),
                         hi_slope, slope_cap)

    inside = interp_cubic(log_x_query, log_x, log_f)
    above = log_f[-1] + hi_slope * (log_x_query - log_x[-1])
    below = log_f[0] + lo_slope * (log_x_query - log_x[0])
    out = jnp.where(log_x_query > log_x[-1], above, inside)
    return jnp.where(log_x_query < log_x[0], below, out)


def hankel(rule: HankelRule, r, x_grid, f_grid, *, power: float,
           slope_cap: float = -3.0):
    r""":math:`\int_0^\infty dt\;t^{\,\rm power}\,F(t/r)\,J_\nu(t)`, shape ``(Nr,)``.

    ``F`` is given on ``x_grid`` and interpolated in log-log, with the power-law
    continuation above.  The whole traced path is one ``interp`` and one
    ``einsum``: no Bessel function, no gamma function, no FFT.
    """
    r = jnp.atleast_1d(jnp.asarray(r))
    log_x = jnp.log(jnp.asarray(x_grid))
    f = jnp.asarray(f_grid)
    log_f = _safe_log(f)
    # (Nr, N): the argument of F at every node, for every output radius.
    log_arg = jnp.log(rule.x)[None, :] - jnp.log(r)[:, None]
    vals = jnp.exp(_log_interp_with_tail(log_arg, log_x, log_f,
                                         slope_cap=slope_cap))
    vals = _signed_fallback(vals, log_arg, log_x, f)
    return jnp.einsum("n,n,rn->r", rule.w, rule.x ** power, vals)


def _signed_fallback(vals, log_q, log_x, f):
    r"""Cubic interpolation of the *value* wherever the log-cubic cannot be trusted.

    :func:`hankel` interpolates :math:`\log F`, which a spectrum that changes
    sign does not have.  :func:`_safe_log` floors the non-positive nodes at
    ``peak * 1e-300`` (float64) or the smallest normal (float32), so a sign
    change becomes a step of ~690 (float64) or ~60 (float32) in
    :math:`\log F`, and the cubic overshoots it by a sizeable fraction of
    the step -- tens of e-folds above the spectrum's own peak in float64.

    **Measured on a pressure profile with a central depression** (ggah_cal's
    M*>10.5 CAP MAP, ``alpha_in_p = -0.79``): :math:`P_{gy}(k) < 0` above
    :math:`k \approx 140\,h/{\rm Mpc}`, which at :math:`z = 0.053` the ACT beam
    still passes at 1e-4, and :math:`w(1.9')` came out as -6.0e3 against a
    brute-force -- and positive -- 4.1e-7.  Float32 survived only because its
    floor makes the step ten times smaller.

    So every query whose cubic stencil -- nodes ``i-1 .. i+2`` of
    :func:`~ggah_mod.numerics.interp_cubic`'s interval -- touches a node with
    ``F <= 0`` takes the same :math:`C^1` cubic applied to ``F`` itself: no
    log, so no floor and no step, and the sign is kept.  Against the closed
    form of ``tests/test_transforms.py::TestASignChangingSpectrum`` it is good
    to 6e-4 on ``DEFAULT_WTHETA_ELL`` (linear in the value was 1e-2).  It is
    only ever applied next to a non-positive node.  **A spectrum positive
    everywhere selects the log-cubic at every query, so its result is
    unchanged bit for bit.**  Below the grid a
    stencil meeting a non-positive node takes the edge value, and above it the
    power-law tail is kept: its slope already falls back to ``slope_cap`` when
    the last nodes are not usable.
    """
    bad = ~(f > 0)
    n = log_x.shape[0]
    i = jnp.clip(jnp.searchsorted(log_x, log_q) - 1, 0, n - 2)
    hit = (bad[jnp.clip(i - 1, 0, n - 1)] | bad[i] | bad[i + 1]
           | bad[jnp.clip(i + 2, 0, n - 1)])
    inside = (log_q >= log_x[0]) & (log_q <= log_x[-1])
    lin = interp_cubic(log_q, log_x, f)
    out = jnp.where(inside & hit, lin, vals)
    below = (log_q < log_x[0]) & (bad[0] | bad[1])
    return jnp.where(below, f[0], out)


def make_hankel(order: float = 0.5, *, engine: str | None = None,
                backend=None, n: int | None = None,
                h: float | None = None) -> HankelRule:
    """Build an Ogata rule from the flavour's declared knobs.

    ``n`` is :attr:`~ggah_mod.backend.Backend.n_hankel` and ``h`` is
    :attr:`~ggah_mod.backend.Backend.hankel_h`.  They are **not** the same kind
    of knob: ``h`` sets the accuracy and ``n`` sets the reach.  See that field's
    docstring for the measured table.
    """
    from ..backend import resolve_backend

    b = resolve_backend(backend)
    engine = engine or b.hankel
    if engine != "quadrature":
        raise ValueError(
            f"`make_hankel` builds the Ogata rule; {engine!r} is built by "
            f"`make_fftlog`, which additionally needs the input grid because "
            f"its coefficients depend on the log spacing.  The transforms "
            f"dispatch on `Backend.hankel` and call the right one.")
    n = int(b.n_hankel if n is None else n)
    h = float(b.hankel_h if h is None else h)
    x, w = _ogata_nodes(float(order), n, h)
    return HankelRule(jnp.asarray(x), jnp.asarray(w), float(order), engine)


# =========================================================================
# The four transforms
# =========================================================================
#
# Each is `int dx x^p F(x) K(xr)`, and each engine gets the same three numbers:
# the kernel, the power, and the order.  The dispatch is one function, so a
# transform added later cannot pick up a different convention for one engine.

#: ``name -> (kernel, order, power, prefactor(r))``.
#:
#: ``power`` is the exponent in mcfit's normal form
#: ``G(y) = int F(x) K(xy) dx/x``, i.e. one more than the power of x in the
#: integral as usually written.  Stated here once because the two differ by one
#: and the difference is invisible in a shape comparison.
_TRANSFORMS = {
    "xi": ("spherical", 0.0, 3.0),
    "sigma": ("bessel", 0.0, 2.0),
    "delta_sigma": ("bessel", 2.0, 2.0),
    "wtheta": ("bessel", 0.0, 2.0),
}


def _transform(name, r, x_grid, f_grid, *, rule=None, backend=None,
               slope_cap: float = -3.0):
    """Evaluate one transform at ``r``, by whichever engine the flavour names."""
    from ..backend import resolve_backend

    kind, order, power = _TRANSFORMS[name]
    b = resolve_backend(backend)
    engine = b.hankel if rule is None else rule.engine
    r = jnp.atleast_1d(jnp.asarray(r))

    if engine == "quadrature":
        # The Ogata rule works in `t = x r`, where a spherical j_nu becomes
        # J_{nu+1/2} with an extra sqrt(pi/2t).  Folding that in here keeps the
        # two engines' entry points identical.
        if kind == "spherical":
            rule = rule or make_hankel(order + 0.5, backend=b)
            integral = hankel(rule, r, x_grid, f_grid, power=power - 1.5,
                              slope_cap=slope_cap)
            return jnp.sqrt(0.5 * jnp.pi) * integral / r ** power
        rule = rule or make_hankel(order, backend=b)
        integral = hankel(rule, r, x_grid, f_grid, power=power - 1.0,
                          slope_cap=slope_cap)
        # t = x r turns `int dx x^(p) F K(xr)` into `r^-(p+1) int dt t^p F K(t)`,
        # and `power` is p+1 -- the exponent is the same for both kernels.
        return integral / r ** power

    rule = rule or make_fftlog(order, x_grid, kind=kind, power=power,
                               backend=b)
    g = fftlog(rule, f_grid, slope_cap=slope_cap)
    # FFTLog answers on its own reciprocal grid; the caller's radii come from a
    # C1 interpolation of it, which is where a linear one would put the engine's
    # error straight back.
    #
    # Log in the *abscissa*, linear in the *value*.  Both halves matter: the
    # grid is log-spaced so the abscissa must be, and the value must not be,
    # because these outputs change sign.  `xi(r)` crosses zero near the baryon
    # scale and `DeltaSigma` can too; taking a log there gives a floor where
    # there should be a zero crossing, and the result looks like a feature.
    return interp_cubic(jnp.log(r), jnp.log(rule.y), g)


def pk_to_xi(r, k, pk, *, rule=None, backend=None):
    r""":math:`\xi(r) = \frac{1}{2\pi^2}\int dk\,k^2P(k)\,j_0(kr)`."""
    return _transform("xi", r, k, pk, rule=rule, backend=backend) \
        / (2.0 * jnp.pi ** 2)


def pk_to_sigma(rp, k, pk, rho_m, *, rule=None, backend=None):
    r""":math:`\Sigma(R) = \frac{\bar\rho_m}{2\pi}\int dk\,k\,P(k)\,J_0(kR)`.

    Comoving surface density, in :math:`(M_\odot/h)(\mathrm{Mpc}/h)^{-2}`.
    """
    return rho_m * _transform("sigma", rp, k, pk, rule=rule, backend=backend) \
        / (2.0 * jnp.pi)


def pk_to_delta_sigma(rp, k, pk, rho_m, *, rule=None, backend=None):
    r""":math:`\Delta\Sigma(R) = \frac{\bar\rho_m}{2\pi}\int dk\,k\,P(k)\,J_2(kR)`.

    **One transform, not four steps.**  :math:`J_2` *is* what
    :math:`\bar\Sigma(<R) - \Sigma(R)` encodes, so taking it directly removes
    the line-of-sight truncation, the interpolation onto
    :math:`\sqrt{R^2+\chi^2}`, and the cumulative integral from :math:`R = 0`
    that needs :math:`\Sigma` below the smallest tabulated radius -- three error
    sources the predecessor carried, one of which (a linear :math:`\chi` grid
    with :math:`\Delta\chi \gg R_{\min}`) it records as a ~10x overestimate.

    The real-space route is kept as an independent cross-check, not as a second
    production path.
    """
    return rho_m * _transform("delta_sigma", rp, k, pk, rule=rule,
                              backend=backend) / (2.0 * jnp.pi)


def cl_to_wtheta(theta, ell, cl, *, rule=None, backend=None):
    r""":math:`w(\theta) = \frac{1}{2\pi}\int d\ell\,\ell\,C_\ell\,J_0(\ell\theta)`."""
    return _transform("wtheta", theta, ell, cl, rule=rule, backend=backend) \
        / (2.0 * jnp.pi)


# =========================================================================
# The one special function with a traced argument
# =========================================================================

@functools.lru_cache(maxsize=4)
def _de_nodes(n: int, h: float, t_max: float = 40.0):
    r"""Double-exponential nodes on :math:`(0, \infty)`.  numpy, built once.

    Truncated at ``t_max``: past it ``cosh t`` overflows float64 while
    :math:`e^{-x\cosh t}` has been exactly zero for thirty decades, so the
    nodes there contribute nothing and cost a ``nan``.
    """
    k = np.arange(-n, n + 1, dtype=np.float64) * h
    # Clamped before the exponential, not after: `exp` of the unclamped
    # argument overflows to `inf` and warns, and the nodes it would have
    # produced are past `t_max` anyway.
    arg = np.minimum(0.5 * np.pi * np.sinh(k), np.log(t_max))
    t = np.exp(arg)
    w = 0.5 * np.pi * np.cosh(k) * t * h
    keep = np.isfinite(t) & np.isfinite(w) & (w > 0) & (t < t_max)
    return np.ascontiguousarray(t[keep]), np.ascontiguousarray(w[keep])


def _log_cosh(z):
    r""":math:`\ln\cosh z`, finite for every ``z``.

    :math:`|z| + \ln(1+e^{-2|z|}) - \ln 2`.  Written this way because
    ``log(cosh(z))`` overflows to ``inf`` at :math:`|z| \approx 710` and the
    quantity itself is only ~710 there.
    """
    a = jnp.abs(z)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - jnp.log(2.0)


def bessel_k(nu, x, *, n: int = 256, h: float = 0.02):
    r""":math:`K_\nu(x) = \int_0^\infty e^{-x\cosh t}\cosh(\nu t)\,dt`.

    The **one** Bessel function layer 5 evaluates at a traced argument, and the
    reason is the King PSF: :math:`B_\ell` for a King profile is
    :math:`2^{2-\alpha}\Gamma(\alpha-1)^{-1}(\ell\theta_c)^{\alpha-1}
    K_{\alpha-1}(\ell\theta_c)`, and :math:`\theta_c` is *fitted*.

    The integral representation rather than a rational approximation, because it
    is :math:`C^\infty` and differentiable in :math:`\nu` **as well as**
    :math:`x` -- so the King slope becomes a free parameter instead of a static
    choice.  The predecessor could do neither: it reached for
    ``scipy.special.kv``, which has no JAX path, and hard-coded the
    :math:`\alpha = 3/2` case where the answer is elementary.

    That elementary case, :math:`K_{1/2}(x) = \sqrt{\pi/2x}\,e^{-x}`, is the
    closed-form check.  Measured against it, ``h`` sets the accuracy and ``n``
    does not -- the same split as
    :attr:`~ggah_mod.backend.Backend.hankel_h`, and for the same reason:

    ====== ============ ==============
    ``h``  err at 1e-3  err at x >= 0.1
    ====== ============ ==============
    0.06   3.0e-3       1.7e-6
    0.03   3.4e-6       1.2e-10
    0.02   1.4e-8       machine
    ====== ============ ==============

    The small-argument end is the hard one, and it is the one the King PSF
    lives at: :math:`\ell\theta_c` is 4e-3 at :math:`\ell = 100` for an
    8.64-arcsecond core.

    **The whole integrand lives in the exponent**, and that is not tidiness.
    Written as ``exp(-x cosh t) * cosh(nu t)`` the two factors overflow and
    underflow at the same nodes, giving ``inf * 0 = nan`` -- and masking it
    afterwards with ``jnp.where(isfinite, val, 0)`` fixes the *value* while
    leaving the *gradient* ``nan``, because reverse-mode differentiates both
    branches of a ``where``.  That is a silent failure of exactly the kind this
    package hunts: the forward pass looks right and only ``jax.grad`` is wrong.
    """
    t, w = _de_nodes(int(n), float(h))
    t, w = jnp.asarray(t), jnp.asarray(w)
    x = jnp.atleast_1d(jnp.asarray(x))[:, None]
    nu = jnp.asarray(nu)
    log_val = -x * jnp.cosh(t)[None, :] + _log_cosh(nu * t)[None, :]
    return jnp.sum(jnp.exp(log_val) * w[None, :], axis=-1)
