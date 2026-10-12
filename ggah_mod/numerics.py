"""Numeric helpers shared across the layers.

Small, dependency-free pieces that more than one layer needs, written once
rather than twice.  Each encodes a correctness detail that, got wrong, leaves no
trace in the output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["arctan", "log_grid", "lin_weights", "smoothstep", "soft_saturate",
           "hermite", "hermite_slopes", "interp_cubic", "invert_monotone",
           "require_x64", "N_BISECT"]


def require_x64(quantity: str, *, order: str, in_range: str = "") -> None:
    r"""Refuse a quantity single precision cannot hold, rather than return ``inf``.

    :math:`L_X` is of order :math:`10^{44}` erg/s and its square
    :math:`10^{88}`, against a largest float32 of :math:`3.4\times10^{38}`, so
    no arrangement of the integral returns them in single precision (in
    :meth:`~ggah_mod.sectors.gas.HotGasDPM.gas_mass`, by contrast, only an
    *intermediate* overflowed, and regrouping the arithmetic was enough).  We
    refuse rather than change the units, which would make every published value
    mean something else.

    ``tests/conftest.py`` turns x64 on for every test, as does the paper's
    ``measure_accuracy.py``, while an ordinary ``import ggah_mod`` does not, so
    ``tests/test_single_precision.py`` runs the checks in a subprocess with x64
    off.

    Two sectors call it: the hot gas for :math:`L_X`, and the AGN sector for
    the same luminosity as a point source and for its square.

    Parameters
    ----------
    quantity : str
        The name of the quantity being refused, for the message.
    order : str
        Its order of magnitude with units, e.g. ``"1e44 erg/s"``.  Stated by the
        caller because only the caller knows it.
    in_range : str
        What *is* computable in single precision nearby, so the message points
        somewhere rather than only refusing.  Optional.
    """
    if not jax.config.jax_enable_x64:
        tail = f"  {in_range}" if in_range else ""
        raise RuntimeError(
            f"{quantity} needs 64-bit JAX: it is of order {order} and the "
            f"largest float32 is 3.4e38, so in single precision it can only "
            f"come back as `inf`.  Enable x64 before the first JAX call -- "
            f"`jax.config.update(\"jax_enable_x64\", True)` or "
            f"`JAX_ENABLE_X64=1` in the environment.{tail}"
        )


def log_grid(lo, hi, n):
    r"""``n`` log-spaced points from ``lo`` to ``hi``, as a **numpy float64** array.

    ``lo`` and ``hi`` are the **bounds themselves**, not their exponents.

    Built in numpy whatever JAX's dtype configuration.  ``jnp.logspace`` under
    the default float32 runs both the ``linspace`` and the ``power`` in float32,
    and their errors compound: the shipped ``k`` grid's nodes are then misplaced
    in :math:`\ln k` by 5-8 eps32, six times the 1.0 eps32 of the same grid
    computed exactly and rounded once.

    That matters wherever the grid is a **transform abscissa**:
    :func:`~ggah_mod.observables.transforms.make_fftlog` measures the log
    spacing and substitutes an ideal grid for it, so the caller's distance from
    that ideal is the sampling error.  It does not matter for a table that is
    only interpolated, and those sites do not use this function.

    A caller needing Python floats independent of the dtype active at import
    gets them here, and one needing an array converts once, at its own
    boundary: routing the grid through ``jnp`` first would round it to float32
    again, as :data:`~ggah_mod.observables.spec.DEFAULT_WTHETA_ELL` once did.
    """
    return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))


def lin_weights(x, grid):
    """Index and fraction for a clamped linear interpolation.

    The fraction is differentiable in ``x`` and the index is not, so the
    gradient flows through the value rather than through the lookup.

    **The clamp is on x, not on the fraction, and uses ``where`` rather than
    ``clip``.**  Getting either wrong halves a gradient silently:

    * clamping the fraction with ``jnp.clip(t, 0, 1)`` puts it exactly on the
      clip boundary, at 1.0, whenever ``x`` lands on a grid node.  ``clip`` is
      ``minimum(maximum(...))``, and JAX splits the gradient of a ``minimum``
      tie **50/50** between its arguments, so the derivative comes back at
      exactly half its true value;
    * the same tie arises at the ends of the grid, where the clamp is needed;
      ``where`` selects one branch and carries the full gradient.

    A node at the fiducial is the ordinary case: the neutrino-ratio table's mass
    grid has one at ``Sigma m_nu = 0.06`` eV, where a clipped fraction gives
    every Fisher forecast in the neutrino mass a derivative wrong by a factor of
    two.
    """
    lo, hi = grid[0], grid[-1]
    x_c = jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))
    i = jnp.clip(jnp.searchsorted(grid, x_c) - 1, 0, grid.size - 2)
    t = (x_c - grid[i]) / (grid[i + 1] - grid[i])
    return i, t


def smoothstep(x, lo, hi):
    r"""Hermite ramp from 0 at ``lo`` to 1 at ``hi``, :math:`t^2(3-2t)`.

    :math:`t` is the position of ``x`` between ``lo`` and ``hi``, clipped to
    [0, 1].  :math:`C^1` at both ends, where the derivative vanishes and a
    linear ramp's jumps.  Used to taper a tabulated correction to zero across
    the edge of its support instead of stepping it.
    """
    t = jnp.clip((jnp.asarray(x) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hermite(t, y0, y1, m0, m1, h):
    r"""Cubic Hermite basis on one interval, :math:`t \in [0,1]`.

    ``y0`` and ``y1`` are the end values, ``m0`` and ``m1`` the end slopes and
    ``h`` the interval width.  :math:`C^1` by construction.  Two modules use
    it: :mod:`ggah_mod.sectors.cooling` for the APEC table's temperature axis
    and :mod:`ggah_mod.observables.transforms` for the log-log interpolation of
    a spectrum.
    """
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0
            + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)


def hermite_slopes(x, y):
    r"""Three-point weighted slopes for :func:`interp_cubic`, shape ``y.shape``.

    ``y`` may carry trailing axes (a table of values on the grid ``x`` along
    its leading axis) and gets one slope per element:

    .. math::  m_i = \frac{h_i\,\delta_{i-1} + h_{i-1}\,\delta_i}{h_{i-1}+h_i}

    the standard non-uniform-grid finite difference, with
    :math:`h_i = x_{i+1} - x_i` and :math:`\delta_i = (y_{i+1} - y_i)/h_i`,
    one-sided at the ends.

    The slopes are **not** limited to be monotone.  A Fritsch-Carlson limiter
    suits a *table* being made monotone (:mod:`ggah_mod.sectors.cooling` uses
    one), but it selects branches with ``min``/``max`` on values, and every such
    tie splits a gradient 50/50.  The quantity interpolated here is a traced
    spectrum with genuine curvature (the baryon wiggles are not monotone), so
    the limiter would be active at the fiducial; unlimited Hermite slopes have
    no branch and so no tie.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    # the grid is the leading axis; any trailing axes are carried along
    h = jnp.diff(x).reshape((-1,) + (1,) * (y.ndim - 1))
    d = jnp.diff(y, axis=0) / h
    interior = (h[1:] * d[:-1] + h[:-1] * d[1:]) / (h[:-1] + h[1:])
    return jnp.concatenate([d[:1], interior, d[-1:]])


def interp_cubic(x_query, x, y, slopes=None):
    r""":math:`C^1` cubic interpolation of ``y(x)`` at ``x_query``.

    A drop-in for :func:`jax.numpy.interp` whose error falls as
    :math:`(\Delta x)^4` rather than :math:`(\Delta x)^2`: on a Gaussian
    transform pair over the shipped k grid, 1e-8 against 8.9e-5 for linear
    interpolation, on the same grid and at the same cost, since the expensive
    part is the ``searchsorted`` both share.  ``slopes``, when given, replaces
    :func:`hermite_slopes`.

    Ends are clamped by index, exactly as :func:`lin_weights` clamps: the query,
    never the fraction.  Outside the grid the caller continues the function
    itself, as :func:`~ggah_mod.observables.transforms._log_interp_with_tail`
    does with a power-law tail.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    m = hermite_slopes(x, y) if slopes is None else jnp.asarray(slopes)
    h_all = jnp.diff(x)
    i = jnp.clip(jnp.searchsorted(x, x_query) - 1, 0, x.size - 2)
    h = h_all[i]
    t = (x_query - x[i]) / h
    return hermite(t, y[i], y[i + 1], m[i], m[i + 1], h)



#: Bisection steps before the Newton polish.  60 halves the bracket by 1e-18,
#: far below float64 resolution on a log axis -- and the Newton step supplies
#: the gradient, so extra iterations would buy nothing either way.  Static, so
#: the loop has a fixed trip count and never branches on a traced value.
N_BISECT = 60


def invert_monotone(f, y, lo, hi, n_steps: int = N_BISECT):
    r"""Solve :math:`f(x) = y` for a monotonically **increasing** ``f``.

    Parameters
    ----------
    f : callable
        Traceable, scalar-to-scalar, broadcasting over arrays, and increasing
        in ``x`` over ``[lo, hi]``.
    y : array
        Target value(s).
    lo, hi : float or array
        Bracket.  Fixed and data-independent, so the loop has no
        value-dependent control flow and jits without unrolling.
    n_steps : int
        Number of bisection steps; static.

    Notes
    -----
    We bracket the root by ``n_steps`` bisections under ``stop_gradient`` and
    take one Newton step.  The Newton step supplies the derivative: without it
    the return value is the same and its derivative is exactly zero.  It takes
    :math:`\partial f/\partial x` from ``jax.grad(f)``, not from a finite
    difference, so the derivative is exact and the step is a Newton step rather
    than a secant one.
    """
    y = jnp.asarray(y, dtype=float)
    shape = jnp.shape(y)

    def body(_, bounds):
        a, b = bounds
        mid = 0.5 * (a + b)
        go_right = f(mid) < y            # f increasing: root lies right
        return jnp.where(go_right, mid, a), jnp.where(go_right, b, mid)

    a = jnp.full(shape, lo, dtype=float)
    b = jnp.full(shape, hi, dtype=float)
    a, b = jax.lax.fori_loop(0, n_steps, body, (a, b))

    x0 = jax.lax.stop_gradient(0.5 * (a + b))
    # d f/d x on the bracketed point, exactly.  `f` is applied elementwise, so
    # the elementwise derivative is the diagonal of its Jacobian, which
    # `grad(sum(f))` gives in one pass without materialising the rest.
    df = jax.grad(lambda x: jnp.sum(f(x)))(x0)
    return x0 - (f(x0) - y) / df

def soft_saturate(x, ceiling):
    r"""Approach ``ceiling`` from below, smoothly:
    :math:`c\,[1 - e^{-x/c}]`.

    A drop-in for ``jnp.minimum(ceiling, x)`` wherever the minimum is a
    *physical* saturation rather than a domain guard, with three properties the
    hard version lacks:

    * exact in the unsaturated limit: :math:`x \ll c` gives :math:`x` to
      :math:`O(x^2/c)`;
    * no tie.  ``jnp.minimum`` at :math:`x = c` splits its gradient 50/50, and
      beyond it the derivative in ``x`` is zero.  Here it is
      :math:`e^{-x/c} > 0` everywhere;
    * :math:`C^\infty`, so second derivatives, which a Fisher forecast takes,
      exist too.

    Saturation is asymptotic rather than exact, which suits a baryon fraction or
    an obscured fraction: neither reaches its ceiling and stops.

    It lives here rather than in :mod:`ggah_mod.sectors.energetics`, its first
    user, so that the layer-3 modules sharing it do not import one another.
    """
    c = jnp.asarray(ceiling)
    return c * -jnp.expm1(-jnp.asarray(x) / c)


def arctan(x):
    r""":math:`\arctan x`, without the ``atan`` primitive.

    jaxlib 0.10.2, the newest that supports Python 3.11 and the one the shared
    development environment is held at, miscompiles ``jnp.arctan`` on a CPU
    with vector instructions, and returns finite, wrong numbers without
    raising: on 64 elements or more half the outputs come back zero and most of
    the rest are wrong, and inside a fused kernel it fails on a dozen (the
    Hernquist lensing profile was off by up to :math:`10^3`).
    ``jnp.arctan2(x, 1)`` lowers to the same thing and fails the same way.
    jaxlib 0.11 is correct.

    Built instead from ``arcsin`` and ``arccos``, each where it is well
    conditioned: :math:`\arcsin(x/\sqrt{1+x^2})` for :math:`|x| \le 1`, whose
    argument stays within :math:`\pm1/\sqrt2`, and
    :math:`{\rm sign}(x)\arccos(1/\sqrt{1+x^2})` beyond, whose argument stays
    below :math:`1/\sqrt2`.  Each branch is evaluated on an argument it is
    defined for, so neither leaks a ``nan`` gradient into the other.  Agrees
    with numpy to round-off, value and gradient; ``tests/test_numerics.py``
    pins both, and refuses ``jnp.arctan`` anywhere in the package.
    """
    x = jnp.asarray(x)
    big = jnp.abs(x) > 1.0
    x_in = jnp.where(big, 0.0, x)
    x_out = jnp.where(big, x, 2.0)
    inner = jnp.arcsin(x_in * jax.lax.rsqrt(1.0 + x_in * x_in))
    outer = jnp.sign(x_out) * jnp.arccos(jax.lax.rsqrt(1.0 + x_out * x_out))
    return jnp.where(big, outer, inner)
