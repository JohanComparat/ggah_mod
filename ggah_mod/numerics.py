"""Numeric helpers shared across the layers.

Small, dependency-free pieces that more than one layer needs.  They live here
rather than being written twice: each one below encodes a correctness detail
that is invisible in the output when it is got wrong, which is exactly the class
of thing this package exists to settle once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["log_grid", "lin_weights", "smoothstep", "soft_saturate", "hermite",
           "hermite_slopes", "interp_cubic", "invert_monotone",
           "require_x64", "N_BISECT"]


def require_x64(quantity: str, *, order: str, in_range: str = "") -> None:
    r"""Refuse a quantity single precision cannot hold, rather than return ``inf``.

    Some answers are simply out of float32's range.  :math:`L_X` is of order
    :math:`10^{44}` erg/s and its square :math:`10^{88}`, against a largest
    float32 of :math:`3.4\times10^{38}` -- unlike
    :meth:`~ggah_mod.sectors.gas.HotGasDPM.gas_mass`, where only an
    *intermediate* overflowed and regrouping the arithmetic was enough.  No
    arrangement of the integral returns these numbers in single precision, so
    the only honest options are to refuse or to change the units, and changing
    the units would make every published value mean something else.

    Silently returning ``inf`` is the failure this package exists to avoid, and
    it is a failure the test suite could not see on its own: ``tests/conftest.py``
    turns x64 on for every test, as does the paper's ``measure_accuracy.py``, so
    both ran in double precision while an ordinary ``import ggah_mod`` did not.
    ``tests/test_single_precision.py`` runs the checks in a subprocess with x64
    off, which is the only way to see it.

    It lives here rather than in one sector because two sectors need it: the hot
    gas for :math:`L_X`, and the AGN sector for the same luminosity as a point
    source and for its square.  Two copies would be two messages to keep true.

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
    r"""``n`` log-spaced points from ``lo`` to ``hi`` -- **numpy float64**.

    numpy and not ``jnp``, unconditionally, and that is the whole function.  A
    grid built with ``jnp.logspace`` inherits JAX's dtype configuration, so
    under the default (float32) the ``linspace`` and the ``power`` both happen
    *in* float32 and their errors compound: the shipped ``k`` grid's nodes are
    then misplaced in :math:`\ln k` by 5-8 eps32, against 1.0 eps32 for the same
    grid computed exactly and rounded once.  Six times the deviation, for nodes
    that are identical to float32's own resolution.

    That matters wherever the grid is a **transform abscissa**.
    :func:`~ggah_mod.observables.transforms.make_fftlog` measures the log
    spacing and then substitutes an ideal grid for it, so the caller's distance
    from that ideal *is* the sampling error.  It does not matter for a
    tabulation that is only interpolated off, and this is deliberately not used
    at those sites.

    Returns numpy rather than ``jnp`` because the dtype is the point: a caller
    needing Python floats that do not depend on the dtype active at import gets
    them from here, and one needing an array converts once, at its own
    boundary.  Routing through ``jnp`` first would round to float32 and hand
    back exactly the values this exists to avoid --- which is the bug
    :data:`~ggah_mod.observables.spec.DEFAULT_WTHETA_ELL` had.

    ``lo`` and ``hi`` are the **bounds themselves**, not their exponents.
    """
    return np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))


def lin_weights(x, grid):
    """Index and fraction for a clamped linear interpolation.

    The fraction is differentiable in ``x``; the index is not, which is what
    makes the gradient flow correctly through the value rather than the lookup.

    **The clamp is on x, not on the fraction, and uses `where` rather than
    `clip`.**  Both details are load-bearing, and getting either wrong halves a
    gradient silently:

    * clamping the fraction with ``jnp.clip(t, 0, 1)`` means that when ``x``
      lands exactly on a grid node the fraction is exactly 1.0 -- sitting on the
      clip boundary.  ``clip`` is ``minimum(maximum(...))``, and JAX splits the
      gradient of a ``minimum`` tie **50/50** between its arguments, so the
      derivative comes back exactly half its true value.
    * the same tie arises at the ends of the grid, where the clamp is genuinely
      needed; ``where`` selects a branch cleanly and carries the full gradient.

    This is not hypothetical.  The neutrino-ratio table's mass grid has a node
    at ``Sigma m_nu = 0.06`` eV, which is the fiducial, so every Fisher forecast
    differentiating the neutrino mass at the fiducial got a derivative wrong by
    a factor of two -- with no symptom other than the number.
    """
    lo, hi = grid[0], grid[-1]
    x_c = jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))
    i = jnp.clip(jnp.searchsorted(grid, x_c) - 1, 0, grid.size - 2)
    t = (x_c - grid[i]) / (grid[i + 1] - grid[i])
    return i, t


def smoothstep(x, lo, hi):
    r"""Hermite ramp from 0 at ``lo`` to 1 at ``hi``, :math:`t^2(3-2t)`.

    :math:`C^1` at both ends -- the derivative vanishes there rather than
    jumping -- which is what distinguishes it from a linear ramp when the result
    is differentiated.  Used to taper a tabulated correction to zero across the
    edge of its support instead of stepping it.
    """
    t = jnp.clip((jnp.asarray(x) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hermite(t, y0, y1, m0, m1, h):
    r"""Cubic Hermite basis on one interval, :math:`t \in [0,1]`.

    :math:`C^1` by construction.  One implementation, because there were two:
    :mod:`ggah_mod.sectors.cooling` needs it for the APEC table's temperature
    axis and :mod:`ggah_mod.observables.transforms` for the log-log
    interpolation of a spectrum, and this module exists for exactly that.
    """
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0
            + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)


def hermite_slopes(x, y):
    r"""Three-point weighted slopes for :func:`interp_cubic`, shape ``y.shape``.

    ``y`` may carry trailing axes -- a table of values on the grid ``x`` along
    its leading axis -- and gets one slope per element.

    .. math::  m_i = \frac{h_i\,\delta_{i-1} + h_{i-1}\,\delta_i}{h_{i-1}+h_i}

    the standard non-uniform-grid finite difference, one-sided at the ends.

    **Not** limited to be monotone, and that is deliberate here.  A
    Fritsch-Carlson limiter is the right choice for a *table* being made
    monotone -- :mod:`ggah_mod.sectors.cooling` uses one -- but it selects
    branches with ``min``/``max`` on values, and every such tie is a gradient
    split 50/50.  The quantity interpolated here is a traced spectrum with
    genuine curvature (the baryon wiggles are not monotone), so the limiter
    would be active at the fiducial rather than at an edge case.  Unlimited
    Hermite has no branch and therefore no tie.
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
    :math:`(\Delta x)^4` rather than :math:`(\Delta x)^2`.  Measured on a
    Gaussian transform pair over the shipped k grid, that is 8.9e-5 against
    linear and 1e-8 against this -- for the same grid and the same cost, since
    the expensive part is the ``searchsorted`` both share.

    Ends are clamped by index, exactly as :func:`lin_weights` clamps: the query,
    never the fraction.  Outside the grid the caller is expected to continue the
    function itself -- see
    :func:`~ggah_mod.observables.transforms._log_interp_with_tail`, where a
    power-law tail matters more than any interpolation scheme.
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
        Static.

    Notes
    -----
    **The Newton step is the point of this function.**  Without it the return
    value is identical and its derivative is exactly zero -- see the module
    docstring for what that cost the predecessor.

    ``jax.grad`` of the derivative estimate uses ``jax.grad(f)`` rather than a
    finite difference, so :math:`\partial f/\partial x` is exact and the polish
    is a true Newton step rather than a secant one.
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

    * exact in the unsaturated limit -- :math:`x \ll c` gives :math:`x` to
      :math:`O(x^2/c)`, so nothing is distorted where nothing is saturating;
    * no tie.  ``jnp.minimum`` at :math:`x = c` splits its gradient 50/50, and
      beyond it the derivative in ``x`` is zero.  Here it is
      :math:`e^{-x/c} > 0` everywhere;
    * :math:`C^\infty`, so second derivatives -- which a Fisher forecast takes
      -- exist too.

    The cost is that saturation is asymptotic rather than exact.  For a baryon
    fraction or an obscured fraction that is the physically sensible statement
    anyway: neither hits its ceiling and stops.

    Moved here from :mod:`ggah_mod.sectors.energetics`, which needed it first
    and is no longer its only consumer -- layer 3 acquiring an import cycle to
    share a four-line numeric device would be the wrong way round.
    """
    c = jnp.asarray(ceiling)
    return c * -jnp.expm1(-jnp.asarray(x) / c)
