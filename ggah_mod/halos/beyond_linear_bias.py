r"""Beyond-linear halo bias -- Mead & Verde (2021), arXiv:2011.08858.

The standard two-halo term assumes haloes are linearly biased tracers of the
linear field, :math:`P_{hh} = b_1 b_2 P_{\rm lin}`.  Measured against N-body it
is not, and the residual

.. math::

    P_{hh}(M_1, M_2, k) = b(M_1)\,b(M_2)\,P_{\rm lin}(k)
                          \bigl[1 + \beta^{\rm NL}(k, \nu_1, \nu_2)\bigr]

is tabulated here, from the MultiDark MDR1 simulation (2048^3 particles in a
1 Gpc/h box, Rockstar haloes on the virial criterion).

Redshift
--------
:math:`\beta^{\rm NL}` is **not** redshift-universal, even expressed in peak
height and divided by :math:`P_{\rm lin}`.  Mead & Verde say so directly
(Appendix B: the z = 0 function used at other redshifts "did not work as well
... hinting that the correction has a significant redshift dependence"), and the
tables bear it out -- at :math:`\nu = 2`, :math:`k = 0.74\,h\,{\rm Mpc}^{-1}`,
:math:`\beta^{\rm NL}` runs 0.37 at z = 0 to 1.32 at z = 1.28, a 4-34 sigma
shift against the published errors, while adjacent snapshots
(:math:`\Delta z = 0.014`) agree to :math:`\lesssim 1\sigma`.  The tabulated
:math:`\nu` support moves too, from [0.85, 3.71] at z = 0 to [1.33, 4.27] at
z = 1.  All **35** public snapshots are therefore shipped, a = 0.257 to 1.001.

Past either end of the sequence the table is **extrapolated linearly in the
MultiDark growth** ``g``, not clamped.  The case is common rather than exotic:
PLANCK18 at z = 0 matches MultiDark at g = 1.09, 0.09 past the last snapshot,
because the rescaling shortens its lengths (s = 0.86) and must raise the
amplitude to compensate.  Each element's slope is a least-squares line in g,
weighted by the published errors, over the last :data:`G_BASELINE` of the
sequence, and the line is anchored at the end snapshot so the table stays
continuous there.  The bins' peak heights move with it: the lowest bin is
MultiDark's fixed 512-particle mass, so its :math:`\nu` scales as 1/g, and the
bins above it keep their spacing.  Beyond :data:`G_REACH` the table is held,
and that, or the clamp, is what the boundary warning reports; carrying it within
the reach is silent, because PLANCK18 needs it at every z below 0.14.

Held out, the rule does what the clamp cannot.  Fitted on snapshots that end at
g = 0.907 and carried to the z = 0 snapshot (the same 0.09), the four
lowest-:math:`\nu` bins -- which carry the matter spectrum -- miss by 1.4, 0.6,
0.8 and 1.8 sigma rms over k > 0.08 h/Mpc, against 4.6, 1.8, 3.2 and 3.2 for
holding the anchor; the peak heights miss by 0.004 to 0.07, against 0.05 to
0.2 for a straight line.  The top two bins scatter by ten sigma between
adjacent snapshots whichever rule is used, so no rule is better there.
``table_at(..., extrapolate=False)`` restores the clamp, which is what the
reference implementation does.

Cosmology, and how it is absorbed
---------------------------------
The measurement is conditioned on MDR1's own WMAP5 cosmology (:data:`MDR1`), so
using it at another one is an approximation.  The correction applied here is the
Angulo & White (2010) rescaling, which is what the reference implementation
does by default (``bnl_method_rescale``): find the length scaling ``s`` and the
MultiDark scale factor ``a'`` whose :math:`\sigma(R)` best reproduces the
target's over :math:`R \in [1, 10]\,{\rm Mpc}/h`,

.. math::

    C(s, a') = \frac{1}{\ln(R_2/R_1)}\int_{\ln R_1}^{\ln R_2}
               \left[1 - \frac{\sigma_{\rm MD}(R/s,\,a')}
                              {\sigma_{\rm tgt}(R,\,a)}\right]^2 {\rm d}\ln R ,

then read the table at :math:`k_{\rm MD} = s\,k_{\rm tgt}` and blend the two
snapshots bracketing ``a'``.  This absorbs the *shape* of the mismatch -- n_s
and Omega_m, not only the amplitude.

Smooth in the cosmology
-----------------------
Until 0.9.7 every step of the rescaling was C^0 in the cosmology, and the
derivatives of everything downstream -- every spectrum, ``w_p`` and
:math:`\Delta\Sigma` -- had kinks on parameter scales of 1e-4 to 1e-3.
Autodiff returned the exact local slope; a central difference wider than the
kink spacing returned something else, by up to a factor of two for n_s on the
lensing rows (found by ggah_sens_study; switching this term off made every
cosmology derivative converge as :math:`h^2`).  Four readings, each now C^1:

* ``s`` was the parabola through three grid costs around the argmin, and its
  slope jumped 3-4 per cent whenever the argmin moved a cell -- at PLANCK18,
  z = 0.13, 2.9e-4 away in n_s, 2.0e-4 in Omega_m, 5.4e-4 in h and 1.5e-4 in
  Omega_b.  Now Newton steps on a C^2 cost (:data:`N_NEWTON`): ``s`` is the
  minimum to round-off, and on a target built from the table itself ``s =
  lambda`` to 2e-14 with ``ds/dlambda = 1`` to 7e-8 (the vertex: 7.9e-5 and
  2.4e-2).  ``ln sigma_MD`` is a C^2 spline for it.
* ``beta`` was read linearly in ln k at ``s * k``, so ``d ln P / d theta``
  stepped wherever that crossed a node; now a C^1 Hermite (:data:`K_INTERP`).
* the snapshots were blended linearly in ``g``, and joined the extrapolation
  with a kink; now a Hermite whose end slopes *are* the extrapolation's
  (:data:`G_INTERP`).
* the peak heights were projected with hat functions (:data:`NU_INTERP`); now
  Hermite weights, still a partition of unity with the same edge rules.

Each reading reproduces the tabulated nodes exactly.  Between them it reads
the noisy table differently from a straight line: at PLANCK18 P_gg moves by up
to 1.1 per cent at k = 0.14 h/Mpc, ``w_p`` and :math:`\Delta\Sigma` by 0.2 per
cent, ``s`` by 4.4e-5.  C^1 removes the steps, not the table's noise: the
derivatives still wiggle on the table's own scales (0.14-0.35 in ln k, ~0.01 in
``g``).  ``table_at(..., interp="linear")``, the ``*_INTERP`` switches and
``N_NEWTON = 0`` restore the reference's reading, which the port-parity suite
uses.

Support and edge treatment
--------------------------
25 wavenumbers, :math:`k \in [0.0063, 0.738]\,h\,{\rm Mpc}^{-1}`, and 8
peak-height bins per snapshot.  Outside that box:

* ``k`` below :data:`K_MIN_BNL` -- tapered smoothly to zero.  The reference
  applies the same cut as a hard step (``kmin_bnl``); the ramp keeps the
  correction continuous, which matters when it is differentiated.  The cut is
  independently justified by the published errors: below
  :math:`k \approx 0.02\,h\,{\rm Mpc}^{-1}` the measurement is consistent with
  zero (SNR 0.2-0.9), so the tabulated values there are noise, not a
  large-scale limit.
* ``k`` above the table -- clamped to the last tabulated value, matching the
  reference's ``fix_maximum``.  The one-halo term dominates there anyway.
* ``nu`` below the table -- linear extrapolation, the paper's preferred
  "extrapolated to low halo mass" variant and the reference's ``iextrap_lin``.
* ``nu`` above the table -- held constant.  The paper states results are
  unchanged whether :math:`\beta^{\rm NL}` is extrapolated above
  :math:`\nu \simeq 4` or fixed to zero, so the continuous choice is safe.
* ``nu`` outside :data:`NU_HARD` -- identically zero (``numin_bnl``/
  ``numax_bnl``).  Realistic haloes never reach either bound.

Measurement quality
-------------------
``sigma_beta`` carries the published 1-sigma errors: 0.5-2 per cent for
:math:`\nu \lesssim 2`, but 10-100 per cent for the top three bins
(:math:`\nu \gtrsim 2.9`), where the signal-to-noise on
:math:`\beta^{\rm NL}` itself is only 1-5.  ``nu_max_trust`` truncates that
corner; it is off by default, because dropping bins is a modelling choice and
because doing it carelessly reintroduces the very edge the rules above remove.

Data
----
``build()`` regenerates the table from the upstream repository.  Variants that
are public but not shipped: ``BNL_folded`` extends k to 1.48 h/Mpc,
``BNL_M200`` uses the M200 halo definition, and ``BNL_DQ`` is a Dark Quest
emulator version -- smooth, shot-noise-free, 64 wavenumbers out to k = 10 h/Mpc,
but only for z <= 1.445.
"""

from __future__ import annotations

import functools
import pathlib
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics import hermite, hermite_slopes, lin_weights, smoothstep

__all__ = ["load", "BetaNLTable", "MDR1", "MDR1_SIGMA8", "K_MIN_BNL", "NU_HARD",
           "R_RESCALE", "S_RANGE", "rescaling_cost", "match", "table_at",
           "beta_nl", "project_weights", "correction_2h_fused",
           "correction_2h_gg", "correction_2h_gm", "build", "G_BASELINE",
           "G_REACH", "N_NEWTON", "K_INTERP", "G_INTERP", "NU_INTERP"]

_NPZ = pathlib.Path(__file__).resolve().parent.parent / "data" / "bnl" / "mdr1_bnl.npz"
_URL = "https://raw.githubusercontent.com/alexander-mead/BNL/master/data"

#: Wavenumber [h/Mpc] below which beta^NL is taken to be zero (``kmin_bnl``).
K_MIN_BNL = 0.08
#: Half-width of the taper, as a factor on :data:`K_MIN_BNL`.
K_TAPER = 1.25
#: Hard peak-height bounds outside which beta^NL vanishes (``numin``/``numax``).
NU_HARD = (0.0, 10.0)
#: Radii [Mpc/h] over which the rescaling matches sigma(R) (``bnl_rescale_R1/R2``).
R_RESCALE = (1.0, 10.0)
#: Range of the length rescaling searched (``smin_rescale``/``smax_rescale``).
S_RANGE = (0.33, 3.00)
#: Points in the ``s`` search -- the reference's own resolution (``ns_rescale``).
N_S = 268
#: Quadrature nodes for the ln R integral in the cost.
N_R_COST = 48
#: Newton steps on the rescaling cost after the grid search (0.9.7).  0 is
#: 0.9.4's parabolic vertex alone, whose slope in the cosmology jumped each
#: time the grid argmin moved one cell -- see "Smooth in the cosmology" above.
N_NEWTON = 2
#: How ``beta`` is read between the tabulated wavenumbers and between the
#: snapshots: ``"cubic"`` (C^1 Hermite, 0.9.7) or ``"linear"`` (the reference's
#: reading, and 0.9.4's).  Module switches, read when a function is traced;
#: ``table_at(..., interp=)`` overrides both for one call.
K_INTERP = "cubic"
G_INTERP = "cubic"
#: How a peak height is projected onto the tabulated bins (:func:`project_weights`):
#: ``"cubic"`` (C^1 Hermite weights, 0.9.7) or ``"linear"`` (hat functions, the
#: reference's and 0.9.4's).
NU_INTERP = "cubic"
#: Width in g of the snapshots each end's extrapolation slope is fitted over.
#: 0.2 and 0.3 both beat the clamp by a factor of two to four on the lowest
#: bins in the hold-out of the module docstring; 0.1 is too few snapshots at
#: the z > 2.9 end, 0.45 starts to see the curvature.
G_BASELINE = 0.3
#: How far past either end of the sequence, in g, the table is carried before
#: it is held.  No further than the baseline its slope was measured over.
G_REACH = 0.3

#: MDR1's own cosmology (Mead & Verde 2021 sec. 3.2): WMAP5, flat LambdaCDM,
#: no massive neutrinos.  ``sigma_8`` is *not* a field of ``Cosmology`` -- it is
#: an output of this package -- so it lives separately and ``build()`` turns it
#: into an amplitude with ``ln10A_s_for_sigma8``.
#:
#: **The rescaling does not carry curvature, and cannot be asked to.**
#: Angulo & White (2010) match :math:`\sigma(R)` over
#: :data:`R_RESCALE`, which is a statement about the amplitude and shape of the
#: linear field and none at all about geometry.  So a curved target is matched
#: to a flat simulation on the one axis the method looks at, and the
#: :math:`\beta^{\rm NL}` table it returns stays MDR1's -- flat, WMAP5,
#: massless.  That is a scope statement rather than a defect: the correction is
#: a measured non-linear bias ratio, not a fitted surface in the cosmological
#: parameters, so there is no curvature axis it is silently extrapolating along.
MDR1 = dict(Omega_m=0.27, Omega_b=0.0469, h=0.70, n_s=0.95, sum_mnu=0.0)
MDR1_SIGMA8 = 0.82

#: The 35 snapshots carrying a beta^NL measurement, and their scale factors.
#: Both are hardcoded in the reference implementation (``hmx.f90``,
#: ``init_BNL``) as ``snaps``/``as_MD`` and agree with the ``MDR1_redshifts.csv``
#: shipped alongside the tables.
SNAPS = (36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 67,
         68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85)
A_MD = (0.257, 0.287, 0.318, 0.348, 0.378, 0.409, 0.439, 0.470, 0.500, 0.530,
        0.561, 0.591, 0.621, 0.652, 0.682, 0.713, 0.728, 0.743, 0.758, 0.773,
        0.788, 0.804, 0.819, 0.834, 0.849, 0.864, 0.880, 0.895, 0.910, 0.925,
        0.940, 0.956, 0.971, 0.986, 1.001)

_NBIN, _NK = 8, 25
_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """The distilled table, as a dict of ``jnp`` arrays.  Cached per process.

    Built under ``jax.ensure_compile_time_eval``: the cache outlives any trace,
    so a first call from inside a ``jit`` must still store concrete arrays.
    Without it the cache held that trace's tracers and every later call leaked
    them -- which the default-on beyond-linear term made the common path.
    """
    if not _NPZ.exists():
        raise FileNotFoundError(
            f"{_NPZ} is missing; regenerate it with\n"
            f"    python -m ggah_mod.halos.beyond_linear_bias")
    with np.load(_NPZ) as f, jax.ensure_compile_time_eval():
        return {k: jnp.asarray(f[k]) for k in f.files}


def _line_slope(x, y, var):
    """Weighted least-squares slope of ``y`` against ``x``, per element.

    ``x`` has shape ``(S,)``, ``y`` and ``var`` shape ``(S, ...)``; weights are
    ``1/var``.  numpy, float64: it runs once, on the table.
    """
    w = 1.0 / var
    ex = (-1,) + (1,) * (y.ndim - 1)
    xm = (w * x.reshape(ex)).sum(axis=0) / w.sum(axis=0)
    ym = (w * y).sum(axis=0) / w.sum(axis=0)
    dx = x.reshape(ex) - xm
    return (w * dx * (y - ym)).sum(axis=0) / (w * dx ** 2).sum(axis=0)


@functools.lru_cache(maxsize=1)
def _extrapolation() -> dict:
    r"""What the table needs past its ends: a slope and a peak-height anchor.

    For each end, ``slope_<end>`` is :math:`\partial\beta^{\rm NL}/\partial g`
    per element, shape ``(Nk_table, nbin, nbin)``: a least-squares line through
    the snapshots within :data:`G_BASELINE` of that end, weighted by the
    inverse published variance.  ``nu0_<end>`` is the lowest bin's peak height
    at the end snapshot and ``g_<end>`` its growth.  Computed once, in numpy,
    and stored as concrete arrays for the same reason :func:`load` is.
    """
    tab = load()
    g = np.asarray(tab["g_md"], dtype=np.float64)
    beta = np.asarray(tab["beta"], dtype=np.float64)
    # The error file's column is signed in places; its square is the variance.
    var = np.asarray(tab["sigma_beta"], dtype=np.float64) ** 2
    nu = np.asarray(tab["nu"], dtype=np.float64)
    out = {}
    with jax.ensure_compile_time_eval():
        for end, sel, i in (("hi", g >= g[-1] - G_BASELINE, -1),
                            ("lo", g <= g[0] + G_BASELINE, 0)):
            out[f"slope_{end}"] = jnp.asarray(
                _line_slope(g[sel], beta[sel], var[sel]))
            out[f"nu0_{end}"] = jnp.asarray(nu[i, 0])
            out[f"g_{end}"] = jnp.asarray(g[i])
    return out


def _extrapolate(grid, nu, g, nb):
    """``grid`` and ``nu``, blended at the clamped ``g``, carried to ``g``.

    Inside the sequence both corrections are exactly zero, so this is the
    identity there.  Branches are ``where``, not ``maximum``/``clip``: a tie at
    the boundary would otherwise split the gradient 50/50 (see
    :func:`~ggah_mod.numerics.lin_weights`).
    """
    ex = _extrapolation()
    lo, hi = ex["g_lo"], ex["g_hi"]
    gc = jnp.where(g < lo - G_REACH, lo - G_REACH,
                   jnp.where(g > hi + G_REACH, hi + G_REACH, g))
    up = jnp.where(gc > hi, gc - hi, 0.0)
    dn = jnp.where(gc < lo, gc - lo, 0.0)
    grid = grid + (up * ex["slope_hi"] + dn * ex["slope_lo"])[:, :nb, :nb]
    # The lowest bin is a fixed mass, so its nu goes as 1/g; the rest keep
    # their spacing, so the whole grid shifts with it.
    shift = (jnp.where(gc > hi, ex["nu0_hi"] * (hi / gc - 1.0), 0.0)
             + jnp.where(gc < lo, ex["nu0_lo"] * (lo / gc - 1.0), 0.0))
    return grid, nu + shift


class BetaNLTable(NamedTuple):
    r"""beta^NL resolved for one redshift and cosmology.

    ``beta`` is already interpolated onto the caller's ``k`` grid, rescaled and
    tapered, so a correction integral only has to contract it with weights
    projected onto ``nu``.

    Attributes
    ----------
    nu    : (nbin,)  peak heights of the tabulated bins, blended across ``a_md``
    beta  : (Nk, nbin, nbin)  on the caller's k grid
    s     : the AW10 length rescaling; ``k_table = s * k_target``
    a_md  : the matched MultiDark scale factor; past either end of the
            sequence, the end snapshot's (``g`` says how far past)
    w     : blend weight between snapshots ``i0`` and ``i1``
    cost  : the rescaling cost at the solution -- 0 is a perfect match
    g     : the matched MultiDark growth, unclamped; ``g_md`` spans the table
    """

    nu: jnp.ndarray
    beta: jnp.ndarray
    s: jnp.ndarray
    a_md: jnp.ndarray
    w: jnp.ndarray
    i0: jnp.ndarray
    i1: jnp.ndarray
    cost: jnp.ndarray
    g: jnp.ndarray = None


# ---------------------------------------------------------------------------
# The Angulo & White (2010) rescaling
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _ln_sigma_spline():
    """Not-a-knot cubic-spline coefficients of ln sigma_MD(ln R), per interval.

    Built once in float64 and stored concrete, for the reason :func:`load` is.
    The spline moves the values by ~1e-5 against the linear reading (256 nodes,
    Delta ln R = 0.02); what it changes is the *curvature*: a C^2 cost is what
    lets :func:`match` take Newton steps and the implicit-function derivative.
    """
    from scipy.interpolate import CubicSpline

    tab = load()
    x = np.asarray(tab["ln_r"], dtype=np.float64)
    c = CubicSpline(x, np.asarray(tab["ln_sigma_md"], dtype=np.float64),
                    bc_type="not-a-knot").c                         # (4, N-1)
    with jax.ensure_compile_time_eval():
        return jnp.asarray(c)


def _ln_sigma_md(ln_r, tab):
    """ln sigma_MD(R) at a = 1.001: a C^2 cubic spline in ln R (0.9.7).

    With :data:`N_NEWTON` = 0 the 0.9.4 reading, linear in ln R, is kept: the
    cost is then only C^0 in ``s``, with a kink wherever ``ln R - ln s`` crosses
    a node, which the parabolic vertex never looked at.  Both are clamped by
    ``lin_weights`` at the ends, which the cost's range never reaches.
    """
    i, t = lin_weights(ln_r, tab["ln_r"])
    if N_NEWTON == 0:
        y = tab["ln_sigma_md"]
        return (1.0 - t) * y[i] + t * y[i + 1]
    c = _ln_sigma_spline()
    dx = t * (tab["ln_r"][i + 1] - tab["ln_r"][i])
    return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]


def rescaling_cost(s, g, ln_r_nodes, ln_sigma_tgt, tab):
    r"""The AW10 cost :math:`C(s, a')`, with ``g`` the MDR1 growth at ``a'``.

    ``g`` enters only as a multiplicative amplitude on
    :math:`\sigma_{\rm MD}`, which is exact for MDR1: it is flat LambdaCDM with
    no massive neutrinos, so its linear growth is scale-independent and
    :math:`\sigma_{\rm MD}(R, a') = g(a')\,\sigma_{\rm MD}(R, 1)`.
    ``build()`` verifies that factorisation against the spectra it computes and
    records the residual in the table.
    """
    u = jnp.exp(_ln_sigma_md(ln_r_nodes - jnp.log(s), tab) - ln_sigma_tgt)
    return jnp.mean((1.0 - g * u) ** 2)


def _cost_and_best_g(s, ln_r_nodes, ln_sigma_tgt, tab):
    r"""Cost minimised over ``g`` in closed form, and the minimising ``g``.

    Because :math:`g` appears only linearly inside the square, the cost is a
    parabola in it and its minimum is exact:
    :math:`g^\star = \langle u\rangle / \langle u^2\rangle`, giving
    :math:`C = 1 - \langle u\rangle^2/\langle u^2\rangle`.  That collapses the
    reference's two-dimensional grid search to one dimension **without
    approximating its cost function** -- the same :math:`C`, minimised exactly
    in one of its two arguments rather than sampled on the 35 snapshots.

    ``g`` is deliberately *unconstrained* here.  Restricting it to the range the
    snapshot sequence spans would conflate two separate things: the length
    rescaling ``s`` is a statement about the shape of :math:`\sigma(R)` and
    does not care which snapshots exist, while the clamp is a statement about
    what the table can reach.  Clamping inside the fit biases ``s`` for any
    target sitting on the boundary -- which includes MDR1 itself at z = 0, the
    one case where the answer is known exactly.  The clamp belongs in
    :func:`match`, where ``g`` is mapped onto the snapshots.
    """
    u = jnp.exp(_ln_sigma_md(ln_r_nodes[None, :] - jnp.log(s)[:, None], tab)
                - ln_sigma_tgt[None, :])
    mu, mu2 = jnp.mean(u, axis=1), jnp.mean(u ** 2, axis=1)
    g_star = mu / mu2
    return 1.0 - mu ** 2 / mu2, g_star


def _parabolic_vertex(y_lo, y_mid, y_hi, x_mid, dx):
    """Vertex of the parabola through three equally spaced points.

    Refines a grid argmin to sub-cell precision and, unlike the argmin itself,
    is differentiable: the three ordinates carry the gradient while the index
    that selected them does not.  Degenerate (flat or non-convex) triples fall
    back to the grid point.
    """
    denom = y_lo - 2.0 * y_mid + y_hi
    shift = jnp.where(jnp.abs(denom) > 1e-30,
                      0.5 * (y_lo - y_hi) / jnp.where(denom == 0, 1.0, denom),
                      0.0)
    return x_mid + dx * jnp.clip(shift, -1.0, 1.0)


def _solve_s(ln_r_nodes, ln_sigma_tgt, tab):
    """The length rescaling for a target ``ln sigma(R)`` on ``ln_r_nodes``.

    The grid search, its parabolic vertex, and the Newton polish -- the part of
    :func:`match` that does not need a spectrum, so a target built from the
    table itself (``s`` known exactly) can be solved directly.
    """
    s_grid = jnp.linspace(S_RANGE[0], S_RANGE[1], N_S)
    cost, _ = _cost_and_best_g(s_grid, ln_r_nodes, ln_sigma_tgt, tab)
    j = jax.lax.stop_gradient(jnp.clip(jnp.argmin(cost), 1, N_S - 2))
    ds = s_grid[1] - s_grid[0]
    vertex = jnp.clip(_parabolic_vertex(cost[j - 1], cost[j], cost[j + 1],
                                        s_grid[j], ds), *S_RANGE)
    return _newton_polish(vertex, ds, ln_r_nodes, ln_sigma_tgt, tab)


def _newton_polish(vertex, ds, ln_r_nodes, ln_sigma_tgt, tab):
    r""":data:`N_NEWTON` Newton steps on :math:`C(s)` from the grid vertex.

    The seed is the vertex under ``stop_gradient`` and every step but the last
    is too, so the returned ``s`` carries the gradient of the last step alone:
    :math:`s = s_0 - C'(s_0)/C''(s_0)`, whose derivative at convergence is the
    implicit-function one, :math:`ds/d\theta = -\partial_\theta C'/C''` --
    the idiom of :func:`~ggah_mod.numerics.invert_monotone`.  The vertex is off
    by ~7e-5 in ``s``; one step leaves ~6e-9, two leave round-off, so ``s`` no
    longer remembers which grid cell the search stopped in.

    A step is taken only where the cost is convex and the step stays inside one
    grid cell; elsewhere the vertex is returned **with its own gradient**,
    never a silently zero one.
    """
    if N_NEWTON == 0:
        return vertex

    def cost_at(x):
        return _cost_and_best_g(jnp.atleast_1d(x), ln_r_nodes, ln_sigma_tgt, tab)[0][0]

    def step(x):
        c1, c2 = jax.jvp(jax.grad(cost_at), (x,), (jnp.ones_like(x),))
        ok = (c2 > 0) & (jnp.abs(c1) < c2 * ds)
        return x - jnp.where(ok, c1 / jnp.where(ok, c2, 1.0), 0.0), ok

    x = jax.lax.stop_gradient(vertex)
    for _ in range(N_NEWTON - 1):
        x = jax.lax.stop_gradient(step(x)[0])
    s_n, ok = step(x)
    inside = ok & (s_n > S_RANGE[0]) & (s_n < S_RANGE[1])
    return jnp.where(inside, s_n, vertex)


def match(k, pk_cb, return_g=False):
    r"""Solve the AW10 rescaling against a target spectrum.

    Returns ``(s, a_md, i0, i1, w, cost)``: the length rescaling, the matched
    MultiDark scale factor, the two snapshots bracketing it, the blend weight,
    and the cost at the solution.

    Parameters
    ----------
    k : array, shape (Nk,)
        Wavenumbers [h/Mpc] of the target spectrum, log-spaced.
    pk_cb : array, shape (Nk,)
        The target's **cold** spectrum on ``k``, at the redshift wanted.  The
        redshift enters only through this array -- the same convention as
        :func:`~ggah_mod.halos.variance.sigma_of_mass`, and for the same
        reason: a backend with ``has_native_z = False`` cannot be asked for a
        redshift, and this layer does not decide how one is obtained.

        Cold rather than total, matching what :math:`\sigma(M)` uses: the
        rescaling is matching the field haloes form from.  MDR1 has no massive
        neutrinos, so cold and total coincide on the other side.

    Notes
    -----
    The ``s`` search is the reference's own -- :data:`N_S` points across
    :data:`S_RANGE` -- refined past the grid by a parabolic vertex and then by
    :data:`N_NEWTON` Newton steps on the cost (:func:`_newton_polish`).  The
    index of the grid minimum is discrete and carries no gradient.  Until 0.9.7
    the vertex was the answer, and its slope in the cosmology jumped by 3-4 per
    cent each time the index moved one cell -- at PLANCK18 the fiducial sat
    2e-4 in Omega_m and 3e-4 in n_s from such a switch.  The Newton steps make
    ``s`` the cost's true minimum, so it is C^1 in whatever ``pk_cb`` is.
    """
    from ..cosmology.amplitude import sigma2_tophat

    tab = load()
    k = jnp.asarray(k)
    ln_r_nodes = jnp.linspace(jnp.log(R_RESCALE[0]), jnp.log(R_RESCALE[1]),
                              N_R_COST)
    ln_sigma_tgt = 0.5 * jnp.log(
        sigma2_tophat(jnp.asarray(pk_cb), k, jnp.exp(ln_r_nodes)))

    s = _solve_s(ln_r_nodes, ln_sigma_tgt, tab)

    # Re-evaluate at the refined s so g and the cost belong to the s returned.
    cost_s, g_s = _cost_and_best_g(jnp.atleast_1d(s), ln_r_nodes,
                                   ln_sigma_tgt, tab)
    g, cost_best = g_s[0], cost_s[0]

    # lin_weights clamps g to the tabulated range; table_at carries the table
    # past it, and _check_solution says so.
    i0, w = lin_weights(g, tab["g_md"])
    a_md = (1.0 - w) * tab["a"][i0] + w * tab["a"][i0 + 1]
    if return_g:
        return s, a_md, i0, i0 + 1, w, cost_best, g
    return s, a_md, i0, i0 + 1, w, cost_best


def _check_solution(s, g, cost, tab, extrapolate=True):
    """Warn once when the solution is pinned by a bound rather than found."""
    s, g, cost = float(s), float(g), float(cost)
    lo, hi = float(tab["g_md"][0]), float(tab["g_md"][-1])
    # Within G_REACH the extrapolation is the table's designed behaviour, not a
    # boundary: PLANCK18 is past the sequence for every z below 0.14, so a
    # warning there would fire on the default path of every low-redshift
    # analysis.  What is said is the clamp, which is a choice, and the reach,
    # past which the table stops following the target.
    past = lo - g if g < lo else g - hi
    if past > 0 and (not extrapolate or past > G_REACH):
        end = "below" if g < lo else "above"
        if not extrapolate:
            rule, how = "clamp", "clamped to the end snapshot"
        else:
            rule, how = "reach", (
                f"extrapolated to G_REACH = {G_REACH} past its end and held "
                f"there; the target is {past:.3f} past")
        _warn_once(f"bnl-g-{end}-{rule}", (
            f"beta^NL: the target amplitude sits {end} the MultiDark sequence "
            f"(g = {g:.4f}, range [{lo:.4f}, {hi:.4f}], z = {float(tab['z'][0]):.3f} "
            f"to {float(tab['z'][-1]):.3f}); the table is {how}."))
    if not S_RANGE[0] * 1.001 < s < S_RANGE[1] * 0.999:
        _warn_once("bnl-s-bound", (
            f"beta^NL: the AW10 rescaling hit its search bound (s = {s:.3f}, "
            f"range {S_RANGE}). The reference implementation treats this as an "
            "error; the requested cosmology is too far from MultiDark for the "
            "rescaling to be meaningful."))
    if cost > 0.01:
        _warn_once("bnl-cost", (
            f"beta^NL: the rescaling residual is large (C = {cost:.4f}, i.e. a "
            f"~{100 * cost ** 0.5:.1f}% rms sigma(R) mismatch over "
            f"R in {R_RESCALE} Mpc/h). beta^NL is being read from a simulation "
            "whose clustering does not resemble the requested cosmology."))


# ---------------------------------------------------------------------------
# Table evaluation
# ---------------------------------------------------------------------------

def _k_taper(k):
    """Smoothstep from 0 to 1 across the :data:`K_MIN_BNL` ramp, in log k."""
    lk = jnp.log(jnp.maximum(jnp.asarray(k), 1e-300))
    return smoothstep(lk, jnp.log(K_MIN_BNL / K_TAPER), jnp.log(K_MIN_BNL * K_TAPER))


def _interp_k(grid, k, ln_k_ref, interp=None):
    """``(Nk_ref, b, b)`` onto ``k`` in log k, with the k edge rules.

    ``k`` is clamped to the tabulated range at both ends -- the reference's
    ``fix_maximum`` -- and the low-k taper is applied afterwards, so the
    clamping at the bottom never shows: everything below the ramp is zero.

    ``interp`` (default :data:`K_INTERP`): ``"linear"`` is the reference's
    reading; ``"cubic"`` is a C^1 Hermite with unlimited three-point slopes
    (the ln k nodes are not uniform, 0.14-0.35 apart) and zero slope at both
    end nodes, so that the clamp is C^1 too.  The table is read at
    ``s * k_target``, so with the linear reading every ``d ln P/d theta`` had
    a step wherever ``s * k`` crossed a node (k = 0.10, 0.12, 0.15, 0.18, 0.22
    h/Mpc at PLANCK18).  Unlimited slopes are linear in the table, so the
    nodes are reproduced exactly and a limiter's branch switches cannot enter.
    """
    interp = K_INTERP if interp is None else interp
    k = jnp.asarray(k)
    i, t = lin_weights(jnp.log(jnp.maximum(k, 1e-300)), ln_k_ref)
    if interp == "linear":
        beta = (1.0 - t)[:, None, None] * grid[i] + t[:, None, None] * grid[i + 1]
    elif interp == "cubic":
        m = hermite_slopes(ln_k_ref, grid).at[0].set(0.0).at[-1].set(0.0)
        h = (ln_k_ref[i + 1] - ln_k_ref[i])[:, None, None]
        beta = hermite(t[:, None, None], grid[i], grid[i + 1], m[i], m[i + 1], h)
    else:
        raise ValueError(f"interp {interp!r}: 'linear' or 'cubic'")
    return beta * _k_taper(k)[:, None, None]


@functools.lru_cache(maxsize=1)
def _g_slopes() -> dict:
    r"""Hermite slopes of ``beta`` and ``nu`` along the snapshots, in ``g``.

    Three-point slopes on the non-uniform ``g_md`` grid, except at the two end
    snapshots, where the slope is the extrapolation's own
    (:func:`_extrapolation`): ``slope_<end>`` for ``beta`` and
    :math:`-\nu_0/g_{\rm end}` for ``nu`` (the derivative of the peak-height
    shift at the end).  So the blended table and its extrapolation join C^1.
    """
    tab = load()
    ex = _extrapolation()
    g = np.asarray(tab["g_md"], dtype=np.float64)
    beta = np.asarray(tab["beta"], dtype=np.float64)
    nu = np.asarray(tab["nu"], dtype=np.float64)
    with jax.ensure_compile_time_eval():
        mb = np.array(hermite_slopes(jnp.asarray(g), jnp.asarray(beta)))
        mn = np.array(hermite_slopes(jnp.asarray(g), jnp.asarray(nu)))
        mb[0], mb[-1] = np.asarray(ex["slope_lo"]), np.asarray(ex["slope_hi"])
        mn[0] = -float(ex["nu0_lo"]) / float(ex["g_lo"])
        mn[-1] = -float(ex["nu0_hi"]) / float(ex["g_hi"])
        return {"beta": jnp.asarray(mb), "nu": jnp.asarray(mn)}


def _blend_in_g(tab, i0, i1, w, nb, interp):
    """``(nu, beta)`` between snapshots ``i0`` and ``i1`` at fraction ``w``.

    ``"linear"`` is 0.9.4's blend, C^0 at every snapshot; ``"cubic"`` a Hermite
    with :func:`_g_slopes`.  A pinned snapshot has ``w = 0`` and returns the
    snapshot bit for bit either way (``i0 == i1`` gives ``h = 0``).
    """
    if interp == "linear":
        nu = ((1.0 - w) * tab["nu"][i0] + w * tab["nu"][i1])[:nb]
        grid = ((1.0 - w) * tab["beta"][i0] + w * tab["beta"][i1])[:, :nb, :nb]
        return nu, grid
    if interp != "cubic":
        raise ValueError(f"interp {interp!r}: 'linear' or 'cubic'")
    m = _g_slopes()
    h = tab["g_md"][i1] - tab["g_md"][i0]
    nu = hermite(w, tab["nu"][i0], tab["nu"][i1], m["nu"][i0], m["nu"][i1], h)[:nb]
    grid = hermite(w, tab["beta"][i0], tab["beta"][i1], m["beta"][i0],
                   m["beta"][i1], h)[:, :nb, :nb]
    return nu, grid


def _trusted_bins(tab, nu_max_trust) -> int:
    """Leading bins whose median fractional error is within ``nu_max_trust``."""
    if nu_max_trust is None:
        return tab["nu"].shape[1]
    diag = jnp.einsum("skbb->skb", tab["beta"])
    sig = jnp.einsum("skbb->skb", tab["sigma_beta"])
    med = jnp.median(sig / jnp.maximum(jnp.abs(1.0 + diag), 1e-30), axis=(0, 1))
    bad = np.flatnonzero(np.asarray(med) > nu_max_trust)
    n = int(bad[0]) if bad.size else int(med.size)
    if n < 2:
        raise ValueError(
            f"nu_max_trust={nu_max_trust} leaves {n} bins; at least 2 are "
            "needed to interpolate in nu")
    return n


def table_at(k_out, k=None, pk_cb=None, snap=None, nu_max_trust=None,
             check=True, extrapolate=True, interp=None) -> BetaNLTable:
    """Resolve beta^NL for one target spectrum, on the caller's ``k_out``.

    Parameters
    ----------
    k_out : array
        Wavenumbers [h/Mpc] the correction is wanted at.
    k, pk_cb : array
        The target's cold spectrum, at the redshift wanted.  See :func:`match`.
    snap : int, optional
        Pin a MultiDark snapshot and skip the rescaling entirely (``s = 1``) --
        the reproducible mode, and what the parity suite compares against
        ``hod_mod``.  ``k``/``pk_cb`` are then not needed.
    nu_max_trust : float, optional
        Discard tabulated bins whose median fractional error exceeds this,
        holding beta^NL constant above the last retained bin.  Off by default.
    check : bool
        Run the boundary diagnostics.  They read concrete floats, so they are
        skipped whenever the solution is traced -- which under ``jit`` it is
        even for a concrete spectrum, since every operation is staged.
    extrapolate : bool
        Carry the table linearly in ``g`` past the ends of the sequence (the
        module docstring's rule) rather than clamping it to the end snapshot,
        which is the reference's behaviour.  A pinned ``snap`` is never
        extrapolated.
    interp : {None, "linear", "cubic"}
        How ``beta`` is read between tabulated wavenumbers and between
        snapshots.  ``None`` takes :data:`K_INTERP` and :data:`G_INTERP`
        (``"cubic"``, C^1, since 0.9.7); ``"linear"`` is the reference's
        reading, which the port-parity suite asks for.
    """
    tab = load()
    nb = _trusted_bins(tab, nu_max_trust)

    if snap is not None:
        hits = np.flatnonzero(np.asarray(tab["snap"]) == int(snap))
        if hits.size == 0:
            raise ValueError(f"snapshot {snap} is not tabulated; available "
                             f"{int(tab['snap'][0])}..{int(tab['snap'][-1])}")
        i0 = int(hits[0])
        i1, w = min(i0 + 1, tab["a"].size - 1), jnp.asarray(0.0)
        s, a_md, cost = jnp.asarray(1.0), tab["a"][i0], jnp.asarray(0.0)
        g = tab["g_md"][i0]
    else:
        if k is None or pk_cb is None:
            raise ValueError(
                "table_at needs a target spectrum (k, pk_cb) to run the "
                "rescaling; pass snap=<n> to pin a snapshot instead")
        s, a_md, i0, i1, w, cost, g = match(k, pk_cb, return_g=True)
        traced = any(isinstance(x, jax.core.Tracer) for x in (s, g, cost))
        if check and not traced:
            _check_solution(s, g, cost, tab, extrapolate)

    k_interp = K_INTERP if interp is None else interp
    g_interp = G_INTERP if interp is None else interp
    nu, grid = _blend_in_g(tab, i0, i1, w, nb, g_interp)
    if snap is None and extrapolate:
        grid, nu = _extrapolate(grid, nu, g, nb)
    # k_table = s * k_target.  Fixed by construction, not by reading a comment:
    # a target whose lengths are all lambda times MultiDark's returns s =
    # lambda, so a target length is s MultiDark lengths and a target wavenumber
    # is k_MD/s.  tests/test_beyond_linear_bias.py pins it.
    beta = _interp_k(grid, s * jnp.asarray(k_out), tab["ln_k"], k_interp)
    # i0/i1 stay as they arrive -- a traced index under jit, a Python int for a
    # pinned snapshot.  Forcing int() here would make the whole table
    # un-jittable for the sake of a field nothing computes with.
    return BetaNLTable(nu=nu, beta=beta, s=s, a_md=a_md, w=w,
                       i0=i0, i1=i1, cost=cost, g=g)


def project_weights(nu, nu_ref, interp=None):
    r"""Interpolation weights from a fine ``nu`` grid onto the tabulated bins.

    Shape ``(N, nbin)``.  This is where the ``nu`` edge rule lives: the fraction
    is allowed to go **negative** below the first bin, which is linear
    extrapolation, and is capped at 1 above the last, which holds it constant.
    Outside :data:`NU_HARD` the row is zero.

    Deliberately *not* :func:`~ggah_mod.numerics.lin_weights`, whose clamp is
    the right rule for a table with no meaning outside its grid.  This table has
    one below its grid -- the paper's own preferred variant extrapolates
    :math:`\beta^{\rm NL}` to low halo mass -- so clamping here would silently
    truncate the low-mass end of every mass integral.

    ``interp`` (default :data:`NU_INTERP`): ``"linear"`` is the hat functions
    of the reference and of 0.9.4, whose weights are C^0 in ``nu``.  Both the
    field's peak heights and the bins' move with the cosmology, so every
    crossing of a mass node over a bin put a small kink into every cosmology
    derivative -- dense and each 1/n_m of the integral.  ``"cubic"`` (0.9.7)
    gives the weights of a C^1 Hermite with three-point slopes (linear in the
    tabulated values, so still a weight matrix), the one-sided secant at the
    first bin so that the linear extrapolation below joins C^1, and zero slope
    at the last so that the hold above does.
    """
    interp = NU_INTERP if interp is None else interp
    nu = jnp.atleast_1d(jnp.asarray(nu))
    nb = nu_ref.size
    i = jnp.clip(jnp.searchsorted(nu_ref, nu) - 1, 0, nb - 2)
    h = nu_ref[i + 1] - nu_ref[i]
    t = (nu - nu_ref[i]) / h
    rows = jnp.arange(nu.size)
    inside = (nu >= NU_HARD[0]) & (nu <= NU_HARD[1])
    if interp == "linear":
        t = jnp.minimum(t, 1.0)
        phi = jnp.zeros((nu.size, nb))
        phi = phi.at[rows, i].set(1.0 - t).at[rows, i + 1].add(t)
        return phi * inside[:, None]
    if interp != "cubic":
        raise ValueError(f"interp {interp!r}: 'linear' or 'cubic'")
    eye = jnp.eye(nb)
    S = hermite_slopes(nu_ref, eye).at[-1].set(0.0)      # m = S @ y, (nb, nb)
    below = t < 0.0                                       # only in the first bin
    tc = jnp.where(t > 1.0, 1.0, jnp.where(below, 0.0, t))
    t2, t3 = tc * tc, tc * tc * tc
    h00, h10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + tc
    h01, h11 = -2 * t3 + 3 * t2, t3 - t2
    cub = (h00[:, None] * eye[i] + h01[:, None] * eye[i + 1]
           + h[:, None] * (h10[:, None] * S[i] + h11[:, None] * S[i + 1]))
    lin = (1.0 - t)[:, None] * eye[i] + t[:, None] * eye[i + 1]
    phi = jnp.where(below[:, None], lin, cub)
    return phi * inside[:, None]


def beta_nl(k, nu1, nu2, table: BetaNLTable = None, **kw):
    """Interpolate beta^NL onto arbitrary ``(k, nu1, nu2)``, shape ``(Nk, N1, N2)``.

    ``table`` comes from :func:`table_at`; without one, ``kw`` is passed
    straight to it.
    """
    if table is None:
        table = table_at(k, **kw)
    return jnp.einsum("ia,kab,jb->kij",
                      project_weights(nu1, table.nu), table.beta,
                      project_weights(nu2, table.nu))


# ---------------------------------------------------------------------------
# Two-halo correction integrals
# ---------------------------------------------------------------------------

def correction_2h_fused(nu, w_a, w_b, table: BetaNLTable):
    r"""The correction for two **fused** weights, each shape ``(Nk, NM)``.

    .. math::

        \delta(k) = \sum_{ij} W_{a,i}(k)\,W_{b,j}(k)\,
                    \beta^{\rm NL}(k, \nu_i, \nu_j)

    with :math:`W_{a,i}(k)` already the whole integrand of :math:`I_a(k)` at
    node :math:`i` -- measure, mass function, bias and profile.  The two
    wrappers below factor it as ``weights[None, :] * uk``, which a
    scale-dependent bias cannot be written as: an ``(Nk, NM)`` bias makes
    ``weights`` two-dimensional and their contraction a shape error.  A
    fused weight has no such limit, and it is also where a point mass at a
    grid node -- the low-mass completion of
    :mod:`~ggah_mod.spectra.bnl` -- is simply an addition to one column.
    """
    phi = project_weights(nu, table.nu)
    wa = jnp.asarray(w_a) @ phi
    wb = wa if w_b is w_a else jnp.asarray(w_b) @ phi
    return jnp.einsum("ka,kab,kb->k", wa, table.beta, wb)


def correction_2h_gg(nu, weights, uk, table: BetaNLTable):
    r"""Additive correction to :math:`P_{uu}^{2h}(k)/P_{\rm lin}(k)`.

    .. math::

        \delta(k) = \sum_i \sum_j w_i(k)\,w_j(k)\,\beta^{\rm NL}(k, \nu_i, \nu_j)

    evaluated by projecting the weights onto the 8-bin grid first, which is
    exact -- the interpolation factorises over the two ``nu`` axes -- and turns
    an :math:`O(N_k N_M^2)` contraction into :math:`O(N_k N_M \cdot 8)`.

    Parameters
    ----------
    nu      : (NM,)      peak heights on the halo mass grid
    weights : (NM,)      e.g. ``dndm * N_tot * b / n_bar``, **including** the dm factor
    uk      : (Nk, NM)   the tracer's Fourier profile
    """
    w_eff = jnp.asarray(weights)[None, :] * jnp.asarray(uk)
    W = w_eff @ project_weights(nu, table.nu)
    return jnp.einsum("ka,kab,kb->k", W, table.beta, W)


def correction_2h_gm(nu, weights_a, weights_b, uk_a, uk_b, table: BetaNLTable):
    r"""The same for two different tracers -- the asymmetric double integral."""
    wa = jnp.asarray(weights_a)[None, :] * jnp.asarray(uk_a)
    wb = jnp.asarray(weights_b)[None, :] * jnp.asarray(uk_b)
    phi = project_weights(nu, table.nu)
    return jnp.einsum("ka,kab,kb->k", wa @ phi, table.beta, wb @ phi)


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------

#: Points in the stored ln R grid, and its span [Mpc/h].  The span must cover
#: R/s for every R in :data:`R_RESCALE` and every s in :data:`S_RANGE`, i.e.
#: [1/3.0, 10/0.33] = [0.33, 30.3], with a margin.
N_R_TABLE = 256
R_TABLE = (0.25, 40.0)


def _read_bnl(path, ncol):
    """A ``*_bnl.dat`` file as ``(k, [col, ...])`` in ``(n_k, nbin, nbin)`` layout.

    Upstream writes a triple loop over (bin, bin, k), so the flat table reshapes
    to ``(64, 25, ncol)`` and then to ``(8, 8, 25)``.  beta^NL is symmetric under
    exchange of its two mass arguments, so which bin axis is the outer loop does
    not matter.
    """
    raw = np.loadtxt(path)
    if raw.shape != (_NBIN * _NBIN * _NK, ncol):
        raise ValueError(f"{path.name}: expected "
                         f"{(_NBIN * _NBIN * _NK, ncol)}, got {raw.shape}")
    raw = raw.reshape(_NBIN * _NBIN, _NK, ncol)
    cols = [raw[:, :, c].reshape(_NBIN, _NBIN, _NK).transpose(2, 0, 1)
            for c in range(1, ncol)]
    return raw[0, :, 0], cols


def _fetch(rel, cache: pathlib.Path) -> pathlib.Path:
    """``rel`` from the cache, downloaded from the upstream repository if absent."""
    dest = cache / rel
    if not (dest.exists() and dest.stat().st_size):
        _download(f"{_URL}/{rel}", dest)
    return dest


def _download(url: str, dest: pathlib.Path) -> None:  # pragma: no cover
    """The one step of :func:`build` no test takes, because it needs the
    network; everything after it is tested on a synthetic cache."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())


def build(out=None, cache=None, pk_name: str = "camb"):
    """Regenerate the distilled table from the upstream ascii.

    Downloads the 35 snapshot pairs and their error tables, and computes MDR1's
    own :math:`\\sigma(R, a)` -- which the rescaling needs and which no upstream
    file carries.  MDR1 is built as a first-class
    :class:`~ggah_mod.cosmology.parameters.Cosmology`: its
    :math:`\\sigma_8 = 0.82` is turned into an amplitude once, up front, with
    :func:`~ggah_mod.cosmology.amplitude.ln10A_s_for_sigma8`, because
    :math:`\\sigma_8` is an output of this package and never an input.

    The growth factorisation the runtime cost relies on --
    :math:`\\sigma_{\\rm MD}(R, a) = g(a)\\,\\sigma_{\\rm MD}(R, 1)` -- is
    **verified** here against the spectra actually computed, and the residual is
    stored as ``g_residual`` rather than asserted.
    """
    import tempfile

    from ..cosmology.amplitude import ln10A_s_for_sigma8, sigma_tophat
    from ..cosmology.parameters import Cosmology
    from ..cosmology.power import make_pk

    out = _NPZ if out is None else pathlib.Path(out)
    cache = pathlib.Path(tempfile.mkdtemp(prefix="bnl-")) if cache is None \
        else pathlib.Path(cache)

    # -- the tabulated beta^NL ------------------------------------------
    nu = np.empty((len(SNAPS), _NBIN))
    bias = np.empty_like(nu)
    log10m = np.empty_like(nu)
    beta = np.empty((len(SNAPS), _NK, _NBIN, _NBIN))
    sigma_beta = np.empty_like(beta)
    k_ref = None

    for i, snap in enumerate(SNAPS):
        stem = f"MDR1_rockstar_{snap}"
        # binstats columns: log10 m_min, log10 m_max, log10 m,
        #                   nu_min, nu_max, nu, b, r_v
        bs = np.loadtxt(_fetch(f"BNL/M512/{stem}_binstats.dat", cache))
        log10m[i], nu[i], bias[i] = bs[:, 2], bs[:, 5], bs[:, 6]

        k, (one_plus,) = _read_bnl(_fetch(f"BNL/M512/{stem}_bnl.dat", cache), 2)
        beta[i] = one_plus - 1.0
        k_err, (one_plus_e, sig) = _read_bnl(
            _fetch(f"BNL_errors/M512/{stem}_bnl.dat", cache), 3)
        sigma_beta[i] = sig

        # The two variants are the same measurement written at different
        # precision: they differ by a fixed ~3.6e-6 relative offset on 384 of
        # the 1600 rows, five orders of magnitude below sigma.  The tolerance
        # is set to catch a genuinely different measurement, not that.
        if not np.allclose(one_plus, one_plus_e, rtol=1e-4):
            raise ValueError(f"{stem}: BNL and BNL_errors disagree on beta^NL")
        if k_ref is None:
            k_ref = k
        elif not (np.allclose(k, k_ref) and np.allclose(k_err, k_ref)):
            raise ValueError(f"{stem}: k grid differs from snapshot {SNAPS[0]}")

    if not np.all(np.diff(nu, axis=1) > 0):
        raise ValueError("nu bins are not strictly increasing in every snapshot")

    # -- MDR1's own sigma(R, a) -----------------------------------------
    pk = make_pk(pk_name)
    mdr1 = Cosmology.create(**MDR1)
    mdr1 = mdr1.replace(ln10A_s=ln10A_s_for_sigma8(MDR1_SIGMA8, mdr1, pk))

    k_sig = np.logspace(-4.0, 2.0, 512)
    r_tab = np.logspace(np.log10(R_TABLE[0]), np.log10(R_TABLE[1]), N_R_TABLE)
    # MultiDark records the z = 0 snapshot's expansion factor as 1.001 -- an
    # output-rounding artefact; its own redshift table lists that snapshot at
    # zred = 0.  Taking 1/a - 1 literally therefore asks for a *negative*
    # redshift, which a Boltzmann solver refuses (CAMB segfaults on it).
    #
    # Clamping only the redshift would leave a 0.1% inconsistency between the
    # scale factor the table reports and the one its sigma was evaluated at,
    # worth ~0.5% in s at exactly z = 0.  So the working scale factor is
    # derived from the clamped redshift, and the raw expansion factors are kept
    # alongside as provenance -- they are what the reference hardcodes.
    z_md = np.maximum(0.0, 1.0 / np.asarray(A_MD) - 1.0)
    a_md = 1.0 / (1.0 + z_md)
    # MDR1 has no massive neutrinos, so the cold and total spectra coincide.
    pk_md = np.asarray(pk.pk(k_sig, z_md, mdr1))                # (35, n_k)
    sig = np.stack([np.asarray(sigma_tophat(pk_md[i], k_sig, r_tab))
                    for i in range(len(A_MD))])                 # (35, N_R)

    # g(a) = sigma(R, a)/sigma(R, a_max), which is R-independent exactly when
    # growth is scale-independent.  Verify rather than assume.
    g_of_r = sig / sig[-1][None, :]
    g_md = g_of_r.mean(axis=1)
    g_residual = float(np.max(np.abs(g_of_r / g_md[:, None] - 1.0)))
    if g_residual > 1e-3:
        raise ValueError(
            "sigma_MD(R, a) does not factorise into g(a) sigma_MD(R) to 0.1% "
            f"(max residual {g_residual:.2e}); the runtime cost function "
            "assumes it does")
    if not np.all(np.diff(g_md) > 0):
        raise ValueError("g(a) is not strictly increasing")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        snap=np.asarray(SNAPS, dtype=np.int16),
        a=a_md,
        a_raw=np.asarray(A_MD, dtype=np.float64),
        z=z_md,
        k=k_ref.astype(np.float64),
        ln_k=np.log(k_ref).astype(np.float64),
        nu=nu, bias=bias, log10m=log10m,
        # float64 so a pinned snapshot stays bit-exact against the upstream
        # ascii; sigma_beta is a diagnostic that never enters a prediction.
        beta=beta.astype(np.float64),
        sigma_beta=sigma_beta.astype(np.float32),
        ln_r=np.log(r_tab),
        ln_sigma_md=np.log(sig[-1]),
        g_md=g_md,
        g_residual=np.asarray(g_residual),
        mdr1_ln10A_s=np.asarray(float(mdr1.ln10A_s)),
        mdr1_Omega_m=np.asarray(MDR1["Omega_m"]),
        mdr1_Omega_b=np.asarray(MDR1["Omega_b"]),
        mdr1_h=np.asarray(MDR1["h"]),
        mdr1_n_s=np.asarray(MDR1["n_s"]),
        mdr1_sigma8=np.asarray(MDR1_SIGMA8),
    )
    load.cache_clear()
    _extrapolation.cache_clear()
    return out


if __name__ == "__main__":  # pragma: no cover
    print("wrote", build())
