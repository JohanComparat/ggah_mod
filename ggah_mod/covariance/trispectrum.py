r"""The one-halo trispectrum: the four-point rule, and the connected covariance.

.. math::

    T^{1h}_{abcd}(k_1, k_2) = \int dM\,\frac{dn}{dM}\,
        \mathcal{Q}_{abcd}(k_1, k_1, k_2, k_2\,|\,M)

is the parallelogram configuration :math:`(\mathbf{k}_1, -\mathbf{k}_1,
\mathbf{k}_2, -\mathbf{k}_2)` a power-spectrum covariance reads.  One halo's
profiles are isotropic, so the one-halo term depends on the two magnitudes
alone and needs no angular average.

The four-point rule
-------------------

:math:`\mathcal{Q}` is :func:`~ggah_mod.spectra.pair.pair_1h` with four legs.
Every leg takes either a tracer's point part or its extended part, and the
product of the four is kept unless two legs would be **the same object**:

* two point legs of one discrete sample -- a halo has one central, so
  :math:`\langle N_c(N_c-1)\rangle = 0`;
* nothing else -- satellites are Poisson about the central, so every factorial
  moment of theirs is the power of the mean, and legs of disjoint samples or of
  a continuous field are independent.

For the auto-trispectrum of one galaxy sample, with :math:`p` and :math:`e` the
normalised point and extended weights,

.. math::

    \mathcal{Q} = e_1^2e_2^2 + 2p_1e_1e_2^2 + 2e_1^2p_2e_2 ,

which is :math:`\langle N(N-1)(N-2)(N-3)\rangle` over the four distinct objects,
and it vanishes for a Bernoulli sample with nothing on a profile, as the pair
rule does.  For a continuous field it is the product of the four weights.

The **product of two pair rules**, which is what CCL's ``halomod_Tk3D_1h``
integrates for an HOD, has one more term, :math:`4p_1p_2e_1e_2`: the central
counted in both pairs.  That term belongs to the family below.

One shared object
-----------------

The rule counts four *distinct* objects.  A discrete sample also has the terms
in which the two pairs **share** one object -- a galaxy paired with one
neighbour in bin :math:`i` and with another in bin :math:`j` -- which are the
one-halo part of the shot-noise trispectrum of Lacasa (2018).  The shared
object sits at :math:`\mathbf{k}_1 + \mathbf{k}_2`, so its profile enters at
:math:`|\mathbf{k}_1 + \mathbf{k}_2|` and is not separable.  For projected
statistics the angular average is one Hankel pair (Graf's addition theorem),

.. math::

    \big\langle u(|\mathbf{k}_1+\mathbf{k}_2|)\big\rangle_\phi =
    \int d^2R\;\Sigma_u(R)\,J_0(k_1R)\,J_0(k_2R),

with :math:`\Sigma_u` the projected profile, so the term is a radial integral
of three real-space functions:

.. math::

    {\rm Cov}_{ij} = \frac{1}{V}\sum \int dM\,\frac{dn}{dM}\int d^2R\;
        \Sigma_m(R|M)\,A_i(R|M)\,B_j(R|M),

:math:`\Sigma_m` the shared object's weight profile and :math:`A_i` the bin
:math:`i` kernel applied to the other leg's profile about it
(:func:`shared_object_cng`).  Each factor is split into its value at the top of
the wavenumber grid, a point at the centre whose real-space kernel is known in
closed form, and a remainder that falls to zero there and is transformed
numerically -- so a central's :math:`\delta` and a bin's sharp edges never pass
through a truncated Bessel sum.

The shared object carries :math:`\langle N\rangle\ell^2` rather than
:math:`\langle N\rangle\ell`.  The contract separates the two only through
:attr:`~ggah_mod.sectors.protocol.TracerWeights.self_pair`, so the ratio
:math:`\ell = {\rm self\_pair}/(w_{\rm point} + w_{\rm ext})_{k\to0}` is
applied to both parts: exact for number counts and for a tracer with a single
part, and an average :math:`\ell` for a weighted tracer whose centrals and
satellites differ.

What is not here
----------------

The two- to four-halo trispectrum; the shared-object terms of a harmonic
block, whose angular average is three-dimensional; and the term in which the
two pairs share *both* objects, which is not a one-halo term at all but the
clustering of the pair
(:func:`~ggah_mod.covariance.gaussian.wp_noise_term`).

Nor is the one-halo transition of :mod:`~ggah_mod.spectra.transition` applied:
it is a fit to :math:`P(k)` and says nothing about the four-point function.

Projection
----------

Every factor of :math:`\mathcal{Q}` is a product of a function of :math:`k_1`
and one of :math:`k_2`, so the covariance of two binned statistics never forms
:math:`T` at all.  Each side is contracted with its statistic's kernel at every
halo mass, and the two are joined by one mass sum:

.. math::

    {\rm Cov}_{ij} = \frac{1}{V}\sum_\alpha \int dM\,\frac{dn}{dM}\,
        \big[K_i\cdot F_\alpha\big](M)\,\big[K_j\cdot G_\alpha\big](M) .
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special
import jax
import jax.numpy as jnp

from ..spectra.pair import normalised_parts
from ..spectra.spec import overlap_of
from ..spectra.tracers import build_weights, resolve

__all__ = ["legs", "side_factors", "one_halo_factors", "one_halo_trispectrum",
           "interpolate_factor", "contract_cng", "RealKernel", "real_kernel",
           "shared_object_cng"]

_TAGS = ("point", "extended")


def legs(field, tracer, sectors: dict, params: dict):
    """``((Component, TracerWeights), ...)`` for one tracer at one field."""
    t = resolve(tracer)
    return tuple((c, build_weights(c, field, sectors, params))
                 for c in t.components)


def _overlap(la, lb, overlaps):
    (ca, wa), (cb, wb) = la, lb
    key = (ca.population, cb.population)
    ov = overlaps.get(key) or overlaps.get(key[::-1]) or overlap_of(
        ca, cb, a_discrete=wa.discrete, b_discrete=wb.discrete)
    if ov == "nested":
        raise NotImplementedError(
            f"a nested pair of samples ({ca.label()!r} within {cb.label()!r} "
            f"or the reverse) in a four-point function: which objects are "
            f"shared is not in the contract, for the reason "
            f"`ggah_mod.spectra.pair.pair_1h` gives.")
    if ov not in ("identical", "none"):
        raise ValueError(f"unknown overlap {ov!r}")
    return ov


def _same_object_possible(la, lb, overlaps) -> bool:
    """Can a point leg of ``la`` and one of ``lb`` be the same object?"""
    return la[1].discrete and _overlap(la, lb, overlaps) == "identical"


def side_factors(la, lb, n_k: int, overlaps=None):
    r"""The legs at one wavenumber: ``[((tag_a, tag_b), array (Nk, NM))]``.

    Point-point is dropped for one discrete sample -- the pair rule -- and a
    part the tracer does not have (``None``) contributes no factor at all.
    """
    overlaps = overlaps or {}
    pa = dict(zip(_TAGS, normalised_parts(la[1], n_k)))
    pb = dict(zip(_TAGS, normalised_parts(lb[1], n_k)))
    same = _same_object_possible(la, lb, overlaps)
    out = []
    for ta in _TAGS:
        for tb in _TAGS:
            x, y = pa[ta], pb[tb]
            if x is None or y is None:
                continue
            if same and ta == tb == "point":
                continue
            out.append(((ta, tb), x * y))
    return out


def one_halo_factors(field, a, b, c, d, sectors: dict, params: dict, *,
                     overlaps=None, cache=None):
    r""":math:`\mathcal{Q}_{abcd}` as ``[(F, G)]``, each ``(Nk, NM)``, with
    :math:`\mathcal{Q}(k_1, k_2) = \sum F(k_1)\,G(k_2)`.

    ``a, b`` are the legs at :math:`k_1` and ``c, d`` those at :math:`k_2`.
    Tracers with several components are expanded over every component
    quadruple, as :func:`~ggah_mod.spectra.tracers.spectrum` expands a pair.
    ``cache`` maps a tracer name to its :func:`legs`, so weights shared by
    several blocks are built once.
    """
    overlaps = overlaps or {}
    cache = {} if cache is None else cache
    n_k = int(jnp.atleast_1d(field.k).shape[-1])

    def legs_of(name):
        key = name if isinstance(name, str) else id(name)
        if key not in cache:
            cache[key] = legs(field, name, sectors, params)
        return cache[key]

    out = []
    for la in legs_of(a):
        for lb in legs_of(b):
            left = side_factors(la, lb, n_k, overlaps)
            if not left:
                continue
            for lc in legs_of(c):
                for ld in legs_of(d):
                    right = side_factors(lc, ld, n_k, overlaps)
                    if not right:
                        continue
                    clash = [[_same_object_possible(x, y, overlaps)
                              for y in (lc, ld)] for x in (la, lb)]
                    for tags_l, f in left:
                        g = None
                        for tags_r, h in right:
                            if any(clash[i][j] and tags_l[i] == "point"
                                   and tags_r[j] == "point"
                                   for i in range(2) for j in range(2)):
                                continue
                            g = h if g is None else g + h
                        if g is not None:
                            out.append((f, g))
    return out


def one_halo_trispectrum(field, a, b, c, d, sectors: dict, params: dict, *,
                         overlaps=None):
    r""":math:`T^{1h}_{abcd}(k_1, k_2)` on the field's grid, shape ``(Nk, Nk)``.

    For inspection and for comparison with other codes; the covariance itself
    never forms this array.
    """
    w_m = field.dndm * field.quadrature_measure()
    t = jnp.zeros((field.k.shape[-1],) * 2)
    for f, g in one_halo_factors(field, a, b, c, d, sectors, params,
                                 overlaps=overlaps):
        t = t + (f * w_m) @ g.T
    return t


def interpolate_factor(k_query, k, factor):
    r"""A factor ``(Nk, NM)`` read at ``k_query`` (any shape), linear in
    :math:`\ln k`, shape ``k_query.shape + (NM,)``.

    Held at its first value below the grid, where a one-halo factor has already
    reached its :math:`k \to 0` limit, and zero above it, where the grid ends
    and no profile is known.
    """
    lk = jnp.log(jnp.asarray(k))
    lq = jnp.log(jnp.asarray(k_query))

    def col(v):
        return jnp.interp(lq, lk, v, left=v[0], right=0.0)

    return jnp.moveaxis(jax.vmap(col, in_axes=1)(factor), 0, -1)


def contract_cng(kernel_i, kernel_j, k_nodes, field, factors, volume):
    r"""The one-halo connected covariance of two binned statistics.

    Parameters
    ----------
    kernel_i, kernel_j : array (Ni, Nn), (Nj, Nn)
        Quadrature weights times the bin-averaged kernel at ``k_nodes``, so
        that a statistic is ``kernel @ P(k_nodes)``.
    k_nodes : array (Nn,)
    field : HaloField
    factors : list of (F, G)
        From :func:`one_halo_factors`.
    volume : float
        The volume (or solid angle, for a harmonic block) the estimator is
        averaged over.
    """
    w_m = field.dndm * field.quadrature_measure()
    cov = jnp.zeros((kernel_i.shape[0], kernel_j.shape[0]))
    for f, g in factors:
        a = kernel_i @ interpolate_factor(k_nodes, field.k, f)
        b = kernel_j @ interpolate_factor(k_nodes, field.k, g)
        cov = cov + (a * w_m) @ b.T
    return cov / volume


# --------------------------------------------------------------------------
# one shared object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RealKernel:
    r"""One binned projected statistic, in both spaces.

    ``matrix @ P(k)`` on the nodes ``k`` is the statistic, with ``matrix``
    holding the coefficient, the :math:`k\,dk/2\pi` weight ``k_weight`` and
    the bin-averaged :math:`\bar J_\nu`.  :meth:`real` is the same kernel acting on the
    projected correlation function, :math:`\int d^2R\,a_i(R)\,\xi(R)`.
    """

    k: np.ndarray
    k_weight: np.ndarray
    matrix: np.ndarray
    order: int
    edges: np.ndarray
    coefficient: float

    def real(self, r) -> np.ndarray:
        r"""``(n_bins, n_r)``.  Order 0 is the annulus mean,
        :math:`1_{A_i}(R)/A_i`; order 2 is :math:`\Delta\Sigma`'s
        disc-minus-ring, :math:`[2\ln(r_{\rm hi}/\max(R, r_{\rm lo}))
        \,H(r_{\rm hi}-R) - 1_{A_i}(R)]/A_i`."""
        r = np.asarray(r, float)[None, :]
        lo, hi = self.edges[:-1, None], self.edges[1:, None]
        area = np.pi * (hi ** 2 - lo ** 2)
        inside = ((r >= lo) & (r < hi)).astype(float)
        if self.order == 0:
            out = inside / area
        elif self.order == 2:
            out = (2.0 * np.log(hi / np.maximum(r, lo)) * (r < hi) - inside) / area
        else:
            raise ValueError(f"kernel order {self.order}")
        return self.coefficient * out


def real_kernel(quad, order: int, coefficient: float = 1.0) -> RealKernel:
    """A :class:`RealKernel` from a
    :class:`~ggah_mod.covariance.lensing.LensingQuadrature`."""
    jbar = {0: quad.j0, 2: quad.j2}[order]
    return RealKernel(np.asarray(quad.k), np.asarray(quad.weight),
                      coefficient * quad.weight * jbar, order,
                      np.asarray(quad.edges, float), float(coefficient))


def _radial_nodes(edges, r_max, r_floor, per_interval):
    """Gauss-Legendre in R with a break at every bin edge below ``r_max``;
    logarithmic below the first edge, where profiles are steep."""
    br = np.unique(np.concatenate([[r_floor], edges[edges < r_max], [r_max]]))
    x, w = np.polynomial.legendre.leggauss(per_interval)
    rs, ws = [], []
    for i, (a, b) in enumerate(zip(br[:-1], br[1:])):
        if i == 0:
            la, lb = np.log(a), np.log(b)
            rr = np.exp(0.5 * (lb - la) * (x + 1.0) + la)
            ww = 0.5 * (lb - la) * w * rr
        else:
            rr = 0.5 * (b - a) * (x + 1.0) + a
            ww = 0.5 * (b - a) * w
        rs.append(rr)
        ws.append(ww)
    r = np.concatenate(rs)
    return r, np.concatenate(ws) * 2.0 * np.pi * r


def shared_object_cng(field, ker_i: RealKernel, ker_j: RealKernel, a, b, c, d,
                      sectors: dict, params: dict, volume, *, overlaps=None,
                      cache=None, r_max: float = 10.0, r_floor: float = 1e-4,
                      per_interval: int = 12):
    r"""The one-halo covariance terms with one object in both pairs.

    ``a, b`` are the tracers of the statistic binned by ``ker_i``, ``c, d``
    those of ``ker_j``.  Every way one leg of each pair can be the same object
    is summed -- the same population, discrete -- with the other two legs
    distinct from it and from each other.  See the module docstring for the
    method; ``r_max`` [Mpc/h] bounds the shared object's distance from its
    halo centre, and must exceed the largest halo on the mass grid.
    """
    overlaps = overlaps or {}
    cache = {} if cache is None else cache
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    if not np.array_equal(ker_i.k, ker_j.k):
        raise ValueError("the two kernels must share their wavenumber nodes")
    kn = jnp.asarray(ker_i.k)
    edges = np.concatenate([ker_i.edges, ker_j.edges])
    r, w_r = _radial_nodes(edges, r_max, r_floor, per_interval)
    j0 = jnp.asarray(special.j0(np.outer(r, ker_i.k)))            # (NR, Nn)
    w_m = field.dndm * field.quadrature_measure()

    def legs_of(name):
        key = name if isinstance(name, str) else id(name)
        if key not in cache:
            cache[key] = legs(field, name, sectors, params)
        return cache[key]

    def parts(leg):
        return dict(zip(_TAGS, normalised_parts(leg[1], n_k)))

    split = {}

    def split_part(leg, tag):
        key = (id(leg[1]), tag)
        if key not in split:
            f = parts(leg)[tag]
            f_inf = f[-1]                                             # (NM,)
            f_nodes = interpolate_factor(kn, field.k, f)              # (Nn, NM)
            split[key] = (f_inf, f_nodes - f_inf)
        return split[key]

    side = {}

    def around(leg, tag, ker, label):
        """A(R|M) = kernel applied to the leg's profile about the shared
        object: (n_bins, NR, NM)."""
        key = (id(leg[1]), tag, label)
        if key not in side:
            f_inf, f_rest = split_part(leg, tag)
            numeric = jnp.einsum("bn,nm,rn->brm", jnp.asarray(ker.matrix),
                                 f_rest, j0)
            side[key] = numeric + jnp.asarray(ker.real(r))[:, :, None] * f_inf
        return side[key]

    def at_centre(leg, tag, ker):
        """A(0|M): (n_bins, NM)."""
        f_inf, f_rest = split_part(leg, tag)
        return jnp.asarray(ker.matrix) @ f_rest \
            + jnp.asarray(ker.real(np.zeros(1)))[:, 0, None] * f_inf

    nodes_weight = ker_i.k_weight

    def shared_profile(leg, tag):
        """(point part (NM,), projected remainder (NR, NM)), both times l/n."""
        sp = leg[1].self_pair
        if sp is None:
            raise ValueError(f"{leg[0].label()!r} is discrete but declares no "
                             f"self_pair, so the weight of a shared object is "
                             f"unknown")
        p0, e0 = normalised_parts(leg[1], n_k)
        total0 = sum(x[0] for x in (p0, e0) if x is not None) * leg[1].norm
        ell = jnp.where(total0 > 0, sp / jnp.where(total0 > 0, total0, 1.0), 0.0)
        f_inf, f_rest = split_part(leg, tag)
        scale = ell / leg[1].norm
        sigma = j0 @ (jnp.asarray(nodes_weight)[:, None] * f_rest)   # (NR, NM)
        return f_inf * scale, sigma * scale

    cov = jnp.zeros((ker_i.matrix.shape[0], ker_j.matrix.shape[0]))
    for la in legs_of(a):
        for lb in legs_of(b):
            for lc in legs_of(c):
                for ld in legs_of(d):
                    for x, xp in ((la, lb), (lb, la)):
                        for y, yp in ((lc, ld), (ld, lc)):
                            if not _same_object_possible(x, y, overlaps):
                                continue
                            if x[0] != y[0]:
                                raise NotImplementedError(
                                    f"{x[0].label()!r} and {y[0].label()!r} "
                                    f"are the same objects with different "
                                    f"weights; a shared object's weight needs "
                                    f"one per-object weight")
                            px, pxp, pyp = parts(x), parts(xp), parts(yp)
                            for tm in _TAGS:
                                if px[tm] is None:
                                    continue
                                m_inf, m_sig = shared_profile(x, tm)
                                for tx in _TAGS:
                                    if pxp[tx] is None or (
                                            tm == tx == "point"
                                            and _same_object_possible(x, xp, overlaps)):
                                        continue
                                    for ty in _TAGS:
                                        if pyp[ty] is None or (
                                                tm == ty == "point"
                                                and _same_object_possible(x, yp, overlaps)) or (
                                                tx == ty == "point"
                                                and _same_object_possible(xp, yp, overlaps)):
                                            continue
                                        ai = around(xp, tx, ker_i, "i")
                                        bj = around(yp, ty, ker_j, "j")
                                        a0 = at_centre(xp, tx, ker_i)
                                        b0 = at_centre(yp, ty, ker_j)
                                        point = (a0 * m_inf * w_m) @ b0.T
                                        spread = jnp.einsum(
                                            "r,rm,irm,jrm,m->ij", jnp.asarray(w_r),
                                            m_sig, ai, bj, w_m)
                                        cov = cov + point + spread
    return cov / volume

