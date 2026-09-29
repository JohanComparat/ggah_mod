r"""The two-, three- and four-halo trispectrum.

The halo model's trispectrum beyond one halo, in the parallelogram
configuration :math:`(\mathbf{k}_1, -\mathbf{k}_1, \mathbf{k}_2, -\mathbf{k}_2)`
and averaged over the angle between :math:`\mathbf{k}_1` and :math:`\mathbf{k}_2`
in their plane -- the average a projected statistic needs, and the one CCL's
``halomod_trispectrum_2h_22``, ``_2h_13``, ``_3h`` and ``_4h`` take (Takada &
Hu 2013; Cooray & Hu 2001).  With :math:`a, b` the legs at :math:`k_1`,
:math:`c, d` those at :math:`k_2`, and :math:`\langle\cdot\rangle` that average:

.. math::

    T^{2h}_{22} &= \langle P_{\rm L}(|\mathbf{k}_1+\mathbf{k}_2|)\rangle
        \big[I_{ac}I_{bd} + I_{ad}I_{bc}\big],\\
    T^{2h}_{13} &= P_{\rm L}(k_1)\big[I_a I_{bcd} + I_b I_{acd}\big]
        + P_{\rm L}(k_2)\big[I_c I_{dab} + I_d I_{cab}\big],\\
    T^{3h} &= \langle B^{\rm PT}\rangle
        \big[I_aI_cI_{bd} + I_aI_dI_{bc} + I_bI_cI_{ad} + I_bI_dI_{ac}\big],\\
    T^{4h} &= \langle T^{\rm PT}\rangle\,I_aI_bI_cI_d ,

with the bias-weighted moments

.. math::

    I_x(k) = \int dM\,\frac{dn}{dM}\,b\,W_x(k|M), \quad
    I_{xy}(k_1,k_2) = \int dM\,\frac{dn}{dM}\,b\,\mathcal{Q}_{xy}(k_1,k_2|M),

and :math:`I_{xyz}(k;k',k')` likewise.  :math:`I_x` is
:func:`~ggah_mod.spectra.pk.i_of_k`, low-mass counterterm and ``bias_weight``
included, so the two-halo term of every spectrum and of its covariance is the
same integral -- except for the neutrinos' linear leg
(:mod:`~ggah_mod.spectra.neutrinos`), which :math:`I_x` here does **not**
carry.  That is deliberate: the leg is a linear, fully correlated addition to
the two-halo *spectrum*, and its connected four-point function is second order
in :math:`f_\nu` on top of tree-level terms that are already sub-dominant --
0.9 per cent of an amplitude at the minimal mass enters the covariance at the
:math:`10^{-4}` level.  Carrying it here would need the same leg on every
moment :math:`I_{xy}`, which has no meaning: the neutrinos are in no halo.
:math:`\langle B^{\rm PT}\rangle` and
:math:`\langle T^{\rm PT}\rangle` are the tree-level bispectrum and trispectrum
of the linear spectrum the options select, in the forms of CCL
(Takada & Hu 2013, Eq. 30), with that spectrum read through a natural cubic
spline in log-log: a squeezed configuration cancels at fourth order in the
ratio of its wavenumbers, and what survives depends on the spectrum's curvature.

The mass below the grid
-----------------------

:math:`I_x` carries the counterterm's point mass at :math:`M_{\min}`, and
:math:`I_{xy}` and :math:`I_{xyz}` carry the same one: substituting
:math:`n \to n + A\,\delta_D(M - M_{\min})/[b\,M_{\min}/\bar\rho_{cb}]`
(Mead & Verde 2021, Eq. A4) into a moment of several legs gives
:math:`A(\bar\rho_{cb}/M_{\min})\prod W(k|M_{\min})`, which is CCL's
``_integrate_over_mbf``.  Only when every leg is continuous -- a discrete
tracer does not continue below the grid -- and with
``two_halo_consistency="linear_deficit"``.  A leg that declares
``w_unresolved`` is read at :math:`M_{\min}` through it.

Moments of several legs count distinct objects
----------------------------------------------

:math:`\mathcal{Q}_{xy}` and :math:`\mathcal{Q}_{xyz}` follow the rule of
:mod:`~ggah_mod.covariance.trispectrum`: every leg takes a point or an extended
part, and no two point legs of one discrete sample are kept.  CCL writes
:math:`I_{xyz}` as the product of a one- and a two-point moment, which counts a
central twice; for a continuous field the two agree.  The terms in which a
galaxy is shared between the two pairs across halos are not these, and are not
written.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..observables.transforms import _log_interp_with_tail, _safe_log
from ..spectra.counterterm import mass_deficit
from ..spectra.pair import _total, normalised_parts
from ..spectra.pk import i_of_k, linear_spectrum
from .trispectrum import _same_object_possible, _TAGS, interpolate_factor, legs

__all__ = ["pt_power_sum", "pt_bispectrum", "pt_trispectrum", "pt_kernels",
           "multi_halo_trispectrum"]


def _angles(n: int):
    """Gauss-Legendre on :math:`[0, \\pi]`, weights normalised to one."""
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(0.5 * np.pi * (x + 1.0)), jnp.asarray(0.5 * w)


def _spline_coefficients(x, y):
    r"""Second derivatives of the natural cubic spline through ``(x, y)``.

    :math:`C^2`, and that is the point.  The tree-level trispectrum of a
    squeezed configuration cancels at order :math:`(k_{\rm small}/k_{\rm
    large})^4`, and what survives depends on the curvature of
    :math:`\ln P`; the :math:`C^1` interpolation layer 5 uses elsewhere has a
    curvature that jumps at every node, which moved the four-halo term by up
    to 20% there.  One dense solve on the field's grid.
    """
    h = jnp.diff(x)
    n = x.size
    a = jnp.zeros((n, n))
    rhs = jnp.zeros(n)
    idx = jnp.arange(1, n - 1)
    a = a.at[0, 0].set(1.0).at[n - 1, n - 1].set(1.0)
    a = a.at[idx, idx - 1].set(h[:-1] / 6.0)
    a = a.at[idx, idx].set((h[:-1] + h[1:]) / 3.0)
    a = a.at[idx, idx + 1].set(h[1:] / 6.0)
    rhs = rhs.at[idx].set((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    return jnp.linalg.solve(a, rhs)


def _spline_eval(xq, x, y, m2):
    i = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, x.size - 2)
    h = x[i + 1] - x[i]
    t_hi = (xq - x[i]) / h
    t_lo = 1.0 - t_hi
    return (t_lo * y[i] + t_hi * y[i + 1]
            + ((t_lo ** 3 - t_lo) * m2[i] + (t_hi ** 3 - t_hi) * m2[i + 1])
            * h ** 2 / 6.0)


class _Spectrum:
    """:math:`P_{\rm L}` for the PT kernels: a natural cubic spline in
    log-log inside the grid, the layer-5 power-law continuation outside."""

    def __init__(self, log_k, log_p):
        self.log_k, self.log_p = log_k, log_p
        self.m2 = _spline_coefficients(log_k, log_p)

    def __call__(self, kq):
        lq = jnp.log(kq)
        inside = (lq >= self.log_k[0]) & (lq <= self.log_k[-1])
        spline = _spline_eval(jnp.clip(lq, self.log_k[0], self.log_k[-1]),
                              self.log_k, self.log_p, self.m2)
        tail = _log_interp_with_tail(lq, self.log_k, self.log_p, slope_cap=-3.0)
        return jnp.exp(jnp.where(inside, spline, tail))


def _power(kq, log_k, log_p):
    return log_p(kq) if isinstance(log_p, _Spectrum) else jnp.exp(
        _log_interp_with_tail(jnp.log(kq), log_k, log_p, slope_cap=-3.0))


def _kr_f2(k1, k2, cth):
    r""":math:`|\mathbf{k}_1+\mathbf{k}_2|` and CCL's :math:`F_2`, with
    ``kk = k2`` on the second axis and ``kp = k1`` on the first."""
    kk = k2[None, :]
    kp = k1[:, None]
    kr2 = kk ** 2 + kp ** 2 + 2.0 * kk * kp * cth
    safe = jnp.where(kr2 > 0, kr2, 1.0)
    f2 = 5.0 / 7.0 - 0.5 * (1.0 + kk ** 2 / safe) * (1.0 + kp / kk * cth) \
        + 2.0 / 7.0 * kk ** 2 / safe * (1.0 + kp / kk * cth) ** 2
    f2 = jnp.where(kr2 > 0, f2, 13.0 / 28.0)
    return jnp.sqrt(jnp.where(kr2 > 0, kr2, 0.0)), f2


def pt_power_sum(k1, k2, log_k, log_p, n_angle: int = 32):
    r""":math:`\langle P_{\rm L}(|\mathbf{k}_1+\mathbf{k}_2|)\rangle`,
    ``(N1, N2)``."""
    th, w = _angles(n_angle)

    def one(t):
        kr, _ = _kr_f2(k1, k2, jnp.cos(t))
        return _power(jnp.where(kr > 0, kr, jnp.exp(log_k[0])), log_k, log_p)

    return jnp.einsum("a,aij->ij", w, jax.vmap(one)(th))


def pt_bispectrum(k1, k2, log_k, log_p, n_angle: int = 32):
    r"""The averaged tree-level :math:`B` of the three-halo term, ``(N1, N2)``:
    CCL's ``6/7 P P + 2 P <P F_2>``, symmetrised."""
    th, w = _angles(n_angle)
    p1 = _power(k1, log_k, log_p)[:, None]
    p2 = _power(k2, log_k, log_p)[None, :]

    def one(t):
        kr, f2 = _kr_f2(k1, k2, jnp.cos(t))
        return _power(jnp.where(kr > 0, kr, jnp.exp(log_k[0])), log_k, log_p) * f2

    p3 = jnp.einsum("a,aij->ij", w, jax.vmap(one)(th))
    # CCL: Bpt = 6/7 pk pk.T + 2 pk P3, pk on the second axis; then + transpose.
    # Here the arrays are rectangular, so the transpose is the same expression
    # with the roles of k1 and k2 exchanged.
    b = 6.0 / 7.0 * p1 * p2 + 2.0 * p2 * p3

    def one_swapped(t):
        kr, f2 = _kr_f2(k2, k1, jnp.cos(t))
        return _power(jnp.where(kr > 0, kr, jnp.exp(log_k[0])), log_k, log_p) * f2

    p3s = jnp.einsum("a,aij->ij", w, jax.vmap(one_swapped)(th)).T
    return b + 6.0 / 7.0 * p1 * p2 + 2.0 * p1 * p3s


def pt_trispectrum(k1, k2, log_k, log_p, n_angle: int = 32):
    r"""The averaged tree-level :math:`T` of the four-halo term, ``(N1, N2)``,
    in CCL's ``t1113 + t1122`` form, symmetrised."""
    th, w = _angles(n_angle)

    def half(ka, kb):
        """CCL's expressions with ``kk = kb`` (second axis), ``kp = ka``."""
        pk = _power(kb, log_k, log_p)[None, :]
        pkt = _power(ka, log_k, log_p)[:, None]
        r = ka[:, None] / kb[None, :]

        def terms(t):
            cth = jnp.cos(t)
            kr, f2 = _kr_f2(ka, kb, cth)
            _, f2t = _kr_f2(kb, ka, cth)
            pkr = _power(jnp.where(kr > 0, kr, jnp.exp(log_k[0])), log_k, log_p)
            den = 1.0 + r ** 2 + 2.0 * r * cth
            safe = jnp.where(den > 0, den, 1.0)
            intd = (5.0 * r + (7.0 - 2.0 * r ** 2) * cth) / safe \
                * (3.0 / 7.0 * r + 0.5 * (1.0 + r ** 2) * cth
                   + 4.0 / 7.0 * r * cth ** 2)
            intd = jnp.where(den > 0, intd, 0.0)
            return pkr * f2 ** 2, pkr * f2 * f2t.T, intd

        p4a, p4x, xi = jax.vmap(terms)(th)
        p4a = jnp.einsum("a,aij->ij", w, p4a)
        p4x = jnp.einsum("a,aij->ij", w, p4x)
        x = -7.0 / 4.0 * (1.0 + r ** 2) + jnp.einsum("a,aij->ij", w, xi)
        t1113 = 4.0 / 9.0 * pk ** 2 * pkt * x
        t1122 = 8.0 * (pk ** 2 * p4a + pk * pkt * p4x)
        return t1113 + t1122

    return half(k1, k2) + half(k2, k1).T


# --------------------------------------------------------------------------
# bias-weighted moments
# --------------------------------------------------------------------------

class _Moments:
    """The :math:`I` integrals of one field, cached per tracer and wavenumber
    set."""

    def __init__(self, field, sectors, params, options, overlaps, cache):
        self.field = field
        self.sectors = sectors
        self.params = params
        self.options = options
        self.overlaps = overlaps
        self.legs_cache = cache
        self.w_b = field.dndm * field.bias * field.quadrature_measure()
        self.n_k = int(jnp.atleast_1d(field.k).shape[-1])
        self.point_mass = mass_deficit(field) * field.rho_cold / field.m[0]
        self._parts = {}
        self._one = {}

    def legs(self, name):
        key = name if isinstance(name, str) else id(name)
        if key not in self.legs_cache:
            self.legs_cache[key] = legs(self.field, name, self.sectors,
                                        self.params)
        return self.legs_cache[key]

    def parts(self, leg, label, kq):
        key = (id(leg[1]), label)
        if key not in self._parts:
            raw = dict(zip(_TAGS, normalised_parts(leg[1], self.n_k)))
            self._parts[key] = {t: None if v is None else
                                interpolate_factor(kq, self.field.k, v)
                                for t, v in raw.items()}
        return self._parts[key]

    def low_mass(self, leg, label, kq):
        r""":math:`W(k|M_{\min})` of one leg at ``kq``, or ``None``.

        ``None`` for a discrete leg and with the counterterm off, which is how
        the multi-leg completion is skipped.  A declared ``w_unresolved`` is
        per unit of the cold density, so it is taken back to one halo of mass
        :math:`M_{\min}` before it is read.
        """
        w = leg[1]
        if w.discrete or self.options.two_halo_consistency != "linear_deficit":
            return None
        key = (id(w), label, "low_mass")
        if key not in self._parts:
            if w.w_unresolved is not None:
                v = (jnp.broadcast_to(jnp.asarray(w.w_unresolved), (self.n_k,))
                     * self.field.m[0] / self.field.rho_cold)
            else:
                v = _total(*normalised_parts(w, self.n_k))[:, 0]
            lk = jnp.log(self.field.k)
            self._parts[key] = jnp.interp(jnp.log(kq), lk, v, left=v[0],
                                          right=0.0)
        return self._parts[key]

    def one(self, name, label, kq):
        key = (name if isinstance(name, str) else id(name), label)
        if key not in self._one:
            c = self.options.two_halo_consistency
            lk = jnp.log(self.field.k)
            total = sum(i_of_k(self.field, w, consistency=c)
                        for _, w in self.legs(name))
            self._one[key] = jnp.interp(jnp.log(kq), lk, total,
                                        left=total[0], right=0.0)
        return self._one[key]

    def two(self, x, y, lx, ly, kx, ky):
        """``I_xy``, x at ``kx`` (first axis), y at ``ky``."""
        out = jnp.zeros((kx.size, ky.size))
        for leg_x in self.legs(x):
            for leg_y in self.legs(y):
                px = self.parts(leg_x, lx, kx)
                py = self.parts(leg_y, ly, ky)
                same = _same_object_possible(leg_x, leg_y, self.overlaps)
                for tx in _TAGS:
                    for ty in _TAGS:
                        if px[tx] is None or py[ty] is None:
                            continue
                        if same and tx == ty == "point":
                            continue
                        out = out + (px[tx] * self.w_b) @ py[ty].T
                ox = self.low_mass(leg_x, lx, kx)
                oy = self.low_mass(leg_y, ly, ky)
                if ox is not None and oy is not None:
                    out = out + self.point_mass * jnp.outer(ox, oy)
        return out

    def three(self, x, y, z, lx, lyz, kx, kyz):
        """``I_xyz``, x at ``kx`` (first axis), y and z at ``kyz``."""
        out = jnp.zeros((kx.size, kyz.size))
        for leg_x in self.legs(x):
            for leg_y in self.legs(y):
                for leg_z in self.legs(z):
                    px = self.parts(leg_x, lx, kx)
                    py = self.parts(leg_y, lyz, kyz)
                    pz = self.parts(leg_z, lyz, kyz)
                    sxy = _same_object_possible(leg_x, leg_y, self.overlaps)
                    sxz = _same_object_possible(leg_x, leg_z, self.overlaps)
                    syz = _same_object_possible(leg_y, leg_z, self.overlaps)
                    for tx in _TAGS:
                        for ty in _TAGS:
                            for tz in _TAGS:
                                if px[tx] is None or py[ty] is None or pz[tz] is None:
                                    continue
                                if (sxy and tx == ty == "point") or \
                                        (sxz and tx == tz == "point") or \
                                        (syz and ty == tz == "point"):
                                    continue
                                out = out + (px[tx] * self.w_b) @ (py[ty] * pz[tz]).T
                    ox = self.low_mass(leg_x, lx, kx)
                    oy = self.low_mass(leg_y, lyz, kyz)
                    oz = self.low_mass(leg_z, lyz, kyz)
                    if ox is not None and oy is not None and oz is not None:
                        out = out + self.point_mass * jnp.outer(ox, oy * oz)
        return out


def pt_kernels(field, k1, k2, *, options, n_angle: int = 32,
               parts=("2h_22", "2h_13", "3h", "4h")):
    r"""The tracer-independent factors of :func:`multi_halo_trispectrum` on
    one pair of wavenumber sets: the averaged :math:`P_{\rm L}`, bispectrum
    and trispectrum.  Computed once per field and reused for every tracer
    quadruple, which is where the cost is."""
    k1 = jnp.atleast_1d(jnp.asarray(k1))
    k2 = jnp.atleast_1d(jnp.asarray(k2))
    log_k = jnp.log(field.k)
    spec = _Spectrum(log_k, _safe_log(linear_spectrum(field, options)))
    out = {"p1": spec(k1), "p2": spec(k2)}
    if "2h_22" in parts:
        out["power_sum"] = pt_power_sum(k1, k2, log_k, spec, n_angle)
    if "3h" in parts:
        out["bispectrum"] = pt_bispectrum(k1, k2, log_k, spec, n_angle)
    if "4h" in parts:
        out["trispectrum"] = pt_trispectrum(k1, k2, log_k, spec, n_angle)
    return out


def multi_halo_trispectrum(field, a, b, c, d, sectors: dict, params: dict,
                           k1, k2, *, options, overlaps=None, cache=None,
                           n_angle: int = 32, parts=("2h_22", "2h_13", "3h", "4h"),
                           pt=None):
    r""":math:`T^{2h}+T^{3h}+T^{4h}` of the tracers ``a, b`` at ``k1`` and
    ``c, d`` at ``k2``, shape ``(N1, N2)``.  ``parts`` selects terms, for
    inspection; the covariance takes all four.  ``pt`` is :func:`pt_kernels`
    on the same wavenumbers, when it has been computed already."""
    overlaps = overlaps or {}
    cache = {} if cache is None else cache
    k1 = jnp.atleast_1d(jnp.asarray(k1))
    k2 = jnp.atleast_1d(jnp.asarray(k2))
    m = _Moments(field, sectors, params, options, overlaps, cache)
    if pt is None:
        pt = pt_kernels(field, k1, k2, options=options, n_angle=n_angle,
                        parts=parts)
    out = jnp.zeros((k1.size, k2.size))

    ia, ib = m.one(a, 1, k1), m.one(b, 1, k1)
    ic, id_ = m.one(c, 2, k2), m.one(d, 2, k2)
    if "2h_22" in parts or "3h" in parts:
        iac, ibd = m.two(a, c, 1, 2, k1, k2), m.two(b, d, 1, 2, k1, k2)
        iad, ibc = m.two(a, d, 1, 2, k1, k2), m.two(b, c, 1, 2, k1, k2)
    if "2h_22" in parts:
        out = out + pt["power_sum"] * (iac * ibd + iad * ibc)
    if "2h_13" in parts:
        p1 = pt["p1"][:, None]
        p2 = pt["p2"][None, :]
        out = out + p1 * (ia[:, None] * m.three(b, c, d, 1, 2, k1, k2)
                          + ib[:, None] * m.three(a, c, d, 1, 2, k1, k2))
        out = out + p2 * (ic[None, :] * m.three(d, a, b, 2, 1, k2, k1).T
                          + id_[None, :] * m.three(c, a, b, 2, 1, k2, k1).T)
    if "3h" in parts:
        out = out + pt["bispectrum"] * (ia[:, None] * ic[None, :] * ibd
                           + ia[:, None] * id_[None, :] * ibc
                           + ib[:, None] * ic[None, :] * iad
                           + ib[:, None] * id_[None, :] * iac)
    if "4h" in parts:
        out = out + pt["trispectrum"] * (ia * ib)[:, None] * (ic * id_)[None, :]
    return out
