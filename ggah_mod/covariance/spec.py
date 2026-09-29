r"""From a data vector's spec to its covariance matrix.

:func:`covariance` takes the same
:class:`~ggah_mod.observables.spec.ObservableSpec` a model is compiled from and
returns ``params -> Cov``, laid out in the spec's own order, so a data vector and
its covariance cannot disagree about which row is which.

What a covariance needs that a model does not
---------------------------------------------

The covariance of :math:`C^{AB}` and :math:`C^{CD}` needs :math:`C^{AC}`,
:math:`C^{BD}`, :math:`C^{AD}` and :math:`C^{BC}`, most of which no statistic
asks for.  They are added to the spec as auxiliary statistics and compiled
through layer 5's own :func:`~ggah_mod.observables.spec.make_model`, so every
spectrum a covariance reads is resolved, de-duplicated and projected exactly as
a data-vector spectrum is -- on the same grid, with the same kernels and beams.
A map is a ``(tracer, kernel, beam)`` triple, because a beam is part of what
was measured.

:math:`w_p` and :math:`\Delta\Sigma` need 3D spectra, read from layer 4 at the
statistic's redshift; the line-of-sight convergence behind a :math:`\Delta\Sigma`
is a :math:`C_\ell` and goes through layer 5 like any other.

Terms
-----

``"gaussian"``
    The disconnected term with declared noise
    (:mod:`~ggah_mod.covariance.gaussian`, :mod:`~ggah_mod.covariance.lensing`).
``"one_halo"``
    The one-halo trispectrum of four distinct objects
    (:mod:`~ggah_mod.covariance.trispectrum`).  For :math:`C_\ell` in the Limber
    approximation,

    .. math::

        {\rm Cov}\big[C^{ab}_{\ell_1}, C^{cd}_{\ell_2}\big] =
        \frac{1}{4\pi f_{\rm sky}}\int d\chi\,
        \frac{W_aW_bW_cW_d}{\chi^6}\,
        T^{1h}_{abcd}\!\left(\frac{\ell_1+\tfrac12}{\chi},
        \frac{\ell_2+\tfrac12}{\chi}\right),

    and for :math:`w_p` and :math:`\Delta\Sigma`, with bin-averaged :math:`\bar
    J_0` and :math:`\bar J_2`,

    .. math::

        {\rm Cov}\big[X_i, Y_j\big] = \frac{c_Xc_Y}{V}
        \int\frac{d^2k_1}{(2\pi)^2}\frac{d^2k_2}{(2\pi)^2}\,
        \bar J_X(k_1; i)\,\bar J_Y(k_2; j)\,T^{1h}(k_1, k_2),

    :math:`c_{w_p} = 1` and :math:`c_{\Delta\Sigma} = \bar\rho_m`.  The
    transverse wavenumbers stand in for the 3D ones: a halo is far smaller than
    :math:`\Pi_{\max}`, so the line-of-sight window keeps every one-halo pair.
    A photometric dispersion is the exception, and it scales each :math:`w_p`
    leg by the fraction of pairs it leaves inside the window,
    :math:`{\rm erf}(\Pi_{\max}/2\sigma)`.  :math:`V` is the lens volume, or
    the lens volume inside the lens-source overlap for two :math:`\Delta\Sigma`
    blocks, as for the Gaussian term.

``"shared_pair"``
    For :math:`w_p`: the clustering of a pair counted in two overlapping bins,
    :math:`N^2(2\Pi_{\max} + w_p)` in place of :math:`N^2\,2\Pi_{\max}`
    (:func:`~ggah_mod.covariance.gaussian.wp_noise_term`).
``"one_halo_shared"``
    For :math:`w_p` and :math:`\Delta\Sigma`: the one-halo terms in which the
    two pairs share one object
    (:func:`~ggah_mod.covariance.trispectrum.shared_object_cng`).
``"multi_halo"``
    The two-, three- and four-halo trispectrum with tree-level PT
    (:mod:`~ggah_mod.covariance.multihalo`), projected as the one-halo term is.
``"super_sample"``
    The response to a mode longer than the survey, with :math:`\sigma_b^2`
    from the footprint's mask power (:mod:`~ggah_mod.covariance.supersample`).
    A geometry known only by its area refuses: a disc of the same area is a
    choice, made by passing :func:`~ggah_mod.covariance.geometry.spherical_cap`.
    Real-space responses are bin-averaged over each annulus and transformed
    by layer 5, :math:`\Pi_{\max}` window included.

Every result lists in :attr:`Covariance.missing` the terms a complete
covariance has and it does not.

Noise is declared, never defaulted
----------------------------------

Every map in a :math:`C_\ell` block needs an entry in ``cl_noise`` and every
tracer in a real-space block an entry in ``tracer_noise``, zero included.  A
covariance with a forgotten shot noise is not slightly wrong on small scales; it
is wrong by orders of magnitude there, and it inverts without complaint.  Noise
is assigned to an auto of one map only: two differently named tracers are taken
to be disjoint samples.

What is not done, and says so
-----------------------------

* :math:`\xi`, :math:`w(\theta)` and the kSZ aperture, and the cross-covariance
  between a :math:`C_\ell` block and a real-space block: each raises rather
  than returning a zero block that would read as "no correlation".
* Real-space blocks at different redshifts or :math:`\Pi_{\max}`: their volumes
  and line-of-sight windows overlap in ways a single slab cannot state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from ..observables.spec import Cl, ObservableSpec, Statistic, make_model
from .gaussian import band_modes, cl_gaussian, make_wp_kernel, wp_gaussian
from .geometry import SurveyGeometry, projected_sigma2_b
from ..backend import resolve_backend
from ..spectra.spec import PkOptions

__all__ = ["Covariance", "covariance", "gaussian_covariance", "TERMS",
           "NOT_IMPLEMENTED"]

_SUPPORTED = ("cl", "wp", "delta_sigma")

#: The terms :func:`covariance` can include, with the name each is listed
#: under when it is left out.
TERMS = {
    "gaussian": "Gaussian",
    "shared_pair": "clustering of a pair shared by two bins",
    "one_halo": "one-halo trispectrum",
    "one_halo_shared": "one-halo terms with one object shared between the two pairs",
    "multi_halo": "two- to four-halo trispectrum",
    "super_sample": "super-sample",
}

#: Terms of a complete covariance that no option includes yet.
NOT_IMPLEMENTED = (
    "two- and three-halo terms with one object shared between the two pairs",
)


def _map(s: Statistic, leg: str):
    if leg == "a":
        return (s.a, s.kernel_a, s.beam_a)
    return (s.b, s.kernel_b, s.beam_b)


def _map_key(m) -> tuple:
    return tuple("" if v is None else str(v) for v in m)


@dataclass
class Covariance:
    r"""``params -> Cov``, with the bookkeeping that says what is in it.

    Attributes
    ----------
    names, slices : the data vector's layout, identical to the model's.
    terms : the terms included, as keys of :data:`TERMS`.
    missing : the terms a complete covariance has and this one does not.
    geometry : the survey the mode count and volume came from.
    """

    names: tuple[str, ...]
    slices: tuple[tuple[str, int, int], ...]
    geometry: SurveyGeometry
    terms: tuple[str, ...]
    missing: tuple[str, ...]
    _fn: object = field(default=None, repr=False)

    def __call__(self, params):
        return self._fn(params)

    @property
    def size(self) -> int:
        return self.slices[-1][2] if self.slices else 0


def _check_edges(s: Statistic, edges):
    if s.name not in edges:
        raise ValueError(
            f"{s.name!r} has no bin edges. A covariance counts modes or pairs "
            f"per bin, and a bin centre does not say how wide the bin is; pass "
            f"`edges={{{s.name!r}: ...}}`.")
    e = np.asarray(edges[s.name], dtype=float)
    if e.size != s.n + 1:
        raise ValueError(f"{s.name!r}: {e.size} edges for {s.n} bins")
    x = np.asarray(s.x)
    if np.any(x < e[:-1]) or np.any(x >= e[1:]):
        raise ValueError(f"{s.name!r}: a bin centre lies outside its edges")
    return e


def gaussian_covariance(spec: ObservableSpec, **kw) -> Covariance:
    """:func:`covariance` with the Gaussian term alone."""
    return covariance(spec, terms=("gaussian",), **kw)


def _check_terms(terms) -> tuple[str, ...]:
    terms = tuple(terms)
    unknown = [x for x in terms if x not in TERMS]
    if unknown or not terms or len(set(terms)) != len(terms):
        raise ValueError(f"terms {terms!r}: expected distinct names from "
                         f"{sorted(TERMS)}")
    return terms


def _missing(terms) -> tuple[str, ...]:
    return tuple(v for k, v in TERMS.items() if k not in terms) + NOT_IMPLEMENTED


def covariance(spec: ObservableSpec, *, terms, geometry: SurveyGeometry,
               edges: dict, fields_at, sectors, cosmo,
               kernels=None, beams=None, backend=None, options=None,
               z_proj=(), cl_noise=None, tracer_noise=None,
               sources=None, lensing_geometry=None, wp_sigma_los=None,
               wp_kernel_kw=None, wp_noise=None, overlaps=None) -> Covariance:
    r"""Compile the covariance of ``spec`` with the chosen ``terms``.

    Parameters
    ----------
    spec : ObservableSpec
        The data vector: ``cl``, ``wp`` and ``delta_sigma`` statistics.
    terms : tuple of str
        Keys of :data:`TERMS`.  Required: which terms a covariance holds is
        the first thing a reader of a fit needs to know.
    geometry : SurveyGeometry
        :math:`f_{\rm sky}` for :math:`C_\ell`, the slab volume for :math:`w_p`.
    edges : dict
        ``name -> edges``: integer multipole edges for a ``cl``, :math:`r_p`
        edges [Mpc/h] for a ``wp``.
    cl_noise : dict
        ``(tracer, kernel) -> float`` noise power [sr], for every map in a
        ``cl`` block.
    tracer_noise : dict
        ``tracer -> float``, :math:`1/\bar n` [(Mpc/h)\ :sup:`3`] for a
        discrete tracer and ``0.0`` for a field, for every tracer in a ``wp``
        block and every lens of a ``delta_sigma`` block.
    sources : dict
        ``name -> SourceSample`` for every ``delta_sigma`` statistic.
    lensing_geometry : SurveyGeometry
        The lens-source overlap, with the lens redshift slab.  Required with
        ``delta_sigma``.
    wp_sigma_los : dict
        ``name -> sigma`` [Mpc/h], a photometric line-of-sight dispersion for a
        ``wp`` block; see :func:`~ggah_mod.covariance.gaussian.make_wp_kernel`.
    overlaps : dict
        As for :func:`~ggah_mod.spectra.tracers.spectrum`, for the one-halo
        trispectrum's same-object rule.
    fields_at, sectors, kernels, beams, cosmo, backend, options, z_proj
        As for :func:`~ggah_mod.observables.spec.make_model`.
    """
    from .trispectrum import (contract_cng, interpolate_factor, one_halo_factors,
                              real_kernel, shared_object_cng)
    from .multihalo import multi_halo_trispectrum, pt_kernels

    terms = _check_terms(terms)
    gauss, one_halo = "gaussian" in terms, "one_halo" in terms
    shared_pair = "shared_pair" in terms
    one_halo_shared = "one_halo_shared" in terms
    super_sample = "super_sample" in terms
    multi_halo = "multi_halo" in terms
    overlaps = dict(overlaps or {})
    stats = spec.statistics
    for s in stats:
        if s.kind not in _SUPPORTED:
            raise NotImplementedError(
                f"{s.name!r} is a {s.kind!r} statistic; the Gaussian covariance "
                f"is implemented for {list(_SUPPORTED)} so far. A zero block "
                f"would read as 'uncorrelated', so this refuses instead.")
    kinds = {s.kind for s in stats}
    if "cl" in kinds and kinds != {"cl"}:
        raise NotImplementedError(
            "this spec mixes harmonic C_l and real-space blocks. They share "
            "galaxies, so their cross-covariance is not zero, and the joint "
            "projected/3D treatment it needs is not written yet.")

    e_of = {s.name: _check_edges(s, edges) for s in stats}
    slices = spec.resolve(z_proj=tuple(float(z) for z in z_proj)).slices
    offset = {name: start for name, start, _ in slices}
    total = slices[-1][2]

    # ---------------------------------------------------------------- C_ell
    if kinds == {"cl"}:
        if shared_pair or one_halo_shared:
            raise NotImplementedError(
                "the shared-object terms of a C_l covariance -- the shot-noise "
                "trispectrum, with profiles at |l1 + l2| averaged over three "
                "dimensions -- are not written; they exist for w_p and Delta "
                "Sigma. Leave them out and they are listed in `missing`.")
        cl_noise = dict(cl_noise or {})
        maps = []
        for s in stats:
            for leg in ("a", "b"):
                m = _map(s, leg)
                if m not in maps:
                    maps.append(m)
        missing = [(t, k) for t, k, _ in maps if (t, k) not in cl_noise]
        if missing:
            raise ValueError(
                f"no noise declared for maps {missing}. Pass cl_noise[(tracer, "
                f"kernel)], 0.0 for a map with none: a forgotten shot or shape "
                f"noise is the error an analytic covariance cannot recover "
                f"from.")
        blocks, aux, aux_name = [], [], {}

        def need(m1, m2, s):
            key = (tuple(sorted((_map_key(m1), _map_key(m2)))), s.x)
            if key not in aux_name:
                name = f"__cov__{len(aux_name)}"
                aux_name[key] = name
                aux.append(Cl(name, m1[0], m2[0], s.x, kernel_a=m1[1],
                              kernel_b=m2[1], beam_a=m1[2], beam_b=m2[2]))
            return aux_name[key]

        for i, s in enumerate(stats):
            for t in stats[i:]:
                if s.x != t.x or not np.array_equal(e_of[s.name], e_of[t.name]):
                    raise ValueError(
                        f"{s.name!r} and {t.name!r} are on different multipole "
                        f"bands. The Gaussian term couples equal multipoles "
                        f"only, so the cross block of overlapping but unequal "
                        f"bands is not a diagonal; bin both identically.")
                A, B, C, D = _map(s, "a"), _map(s, "b"), _map(t, "a"), _map(t, "b")
                blocks.append((s, t, need(A, C, s), need(B, D, s),
                               need(A, D, s), need(B, C, s),
                               (A, B, C, D)))

        model = make_model(ObservableSpec(tuple(aux)), fields_at=fields_at,
                           sectors=sectors, kernels=kernels, beams=beams,
                           cosmo=cosmo, backend=backend, options=options,
                           z_proj=z_proj)

        def noise_of(m1, m2):
            if _map_key(m1) != _map_key(m2):
                return 0.0
            return float(cl_noise[(m1[0], m1[1])])

        z_nodes = tuple(float(z) for z in z_proj)
        solid_angle = 4.0 * np.pi * geometry.f_sky

        def beam(m, ell):
            return 1.0 if m[2] is None else jnp.asarray(beams[m[2]](ell))

        def cng_block(params, fields, caches, s, t, A, B, C, D):
            ka, kb = kernels[A[1]], kernels[B[1]]
            kc, kd = kernels[C[1]], kernels[D[1]]
            chi = jnp.asarray(ka.chi)
            if not all(k.chi.shape == chi.shape for k in (kb, kc, kd)):
                raise ValueError(f"{s.name!r} x {t.name!r}: kernels on "
                                 f"different chi grids")
            l1, l2 = jnp.asarray(s.x), jnp.asarray(t.x)
            opts = options or PkOptions.from_backend(resolve_backend(backend))
            rows = []
            for i, fld in enumerate(fields):
                w_m = fld.dndm * fld.quadrature_measure()
                blk = jnp.zeros((s.n, t.n))
                k1, k2 = (l1 + 0.5) / chi[i], (l2 + 0.5) / chi[i]
                if one_halo:
                    for f, g in one_halo_factors(fld, A[0], B[0], C[0], D[0],
                                                 sectors, params,
                                                 overlaps=overlaps,
                                                 cache=caches[i]):
                        f1 = interpolate_factor(k1, fld.k, f)
                        g2 = interpolate_factor(k2, fld.k, g)
                        blk = blk + (f1 * w_m) @ g2.T
                if multi_halo:
                    blk = blk + multi_halo_trispectrum(
                        fld, A[0], B[0], C[0], D[0], sectors, params, k1, k2,
                        options=opts, overlaps=overlaps, cache=caches[i])
                rows.append(blk)
            weight = (jnp.asarray(ka.w) * jnp.asarray(kb.w) * jnp.asarray(kc.w)
                      * jnp.asarray(kd.w) / chi ** 6)
            out = jnp.trapezoid(weight[:, None, None] * jnp.stack(rows), chi,
                                axis=0) / solid_angle
            b1 = beam(A, l1) * beam(B, l1)
            b2 = beam(C, l2) * beam(D, l2)
            return out * jnp.outer(b1 * jnp.ones(s.n), b2 * jnp.ones(t.n))

        def ssc_harmonic(params, fields):
            from ..cosmology import background as bg
            from ..spectra.pk import linear_spectrum
            from .supersample import response
            opts = options or PkOptions.from_backend(resolve_backend(backend))
            chi = jnp.asarray(kernels[blocks[0][6][0][1]].chi)
            p_lin = jnp.stack([linear_spectrum(f, opts) for f in fields])
            chi_t = bg.transverse_distance(chi, cosmo)
            s2 = projected_sigma2_b(geometry, chi_t, fields[0].k, p_lin)
            resp = {}

            from ..spectra.bnl import table_for
            tables = [table_for(f, opts) for f in fields]

            def r_of(x, y):
                key = tuple(sorted((x, y)))
                if key not in resp:
                    rows = []
                    for f, tab in zip(fields, tables):
                        g, l = response(f, key[0], key[1], sectors, params,
                                        options=opts, overlaps=overlaps,
                                        bnl_table=tab)
                        rows.append(g - l)
                    resp[key] = jnp.stack(rows)                   # (Nz, Nk)
                return resp[key]

            out = jnp.zeros((total, total))
            lk = jnp.log(fields[0].k)
            for s, t, _, _, _, _, (A, B, C, D) in blocks:
                ka, kb = kernels[A[1]], kernels[B[1]]
                kc, kd = kernels[C[1]], kernels[D[1]]
                l1, l2 = jnp.asarray(s.x), jnp.asarray(t.x)
                rab, rcd = r_of(A[0], B[0]), r_of(C[0], D[0])

                def at(rows, ell):
                    q = jnp.log((ell[None, :] + 0.5) / chi[:, None])
                    return jnp.stack([jnp.where(
                        (q[i] >= lk[0]) & (q[i] <= lk[-1]),
                        jnp.interp(q[i], lk, rows[i]), 0.0)
                        for i in range(chi.size)])                # (Nz, Nl)

                w = (jnp.asarray(ka.w) * jnp.asarray(kb.w) * jnp.asarray(kc.w)
                     * jnp.asarray(kd.w) / chi ** 4) * s2
                blk = jnp.trapezoid(w[:, None, None] * at(rab, l1)[:, :, None]
                                    * at(rcd, l2)[:, None, :], chi, axis=0)
                blk = blk * jnp.outer(beam(A, l1) * beam(B, l1) * jnp.ones(s.n),
                                      beam(C, l2) * beam(D, l2) * jnp.ones(t.n))
                i0, j0 = offset[s.name], offset[t.name]
                out = out.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    out = out.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            return out

        def fn(params):
            cov = jnp.zeros((total, total))
            if gauss:
                cls, _ = model(params)
                for s, t, ac, bd, ad, bc, (A, B, C, D) in blocks:
                    n = band_modes(e_of[s.name], geometry.f_sky)
                    d = cl_gaussian(cls[ac] + noise_of(A, C),
                                    cls[bd] + noise_of(B, D),
                                    cls[ad] + noise_of(A, D),
                                    cls[bc] + noise_of(B, C), n)
                    i0, j0 = offset[s.name], offset[t.name]
                    idx = jnp.arange(s.n)
                    cov = cov.at[i0 + idx, j0 + idx].add(d)
                    if s.name != t.name:
                        cov = cov.at[j0 + idx, i0 + idx].add(d)
            if super_sample:
                if not z_nodes:
                    raise ValueError("the super-sample C_l term needs `z_proj`")
                fields = tuple(fields_at(z_nodes))
                cov = cov + ssc_harmonic(params, fields)
            if one_halo or multi_halo:
                if not z_nodes:
                    raise ValueError("the trispectrum C_l terms need `z_proj`")
                fields = tuple(fields_at(z_nodes))
                caches = [{} for _ in fields]
                for s, t, _, _, _, _, (A, B, C, D) in blocks:
                    blk = cng_block(params, fields, caches, s, t, A, B, C, D)
                    i0, j0 = offset[s.name], offset[t.name]
                    cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                    if s.name != t.name:
                        cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            return cov

        return Covariance(tuple(s.name for s in stats), tuple(slices),
                          geometry, terms, _missing(terms), _fn=fn)

    # ------------------------------------------------------ w_p and Delta Sigma
    from ..cosmology import background
    from ..spectra.tracers import spectrum as pk_spectrum
    from .gaussian import make_wp_kernel, wp_gaussian
    from .lensing import (delta_sigma_gaussian, make_lensing_quadrature,
                          wp_delta_sigma_gaussian)

    if wp_noise is not None:
        raise ValueError("`wp_noise` is now `tracer_noise`: the same 1/n_bar "
                         "serves w_p and the lens term of Delta Sigma.")
    tracer_noise = dict(tracer_noise or {})
    sources = dict(sources or {})
    wps = [s for s in stats if s.kind == "wp"]
    dss = [s for s in stats if s.kind == "delta_sigma"]
    tracers = sorted({n for s in wps for n in (s.a, s.b)} | {s.a for s in dss})
    missing = [n for n in tracers if n not in tracer_noise]
    if missing:
        raise ValueError(
            f"no noise declared for tracers {missing}. Pass tracer_noise[tracer] "
            f"= 1/n_bar for a discrete sample and 0.0 for a field.")
    for s in dss:
        if s.b != "matter":
            raise ValueError(f"{s.name!r}: Delta Sigma is a galaxy-matter "
                             f"statistic, so its second tracer must be 'matter'.")
        if s.name not in sources:
            raise ValueError(
                f"{s.name!r} has no SourceSample. Its covariance is set by the "
                f"sources -- shape noise, Sigma_crit, the lensing window -- as "
                f"much as by the lenses; pass `sources={{{s.name!r}: ...}}`.")
    if dss and lensing_geometry is None:
        raise ValueError(
            "Delta Sigma blocks need `lensing_geometry`: the lens-source overlap "
            "is the area the lensing signal is measured on, and it is not the "
            "lens survey's area.")
    if len({sources[s.name].name for s in dss}) > 1:
        raise NotImplementedError(
            "Delta Sigma blocks from different source samples: their overlaps "
            "and shared shape noise are not modelled yet.")
    zs = {float(s.z) for s in stats}
    if len(zs) != 1:
        raise NotImplementedError(
            f"real-space blocks at redshifts {sorted(zs)}: one lens slab only, "
            f"so far.")
    pis = {float(s.pi_max) for s in wps}
    if len(pis) > 1:
        raise NotImplementedError(f"w_p blocks at pi_max {sorted(pis)}: one "
                                  f"line-of-sight window only, so far.")
    z = zs.pop()
    b = resolve_backend(backend)
    options = options or PkOptions.from_backend(b)
    kw = dict(k_min=b.k_min, k_max=b.k_max)
    kw.update(wp_kernel_kw or {})
    sigma_los = dict(wp_sigma_los or {})
    wp_kernel = {s.name: make_wp_kernel(e_of[s.name], s.pi_max,
                                        sigma_los=float(sigma_los.get(s.name, 0.0)),
                                        **kw) for s in wps}
    quad = {s.name: make_lensing_quadrature(e_of[s.name], k_min=b.k_min,
                                            k_max=b.k_max) for s in stats}
    v_lens = geometry.volume(cosmo)
    v_overlap = lensing_geometry.volume(cosmo) if dss else None
    chi_l = float(background.transverse_distance(
        background.comoving_distance(z, cosmo), cosmo)[0])
    rho_m = float(cosmo.rho_matter)

    # Line-of-sight convergence, through layer 5, when a source asks for it.
    kappa_model, kappa_name = None, {}
    kappa_stats = []
    for s in dss:
        src = sources[s.name]
        if src.kappa_kernel is not None and src.name not in kappa_name:
            ell = np.clip(quad[s.name].k * chi_l, 1.0, None)
            nm = f"__cov__kappa_{len(kappa_name)}"
            kappa_name[src.name] = nm
            kappa_stats.append(Cl(nm, "matter", "matter", tuple(float(v) for v in ell),
                                  kernel_a=src.kappa_kernel,
                                  kernel_b=src.kappa_kernel))
    if kappa_stats:
        kappa_model = make_model(ObservableSpec(tuple(kappa_stats)),
                                 fields_at=fields_at, sectors=sectors,
                                 kernels=kernels, beams=beams, cosmo=cosmo,
                                 backend=backend, options=options, z_proj=z_proj)

    # One-halo kernels: a statistic is kernel @ P on the quadrature nodes.
    from ..observables.real_space import SIGMA_UNIT
    from scipy.special import erf

    def cng_kernel(s):
        q = quad[s.name]
        if s.kind == "wp":
            sig = float(sigma_los.get(s.name, 0.0))
            inside = 1.0 if sig == 0.0 else float(erf(s.pi_max / (2.0 * sig)))
            return real_kernel(q, 0, inside)
        return real_kernel(q, 2, rho_m * SIGMA_UNIT)

    k_cng = {s.name: cng_kernel(s) for s in stats} \
        if (one_halo or one_halo_shared or multi_halo) else {}

    # Shared-pair term: w_p of the pair integrated over each bin intersection.
    from ..observables.real_space import wp as wp_model
    from .gaussian import annulus_intersection_nodes
    shared_nodes = {}
    if shared_pair:
        for i, s in enumerate(wps):
            for t in wps[i:]:
                if {s.a, s.b} == {t.a, t.b} and tracer_noise[s.a] and tracer_noise[s.b]:
                    r, w = annulus_intersection_nodes(e_of[s.name], e_of[t.name])
                    sig = float(sigma_los.get(s.name, 0.0))
                    if sig != float(sigma_los.get(t.name, 0.0)):
                        raise NotImplementedError(
                            f"{s.name!r} and {t.name!r} share pairs but not a "
                            f"line-of-sight dispersion")
                    inside = 1.0 if sig == 0.0 else float(erf(s.pi_max / (2.0 * sig)))
                    shared_nodes[(s.name, t.name)] = (r, inside * w)

    # Super-sample: bin-averaged responses and the slab variances.
    if super_sample:
        from ..observables.real_space import delta_sigma as ds_model
        from .supersample import response, slab_sigma2_b
        for s in wps:
            if float(sigma_los.get(s.name, 0.0)) > 0.0:
                raise NotImplementedError(
                    f"{s.name!r}: the super-sample response of a w_p smeared "
                    f"along the line of sight is not written, and the layer-5 "
                    f"w_p it would transform is not smeared either.")
        mean_nodes = {}
        for s in stats:
            r, w = annulus_intersection_nodes(e_of[s.name], e_of[s.name])
            area = np.pi * (e_of[s.name][1:] ** 2 - e_of[s.name][:-1] ** 2)
            mean_nodes[s.name] = (r, np.einsum("iim->im", w) * area[:, None])

    pairs = [(s, t) for i, s in enumerate(stats) for t in stats[i:]]
    need = set()
    for s, t in pairs:
        for x in (s.a, s.b):
            for y in (t.a, t.b):
                need.add(tuple(sorted((x, y))))

    def fn(params):
        (fld,) = tuple(fields_at((z,)))
        from ..spectra.bnl import table_for
        tab = table_for(fld, options)
        pk = {p: pk_spectrum(fld, p[0], p[1], sectors, params,
                             options=options, bnl_table=tab).total
              for p in sorted(need)}
        k = fld.k
        if kappa_model is not None:
            kap, _ = kappa_model(params)

        def P(x, y):
            return pk[tuple(sorted((x, y)))]

        def N(x, y):
            return float(tracer_noise[x]) if x == y else 0.0

        def p_ss(s):
            src = sources[s.name]
            if src.kappa_kernel is None:
                return jnp.zeros(quad[s.name].k.size)
            return src.sigma_crit ** 2 * chi_l ** 2 * kap[kappa_name[src.name]]

        cov = jnp.zeros((total, total))
        cache = {}
        pt = None
        if multi_halo:
            kn = jnp.asarray(quad[stats[0].name].k)
            pt = pt_kernels(fld, kn, kn, options=options)
        if super_sample:
            s2_lens = slab_sigma2_b(geometry, cosmo, fields_at, options)
            s2_overlap = slab_sigma2_b(lensing_geometry, cosmo, fields_at,
                                       options) if dss else None
            resp = {}
            for s in stats:
                r, w = mean_nodes[s.name]
                b_name = "matter" if s.kind == "delta_sigma" else s.b
                gain, loss = response(fld, s.a, b_name, sectors, params,
                                      options=options, overlaps=overlaps,
                                      bnl_table=tab)
                if s.kind == "wp":
                    def tr(p):
                        return wp_model(jnp.asarray(r), (k, p), pi_max=s.pi_max,
                                        backend=b)
                else:
                    def tr(p):
                        return ds_model(jnp.asarray(r), (k, p), cosmo, backend=b)
                resp[s.name] = jnp.asarray(w) @ (tr(gain) - tr(loss))
            for s, t in pairs:
                s2 = s2_overlap if s.kind == t.kind == "delta_sigma" else s2_lens
                blk = s2 * jnp.outer(resp[s.name], resp[t.name])
                i0, j0 = offset[s.name], offset[t.name]
                cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
        for s, t in pairs:
            v = v_overlap if s.kind == t.kind == "delta_sigma" else v_lens
            i0, j0 = offset[s.name], offset[t.name]
            if one_halo:
                blk = contract_cng(
                    jnp.asarray(k_cng[s.name].matrix),
                    jnp.asarray(k_cng[t.name].matrix), quad[s.name].k, fld,
                    one_halo_factors(fld, s.a, s.b, t.a, t.b, sectors, params,
                                     overlaps=overlaps, cache=cache), v)
                cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            if multi_halo:
                kn = jnp.asarray(quad[s.name].k)
                tk = multi_halo_trispectrum(fld, s.a, s.b, t.a, t.b, sectors,
                                            params, kn, kn, options=options,
                                            overlaps=overlaps, cache=cache, pt=pt)
                blk = jnp.asarray(k_cng[s.name].matrix) @ tk \
                    @ jnp.asarray(k_cng[t.name].matrix).T / v
                cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            if one_halo_shared:
                blk = shared_object_cng(fld, k_cng[s.name], k_cng[t.name],
                                        s.a, s.b, t.a, t.b, sectors, params, v,
                                        overlaps=overlaps, cache=cache)
                cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            if (s.name, t.name) in shared_nodes:
                r, w = shared_nodes[(s.name, t.name)]
                w_ov = jnp.asarray(w) @ wp_model(jnp.asarray(r), (k, P(s.a, s.b)),
                                                 pi_max=s.pi_max, backend=b)
                blk = (N(s.a, t.a) * N(s.b, t.b) + N(s.a, t.b) * N(s.b, t.a)) \
                    * w_ov / v_lens
                i0, j0 = offset[s.name], offset[t.name]
                cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
                if s.name != t.name:
                    cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
            if not gauss:
                continue
            if s.kind == "wp" and t.kind == "wp":
                blk = wp_gaussian(wp_kernel[s.name], v_lens, k,
                                  P(s.a, t.a), P(s.b, t.b), P(s.a, t.b), P(s.b, t.a),
                                  N(s.a, t.a), N(s.b, t.b), N(s.a, t.b), N(s.b, t.a))
            elif s.kind == "delta_sigma" and t.kind == "delta_sigma":
                src = sources[s.name]
                blk = delta_sigma_gaussian(
                    quad[s.name], quad[t.name], v_overlap, k,
                    P(s.a, t.a), N(s.a, t.a), P(s.a, "matter"), P(t.a, "matter"),
                    rho_m, p_ss(s), src.shape_noise(chi_l), src.delta_pi2)
            else:
                w, d = (s, t) if s.kind == "wp" else (t, s)
                src = sources[d.name]
                blk = wp_delta_sigma_gaussian(
                    quad[w.name], quad[d.name], v_lens, k,
                    P(w.a, d.a), N(w.a, d.a), P(w.b, "matter"),
                    P(w.b, d.a), N(w.b, d.a), P(w.a, "matter"),
                    rho_m, src.delta_pi_cross)
                if s.kind != "wp":
                    blk = blk.T
            i0, j0 = offset[s.name], offset[t.name]
            cov = cov.at[i0:i0 + s.n, j0:j0 + t.n].add(blk)
            if s.name != t.name:
                cov = cov.at[j0:j0 + t.n, i0:i0 + s.n].add(blk.T)
        return cov

    return Covariance(tuple(s.name for s in stats), tuple(slices),
                      geometry, terms, _missing(terms), _fn=fn)
