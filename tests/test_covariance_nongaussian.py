r"""Verification: layer 6's non-Gaussian terms.

The same three kinds of check as ``tests/test_covariance.py``.  **Reductions**
of the four-point rule to the pair rule, to a product of weights and to zero.
**CCL**, for the harmonic projection and for the halo-mass integral, fed the
same trispectrum and the same ingredients so that only conventions are
compared.  And **processes whose covariance is known**: a Poisson-cluster field,
whose connected four-point function is the one-halo trispectrum and nothing
else, and a halo point process -- Poisson halos, Bernoulli centrals, Poisson
satellites -- whose pair counts carry every discrete term at once.
"""

import numpy as np
import pytest
from scipy import special

import jax.numpy as jnp
from types import SimpleNamespace

import ggah_mod.sectors as S
from ggah_mod.backend import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.covariance import (
    FULL_SKY_DEG2, LS10_SAMPLES, NOT_IMPLEMENTED, TERMS, SourceSample,
    contract_cng, covariance, from_area, galactic_latitude_cut, ls10_sample,
    make_lensing_quadrature, one_halo_factors, one_halo_trispectrum,
    projected_sigma2_b, real_kernel, shared_object_cng, spherical_cap,
)
from ggah_mod.covariance.supersample import response, slab_sigma2_b
from ggah_mod.covariance.multihalo import (
    _Spectrum, _spline_coefficients, multi_halo_trispectrum,
)
from ggah_mod.spectra.pk import linear_spectrum
from ggah_mod.spectra.spec import PkOptions
from ggah_mod.spectra.tracers import spectrum
from ggah_mod.covariance.gaussian import (
    annulus_intersection_nodes, annulus_overlap, make_wp_kernel, wp_noise_term,
)
from ggah_mod.covariance.trispectrum import legs
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.observables import (
    Cl, DeltaSigma, ObservableSpec, Wp, limber_grid, lensing_efficiency,
    number_counts,
)
from ggah_mod.sectors.protocol import TracerWeights
from ggah_mod.spectra.pair import normalised_parts, pair_1h
from ggah_mod.spectra.spec import Component


@pytest.fixture(scope="module")
def halo():
    pk = make_pk("emu_pk")
    gal = S.GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
    gp = dict(S.galaxy_defaults("zumandelbaum15"), log10m_star_thresh=10.0)
    fld = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.13)
    f_cen, _ = gal.stellar_fraction(fld, gp)
    params = {"galaxies": gp, "matter": {"split": S.BaryonSplit.from_hot(
        PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(fld.n_m, 0.1),
        f_star_cen=f_cen)}}
    sectors = {"galaxies": gal, "matter": S.MatterField()}
    return dict(pk=pk, field=fld, params=params, sectors=sectors,
                fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz))


# --------------------------------------------------------------------------
# the four-point rule
# --------------------------------------------------------------------------

class TestTheFourPointRule:
    def test_a_continuous_field_is_the_product_of_its_weights(self, halo):
        f = halo["field"]
        (_, w), = legs(f, "matter", halo["sectors"], halo["params"])
        t = np.asarray(w.total(f.k.shape[-1]))
        want = (t ** 2 * np.asarray(f.dndm * f.quadrature_measure())) @ (t ** 2).T
        got = np.asarray(one_halo_trispectrum(f, *("matter",) * 4,
                                              halo["sectors"], halo["params"]))
        assert np.allclose(got, want, rtol=1e-5)

    def test_one_sample_is_the_pair_product_less_the_shared_central(self, halo):
        r"""CCL integrates :math:`{\rm pair}(k_1)\,{\rm pair}(k_2)`, which counts
        the central in both pairs; four distinct objects are that less
        :math:`4p_1e_1p_2e_2`."""
        f = halo["field"]
        nk = f.k.shape[-1]
        (_, w), = legs(f, "galaxies", halo["sectors"], halo["params"])
        pair = np.asarray(pair_1h(w, w, nk, overlap="identical"))
        p, e = [np.asarray(x) for x in normalised_parts(w, nk)]
        wm = np.asarray(f.dndm * f.quadrature_measure())
        want = (pair * wm) @ pair.T - 4.0 * ((p * e) * wm) @ (p * e).T
        got = np.asarray(one_halo_trispectrum(f, *("galaxies",) * 4,
                                              halo["sectors"], halo["params"]))
        assert np.allclose(got, want, rtol=1e-4, atol=1e-6 * np.abs(want).max())

    def test_two_centrals_of_one_halo_are_one_object(self, halo):
        """Centrals crossed with satellites have a one-halo pair, but four
        distinct objects with two centrals in one halo do not exist."""
        f = halo["field"]
        assert one_halo_factors(f, "centrals", "satellites", "centrals",
                                "satellites", halo["sectors"],
                                halo["params"]) == []
        assert one_halo_factors(f, *("centrals",) * 4, halo["sectors"],
                                halo["params"]) == []

    def test_a_nested_pair_is_refused(self, halo):
        with pytest.raises(NotImplementedError, match="nested"):
            one_halo_factors(halo["field"], "galaxies", "centrals", "galaxies",
                             "centrals", halo["sectors"], halo["params"])


class TestAgainstCCL:
    def test_the_mass_integral_matches_ccl(self, halo):
        """CCL's ``halomod_trispectrum_1h`` on this field's mass function and
        weights.  What differs is the quadrature, and CCL's re-interpolation of
        the tables onto its own mass grid."""
        ccl = pytest.importorskip("pyccl")
        f = halo["field"]
        h = PLANCK18.h
        m, k = np.asarray(f.m), np.asarray(f.k)
        nk = k.size
        tables = {}
        for name, ov in (("galaxies", "identical"), ("matter", "none")):
            (_, w), = legs(f, name, halo["sectors"], halo["params"])
            tables[name] = np.asarray(pair_1h(w, w, nk, overlap=ov)).T
        wm = np.asarray(f.dndm * f.quadrature_measure())
        md = ccl.halos.MassDef200m
        dndlog10m = np.asarray(f.dndm) * m * np.log(10.0)

        class MF(ccl.halos.MassFunc):
            name = "ggah"

            def _check_mass_def_strict(self, mass_def):
                return False

            def __call__(self, cosmo, M, a):
                return np.interp(np.log(np.atleast_1d(M) * h), np.log(m),
                                 dndlog10m) * h ** 3

        class One(ccl.halos.HaloProfile):
            def _fourier(self, cosmo, kk, M, a):
                return np.ones((np.atleast_1d(M).size, np.atleast_1d(kk).size))

        class Pair(ccl.halos.Profile2pt):
            def __init__(self, table):
                super().__init__()
                self.table = table

            def fourier_2pt(self, cosmo, kk, M, a, prof, *, prof2=None):
                lm = np.log(np.atleast_1d(M) * h)
                on_m = np.array([np.interp(lm, np.log(m), col)
                                 for col in self.table.T]).T
                lk = np.log(np.atleast_1d(kk) / h)
                return np.array([np.interp(lk, np.log(k), row)
                                 for row in on_m]) / h ** 6

        cosmo = ccl.Cosmology(Omega_c=PLANCK18.Omega_cdm, Omega_b=PLANCK18.Omega_b,
                              h=h, n_s=PLANCK18.n_s, A_s=2.1e-9,
                              transfer_function="eisenstein_hu")
        hmc = ccl.halos.HMCalculator(
            mass_function=MF(mass_def=md),
            halo_bias=ccl.halos.HaloBiasTinker10(mass_def=md), mass_def=md,
            log10M_min=np.log10(m[0] / h), log10M_max=np.log10(m[-1] / h),
            nM=m.size)
        get = hmc._get_mass_function

        def no_low_mass_deficit(*a, **kw):     # CCL adds rho-deficit/M_min
            get(*a, **kw)
            hmc._mf0 = 0.0
        hmc._get_mass_function = no_low_mass_deficit
        sel = np.arange(0, nk, 64)
        for name, table in tables.items():
            ours = (table.T * wm) @ table
            theirs = ccl.halos.halomod_trispectrum_1h(
                cosmo, hmc, k[sel] * h, 1.0 / 1.13, One(mass_def=md),
                prof12_2pt=Pair(table)) * h ** 9
            assert np.allclose(ours[np.ix_(sel, sel)], theirs, rtol=5e-3)

    @pytest.mark.slow
    def test_the_harmonic_block_matches_ccl(self, halo):
        r"""The Limber projection and :math:`1/4\pi f_{\rm sky}`, against CCL's
        ``angular_cl_cov_cNG`` given this package's trispectrum and radial
        kernels.  The residual is CCL interpolating the trispectrum between
        redshift nodes and integrating a spline where this uses a trapezoid."""
        ccl = pytest.importorskip("pyccl")
        h = PLANCK18.h
        z, chi = limber_grid(PLANCK18, z_max=1.2, n=48, backend=DIFFERENTIABLE,
                             z_min=0.02)
        zn = np.asarray(z)
        kern = {"counts": number_counts(z, chi, jnp.exp(-0.5 * ((z - 0.35) / 0.08) ** 2)),
                "wl": lensing_efficiency(z, chi, jnp.exp(-0.5 * ((z - 0.9) / 0.15) ** 2),
                                         PLANCK18)}
        edges = np.array([200, 400, 800, 1600])
        ell = tuple(0.5 * (edges[1:] + edges[:-1]))
        spec = ObservableSpec((
            Cl("gg", "galaxies", "galaxies", ell, kernel_a="counts", kernel_b="counts"),
            Cl("kk", "matter", "matter", ell, kernel_a="wl", kernel_b="wl"),
        ))
        geo = galactic_latitude_cut(20.0, ell_max=16)
        cov = covariance(spec, terms=("one_halo",), geometry=geo,
                         edges={s.name: edges for s in spec.statistics},
                         fields_at=halo["fields_at"], sectors=halo["sectors"],
                         kernels=kern, cosmo=PLANCK18, backend=DIFFERENTIABLE,
                         z_proj=z, cl_noise={("galaxies", "counts"): 0.0,
                                             ("matter", "wl"): 0.0})
        c = np.asarray(cov(halo["params"]))
        za = np.linspace(0.02, 1.2, 24)
        flds = make_fields(PLANCK18, DIFFERENTIABLE, halo["pk"], tuple(za))
        cosmo = ccl.Cosmology(Omega_c=PLANCK18.Omega_cdm, Omega_b=PLANCK18.Omega_b,
                              h=h, n_s=PLANCK18.n_s, A_s=2.1e-9, m_nu=PLANCK18.sum_mnu,
                              transfer_function="eisenstein_hu")

        def tk(name):
            t = np.array([np.asarray(one_halo_trispectrum(
                f, *(name,) * 4, halo["sectors"], halo["params"])) for f in flds])
            return ccl.Tk3D(a_arr=1.0 / (1.0 + za[::-1]),
                            lk_arr=np.log(np.asarray(flds[0].k) * h),
                            tkk_arr=t[::-1] / h ** 9, is_logt=False,
                            extrap_order_lok=0, extrap_order_hik=0)

        def tracer(name):
            tr = ccl.Tracer()
            tr.add_tracer(cosmo, kernel=(np.asarray(chi) / h,
                                         np.asarray(kern[name].w) * h))
            return tr

        sl = {n: slice(a, b) for n, a, b in cov.slices}
        for name, tr, ker in (("gg", "galaxies", "counts"), ("kk", "matter", "wl")):
            t = tracer(ker)
            theirs = ccl.angular_cl_cov_cNG(cosmo, t, t, tracer3=t, tracer4=t,
                                            ell=np.asarray(ell), t_of_kk_a=tk(tr),
                                            fsky=geo.f_sky,
                                            integration_method="spline")
            assert np.allclose(c[sl[name], sl[name]], theirs, rtol=0.03)


# --------------------------------------------------------------------------
# processes whose covariance is known
# --------------------------------------------------------------------------

class TestAPoissonClusterField:
    def test_the_projection_is_the_discrete_mode_sum(self):
        r"""Halos at Poisson positions, each a Gaussian lump of mass: every
        connected cumulant is one-halo, so the connected covariance of the
        grid's own estimator is exactly
        :math:`\sum_c n_c s_is_j/(N^6a^6)` with :math:`s_i = \sum_k
        C_i(k)F_c(k)` over the grid's modes.  Against it, the continuum
        contraction of :func:`contract_cng`, for annulus-mean (:math:`\bar J_0`)
        and disc-minus-ring (:math:`\bar J_2`) kernels.  The residual is the
        grid resolving each annulus edge."""
        L, N = 900.0, 128
        a = L / N
        M = np.array([1.0, 8.0, 40.0])
        rs = np.array([10.0, 20.0, 45.0])
        n = np.array([3e-3, 3e-4, 2e-5])
        rho = np.sum(n * M)
        kx = 2 * np.pi * np.fft.fftfreq(N, d=a)
        K = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
        edges = np.array([45.0, 75.0, 120.0, 180.0])
        d = np.fft.fftfreq(N, d=1.0 / N) * a
        r = np.sqrt(d[:, None] ** 2 + d[None, :] ** 2).ravel()
        q = make_lensing_quadrature(edges, k_min=1e-4, k_max=np.pi / a * np.sqrt(2),
                                    n=1024)
        k0, k2 = real_kernel(q, 0), real_kernel(q, 2)
        W = np.vstack([k0.real(r), k2.real(r)]) * a * a
        Ck = np.array([np.fft.fft2(w.reshape(N, N)).real for w in W])
        exact = np.zeros((6, 6))
        for c in range(3):
            s = np.einsum("ixy,xy->i", Ck, (M[c] / rho) ** 2
                          * np.exp(-(K * rs[c]) ** 2))
            exact += n[c] * np.outer(s, s)
        exact /= N ** 6 * a ** 6
        kf = np.logspace(-4, np.log10(q.k[-1] * 1.01), 800)
        fld = SimpleNamespace(k=jnp.asarray(kf), dndm=jnp.asarray(n),
                              quadrature_measure=lambda: jnp.ones(3))
        F = jnp.asarray((M / rho) ** 2 * np.exp(-(kf[:, None] * rs) ** 2))
        J = jnp.asarray(np.vstack([k0.matrix, k2.matrix]))
        ours = np.asarray(contract_cng(J, J, q.k, fld, [(F, F)], L * L))
        assert np.allclose(ours, exact, rtol=0.05)


def _halo_process():
    """Three halo classes: abundance, central probability, mean satellites and
    the satellites' Gaussian dispersion per axis."""
    return dict(nh=np.array([2e-4, 3e-5, 3e-6]), Nc=np.array([0.8, 1.0, 1.0]),
                Ns=np.array([0.3, 4.0, 25.0]), rs=np.array([12.0, 25.0, 45.0]),
                L=1600.0, edges=np.array([30.0, 50.0, 80.0, 125.0]))


def _process_model(hp, q):
    nbar = np.sum(hp["nh"] * (hp["Nc"] + hp["Ns"]))
    kf = np.logspace(-4, np.log10(q.k[-1]), 600)
    u = np.exp(-0.5 * (kf[:, None] * hp["rs"][None, :]) ** 2)
    w = TracerWeights(jnp.asarray(hp["Nc"]), jnp.asarray(hp["Ns"][None, :] * u),
                      jnp.asarray(nbar), True, None, "g",
                      jnp.asarray(hp["Nc"] + hp["Ns"]))
    fld = SimpleNamespace(k=jnp.asarray(kf), dndm=jnp.asarray(hp["nh"]),
                          quadrature_measure=lambda: jnp.ones(3))
    cache = {"g": ((Component("galaxies", "total", population="g"), w),)}
    return nbar, fld, cache


class TestAHaloPointProcess:
    def test_the_shared_object_term_is_its_fourier_sum(self):
        r"""For Gaussian satellites the angular average at the shared object is
        closed-form, :math:`e^{-(k_1^2+k_2^2)r_s^2/2}I_0(k_1k_2r_s^2)`, so the
        real-space route of :func:`shared_object_cng` -- points split off,
        remainders transformed -- is checked against the double Fourier sum."""
        hp = _halo_process()
        q = make_lensing_quadrature(hp["edges"], k_min=1e-4, k_max=60.0, n=512)
        nbar, fld, cache = _process_model(hp, q)
        A = hp["L"] ** 2
        for order in (0, 2):
            ker = real_kernel(q, order)
            ours = np.asarray(shared_object_cng(fld, ker, ker, *("g",) * 4, {}, {},
                                                A, cache=cache, r_max=300.0))
            k = q.k
            want = np.zeros_like(ours)
            for c in range(3):
                rs, Nc, Ns = hp["rs"][c], hp["Nc"][c], hp["Ns"][c]
                p = np.full_like(k, Nc / nbar)
                e = Ns * np.exp(-0.5 * (k * rs) ** 2) / nbar
                ubar = np.exp(-0.5 * (k[:, None] - k[None, :]) ** 2 * rs ** 2) \
                    * special.i0e(np.outer(k, k) * rs ** 2)
                sat = Ns / nbar ** 2 * ubar * (np.outer(p + e, p + e) - np.outer(p, p))
                cen = Nc / nbar ** 2 * np.outer(e, e)
                want += hp["nh"][c] * 4.0 * ker.matrix @ (sat + cen) @ ker.matrix.T
            assert np.allclose(ours, want / A, rtol=2e-3)

    @pytest.mark.slow
    def test_its_pair_counts_scatter_as_the_sum_of_every_term(self):
        r"""Realisations of the process, pair counts on exact positions in a
        periodic box, and the disc-minus-ring statistic -- which needs no mean
        density, so the box's own :math:`k = 0` mode does not enter.  Its
        variance is the Gaussian term, the shared pair, the four distinct
        objects and the shared object together; leaving out the last is
        visible."""
        spatial = pytest.importorskip("scipy.spatial")
        hp = _halo_process()
        L, edges = hp["L"], hp["edges"]
        q = make_lensing_quadrature(edges, k_min=1e-4, k_max=60.0, n=768)
        nbar, fld, cache = _process_model(hp, q)
        A = L * L
        ker = real_kernel(q, 2)
        rng = np.random.default_rng(11)
        n_real = 2500
        est = np.empty((n_real, edges.size - 1))
        for it in range(n_real):
            xs, ys = [], []
            for c in range(3):
                nh = rng.poisson(hp["nh"][c] * A)
                hx, hy = rng.uniform(0, L, nh), rng.uniform(0, L, nh)
                cen = rng.random(nh) < hp["Nc"][c]
                ns = rng.poisson(hp["Ns"][c], nh)
                xs += [hx[cen], np.repeat(hx, ns) + rng.normal(0, hp["rs"][c], ns.sum())]
                ys += [hy[cen], np.repeat(hy, ns) + rng.normal(0, hp["rs"][c], ns.sum())]
            pts = np.column_stack([np.concatenate(xs) % L, np.concatenate(ys) % L])
            pr = spatial.cKDTree(pts, boxsize=L).query_pairs(edges[-1],
                                                             output_type="ndarray")
            dv = pts[pr[:, 0]] - pts[pr[:, 1]]
            dv -= L * np.round(dv / L)
            est[it] = 2.0 * ker.real(np.hypot(dv[:, 0], dv[:, 1])).sum(1) / (nbar ** 2 * A)
        mc = np.diag(np.cov(est.T))

        P = np.sum(hp["nh"] * (2 * hp["Nc"] * hp["Ns"] * np.exp(-0.5 * (q.k[:, None] * hp["rs"]) ** 2)
                               + hp["Ns"] ** 2 * np.exp(-(q.k[:, None] * hp["rs"]) ** 2)), 1) / nbar ** 2
        noise = 1.0 / nbar
        rr = np.linspace(1e-3, edges[-1], 100001)
        kr = ker.real(rr) * 2 * np.pi * rr * (rr[1] - rr[0])
        xi = special.j0(np.outer(rr, q.k)) @ (q.weight * P)
        gauss = np.einsum("in,n,jn->ij", ker.matrix, 2 * P ** 2 + 4 * P * noise, q.j2) / A \
            + 2 * noise ** 2 * kr @ ker.real(rr).T / A
        pair = 2 * noise ** 2 * (kr * xi) @ ker.real(rr).T / A
        K = jnp.asarray(ker.matrix)
        four = np.asarray(contract_cng(K, K, q.k, fld, one_halo_factors(
            fld, *("g",) * 4, {}, {}, cache=cache), A))
        shared = np.asarray(shared_object_cng(fld, ker, ker, *("g",) * 4, {}, {}, A,
                                              cache=cache, r_max=300.0))
        full = np.diag(gauss + pair + four + shared)
        assert np.allclose(mc, full, rtol=0.06)
        assert np.all(mc / np.diag(gauss + pair + four) > 1.1)


# --------------------------------------------------------------------------
# the shared pair
# --------------------------------------------------------------------------

class TestTheSharedPair:
    def test_intersection_nodes_integrate_the_overlap(self):
        e1, e2 = np.array([1.0, 2.0, 4.0]), np.array([1.5, 3.0])
        r, w = annulus_intersection_nodes(e1, e2)
        assert np.allclose(w.sum(-1), annulus_overlap(e1, e2), rtol=1e-12)
        # r^2 over an intersection, exactly
        lo, hi = 1.5, 2.0
        want = 0.5 * np.pi * (hi ** 4 - lo ** 4) / (np.pi * 3.0 * np.pi * 6.75)
        assert (w @ r ** 2)[0, 0] == pytest.approx(want, rel=1e-12)

    def test_a_pair_is_counted_where_its_members_are(self):
        """N^2 (2 Pi + w_p): zero clustering is the Poisson term."""
        kern = make_wp_kernel([1.0, 2.0, 3.0], 50.0, n_perp=16, n_tail=4)
        poisson = np.asarray(wp_noise_term(kern, 1e6, 2.0, 2.0, 2.0, 2.0))
        r, w = annulus_intersection_nodes(kern.edges, kern.edges)
        clustered = np.asarray(wp_noise_term(kern, 1e6, 2.0, 2.0, 2.0, 2.0,
                                             w @ np.full(r.size, 100.0)))
        assert np.allclose(np.diag(clustered), np.diag(poisson) * (1 + 100.0 / 100.0))
        assert np.allclose(clustered - np.diag(np.diag(clustered)), 0.0)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

EDGES = np.logspace(np.log10(0.1), np.log10(10.0), 6)
RP = np.sqrt(EDGES[1:] * EDGES[:-1])
REAL = ObservableSpec((Wp("wp", "galaxies", "galaxies", RP, pi_max=67.4, z=0.13),
                       DeltaSigma("ds", "galaxies", "matter", RP, z=0.13)))


def _discs():
    """The LS10 sample and a 400 deg^2 overlap as discs: the super-sample term
    needs a mask, and a disc of the right area is the stated stand-in."""
    area = ls10_sample("10.0")
    geo = spherical_cap(area.area_sr / (4 * np.pi), name="LS10 disc",
                        z_min=area.z_min, z_max=area.z_max, ell_max=1024)
    overlap = spherical_cap(400.0 / FULL_SKY_DEG2, name="overlap disc",
                            z_min=area.z_min, z_max=area.z_max, ell_max=2048)
    return area, geo, overlap


def _real_cov(halo, terms, **extra):
    area, geo, overlap = _discs()
    nbar = LS10_SAMPLES["10.0"][3] / float(area.volume(PLANCK18))
    kw = dict(geometry=geo, edges={"wp": EDGES, "ds": EDGES},
              fields_at=halo["fields_at"], sectors=halo["sectors"], cosmo=PLANCK18,
              backend=DIFFERENTIABLE, tracer_noise={"galaxies": 1.0 / nbar},
              sources={"ds": SourceSample("s", 5e15, 0.28, 15.0, 700.0, 134.8,
                                          None, "test")},
              lensing_geometry=overlap)
    kw.update(extra)
    return covariance(REAL, terms=terms, **kw)


class TestAssembly:
    def test_every_term_together_is_positive_definite(self, halo):
        cov = _real_cov(halo, tuple(TERMS))
        assert cov.missing == NOT_IMPLEMENTED
        c = np.asarray(cov(halo["params"]))
        assert np.allclose(c, c.T, rtol=1e-6, atol=0.0)
        assert np.linalg.eigvalsh(c).min() > 0.0

    def test_the_terms_add(self, halo):
        whole = np.asarray(_real_cov(halo, tuple(TERMS))(halo["params"]))
        parts = sum(np.asarray(_real_cov(halo, (t,))(halo["params"])) for t in TERMS)
        assert np.allclose(whole, parts, rtol=1e-6)

    def test_the_one_halo_term_dominates_small_scale_wp(self, halo):
        g = np.diag(np.asarray(_real_cov(halo, ("gaussian",))(halo["params"])))
        t = np.diag(np.asarray(_real_cov(halo, ("one_halo",))(halo["params"])))
        assert t[0] > g[0]
        assert t[4] < 0.1 * g[4]

    def test_the_super_sample_block_is_one_response_times_its_variance(self, halo):
        c = np.asarray(_real_cov(halo, ("super_sample",))(halo["params"]))
        for blk in (c[:5, :5], c[5:, 5:]):
            s = np.linalg.svd(blk, compute_uv=False)
            assert s[1] < 1e-6 * s[0]
        # the cross block shares the lens footprint's variance with the w_p block
        _, geo, overlap = _discs()
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        s_lens = float(slab_sigma2_b(geo, PLANCK18, halo["fields_at"], opts))
        s_over = float(slab_sigma2_b(overlap, PLANCK18, halo["fields_at"], opts))
        r_wp = np.sqrt(np.diag(c[:5, :5]) / s_lens) * np.sign(np.diag(c[:5, 5:]))
        r_ds = np.sqrt(np.diag(c[5:, 5:]) / s_over)
        assert np.allclose(np.abs(c[:5, 5:]), s_lens * np.abs(np.outer(r_wp, r_ds)),
                           rtol=1e-5)

    def test_a_footprint_known_by_its_area_alone_refuses_the_super_sample_term(self, halo):
        area, _, overlap = _discs()
        with pytest.raises(ValueError, match="mask power"):
            _real_cov(halo, ("super_sample",), geometry=area)(halo["params"])

    def test_a_smeared_wp_refuses_the_super_sample_term(self, halo):
        with pytest.raises(NotImplementedError, match="smeared"):
            _real_cov(halo, ("super_sample",), wp_sigma_los={"wp": 30.0})

    def test_a_photometric_dispersion_scales_the_one_halo_wp_by_erf_squared(self, halo):
        from scipy.special import erf
        a = np.asarray(_real_cov(halo, ("one_halo",))(halo["params"]))
        b = np.asarray(_real_cov(halo, ("one_halo",), wp_sigma_los={"wp": 40.0})(halo["params"]))
        f = erf(67.4 / 80.0)
        assert np.allclose(b[:5, :5], f ** 2 * a[:5, :5], rtol=1e-6)
        assert np.allclose(b[:5, 5:], f * a[:5, 5:], rtol=1e-6)
        assert np.allclose(b[5:, 5:], a[5:, 5:], rtol=1e-6)

    def test_unknown_or_empty_terms_are_refused(self, halo):
        with pytest.raises(ValueError, match="terms"):
            _real_cov(halo, ("gaussian", "trispectrum"))
        with pytest.raises(ValueError, match="terms"):
            _real_cov(halo, ())

    def test_a_harmonic_shared_object_term_is_refused(self, halo):
        spec = ObservableSpec((Cl("gg", "galaxies", "galaxies", (300.0,),
                                  kernel_a="c", kernel_b="c"),))
        with pytest.raises(NotImplementedError, match="shared-object"):
            covariance(spec, terms=("gaussian", "one_halo_shared"),
                       geometry=galactic_latitude_cut(20.0, ell_max=16),
                       edges={"gg": [200, 400]}, fields_at=halo["fields_at"],
                       sectors=halo["sectors"], cosmo=PLANCK18,
                       cl_noise={("galaxies", "c"): 0.0})


# --------------------------------------------------------------------------
# super-sample
# --------------------------------------------------------------------------

class TestTheResponse:
    @pytest.mark.parametrize("tracer", ["matter", "galaxies"])
    def test_on_large_scales_it_is_growth_dilation_and_the_mean_density(self, halo, tracer):
        r""":math:`47/21 - \tfrac13 d\ln P_{\rm L}/d\ln k`, less :math:`2b` for
        a sample whose mean density is measured in the survey.

        A statement about the two-halo term, so the one-halo plateau is damped
        here (``mead20``); with the default undamped sum the halo sample
        variance keeps a large-scale constant, which the next test pins."""
        f = halo["field"]
        opts = PkOptions.from_backend(DIFFERENTIABLE, one_halo_transition="mead20")
        gain, loss = response(f, tracer, tracer, halo["sectors"], halo["params"],
                              options=opts)
        p = np.asarray(spectrum(f, tracer, tracer, halo["sectors"], halo["params"],
                                options=opts).total)
        k = np.asarray(f.k)
        pl = np.asarray(linear_spectrum(f, opts))
        i = np.searchsorted(k, [1e-3, 3e-3])
        want = 47 / 21 - np.gradient(np.log(pl), np.log(k))[i] / 3
        if tracer == "galaxies":
            want = want - 2.0 * np.sqrt(p[i] / pl[i])
        got = (np.asarray(gain) - np.asarray(loss))[i] / p[i]
        assert np.allclose(got, want, rtol=2e-3)

    @pytest.mark.parametrize("tracer", ["matter", "galaxies"])
    def test_with_the_default_sum_the_halo_sample_variance_stays(self, halo, tracer):
        r"""Undamped, the one-halo response :math:`\int dM\,n\,b\,W_aW_b` keeps
        its :math:`k\to0` value, as it does in CCL, which damps nothing.  So on
        large scales the response is growth and dilation on the two-halo term,
        plus that constant, less :math:`2bP` for a counted sample."""
        from ggah_mod.spectra.pair import pair_1h
        from ggah_mod.spectra.tracers import build_weights, resolve
        from ggah_mod.spectra.pk import two_halo_amplitude
        f = halo["field"]
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        assert opts.one_halo_transition == "none"
        gain, loss = response(f, tracer, tracer, halo["sectors"], halo["params"],
                              options=opts)
        ps = spectrum(f, tracer, tracer, halo["sectors"], halo["params"],
                      options=opts)
        k = np.asarray(f.k)
        pl = np.asarray(linear_spectrum(f, opts))
        i = np.searchsorted(k, [1e-3, 3e-3])
        (c,) = resolve(tracer).components
        w = build_weights(c, f, halo["sectors"], halo["params"])
        pair = pair_1h(w, w, k.size, overlap="identical")
        hsv = np.asarray(f.integrate(f.dndm * f.bias * pair, axis=-1))
        growth = 47 / 21 - np.gradient(np.log(pl), np.log(k)) / 3
        want = growth[i] * np.asarray(ps.two_halo)[i] + hsv[i]
        if tracer == "galaxies":
            # The mean density responds with one number, the large-scale bias,
            # which the response reads at the grid's first node.
            b = float(np.asarray(two_halo_amplitude(f, w, opts))[0])
            want = want - 2.0 * b * np.asarray(ps.total)[i]
        got = (np.asarray(gain) - np.asarray(loss))[i]
        assert np.all(hsv[i] > 0.0)
        assert np.allclose(got, want, rtol=1e-8)


class TestTheSlabVariance:
    def test_it_is_the_exact_spherical_bessel_sum(self, halo):
        r"""A thin, wide slab, where Limber's approximation along the line of
        sight is 38% low: the hybrid against the full sum over multipoles, for
        one spectrum at every distance and a mask truncated where both stop."""
        from scipy import special
        from ggah_mod.cosmology import background
        geo = spherical_cap(0.46, z_min=0.05, z_max=0.18, ell_max=120)
        f = halo["field"]
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        ours = float(slab_sigma2_b(geo, PLANCK18, lambda zz: tuple(f for _ in zz),
                                   opts))
        chi1, chi2 = (float(np.asarray(background.comoving_distance(z, PLANCK18))[0])
                      for z in (0.05, 0.18))
        k = np.concatenate([np.geomspace(1e-5, 1e-3, 40, endpoint=False),
                            np.arange(1e-3, 0.4, 5e-4)])
        pk = np.exp(np.interp(np.log(k), np.log(np.asarray(f.k)),
                              np.log(np.asarray(linear_spectrum(f, opts)))))
        x, w = np.polynomial.legendre.leggauss(300)
        chi = 0.5 * (chi2 - chi1) * (x + 1) + chi1
        wc = 0.5 * (chi2 - chi1) * w
        wl = np.asarray(geo.mask_wl)
        total = 0.0
        for ell in range(wl.size):
            rad = (special.spherical_jn(ell, np.outer(k, chi)) * chi ** 2) @ wc
            total += wl[ell] * np.trapezoid(k ** 2 * pk * rad ** 2, k)
        exact = 2 / np.pi * total / np.sum(wc * chi ** 2) ** 2
        assert ours == pytest.approx(exact, rel=2e-3)

    def test_limber_alone_would_have_been_wrong(self, halo):
        geo = spherical_cap(0.46, z_min=0.05, z_max=0.18, ell_max=120)
        f = halo["field"]
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        const = lambda zz: tuple(f for _ in zz)
        hybrid = float(slab_sigma2_b(geo, PLANCK18, const, opts))
        limber = float(slab_sigma2_b(geo, PLANCK18, const, opts, ell_exact=0))
        assert limber < 0.8 * hybrid


class TestTheHarmonicSuperSampleAgainstCCL:
    @pytest.mark.slow
    def test_it_matches_ccl_on_the_same_response_and_mask(self, halo):
        r"""CCL's ``angular_cl_cov_SSC`` given this package's response, radial
        kernels and :math:`\sigma_b^2(\chi)` of the :math:`|b|>20^\circ` mask.
        CCL interpolates :math:`\sigma_b^2` in the scale factor, and at low
        redshift -- where a lensing block's integrand starts -- that needs a
        logarithmic grid."""
        ccl = pytest.importorskip("pyccl")
        from ggah_mod.cosmology import background
        h = PLANCK18.h
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        z, chi = limber_grid(PLANCK18, z_max=1.2, n=160, backend=DIFFERENTIABLE,
                             z_min=0.005)
        kern = {"counts": number_counts(z, chi, jnp.exp(-0.5 * ((z - 0.35) / 0.08) ** 2)),
                "wl": lensing_efficiency(z, chi, jnp.exp(-0.5 * ((z - 0.9) / 0.15) ** 2),
                                         PLANCK18)}
        edges = np.array([60, 120, 240, 480])
        ell = tuple(0.5 * (edges[1:] + edges[:-1]))
        spec = ObservableSpec((
            Cl("kk", "matter", "matter", ell, kernel_a="wl", kernel_b="wl"),
            Cl("gk", "galaxies", "matter", ell, kernel_a="counts", kernel_b="wl"),
        ))
        geo = galactic_latitude_cut(20.0, ell_max=2048)
        cov = covariance(spec, terms=("super_sample",), geometry=geo,
                         edges={s.name: edges for s in spec.statistics},
                         fields_at=halo["fields_at"], sectors=halo["sectors"],
                         kernels=kern, cosmo=PLANCK18, backend=DIFFERENTIABLE,
                         z_proj=z, cl_noise={("galaxies", "counts"): 0.0,
                                             ("matter", "wl"): 0.0})
        c = np.asarray(cov(halo["params"]))
        za = np.geomspace(0.005, 1.2, 96)
        flds = make_fields(PLANCK18, DIFFERENTIABLE, halo["pk"], tuple(za))
        k = np.asarray(flds[0].k)

        def resp(a, b):
            rows = [np.asarray(g) - np.asarray(l) for g, l in (
                response(f, a, b, halo["sectors"], halo["params"], options=opts)
                for f in flds)]
            return np.array(rows)[::-1] / h ** 3

        s2 = np.asarray(projected_sigma2_b(
            geo, background.comoving_distance(jnp.asarray(za), PLANCK18), flds[0].k,
            jnp.stack([linear_spectrum(f, opts) for f in flds]))) / h
        cosmo = ccl.Cosmology(Omega_c=PLANCK18.Omega_cdm, Omega_b=PLANCK18.Omega_b,
                              h=h, n_s=PLANCK18.n_s, A_s=2.1e-9, m_nu=PLANCK18.sum_mnu,
                              transfer_function="eisenstein_hu")
        a_arr = 1.0 / (1.0 + za[::-1])

        def tracer(name):
            tr = ccl.Tracer()
            tr.add_tracer(cosmo, kernel=(np.asarray(chi) / h,
                                         np.asarray(kern[name].w) * h))
            return tr

        tg, tl = tracer("counts"), tracer("wl")
        r_mm, r_gm = resp("matter", "matter"), resp("galaxies", "matter")
        sl = {n: slice(a, b) for n, a, b in cov.slices}
        for name, (t1, t2), r in (("kk", (tl, tl), r_mm), ("gk", (tg, tl), r_gm)):
            tk = ccl.Tk3D(a_arr=a_arr, lk_arr=np.log(k * h), pk1_arr=r, pk2_arr=r,
                          is_logt=False, extrap_order_lok=0, extrap_order_hik=0)
            theirs = ccl.angular_cl_cov_SSC(cosmo, t1, t2, ell=np.asarray(ell),
                                            t_of_kk_a=tk, sigma2_B=(a_arr, s2[::-1]),
                                            integration_method="spline")
            assert np.allclose(c[sl[name], sl[name]], theirs.T, rtol=0.01)


# --------------------------------------------------------------------------
# two to four halos
# --------------------------------------------------------------------------

class TestTheMultiHaloTrispectrum:
    @pytest.fixture(scope="class", params=["none", "linear_deficit"])
    def matter_vs_ccl(self, halo, request):
        """This field's matter weights, mass function, bias and linear spectrum,
        in CCL's ``halomod_trispectrum_*`` and in ours.

        Twice: with the low-mass point mass off on both sides, and on on both
        -- CCL's ``_mbf0``, which completes every bias-weighted moment, against
        this package's counterterm and its multi-leg completion.  CCL's number
        -count extension ``_mf0`` has no counterpart here and stays off."""
        ccl = pytest.importorskip("pyccl")
        f = halo["field"]
        h = PLANCK18.h
        m, k = np.asarray(f.m), np.asarray(f.k)
        # The halo integral on both sides: CCL has no neutrino leg and no
        # beyond-linear term.
        completed = request.param == "linear_deficit"
        opts = PkOptions(two_halo_consistency=request.param,
                         neutrino_two_halo="none", bnl=False)
        (_, w), = legs(f, "matter", halo["sectors"], halo["params"])
        W = np.asarray(w.total(k.size))
        dndlog10m = np.asarray(f.dndm) * m * np.log(10.0)
        bias = np.asarray(f.bias)
        md = ccl.halos.MassDef200m

        def on_mass(M):
            return np.log(np.atleast_1d(M) * h)

        class MF(ccl.halos.MassFunc):
            name = "ggah"

            def _check_mass_def_strict(self, mass_def):
                return False

            def __call__(self, cosmo, M, a):
                return np.interp(on_mass(M), np.log(m), dndlog10m) * h ** 3

        class HB(ccl.halos.HaloBias):
            name = "ggah"

            def _check_mass_def_strict(self, mass_def):
                return False

            def __call__(self, cosmo, M, a):
                return np.interp(on_mass(M), np.log(m), bias)

        class Prof(ccl.halos.HaloProfile):
            def _fourier(self, cosmo, kk, M, a):
                rows = np.array([np.interp(on_mass(M), np.log(m), r) for r in W])
                lk = np.log(np.atleast_1d(kk) / h)
                return np.array([np.interp(lk, np.log(k), c, right=0.0)
                                 for c in rows.T]) / h ** 3

        cosmo = ccl.Cosmology(Omega_c=PLANCK18.Omega_cdm, Omega_b=PLANCK18.Omega_b,
                              h=h, n_s=PLANCK18.n_s, A_s=2.1e-9,
                              transfer_function="eisenstein_hu")
        hmc = ccl.halos.HMCalculator(mass_function=MF(mass_def=md),
                                     halo_bias=HB(mass_def=md), mass_def=md,
                                     log10M_min=np.log10(m[0] / h),
                                     log10M_max=np.log10(m[-1] / h), nM=m.size)
        get = hmc._get_ingredients

        def no_low_mass_extension(cosmo, a, *, get_bf):
            get(cosmo, a, get_bf=get_bf)
            hmc._mf0 = 0.0
            if get_bf and not completed:
                hmc._mbf0 = 0.0
        hmc._get_ingredients = no_low_mass_extension
        pl = np.asarray(linear_spectrum(f, opts), dtype=np.float64)
        a_nodes = np.linspace(0.5, 1.0, 8)
        pk2d = ccl.Pk2D(a_arr=a_nodes, lk_arr=np.log(k * h).astype(np.float64),
                        pk_arr=np.tile(np.log(pl / h ** 3), (8, 1)), is_logp=True)
        kq = np.geomspace(3e-3, 1.0, 9)
        a0 = 1.0 / 1.13
        prof = Prof(mass_def=md)
        theirs = {
            "2h_22": ccl.halos.halomod_trispectrum_2h_22,
            "2h_13": ccl.halos.halomod_trispectrum_2h_13,
            "3h": ccl.halos.halomod_trispectrum_3h,
            "4h": ccl.halos.halomod_trispectrum_4h,
        }
        theirs = {p: fn(cosmo, hmc, kq * h, a0, prof, p_of_k_a=pk2d) * h ** 9
                  for p, fn in theirs.items()}
        return kq, opts, theirs

    @pytest.mark.parametrize("part", ["2h_22", "2h_13", "3h", "4h"])
    def test_each_term_matches_ccl_for_matter(self, halo, matter_vs_ccl, part):
        """A continuous field, where CCL's product approximation of the
        three-leg moment is exact; the wavenumbers stop short of the most
        squeezed pairs, where both codes read the curvature of their own
        interpolant."""
        kq, opts, theirs = matter_vs_ccl
        ours = np.asarray(multi_halo_trispectrum(
            halo["field"], *("matter",) * 4, halo["sectors"], halo["params"],
            kq, kq, options=opts, parts=(part,), n_angle=64))
        assert np.allclose(ours, theirs[part], rtol=5e-3)

    def test_the_multi_leg_moments_carry_the_point_mass(self, halo):
        r"""The completion of :math:`I_{xy}` and :math:`I_{xyz}`, exactly, and
        its size: for matter it is 3e-4 of the two-leg moment, below the CCL
        comparison's tolerance, which is why it is checked here directly."""
        from ggah_mod.covariance.multihalo import _Moments
        from ggah_mod.spectra.counterterm import mass_deficit
        f = halo["field"]
        kq = jnp.asarray(np.geomspace(3e-3, 1.0, 5))
        on = _Moments(f, halo["sectors"], halo["params"],
                      PkOptions(two_halo_consistency="linear_deficit"), {}, {})
        off = _Moments(f, halo["sectors"], halo["params"],
                       PkOptions(two_halo_consistency="none"), {}, {})
        (_, w), = legs(f, "matter", halo["sectors"], halo["params"])
        w0 = np.interp(np.log(np.asarray(kq)), np.log(np.asarray(f.k)),
                       np.asarray(w.total(f.k.shape[0]))[:, 0])
        a = float(mass_deficit(f) * f.rho_cold / f.m[0])
        two_on = np.asarray(on.two("matter", "matter", "a", "b", kq, kq))
        two_off = np.asarray(off.two("matter", "matter", "a", "b", kq, kq))
        assert np.allclose(two_on - two_off, a * np.outer(w0, w0), rtol=1e-8)
        assert 0.0 < np.max((two_on - two_off) / two_off) < 1e-3
        three_on = np.asarray(on.three("matter", "matter", "matter", "a", "b",
                                       kq, kq))
        three_off = np.asarray(off.three("matter", "matter", "matter", "a", "b",
                                         kq, kq))
        assert np.allclose(three_on - three_off, a * np.outer(w0, w0 * w0),
                           rtol=1e-8)

    def test_a_discrete_leg_gets_no_point_mass(self, halo):
        from ggah_mod.covariance.multihalo import _Moments
        f = halo["field"]
        kq = jnp.asarray(np.geomspace(3e-3, 1.0, 5))
        on = _Moments(f, halo["sectors"], halo["params"],
                      PkOptions(two_halo_consistency="linear_deficit"), {}, {})
        off = _Moments(f, halo["sectors"], halo["params"],
                       PkOptions(two_halo_consistency="none"), {}, {})
        assert np.array_equal(
            np.asarray(on.two("galaxies", "matter", "a", "b", kq, kq)),
            np.asarray(off.two("galaxies", "matter", "a", "b", kq, kq)))

    def test_an_auto_trispectrum_is_symmetric(self, halo):
        kq = np.geomspace(3e-3, 1.0, 7)
        t = np.asarray(multi_halo_trispectrum(
            halo["field"], *("galaxies",) * 4, halo["sectors"], halo["params"],
            kq, kq, options=PkOptions.from_backend(DIFFERENTIABLE)))
        assert np.allclose(t, t.T, rtol=1e-5)

    def test_two_centrals_are_not_two_objects_in_the_moments(self, halo):
        """Centrals only: every multi-leg moment of one halo vanishes, so only
        the four-halo term, built from one-leg moments, survives."""
        kq = np.geomspace(3e-3, 0.3, 5)
        opts = PkOptions.from_backend(DIFFERENTIABLE)
        f = halo["field"]
        for part in ("2h_22", "2h_13", "3h"):
            t = np.asarray(multi_halo_trispectrum(
                f, *("centrals",) * 4, halo["sectors"], halo["params"], kq, kq,
                options=opts, parts=(part,)))
            assert np.all(t == 0.0)
        assert np.all(np.asarray(multi_halo_trispectrum(
            f, *("centrals",) * 4, halo["sectors"], halo["params"], kq, kq,
            options=opts, parts=("4h",))) != 0.0)

    def test_the_spline_has_a_continuous_curvature(self):
        x = jnp.linspace(0.0, 3.0, 40)
        y = jnp.sin(x)
        s = _Spectrum(x, y)
        xq = jnp.linspace(0.05, 2.95, 400)
        got = np.log(np.asarray(s(jnp.exp(xq))))   # the class works in log k
        assert np.allclose(got, np.sin(np.asarray(xq)), atol=2e-4)
        m2 = np.asarray(_spline_coefficients(x, y))
        assert np.allclose(m2[5:-5], -np.sin(np.asarray(x))[5:-5], atol=5e-3)
