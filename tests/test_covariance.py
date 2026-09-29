r"""Verification: layer 6's geometry and Gaussian covariance.

Three kinds of check, and the third is the one that matters.  Closed forms
(areas, monopoles, the Poisson pair count, kernel limits) say the pieces are
the pieces.  External codes (CCL's projected super-sample variance, jax-cosmo's
Gaussian ``C_l`` covariance, healpy's harmonic transforms) say the conventions
are the literature's.  **Gaussian realisations** -- on the sphere through healpy,
and in a periodic box through the exact covariance of its discrete modes -- say
the formulae describe how a Gaussian field actually scatters, which no amount of
agreement between two analytic codes can establish.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import ggah_mod.sectors as S
from ggah_mod.backend import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.covariance import (
    LS10_SAMPLES, annulus_j0, annulus_j2, band_modes, cl_gaussian, from_area,
    galactic_latitude_cut, gaussian_covariance, ls10_sample, make_wp_kernel,
    projected_sigma2_b, spherical_cap, wp_gaussian, wp_noise_term,
)
from ggah_mod.observables import (
    Cl, DeltaSigma, ObservableSpec, Wp, Xi, cmb_lensing, limber_grid,
    number_counts,
)
from ggah_mod.covariance.gaussian import annulus_overlap
from ggah_mod.covariance.lensing import (
    SourceSample, delta_sigma_gaussian, effective_sigma_crit,
    make_lensing_quadrature, sigma_crit_comoving, wp_delta_sigma_gaussian,
)
from ggah_mod.observables.real_space import SIGMA_UNIT


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

class TestGeometry:
    def test_the_galactic_cut_area_is_one_minus_sin_bmin(self):
        for b in (10.0, 20.0, 30.0):
            g = galactic_latitude_cut(b, ell_max=64)
            assert g.f_sky == pytest.approx(1.0 - np.sin(np.deg2rad(b)), rel=1e-14)

    @pytest.mark.parametrize("geo", [galactic_latitude_cut(20.0, ell_max=64),
                                     spherical_cap(0.3, ell_max=64)])
    def test_every_footprint_has_monopole_one_over_four_pi(self, geo):
        """With coefficients normalised by the footprint's own area the
        monopole is the same for every mask, which is what makes the
        normalisation checkable at all."""
        assert geo.mask_wl[0] == pytest.approx(1.0 / (4.0 * np.pi), rel=1e-13)

    def test_the_galactic_cut_has_no_odd_multipoles(self):
        """Mirror-symmetric caps; an odd multipole here would be a sign error
        in the southern cap, not a small number."""
        wl = np.asarray(galactic_latitude_cut(20.0, ell_max=200).mask_wl)
        assert np.all(wl[1::2] == 0.0)
        assert np.all(wl[0::2][:50] > 0.0)

    def test_the_analytic_mask_matches_a_pixelised_one(self):
        hp = pytest.importorskip("healpy")
        from ggah_mod.covariance import from_healpix_mask
        nside = 256
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        mask = (np.abs(90.0 - np.rad2deg(theta)) > 20.0).astype(float)
        pix = np.asarray(from_healpix_mask(mask, name="pix", ell_max=40,
                                           provenance="test").mask_wl)
        # A HEALPix cut is a staircase of rings, so its edge is not at 20 deg
        # exactly; compare at the latitude that has the pixel map's area, or
        # the zeros of the mask power move and a ratio near one blows up.
        b_eff = np.rad2deg(np.arcsin(1.0 - mask.mean()))
        ana = np.asarray(galactic_latitude_cut(b_eff, ell_max=40).mask_wl)
        even = np.arange(0, 21, 2)
        assert np.allclose(pix[even], ana[even], rtol=1e-3)

    def test_the_ls10_samples_each_carry_their_own_area(self):
        """The areas differ because each is counted from that sample's own
        randoms; one area for the survey would be wrong for four of five."""
        areas = {k: ls10_sample(k).area_deg2 for k in LS10_SAMPLES}
        assert len(set(round(a, 1) for a in areas.values())) == len(areas)
        g = ls10_sample("10.0")
        assert (g.z_min, g.z_max) == (0.05, 0.18)
        assert "area_deg2" in g.provenance

    def test_a_flat_slab_has_the_flat_volume(self):
        from ggah_mod.cosmology import background
        g = ls10_sample("10.0")
        chi = np.asarray(background.comoving_distance(jnp.asarray([0.05, 0.18]),
                                                      PLANCK18))
        flat = g.f_sky * 4.0 / 3.0 * np.pi * (chi[1] ** 3 - chi[0] ** 3)
        assert float(g.volume(PLANCK18)) == pytest.approx(flat, rel=1e-6)

    def test_a_footprint_without_a_slab_has_no_volume(self):
        with pytest.raises(ValueError, match="no redshift slab"):
            galactic_latitude_cut(20.0, ell_max=8).volume(PLANCK18)

    def test_the_super_sample_variance_needs_a_mask(self):
        g = from_area(1000.0, name="x", provenance="test")
        with pytest.raises(ValueError, match="not a default"):
            projected_sigma2_b(g, [100.0], np.logspace(-3, 0, 10),
                               np.ones((1, 10)))


class TestAFootprintFromRandoms:
    def test_uniform_randoms_on_a_disc_give_the_disc(self):
        """Area to the pixelisation of the edge, and the mask power of the
        analytic cap after the randoms' shot noise is removed."""
        hp = pytest.importorskip("healpy")
        from ggah_mod.covariance import from_randoms
        rng = np.random.default_rng(4)
        f_sky = 0.2
        n = 4_000_000
        mu = rng.uniform(1.0 - 2.0 * f_sky, 1.0, n)
        ra = rng.uniform(0.0, 360.0, n)
        dec = np.rad2deg(np.arcsin(mu))
        got = from_randoms(ra, dec, nside=64, name="disc randoms", provenance="test")
        assert got.f_sky == pytest.approx(f_sky, rel=0.01)
        want = np.asarray(spherical_cap(f_sky, ell_max=60).mask_wl)
        wl = np.asarray(got.mask_wl)[:61]
        low = np.arange(0, 21)
        assert np.allclose(wl[low], want[low], rtol=0.03, atol=1e-3 * want[0])

    def test_too_coarse_a_catalogue_is_refused(self):
        pytest.importorskip("healpy")
        from ggah_mod.covariance import from_randoms
        with pytest.raises(ValueError, match="fully covered"):
            from_randoms(np.array([10.0, 11.0]), np.array([5.0, 5.0]), nside=64,
                         name="two points", provenance="test")


class TestTheProjectedVarianceAgainstCCL:
    @pytest.fixture(scope="class")
    def ccl_setup(self):
        ccl = pytest.importorskip("pyccl")
        cosmo = ccl.Cosmology(Omega_c=0.26, Omega_b=0.049, h=0.677,
                              n_s=0.9665, sigma8=0.81,
                              transfer_function="eisenstein_hu")
        a = np.array([0.9, 0.6])
        chi = ccl.comoving_radial_distance(cosmo, a)
        k = np.logspace(-5, 1, 3000)
        h = cosmo["h"]
        plin = np.array([ccl.linear_matter_power(cosmo, k, x) for x in a]) * h ** 3
        return ccl, cosmo, a, chi * h, k / h, plin, h

    def test_it_matches_ccl_from_the_same_mask(self, ccl_setup):
        """Same mask power in, so this checks the sum, the interpolation and
        the h units -- ours are Mpc/h, CCL's Mpc."""
        ccl, cosmo, a, chi, k, plin, h = ccl_setup
        g = galactic_latitude_cut(20.0, ell_max=2048)
        ours = np.asarray(projected_sigma2_b(g, chi, k, plin)) / h
        theirs = ccl.sigma2_B_from_mask(cosmo, a, mask_wl=np.asarray(g.mask_wl))
        assert np.allclose(ours, theirs, rtol=1e-3)

    def test_a_small_disc_matches_ccls_flat_sky_disc(self, ccl_setup):
        ccl, cosmo, a, chi, k, plin, h = ccl_setup
        ours = np.asarray(projected_sigma2_b(spherical_cap(0.002, ell_max=4096),
                                             chi, k, plin)) / h
        assert np.allclose(ours, ccl.sigma2_B_disc(cosmo, a, fsky=0.002),
                           rtol=3e-3)

    def test_a_wide_disc_does_not_and_that_is_ccls_flat_sky(self, ccl_setup):
        r"""CCL's disc is :math:`[2J_1(kR)/kR]^2`, a flat-sky window.  At the
        :math:`f_{\rm sky}` of the extragalactic sky the exact cap is tens of
        per cent higher at low redshift, which is why wide footprints use the
        mask sum here and not a disc."""
        ccl, cosmo, a, chi, k, plin, h = ccl_setup
        ours = np.asarray(projected_sigma2_b(spherical_cap(0.658, ell_max=4096),
                                             chi, k, plin)) / h
        ratio = ours / ccl.sigma2_B_disc(cosmo, a, fsky=0.658)
        assert ratio[0] > 1.1


# --------------------------------------------------------------------------
# harmonic space
# --------------------------------------------------------------------------

class TestHarmonic:
    def test_band_modes_is_exact_for_integer_edges(self):
        edges = np.array([2, 5, 11])
        full = np.array([sum(2 * l + 1 for l in range(2, 5)),
                         sum(2 * l + 1 for l in range(5, 11))])
        assert np.array_equal(band_modes(edges, 1.0), full)

    def test_fractional_edges_are_refused(self):
        with pytest.raises(ValueError, match="integers"):
            band_modes([2.5, 10.0], 1.0)

    def test_it_matches_jax_cosmo(self):
        """jax-cosmo counts modes as (2l+1) * gradient(l) * f_sky and this
        package as f_sky (l_hi^2 - l_lo^2); with that one conversion the two
        agree to rounding, block by block."""
        jca = pytest.importorskip("jax_cosmo.angular_cl")
        edges = np.array([20, 40, 70, 110, 170, 260])
        ell = 0.5 * (edges[1:] + edges[:-1])
        rng = np.random.default_rng(3)
        c = np.abs(rng.normal(1e-6, 3e-7, (3, ell.size)))
        noise = np.stack([np.full(ell.size, 2e-7), np.zeros(ell.size),
                          np.full(ell.size, 5e-8)])

        class Probe:
            n_tracers = 1

        jc = np.asarray(jca.gaussian_cl_covariance(
            jnp.asarray(ell), [Probe(), Probe()], jnp.asarray(c),
            jnp.asarray(noise), f_sky=0.4, sparse=False))
        conv = (2 * ell + 1) * np.gradient(ell) / (edges[1:] ** 2 - edges[:-1] ** 2)
        obs = c + noise
        n = band_modes(edges, 0.4)
        # block (gg, kk): C^{gk} C^{gk} + C^{gk} C^{gk}, index 0 and 2 of three
        ours = np.asarray(cl_gaussian(obs[1], obs[1], obs[1], obs[1], n))
        nb = ell.size
        assert np.allclose(ours, np.diag(jc[:nb, 2 * nb:]) * conv, rtol=1e-12)
        ours = np.asarray(cl_gaussian(obs[0], obs[0], obs[0], obs[0], n))
        assert np.allclose(ours, np.diag(jc[:nb, :nb]) * conv, rtol=1e-12)

    def test_a_full_sky_gaussian_field_scatters_this_much(self):
        hp = pytest.importorskip("healpy")
        rng_seed = 11
        np.random.seed(rng_seed)
        lmax, nreal = 128, 400
        ell = np.arange(lmax + 1)
        cl = 1e-5 * ((ell + 10.0) / 50.0) ** -1.4
        edges = np.arange(8, 129, 20)
        lo, hi = edges[:-1], edges[1:]
        w = 2 * ell + 1

        def bands(c):
            return np.array([np.sum((w * c)[a:b]) / np.sum(w[a:b])
                             for a, b in zip(lo, hi)])

        est = np.array([bands(hp.alm2cl(hp.synalm(cl, lmax=lmax)))
                        for _ in range(nreal)])
        ana = np.asarray(cl_gaussian(bands(cl), bands(cl), bands(cl),
                                     bands(cl), band_modes(edges, 1.0)))
        ratio = est.var(axis=0, ddof=1) / ana
        mc = np.sqrt(2.0 / nreal)
        assert abs(ratio.mean() - 1.0) < 2.5 * mc / np.sqrt(ratio.size)
        assert np.all(np.abs(ratio - 1.0) < 4.0 * mc)


# --------------------------------------------------------------------------
# real space
# --------------------------------------------------------------------------

class TestRealSpace:
    def test_bin_averaged_kernels_have_their_limits(self):
        k = np.array([1e-8])
        assert annulus_j0(k, [1.0], [2.0])[0, 0] == pytest.approx(1.0, abs=1e-12)
        assert annulus_j2(k, [1.0], [2.0])[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_bin_averaged_kernels_match_direct_integration(self):
        from scipy import integrate, special
        lo, hi = 0.7, 1.3
        for k in (0.5, 3.0, 20.0):
            area = 0.5 * (hi ** 2 - lo ** 2)
            j0 = integrate.quad(lambda r: r * special.j0(k * r), lo, hi,
                                limit=200)[0] / area
            j2 = integrate.quad(lambda r: r * special.jv(2, k * r), lo, hi,
                                limit=200)[0] / area
            assert annulus_j0([k], [lo], [hi])[0, 0] == pytest.approx(j0, rel=1e-9)
            assert annulus_j2([k], [lo], [hi])[0, 0] == pytest.approx(j2, rel=1e-9)

    def test_the_noise_term_is_the_poisson_pair_count(self):
        r"""Pure noise :math:`1/\bar n`: :math:`{\rm Var}[w_p] =
        (2\Pi)^2/N_{\rm pairs}` with :math:`N_{\rm pairs} = \bar n^2 V A\,2\Pi/2`."""
        edges = np.array([1.0, 2.0, 4.0])
        ker = make_wp_kernel(edges, 60.0, n_perp=32, n_panels=8)
        nbar, vol = 1e-3, 1e8
        got = np.asarray(wp_noise_term(ker, vol, 1 / nbar, 1 / nbar, 1 / nbar,
                                       1 / nbar))
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        pairs = nbar ** 2 * vol * area * 2 * 60.0 / 2.0
        assert np.allclose(np.diag(got), (2 * 60.0) ** 2 / pairs, rtol=1e-12)
        assert np.all(got[~np.eye(2, dtype=bool)] == 0.0)

    def test_the_wp_quadrature_has_converged(self):
        edges = np.logspace(np.log10(0.5), np.log10(40.0), 9)
        k = np.logspace(-4, np.log10(200.0), 1024)
        p = 2e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.6) + 50.0 / (1 + (k / 3.0) ** 2)
        a = np.asarray(wp_gaussian(make_wp_kernel(edges, 67.4), 2.6e8, k,
                                   p, p, p, p, 94.0, 94.0, 94.0, 94.0))
        b = np.asarray(wp_gaussian(make_wp_kernel(edges, 67.4, n_perp=768,
                                                  nodes_per_panel=10, n_tail=96),
                                   2.6e8, k, p, p, p, p, 94.0, 94.0, 94.0, 94.0))
        assert np.allclose(np.diag(a), np.diag(b), rtol=2e-3)

    def test_a_gaussian_box_has_this_wp_covariance(self):
        r"""The continuum formula against the **exact** covariance of a periodic
        box's discrete modes -- no Monte Carlo noise -- for :math:`w_p`
        measured on the same grid, with the grid's own line-of-sight window.
        Cell-discretised annuli leave two per cent."""
        L, N, PI = 1000.0, 128, 100.0
        v = (L / N) ** 3

        def P(k):
            return 2e4 * (k / 0.02) / (1.0 + (k / 0.02) ** 2.6)

        kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        kz = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
        K = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2
                    + kz[None, None, :] ** 2)
        PK = np.where(K > 0, P(np.where(K > 0, K, 1.0)), 0.0)
        edges = np.array([30.0, 45.0, 67.0, 100.0])
        d = np.fft.fftfreq(N, d=1.0 / N) * (L / N)
        rperp = np.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        slab = np.abs(d) <= PI + 1e-9
        kbin = []
        for i in range(edges.size - 1):
            ann = (rperp >= edges[i]) & (rperp < edges[i + 1])
            c = np.zeros((N, N, N))
            c[ann[:, :, None] & slab[None, None, :]] = (L / N) / ann.sum()
            kbin.append(np.fft.rfftn(c).real)
        w_half = np.ones_like(K)
        w_half[:, :, 1:] = 2.0
        nb = len(kbin)
        exact = np.array([[2.0 / (N ** 6 * v ** 2)
                           * np.sum(w_half * PK ** 2 * kbin[i] * kbin[j])
                           for j in range(nb)] for i in range(nb)])
        kg = np.logspace(-4, np.log10(np.pi * N / L * np.sqrt(3)), 2000)
        # The grid's line-of-sight window is whole cells: 25 cells, not 2 Pi.
        pi_grid = slab.sum() * (L / N) / 2.0
        ker = make_wp_kernel(edges, pi_grid, k_min=1e-4, k_max=kg[-1],
                             n_perp=512)
        pg = P(kg)
        ana = np.asarray(wp_gaussian(ker, L ** 3, kg, pg, pg, pg, pg))
        assert np.allclose(np.diag(ana), np.diag(exact), rtol=0.03)
        ce = exact / np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
        ca = ana / np.sqrt(np.outer(np.diag(ana), np.diag(ana)))
        assert np.allclose(ca, ce, atol=0.01)


# --------------------------------------------------------------------------
# assembly from a spec
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    pk = make_pk("emu_pk")
    gal = S.GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
    gp = dict(S.galaxy_defaults("zumandelbaum15"), log10m_star_thresh=10.0)
    fld = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.13)
    f_cen, _ = gal.stellar_fraction(fld, gp)
    params = {"galaxies": gp, "matter": {"split": S.BaryonSplit.from_hot(
        PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(fld.n_m, 0.1),
        f_star_cen=f_cen)}}
    sectors = {"galaxies": gal, "matter": S.MatterField()}
    z, chi = limber_grid(PLANCK18, z_max=1.0, n=8, backend=DIFFERENTIABLE)
    kernels = {"counts": number_counts(z, chi,
                                       jnp.exp(-0.5 * ((z - 0.3) / 0.1) ** 2)),
               "cmbk": cmb_lensing(z, chi, PLANCK18)}
    return dict(params=params, sectors=sectors, kernels=kernels, z=z,
                fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz))


EDGES = np.array([20, 50, 100, 200, 400])
ELL = 0.5 * (EDGES[1:] + EDGES[:-1])
CL_SPEC = ObservableSpec((
    Cl("kk", "matter", "matter", ELL, kernel_a="cmbk", kernel_b="cmbk"),
    Cl("gg", "galaxies", "galaxies", ELL, kernel_a="counts", kernel_b="counts"),
    Cl("gk", "galaxies", "matter", ELL, kernel_a="counts", kernel_b="cmbk"),
))
NOISE = {("galaxies", "counts"): 1e-6, ("matter", "cmbk"): 1e-7}


def _cl_cov(built, spec=CL_SPEC, noise=NOISE, edges=None):
    return gaussian_covariance(
        spec, geometry=galactic_latitude_cut(20.0, ell_max=16),
        edges=edges or {s.name: EDGES for s in spec.statistics},
        fields_at=built["fields_at"], sectors=built["sectors"],
        kernels=built["kernels"], cosmo=PLANCK18, backend=DIFFERENTIABLE,
        z_proj=built["z"], cl_noise=noise)


class TestAssembly:
    def test_a_cl_covariance_is_symmetric_and_positive_definite(self, built):
        c = np.asarray(_cl_cov(built)(built["params"]))
        assert c.shape == (12, 12)
        assert np.allclose(c, c.T)
        assert np.linalg.eigvalsh(c).min() > 0.0

    def test_rows_follow_the_spec_order(self, built):
        cov = _cl_cov(built)
        assert cov.names == ("kk", "gg", "gk")
        assert cov.slices == CL_SPEC.resolve(
            z_proj=tuple(float(v) for v in built["z"])).slices

    def test_it_says_which_terms_it_lacks(self, built):
        cov = _cl_cov(built)
        assert cov.terms == ("gaussian",)
        assert "super-sample" in cov.missing
        assert "one-halo trispectrum" in cov.missing

    def test_a_missing_noise_is_refused(self, built):
        with pytest.raises(ValueError, match="no noise declared"):
            _cl_cov(built, noise={("matter", "cmbk"): 1e-7})

    def test_unequal_bands_are_refused(self, built):
        spec = ObservableSpec((
            Cl("a", "matter", "matter", ELL, kernel_a="cmbk", kernel_b="cmbk"),
            Cl("b", "matter", "matter", ELL + 1.0, kernel_a="cmbk",
               kernel_b="cmbk"),
        ))
        with pytest.raises(ValueError, match="different multipole"):
            _cl_cov(built, spec=spec, noise={("matter", "cmbk"): 1e-7},
                    edges={"a": EDGES, "b": EDGES})

    def test_an_unsupported_statistic_is_refused(self, built):
        spec = ObservableSpec((Xi("xi", "galaxies", "galaxies", [1.0], z=0.13),))
        with pytest.raises(NotImplementedError, match="zero block"):
            gaussian_covariance(spec, geometry=ls10_sample("10.0"),
                                edges={"xi": [0.5, 2.0]}, fields_at=None,
                                sectors={}, cosmo=PLANCK18)

    def test_mixing_cl_and_wp_is_refused(self, built):
        spec = ObservableSpec((
            CL_SPEC.statistics[0],
            Wp("wp", "galaxies", "galaxies", [1.0], pi_max=67.4, z=0.13),
        ))
        with pytest.raises(NotImplementedError, match="cross-covariance"):
            gaussian_covariance(spec, geometry=ls10_sample("10.0"),
                                edges={"kk": EDGES, "wp": [0.5, 2.0]},
                                fields_at=None, sectors={}, cosmo=PLANCK18)

    def test_a_wp_covariance_is_symmetric_and_positive_definite(self, built):
        edges = np.logspace(-1, np.log10(30.0), 9)
        rp = np.sqrt(edges[1:] * edges[:-1])
        spec = ObservableSpec((Wp("wp", "galaxies", "galaxies", rp,
                                  pi_max=67.4, z=0.13),))
        geo = ls10_sample("10.0")
        nbar = LS10_SAMPLES["10.0"][3] / float(geo.volume(PLANCK18))
        cov = gaussian_covariance(spec, geometry=geo, edges={"wp": edges},
                                  fields_at=built["fields_at"],
                                  sectors=built["sectors"], cosmo=PLANCK18,
                                  backend=DIFFERENTIABLE,
                                  tracer_noise={"galaxies": 1.0 / nbar})
        c = np.asarray(cov(built["params"]))
        assert np.allclose(c, c.T)
        assert np.linalg.eigvalsh(c).min() > 0.0
        # the noise floor is a lower bound on every diagonal element
        floor = np.diag(np.asarray(wp_noise_term(
            make_wp_kernel(edges, 67.4, n_perp=8, n_panels=8),
            geo.volume(PLANCK18), 1 / nbar, 1 / nbar, 1 / nbar, 1 / nbar)))
        assert np.all(np.diag(c) >= floor * (1 - 1e-12))

    def test_a_wp_tracer_without_noise_is_refused(self, built):
        spec = ObservableSpec((Wp("wp", "galaxies", "galaxies", [1.0],
                                  pi_max=67.4, z=0.13),))
        with pytest.raises(ValueError, match="no noise declared"):
            gaussian_covariance(spec, geometry=ls10_sample("10.0"),
                                edges={"wp": [0.5, 2.0]},
                                fields_at=built["fields_at"],
                                sectors=built["sectors"], cosmo=PLANCK18,
                                backend=DIFFERENTIABLE)


# --------------------------------------------------------------------------
# galaxy-galaxy lensing
# --------------------------------------------------------------------------

class TestLensing:
    def test_sigma_crit_matches_astropy(self):
        astropy = pytest.importorskip("astropy")
        import astropy.constants as const
        import astropy.units as u
        from astropy.cosmology import Planck18 as P18
        zl, zs = 0.13, 0.8
        dl = P18.angular_diameter_distance(zl)
        ds = P18.angular_diameter_distance(zs)
        dls = P18.angular_diameter_distance_z1z2(zl, zs)
        phys = (const.c ** 2 / (4 * np.pi * const.G) * ds / (dl * dls)).to(
            u.Msun / u.Mpc ** 2).value
        comoving_h = phys / (1 + zl) ** 2 / P18.h
        assert float(sigma_crit_comoving(zl, zs, PLANCK18)) == pytest.approx(
            comoving_h, rel=1e-3)

    def test_sources_in_front_of_the_lens_do_not_lens(self):
        assert np.isinf(float(sigma_crit_comoving(0.3, 0.2, PLANCK18)))
        zs = np.linspace(0.01, 0.29, 50)
        assert np.isinf(float(effective_sigma_crit(0.3, zs, np.ones_like(zs),
                                                   PLANCK18)))

    def test_annulus_overlap_is_the_closure_relation(self):
        e = np.array([1.0, 2.0, 3.0])
        m = annulus_overlap(e, e)
        area = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
        assert np.allclose(m, np.diag(1.0 / area))
        # a coarse bin containing both fine ones
        m2 = annulus_overlap(e, np.array([1.0, 3.0]))
        assert np.allclose(m2[:, 0], area / (area * area.sum()))

    def test_pure_noise_is_the_pair_count_shape_noise(self):
        r"""With every spectrum zero, :math:`{\rm Var}[\Delta\Sigma] =
        \Sigma_{\rm crit}^2\sigma_e^2/N_{\rm pairs}` with
        :math:`N_{\rm pairs} = N_{\rm lens}\,n_s A`."""
        edges = np.array([0.1, 0.3, 1.0])
        q = make_lensing_quadrature(edges, n=64)
        k = np.logspace(-4, 2, 50)
        zero = np.zeros(50)
        nbar, vol, n_sigma = 1e-2, 3e6, 7.0e30
        cov = np.asarray(delta_sigma_gaussian(q, q, vol, k, zero, 1 / nbar,
                                              zero, zero, 1.0, np.zeros(64),
                                              n_sigma, 700.0))
        area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        want = n_sigma / (nbar * vol * area) * SIGMA_UNIT ** 2
        assert np.allclose(np.diag(cov), want, rtol=1e-12)

    def test_shape_noise_is_per_comoving_area_at_the_lens(self):
        src = SourceSample("s", sigma_crit=5e15, sigma_e=0.3,
                           n_eff_arcmin2=10.0, delta_pi2=700.0,
                           delta_pi_cross=100.0, kappa_kernel=None,
                           provenance="test")
        chi = 400.0
        n_plane = 10.0 * (180 * 60 / np.pi) ** 2 / chi ** 2
        assert src.shape_noise(chi) == pytest.approx(5e15 ** 2 * 0.09 / n_plane)

    def test_a_2d_gaussian_grid_has_this_delta_sigma_covariance(self):
        r"""Two correlated 2D Gaussian fields on a periodic grid, with
        :math:`\Delta\Sigma = \bar\Sigma(<R) - \Sigma(R)` estimated from the
        cross-correlation on the grid itself.  The exact covariance of that
        estimator over the grid's discrete modes against
        :func:`delta_sigma_gaussian` and :func:`wp_delta_sigma_gaussian`: it
        checks the :math:`\bar J_2` kernels, the Wick factors and the
        normalisation, and leaves only the line-of-sight approximation of
        Singh et al. untested, which a 2D field has no line of sight for."""
        L, N = 2400.0, 512
        a = (L / N) ** 2
        kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        ky = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
        K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

        def pgg(k):
            return 4e3 * (k / 0.03) / (1 + (k / 0.03) ** 2.4)

        def pss(k):
            return 2e3 / (1 + (k / 0.05) ** 1.8)

        def pgs(k):
            return 0.5 * np.sqrt(pgg(k) * pss(k))

        Kp = np.where(K > 0, K, 1.0)
        Pgg, Pss, Pgs = [np.where(K > 0, f(Kp), 0.0) for f in (pgg, pss, pgs)]
        # annuli six cells wide: the residual is the grid's pixelisation of
        # the annulus edges, 6.5% at four cells
        edges = np.array([60.0, 90.0, 135.0])
        d = np.fft.fftfreq(N, d=1.0 / N) * (L / N)
        r = np.sqrt(d[:, None] ** 2 + d[None, :] ** 2).ravel()
        order = np.argsort(r, kind="stable")
        rs = r[order]
        last = np.searchsorted(rs, rs, side="right")

        def c_ds(lo, hi):
            ann = (rs >= lo) & (rs < hi)
            n_a = ann.sum()
            inc = np.zeros(r.size)
            idx = np.flatnonzero(ann)
            np.add.at(inc, last[idx] - 1, 1.0 / (n_a * last[idx]))
            c = np.cumsum(inc[::-1])[::-1] - ann / n_a
            out = np.empty(r.size)
            out[order] = c
            return out.reshape(N, N)

        def c_mean(lo, hi):
            ann = (r >= lo) & (r < hi)
            return (ann / ann.sum()).reshape(N, N)

        nb = edges.size - 1
        kds = [np.fft.rfftn(c_ds(edges[i], edges[i + 1])).real for i in range(nb)]
        kw = [np.fft.rfftn(c_mean(edges[i], edges[i + 1])).real for i in range(nb)]
        wh = np.ones_like(K)
        wh[:, 1:] = 2.0
        pref = 1.0 / (N ** 4 * a ** 2)
        ex_ds = np.array([[pref * np.sum(wh * (Pgg * Pss + Pgs ** 2) * kds[i] * kds[j])
                           for j in range(nb)] for i in range(nb)])
        ex_x = np.array([[pref * np.sum(wh * 2 * Pgg * Pgs * kw[i] * kds[j])
                          for j in range(nb)] for i in range(nb)])
        kg = np.logspace(-4, np.log10(np.pi * N / L * np.sqrt(2)), 3000)
        q = make_lensing_quadrature(edges, k_min=1e-4, k_max=kg[-1], n=1024)
        ds = np.asarray(delta_sigma_gaussian(q, q, L ** 2, kg, pgg(kg), 0.0,
                                             pgs(kg), pgs(kg), 1.0, pss(q.k),
                                             0.0, 1.0)) / SIGMA_UNIT ** 2
        x = np.asarray(wp_delta_sigma_gaussian(q, q, L ** 2, kg, pgg(kg), 0.0,
                                               pgs(kg), pgg(kg), 0.0, pgs(kg),
                                               1.0, 1.0)) / SIGMA_UNIT
        assert np.allclose(ds, ex_ds, rtol=0.025)
        assert np.allclose(x, ex_x, rtol=0.025)


class TestPhotometricLineOfSight:
    def test_zero_dispersion_is_the_spectroscopic_kernel(self):
        edges = np.array([1.0, 3.0, 9.0])
        k = np.logspace(-4, 2, 400)
        p = 1e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.5)
        a = np.asarray(wp_gaussian(make_wp_kernel(edges, 60.0), 1e8, k, p, p, p, p))
        b = np.asarray(wp_gaussian(make_wp_kernel(edges, 60.0, sigma_los=0.0),
                                   1e8, k, p, p, p, p))
        assert np.array_equal(a, b)

    def test_a_photometric_smearing_lowers_the_sample_variance(self):
        """Smearing moves pairs out of the pi_max window, so less signal is
        inside it and the signal variance falls, monotonically in sigma."""
        edges = np.array([5.0, 10.0, 20.0])
        k = np.logspace(-4, 2, 400)
        p = 1e4 * (k / 0.02) / (1 + (k / 0.02) ** 2.5)
        var = [np.diag(np.asarray(wp_gaussian(
            make_wp_kernel(edges, 60.0, sigma_los=s), 1e8, k, p, p, p, p)))
            for s in (0.0, 20.0, 60.0)]
        assert np.all(var[0] > var[1]) and np.all(var[1] > var[2])

    def test_a_negative_dispersion_is_refused(self):
        with pytest.raises(ValueError, match="dispersion"):
            make_wp_kernel([1.0, 2.0], 60.0, sigma_los=-1.0)


class TestLensingAssembly:
    def test_delta_sigma_needs_its_sources_and_overlap(self, built):
        spec = ObservableSpec((DeltaSigma("ds", "galaxies", "matter", [1.0],
                                          z=0.13),))
        kw = dict(geometry=ls10_sample("10.0"), edges={"ds": [0.5, 2.0]},
                  fields_at=built["fields_at"], sectors=built["sectors"],
                  cosmo=PLANCK18, backend=DIFFERENTIABLE,
                  tracer_noise={"galaxies": 94.0})
        with pytest.raises(ValueError, match="no SourceSample"):
            gaussian_covariance(spec, **kw)
        src = SourceSample("s", 5e15, 0.3, 10.0, 700.0, 100.0, None, "test")
        with pytest.raises(ValueError, match="lensing_geometry"):
            gaussian_covariance(spec, sources={"ds": src}, **kw)

    def test_the_old_noise_keyword_is_refused_with_its_new_name(self, built):
        spec = ObservableSpec((Wp("wp", "galaxies", "galaxies", [1.0],
                                  pi_max=67.4, z=0.13),))
        with pytest.raises(ValueError, match="tracer_noise"):
            gaussian_covariance(spec, geometry=ls10_sample("10.0"),
                                edges={"wp": [0.5, 2.0]},
                                fields_at=built["fields_at"],
                                sectors=built["sectors"], cosmo=PLANCK18,
                                wp_noise={"galaxies": 94.0})

    def test_a_joint_wp_delta_sigma_covariance_is_positive_definite(self, built):
        edges = np.logspace(-1, np.log10(20.0), 7)
        rp = np.sqrt(edges[1:] * edges[:-1])
        spec = ObservableSpec((
            Wp("wp", "galaxies", "galaxies", rp, pi_max=67.4, z=0.13),
            DeltaSigma("ds", "galaxies", "matter", rp, z=0.13),
        ))
        geo = ls10_sample("10.0")
        nbar = LS10_SAMPLES["10.0"][3] / float(geo.volume(PLANCK18))
        src = SourceSample("s", 5e15, 0.28, 15.0, 700.0, 134.8, None, "test")
        overlap = from_area(400.0, name="overlap", z_min=geo.z_min,
                            z_max=geo.z_max, provenance="test")
        cov = gaussian_covariance(spec, geometry=geo,
                                  edges={"wp": edges, "ds": edges},
                                  fields_at=built["fields_at"],
                                  sectors=built["sectors"], cosmo=PLANCK18,
                                  backend=DIFFERENTIABLE,
                                  tracer_noise={"galaxies": 1.0 / nbar},
                                  sources={"ds": src}, lensing_geometry=overlap)
        c = np.asarray(cov(built["params"]))
        assert c.shape == (12, 12)
        assert np.allclose(c, c.T)
        assert np.linalg.eigvalsh(c).min() > 0.0
