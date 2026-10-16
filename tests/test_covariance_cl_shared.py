r"""Verification: the shared-object and shared-pair terms of a harmonic block.

Three kinds of check.  **Closed forms** of the angular averages, for a Gaussian
profile and a power-law spectrum.  **Reductions**: contracted with the
bin-averaged kernels of a projected statistic, the harmonic shared-object term
is the real-space one of :func:`shared_object_cng`.  And a **process whose
covariance is known**, a halo point process in a periodic box, whose band
powers scatter as the Gaussian term, the shared pair, the four distinct
objects and the shared object together.  Then the assembly in
:func:`covariance`, with the slab super-sample option.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import special

import jax.numpy as jnp

import ggah_mod.sectors as S
from ggah_mod.backend import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.covariance import (
    covariance, make_lensing_quadrature, one_halo_factors, real_kernel,
    shared_object_cng, spherical_cap,
)
from ggah_mod.covariance.spec import _slab_weights
from ggah_mod.covariance.trispectrum import (
    angle_average, interpolate_factor, shared_object_harmonic, shared_pair_harmonic,
)
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.observables import Cl, ObservableSpec, limber_grid, number_counts
from ggah_mod.sectors.protocol import TracerWeights
from ggah_mod.spectra.spec import Component


# --------------------------------------------------------------------------
# closed forms
# --------------------------------------------------------------------------

class TestTheAngularAverages:
    def test_a_gaussian_profile_averages_to_a_bessel_function(self):
        r""":math:`\langle e^{-q^2s^2/2}\rangle_\phi =
        e^{-(k_1^2+k_2^2)s^2/2}I_0(k_1k_2s^2)`."""
        s = 1.3
        k = np.geomspace(1e-3, 50.0, 2000)
        f = np.exp(-0.5 * (k * s) ** 2)
        k1 = np.array([0.05, 0.3, 0.9, 1.7])
        k2 = np.array([0.1, 0.8, 1.6])
        got = np.asarray(angle_average(k1, k2, k, f, n_angle=128))
        want = np.exp(-0.5 * (k1[:, None] - k2[None, :]) ** 2 * s ** 2) \
            * special.i0e(np.outer(k1, k2) * s ** 2)
        assert np.allclose(got, want, rtol=2e-4, atol=1e-10)

    def test_a_point_part_is_its_own_average(self):
        k = np.geomspace(1e-3, 50.0, 200)
        f = np.ones((k.size, 3)) * np.array([1.0, 2.0, 5.0])
        got = np.asarray(angle_average([0.1, 1.0], [0.2, 3.0], k, f, n_angle=16))
        assert np.allclose(got, np.broadcast_to([1.0, 2.0, 5.0], (2, 2, 3)), rtol=1e-12)

    def test_an_inverse_square_spectrum_averages_to_the_difference_of_squares(self):
        r""":math:`\langle 1/|\boldsymbol\ell_1+\boldsymbol\ell_2|^2\rangle_\phi =
        1/|\ell_1^2-\ell_2^2|`, away from :math:`\ell_1 = \ell_2`."""
        ell = np.geomspace(1.0, 1e4, 4000)
        l1, l2 = np.array([100.0, 900.0]), np.array([300.0, 2000.0])
        got = np.asarray(shared_pair_harmonic(l1, l2, ell, 1.0 / ell ** 2, n_angle=512))
        want = 1.0 / np.abs(l1[:, None] ** 2 - l2[None, :] ** 2)
        assert np.allclose(got, want, rtol=2e-3)

    def test_a_steep_spectrum_on_the_diagonal_converges(self):
        r"""A spectrum that falls by three decades from low multipoles, read on
        the diagonal, against an adaptive quadrature of the same average."""
        from scipy import integrate

        ell = np.geomspace(1.0, 2e4, 3000)
        cl = 1e-4 / (1.0 + (ell / 30.0) ** 1.5)
        l1 = np.array([200.0, 1500.0, 6000.0])
        got = np.asarray(shared_pair_harmonic(l1, l1, ell, cl, n_angle=128))

        def avg(L):
            f = lambda ph: np.interp(np.log(np.sqrt(max(2 * L * L * (1 + np.cos(ph)), 1e-30))),  # noqa: E731
                                     np.log(ell), cl)
            return integrate.quad(f, 0.0, np.pi, limit=500, points=[np.pi - 1e-2,
                                                                    np.pi - 1e-3])[0] / np.pi

        want = np.array([avg(L) for L in l1])
        assert np.allclose(np.diag(got), want, rtol=2e-3)

    def test_the_slab_weights_split_the_trapezoid(self):
        chi = np.array([100.0, 200.0, 300.0, 450.0, 600.0])
        w_in, w_out = (np.asarray(x) for x in _slab_weights(chi, (200.0, 450.0)))
        f = chi ** 2
        assert np.isclose((w_in + w_out) @ f, np.trapezoid(f, chi), rtol=1e-12)
        assert np.isclose(w_in @ np.ones_like(chi), 250.0)
        w_in, w_out = (np.asarray(x) for x in _slab_weights(chi, (0.0, 1e4)))
        assert np.allclose(w_out, 0.0)


# --------------------------------------------------------------------------
# a halo point process: Poisson haloes, Bernoulli centrals, Poisson satellites
# with a Gaussian dispersion
# --------------------------------------------------------------------------

def _process():
    return dict(nh=np.array([2e-4, 3e-5, 3e-6]), Nc=np.array([0.8, 1.0, 1.0]),
                Ns=np.array([0.3, 4.0, 25.0]), rs=np.array([12.0, 25.0, 45.0]))


def _model(hp, k_max):
    nbar = float(np.sum(hp["nh"] * (hp["Nc"] + hp["Ns"])))
    kf = np.geomspace(1e-4, k_max, 600)
    u = np.exp(-0.5 * (kf[:, None] * hp["rs"][None, :]) ** 2)
    w = TracerWeights(jnp.asarray(hp["Nc"]), jnp.asarray(hp["Ns"][None, :] * u),
                      jnp.asarray(nbar), True, None, "g",
                      jnp.asarray(hp["Nc"] + hp["Ns"]))
    fld = SimpleNamespace(k=jnp.asarray(kf), dndm=jnp.asarray(hp["nh"]),
                          quadrature_measure=lambda: jnp.ones(3))
    cache = {"g": ((Component("galaxies", "total", population="g"), w),)}
    return nbar, fld, cache


def _shared_closed_form(hp, nbar, k1, k2):
    r"""The shared-object trispectrum of the process, every leg pairing, with the
    closed-form average of the Gaussian satellites."""
    out = np.zeros((k1.size, k2.size))
    for c in range(3):
        rs, Nc, Ns = hp["rs"][c], hp["Nc"][c], hp["Ns"][c]
        p1, p2 = np.full_like(k1, Nc / nbar), np.full_like(k2, Nc / nbar)
        e1 = Ns * np.exp(-0.5 * (k1 * rs) ** 2) / nbar
        e2 = Ns * np.exp(-0.5 * (k2 * rs) ** 2) / nbar
        ubar = np.exp(-0.5 * (k1[:, None] - k2[None, :]) ** 2 * rs ** 2) \
            * special.i0e(np.outer(k1, k2) * rs ** 2)
        sat = Ns / nbar ** 2 * ubar * (np.outer(p1 + e1, p2 + e2) - np.outer(p1, p2))
        cen = Nc / nbar ** 2 * np.outer(e1, e2)
        out += hp["nh"][c] * 4.0 * (sat + cen)
    return out


class TestTheSharedObjectInAHarmonicBlock:
    def test_it_is_the_closed_form_of_the_process(self):
        hp = _process()
        nbar, fld, cache = _model(hp, 5.0)
        k1, k2 = np.geomspace(2e-3, 0.3, 17), np.geomspace(3e-3, 0.25, 13)
        got = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, k1, k2,
                                                same=[[True, True], [True, True]],
                                                cache=cache, n_angle=256))
        want = _shared_closed_form(hp, nbar, k1, k2)
        assert np.allclose(got, want, rtol=1e-3, atol=1e-6 * np.abs(want).max())

    def test_contracted_with_projected_kernels_it_is_the_real_space_term(self):
        r"""The reduction to :math:`w_p` and :math:`\Delta\Sigma`: through the
        bin-averaged kernels, the harmonic term is :func:`shared_object_cng`'s."""
        hp = _process()
        edges = np.array([30.0, 50.0, 80.0, 125.0])
        q = make_lensing_quadrature(edges, k_min=1e-4, k_max=1.0, n=256)
        nbar, fld, cache = _model(hp, 1.0)
        A = 1600.0 ** 2
        t = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, q.k, q.k,
                                              same=[[True, True], [True, True]],
                                              cache=cache, n_angle=128))
        for order in (0, 2):
            ker = real_kernel(q, order)
            ours = ker.matrix @ t @ ker.matrix.T / A
            real = np.asarray(shared_object_cng(fld, ker, ker, *("g",) * 4, {}, {}, A,
                                                cache=cache, r_max=300.0))
            assert np.allclose(ours, real, rtol=5e-3)

    def test_disjoint_maps_share_no_object(self):
        hp = _process()
        nbar, fld, cache = _model(hp, 1.0)
        k = np.geomspace(1e-3, 0.2, 5)
        z = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, k, k,
                                              same=[[False, False], [False, False]],
                                              cache=cache))
        assert np.all(z == 0.0)
        one = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, k, k,
                                                same=[[True, False], [False, False]],
                                                cache=cache, n_angle=128))
        four = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, k, k,
                                                 same=[[True, True], [True, True]],
                                                 cache=cache, n_angle=128))
        assert np.allclose(4.0 * one, four, rtol=1e-10)

    @pytest.mark.slow
    def test_band_powers_scatter_as_the_sum_of_every_term(self):
        r"""Realisations of the process in a periodic box, and the band powers of
        its distinct pairs, :math:`(|\sum_j e^{-i\mathbf{k}\cdot\mathbf{x}_j}|^2
        - N)/\bar n^2A`, from direct sums on the box's modes.  Their covariance is
        the Gaussian term, the shared pair, the four distinct objects and the
        shared object, each averaged over the modes of the two bands; leaving
        out the shared object or the shared pair is visible."""
        hp = _process()
        L = 1600.0
        A = L * L
        kf = 2 * np.pi / L
        edges = np.array([4.5, 9.0, 15.0, 24.0]) * kf
        nbar, fld, cache = _model(hp, 2.0)
        n = np.arange(-24, 25)
        kx, ky = np.meshgrid(n * kf, n * kf, indexing="ij")
        kx, ky = kx.ravel(), ky.ravel()
        kk = np.hypot(kx, ky)
        half = (ky > 0) | ((ky == 0) & (kx > 0))
        band = np.digitize(kk, edges) - 1
        sel = half & (band >= 0) & (band < edges.size - 1)
        kx, ky, kk, band = kx[sel], ky[sel], kk[sel], band[sel]
        nb = edges.size - 1
        rng = np.random.default_rng(5)
        n_real = 3000
        est = np.empty((n_real, nb))
        for it in range(n_real):
            xs, ys = [], []
            for c in range(3):
                nh = rng.poisson(hp["nh"][c] * A)
                hx, hy = rng.uniform(0, L, nh), rng.uniform(0, L, nh)
                cen = rng.random(nh) < hp["Nc"][c]
                ns = rng.poisson(hp["Ns"][c], nh)
                xs += [hx[cen], np.repeat(hx, ns) + rng.normal(0, hp["rs"][c], ns.sum())]
                ys += [hy[cen], np.repeat(hy, ns) + rng.normal(0, hp["rs"][c], ns.sum())]
            x, y = np.concatenate(xs), np.concatenate(ys)
            amp = np.exp(-1j * (np.outer(kx, x) + np.outer(ky, y))).sum(1)
            pk = (np.abs(amp) ** 2 - x.size) / (nbar ** 2 * A)
            est[it] = np.bincount(band, pk, nb) / np.bincount(band, None, nb)
        mc = np.cov(est.T)

        # the theory on the box's modes: every distinct |k| of a band, with its count
        mags = [np.unique(np.round(kk[band == b], 10), return_counts=True) for b in range(nb)]
        P = lambda k: np.sum(hp["nh"] * (2 * hp["Nc"] * hp["Ns"]  # noqa: E731
                                         * np.exp(-0.5 * (k[:, None] * hp["rs"]) ** 2)
                                         + hp["Ns"] ** 2 * np.exp(-(k[:, None] * hp["rs"]) ** 2)),
                             1) / nbar ** 2
        kg = np.geomspace(1e-4, 2.0, 4000)
        noise = 1.0 / nbar
        fac = one_halo_factors(fld, *("g",) * 4, {}, {}, cache=cache)
        theory = {t: np.zeros((nb, nb)) for t in ("gauss", "pair", "four", "shared")}
        for i in range(nb):
            ki, ci = mags[i]
            # every mode of the half plane is one independent complex mode
            theory["gauss"][i, i] = np.sum(ci * (P(ki) + noise) ** 2) / ci.sum() ** 2
            for j in range(nb):
                kj, cj = mags[j]
                wij = np.outer(ci, cj) / (ci.sum() * cj.sum())
                t4 = sum(np.asarray(interpolate_factor(ki, fld.k, f)) * hp["nh"]
                         @ np.asarray(interpolate_factor(kj, fld.k, g)).T for f, g in fac)
                ts = np.asarray(shared_object_harmonic(fld, *("g",) * 4, {}, {}, ki, kj,
                                                       same=[[True, True], [True, True]],
                                                       cache=cache, n_angle=256))
                tp = 2 * noise ** 2 * np.asarray(shared_pair_harmonic(ki, kj, kg, P(kg),
                                                                      n_angle=256))
                theory["four"][i, j] = np.sum(wij * t4) / A
                theory["shared"][i, j] = np.sum(wij * ts) / A
                theory["pair"][i, j] = np.sum(wij * tp) / A
        full = sum(theory.values())
        err = np.sqrt((np.outer(np.diag(mc), np.diag(mc)) + mc ** 2) / n_real)
        assert np.all(np.abs(mc - full) < 4.0 * err)
        assert np.any(np.abs(mc - (full - theory["shared"])) > 6.0 * err)


# --------------------------------------------------------------------------
# the assembly in covariance()
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def halo():
    pk = make_pk("emu_pk")
    gal = S.GalaxySector("zumandelbaum15", shmr="zu15", backend=DIFFERENTIABLE)
    gp = dict(S.galaxy_defaults("zumandelbaum15"), log10m_star_thresh=10.5)
    fld = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.2)
    f_cen, _ = gal.stellar_fraction(fld, gp)
    params = {"galaxies": gp, "matter": {"split": S.BaryonSplit.from_hot(
        PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(fld.n_m, 0.1),
        f_star_cen=f_cen)}}
    return dict(pk=pk, params=params,
                sectors={"galaxies": gal, "matter": S.MatterField()},
                fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz))


EDGES = np.array([100, 200, 400, 800, 1600, 3200])
ELL = tuple(np.sqrt(EDGES[1:] * EDGES[:-1]))


def _cl_cov(halo, terms, *, nbar_sr=2e6, **kw):
    z, chi = limber_grid(PLANCK18, z_max=0.3, n=12, backend=DIFFERENTIABLE, z_min=0.1)
    kern = {"c": number_counts(z, chi, jnp.ones_like(z))}
    spec = ObservableSpec((Cl("gg", "galaxies", "galaxies", ELL, kernel_a="c",
                              kernel_b="c"),
                           Cl("gm", "galaxies", "matter", ELL, kernel_a="c",
                              kernel_b="c")))
    geo = spherical_cap(0.3, name="cap", z_min=0.1, z_max=0.3, ell_max=512)
    args = dict(terms=terms, geometry=geo, edges={"gg": EDGES, "gm": EDGES},
                fields_at=halo["fields_at"], sectors=halo["sectors"], kernels=kern,
                cosmo=PLANCK18, backend=DIFFERENTIABLE, z_proj=z,
                cl_noise={("galaxies", "c"): 1.0 / nbar_sr, ("matter", "c"): 0.0})
    args.update(kw)
    return np.asarray(covariance(spec, **args)(halo["params"]))


class TestTheHarmonicAssembly:
    def test_the_shot_noise_terms_make_the_one_halo_block_a_variance(self, halo):
        """Gaussian and one-halo terms with the two shared terms are positive
        definite; for a sparse sample the four-object term alone takes it
        negative or near it."""
        terms = ("gaussian", "one_halo", "one_halo_shared", "shared_pair")
        c = _cl_cov(halo, terms, nbar_sr=2e5)
        assert np.allclose(c, c.T, rtol=1e-10, atol=0.0)
        d = np.sqrt(np.diag(c))
        assert np.linalg.eigvalsh(c / np.outer(d, d)).min() > 0.0

    def test_the_terms_add(self, halo):
        terms = ("gaussian", "one_halo", "one_halo_shared", "shared_pair")
        whole = _cl_cov(halo, terms)
        parts = sum(_cl_cov(halo, (t,)) for t in terms)
        assert np.allclose(whole, parts, rtol=1e-9, atol=1e-12 * np.abs(whole).max())

    def test_the_shared_pair_is_the_discrete_auto_alone(self, halo):
        c = _cl_cov(halo, ("shared_pair",))
        n = len(ELL)
        assert np.abs(c[:n, :n]).max() > 0.0
        assert np.all(c[n:, :] == 0.0) and np.all(c[:, n:] == 0.0)

    def test_the_shared_object_is_where_one_galaxy_leg_meets_another(self, halo):
        c = _cl_cov(halo, ("one_halo_shared",))
        n = len(ELL)
        assert np.all(np.diag(c[:n, :n]) > 0.0) and np.all(np.diag(c[n:, n:]) > 0.0)
        assert np.abs(c[:n, n:]).max() > 0.0

    def test_a_slab_inside_every_kernel_is_one_coherent_mode(self, halo):
        slab = _cl_cov(halo, ("super_sample",), cl_super_sample="slab")
        s = np.linalg.svd(slab, compute_uv=False)
        assert s[1] < 1e-8 * s[0]

    def test_a_slab_outside_every_kernel_leaves_limber(self, halo):
        lim = _cl_cov(halo, ("super_sample",))
        far = spherical_cap(0.3, name="cap", z_min=0.5, z_max=0.6, ell_max=512)
        slab = _cl_cov(halo, ("super_sample",), cl_super_sample="slab", geometry=far)
        assert np.allclose(slab, lim, rtol=1e-10, atol=1e-12 * np.abs(lim).max())

    def test_the_slab_needs_a_slab(self, halo):
        geo = spherical_cap(0.3, name="cap", ell_max=512)
        with pytest.raises(ValueError, match="slab"):
            _cl_cov(halo, ("super_sample",), cl_super_sample="slab", geometry=geo)
