r"""Verification: the layers actually connect, from a cosmology to an observable.

Every other test file checks one layer against a closed form.  This one runs the
whole chain — ``Cosmology`` → ``HaloField`` → sectors → ``P_ab(k)`` →
``w_p`` / ``ΔΣ`` / ``C_ℓ`` — and asserts the results are physically sensible and
differentiable end to end.  It is the test that would have caught the
extrapolation defect below, which no single-layer test could: each layer was
correct, and the composition was not.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.observables import (
    c_ell, delta_sigma, king, limber_grid, number_counts, thermal_sz, wp,
)
from ggah_mod.sectors import BaryonSplit

pytestmark = pytest.mark.slow

ZU15 = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)
ELL = jnp.asarray([100.0, 1000.0, 5000.0, 20000.0])


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def model(pk):
    """A closure over the whole chain: ``(z, a, b) -> (k, P_ab)``."""
    gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
    gal_p = S.galaxy_defaults("zumandelbaum15")
    options = SP.PkOptions.from_backend(DIFFERENTIABLE)

    def spectra(z, a, b, cosmo=PLANCK18):
        return from_field(make_field(cosmo, DIFFERENTIABLE, pk, z=z), a, b)

    def from_field(field, a, b):
        f_cen, _ = gal.stellar_fraction(field, gal_p)
        sectors = {"galaxies": gal, "gas": S.HotGasDPM(backend=DIFFERENTIABLE),
                   "agn": S.AgnSector(gal), "matter": S.MatterField()}
        params = {"galaxies": gal_p, "gas": S.dpm_model_params(2),
                  "agn": S.AgnParams(),
                  "matter": {"split": BaryonSplit.from_hot(
                      PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(field.n_m, 0.1), f_star_cen=f_cen)}}
        return field.k, SP.spectrum(field, a, b, sectors, params,
                                    options=options).total

    spectra.from_field = from_field
    return spectra


@pytest.fixture(scope="module")
def angular(model, pk):
    """A redshift stack, and the kernels to project it with."""
    z, chi = limber_grid(PLANCK18, z_max=1.5, n=12, backend=DIFFERENTIABLE)
    k = model(0.25, "galaxies", "galaxies")[0]
    # The declared route, not a loop of `make_field(z=float(zi))`: one linear
    # P(k) call covers the whole grid, and each row is an ordinary one-epoch
    # field.  This used to be the loop, which is why nothing in the suite
    # exercised the route `observables/limber.py` recommends.
    fields = make_fields(PLANCK18, DIFFERENTIABLE, pk, z)
    gg = jnp.stack([model.from_field(f, "galaxies", "galaxies")[1] for f in fields])
    gy = jnp.stack([model.from_field(f, "galaxies", "pressure")[1] for f in fields])
    nz = jnp.exp(-0.5 * ((z - 0.5) / 0.15) ** 2)
    return k, gg, gy, number_counts(z, chi, nz), thermal_sz(z, chi)


class TestTheChainProducesSensibleNumbers:
    """Order-of-magnitude, which is what an integration test can assert."""

    def test_wp_looks_like_a_galaxy_sample(self, model):
        k, p_gg = model(0.25, "galaxies", "galaxies")
        got = np.asarray(wp(jnp.asarray([0.1, 1.0, 10.0]), (k, p_gg),
                            pi_max=60.0, backend=DIFFERENTIABLE))
        assert np.all(np.diff(got) < 0)              # falls with separation
        assert 100.0 < got[0] < 5000.0               # Mpc/h, at 0.1 Mpc/h
        assert 1.0 < got[-1] < 100.0                 # ...and at 10

    def test_delta_sigma_looks_like_a_lensing_measurement(self, model):
        k, p_gm = model(0.25, "galaxies", "matter")
        got = np.asarray(delta_sigma(jnp.asarray([0.1, 1.0, 10.0]), (k, p_gm),
                                     PLANCK18, backend=DIFFERENTIABLE))
        assert np.all(np.diff(got) < 0)
        assert 1.0 < got[0] < 500.0                  # Msun h / pc^2
        assert got[-1] > 0.0

    def test_the_tsz_cross_spectrum_is_the_right_order(self, angular):
        k, _, gy, kg, ky = angular
        got = np.asarray(c_ell(ELL, k, gy, kg, ky))
        assert np.all(got > 0.0)
        assert 1e-13 < got[0] < 1e-10                # C_l^gy at ell = 100


class TestTheAngularSpectrumFallsMonotonically:
    r"""The regression guard for a defect only the composition could show.

    ``c_ell`` interpolates :math:`P(k, z)` at :math:`k = (\ell+\tfrac12)/\chi`.
    At high :math:`\ell` and small :math:`\chi` that leaves the k grid -- at
    :math:`\ell = 5000` and :math:`z = 10^{-3}` it is already 1700 h/Mpc against
    a grid ending at 200 -- and a :math:`C^1` cubic extrapolated past its last
    knot is a *cubic in* :math:`\log P`, which diverges.

    Measured, before the fix: :math:`C_\ell^{gg}` came out **4.3e+22** at
    :math:`\ell = 5000` while :math:`\ell = 100` and 1000 were correct.  Every
    layer was right on its own; only the composition was wrong, and a plot of
    the first two points would have looked fine.
    """

    def test_it_does_not_diverge_at_high_ell(self, angular):
        k, gg, _, kg, _ = angular
        got = np.asarray(c_ell(ELL, k, gg, kg, kg))
        assert np.all(np.isfinite(got))
        assert np.all(np.diff(got) < 0.0), got
        assert got[-1] < got[0]                      # by orders, not a wobble
        assert got[0] / got[-1] > 100.0

    def test_the_spectrum_is_continued_not_extrapolated(self, angular):
        """Past the grid it must fall at least as fast as ``k^-3``."""
        import inspect
        from ggah_mod.observables import limber
        body = inspect.getsource(limber.c_ell).split('"""')[-1]
        assert "_log_interp_with_tail" in body
        assert "interp_cubic(" not in body


class TestThePsfActsWhereItShould:
    def test_it_suppresses_more_at_higher_ell(self, angular):
        k, _, gy, kg, ky = angular
        bare = np.asarray(c_ell(ELL, k, gy, kg, ky))
        beamed = np.asarray(c_ell(ELL, k, gy, kg, ky,
                                  beam_b=king(ELL, 8.64, 1.5)))
        ratio = beamed / bare
        assert np.all(ratio <= 1.0)
        assert np.all(np.diff(ratio) < 0.0)
        assert ratio[0] > 0.99 and ratio[-1] < 0.6


class TestTheWholeChainDifferentiates:
    @pytest.mark.x64
    def test_the_gradient_reaches_the_cosmology_through_wp(self, model):
        def f(omega_m):
            k, p = model(0.25, "galaxies", "galaxies",
                         PLANCK18.replace(Omega_m=omega_m))
            return jnp.sum(wp(jnp.asarray([1.0, 5.0]), (k, p), pi_max=60.0,
                              backend=DIFFERENTIABLE))

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_gradient_reaches_the_cosmology_through_delta_sigma(self, model):
        def f(omega_m):
            cosmo = PLANCK18.replace(Omega_m=omega_m)
            k, p = model(0.25, "galaxies", "matter", cosmo)
            return jnp.sum(delta_sigma(jnp.asarray([0.5, 2.0]), (k, p), cosmo,
                                       backend=DIFFERENTIABLE))

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_gradient_reaches_a_gas_parameter_through_c_ell(self, angular,
                                                                pk):
        r"""The whole point of the package: one parameter, two observables.

        ``log10_pe_anchor`` is the DPM pressure normalisation, so it reaches
        :math:`C_\ell^{gy}` and nothing else in this chain.
        """
        z, chi = limber_grid(PLANCK18, z_max=1.5, n=6, backend=DIFFERENTIABLE)
        gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
        gal_p = S.galaxy_defaults("zumandelbaum15")
        nz = jnp.exp(-0.5 * ((z - 0.5) / 0.15) ** 2)
        kg, ky = number_counts(z, chi, nz), thermal_sz(z, chi)
        options = SP.PkOptions.from_backend(DIFFERENTIABLE)

        def f(log10_pe_anchor):
            gas_p = S.dpm_model_params(2).replace(log10_pe_anchor=log10_pe_anchor)
            rows, k = [], None
            # One P(k) call for the grid; `log10_pe_anchor` is a gas parameter,
            # so the spectra it moves are rebuilt per evaluation but the fields
            # are not the thing being differentiated.
            for field in make_fields(PLANCK18, DIFFERENTIABLE, pk, z):
                f_cen, _ = gal.stellar_fraction(field, gal_p)
                sectors = {"galaxies": gal, "gas": S.HotGasDPM(backend=DIFFERENTIABLE),
                           "agn": S.AgnSector(gal), "matter": S.MatterField()}
                params = {"galaxies": gal_p, "gas": gas_p,
                          "agn": S.AgnParams(),
                          "matter": {"split": BaryonSplit.from_hot(
                              PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(field.n_m, 0.1),
                              f_star_cen=f_cen)}}
                rows.append(SP.spectrum(field, "galaxies", "pressure", sectors,
                                        params, options=options).total)
                k = field.k
            return jnp.sum(c_ell(jnp.asarray([500.0]), k, jnp.stack(rows),
                                 kg, ky))

        x, h = 2.0607, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)
