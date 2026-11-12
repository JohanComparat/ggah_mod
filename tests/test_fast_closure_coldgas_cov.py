"""Fast versions of slow paths: the cold gas, the energy closure of the hot gas
and the non-Gaussian covariance of an angular spectrum.

The slow tests run these at the accurate settings and compare with data or
other codes; here each runs on the coarse flavour, on a few nodes, and checks
an identity the code must satisfy at any resolution.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

import ggah_mod.covariance as CV
import ggah_mod.observables as OB
import ggah_mod.sectors as S
from ggah_mod import DIFFERENTIABLE_COARSE as B
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.sectors import coldgas as CG


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, B, pk, z=0.0, calibration="off")


@pytest.fixture(scope="module")
def gal():
    return S.GalaxySector("zumandelbaum15", backend=B, calibration="off")


@pytest.fixture(scope="module")
def gp():
    return S.galaxy_defaults("zumandelbaum15")


# --------------------------------------------------------------------------
# cold gas
# --------------------------------------------------------------------------
class TestColdGas:
    @pytest.fixture(scope="class", params=sorted(CG.RELATIONS))
    def sector(self, request, gal, gp):
        rel = request.param
        per_galaxy = rel in CG.HI_GALAXY
        s = CG.ColdGasSector(rel, calibration="off",
                             **({"galaxies": gal} if per_galaxy else {}))
        return s, CG.ColdGasParams(), (gp if per_galaxy else None)

    def test_the_views_tend_to_their_masses(self, field, sector):
        s, p, g = sector
        hi = s.weights(field, p, view="hi", galaxies_params=g)
        mass = s.weights(field, p, view="mass", galaxies_params=g)
        assert not hi.discrete
        np.testing.assert_allclose(np.asarray(hi.w_extended)[0],
                                   np.asarray(s.m_hi(field, p, g)), rtol=1e-5)
        np.testing.assert_allclose(np.asarray(mass.w_extended)[0],
                                   np.asarray(s.m_neutral(field, p, g)), rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(s.m_neutral(field, p, g)),
            CG.helium_correction()
            * (np.asarray(s.m_hi(field, p, g)) + np.asarray(s.m_h2(field, p, g))),
            rtol=1e-12)
        np.testing.assert_allclose(np.asarray(s.f_cold(field, p, g)),
                                   np.asarray(s.m_neutral(field, p, g)) / np.asarray(field.m),
                                   rtol=1e-12)

    def test_the_cosmic_hi_is_a_small_positive_density(self, field, sector):
        s, p, g = sector
        omega = float(s.omega_hi(field, p, g))
        assert 1e-5 < omega < 1e-2
        assert 0.3 < float(s.bias_hi(field, p, g)) < 2.0
        assert float(s.brightness_temperature(field, p, g)) > 0.0

    def test_the_hi_mass_function_is_per_galaxy(self, field, sector):
        s, p, g = sector
        lg = np.linspace(8.5, 10.5, 5)
        if not s.per_galaxy:
            with pytest.raises(ValueError, match="halo-total"):
                s.hi_mass_function(field, lg, p, g)
            return
        phi = np.asarray(s.hi_mass_function(field, lg, p, g))
        assert np.all(np.isfinite(phi)) and np.all(phi > 0.0)

    @pytest.mark.parametrize("rel, kw, match", [
        ("hipass", {}, "unknown HI relation"),
        ("catinella18", {}, "needs the galaxy sector"),
        ("villaescusa18", {"galaxies": object()}, "halo-total"),
    ])
    def test_a_mismatched_sector_is_refused(self, rel, kw, match):
        with pytest.raises(ValueError, match=match):
            CG.ColdGasSector(rel, **kw)

    def test_a_galaxy_sector_without_a_threshold_is_refused(self):
        with pytest.raises(ValueError, match="threshold"):
            CG.ColdGasSector("catinella18", galaxies=S.GalaxySector("zheng07"))

    def test_an_unknown_view_is_refused(self, field):
        s = CG.ColdGasSector("villaescusa18", calibration="off")
        with pytest.raises(ValueError, match="unknown cold-gas view"):
            s.weights(field, CG.ColdGasParams(), view="h3")


# --------------------------------------------------------------------------
# the energy closure of the hot gas
# --------------------------------------------------------------------------
class TestTheClosure:
    @pytest.fixture(scope="class")
    def closed(self, gal):
        agn = S.AgnSector(gal, calibration="off")
        return S.HotGasDPM(backend=B, feedback="closure", agn=agn, galaxies=gal)

    def test_the_budget_is_between_the_floor_and_the_cosmic_share(self, field, closed, gp):
        f = np.asarray(closed.feedback_budget(field, S.DpmParams(), S.AgnParams(), gp))
        f_b = float(S.cosmic_baryon_fraction(PLANCK18))
        assert np.all(f > 0.0) and np.all(f < f_b)

    def test_the_amplitude_is_one_at_the_pivot(self, field, closed, gp):
        a = np.asarray(closed._feedback_amplitude(field, S.DpmParams(),
                                                  S.AgnParams(), gp))
        m = np.asarray(field.m)
        assert np.interp(np.log10(closed.feedback_pivot), np.log10(m), a) == \
            pytest.approx(1.0, rel=1e-2)

    def test_the_closure_moves_the_amplitude_not_the_shape(self, field, closed, gp):
        free = S.HotGasDPM(backend=B)
        dp = S.DpmParams()
        w_free = np.asarray(free.weights(field, dp, view="pressure").w_extended)
        w_closed = np.asarray(closed.weights(field, dp, view="pressure",
                                             agn_params=S.AgnParams(),
                                             galaxies_params=gp).w_extended)
        a = np.asarray(closed._feedback_amplitude(field, dp, S.AgnParams(), gp))
        ok = w_free[0] > 0
        np.testing.assert_allclose(w_closed[:, ok], a[None, ok] * w_free[:, ok],
                                   rtol=1e-6)

    def test_the_shape_moves_towards_the_dark_matter(self, field, closed, gp):
        dp = S.DpmParams()
        shaped = closed.closure_shape(field, dp, S.AgnParams(), gp)
        a = np.asarray(shaped.alpha_out_n)
        assert a.shape == (field.n_m,)
        lo, hi = sorted((float(dp.alpha_out_n), 3.0))
        assert np.all((a >= lo - 1e-9) & (a <= hi + 1e-9 + abs(dp.alpha_out_n_var) * 6))

    def test_the_closure_needs_its_peers_and_their_parameters(self, field, gal, gp, closed):
        with pytest.raises(ValueError):
            S.HotGasDPM(backend=B, feedback="closure").weights(
                field, S.DpmParams(), agn_params=S.AgnParams(), galaxies_params=gp)
        with pytest.raises(ValueError):
            closed.weights(field, S.DpmParams(), view="pressure")

    def test_a_cold_gas_peer_needs_its_parameters(self, field, gal, gp):
        cold = CG.ColdGasSector("villaescusa18", calibration="off")
        s = S.HotGasDPM(backend=B, feedback="closure",
                        agn=S.AgnSector(gal, calibration="off"), galaxies=gal,
                        coldgas=cold)
        with pytest.raises(ValueError, match="coldgas"):
            s.feedback_budget(field, S.DpmParams(), S.AgnParams(), gp)

    @pytest.mark.parametrize("kw, match", [
        ({"feedback": "winds"}, "feedback"),
        ({"scatter": "lognormal"}, "scatter"),
    ])
    def test_unknown_modes_are_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            S.HotGasDPM(backend=B, **kw)


class TestTheScatter:
    def test_the_two_forms_agree_without_scatter(self, field):
        on = dict(conc=field.conc, mdef=field.mdef)
        dp = S.DpmParams()
        lx = [np.asarray(S.HotGasDPM(backend=B, scatter=s).x_ray_luminosity(
            field.m, field.z, field.cosmo, dp, **on)) for s in ("constant", "isobaric")]
        np.testing.assert_allclose(lx[1], lx[0], rtol=1e-6)

    def test_the_constant_form_is_the_boost(self, field):
        on = dict(conc=field.conc, mdef=field.mdef)
        g = S.HotGasDPM(backend=B)
        lx0 = np.asarray(g.x_ray_luminosity(field.m, field.z, field.cosmo, S.DpmParams(), **on))
        lx1 = np.asarray(g.x_ray_luminosity(field.m, field.z, field.cosmo,
                                            S.DpmParams().replace(sigma_scatter=0.2), **on))
        np.testing.assert_allclose(lx1 / lx0, np.exp((0.2 * np.log(10.0)) ** 2), rtol=1e-6)


# --------------------------------------------------------------------------
# the covariance of an angular spectrum, every term, on a few nodes
# --------------------------------------------------------------------------
class TestTheAngularCovariance:
    @pytest.fixture(scope="class")
    def setup(self, pk, field, gal, gp):
        z, chi = OB.limber_grid(PLANCK18, z_max=0.8, n=4, backend=B)
        kernels = {"counts": OB.number_counts(
            z, chi, np.exp(-0.5 * ((np.asarray(z) - 0.4) / 0.1) ** 2))}
        edges = np.array([20, 60, 180])
        ell = np.sqrt(edges[1:] * edges[:-1])
        spec = OB.ObservableSpec((OB.Cl("cl", "galaxies", "galaxies", ell,
                                        kernel_a="counts", kernel_b="counts"),))
        f_cen, _ = gal.stellar_fraction(field, gp)
        params = {"galaxies": gp}
        common = dict(geometry=CV.spherical_cap(0.1, ell_max=256), edges={"cl": edges},
                      fields_at=lambda zs: make_fields(PLANCK18, B, pk, zs),
                      sectors={"galaxies": gal}, kernels=kernels, cosmo=PLANCK18,
                      backend=B, z_proj=np.asarray(z),
                      cl_noise={("galaxies", "counts"): 1e-6})
        return spec, common, params

    def test_the_terms_add_and_the_result_is_a_covariance(self, setup):
        spec, common, params = setup
        terms = ("gaussian", "one_halo", "multi_halo", "super_sample")
        parts = [np.asarray(CV.covariance(spec, terms=(t,), **common)(params)) for t in terms]
        whole = CV.covariance(spec, terms=terms, **common)
        c = np.asarray(whole(params))
        np.testing.assert_allclose(c, sum(parts), rtol=1e-10)
        np.testing.assert_allclose(c, c.T, rtol=1e-10)
        assert np.all(np.linalg.eigvalsh(c) > 0.0)
        assert np.all(np.diag(parts[1]) > 0.0)

    def test_the_shared_terms_are_refused_for_cl(self, setup):
        spec, common, _ = setup
        with pytest.raises(NotImplementedError):
            CV.covariance(spec, terms=("shared_pair",), **common)

    def test_the_connected_terms_need_a_projection_grid(self, setup):
        spec, common, params = setup
        cov = CV.covariance(spec, terms=("one_halo",), **(common | {"z_proj": ()}))
        with pytest.raises(ValueError, match="z_proj"):
            cov(params)
