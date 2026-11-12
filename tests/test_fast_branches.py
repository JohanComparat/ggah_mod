"""Fast checks of the refusals and the less travelled branches of layers 3 to 6.

Each test pins a refusal by its message, or a branch by a value it must take:
the galaxy sector's stellar masses for every kind of occupation, the
forward stellar-mass relations, the parameter containers, the calibration
policies, the survey geometry, the real-space covariance and ``make_model``.
"""
import warnings

import jax
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
from ggah_mod.halos.calibration import MismatchWarning
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.sectors import calibration as CAL
from ggah_mod.sectors import sham as SH
from ggah_mod.sectors.params import Flat, Gaussian, MultivariateGaussian, Param


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, B, pk, z=0.1, calibration="off")


# --------------------------------------------------------------------------
# the galaxy sector
# --------------------------------------------------------------------------
class TestGalaxySector:
    @pytest.mark.parametrize("kw, match", [
        (dict(model="hod5"), "unknown galaxy model"),
        (dict(model="zheng07", shmr="behroozi10"), "unknown SHMR"),
    ])
    def test_unknown_names_are_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            S.GalaxySector(**kw)

    def test_an_unknown_view_is_refused(self, field):
        g = S.GalaxySector("zheng07", backend=B)
        with pytest.raises(ValueError, match="unknown galaxy view"):
            g.effective_bias(field, S.galaxy_defaults("zheng07"), view="all")

    def test_a_missing_parameter_is_named(self, field):
        p = S.galaxy_defaults("zheng07")
        p.pop("alpha")
        with pytest.raises(ValueError, match="alpha"):
            S.GalaxySector("zheng07", backend=B).occupation(field, p)

    def test_a_model_without_a_relation_has_no_stellar_mass(self, field):
        g = S.GalaxySector("zheng07", backend=B)
        with pytest.raises(ValueError, match="shmr="):
            g.log10_mstar(np.array([12.0]), S.galaxy_defaults("zheng07"), h=PLANCK18.h)
        with pytest.raises(ValueError, match="no stellar-mass scatter"):
            g.log10_mstar_scatter(np.array([12.0]), S.galaxy_defaults("zheng07"),
                                  h=PLANCK18.h)

    @pytest.mark.parametrize("model, name, factor", [
        ("leauthaud12", "sigma_logmstar", 1.0),
        ("zacharegkas25", "width_logmstar", 1.0 / np.sqrt(2.0)),
    ])
    def test_the_scatter_in_each_convention(self, model, name, factor):
        g = S.GalaxySector(model, backend=B)
        p = S.galaxy_defaults(model)
        s = np.asarray(g.log10_mstar_scatter(np.array([11.0, 13.0]), p, h=PLANCK18.h))
        np.testing.assert_allclose(s, p[name] * factor, rtol=1e-12)

    def test_satellite_stars_of_a_non_threshold_model_need_f_sub(self, field):
        g = S.GalaxySector("zheng07", shmr="moster13", backend=B)
        p = S.galaxy_defaults("zheng07")
        with pytest.raises(ValueError, match="f_sub"):
            g.stellar_fraction(field, p, satellites=True)
        f_cen, f_sat = g.stellar_fraction(field, p | {"f_sub": 0.3}, satellites=True)
        assert np.all(np.asarray(f_sat) >= 0.0) and float(np.max(f_sat)) > 0.0
        assert float(np.max(f_cen)) < 0.1

    def test_the_central_csmf_on_a_given_grid(self, field):
        g = S.GalaxySector("zumandelbaum15", backend=B, calibration="off")
        p = S.galaxy_defaults("zumandelbaum15")
        lg, dn = g.central_csmf_msun(field, p)
        grid = np.linspace(9.5, 11.5, 9)
        lg2, dn2 = g.central_csmf_msun(field, p, lg_ms_msun=grid)
        np.testing.assert_allclose(np.asarray(lg2), grid)
        assert np.asarray(dn2).shape == (grid.size, field.n_m)
        assert np.all(np.asarray(dn2) >= 0.0)

    def test_a_mis_centred_galaxy_sample_carries_h_k(self, field):
        g = S.GalaxySector("zumandelbaum15", backend=B, calibration="off")
        p = S.galaxy_defaults("zumandelbaum15")
        w = g.weights(field, p | {"p_off": 0.4, "r_off": 0.3})
        assert np.asarray(w.w_point).shape == (field.k.size, field.n_m)
        np.testing.assert_allclose(np.asarray(w.w_point)[0],
                                   np.asarray(g.weights(field, p).w_point), rtol=1e-5)


# --------------------------------------------------------------------------
# the stellar-mass relations
# --------------------------------------------------------------------------
class TestStellarMassRelations:
    def test_moster13_is_its_normalisation_at_its_pivot(self):
        """At M_h = M_1 both power laws are one, so M_*/M_h = 2N/2 = N."""
        assert 10 ** float(SH.mstar_moster13(11.590) - 11.590) == pytest.approx(0.0351, rel=1e-12)
        z = 1.0
        a = z / (1 + z)
        assert 10 ** float(SH.mstar_moster13(11.590 + 1.195 * a, z=z) - (11.590 + 1.195 * a)) \
            == pytest.approx(0.0351 - 0.0247 * a, rel=1e-10)

    def test_the_early_type_fraction_rises_from_zero_to_one(self):
        f = np.asarray(SH.f_early_type(np.array([9.0, 12.5, 16.0])))
        assert f[0] < 1e-3 and f[2] > 1 - 1e-6
        assert f[1] == pytest.approx(1.0 - np.exp(-1.0))

    @pytest.mark.parametrize("includes", [True, False])
    def test_the_remnant_split_keeps_the_living_stars(self, includes):
        alive, rem = SH.split_stellar(1e10, remnant_fraction=0.2,
                                      includes_remnants=includes)
        total = float(alive) + float(rem)
        assert float(rem) / total == pytest.approx(0.2)
        assert (total if includes else float(alive)) == pytest.approx(1e10)


# --------------------------------------------------------------------------
# the parameter containers
# --------------------------------------------------------------------------
class TestParameterContainers:
    def test_a_joint_prior_has_its_correlation(self):
        mvg = MultivariateGaussian(("a", "b"), jnp.array([0.0, 1.0]),
                                   jnp.array([[1.0, 0.5], [0.5, 4.0]]))
        np.testing.assert_allclose(np.asarray(mvg.correlation()),
                                   [[1.0, 0.25], [0.25, 1.0]], rtol=1e-12)

    def test_clip_is_for_proposals(self):
        p = Param(0.5, (0.0, 1.0), Flat(), why="test")
        np.testing.assert_allclose(np.asarray(p.clip(jnp.array([-1.0, 0.3, 2.0]))),
                                   [0.0, 0.3, 1.0])

    def test_the_registry_views_and_the_pytree(self):
        cls = S.OCCUPATION_PARAMS["zheng07"]
        p = cls()
        assert cls.free_names() == tuple(cls._PARAMS)
        assert cls.defaults() == {n: q.default for n, q in cls._PARAMS.items()}
        assert set(cls.bounds()) == set(cls._PARAMS)
        back = jax.tree_util.tree_map(lambda x: x, p)
        assert back.as_dict() == p.as_dict()
        assert not p.replace(alpha=99.0).within_bounds()
        with pytest.raises(TypeError, match="unknown parameter"):
            p.replace(beta=1.0)

    @pytest.mark.parametrize("kw, err, match", [
        (dict(default=2.0, bounds=(0.0, 1.0)), ValueError, "outside its own bounds"),
        (dict(default=0.5, bounds=(0.0, 1.0), kind="guess"), ValueError, "kind"),
        (dict(default=0.5, bounds=(0.0, 1.0), why=""), ValueError, "why"),
    ])
    def test_a_malformed_param_is_refused(self, kw, err, match):
        kw = {"why": "test"} | kw
        with pytest.raises(err, match=match):
            Param(**kw)

    def test_a_gaussian_log_prior(self):
        cls = S.OCCUPATION_PARAMS["zheng07"]
        lp0 = float(cls().log_prior())
        mvg = MultivariateGaussian(("alpha", "log10m1"), jnp.array([1.1, 12.4]),
                                   jnp.eye(2) * 0.01)
        lp = float(cls().log_prior(joint=[mvg]))
        at_mean = float(cls(alpha=1.1).log_prior(joint=[mvg]))
        assert lp - at_mean == pytest.approx(-0.5 * 0.1 ** 2 / 0.01, rel=1e-10)
        assert lp0 == 0.0                      # every zheng07 prior is flat
        assert np.isfinite(float(Gaussian(0.0, 1.0).log_prob(0.5)))


# --------------------------------------------------------------------------
# the calibration policies
# --------------------------------------------------------------------------
class TestCalibrationPolicies:
    TABLE = {"powell": S.agn.AGN_CALIBRATION["powell"]}

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValueError, match="calibration must be one of"):
            CAL.check_sector_range("AGN", "powell", self.TABLE, [9.0], policy="loud")

    def test_off_unknown_and_inside_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CAL.check_sector_range("AGN", "powell", self.TABLE, [8.0], policy="off")
            CAL.check_sector_range("AGN", "other", self.TABLE, [8.0])
            CAL.check_sector_range("AGN", "powell", self.TABLE, [10.5, 11.5])
            CAL.check_sector_range("AGN", "powell", self.TABLE, [9.95], tolerance=0.1)

    def test_outside_warns_with_its_cost_and_strict_raises(self):
        with pytest.warns(MismatchWarning, match="factor 0.800"):
            CAL.check_sector_range("AGN", "powell", self.TABLE, [8.0], cost=0.8)
        with pytest.warns(MismatchWarning, match="read the docs"):
            CAL.check_sector_range("AGN", "powell", self.TABLE, [8.0], hint="read the docs")
        with pytest.raises(ValueError, match="extrapolation"):
            CAL.check_sector_range("AGN", "powell", self.TABLE, [8.0], policy="strict")


# --------------------------------------------------------------------------
# the survey geometry
# --------------------------------------------------------------------------
class TestSurveyGeometry:
    @pytest.mark.parametrize("kw, match", [
        (dict(area_sr=0.0), "not a solid angle"),
        (dict(area_sr=1.0, z_min=0.1), "both z_min and z_max"),
        (dict(area_sr=1.0, z_min=0.5, z_max=0.2), "not a slab"),
    ])
    def test_a_malformed_footprint_is_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            CV.SurveyGeometry("bad", **kw)

    def test_the_builders_refuse_what_is_not_a_sky(self):
        with pytest.raises(ValueError, match="fraction of the sky"):
            CV.spherical_cap(1.5)
        with pytest.raises(ValueError, match="latitude cut"):
            CV.galactic_latitude_cut(95.0)
        with pytest.raises(KeyError, match="unknown LS10 sample"):
            CV.ls10_sample("12.5")

    def test_a_slab_moves_and_the_volume_needs_one(self):
        cap = CV.spherical_cap(0.1, ell_max=32)
        with pytest.raises(ValueError):
            cap.volume(PLANCK18)
        v1 = float(cap.with_slab(0.1, 0.2).volume(PLANCK18))
        v2 = float(cap.with_slab(0.1, 0.3).volume(PLANCK18))
        assert 0.0 < v1 < v2
        assert cap.area_deg2 == pytest.approx(0.1 * CV.FULL_SKY_DEG2)


# --------------------------------------------------------------------------
# the real-space covariance
# --------------------------------------------------------------------------
class TestRealSpaceCovariance:
    @pytest.fixture(scope="class")
    def setup(self, pk, field):
        gal = S.GalaxySector("zheng07", backend=B)
        split = S.BaryonSplit.from_hot(S.cosmic_baryon_fraction(PLANCK18),
                                       jnp.full(field.n_m, 0.1))
        sectors = {"galaxies": gal, "matter": S.MatterField()}
        params = {"galaxies": S.galaxy_defaults("zheng07"), "matter": {"split": split}}
        edges = np.logspace(-0.5, 1.0, 4)
        rp = np.sqrt(edges[1:] * edges[:-1])
        geo = CV.ls10_sample("10.0")
        z, chi = OB.limber_grid(PLANCK18, z_max=1.5, n=6, backend=B)
        kernels = {"lens": OB.lensing_efficiency(
            z, chi, np.exp(-0.5 * ((np.asarray(z) - 0.8) / 0.2) ** 2), PLANCK18)}
        src = CV.SourceSample("s", 5e15, 0.28, 15.0, 700.0, 134.8, "lens", "test")
        common = dict(geometry=geo, edges={"wp": edges, "ds": edges},
                      fields_at=lambda zs: make_fields(PLANCK18, B, pk, zs),
                      sectors=sectors, cosmo=PLANCK18, backend=B,
                      tracer_noise={"galaxies": 1e-3}, sources={"ds": src},
                      lensing_geometry=CV.from_area(400.0, name="overlap",
                                                    z_min=geo.z_min, z_max=geo.z_max,
                                                    provenance="test"),
                      kernels=kernels, z_proj=np.asarray(z))
        return rp, edges, common, params

    def _spec(self, rp, z_ds=0.13, pi=60.0, b="matter"):
        return OB.ObservableSpec((OB.Wp("wp", "galaxies", "galaxies", rp, pi_max=pi, z=0.13),
                                  OB.DeltaSigma("ds", "galaxies", b, rp, z=z_ds)))

    def test_the_projected_matter_enters_the_lensing_block(self, setup):
        rp, _, common, params = setup
        with_kappa = np.asarray(CV.gaussian_covariance(self._spec(rp), **common)(params))
        src = common["sources"]["ds"]
        bare = CV.SourceSample("s", src.sigma_crit, src.sigma_e, src.n_eff_arcmin2,
                               src.delta_pi2, src.delta_pi_cross, None, "test")
        without = np.asarray(CV.gaussian_covariance(
            self._spec(rp), **(common | {"sources": {"ds": bare}}))(params))
        np.testing.assert_allclose(with_kappa, with_kappa.T, rtol=1e-10)
        assert np.all(np.diag(with_kappa)[3:] > np.diag(without)[3:])
        np.testing.assert_allclose(with_kappa[:3, :3], without[:3, :3], rtol=1e-10)

    @pytest.mark.parametrize("change, err, match", [
        (dict(edges={"ds": np.logspace(-0.5, 1.0, 4)}), ValueError, "no bin edges"),
        (dict(edges={"wp": np.logspace(-0.5, 1.0, 3), "ds": np.logspace(-0.5, 1.0, 4)}),
         ValueError, "edges for"),
        (dict(edges={"wp": np.logspace(0.5, 1.5, 4), "ds": np.logspace(-0.5, 1.0, 4)}),
         ValueError, "outside its edges"),
    ])
    def test_bins_are_checked(self, setup, change, err, match):
        rp, _, common, _ = setup
        with pytest.raises(err, match=match):
            CV.gaussian_covariance(self._spec(rp), **(common | change))

    @pytest.mark.parametrize("spec_kw, err, match", [
        (dict(b="galaxies"), ValueError, "must be 'matter'"),
        (dict(z_ds=0.3), NotImplementedError, "one lens slab"),
    ])
    def test_what_one_slab_cannot_hold_is_refused(self, setup, spec_kw, err, match):
        rp, _, common, _ = setup
        with pytest.raises(err, match=match):
            CV.gaussian_covariance(self._spec(rp, **spec_kw), **common)

    def test_two_line_of_sight_windows_are_refused(self, setup):
        rp, edges, common, _ = setup
        spec = OB.ObservableSpec((
            OB.Wp("wp", "galaxies", "galaxies", rp, pi_max=60.0, z=0.13),
            OB.Wp("wp2", "galaxies", "galaxies", rp, pi_max=100.0, z=0.13)))
        with pytest.raises(NotImplementedError, match="pi_max"):
            CV.gaussian_covariance(spec, **(common | {"edges": {"wp": edges, "wp2": edges}}))

    def test_two_source_samples_are_refused(self, setup):
        rp, edges, common, _ = setup
        src = common["sources"]["ds"]
        other = CV.SourceSample("t", src.sigma_crit, src.sigma_e, src.n_eff_arcmin2,
                                src.delta_pi2, src.delta_pi_cross, None, "test")
        spec = OB.ObservableSpec((
            OB.DeltaSigma("ds", "galaxies", "matter", rp, z=0.13),
            OB.DeltaSigma("ds2", "galaxies", "matter", rp, z=0.13)))
        with pytest.raises(NotImplementedError, match="different source samples"):
            CV.gaussian_covariance(spec, **(common | {
                "edges": {"ds": edges, "ds2": edges},
                "sources": {"ds": src, "ds2": other}}))

    def test_an_empty_covariance_has_size_zero(self):
        assert CV.Covariance((), (), None, (), (), None).size == 0


# --------------------------------------------------------------------------
# make_model
# --------------------------------------------------------------------------
class TestMakeModel:
    @pytest.fixture(scope="class")
    def model_parts(self, pk, field):
        gal = S.GalaxySector("zheng07", backend=B)
        z, chi = OB.limber_grid(PLANCK18, z_max=1.0, n=4, backend=B)
        nz = np.exp(-0.5 * ((np.asarray(z) - 0.4) / 0.1) ** 2)
        spec = OB.ObservableSpec((
            OB.Xi("xi", "galaxies", "galaxies", [1.0, 5.0], z=0.1),
            OB.Cl("cl", "galaxies", "galaxies", [100.0, 300.0],
                  kernel_a="counts", kernel_b="counts")))
        return gal, spec, z, chi, nz, np.asarray(field.k)

    def test_a_traced_cosmology_and_a_kernel_of_the_cosmology(self, pk, model_parts):
        gal, spec, z, chi, nz, k = model_parts
        kernel = lambda c: OB.number_counts(z, OB.limber_grid(c, z_max=1.0, n=4, backend=B)[1], nz)
        model = OB.make_model(
            spec, fields_at=lambda zs, c=PLANCK18: make_fields(c, B, pk, zs),
            sectors={"galaxies": gal}, kernels={"counts": kernel}, cosmo=PLANCK18,
            backend=B, z_proj=np.asarray(z), k=k)
        p = {"galaxies": S.galaxy_defaults("zheng07")}
        _, v0 = model(p)
        _, v1 = model(p, PLANCK18)
        np.testing.assert_allclose(np.asarray(v1), np.asarray(v0), rtol=1e-10)
        _, v2 = model(p, PLANCK18.replace(Omega_m=0.33))
        assert not np.allclose(np.asarray(v2), np.asarray(v0))

    def test_fields_and_wavenumbers_must_match_the_plan(self, pk, model_parts):
        gal, spec, z, chi, nz, k = model_parts
        kw = dict(sectors={"galaxies": gal},
                  kernels={"counts": OB.number_counts(z, chi, nz)},
                  cosmo=PLANCK18, backend=B, z_proj=np.asarray(z))
        p = {"galaxies": S.galaxy_defaults("zheng07")}
        short = OB.make_model(spec, fields_at=lambda zs: make_fields(PLANCK18, B, pk, zs[:-1]), **kw)
        with pytest.raises(ValueError, match="fields for"):
            short(p)
        bad_k = OB.make_model(spec, fields_at=lambda zs: make_fields(PLANCK18, B, pk, zs),
                              k=k[:-1], **kw)
        with pytest.raises(ValueError, match="wavenumbers"):
            bad_k(p)
        no_grid = OB.make_model(spec, fields_at=lambda zs: make_fields(PLANCK18, B, pk, zs),
                                **(kw | {"z_proj": ()}))
        with pytest.raises(ValueError, match="projection grid"):
            no_grid(p)


# --------------------------------------------------------------------------
# the dispersion measure
# --------------------------------------------------------------------------
class TestDispersionMeasure:
    def test_the_mean_electron_density_is_the_baryons_over_mu_e_m_p(self):
        from ggah_mod.sectors.gas import M_PROTON_G, MU_E
        n_e0 = float(OB.mean_electron_density(PLANCK18))
        h0 = 100.0 * PLANCK18.h * 1e5 / 3.0856775814913673e24          # s^-1
        rho_crit = 3.0 * h0 ** 2 / (8.0 * np.pi * 6.6743e-8)             # g cm^-3
        rho_b = PLANCK18.Omega_b * rho_crit
        assert n_e0 == pytest.approx(rho_b / (MU_E * M_PROTON_G), rel=1e-3)
        assert float(OB.mean_electron_density(PLANCK18, f_e=0.5)) == pytest.approx(0.5 * n_e0)

    def test_the_low_redshift_limit_and_the_kernel_integral(self):
        from ggah_mod.cosmology.constants import C_KM_S, MPC_CM
        from ggah_mod.observables.kernels import PC_CM
        n_e0 = float(OB.mean_electron_density(PLANCK18))
        dm = float(OB.mean_dispersion_measure(1e-3, PLANCK18)[0])
        hubble_pc = C_KM_S / (100.0 * PLANCK18.h) * MPC_CM / PC_CM
        assert dm == pytest.approx(n_e0 * hubble_pc * 1e-3, rel=2e-3)
        z, chi = OB.limber_grid(PLANCK18, z_max=1.0, n=400, backend=B)
        w = OB.dispersion_measure(z, chi, PLANCK18)
        integral = float(np.trapezoid(np.asarray(w.w), np.asarray(chi)))
        assert integral == pytest.approx(
            float(OB.mean_dispersion_measure(1.0, PLANCK18)[0]), rel=2e-3)


# --------------------------------------------------------------------------
# small refusals and branches of layer 4
# --------------------------------------------------------------------------
class TestLayerFourBranches:
    @pytest.mark.parametrize("kw, match", [
        (dict(point="half"), "Overlap.point"),
        (dict(shared="some"), "Overlap.shared"),
        (dict(why="  "), None),
    ])
    def test_a_malformed_overlap_is_refused(self, kw, match):
        from ggah_mod.spectra.spec import Overlap
        with pytest.raises(ValueError, match=match):
            Overlap(**({"why": "test"} | kw))

    def test_options_that_count_the_neutrinos_twice_are_refused(self):
        import ggah_mod.spectra as SP
        with pytest.raises(ValueError):
            SP.PkOptions(two_halo_spectrum="total", neutrino_two_halo="linear")

    def test_an_unlabelled_discrete_component_is_refused(self):
        from ggah_mod.spectra.spec import Component, overlap_of
        with pytest.raises(ValueError):
            overlap_of(Component("galaxies", "total"), Component("galaxies", "cen"),
                       a_discrete=True, b_discrete=True)
        assert overlap_of(Component("agn", "counts", population="agn"),
                          Component("agn", "emission", population="agn"),
                          a_discrete=True, b_discrete=True) == "identical"

    def test_the_spectrum_parts_and_the_total_linear_spectrum(self, field):
        import ggah_mod.spectra as SP
        gal = S.GalaxySector("zheng07", backend=B)
        w = gal.weights(field, S.galaxy_defaults("zheng07"))
        p = SP.pk_cross(field, w, w, overlap="identical", options=SP.PkOptions(bnl=False))
        np.testing.assert_allclose(np.asarray(p.with_shot()),
                                   np.asarray(p.total) + float(p.shot), rtol=1e-14)
        opts = SP.PkOptions(two_halo_spectrum="total", neutrino_two_halo="none", bnl=False)
        np.testing.assert_array_equal(np.asarray(SP.linear_spectrum(field, opts)),
                                      np.asarray(field.pk_lin))
        bare = np.asarray(SP.i_of_k(field, w, consistency="none"))
        assert bare.shape == (field.k.size,)

    def test_the_baryon_split_is_a_pytree_and_retains_its_baryons(self, field):
        split = S.BaryonSplit.from_retained(S.cosmic_baryon_fraction(PLANCK18),
                                            jnp.full(field.n_m, 0.1), f_star_cen=0.01)
        back = jax.tree_util.tree_map(lambda x: x, split)
        np.testing.assert_allclose(np.asarray(back.f_ejected), np.asarray(split.f_ejected))
        np.testing.assert_allclose(np.asarray(split.f_retained), 0.1, rtol=1e-12)
        assert float(np.asarray(split.replace(f_cold=0.0).f_cold)) == 0.0
