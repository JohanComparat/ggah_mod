"""Fast checks of layer-3 to layer-5 paths the slow tests alone used to reach.

Mis-centring, the cosmic baryon census, the ejected gas, the pair rule under a
declared :class:`~ggah_mod.spectra.spec.Overlap`, the tracer registry's
refusals and the validation of an :class:`~ggah_mod.observables.spec.Statistic`,
each on the coarse flavour, each asserting a closed-form limit, an identity or
a refusal rather than only running the code.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

import ggah_mod.observables as OB
import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE_COARSE as B
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors.gas import MU_E
from ggah_mod.spectra.pair import pair_1h, shot_noise
from ggah_mod.spectra.spec import Component, Overlap, TracerSpec


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, B, make_pk("emu_pk"), z=0.2, calibration="off")


@pytest.fixture(scope="module")
def gal():
    return S.GalaxySector("zumandelbaum15", backend=B, calibration="off")


@pytest.fixture(scope="module")
def gp():
    return S.galaxy_defaults("zumandelbaum15")


@pytest.fixture(scope="module")
def split(field):
    """A hot fraction of 0.1 everywhere, no stars: a split built on the
    cosmic baryon fraction, so the census closes."""
    return S.BaryonSplit.from_hot(S.cosmic_baryon_fraction(PLANCK18),
                                  jnp.full(field.n_m, 0.1))


# --------------------------------------------------------------------------
# mis-centring
# --------------------------------------------------------------------------
class TestMisCentering:
    def test_an_unknown_convention_is_refused(self):
        with pytest.raises(ValueError, match="mis-centering convention"):
            S.MisCentering("hikage")

    def test_the_two_offset_scales(self, field):
        p = S.MisCenteringParams(p_off=0.3, r_off=0.4, sigma_off=0.7)
        np.testing.assert_allclose(S.MisCentering("more15").offset_scale(field, p),
                                   0.4 * np.asarray(field.r_s), rtol=1e-14)
        np.testing.assert_allclose(
            S.MisCentering("johnston07").offset_scale(field, p), 0.7, rtol=0)

    @pytest.mark.parametrize("convention", ["more15", "johnston07"])
    def test_h_is_one_on_large_scales_and_one_minus_p_on_small(self, field,
                                                               convention):
        h = np.asarray(S.MisCentering(convention).h_k(
            field, S.MisCenteringParams(p_off=0.3)))
        assert h.shape == (field.k.size, field.n_m)
        np.testing.assert_allclose(h[0], 1.0, atol=1e-6)
        assert np.all((h >= 0.7 - 1e-12) & (h <= 1.0 + 1e-12))
        # the largest halo is displaced furthest and reaches 1 - p_off
        assert h[-1, -1] == pytest.approx(0.7, abs=1e-6)

    def test_apply_moves_the_point_part_and_keeps_the_density(self, field, gal, gp):
        w = gal.weights(field, gp, view="cen")
        out = S.MisCentering().apply(w, field, S.MisCenteringParams(p_off=0.3))
        assert np.asarray(out.w_point).shape == (field.k.size, field.n_m)
        assert float(out.norm) == float(w.norm)
        np.testing.assert_allclose(np.asarray(out.w_point)[0],
                                   np.asarray(w.w_point), rtol=1e-5)

    def test_apply_refuses_a_tracer_with_no_point_part(self, field, gal, gp):
        with pytest.raises(ValueError, match="no point component"):
            S.MisCentering().apply(gal.weights(field, gp, view="sat"), field,
                                   S.MisCenteringParams(p_off=0.3))


# --------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------
class TestCensus:
    def test_the_budget_closes(self, field, split):
        c = S.census(field, split)
        assert abs(float(c.closure_residual())) < 1e-6
        assert float(c.omega_total) == pytest.approx(PLANCK18.Omega_b, rel=1e-12)
        assert sum(float(v) for v in c.fractions().values()) == pytest.approx(1.0, rel=1e-12)

    def test_the_outside_row_is_diffuse_when_the_split_has_no_stars(self, field, split):
        out = S.census(field, split).outside_fractions()
        assert float(out["diffuse"]) > 0.0
        for phase in ("star_cen", "star_sat", "cold"):
            assert float(out[phase]) == 0.0

    def test_a_higher_cut_holds_fewer_baryons(self, field, split):
        by_cut = S.census_over_mass_ranges(field, split, m_mins=(1e10, 1e12, 1e14))
        held = [float(by_cut[m].omega_halos) for m in (1e10, 1e12, 1e14)]
        assert held[0] > held[1] > held[2] > 0.0
        assert float(S.census(field, split, m_max=1e12).omega_halos) < held[0]

    def test_the_census_is_a_pytree(self, field, split):
        import jax
        c = S.census(field, split)
        back = jax.tree_util.tree_map(lambda x: x, c)
        assert float(back.omega_halos) == float(c.omega_halos)
        assert back.omega_b == c.omega_b


# --------------------------------------------------------------------------
# the ejected gas
# --------------------------------------------------------------------------
class TestEjecta:
    def test_the_weight_tends_to_the_ejected_mass(self, field, split):
        ej = S.EjectaSector()
        w = ej.weights(field, {"split": split})
        np.testing.assert_allclose(np.asarray(w.w_extended)[0],
                                   np.asarray(ej.mass(field, split)), rtol=1e-6)
        assert not w.discrete
        np.testing.assert_allclose(np.asarray(ej.electron_count(field, split)),
                                   np.asarray(ej.mass(field, split)) / MU_E, rtol=1e-14)

    def test_a_larger_shell_is_suppressed_at_a_smaller_k(self, field, split):
        ej = S.EjectaSector()
        u2 = np.asarray(ej.weights(field, {"split": split, "eta_ej": 2.0}).w_extended)
        u4 = np.asarray(ej.weights(field, {"split": split,
                                           "params": S.EjectaParams(eta_ej=4.0)}).w_extended)
        i = np.searchsorted(np.asarray(field.k), 1.0)
        assert np.all(u4[i] <= u2[i])

    def test_the_unresolved_matter_is_the_diffuse_share(self, field, split):
        ej = S.EjectaSector()
        f_diffuse = float(ej.diffuse_fraction(split))
        assert f_diffuse == pytest.approx(
            float(jnp.atleast_1d(split.f_hot)[0] + jnp.atleast_1d(split.f_ejected)[0]))
        uk = np.ones((field.k.size, field.n_m))
        np.testing.assert_allclose(np.asarray(ej.unresolved_weight(field, split, uk)),
                                   f_diffuse * float(field.rho_cold), rtol=1e-12)

    @pytest.mark.parametrize("params, match", [
        ({}, "needs `split`"),
        (S.EjectaParams(), "needs `split`"),
    ])
    def test_a_missing_split_is_refused(self, field, params, match):
        with pytest.raises(ValueError, match=match):
            S.EjectaSector().weights(field, params)

    def test_an_unknown_view_is_refused(self, field, split):
        with pytest.raises(ValueError, match="unknown ejecta view"):
            S.EjectaSector().weights(field, {"split": split}, view="pressure")


# --------------------------------------------------------------------------
# the pair rule and the shot noise under a declared overlap
# --------------------------------------------------------------------------
class TestDeclaredOverlap:
    @pytest.fixture(scope="class")
    def w(self, field, gal, gp):
        return {v: gal.weights(field, gp, view=v) for v in ("total", "cen", "sat")}

    def test_disjoint_points_give_the_product(self, field, w):
        n = field.k.size
        got = pair_1h(w["total"], w["total"], n,
                      overlap=Overlap(point="disjoint", why="two samples"))
        t = (np.asarray(w["total"].w_point)[None, :]
             + np.asarray(w["total"].w_extended)) / float(w["total"].norm)
        np.testing.assert_allclose(np.asarray(got), t * t, rtol=1e-12)

    def test_one_central_per_halo_is_not_a_pair(self, field, w):
        n = field.k.size
        same = pair_1h(w["total"], w["total"], n,
                       overlap=Overlap(point="same_object", why="one central"))
        prod = pair_1h(w["total"], w["total"], n,
                       overlap=Overlap(point="disjoint", why="two samples"))
        p = np.asarray(w["total"].w_point) / float(w["total"].norm)
        np.testing.assert_allclose(np.asarray(prod) - np.asarray(same),
                                   np.broadcast_to(p * p, (n, p.size)), rtol=1e-10)
        # with a side that has no point part, nothing is subtracted
        sat = pair_1h(w["total"], w["sat"], n,
                      overlap=Overlap(point="same_object", why="test"))
        disj = pair_1h(w["total"], w["sat"], n,
                       overlap=Overlap(point="disjoint", why="test"))
        np.testing.assert_allclose(np.asarray(sat), np.asarray(disj), rtol=0)

    def test_the_auto_pair_rule_by_population(self, field, w):
        n = field.k.size
        assert pair_1h(w["cen"], w["cen"], n, overlap="identical") is None
        e = np.asarray(w["sat"].w_extended) / float(w["sat"].norm)
        np.testing.assert_allclose(
            np.asarray(pair_1h(w["sat"], w["sat"], n, overlap="identical")),
            e * e, rtol=1e-12)

    def test_shared_points_give_the_point_shot_noise(self, field, w):
        shared = Overlap(point="same_object", shared="point", why="same centrals")
        got = float(shot_noise(field, w["cen"], w["cen"], overlap=shared))
        n_c = np.asarray(w["cen"].w_point)
        want = float(field.integrate(field.dndm * n_c * n_c)) / float(w["cen"].norm) ** 2
        assert got == pytest.approx(want, rel=1e-10)
        # <N_c^2> <= <N_c> for a central occupation, so it is below 1/n
        assert got < 1.0 / float(w["cen"].norm)

    def test_a_subset_gives_the_larger_samples_inverse_density(self, field, w):
        nested = Overlap(point="same_object", shared="subset", why="nested thresholds")
        got = float(shot_noise(field, w["cen"], w["total"], overlap=nested))
        want = float(field.integrate(field.dndm * np.minimum(
            np.asarray(w["cen"].self_pair), np.asarray(w["total"].self_pair)))) / (
            float(w["cen"].norm) * float(w["total"].norm))
        assert got == pytest.approx(want, rel=1e-10)
        assert got == pytest.approx(1.0 / float(w["total"].norm), rel=1e-6)

    def test_a_shared_point_needs_a_point_on_both_sides(self, field, w):
        with pytest.raises(ValueError, match="point component"):
            shot_noise(field, w["cen"], w["sat"],
                       overlap=Overlap(point="disjoint", shared="point", why="test"))

    def test_a_shared_point_refuses_a_k_dependent_point(self, field, w):
        moved = S.MisCentering().apply(w["cen"], field, S.MisCenteringParams(p_off=0.3))
        with pytest.raises(ValueError, match="wavenumber-independent"):
            shot_noise(field, moved, w["cen"],
                       overlap=Overlap(point="same_object", shared="point", why="test"))

    def test_a_subset_needs_a_self_pair(self, field, w):
        bare = w["cen"].replace(self_pair=None)
        with pytest.raises(ValueError, match="self_pair"):
            shot_noise(field, bare, w["cen"],
                       overlap=Overlap(point="same_object", shared="subset", why="test"))


# --------------------------------------------------------------------------
# the tracer registry
# --------------------------------------------------------------------------
class TestTracerRegistry:
    def test_resolve_passes_specs_and_components_through(self):
        spec = TracerSpec.one("galaxies", "cen", population="galaxies/cen")
        assert SP.resolve(spec) is spec
        comp = Component("gas", "pressure")
        assert SP.resolve(comp).components == (comp,)
        with pytest.raises(ValueError, match="lensing"):
            SP.resolve("shear")

    def test_the_agn_needs_the_galaxy_parameters(self, field, gal, gp):
        agn = S.AgnSector(gal, calibration="off")
        with pytest.raises(ValueError, match="needs 'galaxies' parameters"):
            SP.spectrum(field, "agn", "agn", {"agn": agn, "galaxies": gal},
                        {"agn": S.AgnParams()})

    def test_an_unknown_agn_view_is_refused(self, field, gal, gp):
        agn = S.AgnSector(gal, calibration="off")
        with pytest.raises(ValueError, match="unknown AGN view"):
            SP.build_weights(Component("agn", "flux", population="agn"), field,
                             {"agn": agn, "galaxies": gal},
                             {"agn": S.AgnParams(), "galaxies": gp})

    def test_the_two_agn_branches_are_one_population(self, field, gal, gp):
        agn = S.AgnSector(gal, calibration="off", obscuration="split")
        sectors = {"agn": agn, "galaxies": gal}
        params = {"agn": S.AgnParams(), "galaxies": gp,
                  "obscuration": S.ObscurationParams()}
        a = TracerSpec.one("agn", "emission_obscured", population="agn/obs",
                           band=SP.Band(0.5, 2.0))
        b = TracerSpec.one("agn", "emission_unobscured", population="agn/unobs",
                           band=SP.Band(0.5, 2.0))
        p = SP.spectrum(field, a, b, sectors, params,
                        options=SP.PkOptions(bnl=False))
        assert np.all(np.isfinite(np.asarray(p.total)))

    def test_electrons_from_two_models_warn(self, field, split):
        sectors = {"gas": S.HotGasDPM(backend=B), "ejecta": S.EjectaSector()}
        params = {"gas": S.DpmParams(), "ejecta": {"split": split}}
        with pytest.warns(RuntimeWarning, match="do not close"):
            SP.spectrum(field, "electrons", "electrons", sectors, params,
                        options=SP.PkOptions(bnl=False))

    def test_the_suppression_is_one_with_every_component_on_the_dark_matter(self, field, split):
        k, s = SP.matter_suppression(field, {"matter": S.MatterField()},
                                     {"matter": {"split": split}},
                                     options=SP.PkOptions(bnl=False))
        np.testing.assert_allclose(np.asarray(s), 1.0, rtol=1e-12)
        with pytest.raises(ValueError, match="matter sector"):
            SP.matter_suppression(field, {}, {})


# --------------------------------------------------------------------------
# the specification of a statistic
# --------------------------------------------------------------------------
class TestStatisticValidation:
    @pytest.mark.parametrize("kw, match", [
        (dict(kind="power"), "unknown statistic kind"),
        (dict(x=()), "empty abscissa"),
        (dict(kind="xi", z=None), "needs a `z`"),
        (dict(kind="cl", z=None), "radial kernel"),
    ])
    def test_a_malformed_statistic_is_refused(self, kw, match):
        base = dict(name="s", kind="xi", a="galaxies", b="galaxies", x=(1.0,), z=0.2)
        with pytest.raises(ValueError, match=match):
            OB.Statistic(**(base | kw))

    def test_duplicate_names_are_refused(self):
        s = OB.Xi("s", "galaxies", "galaxies", [1.0], z=0.2)
        with pytest.raises(ValueError, match="share a name"):
            OB.ObservableSpec((s, s))

    def test_the_tau_ksz_grid_reaches_past_the_outer_aperture(self):
        s = OB.TauKsz("t", "electrons", "galaxies", [1.0, 2.0], z=0.3)
        assert min(s.ell) == pytest.approx(0.1)
        assert max(s.ell) == pytest.approx(np.sqrt(2.0) * 2.0)
        given = OB.TauKsz("t", "electrons", "galaxies", [1.0], z=0.3, r_grid=[0.5, 3.0])
        assert given.ell == (0.5, 3.0)

    def test_the_plan_reads_a_projected_pair_at_every_node(self):
        spec = OB.ObservableSpec((
            OB.Wp("wp", "galaxies", "galaxies", [1.0], pi_max=60.0, z=0.2),
            OB.Cl("cl", "galaxies", "matter", [100.0], kernel_a="a", kernel_b="b"),
        ))
        plan = spec.resolve(z_proj=(0.1, 0.3))
        assert spec.required_tracers() == ("galaxies", "matter")
        assert plan.z_nodes == (0.1, 0.2, 0.3)
        assert len(plan.needs) == 3

    def test_make_model_refuses_the_one_redshift_builder(self):
        spec = OB.ObservableSpec((OB.Xi("s", "galaxies", "galaxies", [1.0], z=0.2),))
        with pytest.raises(ValueError, match="field_at"):
            OB.make_model(spec, fields_at=None, sectors={}, kernels={},
                          cosmo=PLANCK18, field_at=lambda z: None)
