r"""The documented layers' edges: refusals, accessors, and the paths nothing else takes.

Layers 1 and 2 are the public, documented surface of 1.0.0, and a refusal is
part of that contract: a request the model cannot honour raises rather than
returning an approximation.  Each test below pins one such branch, or one small
accessor, that the rest of the suite reached only indirectly or not at all.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18, Cosmology


# ==========================================================================
# The package and its flavours
# ==========================================================================
class TestTheFlavours:
    def test_a_retired_flavour_name_still_resolves_with_a_warning(self):
        import ggah_mod
        from ggah_mod.backend import Backend

        with pytest.warns(DeprecationWarning):
            flavour = ggah_mod.FAST
        assert isinstance(flavour, Backend)

    def test_an_unknown_attribute_is_an_attribute_error(self):
        import ggah_mod

        with pytest.raises(AttributeError):
            ggah_mod.no_such_flavour

    def test_the_beyond_linear_switch_is_a_switch(self):
        from ggah_mod import ACCURATE

        with pytest.raises(ValueError, match="it is a switch"):
            ACCURATE.with_(bnl=1)

    def test_the_spectrum_backend_can_be_chosen_by_environment(self, monkeypatch):
        from ggah_mod.backend import PK_ENV_VAR, resolve_backend

        monkeypatch.setenv(PK_ENV_VAR, "camb")
        assert resolve_backend("accurate").pk == "camb"

    def test_an_unknown_spectrum_backend_in_the_environment_is_refused(
            self, monkeypatch):
        from ggah_mod.backend import PK_ENV_VAR, resolve_backend

        monkeypatch.setenv(PK_ENV_VAR, "halofit")
        with pytest.raises(ValueError, match="not a linear P"):
            resolve_backend("accurate")


# ==========================================================================
# Layer 1
# ==========================================================================
class TestLayerOneEdges:
    def test_a_negative_neutrino_mass_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            Cosmology.create(sum_mnu=-0.1, nu_hierarchy="degenerate")

    def test_the_hubble_distance_in_mpc_divides_by_h(self):
        c = PLANCK18
        assert float(c.hubble_distance_mpc) == pytest.approx(
            float(c.hubble_distance) / c.h, rel=1e-14)

    def test_the_distance_modulus_uses_mpc_not_mpc_over_h(self):
        from ggah_mod.cosmology import distance_modulus, luminosity_distance

        z = np.array([0.1, 1.0])
        want = 5.0 * np.log10(np.asarray(luminosity_distance(z, PLANCK18))
                              / PLANCK18.h * 1e6 / 10.0)
        assert np.allclose(distance_modulus(z, PLANCK18), want, rtol=0, atol=1e-10)

    def test_the_spread_accepts_its_own_wavenumbers(self):
        from ggah_mod.cosmology import growth_scale_spread, make_pk
        from ggah_mod.cosmology.growth import SPREAD_K_RANGE

        pk = make_pk("emu_pk")
        lo, hi = SPREAD_K_RANGE
        k = np.logspace(np.log10(lo), np.log10(hi), 512)
        assert float(growth_scale_spread(0.5, PLANCK18, pk, k=k)) == pytest.approx(
            float(growth_scale_spread(0.5, PLANCK18, pk)), rel=1e-12)

    def test_the_x64_guard_refuses_in_single_precision(self):
        from ggah_mod.numerics import require_x64

        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(RuntimeError, match="the answer"):
                require_x64("the answer", order=1e-8, in_range="nothing else")
        finally:
            jax.config.update("jax_enable_x64", True)


class TestTheSpectrumBackends:
    def test_a_boltzmann_base_asks_for_nothing_and_solves_nothing(self):
        from ggah_mod.cosmology.power import _BoltzmannBase

        base = _BoltzmannBase()
        assert base.precision == {}
        with pytest.raises(NotImplementedError):
            base._solve_uncached((), ())

    def test_camb_is_handed_cpl_dark_energy(self):
        pytest.importorskip("camb")
        from ggah_mod.cosmology.power import camb_input

        pars = camb_input(PLANCK18.replace(w0=-0.9, wa=0.1))
        assert "PPF" in type(pars.DarkEnergy).__name__
        assert pars.DarkEnergy.w == pytest.approx(-0.9)
        assert pars.DarkEnergy.wa == pytest.approx(0.1)

    def test_an_unknown_camb_accuracy_key_is_refused(self):
        pytest.importorskip("camb")
        from ggah_mod.cosmology.power import camb_input

        with pytest.raises(KeyError, match="not a CAMB accuracy parameter"):
            camb_input(PLANCK18, precision={"NotAnAccuracySetting": 2})

    def test_an_emulator_input_with_no_value_here_is_refused(self, monkeypatch):
        """``emu_pk`` refuses a box input its checkpoint lacks when the backend
        is built; this is the second guard, for an input this package cannot
        supply, so the box grows after construction."""
        from emu_pk import box

        from ggah_mod.cosmology import make_pk

        pk = make_pk("emu_pk")
        monkeypatch.setattr(box, "PARAMS", tuple(box.PARAMS) + ("mystery",))
        with pytest.raises(ValueError, match="no value for"):
            pk._params(PLANCK18)


# ==========================================================================
# Layer 2
# ==========================================================================
@pytest.fixture(scope="module")
def emu():
    from ggah_mod.cosmology import make_pk
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(emu):
    from ggah_mod import DIFFERENTIABLE
    from ggah_mod.halos.field import make_field
    return make_field(PLANCK18, DIFFERENTIABLE, emu, z=0.5)


class TestTheHaloField:
    def test_the_physical_radius_is_the_comoving_one_over_one_plus_z(self, field):
        assert np.allclose(field.r_delta_physical,
                           np.asarray(field.r_delta) / 1.5, rtol=1e-14)

    def test_the_circular_velocity_is_gm_over_the_physical_radius(self, field):
        from ggah_mod.cosmology import constants as C

        want = C.G_MPC_KMS2_MSUN * np.asarray(field.m) / np.asarray(field.r_delta_physical)
        assert np.allclose(field.v_delta_squared, want, rtol=1e-14)

    def test_replace_returns_a_field_with_the_change(self, field):
        doubled = field.replace(bias=2.0 * field.bias)
        assert type(doubled) is type(field)
        assert np.allclose(doubled.bias, 2.0 * np.asarray(field.bias))
        assert doubled.hmf_model == field.hmf_model


class TestLayerTwoEdges:
    def test_the_reference_density_of_a_definition(self):
        from ggah_mod.halos.mass_definitions import MassDef, rho_reference

        assert rho_reference("200m", 0.0, PLANCK18) == MassDef.from_string(
            "200m").delta_rho(0.0, PLANCK18)

    def test_a_calibration_note_with_an_empty_part_still_parses(self):
        from ggah_mod.halos.calibration import calibrated_mass_defs
        from ggah_mod.halos.mass_definitions import parse_mass_def

        assert calibrated_mass_defs("SO 200m//vir") == frozenset(
            {parse_mass_def("200m"), parse_mass_def("vir")})

    def test_klypin16_has_a_virial_table(self):
        from ggah_mod.halos.concentration import c_klypin16

        m = np.array([1e12, 1e14])
        c_vir, c_200c = c_klypin16(m, 0.0, mdef="vir"), c_klypin16(m, 0.0, mdef="200c")
        assert np.all(np.isfinite(c_vir)) and np.all(np.asarray(c_vir) > np.asarray(c_200c))

    def test_diemer19_refuses_an_unknown_statistic(self):
        from ggah_mod.halos.concentration import c_diemer19

        with pytest.raises(ValueError, match="statistic"):
            c_diemer19(jnp.array([1.0]), -2.0, 1.0, statistic="mode")

    def test_seppi21_takes_one_redshift_where_it_says_so(self):
        from ggah_mod.halos.concentration import seppi21_shape

        with pytest.raises(ValueError, match="one redshift"):
            seppi21_shape(jnp.array([1.0]), jnp.array([0.0, 0.5]))

    def test_seppi21_weights_stack_over_redshifts(self):
        from ggah_mod.halos.concentration import _s21_z_weights

        z = jnp.array([0.0, 0.5, 1.0])
        w = _s21_z_weights(z)
        assert w.shape[0] == 3
        assert np.allclose(np.asarray(w).sum(axis=-1), 1.0)
        assert np.allclose(w[1], _s21_z_weights(0.5))

    def test_the_gnfw_density_is_nfw_at_one_one_three(self):
        from ggah_mod.halos.profiles import gnfw_rho, nfw_rho

        r = np.logspace(-3, 0, 50)
        assert np.allclose(gnfw_rho(r, 1.0e14, 0.1), nfw_rho(r, 1.0e14, 0.1), rtol=1e-12)

    def test_a_fitting_function_backend_has_the_layers_sigma(self, emu):
        from ggah_mod.halos.mass_function import FittingFunctionHMF
        from ggah_mod.halos.variance import sigma_of_mass

        hmf = FittingFunctionHMF(emu)
        m = np.logspace(11, 15, 5)
        k = hmf._k
        want = sigma_of_mass(m, k, emu.pk_cb(k, 0.0, PLANCK18), PLANCK18.rho_cold)
        assert np.allclose(hmf.sigma(m, 0.0, PLANCK18), want, rtol=1e-12)


# ==========================================================================
# Layer 2: the beyond-linear bias table
# ==========================================================================
@pytest.fixture(scope="module")
def spectrum(emu):
    k = np.logspace(-4, 2, 256)
    return k, np.asarray(emu.pk_cb(k, 0.0, PLANCK18))


class TestTheBetaNLTable:
    def test_a_missing_table_names_how_to_rebuild_it(self, monkeypatch, tmp_path):
        import ggah_mod.halos.beyond_linear_bias as B

        monkeypatch.setattr(B, "_NPZ", tmp_path / "absent.npz")
        B.load.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="regenerate"):
                B.load()
        finally:
            monkeypatch.undo()
            B.load.cache_clear()

    def test_a_rescaling_pinned_by_its_bounds_warns(self, monkeypatch):
        import ggah_mod.halos.beyond_linear_bias as B

        monkeypatch.setattr(B, "_WARNED", set())
        tab = B.load()
        g = float(tab["g_md"][len(tab["g_md"]) // 2])
        with pytest.warns(RuntimeWarning) as seen:
            B._check_solution(B.S_RANGE[0], g, 0.5, tab)
        text = " ".join(str(w.message) for w in seen)
        assert "search bound" in text and "residual is large" in text

    def test_an_unknown_interpolation_is_refused_on_every_axis(self):
        import ggah_mod.halos.beyond_linear_bias as B

        tab = B.load()
        with pytest.raises(ValueError, match="'linear' or 'cubic'"):
            B._interp_k(np.zeros((tab["k"].size, 2, 2)), np.array([0.1]),
                        np.asarray(tab["ln_k"]), interp="quintic")
        with pytest.raises(ValueError, match="'linear' or 'cubic'"):
            B._blend_in_g(tab, 0, 1, 0.5, 4, "quintic")
        with pytest.raises(ValueError, match="'linear' or 'cubic'"):
            B.project_weights(np.array([1.0]), np.asarray(tab["nu"][0]),
                              interp="quintic")

    def test_an_untabulated_snapshot_is_refused(self, spectrum):
        from ggah_mod.halos.beyond_linear_bias import table_at

        k, _ = spectrum
        with pytest.raises(ValueError, match="not tabulated"):
            table_at(k, snap=9999)

    def test_the_rescaling_needs_a_spectrum(self, spectrum):
        from ggah_mod.halos.beyond_linear_bias import table_at

        k, _ = spectrum
        with pytest.raises(ValueError, match="target spectrum"):
            table_at(k)

    def test_beta_builds_its_own_table_when_given_none(self, spectrum):
        """Without a table ``beta_nl`` passes its keywords to :func:`table_at`.
        Its first argument is already ``k``, so the keyword form cannot also
        name the target spectrum's ``k``: it serves a pinned snapshot."""
        from ggah_mod.halos.beyond_linear_bias import beta_nl, load, table_at

        k, _ = spectrum
        snap = int(load()["snap"][-1])
        nu = np.array([1.0, 2.0])
        direct = beta_nl(k, nu, nu, table_at(k, snap=snap))
        assert np.array_equal(beta_nl(k, nu, nu, snap=snap), direct)
