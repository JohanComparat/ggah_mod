r"""What the census predicts that a telescope can answer.

Two observables here, and they are the two the ejected component was built for:
the dispersion measure of a fast radio burst, which counts every free electron
along a line of sight, and the baryonic suppression of P_mm, which is what a
weak-lensing survey has to marginalise over.  Both were unrepresentable before
the matter budget had somewhere to put the gas a halo had lost.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.halos.profiles import ejected_uk
from ggah_mod.observables import real_space
from ggah_mod.observables import (
    dispersion_measure, limber_grid, mean_dispersion_measure,
    mean_electron_density,
)
from ggah_mod.sectors import (
    BaryonSplit, ColdGasParams, ColdGasSector, HotGasDPM, MatterField,
    dpm_model_params,
)
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import sham as SH
from ggah_mod.sectors.census import census

pytestmark = pytest.mark.slow

ZM15 = (12.10, 10.31, 0.33, 0.42, 1.21)
F_B = PLANCK18.Omega_b / PLANCK18.Omega_m
OPTS = SP.PkOptions(two_halo_spectrum="cb",
                    two_halo_consistency="linear_deficit")


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)


@pytest.fixture(scope="module")
def split(field):
    lm = jnp.log10(field.m)
    f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)) / field.m
    f_cold = ColdGasSector("padmanabhan17").f_cold(field, ColdGasParams())
    return BaryonSplit.from_hot(F_B, E.f_gas_sigmoid(lm, F_B),
                                f_star_cen=f_star, f_cold=f_cold)


class TestTheDispersionMeasure:
    def test_the_ionised_fraction_comes_from_the_census(self, field, split):
        """The point of putting a dispersion measure in this thread: `f_e` is
        not a fitted nuisance, it is one minus the census's own stellar and
        neutral rows."""
        c = census(field, split)
        locked = float(c.omega["star_cen"] + c.omega["star_sat"]
                       + c.omega["cold"])
        f_e = 1.0 - locked / c.omega_b
        assert 0.90 < f_e < 0.99
        assert mean_electron_density(PLANCK18, f_e) > 0

    def test_it_is_linear_in_z_at_low_redshift(self):
        """The Macquart relation.  Curvature enters through E(z), which is
        1 at z = 0, so the first departure from linearity is second order."""
        dm = np.asarray(mean_dispersion_measure([0.05, 0.10], PLANCK18, 0.94))
        assert dm[1] / dm[0] == pytest.approx(2.0, rel=0.02)

    def test_the_amplitude_is_the_published_order(self):
        """Compilations quote <DM_cosmic> ~ 850-1000 z pc/cm^3.  Ours is above
        the lower end because `f_e` here counts *every* ionised baryon,
        including the gas still bound in haloes, where the usual f_IGM ~ 0.83
        excludes it."""
        dm = float(mean_dispersion_measure(1.0, PLANCK18, 0.9434)[0])
        assert 800.0 < dm < 1200.0

    def test_the_one_plus_z_is_one_power_not_three(self):
        """Proper density gives (1+z)^3, proper path length (1+z)^-1, and the
        observed frequency-squared scaling another (1+z)^-1.  A model with the
        wrong power is a factor 2 out at z = 1, which is inside the range a
        careless fit would absorb -- so it is pinned against a hand integral
        rather than against a plot."""
        z = np.linspace(0.0, 1.0, 20001)
        from ggah_mod.cosmology.background import hubble_e
        from ggah_mod.observables.kernels import PC_CM
        from ggah_mod.cosmology import constants as C
        want = (mean_electron_density(PLANCK18, 0.9434)
                * C.C_KM_S / (100.0 * PLANCK18.h) * C.MPC_CM
                * np.trapezoid((1.0 + z) / np.asarray(hubble_e(z, PLANCK18)), z)
                / PC_CM)
        got = float(mean_dispersion_measure(1.0, PLANCK18, 0.9434)[0])
        assert got == pytest.approx(want, rel=1e-4)

    def test_the_kernel_integrates_to_the_mean(self):
        """`int W dchi = <DM>`: the kernel is the same integral against dchi
        rather than dz, so the two cannot disagree without one of the unit
        chains being wrong."""
        z, chi = limber_grid(PLANCK18, z_max=1.0, n=2048)
        w = dispersion_measure(z, chi, PLANCK18, 0.9434).w
        got = float(jnp.trapezoid(w, chi))
        want = float(mean_dispersion_measure(1.0, PLANCK18, 0.9434)[0])
        assert got == pytest.approx(want, rel=2e-3)

    def test_the_kernel_is_not_normalised(self, ):
        """Unlike `number_counts`.  Its amplitude *is* the physical electron
        density, which is the quantity the census predicts; normalising would
        discard it."""
        z, chi = limber_grid(PLANCK18, z_max=1.0, n=256)
        a = dispersion_measure(z, chi, PLANCK18, 0.5).w
        b = dispersion_measure(z, chi, PLANCK18, 1.0).w
        assert float(jnp.max(b / a)) == pytest.approx(2.0, rel=1e-12)


class TestTheBaryonicSuppression:
    @pytest.fixture(scope="class")
    def wired(self, field, split):
        gas = HotGasDPM(backend=DIFFERENTIABLE)
        u_gas = gas.mass_uk(field.k, field.m, field.z, field.cosmo,
                            dpm_model_params(2), conc=field.conc)
        u_ej = ejected_uk(field.k, field.r_delta, eta_ej=2.0)
        return ({"matter": MatterField()},
                {"matter": {"split": split, "u_gas": u_gas, "u_ej": u_ej}})

    def test_it_is_one_on_large_scales_to_the_transforms_own_accuracy(
            self, field, wired):
        """Every u -> 1 there, so the profiles are indistinguishable and no
        rearrangement of the baryons can matter.  Nothing imposes this; it
        falls out, which is what makes it a check.

        The tolerance is not a round number.  `at_large_scales` reads k_min
        rather than k = 0, and the transforms depart from unity there by
        O((k_min r)^2) -- the same floor that sets the matter budget's
        mass-conservation residual.  Measured here and used, so the test
        tightens by itself if the grid ever does.
        """
        sectors, params = wired
        floor = max(
            float(jnp.max(jnp.abs(field.u_nfw()[0] - 1.0))),
            float(jnp.max(jnp.abs(params["matter"]["u_gas"][0] - 1.0))),
            float(jnp.max(jnp.abs(params["matter"]["u_ej"][0] - 1.0))),
        )
        assert floor < 1e-5, "the grid has become too coarse to say anything"
        _, S = SP.matter_suppression(field, sectors, params, options=OPTS)
        assert abs(float(S[0]) - 1.0) < 10.0 * floor

    def test_it_suppresses_by_a_believable_amount(self, field, wired):
        """Hydrodynamic simulations put the suppression at a few per cent by
        k = 1 h/Mpc and 15-25% by k = 10.  Asserted as a band: a tighter test
        would be asserting the feedback model rather than the machinery."""
        k, S = SP.matter_suppression(field, *wired, options=OPTS)
        k, S = np.asarray(k), np.asarray(S)
        assert 0.85 < float(np.interp(1.0, k, S)) < 0.99
        assert 0.70 < float(np.interp(10.0, k, S)) < 0.90

    def test_it_is_monotonic_in_k(self, field, wired):
        k, S = SP.matter_suppression(field, *wired, options=OPTS)
        inside = np.asarray(k) < 20.0
        assert np.all(np.diff(np.asarray(S)[inside]) <= 1e-12)

    def test_without_profiles_there_is_no_suppression_at_all(self, field,
                                                             split):
        """The defect this whole thread began with, as a test.

        With every component on u_DM -- which is what the three-way split did
        to the ejected baryons -- moving mass between components is the
        identity, and the model cannot produce a suppression however strong the
        feedback.  A lensing analysis would then have marginalised over a
        parameter that does nothing.
        """
        sectors = {"matter": MatterField()}
        params = {"matter": {"split": split}}
        _, S = SP.matter_suppression(field, sectors, params, options=OPTS)
        np.testing.assert_allclose(np.asarray(S), 1.0, rtol=1e-12)

    def test_a_larger_ejection_radius_suppresses_more(self, field, split):
        sectors = {"matter": MatterField()}
        out = []
        for eta in (1.0, 4.0):
            u_ej = ejected_uk(field.k, field.r_delta, eta_ej=eta)
            k, S = SP.matter_suppression(
                field, sectors, {"matter": {"split": split, "u_ej": u_ej}},
                options=OPTS)
            out.append(float(np.interp(1.0, np.asarray(k), np.asarray(S))))
        assert out[1] < out[0]

    def test_it_refuses_without_the_matter_sector(self, field):
        with pytest.raises(ValueError, match="matter_suppression needs"):
            SP.matter_suppression(field, {}, {}, options=OPTS)


class TestTheStackedKineticSz:
    """The other measurement of where the ejected baryons are."""

    @pytest.fixture(scope="class")
    def p_eg(self, field, split):
        from ggah_mod.sectors import (
            EjectaSector, GalaxySector, galaxy_defaults,
        )
        gal = GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE)
        gp = dict(galaxy_defaults("zheng07"),
                  lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42,
                  gamma=1.21)
        sectors = {"gas": HotGasDPM(backend=DIFFERENTIABLE),
                   "ejecta": EjectaSector(), "galaxies": gal}
        params = {"gas": dpm_model_params(2),
                  "ejecta": {"split": split, "eta_ej": 2.0},
                  "galaxies": gp}
        both = SP.spectrum(field, "electrons", "galaxies", sectors, params,
                           options=OPTS)
        hot = SP.spectrum(field, SP.tracers.resolve("gas"), "galaxies",
                          sectors, params, options=OPTS)
        return both, hot

    def test_the_optical_depth_is_the_measured_order(self, p_eg):
        """Stacked kSZ reports tau ~ 1e-5 to 1e-4 around L* galaxies.  An
        amplitude test rather than a shape one, because the unit chain is what
        it is guarding: getting the mean-density factor wrong put this at
        1e+6, and only its size made that obvious.

        The pinned inner value is 1.42e-4 since ggah_mod 0.8.8, when the gas
        profiles moved onto the halo's own concentration (7.6 at 1e14 against
        the published 4.59): a more concentrated profile puts more electrons
        in the column at 0.05 Mpc/h.  It was 1.1e-4 before, and the order of
        magnitude above is what guards the unit chain."""
        both, _ = p_eg
        rp = np.array([0.05, 0.3, 3.0])
        tau = np.asarray(real_space.tau_ksz(rp, both, PLANCK18))
        assert np.all(tau > 1e-6) and np.all(tau < 1e-3)
        assert tau[0] == pytest.approx(1.42e-4, rel=0.25)

    def test_no_mean_density_is_multiplied_in(self, p_eg):
        """The gas and ejecta sectors weight by mass, so P_eg already carries
        the density that `pk_to_sigma` restores for the matter field.  A second
        factor is ten orders of magnitude, which this pins."""
        from ggah_mod.observables.transforms import pk_to_sigma
        both, _ = p_eg
        rp = np.array([0.3])
        right = float(real_space.tau_ksz(rp, both, PLANCK18)[0])
        wrong_scale = float(pk_to_sigma(rp, both.k, both.total,
                                        PLANCK18.rho_matter)[0])
        assert wrong_scale / float(pk_to_sigma(rp, both.k, both.total, 1.0)[0]) \
            == pytest.approx(float(PLANCK18.rho_matter), rel=1e-12)
        assert right < 1e-3

    def test_it_falls_with_radius(self, p_eg):
        both, _ = p_eg
        rp = np.array([0.05, 0.1, 0.3, 1.0, 3.0])
        tau = np.asarray(real_space.tau_ksz(rp, both, PLANCK18))
        assert np.all(np.diff(tau) < 0)

    def test_the_ejected_component_takes_over_with_radius(self, p_eg):
        """The physical content, and the reason `electrons` is a composite.

        Inside the halo the hot phase dominates the electron column; by a few
        Mpc most of it is gas the halo lost.  A model using `gas:density` alone
        would under-predict the outer profile by more than a factor two, which
        is where stacked kSZ has its constraining power.
        """
        both, hot = p_eg
        rp = np.array([0.05, 3.0])
        r = (np.asarray(real_space.tau_ksz(rp, hot, PLANCK18))
             / np.asarray(real_space.tau_ksz(rp, both, PLANCK18)))
        assert r[0] > 0.7
        assert r[1] < 0.5
        assert r[0] > r[1]

    def test_the_redshift_factor_is_two_powers(self, p_eg):
        """(1+z)^3 from the proper density, (1+z)^-1 from the proper path.  No
        third factor: an optical depth is not redshifted the way an observed
        dispersion measure is."""
        both, _ = p_eg
        rp = np.array([0.3])
        a = float(real_space.tau_ksz(rp, both, PLANCK18, z=0.0)[0])
        b = float(real_space.tau_ksz(rp, both, PLANCK18, z=1.0)[0])
        assert b / a == pytest.approx(4.0, rel=1e-12)
