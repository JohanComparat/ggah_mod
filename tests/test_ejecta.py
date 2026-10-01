r"""The ejected baryons as a tracer, and the composite that needs it.

The census can say how much gas left.  Only a tracer can say where it went, and
every measurement of the ejected component -- stacked kSZ, a dispersion measure,
the suppression of P_mm -- measures where.
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
from ggah_mod.halos.profiles import ejected_uk as EJ_UK
from ggah_mod.sectors import (
    AgnParams, AgnSector, BaryonSplit, EjectaParams, EjectaSector,
    GalaxySector, HotGasDPM, MatterField, DpmParams, galaxy_defaults,
)
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import matter as MT
from ggah_mod.sectors import sham as SH

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
    f_hot = E.f_gas_sigmoid(lm, F_B)
    f_star = jnp.power(10.0, SH.mstar_from_mh_zu15(lm, *ZM15)) / field.m
    return BaryonSplit.from_hot(F_B, f_hot, f_star_cen=f_star)


class TestTheSectorContract:
    def test_it_is_a_peer_like_every_other_sector(self, field, split):
        """Takes a halo field, builds no mass grid, needs no galaxy
        parameters."""
        w = EjectaSector().weights(field, {"split": split})
        assert w.discrete is False
        assert w.w_point is None
        assert np.all(np.isfinite(np.asarray(w.w_extended)))

    def test_it_refuses_to_recompute_the_ejected_fraction(self, field):
        """Two definitions of one number is the defect this package is mostly
        docstrings about."""
        with pytest.raises(ValueError, match="second definition"):
            EjectaSector().weights(field, {})

    def test_an_unknown_view_is_refused(self, field, split):
        with pytest.raises(ValueError, match="unknown ejecta view"):
            EjectaSector().weights(field, {"split": split}, view="pressure")

    def test_density_is_the_same_array_as_mass(self, field, split):
        """One function under two names: the shapes are identical and the
        amplitudes differ by a constant that cancels."""
        p = {"split": split}
        a = EjectaSector().weights(field, p, view="mass")
        b = EjectaSector().weights(field, p, view="density")
        np.testing.assert_array_equal(np.asarray(a.w_extended),
                                      np.asarray(b.w_extended))


class TestItPutsTheGasWhereTheHaloIsNot:
    def test_the_profile_is_flatter_than_the_dark_matter(self, field, split):
        """The whole point.  On u_DM the expelled gas sits where the lensing
        signal is largest; on the Gaussian shell it does not."""
        w = EjectaSector().weights(field, {"split": split})
        amp = np.asarray(split.f_ejected * field.m)
        u_ej = np.asarray(w.w_extended) / amp[None, :]
        u_dm = np.asarray(field.u_nfw())
        i = int(np.argmin(np.abs(np.asarray(field.m) - 1e14)))
        k = np.asarray(field.k)
        mid = np.argmin(np.abs(k - 1.0))
        assert u_ej[mid, i] < u_dm[mid, i]

    def test_a_larger_radius_moves_more_power_to_large_scales(self, field,
                                                             split):
        """Compared on the *profile*, not the weight.

        `f_ejected` goes slightly negative at the top of the mass grid -- the
        sigmoid's ceiling is f_b^cosmic rather than f_b^cosmic - f_star, so the
        most massive haloes are marginally overdrawn -- and `BaryonSplit` does
        not clip it.  Against a negative amplitude a *more* suppressed profile
        gives a *larger* signed weight, so asserting on the weight would be
        asserting the sign of a number this test is not about.
        """
        amp = np.asarray(split.f_ejected * field.m)[None, :]
        near = np.asarray(EjectaSector().weights(
            field, {"split": split, "eta_ej": 1.0}).w_extended) / amp
        far = np.asarray(EjectaSector().weights(
            field, {"split": split, "eta_ej": 5.0}).w_extended) / amp
        assert np.all(far <= near + 1e-15)
        assert np.max(near - far) > 0.5

    def test_the_large_scale_limit_is_the_ejected_mass(self, field, split):
        """u -> 1, so W(k->0) is M_ej itself -- which is what makes the tracer
        and the census row the same number."""
        w = EjectaSector().weights(field, {"split": split})
        got = np.asarray(w.at_large_scales())
        want = np.asarray(split.f_ejected * field.m)

        # Stated as an identity rather than a tolerance: `at_large_scales`
        # reads k_min, not k = 0, so the ratio *is* u_ej(k_min) and nothing
        # else.  Asserting that exactly is stronger than asserting a small
        # difference, because it would catch an amplitude error of any size.
        u_min = np.asarray(EJ_UK(field.k[:1], field.r_delta))[0]
        np.testing.assert_allclose(got, want * u_min, rtol=1e-14)

        # And the departure itself: 5.4e-7 at the largest halo on the shipped
        # grid, growing as (k_min r_ej)^2, so it is the grid's smallest
        # wavenumber rather than anything about the profile.
        assert float(np.max(np.abs(u_min - 1.0))) < 1e-5


class TestTheElectronComposite:
    """kSZ and a dispersion measure see both phases, not one."""

    @pytest.fixture(scope="class")
    def wired(self, field, split):
        gas = HotGasDPM(backend=DIFFERENTIABLE)
        closed = EjectaSector.split_from_gas(
            field, gas, DpmParams(), f_star_cen=split.f_star_cen)
        sectors = {"gas": gas, "ejecta": EjectaSector(),
                   "matter": MatterField()}
        params = {"gas": DpmParams(),
                  "ejecta": {"split": closed},
                  "matter": {"split": closed}}
        return sectors, params

    def test_a_split_built_from_the_gas_closes_the_budget(self, field, wired):
        """The hot atmosphere plus what it does not hold is every non-stellar
        baryon, node by node, to the k_min departure of the two profiles."""
        import warnings
        sectors, params = wired
        split = params["ejecta"]["split"]
        comps = SP.resolve("electrons").components
        held = sum(np.asarray(SP.build_weights(c, field, sectors, params)
                              .at_large_scales()) for c in comps)
        want = (float(MT.cosmic_baryon_fraction(PLANCK18))
                - np.asarray(split.f_star_cen)) * np.asarray(field.m)
        np.testing.assert_allclose(held, want, rtol=1e-5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SP.spectrum(field, "electrons", "electrons", sectors, params,
                        options=OPTS)
        assert not [w for w in caught if "split_from_gas" in str(w.message)]

    def test_two_models_of_the_gas_are_reported(self, field, split):
        """The fixture's split takes its hot gas from a FLAMINGO sigmoid, the
        gas leg from the DPM: the composite counts some gas twice, and says so."""
        sectors = {"gas": HotGasDPM(backend=DIFFERENTIABLE),
                   "ejecta": EjectaSector()}
        params = {"gas": DpmParams(), "ejecta": {"split": split}}
        with pytest.warns(RuntimeWarning, match="split_from_gas"):
            SP.spectrum(field, "electrons", "electrons", sectors, params,
                        options=OPTS)

    @pytest.mark.xfail(strict=True, reason=(
        "the calibrated DpmParams defaults put more hot gas than the cosmic "
        "share at the top of the mass grid (1.03 f_b at 10^16 Msun/h, negative "
        "ejecta above 10^15.95: the calibration's baryon-budget wall is not "
        "fully met there), so "
        "split_from_gas makes the ejected fraction negative and the "
        "composite falls below the hot gas"))
    def test_the_electron_spectrum_exceeds_the_hot_gas_alone(self, field,
                                                             wired):
        """The commonest error the composite exists to prevent: `gas:density`
        is the electrons inside R_Delta, and most of the budget is outside."""
        sectors, params = wired
        both = SP.spectrum(field, "electrons", "electrons", sectors, params,
                           options=OPTS)
        hot = SP.spectrum(field, SP.tracers.resolve("gas"),
                          SP.tracers.resolve("gas"), sectors, params,
                          options=OPTS)
        r = np.asarray(both.total / hot.total)
        assert np.all(r >= 1.0 - 1e-12)
        assert r[0] > 1.5, "the ejected gas should dominate on large scales"

        # And it vanishes on small ones: the Gaussian is zero well inside the
        # halo-model k range, so a small-scale probe sees the hot phase alone.
        # That is the physical content of the composite -- which probe you use
        # decides which phase you are measuring -- and it is why the ratio
        # tends to exactly 1 rather than to something merely close.
        assert r[-1] == pytest.approx(1.0, abs=1e-12)

    def test_it_is_three_pairs_not_one(self, field, wired):
        """P_ee = P_gg + 2P_ge + P_ee, which is the reason a tracer is a list.
        If the cross term were dropped the total would fall short."""
        sectors, params = wired
        tot = SP.spectrum(field, "electrons", "electrons", sectors, params,
                          options=OPTS).total
        gg = SP.spectrum(field, SP.tracers.resolve("gas"),
                         SP.tracers.resolve("gas"), sectors, params,
                         options=OPTS).total
        ee = SP.spectrum(field, "ejecta", "ejecta", sectors, params,
                         options=OPTS).total
        ge = SP.spectrum(field, SP.tracers.resolve("gas"), "ejecta", sectors,
                         params, options=OPTS).total
        np.testing.assert_allclose(np.asarray(tot),
                                   np.asarray(gg + 2 * ge + ee), rtol=1e-10)


class TestParameters:
    def test_eta_ej_carries_a_bound_a_prior_and_a_reason(self):
        p = EjectaParams._PARAMS["eta_ej"]
        assert p.bounds == (1.0, 8.0)
        assert p.why and p.kind == "physical"

    @pytest.mark.x64
    def test_the_ejection_radius_is_differentiable(self, field, split):
        def f(eta):
            w = EjectaSector().weights(field, {"split": split, "eta_ej": eta})
            return jnp.sum(w.w_extended)

        ad = float(jax.grad(f)(2.0))
        fd = float((f(2.0 + 1e-6) - f(2.0 - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-5)
