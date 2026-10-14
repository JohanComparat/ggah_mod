r"""Verification: galaxy X-ray binaries, and the stacked X-ray luminosity of galaxies.

The binaries are Lehmer et al. (2016) Eq. 15 at their Table 3 coefficients,
the star-formation rate is Speagle et al. (2014) Eq. 28, and the stacked
luminosity adds the AGN sector's mean at fixed stellar mass.
"""
import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ggah_mod.sectors import agn as A
from ggah_mod.sectors import xrb as X
from ggah_mod.sectors.galaxies import GalaxySector
from ggah_mod.sectors.params import Gaussian
from test_agn_kbol import AGN_140

Z = 0.135
GAL = GalaxySector("zumandelbaum15")


def _agn(**kw):
    return A.AgnSector(GAL, calibration="off", obscuration="split", **kw)


class TestLehmer16:
    def test_the_defaults_and_priors_are_table_3(self):
        p = X.XrbParams()
        for name, (value, err) in X.LEHMER16.items():
            assert getattr(p, name) == value
            prior = X.XrbParams._PARAMS[name].prior
            assert isinstance(prior, Gaussian)
            assert (prior.mu, prior.sigma) == (value, err)
        assert X.LEHMER16 == {"log10_alpha0": (29.37, 0.15),
                              "gamma": (2.03, 0.60),
                              "log10_beta0": (39.28, 0.05),
                              "delta": (1.31, 0.13)}

    def test_a_galaxy_of_1e10_msun_forming_one_msun_per_year(self):
        p = X.XrbParams()
        at0 = 10.0 ** 39.37 + 10.0 ** 39.28
        assert float(X.lx_xrb(10.0, 1.0, 0.0, p)) == pytest.approx(at0, rel=1e-13)
        assert at0 == pytest.approx(4.2497e39, rel=1e-4)
        at1 = 10.0 ** 39.37 * 2.0 ** 2.03 + 10.0 ** 39.28 * 2.0 ** 1.31
        assert float(X.lx_xrb(10.0, 1.0, 1.0, p)) == pytest.approx(at1, rel=1e-13)

    def test_the_band_conversion_is_their_gamma_2(self):
        """Lehmer et al. (2016) Sect. 3 divide 0.5-7 keV by 1.64 for 2-10 keV."""
        assert X.xrb_band_ratio((0.5, 7.0)) == pytest.approx(1.64, abs=5e-3)
        assert X.xrb_band_ratio((0.5, 2.0)) == pytest.approx(math.log(4) / math.log(5))
        assert X.xrb_band_ratio() == 1.0
        with pytest.raises(ValueError, match="band"):
            X.xrb_band_ratio((2.0, 0.5))

    def test_one_galaxy_scatters_about_the_mean(self):
        p = X.XrbParams()
        lx = jnp.linspace(35.0, 44.0, 9001)
        pdf = X.p_log10_lx_xrb(lx, 10.0, 1.0, 0.5, p)
        assert float(jnp.trapezoid(pdf, lx)) == pytest.approx(1.0, abs=1e-10)
        mean = float(jnp.trapezoid(pdf * 10.0 ** (lx - 39.0), lx)) * 1e39
        assert mean == pytest.approx(float(X.lx_xrb(10.0, 1.0, 0.5, p)), rel=1e-8)


class TestSpeagle14:
    def test_the_age_is_their_cosmology(self):
        from astropy.cosmology import FlatLambdaCDM
        ref = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=0.0)
        for z in (0.0, 0.135, 1.0, 3.0):
            assert float(X.age_speagle14(z)) == pytest.approx(
                ref.age(z).value, rel=1e-6)

    def test_the_main_sequence_is_eq_28(self):
        for z, lm in ((0.135, 10.0), (1.0, 9.7), (2.0, 11.1)):
            t = float(X.age_speagle14(z))
            ref = (0.84 - 0.026 * t) * lm - (6.51 - 0.11 * t)
            assert float(X.log10_sfr_main_sequence(lm, z)) == pytest.approx(
                ref, abs=1e-12)

    def test_the_mean_sfr(self):
        p = X.XrbParams()
        ms = float(10.0 ** X.log10_sfr_main_sequence(10.5, 0.5))
        lognormal = math.exp(0.5 * (math.log(10.0) * 0.2) ** 2)
        assert float(X.mean_sfr(10.5, 0.5, p)) == pytest.approx(ms * lognormal,
                                                                rel=1e-12)
        q = p.replace(sigma_sfr=0.0)
        assert float(X.mean_sfr(10.5, 0.5, q)) == pytest.approx(ms, rel=1e-12)
        assert float(X.mean_sfr(10.5, 0.5, q, red_fraction=1.0)) == \
            pytest.approx(0.1 * ms, rel=1e-12)
        assert float(X.mean_sfr(10.5, 0.5, q, red_fraction=0.5)) == \
            pytest.approx(0.55 * ms, rel=1e-12)


class TestStackedLx:
    MS = jnp.array([8.0, 9.0, 10.0, 10.5, 11.0, 11.5])

    def test_without_an_agn_it_is_the_binaries_at_the_mean_sfr(self):
        p = X.XrbParams()
        got = X.stacked_lx(self.MS, Z, p, red_fraction=0.3)
        ref = X.lx_xrb(self.MS, X.mean_sfr(self.MS, Z, p, 0.3), Z, p)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-13)
        c = X.stacked_lx(self.MS, Z, p, band=(0.5, 2.0), components=True)
        np.testing.assert_array_equal(np.asarray(c["agn"]), 0.0)
        np.testing.assert_allclose(np.asarray(c["lmxb"] + c["hmxb"]),
                                   np.asarray(c["total"]), rtol=1e-15)

    def test_the_agn_term_is_the_mean_of_the_fixed_mstar_distribution(self):
        agn, ap = _agn(), A.AgnParams()
        c = X.stacked_lx(self.MS, Z, agn=agn, components=True)
        prob = agn.p_active_loglx_given_mstar(self.MS, ap, z=Z)
        ref = jnp.sum(prob * 10.0 ** agn.loglx, axis=-1) * agn.dlx
        np.testing.assert_allclose(np.asarray(c["agn"]), np.asarray(ref),
                                   rtol=1e-13)
        f_s = 0.3
        sat = X.stacked_lx(self.MS, Z, agn=agn, satellite_fraction=f_s,
                           components=True)["agn"]
        np.testing.assert_allclose(
            np.asarray(sat), np.asarray(ref * (1 - f_s + f_s * ap.f_duty_sat)),
            rtol=1e-13)
        assert float(X.stacked_lx(7.5, Z, agn=agn, components=True)["agn"]) == 0.0

    def test_another_band_needs_the_split(self):
        lum = A.AgnSector(GAL, calibration="off")
        with pytest.raises(ValueError, match="obscuration='split'"):
            X.stacked_lx(self.MS, Z, agn=lum, band=(0.5, 2.0))
        soft = X.stacked_lx(self.MS, Z, agn=_agn(), band=(0.5, 2.0),
                            components=True)
        hard = X.stacked_lx(self.MS, Z, agn=_agn(), components=True)
        r = np.asarray(soft["agn"] / hard["agn"])
        assert np.all((r > 0.0) & (r < 1.0))

    @pytest.mark.parametrize("kbol, params", [
        pytest.param("constant", AGN_140, id="constant"),
        pytest.param("duras20", {}, id="duras20")])
    def test_binaries_at_low_mass_and_nuclei_at_high_mass(self, kbol, params):
        """Each correction at the parameters fitted with it, at z = 0.135:
        at the defaults (kbol='duras20') the two are equal at log M* = 8.21,
        and the nuclei are 21 times the binaries at 10^10 Msun and 65 times at
        10^11; at kbol='constant' and the parameters of 1.4.0, 8.15, 18 and
        60."""
        ap = A.AgnParams(**params)
        c = X.stacked_lx(jnp.array([7.9, 10.0, 11.0]), Z,
                         agn=_agn(kbol=kbol), agn_params=ap, components=True)
        assert float(c["agn"][0]) == 0.0
        assert np.all(np.asarray(c["agn"][1:] > 10.0 * c["xrb"][1:]))
        low = X.stacked_lx(8.0, Z, agn=_agn(kbol=kbol), agn_params=ap,
                           components=True)
        assert float(low["xrb"]) > float(low["agn"])

    def test_the_red_fraction_from_the_galaxy_sector(self):
        """The recipe :func:`~ggah_mod.sectors.xrb.mean_sfr` gives, on a
        coarse field."""
        from ggah_mod import DIFFERENTIABLE
        from ggah_mod.cosmology import PLANCK18
        from ggah_mod.cosmology.power import make_pk
        from ggah_mod.halos.field import make_field
        from ggah_mod.sectors.galaxies import galaxy_defaults

        field = make_field(PLANCK18, DIFFERENTIABLE.with_(n_m=64), make_pk("emu_pk"),
                           z=Z)
        red = GalaxySector("zumandelbaum16_red")
        lg, c_r, s_r = red.stellar_mass_function(
            field, galaxy_defaults("zumandelbaum16_red"), msun=True)
        _, c_a, s_a = GalaxySector("zumandelbaum15").stellar_mass_function(
            field, galaxy_defaults("zumandelbaum15"), msun=True)
        ms = self.MS[1:]                       # inside the function's grid
        assert float(lg[0]) < float(ms[0]) and float(lg[-1]) > float(ms[-1])
        f_red = jnp.interp(ms, lg, (c_r + s_r) / (c_a + s_a))
        assert np.all((np.asarray(f_red) > 0.0) & (np.asarray(f_red) < 1.0))
        a = X.stacked_lx(ms, Z, red_fraction=f_red)
        b = X.stacked_lx(ms, Z)
        assert np.all(np.asarray(a) < np.asarray(b))


@pytest.mark.x64
def test_the_derivatives_are_central_differences():
    """In the seven XRB, twenty AGN and nine obscuration parameters, at the
    defaults, in the soft band (the AGN band factor included)."""
    agn = _agn(n_lam=96, n_lx=160)
    theta, unravel = ravel_pytree((X.XrbParams(), A.AgnParams(),
                                   A.ObscurationParams()))
    ms = jnp.array([9.0, 10.0, 11.0])

    def out(t):
        px, pa, po = unravel(t)
        c = X.stacked_lx(ms, 0.5, px, agn=agn, agn_params=pa, obsc=po,
                         red_fraction=0.4, band=(0.5, 2.0),
                         satellite_fraction=0.2, components=True)
        return jnp.log(jnp.stack([c["xrb"], c["agn"]]) / 1e40)

    f = jax.jit(out)
    ad = np.asarray(jax.jacfwd(f)(theta))
    # The kernel is linear between its nodes, and at the defaults the
    # log L_X = 43.38 point of the M* = 10^9 row sits 1.3e-6 dex from one, so
    # a step that moves the kernel by more (mu_bh, al_bh, log10_lstar_mslope)
    # straddles the kink: at 1e-5 the difference is the mean of the two
    # one-sided slopes, 7e-6 from jacfwd's, which is the slope of the segment.
    # At 1e-7 none straddles it and every entry converges, to 9e-8; below
    # 5e-8 round-off takes over.
    cols = []
    for i in range(theta.size):
        step = 1e-7 * max(1.0, abs(float(theta[i])))
        e = jnp.zeros_like(theta).at[i].set(step)
        cols.append((f(theta + e) - f(theta - e)) / (2 * step))
    fd = np.asarray(jnp.stack(cols, axis=-1))
    np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-9)


def test_it_is_exported_with_the_sectors():
    import ggah_mod.sectors as S
    assert S.XrbParams is X.XrbParams and S.stacked_lx is X.stacked_lx
