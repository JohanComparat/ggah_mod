r"""Verification: the AGN sector's bolometric correction and its soft band.

``AgnSector(kbol="duras20")``, the default, is the general
:math:`K_X(L_{\rm bol})` of Duras et al. (2020, Eq. 2 and Table 1), with the
Jacobian of the change of variable from :math:`\log L_{\rm bol}` to
:math:`\log L_X`.  ``kbol="constant"`` is ``k_bol`` at every luminosity and,
at the parameters of 1.4.0, must be 1.4.0 bit for bit.  At ``obscuration="split"`` the 0.5--2 keV XLF is the sum of
the Comparat et al. (2019) classes, each moved by its absorbed band ratio
(:meth:`~ggah_mod.sectors.agn.AgnSector.band_shift`).
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import agn as A
from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

Z = 0.135
GAL = GalaxySector("zumandelbaum15")
GP = galaxy_defaults("zumandelbaum15")
#: Duras et al. (2020) Table 1, general sample, K_X(L_bol).
A_D, B_D, C_D = 10.96, 11.93, 17.79
#: The fitted defaults of 1.4.0, fitted with k_bol = 20 (``kbol="constant"``):
#: the parameters a claim about the constant correction's own model is made at.
#: Every other AgnParams and ObscurationParams default is unchanged since.
AGN_140 = dict(mu_bh=7.601, al_bh=1.061, sig_bh=0.597, log10_lstar=-0.999,
               log10_lstar_mslope=-1.123, delta1=0.37, delta2=2.581,
               log10_ferdf=-0.7, log10_ferdf_mslope=1.401, gamma_x=1.723,
               f_duty_sat=0.321)
OBSCURATION_140 = dict(f_faint_norm=1.0, l_blend_0=43.184, l_blend_width=0.691,
                       l_blend_m_amp=0.119)


def _shift_140(self, t_grid, k, p, mean_log10_mbh, log10_mstar):
    """``AgnSector._shift`` of ggah_mod 1.4.0, verbatim."""
    logk = jnp.log10(A.LBOL_COEF / p.k_bol)
    t = (self.loglx[None, :] - logk
         - jnp.atleast_1d(mean_log10_mbh)[:, None]
         - jnp.atleast_1d(A._lstar_tilt(log10_mstar, p))[:, None])
    if k.ndim == 1:
        return jnp.interp(t.ravel(), t_grid, k, left=0.0,
                          right=0.0).reshape(t.shape)
    return jax.vmap(lambda tt, kk: jnp.interp(tt, t_grid, kk, left=0.0,
                                              right=0.0))(t, k)


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                      make_pk("emu_pk"), z=Z)


def _sectors(kbol):
    kw = dict(calibration="off", kbol=kbol)
    return dict(
        lum=A.AgnSector(GAL, **kw),
        split=A.AgnSector(GAL, obscuration="split", **kw),
        cat=A.AgnSector(GAL, obscuration="split", selection="catalogue",
                        catalogue_flux_limit=True, **kw))


def _luminosity_outputs(sec, field, p, o):
    """Every output that reads the luminosity kernel."""
    lum, split, cat = sec["lum"], sec["split"], sec["cat"]
    lm, h = jnp.log10(field.m), field.cosmo.h
    out = dict(
        p_loglx_given_m=lum.p_loglx_given_m(lm, p, GP, h=h, z=field.z),
        p_active=lum.p_active_loglx_given_m(lm, p, GP, h=h, z=field.z),
        satellite_kernel=lum._satellite_kernel(field, p, GP)[2],
        xlf_hard=lum.xlf(field, p, GP)[1],
        xlf_soft_grid=lum.xlf(field, p, GP, band="soft")[0],
        xlf_soft=lum.xlf(field, p, GP, band="soft")[1],
        xlf_aird15=split.xlf(field, p, GP, observed="aird15", obsc=o)[1],
        xlf_catalogue=cat.xlf(field, p, GP, observed=True, obsc=o)[1],
        occupation=lum.occupation(field, p, GP)[0],
        occupation_lx=lum.occupation(field, p, GP)[1],
        satellite_occupation=lum.satellite_occupation(field, p, GP),
        catalogue_occupation=cat.occupation(field, p, GP, obsc=o)[0],
        validity_cost=lum.validity_cost(field, p, GP, quantity="l_x"))
    w = lum.weights(field, p, GP)
    out.update(w_point=w.w_point, w_extended=w.w_extended, self_pair=w.self_pair)
    wc = cat.weights(field, p, GP, obsc=o)
    out.update(cat_point=wc.w_point, cat_extended=wc.w_extended)
    e = split.emission_weights(field, p, GP, band=(0.5, 2.0), obsc=o)
    out.update(emission=e.w_point, emission_extended=e.w_extended,
               emission_pair=e.self_pair)
    return {k: np.asarray(v) for k, v in out.items()}


class TestDuras20IsTheShippedModelAndConstantIs140:
    def test_duras20_is_the_default_and_the_modes_are_checked(self):
        assert A.AgnSector(GAL).kbol == "duras20"
        assert A.KBOL_MODES == ("constant", "duras20")
        with pytest.raises(ValueError, match="unknown kbol"):
            A.AgnSector(GAL, kbol="marconi04")

    @pytest.mark.x64
    def test_every_luminosity_output_is_1_4_0_bit_for_bit(self, field,
                                                          monkeypatch):
        """``kbol="constant"`` at the parameters of 1.4.0 is the 1.4.0 model."""
        p = A.AgnParams(**AGN_140)
        o = A.ObscurationParams(**OBSCURATION_140)
        with monkeypatch.context() as m:
            m.setattr(A.AgnSector, "_shift", _shift_140)
            old = _luminosity_outputs(_sectors("constant"), field, p, o)
        new = _luminosity_outputs(_sectors("constant"), field, p, o)
        assert set(old) == set(new)
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)

    def test_the_unsplit_soft_band_is_still_the_unabsorbed_shift(self, field):
        lum = A.AgnSector(GAL, calibration="off")
        p = A.AgnParams()
        g_h, phi_h = lum.xlf(field, p, GP)
        g_s, phi_s = lum.xlf(field, p, GP, band="soft")
        np.testing.assert_array_equal(np.asarray(phi_s), np.asarray(phi_h))
        np.testing.assert_array_equal(np.asarray(g_s),
                                      np.asarray(g_h + jnp.log10(p.k_h2s)))


class TestDurasBolometricCorrection:
    def test_it_is_their_formula_inside_their_range(self):
        lb = np.linspace(7.5, 14.5, 141) + np.log10(C.L_SUN_ERG_S)
        u = lb - np.log10(3.828e33)
        ref = A_D * (1.0 + (u / B_D) ** C_D)
        got = 10.0 ** np.asarray(A.log10_kx_duras20(jnp.asarray(lb)))
        np.testing.assert_allclose(got, ref, rtol=1e-13)
        assert A.DURAS20_KX == {"a": A_D, "b": B_D, "c": C_D}
        assert A.DURAS20_LOG_LBOL_RANGE == (7.5, 14.5)
        assert C.L_SUN_ERG_S == 3.828e33

    def test_the_slope_is_its_derivative(self):
        lb = jnp.linspace(36.0, 52.0, 321)
        ad = jax.vmap(jax.grad(A.log10_kx_duras20))(lb)
        np.testing.assert_allclose(np.asarray(A.dlog10_kx_duras20(lb)),
                                   np.asarray(ad), rtol=1e-12, atol=1e-15)

    def test_it_continues_linearly_and_smoothly_outside(self):
        lo, hi = (x + np.log10(C.L_SUN_ERG_S) for x in A.DURAS20_LOG_LBOL_RANGE)
        for edge, sign in ((lo, -1.0), (hi, 1.0)):
            k0 = float(A.log10_kx_duras20(edge))
            s0 = float(A.dlog10_kx_duras20(edge))
            for d in (0.5, 2.0, 6.0):
                x = edge + sign * d
                assert float(A.log10_kx_duras20(x)) == pytest.approx(
                    k0 + s0 * sign * d, rel=1e-13)
                assert float(A.dlog10_kx_duras20(x)) == s0
            # C^1 across the edge: the slope just inside tends to the edge's.
            inside = float(A.dlog10_kx_duras20(edge - sign * 1e-7))
            assert inside == pytest.approx(s0, rel=1e-5, abs=1e-9)
        assert float(A.dlog10_kx_duras20(hi)) == pytest.approx(0.51677, abs=1e-5)

    def test_l_x_is_monotonic_in_l_bol(self):
        """d log K/d log L_bol stays in [0, 0.522], so log L_X(log L_bol) has
        slope 0.478--1 everywhere, including the continuation."""
        lb = jnp.linspace(30.0, 60.0, 30001)
        s = np.asarray(A.dlog10_kx_duras20(lb))
        assert s.min() >= 0.0 and s.max() < 0.5217
        i = int(np.argmax(s))
        assert float(lb[i]) - np.log10(C.L_SUN_ERG_S) == pytest.approx(13.98, abs=0.01)
        lx = np.asarray(lb - A.log10_kx_duras20(lb))
        assert np.all(np.diff(lx) > 0.0)

    def test_the_inverse_and_its_derivative(self):
        lx = jnp.linspace(38.0, 48.0, 101)
        lb = A.log10_lbol_of_lx_duras20(lx)
        np.testing.assert_allclose(np.asarray(lb - A.log10_kx_duras20(lb)),
                                   np.asarray(lx), rtol=0, atol=1e-12)
        ad = np.asarray(jax.vmap(jax.grad(A.log10_lbol_of_lx_duras20))(lx))
        h = 1e-5
        fd = np.asarray((A.log10_lbol_of_lx_duras20(lx + h)
                         - A.log10_lbol_of_lx_duras20(lx - h)) / (2 * h))
        np.testing.assert_allclose(ad, fd, rtol=1e-7)
        np.testing.assert_allclose(
            ad, 1.0 / (1.0 - np.asarray(A.dlog10_kx_duras20(lb))), rtol=1e-12)
        assert ad.min() >= 1.0 and ad.max() < 2.091

    def test_the_numbers_quoted(self):
        lb = A.log10_lbol_of_lx_duras20(jnp.asarray([42.0, 44.0, 45.0]))
        k = 10.0 ** np.asarray(A.log10_kx_duras20(lb))
        np.testing.assert_allclose(k, [11.14, 18.54, 99.08], rtol=1e-3)


@pytest.fixture(scope="module")
def pair():
    return (A.AgnSector(GAL, calibration="off", kbol="constant"),
            A.AgnSector(GAL, calibration="off", kbol="duras20"))


@pytest.fixture(scope="module")
def split():
    return A.AgnSector(GAL, calibration="off", obscuration="split")


class TestDurasKernel:
    def test_the_sector_carries_the_inverse_and_the_jacobian(self, pair):
        _, d = pair
        np.testing.assert_allclose(
            np.asarray(d._log10_lbol),
            np.asarray(A.log10_lbol_of_lx_duras20(d.loglx)), rtol=0, atol=0)
        h = 1e-5
        fd = np.asarray((A.log10_lbol_of_lx_duras20(d.loglx + h)
                         - A.log10_lbol_of_lx_duras20(d.loglx - h)) / (2 * h))
        np.testing.assert_allclose(np.asarray(d._dlog10_lbol_dlog10_lx), fd,
                                   rtol=1e-7)

    def test_the_kernel_integrates_to_the_same_total(self, pair):
        """The Jacobian is right: a kernel whose support lies inside the
        grid integrates to one over log L_X in both modes, as it does over
        log L_bol."""
        const, dur = pair
        p = A.AgnParams(sig_bh=0.3)
        t, k = dur._kernel(p, 0.3, z=Z)
        assert float(jnp.sum(k) * dur.dlam) == pytest.approx(1.0, abs=1e-10)
        mean = jnp.asarray([7.0, 7.5, 8.0, 8.5])
        ms = jnp.full(4, 10.5)
        for sec in pair:
            prob = sec._shift(t, k, p, mean, ms)
            total = np.asarray(jnp.trapezoid(prob, sec.loglx, axis=-1))
            np.testing.assert_allclose(total, 1.0, atol=2e-4)

    def test_the_cumulative_distribution_is_the_bolometric_one(self, pair):
        """P(< L_X) on the L_X grid equals P(< L_bol(L_X)) of the kernel in
        log L_bol: the change of variable, Jacobian included."""
        _, dur = pair
        p = A.AgnParams(sig_bh=0.3)
        t, k = dur._kernel(p, 0.3, z=Z)
        mean = jnp.asarray([8.0])
        prob = np.asarray(dur._shift(t, k, p, mean, jnp.asarray([10.5])))[0]
        lx = np.asarray(dur.loglx)
        cdf_x = np.concatenate([[0.0], np.cumsum(0.5 * (prob[1:] + prob[:-1])
                                                 * np.diff(lx))])
        # The same kernel in y = log L_bol - log 1.26e38 - mean, on its grid.
        tt, kk = np.asarray(t), np.asarray(k)
        cdf_t = np.concatenate([[0.0], np.cumsum(0.5 * (kk[1:] + kk[:-1])
                                                 * np.diff(tt))])
        y = np.asarray(dur._log10_lbol) - np.log10(A.LBOL_COEF) - 8.0
        ref = np.interp(y, tt, cdf_t) - np.interp(y[0], tt, cdf_t)
        np.testing.assert_allclose(cdf_x, ref, atol=3e-4)

    def test_the_bolometric_correction_method(self, pair):
        const, dur = pair
        p = A.AgnParams(k_bol=15.0)
        lb = jnp.linspace(42.0, 47.0, 6)
        np.testing.assert_allclose(np.asarray(const.log10_kbol(lb, p)),
                                   np.full(6, np.log10(15.0)), rtol=1e-15)
        np.testing.assert_array_equal(np.asarray(dur.log10_kbol(lb, p)),
                                      np.asarray(A.log10_kx_duras20(lb)))

    def test_it_moves_the_xlf_where_k_x_departs_from_20(self, pair, field):
        """Duras's K_X is 11 at 1e42 and 99 at 1e45: the function rises at the
        faint end and falls at the bright end against k_bol = 20."""
        const, dur = pair
        p = A.AgnParams()
        g, phi_c = const.xlf(field, p, GP)
        _, phi_d = dur.xlf(field, p, GP)
        r = np.asarray(phi_d) / np.asarray(phi_c)
        g = np.asarray(g)
        assert r[np.argmin(np.abs(g - 42.0))] > 1.0
        assert r[np.argmin(np.abs(g - 45.0))] < 0.5

    @pytest.mark.x64
    @pytest.mark.parametrize("key", ["mu_bh", "sig_bh", "log10_lstar",
                                     "log10_lstar_mslope", "delta2",
                                     "log10_ferdf", "k_bol"])
    def test_gradients_match_finite_differences(self, pair, field, key):
        _, dur = pair
        p = A.AgnParams()

        def f(v):
            q = p.replace(**{key: v})
            g, phi = dur.xlf(field, q, GP)
            return (jnp.sum(phi * (g > 42.0)) * dur.dlx
                    + 1e-44 * jnp.sum(dur.l_x_agn(field, q, GP)))

        x = float(getattr(p, key))
        h = 1e-6 * max(abs(x), 1.0)
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad), key
        if key == "k_bol":
            assert ad == 0.0      # not read at kbol="duras20"
            return
        assert ad != 0.0, key
        assert ad == pytest.approx(fd, rel=1e-5), key


class TestTheSoftBandThroughTheObscurationClasses:
    def test_the_shift_is_the_absorbed_band_ratio(self, split):
        p = A.AgnParams()
        hard = float(A.band_energy(2.0, 10.0, p.gamma_x))
        unabs = float(A.band_energy(0.5, 2.0, p.gamma_x))
        for frame, z in (("rest", 0.0), ("observer", 0.36)):
            sh = split.band_shift(p, 0.36, frame=frame)
            for b, (nodes, _) in split._class_nodes().items():
                lin = np.asarray(A.absorbed_band_energy(
                    0.5, 2.0, p.gamma_x, nodes[:, None], z)).reshape(-1)
                if b != "unobscured":
                    lin = lin + A.F_SCATTER * (1.0 + z) ** (2.0 - p.gamma_x) * unabs
                np.testing.assert_allclose(np.asarray(sh[b][0]),
                                           np.log10(lin / hard), atol=1e-10,
                                           err_msg=f"{frame} {b}")
        thick = np.asarray(split.band_shift(p, 0.36)["thick"][0])
        np.testing.assert_allclose(thick, np.log10(A.F_SCATTER * float(p.k_h2s)),
                                   atol=1e-8)

    def test_no_column_and_no_scatter_is_the_unabsorbed_ratio(self):
        thin = {b: (-12.0, -11.0) for b in A.CATALOGUE_BRANCHES}
        sec = A.AgnSector(GAL, calibration="off", obscuration="split",
                          catalogue_nh_classes=thin)
        p = A.AgnParams()
        for frame, z in (("rest", 0.0), ("observer", 0.36)):
            sh = sec.band_shift(p, 0.36, frame=frame, f_scatter=0.0)
            ref = (float(jnp.log10(p.k_h2s))
                   + (2.0 - p.gamma_x) * np.log10(1.0 + z))
            for b in A.CATALOGUE_BRANCHES:
                np.testing.assert_allclose(np.asarray(sh[b][0]), ref,
                                           atol=1e-10)
        with pytest.raises(ValueError, match="frame"):
            sec.band_shift(p, 0.36, frame="lab")

    def test_counts_above_a_soft_threshold_are_the_classes_moved(self, split,
                                                                 field):
        """N(> L_s) = sum_b sum_i w_i N_b(> L_s - Delta_{b,i}): each class keeps
        its sources and moves them by its band ratio."""
        p, o = A.AgnParams(), A.ObscurationParams()
        lm = jnp.log10(field.m)
        prob = split.p_active_loglx_given_m(lm, p, GP, h=field.cosmo.h, z=Z)
        dn = field.dndm * field.m * np.log(10.0)
        phi_b = split._branch_xlfs(field, p, GP, o, lm, prob, dn)
        g, phi_s = split.xlf(field, p, GP, band="soft", obsc=o)
        g = np.asarray(g)

        def above(phi, l0):
            phi = np.asarray(phi)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (phi[1:] + phi[:-1])
                                                   * np.diff(g))])
            return cum[-1] - np.interp(l0, g, cum)

        sh = split.band_shift(p, Z, frame="rest")
        for l0 in (42.0, 43.0, 44.0):
            ref = sum(float(w) * above(phi_b[b], l0 - float(d))
                      for b in A.CATALOGUE_BRANCHES
                      for d, w in zip(*sh[b]))
            assert above(phi_s, l0) == pytest.approx(ref, rel=2e-3)

    def test_absorption_only_removes_sources_above_a_soft_threshold(self,
                                                                    split,
                                                                    field):
        p = A.AgnParams()
        g, phi_s = split.xlf(field, p, GP, band="soft")
        gu, phi_u = A.AgnSector(GAL, calibration="off").xlf(field, p, GP,
                                                            band="soft")

        def above(grid, phi, l0):
            grid, phi = np.asarray(grid), np.asarray(phi)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (phi[1:] + phi[:-1])
                                                   * np.diff(grid))])
            return cum[-1] - np.interp(l0, grid, cum)

        for l0 in (41.0, 42.0, 43.0, 44.0, 45.0):
            assert above(g, phi_s, l0) < above(gu, phi_u, l0)

    @pytest.mark.x64
    def test_gamma_x_reaches_the_soft_band(self, split, field):
        p = A.AgnParams()

        def f(v):
            g, phi = split.xlf(field, p.replace(gamma_x=v), GP, band="soft")
            return jnp.sum(phi * (g > 42.0)) * split.dlx

        x = float(p.gamma_x)
        ad = float(jax.grad(f)(x))
        fd = float((f(x + 1e-6) - f(x - 1e-6)) / 2e-6)
        assert ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)
