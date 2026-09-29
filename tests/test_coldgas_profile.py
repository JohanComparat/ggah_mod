r"""The neutral gas's profile: an exponential, with one parameter.

Three claims carry this module, and each is a test rather than a sentence.

**The profile adds a shape and no amplitude.**  ``weights`` multiplies a mass by
a transform equal to one at k = 0, so the profile's own normalisation cancels; a
parameter for it would be degenerate with ``alpha_hi``.  Everything on the mass
side must therefore be exactly -- not approximately -- independent of it.

**The closed form is the profile.**  It is the untruncated exponential of
Padmanabhan, Refregier & Amara (2017), and it stands in for the one truncated at
the halo only above a concentration -- the paper's own c_HI > 10.  The
parameter's bounds are the paper's flat prior, which sits well above that, so
the floor is a prior and not a numerical limit.

**The hot gas's shape was the wrong borrowing,** measured rather than argued:
kept here as a test so the reason for the choice survives the choice.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos import mass_definitions as MD
from ggah_mod.halos.field import make_field
from ggah_mod.halos.profiles import gnfw_shape, profile_uk_gl
from ggah_mod.sectors import BaryonSplit, ColdGasParams, ColdGasSector
from ggah_mod.sectors import coldgas as CG
from ggah_mod.sectors.matter import MatterField, matter_weights

pytestmark = pytest.mark.slow

F_B = PLANCK18.Omega_b / PLANCK18.Omega_m


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)


@pytest.fixture(scope="module")
def sector():
    return ColdGasSector("padmanabhan17")


def _near(field, lg):
    return int(np.argmin(np.abs(np.log10(np.asarray(field.m)) - lg)))


def _truncated(field, c_hi_0):
    """The same exponential, cut at the halo boundary, by quadrature."""
    r_delta = jnp.asarray(field.r_delta)
    r_s = r_delta / CG.c_hi_padmanabhan(field.m, field.z, c_hi_0=c_hi_0)
    return np.asarray(profile_uk_gl(
        field.k, r_delta, lambda r: jnp.exp(-r / r_s[:, None]), n_gl=400))


class TestTheConcentration:
    def test_c_hi_0_is_not_the_concentration(self):
        """Padmanabhan, Refregier & Amara (2017) Eq. (3) carries a factor 4: at
        the pivot and z = 0 the concentration is four times the parameter."""
        c = float(CG.c_hi_padmanabhan(1e11, 0.0, c_hi_0=CG.C_HI_0)[0])
        assert c == pytest.approx(4.0 * CG.C_HI_0, rel=1e-12)

    def test_it_falls_with_mass_and_with_redshift(self):
        m = jnp.asarray([1e11, 1e13, 1e15])
        c0 = np.asarray(CG.c_hi_padmanabhan(m, 0.0))
        assert np.all(np.diff(c0) < 0)
        c1 = np.asarray(CG.c_hi_padmanabhan(m, 1.0))
        assert np.allclose(c1 / c0, 2.0 ** -1.45, rtol=1e-12)

    def test_the_fiducial_is_the_published_one_moved_to_r200m(self):
        """Not 28.65: Padmanabhan, Refregier & Amara (2017) put r_s on the
        virial radius and the field carries R_200m, so the value, its width and
        the published flat prior [20, 400] all carry the ratio."""
        q = CG.R200M_OVER_RVIR_Z0
        assert CG.C_HI_0 == pytest.approx(28.65 * q, rel=1e-12)
        p = ColdGasParams._PARAMS["c_hi_0"]
        assert p.default == CG.C_HI_0
        assert p.prior.sigma == pytest.approx(1.76 * q, rel=1e-12)
        assert p.bounds == pytest.approx((20.0 * q, 400.0 * q), rel=1e-12)

    def test_the_ratio_is_what_the_mass_definitions_give(self):
        """A derivable number written as a literal goes stale silently unless
        something re-derives it."""
        m = jnp.asarray([1e11, 1e13, 1e15])
        r2 = np.asarray(MD.parse_mass_def("200m").radius(m, 0.0, PLANCK18))
        rv = np.asarray(MD.parse_mass_def("vir").radius(m, 0.0, PLANCK18))
        np.testing.assert_allclose(r2 / rv, CG.R200M_OVER_RVIR_Z0, atol=5e-5)

    def test_the_absorption_is_exact_only_at_z_zero(self):
        """The constant's docstring says so; this measures it.  Above z ~ 0.5
        the absorbed value makes the profile more compact than the paper
        intended, by 18 per cent at z = 1."""
        m = jnp.asarray([1e12])
        q = {z: float(np.asarray(MD.parse_mass_def("200m").radius(m, z, PLANCK18))[0]
                      / np.asarray(MD.parse_mass_def("vir").radius(m, z, PLANCK18))[0])
             for z in (0.0, 1.0, 5.0)}
        assert q[0.0] == pytest.approx(CG.R200M_OVER_RVIR_Z0, abs=5e-5)
        assert q[1.0] == pytest.approx(1.0032, abs=5e-4)
        assert q[5.0] < 1.0


class TestTheTransform:
    def test_it_goes_to_one_as_k_goes_to_zero(self, field, sector):
        """Two statements, and the second needs the right k to mean anything.

        At the grid's smallest k the transform is one to within 3.6e-10 at the
        largest haloes and to a few ULP at the smallest (1.8e-15) -- which is
        also why the approach to one cannot be tested there: at the low-mass
        end 1 - u is quantised in steps of the float spacing below one.  So the
        leading order, 1 - u = 2x^2 - 3x^4, is checked where x^2 is small but
        resolved at every mass.
        """
        p = ColdGasParams()
        u = np.asarray(sector.u_k(field, p))
        k = np.asarray(field.k)
        assert u.shape == (k.size, np.asarray(field.m).size)
        assert np.all(np.isfinite(u)) and np.all(u > 0) and np.all(u <= 1)
        assert np.max(1.0 - u[0]) < 1e-9

        r_s = (np.asarray(field.r_delta) / np.asarray(sector.c_hi(field, p)))
        j = int(np.argmin(np.abs(k * r_s.max() - 1e-3)))     # x^2 ~ 1e-6
        x2 = (k[j] * r_s) ** 2
        assert 1e-9 < x2.max() < 1e-5
        # r_s spans a factor ~450 over the grid, so no one k resolves x^2 at
        # every mass: relative where it is resolved, a few ULP where it is not.
        np.testing.assert_allclose(1.0 - u[j], 2.0 * x2 - 3.0 * x2 ** 2,
                                   rtol=1e-6, atol=1e-15)

    def test_it_is_the_closed_form(self, field, sector):
        p = ColdGasParams()
        r_s = (np.asarray(field.r_delta) / np.asarray(sector.c_hi(field, p)))
        k = np.asarray(field.k)
        want = (1.0 + (k[:, None] * r_s[None, :]) ** 2) ** -2
        np.testing.assert_allclose(np.asarray(sector.u_k(field, p)), want,
                                   rtol=1e-12)

    @pytest.mark.parametrize("c_hi_0, tol", [
        (CG.C_HI_0, 1e-12),
        (ColdGasParams._PARAMS["c_hi_0"].bounds[0], 1e-8),
        (ColdGasParams._PARAMS["c_hi_0"].bounds[1], 1e-12),
        (10.0, 1e-3),
    ])
    def test_the_untruncated_form_stands_in(self, field, sector, c_hi_0, tol):
        """The paper integrates to infinity and says its closed form holds for
        c_HI > 10.  Measured against the profile cut at R_Delta: 1.8e-14 at the
        fiducial, 9e-10 at the prior's floor, 4e-15 at its ceiling -- and still
        8.7e-4 at 10, well below the floor."""
        k = np.asarray(field.k)
        u = np.asarray(sector.u_k(field, ColdGasParams(c_hi_0=c_hi_0)))
        err = np.max(np.abs(u - _truncated(field, c_hi_0))[k <= 10.0])
        assert err < tol

    def test_below_ten_the_stand_in_fails(self, field, sector):
        """Where the paper's c_HI > 10 bites: at 6 the closed form is off by
        per cent and the HI is as extended as the dark matter.  The published
        prior's floor sits well above this, so it is not a numerical limit."""
        k = np.asarray(field.k)
        u = np.asarray(sector.u_k(field, ColdGasParams(c_hi_0=6.0)))
        err = np.max(np.abs(u - _truncated(field, 6.0))[k <= 10.0])
        assert err > 1e-2
        assert ColdGasParams._PARAMS["c_hi_0"].bounds[0] > 6.0

    def test_it_is_not_the_dark_matter_transform(self, field, sector):
        """What this replaced.  At 1e13 and k = 10 the HI keeps nearly all its
        power where the dark matter has lost most of its."""
        i = _near(field, 13.0)
        j = int(np.argmin(np.abs(np.asarray(field.k) - 10.0)))
        u = float(np.asarray(sector.u_k(field, ColdGasParams()))[j, i])
        nfw = float(np.asarray(field.u_nfw())[j, i])
        assert u > 1.5 * nfw

    def test_concentration_matters_at_small_scales(self, field, sector):
        i = _near(field, 13.0)
        j = int(np.argmin(np.abs(np.asarray(field.k) - 30.0)))
        lo = float(np.asarray(sector.u_k(field, ColdGasParams(c_hi_0=25.0)))[j, i])
        hi = float(np.asarray(sector.u_k(field, ColdGasParams(c_hi_0=60.0)))[j, i])
        assert hi > lo, "a more concentrated profile keeps more power at high k"


class TestWhyNotTheHotGasShape:
    """The first design borrowed the hot gas's generalised NFW.  At the same
    concentration it is nine times less compact than the profile the
    concentration was fitted with, because an outer slope below three keeps the
    enclosed mass growing to the truncation.  Kept as a test so the reason
    survives the change."""

    def _r_half(self, rho, r):
        m = np.cumsum(rho * r ** 2)
        return float(np.interp(0.5, m / m[-1], r))

    def test_the_gas_shape_does_not_compact_at_hi_concentrations(self):
        r = np.linspace(1e-6, 1.0, 200001)
        c = 114.6
        exp_half = self._r_half(np.exp(-r * c), r)
        gnfw_half = self._r_half(np.asarray(gnfw_shape(r * c, 1.0, 1.9, 2.7)), r)
        assert exp_half == pytest.approx(0.0233, abs=5e-4)
        assert gnfw_half == pytest.approx(0.2056, abs=5e-4)
        assert gnfw_half / exp_half > 8.0


class TestNoAmplitude:
    """The profile's normalisation is not a parameter, because it cancels."""

    @pytest.mark.parametrize("rel", ["padmanabhan17", "villaescusa18"])
    def test_the_mass_side_does_not_see_the_profile(self, field, rel):
        """Bit for bit, not approximately.  Omega_HI, b_HI and f_cold are
        integrals over M with no transform in them."""
        s = ColdGasSector(rel, calibration="off")
        a, b = ColdGasParams(), ColdGasParams(c_hi_0=60.0)
        assert float(s.omega_hi(field, a)) == float(s.omega_hi(field, b))
        assert float(s.bias_hi(field, a)) == float(s.bias_hi(field, b))
        np.testing.assert_array_equal(np.asarray(s.f_cold(field, a)),
                                      np.asarray(s.f_cold(field, b)))

    def test_the_weight_is_the_mass_times_a_unit_transform(self, field, sector):
        """``w_extended / m_hi`` is the transform itself, so there is nowhere
        for a separate profile amplitude to act."""
        p = ColdGasParams()
        w = np.asarray(sector.weights(field, p, view="hi").w_extended)
        m_hi = np.asarray(sector.m_hi(field, p))
        np.testing.assert_allclose(w / m_hi[None, :],
                                   np.asarray(sector.u_k(field, p)), rtol=1e-12)

    def test_every_view_uses_the_same_transform(self, field, sector):
        p = ColdGasParams()
        mass = {"hi": sector.m_hi, "h2": sector.m_h2, "mass": sector.m_neutral}
        u = {v: np.asarray(sector.weights(field, p, view=v).w_extended)
             / np.asarray(mass[v](field, p))[None, :]
             for v in ColdGasSector.VIEWS}
        np.testing.assert_allclose(u["hi"], u["mass"], rtol=1e-12)
        np.testing.assert_allclose(u["h2"], u["mass"], rtol=1e-12)


class TestTheMatterBudget:
    @pytest.fixture(scope="class")
    def split(self, field):
        f_cold = ColdGasSector("padmanabhan17").f_cold(field, ColdGasParams())
        lm = jnp.log10(field.m)
        from ggah_mod.sectors import energetics as E
        # f_star_sat must be NON-ZERO.  With it at zero, (0 + f) u and
        # 0 u + f u are the same float, and the bit-identity test below passed
        # against the very rewrite it exists to refuse -- which is how it was
        # found to be checking nothing.
        return BaryonSplit.from_hot(F_B, E.f_gas_sigmoid(lm, F_B),
                                    f_star_cen=0.01, f_star_sat=0.0037,
                                    f_cold=f_cold)

    def test_no_profile_is_bit_identical_to_before(self, field, split):
        """``u_cold`` defaults to ``u_sat``, and the default path keeps the
        shipped expression literally -- `(a + b) u` is not `a u + b u` in
        floating point; measured, the obvious rewrite moves 619 of 131072
        weights by up to 2.9e-11.  Only a test when ``f_star_sat`` is non-zero;
        see the fixture."""
        assert float(np.min(np.asarray(split.f_star_sat))) > 0.0
        u_sat = field.u_nfw() * 0.97
        new = np.asarray(matter_weights(field, split, u_sat=u_sat).w_extended)
        m_over_rho = field.m / field.rho_matter
        u_dm = field.u_nfw()
        old = np.asarray(m_over_rho * (split.f_collisionless * u_dm
                                       + split.f_hot * u_dm
                                       + (split.f_star_sat + split.f_cold) * u_sat
                                       + split.f_ejected * u_dm))
        np.testing.assert_array_equal(new, old)

    def test_the_neutral_gas_can_be_put_on_its_own_profile(self, field, split,
                                                          sector):
        u_cold = sector.u_k(field, ColdGasParams())
        base = np.asarray(matter_weights(field, split).w_extended)
        moved = np.asarray(matter_weights(field, split,
                                          u_cold=u_cold).w_extended)
        assert np.max(np.abs(moved - base)) > 0.0
        # ... and only through the cold-gas term: at k -> 0, where every
        # transform is one, it is the same mass either way.
        np.testing.assert_allclose(moved[0], base[0], rtol=1e-6)

    def test_one_profile_for_one_gas(self, field, split, sector):
        """The standalone tracer used u_nfw while the matter budget used u_sat;
        with ``u_cold`` the matter field's neutral-gas term is the tracer's own
        weight, scaled to a fraction of the matter density."""
        p = ColdGasParams()
        u_cold = sector.u_k(field, p)
        with_cold = np.asarray(matter_weights(field, split,
                                              u_cold=u_cold).w_extended)
        without = np.asarray(matter_weights(
            field, split.replace(f_cold=jnp.zeros_like(split.f_cold)),
            u_cold=u_cold).w_extended)
        cold_term = with_cold - without
        m_over_rho = np.asarray(field.m / field.rho_matter)
        want = (np.asarray(split.f_cold) * m_over_rho)[None, :] * np.asarray(u_cold)
        np.testing.assert_allclose(cold_term, want, rtol=1e-9, atol=1e-30)

    def test_the_field_built_with_the_sector_asks_it(self, field, split, sector):
        """The default when the neutral-gas sector is there: its exponential.

        Not the satellites' profile.  The two part company where the halo
        stops being the right scale: c_HI ~ 30 against a halo concentration of
        about 5.
        """
        p = ColdGasParams()
        want = np.asarray(matter_weights(field, split,
                                         u_cold=sector.u_k(field, p)).w_extended)
        got = np.asarray(MatterField(coldgas=sector).weights(
            field, {"split": split}, coldgas_params=p).w_extended)
        np.testing.assert_array_equal(got, want)
        on_sat = np.asarray(matter_weights(field, split).w_extended)
        assert np.max(np.abs(got - on_sat)) > 0.0

    def test_without_the_sector_it_is_still_the_satellites(self, field, split):
        """The fraction does not carry a shape, so nothing is invented."""
        want = np.asarray(matter_weights(field, split).w_extended)
        got = np.asarray(MatterField().weights(field, {"split": split}).w_extended)
        np.testing.assert_array_equal(got, want)

    def test_an_explicit_profile_wins_over_the_sector(self, field, split, sector):
        """A caller who passes one has said what they mean."""
        mine = sector.u_k(field, ColdGasParams()) * 0.5 + 0.5
        want = np.asarray(matter_weights(field, split, u_cold=mine).w_extended)
        got = np.asarray(MatterField(coldgas=sector).weights(
            field, {"split": split, "u_cold": mine},
            coldgas_params=ColdGasParams()).w_extended)
        np.testing.assert_array_equal(got, want)

    def test_the_alignment_field_reads_the_same_gas(self, field, split, sector):
        """P_II is the matter field scaled, so it may not carry a second one."""
        from ggah_mod.sectors.alignments import IntrinsicAlignmentSector
        p = ColdGasParams()
        params = {"split": split, "growth": 0.95}
        a = np.asarray(IntrinsicAlignmentSector(coldgas=sector).weights(
            field, params, coldgas_params=p).w_extended)
        b = np.asarray(IntrinsicAlignmentSector().weights(field, params).w_extended)
        assert np.max(np.abs(a - b)) > 0.0
        m = np.asarray(MatterField(coldgas=sector).weights(
            field, {"split": split}, coldgas_params=p).w_extended)
        ratio = a / m
        np.testing.assert_allclose(ratio, ratio[0, 0], rtol=1e-6)

    def test_the_matter_sector_forwards_it(self, field, split, sector):
        u_cold = sector.u_k(field, ColdGasParams())
        direct = np.asarray(matter_weights(field, split,
                                           u_cold=u_cold).w_extended)
        via = np.asarray(MatterField().weights(
            field, {"split": split, "u_cold": u_cold}).w_extended)
        np.testing.assert_array_equal(direct, via)


class TestLayerFour:

    @pytest.fixture(scope="class")
    def split(self, field):
        from ggah_mod.sectors import energetics as E
        f_cold = ColdGasSector("padmanabhan17").f_cold(field, ColdGasParams())
        lm = jnp.log10(field.m)
        return BaryonSplit.from_hot(F_B, E.f_gas_sigmoid(lm, F_B),
                                    f_star_cen=0.01, f_star_sat=0.0037,
                                    f_cold=f_cold)

    def test_layer_four_hands_the_sector_its_own_parameters(self, field, split,
                                                            sector):
        """The peer is what makes P_mm and P_HI one gas distribution."""
        from ggah_mod.spectra.spec import TracerSpec
        from ggah_mod.spectra.tracers import build_weights
        comp = TracerSpec.one("matter", None, name="matter").components[0]
        p = ColdGasParams()
        params = {"matter": {"split": split}, "coldgas": p}
        with_peer = np.asarray(build_weights(
            comp, field, {"matter": MatterField(coldgas=sector),
                          "coldgas": sector}, params).w_extended)
        want = np.asarray(matter_weights(field, split,
                                         u_cold=sector.u_k(field, p)).w_extended)
        np.testing.assert_array_equal(with_peer, want)
        alone = np.asarray(build_weights(
            comp, field, {"matter": MatterField()},
            {"matter": {"split": split}}).w_extended)
        assert np.max(np.abs(with_peer - alone)) > 0.0

    def test_two_neutral_gas_models_in_one_spectrum_are_refused(self, field,
                                                               split, sector):
        from ggah_mod.spectra.spec import TracerSpec
        from ggah_mod.spectra.tracers import build_weights
        comp = TracerSpec.one("matter", None, name="matter").components[0]
        other = ColdGasSector("padmanabhan17")
        with pytest.raises(ValueError, match="two different\\s+profiles"):
            build_weights(comp, field,
                          {"matter": MatterField(coldgas=sector),
                           "coldgas": other},
                          {"matter": {"split": split}, "coldgas": ColdGasParams()})

    def test_the_suppression_keeps_its_reference(self, field, split, sector):
        """S(k)'s reference is every component on u_DM, the cold gas included.

        The profile arrives as a peer rather than in the matter block, so
        `matter_suppression` has to drop the peer as well as the block's
        transforms; otherwise the ratio would hold the neutral gas fixed on its
        exponential and report a suppression that is partly no suppression.
        """
        from ggah_mod.spectra import tracers as TR
        from ggah_mod.spectra.spec import PkOptions
        opts = PkOptions.from_backend(DIFFERENTIABLE, bnl=False)
        params = {"matter": {"split": split}, "coldgas": ColdGasParams()}
        sectors = {"matter": MatterField(coldgas=sector), "coldgas": sector}
        k, s_peer = TR.matter_suppression(field, sectors, params, options=opts)
        _, s_bare = TR.matter_suppression(
            field, {"matter": MatterField()}, {"matter": {"split": split}},
            options=opts)
        s_peer, s_bare = np.asarray(s_peer), np.asarray(s_bare)
        assert abs(s_peer[0] - 1.0) < 1e-6 and abs(s_bare[0] - 1.0) < 1e-6
        # The reference is the same spectrum either way: the difference between
        # the two suppressions is the numerator alone.
        p_dmo = TR.spectrum(field, "matter", "matter", sectors,
                            {"matter": {"split": split}}, options=opts)
        p_dmo_bare = TR.spectrum(field, "matter", "matter",
                                 {"matter": MatterField()},
                                 {"matter": {"split": split}}, options=opts)
        np.testing.assert_allclose(np.asarray(p_dmo.total),
                                   np.asarray(p_dmo_bare.total), rtol=1e-12)
        assert np.max(np.abs(s_peer - s_bare)) > 0.0

    def test_a_21cm_spectrum_builds_on_its_own(self, field):
        from ggah_mod.spectra.spec import TracerSpec
        from ggah_mod.spectra.tracers import build_weights
        comp = TracerSpec.one("coldgas", "hi", name="hi").components[0]
        w = build_weights(comp, field,
                          {"coldgas": ColdGasSector("padmanabhan17")},
                          {"coldgas": ColdGasParams()})
        assert np.all(np.isfinite(np.asarray(w.w_extended)))
