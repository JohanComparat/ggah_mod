r"""Verification: intrinsic alignments as a sector.

``PLAN.md`` item **E3**, and the benchmark's reason: *"A weak-lensing likelihood
cannot be written without one."*  What is worth testing is not the algebra of
the NLA model, which is one line, but the two claims the *shape* of the
implementation makes: that the sign is right, and that writing it as a tracer
rather than as a correction gets :math:`P_{II}` and :math:`P_{gI}` out of layer
4's existing integral with no new code there.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.growth import growth_factor
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors.alignments import C1_RHO_CRIT, alignment_bias

Z = 0.5


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk=pk, z=Z, calibration="off")


@pytest.fixture(scope="module")
def growth(pk):
    return float(growth_factor(Z, PLANCK18, pk))


@pytest.fixture(scope="module")
def setup(field, growth):
    split = S.BaryonSplit.from_hot(0.16, jnp.full(field.n_m, 0.1))
    gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
    sectors = {
        "galaxies": gal,
        "gas": S.HotGasDPM(backend=DIFFERENTIABLE), "agn": S.AgnSector(gal),
        "matter": S.MatterField(),
        # The plain NLA: these tests are of the tracer's algebra, which the
        # red population's amplitude only rescales (TestOnlyRedGalaxiesAlign).
        "alignments": S.IntrinsicAlignmentSector(population="all"),
    }
    params = {
        "galaxies": S.galaxy_defaults("zumandelbaum15"),
        "gas": S.DpmParams(), "agn": S.AgnParams(),
        "matter": {"split": split},
        "alignments": {"split": split, "growth": growth,
                       "params": S.IaParams(a_ia=1.0)},
    }
    return sectors, params


class TestTheSignIsThePhysics:
    r"""Galaxies align *along* the stretch; lensing shears *across* it."""

    def test_the_bias_is_negative_for_a_positive_amplitude(self, growth):
        b = float(alignment_bias(Z, growth, S.IaParams(a_ia=1.0)))
        assert b < 0.0
        assert b == pytest.approx(
            -C1_RHO_CRIT * 0.3 / growth, rel=1e-12)

    def test_the_cross_with_galaxies_anticorrelates(self, field, setup):
        """Which is what makes intrinsic alignment a *contaminant* of
        galaxy-galaxy lensing rather than an addition to it.  A model that came
        out positive would be adding to the signal it exists to subtract."""
        sectors, params = setup
        gm = np.asarray(SP.spectrum(field, "galaxies", "matter", sectors,
                                    params).total)
        gi = np.asarray(SP.spectrum(field, "galaxies", "alignments", sectors,
                                    params).total)
        assert np.all(gm > 0.0)
        assert np.all(gi < 0.0)

    def test_a_negative_amplitude_flips_it(self, field, setup, growth):
        """Both signs are physical and the bound admits both: a prior excluding
        the sign no measurement supports would make a null result
        unrepresentable."""
        sectors, params = setup
        flipped = dict(params, alignments=dict(
            params["alignments"], params=S.IaParams(a_ia=-1.0)))
        gi = np.asarray(SP.spectrum(field, "galaxies", "alignments", sectors,
                                    flipped).total)
        assert np.all(gi > 0.0)

    def test_the_auto_spectrum_is_positive(self, field, setup):
        """`P_II` goes as `b_I^2`, so it is positive whatever the sign of the
        amplitude -- which is why the cross is the one that identifies it."""
        sectors, params = setup
        ii = np.asarray(SP.spectrum(field, "alignments", "alignments", sectors,
                                    params).total)
        assert np.all(ii > 0.0)


class TestItIsATracerAndThatIsWhyThereIsNoLayerFourCode:
    r"""The shape claim, measured: layer 4's one integral does the work."""

    def test_the_cross_is_exactly_b_i_times_the_matter_cross(
            self, field, setup, growth):
        sectors, params = setup
        b_i = float(alignment_bias(Z, growth, S.IaParams(a_ia=1.0)))
        gm = np.asarray(SP.spectrum(field, "galaxies", "matter", sectors,
                                    params).total)
        gi = np.asarray(SP.spectrum(field, "galaxies", "alignments", sectors,
                                    params).total)
        assert np.allclose(gi / gm, b_i, rtol=1e-12)

    def test_the_auto_is_exactly_b_i_squared_times_the_matter_auto(
            self, field, setup, growth):
        sectors, params = setup
        b_i = float(alignment_bias(Z, growth, S.IaParams(a_ia=1.0)))
        mm = np.asarray(SP.spectrum(field, "matter", "matter", sectors,
                                    params).total)
        ii = np.asarray(SP.spectrum(field, "alignments", "alignments", sectors,
                                    params).total)
        assert np.allclose(ii / mm, b_i ** 2, rtol=1e-12)

    def test_it_is_a_sector_by_the_protocol(self):
        from ggah_mod.sectors.protocol import Sector

        s = S.IntrinsicAlignmentSector()
        assert isinstance(s, Sector)
        assert s.name == "alignments" and s.differentiable is True

    def test_lensing_is_still_refused_and_this_is_not(self):
        """The registry refuses shear and convergence because they are not 3D
        fields -- they are the matter field under a radial kernel, which is
        layer 5's business.  An intrinsic alignment *is* a 3D field: it is a
        property of the galaxies at their own redshift.  The distinction is the
        reason this entry is allowed and those are not."""
        from ggah_mod.spectra.tracers import resolve

        assert resolve("alignments").name == "alignments"
        for absent in ("shear", "convergence", "lensing"):
            with pytest.raises(ValueError, match="unknown tracer"):
                resolve(absent)


class TestWhatItRefusesToGuess:
    """Two required inputs, each for a reason the module states."""

    def test_it_needs_the_matter_split(self, field, growth):
        """The *same* split the matter field uses, so `P_gI` and `P_gm`
        describe one universe."""
        with pytest.raises(ValueError, match="needs `split`"):
            S.IntrinsicAlignmentSector().weights(field, {"growth": growth})

    def test_it_needs_the_growth_factor(self, field):
        """It cannot be taken from the field, and the two conventions differ by
        about 25 per cent -- a systematic wearing a plausible number."""
        split = S.BaryonSplit.from_hot(0.16, jnp.full(field.n_m, 0.1))
        with pytest.raises(ValueError, match="needs `growth`"):
            S.IntrinsicAlignmentSector().weights(field, {"split": split})

    def test_omega_m_ref_is_a_convention_not_the_cosmology(self):
        r"""Declared rather than read off ``cosmo.Omega_m``: the published NLA
        convention fixes it, so an :math:`A_{\rm IA}` carried in from a survey
        means what that survey meant.  Using the sampled value instead is a
        defensible different convention and a silent change of meaning."""
        p = S.IaParams()
        assert p.omega_m_ref == 0.3
        assert float(PLANCK18.Omega_m) != p.omega_m_ref
        assert "convention" in S.IaParams._PARAMS["omega_m_ref"].why


class TestItDifferentiates:
    """Layer 3's rule, and the amplitude is the parameter a survey fits."""

    @pytest.mark.parametrize("name", ["a_ia", "eta_ia", "beta_ia"])
    def test_every_free_parameter_carries_a_gradient(self, growth, name):
        def f(v):
            p = S.IaParams(**{name: v, "beta_ia": 1.0 if name != "beta_ia" else v,
                              "l_over_l0": 1.5})
            return alignment_bias(Z, growth, p)

        g = float(jax.grad(f)(0.5))
        assert np.isfinite(g) and g != 0.0

    def test_the_amplitude_reaches_the_spectrum(self, field, setup, growth):
        sectors, params = setup

        def p_gi(a):
            q = dict(params, alignments=dict(params["alignments"],
                                             params=S.IaParams(a_ia=a)))
            return jnp.sum(SP.spectrum(field, "galaxies", "alignments",
                                       sectors, q).total)

        g = float(jax.grad(p_gi)(1.0))
        assert np.isfinite(g) and g < 0.0


#: Wright et al. (2025, KiDS-Legacy), Table B.1: per tomographic bin, the mean
#: redshift, log10 M_eff [Msun/h] and the red fraction of the shape sample.
WRIGHT25_B1 = ((0.34, 11.69, 0.15), (0.48, 12.46, 0.20), (0.59, 12.76, 0.17),
               (0.79, 12.93, 0.24), (0.94, 13.08, 0.19), (1.23, 13.21, 0.03))


class TestOnlyRedGalaxiesAlignByDefault:
    r"""The NLA-M model of Wright et al. (2025, Eq. 11): the red fraction
    aligns, with the amplitude of Fortuna et al. (2025) scaled by the red
    galaxies' mean host halo mass; the blue galaxies do not."""

    def test_the_default_is_red_with_fortunas_amplitude(self):
        assert S.IntrinsicAlignmentSector().population == "red"
        p = S.IaParams()
        assert (p.a_ia, p.beta_m, p.log10_mh_pivot) == (5.74, 0.44, 13.5)
        with pytest.raises(ValueError, match="unknown population"):
            S.IntrinsicAlignmentSector(population="blue")

    @pytest.mark.parametrize("z, log10_mh, f_r", WRIGHT25_B1)
    def test_it_is_wrights_eq_11_at_their_table_b1(self, z, log10_mh, f_r):
        """Their A_IA,total (Eq. 12) at the prior's means, times the NLA
        normalisation."""
        g = 0.8
        a_total = 5.74 * f_r * (10.0 ** (log10_mh - 13.5)) ** 0.44
        p = S.IaParams(z_pivot=z)     # no redshift term: eta_ia = 0 either way
        b = float(alignment_bias(z, g, p, f_r, log10_mh))
        assert b == pytest.approx(-a_total * C1_RHO_CRIT * 0.3 / g, rel=1e-12)

    def test_the_red_and_blue_fractions_sum_to_one(self, field):
        ia = S.IntrinsicAlignmentSector(calibration="off")
        f_red, log10_mh = ia.red_sample(field)
        p = S.galaxy_defaults("zumandelbaum16_blue")
        blue = S.GalaxySector("zumandelbaum16_blue", calibration="off")
        every = S.GalaxySector("zumandelbaum15", calibration="off")
        f_blue = float(blue.number_density(field, p) / every.number_density(field, p))
        assert 0.0 < float(f_red) < 1.0
        assert float(f_red) + f_blue == pytest.approx(1.0, rel=1e-10)
        assert 11.0 < float(log10_mh) < 14.5

    def test_a_more_massive_sample_is_redder_and_in_heavier_haloes(self, field):
        ia = S.IntrinsicAlignmentSector(calibration="off")
        lo = ia.red_sample(field, {"log10m_star_thresh": 9.5})
        hi = ia.red_sample(field, {"log10m_star_thresh": 10.8})
        assert float(hi[0]) > float(lo[0]) and float(hi[1]) > float(lo[1])

    def test_the_spectrum_carries_the_red_amplitude(self, field, setup, growth):
        """P_gI(red) / P_gI(all) at one IaParams is f_red times the mass
        term, and nothing else changes."""
        sectors, params = setup
        red = S.IntrinsicAlignmentSector(calibration="off")
        f_red, log10_mh = red.red_sample(field)
        ia = S.IaParams()
        q = dict(params, alignments=dict(params["alignments"], params=ia))
        gi_all = np.asarray(SP.spectrum(field, "galaxies", "alignments", sectors, q).total)
        gi_red = np.asarray(SP.spectrum(field, "galaxies", "alignments",
                                        dict(sectors, alignments=red), q).total)
        want = float(f_red) * 10.0 ** (0.44 * (float(log10_mh) - 13.5))
        assert np.allclose(gi_red / gi_all, want, rtol=1e-10)

    def test_given_numbers_take_precedence(self, field, setup):
        _, params = setup
        red = S.IntrinsicAlignmentSector(calibration="off")
        q = dict(params["alignments"], params=S.IaParams(), f_red=0.2, log10_mh=13.5)
        w = red.weights(field, q)
        a = S.IntrinsicAlignmentSector(population="all").weights(
            field, dict(q, params=S.IaParams(a_ia=5.74 * 0.2)))
        assert np.allclose(np.asarray(w.w_extended), np.asarray(a.w_extended), rtol=1e-12)

    def test_the_quenching_reaches_the_amplitude(self, field, growth):
        """f_red is a function of the galaxy sector's quenching parameters, and
        differentiable in them."""
        ia = S.IntrinsicAlignmentSector(calibration="off")
        p0 = S.galaxy_defaults("zumandelbaum16_red")
        name = next(n for n in ("lg_mh_qc", "lg_mh_qs") if n in p0)

        def b(v):
            f, m = ia.red_sample(field, {name: v})
            return alignment_bias(Z, growth, S.IaParams(), f, m)

        g = float(jax.grad(b)(float(p0[name])))
        assert np.isfinite(g) and g != 0.0
