r"""The halo field: sigma(M) -> dn/dM -> b(M).

The centrepiece is :class:`TestPressSchechterIsExact`.  Press-Schechter with its
own peak-background-split bias is analytically closed, so the whole chain --
variance, the mass-function algebra, the bias, and both integrals -- can be
checked against closed form with no reference package and no tolerance chosen by
taste.  For a spectrum truncated at :math:`\sigma_{\max}` (which every real
spectrum is, because the k grid ends):

.. math::

    \int \frac{M}{\bar\rho}\frac{dn}{dM}dM &= \mathrm{erfc}(\nu_0/\sqrt2) \\
    \int b\,\frac{M}{\bar\rho}\frac{dn}{dM}dM
      &= \mathrm{erfc}(\nu_0/\sqrt2)
       + \frac{1}{\delta_c}\sqrt{\frac{2}{\pi}}\,\nu_0 e^{-\nu_0^2/2}

with :math:`\nu_0 = \delta_c/\sigma_{\max}`.  Both tend to 1 as the spectrum
extends to arbitrarily small scales, which is the peak-background split.
"""
import numpy as np
import pytest
from scipy.special import erfc

from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.cosmology.power import make_pk
from ggah_mod.cosmology.amplitude import sigma_tophat
from ggah_mod.halos import (
    lagrangian_radius, mass_from_radius, sigma_of_mass, dln_sigma_dln_mass,
    dndm, make_bias, make_multiplicity, mass_weighted_bias, mass_fraction,
    DELTA_C,
)

pytestmark = pytest.mark.slow

_K = np.logspace(-4, np.log10(300.0), 1400)
_M = np.logspace(10, 16, 200)


@pytest.fixture(scope="module")
def spectrum():
    pk = make_pk("class", k_max=300.0, n_k=1200)
    return np.asarray(pk.pk_cb(_K, 0.0, PLANCK18))


class TestLagrangianRadius:
    def test_round_trip(self):
        c = PLANCK18
        r = lagrangian_radius(_M, c.rho_cold)
        np.testing.assert_allclose(np.asarray(mass_from_radius(r, c.rho_cold)),
                                   _M, rtol=1e-12)

    def test_uses_the_cold_density_not_the_total(self):
        """Halos form out of the matter that collapses; neutrinos do not.
        Passing rho_matter instead is a 0.15% error in R, and 0.5% in M."""
        c = PLANCK18
        r_cold = float(lagrangian_radius(1e13, c.rho_cold))
        r_tot = float(lagrangian_radius(1e13, c.rho_matter))
        assert r_cold > r_tot
        assert abs(r_cold / r_tot - 1.0) == pytest.approx(
            (c.rho_matter / c.rho_cold) ** (1 / 3) - 1.0, rel=1e-9)


class TestVariance:
    def test_decreasing_in_mass(self, spectrum):
        s = np.asarray(sigma_of_mass(_M, _K, spectrum, PLANCK18.rho_cold))
        assert np.all(np.diff(s) < 0)

    def test_derivative_matches_finite_difference(self, spectrum):
        """Autodiff, not a finite difference of an integral -- which is the
        classic way to get percent noise into dn/dM."""
        c = PLANCK18
        ad = np.asarray(dln_sigma_dln_mass(_M, _K, spectrum, c.rho_cold))
        for target in (1e11, 1e13, 1e15):
            h = 1e-4
            f = lambda mm: np.log(float(sigma_tophat(
                spectrum, _K, float(lagrangian_radius(mm, c.rho_cold)))))
            fd = (f(target * (1 + h)) - f(target * (1 - h))) / (2 * h)
            got = float(np.interp(np.log10(target), np.log10(_M), ad))
            # The finite difference is the approximation here, not the
            # autodiff: its O(h^2) truncation error is the residual.
            assert got == pytest.approx(fd, rel=1e-5)

    def test_derivative_is_negative(self, spectrum):
        d = np.asarray(dln_sigma_dln_mass(_M, _K, spectrum, PLANCK18.rho_cold))
        assert np.all(d < 0)

    def test_construction_is_not_separable(self):
        """sigma(M,z) is built from P_cb(k,z) at the redshift wanted, so
        sigma(M,z)/sigma(M,0) is free to depend on M -- and with a large enough
        neutrino mass it must.

        Checked at 0.3 eV rather than the fiducial 0.06 eV, because the top-hat
        window averages most of the scale-dependence away: the mass-dependence
        is 3e-5 at 0.06 eV but 1.4e-3 at 0.30 eV.  A test at the fiducial mass
        would be indistinguishable from quadrature noise and would pass against
        a separable implementation too.
        """
        c = Cosmology.create(sum_mnu=0.30)
        pk = make_pk("class")
        p0 = np.asarray(pk.pk_cb(_K, 0.0, c))
        p1 = np.asarray(pk.pk_cb(_K, 3.0, c))
        r = (np.asarray(sigma_of_mass(_M, _K, p1, c.rho_cold))
             / np.asarray(sigma_of_mass(_M, _K, p0, c.rho_cold)))
        assert np.max(np.abs(r / np.median(r) - 1.0)) > 5e-4, (
            "sigma(M,z)/sigma(M,0) is mass-independent even at 0.3 eV -- a "
            "scalar growth factor has been reintroduced somewhere")

    def test_scale_dependence_is_small_at_the_fiducial_mass(self):
        """The honest counterpart: at 0.06 eV the separable form would have
        been wrong by only ~5e-5.  Pinned so that if this ever grows, the
        reason gets looked at."""
        c = PLANCK18
        pk = make_pk("class")
        p0 = np.asarray(pk.pk_cb(_K, 0.0, c))
        p1 = np.asarray(pk.pk_cb(_K, 3.0, c))
        r = (np.asarray(sigma_of_mass(_M, _K, p1, c.rho_cold))
             / np.asarray(sigma_of_mass(_M, _K, p0, c.rho_cold)))
        assert np.max(np.abs(r / np.median(r) - 1.0)) < 2e-4


class TestPressSchechterIsExact:
    """The analytic case, checked in closed form."""

    @staticmethod
    def _integrals(spectrum, lo=2, hi=17, n=1200):
        c = PLANCK18
        m = np.logspace(lo, hi, n)
        s = np.asarray(sigma_of_mass(m, _K, spectrum, c.rho_cold))
        d = np.asarray(dln_sigma_dln_mass(m, _K, spectrum, c.rho_cold))
        n_m = np.asarray(dndm(m, s, d, c.rho_cold, model="press74"))
        b = np.asarray(make_bias("press74")(s))
        return (float(mass_fraction(m, n_m, c.rho_cold)),
                float(mass_weighted_bias(m, n_m, b, c.rho_cold)),
                float(s.max()))

    def test_mass_fraction_matches_closed_form(self, spectrum):
        mf, _, smax = self._integrals(spectrum)
        nu0 = DELTA_C / smax
        assert mf == pytest.approx(erfc(nu0 / np.sqrt(2.0)), rel=2e-3)

    def test_bias_integral_matches_closed_form(self, spectrum):
        _, bb, smax = self._integrals(spectrum)
        nu0 = DELTA_C / smax
        expect = (erfc(nu0 / np.sqrt(2.0))
                  + np.sqrt(2.0 / np.pi) * nu0 * np.exp(-0.5 * nu0 ** 2) / DELTA_C)
        assert bb == pytest.approx(expect, rel=2e-3)

    def test_both_tend_to_one_as_the_spectrum_extends(self, spectrum):
        """The peak-background split.  The deficit at a finite k grid is not an
        error -- it is the mass below the smallest resolved scale."""
        narrow = self._integrals(spectrum, lo=11)
        wide = self._integrals(spectrum, lo=2)
        assert wide[0] > narrow[0] and wide[1] > narrow[1]
        assert wide[0] < 1.0 and wide[1] < 1.0


class TestMassFunction:
    @pytest.mark.parametrize("model", ["tinker08", "press74", "sheth99"])
    def test_positive_and_falling(self, spectrum, model):
        c = PLANCK18
        s = np.asarray(sigma_of_mass(_M, _K, spectrum, c.rho_cold))
        d = np.asarray(dln_sigma_dln_mass(_M, _K, spectrum, c.rho_cold))
        n = np.asarray(dndm(_M, s, d, c.rho_cold, model=model))
        assert np.all(n > 0) and np.all(np.diff(n) < 0)

    def test_unknown_model_refused(self):
        with pytest.raises(ValueError, match="unknown mass function"):
            make_multiplicity("tinker08b")

    def test_tinker08_covers_the_whole_table(self):
        """Superseded an earlier version that raised for Delta != 200.  The
        full Table 2 interpolation is now ported, so an intermediate
        overdensity is answered rather than refused -- and 500 sits between
        tabulated rows, which is the case interpolation exists for."""
        f = make_multiplicity("tinker08")
        v500 = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, 500.0)))[0])
        v400 = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, 400.0)))[0])
        v600 = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, 600.0)))[0])
        assert v600 < v500 < v400

    def test_redshift_evolution_is_present(self, spectrum):
        f = make_multiplicity("tinker08")
        s = np.array([1.0])
        assert float(np.asarray(f(s, z=0.0))[0]) != float(np.asarray(f(s, z=1.0))[0])


class TestBias:
    @pytest.mark.parametrize("model", ["tinker10", "sheth99", "press74"])
    def test_increases_with_mass(self, spectrum, model):
        s = np.asarray(sigma_of_mass(_M, _K, spectrum, PLANCK18.rho_cold))
        b = np.asarray(make_bias(model)(s))
        assert np.all(np.diff(b) > 0)

    def test_crosses_unity_at_nu_equals_one(self):
        """b = 1 exactly where sigma = delta_c, for the analytic model.

        Evaluated at sigma = delta_c directly rather than interpolating a mass
        grid twice, which was worth 2e-4 of interpolation error.
        """
        b = float(np.asarray(make_bias("press74")(np.array([DELTA_C])))[0])
        assert b == pytest.approx(1.0, abs=1e-14)

    def test_crosses_unity_near_m_star_on_a_real_spectrum(self, spectrum):
        s = np.asarray(sigma_of_mass(_M, _K, spectrum, PLANCK18.rho_cold))
        b = np.asarray(make_bias("press74")(s))
        m_star = np.interp(-DELTA_C, -s, _M)
        assert float(np.interp(np.log10(m_star), np.log10(_M), b)) == pytest.approx(1.0, abs=1e-3)

    def test_unknown_model_refused(self):
        with pytest.raises(ValueError, match="unknown bias model"):
            make_bias("tinker11")


class TestAgainstColossus:
    """An independent implementation, so a disagreement is a bug somewhere."""

    def test_sigma_and_bias(self, spectrum):
        colossus = pytest.importorskip("colossus")
        from colossus.cosmology import cosmology as cc
        from colossus.lss import bias as cbias

        c = PLANCK18
        cc.setCosmology("ggah_test", {
            "flat": True, "H0": 100 * c.h, "Om0": c.Omega_m, "Ob0": c.Omega_b,
            "sigma8": 0.8021, "ns": c.n_s, "relspecies": False, "persistence": ""})
        m = np.logspace(11, 15, 9)
        s_ours = np.asarray(sigma_of_mass(m, _K, spectrum, c.rho_cold))
        r = np.asarray(lagrangian_radius(m, c.rho_cold))
        s_col = cc.getCurrent().sigma(r, 0.0)
        # COLOSSUS uses an EH98 transfer function by default; ~1% is its shape
        # error against CLASS, not ours.
        assert np.max(np.abs(s_ours / s_col - 1.0)) < 0.02

        b_ours = np.asarray(make_bias("tinker10")(s_ours))
        b_col = cbias.haloBias(m, model="tinker10", z=0.0, mdef="200m")
        assert np.max(np.abs(b_ours / b_col - 1.0)) < 0.02


class TestAllMultiplicityFunctions:
    """Seventeen fits.  What can be asserted about all of them at once is shape
    and hygiene, not accuracy -- they disagree with each other by design, and
    each is right for its own halo definition.

    Sixteen take ``(sigma, z)``.  The seventeenth is recalibrated per cosmology
    and takes one, which is the whole point of it; these tests read
    ``COSMOLOGY_DEPENDENT_MULTIPLICITY`` rather than assuming a uniform
    signature, in the same way ``make_field`` does.
    """

    @staticmethod
    def _kw(name):
        from ggah_mod.cosmology import PLANCK18
        from ggah_mod.halos.mass_function import (
            COSMOLOGY_DEPENDENT_MULTIPLICITY)

        return {"cosmo": PLANCK18} if name in COSMOLOGY_DEPENDENT_MULTIPLICITY \
            else {}

    @pytest.mark.parametrize("name", sorted(
        __import__("ggah_mod.halos", fromlist=["MULTIPLICITY"]).MULTIPLICITY))
    def test_positive_finite_and_falling_at_high_nu(self, name):
        from ggah_mod.halos import make_multiplicity
        s = np.logspace(-0.6, 0.7, 40)          # sigma in [0.25, 5]
        f = np.atleast_1d(np.asarray(
            make_multiplicity(name)(s, 0.0, **self._kw(name))))
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        # The exponential cutoff: rare, massive halos (small sigma) are rarer.
        assert f[0] < f[len(f) // 2]

    @pytest.mark.parametrize("name", sorted(
        __import__("ggah_mod.halos", fromlist=["MULTIPLICITY"]).MULTIPLICITY))
    def test_order_of_magnitude_at_sigma_one(self, name):
        """Every fit is a multiplicity function of the same quantity, so at
        sigma = 1 they must agree to within a factor of two even though they
        were calibrated on different halo finders."""
        from ggah_mod.halos import make_multiplicity
        f = float(np.atleast_1d(np.asarray(make_multiplicity(name)(
            np.array([1.0]), 0.0, **self._kw(name))))[0])
        assert 0.15 < f < 0.5, f"{name}: f(sigma=1) = {f}"

    @pytest.mark.parametrize("name", sorted(
        __import__("ggah_mod.halos", fromlist=["MULTIPLICITY"]).MULTIPLICITY))
    def test_differentiable_in_sigma(self, name):
        """All seventeen are pure JAX, so any of them can sit inside DIFFERENTIABLE."""
        import jax
        import jax.numpy as jnp
        from ggah_mod.halos import make_multiplicity
        fn = make_multiplicity(name)
        kw = self._kw(name)
        g = jax.grad(lambda s: jnp.sum(
            jnp.atleast_1d(fn(jnp.array([s]), 0.0, **kw))))(1.0)
        assert np.isfinite(float(g)) and float(g) != 0.0

    def test_every_model_declares_its_calibration(self):
        """A fit applied outside the halo definition it was calibrated on is a
        systematic error; the table is what makes that checkable."""
        from ggah_mod.halos import MULTIPLICITY, CALIBRATION
        assert set(CALIBRATION) == set(MULTIPLICITY)
        for name, (defn, (z_lo, z_hi)) in CALIBRATION.items():
            assert isinstance(defn, str) and defn
            assert z_lo <= z_hi

    def test_tinker08_interpolates_over_overdensity(self):
        """The Table 2 parameters move by a factor of two across Delta, so
        pinning 200m and using it at 800 is not a small error."""
        from ggah_mod.halos import make_multiplicity
        f = make_multiplicity("tinker08")
        vals = [float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, d)))[0])
                for d in (200.0, 400.0, 800.0, 3200.0)]
        assert vals == sorted(vals, reverse=True)
        assert vals[0] / vals[-1] > 3.0

    def test_despali16_responds_to_the_overdensity_ratio(self):
        from ggah_mod.halos import make_multiplicity
        f = make_multiplicity("despali16")
        a = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, 1.0)))[0])
        b = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, 2.0)))[0])
        assert a != b

    def test_bocquet16_hydro_differs_from_dmo(self):
        from ggah_mod.halos import make_multiplicity
        f = make_multiplicity("bocquet16")
        dmo = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, False)))[0])
        hyd = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0, True)))[0])
        assert dmo != hyd

    @pytest.mark.parametrize("name", ["crocce10", "bhattacharya11", "tinker08",
                                      "bocquet16", "rodriguezpuebla16",
                                      "yung24", "yung25", "seppi20"])
    def test_redshift_dependent_models_respond_to_z(self, name):
        from ggah_mod.halos import make_multiplicity
        f = make_multiplicity(name)
        a = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0)))[0])
        b = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 1.0)))[0])
        assert a != b

    @pytest.mark.parametrize("name", ["press74", "sheth99", "jenkins01",
                                      "warren06", "angulo12", "watson13",
                                      "comparat17"])
    def test_redshift_independent_models_ignore_z(self, name):
        """Accepting z for a uniform interface and ignoring it is a real
        property: comparat17 is a z = 0 fit, and using it at z = 1 is an
        extrapolation the fit does not support."""
        from ggah_mod.halos import make_multiplicity
        f = make_multiplicity(name)
        a = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 0.0)))[0])
        b = float(np.atleast_1d(np.asarray(f(np.array([1.0]), 2.0)))[0])
        assert a == b


class TestMatchedBias:
    def test_partners_are_real_bias_models(self):
        from ggah_mod.halos import MATCHED_BIAS, BIAS, MULTIPLICITY
        for mf_name, b_name in MATCHED_BIAS.items():
            assert mf_name in MULTIPLICITY
            assert b_name in BIAS

    def test_absent_partner_returns_none(self):
        from ggah_mod.halos import matched_bias_for
        assert matched_bias_for("watson13") is None
        assert matched_bias_for("tinker08") == "tinker10"

    @pytest.mark.slow
    def test_matched_pairs_beat_ad_hoc_ones_on_the_consistency_integral(self, spectrum):
        """The peak-background split is what a matched pair satisfies and an
        ad hoc one does not.  Measured over a finite mass range, so neither
        reaches 1 -- the point is which gets closer."""
        from ggah_mod.halos import (make_bias, mass_weighted_bias, mass_fraction,
                                    matched_bias_for)
        c = PLANCK18
        mm = np.logspace(2, 17, 900)
        s = np.asarray(sigma_of_mass(mm, _K, spectrum, c.rho_cold))
        d = np.asarray(dln_sigma_dln_mass(mm, _K, spectrum, c.rho_cold))

        def ratio(mf_name, b_name):
            n = np.asarray(dndm(mm, s, d, c.rho_cold, model=mf_name))
            b = np.asarray(make_bias(b_name)(s))
            return (float(mass_weighted_bias(mm, n, b, c.rho_cold))
                    / float(mass_fraction(mm, n, c.rho_cold)))

        for mf_name in ("press74", "sheth99"):
            matched = ratio(mf_name, matched_bias_for(mf_name))
            ad_hoc = ratio(mf_name, "sheth01")
            assert abs(matched - 1.10) < abs(ad_hoc - 1.10)
