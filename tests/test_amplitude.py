"""The top-hat variance, and the amplitudes derived from it.

No Boltzmann solver: these run on analytic spectra, so they test the quadrature
and the conventions rather than the physics.
"""
import numpy as np
import pytest

from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.amplitude import (
    tophat_window, sigma2_tophat, sigma_tophat, sigma8, s8,
    ln10A_s_for_sigma8,
)

_K = np.logspace(-4, 2, 2048)


def _power_law(k, n=-1.5, amp=1e4):
    return amp * k ** n


class TestTopHatWindow:
    def test_limit_at_zero_is_one(self):
        """The closed form is 0/0 here; the series branch must carry it."""
        assert float(tophat_window(0.0)) == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("x", [1e-8, 1e-5, 1e-3, 0.05, 0.0999])
    def test_series_branch_matches_the_exact_series(self, x):
        assert float(tophat_window(x)) == pytest.approx(
            1.0 - x ** 2 / 10.0 + x ** 4 / 280.0, rel=1e-14)

    def test_branches_agree_at_the_switch(self):
        """A step at x = 0.1 would put a step in every sigma(M).

        Both branches are evaluated at the *same* x -- comparing W(0.0999) with
        W(0.1001) instead measures the slope of W, which at 4e-6 swamps the
        thing being tested.  The residual here is the first dropped series
        term, x^6/15120 = 6.6e-11.
        """
        x = 0.1
        series = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0
        closed = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        assert abs(series - closed) < 1e-10
        assert float(tophat_window(x)) == pytest.approx(closed, abs=1e-10)

    @pytest.mark.parametrize("x", [0.5, 1.0, 5.0, 50.0])
    def test_closed_form_branch(self, x):
        assert float(tophat_window(x)) == pytest.approx(
            3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3, rel=1e-13)

    def test_decays_at_large_x(self):
        assert abs(float(tophat_window(200.0))) < 1e-3

    def test_no_catastrophic_cancellation(self):
        """The closed form loses ~12 digits by x ~ 1e-4; the series must not."""
        x = 1e-4
        naive = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
        ours = float(tophat_window(x))
        assert ours == pytest.approx(1.0 - x ** 2 / 10.0, rel=1e-15)
        assert abs(ours - 1.0) < 1e-8 and abs(naive - 1.0) < 1e-3  # naive is noisy


class TestSigma2:
    def test_scales_linearly_with_amplitude(self):
        p = _power_law(_K)
        a = float(sigma2_tophat(p, _K, 8.0))
        b = float(sigma2_tophat(3.0 * p, _K, 8.0))
        assert b / a == pytest.approx(3.0, rel=1e-12)

    def test_decreases_with_radius(self):
        p = _power_law(_K)
        s = np.asarray(sigma2_tophat(p, _K, np.array([1.0, 4.0, 8.0, 16.0])))
        assert np.all(np.diff(s) < 0)

    def test_vectorised_matches_scalar(self):
        p = _power_law(_K)
        radii = np.array([2.0, 8.0, 30.0])
        vec = np.asarray(sigma2_tophat(p, _K, radii))
        for i, r in enumerate(radii):
            assert vec[i] == pytest.approx(float(sigma2_tophat(p, _K, r)), rel=1e-12)

    def test_scalar_radius_returns_scalar(self):
        assert np.ndim(sigma2_tophat(_power_law(_K), _K, 8.0)) == 0

    @pytest.mark.parametrize("n_k,tol", [(512, 3e-6), (1024, 2e-7), (2048, 2e-7)])
    def test_quadrature_convergence_is_what_is_claimed(self, n_k, tol):
        """Measured against a 16384-point reference, so the numbers in the
        Backend grid choices mean something.

        512 points over six decades reaches 1.5e-6 and 1024 reaches 7.3e-8 --
        which is a genuine, quantified difference between the DIFFERENTIABLE and ACCURATE
        flavours, and belongs in the parity budget rather than in a comment.
        """
        ref_k = np.logspace(-4, 2, 16384)
        ref = float(sigma_tophat(_power_law(ref_k), ref_k, 8.0))
        k = np.logspace(-4, 2, n_k)
        assert float(sigma_tophat(_power_law(k), k, 8.0)) == pytest.approx(ref, rel=tol)

    def test_sigma_is_the_root(self):
        p = _power_law(_K)
        assert float(sigma_tophat(p, _K, 8.0)) ** 2 == pytest.approx(
            float(sigma2_tophat(p, _K, 8.0)), rel=1e-14)


class TestDerivedAmplitudes:
    def test_sigma8_is_sigma_at_R8(self):
        p = _power_law(_K)
        assert float(sigma8(p, _K)) == pytest.approx(
            float(sigma_tophat(p, _K, C.R8)), rel=1e-15)

    def test_s8_definition(self, cosmo):
        p = _power_law(_K)
        assert float(s8(p, _K, cosmo)) == pytest.approx(
            float(sigma8(p, _K)) * np.sqrt(cosmo.Omega_m / 0.3), rel=1e-14)

    def test_s8_equals_sigma8_at_the_pivot(self, cosmo):
        p = _power_law(_K)
        c = cosmo.replace(Omega_m=0.3)
        assert float(s8(p, _K, c)) == pytest.approx(float(sigma8(p, _K)), rel=1e-14)


class TestSolvingForTheAmplitude:
    r"""``ln10A_s_for_sigma8`` -- the only supported way in from a sigma_8.

    Every line of its loop was unexecuted by this suite until now, which is
    the kind of gap a percentage hides: the function is small, public, and the
    documented entry point for anyone migrating a sigma_8-parameterised model.
    """

    @staticmethod
    def _pk():
        from conftest import AnalyticPk
        return AnalyticPk()

    @pytest.mark.parametrize("target", [0.75, 0.8111, 0.88])
    def test_it_hits_the_requested_sigma8(self, target):
        from ggah_mod.cosmology import PLANCK18

        pk = self._pk()
        lnA = ln10A_s_for_sigma8(target, PLANCK18, pk)
        k = np.logspace(-4, 2, 512)
        got = sigma8(np.asarray(pk.pk(k, 0.0, PLANCK18.replace(ln10A_s=lnA))), k)
        assert float(got) == pytest.approx(target, rel=1e-6)

    def test_it_converges_in_essentially_one_step(self):
        r"""The docstring's claim, tested rather than trusted.

        :math:`\sigma_8^2 \propto A_s`, so a single evaluation fixes the
        answer and the loop exists only to absorb a backend's non-linearity.
        With ``max_iter=2`` there is one update and one confirmation, so this
        passing *is* the linearity claim.
        """
        from ggah_mod.cosmology import PLANCK18

        lnA = ln10A_s_for_sigma8(0.85, PLANCK18, self._pk(), max_iter=2)
        assert np.isfinite(lnA)

    def test_it_raises_rather_than_returning_an_unconverged_amplitude(self):
        """A tolerance it cannot reach must not come back as a number.

        ``tol=0`` is unreachable in floating point, so this drives the loop to
        exhaustion -- the one path that distinguishes "solved" from "gave up".
        """
        from ggah_mod.cosmology import PLANCK18

        with pytest.raises(RuntimeError, match="did not converge"):
            ln10A_s_for_sigma8(0.85, PLANCK18, self._pk(), tol=0.0, max_iter=3)
