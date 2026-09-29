r"""Parity: the CLF family and the AGN kernel, against `hod_mod`.

Both are transcriptions, so the bar is round-off rather than a tolerance.

The AGN comparison is deliberately of the **kernel** alone, not of the
luminosity function.  The XLF folds in a halo mass function, and comparing it
would measure the two packages' HMFs as much as their AGN chains -- so a
disagreement would be uninterpretable.  ``P(log L_X | M_h)`` is
HMF-independent, and it is the whole of what was ported.
"""
import numpy as np
import pytest

import jax.numpy as jnp

from ggah_mod.sectors import agn as A
from ggah_mod.sectors import clf as C

pytestmark = pytest.mark.parity

hod_clf = pytest.importorskip(
    "hod_mod.connection.clf",
    reason="hod_mod is a test-only dependency; install it to run parity")
hod_powell = pytest.importorskip("hod_mod.agn.powell")

LOG10M = jnp.asarray(np.linspace(11.0, 15.5, 14))


def _same(got, want, tag="", atol_on_peak=1e-12):
    got, want = np.asarray(got), np.asarray(want)
    peak = max(float(np.max(np.abs(want))), 1e-300)
    assert np.max(np.abs(got - want)) / peak < atol_on_peak, (
        f"{tag}: {np.max(np.abs(got - want)) / peak:.3e}")


#: ``ggah_mod`` name -> ``hod_mod`` name, where the two disagree.
#:
#: ``PLAN.md`` item **G9** renamed the CLF faint-end slope and its
#: normalisation, because ``alpha_sat`` also named a *positive* satellite index
#: in the occupations and ``b_sat`` differed from their ``bsat`` by one
#: underscore.  The predecessor keeps both old spellings.
#:
#: Translating rather than reverting is what a parity suite is for: the numbers
#: must agree and the names need not, and matching the predecessor's names to
#: make a test pass would import the collision along with them.
_HOD_MOD_NAMES = {"alpha_faint": "alpha_sat", "phi_s_amp": "b_sat"}


def _for_hod_mod(params: dict) -> dict:
    """The same values under the predecessor's spelling."""
    return {_HOD_MOD_NAMES.get(k, k): v for k, v in params.items()}


class TestConditionalLuminosityFunction:
    P = C.CLF_DEFAULTS["cacciato09"]

    def test_central_luminosity(self):
        k = ("log10l0", "log10m1", "alpha_cen", "beta_cen")
        p = {n: self.P[n] for n in k}
        _same(C.log10_lc(LOG10M, **p), hod_clf.log10_lc(LOG10M, **p))

    def test_central_occupation(self):
        k = ("log10l_lim", "log10l0", "log10m1", "alpha_cen", "beta_cen",
             "sigma_c")
        p = {n: self.P[n] for n in k}
        _same(C.clf_central_mean(LOG10M, **p),
              hod_clf.clf_central_mean(LOG10M, **_for_hod_mod(p)))

    def test_satellite_occupation(self):
        _same(C.clf_satellite_mean(LOG10M, **self.P),
              hod_clf.clf_satellite_mean(LOG10M, **_for_hod_mod(self.P)))

    @pytest.mark.parametrize("alpha_s", [-1.3, -1.15, -0.5, 0.5, 1.2])
    def test_the_incomplete_gamma_including_its_recurrence(self, alpha_s):
        """`a = (alpha_s+1)/2` is negative for every published fit, which is
        outside `gammaincc`'s domain -- so the recurrence branch is the one
        that actually runs in production."""
        a = (alpha_s + 1.0) / 2.0
        x = jnp.asarray([1e-3, 0.1, 1.0, 5.0, 20.0])
        _same(C.upper_gamma(a, x), hod_clf._upper_gamma(jnp.asarray(a), x),
              f"alpha_s={alpha_s}")

    def test_the_recurrence_branch_is_the_one_being_exercised(self):
        """Or the test above is checking `gammaincc` against itself."""
        assert (C.CLF_DEFAULTS["cacciato09"]["alpha_faint"] + 1.0) / 2.0 < 0.0
        assert (C.CLF_DEFAULTS["vandenbosch13"]["alpha_faint"] + 1.0) / 2.0 < 0.0


class TestAgnKernel:
    """`P(log L_X | M_h)` -- the whole of the ported chain."""

    def test_it_matches_the_reference_implementation(self):
        from ggah_mod.sectors.sham import mstar_girelli20

        class _StubHmf:
            """The kernel never touches the mass function; this keeps the
            comparison from measuring two packages' HMFs against each other."""

            def dndm(self, m, *a, **k):
                return np.ones_like(np.asarray(m))

            def bias(self, m, *a, **k):
                return np.ones_like(np.asarray(m))

        theta = dict(Omega_m=0.31, Omega_b=0.0493, h=0.6736, n_s=0.9649,
                     ln10_10_As=3.044, sum_mnu=0.06, w0=-1.0, wa=0.0)
        ref = hod_powell.PowellAGNModel(
            theta_cosmo=theta, hmf=_StubHmf(), z_mean=0.135,
            log10lx_min=42.0, log10m_min=10.5, log10m_max=15.5, n_m=220,
            n_lx=560, loglx_lo=40.0, loglx_hi=47.0, n_lam=240,
            loglam_min=-3.0, loglam_max=1.5)
        want = np.asarray(ref._p_lx_given_m())

        from ggah_mod.sectors.galaxies import GalaxySector
        # The reference has no stellar-mass cut, so this sector's sits far
        # below every galaxy, where the central fraction is 1 exactly.
        sector = A.AgnSector(GalaxySector("zumandelbaum15"), n_lam=240,
                             loglam_min=-3.0, loglam_max=1.5,
                             n_lx=560, loglx_min=40.0, loglx_max=47.0,
                             lg_mstar_min=0.0)
        # match the reference's SHMR: it uses Girelli on a *physical* mass.
        # Replaced on the instance, because this compares the kernel given a
        # stellar mass and the reference's is not any galaxy sector's.
        # `z=None` is part of the real signature: the sector forwards the
        # field's redshift to the galaxy sector rather than dropping it, and a
        # stub that cannot take it would pass only by being narrower than the
        # thing it stands in for.
        sector.log10_mstar = (
            lambda lm, _gp, h=None, z=None:
            mstar_girelli20(lm - np.log10(0.6736), 0.135))
        p = A.AgnParams(mu_bh=7.76, al_bh=0.67, sig_bh=0.33, sigma_ms=0.20,
                        rho=0.0, log10_lstar=float(np.log10(0.13)),
                        delta1=0.30, delta2=3.70)
        got = np.asarray(sector.p_loglx_given_m(
            jnp.asarray(np.asarray(ref.log10m)), p, None, h=0.6736))

        assert got.shape == want.shape
        _same(got, want, "P(logLx|Mh)", atol_on_peak=1e-8)

    def test_the_eddington_ratio_distribution(self):
        loglam = jnp.asarray(np.linspace(-3.0, 1.5, 40))
        got = A.erdf(loglam, np.log10(0.13), 0.30, 3.70)
        want = hod_powell.erdf_dloglambda(np.asarray(loglam),
                                          np.log10(0.13), 0.30, 3.70)
        _same(got, want, "ERDF")
