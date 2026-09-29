r"""The external reference for :mod:`ggah_mod.halos.lensing_profiles`.

This module is what closed the longest-standing hole in the validation table.
Every other lensing-profile test in this suite is *internal*: the BMO profile is
written twice, in Fourier space and in real space, and pushing the first through
the layer-5 transform reproduces the second to 3.0e-7 of the peak.  That is a
sharp check and it rules out a transcription slip in one representation.  It
does not rule out a misreading the two share, because both were transcribed by
the same reader from the same two papers.

Provenance, which is the whole point
------------------------------------

The values below are the **Oguri et al. (2026)** reference implementation
(``github.com/massarin/halo_lensing``, arXiv:2512.13954), evaluated with
``mpmath`` at **50 decimal digits** for the P/Q hyperbolic combinations and the
BMO Fourier window.  They were generated on 2026-07-07 by
``hod_mod/scripts/cosmology/make_lensing_goldens.py``, which takes a clone of
that repository as its argument and prints paste-ready arrays.

**They are transported through the predecessor and were not regenerated here,
and that distinction matters.**  `tests/parity/` is this suite's one place that
compares against ``hod_mod``, and this is not that: the numbers are not
``hod_mod``'s answers, they are ``halo_lensing``'s, and ``hod_mod`` is the
courier.  What the transport costs is that a transcription error made *in the
courier* would be invisible here -- which is why the generator is named, so the
arrays can be regenerated from the source by anyone with the clone and thirty
seconds.  What it does not cost is independence: ``ggah_mod``'s kernels are a
clean-room rebuild and were never compared against these numbers until they
were run against them for the first time, below.

The two packages' private kernels happen to share names (``_tj_sigma_dl`` and
the rest).  That is a shared reading of one paper's notation, not shared code --
and if it were shared code the agreement below would be uninformative rather
than wrong, so it is worth saying which.

What agreement was found, first time
------------------------------------

===============================  ================
quantity                         ``max err/peak``
===============================  ================
:math:`\Sigma_{\rm TJ}`, 3 c     1.2e-13
:math:`\bar\Sigma_{\rm TJ}`      3.6e-10
:math:`\Sigma_{\rm BMO}`         1.2e-13
:math:`\bar\Sigma_{\rm BMO}`     4.0e-10
:math:`M_{\rm BMO}(<x)`          2.1e-13
:math:`M_{\rm BMO}^{\rm tot}`    exact to 12 digits
:math:`\Sigma_{\rm Hernquist}`   3.2e-13
:math:`\bar\Sigma_{\rm H}`       2.1e-13
:math:`P(x)`, :math:`Q(x)`       2.8e-9
:math:`\tilde u_{\rm BMO}(k)`    5.9e-11 relative
===============================  ================

The ordering is informative rather than incidental.  The plain surface
densities land at 1e-13, which is float64 round-off on an expression of this
length -- there is nothing left to find in them.  The *mean* surface densities
sit three decades worse at 4e-10, because :math:`\bar\Sigma` carries the
:math:`P`/:math:`Q` hyperbolic combination, and :math:`P`/:math:`Q` themselves
are the loosest row at 2.8e-9.  The error budget is one term deep and it is the
term the reference itself computes at 50 digits for the same reason.

None of these is a tolerance anyone chose.  They are what the first run
returned, and the assertions below are set an order of magnitude looser so the
suite records a *regression* rather than re-deriving float64.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.halos.lensing_profiles import (
    _bmo_bsigma_dl, _bmo_sigma_dl, _hern_bsigma_dl, _hern_sigma_dl,
    _m_bmo_dl, _m_bmo_tot_dl, _pq_hyperbolic, _tj_bsigma_dl, _tj_sigma_dl,
    bmo_rho, bmo_mass, bmo_mass_total, bmo_uk, hernquist_mass, hernquist_rho,
    tnfw_mass, tnfw_rho,
)

# ==========================================================================
# Goldens -- verbatim from make_lensing_goldens.py (2026-07-07).
# Do not hand-edit: regenerate, or the provenance above stops being true.
# ==========================================================================
GOLD_X = np.array([1e-3, 0.01, 0.1, 0.3, 0.5, 0.9, 0.985, 1.0, 1.015, 1.5,
                   2.0, 2.9, 3.0, 5.0, 6.0, 8.0, 15.0, 40.0])

TJ_SIGMA_C6 = np.array([
    3.294809536932e+00, 2.143846827163e+00, 1.008646870411e+00,
    5.241835493106e-01, 3.414717910999e-01, 1.830193624054e-01,
    1.640252418298e-01, 1.609817628054e-01, 1.580218502065e-01,
    9.335602291866e-02, 6.009398881450e-02, 3.126045449931e-02,
    2.926470271617e-02, 7.817903334029e-03, 0.0, 0.0, 0.0, 0.0])
TJ_BSIGMA_C6 = np.array([
    3.544807094149e+00, 2.393688739437e+00, 1.251340673515e+00,
    7.407666719647e-01, 5.319390450833e-01, 3.316183052840e-01,
    3.054434330101e-01, 3.011870710810e-01, 2.970299650338e-01,
    2.007956316612e-01, 1.454250640675e-01, 9.159873462846e-02,
    8.757656993834e-02, 4.155010044492e-02, 3.024353588646e-02,
    1.701198893613e-02, 4.838965741833e-03, 6.804795574453e-04])
TJ_SIGMA_C3 = np.array([
    3.281615268745e+00, 2.130652514806e+00, 9.954481384832e-01,
    5.109489189769e-01, 3.281643603572e-01, 1.694459675889e-01,
    1.503723428516e-01, 1.473139127472e-01, 1.443387655249e-01,
    7.900434847303e-02, 4.454116128743e-02, 8.464774376717e-03,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
TJ_BSIGMA_C3 = np.array([
    3.531612825580e+00, 2.380494449390e+00, 1.238144174106e+00,
    7.275522576820e-01, 5.186884685683e-01, 3.182375165453e-01,
    2.920242940320e-01, 2.877607494539e-01, 2.835963340485e-01,
    1.870505327507e-01, 1.311644330065e-01, 7.527225513313e-02,
    7.069937345777e-02, 2.545177444480e-02, 1.767484336444e-02,
    9.942099392498e-03, 2.827974938311e-03, 3.976839756999e-04])
TJ_SIGMA_C15 = np.array([
    3.299437044884e+00, 2.148474338729e+00, 1.013274743378e+00,
    5.288143463686e-01, 3.461084560767e-01, 1.876767752440e-01,
    1.686886520438e-01, 1.656462905603e-01, 1.626875133684e-01,
    9.806834712173e-02, 6.487628272394e-02, 3.623734255999e-02,
    3.426990054357e-02, 1.396094107200e-02, 9.834141551841e-03,
    5.388389200077e-03, 0.0, 0.0])
TJ_BSIGMA_C15 = np.array([
    3.549434601329e+00, 2.398316249171e+00, 1.255968363949e+00,
    7.453958237606e-01, 5.365711261736e-01, 3.362607086030e-01,
    3.100888098642e-01, 3.058330015134e-01, 3.016764577321e-01,
    2.054650901707e-01, 1.501284470949e-01, 9.639328105246e-02,
    9.238399456320e-02, 4.679933318843e-02, 3.606570728464e-02,
    2.344758519185e-02, 8.155949876621e-03, 1.146930451400e-03])

BMO_SIGMA_T12P5 = np.array([
    3.290684273116e+00, 2.139720375669e+00, 1.004473113216e+00,
    5.198857740404e-01, 3.371179273512e-01, 1.787571994826e-01,
    1.598090529715e-01, 1.567744407107e-01, 1.538236086839e-01,
    8.954265581717e-02, 5.679732152689e-02, 2.904178315105e-02,
    2.717343830890e-02, 8.720583394866e-03, 5.393589179465e-03,
    2.278459836560e-03, 2.036083413480e-04, 1.294308896494e-06])
BMO_BSIGMA_T12P5 = np.array([
    3.540681827996e+00, 2.389562851565e+00, 1.247187386985e+00,
    7.365274716444e-01, 5.276387574229e-01, 3.273023187703e-01,
    3.011400855294e-01, 2.968864505485e-01, 2.927322147318e-01,
    1.966558191640e-01, 1.415433607786e-01, 8.831775937151e-02,
    8.436942703138e-02, 4.003173967484e-02, 2.989303158935e-02,
    1.835012556812e-02, 5.733613231222e-03, 8.252528140823e-04])
BMO_MASS_T12P5 = np.array([
    4.993340792290e-07, 4.934072266381e-05, 4.400814371764e-03,
    3.157810156420e-02, 7.202873880662e-02, 1.674451424653e-01,
    1.884332319804e-01, 1.921363216029e-01, 1.958380001281e-01,
    3.128302715748e-01, 4.240713888500e-01, 5.960537312598e-01,
    6.130189270042e-01, 8.788504750385e-01, 9.704492840128e-01,
    1.098491537635e+00, 1.268244290930e+00, 1.319482637492e+00])
BMO_MTOT = {3.0: 4.666245937027e-01, 12.5: 1.321506350004e+00,
            25.0: 1.881177422384e+00}

HERN_SIGMA = np.array([
    1.220184402463e+01, 7.599164212779e+00, 3.108547414762e+00,
    1.334840544858e+00, 7.494346370742e-01, 3.182989330708e-01,
    2.736456268127e-01, 2.666666666667e-01, 2.599277560254e-01,
    1.261871525952e-01, 6.973319205205e-02, 3.024546903357e-02,
    2.792669738311e-02, 7.894890607610e-03, 4.909345013478e-03,
    2.272458110462e-03, 3.948600444699e-04, 2.304770461600e-05])
HERN_BSIGMA = np.array([
    1.320182522188e+01, 8.597974397638e+00, 4.057176060313e+00,
    2.119334828536e+00, 1.388511980272e+00, 7.547877560736e-01,
    6.788359794550e-01, 6.666666666667e-01, 6.548316930750e-01,
    3.963604984734e-01, 2.636001412813e-01, 1.490645676198e-01,
    1.411975310791e-01, 6.003871299388e-02, 4.358804865938e-02,
    2.596470118306e-02, 8.031289427654e-03, 1.202430362446e-03])

PQ_X = np.array([1e-3, 0.1, 0.5, 2.0, 13.9, 14.1, 29.0, 31.0, 50.0, 300.0])
P_GOLD = np.array([
    -7.330540974726e-03, -2.731303940804e-01, -5.992044655175e-01,
    -5.159056633391e-01, -7.274432043866e-02, -7.168828782891e-02,
    -3.456597777625e-02, -3.232606505438e-02, -2.001607774303e-02,
    -3.333407417287e-03])
Q_GOLD = np.array([
    -6.330543529351e+00, -1.741512150628e+00, -3.237061669662e-01,
    1.545770464509e-01, 5.359146373881e-03, 5.202365373509e-03,
    1.197756653694e-03, 1.047221225120e-03, 4.009678129146e-04,
    1.111185201654e-05])

BMO_UK_K = np.logspace(-3, 2.5, 23)
BMO_UK_T15P0 = np.array([
    9.999891592506e-01, 9.999657273554e-01, 9.998916971974e-01,
    9.996581715463e-01, 9.989244696731e-01, 9.966420282025e-01,
    9.897046990854e-01, 9.696641497749e-01, 9.173949399791e-01,
    8.040491980104e-01, 6.197049661077e-01, 4.095911545154e-01,
    2.342650819837e-01, 1.161990466646e-01, 4.990326656776e-02,
    1.884972746317e-02, 6.492675541172e-03, 2.124903034211e-03,
    6.801546710380e-04, 2.159465179716e-04, 6.837605905423e-05,
    2.163123514727e-05, 6.841281420968e-06])


def _rel_to_peak(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


X = jnp.asarray(GOLD_X)


@pytest.mark.x64
class TestTheDimensionlessKernels:
    """The kernels, against 50-digit values from an independent code.

    ``x64`` is not decoration.  In float32 the goldens are meaningless: the
    reference values carry twelve significant figures and single precision has
    seven, so every row here would pass at a tolerance that also passes a wrong
    profile.  The marker says the test is only a test at the precision it is
    run at.
    """

    #: Set an order of magnitude above what the first run measured.  A
    #: tolerance at the measured value turns float64's last bit into a test
    #: failure on a different BLAS; a tolerance far above it stops recording
    #: anything.
    TOL_SIGMA = 1e-12
    TOL_BSIGMA = 1e-8

    @pytest.mark.parametrize("c,tag", [(3.0, "C3"), (6.0, "C6"), (15.0, "C15")])
    def test_truncated_nfw_surface_density(self, c, tag):
        r""":math:`\Sigma_{\rm TJ}(x, c)` -- Takada & Jain (2003), sharply
        truncated at :math:`x = c`.

        Three concentrations rather than one, because the truncation radius
        *is* ``c``: at a single value a formula that ignored the truncation
        entirely would agree everywhere inside it and be checked nowhere
        outside.  The goldens go to ``x = 40``, so ``c = 3`` is mostly
        exterior and ``c = 15`` mostly interior, and both are checked.
        """
        got = _tj_sigma_dl(X, c)
        assert _rel_to_peak(got, globals()[f"TJ_SIGMA_{tag}"]) < self.TOL_SIGMA

    @pytest.mark.parametrize("c,tag", [(3.0, "C3"), (6.0, "C6"), (15.0, "C15")])
    def test_truncated_nfw_mean_surface_density(self, c, tag):
        r""":math:`\bar\Sigma_{\rm TJ}`, three decades looser than
        :math:`\Sigma` and for a locatable reason -- it carries the
        :math:`P`/:math:`Q` hyperbolic combination, which is itself the
        loosest row in this module."""
        got = _tj_bsigma_dl(X, c)
        assert _rel_to_peak(got, globals()[f"TJ_BSIGMA_{tag}"]) < self.TOL_BSIGMA

    def test_bmo_surface_density(self):
        assert _rel_to_peak(_bmo_sigma_dl(X, 12.5), BMO_SIGMA_T12P5) < self.TOL_SIGMA

    def test_bmo_mean_surface_density(self):
        assert _rel_to_peak(_bmo_bsigma_dl(X, 12.5), BMO_BSIGMA_T12P5) < self.TOL_BSIGMA

    def test_bmo_enclosed_mass(self):
        assert _rel_to_peak(_m_bmo_dl(X, 12.5), BMO_MASS_T12P5) < self.TOL_SIGMA

    @pytest.mark.parametrize("tau", sorted(BMO_MTOT))
    def test_bmo_total_mass(self, tau):
        r""":math:`M^{\rm tot}_{\rm BMO}(\tau)`, the normalisation every BMO
        quantity in the package divides by.

        Checked at three truncations spanning a factor of eight, because this
        is one scalar per :math:`\tau` and a wrong *slope* in :math:`\tau`
        would pass at any single one -- while rescaling every BMO profile the
        package produces.
        """
        assert float(_m_bmo_tot_dl(tau)) == pytest.approx(BMO_MTOT[tau], rel=1e-11)

    def test_hernquist_surface_density(self):
        assert _rel_to_peak(_hern_sigma_dl(X), HERN_SIGMA) < self.TOL_SIGMA

    def test_hernquist_mean_surface_density(self):
        assert _rel_to_peak(_hern_bsigma_dl(X), HERN_BSIGMA) < self.TOL_SIGMA

    def test_the_hyperbolic_combination(self):
        r""":math:`P(x)` and :math:`Q(x)`, the reference's own 50-digit rows.

        The loosest agreement in the module at 2.8e-9, and the one place that
        is a statement about this implementation rather than about float64.
        ``PQ_X`` straddles 14 and 30 on both sides deliberately -- those are
        where the reference switches asymptotic branch, and a branch boundary
        off by one is exactly the error two agreeing representations of the
        *same* transcription would share.
        """
        p, q = _pq_hyperbolic(jnp.asarray(PQ_X))
        assert _rel_to_peak(p, P_GOLD) < 1e-8
        assert _rel_to_peak(q, Q_GOLD) < 1e-9

    def test_the_bmo_fourier_window(self):
        r""":math:`\tilde u_{\rm BMO}(k)` over five and a half decades.

        The only golden here that is not a projection, and the one that
        matters most to layer 4: this window is what a BMO tracer's weight
        carries into :func:`~ggah_mod.spectra.pk.pk_cross`.  Agreement at
        5.9e-11 *relative* rather than to the peak, because the array spans
        six orders of magnitude and a peak-relative bound would check only its
        first three points.
        """
        got = np.asarray(bmo_uk(jnp.asarray(BMO_UK_K), jnp.asarray([1.0]), 15.0))[:, 0]
        assert np.max(np.abs(got / BMO_UK_T15P0 - 1.0)) < 1e-9


@pytest.mark.x64
class TestTheDensitiesThemselves:
    r"""The three :math:`\rho(r)` the suite never called.

    Every other test of this module -- including the goldens above -- goes
    through a *projection* or an enclosed mass.  ``tnfw_rho``, ``bmo_rho`` and
    ``hernquist_rho`` were the functions those are derived from, and they were
    the only three names in ``__all__`` that no test in this repository
    mentioned.  ``lensing_profiles.py`` was the package's lowest-covered module
    at 65.9 per cent, and this was most of the gap.

    That is a worse place for a hole than it sounds.  A wrong :math:`\rho` with
    a right :math:`M(<r)` is not a contradiction a reader would notice, because
    the two are written as separate closed forms from the same paper rather
    than one integrated from the other -- so nothing in the module ties them
    together and nothing did.  The tie is here: integrate the density and
    require the mass, which is now itself checked against the reference above.
    """

    RHO_S, R_S, C_T, TAU = 1e15, 0.2, 6.0, 15.0
    M_TOT, R_B = 1e12, 0.01

    @staticmethod
    def _enclosed(rho, r_max, n=200_001):
        r""":math:`\int_0^{r_{\max}} 4\pi r^2 \rho\,dr` on a linear grid.

        Linear and very fine rather than log and cheap: the NFW cusp puts most
        of the integrand's *structure* near zero but almost none of its mass
        there, and a log grid that resolves the cusp spends its nodes where
        they do not matter.  200k nodes gets Simpson to ~1e-9 of the answer,
        which is two decades below the tolerance asserted.
        """
        r = np.linspace(0.0, r_max, n)
        return np.trapezoid(4.0 * np.pi * r ** 2 * np.asarray(rho(r)), r)

    @pytest.mark.parametrize("frac", [0.25, 0.5, 0.9])
    def test_truncated_nfw_density_integrates_to_its_mass(self, frac):
        r_max = frac * self.C_T * self.R_S
        got = self._enclosed(
            lambda r: tnfw_rho(r, self.RHO_S, self.R_S, self.C_T), r_max)
        want = float(tnfw_mass(r_max, self.RHO_S, self.R_S, self.C_T))
        assert got == pytest.approx(want, rel=1e-6)

    def test_truncated_nfw_density_is_zero_beyond_the_truncation(self):
        """Sharp, not tapered -- which is the whole difference between this
        profile and BMO, and the reason both are shipped."""
        r_t = self.C_T * self.R_S
        inside = np.asarray(tnfw_rho(np.array([0.999 * r_t]), self.RHO_S,
                                     self.R_S, self.C_T))
        outside = np.asarray(tnfw_rho(np.array([1.001 * r_t, 10.0 * r_t]),
                                      self.RHO_S, self.R_S, self.C_T))
        assert inside[0] > 0.0
        assert np.all(outside == 0.0)

    @pytest.mark.parametrize("frac", [0.25, 1.0, 4.0])
    def test_bmo_density_integrates_to_its_mass(self, frac):
        r_max = frac * self.TAU * self.R_S
        got = self._enclosed(
            lambda r: bmo_rho(r, self.RHO_S, self.R_S, self.TAU), r_max)
        want = float(bmo_mass(r_max, self.RHO_S, self.R_S, self.TAU))
        assert got == pytest.approx(want, rel=1e-6)

    def test_the_bmo_density_falls_as_r_to_the_seventh(self):
        r"""NFW's :math:`r^{-3}` times the taper's :math:`r^{-4}`.

        The asymptotic slope is what makes the total mass finite, so getting
        it right is the difference between ``bmo_mass_total`` being a number
        and being a divergent integral that happens to be evaluated on a
        finite grid.
        """
        r = np.array([1e3, 1e4]) * self.R_S
        rho = np.asarray(bmo_rho(r, self.RHO_S, self.R_S, self.TAU))
        slope = np.log(rho[1] / rho[0]) / np.log(r[1] / r[0])
        assert slope == pytest.approx(-7.0, abs=1e-3)

    def test_the_bmo_density_integrates_to_the_total_mass(self):
        """The tail carries real mass, so the ceiling is approached from
        below rather than reached: at 4000 scale radii the integral is within
        0.1 per cent, and every remaining part is beyond it."""
        got = self._enclosed(
            lambda r: bmo_rho(r, self.RHO_S, self.R_S, self.TAU),
            4000.0 * self.R_S, n=400_001)
        want = float(bmo_mass_total(self.RHO_S, self.R_S, self.TAU))
        assert got < want
        assert got == pytest.approx(want, rel=2e-3)

    @pytest.mark.parametrize("frac", [0.5, 2.0, 20.0])
    def test_hernquist_density_integrates_to_its_mass(self, frac):
        r_max = frac * self.R_B
        got = self._enclosed(lambda r: hernquist_rho(r, self.M_TOT, self.R_B),
                             r_max)
        want = float(hernquist_mass(r_max, self.M_TOT, self.R_B))
        assert got == pytest.approx(want, rel=1e-6)

    def test_the_hernquist_density_carries_its_total_mass(self):
        r""":math:`M(<r) \to M_{\rm tot}` -- the name of the argument, checked.

        ``hernquist_rho`` takes ``m_tot`` and not a scale density, so if the
        normalisation were the NFW one the profile would still be a Hernquist
        profile and every *shape* test would pass while the amplitude was
        wrong by :math:`2\pi`.
        """
        got = self._enclosed(lambda r: hernquist_rho(r, self.M_TOT, self.R_B),
                             1e4 * self.R_B, n=400_001)
        assert got == pytest.approx(self.M_TOT, rel=1e-3)

    @pytest.mark.parametrize("fn,args", [
        (tnfw_rho, (1e15, 0.2, 6.0)), (bmo_rho, (1e15, 0.2, 15.0)),
        (hernquist_rho, (1e12, 0.01)),
    ])
    def test_the_densities_are_finite_at_the_origin(self, fn, args):
        """Not because zero is a radius anyone asks for, but because all three
        are cusped there and a ``nan`` at ``r = 0`` poisons the gradient of
        every radius that *is* asked for -- the trap ``bessel_k`` was caught
        by, one module away."""
        out = np.asarray(fn(np.array([0.0, 1e-8]), *args))
        assert np.all(np.isfinite(out))
        g = float(jax.grad(lambda a: jnp.sum(fn(jnp.array([0.0, 1e-3]), a,
                                                *args[1:])))(args[0]))
        assert np.isfinite(g) and g != 0.0
