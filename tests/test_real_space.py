r"""Verification: the real-space projections, and the profile they finally pin.

The headline is closed-form and needs no reference data.  The BMO (Baltz,
Marshall & Oguri) truncated profile is written **twice** in this package -- as a
Fourier transform in :func:`~ggah_mod.halos.lensing_profiles.bmo_uk` and as
:math:`\Sigma`, :math:`\Delta\Sigma` in real space -- from two different papers'
algebra.  Feeding the first through
:func:`~ggah_mod.observables.transforms.pk_to_delta_sigma` must reproduce the
second, and it does, to :math:`10^{-4}` or better through **both** engines.

That matters beyond this file.  ``PLAN.md`` records ``lensing_profiles`` as
*"present but not validated ... covered by smoke tests that check it runs and
satisfies its own analytic limits.  Those catch a transcription error; they
would not catch a wrong coefficient.  Validation against the goldens waits until
layer 5 consumes it."*  Layer 5 has now consumed it, and a wrong coefficient in
either representation would show here as a ratio that is not one.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from ggah_mod.backend import ACCURATE, DIFFERENTIABLE, DIFFERENTIABLE_COARSE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.halos.lensing_profiles import (
    bmo_delta_sigma, bmo_mass_total, bmo_sigma, bmo_uk,
)
from ggah_mod.halos.profiles import nfw_scale_density
from ggah_mod.observables import real_space as RS
from ggah_mod.observables.transforms import pk_to_delta_sigma, pk_to_sigma

#: A cluster-scale halo, and a truncation at 3 R_delta.
M, C, R_DELTA, TAU = 1.0e14, 5.0, 1.0, 3.0
#: The Gaussian pair's width [Mpc/h], for the `w_p` closed form.
A = 5.0


@pytest.fixture(scope="module")
def bmo():
    rho_s, r_s = nfw_scale_density(M, C, R_DELTA)
    rho_s, r_s = float(rho_s), float(r_s)
    k = jnp.logspace(-4, 4, 4096)
    pk = float(bmo_mass_total(rho_s, r_s, TAU)) * bmo_uk(k, jnp.asarray([r_s]),
                                                          TAU)[:, 0]
    return rho_s, r_s, k, pk


@pytest.fixture(scope="module")
def bmo_pk(bmo):
    """The same profile as a genuine ``P_gm(k)`` in (Mpc/h)^3.

    ``bmo`` carries ``rho_m * P``; the projections take ``P`` and supply the
    density themselves, so dividing here is not bookkeeping -- it is the
    difference between a spectrum and a surface density.
    """
    _, _, k, pk = bmo
    return k, pk / PLANCK18.rho_matter


@pytest.fixture(scope="module")
def gaussian():
    k = jnp.logspace(-5, 4, 4096)
    return k, (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)


def _rel(got, want):
    return float(np.max(np.abs(np.asarray(got) / np.asarray(want) - 1.0)))


def _rel_to_peak(got, want):
    """Max absolute difference, relative to the **peak** of ``want``.

    The metric ``tests/parity/test_clf_and_agn.py`` already uses, and the right
    one wherever a profile falls by orders of magnitude across the range: a
    per-point relative error at the far tail measures the tail's own smallness,
    not the agreement.
    """
    want = np.asarray(want)
    return float(np.max(np.abs(np.asarray(got) - want)) / np.max(np.abs(want)))


class TestTheProfileIsPinnedInBothSpaces:
    r"""One profile, two independent derivations, two independent engines."""

    R = jnp.asarray([0.05, 0.1, 0.2, 0.5])

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_delta_sigma_reproduces_the_closed_form(self, flavour, bmo):
        rho_s, r_s, k, pk = bmo
        got = pk_to_delta_sigma(self.R, k, pk, 1.0, backend=flavour)
        want = bmo_delta_sigma(self.R, rho_s, r_s, TAU)
        assert _rel_to_peak(got, want) < 3e-4

    def test_the_ogata_deficit_was_the_reach_and_is_gone(self, bmo):
        r"""**A finding that closed.**

        This test used to pin an unexplained defect: the Ogata rule was
        uniformly *low* against the closed form by an amount that grew with
        radius and plateaued -- :math:`6.3, 12.7, 14.0, 14.0 \times 10^{9}` at
        :math:`R = 0.05, 0.1, 0.2, 0.5` Mpc/h, 1.2e-4 of the peak -- and its
        docstring recorded that it was *"not the k range, not the node count,
        and not the profile's own consistency"*.

        It was the node count, and the reason the check missed it is instructive.
        ``n_hankel`` was varied at 256, 512 and 1024 with ``h = 0.001``, which is
        ``h*N`` = 0.26, 0.51 and 1.02 -- every one of them below the ~2.5 where
        :attr:`~ggah_mod.backend.Backend.hankel_h` says the rule saturates.  The
        knob was moved inside the broken regime and concluded not to matter.

        Measured across the threshold, same profile and same ``k`` grid:

        ========  ======  ===============  ==============
        ``N``     ``hN``  deficit at each R  ``|err|/peak``
        ========  ======  ===============  ==============
        512       0.51    all positive     7.5e-5
        1024      1.02    all positive     1.1e-5
        2048      2.05    mixed sign       3.4e-7
        4096      4.10    mixed sign       3.0e-7
        ========  ======  ===============  ==============

        ``ACCURATE`` ships 4096 now, and the deficit is gone: not merely smaller
        but no longer one-signed, which is what says a truncation was removed
        rather than a tolerance loosened.

        The engines' error *shapes* still differ, which was the other half of
        the original claim and remains true -- but not symmetrically any more.
        Ogata is now uniformly good; FFTLog's error is concentrated at the small
        end of its reciprocal grid (2.6e10 at :math:`R = 0.05` against 1e5 at
        0.5), which is the open row of the parity budget, PLAN.md **B4**.
        """
        rho_s, r_s, k, pk = bmo
        want = np.asarray(bmo_delta_sigma(self.R, rho_s, r_s, TAU))
        ogata = np.asarray(pk_to_delta_sigma(self.R, k, pk, 1.0,
                                             backend=ACCURATE))
        fft = np.asarray(pk_to_delta_sigma(self.R, k, pk, 1.0,
                                           backend=DIFFERENTIABLE))

        assert _rel_to_peak(ogata, want) < 1e-6         # was 3e-4
        assert _rel_to_peak(fft, want) < 1e-6           # was 3e-4, then padded

        # The deficit is no longer one-signed: a truncation was removed.
        deficit = want - ogata
        assert np.any(deficit > 0.0) and np.any(deficit < 0.0)

        # The engines are now within a small factor of each other against a
        # closed form, where Ogata used to be a hundredfold better.  That is
        # `Backend.fftlog_pad_decades`, and this profile is an independent
        # check on it: `test_parity_budget` compares the engines to each other,
        # and this compares each to an analytic Delta Sigma.
        assert _rel_to_peak(fft, want) < 5.0 * _rel_to_peak(ogata, want)

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_sigma_reproduces_the_closed_form(self, flavour, bmo):
        rho_s, r_s, k, pk = bmo
        got = pk_to_sigma(self.R, k, pk, 1.0, backend=flavour)
        assert _rel_to_peak(got, bmo_sigma(self.R, rho_s, r_s, TAU)) < 1e-3

    def test_it_is_sharp_enough_to_catch_a_wrong_coefficient(self, bmo):
        """A smoke test that checks limits would pass a profile scaled by any
        constant.  This does not: a 1% error in either representation shows."""
        rho_s, r_s, k, pk = bmo
        got = pk_to_delta_sigma(self.R, k, pk, 1.0, backend=ACCURATE)
        want = bmo_delta_sigma(self.R, rho_s, r_s, TAU)
        assert _rel_to_peak(got, want) < 3e-4
        assert _rel_to_peak(got, 1.01 * want) > 5e-3


class TestWpNeedsItsAperture:
    r""":math:`\pi_{\max}` is a survey property and has no default."""

    def test_it_refuses_without_one(self, gaussian):
        with pytest.raises(TypeError):
            RS.wp(jnp.asarray([1.0]), gaussian)

    def test_an_explicit_none_is_refused_with_the_reason(self, gaussian):
        with pytest.raises(ValueError, match="property of the measurement"):
            RS.wp(jnp.asarray([1.0]), gaussian, pi_max=None)

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    @pytest.mark.parametrize("pi_max", [40.0, 100.0])
    def test_against_the_closed_form(self, pi_max, flavour, gaussian):
        r"""For a Gaussian :math:`\xi`,

        .. math::

            w_p = \sqrt{2\pi}\,a\,e^{-r_p^2/2a^2}\,
                  \mathrm{erf}\!\big(\pi_{\max}/a\sqrt2\big)

        -- the infinite-aperture answer times an error function, so the test is
        sensitive to the truncation and not only to :math:`\xi`.

        **Parametrised over both flavours, which it was not.**
        :math:`\Sigma` and :math:`\Delta\Sigma` above are checked on both
        engines; ``w_p`` was pinned to ``ACCURATE``, so the FFTLog route -- the
        default on the only differentiable flavour -- had never once been
        compared against its analytic answer.  That gap is why a refusal on
        that route could sit in the shipped package unnoticed.
        """
        rp = jnp.asarray([1.0, 2.0, 5.0])
        got = RS.wp(rp, gaussian, pi_max=pi_max, backend=flavour)
        want = (jnp.sqrt(2 * jnp.pi) * A * jnp.exp(-0.5 * (rp / A) ** 2)
                * erf(pi_max / (A * jnp.sqrt(2.0))))
        assert _rel(got, want) < 1e-4

    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: f"{b.name}:{b.hankel}")
    def test_a_larger_aperture_gives_more_signal(self, flavour, gaussian):
        rp = jnp.asarray([1.0, 3.0])
        small = RS.wp(rp, gaussian, pi_max=10.0, backend=flavour)
        large = RS.wp(rp, gaussian, pi_max=100.0, backend=flavour)
        assert np.all(np.asarray(large) > np.asarray(small))


class TestWpLineOfSight:
    r"""The line-of-sight integral below :math:`r_p \approx 0.2` Mpc/h (0.9.7).

    Until 0.9.7 ``wp`` integrated on a grid linear in :math:`\pi`, ``n_pi``
    nodes from 0 to :math:`\pi_{\max}`: 0.196 Mpc/h apart at
    :math:`\pi_{\max} = 100` on ``ACCURATE``.  The integrand peaks at
    :math:`\pi \lesssim r_p`, so where :math:`r_p` is smaller than a node
    spacing the trapezoid saw the peak at one node and extrapolated it across a
    whole interval.  The closed-form test above runs at :math:`r_p` = 1-5
    Mpc/h, where that never happens, and a galaxy sample's ``w_p`` was high by
    1.9-3.9x at 0.011 Mpc/h with nothing failing (``ggah_sens_study``, Tier 1).

    Both tests below reach 0.005 Mpc/h, below the smallest separation any data
    set asks for (LS10: 0.0078 Mpc/h).
    """

    #: The bounds measured when the substitution landed, per flavour's
    #: ``(n_pi, n_r_tab)``: the power law 5.1e-6 / 5.3e-6 / 7.4e-5, the narrow
    #: Gaussian 1.8e-6 / 1.4e-6 / 1.8e-4.  COARSE's 128 nodes and 128-point
    #: table are the limit; the other two are the table's cubic interpolation.
    BOUND = {"accurate": (1e-5, 5e-6), "differentiable": (1e-5, 5e-6),
             "differentiable_coarse": (1.5e-4, 5e-4)}
    FLAVOURS = [ACCURATE, DIFFERENTIABLE, DIFFERENTIABLE_COARSE]
    RP = np.array([0.005, 0.0078, 0.011, 0.02, 0.05, 0.1, 1.0, 10.0])

    @staticmethod
    def _power_law(rp, gamma, pi_max, r0=5.0):
        r""":math:`\xi = (r/r_0)^{-\gamma}` to a finite :math:`\pi_{\max}`:
        with :math:`t = r_p^2/(r_p^2+\pi^2)` the integral is an incomplete beta,
        :math:`w_p = r_p(r_0/r_p)^\gamma B(\tfrac{\gamma-1}2,\tfrac12)
        [1 - I_{t_0}(\tfrac{\gamma-1}2,\tfrac12)]`, :math:`t_0 =
        r_p^2/(r_p^2+\pi_{\max}^2)`."""
        from scipy.special import beta, betainc

        a = 0.5 * (gamma - 1.0)
        t0 = rp ** 2 / (rp ** 2 + pi_max ** 2)
        return rp * (r0 / rp) ** gamma * beta(a, 0.5) * (1.0 - betainc(a, 0.5, t0))

    @pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda b: b.name)
    def test_a_steep_power_law_to_a_finite_aperture(self, flavour):
        """Engine-free: the quadrature alone, on the flavour's own table."""
        worst = 0.0
        for gamma in (1.8, 2.4):
            for pi_max in (40.0, 67.4, 100.0):
                r_tab = jnp.logspace(-3.0, 2.5, flavour.n_r_tab)
                xi_tab = (r_tab / 5.0) ** (-gamma)
                got = RS._los_integral(jnp.asarray(self.RP), r_tab, xi_tab, pi_max,
                                       flavour.n_pi)
                worst = max(worst, _rel(got, self._power_law(self.RP, gamma, pi_max)))
        assert worst < self.BOUND[flavour.name][0], (
            f"{flavour.name}: power-law w_p off by {worst:.2e}")

    def test_the_linear_grid_was_the_defect(self):
        """The record: 0.9.4's grid on the same power law, 2.8-15x high below 0.02."""
        pi = np.linspace(0.0, 100.0, ACCURATE.n_pi)
        rp = self.RP[self.RP < 0.02]
        for gamma in (1.8, 2.4):
            old = np.array([2.0 * np.trapezoid((np.sqrt(r ** 2 + pi ** 2) / 5.0) ** (-gamma),
                                               pi) for r in rp])
            assert np.all(old / self._power_law(rp, gamma, 100.0) > 2.5)

    @pytest.mark.parametrize("pi_max", [40.0, 100.0])
    @pytest.mark.parametrize("flavour", FLAVOURS, ids=lambda b: b.name)
    def test_a_narrow_gaussian_through_the_transform(self, flavour, pi_max):
        r"""End to end, Hankel engine included, for a pair width :math:`a` =
        0.05 Mpc/h -- a quarter of the old node spacing -- and :math:`r_p` down
        to 0.005: :math:`w_p = \sqrt{2\pi}\,a\,e^{-r_p^2/2a^2}\,
        {\rm erf}(\pi_{\max}/a\sqrt2)` for a unit-amplitude Gaussian
        :math:`\xi`."""
        a = 0.05
        k = jnp.logspace(-4, np.log10(200.0), 4096)
        pk = (2 * jnp.pi * a * a) ** 1.5 * jnp.exp(-0.5 * (k * a) ** 2)
        rp = jnp.asarray([0.005, 0.0078, 0.01, 0.02, 0.05, 0.1])
        got = RS.wp(rp, (k, pk), pi_max=pi_max, backend=flavour)
        want = (jnp.sqrt(2 * jnp.pi) * a * jnp.exp(-0.5 * (rp / a) ** 2)
                * erf(pi_max / (a * jnp.sqrt(2.0))))
        assert _rel(got, want) < self.BOUND[flavour.name][1]


class TestSigmaHadADeadFlagOnce:
    r"""The defect that started this, kept because the fix moved twice.

    ``sigma``'s branch read ``cosmo.rho_matter if comoving else
    cosmo.rho_matter`` -- the same expression on both sides -- so a caller
    asking for proper units got the comoving number with nothing in it saying
    so.  Line coverage rated that line fully exercised: it ran on every call,
    and both "branches" were taken in the only sense coverage measures.

    It was first fixed by refusing ``comoving=False`` outright, because the
    conversion needs a redshift the function had no way to learn.  It now
    *answers*, because the redshift is passed in -- see
    :class:`TestSigmaInProperUnits`.  What survives here is the part that was
    true at every stage: the default is comoving, and it answers.
    """

    def test_the_default_is_comoving_and_answers(self, bmo_pk):
        got = np.asarray(RS.sigma(jnp.asarray([0.5, 1.0, 2.0]), bmo_pk,
                                  PLANCK18))
        assert got.shape == (3,)
        assert np.all(np.isfinite(got)) and np.all(got > 0.0)

    def test_the_flag_is_not_inert_any_more(self, bmo_pk):
        """The one assertion that would have failed against the original.

        Whatever ``comoving=False`` does now, it must not be "the same thing
        as ``comoving=True``" -- which is precisely what it used to do.
        """
        rp = jnp.asarray([0.5, 1.0, 2.0])
        com = np.asarray(RS.sigma(rp, bmo_pk, PLANCK18))
        pro = np.asarray(RS.sigma(rp, bmo_pk, PLANCK18, comoving=False, z=1.0))
        assert not np.array_equal(com, pro)


class TestSigmaInProperUnits:
    r"""``comoving=False`` answers now, given the redshift it needs.

    ``pk_to_sigma`` integrates a comoving :math:`\bar\rho_m` against a comoving
    :math:`k`, so its answer is a mass per *comoving* area.  The same mass
    occupies a proper area smaller by :math:`(1+z)^2`, which is the whole
    conversion -- and the reason ``z`` had to be plumbed in rather than
    inferred: nothing else in the call carries one.
    """

    RP = jnp.asarray([0.5, 1.0, 2.0])

    def test_proper_is_the_comoving_one_times_one_plus_z_squared(self, bmo_pk):
        for z in (0.25, 1.0, 2.5):
            com = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18))
            pro = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18,
                                      comoving=False, z=z))
            np.testing.assert_allclose(pro / com, (1.0 + z) ** 2, rtol=1e-12)

    def test_at_z_zero_the_two_agree_exactly(self, bmo_pk):
        """Not approximately: ``(1 + 0)^2`` is 1 and multiplying by it is
        exact, so a discrepancy here would mean the conversion is reaching
        something other than the factor it claims to be."""
        com = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18))
        pro = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18,
                                  comoving=False, z=0.0))
        assert np.array_equal(com, pro)

    def test_the_radius_is_not_converted_with_the_area(self, bmo_pk):
        r"""The ambiguity this could have hidden.

        Proper units change what the mass is divided *by*; they do not move
        where it was evaluated.  If the function silently rescaled ``rp`` to a
        proper radius as well, the ratio to the comoving answer would stop
        being a constant -- :math:`\Sigma` is not a power law in :math:`R`, so
        evaluating it at :math:`R/(1+z)` and dividing shows up immediately as
        an :math:`R`-dependent ratio.  A flat ratio is the evidence that only
        the units moved.
        """
        z = 1.5
        com = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18))
        pro = np.asarray(RS.sigma(self.RP, bmo_pk, PLANCK18,
                                  comoving=False, z=z))
        ratio = pro / com
        assert np.ptp(ratio) < 1e-12 * ratio.mean(), ratio

    def test_it_still_refuses_a_missing_or_pointless_z(self, bmo_pk):
        """Both halves of the contract, since each is a different mistake."""
        with pytest.raises(ValueError, match="needs a `z`"):
            RS.sigma(self.RP, bmo_pk, PLANCK18, comoving=False)
        with pytest.raises(ValueError, match="no effect"):
            RS.sigma(self.RP, bmo_pk, PLANCK18, comoving=True, z=0.5)

    @pytest.mark.x64
    def test_the_redshift_carries_a_gradient(self, bmo_pk):
        """`z` enters as a factor, so it must be traceable.

        A forecast that varies a bin's redshift -- or marginalises over its
        mean -- differentiates through exactly this, and a `float(z)` anywhere
        in the conversion would have killed it silently.
        """
        f = lambda zz: jnp.sum(RS.sigma(self.RP, bmo_pk, PLANCK18,
                                        comoving=False, z=zz))
        g = float(jax.grad(f)(0.5))
        assert np.isfinite(g) and g != 0.0
        # d/dz of (1+z)^2 S is 2(1+z) S, and S is z-independent here.
        base = float(jnp.sum(RS.sigma(self.RP, bmo_pk, PLANCK18)))
        assert g == pytest.approx(2.0 * 1.5 * base, rel=1e-10)


class TestDeltaSigmaInProperUnits:
    r"""The same conversion as :class:`TestSigmaInProperUnits`, on the quantity
    that actually gets fitted.

    :math:`\Delta\Sigma` is what a lensing survey reports, so a
    redshift-binned fit is where the proper/comoving choice has to be stated
    rather than defaulted -- and where getting it wrong is a coherent
    :math:`(1+z)^2` per bin, which a free amplitude would happily absorb.
    """

    RP = jnp.asarray([0.5, 1.0, 2.0])

    @pytest.mark.parametrize("route", ["delta_sigma", "delta_sigma_via_abel"])
    @pytest.mark.parametrize("z", [0.25, 1.0, 2.5])
    def test_proper_is_comoving_times_one_plus_z_squared(self, route, z,
                                                         bmo_pk):
        fn = getattr(RS, route)
        com = np.asarray(fn(self.RP, bmo_pk, PLANCK18))
        pro = np.asarray(fn(self.RP, bmo_pk, PLANCK18, comoving=False, z=z))
        np.testing.assert_allclose(pro / com, (1.0 + z) ** 2, rtol=1e-12)

    def test_the_cross_check_is_unaffected_by_the_convention(self, bmo_pk):
        """The reason the Abel route had to gain the flag too.

        These two exist to be compared: one is production, the other the
        independent long way round.  If only one carried the conversion, asking
        either for proper units would make them disagree by 2.25 -- and that
        reads as a defect in a transform rather than in a unit.  Their
        agreement must be the *same* number under both conventions.
        """
        direct_c = np.asarray(RS.delta_sigma(self.RP, bmo_pk, PLANCK18))
        abel_c = np.asarray(RS.delta_sigma_via_abel(self.RP, bmo_pk, PLANCK18))
        kw = dict(comoving=False, z=0.7)
        direct_p = np.asarray(RS.delta_sigma(self.RP, bmo_pk, PLANCK18, **kw))
        abel_p = np.asarray(RS.delta_sigma_via_abel(self.RP, bmo_pk,
                                                    PLANCK18, **kw))
        np.testing.assert_allclose(abel_p / direct_p, abel_c / direct_c,
                                   rtol=1e-12)

    @pytest.mark.parametrize("route", ["sigma", "delta_sigma",
                                       "delta_sigma_via_abel"])
    def test_every_route_refuses_the_same_two_misuses(self, route, bmo_pk):
        """One helper, so one contract -- asserted on all three rather than
        on the one that happened to be written first."""
        fn = getattr(RS, route)
        with pytest.raises(ValueError, match="needs a `z`"):
            fn(self.RP, bmo_pk, PLANCK18, comoving=False)
        with pytest.raises(ValueError, match="no effect"):
            fn(self.RP, bmo_pk, PLANCK18, comoving=True, z=0.5)

    @pytest.mark.parametrize("route", ["sigma", "delta_sigma"])
    def test_the_message_names_the_function_it_came_from(self, route, bmo_pk):
        """The helper is shared; the message must still say who refused.

        A single message reading "sigma()" out of a `delta_sigma` call would
        send the reader to the wrong function -- the sort of small dishonesty
        a shared helper invites.
        """
        with pytest.raises(ValueError, match=route + r"\(comoving=False\)"):
            getattr(RS, route)(self.RP, bmo_pk, PLANCK18, comoving=False)


class TestTheTwoDeltaSigmaRoutes:
    r"""The direct :math:`J_2` transform, against the long way round."""

    R = jnp.asarray([0.2, 0.5, 1.0, 2.0])

    def test_they_agree_to_a_few_per_mille(self, bmo, bmo_pk):
        """Measured, not assumed.  The real-space route carries three error
        sources the transform does not -- the line-of-sight truncation, the
        interpolation onto sqrt(R^2+chi^2), and a cumulative integral started
        below the smallest tabulated radius.  2.2e-3 measured; this bound was
        5% while the chi grid's fixed 1e-3 floor cost 1.3% at R = 0.2."""
        direct = RS.delta_sigma(self.R, bmo_pk, PLANCK18, backend=ACCURATE)
        abel = RS.delta_sigma_via_abel(self.R, bmo_pk, PLANCK18,
                                       backend=ACCURATE)
        assert _rel(abel, direct) < 5e-3

    @pytest.mark.parametrize("backend", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: b.name)
    def test_the_line_of_sight_starts_below_the_smallest_radius(
            self, bmo, bmo_pk, backend):
        r"""Down to the LS10 reach, where a fixed floor shows.

        With the chi grid's log half starting at 1e-3 Mpc/h whatever ``rp``
        was, the missing :math:`\int_0^{10^{-3}}\xi\,d\chi` cost about
        :math:`10^{-3}/R`: 18% at 0.01, 9.7% at 0.02, 4.2% at 0.05.  Tied to
        three decades below the smallest ``rp``, 0.26% at worst on either
        flavour.  The radii of the test above, 0.2-2, cannot see it at the
        few-percent level that bound used to allow.
        """
        rho_s, r_s, _, _ = bmo
        R = jnp.asarray([0.01, 0.02, 0.05])
        want = bmo_delta_sigma(R, rho_s, r_s, TAU) * RS.SIGMA_UNIT
        abel = RS.delta_sigma_via_abel(R, bmo_pk, PLANCK18, backend=backend)
        assert _rel(abel, want) < 5e-3

    def test_the_gap_between_them_is_why_one_is_production(self, bmo, bmo_pk):
        """The direct route is 2e-8 of the peak from the closed form on these
        radii (1.2e-4 before the Ogata reach was fixed) and the long way round
        is about a percent -- orders of magnitude apart, and asserted here with
        a wide margin: under 3e-4, and at least twentyfold.  A preference
        stated as a number."""
        rho_s, r_s, _, _ = bmo
        want = bmo_delta_sigma(self.R, rho_s, r_s, TAU) * RS.SIGMA_UNIT
        direct = RS.delta_sigma(self.R, bmo_pk, PLANCK18, backend=ACCURATE)
        abel = RS.delta_sigma_via_abel(self.R, bmo_pk, PLANCK18,
                                       backend=ACCURATE)
        assert _rel_to_peak(direct, want) < 3e-4
        assert _rel_to_peak(abel, want) > 20.0 * _rel_to_peak(direct, want)

    def test_the_direct_route_is_the_production_one(self):
        """One production path; the other is a cross-check, and its name says
        so."""
        assert RS.delta_sigma.__name__ == "delta_sigma"
        assert "cross-check" in RS.delta_sigma_via_abel.__doc__


class TestUnits:
    def test_sigma_is_in_msun_per_parsec_squared(self, bmo_pk):
        r""":math:`(M_\odot/h)(\mathrm{pc}/h)^{-2}`, the unit lensing is quoted
        in.  The predecessor wrote the ``1e-12`` inline at three call sites."""
        assert RS.SIGMA_UNIT == 1e-12
        got = RS.delta_sigma(jnp.asarray([0.5]), bmo_pk, PLANCK18,
                             backend=ACCURATE)
        # A 1e14 Msun/h halo at 0.5 Mpc/h: tens of Msun/pc^2.
        assert 1.0 < float(got[0]) < 1000.0

    def test_it_accepts_a_power_spectrum_object(self, bmo_pk):
        """`PowerSpectrum` or a bare `(k, P)` pair, so layer 4's output goes
        straight in."""
        from ggah_mod.spectra.pk import PowerSpectrum
        k, pk = bmo_pk
        obj = PowerSpectrum(k=k, one_halo=pk, two_halo=jnp.zeros_like(pk),
                            shot=jnp.asarray(0.0))
        a = RS.delta_sigma(jnp.asarray([0.5]), obj, PLANCK18, backend=ACCURATE)
        b = RS.delta_sigma(jnp.asarray([0.5]), (k, pk), PLANCK18,
                           backend=ACCURATE)
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-14)


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("flavour", [ACCURATE, DIFFERENTIABLE],
                             ids=lambda b: b.hankel)
    def test_the_gradient_flows_through_wp(self, flavour, gaussian):
        k, _ = gaussian
        rp = jnp.asarray([1.0, 3.0])

        def f(amp):
            pk = amp * (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)
            return jnp.sum(RS.wp(rp, (k, pk), pi_max=60.0, backend=flavour))

        h = 1e-6
        ad = float(jax.grad(f)(1.0))
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-5)

    @pytest.mark.x64
    def test_the_gradient_flows_through_delta_sigma(self, bmo_pk):
        k, pk = bmo_pk
        R = jnp.asarray([0.2, 1.0])

        def f(amp):
            return jnp.sum(RS.delta_sigma(R, (k, amp * pk), PLANCK18,
                                          backend=ACCURATE))

        h = 1e-6
        ad = float(jax.grad(f)(1.0))
        fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
        assert ad == pytest.approx(fd, rel=1e-5)

    def test_jit_is_value_identical(self, bmo_pk):
        k, pk = bmo_pk
        R = jnp.asarray([0.2, 1.0])
        f = lambda p: RS.delta_sigma(R, (k, p), PLANCK18, backend=ACCURATE)
        assert np.allclose(np.asarray(jax.jit(f)(pk)), np.asarray(f(pk)),
                           rtol=1e-12)


class TestTheCompensatedAperture:
    r"""The filter that makes ``tau_ksz`` comparable to a measurement.

    ``PLAN.md`` item **D1**, and its reason: a stacked kinetic-SZ measurement
    cannot quote :math:`\tau` at a radius, because the estimator that removes
    the primary CMB *is* the aperture filter.  So the package computed a
    quantity nobody reports and the conversion was missing rather than wrong.
    """

    R_AP = jnp.asarray([0.3, 0.7, 1.5, 3.0])

    @staticmethod
    def _grid():
        return jnp.geomspace(1e-4, 20.0, 6000)

    def test_a_flat_profile_gives_zero(self):
        r"""The whole point of the :math:`\sqrt2`: disc and annulus have equal
        area, so anything flat across the aperture cancels.  The CMB is flat on
        these scales and a halo is not, which is why the filter works."""
        rp = self._grid()
        cap = np.asarray(RS.compensated_aperture(self.R_AP, rp,
                                              jnp.ones_like(rp)))
        scale = 2.0 * np.pi * np.asarray(self.R_AP) ** 2
        assert np.all(np.abs(cap) / scale < 1e-5)

    def test_the_outer_radius_is_root_two_in_double_precision(self):
        """``math.sqrt``, not ``jnp.sqrt``.

        Written with the latter it evaluated at import in whatever dtype JAX
        was configured for and came out 1.4142135381698608 -- float32 widened
        into a float64 Python float, with nothing recording where the digits
        went.  That is the trap ``DEFAULT_WTHETA_ELL`` documents, hit one
        module away from the paragraph describing it.
        """
        assert RS.CAP_OUTER == pytest.approx(np.sqrt(2.0), rel=0, abs=1e-16)
        assert RS.CAP_OUTER != np.float32(np.sqrt(2.0)).item()

    def test_a_profile_inside_the_aperture_is_recovered_whole(self):
        """With no signal in the annulus there is nothing to subtract, so the
        filter returns the full projected integral."""
        rp = self._grid()
        compact = jnp.exp(-0.5 * (rp / 0.02) ** 2)
        got = float(RS.compensated_aperture(1.0, rp, compact)[0])
        want = 2.0 * np.pi * float(jnp.trapezoid(compact * rp, rp))
        assert got == pytest.approx(want, rel=1e-5)

    def test_it_subtracts_a_background_that_a_profile_at_a_radius_would_keep(self):
        """A halo plus a constant sheet: the filter returns the halo alone,
        which is the difference between what is computed and what is measured.
        """
        rp = self._grid()
        halo = jnp.exp(-0.5 * (rp / 0.2) ** 2)
        sheet = 0.05
        alone = np.asarray(RS.compensated_aperture(self.R_AP, rp, halo))
        with_bg = np.asarray(RS.compensated_aperture(self.R_AP, rp, halo + sheet))
        scale = 2.0 * np.pi * np.asarray(self.R_AP) ** 2 * sheet
        assert np.all(np.abs(with_bg - alone) / scale < 1e-4)

    def test_a_short_grid_is_refused_rather_than_truncated(self):
        """``tau_ksz_cap`` checks the override instead of trusting it: a grid
        stopping inside ``sqrt2 * R_ap`` truncates the annulus, so the filter
        subtracts less background than it should and returns a CAP that is too
        large -- in a way no plot shows."""
        rho_s, r_s, k, pk = None, None, jnp.geomspace(1e-3, 1e3, 256), None
        with pytest.raises(ValueError, match="compensated aperture needs"):
            RS.tau_ksz_cap(jnp.asarray([2.0]), (k, jnp.ones_like(k)), PLANCK18,
                        rp=jnp.geomspace(0.1, 2.5, 64))

    def test_it_differentiates_in_the_aperture(self, bmo):
        """An aperture is a survey property a fit may want to marginalise
        over, so a structural zero here would be a flat direction.

        The grid has to be passed explicitly to differentiate in ``r_ap``,
        because ``tau_ksz_cap`` otherwise derives the grid's *endpoints* from
        the apertures -- and a grid is a static choice, for the same reason
        ``Statistic.x`` is a tuple of floats. Deriving one from a tracer would
        make the output shape depend on a value.
        """
        rho_s, r_s, k, pk = bmo
        rp = jnp.geomspace(0.01, 5.0, 512)
        g = jax.grad(lambda a: RS.tau_ksz_cap(jnp.asarray([a]), (k, pk),
                                              PLANCK18, rp=rp)[0])(1.0)
        assert np.isfinite(float(g)) and abs(float(g)) > 0.0

    def test_a_traced_aperture_without_a_grid_says_why(self):
        """Rather than a ConcretizationTypeError from four frames down."""
        k = jnp.geomspace(1e-3, 1e3, 256)
        with pytest.raises(TypeError, match="grid is a \\*static\\* choice"):
            jax.grad(lambda a: RS.tau_ksz_cap(jnp.asarray([a]),
                                              (k, jnp.ones_like(k)),
                                              PLANCK18)[0])(1.0)
