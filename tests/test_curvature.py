r"""Verification: :math:`\Omega_k`, and the four places it had to reach at once.

``PLAN.md`` item **E2**.  The benchmark called this the only *structural* gap --
"``Cosmology`` derives ``Omega_de`` from flatness, so there is no slot for
curvature and the package cannot be used for a curvature constraint at all" --
and the paper made the point that decided its shape: the cost is not any one of
the pieces but that **four must change together**.  A package with three of them
would report a constraint that was partly flat.

So the tests come in two halves.  One asserts each piece does what it should.
The other asserts that with :math:`\Omega_k = 0` **nothing moved, bit for bit**,
which is the only gate that shows the four were added without disturbing the
geometry everything else was measured in.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.cosmology.power import _nu_ratios
from ggah_mod.cosmology.background import (
    _sinhc, angular_diameter_distance, comoving_distance,
    comoving_distance_z1z2, comoving_volume_element, hubble_e,
    luminosity_distance, transverse_distance,
)

Z = jnp.linspace(0.01, 3.0, 40)
OPEN = Cosmology.create(Omega_k=0.05)
CLOSED = Cosmology.create(Omega_k=-0.05)


class TestFlatIsUntouched:
    r"""The gate: every flat number is what it was, to the last bit.

    Not "to a tolerance".  Each of the four changes is an addition of exactly
    zero or a multiplication by exactly one when :math:`\Omega_k = 0`, which is
    exact in floating point -- so anything less than bit-for-bit would mean a
    piece was restructured rather than extended, and the whole validation table
    of this package would need re-measuring.
    """

    def test_the_transverse_distance_is_the_radial_one(self):
        chi = comoving_distance(Z, PLANCK18)
        assert np.array_equal(np.asarray(transverse_distance(chi, PLANCK18)),
                              np.asarray(chi))

    def test_the_distances_are_unchanged(self):
        chi = np.asarray(comoving_distance(Z, PLANCK18))
        z = np.asarray(Z)
        assert np.array_equal(
            np.asarray(angular_diameter_distance(Z, PLANCK18)), chi / (1 + z))
        assert np.array_equal(
            np.asarray(luminosity_distance(Z, PLANCK18)), chi * (1 + z))

    def test_the_two_redshift_distance_is_still_a_difference(self):
        got = np.asarray(comoving_distance_z1z2(0.3, 1.2, PLANCK18))
        want = (np.asarray(comoving_distance(1.2, PLANCK18))
                - np.asarray(comoving_distance(0.3, PLANCK18)))
        # **Within one ulp, and not bit-for-bit, because of the jit boundary.**
        # `transverse_distance` *is* the identity at `Omega_k = 0` -- computed
        # eagerly the two sides agree bit for bit -- but
        # `comoving_distance_z1z2` is `@jax.jit`, so XLA fuses both quadratures
        # and the difference into one computation and may round once where the
        # right-hand side rounds twice.  Demanding equality across that
        # boundary asserts something about XLA's fusion rather than about
        # curvature, and it held for as long as it did by luck.
        assert np.allclose(got, want, rtol=4e-16, atol=0.0)

    def test_e_of_z_is_still_exactly_one_at_zero(self):
        """``Omega_de`` closes against the curvature term too now, so this is
        the check that the closure and the sum still agree."""
        assert float(hubble_e(0.0, PLANCK18)) == 1.0

    def test_the_lensing_kernel_is_unchanged(self):
        from ggah_mod.observables.kernels import lensing_efficiency, limber_grid

        z, chi = limber_grid(PLANCK18, z_max=2.0, n=48)
        nz = jnp.exp(-0.5 * ((z - 0.6) / 0.2) ** 2)
        k = lensing_efficiency(z, chi, nz, PLANCK18)

        # Rebuilt the way it was written before E2: the bare ratio.
        n_chi = (jnp.asarray(nz) * jnp.gradient(z) / jnp.gradient(chi))
        n_chi = n_chi / jnp.trapezoid(n_chi, chi)
        ratio = (chi[None, :] - chi[:, None]) / chi[None, :]
        geom = jnp.where(chi[None, :] > chi[:, None], ratio, 0.0)
        g = jnp.trapezoid(n_chi[None, :] * geom, chi, axis=-1)
        amp = 1.5 * PLANCK18.Omega_m / PLANCK18.hubble_distance ** 2
        want = amp * chi * (1.0 + z) * g
        assert np.allclose(np.asarray(k.w), np.asarray(want), rtol=1e-14)


class TestTheEntireFunctionThatMakesItOneExpression:
    r""":math:`s(x)=\sinh\sqrt x/\sqrt x`, which is entire.

    Open, flat and closed are one analytic expression rather than three
    branches, and :math:`x=0` is an interior point rather than a special case --
    which is what makes the flat limit exact instead of approached.
    """

    def test_it_is_exactly_one_at_zero(self):
        assert float(_sinhc(0.0)) == 1.0

    @pytest.mark.parametrize("x", [-4.0, -1.0, -1e-2, 1e-2, 1.0, 4.0])
    def test_it_matches_the_closed_form_on_both_sides(self, x):
        r = np.sqrt(abs(x))
        want = np.sinh(r) / r if x > 0 else np.sin(r) / r
        assert float(_sinhc(x)) == pytest.approx(want, rel=1e-14)

    def test_the_two_representations_agree_where_they_change_over(self):
        r"""The series and the closed form meet without a step.

        Checked at the *same* :math:`x`, on both sides of the threshold, rather
        than by comparing the function at two nearby points -- which is what a
        first version of this did, and which measures :math:`s'(x)\,\Delta x`
        (about 3e-7 here) rather than any discontinuity.
        """
        eps = 1e-3
        for x in (-eps, -eps * 0.5, eps * 0.5, eps):
            r = np.sqrt(abs(x))
            closed = np.sinh(r) / r if x > 0 else np.sin(r) / r
            assert float(_sinhc(x)) == pytest.approx(closed, rel=1e-13), x

    def test_it_has_no_step_at_the_branch_threshold(self):
        r"""Straddling the switch: what crosses it is the function's own slope
        and nothing else.

        The difference across :math:`x = \pm10^{-3}` is :math:`3.33\times
        10^{-10}`, which looks like a step until it is compared with
        :math:`s'(x)\,\Delta x = 2\times10^{-9}/6 = 3.33\times10^{-10}`.  It is
        the slope, to five figures.  So the test is not "the jump is small" --
        it is "the jump **is** the slope", which is the statement that no step
        is hiding underneath a small number.
        """
        eps, d = 1e-3, 1e-9
        for sign in (+1.0, -1.0):
            lo = float(_sinhc(sign * (eps - d)))
            hi = float(_sinhc(sign * (eps + d)))
            # s(x) = 1 + x/6 + ..., so ds/dx -> 1/6 near zero.
            expected = sign * 2.0 * d / 6.0
            assert (hi - lo) == pytest.approx(expected, rel=1e-4)

    def test_both_branches_are_guarded(self):
        """``where`` evaluates both, and a ``nan`` in the unused one still
        poisons the gradient of the used one -- the trap ``bessel_k`` was once
        caught by."""
        for x in (-2.0, 0.0, 2.0):
            g = float(jax.grad(lambda v: _sinhc(v))(x))
            assert np.isfinite(g)


class TestCurvatureDoesSomething:
    """The other half: with a non-zero value, each piece moves the right way."""

    def test_open_stretches_and_closed_shrinks_the_transverse_distance(self):
        chi_o = comoving_distance(1.0, OPEN)
        chi_c = comoving_distance(1.0, CLOSED)
        assert float(transverse_distance(chi_o, OPEN)[0]) > float(chi_o[0])
        assert float(transverse_distance(chi_c, CLOSED)[0]) < float(chi_c[0])

    def test_the_expansion_carries_the_curvature_term(self):
        """``+ Omega_k (1+z)^2`` -- an open universe expands faster at fixed
        matter density, so its comoving distance to a given redshift is
        shorter."""
        assert float(hubble_e(1.0, OPEN)) > float(hubble_e(1.0, PLANCK18))
        assert float(comoving_distance(1.0, OPEN)[0]) < \
            float(comoving_distance(1.0, PLANCK18)[0])

    def test_the_budget_still_closes(self):
        for c in (OPEN, CLOSED, PLANCK18):
            assert float(hubble_e(0.0, c)) == pytest.approx(1.0, abs=1e-12)
            total = (float(c.Omega_gamma) + float(c.Omega_cb)
                     + float(c.Omega_nu_today) + float(c.Omega_k)
                     + float(c.Omega_de))
            assert total == pytest.approx(1.0, abs=1e-12)

    def test_the_volume_element_uses_the_transverse_distance(self):
        """A solid angle subtends an *area*, so it is f_K squared."""
        for c in (OPEN, CLOSED):
            chi = comoving_distance(1.0, c)
            want = (c.hubble_distance
                    * float(transverse_distance(chi, c)[0]) ** 2
                    / float(hubble_e(1.0, c)))
            assert float(comoving_volume_element(1.0, c)[0]) == \
                pytest.approx(want, rel=1e-12)

    def test_it_carries_a_gradient(self):
        """A curvature constraint needs one, which is the point of adding the
        parameter rather than the geometry."""
        def d_a(ok):
            return angular_diameter_distance(
                1.0, PLANCK18.replace(Omega_k=ok))[0]

        g = float(jax.grad(d_a)(0.0))
        assert np.isfinite(g) and abs(g) > 1.0


class TestTheSpectrumIsNotSilentlyFlat:
    r"""The fifth piece, and the one that would have made this half-done.

    The four changes above curve the *background*.  Both Boltzmann solvers hard
    coded ``Omega_k = 0`` with a comment saying "the package is flat
    throughout", so without this the package would have paired a curved
    :math:`E(z)` with a flat :math:`P(k)` -- a constraint partly in one geometry
    and partly in the other, which is exactly what doing the four together was
    meant to prevent.

    **Every backend now carries curvature, and ``emu_pk`` is the one that
    changed.**  It declared ``supports_curvature = False`` and refused, from
    item E2 until ``emu_pk`` 2.0.0 shipped a checkpoint trained with
    :math:`\Omega_k` on 2026-09-10.  That is the single event ``CROSS_REPO.md``
    X9 and X10 name as reopening layers 1 and 2.  The refusal machinery stays --
    it is what the declaration is *for* -- and what these tests now assert is
    that the declaration follows the installed box rather than this file.
    """

    def test_the_capability_is_declared_not_sniffed(self):
        from ggah_mod.cosmology.power import make_pk

        assert make_pk("class").supports_curvature is True
        assert make_pk("camb").supports_curvature is True
        assert make_pk("emu_pk").supports_curvature is True

    def test_the_differentiable_backend_now_carries_curvature(self):
        """It refused this until 2.0.0.  What makes the answer trustworthy is
        not that it returns one but that it returns a *different* one: a
        network ignoring the input would give back the flat spectrum."""
        from ggah_mod.cosmology.power import make_pk

        k = np.logspace(-3, 0, 8)
        emu = make_pk("emu_pk")
        flat = np.asarray(emu.pk(k, 0.0, PLANCK18.replace(Omega_k=0.0)))
        open_ = np.asarray(emu.pk(k, 0.0, OPEN))
        closed = np.asarray(emu.pk(k, 0.0, CLOSED))
        assert np.all(open_ > 0.0) and np.all(closed > 0.0)
        assert np.max(np.abs(open_ / flat - 1.0)) > 1e-3
        assert np.max(np.abs(closed / flat - 1.0)) > 1e-3
        # And the two sides are not the same answer wearing a sign.
        assert np.max(np.abs(open_ / closed - 1.0)) > 1e-3

    def test_the_refusal_still_exists_for_a_backend_that_cannot(self):
        """The machinery is what the declaration is for, and no shipped backend
        exercises it any more.  A stub keeps it reachable, so the guard cannot
        rot into a branch nothing has taken since the day it stopped firing."""
        from ggah_mod.cosmology.power import _check_curvature

        class Flat:
            name = "stub"
            supports_curvature = False

        _check_curvature(Flat(), PLANCK18)          # flat is fine
        with pytest.raises(ValueError, match="supports_curvature"):
            _check_curvature(Flat(), OPEN)

    def test_it_still_answers_for_a_flat_cosmology(self):
        from ggah_mod.cosmology.power import make_pk

        p = np.asarray(make_pk("emu_pk").pk(np.logspace(-3, 0, 8), 0.0,
                                            PLANCK18))
        assert p.shape == (8,) and np.all(p > 0.0)


class TestTheMappingFollowsTheEmulatorsOwnBox:
    r"""What ``emu_pk`` is handed, and what happens when its box grows.

    ``emu_pk`` 2.0 appends ``Omega_k``, ``nu_r1`` and ``nu_r2`` to
    :data:`emu_pk.box.PARAMS`, leaving the first eight entries and their
    meanings alone so that an existing mapping keeps working.  This package
    builds its vector *from* that tuple, so the append costs no edit here -- and
    the capability flag, which is a claim rather than a lookup, is checked
    against the box so it cannot go stale in the other direction.
    """

    def test_the_vector_is_exactly_as_long_as_the_box(self):
        """A short vector is not an error to JAX: ``_forward`` clamps the gather
        and returns a finite, smooth, wrong spectrum, measured there at 10.3 per
        cent in P(0.05).  Length is the only thing standing between that and a
        published number."""
        from emu_pk import box
        from ggah_mod.cosmology.power import make_pk

        assert len(make_pk("emu_pk")._params(PLANCK18)) == len(box.PARAMS)

    def test_the_declared_capability_is_the_installed_box(self):
        """**This is the reminder, and it is meant to fail.**  The day an
        ``emu_pk`` carrying ``Omega_k`` is installed, this goes red and says
        that ``supports_curvature`` is a stale ``False`` -- which is the moment
        to flip it, lift the pin in ``pyproject.toml``, and re-run Sec. 8 of
        the paper rather than edit it.  A capability that drifted the other way
        would return the flat spectrum in silence."""
        from emu_pk import box
        from ggah_mod.cosmology.power import make_pk

        emu = make_pk("emu_pk")
        assert emu.supports_curvature == ("Omega_k" in box.PARAMS)
        assert emu.supports_nondegenerate_nu == ("nu_r1" in box.PARAMS)

    @pytest.mark.parametrize("hierarchy", ["degenerate", "massless"])
    def test_the_degenerate_conventions_are_exactly_one_third(self, hierarchy):
        """A massless cosmology *is* the degenerate one -- ``nu_masses``
        returns (0, 0, 0) in every ordering -- so the 0/0 at the origin has an
        answer rather than a limit."""
        from ggah_mod.cosmology.power import _nu_ratios

        c = Cosmology.create(sum_mnu=0.0 if hierarchy == "massless" else 0.12,
                             nu_hierarchy=hierarchy)
        r1, r2 = _nu_ratios(c)
        assert float(r1) == pytest.approx(1 / 3, rel=1e-12)
        assert float(r2) == pytest.approx(1 / 3, rel=1e-12)

    @pytest.mark.parametrize("hierarchy", ["normal", "inverted"])
    def test_a_split_lands_inside_the_boxs_ordered_simplex(self, hierarchy):
        """``box.sample`` rejects the corner where r2 < r1 or r2 > (1-r1)/2, so
        r1 <= 1/3 is the constraint rather than a choice.  A physical splitting
        has to satisfy it or the network is being asked outside its design."""
        from ggah_mod.cosmology.power import _nu_ratios

        r1, r2 = (float(v) for v in
                  _nu_ratios(Cosmology.create(sum_mnu=0.12,
                                              nu_hierarchy=hierarchy)))
        r3 = 1.0 - r1 - r2
        assert r1 <= r2 <= r3
        assert 0.0 <= r1 <= 1 / 3 and 0.0 <= r2 <= 0.5
        assert r1 + r2 + r3 == pytest.approx(1.0, rel=1e-14)

    def test_the_equal_mass_ratio_is_not_one_ulp_outside_its_own_bound(self):
        """The box's upper bound on ``nu_r1`` **is** 1/3, and the degenerate
        hierarchy sits exactly on it.  Computing it as ``m[0] / sum`` returns
        0.33333333333333337 at 0.06 and 0.12 eV -- one ulp above -- so the most
        ordinary cosmology in the package lands outside the box by
        construction.  ``emu_pk``'s bounds carry float32 slack and absorb it
        today, which makes it luck; any stricter check downstream refuses the
        fiducial.  Found by the session running the derivative scan, whose own
        first bounds check refused it.
        """
        box = pytest.importorskip("emu_pk.box")
        hi1, hi2 = box.BOX["nu_r1"][1], box.BOX["nu_r2"][1]
        for mnu in (0.0, 0.001, 0.03, 0.06, 0.12, 0.2, 0.3, 0.45, 0.6):
            for h in ("degenerate", "normal", "inverted", "massless"):
                try:
                    c = Cosmology.create(sum_mnu=mnu, nu_hierarchy=h)
                except ValueError:
                    continue            # below this ordering's own mass floor
                r1, r2 = (float(v) for v in _nu_ratios(c))
                assert r1 <= hi1, (mnu, h, repr(r1))
                assert r2 <= hi2, (mnu, h, repr(r2))
                assert 0.0 <= r1 <= r2 <= 1.0 - r1 - r2 + 1e-15, (mnu, h)

    def test_the_ratios_carry_a_gradient_and_the_origin_carries_no_nan(self):
        """Written as a ``where`` on an already-safe denominator, because a NaN
        in the untaken branch poisons the gradient through both -- the care
        ``_sinhc`` takes at x = 0, for the same reason."""
        from ggah_mod.cosmology.power import _nu_ratios

        def r1_of(s):
            return _nu_ratios(Cosmology.create(sum_mnu=s,
                                               nu_hierarchy="normal"))[0]

        assert float(jax.grad(r1_of)(0.12)) != 0.0
        assert np.isfinite(float(jax.grad(r1_of)(0.0)))
