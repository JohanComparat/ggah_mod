r"""Beyond-linear halo bias: the table, the AW10 rescaling, and the edge rules.

The load-bearing test here is :class:`TestRescalingDirection`.  Upstream states
the direction of the length rescaling three inconsistent ways -- the
``calculate_rescaling_parameters`` header says "R -> R/s (or k -> s*k)", the
``smin_rescale`` comment says "R -> sR", and ``init_BNL`` multiplies the table's
own k axis by ``s`` before interpolating, which composes to the opposite of what
the cost integrand implies.  Only the cost integrand is unambiguous, so the
direction is pinned here by construction rather than inherited from any of them.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from ggah_mod.halos import beyond_linear_bias as B
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from conftest import AnalyticPk


@pytest.fixture(scope="module")
def tab():
    return B.load()


@pytest.fixture(scope="module")
def analytic():
    return AnalyticPk()


@pytest.fixture(scope="module")
def spectrum(analytic):
    """A differentiable target spectrum: (k, P_cb) at the fiducial cosmology."""
    k = jnp.logspace(-4.0, 2.0, 512)
    return k, jnp.asarray(analytic.pk_cb(k, 0.0, PLANCK18))


# ---------------------------------------------------------------------------
# The distilled table
# ---------------------------------------------------------------------------

class TestTable:

    def test_covers_all_35_snapshots(self, tab):
        assert tab["snap"].shape == (35,)
        assert int(tab["snap"][0]) == 36 and int(tab["snap"][-1]) == 85
        assert float(tab["z"][-1]) == pytest.approx(0.0)
        assert float(tab["z"][0]) == pytest.approx(2.891, abs=1e-3)
        assert np.all(np.diff(np.asarray(tab["a"])) > 0)

    def test_shapes(self, tab):
        assert tab["beta"].shape == (35, 25, 8, 8)
        assert tab["sigma_beta"].shape == tab["beta"].shape
        assert tab["nu"].shape == (35, 8)
        assert tab["ln_sigma_md"].shape == tab["ln_r"].shape

    def test_beta_is_symmetric_in_its_two_mass_arguments(self, tab):
        """beta^NL(k, nu1, nu2) == beta^NL(k, nu2, nu1), as the data is.

        Not a tautology: the upstream file is a flat list and the reshape that
        recovers the two bin axes could transpose them against the k axis.  A
        botched reshape shows up here and essentially nowhere else.
        """
        beta = np.asarray(tab["beta"])
        np.testing.assert_allclose(beta, beta.transpose(0, 1, 3, 2), atol=0)

    def test_nu_bins_increase_within_every_snapshot(self, tab):
        assert np.all(np.diff(np.asarray(tab["nu"]), axis=1) > 0)

    def test_nu_support_moves_with_redshift(self, tab):
        """The reason a z = 0-only table cannot stand in for the others."""
        nu = np.asarray(tab["nu"])
        assert nu[-1, 0] == pytest.approx(0.846, abs=1e-2)     # z = 0
        assert nu[-1, -1] == pytest.approx(3.711, abs=1e-2)
        assert nu[8, 0] > 1.3                                   # z = 1
        assert nu[8, -1] > 4.2

    def test_growth_factorisation_holds(self, tab):
        """sigma_MD(R, a) = g(a) sigma_MD(R, 1), which the cost function assumes."""
        assert float(tab["g_residual"]) < 1e-4
        g = np.asarray(tab["g_md"])
        assert np.all(np.diff(g) > 0)
        assert g[-1] == pytest.approx(1.0)

    def test_mdr1_amplitude_was_solved_not_assumed(self, tab):
        assert float(tab["mdr1_sigma8"]) == pytest.approx(0.82)
        # sigma_8 = 0.82 is well above Planck's, so the amplitude must exceed
        # the 3.044 default it was solved from.
        assert float(tab["mdr1_ln10A_s"]) > 3.10


# ---------------------------------------------------------------------------
# The Angulo & White (2010) rescaling
# ---------------------------------------------------------------------------

def _ln_sigma_tgt_from_table(tab, lam, g):
    """A target whose sigma(R) is exactly ``g * sigma_MD(R/lam)``.

    Built straight from the stored MDR1 profile, so the exact answer is known
    without involving a Boltzmann solver: the cost is minimised at ``s = lam``
    with ``g* = g``, and the residual is zero.
    """
    nodes = jnp.linspace(jnp.log(B.R_RESCALE[0]), jnp.log(B.R_RESCALE[1]),
                         B.N_R_COST)
    return nodes, B._ln_sigma_md(nodes - jnp.log(lam), tab) + jnp.log(g)


class TestRescalingDirection:
    """What ``s`` means, fixed by construction rather than by reading a comment."""

    @pytest.mark.parametrize("lam", [0.7, 0.85, 1.0, 1.2, 1.4])
    def test_recovers_a_known_length_rescaling(self, tab, lam):
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, lam, 1.0)
        s_grid = jnp.linspace(*B.S_RANGE, B.N_S)
        cost, _ = B._cost_and_best_g(s_grid, nodes, ln_sig, tab)
        j = int(jnp.argmin(cost))
        s = B._parabolic_vertex(cost[j - 1], cost[j], cost[j + 1],
                                s_grid[j], s_grid[1] - s_grid[0])
        assert float(s) == pytest.approx(lam, rel=2e-3)
        assert float(cost[j]) < 1e-8

    @pytest.mark.parametrize("g", [0.4, 0.65, 0.9])
    def test_recovers_a_known_amplitude(self, tab, g):
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, 1.0, g)
        _, g_star = B._cost_and_best_g(jnp.array([1.0]), nodes, ln_sig, tab)
        assert float(g_star[0]) == pytest.approx(g, rel=1e-6)

    def test_amplitude_and_length_are_recovered_together(self, tab):
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, 1.25, 0.6)
        s_grid = jnp.linspace(*B.S_RANGE, B.N_S)
        cost, g_star = B._cost_and_best_g(s_grid, nodes, ln_sig, tab)
        j = int(jnp.argmin(cost))
        assert float(s_grid[j]) == pytest.approx(1.25, abs=0.011)
        assert float(g_star[j]) == pytest.approx(0.6, rel=1e-3)

    def test_the_table_is_read_at_s_times_k_not_k_over_s(self, tab, spectrum):
        """The composition, which is the half upstream states inconsistently.

        ``s = lambda`` is fixed by the tests above.  What remains is which way
        it is applied, and the two choices are not subtly different: they read
        the table an octave apart.  Where beta^NL rises with k, reading at
        ``s*k`` (a smaller wavenumber) must give a *smaller* correction than
        reading at ``k`` -- and reading at ``k/s`` a larger one.

        **The premise is asserted, not assumed.**  beta^NL turns over, so the
        ordering is a local claim rather than a global one, and which side of
        the turnover ``k/s`` lands on depends on the fitted ``s`` and therefore
        on the target spectrum.  An earlier version of this test hard-coded
        ``k = 0.3``, which sat on the rising side for one stub spectrum and
        past the peak for another -- a test that passes or fails on the choice
        of fixture is testing the fixture.  The rise is now checked first.
        """
        k, pk_cb = spectrum
        k_out = jnp.array([0.2])
        # The grid is rebuilt below from (i0, i1, w), the clamped blend; this
        # target sits past the sequence's end, so the growth extrapolation is
        # pinned off -- it is a statement about g, and this test is about k.
        t = B.table_at(k_out, k, pk_cb, check=False, extrapolate=False)
        assert float(t.s) < 0.99, "this target must actually be rescaled"

        grid = ((1.0 - t.w) * tab["beta"][t.i0] + t.w * tab["beta"][t.i1])
        unrescaled = B._interp_k(grid, k_out, tab["ln_k"])
        over_s = B._interp_k(grid, k_out / t.s, tab["ln_k"])

        b, b_none, b_wrong = (float(x[0, 3, 3])
                              for x in (t.beta, unrescaled, over_s))
        # The premise: beta^NL is rising across the whole octave the two
        # readings span.  If this fails the ordering below means nothing.
        span = jnp.asarray([t.s * k_out[0], k_out[0], k_out[0] / t.s])
        rising = np.asarray([float(B._interp_k(grid, jnp.array([x]),
                                               tab["ln_k"])[0, 3, 3])
                             for x in span])
        assert np.all(np.diff(rising) > 0), (
            f"beta^NL is not monotone over {np.asarray(span)}; the ordering "
            "claim does not apply at this wavenumber")
        assert b_none > 0.0
        assert b < b_none < b_wrong               # the ordering is the claim
        assert b == pytest.approx(
            float(B._interp_k(grid, t.s * k_out, tab["ln_k"])[0, 3, 3]))


class TestRescalingSolver:

    def test_cost_is_zero_for_a_perfect_match(self, tab):
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, 1.0, 1.0)
        cost, _ = B._cost_and_best_g(jnp.array([1.0]), nodes, ln_sig, tab)
        assert float(cost[0]) < 1e-12

    def test_amplitude_is_unconstrained_inside_the_fit(self, tab):
        """g must not be clamped where s is solved.

        Clamping there biases ``s`` for any target sitting on the boundary --
        which includes MultiDark itself at z = 0, the one case with a known
        exact answer.
        """
        g_hi = float(tab["g_md"][-1])
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, 1.0, 1.5 * g_hi)
        _, g_star = B._cost_and_best_g(jnp.array([1.0]), nodes, ln_sig, tab)
        assert float(g_star[0]) > g_hi

    def test_out_of_range_amplitude_warns_and_is_held(self, tab, spectrum):
        k, pk_cb = spectrum
        B._WARNED.clear()
        with pytest.warns(RuntimeWarning, match="MultiDark sequence"):
            B.table_at(jnp.array([0.3]), k, 1e-4 * pk_cb, check=True)
        B._WARNED.clear()

    def test_closed_form_minimum_is_the_reference_cost(self, tab):
        """The 1-D reduction is exact, not an approximation.

        :func:`rescaling_cost` is a literal transcription of the reference's
        integrand -- ``(1 - sigma_MD(R/s, a')/sigma_tgt(R, a))^2`` averaged over
        ln R.  The solver never evaluates it: it uses the closed form obtained
        by minimising analytically over the amplitude, which is what replaces
        the reference's 35-point scan of ``a'``.  If the two disagreed, the
        collapse from two dimensions to one would be quietly changing the
        quantity being minimised.
        """
        nodes, ln_sig = _ln_sigma_tgt_from_table(tab, 1.25, 0.6)
        c_closed, g_star = B._cost_and_best_g(jnp.array([1.25]), nodes,
                                              ln_sig, tab)
        c_literal = B.rescaling_cost(1.25, g_star[0], nodes, ln_sig, tab)
        assert float(c_literal) == pytest.approx(float(c_closed[0]), abs=1e-14)

        # and it really is the minimum over g, not just some value of it
        for factor in (0.5, 0.9, 1.1, 2.0):
            worse = B.rescaling_cost(1.25, factor * g_star[0], nodes,
                                     ln_sig, tab)
            assert float(worse) > float(c_closed[0])

    def test_parabolic_vertex_handles_a_flat_triple(self):
        v = B._parabolic_vertex(jnp.array(1.0), jnp.array(1.0), jnp.array(1.0),
                                jnp.array(0.5), 0.01)
        assert float(v) == pytest.approx(0.5)

    @pytest.mark.slow
    def test_self_consistency_against_a_boltzmann_mdr1(self, tab):
        """Target = MDR1 itself must give s = 1 and recover the scale factor."""
        from ggah_mod.cosmology.parameters import Cosmology

        pk = make_pk("camb")
        mdr1 = Cosmology.create(**B.MDR1).replace(
            ln10A_s=float(tab["mdr1_ln10A_s"]))
        k = jnp.logspace(-4.0, 2.0, 512)
        for i in (0, 8, 20, 34):
            z = float(tab["z"][i])
            s, a_md, _, _, _, cost = B.match(k, pk.pk_cb(k, z, mdr1))
            assert float(s) == pytest.approx(1.0, abs=1e-3)
            assert float(a_md) == pytest.approx(float(tab["a"][i]), rel=1e-3)
            assert float(cost) < 1e-8


# ---------------------------------------------------------------------------
# Edge rules
# ---------------------------------------------------------------------------

class TestEdgeRules:

    @pytest.fixture(scope="class")
    def pinned(self):
        return B.table_at(jnp.logspace(-3, 0.7, 400), snap=85)

    def test_taper_endpoints(self):
        assert float(B._k_taper(jnp.array([B.K_MIN_BNL / B.K_TAPER]))[0]) == 0.0
        assert float(B._k_taper(jnp.array([B.K_MIN_BNL * B.K_TAPER]))[0]) == 1.0
        assert float(B._k_taper(jnp.array([B.K_MIN_BNL]))[0]) == pytest.approx(0.5)

    def test_below_the_taper_beta_vanishes(self):
        t = B.table_at(jnp.array([1e-4, 1e-3, B.K_MIN_BNL / 1.26]), snap=85)
        np.testing.assert_allclose(np.asarray(t.beta), 0.0, atol=1e-12)

    def test_above_the_table_k_is_clamped_not_zeroed(self, tab):
        k_max = float(tab["k"][-1])
        edge = B.table_at(jnp.array([k_max]), snap=85).beta[0]
        beyond = B.table_at(jnp.array([5.0, 50.0]), snap=85).beta
        assert np.abs(np.asarray(edge)).max() > 1e-3
        np.testing.assert_allclose(np.asarray(beyond[0]), np.asarray(edge),
                                   rtol=1e-12)
        np.testing.assert_allclose(np.asarray(beyond[1]), np.asarray(edge),
                                   rtol=1e-12)

    def test_beta_is_continuous_in_k(self, pinned):
        """No step anywhere -- at the taper, or at the top of the table."""
        b = np.asarray(pinned.beta)[:, 1, 1]
        step = np.abs(np.diff(b))
        assert step.max() < 25.0 * np.median(step[step > 0])

    def test_nu_below_the_table_extrapolates(self, pinned):
        """Linear extrapolation, not a mask to zero.

        Clamping here would silently truncate the low-mass end of every mass
        integral, which is the paper's own preferred variant of the correction.
        """
        nu_ref = pinned.nu
        phi = np.asarray(B.project_weights(jnp.array([nu_ref[0] - 0.3]), nu_ref))
        assert phi.sum() == pytest.approx(1.0)
        assert phi[0, 0] > 1.0 and phi[0, 1] < 0.0

    def test_nu_above_the_table_is_held_constant(self, pinned):
        nu_ref = pinned.nu
        edge = np.asarray(B.project_weights(jnp.array([nu_ref[-1]]), nu_ref))
        above = np.asarray(B.project_weights(jnp.array([nu_ref[-1] + 2.0]), nu_ref))
        np.testing.assert_allclose(above, edge, atol=1e-12)
        assert above[0, -1] == pytest.approx(1.0)

    def test_nu_outside_the_hard_bounds_is_zero(self, pinned):
        phi = B.project_weights(jnp.array([-0.5, B.NU_HARD[1] + 1.0]), pinned.nu)
        np.testing.assert_array_equal(np.asarray(phi), 0.0)

    def test_nu_projection_is_continuous_across_both_edges(self, pinned):
        nu_ref = pinned.nu
        for edge in (float(nu_ref[0]), float(nu_ref[-1])):
            phi = np.asarray(B.project_weights(
                jnp.linspace(edge - 0.05, edge + 0.05, 401), nu_ref))
            assert np.abs(np.diff(phi, axis=0)).max() < 1e-2


class TestExtrapolationInGrowth:
    """Past either end of the MultiDark sequence the table is carried, not held.

    Targets are placed on the sequence by scaling one spectrum: ``s`` depends
    on the shape of sigma(R) only, so a factor ``f**2`` on the spectrum moves
    ``g`` by ``f`` and nothing else.
    """

    K_OUT = jnp.array([0.1, 0.3, 0.5, 0.7])

    @pytest.fixture(scope="class")
    def at(self, spectrum):
        k, pk_cb = spectrum
        g0 = float(B.table_at(self.K_OUT, k, pk_cb, check=False).g)

        def at(g, **kw):
            return B.table_at(self.K_OUT, k, (g / g0) ** 2 * pk_cb,
                              check=False, **kw)
        return at

    @pytest.fixture(scope="class")
    def ends(self, tab):
        return float(tab["g_md"][0]), float(tab["g_md"][-1])

    def test_it_is_the_identity_inside_the_sequence(self, at):
        for g in (0.4, 0.8, 0.99):
            a, b = at(g), at(g, extrapolate=False)
            assert float(a.g) == pytest.approx(g, rel=1e-9)
            assert np.array_equal(np.asarray(a.beta), np.asarray(b.beta))
            assert np.array_equal(np.asarray(a.nu), np.asarray(b.nu))

    def test_it_is_continuous_at_both_ends(self, at, ends):
        """The step across each end shrinks with the distance straddled.

        A jump would not: it would stay put as the two points close in.  The
        steps are not compared with a tolerance, because both slopes are large
        at the ends -- an element near 170 at the high-z end moves by 4e-4
        across a 2e-7 straddle.
        """
        def jump(end, sign, h):
            a, b = at(end - sign * h), at(end + sign * h)
            return (np.max(np.abs(np.asarray(b.beta) - np.asarray(a.beta))),
                    np.max(np.abs(np.asarray(b.nu) - np.asarray(a.nu))))

        for end, sign in zip(ends, (-1.0, 1.0)):
            wide, narrow = jump(end, sign, 1e-6), jump(end, sign, 1e-8)
            for w, n in zip(wide, narrow):
                assert n < 0.02 * w
                assert n < 1e-4

    def test_it_is_linear_in_g_with_the_fitted_slope(self, at, ends, tab):
        """beta(g2) - beta(g1) is the slope times g2 - g1, element by element."""
        hi = ends[1]
        a, b = at(hi + 0.05), at(hi + 0.2)
        ex = B._extrapolation()
        s = np.asarray(B.table_at(self.K_OUT, snap=85).s)
        assert float(s) == 1.0
        # The slope lives on the table's own k grid; read it where the pair is.
        slope_k = np.asarray(B._interp_k(ex["slope_hi"], float(a.s) * self.K_OUT,
                                         tab["ln_k"]))
        got = np.asarray(b.beta) - np.asarray(a.beta)
        assert np.allclose(got, slope_k * (float(b.g) - float(a.g)), atol=1e-9)

    def test_the_slope_is_a_weighted_least_squares_line(self, tab):
        g = np.asarray(tab["g_md"])
        sel = g >= g[-1] - B.G_BASELINE
        sig = np.asarray(tab["sigma_beta"], dtype=float)
        slope = np.asarray(B._extrapolation()["slope_hi"])
        for ik, i, j in ((22, 0, 0), (19, 2, 3), (24, 5, 5)):
            y = np.asarray(tab["beta"])[sel, ik, i, j]
            ref = np.polyfit(g[sel], y, 1, w=1.0 / np.abs(sig[sel, ik, i, j]))[0]
            assert slope[ik, i, j] == pytest.approx(ref, rel=1e-9)

    def test_the_peak_heights_follow_the_lowest_bins_mass(self, at, ends):
        """nu_0 goes as 1/g past the end; the spacing is the end snapshot's."""
        lo, hi = ends
        for g, end, i in ((hi + 0.15, hi, -1), (lo - 0.1, lo, 0)):
            t, e = at(g), B.load()
            nu_end = np.asarray(e["nu"][i])
            assert float(t.nu[0]) == pytest.approx(nu_end[0] * end / g, rel=1e-9)
            assert np.allclose(np.diff(np.asarray(t.nu)), np.diff(nu_end),
                               atol=1e-12)

    def test_it_is_held_beyond_the_reach(self, at, ends):
        hi = ends[1]
        a, b = at(hi + B.G_REACH), at(hi + B.G_REACH + 0.2)
        assert np.allclose(np.asarray(a.beta), np.asarray(b.beta), atol=1e-12)
        assert np.allclose(np.asarray(a.nu), np.asarray(b.nu), atol=1e-12)

    def test_the_clamp_is_still_available_and_says_so(self, spectrum, at, ends):
        k, pk_cb = spectrum
        hi = ends[1]
        clamped = at(hi + 0.1, extrapolate=False)
        end = B.table_at(self.K_OUT, snap=85)
        assert np.allclose(np.asarray(clamped.nu), np.asarray(end.nu))
        B._WARNED.clear()
        g0 = float(B.table_at(self.K_OUT, k, pk_cb, check=False).g)
        f2 = ((hi + 0.1) / g0) ** 2
        with pytest.warns(RuntimeWarning, match="clamped to the end snapshot"):
            B.table_at(self.K_OUT, k, f2 * pk_cb, extrapolate=False)
        # Within the reach it is the default path and says nothing -- a
        # downstream suite that errors on RuntimeWarning runs at z = 0.135.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            B.table_at(self.K_OUT, k, f2 * pk_cb)
        f2_far = ((hi + B.G_REACH + 0.1) / g0) ** 2
        with pytest.warns(RuntimeWarning, match="held there"):
            B.table_at(self.K_OUT, k, f2_far * pk_cb)
        B._WARNED.clear()

    def test_a_pinned_snapshot_is_never_extrapolated(self):
        a = B.table_at(self.K_OUT, snap=85)
        b = B.table_at(self.K_OUT, snap=85, extrapolate=False)
        assert np.array_equal(np.asarray(a.beta), np.asarray(b.beta))

    def test_the_amplitude_gradient_survives_past_the_end(self, analytic, ends):
        """On the clamp this derivative was zero; past the end it is not."""
        k = jnp.logspace(-4, 2, 512)
        base = analytic.pk_cb(k, 0.0, PLANCK18)
        g0 = float(B.table_at(self.K_OUT, k, base, check=False).g)
        scale = ((ends[1] + 0.08) / g0) ** 2

        def f(lnA):
            c = PLANCK18.replace(ln10A_s=lnA)
            t = B.table_at(self.K_OUT, k, scale * analytic.pk_cb(k, 0.0, c),
                           check=False)
            return jnp.sum(t.beta[:, 1, 1]) + t.nu[0]

        grad = float(jax.grad(f)(3.044))
        h = 2e-3
        fd = (float(f(3.044 + h)) - float(f(3.044 - h))) / (2 * h)
        assert abs(grad) > 1e-2
        assert grad == pytest.approx(fd, rel=1e-5)

    def test_the_hold_out_behind_the_docstring(self, tab):
        """Carried 0.09 in g from a sequence cut at g = 0.907, the rule beats
        holding the anchor on the four lowest bins, which is where the matter
        spectrum's correction lives; the docstring quotes these numbers."""
        g = np.asarray(tab["g_md"])
        beta = np.asarray(tab["beta"])
        sig = np.asarray(tab["sigma_beta"], dtype=float)
        k = np.asarray(tab["k"])
        end = int(np.argmin(np.abs(g - 0.907)))
        sel = np.flatnonzero((g <= g[end]) & (g >= g[end] - B.G_BASELINE))
        slope = B._line_slope(g[sel], beta[sel], sig[sel] ** 2)
        dg = g[-1] - g[end]
        noise = np.sqrt(sig[-1] ** 2 + sig[end] ** 2)
        band = k >= B.K_MIN_BNL

        def rms(pred):
            chi = ((pred - beta[-1]) / noise)[band]
            return np.array([np.sqrt(np.mean(chi[:, b, b] ** 2))
                             for b in range(4)])

        carried, held = rms(beta[end] + slope * dg), rms(beta[end])
        assert np.round(carried, 1).tolist() == [1.4, 0.6, 0.8, 1.8]
        assert np.round(held, 1).tolist() == [4.6, 1.8, 3.2, 3.2]

        nu = np.asarray(tab["nu"])
        shifted = nu[end] + nu[end, 0] * (g[end] / g[-1] - 1.0)
        miss = shifted - nu[-1]
        assert np.max(np.abs(miss)) < 0.071 and np.min(np.abs(miss)) < 0.005


class TestNuMaxTrust:

    def test_off_by_default(self, tab):
        assert B.table_at(jnp.array([0.3]), snap=85).nu.size == 8

    def test_truncates_the_noise_dominated_corner(self):
        t = B.table_at(jnp.array([0.3]), snap=85, nu_max_trust=0.5)
        assert 2 <= t.nu.size < 8
        assert t.beta.shape == (1, t.nu.size, t.nu.size)

    def test_truncation_holds_rather_than_zeroes(self):
        """Dropping bins must not reintroduce the edge the rules remove."""
        full = B.table_at(jnp.array([0.3]), snap=85)
        trim = B.table_at(jnp.array([0.3]), snap=85, nu_max_trust=0.5)
        phi = np.asarray(B.project_weights(full.nu[-1:], trim.nu))
        assert phi[0, -1] == pytest.approx(1.0)

    def test_too_aggressive_is_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            B.table_at(jnp.array([0.3]), snap=85, nu_max_trust=1e-6)

    def test_errors_grow_towards_high_nu(self, tab):
        """The reason the knob exists."""
        diag = np.einsum("skbb->skb", np.asarray(tab["beta"]))
        sig = np.einsum("skbb->skb", np.asarray(tab["sigma_beta"]))
        med = np.median(sig / np.maximum(np.abs(1.0 + diag), 1e-30), axis=(0, 1))
        assert med[0] < 0.05
        assert med[-1] > 10.0 * med[0]


# ---------------------------------------------------------------------------
# The correction integrals
# ---------------------------------------------------------------------------

class TestCorrections:

    @pytest.fixture(scope="class")
    def setup(self):
        k = jnp.logspace(-2, 0, 40)
        nu = jnp.linspace(0.9, 3.5, 60)
        t = B.table_at(k, snap=85)
        return k, nu, t, jnp.ones(60) / 60.0, jnp.ones((40, 60))

    def test_gg_shape(self, setup):
        k, nu, t, w, uk = setup
        assert B.correction_2h_gg(nu, w, uk, t).shape == (40,)

    def test_gm_shape(self, setup):
        k, nu, t, w, uk = setup
        assert B.correction_2h_gm(nu, w, w, uk, uk, t).shape == (40,)

    def test_gm_reduces_to_gg_for_one_tracer(self, setup):
        k, nu, t, w, uk = setup
        np.testing.assert_allclose(
            np.asarray(B.correction_2h_gm(nu, w, w, uk, uk, t)),
            np.asarray(B.correction_2h_gg(nu, w, uk, t)), rtol=1e-12)

    def test_vanishes_below_the_taper(self):
        k = jnp.array([1e-5, 1e-3])
        nu = jnp.linspace(1.0, 3.0, 40)
        t = B.table_at(k, snap=85)
        d = B.correction_2h_gg(nu, jnp.ones(40) / 40, jnp.ones((2, 40)), t)
        np.testing.assert_allclose(np.asarray(d), 0.0, atol=1e-12)

    def test_matches_the_explicit_double_sum(self, setup):
        """The coarse-grid projection is exact, not an approximation.

        The nu interpolation factorises over the two axes, so contracting the
        8x8 table with projected weights equals the full (NM, NM) double sum.
        """
        k, nu, t, w, uk = setup
        beta = np.asarray(B.beta_nl(k, nu, nu, table=t))       # (Nk, NM, NM)
        w_eff = np.asarray(w)[None, :] * np.asarray(uk)
        explicit = np.einsum("ki,kij,kj->k", w_eff, beta, w_eff)
        np.testing.assert_allclose(np.asarray(B.correction_2h_gg(nu, w, uk, t)),
                                   explicit, rtol=1e-10)


# ---------------------------------------------------------------------------
# The layer's differentiability contract
# ---------------------------------------------------------------------------

class TestDifferentiable:
    """beta^NL adds no exception to L2: it jits and it differentiates.

    The amplitude dependence is real -- a more clustered cosmology sits at a
    different point on the MultiDark sequence -- so a zero gradient here would
    mean the dependence had been dropped, not that there is none.  The target
    is scaled to sit *inside* the sequence, so the test does not lean on the
    extrapolation rule; :class:`TestExtrapolationInGrowth` differentiates past
    the end.
    """

    @staticmethod
    def _f(analytic, k, k_out, scale):
        def f(lnA):
            c = PLANCK18.replace(ln10A_s=lnA)
            t = B.table_at(k_out, k, scale * analytic.pk_cb(k, 0.0, c), check=False)
            return jnp.sum(t.beta[:, 2, 2])
        return f

    def test_jit_equals_eager(self, analytic):
        k = jnp.logspace(-4, 2, 512)
        f = self._f(analytic, k, jnp.logspace(-1.2, -0.2, 12), 0.35)
        assert float(jax.jit(f)(3.044)) == pytest.approx(float(f(3.044)),
                                                         rel=1e-9)

    @pytest.mark.x64
    def test_gradient_matches_finite_differences(self, analytic):
        """In ln10A_s, the one direction in which ``s`` cannot move.

        The AW10 cost is invariant under sigma_tgt -> lambda sigma_tgt (the
        fitted amplitude absorbs it), so d ln s / d ln10A_s = 0 exactly and this
        test could never have seen the defect of 0.9.4 -- the grid argmin
        moving under n_s, Omega_m, h and Omega_b, with ``s``'s slope jumping at
        every switch.  :class:`TestSmoothInCosmology` covers those.  The step is
        2e-4 since the snapshot blend became a cubic (0.9.7): its curvature in
        ``g`` is real, and 2e-3 was reading it as a difference.
        """
        k = jnp.logspace(-4, 2, 512)
        f = self._f(analytic, k, jnp.logspace(-1.2, -0.2, 12), 0.35)
        g = float(jax.grad(f)(3.044))
        h = 2e-4
        fd = (float(f(3.044 + h)) - float(f(3.044 - h))) / (2 * h)
        assert abs(g) > 1e-3, "the amplitude dependence has been dropped"
        assert g == pytest.approx(fd, rel=1e-5)

    def test_corrections_differentiate(self, analytic):
        k = jnp.logspace(-2, 0, 20)
        nu = jnp.linspace(1.0, 3.0, 30)
        uk = jnp.ones((20, 30))

        def f(amp):
            t = B.table_at(k, snap=85)
            return jnp.sum(B.correction_2h_gg(nu, amp * jnp.ones(30) / 30, uk, t))

        assert np.isfinite(float(jax.grad(f)(1.0)))
        assert float(jax.grad(f)(1.0)) != 0.0


# ---------------------------------------------------------------------------
# Smooth in the cosmology (0.9.7)
# ---------------------------------------------------------------------------

def _legacy(monkeypatch):
    """0.9.4's matcher and readings: the parabolic vertex, linear in k and g."""
    monkeypatch.setattr(B, "N_NEWTON", 0)
    monkeypatch.setattr(B, "K_INTERP", "linear")
    monkeypatch.setattr(B, "G_INTERP", "linear")
    monkeypatch.setattr(B, "NU_INTERP", "linear")


@pytest.mark.x64
class TestSmoothInCosmology:
    r"""The rescaling is C^1 in every cosmological parameter.

    Until 0.9.7 ``s`` was the vertex of a parabola through three costs on the
    grid around its argmin.  The index carries no gradient, so ``s`` was
    differentiable -- but each time the argmin moved one cell the vertex was
    taken through a different triple and its slope jumped, by 3-4 per cent;
    ``g``, re-evaluated at ``s``, jumped with it.  At PLANCK18, z = 0.13 the
    fiducial sat 2.9e-4 in n_s, 2.0e-4 in Omega_m, 5.4e-4 in h and -1.5e-4 in
    Omega_b from such a switch, so every central difference wider than that
    straddled one: autodiff and differences disagreed by up to a factor of two
    on the lensing rows (ggah_sens_study, Study 1).  The table was also read
    linearly at ``s * k``, so ``d ln P / d theta`` stepped wherever that crossed
    a tabulated wavenumber.  Each test finds the legacy code's own switch, so
    it does not depend on where the fiducial happens to sit.
    """

    K_OUT = jnp.asarray([0.11, 0.14, 0.2, 0.3])
    #: Both metrics are relative to the largest derivative, not per element:
    #: an element whose derivative crosses zero turns curvature into a large
    #: ratio.  Slope jump 1e-7 either side of the switch: 3.4e-6 (n_s) and
    #: 6.3e-6 (Omega_m) against 5.3e-2 and 2.8e-2 for the legacy code.  AD
    #: against a central difference of 2e-4 straddling it: 7.0e-6 and 1.6e-4
    #: against 1.9e-2 and 1.0e-2 -- which the legacy code keeps at 1e-3 too,
    #: the signature of a kink rather than of curvature.
    SLOPE_BOUND, SLOPE_LEGACY = 1e-4, 1e-2
    FD_STEP, FD_BOUND, FD_LEGACY = 2e-4, 1e-3, 5e-3

    @staticmethod
    def _target(analytic, name):
        k = jnp.logspace(-4.0, 2.0, 512)

        def ln_sig(v):
            c = PLANCK18.replace(**{name: v})
            nodes = jnp.linspace(jnp.log(B.R_RESCALE[0]), jnp.log(B.R_RESCALE[1]),
                                 B.N_R_COST)
            from ggah_mod.cosmology.amplitude import sigma2_tophat
            pk_cb = 0.35 * analytic.pk_cb(k, 0.0, c)
            return nodes, 0.5 * jnp.log(sigma2_tophat(pk_cb, k, jnp.exp(nodes))), pk_cb
        return k, ln_sig

    def _legacy_switch(self, analytic, name, tab):
        """The parameter value where the legacy grid argmin moves one cell."""
        _, ln_sig = self._target(analytic, name)
        s_grid = jnp.linspace(B.S_RANGE[0], B.S_RANGE[1], B.N_S)

        @jax.jit
        def j_of(v):
            nodes, ls, _ = ln_sig(v)
            # 0.9.4's cost: ln sigma_MD linear in ln R
            i, t = B.lin_weights(nodes[None, :] - jnp.log(s_grid)[:, None], tab["ln_r"])
            y = tab["ln_sigma_md"]
            u = jnp.exp((1.0 - t) * y[i] + t * y[i + 1] - ls[None, :])
            mu, mu2 = jnp.mean(u, axis=1), jnp.mean(u ** 2, axis=1)
            return jnp.argmin(1.0 - mu ** 2 / mu2)

        v0 = float(getattr(PLANCK18, name))
        j0 = int(j_of(v0))
        for d in np.arange(1e-4, 3e-2, 1e-4):
            for sgn in (1.0, -1.0):
                if int(j_of(v0 + sgn * d)) != j0:
                    lo, hi = v0 + sgn * (d - 1e-4), v0 + sgn * d
                    for _ in range(45):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if int(j_of(mid)) == j0 else (lo, mid)
                    return 0.5 * (lo + hi)
        raise AssertionError(f"no grid switch within 3e-2 of the fiducial {name}")

    def _f(self, analytic, name):
        k, ln_sig = self._target(analytic, name)

        def f(v):
            t = B.table_at(self.K_OUT, k, ln_sig(v)[2], check=False)
            return jnp.concatenate([jnp.atleast_1d(t.s), jnp.atleast_1d(t.g),
                                    t.beta[:, 0, 0], t.beta[:, 1, 1]])
        return f

    def test_s_is_the_cost_minimum_across_grid_cells(self, tab, monkeypatch):
        """On a target built from the table, ``s = lambda`` exactly; ds/dlambda = 1."""
        lams = np.linspace(0.85, 0.87, 21)          # two cells of the s grid

        def solve(lam):
            nodes, ln_sig = _ln_sigma_tgt_from_table(tab, lam, 1.0)
            return B._solve_s(nodes, ln_sig, tab)

        s = np.array([float(jax.jit(solve)(x)) for x in lams])
        d = np.array([float(jax.jit(jax.grad(solve))(x)) for x in lams])
        assert np.max(np.abs(s - lams)) < 1e-12        # measured 2.4e-14
        assert np.max(np.abs(d - 1.0)) < 1e-6          # measured 6.6e-8
        _legacy(monkeypatch)
        d_old = np.array([float(jax.jit(jax.grad(solve))(x)) for x in lams])
        assert np.max(np.abs(d_old - 1.0)) > 1e-2      # measured 2.4e-2

    @pytest.mark.parametrize("name", ["n_s", "Omega_m"])
    def test_one_sided_slopes_agree_at_the_legacy_switch(self, analytic, tab, name,
                                                         monkeypatch):
        sw = self._legacy_switch(analytic, name, tab)
        def jump():
            df = jax.jit(jax.jacfwd(self._f(analytic, name)))
            L, R = np.asarray(df(sw - 1e-7)), np.asarray(df(sw + 1e-7))
            return np.max(np.abs(R - L)) / np.max(np.abs(L)), abs(R[0] / L[0] - 1.0)

        assert jump()[0] < self.SLOPE_BOUND
        _legacy(monkeypatch)
        assert jump()[1] > self.SLOPE_LEGACY, "the legacy s had no kink"

    @pytest.mark.parametrize("name", ["n_s", "Omega_m"])
    def test_autodiff_matches_a_difference_across_the_switch(self, analytic, tab, name,
                                                             monkeypatch):
        sw = self._legacy_switch(analytic, name, tab)
        h = self.FD_STEP
        th = sw + 0.3 * h                           # the difference straddles sw

        def rel():
            f = self._f(analytic, name)
            ad = np.asarray(jax.jit(jax.jacfwd(f))(th))
            fd = (np.asarray(f(th + h)) - np.asarray(f(th - h))) / (2 * h)
            return np.max(np.abs(fd - ad)) / np.max(np.abs(ad))

        assert rel() < self.FD_BOUND
        _legacy(monkeypatch)
        assert rel() > self.FD_LEGACY


class TestReadingIsC1:
    """The table is read C^1 in ln k and in g, and exactly at its nodes."""

    def test_in_ln_k(self, tab):
        grid = tab["beta"][-1]                       # snapshot 85, all bins
        lk = tab["ln_k"]
        cub = lambda x: B._interp_k(grid, jnp.exp(jnp.atleast_1d(x)), lk, "cubic")[0]
        lin = lambda x: B._interp_k(grid, jnp.exp(jnp.atleast_1d(x)), lk, "linear")[0]
        d = jax.jacfwd(cub)
        scale = max(float(jnp.max(jnp.abs(d(x)))) for x in np.asarray(lk[2:-1]))
        above_taper = np.asarray(lk) > np.log(B.K_MIN_BNL * B.K_TAPER)
        for x in np.asarray(lk)[above_taper][1:-1]:
            np.testing.assert_allclose(cub(x), lin(x), rtol=1e-12, atol=1e-15)
            jump = float(jnp.max(jnp.abs(d(x + 1e-8) - d(x - 1e-8))))
            assert jump < 1e-5 * scale, f"slope jumps at ln k = {x:.3f}"
        # the clamp at the top node is C^1: zero slope on both sides
        top = float(lk[-1])
        assert float(jnp.max(jnp.abs(d(top - 1e-8)))) < 1e-6 * scale
        assert float(jnp.max(jnp.abs(d(top + 1e-3)))) == 0.0

    def test_in_g_and_into_the_extrapolation(self, tab):
        g_md = tab["g_md"]
        nb = tab["nu"].shape[1]

        def F(g):
            i0, w = B.lin_weights(g, g_md)
            nu, grid = B._blend_in_g(tab, i0, i0 + 1, w, nb, "cubic")
            grid, nu = B._extrapolate(grid, nu, g, nb)
            return jnp.concatenate([nu, grid.ravel()])

        dF = jax.jacfwd(F)
        scale = float(jnp.max(jnp.abs(dF(g_md[len(g_md) // 2]))))
        for g in list(np.asarray(g_md)[1:-1]) + [float(g_md[0]), float(g_md[-1])]:
            jump = float(jnp.max(jnp.abs(dF(g + 1e-9) - dF(g - 1e-9))))
            assert jump < 1e-4 * scale, f"slope jumps at g = {g:.4f}"
        # peak heights stay ordered, so the nu projection stays well defined
        for g in np.linspace(float(g_md[0]) - 0.2, float(g_md[-1]) + 0.2, 400):
            assert np.all(np.diff(np.asarray(F(g)[:nb])) > 0)

    def test_in_nu(self, tab):
        """The projection weights: C^1 across every bin, a partition of unity,
        the linear extrapolation below the first bin and the hold above the last."""
        nu_ref = tab["nu"][-1]
        P = lambda x: B.project_weights(jnp.atleast_1d(x), nu_ref, "cubic")[0]
        dP = jax.jacfwd(P)
        scale = float(jnp.max(jnp.abs(dP(0.5 * (nu_ref[3] + nu_ref[4])))))
        for x in np.asarray(nu_ref):
            assert float(jnp.max(jnp.abs(dP(x + 1e-9) - dP(x - 1e-9)))) < 1e-5 * scale
        grid = np.linspace(float(nu_ref[0]) - 0.5, float(nu_ref[-1]) + 0.5, 301)
        W = np.asarray(B.project_weights(jnp.asarray(grid), nu_ref, "cubic"))
        np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-12)
        # nodes are reproduced exactly, as the hat functions do
        np.testing.assert_allclose(np.asarray(B.project_weights(nu_ref, nu_ref, "cubic")),
                                   np.eye(nu_ref.size), atol=1e-12)
        above = grid > float(nu_ref[-1])
        np.testing.assert_allclose(W[above], np.eye(nu_ref.size)[-1][None, :].repeat(
            above.sum(), 0), atol=1e-15)
        below = grid < float(nu_ref[0])
        np.testing.assert_allclose(W[below], np.asarray(
            B.project_weights(jnp.asarray(grid[below]), nu_ref, "linear")), atol=1e-14)

    def test_a_pinned_snapshot_is_bit_exact(self, tab):
        for snap in (36, 60, 85):
            a = B.table_at(jnp.logspace(-1, -0.2, 7), snap=snap, interp="cubic")
            b = B.table_at(jnp.logspace(-1, -0.2, 7), snap=snap, interp="linear")
            np.testing.assert_array_equal(np.asarray(a.nu), np.asarray(b.nu))


@pytest.mark.slow
@pytest.mark.x64
@pytest.mark.parametrize("z", [0.13, 0.35])
def test_what_the_smooth_reading_moves(z, monkeypatch):
    r"""How far 0.9.7's C^1 readings move the model, against 0.9.4's.

    The nodes are reproduced exactly; between them the cubic reads the noisy
    table differently from a straight line (its beta^NL nodes at k = 0.09-0.19
    zig-zag by about their own error bar).  Measured at PLANCK18 with zheng07's
    defaults: ``s`` by 4.4e-5, P_gg by up to 1.14e-2 (at k = 0.14 h/Mpc, z =
    0.13; 1.11e-2 at 0.35), w_p by 1.5e-3 / 1.1e-3 and Delta Sigma by 1.6e-3 /
    2.2e-3 over 0.1-30 Mpc/h.  A model change at the table's own noise level,
    made so the cosmology derivatives are C^1.
    """
    import ggah_mod.spectra as SP
    from ggah_mod.backend import ACCURATE
    from ggah_mod.halos.field import make_field
    from ggah_mod.observables.real_space import delta_sigma, wp
    from ggah_mod.sectors import BaryonSplit, GalaxySector, MatterField, galaxy_defaults
    from ggah_mod.spectra.bnl import table_for

    f = make_field(PLANCK18, ACCURATE, make_pk("emu_pk"), z=z, calibration="off")
    g = GalaxySector("zheng07", shmr="zu15", backend=ACCURATE,
                     calibration="off").weights(f, galaxy_defaults("zheng07"))
    m = MatterField().weights(f, {"split": BaryonSplit.from_hot(0.0, jnp.zeros(f.n_m))})
    R = np.logspace(-1, 1.5, 30)

    def run():
        opts = SP.PkOptions()
        tab = table_for(f, opts)
        gg = SP.pk_cross(f, g, g, overlap="identical", options=opts, bnl_table=tab)
        gm = SP.pk_cross(f, g, m, overlap="none", options=opts, bnl_table=tab)
        return (float(tab.s), np.asarray(gg.total),
                np.asarray(wp(R, (f.k, gg.total), pi_max=60.0, backend=ACCURATE)),
                np.asarray(delta_sigma(R, (f.k, gm.total), PLANCK18, backend=ACCURATE)))

    new = run()
    _legacy(monkeypatch)
    old = run()
    k = np.asarray(f.k)
    sel = (k > 0.05) & (k < 5.0)
    assert abs(new[0] - old[0]) < 1e-4
    assert np.max(np.abs(new[1] / old[1] - 1)[sel]) < 1.5e-2
    assert np.max(np.abs(new[2] / old[2] - 1)) < 2e-3
    assert np.max(np.abs(new[3] / old[3] - 1)) < 3e-3

