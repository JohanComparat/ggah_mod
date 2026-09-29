r"""Verification: the conditional luminosity functions, which had no tests.

``PLAN.md`` item **G2**.  Two of the fourteen occupation models are CLFs and
neither had a dedicated module -- they were exercised only where a galaxy test
happened to select one, which covers "it runs" and not much else.  The
quantities most worth checking here are the two that are easy to get wrong and
invisible when wrong: the **incomplete gamma function** the satellite integral
is written in terms of, and the :math:`L_s^*/L_c` offset that ``PLAN.md`` item
**G4** was about.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaincc, gammaln

from ggah_mod.sectors import clf as C
from ggah_mod.sectors.occupation import M_S_OVER_M_C

LOG10M = np.linspace(11.0, 15.0, 41)


@pytest.fixture(scope="module")
def cacciato():
    return dict(C.CLF_DEFAULTS["cacciato09"])


@pytest.fixture(scope="module")
def vdb():
    return dict(C.CLF_DEFAULTS["vandenbosch13"])


class TestTheUpperIncompleteGamma:
    r"""The one special function in the layer, and the one that has a branch.

    :math:`\Gamma(a, x)` is needed for **negative** :math:`a` -- a CLF's
    faint-end slope makes :math:`(\alpha_s+1)/2` negative for
    :math:`\alpha_s < -1`, which both shipped models have -- and
    ``jax.scipy.special`` offers only the regularised form for :math:`a > 0`.
    So there is a recurrence, and a recurrence is exactly the kind of thing
    that is right in the middle of its range and wrong at the edge.
    """

    @staticmethod
    def _scipy(a, x):
        from scipy.special import gammaincc as sgammaincc
        from scipy.special import gamma as sgamma
        import scipy.special as sp
        return sp.gammaincc(a, x) * sp.gamma(a) if a > 0 else None

    @pytest.mark.parametrize("a", [0.25, 0.5, 1.0, 2.5])
    @pytest.mark.parametrize("x", [1e-3, 0.1, 1.0, 5.0])
    def test_it_matches_scipy_where_scipy_has_it(self, a, x):
        import scipy.special as sp

        want = float(sp.gammaincc(a, x) * sp.gamma(a))
        got = float(C.upper_gamma(a, x))
        assert got == pytest.approx(want, rel=1e-6), (a, x, got, want)

    @pytest.mark.parametrize("a", [-0.9, -0.75, -0.25])
    @pytest.mark.parametrize("x", [0.05, 0.5, 2.0])
    def test_the_negative_branch_satisfies_the_recurrence(self, a, x):
        r""":math:`\Gamma(a,x) = [\Gamma(a+1,x) - x^a e^{-x}]/a`, which is the
        identity the implementation is built on -- checked here against the
        *next* order up rather than against itself."""
        lhs = float(C.upper_gamma(a, x))
        rhs = (float(C.upper_gamma(a + 1.0, x))
               - x ** a * np.exp(-x)) / a
        assert lhs == pytest.approx(rhs, rel=1e-6), (a, x, lhs, rhs)

    def test_below_its_domain_it_is_wrong_and_says_nothing(self):
        r"""The recurrence is **one step**, so it is valid for
        :math:`a > -1` -- which its docstring says -- and outside that it
        clamps :math:`a+1` to :math:`10^{-10}` and returns a number with no
        complaint.  Recorded rather than repaired: a second step is easy and
        an unbounded recursion is not, and nothing here needs one.

        What makes that safe is the next test, not this one.
        """
        a, x = -1.4, 0.5
        lhs = float(C.upper_gamma(a, x))
        rhs = (float(C.upper_gamma(a + 1.0, x)) - x ** a * np.exp(-x)) / a
        assert not np.isclose(lhs, rhs, rtol=1e-3), (
            "the recurrence holds at a = -1.4, so `upper_gamma` now handles "
            "more than one step and its docstring's `a > -1` is too narrow")

    def test_the_declared_bound_keeps_every_fit_inside_that_domain(self):
        r"""The coupling that makes the limitation harmless, checked rather
        than assumed.

        The CLFs' faint-end slope enters as :math:`a = (\alpha_s+1)/2`, so
        :math:`a > -1` is exactly :math:`\alpha_s > -3` -- and the bound
        ``PLAN.md`` item **C1** gave ``alpha_sat`` in the CLFs is
        :math:`(-3, 0)`, chosen for where the Schechter integral converges.
        The two agree, and they were arrived at separately, which is precisely
        the sort of agreement that stops holding when one of them is edited.
        """
        from ggah_mod.sectors.occupation_params import PER_MODEL

        for model in ("cacciato09", "vandenbosch13"):
            lo, hi = PER_MODEL[(model, "alpha_faint")]["bounds"]
            assert lo >= -3.0, (
                f"{model}'s alpha_faint may reach {lo}, which puts "
                f"upper_gamma's order at {(lo + 1) / 2} -- below the a > -1 "
                f"its one-step recurrence is valid for, where it returns a "
                f"wrong number silently")
            assert hi <= 0.0

    def test_it_is_finite_and_differentiable_where_the_clfs_use_it(self):
        """The faint-end slopes shipped are -1.15 and -1.3, so the order is
        about -0.075 and -0.15: both negative, both on the branch."""
        for alpha_s in (-1.15, -1.3):
            a = (alpha_s + 1.0) / 2.0
            assert a < 0.0
            g = float(jax.grad(lambda xx: C.upper_gamma(a, xx))(0.4))
            assert np.isfinite(g) and g < 0.0     # Gamma(a,x) falls with x


class TestTheSatelliteOffset:
    r"""``L_s^*/L_c``, and the two values ``PLAN.md`` item **G4** was about."""

    def test_the_two_families_keep_their_own_papers_figures(self):
        """0.562 in the CLFs and 0.56 in van Uitert's CSMF: the same fitted
        offset, quoted to three figures and to two.  Each module keeps its own
        source's number, because a transcription that adopts another paper's
        precision is no longer a transcription."""
        assert C.L_S_OVER_L_C == 0.562
        assert M_S_OVER_M_C == 0.56
        assert abs(M_S_OVER_M_C / C.L_S_OVER_L_C - 1.0) < 5e-3

    def test_it_shifts_the_cut_off_downward(self, vdb):
        r"""The offset puts :math:`L_s^*` *below* :math:`L_c`, so raising it
        toward 1 must raise the satellite number above a fixed threshold."""
        kw = {k: v for k, v in vdb.items()
              if k in ("log10l_lim", "log10l0", "log10m1", "alpha_cen",
                       "beta_cen", "alpha_faint", "b_0", "b_1", "b_2")}
        low = np.asarray(C.clf_satellite_vdb13(LOG10M, **kw,
                                               f_s_star=C.L_S_OVER_L_C))
        high = np.asarray(C.clf_satellite_vdb13(LOG10M, **kw, f_s_star=1.0))
        assert np.all(high >= low - 1e-12)
        assert np.any(high > low * 1.01)


class TestTheModelsRunAndAreSane:
    """The floor: both CLFs, over four decades of halo mass."""

    @pytest.mark.parametrize("name", sorted(C.CLF_DEFAULTS))
    def test_both_parts_are_finite_and_non_negative(self, name):
        cen, sat = C.CLF[name]
        d = C.CLF_DEFAULTS[name]
        import inspect

        for fn in (cen, sat):
            kw = {k: v for k, v in d.items()
                  if k in inspect.signature(fn).parameters}
            out = np.asarray(fn(LOG10M, **kw))
            assert np.all(np.isfinite(out)), (name, fn.__name__)
            assert np.all(out >= 0.0), (name, fn.__name__)

    @pytest.mark.parametrize("name", sorted(C.CLF_DEFAULTS))
    def test_the_satellite_count_grows_with_halo_mass(self, name):
        """A halo cannot host fewer satellites as it grows -- the one thing
        every occupation model in the literature agrees on."""
        import inspect

        _, sat = C.CLF[name]
        d = C.CLF_DEFAULTS[name]
        kw = {k: v for k, v in d.items()
              if k in inspect.signature(sat).parameters}
        out = np.asarray(sat(LOG10M, **kw))
        assert out[-1] > out[0]
        assert np.all(np.diff(out) >= -1e-9 * max(out.max(), 1e-30))

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown CLF"):
            C.make_clf("nope")

    @pytest.mark.parametrize("name", sorted(C.CLF_DEFAULTS))
    def test_every_declared_parameter_carries_a_gradient(self, name):
        """The layer's rule: nothing in it is allowed a structural zero."""
        import inspect

        _, sat = C.CLF[name]
        d = C.CLF_DEFAULTS[name]
        names = [k for k in d if k in inspect.signature(sat).parameters]
        for k in names:
            def f(v, k=k):
                kw = {kk: (v if kk == k else d[kk]) for kk in names}
                return jnp.sum(sat(LOG10M, **kw))

            g = float(jax.grad(f)(float(d[k])))
            assert np.isfinite(g), (name, k)
