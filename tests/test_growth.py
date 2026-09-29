"""One growth route, and the ambiguity it reports rather than hides."""
import numpy as np
import pytest

from ggah_mod.cosmology import Cosmology
from ggah_mod.cosmology.power import make_pk
from ggah_mod.cosmology.growth import (
    growth_factor, growth_scale_spread, SPREAD_K_RANGE,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def klass():
    return make_pk("class")


class TestGrowthFactor:
    def test_unity_today(self, klass, cosmo):
        assert growth_factor(0.0, cosmo, klass) == pytest.approx(1.0, abs=1e-12)

    def test_decreasing(self, klass, cosmo):
        d = growth_factor(np.array([0.0, 0.5, 1.0, 2.0, 3.0]), cosmo, klass)
        assert np.all(np.diff(d) < 0)

    def test_matter_domination_limit(self, klass, cosmo):
        """D ~ 1/(1+z) deep in matter domination; at z = 3 it should be close."""
        d = growth_factor(3.0, cosmo, klass)
        assert 0.9 < d * 4.0 / 1.0 < 1.4

    def test_vector_matches_scalar(self, klass, cosmo):
        """Exactly equal, and this is the test that found the reason it was not.

        `growth_factor` batches now -- one backend call for the array rather
        than one per entry -- so the two sides of this comparison come from two
        *different* solves, `(0.5, 2.0)` and `(0.5,)`.  That made it fail at
        4.8e-8, because `ClassPk` set `z_max_pk = max(z.max(), 1.0)` and
        `z_max_pk` is a solver setting that changes CLASS's output sampling: the
        spectrum depended on which redshifts were requested *together*.

        `z_max_pk` is now the constant `power.Z_MAX_PK`, so it does not, and
        this is back to a tolerance six orders of magnitude tighter than the
        1e-6 the defect forced.

        Not *exact*, and the reason is worth stating so the next person does not
        tighten it and wonder: the two P(k) rows underneath are bit-identical --
        `tests/test_power.py::TestTheSpectrumDoesNotDependOnTheRequest` asserts
        exactly that, and it is the property this test is downstream of -- but
        the reduction on top of them is not.  `sigma_tophat` runs over a stack of
        one row here and two there, and XLA is entitled to order the trapezoid
        differently.  Measured: **1.1e-16**, one ulp of float64.
        """
        v = growth_factor(np.array([0.5, 2.0]), cosmo, klass)
        assert v[0] == pytest.approx(growth_factor(0.5, cosmo, klass), rel=1e-14)

    def test_cold_and_total_differ_when_massive(self, klass, cosmo):
        t = growth_factor(1.0, cosmo, klass, "total")
        c = growth_factor(1.0, cosmo, klass, "cold")
        assert t != c

    def test_cold_equals_total_when_massless(self, klass, massless):
        t = growth_factor(1.0, massless, klass, "total")
        c = growth_factor(1.0, massless, klass, "cold")
        assert t == pytest.approx(c, rel=1e-14)

    def test_bad_variant_refused(self, klass, cosmo):
        with pytest.raises(ValueError, match="variant"):
            growth_factor(1.0, cosmo, klass, "warm")


class TestNoSilentFallback:
    """The predecessor let a backend with no redshift dependence fall back to a
    fitting formula, with nothing in the output recording which had run."""

    def test_backend_without_native_z_is_refused(self, cosmo):
        class Stub:
            has_native_z = False
            differentiable = True
            name = "stub"
            def pk(self, k, z, c): return np.ones_like(np.atleast_1d(k))
            def pk_cb(self, k, z, c): return np.ones_like(np.atleast_1d(k))

        with pytest.raises(ValueError, match="has_native_z"):
            growth_factor(1.0, cosmo, Stub())

    def test_no_mode_argument_exists(self):
        """Three selectable routes was the design that hid which one ran."""
        import inspect
        assert "mode" not in inspect.signature(growth_factor).parameters


class TestScaleDependenceIsReported:
    def test_massless_spread_is_numerically_zero(self, klass, massless):
        """Without mass, growth really is scale-independent; what is left is
        the solver's own floor."""
        assert growth_scale_spread(1.0, massless, klass) < 3e-4

    def test_massive_spread_is_well_above_the_floor(self, klass, cosmo):
        floor = growth_scale_spread(1.0, Cosmology.create(sum_mnu=0.0), klass)
        assert growth_scale_spread(1.0, cosmo, klass) > 5.0 * floor

    def test_spread_grows_with_mass(self, klass):
        s = [growth_scale_spread(1.0, Cosmology.create(sum_mnu=m), klass)
             for m in (0.0, 0.06, 0.30)]
        assert s[0] < s[1] < s[2]

    def test_spread_grows_with_redshift(self, klass, cosmo):
        assert (growth_scale_spread(0.5, cosmo, klass)
                < growth_scale_spread(3.0, cosmo, klass))

    def test_window_excludes_the_solver_edges(self):
        """Outside this window the ratio is dominated by sampling artefacts,
        not neutrinos -- measured at 1.5e-3 above k = 10 with zero mass."""
        assert SPREAD_K_RANGE == (1e-3, 10.0)
