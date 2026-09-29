r"""Verification: mis-centering, and the branch it replaces.

The headline is closed-form and needs no reference: :math:`h(k\to0) = 1`
**identically**, for every :math:`p_{\rm off}` and every offset scale, because
the two terms sum to :math:`(1-p) + p`.  That is what makes this a modifier
rather than a sector -- a mis-centred tracer has the same mean density and the
same linear bias as a centred one, and only its k dependence differs.

The second is structural.  The predecessor selected between two published
conventions by ``dict.get``: ``if p_off > 0 and R_off > 0`` took More et al.,
``elif p_off > 0 and sigma_off > 0`` took Johnston et al., and setting both
silently took the first.  Here the convention is static and the parameters are
separate, so :class:`TestTheConventionIsNamedNotInferred` can assert the two
give *different* answers from the same parameter set -- which is the property a
dict-key branch destroys.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import galaxy_defaults
from ggah_mod.sectors.galaxies import GalaxySector
from ggah_mod.sectors.miscentering import (
    MISCENTERING_CONVENTIONS, MisCentering, MisCenteringParams,
)
from ggah_mod.sectors.protocol import ProfileModifier

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0)


@pytest.fixture(scope="module")
def galaxies():
    return GalaxySector("zheng07", backend=DIFFERENTIABLE)


class TestTheLargeScaleLimitIsExact:
    """`h(k -> 0) = 1`, algebraically, not asymptotically."""

    @pytest.mark.parametrize("convention", MISCENTERING_CONVENTIONS)
    @pytest.mark.parametrize("p_off", [0.0, 0.3, 1.0])
    def test_it_holds_for_every_offset_fraction(self, convention, p_off, field):
        h = MisCentering(convention).h_k(field, MisCenteringParams(p_off=p_off))
        assert h.shape == (field.n_k, field.n_m)
        assert np.allclose(np.asarray(h[0]), 1.0, atol=1e-8)

    def test_it_only_ever_suppresses(self, field):
        """A displaced central moves power to larger scales; it makes none."""
        h = MisCentering("more15").h_k(field, MisCenteringParams(p_off=0.4))
        assert np.all(np.asarray(h) <= 1.0 + 1e-12)
        assert np.all(np.asarray(h) >= 0.0)

    def test_no_offset_is_the_identity(self, field):
        h = MisCentering("more15").h_k(field, MisCenteringParams(p_off=0.0))
        assert np.allclose(np.asarray(h), 1.0, rtol=0, atol=0)


class TestTheConventionIsNamedNotInferred:
    """Two published models, one static field, no dict-key branch."""

    def test_it_satisfies_the_protocol(self):
        assert isinstance(MisCentering("more15"), ProfileModifier)

    def test_an_unknown_convention_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown mis-centering convention"):
            MisCentering("johnston2007")

    def test_the_two_conventions_disagree_on_the_same_parameters(self, field):
        """They differ by a factor of c(M).  A branch chosen by which key
        happened to be set could not have told them apart."""
        p = MisCenteringParams(p_off=0.5)
        a = MisCentering("more15").h_k(field, p)
        b = MisCentering("johnston07").h_k(field, p)
        assert not np.allclose(np.asarray(a), np.asarray(b), rtol=1e-3)

    def test_the_offset_scale_follows_the_halo_only_in_more15(self, field):
        p = MisCenteringParams(r_off=0.25, sigma_off=0.4)
        rel = np.asarray(MisCentering("more15").offset_scale(field, p))
        fixed = np.asarray(MisCentering("johnston07").offset_scale(field, p))
        assert rel.max() / rel.min() > 10.0        # scales with r_s
        assert np.allclose(fixed, fixed[0])        # one length for every halo


class TestItModifiesWithoutCreating:
    """It displaces centrals; it does not add or remove any."""

    def test_the_normalisation_is_untouched(self, field, galaxies):
        w = galaxies.weights(field, galaxy_defaults("zheng07"))
        got = MisCentering("more15").apply(w, field, MisCenteringParams(p_off=0.3))
        assert float(got.norm) == float(w.norm)

    def test_the_large_scale_weight_is_untouched(self, field, galaxies):
        w = galaxies.weights(field, galaxy_defaults("zheng07"))
        got = MisCentering("more15").apply(w, field, MisCenteringParams(p_off=0.3))
        assert np.allclose(np.asarray(got.at_large_scales()),
                           np.asarray(w.at_large_scales()), rtol=1e-10)

    def test_the_small_scale_weight_is_suppressed(self, field, galaxies):
        w = galaxies.weights(field, galaxy_defaults("zheng07"))
        got = MisCentering("more15").apply(w, field, MisCenteringParams(p_off=0.5))
        assert np.all(np.asarray(got.total()[-1]) < np.asarray(w.total()[-1]))

    def test_a_tracer_with_no_centre_is_refused(self, field, galaxies):
        """A satellite-only tracer has no object at the halo centre, so
        applying this would silently do nothing."""
        w = galaxies.weights(field, galaxy_defaults("zheng07"), view="sat")
        with pytest.raises(ValueError, match="no point component"):
            MisCentering("more15").apply(w, field, MisCenteringParams(p_off=0.3))


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("name,x", [("p_off", 0.3), ("r_off", 0.25),
                                        ("sigma_off", 0.4)])
    def test_the_gradient_reaches_every_parameter(self, name, x, field):
        convention = "johnston07" if name == "sigma_off" else "more15"
        base = dict(p_off=0.3, r_off=0.25, sigma_off=0.4)

        def f(value):
            p = MisCenteringParams(**{**base, name: value})
            return jnp.log(jnp.sum(MisCentering(convention).h_k(field, p)))

        h = 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_the_static_convention_is_not_a_leaf(self):
        """A string among the pytree leaves would fail at the first jit
        boundary rather than at construction."""
        leaves = jax.tree_util.tree_leaves(MisCenteringParams())
        assert all(not isinstance(x, str) for x in leaves)
        assert len(leaves) == 3

    def test_jit_is_value_identical(self, field):
        p = MisCenteringParams(p_off=0.3)
        mc = MisCentering("more15")
        eager = np.asarray(mc.h_k(field, p))
        jitted = np.asarray(jax.jit(lambda q: mc.h_k(field, q))(p))
        assert np.allclose(eager, jitted, rtol=1e-12)
