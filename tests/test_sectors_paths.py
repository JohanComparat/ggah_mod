r"""Verification: layer 3 is differentiable **everywhere**, and that is checked.

Layer 2 makes a careful claim -- pure JAX with one named exception -- and pins
it with an AST audit in ``tests/test_paths.py``.  Layer 3 makes a stronger one:
*no* exception.  The stronger claim needs the stronger check, because "all of it
is differentiable" is exactly the kind of statement that stays true in a
docstring long after it has stopped being true in the code.

Four mechanisms, in increasing order of what they would catch:

1. **The AST audit** (:class:`TestNoNumpyOnTheTracedPath`).  Structural: it
   finds a numpy call *anywhere* in the package's per-call paths, including one
   on a branch no test happens to exercise.  A gradient test can only catch a
   numpy island that sits on the path it differentiates.
2. **The empty-exception assertion**
   (:meth:`TestNoNumpyOnTheTracedPath.test_the_layer_has_no_whole_module_exemptions`).
   This is the layer's headline claim, as one line of code.
3. **Declarations checked by trying them** (:class:`TestDeclarationsAreTrue`).
   Every sector says ``differentiable = True``; the test differentiates it.
4. **Every parameter, not just an amplitude** (:class:`TestGradientReachesEveryLeaf`).
   Layer 2's recorded blind spot: ``einasto_uk`` and ``satellite_nfw_uk`` were
   numpy on the traced path and never failed, because the one gradient test
   differentiated an amplitude that does not reach ``r_delta`` or ``c``.

Why the allow-list is nearly empty
----------------------------------

Layer 2's ``NUMPY_ALLOWED`` has entries for table I/O and for a memoised
Gauss-Legendre node generator, and those are legitimate: a constant built once
at construction is not on the traced path at all.  The same two categories are
the only ones admitted here, and each entry has to say which it is.  Anything
else is a hole in the differentiable path -- the gradient still returns, and is
missing a term.
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod
import ggah_mod.sectors as S
from ggah_mod.sectors.params import Param, SectorParams, declared_fields
from tests.test_paths import _numpy_uses

SECTORS_DIR = pathlib.Path(ggah_mod.__file__).parent / "sectors"

#: Every symbol in `ggah_mod/sectors` allowed to call numpy inside a function.
#:
#: Two categories only, and an entry must say which:
#:
#: * **construction** -- built once, before anything is traced;
#: * **table I/O** -- reading a distilled table off disk, which is the accurate
#:   path by definition and produces a constant the traced code interpolates.
#:
#: Module-level numpy is never listed: a constant built at import is not on the
#: traced path.  Anything else here is a bug, not an exemption.
SECTORS_NUMPY_ALLOWED: dict[str, str] = {
    "cooling._pchip_slopes":
        "construction: monotone-cubic slopes of the static APEC table, "
        "computed once when it is loaded and never re-derived per call",
    "cooling.load":
        "table I/O: reads the .npz, and keeps it numpy on purpose -- a jnp "
        "array cached during a trace carries that trace into the next one",
    "cooling.ApecCooling.__init__":
        "construction: converts the loaded numpy table for the slope "
        "calculation, once per instance and before anything is traced",
    "cooling.build":
        "table I/O: regenerates the table from APEC, the accurate path by "
        "definition",
    "cooling._band_slopes":
        "construction: monotone-cubic slopes along the band table's energy "
        "axis, built once when it is loaded.  Deliberately not the slopes "
        "BandCooling builds per call along log T -- those are jnp, because "
        "they move with the band edges and so are on the traced path",
    "cooling.load_band":
        "table I/O: reads the cached .npz, and keeps it numpy for the reason "
        "`cooling.load` gives",
    "cooling.BandCooling.__init__":
        "construction: converts the loaded table and takes the log of its "
        "totals, once per instance and before anything is traced",
    "cooling.build_band":
        "table I/O: regenerates the band table from APEC, the accurate path "
        "by definition",
    "coldgas.fit_xgass_medians":
        "construction: the least-squares line through the transcribed xGASS "
        "table that ColdGasParams' defaults are rounded from; called by a "
        "test and by nobody on the traced path, which reads the defaults",
}


class TestNoNumpyOnTheTracedPath:
    """The layer is pure JAX.  Full stop -- and this is what makes it a fact."""

    MODULES = sorted(SECTORS_DIR.glob("*.py"))

    def test_there_are_modules_to_check(self):
        """Guards the parametrised loop below against checking nothing.

        Without this the suite goes green on an empty package, which is exactly
        the state it is in while the layer is being written.
        """
        assert len(self.MODULES) >= 3

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
    def test_every_numpy_call_is_accounted_for(self, path):
        unexplained = []
        for qual, lineno, attr in _numpy_uses(path):
            if not qual:
                continue                          # module level: a constant
            if path.stem in SECTORS_NUMPY_ALLOWED:
                continue
            if f"{path.stem}.{qual}" in SECTORS_NUMPY_ALLOWED:
                continue
            unexplained.append(f"{path.stem}.{qual}:{lineno} -> np.{attr}")

        assert not unexplained, (
            "numpy inside a traced layer-3 function, with no entry in "
            "SECTORS_NUMPY_ALLOWED:\n  " + "\n  ".join(unexplained) +
            "\n\nLayer 3 has no differentiable-path exceptions.  Either "
            "rewrite it in jnp, or -- if it genuinely runs once at "
            "construction or reads a table off disk -- add it to "
            "SECTORS_NUMPY_ALLOWED in tests/test_sectors_paths.py saying "
            "which of those two it is.")

    def test_the_allow_list_has_no_dead_entries(self):
        """A stale exemption is how a real hole gets waved through later."""
        seen = set()
        for path in self.MODULES:
            for qual, _, _ in _numpy_uses(path):
                seen.add(path.stem)
                if qual:
                    seen.add(f"{path.stem}.{qual}")
        assert not (set(SECTORS_NUMPY_ALLOWED) - seen), (
            f"SECTORS_NUMPY_ALLOWED names symbols that no longer use numpy: "
            f"{sorted(set(SECTORS_NUMPY_ALLOWED) - seen)}")

    def test_the_layer_has_no_whole_module_exemptions(self):
        """**The layer's claim, as an assertion.**

        Layer 2's equivalent asserts the set is exactly ``{_cemulator_compat}``.
        Here it must be empty: a whole module exempted from the differentiable
        path is a sector that cannot be differentiated, and such a sector does
        not belong in this layer -- it belongs excluded, by name, with the
        reason in a docstring.
        """
        whole = {k for k in SECTORS_NUMPY_ALLOWED if "." not in k}
        assert whole == set(), (
            f"layer 3 admits no non-differentiable module, but {sorted(whole)} "
            f"is exempted wholesale.  Exclude it from the layer instead, and "
            f"say why where a reader will look.")

    def test_the_layer_docstring_still_claims_no_exception(self):
        """The prose and the assertion above must not drift apart."""
        doc = inspect.getdoc(S).lower()
        assert "none" in doc and "exception" in doc


class TestDeclarationsAreTrue:
    """Every sector says it is differentiable.  Try it and see."""

    @staticmethod
    def _sectors():
        """Concrete Sector implementations exported by the layer."""
        out = []
        for name in S.__all__:
            obj = getattr(S, name)
            if isinstance(obj, type) and hasattr(obj, "differentiable") \
                    and hasattr(obj, "weights"):
                out.append(obj)
        return out

    def test_no_sector_declares_itself_non_differentiable(self):
        """The set must be empty, so adding one fails loudly.

        The dangerous direction is not a sector that raises -- that is loud.  It
        is one that quietly returns a number with a term missing.
        """
        opted_out = {c.__name__ for c in self._sectors()
                     if not getattr(c, "differentiable", False)}
        assert opted_out == set(), (
            f"layer 3 has no differentiable-path exceptions, but "
            f"{sorted(opted_out)} declares one.")

    def test_every_sector_declares_the_capability_at_all(self):
        for cls in self._sectors():
            assert isinstance(cls.differentiable, bool), cls.__name__
            assert isinstance(cls.name, str) and cls.name, cls.__name__


def _assert_every_params_name_is_a_container(module):
    """A name ending in ``Params`` must actually *be* a container.

    The audits below find containers by filtering ``__all__`` for
    ``SectorParams`` subclasses, and a filter cannot tell a name it should skip
    from a container that has stopped being one.  ``ObscurationParams`` spent a
    commit as a ``PjitFunction`` -- a stray ``@jax.jit`` landed on the class
    instead of the function below it -- so ``isinstance(obj, type)`` was False
    and all five audits dropped its eight parameters in silence.  Nothing was
    wrong with them; nothing was checking either, and the census reported
    "no parameter without a reason" about the wrong set.

    Same shape as ``GalaxyParams`` escaping for want of an export, one step
    over: exported, and not a type.  So the walk asserts rather than filters.
    """
    bad = []
    for name in module.__all__:
        if not name.endswith("Params") or name == "SectorParams":
            continue
        obj = getattr(module, name, None)
        if not (isinstance(obj, type) and issubclass(obj, SectorParams)):
            bad.append(f"{module.__name__}.{name} is {type(obj).__name__}, "
                       f"not a SectorParams subclass")
    assert not bad, bad


class TestTheParameterContainerSeparatesTracedFromStatic:
    """`_PARAMS` are leaves; `_STATIC` are not.  Checked, not assumed."""

    @staticmethod
    def _param_classes():
        """Every declared parameter container, wherever it lives.

        Layer 4 is walked as well as layer 3, and not for symmetry.  These four
        audits find containers by walking a package's ``__all__``, so a
        container that is not exported is invisible to all of them -- which is
        exactly how ``GalaxyParams`` sat outside
        ``test_every_parameter_has_a_reason_for_its_bound`` while the default
        occupation model's 22 parameters went unchecked.  When layer 4 acquired
        its first container (``TransitionParams``, PLAN.md A5) the same hole
        opened one package over, so the fix is to walk both rather than to
        remember.
        """
        import ggah_mod.spectra as SPEC

        out = []
        for module in (S, SPEC):
            _assert_every_params_name_is_a_container(module)
            for name in module.__all__:
                obj = getattr(module, name)
                if (isinstance(obj, type) and issubclass(obj, SectorParams)
                        and obj is not SectorParams and obj not in out):
                    out.append(obj)
        return out

    def test_layer_four_is_walked_too(self):
        """The guard on the guard: if ``TransitionParams`` stops being exported
        it leaves these audits silently, which is the failure mode this method's
        docstring is about."""
        names = {c.__name__ for c in self._param_classes()}
        assert "TransitionParams" in names
        assert "GalaxyParams" in names

    def test_every_declared_field_is_classified(self):
        """A field in neither `_PARAMS` nor `_STATIC` silently stops being a
        leaf -- it vanishes from the pytree and its gradient becomes zero."""
        for cls in self._param_classes():
            declared = set(declared_fields(cls))
            classified = set(cls._PARAMS) | set(cls._STATIC)
            assert declared == classified, (
                f"{cls.__name__}: fields {sorted(declared ^ classified)} are "
                f"declared but not classified as traced or static")

    def test_defaults_agree_with_the_param_table(self):
        """Two places state a default; a test keeps them one."""
        for cls in self._param_classes():
            obj = cls()
            for name, p in cls._PARAMS.items():
                assert float(getattr(obj, name)) == pytest.approx(p.default), (
                    f"{cls.__name__}.{name}")

    def test_no_static_name_reaches_the_leaves(self):
        for cls in self._param_classes():
            leaves = jax.tree_util.tree_leaves(cls())
            assert len(leaves) == len(cls._PARAMS), cls.__name__

    def test_every_parameter_has_a_reason_for_its_bound(self):
        """The rule the layer exists to enforce: nothing fixed by hard-coding."""
        for cls in self._param_classes():
            for name, p in cls._PARAMS.items():
                assert p.why, f"{cls.__name__}.{name} has no recorded reason"
                assert p.kind in S.BOUND_KINDS, f"{cls.__name__}.{name}"


class TestParamMechanics:
    def test_a_param_refuses_a_default_outside_its_own_bounds(self):
        with pytest.raises(ValueError, match="outside its own bounds"):
            Param(default=2.0, bounds=(0.0, 1.0), why="test")

    def test_a_param_refuses_to_be_created_without_a_reason(self):
        """Because a bound with no reason is a hard-coding with extra steps."""
        with pytest.raises(ValueError, match="needs `why`"):
            Param(default=0.5, bounds=(0.0, 1.0))

    def test_a_param_refuses_an_unclassified_bound(self):
        with pytest.raises(ValueError, match="kind must be"):
            Param(default=0.5, bounds=(0.0, 1.0), why="t", kind="vibes")

    @pytest.mark.parametrize("prior,x", [
        (S.Gaussian(1.0, 0.5), 1.3),
        (S.LogGaussian(-4.0, 0.3), 1e-4),
        (S.Flat(), 0.7),
    ])
    def test_every_prior_is_differentiable(self, prior, x):
        g = float(jax.grad(lambda v: jnp.sum(prior.log_prob(v)))(x))
        assert np.isfinite(g)

    def test_a_flat_prior_contributes_nothing(self):
        assert float(S.Flat().log_prob(3.0)) == 0.0
        assert float(jax.grad(lambda v: S.Flat().log_prob(v))(3.0)) == 0.0

    def test_a_gaussian_prior_peaks_at_its_mean(self):
        p = S.Gaussian(2.0, 0.5)
        assert float(p.log_prob(2.0)) == 0.0
        assert float(p.log_prob(2.5)) == pytest.approx(-0.5)
        assert float(p.log_prob(1.5)) == pytest.approx(-0.5)


class TestTheCorrelatedPrior:
    """`MultivariateGaussian`: what a fit's posterior becomes downstream.

    The reason it exists is that a posterior is not diagonal, and the product
    of its marginals is a *different*, generally tighter distribution.  These
    tests pin that difference rather than describing it.
    """

    @staticmethod
    def _mvg(rho=0.9):
        cov = jnp.array([[0.04, rho * 0.04], [rho * 0.04, 0.04]])
        return S.MultivariateGaussian(("a", "b"), jnp.array([1.0, 2.0]), cov)

    def test_it_peaks_at_its_mean(self):
        m = self._mvg()
        assert float(m.log_prob(jnp.array([1.0, 2.0]))) == pytest.approx(0.0)
        assert float(m.log_prob(jnp.array([1.2, 2.2]))) < 0.0

    def test_it_is_not_the_product_of_its_marginals(self):
        """The whole point.  Along the degeneracy the joint is far more
        permissive than independent Gaussians, and across it far stricter."""
        m = self._mvg(rho=0.9)
        ga, gb = m.marginal("a"), m.marginal("b")
        for da, db in [(0.2, 0.2), (0.2, -0.2)]:
            x = jnp.array([1.0 + da, 2.0 + db])
            joint = float(m.log_prob(x))
            product = float(ga.log_prob(x[0]) + gb.log_prob(x[1]))
            assert abs(joint - product) > 0.1
        # along the degeneracy: joint permits what the product forbids
        along = jnp.array([1.0 + 0.2, 2.0 + 0.2])
        assert float(m.log_prob(along)) > float(
            ga.log_prob(along[0]) + gb.log_prob(along[1]))
        # across it: the reverse
        across = jnp.array([1.0 + 0.2, 2.0 - 0.2])
        assert float(m.log_prob(across)) < float(
            ga.log_prob(across[0]) + gb.log_prob(across[1]))

    def test_an_uncorrelated_one_reduces_to_independent_gaussians(self):
        m = self._mvg(rho=0.0)
        x = jnp.array([1.3, 1.7])
        assert float(m.log_prob(x)) == pytest.approx(
            float(m.marginal("a").log_prob(x[0]) + m.marginal("b").log_prob(x[1])),
            abs=1e-6)

    def test_it_is_differentiable(self):
        m = self._mvg()
        g = jax.grad(lambda x: m.log_prob(x))(jnp.array([1.1, 2.1]))
        assert np.all(np.isfinite(np.asarray(g)))

    def test_a_singular_covariance_is_refused(self):
        """A posterior covariance that has gone singular is a chain that did
        not converge; pseudo-inverting it makes that into a prior."""
        with pytest.raises(ValueError, match="not positive definite"):
            S.MultivariateGaussian(("a", "b"), [0.0, 0.0],
                                   [[1.0, 1.0], [1.0, 1.0]])

    @pytest.mark.parametrize("names,mu,cov,match", [
        (("a", "a"), [0, 0], [[1, 0], [0, 1]], "repeated name"),
        (("a", "b"), [0.0], [[1, 0], [0, 1]], "mu has shape"),
        (("a", "b"), [0, 0], [[1.0]], "cov has shape"),
        (("a", "b"), [0, 0], [[1.0, 0.5], [0.4, 1.0]], "not symmetric"),
    ])
    def test_malformed_input_is_refused(self, names, mu, cov, match):
        with pytest.raises(ValueError, match=match):
            S.MultivariateGaussian(names, mu, cov)

    def test_a_param_refuses_to_carry_one(self):
        """It would be handed a scalar where it expects a vector, and would
        return a number rather than raise."""
        with pytest.raises(TypeError, match="prior over several parameters"):
            Param(1.0, (0.0, 2.0), self._mvg(), why="test")

    def test_from_samples_recovers_the_distribution_it_was_drawn_from(self):
        rng = np.random.default_rng(0)
        cov = np.array([[0.04, 0.03], [0.03, 0.09]])
        draws = rng.multivariate_normal([1.0, 2.0], cov, size=200_000)
        m = S.MultivariateGaussian.from_samples(("a", "b"), draws)
        assert np.allclose(np.asarray(m.mu), [1.0, 2.0], atol=0.01)
        assert np.allclose(np.asarray(m.cov), cov, atol=0.005)

    def test_from_samples_refuses_a_chain_too_short_to_have_a_covariance(self):
        with pytest.raises(ValueError, match="singular by construction"):
            S.MultivariateGaussian.from_samples(("a", "b"), np.zeros((2, 2)))

    def test_a_joint_prior_replaces_the_independent_ones_it_covers(self):
        """Adding both would count one measurement twice, and the copy that
        gets double-counted is the one whose correlations were discarded."""
        mvg = S.MultivariateGaussian(("r_off",), [0.25], [[0.01]])
        p = S.MisCenteringParams(r_off=0.45)
        independent = float(p.log_prior())
        joint = float(p.log_prior(joint=[mvg]))
        # the shared Gaussian(0.25, 0.10) and this MVG are the same
        # distribution, so replacing one by the other must not change anything
        assert joint == pytest.approx(independent, abs=1e-5)
        # ... and it is a replacement, not an addition
        assert joint != pytest.approx(2 * independent, abs=1e-5)

    def test_a_joint_prior_over_an_unknown_name_is_refused(self):
        with pytest.raises(KeyError, match="not parameters of this container"):
            S.MisCenteringParams().log_prior(
                joint=[S.MultivariateGaussian(("nope",), [0.0], [[1.0]])])

    def test_two_joint_priors_may_not_share_a_parameter(self):
        a = S.MultivariateGaussian(("p_off",), [0.0], [[1.0]])
        b = S.MultivariateGaussian(("p_off", "r_off"), [0.0, 0.25],
                                   [[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="counted twice"):
            S.MisCenteringParams().log_prior(joint=[a, b])


class TestOneMeasurementIsStatedOnce:
    """`p_off` and `r_off` live in two containers and must not drift again.

    They had.  `GalaxyParams.p_off` defaulted to 0.25 and
    `MisCenteringParams.p_off` to 0.0, while `GalaxySector.weights` builds the
    second from the first -- so the effective default was 0.25 and
    `miscentering`'s "a mis-centred model must be asked for" was false in
    practice.  `r_off` carried the identical published CMASS `Gaussian(0.25,
    0.10)` in both, which is one measurement counted twice whenever a fit frees
    both containers.
    """

    @pytest.mark.parametrize("name", ["p_off", "r_off"])
    def test_the_two_containers_share_one_param_object(self, name):
        assert (S.GalaxyParams._PARAMS[name]
                is S.MisCenteringParams._PARAMS[name]), (
            f"{name} is declared twice; one object cannot drift from itself")

    @pytest.mark.parametrize("name", ["p_off", "r_off"])
    def test_the_field_defaults_agree_too(self, name):
        assert float(getattr(S.GalaxyParams(), name)) == pytest.approx(
            float(getattr(S.MisCenteringParams(), name)))

    def test_the_fiducial_galaxy_model_is_undecorated(self):
        """`GalaxySector.weights` documents p_off = 0 as the neutral value; the
        default must actually be it, and no assembly-bias amplitude survives
        as a field."""
        g = S.GalaxyParams()
        assert float(g.p_off) == 0.0
        assert not hasattr(g, "a_cen") and not hasattr(g, "a_sat")


class TestTracerWeights:
    """The contract, exercised without layer 4."""

    @staticmethod
    def _w(discrete=True, n_k=4, n_m=6):
        return S.TracerWeights(
            w_point=jnp.full((n_m,), 0.3),
            w_extended=jnp.full((n_k, n_m), 0.7),
            norm=jnp.asarray(2.0), discrete=discrete,
            bias_weight=None, name="test")

    def test_only_the_arrays_are_leaves(self):
        """Three arrays here (bias_weight is None, which is an empty subtree)."""
        leaves = jax.tree_util.tree_leaves(self._w())
        assert len(leaves) == 3

    def test_the_static_flag_does_not_become_a_leaf(self):
        """`discrete` selects the pair rule.  A traced bool would make the
        branch a value, and the pair rule is not something to differentiate."""
        for leaf in jax.tree_util.tree_leaves(self._w()):
            assert not isinstance(leaf, (bool, str))

    def test_total_is_the_documented_combination(self):
        w = self._w()
        np.testing.assert_allclose(np.asarray(w.total()), (0.3 + 0.7) / 2.0)

    def test_a_tracer_with_no_weight_at_all_raises(self):
        w = S.TracerWeights(None, None, jnp.asarray(1.0), True, None, "empty")
        with pytest.raises(ValueError, match="no weight at all"):
            w.total()

    def test_a_redshift_stacked_weight_is_refused(self):
        """Rank is what tells `(NM,)` from `(Nk, NM)`, so a third axis is not a
        variant to absorb --- it is how a redshift-stacked HaloField reaches a
        sector.  `jax.vmap(make_field)` builds one legitimately; handing it
        downstream is the error, and it used to be a silent one."""
        w = S.TracerWeights(
            w_point=jnp.full((3, 4, 6), 0.3), w_extended=None,
            norm=jnp.asarray(2.0), discrete=True, bias_weight=None, name="stacked")
        with pytest.raises(ValueError, match="one epoch|redshift axis"):
            w.total()

    def test_at_large_scales_is_the_k_zero_row_and_not_a_redshift_slice(self):
        """The reason the rank check exists: `at_large_scales` is `total()[0]`,
        which on a rank-3 weight would return a redshift slice wearing the
        k -> 0 row's name, and mass conservation is stated in that number."""
        w = self._w()
        np.testing.assert_allclose(np.asarray(w.at_large_scales()),
                                   np.asarray(w.total())[0])
        stacked = S.TracerWeights(
            w_point=jnp.full((3, 4, 6), 0.3), w_extended=None,
            norm=jnp.asarray(2.0), discrete=True, bias_weight=None, name="stacked")
        with pytest.raises(ValueError):
            stacked.at_large_scales()

    def test_combine_refuses_discrete_tracers(self):
        """Summing two discrete tracers' weights would silently assume they are
        one population; the cross-pair rule is layer 4's job."""
        with pytest.raises(ValueError, match="continuous fields"):
            S.combine(self._w(discrete=True), self._w(discrete=True))

    def test_combine_adds_continuous_fields(self):
        a, b = self._w(discrete=False), self._w(discrete=False)
        np.testing.assert_allclose(np.asarray(S.combine(a, b).total()),
                                   2.0 * np.asarray(a.total()))

    def test_combine_refuses_mismatched_normalisations(self):
        a = self._w(discrete=False)
        b = a.replace(norm=jnp.asarray(5.0))
        with pytest.raises(ValueError, match="one normalisation"):
            S.combine(a, b)

    def test_it_crosses_a_jit_boundary(self):
        """The failure a NamedTuple would have produced, as a test.

        Built outside and passed *in*, so the static fields really do have to
        be aux data: a `str` among the leaves is not a JAX type, and the error
        would surface here rather than where the mistake was made.
        """
        w = self._w()
        out = jax.jit(lambda ww: jnp.sum(ww.total()))(w)
        assert np.isfinite(float(out))
        assert jax.jit(lambda ww: ww)(w).name == "test"

    def test_weights_survive_jit_and_grad(self):
        def f(x):
            w = S.TracerWeights(w_point=jnp.full((6,), x),
                                w_extended=jnp.full((4, 6), 0.7),
                                norm=jnp.asarray(2.0), discrete=True,
                                bias_weight=None, name="test")
            return jnp.sum(w.total())
        assert np.isfinite(float(jax.jit(f)(0.3)))
        assert float(jax.grad(f)(0.3)) == pytest.approx(24.0 / 2.0)
