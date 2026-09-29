r"""Verification: the resolver, and the ordering trap it exists to avoid.

Two things matter here and neither is arithmetic.

**Each `P_ab(k, z)` is computed once.** ``w_p^{gg}``, ``ΔΣ^{gm}`` and
``C_ℓ^{gy}`` share tracers and redshifts; resolving them independently would
build the galaxy weights three times.

**JAX flattens a ``dict`` sorted by key, not by insertion order.** So
``jax.jacfwd`` of a ``dict[name, array]`` silently reorders its rows against the
order the spec was written in, and a Fisher matrix built from it stops matching
its covariance — with nothing raising, because both are still matrices of the
right size. :class:`TestTheOrderingTrap` chooses statistic names whose spec
order and sorted order **differ**, so a test that passed under either would not
be testing anything.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.sectors as S
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.observables import spec as SPECM
from ggah_mod.observables import real_space as RS
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field, make_fields
from ggah_mod.observables import (
    king, limber_grid, number_counts, thermal_sz,
)
from ggah_mod.observables.spec import (
    Cl, DeltaSigma, ObservableSpec, Statistic, Wp, WTheta, Xi, make_model,
)

pytestmark = pytest.mark.slow

ZU15 = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)


#: Names chosen so spec order != sorted order.
SPEC = ObservableSpec((
    Wp("wp_gg", "galaxies", "galaxies", [0.1, 1.0, 10.0], pi_max=60.0, z=0.25),
    DeltaSigma("ds_gm", "galaxies", "matter", [0.5, 2.0], z=0.25),
    Cl("cl_gy", "galaxies", "pressure", [100.0, 1000.0],
       kernel_a="counts", kernel_b="tsz", beam_b="erosita"),
))


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def projection():
    z, chi = limber_grid(PLANCK18, z_max=1.5, n=6, backend=DIFFERENTIABLE)
    nz = jnp.exp(-0.5 * ((z - 0.5) / 0.15) ** 2)
    return z, {"counts": number_counts(z, chi, nz), "tsz": thermal_sz(z, chi)}


@pytest.fixture(scope="module")
def built(pk, projection):
    z, kernels = projection
    gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
    gal_p = S.galaxy_defaults("zumandelbaum15")
    sectors = {"galaxies": gal, "gas": S.HotGasDPM(backend=DIFFERENTIABLE),
               "agn": S.AgnSector(gal), "matter": S.MatterField()}
    model = make_model(
        SPEC, fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz),
        sectors=sectors, kernels=kernels,
        beams={"erosita": lambda ell: king(ell, 8.64, 1.5)},
        cosmo=PLANCK18, backend=DIFFERENTIABLE, z_proj=z)
    field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.25)
    f_cen, _ = gal.stellar_fraction(field, gal_p)
    params = {"galaxies": gal_p, "gas": S.dpm_model_params(2),
              "agn": S.AgnParams(),
              "matter": {"split": S.BaryonSplit.from_hot(
                  PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(field.n_m, 0.1), f_star_cen=f_cen)}}
    return model, params


class TestTheResolver:
    def test_it_deduplicates_the_pairs(self, projection):
        z, _ = projection
        plan = SPEC.resolve(z_proj=tuple(float(v) for v in z))
        assert plan.pairs == (("galaxies", "galaxies"),
                              ("galaxies", "matter"),
                              ("galaxies", "pressure"))

    def test_a_repeated_pair_is_resolved_once(self, projection):
        z, _ = projection
        spec = ObservableSpec((
            Wp("a", "galaxies", "galaxies", [1.0], pi_max=60.0, z=0.2),
            Xi("b", "galaxies", "galaxies", [1.0], z=0.2),
        ))
        assert spec.resolve(z_proj=tuple(float(v) for v in z)).pairs == \
            (("galaxies", "galaxies"),)

    def test_the_pair_is_canonically_ordered(self):
        """Sorted, not a `frozenset`: a one-element set cannot tell an auto
        from a cross of two tracers with the same name, and the auto is exactly
        the case with a different pair rule."""
        one = ObservableSpec((DeltaSigma("x", "galaxies", "matter", [1.0],
                                         z=0.2),)).resolve()
        other = ObservableSpec((DeltaSigma("x", "matter", "galaxies", [1.0],
                                           z=0.2),)).resolve()
        assert one.pairs == other.pairs

    def test_the_projection_grid_is_kept_apart(self, projection):
        r"""A ``C_l`` stack lines up row-for-row with its kernels' chi grid; a
        real-space statistic at another redshift joins ``z_nodes`` without
        belonging in that stack."""
        z, _ = projection
        plan = SPEC.resolve(z_proj=tuple(float(v) for v in z))
        assert len(plan.z_proj) == len(z)
        assert 0.25 in plan.z_nodes and 0.25 not in plan.z_proj
        assert len(plan.z_nodes) == len(z) + 1

    def test_a_projected_statistic_without_a_grid_is_refused(self, built):
        model, params = built
        spec = ObservableSpec((SPEC.statistics[2],))
        m = make_model(spec, fields_at=lambda zz: (None,) * len(zz),
                       sectors={}, kernels={},
                       cosmo=PLANCK18, backend=DIFFERENTIABLE)
        with pytest.raises(ValueError, match="no projection grid"):
            m(params)

    def test_duplicate_names_are_refused(self):
        """Names index the output dict and the slices, so a duplicate silently
        drops one."""
        with pytest.raises(ValueError, match="share a name"):
            ObservableSpec((
                Wp("x", "galaxies", "galaxies", [1.0], pi_max=60.0, z=0.2),
                Xi("x", "galaxies", "galaxies", [1.0], z=0.2)))


class TestComplementaryProbesAtDifferentRedshifts:
    r"""Fitting several statistics measured at *different* redshifts.

    The case this package exists for, and the one nothing here covered: every
    other spec in this file sits at a single redshift, so the resolver's
    redshift handling was exercised only where it could not go wrong.

    `Statistic.z` was already per-probe and already required for a real-space
    kind.  What was missing is that `make_model` computed `pairs x z_nodes` --
    the product rather than the set -- so a spec with three probes at three
    redshifts evaluated every pair at every one of them.
    """

    #: Clustering and lensing in a low-z bin, clustering in a higher one, and
    #: an AGN correlation higher still.  Deliberately *not* a grid: the point
    #: is that no probe wants all the redshifts.
    MIXED = ObservableSpec((
        Wp("wp_lo", "galaxies", "galaxies", [0.5, 1.0, 5.0], pi_max=80.0, z=0.25),
        DeltaSigma("ds_lo", "galaxies", "matter", [0.5, 1.0], z=0.25),
        Wp("wp_hi", "galaxies", "galaxies", [0.5, 1.0, 5.0], pi_max=80.0, z=0.80),
        Xi("xi_agn", "agn", "agn", [1.0, 5.0], z=1.50),
    ))

    def test_the_plan_is_sparse_not_the_product(self):
        plan = self.MIXED.resolve()
        assert len(plan.pairs) == 3 and len(plan.z_nodes) == 3
        # 3 x 3 = 9 if the product were the work; four statistics read four.
        assert len(plan.needs) == 4, plan.needs

    def test_each_redshift_carries_only_its_own_pairs(self):
        plan = self.MIXED.resolve()
        assert plan.pairs_at(0.25) == (("galaxies", "galaxies"),
                                       ("galaxies", "matter"))
        assert plan.pairs_at(0.80) == (("galaxies", "galaxies"),)
        assert plan.pairs_at(1.50) == (("agn", "agn"),)
        # And nothing is claimed at a redshift no statistic asked for.
        assert plan.pairs_at(0.5) == ()

    def test_a_projected_statistic_still_wants_every_node(self, projection):
        """The asymmetry that makes the sparse set worth having.

        A `C_l` reads its pair at *all* of `z_proj` -- it is an integral over
        the kernel -- while a real-space probe reads one node.  A sparse set
        that dropped either would be wrong in opposite directions.
        """
        z, _ = projection
        spec = ObservableSpec((
            Wp("wp", "galaxies", "galaxies", [1.0], pi_max=60.0, z=0.25),
            Cl("cl", "galaxies", "pressure", [100.0],
               kernel_a="counts", kernel_b="tsz"),
        ))
        plan = spec.resolve(z_proj=tuple(float(v) for v in z))
        gg = [zz for pair, zz in plan.needs if pair == ("galaxies", "galaxies")]
        gp = [zz for pair, zz in plan.needs
              if pair == ("galaxies", "pressure")]
        assert gg == [0.25]
        assert sorted(gp) == sorted(float(v) for v in z)

    def test_the_redshifts_are_derived_from_the_probes(self):
        """`z_nodes` is the union of what was asked for, and nothing else.

        There is no second place to declare the redshift set, so it cannot
        drift from the statistics that use it.
        """
        plan = self.MIXED.resolve()
        assert plan.z_nodes == (0.25, 0.80, 1.50)
        assert set(plan.z_nodes) == {z for _, z in plan.needs}

    def test_the_same_statistic_at_two_redshifts_gives_two_answers(self, pk):
        """The assertion that would catch a lookup collapsing to one z.

        `wp_lo` and `wp_hi` differ only in redshift, so if the cache key or the
        sparse set ever confused them this returns one array twice -- which is
        a plausible failure and an invisible one, since both are finite and of
        the right shape.
        """
        gal = S.GalaxySector("zheng07", shmr="zu15", backend=DIFFERENTIABLE,
                             calibration="off")
        spec = ObservableSpec((
            Wp("wp_lo", "galaxies", "galaxies", [1.0, 5.0], pi_max=60.0, z=0.25),
            Wp("wp_hi", "galaxies", "galaxies", [1.0, 5.0], pi_max=60.0, z=1.50),
        ))
        model = make_model(
            spec, fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz),
            sectors={"galaxies": gal}, kernels={}, cosmo=PLANCK18,
            backend=DIFFERENTIABLE)
        out, _ = model({"galaxies": dict(S.galaxy_defaults("zheng07"), **ZU15)})
        lo, hi = np.asarray(out["wp_lo"]), np.asarray(out["wp_hi"])
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert not np.allclose(lo, hi), "the two redshifts returned one answer"
        # Clustering amplitude falls with redshift for a fixed occupation.
        assert np.all(hi < lo)

    def test_the_data_vector_still_follows_spec_order(self, pk):
        """The trap this module's docstring is about, under a multi-z spec.

        Sorted by key, `wp_hi` precedes `wp_lo`; in spec order it does not.
        The slices have to follow the spec or a Fisher matrix stops matching
        its covariance, and adding redshifts must not change that.
        """
        plan = self.MIXED.resolve()
        names = [n for n, _, _ in plan.slices]
        assert names == ["wp_lo", "ds_lo", "wp_hi", "xi_agn"]
        assert names != sorted(names)
        assert plan.size == 3 + 2 + 3 + 2


class TestTheSpecIsStatic:
    def test_it_has_no_pytree_leaves(self):
        """A traced `ell` grid breaks `jacfwd`'s fixed output shape as surely
        as a traced band edge would."""
        for s in SPEC.statistics:
            assert jax.tree_util.tree_leaves(s) in ([], [s])
        assert jax.tree_util.tree_leaves(SPEC) in ([], [SPEC])

    def test_every_statistic_is_hashable(self):
        assert isinstance(hash(SPEC), int)
        for s in SPEC.statistics:
            assert isinstance(hash(s), int)

    def test_a_real_space_statistic_needs_a_redshift(self):
        with pytest.raises(ValueError, match="needs a `z`"):
            Statistic("x", "wp", "galaxies", "galaxies", (1.0,), pi_max=60.0)

    def test_a_projected_statistic_needs_its_kernels(self):
        with pytest.raises(ValueError, match="radial\\s+kernel"):
            Statistic("x", "cl", "galaxies", "pressure", (100.0,))

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="unknown statistic kind"):
            Statistic("x", "bispectrum", "a", "b", (1.0,), z=0.1)


class TestTheOrderingTrap:
    r"""The reason a dict alone is not enough."""

    def test_the_spec_order_is_not_the_sorted_order(self):
        """Otherwise the test below would pass under either convention."""
        names = [s.name for s in SPEC.statistics]
        assert names != sorted(names)

    def test_the_slices_follow_the_spec(self, built):
        model, _ = built
        assert [n for n, _, _ in model.plan.slices] == \
            [s.name for s in SPEC.statistics]

    def test_the_vector_matches_the_slices(self, built):
        model, params = built
        out, vector = model(params)
        for name, lo, hi in model.plan.slices:
            assert np.allclose(np.asarray(vector[lo:hi]),
                               np.asarray(out[name]), rtol=1e-14)

    def test_a_dict_would_have_reordered_it(self, built):
        """Stated as a fact about JAX, so the reason survives."""
        model, params = built
        out, _ = model(params)
        flattened = jax.tree_util.tree_leaves(out)
        by_sorted = [out[n] for n in sorted(out)]
        assert all(np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(flattened, by_sorted))
        assert sorted(out) != [s.name for s in SPEC.statistics]


class TestTheModelIsOneFunctionOfTheParameters:
    def test_it_returns_sensible_numbers(self, built):
        model, params = built
        out, vector = model(params)
        assert set(out) == {"wp_gg", "ds_gm", "cl_gy"}
        assert vector.shape == (7,)
        assert np.all(np.isfinite(np.asarray(vector)))
        assert np.all(np.asarray(out["wp_gg"]) > 0)
        assert np.all(np.asarray(out["cl_gy"]) > 0)

    @pytest.mark.x64
    def test_the_jacobian_has_the_right_sparsity(self, built):
        r"""``log10_pe_anchor`` is the DPM pressure normalisation, so it reaches the
        tSZ cross and nothing else in this vector.  A Jacobian whose *zeros*
        are in the right places is a stronger statement than one whose values
        are finite."""
        model, params = built

        def data_vector(log10_pe_anchor):
            p = dict(params)
            p["gas"] = params["gas"].replace(log10_pe_anchor=log10_pe_anchor)
            return model(p)[1]

        jac = np.asarray(jax.jacfwd(data_vector)(2.0607))
        assert jac.shape == (7,)
        cl_lo, cl_hi = model.plan.slices[-1][1], model.plan.slices[-1][2]
        assert np.all(jac[cl_lo:cl_hi] != 0.0)
        assert np.all(jac[:cl_lo] == 0.0)

    @pytest.mark.x64
    def test_the_jacobian_is_shape_stable(self, built):
        """Which is the point of a static spec: `jacfwd` of a model whose
        output shape depended on the parameters could not be assembled into a
        Fisher matrix at all."""
        model, params = built

        def data_vector(f_gas_amp):
            # Scale the *hot* component and let the split re-close around it:
            # rebuilding through `from_hot` is what keeps the budget exact
            # while one phase is varied, which a dict update could not do.
            p = dict(params)
            base = params["matter"]["split"]
            p["matter"] = {"split": S.BaryonSplit.from_hot(
                PLANCK18.Omega_b / PLANCK18.Omega_m, f_gas_amp * base.f_hot,
                f_star_cen=base.f_star_cen, f_star_sat=base.f_star_sat,
                f_cold=base.f_cold)}
            return model(p)[1]

        for x in (0.9, 1.0, 1.1):
            assert np.asarray(jax.jacfwd(data_vector)(x)).shape == (7,)


class TestTheResolverAsksOnce:
    r"""One layer-1 call for the projection grid, not one per node.

    `make_model` used to take ``field_at: z -> HaloField`` and call it once per
    redshift.  Layer 2 is one epoch per object, so that shape was right; what
    was wrong is that each call then made its own linear-P(k) request, and a
    Boltzmann solve is memoised on the whole redshift tuple.  On the ACCURATE
    flavour, whose ``n_z_proj`` is 64, that was 64 solves for every evaluation
    of the model --- inside the likelihood, not once at setup.
    """

    @staticmethod
    def _model(fields_at, projection):
        z, kernels = projection
        gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
        sectors = {"galaxies": gal, "gas": S.HotGasDPM(backend=DIFFERENTIABLE),
                   "agn": S.AgnSector(gal), "matter": S.MatterField()}
        return make_model(
            SPEC, fields_at=fields_at, sectors=sectors, kernels=kernels,
            beams={"erosita": lambda ell: king(ell, 8.64, 1.5)},
            cosmo=PLANCK18, backend=DIFFERENTIABLE, z_proj=z)

    def test_the_builder_is_handed_every_node_at_once(self, pk, projection, built):
        _, params = built
        seen = []

        def spy(zz):
            seen.append(tuple(zz))
            return make_fields(PLANCK18, DIFFERENTIABLE, pk, jnp.asarray(zz))

        model = self._model(spy, projection)
        model(params)
        assert len(seen) == 1, f"builder called {len(seen)} times, not once"
        assert seen[0] == model.plan.z_nodes

    def test_field_at_is_refused_by_name(self):
        with pytest.raises(ValueError, match="`field_at` is now `fields_at`"):
            make_model(SPEC, fields_at=lambda zz: (), field_at=lambda z: None,
                       sectors={}, kernels={}, cosmo=PLANCK18,
                       backend=DIFFERENTIABLE, z_proj=(0.1,))

    def test_a_short_answer_is_refused(self, projection, built):
        """The correspondence is positional, so a short tuple would pair a
        spectrum with the wrong redshift rather than raise."""
        _, params = built
        model = self._model(lambda zz: (), projection)
        with pytest.raises(ValueError, match="positional"):
            model(params)

    # The two below need neither a spectrum backend nor a sector, so they run
    # wherever the package imports.  The `fields_at` contract is the thing this
    # class exists for, and it should not be reachable only through a full model.

    def test_the_builder_is_asked_once_for_the_whole_node_set(self):
        """The spy returns nothing, so the length check stops the model before
        it needs a sector -- which is what lets this test stand on its own."""
        seen = []

        def spy(zz):
            seen.append(tuple(zz))
            return ()

        model = make_model(
            ObservableSpec([Wp("wp", "galaxies", "galaxies", [1.0, 2.0],
                               pi_max=60.0, z=0.3)]),
            fields_at=spy, sectors={}, kernels={}, cosmo=PLANCK18,
            backend=DIFFERENTIABLE)
        with pytest.raises(ValueError, match="positional"):
            model({})
        assert len(seen) == 1
        assert seen[0] == model.plan.z_nodes == (0.3,)

    def test_a_long_answer_is_refused_too(self):
        model = make_model(
            ObservableSpec([Wp("wp", "galaxies", "galaxies", [1.0],
                               pi_max=60.0, z=0.3)]),
            fields_at=lambda zz: (None, None), sectors={}, kernels={},
            cosmo=PLANCK18, backend=DIFFERENTIABLE)
        with pytest.raises(ValueError, match="positional"):
            model({})


class TestTheStatisticsBlockDAdded:
    r"""``tau(R)`` in the resolver, and two things that were already there.

    ``PLAN.md`` block **D**, and two of its four items turned out to be done
    already -- which is worth as much as the one that was not, because both
    were written down as missing.
    """

    def test_tau_ksz_is_a_statistic_the_spec_can_name(self):
        """It was a function a caller invoked, so it could not appear in a data
        vector and the aperture convention was the caller's problem rather than
        the declaration's.  PLAN.md **D1**."""
        s = SPECM.TauKsz("tau", "galaxies", "electrons", [0.5, 1.0, 2.0], z=0.5)
        assert s.kind == "tau_ksz" and "tau_ksz" in SPECM.KINDS
        assert s.x == (0.5, 1.0, 2.0)
        assert s.n == 3

    def test_its_abscissa_is_an_aperture_and_its_grid_is_in_the_spec(self):
        """The grid is part of the spec for the same reason ``WTheta``'s ``ell``
        is: the statistic inherits its accuracy from it, and a default hidden
        in the model would make that accuracy invisible."""
        s = SPECM.TauKsz("tau", "galaxies", "electrons", [0.5, 4.0], z=0.5)
        assert len(s.ell) == 256
        assert s.ell[0] == pytest.approx(0.05)
        assert s.ell[-1] == pytest.approx(RS.CAP_OUTER * 4.0)

        mine = SPECM.TauKsz("tau", "galaxies", "electrons", [1.0], z=0.5,
                            r_grid=[0.1, 0.5, 1.0, 1.5])
        assert mine.ell == (0.1, 0.5, 1.0, 1.5)

    def test_a_real_space_statistic_still_needs_a_redshift(self):
        with pytest.raises(ValueError, match="needs a `z`"):
            SPECM.Statistic("t", "tau_ksz", "galaxies", "electrons", (1.0,))

    def test_the_dispersion_measure_cross_spectra_were_already_reachable(self):
        r"""**D2 was zero resolver entries away, not one.**

        ``PLAN.md`` carried "the dispersion-measure cross-correlations are one
        resolver entry away", from the paper.  They are none:
        :func:`~ggah_mod.observables.kernels.dispersion_measure` returns an
        ordinary :class:`~ggah_mod.observables.kernels.RadialKernel` and the
        generic ``Cl`` constructor takes it like any other.  What was missing
        was not an entry but the knowledge, and a test.
        """
        import numpy as _np

        from ggah_mod.observables import kernels as K

        z, chi = K.limber_grid(PLANCK18, z_max=1.0, n=8)
        dm = K.dispersion_measure(z, chi, PLANCK18)
        assert dm.name == "dm"

        cross = SPECM.Cl("dm_x_g", "galaxies", "electrons",
                         _np.logspace(1, 3, 6), kernel_a="counts",
                         kernel_b="dm")
        auto = SPECM.Cl("dm_x_dm", "electrons", "electrons",
                        _np.logspace(1, 3, 6), kernel_a="dm", kernel_b="dm")
        plan = SPECM.ObservableSpec([cross, auto]).resolve(
            z_proj=tuple(float(v) for v in z))
        assert ("electrons", "galaxies") in plan.pairs
        assert ("electrons", "electrons") in plan.pairs

    def test_the_plan_asks_per_need_and_not_per_pair(self):
        r"""**D4 was already done too.**

        ``PLAN.md`` carried "a spec mixing one real-space statistic with one
        projected one computes both spectra at all of the projection's
        redshifts including the one it does not want ... the cost scales as the
        product where it should scale as the union."  ``Plan.needs`` is a list
        of ``(pair, z)`` and ``pairs_at`` filters it, so it already scales as
        the union.  Measured here so the claim stops being restated.
        """
        import numpy as _np

        z_proj = tuple(float(v) for v in _np.linspace(0.05, 1.0, 8))
        plan = SPECM.ObservableSpec([
            SPECM.Wp("wp", "galaxies", "galaxies", [1.0, 2.0],
                     pi_max=100.0, z=0.3),
            SPECM.Cl("cl", "galaxies", "matter", [10.0, 100.0],
                     kernel_a="counts", kernel_b="lens"),
        ]).resolve(z_proj=z_proj)

        product = len(plan.pairs) * len(plan.z_nodes)
        assert len(plan.needs) < product
        assert len(plan.needs) == 1 + len(z_proj)          # the union, exactly

        # ...and specifically: the clustering pair is asked for at its own
        # redshift and nowhere else.
        gg = [zz for pair, zz in plan.needs if pair == ("galaxies", "galaxies")]
        assert gg == [0.3]

    def test_it_reaches_a_data_vector_and_a_gradient(self, pk):
        r"""The deliverable: ``tau(R)`` is fittable, not merely computable.

        ``PLAN.md`` **D1** put it this way -- it "cannot appear in a data
        vector", which is what makes an ACT or Simons Observatory stack unable
        to constrain the ejection radius.  It can now, and the gradient in
        ``eta_ej`` is what says the constraint would propagate.
        """
        gal = S.GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
        sectors = {"galaxies": gal, "gas": S.HotGasDPM(backend=DIFFERENTIABLE),
                   "agn": S.AgnSector(gal), "matter": S.MatterField(),
                   "ejecta": S.EjectaSector()}
        spec = SPECM.ObservableSpec([
            SPECM.TauKsz("tau", "galaxies", "electrons", [0.5, 1.0, 2.0],
                         z=0.25)])
        model = make_model(
            spec,
            fields_at=lambda zz: make_fields(PLANCK18, DIFFERENTIABLE, pk, zz),
            sectors=sectors, kernels={}, cosmo=PLANCK18,
            backend=DIFFERENTIABLE)

        split = S.BaryonSplit.from_hot(
            PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(256, 0.1))
        base = {"galaxies": S.galaxy_defaults("zumandelbaum15"),
                "gas": S.dpm_model_params(2), "agn": S.AgnParams(),
                "matter": {"split": split}}

        def vector(eta):
            p = dict(base, ejecta={"split": split,
                                   "params": S.EjectaParams(eta_ej=eta)})
            return model(p)[1]

        v = vector(2.0)
        assert v.shape == (3,)
        assert np.all(np.asarray(v) > 0.0)
        # A kSZ stack constrains where the baryons went, so the response to the
        # ejection radius must not be a structural zero.
        g = float(jax.grad(lambda e: vector(e).sum())(2.0))
        assert np.isfinite(g) and abs(g) > 0.0
