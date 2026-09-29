r"""Verification: the other half of layer 3 has bounds now, and honest ones.

``Param`` applied to five containers and forty-six parameters.  The occupation
and CLF registries are **159 further numbers** across fifteen models, and every
one was a keyword float with no bound, no prior and no recorded reason -- the
half of the layer the technical paper singles out as carrying none.

The risk in fixing that is not omission but *invention*.
``GalaxyParams``'s docstring states it: "a bound invented by a downstream
package looks exactly like a bound a measurement established, and only one of
them may be relaxed on the strength of new data."  So most of what this module
checks is not that bounds exist -- ``test_sectors_paths.py``'s audits already do
-- but that the ones which are *not* measurements do not claim to be.
"""
import inspect

import pytest

from ggah_mod.sectors import occupation_params as OP
from ggah_mod.sectors.clf import CLF, CLF_DEFAULTS
from ggah_mod.sectors.galaxies import GalaxyParams
from ggah_mod.sectors.occupation import DEFAULTS, OCCUPATION
from ggah_mod.sectors.params import BOUND_KINDS, Param, SectorParams

ALL_MODELS = sorted(set(DEFAULTS) | set(CLF_DEFAULTS))


class TestEveryModelHasOne:
    """Fifteen models in, fifteen containers out."""

    def test_the_registries_and_the_containers_agree(self):
        assert set(OP.OCCUPATION_PARAMS) == set(ALL_MODELS)
        assert set(DEFAULTS) <= set(OP.OCCUPATION_PARAMS)
        assert set(CLF_DEFAULTS) <= set(OP.OCCUPATION_PARAMS)

    def test_the_model_registries_are_covered_too(self):
        """``OCCUPATION`` and ``CLF`` hold the callables; ``DEFAULTS`` and
        ``CLF_DEFAULTS`` hold their numbers.  A model in one and not the other
        is a model that cannot be given parameters, or parameters with nothing
        to give them to."""
        assert set(OCCUPATION) == set(DEFAULTS)
        assert set(CLF) == set(CLF_DEFAULTS)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_the_fields_are_exactly_the_registry_defaults(self, model):
        want = DEFAULTS.get(model) or CLF_DEFAULTS[model]
        obj = OP.make_occupation_params(model)
        got = {n: getattr(obj, n) for n in type(obj)._PARAMS}
        assert got == pytest.approx(want)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_the_signature_it_describes_is_the_one_it_matches(self, model):
        """The container must name what the model's own function takes -- a
        bound on a parameter the callable does not have is a bound on nothing.
        """
        fns = OCCUPATION.get(model) or CLF[model]
        taken = set()
        for fn in (fns if isinstance(fns, tuple) else (fns,)):
            taken |= set(inspect.signature(fn).parameters)
        declared = set(type(OP.make_occupation_params(model))._PARAMS)
        assert declared <= taken, (
            f"{model} declares {sorted(declared - taken)}, which its "
            f"occupation does not take")


class TestNoBoundClaimsToBeAMeasurementUnlessItIs:
    r"""The rule ``GalaxyParams`` states, enforced across the other thirteen.

    ``BOUND_KINDS`` makes the distinction machine-readable: a ``prior`` "moves
    when the measurement does", a ``physical`` one when the argument does.  So
    the check is not on the numbers -- which are judgement calls -- but on
    whether a judgement call is *labelled* as a measurement.
    """

    #: The one family whose bounds are published: Zu & Mandelbaum's own uniform
    #: priors, Paper I Table 2 and Paper III Table 2.
    PUBLISHED = tuple(m for m in ALL_MODELS if m.startswith("zumandelbaum"))

    @pytest.mark.parametrize("model", [m for m in ALL_MODELS
                                       if not m.startswith("zumandelbaum")])
    def test_an_unpublished_bound_is_not_labelled_prior(self, model):
        for name, p in type(OP.make_occupation_params(model))._PARAMS.items():
            assert p.kind in ("physical", "definitional", "validity"), (
                f"{model}.{name} is kind={p.kind!r}. Only a bound this package "
                f"can cite a measurement for may be 'prior'; the rest are "
                f"arguments, and the difference is what may be relaxed on new "
                f"data")

    @pytest.mark.parametrize("model", PUBLISHED)
    def test_the_published_family_reuses_the_objects_not_the_numbers(self, model):
        """Shared by identity, the ``P_OFF``/``R_OFF`` idiom: two containers
        holding equal copies drift the moment one is edited."""
        declared = type(OP.make_occupation_params(model))._PARAMS
        # Not "most of the model" but *all* of it: Zu & Mandelbaum fitted every
        # parameter of these three jointly, so there is nothing left for this
        # package to have an opinion about.
        assert set(declared) <= set(GalaxyParams._PARAMS), (
            f"{model} declares {sorted(set(declared) - set(GalaxyParams._PARAMS))} "
            f"which GalaxyParams does not, so those would need a bound of "
            f"their own")
        for name in declared:
            assert declared[name] is GalaxyParams._PARAMS[name], (
                f"{model}.{name} is a copy of GalaxyParams' Param, not the "
                f"same object; they will drift")

    def test_every_parameter_carries_a_unit_or_says_it_is_dimensionless(self):
        for model in ALL_MODELS:
            for name, p in type(OP.make_occupation_params(model))._PARAMS.items():
                assert isinstance(p.unit, str)
                assert p.kind in BOUND_KINDS


class TestTheVocabularyIsComplete:
    """A model added without bounds must fail loudly, and at import."""

    def test_every_registry_name_resolves(self):
        missing = []
        for model in ALL_MODELS:
            names = DEFAULTS.get(model) or CLF_DEFAULTS[model]
            for name in names:
                if (model, name) not in OP.PER_MODEL and name not in OP.VOCABULARY:
                    missing.append(f"{model}.{name}")
        assert not missing, f"no bound declared for {missing}"

    def test_an_undeclared_name_raises_and_says_what_to_do(self):
        with pytest.raises(KeyError, match="no bound is declared"):
            OP._bound_for("zheng07", "a_parameter_nobody_declared", 1.0)

    def test_an_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown occupation or CLF model"):
            OP.make_occupation_params("nope")
        with pytest.raises(ValueError, match="unknown occupation or CLF model"):
            OP.params_for("nope")

    def test_the_vocabulary_is_smaller_than_the_slots_it_covers(self):
        """The point of a vocabulary: one reasoned envelope per *quantity*,
        applied consistently, rather than one judgement per slot."""
        slots = sum(len(DEFAULTS.get(m) or CLF_DEFAULTS[m]) for m in ALL_MODELS)
        assert slots > 150
        assert len(OP.VOCABULARY) < slots / 2


class TestTheThreeCollisionsAreResolved:
    r"""Three names meaning two things each -- found, then renamed.

    Declaring a bound per name is what made them visible: a single name cannot
    take two envelopes, so the moment every parameter needed one, three
    quantities sharing a spelling stopped being a curiosity and became a
    decision.  ``PLAN.md`` items **G5** (which found the first) and **G9**.

    Each is asserted from **both** ends -- the new name present and the old one
    gone -- because a rename that leaves the old spelling reachable somewhere is
    a rename that has not happened.
    """

    def test_the_clf_faint_end_slope_is_no_longer_the_satellite_index(self):
        """``alpha_sat`` was a *positive* satellite power-law index in the
        occupations, published at (0.5, 1.5), and a *negative* Schechter
        faint-end slope in both CLFs.  ``GALAXY_MODELS`` merges the two
        registries, so one lookup reached both."""
        for model in ("cacciato09", "vandenbosch13"):
            names = set(CLF_DEFAULTS[model])
            assert "alpha_faint" in names
            assert "alpha_sat" not in names
            p = type(OP.make_occupation_params(model))._PARAMS["alpha_faint"]
            assert p.default < 0.0 and p.bounds[1] <= 0.0
            assert "faint-end" in p.why

        occ = type(OP.make_occupation_params("leauthaud12"))._PARAMS["alpha_sat"]
        assert occ.default > 0.0 and occ.bounds[0] >= 0.0

    def test_the_two_satellite_normalisations_no_longer_differ_by_a_typo(self):
        """``bsat`` (Zu & Mandelbaum) against ``b_sat`` (Cacciato): one
        underscore apart, which is the trap ``width_logmstar`` was named to
        avoid.  The CLFs' is ``phi_s_amp``, named for what it normalises."""
        assert "bsat" in DEFAULTS["zumandelbaum15"]
        assert "phi_s_amp" in CLF_DEFAULTS["cacciato09"]
        assert "b_sat" not in CLF_DEFAULTS["cacciato09"]
        stripped = {n.replace("_", "") for n in
                    set(DEFAULTS["zumandelbaum15"]) | set(CLF_DEFAULTS["cacciato09"])}
        assert len(stripped) == len(
            set(DEFAULTS["zumandelbaum15"]) | set(CLF_DEFAULTS["cacciato09"])), (
            "two names in these two models differ only by punctuation")

    def test_the_two_shmr_slopes_have_two_names(self):
        """``alpha_shmr`` was +0.3 in ``guo18``/``guo19`` and -1.638 in
        ``zacharegkas25`` -- the Kravtsov et al. (2018) parameterisation rather
        than the Guo one."""
        guo = type(OP.make_occupation_params("guo18"))._PARAMS["alpha_shmr"]
        zac = type(OP.make_occupation_params(
            "zacharegkas25"))._PARAMS["alpha_shmr_k18"]
        assert guo.default > 0.0 > zac.default
        assert "alpha_shmr" not in DEFAULTS["zacharegkas25"]
        assert "Kravtsov" in zac.why

    def test_no_name_in_the_whole_registry_means_two_things(self):
        """The general form, and the one that keeps holding as models are
        added: a name shared by two models must have the same *sign* in both,
        which is the cheapest machine-checkable proxy for "same quantity"."""
        from collections import defaultdict

        signs = defaultdict(set)
        for model in ALL_MODELS:
            for name, value in (DEFAULTS.get(model)
                                or CLF_DEFAULTS[model]).items():
                if value != 0.0:
                    signs[name].add(value > 0.0)
        mixed = sorted(n for n, s in signs.items() if len(s) > 1)
        assert not mixed, (
            f"{mixed} take both signs across the registries, which is how the "
            f"three collisions G5 and G9 renamed were found")

    def test_the_module_records_them_as_renamed_rather_than_open(self):
        doc = OP.__doc__
        for name in ("alpha_faint", "phi_s_amp", "alpha_shmr_k18"):
            assert name in doc
        assert "G9" in doc


class TestTheyAreRealPytrees:
    """A container that is not registered works until someone jits the model."""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_it_flattens_to_its_parameters(self, model):
        import jax

        obj = OP.make_occupation_params(model)
        leaves = jax.tree_util.tree_leaves(obj)
        assert len(leaves) == len(type(obj)._PARAMS)
        assert issubclass(type(obj), SectorParams)

    def test_a_gradient_reaches_a_parameter(self):
        import jax

        cls = OP.OCCUPATION_PARAMS["zheng07"]
        g = jax.grad(lambda p: p.alpha ** 2)(cls())
        assert float(g.alpha) == pytest.approx(2.0 * cls().alpha)
