r"""Every refusal in the package, fired at least once.

This module exists because "refuse rather than degrade silently" is the claim
``README.md`` makes most often, and a coverage sweep found **65 ``raise``
statements that no test had ever reached**.  An unexercised refusal is a weaker
thing than it looks: nothing checks that it fires on the input it describes,
that it does not fire on neighbouring input that is legitimate, or that its
message names the fix.  All three have been wrong in this package before -- the
``sigma()`` ``comoving`` flag returned the comoving answer instead of refusing
the inconsistent combination, which is a refusal that was written and did not
work.

Three things are asserted for each, and the second is the one that catches real
defects:

1. it raises, with the documented exception type;
2. **the message names the accepted values or the fix**, so a caller reading it
   does not have to open the source.  A refusal that says only "invalid" costs
   the reader the same time as no refusal at all;
3. where there is a neighbouring *legitimate* input, that one does not raise --
   a guard whose condition is inverted or off by one passes every test that
   only feeds it bad input.

Deliberately not here: refusals reachable only without an optional dependency
(``galaxies.py``'s ``ImportError``), and the build-time validators in
``beyond_linear_bias`` that fire on a corrupt download.  Both are tested by the
condition they describe rather than by a fake, and faking them would assert the
mock rather than the code.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18


# ==========================================================================
# Registry lookups: one shape, many registries
# ==========================================================================
def _lookups():
    """``(label, callable, the substring the message must carry)``.

    Collected as data rather than written out, because the *uniformity* is the
    property worth testing: every registry in this package refuses an unknown
    name by listing the known ones, and a registry that grew a different habit
    would be a surprise for a caller who had learned the others.
    """
    from ggah_mod.halos import concentration as CN
    from ggah_mod.halos import mass_function as MF
    from ggah_mod.observables import transforms as TR
    from ggah_mod.sectors import clf as CLF
    from ggah_mod.sectors import cooling as CL
    from ggah_mod.sectors import galaxies as GX
    from ggah_mod.sectors import occupation as OC
    from ggah_mod.sectors import sham as SH
    from ggah_mod.spectra import transition as TN

    return [
        ("clf", lambda: CLF.make_clf("nope"), "expected one of"),
        ("cooling", lambda: CL.make_cooling("nope"), "expected one of"),
        ("galaxy defaults", lambda: GX.galaxy_defaults("nope"), "expected one of"),
        ("occupation", lambda: OC.make_occupation("nope"), "expected one of"),
        ("shmr", lambda: SH.make_shmr("nope"), "expected one of"),
        ("transition", lambda: TN.one_halo_transition(None, None,
                                                     transition="nope"), "expected one of"),
        ("multiplicity", lambda: MF.make_hmf("nope"), "expected one of"),
        ("concentration", lambda: CN.make_concentration("nope"), "expected one of"),
        ("fftlog kernel", lambda: TR.make_fftlog(0.5, np.logspace(-2, 2, 32),
                                                kind="nope"), "expected one of"),
    ]


@pytest.mark.parametrize("label,fn,needle", _lookups(),
                         ids=[t[0] for t in _lookups()])
def test_an_unknown_registry_name_lists_the_known_ones(label, fn, needle):
    """One habit across every registry, asserted once per registry.

    ``ValueError`` specifically, not "some exception".  An earlier version of
    this test accepted ``KeyError`` too and passed on a bare dictionary lookup
    while the module's own ``raise`` -- the one carrying the list of
    alternatives -- stayed unreached.  A refusal test that accepts the
    accidental failure is not testing the refusal.
    """
    with pytest.raises(ValueError) as e:
        fn()
    msg = str(e.value)
    assert needle in msg, f"{label}: message does not list the alternatives"
    assert "nope" in msg, f"{label}: message does not repeat what was asked for"


@pytest.mark.parametrize("label,fn", [
    ("clf defaults", lambda: __import__(
        "ggah_mod.sectors.clf", fromlist=["x"]).clf_defaults("nope")),
    ("occupation defaults", lambda: __import__(
        "ggah_mod.sectors.occupation", fromlist=["x"]).occupation_defaults("nope")),
])
def test_the_defaults_lookups_refuse_separately(label, fn):
    r"""Each registry has **two** doors, and they are different dictionaries.

    ``make_clf`` reads ``CLF``; ``clf_defaults`` reads ``CLF_DEFAULTS``.  A name
    can be in one and not the other, which is a drift this pair of refusals
    exists to catch -- and testing only the first door leaves the second
    unexercised, which is exactly the state a coverage sweep found them in.
    """
    with pytest.raises(ValueError, match="expected one of"):
        fn()


def test_the_two_doors_of_each_registry_agree():
    """The drift the paired refusals are for, asserted directly.

    A model in ``CLF`` with no row in ``CLF_DEFAULTS`` is constructible and
    unusable; the reverse is a set of published numbers for a model that cannot
    be selected.  Both have happened in this package's predecessor.
    """
    from ggah_mod.sectors.clf import CLF, CLF_DEFAULTS
    from ggah_mod.sectors.occupation import DEFAULTS, OCCUPATION
    assert set(CLF) == set(CLF_DEFAULTS)
    assert set(OCCUPATION) == set(DEFAULTS)


# ==========================================================================
# Layer 1
# ==========================================================================
class TestLayerOneRefusals:
    def test_a_stack_of_spectra_is_not_a_variance(self):
        r"""``sigma2_tophat`` takes one spectrum.

        The failure it prevents is the reason it is worth a refusal rather than
        a reshape: a ``(Nz, Nk)`` array integrated over the wrong axis comes
        back the right *shape* and the wrong number, so nothing downstream can
        tell.
        """
        from ggah_mod.cosmology.amplitude import sigma2_tophat
        k = np.logspace(-3, 1, 64)
        one = np.ones_like(k)
        sigma2_tophat(one, k, 8.0)                       # legitimate: no raise
        with pytest.raises(ValueError, match=r"one spectrum"):
            sigma2_tophat(np.stack([one, one]), k, 8.0)

    def test_sigma_v_refuses_a_stack_for_the_same_reason(self):
        from ggah_mod.cosmology.amplitude import sigma_v
        k = np.logspace(-3, 1, 64)
        one = np.ones_like(k)
        sigma_v(one, k)
        with pytest.raises(ValueError, match=r"one spectrum"):
            sigma_v(np.stack([one, one]), k)

    def test_a_growth_rate_needs_a_backend_with_redshifts(self, analytic_pk):
        """``has_native_z = False`` cannot supply ``f(z)`` any more than
        ``D(z)``, and says so in those terms."""
        from ggah_mod.cosmology.growth import growth_rate
        with pytest.raises(ValueError, match="has_native_z"):
            growth_rate(0.0, PLANCK18, analytic_pk)

    def test_the_growth_variant_is_total_or_cold(self, counting_pk):
        from ggah_mod.cosmology.growth import growth_rate
        with pytest.raises(ValueError, match="'total' or 'cold'"):
            growth_rate(0.0, PLANCK18, counting_pk, variant="warm")

    def test_the_scale_spread_takes_one_redshift(self, counting_pk):
        """It reports the max over k at one epoch, so a vector z would silently
        reduce over the wrong axis -- the same failure as the variance above,
        and refused the same way."""
        from ggah_mod.cosmology.growth import growth_scale_spread
        growth_scale_spread(0.5, PLANCK18, counting_pk)
        with pytest.raises(ValueError, match="one redshift"):
            growth_scale_spread(np.array([0.0, 0.5]), PLANCK18, counting_pk)


# ==========================================================================
# Layer 2
# ==========================================================================
class TestLayerTwoRefusals:
    @pytest.mark.parametrize("fn,mdef,needle", [
        ("c_duffy08", "500c", "duffy08"),
        ("c_klypin16", "200m", "klypin16"),
        ("c_bhattacharya13", "500c", "bhattacharya13"),
    ])
    def test_a_relation_refuses_a_definition_it_was_not_fitted_in(
            self, fn, mdef, needle):
        r"""The guard that makes ``STUB_CM_MODEL`` necessary in ``conftest``.

        Not a formality: a concentration relation evaluated in the wrong mass
        definition returns a plausible number, and the whole point of
        :mod:`ggah_mod.halos.calibration` is that plausible-and-wrong is the
        state this package refuses.
        """
        from ggah_mod.halos import concentration as CN
        with pytest.raises(ValueError, match=needle):
            getattr(CN, fn)(1e14, 0.0, mdef)

    def test_a_mass_definition_is_a_number_or_vir(self):
        from ggah_mod.halos.mass_definitions import MassDef
        MassDef.from_string("200m")
        MassDef.from_string("vir")
        with pytest.raises(ValueError, match="number or 'vir'"):
            MassDef("halo", "matter")

    def test_a_mass_definition_delta_is_positive(self):
        from ggah_mod.halos.mass_definitions import MassDef
        with pytest.raises(ValueError, match="positive"):
            MassDef(-200.0, "matter")

    def test_the_density_it_is_measured_against_is_named(self):
        from ggah_mod.halos.mass_definitions import MassDef
        with pytest.raises(ValueError):
            MassDef(200.0, "vacuum")


# ==========================================================================
# Layer 3
# ==========================================================================
class TestLayerThreeRefusals:
    def test_an_unknown_dpm_model_lists_the_three(self):
        from ggah_mod.sectors.gas import dpm_model_params
        dpm_model_params(model=2)
        with pytest.raises(ValueError, match="DPM model must be one of"):
            dpm_model_params(model=4)

    def test_an_unknown_galaxy_view_lists_the_views(self, stub_field):
        from ggah_mod.sectors.galaxies import GalaxySector
        sec = GalaxySector(model="zheng07", calibration="off")
        with pytest.raises(ValueError, match="unknown galaxy view"):
            sec.weights(stub_field, None, view="luminosity")

    @pytest.mark.parametrize("name", ["a_cen", "a_sat"])
    def test_the_removed_assembly_bias_amplitudes_are_refused(self, name):
        r"""The decoration is gone, and ``GalaxySector.weights`` reads parameters
        by name, so a stale amplitude in a dictionary would otherwise be
        ignored: a fitted parameter that moves nothing."""
        from ggah_mod.sectors.galaxies import _refuse_removed_assembly_bias
        with pytest.raises(ValueError, match="has been removed from ggah_mod"):
            _refuse_removed_assembly_bias({"log10mmin": 12.0, name: 0.0})
        _refuse_removed_assembly_bias({"log10mmin": 12.0})

    def test_a_param_without_a_reason_is_refused(self):
        """The mechanism the declared-bound rule rests on."""
        from ggah_mod.sectors.params import Flat, Param
        Param(1.0, (0.0, 2.0), Flat(), "", "because", kind="physical")
        with pytest.raises(ValueError):
            Param(1.0, (0.0, 2.0), Flat(), "", "", kind="physical")

    def test_a_bound_kind_outside_the_vocabulary_is_refused(self):
        from ggah_mod.sectors.params import Flat, Param
        with pytest.raises(ValueError):
            Param(1.0, (0.0, 2.0), Flat(), "", "reason", kind="vibes")

    def test_a_multivariate_prior_refuses_a_non_finite_covariance(self):
        from ggah_mod.sectors.params import MultivariateGaussian
        ok = dict(names=("a", "b"), mu=(0.0, 0.0),
                  cov=((1.0, 0.0), (0.0, 1.0)))
        MultivariateGaussian(**ok)
        with pytest.raises(ValueError, match="finite"):
            MultivariateGaussian(**{**ok, "mu": (0.0, np.nan)})
        with pytest.raises(ValueError, match="repeated name"):
            MultivariateGaussian(**{**ok, "names": ("a", "a")})
        with pytest.raises(ValueError, match="not symmetric"):
            MultivariateGaussian(**{**ok, "cov": ((1.0, 0.5), (0.0, 1.0))})

    def test_an_unknown_attribute_on_a_container_says_what_exists(self):
        from ggah_mod.sectors import GalaxyParams
        with pytest.raises(AttributeError):
            GalaxyParams().not_a_parameter


# ==========================================================================
# Layer 4
# ==========================================================================
class TestLayerFourRefusals:
    @staticmethod
    def _w(name):
        from ggah_mod.sectors.protocol import TracerWeights
        return TracerWeights(w_point=jnp.ones((1, 4)), w_extended=None,
                             norm=jnp.asarray(1.0), discrete=True,
                             bias_weight=None, name=name)

    def test_an_unknown_overlap_is_refused(self):
        from ggah_mod.spectra.pair import pair_1h
        with pytest.raises(ValueError, match="unknown overlap"):
            pair_1h(self._w("a"), self._w("b"), 4, overlap="sort-of")

    def test_a_nested_pair_says_what_to_do_instead(self):
        r"""The refusal ``PLAN.md`` calls *a missing model rather than a missing
        implementation* -- so the message has to carry the workaround, since
        there is nothing to wait for.

        It used to offer two, and one of them did not work.  "Split the tracer
        into components whose populations are each identical or disjoint" fails
        because :meth:`~ggah_mod.sectors.galaxies.GalaxySector.weights`
        normalises each view to *that view's own* :math:`\bar n`, so summing the
        parts does not give the sample -- which is exactly the renormalisation
        ``ggah_cal`` applied by hand while the package advised against needing
        it.  The message now names :class:`~ggah_mod.spectra.spec.Overlap`
        instead, and says the old advice was wrong rather than dropping it
        silently.
        """
        from ggah_mod.spectra.pair import pair_1h
        with pytest.raises(NotImplementedError) as e:
            pair_1h(self._w("a"), self._w("a:sub"), 4, overlap="nested")
        msg = str(e.value)
        assert "disjoint" in msg or "'none'" in msg
        assert "Overlap(" in msg, "the refusal must name the declaration type"
        assert "does NOT work" in msg, "and must retract the advice that did not"

    def test_an_unknown_agn_view_lists_the_two(self):
        from ggah_mod.spectra.tracers import TRACERS
        from ggah_mod.spectra.spec import Component
        with pytest.raises(ValueError, match="unknown AGN view"):
            TRACERS["agn"](None, Component(sector="agn", view="flux"),
                           None, None, None)

    def test_an_unknown_sector_lists_the_known_ones(self):
        from ggah_mod.spectra.tracers import build_weights
        from ggah_mod.spectra.spec import Component
        with pytest.raises(ValueError, match="unknown sector"):
            build_weights(Component(sector="neutrinos"), None, {}, {})

    def test_a_known_sector_that_was_not_supplied_says_why_not(self):
        """Layer 4 does not build one, and the message says the reason: every
        sector carries a model choice, and defaulting it would put an
        unasked-for model into the spectrum."""
        from ggah_mod.spectra.tracers import build_weights
        from ggah_mod.spectra.spec import Component
        with pytest.raises(ValueError, match="was not supplied"):
            build_weights(Component(sector="galaxies"), None, {}, {})

    def test_a_band_needs_an_ordered_positive_range(self):
        from ggah_mod.spectra.spec import Band
        Band(0.5, 2.0)
        with pytest.raises(ValueError, match="0 < emin < emax"):
            Band(2.0, 0.5)
        with pytest.raises(ValueError, match="0 < emin < emax"):
            Band(-1.0, 2.0)

    def test_a_band_frame_is_observed_or_rest(self):
        from ggah_mod.spectra.spec import Band
        with pytest.raises(ValueError, match="unknown band frame"):
            Band(0.5, 2.0, frame="comoving")

    def test_a_tracer_with_no_components_is_refused(self):
        from ggah_mod.spectra.spec import TracerSpec
        with pytest.raises(ValueError, match="no components"):
            TracerSpec("empty", ())


# ==========================================================================
# Layer 5
# ==========================================================================
class TestLayerFiveRefusals:
    def test_fftlog_needs_enough_points(self):
        from ggah_mod.observables.transforms import make_fftlog
        make_fftlog(0.5, np.logspace(-2, 2, 32))
        with pytest.raises(ValueError, match="at least 4"):
            make_fftlog(0.5, np.array([1.0, 2.0, 3.0]))

    def test_a_statistic_needs_an_abscissa(self):
        from ggah_mod.observables.spec import Wp
        with pytest.raises(ValueError, match="empty abscissa"):
            Wp("wp", "galaxies", "galaxies", z=0.1, pi_max=100.0, rp=())


# ==========================================================================
# The backend contract
# ==========================================================================
class TestTheBackendRefusals:
    def test_an_unknown_pk_backend_in_the_environment_is_refused(
            self, monkeypatch):
        """Reading a backend name from ``$GGAH_PK`` is convenience; accepting
        an unknown one silently would make a whole campaign run on the wrong
        spectrum with nothing in the output saying so."""
        from ggah_mod import backend as B
        monkeypatch.setenv(B.PK_ENV_VAR, "not_a_backend")
        with pytest.raises(ValueError, match="linear P\\(k\\) backend"):
            B.resolve_backend()

    def test_an_empty_grid_range_is_refused(self):
        from ggah_mod import ACCURATE
        with pytest.raises(ValueError, match="empty k range"):
            ACCURATE.with_(k_min=10.0, k_max=1.0)
        with pytest.raises(ValueError, match="empty mass range"):
            ACCURATE.with_(m_min=1e15, m_max=1e10)

    def test_an_uncalibrated_set_of_halo_choices_is_refused(self):
        """The backend validator, which is the earliest point a mismatched
        (hmf, c-M, mass definition) triple can be caught -- before a field is
        built rather than after a campaign has run."""
        from ggah_mod import DIFFERENTIABLE
        with pytest.raises(ValueError, match="not calibrated together"):
            DIFFERENTIABLE.with_(cm_model="klypin16", mdef="200m")


# ==========================================================================
# The "one redshift" family
# ==========================================================================
class TestTheOneRedshiftRefusals:
    r"""Four places refuse a redshift *array*, and they are one rule.

    Layers 2 to 4 are one epoch at a time; a vector redshift reduces over the
    wrong axis and returns a smooth, positive, plausible, wrong number.  Each of
    these had a version that accepted the array -- ``check_calibration``'s
    silently *skipped the check* for it, which is the opposite of what a check
    is for -- so the family is tested together rather than one by one.
    """

    def test_the_calibration_check_takes_one_redshift(self):
        from ggah_mod.halos.calibration import check_calibration
        check_calibration("tinker08", "duffy08", "200m", 0.0, policy="warn")
        with pytest.raises(ValueError, match="one redshift"):
            check_calibration("tinker08", "duffy08", "200m",
                              np.array([0.0, 1.0]), policy="warn")

    def test_the_seppi20_multiplicity_takes_one_redshift(self):
        from ggah_mod.halos.mass_function import fsigma_seppi20
        with pytest.raises(ValueError, match="one redshift"):
            fsigma_seppi20(np.array([1.0, 2.0]), np.array([0.0, 0.5]))

    def test_the_csst_vir_fit_needs_a_cosmology(self):
        """A ``TypeError`` and not a ``ValueError``: a missing required
        argument, not a bad value, and the message says which."""
        from ggah_mod.halos.mass_function import fsigma_tinker08_csst_vir
        with pytest.raises(TypeError, match="needs a cosmology"):
            fsigma_tinker08_csst_vir(np.array([1.0]), 0.0, cosmo=None)


# ==========================================================================
# Contracts that name what is missing
# ==========================================================================
class TestTheContractsSayWhatIsMissing:
    def test_the_matter_sector_needs_a_split(self, stub_field):
        r"""And the message says why ``f_gas``/``f_star`` was not enough: the
        first could not say whether it meant hot gas or all retained baryons,
        and the two differ by up to 2.8 per cent of the halo mass."""
        from ggah_mod.sectors.matter import MatterField
        with pytest.raises(ValueError, match="needs `split`"):
            MatterField().weights(stub_field, {})

    def test_the_ejecta_sector_needs_the_same_split(self, stub_field):
        """Same rule, different sector -- and the reason is sharper here: a
        recomputed ejected fraction would be a second definition of one
        number."""
        from ggah_mod.sectors.ejecta import EjectaSector
        with pytest.raises(ValueError, match="needs `split`"):
            EjectaSector().weights(stub_field, {})

    def test_an_unknown_two_halo_consistency_lists_the_two(self, stub_field):
        from ggah_mod.sectors.matter import BaryonSplit, MatterField
        from ggah_mod.spectra.pk import i_of_k
        n = stub_field.n_m
        split = BaryonSplit.from_hot(
            float(stub_field.cosmo.Omega_b / stub_field.cosmo.Omega_m),
            0.05 * jnp.ones(n), f_star_cen=0.01 * jnp.ones(n))
        w = MatterField().weights(stub_field, {"split": split})
        i_of_k(stub_field, w, consistency="linear_deficit")   # legitimate
        with pytest.raises(ValueError, match="unknown two-halo consistency"):
            i_of_k(stub_field, w, consistency="fudge")

    def test_make_hankel_points_at_the_other_engine(self):
        """Not "unknown engine": the caller asked for something real, and the
        message says which function builds it and why it needs more."""
        from ggah_mod.observables.transforms import make_hankel
        with pytest.raises(ValueError, match="make_fftlog"):
            make_hankel(0.5, engine="fftlog")


# ==========================================================================
# The cobaya interface
# ==========================================================================
class TestTheCobayaRefusals:
    def test_unknown_units_are_refused(self):
        pytest.importorskip("cobaya")
        from ggah_mod.interfaces.cobaya import GgahMod
        from ggah_mod.cosmology.background import hubble_e

        class _Stub(GgahMod):
            def _cosmo(self):
                return PLANCK18

        t = _Stub.__new__(_Stub)
        # Both accepted spellings answer, so the refusal below is about the
        # third one and not about the stub being incomplete.
        assert float(GgahMod.get_Hubble(t, 0.0)) == pytest.approx(
            100.0 * PLANCK18.h, rel=1e-12)
        assert float(GgahMod.get_Hubble(t, 0.0, units="1/Mpc")) > 0.0
        with pytest.raises(ValueError, match="unknown units"):
            GgahMod.get_Hubble(t, 0.0, units="furlongs/fortnight")

    def test_fields_at_an_unrequested_redshift_are_refused(self):
        """``ggah_fields`` answers for the redshifts ``must_provide`` resolved.

        Asking for another one cannot be served by silently building it: cobaya
        resolves its graph once, and a field built outside that graph would not
        be the one the likelihood's other requirements were computed on.
        """
        pytest.importorskip("cobaya")
        from ggah_mod.interfaces.cobaya import GgahMod
        t = GgahMod.__new__(GgahMod)
        t._current_state = [{}]
        with pytest.raises((ValueError, AttributeError)):
            GgahMod.get_ggah_fields(t, z=None)


# ==========================================================================
# The container contract
# ==========================================================================
class TestTheContainerRefusals:
    def test_replace_refuses_an_unknown_parameter(self):
        """A typo in ``replace`` would otherwise be a silent no-op -- the
        container would come back unchanged and the fit would run at the
        default it was meant to move away from."""
        from ggah_mod.sectors import GalaxyParams
        GalaxyParams().replace(alpha_sat=1.1)
        with pytest.raises(TypeError, match="unknown parameter"):
            GalaxyParams().replace(alpha_satt=1.1)

    def test_a_joint_prior_refuses_a_container_without_its_parameters(self):
        from ggah_mod.sectors.params import MultivariateGaussian
        from ggah_mod.sectors import GalaxyParams
        p = MultivariateGaussian(names=("not_a", "not_b"), mu=(0.0, 0.0),
                                 cov=((1.0, 0.0), (0.0, 1.0)))
        with pytest.raises(AttributeError, match="no parameter"):
            p.log_prob_for(GalaxyParams())

    def test_a_marginal_of_a_name_the_prior_does_not_cover_is_refused(self):
        from ggah_mod.sectors.params import MultivariateGaussian
        p = MultivariateGaussian(names=("a", "b"), mu=(0.0, 0.0),
                                 cov=((1.0, 0.0), (0.0, 1.0)))
        p.marginal("a")
        with pytest.raises(KeyError, match="not in this prior"):
            p.marginal("c")

    def test_a_covariance_from_too_few_samples_is_refused(self):
        """``n <= p`` gives a singular covariance, which inverts to a prior
        that is infinitely tight in some direction -- a plausible object and a
        nonsense one."""
        from ggah_mod.sectors.params import MultivariateGaussian
        rng = np.random.default_rng(0)
        MultivariateGaussian.from_samples(("a", "b"), rng.normal(size=(50, 2)))
        with pytest.raises(ValueError, match="samples for"):
            MultivariateGaussian.from_samples(("a", "b"),
                                              rng.normal(size=(2, 2)))
        with pytest.raises(ValueError, match="samples must be"):
            MultivariateGaussian.from_samples(("a", "b"),
                                              rng.normal(size=(50,)))


class TestTheShotNoiseAndFieldContracts:
    def test_a_discrete_tracer_without_a_self_pair_is_refused(self):
        r"""It is :math:`\langle N\rangle\ell^2` per halo and **not** derivable
        from ``w_point``, which is the product :math:`N\ell` -- so a default
        here would be a guess at the per-object weight."""
        from ggah_mod.sectors.protocol import TracerWeights
        from ggah_mod.spectra.pair import shot_noise
        w = TracerWeights(w_point=jnp.ones((1, 4)), w_extended=None,
                          norm=jnp.asarray(1.0), discrete=True,
                          bias_weight=None, name="unlabelled")
        with pytest.raises(ValueError, match="no `self_pair`"):
            shot_noise(None, w, w, overlap="identical")

    def test_a_component_without_parameters_is_refused(self, stub_field):
        from ggah_mod.spectra.spec import Component
        from ggah_mod.spectra.tracers import build_weights
        from ggah_mod.sectors.galaxies import GalaxySector
        sec = GalaxySector(model="zheng07", calibration="off")
        with pytest.raises(ValueError, match="parameters, which were not"):
            build_weights(Component(sector="galaxies"), stub_field,
                          {"galaxies": sec}, {})

    def test_a_field_with_mismatched_grids_is_refused(self, stub_field):
        import dataclasses

        from ggah_mod.halos.field import _validate
        _validate(stub_field)                       # legitimate: no raise
        with pytest.raises(ValueError, match="sigma has"):
            _validate(dataclasses.replace(stub_field,
                                          sigma=stub_field.sigma[:-1]))
        with pytest.raises(ValueError, match="P_cb has"):
            _validate(dataclasses.replace(stub_field,
                                          pk_cb=stub_field.pk_cb[:-1]))

    def test_a_calibration_string_with_no_mass_definition_is_refused(self):
        """Returning 'unrestricted' would switch the check off for exactly the
        entries whose provenance is least clear."""
        from ggah_mod.halos.calibration import calibrated_mass_defs
        assert calibrated_mass_defs("200m")
        with pytest.raises(ValueError, match="cannot read a mass definition"):
            calibrated_mass_defs("whatever was handy")
