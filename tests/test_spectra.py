r"""Verification: layer 4, the one integral.

Four things are checked that the predecessor's eight hand-written copies could
not be:

**The large-scale limits are closed form.**  :math:`P_{mm}^{2h}(k\to0)/P_{cb}`
must be :math:`(\bar\rho_{cb}/\bar\rho_m)^2 = (1-f_\nu)^2` -- **not** 1, because
:math:`b(M)` is defined against the cold field -- and :math:`P_{mm}^{1h}(k\to0)`
must be :math:`\int dM\,(dn/dM)(M/\bar\rho_m)^2`.  Both come out to 1e-9.

**The central/satellite decomposition is exact.**  ``P_gg = (n_c^2 P_cc +
2n_cn_s P_cs + n_s^2 P_ss)/n_g^2`` holds to 7e-16.  That is a strong statement
about the self-pair rule specifically: it can only hold if the discrete
exclusion is applied consistently across all four spectra, and it is the check
the predecessor's ``P_gg`` could not pass against its own ``P_gX``.

**The two-halo term needs the counterterm, by a factor of two.**  Measured, not
argued: without it :math:`P_{mm}^{2h}(k\to0)/P_{cb}` is 0.4883 where it should
be 0.9909.

**The overlap is declared.**  Crossing two unlabelled discrete tracers raises;
a nested pair raises; siblings are disjoint.
"""
import ast
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import (
    AgnParams, AgnSector, BaryonSplit, GalaxySector, HotGasDPM, MatterField,
    DpmParams, galaxy_defaults,
)
from ggah_mod.spectra.spec import Component, TracerSpec, overlap_of

pytestmark = pytest.mark.slow

#: `zumandelbaum15`'s own SHMR parameters, so a stellar mass asked of the
#: galaxy sector is the one its paper defines.
ZU15 = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0)


@pytest.fixture(scope="module")
def galaxies():
    # A threshold occupation: the AGN chain reads its stellar masses.
    return GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)


@pytest.fixture(scope="module")
def sectors(galaxies):
    return {"galaxies": galaxies, "gas": HotGasDPM(backend=DIFFERENTIABLE),
            "agn": AgnSector(galaxies), "matter": MatterField()}


@pytest.fixture(scope="module")
def gal_params():
    return galaxy_defaults("zumandelbaum15")


@pytest.fixture(scope="module")
def params(field, galaxies, gal_params):
    f_cen, _ = galaxies.stellar_fraction(field, gal_params)
    return {"galaxies": gal_params, "gas": DpmParams(),
            "agn": AgnParams(),
            "matter": {"split": BaryonSplit.from_hot(
                PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(field.n_m, 0.1), f_star_cen=f_cen)}}


@pytest.fixture(scope="module")
def bare_matter(field, gal_params):
    """No gas, no stars: ``W_m = (M/rho_m) u_NFW`` exactly, which is the only
    state the closed forms below are stated in."""
    return {"matter": {"split": BaryonSplit.from_hot(
                0.0, jnp.zeros(field.n_m))},
            "galaxies": gal_params, "gas": DpmParams(),
            "agn": AgnParams()}


#: ``one_halo_transition="none"`` is explicit and load-bearing, not a leftover.
#: Most of this module tests the *integral* -- its closed-form k -> 0 limits, its
#: symmetry, its pair rule -- and those are statements about
#: ``int dM (dn/dM) W_a W_b``, which the transition multiplies afterwards.
#: Leaving it at the shipped default would make every one of them a test of the
#: damping instead, passing or failing for a reason that has nothing to do with
#: what it is named after.  ``TestTheOneHaloTransition`` tests the default.
#:
#: ``neutrino_two_halo="none"`` and ``bnl=False`` for the same reason: this
#: module's closed forms are statements about the cold halo integral, and the
#: neutrino leg and the beyond-linear term are additions to it with classes of
#: their own (``TestTheNeutrinoLeg``, ``TestTheBeyondLinearBiasHook``).
OPTS = SP.PkOptions(two_halo_spectrum="cb", two_halo_consistency="linear_deficit",
                    one_halo_transition="none", neutrino_two_halo="none",
                    bnl=False)
#: The same integral with the neutrino leg on -- the shipped treatment.
NU_OPTS = SP.PkOptions(two_halo_spectrum="cb",
                       two_halo_consistency="linear_deficit",
                       one_halo_transition="none", neutrino_two_halo="linear",
                       bnl=False)


class TestTheClosedFormLimits:
    r"""What :math:`k \to 0` must give, from the algebra alone."""

    def test_the_two_halo_term_tends_to_one_minus_f_nu_squared(
            self, field, sectors, bare_matter):
        r"""**Not** 1.  ``int dM (dn/dM) b M = rho_cb`` -- cold, because that is
        what the mass function integrates to -- while ``W_m`` normalises on
        ``rho_m``.  So the limit is ``(rho_cb/rho_m)^2 = (1-f_nu)^2``, and
        ``P_mm^2h -> (1-f_nu)^2 P_cb`` is the right cold-halo-model answer
        rather than an error to remove."""
        p = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=OPTS)
        got = float(p.two_halo[0] / field.pk_cb[0])
        want = float(PLANCK18.rho_cold / PLANCK18.rho_matter) ** 2
        assert got == pytest.approx(want, rel=1e-6)
        assert want == pytest.approx((1.0 - float(PLANCK18.f_nu)) ** 2, rel=1e-9)

    def test_the_one_halo_term_is_the_second_moment(
            self, field, sectors, bare_matter):
        p = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=OPTS)
        want = float(field.integrate(field.dndm * (field.m / field.rho_matter) ** 2))
        assert float(p.one_halo[0]) == pytest.approx(want, rel=1e-6)

    def test_the_spectrum_is_symmetric(self, field, sectors, params):
        a = SP.spectrum(field, "galaxies", "matter", sectors, params, options=OPTS)
        b = SP.spectrum(field, "matter", "galaxies", sectors, params, options=OPTS)
        assert np.allclose(np.asarray(a.total), np.asarray(b.total), rtol=1e-14)

    def test_the_shot_noise_of_number_counts_is_one_over_n_bar(
            self, field, sectors, params, galaxies, gal_params):
        p = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=OPTS)
        n_bar = float(galaxies.number_density(field, gal_params))
        assert float(p.shot) == pytest.approx(1.0 / n_bar, rel=1e-12)

    def test_shot_noise_is_not_in_the_total(self, field, sectors, params):
        p = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=OPTS)
        assert float(p.with_shot()[0] - p.total[0]) == pytest.approx(
            float(p.shot), rel=1e-12)


class TestThePairRule:
    """The self-pair exclusion, and the identity that pins it."""

    def test_the_central_decomposition_is_exact(
            self, field, sectors, params, galaxies, gal_params):
        r"""``P_gg = (n_c^2 P_cc + 2 n_c n_s P_cs + n_s^2 P_ss)/n_g^2``.

        Each view is normalised by its own ``n_bar``, so the weights come back
        in.  It can only hold if the discrete exclusion is applied consistently
        across all four -- which is the check the predecessor's ``P_gg`` and
        ``P_gX`` could not pass against each other.
        """
        def one(a, b):
            return np.asarray(SP.spectrum(field, a, b, sectors, params,
                                          options=OPTS).one_halo)
        n_c = float(galaxies.number_density(field, gal_params, "cen"))
        n_s = float(galaxies.number_density(field, gal_params, "sat"))
        n_g = float(galaxies.number_density(field, gal_params, "total"))
        rhs = (n_c ** 2 * one("centrals", "centrals")
               + 2 * n_c * n_s * one("centrals", "satellites")
               + n_s ** 2 * one("satellites", "satellites")) / n_g ** 2
        assert np.allclose(one("galaxies", "galaxies"), rhs, rtol=1e-12)

    def test_a_centrals_only_tracer_has_no_one_halo_term(
            self, field, sectors, params):
        """Exactly zero, not small: a Bernoulli central occupation has no
        self-pairs.  `w_extended is None` is the statement, and it survives
        into the arithmetic because it is branched on in Python.  The AGN are
        centrals-only at ``f_duty_sat = 0``, the opt-out."""
        centrals_only = dict(params, agn=AgnParams(f_duty_sat=0.0))
        for name, prm in (("centrals", params), ("agn", centrals_only)):
            p = SP.spectrum(field, name, name, sectors, prm, options=OPTS)
            assert np.all(np.asarray(p.one_halo) == 0.0)

    def test_agn_in_satellite_galaxies_have_a_one_halo_term(
            self, field, sectors, params):
        """By default the satellite galaxies host AGN too, so the AGN auto
        pairs central with satellite and satellite with satellite."""
        p = SP.spectrum(field, "agn", "agn", sectors, params, options=OPTS)
        assert np.all(np.asarray(p.one_halo) > 0.0)

    def test_a_continuous_auto_spectrum_keeps_its_point_square(
            self, field, sectors, params):
        """A field is not a countable population, so nothing is excluded and
        `(w_point + w_ext)^2` is the whole integrand."""
        p = SP.spectrum(field, "matter", "matter", sectors, params, options=OPTS)
        assert float(p.one_halo[0]) > 0.0

    def test_centrals_and_satellites_do_pair(self, field, sectors, params):
        """Siblings, so their cross-spectrum excludes nothing -- and a central
        is never a satellite, so there is nothing to exclude."""
        p = SP.spectrum(field, "centrals", "satellites", sectors, params,
                        options=OPTS)
        assert float(p.one_halo[0]) > 0.0
        assert float(p.shot) == 0.0


class TestTheCounterterm:
    r"""The two-halo normalisation deficit, measured."""

    def test_the_bias_integral_falls_short_on_the_shipped_grid(self, field):
        """0.716, and the three numbers this has been are the whole story.

        =========================================  =====
        ``tinker08`` x ``tinker10``, Delta ignored  0.702
        the same at 200c (Delta_m = 645)           0.530
        the shipped default: 200m, recalibrated    0.716
        =========================================  =====

        The first two differ because the field's mass definition reached the
        pair: the flavours were 200c, which is Delta_m = 645 against the mean
        density, while the abundance and the bias were both being evaluated at
        200 regardless.  The deficit is deeper at the denser boundary.

        The third is *not* a regression to the first.  The default is now 200m,
        where Delta_m is 200 by definition, so the correction that mattered at
        200c has nothing to correct -- and the residual gap between 0.702 and
        0.716 is the recalibrated multiplicity function rather than a
        definition error.  Measured by holding one thing at a time: 200m with
        plain ``tinker08`` gives 0.7013, and swapping in ``tinker08_csst``
        gives 0.7164.  So the definition is worth 0.17 here and the
        recalibration 0.015.
        """
        got = float(SP.bias_consistency(field))
        assert got == pytest.approx(0.716, abs=0.02)
        assert float(SP.mass_deficit(field)) == pytest.approx(1.0 - got, rel=1e-12)

    def test_without_it_the_matter_spectrum_is_short_by_a_factor_of_two(
            self, field, sectors, bare_matter):
        off = SP.PkOptions(two_halo_consistency="none",
                           neutrino_two_halo="none", bnl=False)
        p = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=off)
        got = float(p.two_halo[0] / field.pk_cb[0])
        want = (float(SP.bias_consistency(field))
                * float(PLANCK18.rho_cold / PLANCK18.rho_matter)) ** 2
        assert got == pytest.approx(want, rel=1e-3)
        assert got < 0.55                       # against 0.9909 with it on

    def test_it_is_zero_for_a_discrete_tracer(self, field, sectors, params):
        """A threshold sample does not continue below its own threshold, and a
        count is not a mass.  Zero by return, not by a branch at the call site."""
        w = sectors["galaxies"].weights(field, params["galaxies"])
        ct = SP.low_mass_counterterm(field, w, field.n_k)
        assert np.all(np.asarray(ct) == 0.0)

    def test_it_is_not_clipped_at_zero(self, field):
        """A wide enough mass range overshoots -- 1.065 at m_min = 1e6 -- and a
        negative deficit is information about the (HMF, bias) pair, not a
        number to floor.

        Clipping would be a defect and not a tidy-up: the guarantee is that
        ``I_m(k -> 0)`` is ``rho_cb/rho_m`` *identically*, and a floor at zero
        would bind in exactly the configurations where the integral overshoots,
        leaving the limit wrong by the amount it refused to subtract."""
        src = pathlib.Path(SP.counterterm.__file__).read_text()
        assert "clip" not in src.split('"""')[-1]

    # -- what depends on the mass range, and what does not (PLAN.md A2) ----
    @pytest.mark.slow
    def test_the_normalisation_does_not_depend_on_the_mass_range(self, pk):
        r"""``I_m(k -> 0) = rho_cb/rho_m`` at every ``m_min``, exactly.

        The paper asks the counterterm to "converge, or state what it is
        conditional on".  It converges, and this is the half that does: whatever
        the deficit is, putting it back at ``M_min`` with ``b = 1`` lands the
        limit on the same number.  Four decades of ``m_min``, agreeing to the
        quadrature rather than to the prescription.

        ``k_max`` moves with ``m_min`` because ``check_k_support`` requires it --
        ``sigma(M)`` needs power inside its own window -- which is itself part of
        the answer: the mass range and the k grid are not independent knobs.
        """
        from ggah_mod.sectors import BaryonSplit, MatterField

        target = float(PLANCK18.rho_cold / PLANCK18.rho_matter)
        got = {}
        for m_min, k_max in ((1e10, 200.0), (1e8, 200.0), (1e6, 800.0)):
            b = DIFFERENTIABLE.with_(m_min=m_min, k_max=k_max)
            f = make_field(PLANCK18, b, pk=pk, z=0.0, calibration="off")
            w = MatterField().weights(
                f, {"split": BaryonSplit.from_hot(0.0, jnp.zeros(f.n_m))})
            got[m_min] = (np.asarray(f.k),
                          np.asarray(SP.i_of_k(f, w,
                                               consistency="linear_deficit")))
            assert float(got[m_min][1][0]) == pytest.approx(target, rel=1e-8)

        # The deficit really is different at each -- otherwise this proves
        # nothing about the correction.
        assert float(SP.bias_consistency(
            make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e8), pk=pk, z=0.0,
                       calibration="off"))) > 0.75

        # ...and the *shape* residual is small but not zero: the smallest
        # resolved halo's profile stands in for everything below it, and a
        # smaller halo is more concentrated.
        #
        # Each grid is probed on its own k, because widening m_min forces
        # k_max up and the grids are therefore not the same nodes -- indexing
        # them with one index compares different wavenumbers, which is how this
        # test failed the first time it was run.
        def at(case, k_probe):
            k, i_of = got[case]
            return float(i_of[int(np.argmin(np.abs(k - k_probe)))])

        for k_probe, bound in ((0.1, 1e-5), (1.0, 1e-3), (5.0, 2e-3)):
            assert abs(at(1e6, k_probe) / at(1e10, k_probe) - 1.0) < bound

    # -- the bias override stops here, on purpose (PLAN.md A3) -------------
    def test_the_deficit_ignores_a_bias_override(self, field,
                                                             sectors,
                                                             bare_matter):
        """``i_of_k`` honours ``bias_weight`` and the counterterm does not, and
        that is the correct way round rather than an oversight.

        ``bias_consistency`` is the statement *the mass-weighted bias of all
        matter is one* -- a property of the (mass function, bias) pairing.  A
        decoration re-weights one tracer's bias at fixed halo mass; it does not
        change how much bias-weighted **mass** the grid is missing, and a
        decorated version of that integral has no reason to equal one.  Reading
        the override here would swap a known limit for an unknown one.
        """
        import dataclasses

        w = sectors["matter"].weights(field, bare_matter["matter"])
        assert w.bias_weight is None
        decorated = dataclasses.replace(
            w, bias_weight=field.bias * 1.5, name="matter:decorated")

        plain = np.asarray(SP.low_mass_counterterm(field, w, field.n_k))
        fancy = np.asarray(
            SP.low_mass_counterterm(field, decorated, field.n_k))
        assert np.array_equal(plain, fancy)

        # ...while the main integral does respond, which is the asymmetry.
        assert not np.allclose(
            np.asarray(SP.i_of_k(field, w, consistency="none")),
            np.asarray(SP.i_of_k(field, decorated, consistency="none")))

    def test_no_shipped_sector_can_reach_that_asymmetry(self, field, sectors,
                                                        params, bare_matter):
        """The two halves cannot meet today, and it is contingent, not
        structural: no shipped sector sets ``bias_weight`` on its own weights
        -- the PNG shift is applied by the caller -- and the counterterm is zero
        for a discrete tracer.  Pinned so that a
        sector which starts decorating a continuous tracer has to come here and
        read the paragraph above."""
        from ggah_mod.spectra.tracers import PEERS
        p = dict(params)
        p["matter"] = bare_matter["matter"]
        for name, sector in sectors.items():
            # PEERS values are tuples: the hot gas reads two sectors when it
            # carries a feedback budget, where the AGN chain reads one.
            peer = tuple(p[q] for q in PEERS.get(name, ()))
            w = sector.weights(field, p[name], *peer)
            if w.bias_weight is not None:
                assert w.discrete, (
                    f"sector {name!r} decorates a continuous tracer's bias, so "
                    f"its counterterm now silently ignores the decoration; see "
                    f"counterterm.bias_consistency for which way round is right")


class TestOverlapIsDeclaredNotGuessed:
    """Two discrete tracers must say whether they share objects."""

    def test_an_unlabelled_discrete_pair_raises(self):
        a = Component("galaxies", "total")
        b = Component("agn", "counts", population="agn")
        with pytest.raises(ValueError, match="declare no `population`"):
            overlap_of(a, b, a_discrete=True, b_discrete=True)

    def test_a_continuous_partner_needs_no_label(self):
        a = Component("gas", "pressure")
        b = Component("agn", "counts")
        assert overlap_of(a, b, a_discrete=False, b_discrete=True) == "none"

    @pytest.mark.parametrize("pa,pb,want", [
        ("galaxies/cen", "galaxies/cen", "identical"),
        ("galaxies/cen", "galaxies/sat", "none"),
        ("galaxies", "galaxies/cen", "nested"),
        ("galaxies/cen", "galaxies", "nested"),
        ("galaxies", "agn", "none"),
    ])
    def test_the_population_path_decides(self, pa, pb, want):
        a = Component("galaxies", "total", population=pa)
        b = Component("agn", "counts", population=pb)
        assert overlap_of(a, b, a_discrete=True, b_discrete=True) == want

    def test_a_nested_pair_is_refused_rather_than_guessed(
            self, field, sectors, params):
        """The rule differs per component pair, and `population` says the
        samples overlap -- not which components are shared."""
        with pytest.raises(NotImplementedError, match="nested pair"):
            SP.spectrum(field, "galaxies", "centrals", sectors, params,
                        options=OPTS)


class TestCompositeTracers:
    r"""X-ray is gas **plus** AGN, and that is three pairs, not one."""

    def test_the_xray_tracer_has_two_components(self):
        spec = SP.resolve("xray")
        assert len(spec.components) == 2
        assert {c.sector for c in spec.components} == {"gas", "agn"}

    def test_the_auto_spectrum_expands_to_three_terms(
            self, field, sectors, params):
        r"""``P_XX = P_gg + 2 P_ga + P_aa`` -- exactly what the predecessor's
        ``_pk_tables_XX`` writes out by hand, with three different pair rules
        in it."""
        gas = TracerSpec("g", (Component("gas", "xray"),))
        agn = TracerSpec("a", (Component("agn", "emission", population="agn"),))

        def one(x, y):
            return np.asarray(SP.spectrum(field, x, y, sectors, params,
                                          options=OPTS).total)
        whole = one("xray", "xray")
        parts = one(gas, gas) + 2.0 * one(gas, agn) + one(agn, agn)
        assert np.allclose(whole, parts, rtol=1e-12)

    def test_the_agn_emission_leg_is_discrete(self, field, sectors, params):
        """`forward_jax` squares `X_gas + X_agn` with the central self-pair left
        in, while `_pk_tables_XX` excludes it.  Two implementations of one
        spectrum; here the flag says which, once."""
        agn = TracerSpec("a", (Component("agn", "emission", population="agn"),))
        centrals_only = dict(params, agn=AgnParams(f_duty_sat=0.0))
        p = SP.spectrum(field, agn, agn, sectors, centrals_only, options=OPTS)
        assert np.all(np.asarray(p.one_halo) == 0.0)     # centrals only
        assert float(p.shot) > 0.0                       # but a real self-pair

    def test_its_shot_noise_is_not_one_over_n_bar(self, field, sectors, params):
        r"""``<N> l^2`` with ``l = L_X``, so ``self_pair != w_point``.  This is
        the case :attr:`TracerWeights.self_pair` exists for -- no algebra on the
        fused weight recovers it."""
        agn = TracerSpec("a", (Component("agn", "emission", population="agn"),))
        p = SP.spectrum(field, agn, agn, sectors, params, options=OPTS)
        counts = SP.spectrum(field, "agn", "agn", sectors, params, options=OPTS)
        assert not np.isclose(float(p.shot), float(counts.shot), rtol=1e-3)

    def test_a_named_band_reaches_the_sector(self, field):
        """0.9.2: the band is carried in, through the sector's own spectrum,
        and a sector without the obscured split refuses it -- in the soft
        band the split is the spectrum."""
        from ggah_mod.spectra.tracers import build_weights
        c = Component("agn", "emission", population="agn",
                      band=SP.Band(0.5, 2.0))
        gal = GalaxySector("zumandelbaum15")
        params = {"agn": AgnParams(),
                  "galaxies": galaxy_defaults("zumandelbaum15")}
        w = build_weights(c, field, {"agn": AgnSector(gal, obscuration="split")},
                          params)
        assert w.name == "agn:emission[0.5-2keV/o]" and w.discrete
        with pytest.raises(ValueError, match="obscuration='split'"):
            build_weights(c, field, {"agn": AgnSector(gal)}, params)


class TestTheObscurationBranchesAsTracers:
    r"""0.9.3: the two branch views, and the rule they pair by."""

    @pytest.fixture(scope="class")
    def split_setup(self, field):
        gal = GalaxySector("zumandelbaum15")
        sectors = {"agn": AgnSector(gal, obscuration="split"), "galaxies": gal}
        params = {"agn": AgnParams(),
                  "galaxies": galaxy_defaults("zumandelbaum15")}
        band = SP.Band(0.5, 2.0)
        whole = TracerSpec("x", (Component("agn", "emission", population="agn",
                                           band=band),))
        parts = TracerSpec("x2", (
            Component("agn", "emission_unobscured",
                      population="agn/unobscured", band=band),
            Component("agn", "emission_obscured",
                      population="agn/obscured", band=band)))
        return sectors, params, whole, parts

    @staticmethod
    def _close(p, q):
        for f in ("one_halo", "two_halo", "shot"):
            np.testing.assert_allclose(np.asarray(getattr(p, f)),
                                       np.asarray(getattr(q, f)),
                                       rtol=1e-10, atol=0)

    def test_galaxies_cross_the_two_branches_as_they_cross_the_whole(
            self, field, split_setup):
        sectors, params, whole, parts = split_setup
        a = SP.spectrum(field, "galaxies", whole, sectors, params, options=OPTS)
        b = SP.spectrum(field, "galaxies", parts, sectors, params, options=OPTS)
        self._close(a, b)

    def test_the_branch_auto_spectrum_is_the_whole_one(self, field,
                                                        split_setup):
        """A halo's central AGN is one branch or the other: with the branch
        rule the four pairs add to the unsplit auto-spectrum exactly."""
        sectors, params, whole, parts = split_setup
        a = SP.spectrum(field, whole, whole, sectors, params, options=OPTS)
        b = SP.spectrum(field, parts, parts, sectors, params, options=OPTS)
        self._close(a, b)

    def test_without_the_rule_the_centrals_would_pair(self, field,
                                                       split_setup):
        """The rule is load-bearing: declared independent instead, the two
        branches' centrals pair and the one-halo term comes out high."""
        from ggah_mod.spectra.spec import Overlap

        sectors, params, whole, parts = split_setup
        indep = Overlap(point="disjoint", shared="none",
                        why="deliberately wrong, for this test")
        a = SP.spectrum(field, whole, whole, sectors, params, options=OPTS)
        b = SP.spectrum(field, parts, parts, sectors, params, options=OPTS,
                        overlaps={("agn/unobscured", "agn/obscured"): indep})
        assert np.all(np.asarray(b.one_halo) >= np.asarray(a.one_halo))
        assert not np.allclose(np.asarray(b.one_halo), np.asarray(a.one_halo),
                               rtol=1e-6, atol=0)

    def test_the_views_are_declared(self):
        from ggah_mod.spectra.tracers import SECTOR_VIEWS
        assert {"emission_unobscured", "emission_obscured"} <= set(
            SECTOR_VIEWS["agn"])


class TestTheClosureSeesTheNeutralGasThroughLayerFour:
    """0.9.5: ``coldgas`` is an optional peer of the gas, as of the matter."""

    def test_the_spectrum_hands_the_neutral_gas_to_a_closure_that_asked(
            self, field):
        from ggah_mod.sectors.agn import AGN_PUBLISHED
        from ggah_mod.sectors.coldgas import ColdGasParams, ColdGasSector
        from ggah_mod.sectors.gas import DpmParams, HotGasDPM

        gal = GalaxySector("zumandelbaum15")
        agn = AgnSector(gal, calibration="off")
        cold = ColdGasSector("catinella18", galaxies=gal)
        base = {"agn": AgnParams(**AGN_PUBLISHED),
                "galaxies": galaxy_defaults("zumandelbaum15"),
                "gas": DpmParams()}
        with_hi = HotGasDPM(feedback="closure", agn=agn, galaxies=gal,
                            coldgas=cold)
        sectors = {"gas": with_hi, "galaxies": gal, "agn": agn,
                   "coldgas": cold}
        p = SP.spectrum(field, "galaxies", "pressure", sectors,
                        dict(base, coldgas=ColdGasParams()), options=OPTS)
        assert np.all(np.isfinite(np.asarray(p.total)))
        with pytest.raises(ValueError, match="cold-gas parameters"):
            SP.spectrum(field, "galaxies", "pressure", sectors, base,
                        options=OPTS)


class TestTheLayerKnowsNoSector:
    r""":mod:`~ggah_mod.sectors.protocol` claims layer 4 "never learns which
    sector it is talking to".  Here that is a test rather than a docstring."""

    @staticmethod
    def _identifiers(path):
        """Every name the module's *code* uses: variables, attributes, args.

        Identifiers only -- not docstrings, and not string literals.  The
        distinction is the point.  ``pair.py``'s refusal message explains that
        an X-ray AGN sits in a galaxy, and that is exactly the sort of prose
        this rule is meant to encourage; what it forbids is a code path that
        *dispatches* on which sector it is holding.
        """
        tree = ast.parse(pathlib.Path(path).read_text())
        out = set()
        for node in ast.walk(tree):
            for attr in ("id", "attr", "arg", "name"):
                value = getattr(node, attr, None)
                if isinstance(value, str):
                    out.add(value.lower())
        return out

    def test_the_integral_names_no_sector(self):
        names = " ".join(self._identifiers(SP.pk.__file__))
        for word in ("gas", "agn", "galax", "pressure", "xray", "matter",
                     "dpm", "nfw"):
            assert word not in names, (
                f"ggah_mod/spectra/pk.py has an identifier naming {word!r}. "
                f"The one integral must not know which sector it is "
                f"integrating; the adapters live in spectra/tracers.py.")

    def test_the_pair_rule_names_no_sector_either(self):
        names = " ".join(self._identifiers(SP.pair.__file__))
        for word in ("gas", "agn", "galax", "pressure", "xray", "matter"):
            assert word not in names

    def test_the_self_pair_branch_lives_in_one_module(self):
        """`protocol.py` calls `discrete` "the only physics branch in the
        layer-4 integrand".  It is read in `pair.py` -- the rule itself -- and
        in `counterterm.py`, which needs it to know that a count is not a mass.
        Nowhere else, and `pk.py` above all."""
        assert "discrete" not in self._identifiers(SP.pk.__file__)
        assert "discrete" in self._identifiers(SP.pair.__file__)
        assert "discrete" in self._identifiers(SP.counterterm.__file__)

    def test_layer_four_does_not_import_layer_five(self):
        import ggah_mod.spectra
        root = pathlib.Path(ggah_mod.spectra.__file__).parent
        for path in root.glob("*.py"):
            src = path.read_text()
            assert "observables" not in src.split('"""')[-1], path.name


class TestTheBeyondLinearBiasHook:
    r"""On by default, and its table is built from the field unless passed.

    :math:`\beta^{\rm NL}` is a two-halo correction, not a profile, so it does
    not fit the tracer abstraction and sits behind a hook in
    :mod:`ggah_mod.spectra.bnl`.  The hook was off, and refused to build a
    table, until the low-mass completion made it worth defaulting.
    """

    @pytest.fixture(scope="class")
    def table(self, field):
        from ggah_mod.halos.beyond_linear_bias import table_at
        # `check=False` is required inside a trace: the diagnostics read
        # concrete floats.
        return table_at(field.k, k=field.k, pk_cb=field.pk_cb, check=False)

    def test_it_is_on_by_default(self):
        from ggah_mod import ACCURATE
        assert SP.PkOptions().bnl is True
        for b in (ACCURATE, DIFFERENTIABLE):
            assert b.bnl is True
            assert SP.PkOptions.from_backend(b).bnl is True

    def test_the_table_built_from_the_field_is_the_explicit_one(
            self, field, sectors, params, table):
        """The rescaling reads the target through its cold spectrum alone, and
        the field carries that spectrum; building it here reaches for nothing
        the caller did not pass."""
        on = SP.PkOptions(bnl=True)
        auto = SP.spectrum(field, "galaxies", "matter", sectors, params,
                           options=on)
        given = SP.spectrum(field, "galaxies", "matter", sectors, params,
                            options=on, bnl_table=table)
        assert np.array_equal(np.asarray(auto.total), np.asarray(given.total))

    def test_off_means_off(self, field, sectors, params):
        off = SP.PkOptions(bnl=False)
        p = SP.spectrum(field, "galaxies", "matter", sectors, params, options=off)
        want = SP.pk_2h(field, *(SP.build_weights(SP.resolve(n).components[0],
                                                  field, sectors, params)
                                 for n in ("galaxies", "matter")), options=off)
        assert np.array_equal(np.asarray(p.two_halo), np.asarray(want))

    @pytest.mark.parametrize("pair", [("galaxies", "galaxies"),
                                      ("galaxies", "matter")])
    def test_it_moves_the_two_halo_term_by_a_believable_amount(
            self, pair, field, sectors, params, table):
        r"""Tens of percent on quasi-linear scales, which is what Mead & Verde
        measure; a correction that did nothing would mean the adapter's missing
        ``dM`` had silently zeroed it."""
        a, b = pair
        off = SP.spectrum(field, a, b, sectors, params,
                          options=SP.PkOptions(bnl=False))
        on = SP.spectrum(field, a, b, sectors, params,
                         options=SP.PkOptions(bnl=True), bnl_table=table)
        k = np.asarray(field.k)
        sel = (k > 0.05) & (k < 5.0)
        ratio = np.asarray(on.two_halo)[sel] / np.asarray(off.two_halo)[sel]
        assert np.all(np.isfinite(ratio))
        assert ratio.max() > 1.05          # it does something
        assert ratio.max() < 5.0           # and not something absurd

    def test_it_leaves_the_one_halo_term_alone(
            self, field, sectors, params, table):
        off = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                          options=SP.PkOptions(bnl=False))
        on = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                         options=SP.PkOptions(bnl=True), bnl_table=table)
        assert np.array_equal(np.asarray(on.one_halo), np.asarray(off.one_halo))

    def test_the_quadrature_measure_is_what_makes_the_adapter_right(self, field):
        r"""``correction_2h_*`` sums over the mass index with nothing standing
        in for ``dM``, so its documented ``dndm * N_tot * b / n_bar`` is missing
        one.  With the measure supplied, ``sum(weights * uk)`` is exactly
        ``I(k)`` -- the quantity the correction corrects."""
        from ggah_mod.spectra.bnl import effective_weight
        w = SP.spectrum  # noqa: F841  (readability only)
        from ggah_mod.sectors import matter_weights
        tw = matter_weights(field, BaryonSplit.from_hot(
            0.0, jnp.zeros(field.n_m)))
        dm = field.quadrature_measure()
        by_sum = jnp.sum((field.dndm * field.bias * dm)[None, :]
                         * effective_weight(field, tw), axis=-1)
        by_integral = SP.i_of_k(field, tw, consistency="none")
        assert np.allclose(np.asarray(by_sum), np.asarray(by_integral),
                           rtol=1e-12)



class TestTheUnresolvedMatter:
    r"""What the matter below the grid is made of, as the counterterm reads it.

    The hot-gas views declare nothing there, the ejecta sector carries every
    baryon that is not a star or neutral gas, and the matter field keeps the
    smallest-halo stand-in, which is exact for the total."""

    @pytest.fixture(scope="class")
    def electrons(self, sectors, params):
        from ggah_mod.sectors import EjectaSector
        sec = {**sectors, "ejecta": EjectaSector()}
        par = {**params, "ejecta": {"split": params["matter"]["split"]}}
        return sec, par

    @pytest.mark.parametrize("view", ["pressure", "xray", "mass", "density"])
    def test_the_hot_gas_has_no_counterterm(self, field, sectors, params, view):
        from ggah_mod.spectra.spec import Component
        w = SP.build_weights(Component("gas", view), field, sectors, params)
        assert np.all(np.asarray(w.w_unresolved) == 0.0)
        on = SP.i_of_k(field, w, consistency="linear_deficit")
        off = SP.i_of_k(field, w, consistency="none")
        assert np.array_equal(np.asarray(on), np.asarray(off))

    def test_the_ejecta_carry_the_diffuse_share(self, field, electrons):
        from ggah_mod.halos.profiles import ejected_uk
        from ggah_mod.spectra.spec import Component
        sec, par = electrons
        split = par["ejecta"]["split"]
        w = SP.build_weights(Component("ejecta", "mass"), field, sec, par)
        n_k = int(field.k.shape[0])
        got = SP.low_mass_counterterm(field, w, n_k)
        f_diffuse = float(split.f_hot[0] + split.f_ejected[0])
        want = (SP.mass_deficit(field) * f_diffuse * field.rho_cold
                * ejected_uk(field.k, field.r_delta)[:, 0])
        assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-12)

    def test_the_electrons_count_the_sub_grid_baryons_once(self, field,
                                                           electrons):
        """Every non-stellar, non-neutral baryon below the grid, and nothing
        twice: the sum of the two components' counterterms at k -> 0 is the
        deficit times that share of the cold density."""
        sec, par = electrons
        split = par["ejecta"]["split"]
        n_k = int(field.k.shape[0])
        total = sum(float(SP.low_mass_counterterm(
            field, SP.build_weights(c, field, sec, par), n_k)[0])
            for c in SP.resolve("electrons").components)
        first = lambda x: jnp.atleast_1d(jnp.asarray(x))[0]   # noqa: E731
        share = float(first(split.f_baryon) - first(split.f_star_cen)
                      - first(split.f_star_sat) - first(split.f_cold))
        want = float(SP.mass_deficit(field)) * share * float(field.rho_cold)
        assert total == pytest.approx(want, rel=1e-6)

    def test_the_matter_field_keeps_the_stand_in(self, field, sectors,
                                                 bare_matter):
        w = SP.build_weights(SP.resolve("matter").components[0], field,
                             sectors, bare_matter)
        assert w.w_unresolved is None
        n_k = int(field.k.shape[0])
        got = SP.low_mass_counterterm(field, w, n_k)
        want = (SP.mass_deficit(field) * w.total(n_k)[:, 0]
                * field.rho_cold / field.m[0])
        assert np.array_equal(np.asarray(got), np.asarray(want))

    def test_a_discrete_tracer_cannot_declare_one(self):
        from ggah_mod.sectors import TracerWeights
        with pytest.raises(ValueError, match="discrete and declares"):
            TracerWeights(w_point=jnp.ones(4), w_extended=None,
                          norm=jnp.asarray(1.0), discrete=True,
                          bias_weight=None, name="x",
                          w_unresolved=jnp.zeros(3))

    def test_combining_declared_and_undeclared_is_refused(self):
        from ggah_mod.sectors import TracerWeights, combine
        a = TracerWeights(None, jnp.ones((3, 4)), jnp.asarray(1.0), False,
                          None, "a", w_unresolved=jnp.zeros(3))
        b = TracerWeights(None, jnp.ones((3, 4)), jnp.asarray(1.0), False,
                          None, "b")
        with pytest.raises(ValueError, match="some declare w_unresolved"):
            combine(a, b)
        both = combine(a, a)
        assert np.array_equal(np.asarray(both.w_unresolved), np.zeros(3))


class TestTheLowMassCompletionOfBeyondLinearBias:
    r"""The counterterm's point mass at :math:`M_{\min}` has a beyond-linear
    term too -- Mead & Verde (2021), Eqs. A7-A10 -- and it is carried as an
    addition to the first mass column of the fused weight."""

    @pytest.fixture(scope="class")
    def table(self, field):
        from ggah_mod.halos.beyond_linear_bias import table_at
        return table_at(field.k, k=field.k, pk_cb=field.pk_cb, check=False)

    @staticmethod
    def _four_terms(field, wa, wb, table, consistency="linear_deficit"):
        """I22 + I12 + I21 + I11, written out term by term."""
        from ggah_mod.halos.beyond_linear_bias import beta_nl
        from ggah_mod.spectra.bnl import effective_weight
        from ggah_mod.spectra.counterterm import low_mass_counterterm
        n_k = int(field.k.shape[0])
        dm = field.quadrature_measure()
        grid_a = (field.dndm * field.bias * dm)[None, :] * effective_weight(field, wa)
        grid_b = (field.dndm * field.bias * dm)[None, :] * effective_weight(field, wb)
        beta = beta_nl(field.k, field.nu, field.nu, table=table)      # (Nk, NM, NM)
        i22 = jnp.einsum("ki,kij,kj->k", grid_a, beta, grid_b)
        if consistency == "none":
            return i22
        da = low_mass_counterterm(field, wa, n_k)
        db = low_mass_counterterm(field, wb, n_k)
        i12 = da * jnp.einsum("kj,kj->k", beta[:, 0, :], grid_b)
        i21 = db * jnp.einsum("ki,ki->k", beta[:, :, 0], grid_a)
        i11 = da * db * beta[:, 0, 0]
        return i22 + i12 + i21 + i11

    @pytest.mark.parametrize("pair", [("matter", "matter"),
                                      ("galaxies", "matter"),
                                      ("galaxies", "galaxies")])
    def test_it_is_the_four_terms_of_the_paper(self, pair, field, sectors,
                                               params, table):
        from ggah_mod.halos.beyond_linear_bias import correction_2h_fused
        from ggah_mod.spectra.bnl import completed_weight
        a, b = (SP.build_weights(SP.resolve(n).components[0], field, sectors,
                                 params) for n in pair)
        got = correction_2h_fused(field.nu, completed_weight(field, a),
                                  completed_weight(field, b), table)
        want = self._four_terms(field, a, b, table)
        scale = float(jnp.max(jnp.abs(want)))
        assert scale > 0.0
        assert float(jnp.max(jnp.abs(got - want))) < 1e-12 * scale

    def test_galaxies_have_no_completion_of_their_own(
            self, field, sectors, params, table):
        """A discrete tracer's counterterm is zero, so galaxy x galaxy is the
        grid double integral exactly, and galaxy x matter keeps only the
        matter leg's term."""
        g = SP.build_weights(SP.resolve("galaxies").components[0], field,
                             sectors, params)
        full = self._four_terms(field, g, g, table)
        grid = self._four_terms(field, g, g, table, consistency="none")
        assert np.array_equal(np.asarray(full), np.asarray(grid))

    def test_with_the_counterterm_off_it_is_the_old_kernel(
            self, field, sectors, params, table):
        from ggah_mod.halos.beyond_linear_bias import correction_2h_gm
        from ggah_mod.spectra.bnl import effective_weight, two_halo_correction
        off = SP.PkOptions(two_halo_consistency="none", bnl=True,
                           neutrino_two_halo="none")
        g = SP.build_weights(SP.resolve("galaxies").components[0], field,
                             sectors, params)
        m = SP.build_weights(SP.resolve("matter").components[0], field,
                             sectors, params)
        dm = field.quadrature_measure()
        w = field.dndm * field.bias * dm
        old = field.pk_cb * correction_2h_gm(
            field.nu, w, w, effective_weight(field, g),
            effective_weight(field, m), table)
        new = two_halo_correction(field, g, m, table, options=off)
        # Two contraction orders of the same algebra.  The correction changes
        # sign near the taper, where a purely relative bound reads round-off
        # as disagreement: 1.0e-12 there on the C^1 table of 0.9.7, 1.6e-16 of
        # the peak.  So round-off is bounded against the peak.
        old = np.asarray(old)
        assert np.allclose(np.asarray(new), old, rtol=1e-12,
                           atol=1e-14 * np.max(np.abs(old)))

    def test_it_is_zero_below_the_taper(self, field, sectors, bare_matter,
                                         table):
        from ggah_mod.halos.beyond_linear_bias import K_MIN_BNL, K_TAPER
        from ggah_mod.spectra.bnl import two_halo_correction
        m = SP.build_weights(SP.resolve("matter").components[0], field,
                             sectors, bare_matter)
        d = np.asarray(two_halo_correction(field, m, m, table))
        k = np.asarray(field.k)
        assert np.all(d[k < K_MIN_BNL / K_TAPER / np.asarray(table.s)] == 0.0)

    def test_it_raises_the_matter_spectrum_where_the_paper_says(
            self, field, sectors, bare_matter, table):
        """The completion adds power for matter across the transition, where
        Mead & Verde find at least half of the matter-halo improvement and
        about five times the matter-matter one below the simulation's mass
        limit.  Measured here: +20 per cent against +8 per cent at k = 0.5."""
        from ggah_mod.spectra.bnl import two_halo_correction
        m = SP.build_weights(SP.resolve("matter").components[0], field,
                             sectors, bare_matter)
        on = SP.PkOptions(bnl=True, neutrino_two_halo="none")
        grid = SP.PkOptions(bnl=True, neutrino_two_halo="none",
                            two_halo_consistency="none")
        full = np.asarray(two_halo_correction(field, m, m, table, options=on))
        bare = np.asarray(two_halo_correction(field, m, m, table, options=grid))
        k = np.asarray(field.k)
        sel = (k > 0.2) & (k < 1.0)
        assert np.all(bare[sel] > 0.0)
        assert np.all(full[sel] > bare[sel])

    def test_a_scale_dependent_bias_weight_is_accepted(
            self, field, sectors, params, table):
        """An ``(Nk, NM)`` bias weight made the factorised kernel raise; the
        fused one broadcasts it."""
        from ggah_mod.spectra.bnl import two_halo_correction
        g = SP.build_weights(SP.resolve("galaxies").components[0], field,
                             sectors, params)
        k = field.k[:, None]
        decorated = g.replace(bias_weight=field.bias[None, :]
                              * (1.0 + 1e-3 / k ** 2))
        d = np.asarray(two_halo_correction(field, decorated, decorated, table))
        assert d.shape == (field.k.shape[0],) and np.all(np.isfinite(d))


class TestTheTwoHaloSpectrumChoice:
    r"""``cb`` against ``total``, measured -- and it tracks :math:`(1-f_\nu)^2`."""

    @pytest.mark.parametrize("sum_mnu,f_nu_expected", [(0.06, 4.50e-3),
                                                       (0.30, 2.25e-2)])
    def test_the_ratio_tracks_one_minus_f_nu_squared(
            self, pk, sum_mnu, f_nu_expected, gal_params):
        cosmo = PLANCK18.replace(sum_mnu=sum_mnu)
        fl = make_field(cosmo, DIFFERENTIABLE, pk, z=0.0)
        sec = {"matter": MatterField()}
        par = {"matter": {"split": BaryonSplit.from_hot(
            0.0, jnp.zeros(fl.n_m))}}
        assert float(cosmo.f_nu) == pytest.approx(f_nu_expected, rel=0.05)

        def ratio(which):
            # The cold-only integral on both sides: "total" already contains
            # the neutrinos, and the leg is refused with it.
            o = SP.PkOptions(two_halo_spectrum=which, neutrino_two_halo="none",
                             bnl=False)
            p = SP.spectrum(fl, "matter", "matter", sec, par, options=o)
            return np.asarray(p.two_halo)

        got = ratio("cb")[0] / ratio("total")[0]
        want = float(fl.pk_cb[0] / fl.pk_lin[0])
        assert got == pytest.approx(want, rel=1e-10)

    def test_an_unknown_spectrum_is_refused(self, field, sectors, params):
        with pytest.raises(ValueError, match="unknown two-halo spectrum"):
            SP.spectrum(field, "matter", "matter", sectors, params,
                        options=SP.PkOptions(two_halo_spectrum="nonlinear"))


class TestTheNeutrinoLeg:
    r"""The neutrinos are in no halo, and the two-halo term still sees them.

    :math:`\mathcal I_a = I_a + (\nu_a/n_a)L` with
    :math:`L = \sqrt{P_m/P_{cb}} - \bar\rho_{cb}/\bar\rho_m` -- see
    :mod:`ggah_mod.spectra.neutrinos`.
    """

    @staticmethod
    def _matter(cosmo, pk):
        fl = make_field(cosmo, DIFFERENTIABLE, pk, z=0.0)
        sec = {"matter": MatterField()}
        par = {"matter": {"split": BaryonSplit.from_hot(0.0, jnp.zeros(fl.n_m))}}
        return fl, sec, par

    # (a) the large-scale limit is the total linear spectrum
    @pytest.mark.parametrize("sum_mnu", [0.06, 0.30])
    def test_the_matter_two_halo_term_tends_to_the_total_spectrum(
            self, pk, sum_mnu):
        cosmo = PLANCK18.replace(sum_mnu=sum_mnu)
        fl, sec, par = self._matter(cosmo, pk)
        p = SP.spectrum(fl, "matter", "matter", sec, par, options=NU_OPTS)
        assert float(p.two_halo[0] / fl.pk_lin[0]) == pytest.approx(1.0, abs=1e-8)
        (w,) = [SP.build_weights(c, fl, sec, par)
                for c in SP.resolve("matter").components]
        amp = SP.two_halo_amplitude(fl, w, NU_OPTS)
        # To the counterterm's quadrature, which is what puts I_m(0) at
        # rho_cb/rho_m: 4e-10 measured.
        assert float(amp[0]) == pytest.approx(
            float(jnp.sqrt(fl.pk_lin[0] / fl.pk_cb[0])), rel=5e-9)

    # (b) exactly nothing at zero mass
    def test_at_zero_mass_the_leg_is_an_exact_zero(self, pk, massless):
        fl, sec, par = self._matter(massless, pk)
        leg = np.asarray(SP.neutrino_leg(fl))
        assert np.all(leg == 0.0)
        on = SP.spectrum(fl, "matter", "matter", sec, par, options=NU_OPTS)
        off = SP.spectrum(fl, "matter", "matter", sec, par, options=OPTS)
        assert np.array_equal(np.asarray(on.two_halo), np.asarray(off.two_halo))

    def test_the_guard_is_needed_at_zero_mass(self, pk, massless):
        r"""The reason for the ``where``: the two ``emu_pk`` heads disagree at
        :math:`\Sigma m_\nu = 0` by more than any neutrino could explain, so
        :math:`\sqrt{P_m/P_{cb}} - 1` would be a leg made of network error."""
        fl = make_field(massless, DIFFERENTIABLE, pk, z=0.0)
        mismatch = np.max(np.abs(np.asarray(fl.pk_lin / fl.pk_cb) - 1.0))
        assert mismatch > 1e-4

    @pytest.mark.x64
    def test_the_gradient_at_zero_mass_is_finite(self, pk):
        def leg0(s):
            c = PLANCK18.replace(sum_mnu=s)
            return SP.neutrino_leg(make_field(c, DIFFERENTIABLE, pk, z=0.0))[0]
        g = float(jax.jacfwd(leg0)(0.0))
        assert np.isfinite(g) and g == 0.0

    # (c) its shape
    @pytest.mark.parametrize("sum_mnu", [0.06, 0.15, 0.30])
    def test_the_leg_is_f_nu_on_large_scales_and_vanishes_on_small(
            self, pk, sum_mnu):
        cosmo = PLANCK18.replace(sum_mnu=sum_mnu)
        fl = make_field(cosmo, DIFFERENTIABLE, pk, z=0.0)
        leg = np.asarray(SP.neutrino_leg(fl))
        f_nu = float(cosmo.f_nu)
        k = np.asarray(fl.k)
        assert leg[0] / f_nu == pytest.approx(1.0, abs=5e-3)
        assert abs(leg[np.searchsorted(k, 10.0)]) < 1e-2 * f_nu
        # Decreasing through free streaming, decade by decade: L/f_nu at
        # k = 1e-4, 1e-3, 1e-2, 0.03 is 1.002, 0.914, 0.345, 0.074 at 0.06 eV
        # and 1.001, 0.993, 0.799, 0.497 at 0.30 eV.  Node to node it is not,
        # and past k ~ 0.1 at the minimal mass it is not even signed: the two
        # emu_pk heads' ratio is good to ~1e-4 there (-1.2e-2 f_nu at k = 0.1).
        # That floor is the emulator's, not the neutrinos'.
        dec = leg[np.searchsorted(k, [1e-4, 1e-3, 1e-2, 3e-2])]
        assert np.all(np.diff(dec) < 0.0)
        small = (k > 1.0) & (k < 10.0)
        assert np.max(np.abs(leg[small])) < 1e-2 * f_nu

    # (d) galaxies carry none, and their cross with matter gets it once
    def test_galaxies_carry_no_leg(self, field, sectors, params):
        (w,) = [SP.build_weights(c, field, sectors, params)
                for c in SP.resolve("galaxies").components]
        assert w.neutrino_weight is None
        on = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                         options=NU_OPTS)
        off = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                          options=OPTS)
        assert np.array_equal(np.asarray(on.total), np.asarray(off.total))

    def test_the_galaxy_matter_cross_gets_the_leg_once(
            self, field, sectors, params):
        wg = SP.build_weights(SP.resolve("galaxies").components[0], field,
                              sectors, params)
        wm = SP.build_weights(SP.resolve("matter").components[0], field,
                              sectors, params)
        p = SP.spectrum(field, "galaxies", "matter", sectors, params,
                        options=NU_OPTS)
        ig = SP.i_of_k(field, wg)
        im = SP.i_of_k(field, wm)
        want = field.pk_cb * ig * (im + SP.neutrino_leg(field))
        assert np.allclose(np.asarray(p.two_halo), np.asarray(want),
                           rtol=1e-12, atol=0.0)
        off = SP.spectrum(field, "galaxies", "matter", sectors, params,
                          options=OPTS)
        f = float(PLANCK18.f_nu)
        got = float(p.two_halo[0] / off.two_halo[0]) - 1.0
        assert got == pytest.approx(f / (1.0 - f), rel=5e-3)

    # (e) who carries it
    def test_alignments_carry_b_i_times_the_leg(self, field, sectors, params):
        from ggah_mod.cosmology.growth import growth_factor
        from ggah_mod.sectors import IaParams, IntrinsicAlignmentSector
        from ggah_mod.sectors.alignments import alignment_bias
        growth = float(growth_factor(0.0, PLANCK18, make_pk("emu_pk")))
        ia = IaParams(a_ia=1.0)
        sec = {**sectors, "alignments": IntrinsicAlignmentSector()}
        par = {**params, "alignments": {"split": params["matter"]["split"],
                                        "growth": growth, "params": ia}}
        (w,) = [SP.build_weights(c, field, sec, par)
                for c in SP.resolve("alignments").components]
        b_i = float(alignment_bias(0.0, growth, ia))
        assert float(w.neutrino_weight) == pytest.approx(b_i, rel=1e-14)
        pii = SP.spectrum(field, "alignments", "alignments", sec, par,
                          options=NU_OPTS)
        pmm = SP.spectrum(field, "matter", "matter", sec, par, options=NU_OPTS)
        assert np.allclose(np.asarray(pii.two_halo),
                           b_i ** 2 * np.asarray(pmm.two_halo), rtol=1e-12)

    @pytest.mark.parametrize("name", ["galaxies", "centrals", "satellites",
                                      "gas", "pressure", "xray", "agn"])
    def test_no_cold_tracer_carries_it(self, field, sectors, params, name):
        for c in SP.resolve(name).components:
            w = SP.build_weights(c, field, sectors, params)
            assert w.neutrino_weight is None, c.label()

    def test_only_matter_and_alignments_name_it_among_the_sectors(self):
        """The runtime sweep above covers the tracers this module can build;
        this covers the rest by source, so a sector added later that sets the
        weight is seen here before anyone relies on it."""
        import ggah_mod.sectors as S
        root = pathlib.Path(S.__file__).parent
        naming = sorted(f.name for f in root.glob("*.py")
                        if "neutrino_weight" in f.read_text()
                        and f.name != "protocol.py")
        assert naming == ["alignments.py", "matter.py"]

    def test_pressure_is_bit_identical(self, field, sectors, params):
        on = SP.spectrum(field, "pressure", "pressure", sectors, params,
                         options=NU_OPTS)
        off = SP.spectrum(field, "pressure", "pressure", sectors, params,
                          options=OPTS)
        assert np.array_equal(np.asarray(on.total), np.asarray(off.total))

    def test_electrons_are_bit_identical(self, field, sectors, params):
        from ggah_mod.sectors import EjectaSector
        sec = {**sectors, "ejecta": EjectaSector()}
        par = {**params, "ejecta": {"split": params["matter"]["split"]}}
        on = SP.spectrum(field, "electrons", "electrons", sec, par,
                         options=NU_OPTS)
        off = SP.spectrum(field, "electrons", "electrons", sec, par,
                          options=OPTS)
        assert np.array_equal(np.asarray(on.total), np.asarray(off.total))

    def test_a_discrete_tracer_cannot_carry_it(self):
        from ggah_mod.sectors import TracerWeights
        with pytest.raises(ValueError, match="discrete and carries"):
            TracerWeights(w_point=jnp.ones(4), w_extended=None,
                          norm=jnp.asarray(1.0), discrete=True,
                          bias_weight=None, name="x",
                          neutrino_weight=jnp.asarray(1.0))

    # (f) the gradient
    @pytest.mark.x64
    def test_the_neutrino_mass_gradient_matches_finite_differences(self, pk):
        k_at = np.array([1e-3, 1e-2, 0.1])

        def model(s):
            fl, sec, par = self._matter(PLANCK18.replace(sum_mnu=s), pk)
            p = SP.spectrum(fl, "matter", "matter", sec, par, options=NU_OPTS)
            idx = np.searchsorted(np.asarray(fl.k), k_at)
            return jnp.log(p.total[idx])

        x, h = 0.06, 2e-4
        ad = np.asarray(jax.jacfwd(model)(x))
        fd = np.asarray((model(x + h) - model(x - h)) / (2 * h))
        assert np.all(np.isfinite(ad))
        assert np.allclose(ad, fd, rtol=1e-4, atol=1e-6)

    # (g) off is the cold integral, exactly
    def test_off_is_the_cold_integral(self, field, sectors, bare_matter):
        p = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=OPTS)
        (w,) = [SP.build_weights(c, field, sectors, bare_matter)
                for c in SP.resolve("matter").components]
        i = SP.i_of_k(field, w)
        want = SP.linear_spectrum(field, OPTS) * i * i
        assert np.array_equal(np.asarray(p.two_halo), np.asarray(want))

    # (h) refusals
    def test_total_with_the_leg_is_refused(self):
        with pytest.raises(ValueError, match="counts the neutrinos twice"):
            SP.PkOptions(two_halo_spectrum="total")
        with pytest.raises(ValueError, match="counts the neutrinos twice"):
            DIFFERENTIABLE.with_(two_halo_spectrum="total")

    def test_an_unknown_treatment_is_refused(self, field, sectors, bare_matter):
        bad = SP.PkOptions(neutrino_two_halo="halo", bnl=False)
        with pytest.raises(ValueError, match="unknown neutrino two-halo"):
            SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=bad)
        with pytest.raises(ValueError, match="does not exist"):
            DIFFERENTIABLE.with_(neutrino_two_halo="halo")

    def test_the_backend_offers_exactly_what_the_module_implements(self):
        from ggah_mod.backend import NEUTRINO_TWO_HALO
        assert NEUTRINO_TWO_HALO == frozenset(SP.NEUTRINO_TWO_HALO)

    @pytest.mark.parametrize("flavour", [DIFFERENTIABLE])
    def test_the_flavours_declare_the_leg(self, flavour):
        assert flavour.neutrino_two_halo == "linear"
        assert SP.PkOptions.from_backend(flavour).neutrino_two_halo == "linear"


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("pair", [("matter", "matter"),
                                      ("galaxies", "galaxies"),
                                      ("galaxies", "matter"),
                                      ("matter", "pressure")])
    def test_the_gradient_reaches_the_cosmology(self, pair, pk, gal_params):
        a, b = pair

        def model(omega_m):
            cosmo = PLANCK18.replace(Omega_m=omega_m)
            fl = make_field(cosmo, DIFFERENTIABLE, pk, z=0.0)
            gal = GalaxySector("zumandelbaum15", backend=DIFFERENTIABLE)
            f_cen, _ = gal.stellar_fraction(fl, gal_params)
            sec = {"galaxies": gal, "gas": HotGasDPM(backend=DIFFERENTIABLE),
                   "agn": AgnSector(gal), "matter": MatterField()}
            par = {"galaxies": gal_params, "gas": DpmParams(),
                   "agn": AgnParams(),
                   "matter": {"split": BaryonSplit.from_hot(
                       PLANCK18.Omega_b / PLANCK18.Omega_m, jnp.full(fl.n_m, 0.1), f_star_cen=f_cen)}}
            p = SP.spectrum(fl, a, b, sec, par, options=OPTS)
            return jnp.log(jnp.sum(jnp.abs(p.total)))

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(model)(x))
        fd = float((model(x + h) - model(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_the_power_spectrum_survives_a_jit_boundary(
            self, field, sectors, params):
        p = SP.spectrum(field, "galaxies", "matter", sectors, params,
                        options=OPTS)
        out = jax.jit(lambda q: q.total)(p)
        assert np.allclose(np.asarray(out), np.asarray(p.total), rtol=1e-14)

    def test_the_tracer_labels_are_not_leaves(self, field, sectors, params):
        """A `str` among the pytree leaves fails at the first jit boundary
        rather than at construction."""
        p = SP.spectrum(field, "galaxies", "matter", sectors, params,
                        options=OPTS)
        leaves = jax.tree_util.tree_leaves(p)
        assert all(not isinstance(x, str) for x in leaves)
        assert len(leaves) == 4


class TestTheOneHaloTransition:
    r"""The literal sum by default, and HMcode-2020's damping when asked for.

    The plateau of :math:`P^{1h}(k\to0)` is the integral of a
    :math:`\xi^{1h}` with compact support, so configuration-space statistics
    do not see it, and for rare haloes it is physical.  The damping is kept for
    Fourier-space matter spectra, where mass conservation wants :math:`k^4`.
    """

    DAMPED = SP.PkOptions(two_halo_spectrum="cb",
                          two_halo_consistency="linear_deficit",
                          one_halo_transition="mead20", bnl=False,
                          neutrino_two_halo="none")

    # -- the registry ------------------------------------------------------
    def test_the_registry_and_its_calibration_have_the_same_keys(self):
        from ggah_mod.spectra import transition as TR
        assert set(TR.ONE_HALO_TRANSITION) == set(TR.TRANSITION_CALIBRATION)

    def test_the_backend_offers_exactly_what_the_registry_implements(self):
        """``backend.py`` cannot import layer 4 -- that is the cycle the
        coherence audit forbids -- so the two lists are kept in step here.
        Without this, a flavour could name a prescription that does not exist
        and only fail when someone asked for a spectrum."""
        from ggah_mod.backend import ONE_HALO_TRANSITIONS
        from ggah_mod.spectra import transition as TR
        assert ONE_HALO_TRANSITIONS == frozenset(TR.ONE_HALO_TRANSITION)

    def test_the_default_is_the_literal_sum(self):
        from ggah_mod import ACCURATE
        assert SP.PkOptions().one_halo_transition == "none"
        for b in (ACCURATE, DIFFERENTIABLE):
            assert b.one_halo_transition == "none"

    def test_the_hybrid_is_retired_with_its_reason(self):
        """``mead15`` paired HMcode-2020's form with HMcode-2015's scale; the
        name is refused with that reason rather than as unknown, and a backend
        cannot declare it."""
        from ggah_mod.spectra import transition as TR
        with pytest.raises(ValueError, match="retired.*0.584/sigma_v"):
            TR.one_halo_transition(jnp.ones(8), jnp.linspace(1, 2, 8),
                                   transition="mead15")
        with pytest.raises(ValueError, match="does not exist"):
            DIFFERENTIABLE.with_(one_halo_transition="mead15")
        assert "mead15" not in TR.ONE_HALO_TRANSITION

    def test_a_coefficient_with_no_damping_is_refused(self, field):
        """A parameter that is accepted and unread does nothing."""
        from ggah_mod.spectra import transition as TR
        with pytest.raises(ValueError, match="reads no parameter"):
            TR.one_halo_transition(field.pk_cb, field.k, transition="none",
                                   params=TR.TransitionParams())

    def test_an_unknown_prescription_raises_rather_than_falling_back(self):
        from ggah_mod.spectra import transition as TR
        with pytest.raises(ValueError, match="unknown one-halo transition"):
            TR.one_halo_transition(jnp.ones(8), jnp.linspace(1, 2, 8),
                                   transition="hmcode2020")

    # -- the shape ---------------------------------------------------------
    def test_the_damping_vanishes_at_large_scales_and_is_unity_at_small(
            self, field):
        from ggah_mod.spectra import transition as TR
        t = np.asarray(TR.mead20(field.pk_cb, field.k))
        assert t[0] < 1e-6, "T(k_min) should be negligible, not merely small"
        assert t[-1] == pytest.approx(1.0, rel=1e-9)
        assert np.all((t >= 0.0) & (t <= 1.0))
        assert np.all(np.diff(t) >= 0.0), "the damping must be monotone in k"

    def test_none_is_exactly_one_everywhere(self, field):
        from ggah_mod.spectra import transition as TR
        t = np.asarray(TR.no_transition(field.pk_cb, field.k))
        assert np.array_equal(t, np.ones_like(t))

    # -- the defect ---------------------------------------------------------
    def test_the_damping_removes_the_plateau(self, field, sectors, bare_matter):
        """Undamped, ``P_1h(k -> 0)`` is the second moment and stays there;
        damped, it goes to zero as ``k^4``, which is what mass conservation
        asks of the matter field."""
        plateau = float(field.integrate(
            field.dndm * (field.m / field.rho_matter) ** 2))
        assert plateau > 0.0

        loud = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                           options=OPTS)
        assert float(loud.one_halo[0]) == pytest.approx(plateau, rel=1e-6)

        quiet = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                            options=self.DAMPED)
        assert float(quiet.one_halo[0]) < 1e-6 * plateau

    def test_the_damping_is_a_multiplier_on_the_one_halo_term_alone(
            self, field, sectors, params):
        """It must not reach the two-halo term, which is exact to 0.2 per cent
        against data and has nothing wrong with it, nor the shot noise, which is
        a property of the estimator rather than of the model."""
        loud = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                           options=OPTS)
        quiet = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                            options=self.DAMPED)
        assert np.allclose(np.asarray(loud.two_halo),
                           np.asarray(quiet.two_halo), rtol=1e-14)
        assert float(loud.shot) == pytest.approx(float(quiet.shot), rel=1e-14)

        from ggah_mod.spectra import transition as TR
        ratio = np.asarray(quiet.one_halo) / np.asarray(loud.one_halo)
        assert np.allclose(ratio, np.asarray(TR.mead20(field.pk_cb, field.k)),
                           rtol=1e-10)

    def test_small_scales_are_left_alone(self, field, sectors, params):
        """A transition that moved the one-halo term where the one-halo term is
        the answer would be a different defect, not a repair."""
        loud = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                           options=OPTS)
        quiet = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                            options=self.DAMPED)
        deep = np.asarray(field.k) > 5.0
        assert np.allclose(np.asarray(loud.one_halo)[deep],
                           np.asarray(quiet.one_halo)[deep], rtol=1e-6)

    # -- the option that was planned and is wrong ---------------------------
    def test_subtracting_the_plateau_would_make_the_term_negative(
            self, field, sectors, bare_matter):
        """``PLAN.md`` named "subtract the k -> 0 plateau" as a third entry.
        It is not implemented because it cannot be: ``|u(k|M)| <= u(0|M)`` for
        every halo, so ``P_1h(k) <= P_1h(0)`` *everywhere* -- the plateau is the
        term's maximum, not a floor under it, and subtracting it gives a
        one-halo term negative at every k.  Measured here so the idea is not
        had twice."""
        p = SP.spectrum(field, "matter", "matter", sectors, bare_matter,
                        options=OPTS)
        one = np.asarray(p.one_halo)
        assert np.all(one <= one[0] * (1.0 + 1e-12))
        assert np.all(one - one[0] <= 0.0)

    # -- the scale, measured ------------------------------------------------
    def test_the_damping_scale_is_hmcode_2020s(self, field):
        """``sigma8_cb`` and ``k_*`` at PLANCK18, z = 0, with the digits CLASS
        and CAMB use, so a comparison with either is of the damping and not of
        rounding."""
        from ggah_mod.cosmology.amplitude import sigma8
        from ggah_mod.spectra.transition import K_STAR_COEFF, K_STAR_EXPONENT
        assert (K_STAR_COEFF, K_STAR_EXPONENT) == (0.0561778, -1.0131066)
        s8 = float(sigma8(field.pk_cb, field.k))
        assert s8 == pytest.approx(0.81, abs=0.01)
        k_star = K_STAR_COEFF * s8 ** K_STAR_EXPONENT
        assert k_star == pytest.approx(0.0699, abs=0.001)
        from ggah_mod.spectra.transition import mead20
        i = int(np.argmin(np.abs(np.asarray(field.k) - k_star)))
        assert float(mead20(field.pk_cb, field.k)[i]) == pytest.approx(0.5, abs=0.02)

    def test_the_coefficient_carries_a_gradient(self, field):
        """It is fitted, so a campaign will want to vary it; a structural zero
        here would read as a flat direction rather than as an error."""
        from ggah_mod.spectra.transition import TransitionParams, mead20
        i = int(np.argmin(np.abs(np.asarray(field.k) - 0.07)))
        g = jax.grad(lambda c: mead20(field.pk_cb, field.k,
                                      TransitionParams(k_star_coeff=c))[i])(0.0561778)
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-3

    # -- layer 4's first declared parameter (PLAN.md A5) -------------------
    def test_the_coefficient_is_a_declared_parameter_not_a_bare_default(self):
        """``ggah_cal`` scopes layer 4 out of its campaign because "layer 4
        declares no parameter at all".  It declares one now, with a bound, a
        prior and a reason, like every other number a user might vary."""
        from ggah_mod.spectra.transition import TransitionParams, K_STAR_COEFF

        p = TransitionParams._PARAMS["k_star_coeff"]
        assert TransitionParams().k_star_coeff == K_STAR_COEFF == p.default
        lo, hi = p.bounds
        assert lo < p.default < hi
        assert p.why                        # Param.__post_init__ requires it

    def test_the_value_reaches_the_spectrum_and_moves_it(
            self, field, sectors, params):
        """Declaring it is not enough -- it has to arrive.  ``pk_cross`` takes
        it the way it takes ``bnl_table``, and ``None`` means the shipped
        default rather than a refusal, because a published constant with a
        bound is not a cosmology-dependent table."""
        from ggah_mod.spectra.transition import TransitionParams

        w = sectors["galaxies"].weights(field, params["galaxies"])
        base = SP.pk_cross(field, w, w, overlap="identical",
                           options=self.DAMPED)
        same = SP.pk_cross(field, w, w, overlap="identical",
                           options=self.DAMPED,
                           transition_params=TransitionParams())
        moved = SP.pk_cross(field, w, w, overlap="identical",
                            options=self.DAMPED,
                            transition_params=TransitionParams(k_star_coeff=0.12))

        assert np.array_equal(np.asarray(base.one_halo),
                              np.asarray(same.one_halo))
        assert not np.allclose(np.asarray(base.one_halo),
                               np.asarray(moved.one_halo))
        # A larger coefficient means a larger k_*, so more suppression.
        assert np.all(np.asarray(moved.one_halo)
                      <= np.asarray(base.one_halo) * (1.0 + 1e-12))
        # ...and it must not leak into the other two terms.
        assert np.allclose(np.asarray(base.two_halo),
                           np.asarray(moved.two_halo), rtol=1e-14)

    def test_the_value_is_traced_and_the_prescription_is_not(self):
        """The split ``PkOptions`` already draws, now with a parameter on the
        other side of it: a prescription cannot be differentiated into and a
        coefficient cannot be branched on."""
        from ggah_mod.spectra.transition import TransitionParams

        import dataclasses

        # The value is a registered pytree node, so it flattens to a leaf and
        # a gradient can be taken through it.
        leaves = jax.tree_util.tree_leaves(TransitionParams())
        assert len(leaves) == 1 and float(leaves[0]) == 0.0561778
        assert "one_halo_transition" not in TransitionParams._PARAMS

        # PkOptions is *not* registered, so JAX sees it as one opaque leaf --
        # which is what makes it a static argument rather than a traced one.
        # Every field it holds is a str or a bool; an array there could never
        # be differentiated and would silently force a retrace on every value.
        opts = SP.PkOptions()
        assert type(opts) not in jax.tree_util.__dict__.get("_registry", {})
        for f in dataclasses.fields(opts):
            assert isinstance(getattr(opts, f.name), (str, bool)), (
                f"PkOptions.{f.name} is not a static choice")

    def test_compensation_and_a_positive_xi_1h_cannot_both_hold(
            self, field, sectors, params):
        r"""The two things ``PLAN.md`` asked for are mutually exclusive, and the
        Fourier convention says so in one line.

        :math:`P(0) = \int d^3r\,\xi(r)`.  So a one-halo term with
        :math:`P^{1h}(0) = 0` -- which is the whole point of the damping -- has
        :math:`\int d^3r\,\xi^{1h} = 0`, and since :math:`\xi^{1h} > 0` at small
        separation it **must** change sign.  Conversely a :math:`\xi^{1h}` that
        is positive everywhere has :math:`P^{1h}(0) > 0`, which is the
        white-noise floor.  One or the other.

        Measured here rather than argued.  ``zumandelbaum15`` at its defaults,
        PLANCK18, z = 0, with ``mead20``.  Undamped, ``P^1h(k_min) = 631``;
        damped, it is 2.6e-9.  And ``xi_1h`` changes sign: damped it reaches
        **-8.5e-3 at r = 6.2 Mpc/h**, negative from 4.5 to 63 Mpc/h.  These
        are Sec. 5's numbers, pinned below.  (At the 0.8.5 defaults: 605, then
        607 on emu_pk 2.1, 2.5e-9 and -8.2e-3 at 6.2.  At Paper I's iHOD, the
        default until 0.8.4: 658, 2.8e-9, -9.0e-3 at 6.1, from 4.4 to 63.  Under the
        retired ``mead15``, whose scale was 0.100 rather than 0.070 h/Mpc, it
        was -1.6e-2 at 5.5.)  That is the configuration-space cost of the
        damping, and the reason it is not the default.

        **The plateau is a Fourier-space statement, and this test used to make
        it in configuration space, where it was an artefact.**  It recorded
        ``xi_1h`` undamped as "flat at 2.1e-5 out to 60 Mpc/h -- that constant
        *is* the plateau".  It is not: a constant ``P`` transforms to a delta
        function at the origin, not to a constant ``xi``.  The 2.1e-5 was
        FFTLog wrapping an undecayed integrand (``fftlog_pad_decades``, B4).
        With the padding it reads 1.6e-10, and Ogata -- which never wraps --
        gives -5.6e-10 and was never positive everywhere.  So the old
        ``np.all(xi_1h > 0)`` was passing on the artefact and would have failed
        against the other engine all along.  What survives is the physics: the
        damped sign change is six million times the noise floor and **both
        engines agree about it to 1.4 per cent**.

        The resolution is not that one prescription is wrong: it is that the
        compensated one-halo term is no longer a within-halo pair count.  It is
        that count minus its own large-scale double-count, and only the
        **total** is an observable.  ``xi_total`` stays positive throughout in
        both, which is the statement that matters.
        """
        from ggah_mod.observables.real_space import xi
        from ggah_mod.backend import DIFFERENTIABLE

        r = np.logspace(-1.0, 2.0, 400)
        k = np.asarray(field.k)
        got = {}
        for name, opts in (("none", OPTS), ("mead20", self.DAMPED)):
            p = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                            options=opts)
            got[name] = (
                np.asarray(xi(r, (k, np.asarray(p.one_halo)),
                              backend=DIFFERENTIABLE)),
                np.asarray(xi(r, (k, np.asarray(p.total)),
                              backend=DIFFERENTIABLE)),
                float(p.one_halo[0]))

        one_none, tot_none, p0_none = got["none"]
        one_mead, tot_mead, p0_mead = got["mead20"]

        # The plateau, where it actually lives: Fourier space.
        assert p0_none > 1.0
        assert p0_mead < 1e-6 * p0_none
        # The numbers the paper quotes (Sec. 5), at the defaults.
        assert p0_none == pytest.approx(631.2, rel=2e-3)
        assert one_mead.min() == pytest.approx(-8.548e-3, rel=5e-3)
        assert r[int(np.argmin(one_mead))] == pytest.approx(6.16, abs=0.05)

        # Undamped, xi_1h is positive wherever it is resolvable, and decays to
        # the noise floor rather than to a constant.  The bound is on the
        # *tail's* size, not its sign: at 1e-10 against 1e2 at small r, sign
        # there is arithmetic and not physics.
        assert np.all(one_none[r < 4.0] > 0.0)
        assert np.max(np.abs(one_none[r > 20.0])) < 1e-6 * one_none[0]

        # Damped, the sign change is signal: six orders above that floor.
        floor = float(np.max(np.abs(one_none[r > 20.0])))
        assert one_mead.min() < -1e3 * floor, (
            f"the damped one-halo term reaches {one_mead.min():.2e}, not "
            f"clear of the {floor:.2e} noise floor -- the sign change is what "
            f"this test is about and it has to be measurable")

        # Only the total is observable, and it survives both.
        assert np.all(tot_none > 0.0) and np.all(tot_mead > 0.0)

        # The damping is confined to large separations: inside a halo the two
        # agree, which is where a one-halo term is the answer.
        inner = r < 1.0
        assert np.allclose(one_none[inner], one_mead[inner], rtol=2e-3)

        # And the sign change is not one engine's opinion.  This is the check
        # the old version lacked, and lacking it is how an FFTLog wrap got
        # written down as a physical plateau.
        from ggah_mod.backend import ACCURATE
        other = np.asarray(xi(r, (k, np.asarray(
            SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=self.DAMPED).one_halo)), backend=ACCURATE))
        assert other.min() == pytest.approx(one_mead.min(), rel=0.05)
        assert r[np.argmin(other)] == pytest.approx(r[np.argmin(one_mead)],
                                                    rel=0.05)


class TestLayerFoursParameterIsReachable:
    r"""A declared parameter with no path from the public API is not declared.

    ``TransitionParams.k_star_coeff`` is layer 4's only ``Param``: bounded,
    priored, reasoned, and walked by the parameter audits.  ``pk_cross`` took
    it from the day it landed.  :func:`~ggah_mod.spectra.tracers.spectrum` --
    the entry point ``ggah_cal`` and ``ggah_mod_benchmark`` both use -- did
    **not** forward it for two tiers, while forwarding ``bnl_table`` beside it,
    and the paper described the two arguments as siblings.

    The consequence was not hypothetical.  ``ggah_cal/PLAN.md`` put in writing
    that layer 4 "declares **no** ``Param`` at all", and from where it stands
    that was true.  A parameter is only as declared as its reachability.
    """

    DAMPED = SP.PkOptions(two_halo_spectrum="cb",
                          two_halo_consistency="linear_deficit",
                          one_halo_transition="mead20", bnl=False,
                          neutrino_two_halo="none")

    def test_spectrum_forwards_it(self, field, sectors, params):
        """The regression proper: a non-default value must change the answer."""
        from ggah_mod.spectra.transition import TransitionParams
        a = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=self.DAMPED)
        b = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=self.DAMPED,
                        transition_params=TransitionParams(k_star_coeff=0.2))
        assert not np.allclose(np.asarray(a.one_halo),
                               np.asarray(b.one_halo)), (
            "spectrum() ignored transition_params; k_star_coeff is declared "
            "and unreachable, which is how it came to be described downstream "
            "as a parameter that does not exist")

    def test_it_takes_the_same_road_as_the_bnl_table(self):
        """Both are optional layer-4 inputs to ``pk_cross``, and every wrapper
        must forward both -- or the next one added is forgotten the same way."""
        import inspect
        for fn in (SP.spectrum, SP.matter_suppression):
            src = inspect.getsource(fn)
            if "bnl_table" in src:
                assert "transition_params" in src, (
                    f"{fn.__name__} passes bnl_table but not "
                    f"transition_params")

    def test_the_default_is_the_declared_default(self, field, sectors, params):
        """Passing ``None`` and passing the declared default agree, so nothing
        depends on which of the two a caller writes."""
        from ggah_mod.spectra.transition import TransitionParams
        a = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=self.DAMPED)
        b = SP.spectrum(field, "galaxies", "galaxies", sectors, params,
                        options=self.DAMPED,
                        transition_params=TransitionParams())
        assert np.allclose(np.asarray(a.total), np.asarray(b.total))


class TestTheDeclaredComponentOverlap:
    r"""``Overlap``: the per-component-pair rule ``population`` cannot carry.

    ``overlap_of`` answers from the labels alone and for a nested pair it
    cannot, because the label says the samples overlap and not *which
    components* are shared.  The refusal stands for anything undeclared; this
    is what declaring looks like.

    The acceptance test is an equality, not a property: ``ggah_cal`` has been
    doing this by hand in numpy since its joint covariance went non-positive
    without it, and the declared path has to reproduce that expression.
    """

    @staticmethod
    def _pair(field):
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        gp = galaxy_defaults("zumandelbaum16_red")
        red = GalaxySector("zumandelbaum16_red")
        blue = GalaxySector("zumandelbaum16_blue")
        return (red.weights(field, gp, view="total"),
                blue.weights(field, gp, view="total"),
                red.weights(field, gp, view="cen"),
                blue.weights(field, gp, view="cen"))

    def test_it_reproduces_the_ggah_cal_hand_rolled_subtraction(self, field):
        """`survey_forecast.py::_spectra`, to 1e-12 relative."""
        from ggah_mod.spectra.pk import pk_cross, PkOptions
        from ggah_mod.spectra.spec import Overlap
        wa, wb, ca, cb = self._pair(field)
        o = PkOptions()
        pij = pk_cross(field, wa, wb, overlap="none", options=o)
        cc = pk_cross(field, ca, cb, overlap="none", options=o)
        scale = (ca.norm * cb.norm) / (wa.norm * wb.norm)
        want = np.asarray(pij.one_halo + pij.two_halo) - np.asarray(cc.one_halo) * scale

        got = pk_cross(field, wa, wb, options=o, overlap=Overlap(
            point="same_object",
            why="a halo has one central, so a red central and a blue central "
                "in the same halo are one object and never a distinct pair"))
        got = np.asarray(got.one_halo + got.two_halo)
        assert np.allclose(got, want, rtol=1e-12, atol=0.0)

    def test_disjoint_is_bit_for_bit_the_shipped_string(self, field):
        """The new type must not move a number anyone already had."""
        from ggah_mod.spectra.pair import pair_1h
        from ggah_mod.spectra.spec import Overlap
        wa, wb, _, _ = self._pair(field)
        a = pair_1h(wa, wb, field.n_k, overlap="none")
        b = pair_1h(wa, wb, field.n_k, overlap=Overlap(
            why="two samples with no object in common"))
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_the_cross_shot_noise_is_the_shared_objects(self, field):
        from ggah_mod.spectra.pair import shot_noise
        from ggah_mod.spectra.spec import Overlap
        wa, wb, ca, cb = self._pair(field)
        got = float(shot_noise(field, wa, wb, overlap=Overlap(
            point="same_object", shared="point",
            why="the two central selections are independent at fixed halo mass")))
        want = float(field.integrate(field.dndm * ca.w_point * cb.w_point)
                     / (wa.norm * wb.norm))
        assert got == pytest.approx(want, rel=1e-12)

    def test_nothing_shared_means_no_cross_shot_noise(self, field):
        r"""And for red against blue that is the *physical* declaration.

        The two axes are independent and both have to be stated.  A red central
        and a blue central in one halo are the same object, so the one-halo
        point-point product is not a pair -- but no object is in *both*
        samples, because a central is red or it is blue.  ``point`` and
        ``shared`` answer different questions and this pair needs opposite
        answers to them.
        """
        from ggah_mod.spectra.pair import shot_noise
        from ggah_mod.spectra.spec import Overlap
        wa, wb, _, _ = self._pair(field)
        got = float(shot_noise(field, wa, wb, overlap=Overlap(
            point="same_object", shared="none",
            why="a central is red or blue, so the samples are complementary "
                "and share no object")))
        assert got == 0.0

    def test_a_declaration_needs_a_why(self):
        from ggah_mod.spectra.spec import Overlap
        with pytest.raises(ValueError, match="why"):
            Overlap(point="same_object")

    @pytest.mark.parametrize("kw", [{"point": "sort-of"}, {"shared": "sort-of"}])
    def test_the_members_are_validated(self, kw):
        from ggah_mod.spectra.spec import Overlap
        with pytest.raises(ValueError, match="must be one of"):
            Overlap(why="x", **kw)

    def test_it_is_frozen_and_hashable(self):
        """It lives in a jax treedef, which is why it is not a callback."""
        from ggah_mod.spectra.spec import Overlap
        o = Overlap(why="x")
        assert hash(o) == hash(Overlap(why="x"))
        with pytest.raises(Exception):
            o.point = "same_object"


class TestTheTableUnderJit:
    """A concrete field inside a jitted function still stages every operation,
    so the rescaling's solution is traced and its float-reading diagnostics
    must be skipped -- which is how a calibration's compiled likelihood calls
    this."""

    def test_a_spectrum_on_a_concrete_field_compiles(self, field, sectors,
                                                     params):
        opts = SP.PkOptions(bnl=True)

        def f(scale):
            p = SP.spectrum(field, "galaxies", "matter", sectors, params,
                            options=opts)
            return scale * p.total

        got = np.asarray(jax.jit(f)(1.0))
        want = np.asarray(f(1.0))
        assert np.allclose(got, want, rtol=1e-10)
