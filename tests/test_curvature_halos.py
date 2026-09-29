r"""Verification: curvature and the neutrino ordering reach layer 2.

``PLAN.md`` item **E2** gave layer 1 :math:`\Omega_k` in five places at once,
and the paper's argument for that shape was that a package carrying four of
them would report a constraint that was partly flat.  This file is the sixth
place, one layer down.

Layer 2 reads five things from the cosmology and :math:`\Omega_k` is none of
them, which looks like curvature transparency and is not.  Two pieces here take
a cosmology and answer for it out of a box that has no axis for curvature or
for how a neutrino mass is divided, and a third -- the virial overdensity -- is
a *flat* fitting formula evaluated at an argument that already carries the
curvature.  None of the three fails; each returns a smooth, plausible number
for a different universe.

So the tests come in two halves, as they do one layer up.  One asserts that
each piece now refuses what it cannot answer.  The other asserts that with
:math:`\Omega_k = 0` **nothing moved**, which is the only gate showing the
refusals were added without disturbing the geometry everything else in this
package was measured in.
"""
import math
import pathlib
import re
import warnings
from unittest import mock

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import ACCURATE
from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.cosmology.background import hubble_e
from ggah_mod.halos.calibration import (MismatchWarning,
                                        check_cosmology_support)
from ggah_mod.halos.concentration import CM_CALIBRATION
from ggah_mod.halos.field import _DELTA_ARG, make_field, make_fields
from ggah_mod.halos.mass_definitions import MassDef, delta_vir
from ggah_mod.halos.mass_function import (
    COSMOLOGY_DEPENDENT_MULTIPLICITY, MULTIPLICITY_CAPABILITY,
    VIRIAL_REFERENCED_MULTIPLICITY,
)

from conftest import AnalyticPk


def emu_hmf_version():
    """Its version, for a skip message that says what is missing and why."""
    import types
    try:
        import emu_hmf
        return types.SimpleNamespace(v=getattr(emu_hmf, "__version__", "?"))
    except ImportError:
        return types.SimpleNamespace(v="absent")

#: Inside emu_pk's curvature box, [-0.15, 0.15], and inside the interval the
#: benchmark's own curvature sweep uses.
OPEN = Cosmology.create(Omega_k=0.05, nu_hierarchy="degenerate")
CLOSED = Cosmology.create(Omega_k=-0.05, nu_hierarchy="degenerate")
FLAT = Cosmology.create(nu_hierarchy="degenerate")
SPLIT = Cosmology.create(nu_hierarchy="normal")

#: A cheap field, on the one concentration relation that needs no redshift-aware
#: spectrum.  Nothing here is about the numbers in the field; it is about
#: whether the build is reached at all.
FIELD_KW = dict(cm_model="duffy08", mdef="200m", z=0.0)


def build(cosmo, **kw):
    return make_field(cosmo, ACCURATE, pk=AnalyticPk(), **{**FIELD_KW, **kw})


# ==========================================================================
# The gate
# ==========================================================================

class TestFlatIsUntouched:
    r"""Every flat number is what it was.

    Bit-for-bit rather than to a tolerance, and by construction rather than by
    luck: every path added here is guarded by ``Omega_k == 0``, returns before
    touching a value, and multiplies nothing by anything.  A tolerance would
    mean a formula had been restructured, which would put the whole validation
    table of this package back in question.
    """

    @pytest.mark.parametrize("z, want", [
        (0.0, 102.50497921960846),
        (1.0, 157.93606730963918),
        (2.0, 171.14111025990834),
    ])
    def test_the_virial_overdensity_is_the_number_it_was(self, z, want):
        """Pinned, so a future curved ``delta_vir`` cannot land silently.

        Re-pinned at 0.9.8, and only at :math:`z > 0`: the flat :math:`E(z)`
        itself moved there -- the neutrinos at CLASS's temperature and on the
        exact relic energy integral, the photons on CODATA -- by 1.0e-6 and
        2.2e-6 in these two numbers.  :math:`z = 0` did not move, because
        :math:`E(0) = 1` whatever the densities are.

        The anchoring any general spherical-collapse solve would need is that
        it reduce to *these* values at ``Omega_k = 0``, and a fit is not a
        solution: Bryan & Norman is itself an approximation to the flat answer,
        so an unanchored exact solve moves every one of them.
        """
        assert float(delta_vir(z, PLANCK18)) == want

    def test_the_guard_is_inert_on_every_shipped_default(self):
        for hmf in sorted(COSMOLOGY_DEPENDENT_MULTIPLICITY) + ["tinker08",
                                                               "despali16"]:
            for mdef in ("200m", "200c", "vir"):
                check_cosmology_support(hmf, mdef, PLANCK18)

    def test_a_flat_build_raises_nothing_and_warns_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f = build(FLAT, hmf_model="tinker08")
        assert np.all(np.isfinite(np.asarray(f.dndm)))

    def test_no_ordering_is_refused_at_any_mass(self):
        for h in ("normal", "inverted", "degenerate", "massless"):
            for mnu in (0.0, 0.12):
                if h == "massless" and mnu:
                    continue
                check_cosmology_support(
                    "tinker08_csst", "200m",
                    Cosmology.create(sum_mnu=mnu, nu_hierarchy=h))


# ==========================================================================
# The declaration
# ==========================================================================

class TestWhatEachFitDeclares:
    """The capability is data that is read, not a property that is sniffed."""

    def test_only_the_fits_that_take_a_cosmology_declare_one(self):
        assert set(MULTIPLICITY_CAPABILITY) == COSMOLOGY_DEPENDENT_MULTIPLICITY

    def test_neither_recalibration_claims_an_axis_it_has(self):
        for name, cap in MULTIPLICITY_CAPABILITY.items():
            assert cap.supports_curvature is False, name
            assert cap.supports_nondegenerate_nu is False, name

    def test_the_incapacity_is_emu_hmfs_own_box_and_not_a_local_opinion(self):
        """The fact the refusal rests on is checked against the repository that
        owns it, which is what ``tests/test_cross_repo.py`` does for a claim
        that crosses a boundary.  The day ``emu_hmf`` grows a curvature axis,
        this fails and names the line to change."""
        box = pytest.importorskip(
            "emu_hmf.box",
            reason="emu_hmf is absent, so the declared incapacity is not being "
                   "checked against the box it describes.  This skip is not a "
                   "pass: the table could say anything and nothing would "
                   "disagree.")
        assert "Omega_k" not in box.PARAMS
        assert len(box.PARAMS) == 8
        for cap in MULTIPLICITY_CAPABILITY.values():
            assert cap.supports_curvature == ("Omega_k" in box.PARAMS)

    def test_the_virial_referenced_set_agrees_with_the_argument_table(self):
        """Two records of one fact, and they are asserted against each other
        rather than derived one from the other, because they answer different
        questions: which keyword carries Delta, and what Delta is measured
        against."""
        assert set(VIRIAL_REFERENCED_MULTIPLICITY) == {
            k for k, v in _DELTA_ARG.items() if v == "delta_ratio"}

    def test_no_concentration_relation_needs_a_guard(self):
        """All six reach a cosmology only through sigma, D(z) and f(z), each of
        which carries curvature already.  The three marked False are power laws
        in mass and redshift, so their curvature response is structurally zero
        rather than merely small -- which is a scope statement and not a
        refusal."""
        blind = {n for n, row in CM_CALIBRATION.items() if row[3] is False}
        assert blind == {"duffy08", "dutton14", "klypin16"}


# ==========================================================================
# The refusals
# ==========================================================================

class TestCurvatureIsMeasuredNotRefusedOutright:
    r"""What the flat box costs is a number, and the number decides.

    This class asserted a flat refusal at any :math:`\Omega_k \neq 0` until the
    cost was measured, and the measurement reversed it.  ``tinker08`` is 0.06766
    rms in :math:`\ln f` at 200m where the correction is 0.00518, so refusing
    sent a curved run to a fit thirteen times worse at the *Planck*+BAO bound to
    avoid degrading the correction by 0.4 per cent.  The refusal fired 150 times
    below where the harm begins.

    So the guard is now the crossover: refuse past the curvature at which the
    correction stops beating the fit it replaces, warn below it with the induced
    size, and stay silent when flat.
    """

    def test_a_curved_cosmology_below_the_crossover_builds(self):
        with pytest.warns(MismatchWarning, match="no curvature axis"):
            f = build(OPEN)
        assert np.all(np.isfinite(np.asarray(f.dndm)))

    def test_it_warns_for_closed_as_well_as_open(self):
        with pytest.warns(MismatchWarning, match="no curvature axis"):
            build(CLOSED)

    @pytest.mark.parametrize("ok", [0.002, 0.05, 0.15, 0.30])
    def test_everything_inside_the_crossover_warns(self, ok):
        """0.15 is emu_pk's own box edge and 0.30 is just inside the crossover,
        so *no* curvature the differentiable path can represent is refused."""
        with pytest.warns(MismatchWarning):
            check_cosmology_support(
                "tinker08_csst", "200m",
                Cosmology.create(Omega_k=ok, nu_hierarchy="degenerate"))

    def test_past_the_crossover_it_refuses(self):
        with pytest.raises(ValueError, match="no longer an improvement"):
            check_cosmology_support(
                "tinker08_csst", "200m",
                Cosmology.create(Omega_k=0.31, nu_hierarchy="degenerate"))

    def test_the_crossover_is_derived_from_the_three_measured_numbers(self):
        """Pinned, and derived rather than stored so it cannot drift from them."""
        r = MULTIPLICITY_CAPABILITY["tinker08_csst"].curvature_response
        assert r.crossover == pytest.approx(0.3009, abs=5e-5)
        assert r.crossover == pytest.approx(
            math.sqrt(r.baseline_rms ** 2 - r.val_rms ** 2) / r.coefficient,
            rel=1e-12)
        assert r.total(0.002) == pytest.approx(0.00520, abs=5e-6)

    def test_neither_threshold_rests_on_an_extrapolation(self):
        """The sweep reached 0.45 and both crossings are bracketed: virial
        between 0.10 and 0.15, 200m between 0.35 and 0.40.  200m was the
        extrapolated one for most of a day, which is why the flag stays derived
        per file rather than becoming a constant."""
        for cap in MULTIPLICITY_CAPABILITY.values():
            r = cap.curvature_response
            assert r.crossover_is_measured is True
            assert r.crossover < r.measured_crossing <= r.measured_to

    def test_the_flag_is_derived_and_still_answers_false(self):
        """A newly measured fit has a coefficient before its sweep reaches its
        crossing, so the ``None`` case is reachable and the refusal branch for
        it is not dead code."""
        from ggah_mod.halos.mass_function import CurvatureResponse

        fresh = CurvatureResponse(coefficient=0.2, val_rms=0.005,
                                  baseline_rms=0.07, measured_to=0.05)
        assert fresh.crossover_is_measured is False
        assert fresh.crossover > fresh.measured_to

    def test_the_linear_law_is_early_wherever_the_crossing_was_measured(self):
        """The direction is the whole reason the threshold is defensible: a
        monotonically falling coefficient overestimates the cost, so the law
        refuses early.  A linear crossover *above* a measured one would mean
        that reasoning had inverted."""
        early = {}
        for name, cap in MULTIPLICITY_CAPABILITY.items():
            r = cap.curvature_response
            if r.crossover_is_measured:
                assert r.crossover <= r.measured_crossing, name
                early[name] = 1 - r.crossover / r.measured_crossing
        assert early["tinker08_csst"] == pytest.approx(0.19, abs=0.01)
        assert early["tinker08_csst_vir"] == pytest.approx(0.04, abs=0.01)
        # And more where the law reaches further, which is what a falling
        # coefficient has to do: 200m's crossing is three times virial's.
        assert early["tinker08_csst"] > early["tinker08_csst_vir"]

    def test_the_direction_assertion_is_one_that_can_actually_fail(self):
        """An invariant nobody has seen fail is decoration.

        This re-executes the module with the virial crossing moved *below* its
        own linear crossover -- the inversion the assertion exists to catch --
        and requires the import to raise.  Testing the predicate instead would
        test the predicate; this tests that the guard is wired to it.

        The failure it guards against is the one that would not look like a
        failure: refusing *late* means recommending the uncorrected fit while
        the correction is still the better answer, and every number involved
        would still be plausible.
        """
        import ggah_mod.halos.mass_function as mf

        src = pathlib.Path(mf.__file__).read_text()
        # Matched by name and not by value.  The first version of this test
        # pinned the literal 0.121 and went stale the moment that number was
        # refined to the archive's 0.120979 -- caught only by the inertness
        # guard below, which is the whole reason it is there.
        # One site, the first, now that both files carry a crossing.  Patching
        # both would still raise but would not say which entry did it.
        bad, n = re.subn(r"measured_crossing=[\d.]+",
                         "measured_crossing=0.05", src, count=1)
        assert n == 1, ("the patch matched no site; this test is now inert")
        assert bad != src

        def run(text):
            ns = {"__name__": "ggah_mod.halos._crossover_probe",
                  "__file__": mf.__file__, "__package__": "ggah_mod.halos"}
            exec(compile(text, mf.__file__, "exec"), ns)

        with pytest.raises(AssertionError, match="at or below"):
            run(bad)
        run(src)                # and the shipped numbers are legal

    @pytest.mark.parametrize("fit, ok", [("tinker08_csst", 0.31),
                                        ("tinker08_csst_vir", 0.13)])
    def test_the_refusal_says_the_crossing_was_measured(self, fit, ok):
        """Both files take the measured phrasing now.  The 200m one said the
        opposite this morning, which is why the sentence is chosen per file
        rather than written once."""
        with pytest.raises(ValueError, match="measured, not extrapolated"):
            check_cosmology_support(
                fit, "200m", Cosmology.create(Omega_k=ok,
                                              nu_hierarchy="degenerate"))

    def test_the_extrapolated_phrasing_is_still_reachable(self):
        """Not dead code: it is what any fit says before its sweep reaches its
        crossing, which was 200m's state until this afternoon."""
        from ggah_mod.halos.mass_function import CurvatureResponse, FitCapability
        import ggah_mod.halos.calibration as cal

        table = dict(MULTIPLICITY_CAPABILITY)
        table["tinker08_csst"] = FitCapability(
            False, False, CurvatureResponse(0.2242, 0.00518, 0.06766,
                                            measured_to=0.05))
        with mock.patch.object(cal, "MULTIPLICITY_CAPABILITY", table):
            with pytest.raises(ValueError, match="an extrapolation and not a "
                                                 "measured crossing"):
                check_cosmology_support(
                    "tinker08_csst", "200m",
                    Cosmology.create(Omega_k=0.31, nu_hierarchy="degenerate"))

    @pytest.mark.parametrize("fit, want", [
        ("tinker08_csst", (0.2242, 0.00518, 0.06766, 0.45, 0.371517)),
        ("tinker08_csst_vir", (0.8911, 0.00542, 0.10368, 0.45, 0.120979)),
    ])
    def test_the_restated_values_are_the_ones_received(self, fit, want):
        """Pin every field, so *this* side of the pair can fail today.

        The seam test below compares against ``emu_hmf``'s own constants and is
        the better check, but it **skips** while those constants are
        unreleased -- and a test that cannot fail is indistinguishable from a
        test that passes in a green run.  That is the same failure as a patch
        that matches nothing, and it has already happened once in this file.

        So this one pins the numbers as received, with no dependency to skip
        on.  It catches an accidental edit here; it cannot catch a
        regeneration upstream, which is what the seam test is for.  Until
        ``emu_hmf`` 1.2.0 installs, the drift check is one-sided and it is
        ``emu_hmf`` holding the working half.  1.2.0 is now the floor in
        ``pyproject.toml``, so on a supported install this no longer skips.

        Source: ``emu_hmf`` 40-point design, sweep to 0.45, both crossings
        bracketed by measurement.  Coefficients are the small-curvature mean
        below 0.05 and did not move when the sweep was extended twice.
        """
        r = MULTIPLICITY_CAPABILITY[fit].curvature_response
        got = (r.coefficient, r.val_rms, r.baseline_rms, r.measured_to,
               r.measured_crossing)
        assert got == want

    def test_the_numbers_are_emu_hmfs_numbers(self):
        """The four measured values are *restated* here, not imported, so this
        is where the restatement is checked against its source.

        Importing them instead would remove the drift entirely, and the
        dependency direction allows it -- this package already imports
        ``emu_hmf`` in three places.  It is not done because those imports are
        deliberately *lazy*: sixteen of the eighteen multiplicity functions
        need no emulator, and this table is read at import time by every
        ``make_field`` call including theirs.  Paying an ``emu_hmf`` import to
        run ``tinker08`` would be the wrong trade.

        ``emu_hmf`` pins the same fact from its own side.  Two checks is the
        right number for one number held in two places: the source finds out
        from its suite rather than from a paper, and so does the consumer.

        Skips until the constants exist -- 1.0.0 has none of them, and a
        version-blind assertion would go green against a package that cannot
        disagree.
        """
        target = pytest.importorskip(
            "emu_hmf.target",
            reason="emu_hmf is absent, so the cross-package drift check is not "
                   "running.  This skip is not a pass.")
        wanted = ("OMEGA_K_COST", "OMEGA_K_CROSSOVER", "OMEGA_K_MEASURED_TO",
                  "OMEGA_K_MEASURED_CROSSING")
        missing = [n for n in wanted if not hasattr(target, n)]
        if missing:
            pytest.skip(
                f"emu_hmf {getattr(emu_hmf_version(), 'v', '')} exposes none of "
                f"{missing}, so the two-sided drift check is one-sided and "
                f"emu_hmf is holding the half that works.  A skip here reads "
                f"as green; it is not.")

        # The key each recalibration loads its weights under -- see
        # `fsigma_tinker08_csst` and its virial sibling.
        for fit, key in (("tinker08_csst", "200m"),
                         ("tinker08_csst_vir", "vir")):
            r = MULTIPLICITY_CAPABILITY[fit].curvature_response
            # Exact, not tolerant: a tolerance here would absorb a *redefinition*
            # of the coefficient as though it were rounding.  The one defect
            # this seam has produced was exactly that -- a script meaning over
            # the full sweep instead of |Omega_k| <= 0.05, which moves 200m from
            # 0.2242 to 0.2112 and would sit inside any loose comparison.
            assert r.coefficient == pytest.approx(
                target.OMEGA_K_COST[key], rel=1e-12), fit
            assert r.crossover == pytest.approx(
                target.OMEGA_K_CROSSOVER[key], rel=1e-3), fit
            assert r.measured_to == pytest.approx(
                target.OMEGA_K_MEASURED_TO[key], rel=1e-12), fit
            theirs = target.OMEGA_K_MEASURED_CROSSING[key]
            if theirs is None:
                assert r.measured_crossing is None, fit
            else:
                assert r.measured_crossing == pytest.approx(theirs, rel=1e-9), fit

    def test_the_virial_file_is_four_times_more_sensitive_and_not_pooled(self):
        """A single coefficient for both would understate virial fourfold."""
        a = MULTIPLICITY_CAPABILITY["tinker08_csst"].curvature_response
        b = MULTIPLICITY_CAPABILITY["tinker08_csst_vir"].curvature_response
        assert b.coefficient / a.coefficient == pytest.approx(3.97, abs=0.02)
        assert b.crossover < a.crossover

    def test_the_warning_carries_the_size_rather_than_a_verdict(self):
        with pytest.warns(MismatchWarning) as rec:
            check_cosmology_support("tinker08_csst", "200m", OPEN)
        msg = str(rec[0].message)
        assert "0.00518" in msg and "0.06766" in msg
        assert "not measured" in msg

    def test_the_uncorrected_fit_is_neither_refused_nor_warned(self):
        """``tinker08`` claims no cosmology, so there is nothing to violate."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_cosmology_support("tinker08", "200m", OPEN)
        f = build(OPEN, hmf_model="tinker08")
        assert np.all(np.isfinite(np.asarray(f.dndm)))

    def test_an_unmeasured_missing_axis_is_still_refused(self):
        """The measurement is what converts a refusal into a threshold, so a
        capability with no response recorded keeps the old behaviour."""
        from ggah_mod.halos.mass_function import FitCapability
        import ggah_mod.halos.calibration as cal

        table = dict(MULTIPLICITY_CAPABILITY)
        table["tinker08_csst"] = FitCapability(False, False, None)
        with mock.patch.object(cal, "MULTIPLICITY_CAPABILITY", table):
            with pytest.raises(ValueError, match="nobody has measured"):
                check_cosmology_support("tinker08_csst", "200m", OPEN)

    def test_a_split_ordering_is_not_refused_here(self):
        """The ordering acts on the spectrum, and layer 1 refuses it there.
        Refusing it again on the mass function would be a refusal about the
        caller's P(k) wearing the multiplicity function's name -- see
        ``TestTheOrderingCannotMoveTheCorrection`` for why it cannot move it."""
        check_cosmology_support("tinker08_csst", "200m", SPLIT)

    def test_a_traced_curvature_takes_the_escape_rather_than_the_gradient(self):
        """A construction-time guard, the standing ``_check_curvature`` has.
        Refusing inside a trace would be an exception inside a trace, and every
        realistic path reaches this concretely first."""
        def go(ok):
            check_cosmology_support("tinker08_csst", "200m",
                                    FLAT.replace(Omega_k=ok))
            return ok * 2.0
        assert float(jax.jit(go)(0.05)) == 0.1

    def test_the_virial_refusal_still_ignores_the_calibration_policy(self):
        """The flat ``delta_vir`` is a different defect from the missing box
        axis, and it did not become a threshold: nothing measured it, and
        ``calibration='off'`` is not a claim about geometry."""
        for policy in ("strict", "warn", "off"):
            with pytest.raises(ValueError, match="delta_vir"):
                build(OPEN, mdef="vir", calibration=policy)

    def test_the_csst_backend_still_refuses_outright(self):
        """It is not a correction to a fallback, so there is no crossover to
        compute: nothing says what a curved CSSTemu should be compared against.
        It is also reached through ``make_hmf`` and never through
        ``make_field``, so the guard above never sees it."""
        from ggah_mod.halos.mass_function import CsstHMF
        try:
            hmf = CsstHMF()
        except ImportError as exc:              # CSSTemu is an extra
            pytest.skip(f"CSSTemu is absent, so this backend's own curvature "
                        f"refusal is not exercised -- not a pass.  "
                        f"{str(exc).split('.')[0]}")
        with pytest.raises(ValueError, match="no fallback"):
            hmf._set_cosmology(OPEN)
        hmf._set_cosmology(SPLIT)          # the ordering is not its business

    def test_the_csst_refusal_quotes_the_size_where_one_was_measured(self):
        """The full emulated dn/dlnM and the Castro+23 carrier alone agree to
        one part in 1e4 of the signal, so the carrier's coefficient is this
        backend's whole curvature response and can be quoted.  ``FoFM200c`` has
        none, and says nothing rather than borrowing one."""
        from ggah_mod.halos.mass_function import CsstHMF
        try:
            CsstHMF()
        except ImportError as exc:
            pytest.skip(f"CSSTemu is absent, so the quoted-size asymmetry "
                        f"against FoFM200c is not exercised -- not a pass.  "
                        f"{str(exc).split('.')[0]}")
        for massdef, quoted in (("RockstarM200m", True), ("RockstarMvir", True),
                                ("FoFM200c", False)):
            with pytest.raises(ValueError) as e:
                CsstHMF(massdef)._set_cosmology(OPEN)
            assert ("induced error" in str(e.value)) is quoted, massdef


# ==========================================================================
# The virial overdensity
# ==========================================================================

class TestTheVirialOverdensityIsTheFlatFit:

    def test_it_is_degenerate_in_curvature_at_z_zero(self):
        """E(0) = 1 exactly, by the way ``Omega_de`` closes, so
        ``Omega_m(0) = Omega_m`` whatever the curvature and ``delta_vir`` is
        the same number in all three geometries.

        Asserted rather than merely noted, because it is the trap: a test that
        asked "does delta_vir respond to curvature?" at z = 0 would pass for
        the wrong reason and go on passing after the fit was fixed.
        """
        assert float(delta_vir(0.0, OPEN)) == float(delta_vir(0.0, PLANCK18))
        assert float(delta_vir(0.0, CLOSED)) == float(delta_vir(0.0, PLANCK18))

    def test_it_moves_at_higher_redshift_and_in_the_right_direction(self):
        """Open lowers Omega_m(z), which lowers Delta_vir; closed raises it."""
        z = 1.0
        assert (float(delta_vir(z, OPEN)) < float(delta_vir(z, PLANCK18))
                < float(delta_vir(z, CLOSED)))

    def test_the_coefficients_are_the_flat_pair(self):
        """Bryan & Norman publish two, and only one is here.  Reconstructed
        from the function rather than read off the source, so a change to the
        expression fails this rather than passing it."""
        c, z = PLANCK18, 1.0
        om_z = float(c.Omega_m * (1.0 + z) ** 3 / float(hubble_e(z, c)) ** 2)
        x = om_z - 1.0
        assert float(delta_vir(z, c)) == pytest.approx(
            18.0 * np.pi ** 2 + 82.0 * x - 39.0 * x ** 2, rel=1e-12)

    def test_asking_for_virial_masses_under_curvature_is_refused(self):
        with pytest.raises(ValueError, match="delta_vir"):
            check_cosmology_support("tinker08", "vir", OPEN)

    @pytest.mark.parametrize("mdef", ["200m", "200c", "vir"])
    def test_despali16_is_refused_at_every_mass_definition(self, mdef):
        """It is indexed by Delta/Delta_vir, so ``field._delta_kw`` divides by
        the flat Delta_vir whatever definition the field declares.  Nothing
        cancels away from 'vir': at 200m the numerator is a plain 200."""
        with pytest.raises(ValueError, match="delta_vir"):
            check_cosmology_support("despali16", mdef, OPEN)

    def test_the_ratio_despali16_would_have_been_handed_actually_moves(self):
        """The size of what is being refused, measured rather than asserted."""
        flat = 200.0 / float(MassDef("vir").delta_mean(1.0, PLANCK18))
        open_ = 200.0 / float(MassDef("vir").delta_mean(1.0, OPEN))
        # Re-pinned at 0.9.8 with the flat E(z) above; both moved by 1.1e-6.
        assert flat == pytest.approx(0.9904355992112048, rel=1e-12)
        assert open_ == pytest.approx(0.9674186167833054, rel=1e-12)
        assert abs(open_ / flat - 1.0) > 0.02

    def test_it_is_still_one_jitted_expression_with_a_gradient(self):
        """No branch was added to ``delta_vir``: the refusal lives one level up,
        in the one place that has both the mdef and the cosmology."""
        g = jax.grad(lambda ok: delta_vir(1.0, PLANCK18.replace(Omega_k=ok)))
        d = float(g(0.0))
        assert np.isfinite(d) and d != 0.0


# ==========================================================================
# Why the ordering is declared and not refused
# ==========================================================================

class TestTheOrderingCannotMoveTheCorrection:
    r"""``CROSS_REPO.md`` X14 said layer 2 was untouched by the neutrino split.
    Its *arithmetic* is, and the recalibration was the reason that claim looked
    too wide -- but the reason turns out to run the other way, and it is
    measurable rather than arguable.

    Every entry of the vector ``emu_hmf`` is handed is built from the neutrino
    *sum*, and ``Omega_nu_matter`` is linear in the mass and written in closed
    form, so none of them can see how that sum is divided.  A refusal here would
    therefore refuse a configuration in which the correction returns exactly
    what it returned before.
    """

    def test_the_vector_is_bit_for_bit_the_same_in_every_ordering(self):
        """The load-bearing one.  If ``emu_hmf`` ever changes
        ``theta_from_cosmology`` to carry the ordering, this fails -- and that
        is the moment the question reopens."""
        target = pytest.importorskip(
            "emu_hmf.target",
            reason="emu_hmf is absent, so the invariance that justifies having "
                   "REMOVED the ordering refusal is unverified.  This skip is "
                   "not a pass: without it nothing here checks that the "
                   "correction cannot move with the ordering, which is the "
                   "whole argument for not refusing one.")
        # Each ordering's domain is one interval with its own floor -- 0.058993
        # eV normal, 0.099447 inverted -- so the sums are chosen above both
        # rather than swept blindly.
        for total in (0.12, 0.3):
            ref = np.asarray(target.theta_from_cosmology(
                Cosmology.create(sum_mnu=total, nu_hierarchy="degenerate")))
            for h in ("normal", "inverted"):
                got = np.asarray(target.theta_from_cosmology(
                    Cosmology.create(sum_mnu=total, nu_hierarchy=h)))
                assert np.array_equal(got, ref), (total, h)
        # And at the fiducial, which only the normal ordering reaches.
        assert np.array_equal(
            np.asarray(target.theta_from_cosmology(
                Cosmology.create(sum_mnu=0.06, nu_hierarchy="normal"))),
            np.asarray(target.theta_from_cosmology(
                Cosmology.create(sum_mnu=0.06, nu_hierarchy="degenerate"))))

    def test_the_cold_density_is_what_carries_that(self):
        """``Omega_cb`` is the only entry that could have seen the split, and
        it is the one the closed form makes invariant."""
        for h in ("normal", "inverted"):
            a = Cosmology.create(sum_mnu=0.12, nu_hierarchy=h)
            b = Cosmology.create(sum_mnu=0.12, nu_hierarchy="degenerate")
            assert float(a.Omega_cb) == float(b.Omega_cb)
            assert float(a.rho_cold) == float(b.rho_cold)

    def test_the_measured_cost_leaves_the_published_accuracy_alone(self):
        """``emu_hmf`` measured what the ordering does reach -- f through
        sigma(M) -- after this refusal had already been dropped on the
        invariance argument.  Worst max is 1.15e-4 against a shipped residual
        of 0.00518, so the two combine to the residual unchanged.  Recorded as
        the arithmetic rather than as a sentence, because the sentence is what
        would have to be trusted."""
        worst_max, val_rms = 1.15e-4, 0.00518
        assert worst_max / val_rms < 0.023
        assert math.hypot(val_rms, worst_max) == pytest.approx(val_rms, rel=3e-4)

    def test_the_declaration_survives_as_a_record(self):
        """``supports_nondegenerate_nu`` still says what the box has, and still
        says False.  It drives nothing, and the two facts are not in tension:
        the box has no ordering axis *and* has no need of one."""
        for cap in MULTIPLICITY_CAPABILITY.values():
            assert cap.supports_nondegenerate_nu is False
