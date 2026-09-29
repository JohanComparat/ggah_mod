r"""Does the requested mass definition match what the fit was calibrated in?

Two tables in this layer already record it.
:data:`~ggah_mod.halos.mass_function.CALIBRATION` names, per multiplicity
function, the halo finder and definition it was fitted to and the redshift
range it was fitted over; :data:`~ggah_mod.halos.concentration.CM_CALIBRATION`
does the same for the concentration relations and adds whether the relation can
see a cosmology at all.

Until now **nothing in the package read either of them.**  They were
documentation with a test asserting their contents, which is a weaker thing
than it looks: a table nobody consults cannot prevent the error it describes.
The benchmark measured the cost of exactly that -- an ``81`` per cent apparent
discrepancy in ``duffy08`` that turned out to be a mass-definition mismatch and
not a disagreement at all.  CCL refuses such a pairing outright; this module is
how :func:`~ggah_mod.halos.field.make_field` comes to do the same.

**Why a definition mismatch raises and a redshift extrapolation warns.**  They
are different kinds of wrong.  Evaluating ``diemer19`` -- a relation for
:math:`c_{200c}` -- at :math:`\Delta = 200m` does not return a slightly wrong
concentration, it returns a concentration *for another halo boundary*, and
nothing in the number says so; there is no continuous limit in which it becomes
right.  Running ``tinker08`` at :math:`z = 3` when it was fitted to 2.5 is an
extrapolation of a smooth fit: it may be poor, but it is the same quantity, and
refusing it would make the package unable to say what it thinks happens there.
So: definitions raise, ranges warn, and both name the table entry they came
from so the reader can go and check the fit rather than trust this file.

The check lives here, and is called from
:func:`~ggah_mod.halos.field.make_field`, because that is the one place where
the model names and the requested ``mdef`` are both in scope.
``make_multiplicity`` and ``make_concentration`` take a name and nothing else;
they cannot check what they are never told.
"""

from __future__ import annotations

import re

import warnings

import jax.numpy as jnp

from ..cosmology.parameters import _concrete
from .concentration import CM_CALIBRATION
from .linear_bias import matched_bias_for
from .mass_definitions import VIRIAL, MassDef, parse_mass_def
from .mass_function import (CALIBRATION, MULTIPLICITY_CAPABILITY,
                            VIRIAL_REFERENCED_MULTIPLICITY)

__all__ = ["MismatchWarning", "calibrated_mass_defs", "check_calibration",
           "check_cosmology_support", "check_pairing"]

#: Policies accepted by ``calibration=`` throughout the layer.
POLICIES = ("strict", "warn", "off")


class MismatchWarning(UserWarning):
    """A model is being used outside the range it was calibrated over."""


#: Definitions a fit is valid in, keyed by what its table entry says.  ``None``
#: means unrestricted: an analytic form has no calibration to violate, and a
#: fit published as a function of :math:`\Delta` covers any spherical
#: overdensity.  The strings are parsed rather than duplicated into a second
#: table, because a second table is a thing that drifts.
def calibrated_mass_defs(description: str):
    """Parse a ``CALIBRATION``/``CM_CALIBRATION`` definition string.

    Returns a ``frozenset`` of :class:`~ggah_mod.halos.mass_definitions.MassDef`
    the fit is valid in, or ``None`` when it is unrestricted.  ``"FoF"`` maps to
    the empty set: friends-of-friends is not a spherical overdensity at all, so
    *no* ``mdef`` matches it and every pairing is flagged.

    A description that names a spherical overdensity and yields nothing is a
    :class:`ValueError`, not ``None``.  Those two answers used to be the same
    one, and they mean opposite things: ``None`` disables the check entirely,
    so a description this function could not read silently made a fit look
    valid in every mass definition -- the precise failure the check exists to
    prevent, arrived at through the check itself.  It is not hypothetical:
    ``"SO 200m (Rockstar, CSST/Kun)"`` split on the slash *inside* the
    parenthesis, both halves failed to parse, and the entry came back
    unrestricted.
    """
    s = str(description).strip().lower()
    if s.startswith("analytic") or "universal" in s or "so-any" in s:
        return None
    if "fof" in s and "so" not in s:
        return frozenset()
    # Drop a parenthesised aside before splitting: it carries the suite or the
    # overdensity range, and its punctuation is not the delimiter.
    bare = re.sub(r"\([^)]*\)", " ", s).replace("so ", " ")
    out = set()
    for part in bare.split("/"):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(parse_mass_def(part))
        except ValueError:
            continue
    if not out:
        raise ValueError(
            f"cannot read a mass definition out of {description!r}.  Returning "
            "'unrestricted' here would switch the calibration check off for "
            "this fit rather than report that it could not be read, so it is "
            "refused instead.  Use a form like 'SO 200c', 'SO 200c/vir', "
            "'SO-any', 'FoF b=0.2' or 'analytic', with anything else in "
            "parentheses.")
    return frozenset(out)


def _check_one(name, description, z_range, mdef, z, kind, policy):
    allowed = calibrated_mass_defs(description)
    want = parse_mass_def(mdef)

    if allowed is not None and want not in allowed:
        listed = ("no spherical-overdensity definition -- it is a "
                  "friends-of-friends fit" if not allowed
                  else ", ".join(sorted(repr(d) for d in allowed)))
        msg = (f"{kind} {name!r} was calibrated in {description!r}, which "
               f"covers {listed}; you asked for mdef={str(mdef)!r}.  These are "
               f"different halo boundaries, so the number would be a "
               f"systematic and not a tolerance -- the benchmark measured 81 "
               f"percent for one such pairing.  Either request a definition "
               f"the fit covers, choose a fit calibrated in this one, or pass "
               f"calibration='warn' to take the mismatch knowingly.")
        if policy == "strict":
            raise ValueError(msg)
        warnings.warn(msg, MismatchWarning, stacklevel=3)

    z_lo, z_hi = z_range
    # A rank test first, and a value test second, because `float()` cannot tell
    # the two failures apart: `ConcretizationTypeError` is a `TypeError`, and so
    # is `float()` of a length-2 array.  Answering both with "skip the check"
    # meant that every redshift *stack* silently switched the range check off --
    # on exactly the path, `make_fields`, where the redshifts are concrete and
    # the check can run.  Rank is static, so this branch is safe under trace.
    if jnp.asarray(z).ndim != 0:
        raise ValueError(
            f"check_calibration takes one redshift and got shape "
            f"{jnp.asarray(z).shape}.  It used to accept an array and skip the range "
            f"check for it, which is the opposite of what a check is for; "
            f"loop, and let each redshift answer for itself.")
    try:
        z_f = float(z)
    except TypeError:                       # traced redshift; nothing to check
        return
    if not (z_lo <= z_f <= z_hi):
        warnings.warn(
            f"{kind} {name!r} was fitted over z in [{z_lo:g}, {z_hi:g}] and is "
            f"being evaluated at z = {z_f:g}.  That is an extrapolation of a "
            f"smooth fit rather than a category error, so it is a warning -- "
            f"but it is not covered by anything measured in the paper it comes "
            f"from.", MismatchWarning, stacklevel=3)



# =========================================================================
# Does this layer have an answer at this cosmology?
# =========================================================================

def check_cosmology_support(hmf_model, mdef, cosmo) -> None:
    r"""Refuse, or price, a cosmology this layer's calibrated pieces cannot see.

    The layer-2 half of ``PLAN.md`` item **E2**.  Layer 1 gained
    :attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_k` and a neutrino
    ordering, and layer 2 reads neither -- which looks like curvature
    transparency and is not.  Two pieces here take a cosmology and answer for
    it, and both were fitted in a box with no axis for either parameter:

    * the ``tinker08_csst`` recalibrations, whose eight-parameter box is
      CSSTemu's and carries no curvature.  ``check_box`` cannot catch that,
      because a parameter that is not an axis has no bound to be outside of.
    * :func:`~ggah_mod.halos.mass_definitions.delta_vir`, which is Bryan &
      Norman (1998)'s **flat** coefficient pair evaluated at an
      :math:`\Omega_m(z)` that already carries the curvature through
      :func:`~ggah_mod.cosmology.background.hubble_e`.

    Left alone, each returns a smooth, plausible, wrong number.  That is the
    failure :func:`~ggah_mod.cosmology.power._check_curvature` refuses one layer
    up.

    **The two are not answered the same way, and the difference is a
    measurement.**  A missing axis says a fit cannot see a parameter; it does
    not say what that costs, and here the two came apart by a factor of 150.
    ``tinker08`` is 0.06766 rms in :math:`\ln f` at 200m where the
    recalibration is 0.00518, so a flat refusal sent a curved run to a fit
    thirteen times worse at the *Planck* + BAO bound in order to avoid degrading
    the correction by 0.4 per cent.  So the box question is now the crossover
    of :class:`~ggah_mod.halos.mass_function.CurvatureResponse` -- refuse past
    the curvature at which the correction stops beating the fit it replaces,
    warn below it with the induced size -- while ``delta_vir`` stays an
    unconditional refusal, because nothing has measured *it* and a coefficient
    for one is not a coefficient for the other.

    **The neutrino ordering is declared and not refused, and the difference is
    measured rather than assumed.**  ``emu_hmf``'s
    ``theta_from_cosmology`` returns a vector built from ``Omega_b``,
    ``Omega_cb``, ``h``, ``n_s``, ``A_s``, ``w_0``, ``w_a`` and the neutrino
    *sum*, and every one of those is bit-for-bit invariant under how the sum is
    divided -- ``Omega_nu_matter`` is linear in the mass and is written in
    closed form.  Verified in ``tests/test_curvature_halos.py``: the eight
    numbers are identical for ``normal``, ``inverted`` and ``degenerate`` at the
    same sum.  So the correction *cannot* move with the ordering, and refusing
    it here would be a refusal about the caller's spectrum wearing the mass
    function's name.  The spectrum is where the ordering acts and where it is
    refused: :func:`~ggah_mod.cosmology.power._check_nu_split`.

    What is left is the ordering reaching :math:`f` through :math:`\sigma(M)`,
    and ``emu_hmf`` has since measured that too: normal against degenerate on
    CLASS, over the trained redshifts and fitted peak heights, it is largest at
    the 0.058993 eV floor where the three masses differ most and falls
    monotonically to the box ceiling -- worst max :math:`1.15\times10^{-4}`,
    worst rms :math:`3.5\times10^{-5}`.  That is 2.2 and 0.7 per cent of the
    shipped residual, so in quadrature the published 0.52 and 0.54 per cent are
    unchanged.  The ruling and the measurement agree, and they were made in
    that order.

    **The refusals take no ``policy``, deliberately.**  ``calibration='off'`` is
    a caller saying they accept a model and mass-definition pairing they know is
    mismatched.  It is not a caller saying they accept a flat-box answer for a
    curved universe.  Letting one claim switch off the other is the failure
    :func:`calibrated_mass_defs` already records having been bitten by once, in
    the other direction.  Layer 1's guard has no policy either.  The *warning*
    is an ordinary :class:`MismatchWarning` and can be silenced like any other.

    The escape is to choose a piece that makes no cosmology claim:
    ``hmf_model='tinker08'`` is the same fit uncorrected, and ``'200m'`` and
    ``'200c'`` are exact under curvature because ``hubble_e`` carries the term.
    Past the crossover that escape stops being a fallback and becomes the right
    answer, which is what makes the threshold a threshold.

    Skipped under tracing, where the values are not available -- the same
    construction-time standing ``_check_curvature`` has, and reached
    concretely first by every realistic path.
    """
    cap = MULTIPLICITY_CAPABILITY.get(str(hmf_model).lower())
    ok = _concrete(getattr(cosmo, "Omega_k", 0.0))
    curved = ok is not None and ok != 0.0

    # Two routes to the same flat fit, and the second one does not go through
    # `mdef` at all.  `despali16` is indexed by Delta/Delta_vir, so
    # `field._delta_kw` divides by `MassDef("vir").delta_mean` whatever
    # definition the field declares -- and nothing cancels, because at 200m the
    # numerator is a plain 200.  Refusing only `mdef='vir'` would leave the
    # larger of the two exposures standing.
    if curved:
        via_mdef = parse_mass_def(mdef) == VIRIAL
        via_fit = str(hmf_model).lower() in VIRIAL_REFERENCED_MULTIPLICITY
        if via_mdef or via_fit:
            how = ("mdef='vir'" if via_mdef else
                   f"mass function {str(hmf_model)!r}")
            why = ("" if via_mdef else
                   "  It is indexed by Delta/Delta_vir, so it reaches "
                   "`delta_vir` at *every* mass definition and not only at "
                   "'vir': at 200m and z = 1 the ratio it is handed moves "
                   "0.990 to 0.967 between flat and Omega_k = 0.05.")
            out = ("Use mdef='200m' or '200c'" if via_mdef else
                   "Use hmf_model='tinker08', which takes Delta against the "
                   "mean density directly")
            raise ValueError(
                f"{how} cannot carry Omega_k = {ok:g}.  `delta_vir` is Bryan & "
                f"Norman (1998)'s **flat** fit, 18pi^2 + 82x - 39x^2, and the "
                f"x = Omega_m(z) - 1 it is evaluated at already comes from a "
                f"curved E(z) -- so the two together give a virial boundary "
                f"that is half one geometry and half the other, which is the "
                f"failure PLAN.md item E2 was written as one change to avoid."
                f"{why}  Their second coefficient pair, 60 and -32, is for an "
                f"*open* universe with Lambda = 0, so it is not this cosmology "
                f"either and there is no published pair to switch to.  {out}: "
                f"curvature enters those only through hubble_e, which already "
                f"carries it.  Note delta_vir is degenerate in Omega_k at "
                f"z = 0, where E(0) = 1 exactly, so a flat-looking check there "
                f"proves nothing.")

    # The box arm, second, because the one above is unconditional and emitting a
    # warning immediately before an unrelated refusal is noise.
    if curved and cap is not None and not cap.supports_curvature:
        r = cap.curvature_response
        if r is None:
            raise ValueError(
                f"mass function {str(hmf_model)!r} cannot carry Omega_k = "
                f"{ok:g}: it declares `supports_curvature = False`, its box has "
                f"no curvature axis, and nobody has measured what that costs.  "
                f"It would return the correction fitted for the *flat* "
                f"cosmology carrying the same parameters, against a sigma(M) "
                f"that is not flat.  Use hmf_model='tinker08', which is the "
                f"same fit uncorrected and claims no cosmology, or set "
                f"Omega_k = 0.")
        if abs(ok) > r.crossover:
            raise ValueError(
                f"mass function {str(hmf_model)!r} cannot carry Omega_k = "
                f"{ok:g}: past |Omega_k| = {r.crossover:.4f} the flat-box "
                f"correction is no longer an improvement on the fit it "
                f"replaces.  Its own residual in ln f is {r.val_rms:g}, the "
                f"uncorrected baseline is {r.baseline_rms:g}, and the curvature "
                f"the box cannot see adds {r.coefficient:g} per unit "
                f"|Omega_k| -- {r.total(ok):.5g} in quadrature here.  Beyond "
                f"the crossover the uncorrected fit is the better answer, so "
                f"use hmf_model='tinker08'.  " + (
                    f"That crossover is measured, not extrapolated: the "
                    f"checked range reaches {r.measured_to:g} and brackets the "
                    f"crossing at {r.measured_crossing:g}, so the linear law "
                    f"used here is early by "
                    f"{100 * (1 - r.crossover / r.measured_crossing):.0f} per "
                    f"cent and this refusal is on the conservative side of a "
                    f"real crossing."
                    if r.crossover_is_measured else
                    f"That crossover is an extrapolation and not a measured "
                    f"crossing: the law is fitted over |Omega_k| <= 0.05 and "
                    f"checked to {r.measured_to:g}, where the correction is "
                    f"still winning.  The coefficient falls monotonically, so "
                    f"the law overestimates the cost out here and this refusal "
                    f"lands early rather than late -- deliberately, since past "
                    f"the checked range the package stops knowing."))
        warnings.warn(
            f"mass function {str(hmf_model)!r} has no curvature axis and is "
            f"being used at Omega_k = {ok:g}.  Its residual in ln f goes "
            f"{r.val_rms:g} -> {r.total(ok):.5g}, against {r.baseline_rms:g} "
            f"for the uncorrected fit -- so it is still the better answer here, "
            f"which is why this is a warning and not the refusal it used to be. "
            f"It becomes a refusal past |Omega_k| = {r.crossover:.4f}.  What is "
            f"measured is the carrier's response at fixed sigma; the curvature "
            f"response of the emulator's own residual is not measured, because "
            f"the suite has no curved simulations.  See "
            f"`mass_function.CurvatureResponse`.",
            MismatchWarning, stacklevel=3)


def check_pairing(hmf_model, bias_model):
    r"""Warn when a multiplicity function and a bias fit are not a *pair*.

    A peak-background split derives the two together, and only a pair satisfies
    :math:`\int b\,(M/\bar\rho)\,(\dd n/\dd M)\,\dd M \to 1`.  Mixing them
    is a legitimate choice -- most published mass functions have no matching
    bias at all, and pairing one with ``tinker10`` is the usual thing to do --
    but it moves that integral by an amount nothing downstream will show you,
    because a fitted galaxy bias absorbs it.  How much is a measurement rather
    than a rule of thumb: ``tinker08`` with ``sheth01`` instead of ``tinker10``
    is a couple of per cent at 200c, and the honest way to find out for a given
    pairing is :func:`~ggah_mod.halos.linear_bias.mass_weighted_bias`, which
    reports the closure instead of asserting it.  So: a warning, not a refusal,
    and silence when the mass function has no published partner, because there
    is then nothing to be inconsistent with.
    """
    want = matched_bias_for(hmf_model)
    if want is None or str(bias_model).lower() == want:
        return
    warnings.warn(
        f"mass function {hmf_model!r} was derived with bias {want!r} by a "
        f"peak-background split; you asked for {bias_model!r}.  The pair is "
        f"what makes the bias-weighted mass integral tend to 1, and mixing "
        f"them moves it by an amount a fitted galaxy bias absorbs rather than "
        f"shows.  `linear_bias.mass_weighted_bias` reports the closure if you "
        f"want the number for this combination.",
        MismatchWarning, stacklevel=3)


def check_calibration(hmf_model, cm_model, mdef, z, policy: str = "strict",
                      bias_model=None):
    """Compare the requested ``mdef`` and ``z`` against both tables.

    Parameters
    ----------
    hmf_model, cm_model : str
        Keys of :data:`~ggah_mod.halos.mass_function.CALIBRATION` and
        :data:`~ggah_mod.halos.concentration.CM_CALIBRATION`.
    mdef : str or MassDef
    z : float
        Traced redshifts skip the range check.
    policy : {"strict", "warn", "off"}
        ``"strict"`` raises on a definition mismatch, ``"warn"`` downgrades it,
        ``"off"`` disables the check entirely.  Range checks always warn.
    """
    if policy not in POLICIES:
        raise ValueError(f"calibration must be one of {POLICIES}, got {policy!r}")
    if policy == "off":
        return
    # The two tables put the definition in different columns: CALIBRATION
    # merges finder and definition into one string, CM_CALIBRATION keeps the
    # simulation suite separate from the definitions the relation covers.
    if hmf_model in CALIBRATION:
        description, z_range = CALIBRATION[hmf_model]
        _check_one(hmf_model, description, z_range, mdef, z,
                   "mass function", policy)
    if cm_model in CM_CALIBRATION:
        _, description, z_range, _ = CM_CALIBRATION[cm_model]
        _check_one(cm_model, description, z_range, mdef, z,
                   "concentration relation", policy)
    if bias_model is not None:
        check_pairing(hmf_model, bias_model)
