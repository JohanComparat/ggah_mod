r"""Is this sector model being used at a redshift it was fitted over?

The halo layer asked this question first, and
:mod:`~ggah_mod.halos.calibration` answers it there.  Five registries in *this*
layer record the same kind of provenance --- :data:`OCC_CALIBRATION`,
:data:`CLF_CALIBRATION`, :data:`SHMR_CALIBRATION`, :data:`HI_CALIBRATION` and
:data:`F_GAS_CALIBRATION` --- and until now nothing read any of them.  The
benchmark recorded that as the unclosed half of its calibration finding: *"the
five sector registries are still unread, so the wider point stands."*

**What this module checks, and what it does not.**  It checks the redshift,
because that is the only field for which both a recorded value and a *requested*
value exist.  It does **not** check the selection, even though
:data:`OCC_CALIBRATION` and :data:`CLF_CALIBRATION` record one and a threshold
fit evaluated as a binned one is a genuine category error.  The reason is that
nothing in the package asks for a selection: there is no requested value to
compare against, and inventing an API so this module could have something to
check would be writing the caller to suit the checker.  The field is carried in
:class:`Calibration` and printed in the message, so it is in front of whoever
reads the warning; when a caller does start declaring a selection, the
comparison belongs here.

**Why every mismatch warns and none raise by default.**  The halo module raises
for a mass-definition mismatch because there is no continuous limit in which
:math:`c_{200c}` is a :math:`c_{200m}` --- it is a number for another boundary.
A redshift outside a fitted range is the opposite: it is the same quantity,
extrapolated, and refusing it would stop the package saying what it thinks
happens there.  So the default policy here is ``"warn"`` rather than the halo
layer's ``"strict"``, and ``"strict"`` is available for a caller who wants a
fitted range to be a hard boundary.

**Where it is called from.**  Four of the five registries reach a sector that
holds the model name together with a :class:`~ggah_mod.halos.field.HaloField`
that holds the redshift, so the check happens in ``weights()`` --- the one
method :class:`~ggah_mod.sectors.protocol.Sector` guarantees.
:data:`OCC_CALIBRATION` and :data:`CLF_CALIBRATION` both arrive through
:class:`~ggah_mod.sectors.galaxies.GalaxySector`, whose ``GALAXY_MODELS`` is the
two registries merged, so its lookup tries both; :data:`SHMR_CALIBRATION`
arrives through the same sector's optional ``shmr=``; :data:`HI_CALIBRATION`
through :class:`~ggah_mod.sectors.coldgas.ColdGasSector`.

:data:`F_GAS_CALIBRATION` is the exception, and the reason is worth stating
rather than papering over: **nothing in the package consumes** :data:`F_GAS`.
There is no sector holding one of those names, so there is no field and no
redshift within reach, and the only place such a name is ever resolved is
``make_f_gas`` itself.  That factory therefore takes an optional ``z=`` and
checks when it is given one.  It is a weaker hook than the other four have, and
it is weaker because that registry is further from a caller than the finding
assumed --- which is a fact about the package, not a gap in this module.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import warnings

from ..halos.calibration import POLICIES, MismatchWarning

__all__ = ["Calibration", "MismatchWarning", "POLICIES",
           "check_sector_calibration", "check_sector_range"]


class Calibration(NamedTuple):
    r"""What one sector model was fitted to, and over what range it means anything.

    One record for five registries that used to carry four different tuple
    shapes.  Positional order is ``fit`` then ``z_range`` because those are the
    two every entry has something to say about --- including the entries whose
    answer is "none", which is why ``z_range`` is allowed to be ``None`` rather
    than defaulted to something permissive.

    Parameters
    ----------
    fit : str
        What it was fitted to: the survey, the simulation, the method.
    z_range : tuple of float, or None
        The redshift range the fit covers.  ``None`` means *there is no fitted
        range* --- an analytic null test, not a fit --- and is checked as "never
        warns".  That is deliberately not the same value as a wide range: a
        permissive range asserts the fit was tested over it, and this says the
        question does not apply.
    selection : str
        The sample selection or the variable the fit is in, where the kind of
        model has one.  Recorded and reported, not compared; see the module
        docstring.
    notes : str
        Citation, purpose, or the caveat that does not fit anywhere else.
    cosmology_dependent : bool or None
        Whether the relation can respond to a cosmology at all.  ``None`` where
        the question was never asked of that registry.
    mstar_range : tuple of float, or None
        The :math:`\log_{10}(M_\star/M_\odot)` range the fit covers, where the
        model is one of stellar mass.  ``None`` -- the default, and what all
        five original registries carry -- means the question does not apply to
        that row, exactly as for ``z_range``.

        It is separate from ``selection`` because it is *comparable*: the
        module docstring declines to check a selection on the grounds that
        nothing in the package requests one, and that was true of every
        registry it was written for.  The AGN sector broke the tie by
        evaluating its relation on ``field.m`` and integrating satellites over
        a declared stellar-mass window, so a requested range exists and can be
        compared.  See :func:`check_sector_range`.
    """

    fit: str
    z_range: Optional[tuple[float, float]]
    selection: str = ""
    notes: str = ""
    cosmology_dependent: Optional[bool] = None
    mstar_range: Optional[tuple[float, float]] = None


def check_sector_calibration(kind, name, table, z, policy: str = "warn"):
    """Compare a requested redshift against one registry's fitted range.

    Parameters
    ----------
    kind : str
        What sort of model this is, for the message: ``"occupation"``,
        ``"SHMR"``, ``"HI-halo relation"``, ...
    name : str
        The registry key.  A name the table does not carry is not an error
        here: the registries and their model dicts are asserted equal by the
        sector test suite, so a missing key is that suite's finding, not this
        function's.
    table : mapping of str to Calibration
    z : float or array or tracer
        A traced redshift skips the check.  An array is reduced to the value
        furthest outside the range, since that is the one worth reporting.
    policy : {"strict", "warn", "off"}
        ``"warn"`` is the default and the honest one for an extrapolation;
        ``"strict"`` makes a fitted range a hard boundary.
    """
    if policy not in POLICIES:
        raise ValueError(f"calibration must be one of {POLICIES}, got {policy!r}")
    if policy == "off":
        return

    entry = table.get(str(name).lower())
    if entry is None or entry.z_range is None:
        return

    z_f = _worst(z, entry.z_range)
    if z_f is None:
        return

    z_lo, z_hi = entry.z_range
    if z_lo <= z_f <= z_hi:
        return

    where = f" ({entry.selection})" if entry.selection else ""
    msg = (f"{kind} {name!r} was fitted to {entry.fit!r}{where} over "
           f"z in [{z_lo:g}, {z_hi:g}], and is being evaluated at z = {z_f:g}. "
           f"That is an extrapolation of the same quantity rather than a "
           f"category error, so it is a warning -- but nothing in the paper it "
           f"comes from covers it.  Pass calibration='off' to silence this, or "
           f"'strict' to make the range a boundary.")
    if policy == "strict":
        raise ValueError(msg)
    warnings.warn(msg, MismatchWarning, stacklevel=3)


def check_sector_range(kind, name, table, values, *, field: str = "mstar_range",
                       unit: str = "log10 Msun", policy: str = "warn",
                       cost=None, tolerance: float = 0.0, hint: str = ""):
    r"""Compare a requested range against a registry's fitted one, in any
    variable the record carries.

    :func:`check_sector_calibration` is the redshift specialisation of this and
    stays as it is, so no existing caller moves.  This one takes the field name,
    because the second comparable range to appear was a stellar-mass one and a
    third is likelier than not.

    ``tolerance`` is this package's standing rule made an argument: **a fitted
    range is not a boolean**.  Evaluating a relation 0.1 dex below its floor is
    not the same act as 3 dex below, and a warning that fires identically for
    both is a warning nobody reads.  The default is ``0.0`` -- report any
    excursion -- so the behaviour matches the redshift check until a caller says
    otherwise.

    ``cost`` is the excursion's **measured** price, supplied by the caller and
    printed in the message, because a distance outside a box is not by itself a
    reason to act.  :meth:`~ggah_mod.sectors.agn.AgnSector.validity_cost`
    computes one for the AGN chain.  ``None`` omits the clause rather than
    inventing a number.

    Parameters
    ----------
    kind, name, table
        As :func:`check_sector_calibration`.
    values : float or array or tracer
        The requested values.  A traced value skips the check, for the reason
        :func:`_worst` gives.
    field : str
        The :class:`Calibration` attribute holding the range.
    unit : str
        Printed after the numbers.
    policy : {"strict", "warn", "off"}
    cost : float or None
        What the excursion is worth, measured.
    tolerance : float
        How far outside the range is not worth reporting, in the same units.
    hint : str
        Where to get the measured price, for callers that decline to compute one
        on every call.  Appended to the message.  Kept generic on purpose: the
        first caller's answer lives on its own sector, and naming that method
        here would put one sector inside a function five of them share.
    """
    if policy not in POLICIES:
        raise ValueError(f"calibration must be one of {POLICIES}, got {policy!r}")
    if policy == "off":
        return

    entry = table.get(str(name).lower())
    if entry is None:
        return
    rng = getattr(entry, field, None)
    if rng is None:
        return

    lo, hi = rng
    worst = _worst(values, (lo - tolerance, hi + tolerance))
    if worst is None:
        return
    if lo - tolerance <= worst <= hi + tolerance:
        return

    outside = max(lo - worst, worst - hi)
    if cost is not None:
        priced = (f"  Measured, that extrapolation is worth a factor "
                  f"{float(cost):.3f} on the quantity asked for.")
    elif hint:
        priced = f"  {hint}"
    else:
        priced = ""
    msg = (f"{kind} {name!r} was fitted to {entry.fit!r} over "
           f"{field.replace('_', ' ')} in [{lo:g}, {hi:g}] {unit}, and is being "
           f"evaluated at {worst:g} -- {outside:g} outside it. That is an "
           f"extrapolation of the same quantity rather than a category error, "
           f"so it is a warning.{priced}  Pass calibration='off' to silence "
           f"this, or 'strict' to make the range a boundary.")
    if policy == "strict":
        raise ValueError(msg)
    warnings.warn(msg, MismatchWarning, stacklevel=3)


def _worst(z, z_range):
    """The redshift furthest outside ``z_range``, or ``None`` if it is traced.

    Two jobs in one function because they share the same failure.

    *Which value to report.*  Taking the maximum would be wrong at the low end:
    a grid running from z = 0, checked against a fit that starts at 0.22, is out
    of range at its **smallest** value.  So the entry ranked is the distance
    outside the interval, which is negative everywhere inside it.

    *When not to look at all.*  ``float()`` on a tracer raises
    ``ConcretizationTypeError`` under ``jax.jit`` or ``jax.grad`` and takes the
    gradient with it, so the answer is ``None`` and the caller returns --- the
    same contract as :func:`~ggah_mod.halos.variance.check_k_support`, and safe
    for the same reason: this check only ever reports, so skipping it cannot
    change a number.

    Written in plain Python rather than numpy on purpose.  Layer 3 claims to be
    pure JAX and ``tests/test_sectors_paths.py`` enforces it, allowing numpy
    only for construction or table I/O; this is neither, and a third exception
    category for a function that could avoid the import is a worse trade than
    a ``max`` over a list.
    """
    lo, hi = z_range
    if getattr(z, "ndim", 0) == 0:
        try:
            return float(z)
        except Exception:
            # A traced scalar -- or a plain Python sequence, which has no
            # `ndim` and so arrives here looking like one.  That case used to
            # return `None`, which reads as "traced, skip the check": a guard
            # that silently did nothing for a list.  Fall through and try
            # iterating instead; a real tracer fails that too and still
            # returns `None`.
            pass
    try:
        vals = [float(v) for v in z]
    except Exception:                       # a traced array
        return None
    if not vals:
        return None
    return max(vals, key=lambda v: max(lo - v, v - hi))
