r"""Bounds, priors and reasons for every occupation and CLF model.

The half of layer 3 that carried none.  :class:`~ggah_mod.sectors.params.Param`
has applied to five containers and forty-six parameters;
:data:`~ggah_mod.sectors.occupation.DEFAULTS` and
:data:`~ggah_mod.sectors.clf.CLF_DEFAULTS` are **141 and 18 further numbers**
across fifteen models, every one of them a keyword float with no bound, no
prior and no recorded reason.  This module is the rest, and
``tests/test_sectors_paths.py``'s four container audits now reach it.

One vocabulary, not 159 inventions
----------------------------------

The 141 occupation slots are only **59 distinct names**, because the models
share a vocabulary: a halo-mass pivot is a halo-mass pivot whichever paper wrote
it down.  So the bound is declared once per *quantity* in :data:`VOCABULARY` and
resolved per model, which is both less code and a stronger statement -- a
reasoned envelope applied consistently, rather than fifty-nine separate
judgements that happen to agree.

Where a paper published a range, that range wins
------------------------------------------------

:class:`~ggah_mod.sectors.galaxies.GalaxyParams` already carries Zu & Mandelbaum's
own uniform priors -- Paper I Table 2 and Paper III Table 2 -- and its docstring
states the rule this module has to obey: *"a bound invented by a downstream
package looks exactly like a bound a measurement established, and only one of
them may be relaxed on the strength of new data."*

So the eighteen names that container declares are reused **by object identity**
for the three ``zumandelbaum`` models, the way ``P_OFF`` and ``R_OFF`` are shared
between :mod:`~ggah_mod.sectors.miscentering` and
:mod:`~ggah_mod.sectors.galaxies`.  A test asserts they are the same objects, so
the two cannot drift.

They are **not** reused for other models, and that is the point of doing it by
identity rather than by name.  ``leauthaud12`` also has a ``beta``; it is a
different SHMR's slope fitted to different data, and giving it Zu & Mandelbaum's
box would import a constraint from a measurement that never saw it.  Every
non-published bound is ``kind="physical"`` or ``"definitional"``, which is the
machine-readable version of the same distinction --
:data:`~ggah_mod.sectors.params.BOUND_KINDS` says a ``prior`` moves when the
measurement does and a ``physical`` one moves when the argument does.

Three name collisions this turned up, and all three are renamed
---------------------------------------------------------------

Declaring bounds per name is what makes a name meaning two things visible, and
three do.  None is repaired here because each is a rename with its own blast
radius; they are ``PLAN.md`` item **G9**.

* ``alpha_sat`` was a satellite power-law index in the occupations, published
  at :math:`(0.5, 1.5)` and positive, and a **faint-end slope** in both CLFs, at
  :math:`-1.15` and :math:`-1.3` -- same name, opposite sign, different
  quantity, with :data:`~ggah_mod.sectors.galaxies.GALAXY_MODELS` merging the
  two registries so one lookup reached both.  The CLFs' is ``alpha_faint`` now.
* ``bsat`` (occupations) and ``b_sat`` (Cacciato) differed by one underscore,
  which is the trap ``width_logmstar`` was named to avoid in
  :mod:`~ggah_mod.sectors.occupation`.  The CLFs' is ``phi_s_amp`` now, named
  for what it normalises.
* ``alpha_shmr`` was :math:`+0.3` in ``guo18``/``guo19`` and :math:`-1.638` in
  ``zacharegkas25`` -- the Kravtsov et al. (2018) parameterisation rather than
  the Guo one.  The second is ``alpha_shmr_k18``.

All three are renamed (``PLAN.md`` item **G9**).  :data:`PER_MODEL` survives
because the mechanism is worth keeping: it is where a name that *does* mean two
things gets the envelope each needs, and it now carries the reasons rather than
the workaround.
"""

from __future__ import annotations

from .clf import CLF_CALIBRATION, CLF_DEFAULTS
from .galaxies import GalaxyParams
from .occupation import DEFAULTS, OCC_CALIBRATION
from .params import Flat, Param, SectorParams, sector_params

__all__ = ["VOCABULARY", "PER_MODEL", "OCCUPATION_PARAMS", "make_occupation_params",
           "params_for"]


def _p(bounds, unit, why, kind):
    """A bound without its default; the default comes from the registry."""
    return dict(bounds=bounds, prior=Flat(), unit=unit, why=why, kind=kind)


# -- the families ---------------------------------------------------------
_HALO_MASS = _p(
    (9.0, 16.0), "log10(Msun/h)",
    "a pivot in halo mass. The shipped grid spans 1e10 to 1e16 Msun/h, so a "
    "pivot outside it is not constrained by any halo the model contains -- the "
    "occupation there is an extrapolation of a shape rather than a fit to "
    "anything. The box is one decade below the grid and at its top, so a fit "
    "that wants to leave reaches the boundary and says so instead of running "
    "away. Not any paper's prior",
    "physical")

_STELLAR_MASS = _p(
    (8.0, 12.5), "log10(Msun)",
    "a pivot in stellar mass. Bounded by where a stellar mass function has "
    "support: below 1e8 the samples these models were fitted to do not reach, "
    "and above 1e12.5 there are no galaxies. Not any paper's prior",
    "physical")

_THRESHOLD = _p(
    (8.0, 12.5), "log10(Msun)",
    "the sample selection, not a model parameter: it says which galaxies are "
    "being described. Bounded by validity rather than by physics -- outside the "
    "range the published samples span, the model still evaluates and is no "
    "longer the published model. A parameter rather than a static field because "
    "a forward model may want its gradient when the threshold is itself "
    "uncertain",
    "validity")

_FRACTION = _p(
    (0.0, 1.0), "",
    "a fraction -- a duty cycle, a completeness or a satellite share. Outside "
    "[0, 1] it is not a fraction, which is why this is definitional and never "
    "relaxed. Note the `fc` of the Zu & Mandelbaum occupation is *not* one of "
    "these: it is a normalisation allowed above 1, in the (0.1, 3) box Zu & "
    "Mandelbaum give their own fc, which is a satellite concentration ratio",
    "definitional")

_SCATTER_MASS = _p(
    (0.01, 3.0), "dex",
    "a scatter or erf width in log10 halo mass. Strictly positive by "
    "definition -- a negative width is not a distribution -- and 3 dex is where "
    "the transition is flat across the whole shipped mass range, so the "
    "parameter has stopped doing anything above it",
    "definitional")

_SCATTER_STELLAR = _p(
    (0.01, 2.0), "dex",
    "a scatter or erf width in log10 stellar mass. Strictly positive by "
    "definition, and 2 dex spans the whole stellar mass function, so above it "
    "the model no longer distinguishes the sample from the parent population",
    "definitional")

_SLOPE = _p(
    (-5.0, 5.0), "",
    "a power-law index. The envelope is where the relation stays monotone over "
    "the shipped mass range and the occupation stays finite at both ends; it is "
    "deliberately wide, because a slope is the parameter these models most "
    "often disagree about and a narrow box here would be this package "
    "asserting a result. Not any paper's prior",
    "physical")

_POSITIVE_SLOPE = _p(
    (0.0, 3.0), "",
    "a satellite power-law index, N_sat ~ (M/M1)^alpha. Non-negative because a "
    "halo cannot host fewer satellites as it grows more massive -- that is the "
    "one thing every occupation model in the literature agrees on -- and 3 is "
    "well above the steepest published value. Not any paper's prior",
    "physical")

_AMPLITUDE = _p(
    (0.0, 50.0), "",
    "a positive amplitude multiplying a mass ratio. Non-negative is "
    "definitional for a count; the ceiling is where the term dominates the "
    "occupation across the entire grid and the model has stopped being the one "
    "described. Not any paper's prior",
    "physical")


#: One reasoned bound per quantity, keyed by name.
#:
#: Consulted only where the model has no published range -- see the module
#: docstring.  ``tests/test_occupation_params.py`` asserts every name in every
#: registry resolves, so a model added without declaring its bounds is a test
#: failure rather than a silent gap.
VOCABULARY: dict[str, dict] = {
    # halo-mass pivots
    **{n: _HALO_MASS for n in (
        "log10mmin", "log10m0", "log10m1", "log10m1_sat", "log10m1_shmr",
        "log10m_cut", "log10m_h1", "log10m_inc", "log10m_q", "log10m_sat",
        "lg_m1h", "lg_mh_qc", "lg_mh_qs")},
    # stellar-mass pivots and selections
    **{n: _STELLAR_MASS for n in (
        "log10m_star0", "log10m_star_min_cen", "log10m_star_min_sat",
        "lg_m0star", "log10l0", "log10l_lim")},
    **{n: _THRESHOLD for n in (
        "log10m_star_thresh", "log10m_star_lo", "log10m_star_hi")},
    # fractions
    **{n: _FRACTION for n in ("f_cen", "f_inc", "f_sat")},
    "f_gamma": _p(
        (0.5, 1.0), "",
        "Lange et al. (2025) central completeness: the fraction of haloes "
        "above threshold whose central enters the DESI DR1 catalogue. The box "
        "is the paper's own, which stops at 1 because a completeness above one "
        "is not a completeness -- it would be saying the sample contains "
        "centrals the haloes do not have. The lower half is where a "
        "colour-selected tracer sits", "physical"),
    # scatters
    **{n: _SCATTER_MASS for n in ("sigma_logm",)},
    **{n: _SCATTER_STELLAR for n in (
        "sigma_logmstar", "sigma_lnmstar", "width_logmstar", "sigma_c",
        "sigma_c_cen", "sigma_c_sat")},
    # slopes and indices
    **{n: _POSITIVE_SLOPE for n in ("alpha", "alpha_sat", "alpha_inc")},
    **{n: _SLOPE for n in (
        "alpha_faint", "alpha_s", "alpha_shmr", "alpha_shmr_k18", "alpha_cen", "beta", "beta1", "beta_cen",
        "beta_cut", "beta_sat", "beta_shmr", "gamma", "gamma_shmr", "delta",
        "delta_shmr", "eta", "mu_c", "mu_s", "b0", "b1", "b_0", "b_1", "b_2",
        "log10_beta2")},
    # amplitudes
    **{n: _AMPLITUDE for n in (
        "bsat", "bcut", "phi_s_amp", "B_sat", "B_cut", "kappa", "fc")},
    # normalisations
    "log10eps": _p(
        (-3.0, 0.0), "",
        "log10 of the Kravtsov et al. (2018) SHMR normalisation, the peak "
        "stellar-to-halo mass ratio. Negative by definition -- a halo cannot "
        "turn more than its own mass into stars -- and 1e-3 is an order of "
        "magnitude below the lowest efficiency any measurement reports",
        "physical"),
}

#: Bounds that override :data:`VOCABULARY` for one model, because the name
#: means something else there.  Each is a collision named in the module
#: docstring, and each entry exists so that the collision costs a wrong bound
#: rather than a wrong number.
PER_MODEL: dict[tuple[str, str], dict] = {
    ("cacciato09", "alpha_faint"): _p(
        (-3.0, 0.0), "",
        "the CLF **faint-end slope**, not the occupations' satellite index of "
        "the same name -- negative, and a different quantity. Bounded by where "
        "the Schechter integral converges at the faint end",
        "physical"),
    ("vandenbosch13", "alpha_faint"): _p(
        (-3.0, 0.0), "",
        "the CLF faint-end slope; see cacciato09's entry and the collision "
        "recorded in this module's docstring",
        "physical"),
    ("zacharegkas25", "alpha_shmr_k18"): _p(
        (-5.0, 0.0), "",
        "the Kravtsov et al. (2018) SHMR low-mass slope, which is negative in "
        "that parameterisation -- guo18 and guo19 spell a *positive* slope of "
        "a different SHMR with the same name",
        "physical"),
}


def _bound_for(model: str, name: str, default: float) -> Param:
    """The bound this model's parameter gets, published range first."""
    published = GalaxyParams._PARAMS
    if model.startswith("zumandelbaum") and name in published:
        return published[name]                    # by identity, deliberately
    spec = PER_MODEL.get((model, name)) or VOCABULARY.get(name)
    if spec is None:
        raise KeyError(
            f"{model!r} declares {name!r} and no bound is declared for it. Add "
            f"one to VOCABULARY (if it is a quantity the other models share) or "
            f"to PER_MODEL (if the name means something particular here). A "
            f"number a user may vary is a parameter with a bound, a prior and a "
            f"reason -- that is the rule this module exists to extend to the "
            f"rest of layer 3.")
    return Param(float(default), **spec)


def params_for(model: str) -> dict[str, Param]:
    """``{name: Param}`` for one occupation or CLF model."""
    defaults = DEFAULTS.get(model) or CLF_DEFAULTS.get(model)
    if defaults is None:
        raise ValueError(
            f"unknown occupation or CLF model {model!r}; expected one of "
            f"{sorted(set(DEFAULTS) | set(CLF_DEFAULTS))}")
    return {n: _bound_for(model, n, v) for n, v in defaults.items()}


def _make(model: str) -> type:
    """Build one frozen, registered ``SectorParams`` subclass for ``model``."""
    params = params_for(model)
    cal = OCC_CALIBRATION.get(model) or CLF_CALIBRATION.get(model)
    name = "".join(w.capitalize() for w in model.replace("_", " ").split())
    doc = (f"Declared parameters for the ``{model}`` occupation.\n\n"
           f"    Fitted to {cal.fit}"
           + (f" over z in {cal.z_range}" if cal.z_range else "")
           + (f", {cal.selection}" if cal.selection else "")
           + ".  Defaults are that fit's values, from\n"
           f"    ``occupation.DEFAULTS``; bounds are resolved as this module's\n"
           f"    docstring describes.\n")
    cls = type(f"{name}Params", (SectorParams,), {
        "__doc__": doc,
        "__annotations__": {n: float for n in params},
        "_PARAMS": params,
        "_STATIC": (),
        **{n: p.default for n, p in params.items()},
    })
    cls.__module__ = __name__
    return sector_params(cls)


#: ``model -> SectorParams`` subclass, one per occupation and CLF.
OCCUPATION_PARAMS: dict[str, type] = {
    m: _make(m) for m in list(DEFAULTS) + list(CLF_DEFAULTS)}

# Export each class by name, so the layer-3 container audits -- which walk
# `sectors.__all__` -- reach them.  A container they cannot see is a container
# with no bounds test, which is how `GalaxyParams` went unchecked.
for _model, _cls in OCCUPATION_PARAMS.items():
    globals()[_cls.__name__] = _cls
    __all__.append(_cls.__name__)
del _model, _cls


def make_occupation_params(model: str) -> SectorParams:
    """An instance of ``model``'s container, at its defaults."""
    try:
        return OCCUPATION_PARAMS[model]()
    except KeyError:
        raise ValueError(
            f"unknown occupation or CLF model {model!r}; expected one of "
            f"{sorted(OCCUPATION_PARAMS)}") from None
