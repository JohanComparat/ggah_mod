r"""Galaxies as a tracer: the occupation registry, wired to the contract.

:mod:`~ggah_mod.sectors.occupation` and :mod:`~ggah_mod.sectors.clf` hold fifteen
parameterisations between them and produce
:math:`(\langle N_{\rm cen}\rangle, \langle N_{\rm sat}\rangle)`.  Nothing turned
that pair into :class:`~ggah_mod.sectors.protocol.TracerWeights`, so the package
had a galaxy *model* and no galaxy *tracer* -- layer 4 could compute a gas or an
AGN spectrum and not a galaxy one, which is the reverse of the predecessor's
problem and just as blocking.

This module is that wiring, and nothing else.  It adds no occupation physics: a
test asserts the occupations it returns are identical to calling the registry
function directly.

Three things it has to settle
-----------------------------

**One registry or two?**  One.  :data:`~ggah_mod.sectors.occupation.OCCUPATION`
and :data:`~ggah_mod.sectors.clf.CLF` have the same shape -- ``name ->
(central, satellite)``, both taking ``log10m`` first and their parameters by
keyword -- so a single wrapper serves both, and which registry a name came from
stops being something a caller tracks.  The names are checked to be disjoint at
import, because two registries answering to one name is how a "cacciato09" fit
silently becomes a "zheng07" one.

**Which parameters does a model take?**  Read off its signature, once, at
construction.  The registry's ``DEFAULTS`` carry a *superset* for several models
-- ``zheng07``'s entry has five keys and ``n_cen_zheng07`` takes two of them --
so passing the dict straight through is a ``TypeError`` waiting for the first
caller.  Filtering by signature is introspection at construction time, not on
the traced path.

**Satellite stellar mass is not the halo's SHMR**, and this module does not
pretend otherwise.  :meth:`GalaxySector.stellar_fraction` returns the central
part exactly and the satellite part from the conditional stellar-mass function
the occupation already carries -- see :meth:`_f_star_sat_csmf`.

That route needs **no subhalo mass function**, and this paragraph used to say it
did.  The claim described the SHAM construction, in which a satellite's stellar
mass comes from its own subhalo at infall; an HOD parameterises satellites *per
host* and never resolves one.  Only the second is available here, which is the
opposite of what was written.  The real caveat is that the integral is truncated
at the range the occupation was fitted over.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp

from ..backend import resolve_backend
from ..halos.profiles import satellite_uk
from .calibration import check_sector_calibration
from .clf import CLF, CLF_CALIBRATION, clf_defaults
from .miscentering import P_OFF, R_OFF
from .occupation import OCCUPATION, OCC_CALIBRATION, occupation_defaults
from .params import Flat, Param, SectorParams, sector_params
from .protocol import TracerWeights
from .sham import (SHMR, SHMR_CALIBRATION, SHMR_MASS_UNITS,
                   halo_mass_to_relation, make_shmr, stellar_mass_to_msun)

__all__ = ["GalaxySector", "GalaxyParams", "GALAXY_MODELS", "galaxy_defaults",
           "SATELLITE_PROFILE_PARAMS", "SatelliteProfileParams",
           "THRESHOLD_SHMR", "THRESHOLD_MASS_UNITS"]

#: ``name -> (n_cen, n_sat)`` over both registries.
#:
#: Built by merging rather than by chaining two lookups, so a name collision is
#: an ``ImportError`` here instead of a silent precedence rule discovered later.
GALAXY_MODELS: dict[str, tuple] = {}
_clash = set(OCCUPATION) & set(CLF)
if _clash:
    raise ImportError(
        f"the occupation and CLF registries both define {sorted(_clash)}; one "
        f"name must mean one model, so rename one of them rather than letting "
        f"a lookup order decide")
GALAXY_MODELS.update(OCCUPATION)
GALAXY_MODELS.update(CLF)

#: Satellite-profile freedoms, which belong to the *tracer* rather than to the
#: occupation: how many satellites there are and where they sit are separate
#: questions, and only the second one is a profile.  Defaults recover NFW
#: exactly, so a model that says nothing about them is unchanged.
SATELLITE_PROFILE_PARAMS = ("b_sat_conc", "f_cut_inv", "gamma_inner")


@sector_params
class SatelliteProfileParams(SectorParams):
    r"""Where satellites sit, with bounds and reasons.

    The three freedoms of :func:`~ggah_mod.halos.profiles.satellite_uk`, which
    until now were bare signature defaults -- three numbers with no bound, no
    prior and no recorded reason, in a package whose standing rule is that a
    constant a user might want to vary is a parameter with all three.  They are
    exactly the parameters a mis-centring fit must vary *with* ``p_off``:
    holding the satellite profile at NFW while fitting mis-centring puts the
    profile's error into the displaced fraction, because both act on the same
    scales and only one of them is then allowed to move.

    Every default recovers NFW, so a model that says nothing about satellites
    is unchanged by this container existing.

    Bounds
    ------
    ``b_sat_conc`` and ``gamma_inner`` take the predecessor's boxes
    (``hod_mod``'s ``fit_bgs_multiprobe.py``, satellite extensions A and C),
    whose parameterisation matches this one.

    ``f_cut_inv`` does **not**, for two reasons.  It is the *reciprocal* of a
    truncation scale (see
    :func:`~ggah_mod.halos.profiles.satellite_uk` for why zero had to become
    the interior limit rather than an isolated special case).  And the
    predecessor's ``f_cut`` is an *inner* suppression,
    :math:`1-e^{-r/f_{\rm cut}r_{\rm vir}}`, while this is an *outer*
    truncation -- same name, opposite end of the halo.  Carrying its
    ``(0, 0.3)`` box across would have imported a bound derived for different
    physics, which is the kind of thing a bounds table exists to prevent.
    """

    b_sat_conc: float = 1.0
    f_cut_inv: float = 0.0
    gamma_inner: float = 0.0

    _PARAMS = {
        "b_sat_conc": Param(
            1.0, (0.3, 3.0), Flat(), "",
            "multiplies the halo concentration, c_sat = b * c: above 1 the "
            "satellites are more concentrated than the mass (tidal stripping "
            "retaining inner orbits), below it they are puffed outward. The "
            "box is hod_mod's satellite extension A. Strictly positive is "
            "definitional -- r_s = r_delta / (b*c), so zero is not a radius -- "
            "and 0.3 to 3 is the factor-of-three either side over which the "
            "profile still resembles a satellite distribution", "physical"),
        "f_cut_inv": Param(
            0.0, (0.0, 10.0), Flat(), "1/r_delta",
            "inverse outer truncation scale: the profile carries "
            "exp(-f_cut_inv * r / r_delta), so the truncation radius is "
            "r_delta / f_cut_inv and zero means no truncation. Zero-neutral "
            "and continuous there, which the previous f_cut was not -- it "
            "returned NFW at exactly 0 and a point mass in the limit from "
            "above, with a nan gradient at the default. The upper bound "
            "truncates at 0.1 r_delta, beyond which the satellites are "
            "confined so far inside the halo that the distribution is no "
            "longer describing satellites of it", "physical"),
        "gamma_inner": Param(
            0.0, (0.0, 3.0), Flat(), "",
            "extra inner power (r/r_s)^-gamma on top of NFW, from orbital "
            "energy redistribution (van den Bosch et al. 2005); hod_mod's "
            "satellite extension C. Zero is NFW. Negative is excluded because "
            "it would make the satellite profile shallower than the mass at "
            "small r, which the mechanism it models cannot do", "physical"),
    }
    _STATIC = ()


def galaxy_defaults(name: str) -> dict:
    """The default parameters for one model, from either registry: the
    published fit where one exists, illustrative values otherwise (the
    registries' ``DEFAULTS`` comments say which)."""
    key = str(name).lower()
    if key in OCCUPATION:
        return occupation_defaults(key)
    if key in CLF:
        return clf_defaults(key)
    raise ValueError(f"unknown galaxy model {name!r}; expected one of "
                     f"{sorted(GALAXY_MODELS)}")


def _accepted(fn) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(required, optional)`` keyword names, minus the leading ``log10m``.

    ``inspect`` at construction, never per call.  The registry functions are
    ``jax.jit``-wrapped, so the signature is read off ``__wrapped__`` when one
    is present -- otherwise every model would appear to take ``(*args,
    **kwargs)`` and the filter would pass everything through, which is the
    failure this function exists to prevent and which would look like it
    working.

    **The split matters.**  A parameter with a signature default is a published
    constant the model may be run without: ``clf_satellite_vdb13``'s
    ``f_s_star = 0.562`` is Cacciato et al.'s ratio :math:`L_s^*/L_c`, and
    neither ``CLF_DEFAULTS`` entry lists it; ``n_sat_vanuitert16``'s
    ``f_s_star = 0.56`` is the stellar-mass counterpart, likewise absent from
    ``DEFAULTS``.  Demanding every named parameter would refuse those two of
    the fifteen models over a number they already carry;
    silently dropping the required/optional distinction would instead let a
    genuinely missing parameter through to a ``TypeError`` deep inside a jitted
    kernel.  So: required names must be supplied, optional ones are forwarded
    when present and left to the function otherwise.
    """
    target = inspect.unwrap(fn)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):                      # pragma: no cover
        raise TypeError(
            f"cannot read the signature of {fn!r}, so there is no way to tell "
            f"which parameters it takes; a registry entry must be an ordinary "
            f"function")
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD,
             inspect.Parameter.KEYWORD_ONLY)
    params = [q for q in sig.parameters.values() if q.kind in kinds][1:]
    required = tuple(q.name for q in params if q.default is inspect.Parameter.empty)
    optional = tuple(q.name for q in params if q.default is not inspect.Parameter.empty)
    return required, optional


#: The amplitudes of the assembly-bias decoration this package used to carry.
REMOVED_ASSEMBLY_BIAS = ("a_cen", "a_sat")


def _refuse_removed_assembly_bias(p) -> None:
    r"""Raise if a caller still passes the removed assembly-bias amplitudes.

    The decoration rescaled :math:`b(M)` by :math:`1 + A\,(b-1)/\max(b, 1/2)`.
    It was removed because that kernel has no published source, and its
    amplitudes were named after the Hearin et al. (2016) decorated HOD without
    being that model's amplitudes (paper, App. B).  ``weights`` reads parameters
    by name, so without this check a stale ``a_cen`` in a dictionary would be
    ignored -- a fitted parameter that does nothing, which is the one outcome
    this package refuses.
    """
    stale = sorted(n for n in REMOVED_ASSEMBLY_BIAS if n in p)
    if stale:
        raise ValueError(
            f"{stale} are the amplitudes of an assembly-bias decoration that "
            f"has been removed from ggah_mod: its kernel had no published "
            f"source, and its amplitudes were not those of the Hearin et al. "
            f"(2016) decorated HOD they were named after. The package models no "
            f"assembly bias; drop these keys.")


def _is_active(v) -> bool:
    """Whether a switch parameter is non-zero, when that can be known.

    Under ``jit`` or ``grad`` the value is a tracer and there is nothing to
    compare, so the answer is *yes*: applying a correction that happens to be
    zero costs a multiplication and gives the exactly-neutral answer, while
    skipping it on a guess would make the traced result differ from the
    untraced one and would drop the gradient with respect to the very
    parameter being switched.
    """
    try:
        return float(v) != 0.0
    except Exception:
        return True


@sector_params
class GalaxyParams(SectorParams):
    r"""Zu \& Mandelbaum's iHOD and halo quenching, with bounds and reasons.

    The default galaxy occupation, and the only one in this layer whose whole
    parameter set carries declared bounds.  Every bound below is the **uniform
    prior range the papers themselves used** --- Paper I Table 2 for the
    thirteen iHOD parameters, Paper III Table 2 for the four quenching ones ---
    rather than a range chosen here.  That is the point: a bound invented by a
    downstream package looks exactly like a bound a measurement established,
    and only one of them may be relaxed on the strength of new data.  The
    exception is ``fc``: Paper I's :math:`f_c` is the satellite-to-matter
    concentration ratio (``b_sat_conc`` of :class:`SatelliteProfileParams`
    here), so this central amplitude is an addition to Paper I's model, and
    its (0.1, 1) box is chosen here: the upper edge is definitional, since a
    mean number of centrals cannot exceed one.

    All seventeen are free.  Zu \& Mandelbaum fitted them jointly to clustering
    and galaxy--galaxy lensing across eight stellar-mass samples, so fixing any
    of them at its posterior mean would import a constraint from that data set
    without importing its covariance.

    **The thirteen iHOD defaults are not Zu \& Mandelbaum's** (0.8.5, refitted
    in 1.1.0).  They are the maximum-likelihood point of ggah_cal's
    ``massbins_zu15_gt10.0_sys0.05_nbar-wp`` on ggah_mod 1.1.0.dev2 and
    sum_stat 0.6.0: :math:`\bar n` and :math:`w_p` of all eight LS10 BGS-like
    stellar-mass bins, :math:`10 \le \lg(M_*/M_\odot) < 12`, at
    :math:`0.05 < z < 0.18` (bin mean redshifts 0.134--0.141), the model of a
    bin a difference of two thresholds; jackknife covariance plus a systematic
    error of 5 per cent of every point in quadrature (photometric-redshift and
    stellar-mass errors); :math:`\chi^2 = 195.61` for 192 degrees of freedom,
    rounded here to three decimals.  The four quenching defaults are still
    Paper III's, so the ``zumandelbaum16`` colours combine two fits.  The
    published iHOD values are
    :data:`~ggah_mod.sectors.occupation.ZU15_PUBLISHED`; the 1.1.0.dev1-dev2
    ones, fitted to the five bins above 10^10.6 Msun with the jackknife alone,
    are :data:`~ggah_mod.sectors.occupation.ZU15_LS10_110`, and the 0.8.5 ones
    :data:`~ggah_mod.sectors.occupation.ZU15_LS10_085`.

    The eighteenth field, :attr:`log10m_star_thresh`, is the sample selection
    and not a model parameter: it says which galaxies are being described.  It
    is bounded by validity --- the published samples span
    :math:`8.5 < \lg M_* < 12.0` --- and it is a parameter rather than a static
    field because a forward model may want its gradient when fitting a
    threshold that is itself uncertain.
    """

    # -- the SHMR (Paper I, Eq. 19) ------------------------------------------
    log10m_star_thresh: float = 10.157
    lg_m1h: float = 12.131
    lg_m0star: float = 10.238
    beta: float = 0.605
    delta: float = 0.684
    gamma: float = 2.177
    # -- scatter and the central duty cycle ----------------------------------
    sigma_lnmstar: float = 0.734
    eta: float = -0.143
    fc: float = 0.726
    # -- satellites (Paper I, Eq. 22) ----------------------------------------
    bsat: float = 11.28
    beta_sat: float = 0.658
    bcut: float = 0.716
    beta_cut: float = 0.161
    alpha_sat: float = 1.057
    # -- halo quenching (Paper III, Eqs. 12-13) ------------------------------
    lg_mh_qc: float = 12.20
    mu_c: float = 0.38
    lg_mh_qs: float = 12.17
    mu_s: float = 0.15
    # -- mis-centring (More et al. 2015, Eq. 9) ------------------------------
    # Shared with MisCenteringParams: see miscentering.P_OFF / R_OFF.
    p_off: float = 0.0
    r_off: float = 0.25

    _PARAMS = {
        "log10m_star_thresh": Param(
            10.157, (8.5, 12.0), Flat(), "log10 h^-2 Msun",
            "the published samples span this range (Paper I Table 1); outside "
            "it the model still evaluates but describes a selection nobody "
            "measured.  The default is 10^10.5 Msun at h = 0.6736, a choice "
            "inside the 10^10-10^12 Msun the other defaults are fitted over",
            "validity"),
        "lg_m1h": Param(
            12.131, (9.5, 14.0), Flat(), "log10 h^-1 Msun",
            "Paper I Table 2 uniform prior; characteristic halo mass of the "
            "SHMR", "prior"),
        "lg_m0star": Param(
            10.238, (9.0, 13.0), Flat(), "log10 h^-2 Msun",
            "Paper I Table 2 uniform prior; characteristic stellar mass of "
            "the SHMR", "prior"),
        "beta": Param(
            0.605, (0.0, 2.0), Flat(), "",
            "Paper I Table 2 uniform prior; low-mass slope of the SHMR.  "
            "Negative would make stellar mass fall with halo mass", "physical"),
        "delta": Param(
            0.684, (0.0, 1.5), Flat(), "",
            "Paper I Table 2 uniform prior; controls the high-mass slope",
            "prior"),
        "gamma": Param(
            2.177, (-0.1, 4.9), Flat(), "",
            "Paper I Table 2 uniform prior; controls the intermediate-mass "
            "behaviour", "prior"),
        "sigma_lnmstar": Param(
            0.734, (0.01, 3.0), Flat(), "",
            "Paper I Table 2 uniform prior; a log-normal width, so zero is "
            "not a width and the lower bound is definitional", "definitional"),
        "eta": Param(
            -0.143, (-0.4, 0.4), Flat(), "",
            "Paper I Table 2 uniform prior; the slope with which the scatter "
            "runs with halo mass", "prior"),
        "fc": Param(
            0.726, (0.1, 1.0), Flat(), "",
            "Not in Paper I, whose central occupation has no amplitude: an "
            "addition here, the high-mass limit of <N_cen>, i.e. the central "
            "completeness.  A mean number of centrals cannot exceed one, so the "
            "upper edge is definitional; 0.1 keeps n_bar away from zero.  "
            "Paper I's own f_c is the satellite concentration ratio (bsat_conc "
            "here)", "definitional"),
        "bsat": Param(
            11.28, (0.01, 25.0), Flat(), "",
            "Paper I Table 2 uniform prior; normalises M_sat", "prior"),
        "beta_sat": Param(
            0.658, (0.1, 1.8), Flat(), "",
            "Paper I Table 2 uniform prior; slope of the M_sat scaling",
            "prior"),
        "bcut": Param(
            0.716, (0.0, 6.0), Flat(), "",
            "Paper I Table 2 uniform prior; normalises M_cut", "prior"),
        "beta_cut": Param(
            0.161, (-0.05, 1.50), Flat(), "",
            "Paper I Table 2 uniform prior; slope of the M_cut scaling",
            "prior"),
        "alpha_sat": Param(
            1.057, (0.5, 1.5), Flat(), "",
            "Paper I Table 2 uniform prior; power-law slope of the satellite "
            "occupation", "prior"),
        "lg_mh_qc": Param(
            12.20, (11.0, 15.5), Flat(), "log10 h^-1 Msun",
            "Paper III Table 2 uniform prior; the critical halo mass at which "
            "centrals quench", "prior"),
        "mu_c": Param(
            0.38, (0.0, 3.0), Flat(), "",
            "Paper III Table 2 uniform prior; the powered-exponential index "
            "of the central transition.  Zero makes the red fraction "
            "identically zero at every mass, so the lower bound is "
            "definitional", "definitional"),
        "lg_mh_qs": Param(
            12.17, (11.0, 15.5), Flat(), "log10 h^-1 Msun",
            "Paper III Table 2 uniform prior; the same for satellites, and "
            "the paper's result is that it agrees with the central one",
            "prior"),
        "mu_s": Param(
            0.15, (0.0, 3.0), Flat(), "",
            "Paper III Table 2 uniform prior; the satellite index, which does "
            "*not* agree with the central one -- same threshold, gentler "
            "transition", "definitional"),
        "p_off": P_OFF,
        "r_off": R_OFF,
    }
    _STATIC = ()


#: Where the conditional stellar mass function is integrated, in
#: :math:`\lg M_*/(h^{-2}M_\odot)`.
#:
#: The range Zu \& Mandelbaum's samples span (their Table 1).  Satellites
#: fainter than the lower limit contribute nothing, so
#: :meth:`GalaxySector._f_star_sat_csmf` returns a **lower bound** -- a stated
#: truncation rather than a missing ingredient.  The AGN sector does not use
#: this range: it integrates satellites on its own physical grid from its
#: stellar-mass cut, 10^8 Msun (:data:`~ggah_mod.sectors.agn.LG_MSTAR_MIN`).
CSMF_LG_MSTAR_RANGE = (8.5, 12.0)
#: Nodes on that range.  The integrand is smooth and the quadrature is
#: trapezoidal, so this is convergence rather than resolution.
CSMF_N_MSTAR = 256

#: ``occupation -> (SHMR name, {relation argument: parameter name})`` for the
#: occupations that are **threshold** models: their central term is a stellar-mass
#: relation with scatter, cut at ``log10m_star_thresh``, so they carry a stellar
#: mass for centrals *and*, differentiated in that threshold, the conditional
#: stellar-mass function of their satellites.  The same five that
#: :meth:`GalaxySector.stellar_fraction` routes to :meth:`GalaxySector.satellite_csmf`.
#:
#: The relation is the one the occupation itself evaluates, so the stellar mass
#: this sector reports is the one that counted the galaxies.  The argument map is
#: only non-trivial for ``zacharegkas25``, which calls Kravtsov et al. (2018) with
#: its own parameter names; a test checks every row against its occupation.
THRESHOLD_SHMR: dict[str, tuple[str, dict[str, str]]] = {
    "zumandelbaum15": ("zu15", {}),
    "zumandelbaum16_red": ("zu15", {}),
    "zumandelbaum16_blue": ("zu15", {}),
    "leauthaud12": ("leauthaud12", {}),
    "zacharegkas25": ("kravtsov18", {"log10m1": "log10m1_shmr",
                                     "alpha": "alpha_shmr_k18",
                                     "gamma": "gamma_shmr",
                                     "delta": "delta_shmr"}),
}

#: ``occupation -> (n_halo, n_star, h_ref)``: the units each threshold
#: occupation is written in, which are its relation's
#: (:data:`~ggah_mod.sectors.sham.SHMR_MASS_UNITS`).  Its parameters, its
#: threshold and the halo mass its central and satellite terms are evaluated at
#: are all in these units; :class:`GalaxySector` converts the field's
#: :math:`M_\odot/h` into them, and the stellar masses back out.
#:
#: ``ggah_cal.predict.galaxies.MSTAR_UNITS`` carries the stellar half for its
#: thresholds; the two must agree.
THRESHOLD_MASS_UNITS: dict[str, tuple[int, int, float]] = {
    model: SHMR_MASS_UNITS[relation]
    for model, (relation, _) in THRESHOLD_SHMR.items()
}


class GalaxySector:
    """Galaxies as a discrete tracer.  A **peer**: it owns no grid.

    Parameters
    ----------
    model : str
        Any key of :data:`GALAXY_MODELS`.
    shmr : str, optional
        Which stellar-mass--halo-mass relation :meth:`stellar_fraction` uses.
        There is no default: four of the occupations carry no stellar mass at
        all, and substituting one silently is how a lensing point mass acquires
        a number nobody chose.
    backend : Backend or str, optional
        Supplies ``n_gl`` for the satellite profile quadrature.
    """

    name = "galaxies"
    differentiable = True

    def __init__(self, model: str = "zumandelbaum15", *, shmr: str | None = None,
                 backend=None, calibration: str = "warn"):
        key = str(model).lower()
        if key not in GALAXY_MODELS:
            raise ValueError(f"unknown galaxy model {model!r}; expected one of "
                             f"{sorted(GALAXY_MODELS)}")
        if shmr is not None and str(shmr).lower() not in SHMR:
            raise ValueError(f"unknown SHMR {shmr!r}; expected one of "
                             f"{sorted(SHMR)}")
        if (key in THRESHOLD_SHMR and shmr is not None
                and str(shmr).lower() != THRESHOLD_SHMR[key][0]):
            raise ValueError(
                f"occupation {key!r} counts its galaxies with the "
                f"{THRESHOLD_SHMR[key][0]!r} stellar-mass relation, so "
                f"shmr={shmr!r} would report a second stellar mass for the "
                f"same centrals.  Leave shmr unset; the occupation's own "
                f"relation is used.")
        self.model = key
        self.shmr = None if shmr is None else str(shmr).lower()
        self._cen, self._sat = GALAXY_MODELS[key]
        self._cen_req, self._cen_opt = _accepted(self._cen)
        self._sat_req, self._sat_opt = _accepted(self._sat)
        self._backend = resolve_backend(backend)
        #: Checked in `weights`, not here: the fitted range is a redshift, and
        #: a sector does not learn one until a `HaloField` arrives.
        self.calibration = str(calibration)

    def __repr__(self) -> str:                           # pragma: no cover
        return f"GalaxySector(model={self.model!r}, shmr={self.shmr!r})"

    def _check_calibration(self, field):
        """Both registries this sector draws from, against the field's z.

        ``GALAXY_MODELS`` is ``OCCUPATION`` merged with ``CLF``, so
        ``self.model`` is a key of one or the other and the lookup has to try
        both -- a single table would have to be a third copy of two that are
        maintained beside their models.  ``self.shmr`` is optional and is
        checked only when one was chosen.
        """
        for table, kind in ((OCC_CALIBRATION, "occupation"),
                            (CLF_CALIBRATION, "conditional luminosity function")):
            if self.model in table:
                check_sector_calibration(kind, self.model, table, field.z,
                                         policy=self.calibration)
        if self.shmr is not None:
            check_sector_calibration("SHMR", self.shmr, SHMR_CALIBRATION,
                                     field.z, policy=self.calibration)

    # ------------------------------------------------------------ occupation
    @staticmethod
    def _as_dict(params) -> dict:
        """Accept a mapping or a :class:`SectorParams`; return a mapping."""
        return dict(params if isinstance(params, dict) else params.as_dict())

    def occupation(self, field, params):
        r""":math:`(\langle N_{\rm cen}\rangle, \langle N_{\rm sat}\rangle)`.

        Each model is handed exactly the parameters its own signature names, so
        a ``DEFAULTS`` entry carrying more than one of the pair needs -- which
        several do -- is not an error.
        """
        p = self._as_dict(params)
        log10m = self._native_log10m(field)
        missing = [n for n in set(self._cen_req) | set(self._sat_req)
                   if n not in p]
        if missing:
            raise ValueError(
                f"model {self.model!r} needs {sorted(missing)}, which the "
                f"parameters do not carry; galaxy_defaults({self.model!r}) "
                f"lists the full set")

        def _kw(req, opt):
            return {n: p[n] for n in req} | {n: p[n] for n in opt if n in p}

        n_cen = self._cen(log10m, **_kw(self._cen_req, self._cen_opt))
        n_sat = self._sat(log10m, **_kw(self._sat_req, self._sat_opt))
        return n_cen, n_sat

    def number_density(self, field, params, view: str = "total"):
        r""":math:`\bar n` for one view, in :math:`(\mathrm{Mpc}/h)^{-3}`."""
        return field.number_density(self._occupation_for(field, params, view))

    def effective_bias(self, field, params, view: str = "total"):
        r""":math:`b_{\rm eff}` for one view."""
        return field.effective_bias(self._occupation_for(field, params, view))

    def _occupation_for(self, field, params, view: str):
        n_cen, n_sat = self.occupation(field, params)
        if view == "total":
            return n_cen + n_sat
        if view == "cen":
            return n_cen
        if view == "sat":
            return n_sat
        raise ValueError(f"unknown galaxy view {view!r}; expected one of "
                         f"['cen', 'sat', 'total']")

    # --------------------------------------------------------------- profile
    def satellite_uk(self, field, params):
        r""":math:`u_{\rm sat}(k|M)`, shape ``(Nk, NM)``, :math:`u(k\to0) = 1`.

        Reads the three shape freedoms from ``params`` when they are there and
        uses :func:`~ggah_mod.halos.profiles.satellite_uk`'s own defaults when
        they are not -- which is NFW exactly, not approximately.
        """
        p = self._as_dict(params)
        kw = {n: p[n] for n in SATELLITE_PROFILE_PARAMS if n in p}
        return satellite_uk(field.k, field.r_delta, field.conc,
                            n_gl=self._backend.n_gl, **kw)

    # --------------------------------------------------------- stellar mass
    def _relation(self):
        """``(name, argument map, units)`` of the stellar-mass relation."""
        if self.model in THRESHOLD_SHMR:
            name, rename = THRESHOLD_SHMR[self.model]
        elif self.shmr is not None:
            name, rename = self.shmr, {}
        else:
            raise ValueError(
                f"GalaxySector({self.model!r}) was built without an `shmr=`, so "
                f"there is no stellar mass to report.  Several occupations "
                f"carry none of their own, and choosing one here on their "
                f"behalf would put a number into the lensing point mass that "
                f"nobody selected.  Pass one of {sorted(SHMR)}.")
        return name, rename, SHMR_MASS_UNITS[name]

    def _native_log10m(self, field):
        r"""The field's halo masses in this occupation's units.

        Only a threshold occupation declares units; every other one is written
        in :math:`h^{-1}M_\odot`, the field's own, and sees them unchanged.
        """
        log10m = jnp.log10(field.m)
        if self.model not in THRESHOLD_MASS_UNITS:
            return log10m
        return halo_mass_to_relation(log10m, THRESHOLD_MASS_UNITS[self.model],
                                     field.cosmo.h)

    def _log10_mstar_native(self, log10m_native, params, *, h=None, z=None):
        r""":math:`\log_{10}M_\star^{\rm c}` in the relation's own units, for a
        halo mass already in them -- what the occupation counts centrals with.

        ``h`` and ``z`` are handed to a relation that takes them and was not
        given them in ``params``; the redshift-dependent relations used to be
        evaluated at their signature default, :math:`z = 0`, on every field.
        """
        p = self._as_dict(params)
        name, rename, _ = self._relation()
        fn = make_shmr(name)
        req, opt = _accepted(fn)
        supplied = dict(p)
        if h is not None:
            supplied.setdefault("h", h)
        if z is not None:
            supplied.setdefault("z", z)
        missing = [rename.get(n, n) for n in req
                   if rename.get(n, n) not in supplied]
        if missing:
            raise ValueError(
                f"SHMR {name!r} needs {sorted(missing)}, which the "
                f"parameters do not carry")
        kw = {n: supplied[rename.get(n, n)] for n in req + opt
              if rename.get(n, n) in supplied}
        return fn(jnp.asarray(log10m_native), **kw)

    def log10_mstar_msun(self, log10m, params, *, h, z=None):
        r""":math:`\log_{10}(M_\star^{\rm c}/M_\odot)`, **physical**, for halo
        masses ``log10m`` in :math:`M_\odot/h`.

        The relation is evaluated in its own units
        (:data:`~ggah_mod.sectors.sham.SHMR_MASS_UNITS`), which is what its
        parameters and threshold are written in: the halo mass is converted in,
        the stellar mass out.  For ``zumandelbaum15`` at Planck 2018 the first is
        the identity and the second adds :math:`-2\log h = 0.343` dex.  The AGN
        chain's :math:`M_{\rm BH}`-:math:`M_\star` relation takes it from here.

        :class:`~ggah_mod.sectors.agn.AgnSector` reads it, so a galaxy sample and
        an AGN sample in one spectrum see one stellar mass.  They used to see
        two: the AGN sector evaluated ``zu15`` at its own copy of the fiducial
        parameters, and then in the wrong units.
        """
        _, _, units = self._relation()
        native = self._log10_mstar_native(
            halo_mass_to_relation(log10m, units, h), params, h=h, z=z)
        return stellar_mass_to_msun(native, units, h)

    def log10_mstar(self, log10m, params, *, h, z=None):
        r""":math:`\log_{10}(M_\star^{\rm c}/(M_\odot/h))`: the stellar mass in the
        package's convention, the one :math:`M_h` is in, so that
        :math:`M_\star/M_h` is a fraction.  :meth:`stellar_fraction` uses it.

        A threshold occupation (:data:`THRESHOLD_SHMR`) uses the relation it
        counts its centrals with, under its own parameter names.  Any other
        occupation carries no stellar mass, and uses the ``shmr=`` it was built
        with.  The relation's own units are converted in and out; the
        ``zumandelbaum15`` stellar fraction was a factor :math:`h` low while
        its :math:`h^{-2}M_\odot` was read as :math:`M_\odot/h`.
        """
        return self.log10_mstar_msun(log10m, params, h=h, z=z) + jnp.log10(h)

    @property
    def is_threshold(self) -> bool:
        """Whether this occupation carries a stellar mass for centrals *and* a
        conditional stellar-mass function for satellites (:data:`THRESHOLD_SHMR`)."""
        return self.model in THRESHOLD_SHMR

    def stellar_fraction(self, field, params, *, satellites: bool = False):
        r""":math:`(f_\star^{\rm cen}, f_\star^{\rm sat})` on the mass grid.

        What :func:`~ggah_mod.sectors.matter.matter_weights` needs, and what
        nothing computed before: the test for the whole layer-3 chain
        hand-rolled it from :func:`~ggah_mod.sectors.sham.mstar_from_mh_zu15`
        at the call site.

        The central part is exact:

        .. math::  f_\star^{\rm cen}(M) = N_{\rm cen}(M)\,M_\star(M)/M

        The satellite part is **off unless asked for**, and this is the
        interesting part
        ------------------------------------------------------------------

        A satellite's stellar mass is set by its own subhalo at infall, not by
        its host's SHMR, so the honest calculation is

        .. math::

            f_\star^{\rm sat}(M) = \frac{1}{M}\int dM_{\rm sub}\,
                \frac{dN}{dM_{\rm sub}}(M)\;M_\star(M_{\rm sub})

        over a **subhalo mass function**, which this package does not have --
        and which the exact route below does not need.  See
        :meth:`_f_star_sat_csmf`: an HOD gives the satellite count per host, so
        the conditional stellar-mass function integrates without ever resolving
        a subhalo.  The SHAM construction is the one that needs subhaloes, and
        it is not the one used here.

        The obvious placeholder -- give every satellite the host's central
        stellar mass, :math:`N_{\rm sat}(M)M_\star(M)/M` -- was written first and
        is **wrong by an order of magnitude at the top of the mass range**.
        Measured, ``zheng07`` x ``zu15`` at the published defaults:

        ==========  ========  ==============  =====================
        log10 M     N_sat     M_*(M_h)        N_sat M_*(M_h)/M
        ==========  ========  ==============  =====================
        12           0.33     1.53e10         0.005
        13           3.81     1.27e11         0.050
        14          39.75     5.17e11         0.205
        15         387.40     1.34e12         **0.535**
        ==========  ========  ==============  =====================

        Half the mass of a cluster in satellite stars.  It fails upward because
        :math:`M_\star(M_h)` saturates while :math:`N_{\rm sat}` keeps growing
        linearly, so the product diverges exactly where the term matters most --
        and it would have passed every :math:`k \to 0` check, because mass
        conservation closes by construction whichever component the stars sit
        in.  Recorded here rather than quietly replaced: it is the kind of
        placeholder that reads as physics.

        What is offered instead, and only on request, is the subhalo proxy

        .. math::  f_\star^{\rm sat}(M) = N_{\rm sat}(M)\,M_\star(f_{\rm sub}M)/M

        with :math:`f_{\rm sub}` the typical surviving-subhalo mass fraction --
        one free number, standing in for the integral above, and required
        explicitly rather than defaulted.  With ``satellites=False`` the
        satellite fraction is exactly zero: the centrals-only limit, which is a
        real limit and the one every existing test is stated in.
        """
        log10m = jnp.log10(field.m)
        h, z = field.cosmo.h, field.z
        log10_mstar = self.log10_mstar(log10m, params, h=h, z=z)
        p = self._as_dict(params)
        n_cen, n_sat = self.occupation(field, params)
        f_cen = n_cen * jnp.power(10.0, log10_mstar) / field.m

        if not satellites:
            return f_cen, jnp.zeros_like(f_cen)

        # The honest route, when the occupation can supply it.
        if "log10m_star_thresh" in self._sat_req + self._sat_opt:
            return f_cen, self._f_star_sat_csmf(field, p)

        if "f_sub" not in p:
            raise ValueError(
                "satellites=True needs `f_sub`, the typical surviving-subhalo "
                "mass fraction, in the parameters.  There is no default because "
                "the quantity it stands in for is an integral over a subhalo "
                "mass function this package does not have, and a plausible "
                "number here is indistinguishable from a computed one.  "
                f"Occupation {self.model!r} carries no stellar-mass threshold, "
                "so the conditional stellar mass function route is not "
                "available for it.")
        f_sub = jnp.asarray(p["f_sub"])
        log10m_sub = log10m + jnp.log10(f_sub)
        f_sat = n_sat * jnp.power(
            10.0, self.log10_mstar(log10m_sub, p, h=h, z=z)) / field.m
        return f_cen, f_sat

    def _f_star_sat_csmf(self, field, p):
        r"""Satellite stellar mass per host, from the occupation itself.

        No subhalo mass function, and none needed.  A threshold occupation *is*
        the conditional stellar mass function of its satellites:
        :math:`N_{\rm sat}(>M_*|M_h)` differentiated in its own threshold gives
        :math:`\dd N_{\rm sat}/\dd\lg M_*`, so the first moment is

        .. math::

            f_\star^{\rm sat}(M_h) = \frac{1}{M_h}\int \dd\lg M_*\,
                \frac{\dd N_{\rm sat}}{\dd\lg M_*}(M_*|M_h)\; M_* .

        The claim that this needs a subhalo mass function -- made in this
        docstring, in :mod:`~ggah_mod.sectors.agn`, and in the paper -- was
        wrong.  It described the SHAM route, in which a satellite's stellar
        mass comes from its own subhalo at infall; an HOD parameterises the
        satellites *per host* and never resolves one.  Both are legitimate and
        only the second is available here, which is the opposite of what was
        written.

        **The integral is truncated**, and that is the real caveat.  It runs
        over the range the occupation was fitted on
        (:data:`CSMF_LG_MSTAR_RANGE`), so satellites fainter than its lower
        limit contribute nothing and the result is a lower bound.  That is a
        stated, quantifiable omission; "we do not have a subhalo mass function"
        was not.

        Differentiable throughout: the derivative in the threshold is taken by
        :func:`jax.grad`, not by differencing.
        """
        lg_ms, dn_per_host = self.satellite_csmf_msun(field, p)
        lg_ms = lg_ms + jnp.log10(field.cosmo.h)            # Msun -> Msun/h
        return jnp.trapezoid(
            dn_per_host * jnp.power(10.0, lg_ms)[:, None], lg_ms, axis=0
        ) / field.m

    def satellite_csmf(self, field, params):
        r""":math:`\dd N_{\rm sat}/\dd\lg M_*\,(M_*|M_h)` on the fitted range.

        Returns ``(lg_ms, dn)`` with ``lg_ms`` of shape ``(CSMF_N_MSTAR,)`` over
        :data:`CSMF_LG_MSTAR_RANGE` and ``dn`` of shape ``(CSMF_N_MSTAR, NM)``:
        the satellites per host per dex of stellar mass.  Its first moment is
        :meth:`_f_star_sat_csmf`, and :class:`~ggah_mod.sectors.agn.AgnSector`
        integrates the black-hole chain against it.

        Only a threshold occupation *is* a conditional stellar-mass function, so
        any other raises.
        """
        return self._csmf(self._native_log10m(field), params, "sat")

    def satellite_csmf_msun(self, field, params, lg_ms_msun=None):
        r"""As :meth:`satellite_csmf`, with the stellar-mass axis in **physical**
        :math:`M_\odot`.  ``dn`` is per dex, so the shift of the axis leaves it
        unchanged.

        ``lg_ms_msun`` evaluates it on a caller's grid, in physical
        :math:`\log_{10}(M_*/M_\odot)`, instead of on
        :data:`CSMF_LG_MSTAR_RANGE`: the AGN sector integrates its satellites
        from its own stellar-mass cut, which is physical.
        """
        units = THRESHOLD_MASS_UNITS.get(self.model)
        if lg_ms_msun is None:
            lg_ms, dn = self.satellite_csmf(field, params)
            return stellar_mass_to_msun(lg_ms, units, field.cosmo.h), dn
        lg_ms_msun = jnp.asarray(lg_ms_msun)
        # The occupation's own units, as a shift: `stellar_mass_to_msun` is
        # linear in its argument with unit slope.
        native = lg_ms_msun - stellar_mass_to_msun(0.0, units, field.cosmo.h)
        _, dn = self._csmf(self._native_log10m(field), params, "sat",
                           lg_ms=native)
        return lg_ms_msun, dn

    def central_csmf(self, field, params):
        r""":math:`\dd N_{\rm cen}/\dd\lg M_*\,(M_*|M_h)`, the twin of
        :meth:`satellite_csmf`: the same threshold occupation differentiated in
        its own threshold, on the same grid and with the same shapes.

        For ``zumandelbaum15`` it is the log-normal
        :math:`f_{\rm c}\,\mathcal N(\lg M_*;\ \lg M_*^{\rm c}(M_h),\
        \sigma_{\ln M_*}/\ln 10)`, which is how it is tested.
        """
        return self._csmf(self._native_log10m(field), params, "cen")

    def central_csmf_msun(self, field, params, lg_ms_msun=None):
        r"""As :meth:`central_csmf`, with the stellar-mass axis in **physical**
        :math:`M_\odot`.

        ``lg_ms_msun`` evaluates it on a caller's physical grid, as
        :meth:`satellite_csmf_msun` does: the neutral-gas sector integrates
        centrals and satellites on one grid of its own, so that its HI mass
        function is the stellar-mass function convolved and nothing else.
        """
        units = THRESHOLD_MASS_UNITS[self.model]
        if lg_ms_msun is None:
            lg_ms, dn = self.central_csmf(field, params)
            return stellar_mass_to_msun(lg_ms, units, field.cosmo.h), dn
        lg_ms_msun = jnp.asarray(lg_ms_msun)
        native = lg_ms_msun - stellar_mass_to_msun(0.0, units, field.cosmo.h)
        _, dn = self._csmf(self._native_log10m(field), params, "cen",
                           lg_ms=native)
        return lg_ms_msun, dn

    def stellar_mass_function(self, field, params, *, msun: bool = False):
        r"""The galaxy stellar-mass function, centrals and satellites apart.

        .. math::

            \Phi_x(M_*) = \int \dd M\,\frac{\dd n}{\dd M}\,
            \frac{\dd N_x^{>M_*}}{\dd\lg M_*}(M_*|M), \qquad
            x \in \{{\rm cen}, {\rm sat}\},

        Returns ``(lg_ms, phi_cen, phi_sat)``, ``phi`` in
        :math:`(\mathrm{Mpc}/h)^{-3}\,\mathrm{dex}^{-1}` on
        :data:`CSMF_LG_MSTAR_RANGE`.  ``lg_ms`` is in the occupation's own
        units (:math:`h^{-2}M_\odot` for ``zumandelbaum15``), or in physical
        :math:`M_\odot` with ``msun=True``; ``phi`` is per dex, so the shift of
        the axis leaves it unchanged.  Only a threshold occupation has one.
        """
        log10m = self._native_log10m(field)
        lg_ms, dn_cen = self._csmf(log10m, params, "cen")
        _, dn_sat = self._csmf(log10m, params, "sat")
        phi_cen = field.integrate(field.dndm[None, :] * dn_cen, axis=-1)
        phi_sat = field.integrate(field.dndm[None, :] * dn_sat, axis=-1)
        if msun:
            lg_ms = stellar_mass_to_msun(
                lg_ms, THRESHOLD_MASS_UNITS[self.model], field.cosmo.h)
        return lg_ms, phi_cen, phi_sat

    def _csmf(self, log10m, params, which: str, lg_ms=None):
        """``-dN_x/d threshold`` for ``which`` in ``("cen", "sat")``, on
        :data:`CSMF_LG_MSTAR_RANGE` or on ``lg_ms`` (the occupation's units)."""
        fn, req, opt = {"cen": (self._cen, self._cen_req, self._cen_opt),
                        "sat": (self._sat, self._sat_req, self._sat_opt)}[which]
        if "log10m_star_thresh" not in req + opt:
            raise ValueError(
                f"occupation {self.model!r} carries no stellar-mass threshold, "
                f"so it has no conditional stellar-mass function to "
                f"differentiate; the threshold occupations are "
                f"{sorted(THRESHOLD_SHMR)}")
        p = self._as_dict(params)
        if lg_ms is None:
            lo, hi = CSMF_LG_MSTAR_RANGE
            lg_ms = jnp.linspace(lo, hi, CSMF_N_MSTAR)
        kw = {n: p[n] for n in req + opt
              if n in p and n != "log10m_star_thresh"}

        def n_above(thresh):
            return fn(log10m, log10m_star_thresh=thresh, **kw)

        # -dN/dthreshold at each M_*, for every host mass at once.  `jacrev`
        # rather than `grad`: the occupation returns one number per host, and
        # summing them first would give the derivative of the total instead of
        # the distribution each host carries.
        return lg_ms, -jax.vmap(jax.jacrev(n_above))(lg_ms)   # (n_ms, n_M)

    # ------------------------------------------------------- the contract
    def weights(self, field, params, view: str = "total") -> TracerWeights:
        r"""One of the three views, as :class:`TracerWeights`.

        ``discrete=True`` in every view: galaxies are countable, so the one-halo
        auto-spectrum must not pair a central with itself.  ``norm`` is that
        view's own :math:`\bar n`, so a satellite-only spectrum is per satellite
        rather than per galaxy.

        The ``"cen"`` view leaves ``w_extended`` as ``None`` for the same reason
        :class:`~ggah_mod.sectors.agn.AgnSector` does at ``f_duty_sat = 0``:
        with nothing on a profile, the one-halo auto-spectrum vanishes
        identically, and that is a
        statement the contract should carry rather than a small number layer 4
        computes.
        """
        if view not in ("total", "cen", "sat"):
            raise ValueError(f"unknown galaxy view {view!r}; expected one of "
                             f"['cen', 'sat', 'total']")
        self._check_calibration(field)
        n_cen, n_sat = self.occupation(field, params)
        occ = {"total": n_cen + n_sat, "cen": n_cen, "sat": n_sat}[view]
        n_bar = field.number_density(occ)

        w_point = None if view == "sat" else n_cen
        if view == "cen":
            w_extended = None
        else:
            w_extended = n_sat[None, :] * self.satellite_uk(field, params)

        # Number counts, so the per-object weight is 1 and the self-pair
        # weight is the occupation itself: shot noise comes out as 1/n_bar.
        w = TracerWeights(
            w_point=w_point, w_extended=w_extended, norm=n_bar,
            discrete=True, bias_weight=None, name=f"galaxies:{view}",
            self_pair=occ)

        # Mis-centring.  Part of the default configuration rather than
        # something a caller remembers to apply, and it reduces *exactly* to
        # the undecorated case at p_off = 0, so switching it off is a parameter
        # choice and not a different code path.
        p = self._as_dict(params)
        _refuse_removed_assembly_bias(p)
        p_off = p.get("p_off", 0.0)
        if _is_active(p_off) and w.w_point is not None:
            # Through the modifier that already exists rather than a second
            # implementation here: it multiplies the *point* component by
            # h(k|M) and leaves norm alone, because displacing a central does
            # not create or destroy one.  The "sat" view has no point component
            # and is skipped, which is correct -- there is no central in it to
            # displace.
            from .miscentering import MisCentering, MisCenteringParams

            w = MisCentering(convention="more15").apply(
                w, field, MisCenteringParams(p_off=p_off,
                                             r_off=p.get("r_off", 0.25)))

        return w
