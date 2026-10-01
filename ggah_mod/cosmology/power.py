r"""Linear matter power spectra: the backends, and the contract they share.

Every backend answers the same two questions:

``pk(k, z, cosmo)``
    the **total**-matter spectrum :math:`P_m(k,z)`, in :math:`(\mathrm{Mpc}/h)^3`
    with :math:`k` in :math:`h\,\mathrm{Mpc}^{-1}`;
``pk_cb(k, z, cosmo)``
    the **cold** (CDM + baryon) spectrum :math:`P_{cb}(k,z)`.

Both are needed and they are not interchangeable.  Halos form out of the cold
field, so :math:`\sigma(M)`, :math:`dn/dM` and :math:`b(M)` are built from
:math:`P_{cb}`; lensing and the matter 2-halo term see all the matter, so they
use :math:`P_m`.  At :math:`\Sigma m_\nu = 0.06` eV the two differ by a few
percent over the relevant scales, and using one where the other belongs is a
silent error of that size.

**There is no amplitude branch.**  Every backend is A_s-native.  A cosmology
cannot carry a ``sigma8`` (see
:mod:`ggah_mod.cosmology.parameters`), so the question of which amplitude wins
cannot arise.

Backends declare two capabilities rather than being sniffed for them:

``has_native_z``
    the backend computes its own redshift dependence, so
    :mod:`ggah_mod.cosmology.growth` can read the growth off it rather than
    integrating an approximation;
``differentiable``
    the backend is pure JAX and may appear inside a traced assembly.

A backend that silently ignored ``z`` would return :math:`D(z)=1` and look
perfectly exact, so the declaration is never inferred by calling it.
"""

from __future__ import annotations

import functools
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from .parameters import (Cosmology, LEAF_FIELDS, STATIC_FIELDS, _concrete,
                         nu_energy_factor)

__all__ = ["LinearPowerSpectrum", "ClassPk", "CambPk",
           "GgahEmuPk", "make_pk", "PK_BACKENDS", "Z_MAX_PK",
           "CLASS_PRECISION", "CAMB_PRECISION", "class_input", "camb_input"]


#: Redshift ceiling for CLASS's ``P(k,z)`` output.  **A constant, deliberately.**
#:
#: ``z_max_pk`` is a *solver* setting: it decides how far back CLASS keeps its
#: perturbation output, and with it the time sampling of that output.  This used
#: to be ``max(z.max(), 1.0)`` -- derived from the request -- which made
#: :math:`P(k,z)` a function not of :math:`(\theta, k, z)` but of
#: :math:`(\theta, k, z, \text{whatever else was asked for in the same call})`.
#: The same redshift reached through a one-node request and a two-node one came
#: back different: **9.6e-8** in :math:`\sigma_8`, which the mass function's
#: exponential turns into **1.3e-5** in :math:`\dd n/\dd M`.  A batched result and
#: a looped one therefore disagreed, and the memoisation *had* to carry the whole
#: redshift tuple to be correct rather than merely to be useful.
#:
#: The drift was never a trend, which is what says it is sampling and not
#: physics: measured at ``z_max_pk`` = 1, 2, 5, 10, the values at 2 and 10 agree
#: to every digit and 5 sits above both.  Several settings land on the same
#: discrete output-time grid.
#:
#: Fixing it costs nothing measurable -- 5.8 to 6.4 s per solve across that whole
#: range, which is run-to-run noise -- so there was never anything bought by
#: deriving it.
#:
#: 5.0, and a request above it is **refused** rather than quietly raising the
#: ceiling: extending it on demand is exactly the request-dependence this
#: constant exists to remove.  It covers every projection grid in the package
#: (``limber_grid`` defaults to ``z_max=3.0``).
#:
#: One edge worth knowing: :func:`~ggah_mod.cosmology.growth.growth_rate` asks
#: for a stencil that reaches about 0.2 per cent above the redshift it was
#: given, so its effective ceiling is a hair under this one.  With the headroom
#: above that is a curiosity rather than a limit, and a clear refusal at 4.99 is
#: better than a silent change of sampling at 5.0.
Z_MAX_PK = 5.0



#: The precision the accurate flavour runs CLASS at when no ``precision`` is
#: given: keys of CLASS's ``precisions.h`` on top of its defaults.
#:
#: Chosen by measurement, not by taste.  ``ggah_mod_benchmark``'s precision
#: scan (``scripts/57_precision_scan.py``, rounds 1 and 2, 2026-09-25/28) put
#: every CLASS knob against CLASS's own ``pk_ref.pre`` at eleven cosmologies
#: across the ``emu_pk`` box.  At its defaults CLASS is 2.1e-3 from that
#: reference at the fiducial on 0.01-0.5 h/Mpc and 8.1e-3 at worst over the box
#: (0.6 eV, P_m, z = 0), and the error is almost all in the neutrino sector.
#:
#: **No affordable setting reaches 5e-4 over the whole box**, which was the
#: rule: above 0.3 eV what remains is CLASS's ncdm fluid approximation, and
#: switching it off without raising every truncation with it is worse (1.3e-2),
#: while ``pk_ref`` itself costs hours per massive-neutrino solve.  So the pick
#: is the most accurate rung under five times the default's cost, ties broken
#: by cost: ``tol_ncdm_synchronous = 1e-6``, a thousand times the default and
#: 1.2 times its cost on one Dahu core (37.2 s against 30.5 s).  It takes the
#: fiducial from 2.1e-3 to 6.9e-4 of ``pk_ref`` and the worst cell of the box
#: from 8.1e-3 to 6.9e-3; tightening every ncdm tolerance to 1e-10 buys the
#: same to three digits at 3.2 times.  ``precision={}`` is CLASS's own
#: defaults, which is what ``emu_pk`` was trained on.
CLASS_PRECISION: dict = {"tol_ncdm_synchronous": 1e-6}

#: The same for CAMB: ``CAMBparams.Accuracy`` attributes, plus the four
#: top-level and matter-power switches :func:`camb_input` routes by name.
#:
#: ``lAccuracyBoost = 3``, ``AccuracyBoost = 2`` and no late radiation
#: truncation: the only rung of the scan within 5e-4 of CAMB's own reference
#: (both boosts at 3 with accurate neutrino transfers, a doubled momentum
#: sampling and ``k_per_logint = 30``) at all eleven cosmologies -- 3.9e-4 at
#: worst, 3.2e-5 at the fiducial, against 8.4e-3 and 1.2e-3 at the defaults,
#: for 5.7 times the default's cost (28.0 s against 4.9 s on two Dahu cores).  ``WantCls`` is stated off because that is how
#: the scan solved every rung: with it on CAMB samples its transfer functions
#: differently, and the accuracy measured would not be the accuracy shipped.
#: With both picks, CLASS and CAMB agree to 6.0e-4 at the fiducial, against
#: 2.2e-3 at their defaults; above 0.3 eV the residual is CLASS's.
CAMB_PRECISION: dict = {"lAccuracyBoost": 3.0, "AccuracyBoost": 2.0,
                        "DoLateRadTruncation": False, "WantCls": False}

#: CAMB precision keys that are not ``CAMBparams.Accuracy`` attributes, and
#: where each goes.  The spelling of ``ggah_bench.precision._camb_params``, so
#: a setting the scan measured means the same thing here.
_CAMB_MATTER_KEYS = ("k_per_logint", "accurate_massive_neutrino_transfers")
_CAMB_TOP_KEYS = ("WantCls", "DoLateRadTruncation")


def class_input(cosmo: Cosmology, *, precision=None, output: str = "mPk",
                k_max: float = 300.0) -> dict:
    r"""The CLASS input dict for ``cosmo``: what :class:`ClassPk` sends.

    Public so that nothing else has to restate it.  The technical paper's
    background comparisons call ``classy`` directly with ``output = ""``, and
    until 0.9.8 they spelt the neutrinos their own way; built here they cannot
    drift from the backend.

    **The neutrinos are CLASS's convention and this package's, which are now
    one.**  Three massive states at ``T_ncdm`` =
    :data:`~.constants.T_NCDM_OVER_T_GAMMA`, stated rather than left to CLASS's
    default of the same value, and the massless remainder ``N_ur`` =
    :data:`~.constants.N_UR_REMAINDER`.  CLASS adds back their rest mass,
    :math:`\Sigma m_\nu/(93.143\,{\rm eV}\,h^2)`, which is exactly the
    :attr:`~.parameters.Cosmology.Omega_nu_matter` that
    :attr:`~.parameters.Cosmology.Omega_cdm` subtracted, so CLASS integrates
    :attr:`~.parameters.Cosmology.Omega_m` of total matter.  Until 0.9.8 it
    integrated :math:`\Omega_m - 6.5\times10^{-6}` at the fiducial mass.

    ``precision`` is laid over the physics last, so it can only add solver
    settings; ``None`` is :data:`CLASS_PRECISION`, ``{}`` CLASS's defaults.
    """
    c = cosmo
    params = {"output": output}
    if "mPk" in output:
        params.update({"P_k_max_h/Mpc": k_max * 1.05, "z_max_pk": Z_MAX_PK})
    params.update({
        # Carried, not assumed: a solver told the wrong geometry returns a
        # spectrum for a different universe than the background it is paired
        # with.
        "Omega_k": float(c.Omega_k),
        # See `ClassPk.SBBN_FILE`: a domain limit, not a precision setting.
        "sBBN file": ClassPk._sbbn_path(),
        "h": float(c.h),
        "omega_b": float(c.Omega_b * c.h ** 2),
        "omega_cdm": float(c.Omega_cdm * c.h ** 2),
        "n_s": float(c.n_s),
        "ln10^{10}A_s": float(c.ln10A_s),
        "T_cmb": float(c.T_cmb),
    })
    if float(c.sum_mnu) > 0.0:
        # Three eigenstates carrying the masses `Cosmology.nu_masses` derived,
        # so CLASS and CAMB are handed the same three numbers and a residual
        # between them is the solver rather than the splitting.  Equal masses
        # keep the one-species spelling with a degeneracy of three: the same
        # universe, and the spelling every degenerate measurement was made in.
        t_ncdm = C.T_NCDM_OVER_T_GAMMA
        if c.nu_hierarchy in ("degenerate", "massless"):
            params.update({
                "N_ncdm": 1, "deg_ncdm": float(C.N_NU_MASSIVE),
                "m_ncdm": float(c.sum_mnu) / C.N_NU_MASSIVE,
                "T_ncdm": t_ncdm,
            })
        else:
            masses = np.asarray(c.nu_masses, dtype=float)
            params.update({
                "N_ncdm": int(C.N_NU_MASSIVE),
                "m_ncdm": ", ".join(f"{m:.17g}" for m in masses),
                "T_ncdm": ", ".join([f"{t_ncdm:.17g}"] * int(C.N_NU_MASSIVE)),
            })
        params["N_ur"] = C.N_UR_REMAINDER
    else:
        params["N_ur"] = C.N_EFF
    if (float(c.w0), float(c.wa)) != (-1.0, 0.0):
        params.update({"Omega_Lambda": 0.0, "w0_fld": float(c.w0),
                       "wa_fld": float(c.wa)})
    params.update(CLASS_PRECISION if precision is None else dict(precision))
    return params


def camb_input(cosmo: Cosmology, *, precision=None, redshifts=None,
               k_max: float = 300.0):
    r"""A ``CAMBparams`` for ``cosmo``: what :class:`CambPk` solves.

    Public for the reason :func:`class_input` is.  ``redshifts=None`` builds a
    background-only object; otherwise the matter power is requested at those
    redshifts, in the order given, to ``k_max`` [h/Mpc].

    **The neutrinos are CLASS's, spelt in CAMB's interface**, and until 0.9.8
    they were neither.  CAMB describes each massive eigenstate by a degeneracy
    and a share of ``omnuh2``, the massive states' density *today* --
    rest mass and kinetic energy both -- and works each mass back from its
    share with its own relic energy integral at :math:`(4/11)^{1/3}T_{\rm CMB}`.
    So it is handed

    * a degeneracy :math:`(T_{\rm ncdm}/T_\nu)^4` per state, which puts the
      relativistic density where CLASS's temperature puts it;
    * ``omnuh2`` = the three states' exact density today from
      :func:`~.parameters.nu_energy_factor`, and each state's share of it;
    * the massless remainder as ``num_nu_massless``;
    * ``omch2`` = :attr:`~.parameters.Cosmology.Omega_cdm` :math:`h^2`, the
      same cold density CLASS is given.

    The mass CAMB then infers is :math:`m\,T_\nu/T_{\rm ncdm}`, and the
    dimensionless :math:`m/k_BT` it integrates is exactly CLASS's, so the two
    solve the same neutrinos: :math:`H(z)` agrees to 4e-8 on
    :math:`z \in [0, 1000]`.  Two things were wrong before, and one was a bug.
    ``set_cosmology`` put the heating into :math:`(N_{\rm eff}/3)^{3/4}`,
    :math:`\Sigma m_\nu/93.043` -- a third convention -- so CAMB integrated
    :math:`\Omega_m - 5.0\times10^{-6}`; and the non-degenerate path passed
    *mass* shares as ``nu_mass_fractions``, which CAMB reads as *density*
    shares, so the lightest normal-ordering state was integrated 20 per cent
    light.

    ``precision`` follows ``ggah_bench.precision``: ``k_per_logint`` and
    ``accurate_massive_neutrino_transfers`` go to ``set_matter_power``,
    ``WantCls`` and ``DoLateRadTruncation`` to the parameters, and every other
    key to ``CAMBparams.Accuracy``, where an unknown name raises.  ``None`` is
    :data:`CAMB_PRECISION`, ``{}`` CAMB's defaults.
    """
    import camb

    c = cosmo
    s = dict(CAMB_PRECISION if precision is None else precision)
    pars = camb.CAMBparams()
    massive = float(c.sum_mnu) > 0.0
    pars.set_cosmology(
        H0=100.0 * float(c.h),
        ombh2=float(c.Omega_b * c.h ** 2),
        omch2=float(c.Omega_cdm * c.h ** 2),
        mnu=float(c.sum_mnu),
        num_massive_neutrinos=int(C.N_NU_MASSIVE) if massive else 0,
        nnu=C.N_EFF,
        TCMB=float(c.T_cmb),
        omk=float(c.Omega_k),
    )
    if massive:
        # Each state's density today, from the integral the background uses,
        # at the temperature CLASS uses.  Degenerate masses are three equal
        # states: the same universe as CLASS's one species of degeneracy three.
        per_state = float(c.Omega_nu_massive_rel) / C.N_NU_MASSIVE
        rho = per_state * np.asarray(nu_energy_factor(c.nu_y), dtype=float)
        pars.nu_mass_eigenstates = int(C.N_NU_MASSIVE)
        pars.nu_mass_numbers = [1] * int(C.N_NU_MASSIVE)
        pars.nu_mass_degeneracies = [C.N_MASSIVE_EFF / C.N_NU_MASSIVE] * int(C.N_NU_MASSIVE)
        pars.num_nu_massless = C.N_UR_REMAINDER
        pars.omnuh2 = float(rho.sum()) * float(c.h) ** 2
        pars.nu_mass_fractions = list(rho / rho.sum())
    pars.InitPower.set_params(As=np.exp(float(c.ln10A_s)) * 1e-10,
                              ns=float(c.n_s))
    if (float(c.w0), float(c.wa)) != (-1.0, 0.0):
        pars.set_dark_energy(w=float(c.w0), wa=float(c.wa),
                             dark_energy_model="ppf")
    matter = {k: s.pop(k) for k in _CAMB_MATTER_KEYS if k in s}
    if redshifts is not None:
        pars.set_matter_power(redshifts=list(redshifts), kmax=k_max * 1.05,
                              **matter)
        pars.NonLinear = camb.model.NonLinear_none
    for key in _CAMB_TOP_KEYS:
        if key in s:
            setattr(pars, key, bool(s.pop(key)))
    for key, val in s.items():
        if not hasattr(pars.Accuracy, key):
            raise KeyError(f"{key!r} is not a CAMB accuracy parameter; the "
                           f"keys routed elsewhere are {_CAMB_MATTER_KEYS + _CAMB_TOP_KEYS}")
        setattr(pars.Accuracy, key, val)
    return pars


def _check_curvature(backend, cosmo) -> None:
    """Refuse a curved cosmology on a backend that cannot carry one.

    ``PLAN.md`` item **E2** made :attr:`Cosmology.Omega_k` a parameter and gave
    the background its curvature term.  A spectrum backend that ignored it would
    then pair a *flat* :math:`P(k)` with a *curved* :math:`E(z)` and return a
    constraint that was partly one geometry and partly the other -- which is the
    exact failure the item was written as one change to avoid.

    Declared and not sniffed, like ``has_native_z`` and ``differentiable``:
    CLASS and CAMB both take the parameter and are told it, and so does
    ``emu_pk`` since its 2.0.0 checkpoint was retrained with an ``Omega_k``
    input.  All three shipped backends declare ``supports_curvature = True``;
    the refusal is for a backend that does not.
    """
    ok = getattr(cosmo, "Omega_k", 0.0)
    if isinstance(ok, jax.core.Tracer):
        # A construction-time guard, like `Backend._validate`, and for the same
        # reason: under `jit` the whole Cosmology is traced and a check that
        # branched on a value would have to be a `where` returning a number
        # rather than a refusal.  Every realistic path reaches this concretely
        # first -- `make_pk(...).pk(...)` at setup, and the field build -- so
        # what is skipped here is the second call and not the first.
        return
    ok = float(ok)
    if ok != 0.0 and not getattr(backend, "supports_curvature", False):
        raise ValueError(
            f"{backend.name!r} cannot carry Omega_k = {ok:g}: it declares "
            f"`supports_curvature = False`, and a network trained on a flat box "
            f"has no curvature input to give one. It would return the flat "
            f"spectrum silently, against a background that is not flat. Use a "
            f"Boltzmann backend -- make_pk('class') or make_pk('camb') -- or "
            f"set Omega_k = 0.")



def _nu_ratios(cosmo):
    r"""``(nu_r1, nu_r2)``: how the neutrino sum is divided, lightest first.

    :math:`m_i = r_i \Sigma m_\nu` with :math:`r_3 = 1 - r_1 - r_2`, which is
    the parameterisation ``emu_pk`` 2.0's box adds beside ``sum_mnu`` -- the sum
    keeps its index and its meaning, and what is new is only its division.
    :func:`~ggah_mod.cosmology.parameters.nu_masses` already returns the three
    ascending, which is the order the box wants.

    **The whole difficulty is at zero.**  With :math:`\Sigma m_\nu = 0` the
    ratio is 0/0, and the answer the box wants there is the degenerate
    convention :math:`(1/3, 1/3)` -- a massless cosmology is the degenerate one.
    Written as one ``where`` on a denominator that is *already* safe, rather
    than as a branch: dividing by the raw sum and patching the result afterwards
    still evaluates the division, and a NaN in the untaken branch of a ``where``
    poisons the gradient through both.  The same care ``background._sinhc``
    takes at :math:`x = 0`, for the same reason.
    """
    from .parameters import nu_masses

    third = jnp.asarray(1.0 / 3.0)
    # **The equal-mass case is returned exactly, not divided.**  `nu_masses`
    # gives (s/3, s/3, s/3) here, and `m[0] / s` is not `1/3` in floating point:
    # at 0.06 and 0.12 eV it lands one ulp *above* it, and `1/3` is the box's
    # own upper bound on `nu_r1`, so the most ordinary cosmology in the package
    # sits a hair outside the box by construction.  `emu_pk`'s bounds carry
    # float32 slack and absorb it, which makes this luck rather than
    # correctness -- and any stricter check downstream would refuse the
    # fiducial.  `nu_hierarchy` is a static field, so this is a concrete Python
    # branch even under `jit`, the same standing `_check_nu_split` relies on.
    if cosmo.nu_hierarchy in ("degenerate", "massless"):
        return third, third

    total = jnp.asarray(cosmo.sum_mnu, dtype=float)
    massless = total <= 0.0
    safe = jnp.where(massless, 1.0, total)
    m = nu_masses(safe, cosmo.nu_hierarchy)
    return (jnp.where(massless, third, m[0] / safe),
            jnp.where(massless, third, m[1] / safe))

def _check_nu_split(backend, cosmo) -> None:
    """Refuse a split neutrino mass on a backend trained on degenerate ones.

    The same shape as :func:`_check_curvature`, and for the same reason.
    ``emu_pk`` 1.x carries one ``sum_mnu`` input and its weights were trained
    with ``deg_ncdm = 3``, so a cosmology whose three masses are unequal is out
    of the network's distribution however the sum is passed -- and the network
    would answer anyway, with a spectrum for a universe that is not the one the
    background is being integrated for.

    It keys on the **hierarchy**, not on the masses.  ``nu_hierarchy`` is a
    static field, so it is a concrete string even under ``jit``: this check
    cannot be defeated by tracing, unlike the curvature one, and needs no
    escape.  ``degenerate`` and ``massless`` pass -- they are exactly the
    parameterisation the weights were trained on.
    """
    h = getattr(cosmo, "nu_hierarchy", "degenerate")
    if h in ("degenerate", "massless"):
        return
    # With no mass there is nothing to order: `nu_masses` returns exactly
    # (0, 0, 0) in every hierarchy, so a massless cosmology *is* the degenerate
    # one and sits inside the emulator's training distribution.  Refusing it
    # would refuse the massless control -- which is the one cosmology every
    # backend here agrees on, and the comparison point the whole neutrino
    # sector is measured against.
    #
    # This reads the sum rather than the masses, and only when it is concrete:
    # it is a shortcut past the refusal, so under tracing it declines to take
    # the shortcut rather than guessing.
    s = _concrete(getattr(cosmo, "sum_mnu", 0.0))
    if s == 0.0:
        return
    if not getattr(backend, "supports_nondegenerate_nu", False):
        raise ValueError(
            f"{backend.name!r} cannot carry nu_hierarchy={h!r}: it declares "
            f"`supports_nondegenerate_nu = False`.  Its network takes one "
            f"`sum_mnu` input and was trained with three degenerate species, "
            f"so three unequal masses are outside the box it was shown and it "
            f"would return a spectrum for a different universe than the "
            f"background is integrating.  Use a Boltzmann backend -- "
            f"make_pk('class') or make_pk('camb'), which take the three masses "
            f"-- or declare nu_hierarchy='degenerate' to take the equal-mass "
            f"approximation knowingly.  A retrained emulator closes this.")


@runtime_checkable
class LinearPowerSpectrum(Protocol):
    """The contract every linear-P(k) backend satisfies."""

    name: str
    has_native_z: bool
    differentiable: bool
    #: Whether the backend can carry :attr:`Cosmology.Omega_k`.
    #:
    #: Declared, never sniffed -- the same rule as the other two, and for the
    #: sharper reason: a backend that silently ignored curvature would return a
    #: *flat* spectrum for a curved cosmology whose background was curved, and
    #: the resulting constraint would be partly one geometry and partly the
    #: other.  ``PLAN.md`` item **E2**.
    supports_curvature: bool
    #: Whether the backend can carry three *unequal* neutrino masses.
    #:
    #: Declared for the same reason as the two above: an emulator handed a
    #: split it was never trained on answers rather than refusing, and the
    #: answer is a spectrum for a different universe than the background.
    supports_nondegenerate_nu: bool

    def pk(self, k, z, cosmo: Cosmology): ...
    def pk_cb(self, k, z, cosmo: Cosmology): ...


#: ``float(x)`` if ``x`` is a concrete value, else ``None``.
#:
#: Validation that calls ``float()`` on a traced value raises
#: ``ConcretizationTypeError`` and kills the gradient -- which is the whole
#: point of the differentiable flavour.  Checks that cannot run under tracing
#: must be *skipped* there, not attempted: the check is a courtesy to an
#: interactive user, and a jitted forward model has already been checked once
#: outside the trace.
#:
#: **One definition.**  It lives in :mod:`~ggah_mod.cosmology.parameters`,
#: which needs it for the neutrino refusals and cannot import this module
#: without a cycle, and is re-exported here under the name callers already use.
#: There were two copies of this idiom before ``nu_hierarchy`` needed a third.
concrete = _concrete


def _as_key(cosmo: Cosmology) -> tuple:
    """Cache key covering **every** parameter.

    Not a hand-picked subset.  A cache keyed on part of the cosmology serves a
    spectrum computed at a different one, and the symptom -- a chain that
    explores a direction and sees no response -- reads as a physics result.

    Not every parameter is a float any more: ``nu_hierarchy`` is a string, and
    a key built by ``float()`` over the whole field list raises on it.  The
    split used here is **the pytree's own** -- ``LEAF_FIELDS`` is everything
    not declared static, and ``LEAF_FIELDS + STATIC_FIELDS`` is asserted to be
    the whole dataclass at import -- so this is still not a subset anyone
    chose.  The order is the dataclass order, which is what lets
    ``Cosmology(*key)`` below reconstruct positionally.  A ``str`` is hashable,
    so the ``lru_cache`` is unaffected.
    """
    return (tuple(float(getattr(cosmo, f)) for f in LEAF_FIELDS)
            + tuple(getattr(cosmo, f) for f in STATIC_FIELDS))


class _BoltzmannBase:
    """Shared plumbing: solve once per (cosmology, z-grid), then interpolate.

    The solve costs seconds, so it is memoised on the full cosmology and the
    redshift tuple.  Interpolation is log-log linear in ``k``, which is exact
    for a power law and is what the solver's own sampling assumes.

    **The redshift tuple is part of the key, and that has a consequence worth
    stating rather than leaving to be discovered.**  It is a *cost* consequence
    only: since :data:`Z_MAX_PK` became a constant the spectrum at a given
    redshift no longer depends on which others were asked for beside it, so two
    entries under different tuples hold the same rows where they overlap.  The
    tuple is in the key because the rows returned are the rows requested, not
    because the values would otherwise differ -- which is exactly what it used
    to be for.

    A solve is per *cosmology*:
    the solver integrates the perturbations once and writes out every redshift
    it was asked for, so asking for twelve at once costs one solve and asking
    for them one at a time costs twelve.  Measured with CLASS on the shipped
    grids: **6.0 s against 93.7 s**.  This is why the redshift axis exists in
    layer 1 at all, and why :func:`~ggah_mod.halos.field.make_fields` exists in
    layer 2, where the objects are one epoch each and the loop would otherwise
    be a loop of solves.  Note the key does *not* include ``k`` -- the solve is
    always on ``self._k`` and ``k`` only selects the interpolation -- so two
    callers wanting different wavenumbers at the same redshifts share one solve.

    What batching does not remove: :meth:`ClassPk._solve_uncached` walks
    ``n_k * Nz`` Python-level calls into CLASS to fill its table, and that part
    still grows with the number of redshifts.  It is a small fraction of a solve,
    which is what makes the trade above so one-sided, but it is not nothing.

    Why the key is not the cosmology alone.  It could be, now that the values do
    not depend on the grouping -- a cache that reused any entry whose tuple
    *contains* the redshifts asked for would be exact rather than approximate.
    It is not worth the second cache implementation: the paths that matter
    already hit.  A likelihood evaluating on one grid repeatedly hits on the
    identical tuple, and :func:`~ggah_mod.halos.field.make_fields` has already
    collapsed the per-redshift requests into one.  What would remain is the
    ``(0.0,)`` normalisation :func:`~ggah_mod.cosmology.growth.growth_factor`
    needs, and a projection grid starts at ``z_min = 1e-3`` rather than zero, so
    a superset lookup would not catch that either.
    """

    has_native_z = True
    differentiable = False

    #: Highest redshift this backend will answer at, or ``None`` for no ceiling.
    #: Declared, like every other capability here, rather than discovered by
    #: calling and seeing what comes back.
    z_max: float | None = None

    #: The module-level default ``precision`` falls back to; set by subclasses.
    _DEFAULT_PRECISION: dict = {}

    def __init__(self, k_min: float = 1e-5, k_max: float = 300.0, n_k: int = 1000,
                 cache_size: int = 32, precision=None):
        self.k_min, self.k_max, self.n_k = float(k_min), float(k_max), int(n_k)
        self._k = np.logspace(np.log10(self.k_min), np.log10(self.k_max), self.n_k)
        # Fixed at construction, and a copy: the memoisation below is per
        # instance and keyed on the cosmology alone, so a precision that could
        # change after the first solve would serve spectra of one setting
        # under the name of another.  ``None`` is the package default.
        self.precision = dict(type(self)._default_precision()
                              if precision is None else precision)
        self._solve_cached = functools.lru_cache(maxsize=cache_size)(self._solve_uncached)

    @classmethod
    def _default_precision(cls) -> dict:
        return {}

    # -- subclass hook ------------------------------------------------------
    def _solve_uncached(self, key: tuple, z_key: tuple):
        raise NotImplementedError

    # -- public -------------------------------------------------------------
    def _tables(self, z, cosmo: Cosmology):
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        if self.z_max is not None and z_arr.max() > self.z_max:
            raise ValueError(
                f"{type(self).__name__} answers up to z = {self.z_max:g} and "
                f"was asked for z = {z_arr.max():g}.  The ceiling is a fixed "
                f"solver setting, not a budget to be raised per request: "
                f"raising it is what used to make the spectrum depend on which "
                f"redshifts were asked for together.  Change "
                f"`power.Z_MAX_PK` if the whole package needs a higher one, "
                f"and re-run the validation -- the numbers move at the 1e-7 "
                f"level when it changes.")
        _check_curvature(self, cosmo)
        _check_nu_split(self, cosmo)
        pm, pcb = self._solve_cached(_as_key(cosmo), tuple(np.round(z_arr, 8)))
        return z_arr, pm, pcb

    def _interp(self, k, table_row):
        k = np.atleast_1d(np.asarray(k, dtype=float))
        return np.exp(np.interp(np.log(k), np.log(self._k), np.log(table_row)))

    def pk(self, k, z, cosmo: Cosmology):
        r""":math:`P_m(k,z)` [(Mpc/h)^3]."""
        z_arr, pm, _ = self._tables(z, cosmo)
        out = np.stack([self._interp(k, pm[i]) for i in range(len(z_arr))])
        return out[0] if np.ndim(z) == 0 else out

    def pk_cb(self, k, z, cosmo: Cosmology):
        r""":math:`P_{cb}(k,z)` [(Mpc/h)^3] -- the cold field halos form from."""
        z_arr, _, pcb = self._tables(z, cosmo)
        out = np.stack([self._interp(k, pcb[i]) for i in range(len(z_arr))])
        return out[0] if np.ndim(z) == 0 else out


class ClassPk(_BoltzmannBase):
    """CLASS -- the reference backend.

    ``z_max_pk`` is :data:`Z_MAX_PK`, a constant, so a spectrum here is a
    function of the cosmology and the redshift and of nothing else.

    The only one that treats massive neutrinos exactly rather than through a
    fitting function or a distilled table, and an implementation independent of
    CAMB, which is what makes it able to arbitrate.
    """

    name = "class"
    supports_curvature = True
    supports_nondegenerate_nu = True
    z_max = Z_MAX_PK

    @classmethod
    def _default_precision(cls) -> dict:
        return CLASS_PRECISION

    #: Which of CLASS's tabulated BBN predictions for the primordial helium
    #: fraction to interpolate, as a path relative to the CLASS root (CLASS
    #: prepends its own directory, so this string must start with a slash and
    #: cannot be absolute).
    #:
    #: This is not a taste question, it is a domain question.  CLASS 3.3
    #: defaults to ``sBBN_2025.dat``, which is tabulated over
    #: :math:`0.0073 < \omega_b < 0.03289`; above that ceiling
    #: ``thermodynamics_helium_from_bbn`` refuses the cosmology outright with
    #: "you have asked for an unrealistic high value omega_b".  The refusal is
    #: the right behaviour -- it is loud, and extrapolating a BBN yield would
    #: not be -- but it is a ceiling nothing in this package documented, and it
    #: is low enough to matter: it cuts 15 per cent out of the CSST emulator's
    #: training box, the whole high-:math:`\Omega_b`, high-:math:`H_0` corner.
    #: :class:`CambPk` has no such ceiling (its own BBN interpolator runs past
    #: :math:`\omega_b = 0.045`), so the two backends that share the
    #: ``ACCURATE`` flavour disagreed about which cosmologies exist.
    #:
    #: ``sBBN_2017.dat`` is tabulated to :math:`\omega_b = 0.03993`, covers
    #: every cosmology CAMB accepts within reach of this package, and agrees
    #: with the 2025 default to :math:`1.2\times10^{-4}` in :math:`Y_{He}` where
    #: both are defined (through CLASS, :math:`\omega_b = 0.0075`--0.0328 at
    #: this package's :math:`\Delta N`; it read :math:`4\times10^{-4}` until
    #: 1.1.0, which nothing had measured) -- :math:`2.5\times10^{-5}` in :math:`P(k)`
    #: and :math:`4\times10^{-7}` in :math:`\sigma_8`, four orders of magnitude
    #: below anything this package claims.  Naming it costs that and buys the
    #: corner.
    SBBN_FILE = "/external/bbn/sBBN_2017.dat"

    @staticmethod
    def _sbbn_path() -> str:
        """The BBN table, checked for existence rather than assumed.

        CLASS resolves :attr:`SBBN_FILE` against its own root and reports a
        missing table as a thermodynamics failure a hundred lines into the
        solve, which reads like a physics problem and is not one.  Checking
        here turns it into one sentence naming the file.
        """
        import os
        import classy

        root = os.path.dirname(classy.__file__)
        full = root + ClassPk.SBBN_FILE
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"CLASS's BBN table {full!r} is missing, so the primordial "
                "helium fraction cannot be interpolated in the convention "
                "this package fixes.  Falling back to CLASS's default table "
                "would silently change Y_He by 4e-4 and cap omega_b at "
                "0.03289, so it is refused instead.  Set "
                "ggah_mod.cosmology.power.ClassPk.SBBN_FILE to a table that "
                "exists under {root!r} if your CLASS ships a different set.")
        return ClassPk.SBBN_FILE

    def _solve_uncached(self, key: tuple, z_key: tuple):
        from classy import Class

        c = Cosmology(*key)
        z_arr = np.asarray(z_key, dtype=float)
        # One builder for this and every other CLASS call made on this
        # package's behalf: see `class_input` for the neutrino convention.
        params = class_input(c, precision=self.precision, k_max=self.k_max)

        cl = Class()
        cl.set(params)
        cl.compute()
        try:
            k_phys = self._k * c.h                      # h/Mpc -> 1/Mpc
            h3 = c.h ** 3
            pm = np.array([[cl.pk_lin(kk, zz) for kk in k_phys] for zz in z_arr]) * h3
            if c.sum_mnu > 0.0:
                pcb = np.array([[cl.pk_cb_lin(kk, zz) for kk in k_phys]
                                for zz in z_arr]) * h3
            else:
                pcb = pm.copy()
        finally:
            cl.struct_cleanup()
            cl.empty()
        return pm, pcb


class CambPk(_BoltzmannBase):
    r"""CAMB -- the accurate production backend.

    **Its output redshift list is anchored, for the same reason CLASS's
    ``z_max_pk`` became a constant.**  CAMB integrates the perturbations from
    early times and dumps the transfer function at the redshifts it was asked
    for, so the requested set decides where the integrator places its output
    steps -- and therefore, at round-off, what it reports at any one of them.
    Measured at PLANCK18, ``z = 0.5``, 60 log-spaced k from 1e-3 to 10:

    ==========================  ==================
    requested with 0.5          max rel. deviation
    ==========================  ==================
    ``(0.0, 0.5)``              0
    ``(0.4, 0.5)``              0
    ``(0.5, 0.6)``              4.6e-10
    ``(0.5, 1.0)``              2.8e-7
    ``(0.5, 3.0)``              2.8e-7
    ``(0.0, 0.5, 1.0, 3.0)``    2.8e-7
    ==========================  ==================

    Only a *higher* companion moves it, which is the shape of the cause: the
    highest requested redshift is where the recorded integration begins.  So
    :data:`Z_MAX_PK` is always appended to the request and dropped from the
    answer, the top of the range never moves, and every set above returns the
    ``z = 0.5`` row **bit for bit**.  It costs one extra output redshift.

    This is the same defect CLASS had through ``z_max_pk``, found the same way
    -- by asking for the same redshift in two groupings and comparing -- and
    fixed by the same rule: **a solver setting is not the caller's to set.**
    ``tests/test_power.py`` asserts the property over both backends.
    """

    name = "camb"
    supports_curvature = True
    supports_nondegenerate_nu = True

    @classmethod
    def _default_precision(cls) -> dict:
        return CAMB_PRECISION

    def _solve_uncached(self, key: tuple, z_key: tuple):
        import camb

        c = Cosmology(*key)
        z_arr = np.asarray(z_key, dtype=float)
        # The anchor: solve to a fixed top of range whatever was asked for, so
        # the integrator's output steps do not depend on the request.  See the
        # class docstring for the measurement.
        z_solve = np.unique(np.append(z_arr, Z_MAX_PK))
        # CAMB wants redshifts descending.  One builder for this and every
        # other CAMB call made on this package's behalf: see `camb_input` for
        # the neutrino convention, and for the mass-fraction bug it fixed.
        order = np.argsort(-z_solve)
        pars = camb_input(c, precision=self.precision,
                          redshifts=list(z_solve[order]), k_max=self.k_max)
        results = camb.get_results(pars)

        def _grid(var):
            kh, zs, p = results.get_matter_power_spectrum(
                minkh=self.k_min, maxkh=self.k_max, npoints=self.n_k,
                var1=var, var2=var)
            # get_matter_power_spectrum returns z ascending; map back to our order.
            idx = np.argsort(np.asarray(zs))
            p = np.asarray(p)[idx]
            back = np.searchsorted(np.asarray(zs)[idx], z_arr)
            return np.asarray(kh), p[np.clip(back, 0, len(zs) - 1)]

        k_camb, pm = _grid("delta_tot")
        _, pcb = _grid("delta_nonu")
        # CAMB's own k grid is what it sampled on; resample onto ours.
        pm = np.stack([np.exp(np.interp(np.log(self._k), np.log(k_camb), np.log(row)))
                       for row in pm])
        pcb = np.stack([np.exp(np.interp(np.log(self._k), np.log(k_camb), np.log(row)))
                        for row in pcb])
        return pm, pcb


# =========================================================================
# The differentiable backend
# =========================================================================
#
# One network, and it is `emu_pk`.  Two others stood here and left, for
# opposite reasons.
#
# The analytic Eisenstein & Hu (1998) backend went first: its no-wiggle
# transfer function outlived it for one consumer, the diemer15 concentration
# relation, whose effective slope is a slope of P(k) and so needed a spectrum
# without acoustic wiggles in it.  Diemer & Joyce (2019) defines that slope on
# sigma(M) instead, which is smooth whatever the spectrum does -- so the
# analytic fit lost its last consumer and the package no longer contains one.
#
# CosmoPower-JAX went second, and for the opposite reason: not that nothing
# needed it, but that everything it did `emu_pk` does over a wider box.  It was
# trained on massless LambdaCDM, so it needed the CLASS-distilled correction of
# `emu_pk.ratio` bolted on top to carry neutrinos and dark energy, a
# massless-equivalent cold density substituted underneath, and a power-law
# continuation past its top mode at 14.6 h/Mpc to keep the sigma(M) quadrature
# from integrating a plateau out to k = 200.  All three were scaffolding around
# a training set, and `emu_pk`'s training set does not need them.
#
# Nothing in this package now reads `emu_pk.ratio`.  The correction is still
# there and still correct; it has no work to do once the spectrum it corrected
# is gone.


class GgahEmuPk:
    r"""The differentiable spectrum: eight parameters and the redshift, natively.

    The network was trained on CLASS spectra that already carry massive
    neutrinos and :math:`(w_0, w_a)`, so nothing is corrected on top of it and
    nothing is substituted underneath it.  Two consequences are worth naming,
    because a spectrum assembled the other way pays for both:

    * :math:`P_{cb}` is a **trained output**, not :math:`P_m` times a ratio.
      Two heads of one network, so the cold and total spectra cannot drift
      apart the way two separately corrected quantities can.  At the massless
      corner they agree to 6e-4 -- learned rather than exact, which is the
      honest form of that guarantee and what ``tests/test_backends.py`` pins.
    * the dark-energy response is a network input rather than a table read
      through four Hermite axes, so it costs what any other parameter costs.

    The box runs down to :math:`h = 0.55`, and the training set reaches
    :math:`k = 200\,h\,\mathrm{Mpc}^{-1}` -- which is exactly the
    differentiable flavour's ``k_max``, so the :math:`\sigma(M)` quadrature
    never asks for a mode the network was not trained on and no extrapolation
    is needed at either end.

    ``emu_pk`` is a separate package.  It has to be: generating its training set
    needs ``classy`` and training it needs ``optax``, and neither belongs in a
    dependency chain that exists to be imported by a forecast.
    """

    name = "emu_pk"
    #: **Both true since emu_pk 2.0.0, 2026-09-10.**  Its box gained `Omega_k`,
    #: `nu_r1` and `nu_r2`, and the shipped checkpoint is trained on them --
    #: which is the single event ``CROSS_REPO.md`` X9 and X10 name as reopening
    #: layers 1 and 2.  These are claims about the *installed* box, not about
    #: this class: ``_params`` builds its vector from ``emu_pk.box.PARAMS``, so
    #: against a 1.x install they would be false and
    #: ``tests/test_curvature.py::TestTheMappingFollowsTheEmulatorsOwnBox``
    #: fails rather than letting eight parameters reach a curved cosmology.
    supports_curvature = True
    supports_nondegenerate_nu = True
    has_native_z = True
    differentiable = True

    def __init__(self, weights=None, check_box: bool = True):
        from emu_pk.model import PkEmulator
        self._emu = PkEmulator(weights, check_box=check_box)

    def _params(self, cosmo):
        r"""``Cosmology`` -> the network's inputs, in ``emu_pk.box.PARAMS`` order.

        Physical densities, and :math:`\omega_{cdm}` is the *actual* cold
        density: this network was trained with the neutrino mass as an input,
        so substituting a massless-equivalent density here -- as a network
        trained on massless LambdaCDM would require -- would remove the very
        mass the network is expecting to be told about.

        **The order is read from the box rather than written out here.**
        ``emu_pk`` 2.0 appends ``Omega_k``, ``nu_r1`` and ``nu_r2`` to a tuple
        whose first eight entries and their meanings are unchanged, precisely so
        that a caller's existing mapping keeps working; building the vector from
        that tuple is how this package collects on that promise instead of
        needing an edit to notice.  It also removes the failure the emulator's
        own ``_forward`` guards against -- a *short* vector is not an error to
        JAX, which clamps the gather and returns a finite, smooth, wrong
        spectrum, measured there at 10.3 per cent in :math:`P(0.05)`.

        A name in the box this mapping has no value for is a hard error rather
        than a zero, for the same reason.
        """
        from emu_pk import box

        h2 = cosmo.h ** 2
        known = {
            "omega_b": cosmo.Omega_b * h2,
            "omega_cdm": cosmo.Omega_cdm * h2,
            "h": cosmo.h,
            "n_s": cosmo.n_s,
            "ln10A_s": cosmo.ln10A_s,
            "sum_mnu": cosmo.sum_mnu,
            "w0": cosmo.w0,
            "wa": cosmo.wa,
            "Omega_k": cosmo.Omega_k,
        }
        if "nu_r1" in box.PARAMS or "nu_r2" in box.PARAMS:
            r1, r2 = _nu_ratios(cosmo)
            known["nu_r1"], known["nu_r2"] = r1, r2
        missing = [q for q in box.PARAMS if q not in known]
        if missing:
            raise ValueError(
                f"emu_pk's box names inputs this package has no value for: "
                f"{missing}.  Handing the network a shorter vector than its box "
                f"declares does not raise there -- JAX clamps the gather and "
                f"returns a spectrum -- so the mapping is completed here or "
                f"nowhere.")
        return jnp.stack([jnp.asarray(known[q], dtype=float)
                          for q in box.PARAMS])

    def pk(self, k, z, cosmo):
        _check_curvature(self, cosmo)
        _check_nu_split(self, cosmo)
        return self._emu.predict(k, z, self._params(cosmo), "m")

    def pk_cb(self, k, z, cosmo):
        _check_curvature(self, cosmo)
        _check_nu_split(self, cosmo)
        return self._emu.predict(k, z, self._params(cosmo), "cb")


#: Registry.  ``make_pk`` is how a backend is selected; nothing constructs one
#: of these classes by name elsewhere in the package.
PK_BACKENDS = {"class": ClassPk, "camb": CambPk, "emu_pk": GgahEmuPk}


def make_pk(name: str, **kw) -> LinearPowerSpectrum:
    """Construct a linear-P(k) backend by name.

    ``precision=`` reaches the two Boltzmann backends only.  ``emu_pk`` has
    no solver settings -- its precision is whatever its training set was
    solved at, CLASS's defaults -- and asking it for one is refused rather
    than ignored.
    """
    key = str(name).lower()
    if key not in PK_BACKENDS:
        raise ValueError(f"unknown P(k) backend {name!r}; "
                         f"expected one of {sorted(PK_BACKENDS)}")
    if "precision" in kw and not issubclass(PK_BACKENDS[key], _BoltzmannBase):
        raise ValueError(
            f"{name!r} takes no `precision`: an emulator has no solver "
            f"settings, and emu_pk's are those of its training set (CLASS at "
            f"its defaults).  Use make_pk('class', precision=...) or "
            f"make_pk('camb', precision=...).")
    return PK_BACKENDS[key](**kw)
