r"""The halo field: one object that owns the mass and wavenumber grids.

:class:`HaloField` holds the halo tables of layer 2 on one mass grid, at one
redshift, and :func:`make_field` builds it from a cosmology and a linear power
spectrum.  A layer-3 sector takes a field and owns no grid, so the sectors are
peers: a pure hot-gas or AGN spectrum needs no galaxy parameters, and every
sector of an analysis uses the same mass range.

Scope of the field
------------------

It is not a cache.  Everything on it is computed once by :func:`make_field` and
then read; there is no invalidation, no key and no partial rebuild, so a
forecast varying any parameter (``sum_mnu``, ``w0`` and ``wa`` included) cannot
be served a stale spectrum.

It is not baryonic.  The tables are dark-matter-only: :math:`c(M)`,
:math:`r_\Delta` and :math:`dn/dM` never see the gas fraction.  Layer 3's
:class:`MatterField` reads from this object and never writes back; if it did,
the sector graph would acquire a cycle (``matter -> field -> gas``) and the
evaluation order would stop being well defined.

One redshift per field
----------------------

The redshift axis belongs to layers 1 and 5.  Layer 1 is batched in redshift
because its solver is: a Boltzmann solve is per cosmology, and asking it for
twelve redshifts costs one solve where asking twelve times costs twelve (6.0 s
against 93.7 s, measured with CLASS).  Layer 5 integrates over redshift.
Layers 2, 3 and 4 are one epoch per object.

A redshift axis on :class:`HaloField` would buy no accuracy, since
:math:`\sigma(M,z)` is built from :math:`P_{cb}(k,z)` at the redshift wanted
either way, and it would break layer 4: its contractions are ``(Nk, NM)``, and
a ``(NM,)`` weight is told from a ``(Nk, NM)`` one by the leading axis, a test
that a third axis would turn from raising into silently returning the wrong
branch.  Layer 4 takes no ``z`` argument; layer 3 reads ``field.z`` and carries
none.

So :func:`make_field` refuses a vector ``z``, and there are two ways to cover a
set of redshifts:

``make_fields(cosmo, backend, pk, z_array, ...)``
    one linear-P(k) call for the whole set, then one ordinary field per epoch.
    The only cheap route for CLASS or CAMB, and the one
    :func:`~ggah_mod.observables.spec.make_model` asks its builder for.

``jax.vmap`` over the whole ``z -> spectrum`` closure
    for a differentiable backend, when the redshift belongs inside a trace.

``jax.vmap(make_field)`` on its own is legal and returns a real pytree, with
``Nz`` on the leading axis of every leaf (:attr:`HaloField.n_m` reads
``shape[-1]`` so that it survives).  That object may be read and integrated,
but not handed to a sector or to a layer-4 spectrum, which refuse a rank-3
weight by name.

The pytree and its validation
-----------------------------

Every table is a leaf, so ``jax.grad`` of anything downstream flows through the
field without a dict in the way, and ``jax.jit`` may close over one.  The model
names travel in the treedef as static aux data, so differentiating with respect
to the string ``"tinker08"`` is a structural impossibility rather than a silent
zero.

``__init__`` is branch-free because ``tree_unflatten`` runs inside a trace and
must not branch on values.  Validation lives in :meth:`HaloField.create`, the
same split :class:`~ggah_mod.cosmology.parameters.Cosmology` uses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..backend import resolve_backend
from ..cosmology import growth
from ..cosmology import constants as C
from ..numerics import log_grid
from .calibration import check_calibration, check_cosmology_support
from .concentration import (
    CONCENTRATION, PEAK_HEIGHT_MODELS, make_concentration, n_eff_from_sigma,
)
from .linear_bias import BIAS, make_bias
from .mass_definitions import MassDef, parse_mass_def
from .mass_function import COSMOLOGY_DEPENDENT_MULTIPLICITY, MULTIPLICITY
from .mass_function import _DELTA_ARG as _MF_DELTA_ARG
from .mass_function import dndm as _dndm
from .mass_function import multiplicity_kwargs
from .profiles import nfw_uk
from .variance import (DELTA_C, check_k_support, dln_sigma_dln_mass,
                       sigma_of_mass)

__all__ = ["HaloField", "make_field", "make_fields", "DEFAULT_CM_MODEL"]

#: Concentration relation used when neither the caller nor the backend names one.
#:
#: ``make_field`` prefers :attr:`ggah_mod.backend.Backend.cm_model`, which is
#: what that field is for -- layer 3 is its first consumer, and until now
#: nothing read it.  That is why it was wrong: ``ACCURATE`` asked for
#: ``"diemer19"`` and ``TRACEABLE_CM`` offered ``"diemer19_jax"``, and the
#: registry provides neither.  Both were corrected in layer 2 once this module
#: gave them a reader.
#:
#: The default is a *peak-height* relation rather than an empirical power law
#: because the power laws take no cosmology and cannot respond to one: a
#: free-cosmology chain built on one gets a frozen c(M), and
#: :math:`\partial c/\partial\theta` is identically zero for a reason that has
#: nothing to do with physics.
DEFAULT_CM_MODEL = "bhattacharya13"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HaloField:
    r"""The halo tables on one mass grid, at one redshift.

    Attributes
    ----------
    m : array, shape (NM,) [Msun/h]
        The mass grid, owned here; no sector builds one.
    k : array, shape (Nk,) [h/Mpc]
        The wavenumber grid the spectrum and the profiles are evaluated on.
    z : float
        Redshift.  May be traced.
    sigma, dlns, nu : array, shape (NM,)
        :math:`\sigma(M,z)`, :math:`d\ln\sigma/d\ln M` (negative), and the peak
        height :math:`\nu = \delta_c/\sigma`.
    dndm : array, shape (NM,) [(Mpc/h)^-3 (Msun/h)^-1]
    bias : array, shape (NM,)
        Linear halo bias :math:`b(M)`.
    conc : array, shape (NM,)
        Concentration, in the mass definition :attr:`mdef`.
    r_delta, r_s : array, shape (NM,) [Mpc/h], comoving
        Halo radius and NFW scale radius :math:`r_\Delta/c`.
    pk_cb, pk_lin : array, shape (Nk,)
        The *cold* and *total* linear spectra at ``z`` [(Mpc/h)^3].  Named
        separately because they answer different questions: haloes form from
        the cold field, lensing sees the total.
    cosmo : Cosmology
    mdef, hmf_model, bias_model, cm_model : str
        Static: they live in the treedef, not among the leaves.
    """

    # -- leaves --------------------------------------------------------------
    m: jnp.ndarray
    k: jnp.ndarray
    z: float
    sigma: jnp.ndarray
    dlns: jnp.ndarray
    nu: jnp.ndarray
    dndm: jnp.ndarray
    bias: jnp.ndarray
    conc: jnp.ndarray
    r_delta: jnp.ndarray
    r_s: jnp.ndarray
    pk_cb: jnp.ndarray
    pk_lin: jnp.ndarray
    cosmo: object

    # -- static --------------------------------------------------------------
    mdef: str = "200m"
    hmf_model: str = "tinker08"
    bias_model: str = "tinker10"
    cm_model: str = DEFAULT_CM_MODEL

    # ------------------------------------------------------------- geometry
    @property
    def n_m(self) -> int:
        """Mass-grid size.  Static: it is an array shape.

        Read from ``shape[-1]``, because ``jax.vmap(make_field)`` over the
        redshift puts ``Nz`` on the leading axis of every leaf; ``shape[0]``
        would then return the number of redshifts, a plausible integer.  Such a
        field may be read but not handed downstream (see "One redshift per
        field" in the module docstring).
        """
        return int(self.m.shape[-1])

    @property
    def n_k(self) -> int:
        """Wavenumber-grid size.  Static: it is an array shape.  See :attr:`n_m`."""
        return int(self.k.shape[-1])

    @property
    def ln_m(self):
        r""":math:`\ln M`, the variable every mass integral is taken in."""
        return jnp.log(self.m)

    @property
    def rho_matter(self):
        r"""Comoving :math:`\bar\rho_m` from *total* matter [(Msun/h)/(Mpc/h)^3].

        The density that lenses, and the one
        :class:`~ggah_mod.sectors.matter.MatterField` normalises against, rather
        than :attr:`rho_cold`.
        """
        return self.cosmo.rho_matter

    @property
    def rho_cold(self):
        r"""Comoving :math:`\bar\rho_{cb}`, the density haloes form from."""
        return self.cosmo.rho_cold


    @property
    def r_delta_physical(self):
        r""":math:`r_\Delta` in *physical* Mpc/:math:`h`.

        :attr:`r_delta` is comoving, which is what the profiles want.  A
        binding energy needs the physical radius: :math:`GM/r` with a comoving
        radius is too small by :math:`(1+z)`, no error at :math:`z = 0` and
        100 per cent at :math:`z = 1`.
        """
        return self.r_delta / (1.0 + self.z)

    @property
    def v_delta_squared(self):
        r""":math:`v_\Delta^2 = GM/r_\Delta^{\rm phys}` in :math:`({\rm km/s})^2`.

        The halo's own circular velocity at its boundary, squared, on the mass
        grid.  :math:`h` cancels: a mass in :math:`M_\odot/h` over a radius in
        :math:`{\rm Mpc}/h`.

        A property rather than a leaf, since three leaves already imply it; a
        leaf would touch ``_LEAVES``, ``tree_flatten``, ``tree_unflatten``,
        ``create``, ``_validate``, ``make_field`` and every test that builds a
        field positionally.  It reads :attr:`r_delta` rather than recomputing
        the radius from the mass definition, as
        :mod:`~ggah_mod.sectors.energetics` does with
        ``parse_mass_def(mdef).radius(...)``, so it cannot drift from the field.

        The field stays dark-matter only: ``tests/test_sector_coherence.py``
        asserts that a gas fraction never reaches :math:`c(M)`, ``r_delta`` or
        :math:`dn/dM`.  Reading a velocity out of the field does not change
        that; making ``r_delta`` depend on a baryon fraction would.
        """
        return C.G_MPC_KMS2_MSUN * self.m / self.r_delta_physical

    # -------------------------------------------------------------- profiles
    def u_nfw(self, k=None):
        r"""NFW :math:`u(k|M)`, shape ``(Nk, NM)``, with :math:`u(k\to0) = 1`.

        Uses the field's own ``r_s`` and ``conc``, so every sector that wants
        the dark-matter profile of these haloes gets the same one.  ``k``
        [h/Mpc] defaults to the field's grid.
        """
        k = self.k if k is None else jnp.asarray(k)
        return nfw_uk(k, self.r_s, self.conc)

    # ---------------------------------------------------------- mass integrals
    def integrate(self, integrand, axis: int = -1):
        r""":math:`\int f(M)\,dM`, evaluated as :math:`\int f(M)\,M\,d\ln M`.

        ``integrand`` is a ``dn/dM``-like density on :attr:`m`.  Every mass
        integral goes through this one method, so the ``dM`` versus
        ``d\ln M`` choice is made once; ``tests/test_sector_coherence.py``
        looks for a ``trapezoid`` written at a call site instead.

        ``axis`` is for layer 4, whose integrands are ``(Nk, NM)``: the one-halo
        integral is taken over mass at every wavenumber at once.
        """
        return jnp.trapezoid(jnp.asarray(integrand) * self.m, self.ln_m,
                             axis=axis)

    def quadrature_measure(self):
        r"""The weights :math:`w_i` with :math:`\int f\,dM = \sum_i w_i f_i`.

        :meth:`integrate` in the form a *sum* needs, for the one consumer that
        cannot use the integral itself:
        :func:`~ggah_mod.halos.beyond_linear_bias.correction_2h_gg` contracts
        its weights with a tabulated :math:`\beta^{\rm NL}` and so carries no
        measure of its own.  Its weights, such as ``dndm * N_tot * b / n_bar``,
        must include the ``dM``, and this supplies it.

        Derived from :attr:`ln_m` rather than restated, so the two cannot
        disagree: ``tests/test_field.py`` asserts
        ``integrate(f) == sum(f * quadrature_measure())`` exactly.
        """
        ln_m = self.ln_m
        d = jnp.diff(ln_m)
        w = 0.5 * jnp.concatenate([d[:1], d[:-1] + d[1:], d[-1:]])
        return w * self.m

    def number_density(self, occupation):
        r""":math:`\bar n = \int (dn/dM)\,\langle N(M)\rangle\,dM`."""
        return self.integrate(self.dndm * jnp.asarray(occupation))

    def effective_bias(self, occupation):
        r""":math:`b_{\rm eff} = \int (dn/dM) N b\,dM / \bar n`."""
        occ = jnp.asarray(occupation)
        return (self.integrate(self.dndm * occ * self.bias)
                / self.integrate(self.dndm * occ))

    def effective_mass(self, occupation):
        r""":math:`M_{\rm eff} = \int (dn/dM) N M\,dM / \bar n`."""
        occ = jnp.asarray(occupation)
        return (self.integrate(self.dndm * occ * self.m)
                / self.integrate(self.dndm * occ))

    # ----------------------------------------------------------------- pytree
    _LEAVES = ("m", "k", "z", "sigma", "dlns", "nu", "dndm", "bias", "conc",
               "r_delta", "r_s", "pk_cb", "pk_lin", "cosmo")
    _STATIC = ("mdef", "hmf_model", "bias_model", "cm_model")

    def tree_flatten(self):
        leaves = tuple(getattr(self, n) for n in self._LEAVES)
        aux = tuple(getattr(self, n) for n in self._STATIC)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        # Runs inside a trace: no validation, no branching on values.
        return cls(*leaves, *aux)

    # ------------------------------------------------------------ convenience
    def replace(self, **kw) -> "HaloField":
        """A copy with some fields changed, validated."""
        return _validate(replace(self, **kw))

    @classmethod
    def create(cls, **kw) -> "HaloField":
        """Build a :class:`HaloField`, checking what can be checked.

        Prefer this over the constructor when the arguments come from user
        input: the constructor is kept branch-free so it can run inside a
        ``jit`` trace.
        """
        return _validate(cls(**kw))


def _validate(f: HaloField) -> HaloField:
    for name, registry, what in (
        (f.hmf_model, MULTIPLICITY, "mass function"),
        (f.bias_model, BIAS, "bias model"),
        (f.cm_model, CONCENTRATION, "concentration relation"),
    ):
        if name not in registry:
            raise ValueError(f"unknown {what} {name!r}; expected one of "
                             f"{sorted(registry)}")
    parse_mass_def(f.mdef)                       # raises on a bad string
    if f.m.shape != f.sigma.shape:
        raise ValueError(
            f"mass grid has {f.m.shape} but sigma has {f.sigma.shape}")
    if f.k.shape != f.pk_cb.shape:
        raise ValueError(
            f"k grid has {f.k.shape} but P_cb has {f.pk_cb.shape}")
    return f


#: Which multiplicity functions take the halo boundary as an argument, and
#: under what name.  Everything else in the registry is a fit at one fixed
#: definition, and passing it a Delta would be an error rather than a courtesy.
#: Defined in :mod:`~ggah_mod.halos.mass_function`, which also holds the
#: dispatch both callers use.
_DELTA_ARG = _MF_DELTA_ARG


def _delta_kw(hmf_model, mdef, z, cosmo, hmf_kw):
    r"""Hand the multiplicity function the mass definition the field declares.

    Without it the mass function uses its own default, 200 against the *mean*
    density, whatever the field declares: a ``200c`` field would have 200c radii
    and a 200m abundance, which disagree by 30 per cent in
    :math:`\dd n/\dd M`, while the mass function stays smooth, positive and
    correctly shaped.

    ``tinker08`` is indexed by :math:`\Delta_{\rm m}` and ``despali16`` by
    :math:`\Delta/\Delta_{\rm vir}`; both are taken against the mean density, so
    the conversion is the same one, and both are skipped when the caller has
    supplied the argument explicitly.  The multiplicity functions in
    :data:`~ggah_mod.halos.mass_function.COSMOLOGY_DEPENDENT_MULTIPLICITY` are
    also handed ``cosmo``.  The dispatch is
    :func:`~ggah_mod.halos.mass_function.multiplicity_kwargs`, which
    :class:`~ggah_mod.halos.mass_function.FittingFunctionHMF` uses too.
    """
    # The cosmology-dependent family is handed `cosmo` here because make_field
    # already has it; the alternative is a model that raises unless the user
    # supplies an argument the other fits refuse.
    return multiplicity_kwargs(hmf_model, mdef, z, cosmo, hmf_kw)


def make_field(cosmo, backend=None, pk=None, z=0.0, *,
               mdef=None, hmf_model=None, bias_model=None,
               cm_model=None, m=None, k=None, calibration="strict", **hmf_kw):
    r"""Build a :class:`HaloField`: cosmology and spectrum in, halo tables out.

    This runs the chain layer 2 exposes as loose functions,
    :func:`~ggah_mod.halos.variance.sigma_of_mass` ->
    :func:`~ggah_mod.halos.variance.dln_sigma_dln_mass` ->
    :func:`~ggah_mod.halos.mass_function.dndm` ->
    :func:`~ggah_mod.halos.linear_bias.make_bias` ->
    :meth:`~ggah_mod.halos.mass_definitions.MassDef.radius` -> a concentration
    relation, and packages the result.  It adds no physics:
    ``tests/test_field.py`` asserts the tables are identical to the
    hand-assembled chain.

    Parameters
    ----------
    cosmo : Cosmology
    backend : Backend or str, optional
        Supplies the grids (``n_m``, ``m_min``, ``m_max``, ``n_k``, ``k_min``,
        ``k_max``) and the four model choices it declares: ``mdef``,
        ``hmf_model``, ``bias_model`` and ``cm_model``.  An explicit argument
        overrides any of them.
    pk : LinearPowerSpectrum
        Any backend from :mod:`ggah_mod.cosmology.power`.  The field is
        differentiable exactly when this is.
    z : float
        Redshift.  May be traced.  The spectrum is evaluated *at* ``z`` rather
        than scaled to it by a growth factor, so there is no ``D(z)`` in this
        function.
    mdef : str, optional
        Halo mass definition; defaults to the backend's.
    hmf_model, bias_model : str, optional
        Keys of :data:`~ggah_mod.halos.mass_function.MULTIPLICITY` and
        :data:`~ggah_mod.halos.linear_bias.BIAS`; default to the backend's.
    cm_model : str, optional
        Any key of :data:`~ggah_mod.halos.concentration.CONCENTRATION`;
        defaults to the backend's.  ``bhattacharya13`` is parametrised by the
        growth factor, so it requires a ``pk`` with native redshift support,
        and says so rather than substituting a fitting formula for the growth
        it cannot read.
    m, k : array, optional
        Override the grids entirely [Msun/h, h/Mpc].  ``k`` is checked against
        the mass range by :func:`~ggah_mod.halos.variance.check_k_support`,
        since a narrow grid under-counts :math:`\sigma(M)` without failing.
    calibration : {"strict", "warn", "off"}
        What to do when the requested ``mdef`` is not one the chosen fits were
        calibrated in.  ``"strict"`` refuses, which is CCL's behaviour and this
        package's convention; ``"warn"`` takes the mismatch knowingly.  See
        :mod:`ggah_mod.halos.calibration`.
    **hmf_kw
        Passed to the multiplicity function (``delta`` for ``tinker08``, ...).

    See Also
    --------
    make_fields : the same thing at N redshifts, for one Boltzmann solve.
    """
    s = _resolve(cosmo, backend, pk, mdef, hmf_model, bias_model, cm_model, m, k)
    # `jnp.asarray` first: `jnp.ndim` of a bare Python list deprecation-warns,
    # and a warning ahead of the refusal below is noise in front of the message
    # that is doing the work.  A tracer passes through unchanged, so the rank is
    # still static and the check still runs under `jit` and `vmap`.
    if jnp.asarray(z).ndim != 0:
        raise ValueError(
            f"make_field builds the halo tables at **one** redshift and got an "
            f"array of shape {jnp.asarray(z).shape}.  A HaloField is one epoch by "
            f"construction -- sigma(M), dn/dM and c(M) are each a function of "
            f"mass alone, and layers 3 and 4 read the redshift off the field "
            f"rather than carrying an axis for it.  There are two ways to get "
            f"a stack, and which one you want depends on the spectrum:\n"
            f"  * `make_fields(cosmo, backend, pk, z_array, ...)` -- one "
            f"batched P(k,z) call, then one field per redshift.  Works for "
            f"every backend, and is the only route that is cheap for CLASS or "
            f"CAMB, where a solve is per *cosmology* and a loop of scalar "
            f"calls pays one full solve per entry.\n"
            f"  * `jax.vmap(lambda zz: make_field(..., z=zz))(z_array)` -- for "
            f"a differentiable backend, when you want the redshift axis inside "
            f"a trace.  Note the result is a *batched* HaloField: it may be "
            f"read and integrated, but must not be handed to a layer-3 sector "
            f"or a layer-4 spectrum, which discriminate (NM,) from (Nk, NM) by "
            f"the leading axis.  vmap the whole z -> spectrum closure instead.")

    # Before the spectrum is asked for anything: this refuses a cosmology the
    # calibrated pieces have no axis for, and it does not take `calibration`.
    # Those are two different claims -- see `check_cosmology_support`.
    check_cosmology_support(s.hmf_model, s.mdef, cosmo)
    check_calibration(s.hmf_model, s.cm_model, s.mdef, z, policy=calibration,
                      bias_model=s.bias_model)
    return _field_from_spectra(
        s, z, jnp.asarray(pk.pk_cb(s.k, z, cosmo)),
        jnp.asarray(pk.pk(s.k, z, cosmo)), cosmo, pk, hmf_kw)


def make_fields(cosmo, backend=None, pk=None, z=(0.0,), *,
                mdef=None, hmf_model=None, bias_model=None,
                cm_model=None, m=None, k=None, calibration="strict", **hmf_kw):
    r"""N one-epoch :class:`HaloField`\ s from one linear-P(k) solve.

    Same arguments as :func:`make_field`, same chain, same physics, except that
    ``z`` is a one-dimensional array, and the return is a tuple of fields in its
    order.  It is a loop with the solve hoisted out of it: each field is the
    ordinary single-redshift field :func:`make_field` returns, and nothing here
    has a redshift axis.

    .. rubric:: Cost of a loop of scalar calls

    A Boltzmann solve is per cosmology.  CLASS and CAMB integrate the
    perturbations once and write out every redshift they were asked for, so the
    memoisation in :class:`~ggah_mod.cosmology.power._BoltzmannBase` keys on the
    whole redshift tuple, and a Python loop of scalar :func:`make_field` calls
    pays a full solve per redshift.  The projection grid of the ``ACCURATE``
    flavour is 64 points (:attr:`~ggah_mod.backend.Backend.n_z_proj`), so that
    cost would land on every likelihood evaluation of every projected
    statistic.

    Measured with CLASS at twelve redshifts:

    * the spectrum call alone takes 6.0 s batched against 93.7 s looped, 15.6x,
      which is the solve count;
    * this function takes 21.3 s against 94.1 s, 4.4x.  The difference is work
      that does not batch: filling the table costs ``n_k * Nz`` Python-level
      calls into CLASS either way, and the halo chain (sigma(M), its slope,
      dn/dM, c(M)) is per redshift.  The solves go from 13 to 2.

    The concentration relations fetch their own growth: ``bhattacharya13``
    wants :math:`D(z)` and ``diemer19`` wants :math:`\dd\ln D/\dd\ln a`.  So the
    growth is evaluated once for the whole array here and handed to
    :func:`_concentration`, which otherwise computes it per field.
    ``tests/test_field.py`` asserts the solve count: one for a relation that
    wants no growth and two for one that does, for any number of redshifts,
    against ``N`` and ``2N`` for the loop.  Two rather than three because the
    memoisation key is the cosmology and the redshift tuple, not ``k``: the
    growth uses its own wavenumber grid but the same redshifts, so it reads the
    solve already paid for, and the only extra is the :math:`z = 0`
    normalisation.

    .. rubric:: Equality with the loop

    The output equals the loop's element for element on CLASS as well as on the
    differentiable backends, because ``z_max_pk``, which decides CLASS's output
    sampling, is the constant :data:`~ggah_mod.cosmology.power.Z_MAX_PK` rather
    than taken from the request; taken from the request, it made a twelve-node
    and a one-node call differ by 9.6e-8 in :math:`\sigma_8` and 1.3e-5 in
    ``dndm``.  ``tests/test_field.py`` asserts the equality, and
    ``tests/test_power.py`` the property underneath it.
    """
    s = _resolve(cosmo, backend, pk, mdef, hmf_model, bias_model, cm_model, m, k)
    z_arr = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    if z_arr.ndim != 1:
        raise ValueError(
            f"make_fields takes a one-dimensional array of redshifts; got "
            f"shape {z_arr.shape}.")

    # Once, ahead of the solve, because it is a statement about the cosmology
    # and not about an epoch: every redshift in the stack would give the same
    # answer, and paying for a Boltzmann solve first to raise afterwards would
    # be the expensive way to reach it.
    check_cosmology_support(s.hmf_model, s.mdef, cosmo)

    # One call each, and they share a cache key, so together they are one solve.
    pk_cb = jnp.atleast_2d(jnp.asarray(pk.pk_cb(s.k, z_arr, cosmo)))
    pk_lin = jnp.atleast_2d(jnp.asarray(pk.pk(s.k, z_arr, cosmo)))
    for name, table in (("pk_cb", pk_cb), ("pk", pk_lin)):
        if table.shape != (len(z_arr), len(s.k)):
            raise ValueError(
                f"{type(pk).__name__}.{name} returned {table.shape} for "
                f"{len(z_arr)} redshifts on a grid of {len(s.k)} wavenumbers; "
                f"a vector redshift must give one row per entry.  A backend "
                f"that quietly ignored the redshift axis would return D(z) = 1 "
                f"and look exact, so the shape is checked rather than trusted.")

    d_all, f_all = _growth_for_stack(s.cm_model, z_arr, cosmo, pk)

    fields = []
    for i in range(len(z_arr)):
        z_i = z_arr[i]
        # Per redshift, so each one answers for itself.  `_check_one` used to
        # answer "traced" and "an array" with the same skip -- `float()` raises
        # `TypeError` for both -- so every stacked path had the range check
        # silently switched off.  It now refuses an array and this loop hands it
        # scalars, which is what makes the warning reachable at all here.
        check_calibration(s.hmf_model, s.cm_model, s.mdef, z_i,
                          policy=calibration, bias_model=s.bias_model)
        fields.append(_field_from_spectra(
            s, z_i, pk_cb[i], pk_lin[i], cosmo, pk, hmf_kw,
            growth_d=None if d_all is None else d_all[i],
            growth_f=None if f_all is None else f_all[i]))
    return tuple(fields)


class _Resolved(NamedTuple):
    """What the flavour and the caller settled between them, before any z."""
    b: object
    mdef: str
    hmf_model: str
    bias_model: str
    cm_model: str
    m: jnp.ndarray
    k: jnp.ndarray


def _resolve(cosmo, backend, pk, mdef, hmf_model, bias_model, cm_model, m, k):
    """Everything :func:`make_field` decides that does not depend on ``z``.

    Shared with :func:`make_fields`: the refusals, the four model choices and
    the two grids are the same at every redshift, and one copy means a flavour
    override is honoured on both paths.
    """
    b = resolve_backend(backend)
    if pk is None:
        raise ValueError(
            f"a halo field needs a linear-P(k) backend: pass "
            f"pk=make_pk({b.pk!r}), which is the one {b.name!r} declares.  It "
            f"is not filled in for you, because which spectrum produced a "
            f"number is exactly the thing this package refuses to leave "
            f"implicit -- but the backend does say which it means.")
    if b.traced and not getattr(pk, "differentiable", False):
        raise ValueError(
            f"backend {b.name!r} is traced but {type(pk).__name__} declares "
            f"differentiable = False, so a gradient through this field would "
            f"succeed and silently omit dP/dtheta.  That is the exact failure "
            f"`traced` exists to prevent, and until now nothing checked it "
            f"here: `Backend.pk` was declared, validated against TRACEABLE_PK, "
            f"and read by no module -- so a traced field built on CambPk passed. "
            f"Use one of the differentiable backends ({b.pk!r} is the one this "
            f"flavour declares), or an untraced flavour.")

    # All four come from the flavour unless the caller overrides them.  They
    # used to come from two places -- mdef and cm_model from the backend, the
    # mass function and bias from this signature's defaults -- so overriding
    # the flavour changed two of the four and silently kept the other two.
    mdef = b.mdef if mdef is None else mdef
    cm_model = (b.cm_model or DEFAULT_CM_MODEL) if cm_model is None else cm_model
    hmf_model = b.hmf_model if hmf_model is None else hmf_model
    bias_model = b.bias_model if bias_model is None else bias_model
    # `log_grid` and not `jnp.logspace`: the latter builds in whatever dtype
    # JAX is configured for, and under the default float32 that misplaces `k`'s
    # nodes in ln k by 5-8 eps32 where one rounding of an exact grid gives 1.0.
    # `k` is the correctness case -- it is the FFTLog abscissa for every
    # real-space statistic, and `make_fftlog` reads its spacing.  `m` is
    # consistency: every consumer integrates it against an explicit `ln_m`, so
    # nothing requires it to be uniform.  Both, because they are two lines
    # expressing one intent and this function exists so that a choice cannot be
    # honoured in one and dropped in the other.
    m = jnp.asarray(log_grid(b.m_min, b.m_max, b.n_m)) \
        if m is None else jnp.asarray(m)
    k = jnp.asarray(log_grid(b.k_min, b.k_max, b.n_k)) \
        if k is None else jnp.asarray(k)

    check_k_support(m, k, cosmo.rho_cold)
    return _Resolved(b, mdef, hmf_model, bias_model, cm_model, m, k)


def _growth_for_stack(cm_model, z_arr, cosmo, pk):
    r"""The growth the concentration relation wants, for the whole array at once.

    Returns ``(D, f)``, either of which is ``None`` when this relation does not
    ask for it; four of the six registered relations ask for neither.  The two
    that do would otherwise call :mod:`~ggah_mod.cosmology.growth` once per
    field, reinstating per-z Boltzmann solves behind the batched spectrum call,
    at a cost of ``O(N)``.
    """
    if cm_model == "bhattacharya13":
        return growth.growth_factor(z_arr, cosmo, pk), None
    if cm_model == "diemer19":
        return None, growth.growth_rate(z_arr, cosmo, pk)
    return None, None


def _field_from_spectra(s: _Resolved, z, pk_cb, pk_lin, cosmo, pk, hmf_kw,
                        *, growth_d=None, growth_f=None):
    """The chain itself: two spectra at one redshift in, one field out.

    The single copy of layer 2's assembly: :func:`make_field` and
    :func:`make_fields` differ only in where the two spectra came from, and
    ``tests/test_field.py`` checks the tables built here against the
    hand-assembled chain.
    """
    m, k, mdef = s.m, s.k, s.mdef
    sigma = sigma_of_mass(m, k, pk_cb, cosmo.rho_cold)
    dlns = dln_sigma_dln_mass(m, k, pk_cb, cosmo.rho_cold)
    nu = DELTA_C / sigma

    n_m = _dndm(m, sigma, dlns, cosmo.rho_cold, model=s.hmf_model, z=z,
                **_delta_kw(s.hmf_model, mdef, z, cosmo, hmf_kw))
    # The bias carries the same boundary as the mass function, for the same
    # reason: `bias_tinker10` is a function of log10(Delta) and every fit in the
    # registry accepts one.  A pair calibrated together is only a
    # pair at a common Delta, so handing 645 to the abundance and 200 to the
    # bias breaks exactly the consistency the pairing exists to provide.
    b_m = make_bias(s.bias_model)(sigma,
                                  parse_mass_def(mdef).delta_mean(z, cosmo))

    r_delta = parse_mass_def(mdef).radius(m, z, cosmo)
    conc = _concentration(s.cm_model, m, z, mdef, sigma, cosmo, pk, k, pk_cb,
                          growth_d=growth_d, growth_f=growth_f)

    return HaloField.create(
        m=m, k=k, z=z, sigma=sigma, dlns=dlns, nu=nu, dndm=n_m, bias=b_m,
        conc=conc, r_delta=r_delta, r_s=r_delta / conc,
        pk_cb=pk_cb, pk_lin=pk_lin, cosmo=cosmo,
        mdef=str(mdef), hmf_model=s.hmf_model, bias_model=s.bias_model,
        cm_model=s.cm_model)


def _concentration(name, m, z, mdef, sigma, cosmo, pk, k=None, pk_cb=None,
                   *, growth_d=None, growth_f=None):
    r"""Dispatch over the two concentration families.

    ``growth_d`` and ``growth_f`` are :math:`D(z)` and
    :math:`\dd\ln D/\dd\ln a` already computed by :func:`make_fields`, which
    evaluates the growth once for a whole redshift array.  ``None`` means not
    supplied, and the relation fetches it from the backend, as for
    :func:`make_field`.

    The empirical power laws take ``(m, z, mdef)`` and cannot respond to a
    cosmology; the peak-height relations take ``sigma`` and one physical second
    argument, a different quantity for each: ``diemer19`` wants the effective
    slope of sigma(M) (negative) and the growth rate, ``bhattacharya13`` the
    growth factor (positive), and ``seppi21`` the redshift.  Passing the slope
    where the growth belongs gives NaN, because the growth is raised to a
    fractional power; passing the redshift there gives a smooth, positive,
    wrong answer, since z and D are both positive and of order unity over the
    range these fits cover.  So ``seppi21`` has a branch of its own rather than
    the fall-through at the bottom.
    """
    fn = make_concentration(name)
    if name not in PEAK_HEIGHT_MODELS:
        return fn(m, z, mdef)
    if name == "seppi21":
        # The `has_native_z` refusal below does not apply here.  It exists
        # because `bhattacharya13` and `diemer19` need a D(z) or a dlnD/dlna
        # that this package will not invent from a fitting formula; `seppi21`
        # is tabulated in z directly and never asks for one.  Refusing a
        # backend for lacking something the relation does not want would be a
        # guard copied rather than reasoned.
        return fn(sigma, z, mdef)
    if name == "diemer19":
        if not getattr(pk, "has_native_z", False):
            raise ValueError(
                f"the {name!r} concentration relation needs the growth *rate* "
                f"d ln D / d ln a, and {type(pk).__name__} declares "
                f"has_native_z = False, so it has no redshift dependence to "
                f"differentiate.  Use a backend with native redshift support.")
        # The slope is taken from *this* spectrum and *this* k grid, not from a
        # second set built here -- which is the whole reason the 2019 relation
        # replaced the 2015 one.
        n_eff = n_eff_from_sigma(
            m, cosmo,
            lambda mm: dln_sigma_dln_mass(mm, k, pk_cb, cosmo.rho_cold))
        f_z = growth.growth_rate(z, cosmo, pk) if growth_f is None else growth_f
        return fn(sigma, n_eff, f_z)
    if not getattr(pk, "has_native_z", False):
        raise ValueError(
            f"the {name!r} concentration relation is parameterised by the "
            f"growth factor, and {type(pk).__name__} declares "
            f"has_native_z = False, so there is no growth to read off it.  "
            f"Use a backend with native redshift support, or a relation that "
            f"does not need one -- substituting a fitting formula for D(z) "
            f"here is exactly the shortcut this package removed.")
    # ...and it gets the definition the field declares, like every other fit
    # at this call site.  It did not, for a while: this line called the
    # relation with its own default of 200c whatever the field said, so a
    # field built at 200m carried 200c concentrations -- half as large again
    # at 1e12 Msun/h, and smooth, positive and correctly shaped throughout.
    # That is the same omission Sec. 3.7 records for the abundance, at the
    # same call site, and it survived the fix to that one because the default
    # relation here is calibrated in 200c alone and so never exercised
    # another definition.
    d_z = growth.growth_factor(z, cosmo, pk) if growth_d is None else growth_d
    return fn(sigma, d_z, mdef)
