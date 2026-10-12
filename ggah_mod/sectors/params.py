r"""Parameters that carry their own bounds, priors and reasons.

This module implements the package's fifth standing decision: *nothing is fixed
by hard-coding; a constant a user might want to vary is a parameter with a prior
and a bound*.
Layers 1 and 2 state it without a mechanism, because their constants are
calibrations of published fits and are not meant to vary.
Layer 3 needs one: its whole content is parameters.
The rule was written about the predecessor's gas sector, in which nine
parameters varied and the rest were frozen: the entire pressure shape, the
entire metallicity sector (whose class had no constructor arguments at all),
and two shape parameters held at 0.0.
A parameter frozen at a value nobody chose is indistinguishable, from the
outside, from one that was measured.

Two mechanisms follow.

**A bound needs a reason.** :class:`Param` carries ``why``.
The predecessor's bounds table justified every entry and tagged each as
*definitional*, *physical*, *prior* or *validity*; the justifications are
ported with the numbers rather than re-derived, because a bound whose reason is
lost cannot be revised.

**Differentiability is structural.** A :class:`SectorParams` splits its
contents in two:

* names in ``_PARAMS`` are pytree leaves, which ``jax.grad`` reaches;
* names in ``_STATIC`` live in the treedef, so differentiating one is
  impossible rather than a silent zero.

With a plain ``dict`` of parameters, ``float(params["A_cen"])`` inside a traced
function, on a branch that does not use the value, gives a derivative of
exactly zero with nothing complaining, and a Fisher matrix built on it reports
infinite confidence rather than failing.
``lange25.py`` does this to the two assembly-bias amplitudes the model exists to
vary.

The container pattern follows ``jax_cosmo`` (Campagne et al. 2023, OJAp 6, 15).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import ClassVar

import jax
import jax.numpy as jnp

__all__ = ["Prior", "Flat", "Gaussian", "LogGaussian", "MultivariateGaussian",
           "Param", "SectorParams", "sector_params", "BOUND_KINDS"]

#: Why a bound exists.  Not decoration: the four kinds are revised differently.
#:
#: * ``definitional`` -- outside it the quantity is not the quantity any more
#:   (a fraction above 1, an integral that diverges).  Never relaxed.
#: * ``physical`` -- a physical argument, revisable if the argument is.
#: * ``prior`` -- a measurement.  Moves when the measurement does.
#: * ``validity`` -- the range the fit was calibrated over.  Outside it the
#:   model still evaluates and is no longer the published model.
BOUND_KINDS = ("definitional", "physical", "prior", "validity")


# =========================================================================
# Priors
# =========================================================================

class Prior:
    """A log-density.  Differentiable in the value, always."""

    def log_prob(self, x):                       # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Flat(Prior):
    """Improper uniform: contributes a constant, so a gradient of exactly zero.

    That zero is correct: it is the derivative of a constant, not a dependence
    that fell out of the graph.
    """

    def log_prob(self, x):
        return jnp.zeros_like(jnp.asarray(x, dtype=float))


@dataclass(frozen=True)
class Gaussian(Prior):
    r"""Gaussian log-density, :math:`-\tfrac12((x-\mu)/\sigma)^2`, normalisation
    dropped."""

    mu: float
    sigma: float

    def log_prob(self, x):
        return -0.5 * ((jnp.asarray(x) - self.mu) / self.sigma) ** 2


@dataclass(frozen=True)
class LogGaussian(Prior):
    r"""Gaussian in :math:`\log_{10} x`, for a scale that spans decades.

    Distinct from :class:`Gaussian` on an already-log parameter: this one takes
    the log itself, so it is the right prior for an amplitude stored linearly.
    """

    mu_log10: float
    sigma_log10: float

    def log_prob(self, x):
        lx = jnp.log10(jnp.asarray(x))
        return -0.5 * ((lx - self.mu_log10) / self.sigma_log10) ** 2


@dataclass(frozen=True, eq=False)
class MultivariateGaussian(Prior):
    r"""A correlated Gaussian over several named parameters at once.

    .. math::

        \ln p(\theta) = -\tfrac12 (\theta-\mu)^T C^{-1} (\theta-\mu)

    normalisation dropped, as :class:`Gaussian` drops it.

    Correlated rather than independent Gaussians
    --------------------------------------------

    A posterior is not diagonal.
    When one fit's result becomes the next fit's prior, as a calibration
    campaign chains blocks that share a galaxy sample, thirteen independent
    Gaussians on thirteen correlated parameters are not a conservative
    approximation of the posterior but a *different* and generally tighter
    distribution: the correlated volume is a thin ellipsoid, the product of its
    marginals is the axis-aligned box around it, and the box constrains
    parameter combinations the fit never constrained.

    A prior of the fit, not of a :class:`Param`
    -------------------------------------------

    A ``Param`` describes one number, and :meth:`SectorParams.log_prior`
    evaluates each parameter's prior on its own value.
    A joint prior couples several, so it belongs to the *fit* rather than to
    any one field, and it is passed to :meth:`SectorParams.log_prior` as
    ``joint=``.
    :class:`Param` refuses one outright, because the failure it would otherwise
    produce (``log_prob`` handed a scalar where it expects a vector) gives a
    plausible number rather than an error.

    Parameters
    ----------
    names : tuple[str, ...]
        The parameters this prior is over, in the row order of ``mu`` and
        ``cov``.
        Names rather than positions, because the order a chain was written in
        and the order a container declares its fields in are two different
        orders and nothing checks that they agree.
    mu : array, shape (n,)
        The mean, in the order of ``names``.
    cov : array, shape (n, n)
        Symmetric positive definite.
        Refused otherwise, with the smallest eigenvalue reported: a posterior
        covariance that has gone singular is a chain that did not converge, and
        silently pseudo-inverting it turns that into a prior nobody can
        question.

    Notes
    -----
    The Cholesky factor is taken once, at construction, and reused.
    This is not on a traced path: the prior is a constant of the fit.
    """

    names: tuple
    mu: jnp.ndarray
    cov: jnp.ndarray

    def __post_init__(self):
        names = tuple(str(n) for n in self.names)
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"MultivariateGaussian: repeated name(s) {dup}. One row per "
                f"parameter, or the covariance means two different things in "
                f"two places")
        mu = jnp.atleast_1d(jnp.asarray(self.mu, dtype=float))
        cov = jnp.atleast_2d(jnp.asarray(self.cov, dtype=float))
        n = len(names)
        if mu.shape != (n,):
            raise ValueError(
                f"MultivariateGaussian: {n} names but mu has shape {mu.shape}")
        if cov.shape != (n, n):
            raise ValueError(
                f"MultivariateGaussian: {n} names but cov has shape {cov.shape}")
        if not bool(jnp.all(jnp.isfinite(cov))) or not bool(jnp.all(jnp.isfinite(mu))):
            raise ValueError("MultivariateGaussian: mu and cov must be finite")
        asym = float(jnp.max(jnp.abs(cov - cov.T)))
        scale = float(jnp.max(jnp.abs(cov))) or 1.0
        if asym > 1e-10 * scale:
            raise ValueError(
                f"MultivariateGaussian: cov is not symmetric "
                f"(max |C - C^T| = {asym:g})")
        cov = 0.5 * (cov + cov.T)
        eig = jnp.linalg.eigvalsh(cov)
        lo = float(eig[0])
        if lo <= 0.0:
            raise ValueError(
                f"MultivariateGaussian: cov is not positive definite (smallest "
                f"eigenvalue {lo:.3e}). A posterior covariance that has gone "
                f"singular is a chain that did not converge; pseudo-inverting "
                f"it here would turn that into a prior nobody can question")
        chol = jnp.linalg.cholesky(cov)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "_chol", chol)

    def log_prob(self, x):
        r"""The log-density at ``x``, the values in :attr:`names` order, shape ``(n,)``.

        Solved through the Cholesky factor rather than by forming
        :math:`C^{-1}`: for a posterior covariance with a condition number in
        the thousands, which correlated parameters give, the explicit inverse
        loses digits the triangular solve keeps.
        """
        d = jnp.asarray(x, dtype=float) - self.mu
        z = jax.scipy.linalg.solve_triangular(self._chol, d, lower=True)
        return -0.5 * jnp.sum(z ** 2)

    def log_prob_for(self, container):
        """Evaluate against a :class:`SectorParams` (or any object with the names).

        Gathers by attribute name, so a container that declares its fields in a
        different order from the chain that produced the prior still gets the
        right answer.
        """
        missing = [n for n in self.names if not hasattr(container, n)]
        if missing:
            raise AttributeError(
                f"{type(container).__name__} has no parameter(s) {missing}; a "
                f"joint prior over {list(self.names)} cannot be applied to it")
        return self.log_prob(
            jnp.stack([jnp.asarray(getattr(container, n), dtype=float)
                       for n in self.names]))

    def marginal(self, name: str) -> Gaussian:
        """The one-dimensional marginal for ``name``, as a :class:`Gaussian`.

        For reporting only: summing the marginals is not this prior, so a
        marginal does not belong in a likelihood.
        """
        try:
            i = self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not in this prior; it covers "
                f"{list(self.names)}") from None
        return Gaussian(mu=float(self.mu[i]), sigma=float(jnp.sqrt(self.cov[i, i])))

    def correlation(self):
        """The correlation matrix, for the report."""
        d = jnp.sqrt(jnp.diag(self.cov))
        return self.cov / jnp.outer(d, d)

    @classmethod
    def from_samples(cls, names, samples) -> "MultivariateGaussian":
        """Fit to an MCMC chain: ``samples`` is ``(n_samples, n_names)``.

        The Gaussian approximation to a posterior, the form a downstream block
        can consume.
        A chain whose posterior is not close to Gaussian should not be passed on
        this way; nothing here checks that.
        """
        samples = jnp.asarray(samples, dtype=float)
        names = tuple(names)
        if samples.ndim != 2 or samples.shape[1] != len(names):
            raise ValueError(
                f"samples must be (n_samples, {len(names)}), got {samples.shape}")
        if samples.shape[0] <= samples.shape[1]:
            raise ValueError(
                f"{samples.shape[0]} samples for {samples.shape[1]} parameters: "
                f"the sample covariance is singular by construction")
        mu = jnp.mean(samples, axis=0)
        d = samples - mu
        cov = (d.T @ d) / (samples.shape[0] - 1)
        return cls(names=names, mu=mu, cov=cov)

    def __repr__(self) -> str:                           # pragma: no cover
        return (f"MultivariateGaussian(names={list(self.names)}, "
                f"sigma={[float(s) for s in jnp.sqrt(jnp.diag(self.cov))]})")


# =========================================================================
# A parameter
# =========================================================================

@dataclass(frozen=True)
class Param:
    """One free parameter: its default, its range, its prior, and its reason.

    Parameters
    ----------
    default : float
        The default value, which must lie inside ``bounds``.
    bounds : tuple[float, float]
        Inclusive.
        ``(-inf, inf)`` is allowed and means genuinely unbounded, which is a
        claim: most quantities are not.
    prior : Prior
        Any one-parameter :class:`Prior`; a :class:`MultivariateGaussian` is
        refused.
    unit : str
        ``""`` for dimensionless.
        Written out because the predecessor had a pivot mass in ``Msun`` in one
        line and ``Msun/h`` in the next, inside one function.
    why : str
        Why *this* bound.
        A bound whose reason is lost cannot be revised.
    kind : str
        One of :data:`BOUND_KINDS`.
    """

    default: float
    bounds: tuple[float, float] = (-jnp.inf, jnp.inf)
    prior: Prior = Flat()
    unit: str = ""
    why: str = ""
    kind: str = "physical"

    def __post_init__(self):
        lo, hi = self.bounds
        if not lo <= self.default <= hi:
            raise ValueError(
                f"default {self.default} is outside its own bounds {self.bounds}")
        if self.kind not in BOUND_KINDS:
            raise ValueError(f"kind must be one of {BOUND_KINDS}, got {self.kind!r}")
        if not self.why:
            raise ValueError(
                "a Param needs `why`: the reason for its bound.  This package's "
                "standing rule is that nothing is fixed by hard-coding, and a "
                "bound with no recorded reason is a hard-coding with extra steps "
                "-- nobody can tell later whether it was measured or guessed.")
        if isinstance(self.prior, MultivariateGaussian):
            raise TypeError(
                "a MultivariateGaussian is a prior over several parameters and "
                "cannot belong to one Param.  SectorParams.log_prior evaluates "
                "each parameter's prior on that parameter's own value, so this "
                "would hand a vector prior a scalar and get a number back "
                "rather than an error.  Pass it as "
                "`params.log_prior(joint=[mvg])` instead.")

    def clip(self, x):
        """Project into the bounds; for proposals, never inside a model.

        A ``clip`` on the traced path gives plausible numbers with a dead
        gradient: JAX splits the tie at the boundary 50/50 and the derivative
        outside it is zero.
        Bounds belong in the sampler.
        """
        lo, hi = self.bounds
        return jnp.clip(jnp.asarray(x), lo, hi)


# =========================================================================
# The container
# =========================================================================

class SectorParams:
    """Base for a sector's parameter set: a pytree, split by differentiability.

    Subclasses are frozen dataclasses decorated with :func:`sector_params`, and
    declare:

    * ``_PARAMS``: ``{name: Param}``, in field order, the traced leaves;
    * ``_STATIC``: names that select a *model* rather than a value.
      They live in the treedef, so a recompile is triggered when they change
      and a gradient with respect to one cannot be asked for.

    ``tests/test_sectors_paths.py`` asserts the two together account for every
    dataclass field, so a parameter added to the class and forgotten here is a
    test failure rather than a leaf that silently stops being differentiated.
    """

    _PARAMS: ClassVar[dict[str, Param]] = {}
    _STATIC: ClassVar[tuple[str, ...]] = ()

    # ----------------------------------------------------------------- pytree
    def tree_flatten(self):
        leaves = tuple(getattr(self, n) for n in self._PARAMS)
        aux = tuple(getattr(self, n) for n in self._STATIC)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        """Runs inside a trace: no validation, no branching on values.

        Built with ``object.__new__`` rather than ``__init__`` for that reason:
        ``__init__`` may check things, and under ``jit`` the values it would
        check are tracers.
        """
        obj = object.__new__(cls)
        for name, value in zip(cls._PARAMS, leaves):
            object.__setattr__(obj, name, value)
        for name, value in zip(cls._STATIC, aux):
            object.__setattr__(obj, name, value)
        return obj

    # ------------------------------------------------------------ convenience
    def replace(self, **kw) -> "SectorParams":
        """A copy with some values changed."""
        unknown = set(kw) - set(self._PARAMS) - set(self._STATIC)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.replace: unknown parameter(s) "
                f"{sorted(unknown)}; known are {sorted(self._PARAMS)} and "
                f"config keys {sorted(self._STATIC)}")
        return replace(self, **kw)

    def as_dict(self) -> dict:
        """Every name to its current value.  For reporting, not for the model."""
        return {n: getattr(self, n)
                for n in (*self._PARAMS, *self._STATIC)}

    # ----------------------------------------------------------------- priors
    def log_prior(self, joint=()):
        """Sum of the per-parameter log-priors, plus any joint ones.

        Differentiable in every leaf.

        Parameters
        ----------
        joint : iterable of MultivariateGaussian
            Correlated priors over named subsets of this container's
            parameters, typically an upstream fit's posterior.
            Each one *replaces* the independent priors of the parameters it
            covers: adding both would count the same measurement twice, and the
            version counted twice would be the one whose correlations were
            thrown away.
            A joint prior naming a parameter that is not in ``_PARAMS`` raises
            rather than being silently ignored.

        Notes
        -----
        Bounds are deliberately not applied here.
        A hard rejection is a step, and a step has no useful derivative;
        enforcing a bound inside the model puts a dead gradient into a fit.
        Use :meth:`within_bounds` in the sampler, where a rejection is wanted.
        """
        joint = tuple(joint)
        covered = set()
        for mvg in joint:
            unknown = [n for n in mvg.names if n not in self._PARAMS]
            if unknown:
                raise KeyError(
                    f"{type(self).__name__}.log_prior: joint prior names "
                    f"{unknown}, which are not parameters of this container "
                    f"({sorted(self._PARAMS)})")
            repeated = covered & set(mvg.names)
            if repeated:
                raise ValueError(
                    f"{type(self).__name__}.log_prior: {sorted(repeated)} appear "
                    f"in more than one joint prior, so one measurement would be "
                    f"counted twice")
            covered |= set(mvg.names)

        total = 0.0
        for name, p in self._PARAMS.items():
            if name in covered:
                continue
            total = total + jnp.sum(p.prior.log_prob(getattr(self, name)))
        for mvg in joint:
            total = total + mvg.log_prob_for(self)
        return total

    def within_bounds(self) -> bool:
        """For a sampler's proposal step.  Concrete values only."""
        return all(
            bool(jnp.all((jnp.asarray(getattr(self, n)) >= p.bounds[0])
                         & (jnp.asarray(getattr(self, n)) <= p.bounds[1])))
            for n, p in self._PARAMS.items())

    # --------------------------------------------------------------- registry
    @classmethod
    def bounds(cls) -> dict:
        return {n: p.bounds for n, p in cls._PARAMS.items()}

    @classmethod
    def defaults(cls) -> dict:
        return {n: p.default for n, p in cls._PARAMS.items()}

    @classmethod
    def free_names(cls) -> tuple:
        """The differentiable leaves, in flatten order."""
        return tuple(cls._PARAMS)


def sector_params(cls):
    """Make a :class:`SectorParams` subclass into a frozen dataclass pytree.

    One decorator so the registration and the freezing cannot come apart: an
    unregistered params class works until the model is put under ``jit``, and
    then fails somewhere else.
    """
    return jax.tree_util.register_pytree_node_class(dataclass(frozen=True)(cls))


def declared_fields(cls) -> tuple:
    """Dataclass field names, for the coverage test."""
    return tuple(f.name for f in fields(cls))
