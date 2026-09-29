r"""Which summary statistics are wanted, and hence which :math:`P(k)` to compute.

A list of :class:`Statistic` in, one ``jax.jit``-able function of the parameters
out.  The list is **static** -- every field is a hashable, array-free value, so
the whole spec lives in a treedef and only the parameters are traced.  That is
what fixes the output shapes, which is what makes ``jax.jacfwd`` of a data
vector possible, which is the reason ``DIFFERENTIABLE`` exists.

The resolver's one job
----------------------

De-duplicate.  ``w_p^{gg}``, ``ΔΣ^{gm}`` and ``C_ℓ^{gy}`` share tracers and
redshifts; computing each statistic independently evaluates the galaxy weights
three times and :math:`P_{gg}` twice.  :meth:`ObservableSpec.resolve` unions the
redshifts and canonicalises each pair, so every :math:`P_{ab}(k, z)` is computed
exactly once and every statistic reads the one it needs.

The trap that is not obvious
-----------------------------

**JAX flattens a ``dict`` pytree sorted by key, not by insertion order.**  So
``jax.jacfwd`` of a ``dict[name, array]`` silently reorders its rows against the
order the spec was written in, and a Fisher matrix built from it stops matching
the covariance it is paired with -- with nothing raising, because both are still
matrices of the right size.  :func:`make_model` therefore returns the dict
**and** an explicit concatenation whose :attr:`Plan.slices` come from the spec
order, and ``tests/test_spec.py`` asserts the two orders differ where sorting
would change them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..numerics import log_grid
from .real_space import CAP_OUTER
from ..spectra.spec import PkOptions

__all__ = ["Statistic", "Wp", "DeltaSigma", "Xi", "Cl", "WTheta",
           "ObservableSpec", "Plan", "make_model", "KINDS",
           "DEFAULT_WTHETA_ELL"]

#: The statistics this layer knows how to compute.
KINDS = ("wp", "delta_sigma", "xi", "tau_ksz", "cl", "wtheta")


@dataclass(frozen=True)
class Statistic:
    """One wanted measurement.  Frozen, hashable, and free of arrays.

    ``x`` -- the abscissa, :math:`r_p` or :math:`\\ell` or :math:`\\theta` -- is
    a ``tuple`` of floats rather than an array on purpose: an array here would
    become a pytree leaf, and a traced :math:`\\ell` grid breaks ``jacfwd``'s
    fixed output shape as surely as a traced band edge would.
    """

    name: str
    kind: str
    a: str
    b: str
    x: tuple[float, ...]
    z: float | None = None
    pi_max: float | None = None
    kernel_a: str | None = None
    kernel_b: str | None = None
    beam_a: str | None = None
    beam_b: str | None = None
    #: For ``wtheta``: the intermediate multipole grid the projection runs on.
    #: Part of the spec, not a hidden default, because ``w(theta)`` inherits its
    #: accuracy from it.
    ell: tuple[float, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown statistic kind {self.kind!r}; expected "
                             f"one of {list(KINDS)}")
        if not self.x:
            raise ValueError(f"statistic {self.name!r} has an empty abscissa")
        if self.kind in ("wp", "delta_sigma", "xi", "tau_ksz") \
                and self.z is None:
            raise ValueError(
                f"{self.name!r} is a real-space statistic and needs a `z`: it "
                f"is evaluated at one redshift, and there is no default "
                f"because which one is a property of the sample.")
        if self.kind in ("cl", "wtheta") and (self.kernel_a is None
                                              or self.kernel_b is None):
            raise ValueError(
                f"{self.name!r} is a projected statistic and needs a radial "
                f"kernel for each leg; they are what turn a 3D field into "
                f"something on the sky, and lensing exists only as one.")

    @property
    def n(self) -> int:
        return len(self.x)


def Wp(name, a, b, rp, *, pi_max, z) -> Statistic:
    r""":math:`w_p(r_p;\pi_{\max})`.  ``pi_max`` is required; see
    :func:`~ggah_mod.observables.real_space.wp`."""
    return Statistic(name, "wp", a, b, tuple(float(v) for v in rp), z=float(z),
                     pi_max=float(pi_max))


def DeltaSigma(name, a, b, rp, *, z) -> Statistic:
    r""":math:`\Delta\Sigma(r_p)`.  ``b`` should be ``"matter"``."""
    return Statistic(name, "delta_sigma", a, b,
                     tuple(float(v) for v in rp), z=float(z))


def TauKsz(name, a, b, r_ap, *, z, r_grid=None) -> Statistic:
    r""":math:`\tau_{\rm kSZ}` through a compensated aperture, at ``r_ap``.

    The abscissa is the **aperture radius**, not a separation, because that is
    what a stacked kinetic-SZ measurement reports: the estimator that removes
    the primary CMB is the aperture filter, so a profile at a radius is not the
    quantity anyone measured.  See
    :func:`~ggah_mod.observables.real_space.compensated_aperture`.

    ``b`` should be ``"electrons"`` -- the tracer of every free electron -- and
    ``a`` the galaxy sample being stacked on, for the reason
    :func:`~ggah_mod.observables.real_space.tau_ksz` gives: an auto-spectrum
    silently answers a different question.

    ``r_grid`` is the internal grid the profile is tabulated on before
    filtering, and it is part of the spec for the same reason ``WTheta``'s
    ``ell`` is: the statistic inherits its accuracy from it, and a default
    hidden inside the model would make that accuracy invisible.  ``None`` takes
    a grid spanning a decade inside the smallest aperture out to
    :math:`\sqrt2\times` the largest, which is what the filter needs at each end.
    """
    r_ap = tuple(float(v) for v in r_ap)
    if r_grid is None:
        lo, hi = min(r_ap) / 10.0, CAP_OUTER * max(r_ap)
        r_grid = tuple(float(v) for v in log_grid(lo, hi, 256))
    else:
        r_grid = tuple(float(v) for v in r_grid)
    return Statistic(name, "tau_ksz", a, b, r_ap, z=float(z), ell=r_grid)


def Xi(name, a, b, r, *, z) -> Statistic:
    r""":math:`\xi(r)`."""
    return Statistic(name, "xi", a, b, tuple(float(v) for v in r), z=float(z))


def Cl(name, a, b, ell, *, kernel_a, kernel_b, beam_a=None,
       beam_b=None) -> Statistic:
    r""":math:`C_\ell^{ab}`.  ``beam_*`` name one entry of the beams mapping."""
    return Statistic(name, "cl", a, b, tuple(float(v) for v in ell),
                     kernel_a=kernel_a, kernel_b=kernel_b,
                     beam_a=beam_a, beam_b=beam_b)


#: The default intermediate multipole grid for a ``w(theta)``.
#:
#: Wide, because ``w(theta)`` inherits its accuracy from it: a grid stopping at
#: the multipoles someone happened to plot would truncate the transform rather
#: than the plot.
#:
#: **Built with numpy, and that is not a style choice.**  This was
#: ``10.0 ** jnp.linspace(0.0, 5.0, 256)``, evaluated at import time in whatever
#: dtype JAX was configured for and then widened by ``float()`` -- so the tuple
#: held *float32-rounded values inside float64 Python floats*, with nothing in
#: the object recording where they came from.  Two processes differing only in
#: ``JAX_ENABLE_X64`` projected ``w(theta)`` onto different multipole grids.
#: That is a provenance bug rather than a precision one, and it is exactly the
#: kind the widening hides: ``make_fftlog`` reads float64 off the tuple,
#: applies the tolerance float64 has earned, and refuses a grid whose
#: deviation is 3.3e9 eps64.
DEFAULT_WTHETA_ELL = tuple(float(v) for v in log_grid(1.0, 1e5, 256))


def WTheta(name, a, b, theta, *, kernel_a, kernel_b, ell=None,
           beam_a=None, beam_b=None) -> Statistic:
    r""":math:`w(\theta)`, through :math:`C_\ell`."""
    return Statistic(name, "wtheta", a, b, tuple(float(v) for v in theta),
                     kernel_a=kernel_a, kernel_b=kernel_b,
                     beam_a=beam_a, beam_b=beam_b,
                     ell=DEFAULT_WTHETA_ELL if ell is None
                     else tuple(float(v) for v in ell))


@dataclass(frozen=True)
class Plan:
    """What :meth:`ObservableSpec.resolve` worked out.  All static."""

    statistics: tuple[Statistic, ...]
    #: Canonically ordered, de-duplicated tracer-name pairs.
    pairs: tuple[tuple[str, str], ...]
    #: Every redshift a spectrum is needed at, once -- the projection grid
    #: plus each real-space statistic's own.
    z_nodes: tuple[float, ...]
    #: ``(name, start, stop)`` into the concatenated data vector, **in spec
    #: order** -- not sorted, which is what a dict would give.
    slices: tuple[tuple[str, int, int], ...]
    #: The projection grid alone, **in its own order**.
    #:
    #: Kept apart from :attr:`z_nodes` rather than filtered out of it: a
    #: ``C_l`` stack has to line up row-for-row with its kernels' chi grid, and
    #: a real-space statistic at some other redshift joins ``z_nodes`` without
    #: belonging in that stack.  Merging the two is a shape error on a good day
    #: and a silently misaligned projection on a bad one.
    z_proj: tuple[float, ...] = ()
    #: Every ``(pair, z)`` some statistic actually reads, once, sorted by
    #: ``(z, pair)``.
    #:
    #: :attr:`pairs` and :attr:`z_nodes` are honest summaries -- which tracers
    #: appear, and which redshifts -- but their *product* is not the work.
    #: Fitting complementary probes at different redshifts is exactly the case
    #: where it is much larger: a spec with clustering and lensing at one
    #: redshift, clustering at a second and an AGN correlation at a third needs
    #: four spectra out of a nine-element product, and adding one ``C_l`` on a
    #: 32-node Limber grid makes it 34 out of 99.  Computing the product means
    #: evaluating the tSZ pair at the galaxy redshift and both galaxy pairs at
    #: every projection node, and no statistic ever reads any of them.
    needs: tuple[tuple[tuple[str, str], float], ...] = ()

    @property
    def size(self) -> int:
        return self.slices[-1][2] if self.slices else 0

    def pairs_at(self, z: float) -> tuple[tuple[str, str], ...]:
        """The pairs some statistic reads at ``z``, in :attr:`needs` order.

        A method rather than an inline filter in :func:`make_model` so the
        sparsity is a property of the plan that a test can assert against,
        instead of a loop body's shape.
        """
        zf = float(z)
        return tuple(pair for pair, zz in self.needs if zz == zf)


def _canonical(a: str, b: str) -> tuple[str, str]:
    """``(a, b)`` sorted by name.

    Sorted rather than a ``frozenset``: a set of one element cannot tell
    :math:`P_{gg}` from a cross of two tracers that happen to share a name, and
    the auto case is exactly the one with a different pair rule.
    """
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class ObservableSpec:
    """The list of wanted statistics."""

    statistics: tuple[Statistic, ...]

    def __post_init__(self):
        names = [s.name for s in self.statistics]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"two statistics share a name: {dupes}.  Names index the output "
                f"dict and the data-vector slices, so a duplicate silently "
                f"drops one of them.")

    def resolve(self, *, z_proj: tuple[float, ...] = ()) -> Plan:
        """De-duplicate into a :class:`Plan`.

        ``z_proj`` is the projection grid, supplied by the caller because it
        comes from :func:`~ggah_mod.observables.kernels.limber_grid` and is
        shared by **every** projected statistic -- a pair integrated on
        different grids has an inconsistent cross-covariance.
        """
        pairs, needs, slices, start = [], [], [], 0
        for s in self.statistics:
            pair = _canonical(s.a, s.b)
            if pair not in pairs:
                pairs.append(pair)
            # A projected statistic reads its pair at *every* node of the
            # shared grid; a real-space one at its own redshift and nowhere
            # else.  That asymmetry is the whole point of the sparse set.
            zs = z_proj if s.kind in ("cl", "wtheta") else (s.z,)
            for z in zs:
                entry = (pair, float(z))
                if entry not in needs:
                    needs.append(entry)
            slices.append((s.name, start, start + s.n))
            start += s.n
        # Sorted by (z, pair) so `make_model` can walk one redshift at a time
        # through fields its builder handed back in this same order.
        needs.sort(key=lambda e: (e[1], e[0]))
        z_nodes = sorted({z for _, z in needs})
        return Plan(tuple(self.statistics), tuple(pairs),
                    tuple(z_nodes), tuple(slices),
                    tuple(float(z) for z in z_proj), tuple(needs))

    def required_tracers(self) -> tuple[str, ...]:
        """Every tracer named, once, in first-appearance order."""
        seen = []
        for s in self.statistics:
            for name in (s.a, s.b):
                if name not in seen:
                    seen.append(name)
        return tuple(seen)


def make_model(spec: ObservableSpec, *, fields_at, sectors, kernels,
               beams=None, cosmo, backend=None,
               options: PkOptions | None = None, z_proj=(), field_at=None):
    r"""Compile a spec into ``params -> (dict, data_vector)``.

    Parameters
    ----------
    fields_at : callable
        ``(z0, z1, ...) -> (HaloField, HaloField, ...)``, one field per node in
        the order given.  Supplied, not built here: which spectrum produced a
        number is the thing this package refuses to leave implicit, and layer 5
        is no place to start.

        It takes the **whole** node set rather than one redshift because layer 1
        is batched in the redshift and layer 2 is not.  A Boltzmann solve is
        memoised on the redshift *tuple*
        (:class:`~ggah_mod.cosmology.power._BoltzmannBase`), so a callable asked
        one redshift at a time pays a full solve for each: with this flavour's
        :attr:`~ggah_mod.backend.Backend.n_z_proj` that is one solve per node,
        on every evaluation of the returned model.
        :func:`~ggah_mod.halos.field.make_fields` is the builder this
        expects.
    sectors, kernels, beams : dict
        ``name -> object``.  Same reason.
    z_proj : array
        The shared projection grid from
        :func:`~ggah_mod.observables.kernels.limber_grid`.

    Returns
    -------
    callable
        ``params -> (dict[name, array], concatenated)``.  Both, because a dict
        alone is a trap: JAX flattens it **sorted by key**, so ``jacfwd`` of it
        reorders the rows against the spec.  The concatenation follows
        :attr:`Plan.slices`, which is spec order.
    """
    from ..backend import resolve_backend
    from ..spectra.tracers import spectrum as pk_spectrum
    from .limber import c_ell
    from .real_space import (
        compensated_aperture, delta_sigma, tau_ksz, wp, xi,
    )
    from .transforms import cl_to_wtheta

    if field_at is not None:
        raise ValueError(
            "`field_at` is now `fields_at`, and it takes the whole redshift "
            "set at once rather than one redshift at a time.  The old shape "
            "made one linear-P(k) call per node, and a Boltzmann solve is "
            "memoised on the redshift tuple -- so a 64-node projection grid "
            "cost 64 solves per likelihood evaluation where it now costs two, "
            "and nothing in the output said so.  Pass "
            "`fields_at=lambda zz: make_fields(cosmo, backend, pk, zz)`.")

    b = resolve_backend(backend)
    options = options or PkOptions.from_backend(b)
    beams = beams or {}
    plan = spec.resolve(z_proj=tuple(float(z) for z in z_proj))

    def model(params):
        # One spectrum per (pair, z) that some statistic actually reads.
        #
        # `plan.needs` rather than `pairs x z_nodes`: the product computes
        # spectra nobody reads, and the gap widens precisely as the fit gets
        # more interesting -- see `Plan.needs`.  What this does *not* save is
        # fields: a `HaloField` is built at a redshift if any pair is wanted
        # there, so layer 2's sigma(M), dn/dM and concentration work is
        # unchanged and the saving is entirely layer-4 spectra.
        #
        # The fields themselves come in **one** call for the whole node set,
        # because layer 1 is batched in the redshift and layer 2 is not: asked
        # one at a time, each would make its own linear-P(k) request and a
        # Boltzmann solve is memoised on the redshift tuple.  See `fields_at`.
        fields = tuple(fields_at(plan.z_nodes))
        if len(fields) != len(plan.z_nodes):
            raise ValueError(
                f"fields_at returned {len(fields)} fields for "
                f"{len(plan.z_nodes)} redshifts {plan.z_nodes}; the "
                f"correspondence is positional, so a short or long answer "
                f"would pair a spectrum with the wrong redshift and say "
                f"nothing.")

        from ..spectra.bnl import table_for
        cache = {}
        for z, field in zip(plan.z_nodes, fields):
            # One beyond-linear table per field, not per pair.
            tab = table_for(field, options)
            for a, b_name in plan.pairs_at(z):
                p = pk_spectrum(field, a, b_name, sectors, params,
                                options=options, bnl_table=tab)
                cache[(a, b_name, z)] = (field.k, p.total)

        out = {}
        for s in plan.statistics:
            a, b_name = _canonical(s.a, s.b)
            x = jnp.asarray(s.x)
            if s.kind in ("wp", "delta_sigma", "xi", "tau_ksz"):
                k, p = cache[(a, b_name, float(s.z))]
                if s.kind == "wp":
                    out[s.name] = wp(x, (k, p), pi_max=s.pi_max, backend=b)
                elif s.kind == "delta_sigma":
                    out[s.name] = delta_sigma(x, (k, p), cosmo, backend=b)
                elif s.kind == "tau_ksz":
                    out[s.name] = compensated_aperture(
                        x, jnp.asarray(s.ell),
                        tau_ksz(jnp.asarray(s.ell), (k, p), cosmo, s.z,
                                backend=b))
                else:
                    out[s.name] = xi(x, (k, p), backend=b)
                continue

            if not plan.z_proj:
                raise ValueError(
                    f"{s.name!r} is a projected statistic but the plan has no "
                    f"projection grid; pass `z_proj=` from `limber_grid` so "
                    f"every projected statistic shares one.")
            k = cache[(a, b_name, plan.z_proj[0])][0]
            stack = jnp.stack([cache[(a, b_name, z)][1] for z in plan.z_proj])
            ell = x if s.kind == "cl" else jnp.asarray(s.ell)
            cl = c_ell(ell, k, stack, kernels[s.kernel_a], kernels[s.kernel_b],
                       beam_a=None if s.beam_a is None else beams[s.beam_a](ell),
                       beam_b=None if s.beam_b is None else beams[s.beam_b](ell))
            out[s.name] = cl if s.kind == "cl" else cl_to_wtheta(
                x, ell, cl, backend=b)

        vector = jnp.concatenate([out[name] for name, _, _ in plan.slices])
        return out, vector

    model.plan = plan
    return model

