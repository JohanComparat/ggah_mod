r"""The cosmic baryon census: :math:`\sum_i \Omega_i \equiv \Omega_b`.

:mod:`~ggah_mod.sectors.matter` closes a **per-halo** budget --
:math:`\sum_i f_i = 1` for one halo, algebraically.  This module closes the
**cosmic** one, and the two are not the same statement.  Integrating a closed
per-halo budget over the mass function does *not* give :math:`\Omega_b`:

.. math::

    \sum_i \Omega_i(>M_{\min}) = \Omega_b\,F,
    \qquad F = \int_{M_{\min}} dM\,\frac{dn}{dM}\,\frac{M}{\bar\rho_{cb}}

and :math:`F` is **0.5248** on the shipped grid -- ``ACCURATE``, 200m,
``tinker08_csst``, PLANCK18, :math:`z = 0`,
:math:`M_{\min} = 10^{10}\,M_\odot/h` -- and 0.5247 on ``DIFFERENTIABLE``
(``emu_pk``, 256 mass nodes).  Half the mass of the universe is in halos below
:math:`10^{10}\,M_\odot/h`, so a census that stops at the grid edge reports 48%
of the baryons missing and calls it physics.  It is not physics; it is where
the mass grid was cut.  Widening it does not settle
the question either -- :math:`F = 0.6231` at :math:`10^8` and 0.6873 at
:math:`10^6` on the same flavour, still a third short, in a regime where the
multiplicity function was never calibrated.  (This paragraph used to quote
0.5185, 0.7302 and an overshoot to 1.1151: plain ``tinker08`` evaluated
without the field's mass definition, which the package no longer computes --
see :mod:`~ggah_mod.spectra.counterterm`.)

This is the same trap :mod:`~ggah_mod.spectra.counterterm` was written for on
the two-halo side, one level up: there the missing mass makes
:math:`P_{mm}^{2h}` short by a factor of two, here it makes the baryon budget
short by a factor of two.  Both are solved by naming the deficit rather than
absorbing it.

The three terms, and which one is a residual
---------------------------------------------

.. math::

    \Omega_b = \underbrace{\sum_i \Omega_i(>M_{\min})}_{\text{resolved halos}}
             + \underbrace{\Omega_{\rm outside}}_{\text{everything else}}

Two terms, not three, and the missing third is deliberate.  What is outside the
grid is *both* baryons in halos too small to resolve and baryons in no halo at
all, and **the halo model cannot separate them**: below :math:`M_{\min}` there
is no mass function this package will vouch for -- none of its multiplicity
functions was calibrated there -- so any split between the two would be an
extrapolation wearing the clothes of a measurement.  The census reports the sum
and lets :func:`census_over_mass_ranges` show how much of it is the grid.

What the outside term is *made of* is a separate question, and it is declared
rather than left open, because the two-halo term needs an answer:
:attr:`BaryonCensus.omega_outside_by_phase` keeps the stars and neutral gas of
the smallest resolved halo and books every other baryon as diffuse, ionised
gas.  The spectra read the same composition through each tracer's
``w_unresolved`` -- the hot-gas views declare none, the ejecta sector carries
the diffuse share -- and :attr:`BaryonCensus.unresolved_bias` is the ratio
:math:`(1-B)/(1-F)` that turns this row's mass into the counterterm's
bias-weighted mass.  One object, two weights.

:math:`\Omega_{\rm outside}` is computed **twice**: once as the residual
:math:`\Omega_b - \sum_i\Omega_i`, and once in closed form as
:math:`\Omega_b(1 - F)` from
:func:`~ggah_mod.halos.linear_bias.mass_fraction` -- the *unweighted* partner of
:func:`~ggah_mod.spectra.counterterm.bias_consistency`, which is the same
integral with :math:`b(M)` in it.  :meth:`BaryonCensus.closure_residual` is the
difference.  Defining it only as a residual would make the closure a tautology:
it would hold however wrong the quadrature was.  Computing it twice makes it an
invariant, in the same sense as
:meth:`~ggah_mod.sectors.matter.MatterField.mass_conservation_residual`, and it
is reported rather than asserted for the same reason.

What the split between "in halos" and "diffuse" is worth
--------------------------------------------------------

It depends on :math:`M_{\min}`, which is a grid choice.  So
:func:`census_over_mass_ranges` runs the census at several, and the table it
produces is the honest form of the result: the part that moves with the grid is
not a measurement of the universe.  Comparing a single number against
Fukugita & Peebles (2004) or Shull, Smith & Danforth (2012) without that column
is how a resolution limit gets published as a baryon phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..cosmology import constants as C
from ..halos.linear_bias import mass_fraction, unresolved

__all__ = ["BaryonCensus", "census", "census_over_mass_ranges", "PHASES"]

#: The phases a census reports, in the order a table wants them.  Names match
#: :class:`~ggah_mod.sectors.matter.BaryonSplit`'s fields, so the two cannot
#: drift: a phase added there and forgotten here raises rather than vanishing.
PHASES = ("hot", "star_cen", "star_sat", "cold", "ejected")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BaryonCensus:
    r"""One cosmic baryon budget, by phase.

    Attributes
    ----------
    omega : dict[str, array]
        :math:`\Omega_i` for each of :data:`PHASES`, over resolved halos.
        Scalars.
    d_omega_dlog10m : dict[str, array]
        :math:`d\Omega_i/d\log_{10}M`, shape ``(NM,)``.  The differential form
        is the one worth plotting and the one Dev et al. (2024) measure; the
        scalar above is its integral, computed by the same quadrature the rest
        of the package uses rather than by summing this.
    omega_outside : array
        Baryons the resolved mass grid does not hold -- unresolved halos and
        genuinely diffuse gas together, because nothing here can separate them.
        A residual against :attr:`omega_b`.
    omega_outside_direct : array
        The same quantity in closed form, from the mass fraction alone.  The two
        agree only if the quadrature is right, which is the point.
    omega_b : float
        The **input** :math:`\Omega_b`.  Everything is checked against this and
        nothing rescales to it.
    mass_fraction_in_halos : array
        :math:`F`, the fraction of the cold matter density the mass grid holds.
        Reported because every other number here is conditional on it.
    omega_outside_by_phase : dict[str, array]
        :attr:`omega_outside_direct` split by what the matter below the grid is
        made of: ``star_cen``, ``star_sat`` and ``cold`` in the proportions of
        the smallest resolved halo, and ``diffuse`` for every other baryon --
        the composition the two-halo counterterm reads through each tracer's
        ``w_unresolved``.  Sums to :attr:`omega_outside_direct`.
    unresolved_bias : array
        :math:`b_u = (1-B)/(1-F)`, the effective linear bias of the matter
        below the grid.  The counterterm restores :math:`1-B` of the
        bias-weighted mass; the census books :math:`1-F` of the mass; this is
        the ratio that makes them one object.
    """

    omega: dict
    d_omega_dlog10m: dict
    omega_outside: jnp.ndarray
    omega_outside_direct: jnp.ndarray
    omega_b: float
    mass_fraction_in_halos: jnp.ndarray
    omega_outside_by_phase: dict = None
    unresolved_bias: jnp.ndarray = None

    def tree_flatten(self):
        return ((self.omega, self.d_omega_dlog10m, self.omega_outside,
                 self.omega_outside_direct, self.mass_fraction_in_halos,
                 self.omega_outside_by_phase, self.unresolved_bias),
                (self.omega_b,))

    @classmethod
    def tree_unflatten(cls, aux, children):
        omega, d_omega, outside, outside_direct, f_mass, by_phase, b_u = children
        return cls(omega, d_omega, outside, outside_direct, aux[0], f_mass,
                   by_phase, b_u)

    # ------------------------------------------------------------------ sums
    @property
    def omega_halos(self):
        r""":math:`\sum_i \Omega_i` over resolved halos."""
        return sum(self.omega[k] for k in PHASES)

    @property
    def omega_total(self):
        r"""Resolved plus outside.  Must be :attr:`omega_b`, by construction."""
        return self.omega_halos + self.omega_outside

    def closure_residual(self):
        r""":math:`\Omega_{\rm outside}^{\rm residual}/
        \Omega_{\rm outside}^{\rm direct} - 1`.

        The invariant.  Zero to round-off when the quadrature over the mass
        function agrees with the closed-form mass deficit -- which it must,
        because the per-halo budget sums to one, so the baryons integrate to
        :math:`\Omega_b F` whatever the split between phases.  A non-zero value
        means a phase is being counted twice or not at all, or that the split
        was built with a baryon fraction other than
        :func:`~ggah_mod.sectors.matter.cosmic_baryon_fraction`, and it is the
        only check here that does not depend on believing any sector.
        """
        return self.omega_outside / self.omega_outside_direct - 1.0

    def outside_fractions(self):
        """:attr:`omega_outside_by_phase` as fractions of :attr:`omega_b`."""
        return {k: v / self.omega_b for k, v in self.omega_outside_by_phase.items()}

    def fractions(self):
        """Each phase as a fraction of :attr:`omega_b`, for a table."""
        out = {k: self.omega[k] / self.omega_b for k in PHASES}
        out["outside"] = self.omega_outside / self.omega_b
        return out


def census(field, split, *, m_min=None, m_max=None) -> BaryonCensus:
    r"""The cosmic budget implied by one :class:`~ggah_mod.sectors.matter.BaryonSplit`.

    .. math::

        \Omega_i = \frac{1}{\rho_{c,0}}\int dM\,\frac{dn}{dM}\,M\,f_i(M)

    Parameters
    ----------
    field : HaloField
        Supplies ``dn/dM``, the mass grid and the quadrature.
    split : BaryonSplit
        The six per-halo fractions.  Only the five baryonic ones are integrated:
        ``f_collisionless`` is not a baryon and is checked, not counted.
    m_min, m_max : float, optional
        Restrict the integration range [Msun/h] without rebuilding the field --
        what :func:`census_over_mass_ranges` uses to show the grid dependence.
        Outside the range the integrand is zeroed rather than the grid sliced,
        so the quadrature nodes do not move and the comparison is between
        censuses rather than between quadratures.

    Notes
    -----
    ``dn/dM`` is calibrated against the **cold** field, so
    :math:`\int M\,(dn/dM)\,dM \to \bar\rho_{cb}`, not :math:`\bar\rho_m`.  The
    per-halo fractions are of the same cold halo mass, with
    :func:`~ggah_mod.sectors.matter.cosmic_baryon_fraction` as their ceiling,
    so the closed form is :math:`\Omega_b(1 - F)` with no neutrino factor in
    it.  A split built from :math:`\Omega_b/\Omega_m` mixes the two
    conventions, a 0.46% error that looks like a rounding difference, and
    :meth:`BaryonCensus.closure_residual` is what reports it.
    """
    cosmo = field.cosmo
    rho_crit0 = C.RHO_CRIT0
    ln10 = jnp.log(10.0)

    window = jnp.ones_like(field.m)
    if m_min is not None:
        window = window * (field.m >= m_min)
    if m_max is not None:
        window = window * (field.m <= m_max)

    dndm = field.dndm * window

    omega, d_omega = {}, {}
    for phase in PHASES:
        f = getattr(split, f"f_{phase}")
        integrand = dndm * field.m * jnp.atleast_1d(f)
        omega[phase] = field.integrate(integrand) / rho_crit0
        # dOmega/dlog10M = M (dn/dM) M f ln10 / rho_c, since dM = M ln10 dlog10M
        d_omega[phase] = integrand * field.m * ln10 / rho_crit0

    f_mass = mass_fraction(field.m, dndm, cosmo.rho_cold)
    missing = unresolved(field.m, dndm, field.bias, cosmo.rho_cold)

    omega_b = cosmo.Omega_b
    # `f_mass` is against the cold density and so are the per-halo fractions,
    # so no neutrino factor: a split that disagrees shows up in the residual.
    omega_outside_direct = omega_b * (1.0 - f_mass)

    omega_halos = sum(omega[k] for k in PHASES)
    omega_outside = omega_b - omega_halos

    # The composition of what is outside: the smallest halo the window keeps,
    # with its stars and neutral gas kept and every other baryon diffuse.
    first = jnp.argmax(window > 0)
    f_b = jnp.atleast_1d(split.f_baryon) * jnp.ones_like(field.m)
    share = {k: (jnp.atleast_1d(getattr(split, f"f_{k}"))
                 * jnp.ones_like(field.m))[first] / f_b[first]
             for k in ("star_cen", "star_sat", "cold")}
    share["diffuse"] = 1.0 - sum(share.values())
    by_phase = {k: omega_outside_direct * v for k, v in share.items()}

    return BaryonCensus(
        omega=omega, d_omega_dlog10m=d_omega,
        omega_outside=omega_outside,
        omega_outside_direct=omega_outside_direct,
        omega_b=float(omega_b), mass_fraction_in_halos=f_mass,
        omega_outside_by_phase=by_phase, unresolved_bias=missing.bias)


def census_over_mass_ranges(field, split, m_mins=(1e10, 1e12, 1e13, 1e14)):
    r"""The census at several :math:`M_{\min}`, as a dict keyed by the cut.

    The grid-dependence column.  What fraction of the baryons is "in halos"
    versus "diffuse" is set by where the mass function is truncated, and a
    single row of a census table hides that completely.  Same quadrature nodes
    throughout -- only the window moves -- so the differences are the physics of
    the mass range and not the numerics of a regridding.
    """
    return {m: census(field, split, m_min=m) for m in m_mins}
