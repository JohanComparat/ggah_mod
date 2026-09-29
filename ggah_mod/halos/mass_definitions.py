r"""Spherical-overdensity mass definitions, and conversion between them.

A halo mass is meaningless without saying what boundary it is measured to.
:math:`M_{200m}`, :math:`M_{200c}` and :math:`M_{\rm vir}` are three different
numbers for the same halo, differing by tens of percent, and a mass function or
concentration relation calibrated on one does not describe another.

A :class:`MassDef` is that statement: an overdensity :math:`\Delta` and the
density it is measured against.

Conventions for a mass definition
---------------------------------

:meth:`MassDef.delta_rho` returns a **comoving** reference density in
:math:`(M_\odot/h)/({\rm Mpc}/h)^3`, so

.. math::  r_\Delta = \left(\frac{3M}{4\pi\,\Delta\,\rho_{\rm ref}}\right)^{1/3}

is a comoving radius in Mpc/h, matching every other length in the package.

**"Mean" means total matter.**  The reference density for a ``200m`` definition
is :attr:`~ggah_mod.cosmology.parameters.Cosmology.rho_matter`, not
``rho_cold``: the halo boundary is defined against the mean density of the
universe, which neutrinos contribute to.  The *cold* density belongs to the
Lagrangian radius in :mod:`ggah_mod.halos.variance`, which is about the matter
that collapses.  These are different questions and the package answers them
with different densities.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..cosmology import constants as C
from ..cosmology.background import hubble_e
from .profiles import g_nfw, nfw_scale_density

__all__ = ["MassDef", "parse_mass_def", "rho_reference", "delta_vir",
           "translate_mass", "nfw_params_from_mass", "VIRIAL"]


@jax.jit
def delta_vir(z, cosmo):
    r"""Virial overdensity w.r.t. the **critical** density (Bryan & Norman 1998).

    .. math::  \Delta_{\rm vir} = 18\pi^2 + 82x - 39x^2,\quad
               x = \Omega_m(z) - 1

    with :math:`\Omega_m(z) = \Omega_m(1+z)^3/E^2(z)`.  About 100 at
    :math:`z=0` for Planck parameters, tending to :math:`18\pi^2 \approx 178`
    at high redshift where the universe is matter dominated.

    One definition.  The predecessor wrote this expression in three files, once
    with ``np.pi`` instead of ``jnp.pi``.

    **This is the flat coefficient pair, and it is the layer's one restriction
    under curvature.**  Bryan & Norman fit two: 82 and -39 for
    :math:`\Omega_m + \Omega_\Lambda = 1`, and 60 and -32 for
    :math:`\Omega_m + \Omega_k = 1` with :math:`\Lambda = 0`.  Only the first
    is here, and the :math:`\Omega_m(z)` it is evaluated at comes from
    :func:`~ggah_mod.cosmology.background.hubble_e`, which carries
    :math:`\Omega_k(1+z)^2` since ``PLAN.md`` item **E2**.  So at
    :math:`\Omega_k \neq 0` the argument is curved while the fit is flat, and
    the second pair is no remedy -- a universe with both curvature and dark
    energy is neither of the two models fitted.

    Nothing is branched here: this function is ``@jax.jit`` and a refusal
    inside a trace is an exception inside a trace.  The guard is
    :func:`~ggah_mod.halos.calibration.check_cosmology_support`, called from
    :func:`~ggah_mod.halos.field.make_field`, which refuses ``mdef='vir'`` at
    nonzero curvature and names ``200m`` and ``200c`` -- both exact there,
    because ``hubble_e`` is the only route curvature takes into them.  A caller
    reaching :class:`MassDef` directly is outside that guard, which is the same
    standing :data:`~ggah_mod.halos.variance.DELTA_C` has.
    """
    z = jnp.asarray(z)
    om_z = cosmo.Omega_m * (1.0 + z) ** 3 / hubble_e(z, cosmo) ** 2
    x = om_z - 1.0
    return 18.0 * jnp.pi ** 2 + 82.0 * x - 39.0 * x ** 2


class MassDef:
    """An overdensity and the density it is measured against.

    Parameters
    ----------
    delta : float or ``"vir"``
        Overdensity.  ``"vir"`` selects the Bryan & Norman (1998) value, which
        is redshift dependent and always relative to the critical density.
    rho_type : {"matter", "critical"}
        Ignored when ``delta == "vir"``.

    Examples
    --------
    ``MassDef.from_string("200m")``, ``"200c"``, ``"500c"``, ``"2500c"``,
    ``"vir"``.  Arbitrary :math:`\\Delta` is supported -- the predecessor had a
    general class *and* a narrow three-case function, with the profile and
    lensing paths routed through the narrow one, so ``500c`` was unreachable
    from them despite being implemented.
    """

    __slots__ = ("delta", "rho_type")

    def __init__(self, delta, rho_type: str = "matter"):
        if isinstance(delta, str):
            if delta.lower() not in ("vir", "virial"):
                raise ValueError(f"delta must be a number or 'vir', got {delta!r}")
            self.delta = "vir"
            self.rho_type = "critical"      # virial is always w.r.t. critical
        else:
            if float(delta) <= 0.0:
                raise ValueError(f"delta must be positive, got {delta}")
            if rho_type not in ("matter", "critical"):
                raise ValueError(
                    f"rho_type must be 'matter' or 'critical', got {rho_type!r}")
            self.delta = float(delta)
            self.rho_type = rho_type

    @classmethod
    def from_string(cls, name: str) -> "MassDef":
        """Parse ``"200m"``, ``"500c"``, ``"vir"``, ..."""
        s = str(name).strip().lower()
        if s in ("vir", "virial"):
            return cls("vir")
        if s.endswith("m"):
            return cls(float(s[:-1]), "matter")
        if s.endswith("c"):
            return cls(float(s[:-1]), "critical")
        raise ValueError(
            f"cannot parse mass definition {name!r}; expected e.g. '200m', "
            f"'200c', '500c', '2500c' or 'vir'")

    def __repr__(self) -> str:
        if self.delta == "vir":
            return "MassDef('vir')"
        return f"MassDef({self.delta:g}, {self.rho_type!r})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, MassDef) and self.delta == other.delta
                and self.rho_type == other.rho_type)

    def __hash__(self) -> int:
        return hash((self.delta, self.rho_type))

    def delta_rho(self, z, cosmo):
        r"""``(delta, rho_ref)`` with ``rho_ref`` **comoving** [(Msun/h)/(Mpc/h)^3].

        For a matter definition the comoving reference density is redshift
        independent in h-units; for a critical definition it carries
        :math:`E^2(z)/(1+z)^3`.
        """
        z = jnp.asarray(z)
        if self.delta == "vir":
            delta = delta_vir(z, cosmo)
            rho_ref = C.RHO_CRIT0 * hubble_e(z, cosmo) ** 2 / (1.0 + z) ** 3
            return delta, rho_ref
        if self.rho_type == "matter":
            # Total matter: the halo boundary is set against the mean density
            # of the universe, which neutrinos contribute to.
            return self.delta, cosmo.rho_matter
        return self.delta, C.RHO_CRIT0 * hubble_e(z, cosmo) ** 2 / (1.0 + z) ** 3

    def delta_mean(self, z, cosmo):
        r"""The same boundary, expressed against the **mean matter** density.

        .. math::  \Delta_{\rm m} = \Delta_{\rm ref}\,
                   \bar\rho_{\rm ref} / \bar\rho_{\rm m}

        Published :math:`\Delta`-dependent multiplicity functions are indexed
        this way -- Tinker et al. (2008) tabulate
        :math:`\Delta_{\rm m} \in [200, 3200]` -- so this is the number a fit
        has to be handed, and it is *not* the number in the definition's name:
        :math:`200{\rm c}` is :math:`\Delta_{\rm m} = 645` at :math:`z=0` and
        :math:`\Delta_{\rm vir}` is :math:`331`.  Passing 200 for both is a
        30 per cent error in :math:`\dd n/\dd M` that produces a perfectly
        smooth mass function, which is why it went unnoticed until the halo
        field's own ``mdef`` was compared against what its abundance was
        actually computed at.
        """
        delta, rho_ref = self.delta_rho(z, cosmo)
        return delta * rho_ref / cosmo.rho_matter

    def radius(self, m, z, cosmo):
        r""":math:`r_\Delta` [Mpc/h], comoving."""
        delta, rho_ref = self.delta_rho(z, cosmo)
        return (3.0 * jnp.asarray(m) / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0)

    def mass(self, r, z, cosmo):
        r"""Inverse of :meth:`radius`: :math:`M_\Delta` [Msun/h] from ``r``."""
        delta, rho_ref = self.delta_rho(z, cosmo)
        return (4.0 / 3.0) * jnp.pi * delta * rho_ref * jnp.asarray(r) ** 3


def parse_mass_def(mdef) -> MassDef:
    """Accept a :class:`MassDef` or a string."""
    return mdef if isinstance(mdef, MassDef) else MassDef.from_string(mdef)


#: The virial definition as an *object*, so no other module has to spell it.
#:
#: ``tests/test_one_mass_definition.py`` audits every module for a mass
#: definition written as a string literal, on the rule that this module is the
#: one that knows what the names mean.  A caller asking "is this the virial
#: boundary?" -- :func:`~ggah_mod.halos.calibration.check_cosmology_support`
#: does, because :func:`delta_vir` is the flat fit -- needs to ask it without
#: naming it, and :class:`MassDef` already compares by value.
VIRIAL = MassDef("vir")


def rho_reference(mdef, z, cosmo):
    """``(delta, rho_ref)`` for ``mdef``."""
    return parse_mass_def(mdef).delta_rho(z, cosmo)


def _dg_nfw(y):
    r""":math:`g'(y) = y/(1+y)^2`."""
    y = jnp.asarray(y)
    return y / (1.0 + y) ** 2


@jax.jit
def _solve_radius_ratio(c_in, ratio, n_steps: int = 60):
    r"""Solve :math:`g(cx)/g(c) = {\rm ratio}\,x^3` for :math:`x = r_{\rm out}/r_{\rm in}`.

    :math:`F(x) = g(cx)/g(c) - {\rm ratio}\,x^3` decreases monotonically, so a
    fixed 60-step bisection on :math:`x \in [10^{-3}, 20]` brackets the root
    with no data-dependent control flow.

    **The Newton polish is load-bearing, not cosmetic.**  A bisection depends on
    its inputs only through the sign of :math:`F` at each midpoint, and a
    boolean carries no derivative: ``jax.grad`` through a bare bisection returns
    **exactly zero, silently**.  The predecessor recorded
    :math:`dM_{500c}/d\Omega_m` coming back as 0 against a finite difference of
    :math:`6.5\times10^{13}`.

    Taking one Newton step from a ``stop_gradient``-ed bracket leaves the value
    unchanged to machine precision -- :math:`F(x_0)` is already ~1e-16 -- while
    giving the derivative the implicit-function result
    :math:`dx/d\theta = -(\partial_\theta F)/(\partial_x F)`.
    """
    gc = g_nfw(c_in)

    def F(x):
        return g_nfw(c_in * x) / gc - ratio * x ** 3

    def dF(x):
        return c_in * _dg_nfw(c_in * x) / gc - 3.0 * ratio * x ** 2

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go_right = F(mid) > 0.0            # F decreasing: root lies right
        return jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid)

    shape = jnp.broadcast_shapes(jnp.shape(c_in), jnp.shape(ratio))
    lo = jnp.full(shape, 1e-3)
    hi = jnp.full(shape, 20.0)
    lo, hi = jax.lax.fori_loop(0, n_steps, body, (lo, hi))
    x0 = jax.lax.stop_gradient(0.5 * (lo + hi))
    return x0 - F(x0) / dF(x0)


def translate_mass(m_in, c_in, mdef_in, mdef_out, z, cosmo):
    r"""Convert a halo mass between spherical-overdensity definitions.

    Assumes the halo is NFW with concentration ``c_in`` defined against
    ``mdef_in``, and finds the radius enclosing the target overdensity.

    Returns
    -------
    (m_out, r_out, c_out)
        [Msun/h], comoving [Mpc/h], dimensionless.

    Notes
    -----
    Identical definitions short-circuit, so a round trip is *exact* rather than
    accurate to a tolerance.
    """
    din = parse_mass_def(mdef_in)
    dout = parse_mass_def(mdef_out)
    m_in = jnp.asarray(m_in)
    c_in = jnp.broadcast_to(jnp.asarray(c_in), jnp.shape(m_in))
    r_in = din.radius(m_in, z, cosmo)
    if din == dout:
        return m_in, r_in, c_in

    delta_i, rho_i = din.delta_rho(z, cosmo)
    delta_o, rho_o = dout.delta_rho(z, cosmo)
    ratio = (delta_o * rho_o) / (delta_i * rho_i)
    x = _solve_radius_ratio(c_in, ratio)
    return m_in * g_nfw(c_in * x) / g_nfw(c_in), r_in * x, c_in * x


def nfw_params_from_mass(m, c, z, cosmo, mdef="200c"):
    r""":math:`(\rho_s, r_s, r_\Delta)` for an NFW halo of mass ``m``.

    The bridge from "a mass and a concentration" to "a profile".  Comoving
    radii [Mpc/h]; :math:`\rho_s` in :math:`M_\odot h^2/{\rm Mpc}^3`.
    """
    r_delta = parse_mass_def(mdef).radius(m, z, cosmo)
    rho_s, r_s = nfw_scale_density(m, c, r_delta)
    return rho_s, r_s, r_delta
