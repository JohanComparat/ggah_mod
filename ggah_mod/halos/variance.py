r"""The mass variance :math:`\sigma(M, z)`, built from the spectrum directly.

This module is where the package refuses the usual shortcut.

The standard construction is :math:`\sigma(M,z) = \sigma(M,0)\,D(z)`, with the
growth factor from an ODE or a fitting formula.  That factorisation is exact
only while growth is scale-independent, and massive neutrinos break it: free
streaming suppresses small scales more than large ones, so
:math:`\sqrt{P(k,z)/P(k,0)}` genuinely depends on :math:`k`.

**How much this actually costs, measured**, since the honest answer is smaller
than the argument suggests.  The top-hat window integrates over a broad range
of :math:`k`, which averages most of the scale-dependence away -- by a factor
of about 40 at the minimal mass:

===========  ===  ==========================  =========================
Sigma m_nu   z    spread in :math:`P(k,z)`    spread in :math:`\sigma(M,z)`
===========  ===  ==========================  =========================
0.00 eV      1    1.4e-4                      3.4e-6
0.06 eV      1    1.3e-3                      3.1e-5
0.06 eV      3    3.3e-3                      5.6e-5
0.30 eV      1    6.5e-3                      1.4e-3
0.30 eV      3    1.5e-2                      2.7e-3
===========  ===  ==========================  =========================

(spread = the mass-dependence of :math:`\sigma(M,z)/\sigma(M,0)` over
:math:`10^{10}` to :math:`10^{16}\,M_\odot/h`.)

So at :math:`\Sigma m_\nu = 0.06` eV a separable :math:`\sigma(M,0)D(z)`
would be wrong by only :math:`3\times10^{-5}`, and the case for doing it
properly is not that the error is large -- it is that the error is *unbounded
in advance*: it grows to :math:`10^{-3}` by 0.3 eV, and nothing in a separable
implementation tells you which regime you are in.

:func:`sigma_of_mass` therefore evaluates :math:`P_{cb}(k, z)` **at the
redshift wanted** and integrates it.  There is no growth factor in the halo
sector at all.  The cost is one spectrum evaluation per redshift, which against
a **batched** Boltzmann solve is milliseconds -- so the assumption is removed
for free, which is the actual argument.

The word "batched" is load-bearing, and this file used to say "cached", which
is a different claim and a false one.  A solve is memoised on the whole
redshift *tuple* (:class:`~ggah_mod.cosmology.power._BoltzmannBase`), so a
redshift asked for on its own is a solve of its own: twelve of them cost 93.7 s
against 6.0 s for one call that names all twelve.  What makes the argument
above true is :func:`~ggah_mod.halos.field.make_fields`, which asks once.

**The cold field, not the total.**  A halo's Lagrangian radius is set by the
matter that actually collapses into it, and neutrinos do not.  So the radius
uses :math:`\bar\rho_{cb}` and the spectrum is :math:`P_{cb}`.  Pairing
:math:`P_{cb}` with :math:`\bar\rho_m`, or the reverse, is a percent-level error
that no test catches unless it is looking for it -- which is why
:class:`~ggah_mod.cosmology.parameters.Cosmology` names the two densities
separately and this module takes the one it needs by name.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..cosmology.amplitude import sigma2_tophat, sigma_tophat

__all__ = ["lagrangian_radius", "mass_from_radius", "sigma_of_mass",
           "dln_sigma_dln_mass", "check_k_support", "DELTA_C",
           "K_MAX_R_MIN", "K_MIN_R_MAX"]

#: Spherical-collapse critical overdensity.  Weakly cosmology-dependent
#: (Nakamura & Suto 1997 give a ~1% variation); the constant is what every
#: mass-function and bias calibration in the literature assumes, so using the
#: exact value would make those fits inconsistent with themselves.
#:
#: **Curvature does not change that argument, it strengthens it.**  Spatial
#: curvature moves the collapse threshold by about the same ~1% the other
#: parameters do, and the fits downstream were all calibrated flat at 1.686 --
#: so tracking it here would put this constant and those fits in two different
#: universes, which is worse than the error it removes.  What curvature *does*
#: reach is :math:`\sigma(M)`, through :math:`P_{cb}(k)`, and that is where a
#: curved cosmology is supposed to enter a peak height.
DELTA_C = 1.686


def lagrangian_radius(m, rho_cold):
    r""":math:`R(M) = (3M/4\pi\bar\rho_{cb})^{1/3}` [Mpc/h].

    Pass ``cosmo.rho_cold``.  The comoving radius of the sphere that contained
    ``m`` of *cold* matter before collapse.
    """
    return (3.0 * jnp.asarray(m) / (4.0 * jnp.pi * rho_cold)) ** (1.0 / 3.0)


def mass_from_radius(r, rho_cold):
    r"""Inverse of :func:`lagrangian_radius` [Msun/h]."""
    return (4.0 / 3.0) * jnp.pi * jnp.asarray(r) ** 3 * rho_cold


def sigma_of_mass(m, k, pk_cb, rho_cold):
    r""":math:`\sigma(M)` for one spectrum.

    Parameters
    ----------
    m : array [Msun/h]
    k : array [h/Mpc]
    pk_cb : array
        The **cold** spectrum on ``k``, at the redshift wanted.
    rho_cold : float
        ``cosmo.rho_cold``.
    """
    return sigma_tophat(pk_cb, k, lagrangian_radius(m, rho_cold))


def dln_sigma_dln_mass(m, k, pk_cb, rho_cold):
    r""":math:`d\ln\sigma/d\ln M`, by automatic differentiation.

    Not a finite difference.  :math:`dn/dM` is proportional to this, and a
    finite-difference derivative of an integral is the classic way to put
    percent-level noise into a mass function -- it is a difference of two nearly
    equal quadratures, so it loses precisely the digits the answer needs.

    Since :math:`R \propto M^{1/3}`,
    :math:`d\ln\sigma/d\ln M = \tfrac13\,d\ln\sigma/d\ln R`, and the second
    factor is differentiated exactly.
    """
    k = jnp.asarray(k)
    pk_cb = jnp.asarray(pk_cb)

    def ln_sigma_of_ln_r(ln_r):
        return 0.5 * jnp.log(sigma2_tophat(pk_cb, k, jnp.exp(ln_r)))

    ln_r = jnp.log(lagrangian_radius(m, rho_cold))
    d_ln_sigma_d_ln_r = jax.vmap(jax.grad(ln_sigma_of_ln_r))(jnp.atleast_1d(ln_r))
    out = d_ln_sigma_d_ln_r / 3.0
    return out[0] if jnp.ndim(m) == 0 else out


# =========================================================================
# Does the k grid actually support the masses asked for?
# =========================================================================

#: ``k_max * R`` needed at the *smallest* mass before :func:`sigma_of_mass` has
#: converged.  Measured -- ``measure_accuracy.py::k_support`` in the paper
#: repository -- on **one shared grid**, which is how ``make_field`` uses it:
#: there is a single ``k`` for every mass, so the binding constraint at the top
#: is the smallest halo and at the bottom the largest.  Worst fractional error
#: over :math:`10^{10}\dots10^{16}\,M_\odot/h` at ``PLANCK18`` with CLASS:
#:
#: =============  ==========  =============  ==========
#: ``k_max R``    error       ``k_min R``    error
#: =============  ==========  =============  ==========
#: 2              4.8e-2      1.0            8.6e-2
#: 5              1.1e-3      0.5            1.2e-2
#: 10             6.1e-5      0.1            3.7e-5
#: 20             5.0e-6      0.02           7.3e-7
#: =============  ==========  =============  ==========
#:
#: The thresholds below are the third row of each column: past them the error
#: is far under any tolerance in this package, and short of them it climbs by
#: three orders of magnitude within a factor of five.  They are not punitive --
#: both shipped flavours clear them by more than six times (``k_max R = 61``,
#: ``k_min R = 3.0e-3``), so a grid that trips this guard was narrowed
#: deliberately and the guard is telling its author what it cost.
K_MAX_R_MIN = 10.0

#: ``k_min * R`` allowed at the *largest* mass; see :data:`K_MAX_R_MIN` for the
#: measurement.  The low-:math:`k` end is the one that surprises: a top-hat of
#: radius :math:`R` has :math:`W \simeq 1` for every :math:`k < 1/R`, so
#: truncating there does not taper the integrand, it deletes a finite piece
#: of it.
K_MIN_R_MAX = 0.1


def _concrete(x):
    """``x`` as a numpy array, or ``None`` if it is a tracer."""
    try:
        return np.asarray(jax.device_get(x), dtype=float)
    except Exception:                       # TracerArrayConversionError, ...
        return None


def check_k_support(m, k, rho_cold, what: str = "make_field"):
    r"""Refuse a ``k`` grid too narrow for the masses it will be integrated for.

    :func:`sigma_of_mass` integrates :math:`k^2 P(k) W^2(kR)` over exactly the
    grid it is handed.  Narrow that grid and the integral does not fail -- it
    quietly returns a smaller number, and every rung above it (the mass
    function, the bias, the concentration) inherits the error with no symptom.
    At the extreme it returns zeros: a grid that misses the mass entirely gives
    :math:`\sigma = 0` and :math:`\nu = \infty`.

    ``m`` and ``k`` are static choices, not traced quantities, so this is a
    Python-level check like the backend guards in
    :func:`~ggah_mod.halos.field.make_field`.  If either *is* traced the check
    is skipped rather than forcing a concretisation -- the grids are then the
    caller's own, and so is the responsibility.

    Raises
    ------
    ValueError
        Naming the offending end of the grid, the mass that lost support, and
        the value the grid would need.
    """
    m_np, k_np, rho_np = _concrete(m), _concrete(k), _concrete(rho_cold)
    if (m_np is None or k_np is None or rho_np is None
            or m_np.size == 0 or k_np.size == 0):
        return
    # `rho_cold` is traced whenever the *cosmology* is differentiated, which is
    # most of what this package is for -- so it is a third thing that has to be
    # concrete before the check can run, not an incidental float.
    r = np.asarray(lagrangian_radius(m_np, float(rho_np)), dtype=float)
    r_min, r_max = float(r.min()), float(r.max())
    k_min, k_max = float(k_np.min()), float(k_np.max())
    m_small, m_large = float(m_np.min()), float(m_np.max())

    if k_max * r_min < K_MAX_R_MIN:
        raise ValueError(
            f"{what}: the k grid stops at k_max = {k_max:.4g} h/Mpc, which is "
            f"k_max*R = {k_max * r_min:.3g} at the smallest mass "
            f"{m_small:.3g} Msun/h (R = {r_min:.4g} Mpc/h).  sigma(M) is then "
            f"missing power inside its own window -- the integral does not "
            f"fail, it under-counts, silently, and dn/dM, b and c inherit it. "
            f"The error is 6e-5 at the threshold and 5e-2 by k_max*R = 2. Need "
            f"k_max >= {K_MAX_R_MIN / r_min:.4g} h/Mpc for this mass range, or "
            f"raise m_min.")

    if k_min * r_max > K_MIN_R_MAX:
        raise ValueError(
            f"{what}: the k grid starts at k_min = {k_min:.4g} h/Mpc, which is "
            f"k_min*R = {k_min * r_max:.3g} at the largest mass "
            f"{m_large:.3g} Msun/h (R = {r_max:.4g} Mpc/h).  The top-hat window "
            f"is ~1 for every k below 1/R, so this does not taper the "
            f"integrand -- it deletes a finite piece of it.  The error is 4e-5 "
            f"at the threshold and 9e-2 by k_min*R = 1.  Need "
            f"k_min <= {K_MIN_R_MAX / r_max:.4g} h/Mpc for this mass range, or "
            f"lower m_max.")
