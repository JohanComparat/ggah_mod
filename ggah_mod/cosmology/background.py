r"""Background expansion: :math:`E(z)`, distances, volumes.

w0waCDM, flat or curved, with a **complete** density budget -- photons, the
cold sector, three massive neutrinos with their relativistic transition, the
massless remainder of :math:`N_{\rm eff}`, curvature and CPL dark energy:

.. math::

    E^2(z) = \Omega_\gamma (1+z)^4
           + \Omega_{cb}(1+z)^3
           + \rho_\nu(z)/\rho_{c,0}
           + \Omega_k(1+z)^2
           + \Omega_{\rm DE}\,f_{\rm DE}(z)

with :math:`\Omega_{cb} = \Omega_m - \Omega_\nu^{\rm nr}` because
:class:`~ggah_mod.cosmology.parameters.Cosmology` puts the neutrinos' rest mass
inside :math:`\Omega_m`, and

.. math::

    f_{\rm DE}(z) = (1+z)^{3(1+w_0+w_a)}\exp\!\left(\frac{-3w_a z}{1+z}\right).

The neutrino term is the exact energy integral of each massive state over its
relic spectrum, :func:`~ggah_mod.cosmology.parameters.nu_energy_factor`, on the amplitude
the three would carry were they massless, plus the massless remainder:

.. math::

    \rho_\nu(z)/\rho_{c,0} = \left[\Omega_\nu^{\rm massive,rel}\,
    \tfrac13\textstyle\sum_j F(y_j/(1+z)) + \Omega_{\rm ur}\right](1+z)^4 .

Both asymptotes are exact: :math:`\rho_\nu \propto (1+z)^3` when cold,
:math:`(1+z)^4` when relativistic -- **and the massless case is the massless
answer** rather than zero.

Everything is pure JAX and differentiable with respect to every cosmological
parameter, including ``sum_mnu``, ``w0`` and ``wa``.

Accuracy
--------
:math:`E(0) = 1` identically, because
:attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_de` closes against
:attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_nu_today` -- the density
this module sums -- rather than against the mass budget.  Against CLASS the
neutrino density agrees to the quadrature's 1e-12 wherever CLASS's own momentum
sampling is converged, since the convention and the integral are the same.
Against ``astropy`` it differs by astropy's convention, which puts the heating
of the neutrinos into their degeneracy at :math:`(4/11)^{1/3}T_{\rm CMB}`
rather than into their temperature: 0.47 per cent in the rest mass.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as _np

from . import constants as C
from .parameters import Cosmology, nu_energy_factor_mean

__all__ = ["transverse_distance", 
    "nu_density_shape", "hubble_e", "comoving_distance", "sound_horizon",
    "comoving_distance_z1z2", "angular_diameter_distance",
    "luminosity_distance", "comoving_volume_element", "distance_modulus",
]

# 256-point Gauss-Legendre nodes on [0, 1], built once with numpy so the jaxpr
# carries them as constants.
_GL_N = 256
_x, _w = _np.polynomial.legendre.leggauss(_GL_N)
_GL_X = jnp.asarray(0.5 * (_x + 1.0))
_GL_W = jnp.asarray(0.5 * _w)


def nu_density_shape(z, cosmo: Cosmology):
    r""":math:`\tfrac13\sum_j F(y_j/(1+z))` -- the massive states' energy, **relativistic-normalised**.

    :math:`\rho_{\nu,{\rm massive}}(z)/\rho^{\rm rel}_{\nu,{\rm massive}}(z)`,
    so that multiplying by :attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_nu_massive_rel`
    :math:`(1+z)^4` gives their density.  It tends to 1 at high redshift for any
    mass, and to :math:`\kappa y_0/(1+z)` once the states are cold, which is the
    :math:`(1+z)^{-1}` that turns :math:`(1+z)^4` into :math:`(1+z)^3`.

    **It reads** :attr:`~ggah_mod.cosmology.parameters.Cosmology.nu_y` **rather
    than rebuilding it.**  The same float operations in the same order give the
    same float, which is what makes :math:`E(0)=1` an *identity* rather than a
    tolerance; behind a root-find a second copy would be a second solver.  There
    is one expression.

    The species axis is reduced **here** and does not reach the return value:
    :func:`comoving_distance` hands :func:`hubble_e` a ``z`` of shape
    ``(Nz, 256)``, and a trailing species axis on the way out would arrive at
    that function's ``jnp.sqrt``.
    """
    one_z = 1.0 + jnp.asarray(z)
    return nu_energy_factor_mean(cosmo.nu_y / one_z[..., None])


@jax.jit
def hubble_e(z, cosmo: Cosmology):
    r""":math:`E(z) = H(z)/H_0`.

    Summed as :math:`E^2 = 1 + \sum_i \Omega_i\,[a_i(z) - 1]`, each density
    times its growth *less one*, which is the standard sum rearranged by the
    closure :math:`\sum_i\Omega_i = 1` of
    :attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_de`.  At :math:`z = 0`
    every bracket is ``0.0`` exactly -- ``1.0 ** n - 1.0``, and the neutrino
    density less the same float the closure used -- so :math:`E(0) = 1` is an
    identity of the arithmetic and not of how the rounding happens to fall.
    Summed the plain way it held at the fiducial by luck: adding the massless
    remainder in 0.9.8 moved it to :math:`1 - 1.1\times10^{-16}`.

    At :math:`\Omega_k = 0` the curvature bracket is ``+ 0.0``, which is exact
    in floating point, so a flat cosmology never sees the curvature term.
    """
    one_z = 1.0 + jnp.asarray(z)
    f_de = one_z ** (3.0 * (1.0 + cosmo.w0 + cosmo.wa)) * jnp.exp(
        -3.0 * cosmo.wa * (one_z - 1.0) / one_z)
    # The neutrinos, in the order `Cosmology.Omega_nu_today` sums them, so that
    # at z = 0 this is that property to the last bit and its bracket is zero.
    rho_nu = (cosmo.Omega_nu_massive_rel * one_z ** 4
              * nu_density_shape(one_z - 1.0, cosmo)
              + cosmo.Omega_ur * one_z ** 4)
    e2 = (1.0
          + cosmo.Omega_gamma * (one_z ** 4 - 1.0)
          + cosmo.Omega_cb * (one_z ** 3 - 1.0)
          + (rho_nu - cosmo.Omega_nu_today)
          + cosmo.Omega_k * (one_z ** 2 - 1.0)
          + cosmo.Omega_de * (f_de - 1.0))
    return jnp.sqrt(e2)


@jax.jit
def comoving_distance(z, cosmo: Cosmology):
    r""":math:`\chi(z) = (c/H_0)\int_0^z dz'/E(z')` **[Mpc/h]**.

    Gauss-Legendre on :math:`[0, z]`, vectorised over ``z``.
    """
    z = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    nodes = z[:, None] * _GL_X[None, :]
    integrand = 1.0 / hubble_e(nodes, cosmo)
    integral = z * jnp.sum(integrand * _GL_W[None, :], axis=1)
    return cosmo.hubble_distance * integral


#: Upper limit of :func:`sound_horizon`'s quadrature.  Above it matter is less
#: than 1e-6 of the radiation and the baryon loading :math:`R` is below 1e-7,
#: so the rest of the integral is added in closed form, as pure radiation.
Z_RS_MAX = 1.0e10


@jax.jit
def sound_horizon(z, cosmo: Cosmology):
    r"""Comoving sound horizon :math:`r_s(z)` of the photon-baryon fluid **[Mpc/h]**.

    .. math::

        r_s(z) = \int_z^\infty \frac{c_s(z')\,dz'}{H(z')}, \qquad
        c_s = \frac{c}{\sqrt{3(1+R)}}, \qquad
        R = \frac{3\rho_b}{4\rho_\gamma} = \frac34\,\frac{\Omega_b}{\Omega_\gamma}\,\frac{1}{1+z},

    the distance a sound wave travels before :math:`z` -- Eisenstein & Hu
    (1998) Eq. 6, first equality, with their Eq. 5 for :math:`R`.  At the drag epoch it is the BAO scale,
    :func:`~ggah_mod.cosmology.drag.r_drag`.

    **Over this module's** :math:`E(z)` **and nothing else**, so the neutrinos
    are the exact relic integral at CLASS's temperature, the massless remainder
    of :math:`N_{\rm eff}` is in, and :math:`T_{\rm CMB}` moves :math:`\Omega_\gamma`
    in :math:`R` and in :math:`E` together.  Given the same drag redshift it
    reproduces CLASS's ``rs_d`` to 1e-7 at the fiducial and to 2.2e-6 over the
    flat cosmologies of the ``emu_pk`` training box.

    Gauss-Legendre in :math:`u = \ln(1+z')` on :math:`[\ln(1+z), \ln(1+Z_{\rm RS\_MAX})]`
    -- in :math:`u` the integrand is :math:`(1+z')/E`, which tends to a
    constant over radiation, so 256 nodes span sixteen decades of redshift --
    plus the radiation tail above :data:`Z_RS_MAX`,
    :math:`D_H/\sqrt{3\Omega_r}(1+Z_{\rm RS\_MAX})` with every neutrino
    relativistic.  The tail is 1.9e-7 of :math:`r_s(z_d)`; truncating at
    :math:`10^6` instead would cost 2e-3.

    **No curvature factor.**  CLASS multiplies :math:`dr_s` by
    :math:`\sqrt{1-Kr_s^2}` (``background.c``, marked "TBC"); CAMB does not, and
    neither does this function: the sound horizon is a comoving length, and
    curvature enters where it is turned into an angle, in
    :func:`transverse_distance`.  The two conventions differ by
    :math:`\mp1.8\times10^{-5}` at :math:`\Omega_k = \pm0.1`.
    """
    z = jnp.atleast_1d(jnp.asarray(z, dtype=float))
    u0 = jnp.log1p(z)
    u1 = _np.log1p(Z_RS_MAX)
    u = u0[:, None] + (u1 - u0)[:, None] * _GL_X[None, :]
    one_z = jnp.exp(u)
    r = 0.75 * cosmo.Omega_b / cosmo.Omega_gamma / one_z
    integrand = one_z / (hubble_e(one_z - 1.0, cosmo) * jnp.sqrt(3.0 * (1.0 + r)))
    integral = (u1 - u0) * jnp.sum(integrand * _GL_W[None, :], axis=1)
    tail = 1.0 / (jnp.sqrt(3.0 * (cosmo.Omega_gamma + cosmo.Omega_nu_rel))
                  * (1.0 + Z_RS_MAX))
    return cosmo.hubble_distance * (integral + tail)



def _sinhc(x):
    r""":math:`s(x) = \sinh(\sqrt x)/\sqrt x`, extended to :math:`x \le 0`.

    The single function all three geometries need.  It is **entire** --
    :math:`s(x) = \sum_n x^n/(2n+1)!` -- so open, flat and closed are one
    analytic expression rather than three branches, and :math:`x=0` is an
    interior point rather than a special case.  For :math:`x<0` the series is
    :math:`\sin(\sqrt{-x})/\sqrt{-x}`, which is the closed geometry.

    Evaluated as the series for small :math:`|x|` and in closed form beyond,
    with **both branches guarded**: ``where`` evaluates both, and a ``nan`` in
    the unused one still poisons the gradient of the used one.  That is the same
    trap :func:`~ggah_mod.cosmology.amplitude.tophat_window` documents and
    ``bessel_k`` was once caught by.
    """
    x = jnp.asarray(x)
    small = jnp.abs(x) < 1e-3
    # Six terms take the series to ~1e-22 at |x| = 1e-3, far below float64.
    series = (1.0 + x / 6.0 + x ** 2 / 120.0 + x ** 3 / 5040.0
              + x ** 4 / 362880.0)
    big = jnp.where(small, 1.0, x)              # keep the unused branch finite
    root = jnp.sqrt(jnp.abs(big))
    exact = jnp.where(big > 0.0,
                      jnp.sinh(root) / root,
                      jnp.sin(root) / root)
    return jnp.where(small, series, exact)


@jax.jit
def transverse_distance(chi, cosmo: Cosmology):
    r"""Transverse comoving distance :math:`f_K(\chi)` [Mpc/h].

    .. math::

        f_K(\chi) = \frac{D_H}{\sqrt{\Omega_k}}
            \sinh\!\left(\sqrt{\Omega_k}\,\frac{\chi}{D_H}\right)
        \;=\; \chi\,s\!\left(\Omega_k\,(\chi/D_H)^2\right)

    with :math:`s` of :func:`_sinhc` and :math:`D_H = c/H_0`.  The second form
    is the one implemented, because it makes the flat case an *interior* point
    rather than a limit: at :math:`\Omega_k = 0` the argument is exactly zero,
    :math:`s(0) = 1` exactly, and :math:`f_K = \chi` **bit for bit**.

    **This is the distance that appears in an angle**, and the one
    :math:`\chi(z_2)-\chi(z_1)` was standing in for.  The radial comoving
    distance :func:`comoving_distance` is unchanged by curvature -- it is an
    integral of :math:`dz/E` and :math:`E` carries the curvature term already.
    What curvature changes is how a transverse separation maps onto an angle,
    which is why every consumer here is an angular one: :math:`D_A`,
    :math:`D_L`, the volume element, and layer 5's lensing efficiency.

    ``PLAN.md`` item **E2**, and the reason it was one item rather than four:
    :attr:`~ggah_mod.cosmology.parameters.Cosmology.Omega_de`'s closure,
    :func:`hubble_e`'s :math:`\Omega_k(1+z)^2`, this function, and
    :func:`~ggah_mod.observables.kernels.lensing_efficiency` are each inert on
    their own.  A package with three of the four would report a curvature
    constraint that was partly flat.
    """
    chi = jnp.asarray(chi)
    x = cosmo.Omega_k * (chi / cosmo.hubble_distance) ** 2
    return chi * _sinhc(x)


@jax.jit
def comoving_distance_z1z2(z1, z2, cosmo: Cosmology):
    r"""Transverse comoving distance between two redshifts, :math:`f_K(\Delta\chi)`
    [Mpc/h].

    The radial separation is a difference of integrals; the *transverse* one --
    which is what an angle between two objects at different redshifts sees, and
    what a lensing kernel needs -- is that difference put through
    :func:`transverse_distance`.  Equal to :math:`\chi_2-\chi_1` exactly when
    :math:`\Omega_k = 0`, which is what this function used to return
    unconditionally, with "flat geometry only" as its whole docstring.
    """
    d_chi = comoving_distance(z2, cosmo) - comoving_distance(z1, cosmo)
    return transverse_distance(d_chi, cosmo)


@jax.jit
def angular_diameter_distance(z, cosmo: Cosmology):
    r""":math:`D_A = f_K(\chi)/(1+z)` [Mpc/h].

    The transverse distance, not the radial one: :math:`D_A` is defined by an
    angle.  They coincide when :math:`\Omega_k = 0`.
    """
    return transverse_distance(comoving_distance(z, cosmo), cosmo) / (
        1.0 + jnp.asarray(z))


@jax.jit
def luminosity_distance(z, cosmo: Cosmology):
    r""":math:`D_L = (1+z)f_K(\chi)` [Mpc/h].

    Duality with :func:`angular_diameter_distance`, so it takes the same
    transverse distance.
    """
    return transverse_distance(comoving_distance(z, cosmo), cosmo) * (
        1.0 + jnp.asarray(z))


@jax.jit
def comoving_volume_element(z, cosmo: Cosmology):
    r""":math:`dV_c/(dz\,d\Omega) = (c/H_0)f_K(\chi)^2/E` [(Mpc/h)^3/sr].

    The transverse distance squared: a solid angle subtends an area, and
    that is the distance an area is measured with.
    """
    chi = transverse_distance(comoving_distance(z, cosmo), cosmo)
    return cosmo.hubble_distance * chi ** 2 / hubble_e(z, cosmo)


def distance_modulus(z, cosmo: Cosmology):
    r""":math:`\mu = 5\log_{10}(D_L/10\,{\rm pc})`.

    Uses :math:`D_L` in **Mpc**, not Mpc/h -- a distance modulus is an
    observable and carries no :math:`h`.
    """
    d_l_mpc = luminosity_distance(z, cosmo) / cosmo.h
    return 5.0 * jnp.log10(d_l_mpc * 1.0e5)
