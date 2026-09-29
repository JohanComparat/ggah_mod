r"""Radial kernels: what turns a 3D field into something on the sky.

.. math::

    C_\ell^{ab} = \int d\chi\,
        \frac{W_a(\chi)\,W_b(\chi)}{\chi^2}\,
        P_{ab}\Big(k = \frac{\ell+\tfrac12}{\chi},\, z(\chi)\Big)

**Lensing lives here, not in layer 4.**  Convergence, shear and CMB lensing are
not 3D fields: they are the *matter* field under different :math:`W(\chi)`.
Giving them names in the tracer registry would make :math:`P_{\kappa\kappa}` and
:math:`P_{mm}` two arrays that can drift apart, which is what happened in the
predecessor -- its :math:`C_\ell^{\kappa y}` used a total-matter NFW on the
lensing side and a DPM pressure profile on the :math:`y` side, so one halo had
two gas contents depending on which spectrum was being computed.

One lensing efficiency, and one clip that is not a clip
-------------------------------------------------------

The predecessor has **two** lensing kernels that quadrature :math:`g(\chi)`
differently -- one integrating in :math:`z`, one in :math:`\chi` with an
explicit :math:`dz/d\chi` Jacobian -- in two files, for one quantity.  There is
one here.

Both of them write the lensing geometry as
``jnp.clip((chi_s - chi_l)/chi_s, 0, None)``.  That is the idiom
:mod:`ggah_mod.numerics` documents as splitting a gradient 50/50 at its tie, and
the tie here is :math:`\chi_s = \chi_l` -- the source plane, which every
source-redshift distribution passes through by construction, not at an edge
case.  ``jnp.where`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..cosmology import background
from ..cosmology import constants as C
from ..cosmology.background import transverse_distance

__all__ = ["RadialKernel", "limber_grid", "number_counts",
           "lensing_efficiency", "cmb_lensing", "thermal_sz", "Z_STAR",
           "dispersion_measure", "mean_dispersion_measure",
           "mean_electron_density", "PC_CM"]

#: Redshift of last scattering, for the CMB lensing kernel.  Planck 2018.
Z_STAR = 1089.92


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RadialKernel:
    r"""One tracer's :math:`W(\chi)`, on a shared :math:`(z, \chi)` grid.

    Attributes
    ----------
    z, chi : array (Nz,)
        Redshift and comoving distance [Mpc/h].
    w : array (Nz,)
        The kernel.  Units are whatever makes :math:`\int W_aW_b\,d\chi/\chi^2`
        dimensionless for the pair -- a number-counts kernel is
        :math:`(\mathrm{Mpc}/h)^{-1}`, a lensing one likewise.
    name : str
        Static.
    """

    z: jnp.ndarray
    chi: jnp.ndarray
    w: jnp.ndarray
    name: str = "kernel"

    def tree_flatten(self):
        return ((self.z, self.chi, self.w), (self.name,))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def limber_grid(cosmo, z_max: float = 3.0, n: int | None = None,
                backend=None, z_min: float = 1e-3):
    r"""One :math:`(z, \chi)` grid for **every** statistic in a spec.

    Not one per statistic.  If :math:`C_\ell^{\kappa\kappa}` and
    :math:`C_\ell^{\kappa y}` were integrated on different redshift grids their
    cross-covariance would be inconsistent, and nothing would say so.  It is
    also the minimal-work answer: :math:`P(k,z)` is stacked once.

    ``z_min`` is not zero: :math:`\chi \to 0` makes :math:`k = (\ell+1/2)/\chi`
    diverge and :math:`1/\chi^2` with it, and the integrand's limit is finite
    only because :math:`W(\chi) \to 0` at least as fast.  Starting a hair above
    zero is the honest way to say that, rather than dividing by zero and
    repairing it afterwards.
    """
    from ..backend import resolve_backend

    b = resolve_backend(backend)
    n = int(b.n_z_proj if n is None else n)
    z = jnp.linspace(z_min, z_max, n)
    return z, background.comoving_distance(z, cosmo)


def _normalised(w, chi):
    r"""``w`` scaled so :math:`\int w\,d\chi = 1`."""
    return w / jnp.trapezoid(w, chi)


def number_counts(z, chi, nz, name: str = "counts") -> RadialKernel:
    r""":math:`W(\chi) = dN/d\chi`, normalised on the :math:`\chi` grid.

    ``nz`` is :math:`dN/dz`; the Jacobian is applied here rather than expected
    of the caller, and the normalisation is over :math:`\chi` because that is
    the variable the Limber integral runs in.  Normalising over :math:`z` and
    integrating over :math:`\chi` is a factor of :math:`d z/d\chi` that no shape
    comparison reveals.
    """
    nz = jnp.asarray(nz)
    dz_dchi = jnp.gradient(jnp.asarray(z)) / jnp.gradient(jnp.asarray(chi))
    return RadialKernel(z, chi, _normalised(nz * dz_dchi, chi), name)


def lensing_efficiency(z, chi, nz_source, cosmo,
                       name: str = "lensing") -> RadialKernel:
    r""":math:`W_\kappa(\chi) = \frac{3\Omega_m}{2}\Big(\frac{H_0}{c}\Big)^2
    \frac{\chi}{a}\,g(\chi)`, with

    .. math::  g(\chi) = \int_\chi^\infty d\chi'\,n(\chi')\,
                        \frac{f_K(\chi'-\chi)}{f_K(\chi')}

    The distances are **transverse** ones -- lensing maps an angle -- and the
    leading :math:`\chi` is :math:`f_K(\chi)` for the same reason.  They are the
    radial distance exactly when :math:`\Omega_k = 0`, which is what this
    assumed before ``PLAN.md`` item **E2**.

    :math:`\Omega_m` is the **total** matter density, neutrinos included,
    because that is what lenses -- the same reason
    :class:`~ggah_mod.sectors.matter.MatterField` normalises on ``rho_matter``.
    """
    z, chi = jnp.asarray(z), jnp.asarray(chi)
    n_chi = _normalised(jnp.asarray(nz_source)
                        * jnp.gradient(z) / jnp.gradient(chi), chi)

    # (Nl, Ns): the source is behind the lens, or it does not lens it.  `where`,
    # not `clip`: the tie is chi_s = chi_l, which every n(z) passes through.
    #
    # `f_K` and not the bare difference.  The lensing kernel is built from
    # *transverse* comoving distances, because what a deflection does is map an
    # angle -- and the two coincide only when Omega_k = 0, which is what this
    # line assumed before PLAN.md item E2.  With curvature it is
    # f_K(chi_s - chi_l)/f_K(chi_s), and at Omega_k = 0 `transverse_distance`
    # returns its argument bit for bit, so nothing flat moved.
    d_ls = transverse_distance(chi[None, :] - chi[:, None], cosmo)
    d_s = transverse_distance(chi, cosmo)
    ratio = d_ls / d_s[None, :]
    geometry = jnp.where(chi[None, :] > chi[:, None], ratio, 0.0)
    g = jnp.trapezoid(n_chi[None, :] * geometry, chi, axis=-1)

    amp = 1.5 * cosmo.Omega_m / cosmo.hubble_distance ** 2
    return RadialKernel(z, chi, amp * transverse_distance(chi, cosmo)
                        * (1.0 + z) * g, name)


def cmb_lensing(z, chi, cosmo, z_star: float = Z_STAR,
                name: str = "cmb_lensing") -> RadialKernel:
    r"""The same, with a single source plane at :math:`z_\star`."""
    z, chi = jnp.asarray(z), jnp.asarray(chi)
    chi_star = background.comoving_distance(z_star, cosmo)
    geometry = jnp.where(chi_star > chi, (chi_star - chi) / chi_star, 0.0)
    amp = 1.5 * cosmo.Omega_m / cosmo.hubble_distance ** 2
    return RadialKernel(z, chi, amp * chi * (1.0 + z) * geometry, name)


def thermal_sz(z, chi, name: str = "tsz") -> RadialKernel:
    r"""The Compton-:math:`y` window: **unity**.

    The :math:`\sigma_T/m_ec^2` and the proper-length conversion are already in
    :meth:`~ggah_mod.sectors.gas.HotGasDPM.y_amplitude`, which returns
    :math:`(\mathrm{Mpc}/h)^2` -- so :math:`P_{yy}` is already the spectrum of
    the projected field and the kernel adds nothing.  Written out, with the
    reason, because a window of 1 is otherwise indistinguishable from a window
    somebody forgot.
    """
    z = jnp.asarray(z)
    return RadialKernel(z, jnp.asarray(chi), jnp.ones_like(z), name)


# =========================================================================
# The free electrons: dispersion measure
# =========================================================================

#: One parsec in cm.  A dispersion measure is quoted in pc cm^-3 and nothing
#: else, so the conversion lives here rather than at every call site.
PC_CM = C.MPC_CM / 1e6


def mean_dispersion_measure(z, cosmo, f_e=1.0, n=512):
    r"""The Macquart relation: :math:`\langle{\rm DM}\rangle(z)` [pc cm^-3].

    .. math::

        \langle{\rm DM}\rangle(z) = \frac{c}{H_0}\int_0^z
            \frac{\bar n_{e,0}\,(1+z')}{E(z')}\,dz'

    with :math:`\bar n_{e,0}` the **comoving** mean free-electron density.

    The :math:`(1+z)` is one power, not three and not two, and the arithmetic is
    worth writing down because every wrong answer here is a plausible one.  The
    proper electron density goes as :math:`(1+z)^3`; the proper path length
    contributes :math:`(1+z)^{-1}`; and the dispersion measure is *observed*,
    so the frequency-squared scaling redshifts it by a further
    :math:`(1+z)^{-1}`.  Three minus two is one.  Dropping the last factor is
    the common slip and it is a 100% error at :math:`z = 1`.

    Parameters
    ----------
    z : float or array
        Source redshift.  Evaluated on its own quadrature grid per element, so
        an array of sources costs one pass.
    cosmo : Cosmology
    f_e : float or array
        The fraction of :math:`\Omega_b` that is **diffuse and ionised**.
        Not a free parameter and not a fudge:
        :func:`~ggah_mod.sectors.census.census` computes it, as one minus the
        stellar and neutral shares.  That is the whole reason a dispersion
        measure belongs in this thread -- it is the census's own number,
        multiplied by geometry, compared against a measurement nobody used to
        build it.
    n : int
        Quadrature nodes in z.

    Returns
    -------
    array
        [pc cm^-3].

    Notes
    -----
    Returns the **mean**.  A real sightline scatters around it by of order the
    mean itself, because most of the path is void and the tail is set by how
    many haloes were crossed -- which is a property of the electron
    *distribution*, i.e. of :func:`dispersion_measure`'s kernel crossed with the
    ejecta and gas sectors, and not of this integral.
    """
    z = jnp.atleast_1d(jnp.asarray(z))
    # Comoving mean electron density [cm^-3], from the *input* Omega_b.
    n_e0 = mean_electron_density(cosmo, f_e)
    # c/H0 in cm: (c/100h) Mpc -> cm.
    c_over_h0_cm = C.C_KM_S / (100.0 * cosmo.h) * C.MPC_CM

    # One grid per source, scaled: z' = z * t with t in [0, 1].
    t = jnp.linspace(0.0, 1.0, int(n))
    zz = z[:, None] * t[None, :]
    integrand = (1.0 + zz) / background.hubble_e(zz, cosmo)
    integral = jnp.trapezoid(integrand, zz, axis=-1)
    return n_e0 * c_over_h0_cm * integral / PC_CM


def mean_electron_density(cosmo, f_e=1.0):
    r"""Comoving mean free-electron density :math:`\bar n_{e,0}` [cm^-3].

    .. math::  \bar n_{e,0} = \frac{f_e\,\Omega_b\,\rho_{c,0}}{\mu_e m_p}

    :math:`\mu_e = 1.14` is the mean molecular weight per electron for a fully
    ionised primordial plasma, and it is imported from
    :mod:`~ggah_mod.sectors.gas` rather than restated: the hot gas sector
    already owns that constant, and two copies of it would be free to disagree
    about the helium abundance.

    The :math:`h` bookkeeping, once: :data:`~ggah_mod.cosmology.constants.RHO_CRIT0`
    is in :math:`M_\odot h^{-1}({\rm Mpc}/h)^{-3}`, so a physical density needs
    :math:`h^2`.
    """
    from ..sectors.gas import M_PROTON_G, M_SUN_G, MU_E

    rho_b_cgs = (f_e * cosmo.Omega_b * C.RHO_CRIT0 * cosmo.h ** 2
                 * M_SUN_G / C.MPC_CM ** 3)
    return rho_b_cgs / (MU_E * M_PROTON_G)


def dispersion_measure(z, chi, cosmo, f_e=1.0, name="dm") -> RadialKernel:
    r"""The DM window, for :math:`C_\ell^{\rm DM\,\times\,anything}`.

    .. math::  W_{\rm DM}(\chi) = \bar n_{e,0}\,(1+z)\,\frac{1}{a_{\rm pc}}

    i.e. the integrand of :func:`mean_dispersion_measure` re-expressed against
    :math:`d\chi` rather than :math:`dz`, so that
    :math:`\int W\,d\chi = \langle{\rm DM}\rangle`.  Crossed with the
    ``electrons`` tracer of :mod:`~ggah_mod.spectra.tracers` it gives the
    *fluctuating* part -- the DM--galaxy cross-correlation that FLIMFLAM and
    its successors measure, and the piece that localises the missing baryons
    rather than counting them.

    **Not normalised.**  :func:`number_counts` divides by its own integral
    because a redshift distribution is a probability density; this kernel's
    amplitude is the physical electron density, and normalising it away would
    discard exactly the quantity the census predicts.
    """
    z = jnp.asarray(z)
    n_e0 = mean_electron_density(cosmo, f_e)
    # dchi is comoving Mpc/h; the electron density is comoving cm^-3, so the
    # length conversion carries h as well as Mpc -> pc.
    per_mpc_h = C.MPC_CM / cosmo.h / PC_CM
    w = n_e0 * (1.0 + z) * per_mpc_h
    return RadialKernel(z, jnp.asarray(chi), w, name)
