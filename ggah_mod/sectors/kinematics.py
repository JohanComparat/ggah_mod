r"""Satellite kinematics: the velocity dispersion of satellites around centrals.

The observable of More et al. (2011): stack the satellites found around central
galaxies of one stellar-mass bin and measure the dispersion of their
line-of-sight velocities relative to the central.  Since every central of the
bin sits in a halo drawn from :math:`P(M|M_*^{\rm c})`, the dispersion is an
average over that distribution, and two averages are measured
(More et al. 2011, Sect. 3):

.. math::

    \sigma_{\rm sw}^2 = \frac{\int \dd M\, P(M|M_*^{\rm c})\,
        \langle N_{\rm sat}\rangle(M)\, \sigma_v^2(M)}
        {\int \dd M\, P(M|M_*^{\rm c})\, \langle N_{\rm sat}\rangle(M)},
    \qquad
    \sigma_{\rm hw}^2 = \int \dd M\, P(M|M_*^{\rm c})\, \sigma_v^2(M),

satellite-weighted (every satellite counts once) and host-weighted (every
central counts once).  Their ratio measures the scatter of halo mass at fixed
central stellar mass; together they constrain the stellar-to-halo relation and
its scatter without lensing.

:math:`P(M|M_*^{\rm c})` is the galaxy sector's central term:
:math:`\dd n/\dd M` times the central conditional stellar-mass function
integrated over the bin.  :math:`\langle N_{\rm sat}\rangle(M)` is its satellite
occupation above the spectroscopic limit of the satellites.

The velocity dispersion of a halo is that of satellites on isotropic orbits in
an NFW potential, averaged over an NFW number density within :math:`r_\Delta`,
times a velocity bias :math:`b_v` (:class:`KinematicsParams`):

.. math::

    \sigma_v^2(M) = b_v^2\, K(c)\, V_\Delta^2, \qquad
    K(c) = c\,g(c)^2 \int_0^c x^2\, I(x)\,\dd x, \qquad
    I(x) = \int_x^\infty \frac{m(y)}{y^3(1+y)^2}\,\dd y,

with :math:`V_\Delta^2 = GM/r_\Delta` (:attr:`HaloField.v_delta_squared`),
:math:`m(y) = \ln(1+y) - y/(1+y)` and :math:`g(c) = 1/m(c)`.  This is the
isotropic Jeans equation for an NFW tracer in its own potential (Łokas &
Mamon 2001, Eq. 14 is the closed form of :math:`\sigma_r^2`); for isotropic
orbits the population-averaged line-of-sight dispersion equals the mean of
:math:`\sigma_r^2`.  :math:`I` is universal in :math:`x = cr/r_\Delta`, so it is
tabulated once, and :math:`K` agrees with a direct quadrature to :math:`10^{-6}`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .params import Flat, Param, SectorParams, sector_params

__all__ = ["KinematicsParams", "dispersion_factor", "halo_velocity_dispersion",
           "satellite_velocity_dispersion"]


@sector_params
class KinematicsParams(SectorParams):
    r"""The velocity bias of satellites.

    ``b_v`` multiplies the dark matter's velocity dispersion: 1 is satellites
    that move as the dark matter does at the same radius.  Simulations put
    subhalo velocity bias within 0.9-1.1 (e.g. Wu et al. 2013); dynamical
    friction and tidal stripping push it both ways, so the box is broad.
    """

    b_v: float = 1.0

    _PARAMS = {
        "b_v": Param(
            1.0, (0.5, 1.5), Flat(), "",
            "velocity bias: the satellites' velocity dispersion over the dark "
            "matter's, at the same radius; 1 is unbiased", "physical"),
    }


def _m(y):
    return np.log1p(y) - y / (1.0 + y)


def _tables(n=20001, x_min=1e-5, x_max=1e5):
    r"""``(ln x, ln J(x))`` with :math:`J(x) = \int_0^x u^2 I(u)\,\dd u`."""
    x = np.logspace(np.log10(x_min), np.log10(x_max), n)
    f = _m(x) / (x ** 3 * (1.0 + x) ** 2)
    # I(x): cumulative from the top; the tail beyond x_max is
    # \int_X^\infty (ln y - 1) y^-5 dy, closed form, negligible past 1e5.
    lx = np.log(x)
    seg = 0.5 * (f[1:] * x[1:] + f[:-1] * x[:-1]) * np.diff(lx)
    X = x[-1]
    tail = (4.0 * np.log(X) - 3.0) / (16.0 * X ** 4)
    I = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]]) + tail
    g = x ** 2 * I
    segj = 0.5 * (g[1:] * x[1:] + g[:-1] * x[:-1]) * np.diff(lx)
    # J below x_min: I(u) ~ ln(1/u)/2 for u -> 0, so u^2 I is negligible there
    J = np.concatenate([[0.0], np.cumsum(segj)]) + x[0] ** 3 * (np.log(1 / x[0]) / 6.0 + 1 / 18.0)
    return lx, np.log(J), lx, np.log(I)


_LNX, _LNJ, _LNX_I, _LNI = _tables()


def dispersion_factor(c):
    r""":math:`K(c) = \langle\sigma_r^2\rangle / V_\Delta^2` of isotropic NFW tracers in an NFW
    halo of concentration ``c``, averaged within :math:`r_\Delta`."""
    c = jnp.asarray(c)
    J = jnp.exp(jnp.interp(jnp.log(c), jnp.asarray(_LNX), jnp.asarray(_LNJ)))
    m = jnp.log1p(c) - c / (1.0 + c)
    return c * J / m ** 2


def halo_velocity_dispersion(field, p: KinematicsParams = KinematicsParams()):
    r""":math:`\sigma_v(M) = b_v\sqrt{K(c)\,V_\Delta^2}` in km/s, on the field's mass grid."""
    return p.b_v * jnp.sqrt(dispersion_factor(field.conc) * field.v_delta_squared)


def satellite_velocity_dispersion(field, galaxies, galaxy_params, lg_mstar_edges, *,
                                  lg_mstar_sat_min, p: KinematicsParams = KinematicsParams(),
                                  weighting: str = "satellite", n_nodes: int = 9):
    r"""The satellite- or host-weighted dispersion [km/s] per central stellar-mass bin.

    ``lg_mstar_edges`` are the central bins, in physical
    :math:`\log_{10}(M_*/M_\odot)`; ``lg_mstar_sat_min`` the satellites'
    spectroscopic limit, in the same units.  ``galaxies`` is a threshold
    :class:`~ggah_mod.sectors.galaxies.GalaxySector`.  Returns shape
    ``(len(lg_mstar_edges) - 1,)``.
    """
    from .galaxies import THRESHOLD_MASS_UNITS
    from .sham import stellar_mass_to_msun

    if weighting not in ("satellite", "host"):
        raise ValueError(f"weighting is 'satellite' or 'host', not {weighting!r}")
    e = jnp.asarray(lg_mstar_edges, float)
    nodes = jnp.stack([jnp.linspace(e[i], e[i + 1], n_nodes) for i in range(e.size - 1)])
    _, dn = galaxies.central_csmf_msun(field, galaxy_params, lg_ms_msun=nodes.ravel())
    dn = dn.reshape(nodes.shape + (dn.shape[-1],))                     # (nb, nn, NM)
    p_cen = jnp.trapezoid(dn, nodes[:, :, None], axis=1)              # (nb, NM)
    sig2 = halo_velocity_dispersion(field, p) ** 2
    weight = field.dndm[None, :] * p_cen
    if weighting == "satellite":
        units = THRESHOLD_MASS_UNITS[galaxies.model]
        native = lg_mstar_sat_min - stellar_mass_to_msun(0.0, units, field.cosmo.h)
        gp = dict(galaxy_params if isinstance(galaxy_params, dict) else galaxy_params.as_dict())
        _, n_sat = galaxies.occupation(field, dict(gp, log10m_star_thresh=native))
        weight = weight * n_sat[None, :]
    return jnp.sqrt(field.integrate(weight * sig2[None, :], axis=-1)
                    / field.integrate(weight, axis=-1))
