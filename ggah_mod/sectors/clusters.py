r"""Richness-selected galaxy clusters, built on a threshold galaxy sector.

A photometric cluster finder (redMaPPer, Rykoff et al. 2014) counts the red
galaxies of a halo above a luminosity limit, :math:`\lambda`.  This sector takes
that count from the galaxy model rather than parametrising a mass-richness
relation of its own: the mean richness of a halo is its red satellites above a
stellar-mass limit, times a calibration amplitude :math:`A_\lambda` (the
aperture, the membership probabilities and the luminosity-to-mass conversion),
plus the central:

.. math::

    \bar\lambda(M) = 1 + A_\lambda\,\langle N_{\rm sat}^{\rm red}
        (M_* > M_*^{\lambda})\rangle(M),

with :math:`\ln\lambda` normal about :math:`\ln\bar\lambda` with variance
:math:`\sigma_{\ln\lambda}^2 + (\bar\lambda - 1)/\bar\lambda^2`, an intrinsic
scatter plus the Poisson scatter of the count (the form of Costanzi et al.
2019 in its log-normal limit).  A richness bin is then a tracer with one object
per halo at its centre: :meth:`ClusterSector.weights` gives its
:class:`~ggah_mod.sectors.protocol.TracerWeights`, so the layer-4 spectra give
the stacked :math:`\Delta\Sigma` of the bin and its cross-correlation with
galaxies, and :meth:`ClusterSector.number_density` its abundance.

The galaxy parameters therefore reach the cluster observables through the red
satellite occupation, the same one the galaxy clustering fits; this is what
makes cluster abundances and lensing a test of the halo-quenching model at high
mass.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import ndtr

from .params import Flat, Param, SectorParams, sector_params

__all__ = ["ClusterParams", "ClusterSector"]


@sector_params
class ClusterParams(SectorParams):
    r"""The richness calibration and its intrinsic scatter.

    ``lambda_amp`` multiplies the red satellite count: 1 is a richness equal to
    the red satellites above the membership limit within :math:`r_\Delta`.
    ``sigma_lnlambda`` is the intrinsic scatter of :math:`\ln\lambda` at fixed
    mass, 0.2 for redMaPPer (McClintock et al. 2019 find 0.2-0.3 with
    projection effects).
    """

    lambda_amp: float = 1.0
    sigma_lnlambda: float = 0.2

    _PARAMS = {
        "lambda_amp": Param(
            1.0, (0.2, 5.0), Flat(), "",
            "richness per red satellite above the membership limit: the "
            "aperture, the membership probabilities and the luminosity cut; 1 "
            "counts every red satellite within r_Delta", "physical"),
        "sigma_lnlambda": Param(
            0.2, (0.01, 1.0), Flat(), "",
            "intrinsic scatter of ln(richness) at fixed halo mass, added to "
            "the Poisson scatter of the count", "physical"),
    }


class ClusterSector:
    r"""Clusters selected on richness, on a red threshold galaxy sector.

    Parameters
    ----------
    galaxies : GalaxySector
        A threshold sector whose satellites are the red galaxies a cluster
        finder counts (``zumandelbaum16_red``).
    lg_mstar_member_min : float
        The stellar mass of the membership limit, physical
        :math:`\log_{10}(M_*/M_\odot)`; 10.2 is close to redMaPPer's
        :math:`0.2L_*` at :math:`z\sim0.2`.
    """

    def __init__(self, galaxies, lg_mstar_member_min: float = 10.2):
        if not getattr(galaxies, "is_threshold", False):
            raise ValueError("ClusterSector needs a threshold galaxy sector: the "
                             "richness is its satellite occupation above a "
                             "stellar-mass limit")
        self.galaxies = galaxies
        self.lg_mstar_member_min = float(lg_mstar_member_min)

    def _native(self, h):
        from .galaxies import THRESHOLD_MASS_UNITS
        from .sham import stellar_mass_to_msun
        units = THRESHOLD_MASS_UNITS[self.galaxies.model]
        return self.lg_mstar_member_min - stellar_mass_to_msun(0.0, units, h)

    def mean_richness(self, field, galaxy_params, p: ClusterParams = ClusterParams()):
        r""":math:`\bar\lambda(M)` on the field's mass grid."""
        gp = dict(galaxy_params if isinstance(galaxy_params, dict) else galaxy_params.as_dict())
        _, n_sat = self.galaxies.occupation(
            field, dict(gp, log10m_star_thresh=self._native(field.cosmo.h)))
        return 1.0 + p.lambda_amp * n_sat

    def sigma_lnlambda(self, field, galaxy_params, p: ClusterParams = ClusterParams()):
        r"""The total scatter of :math:`\ln\lambda` at fixed mass."""
        lam = self.mean_richness(field, galaxy_params, p)
        return jnp.sqrt(p.sigma_lnlambda ** 2 + (lam - 1.0) / lam ** 2)

    def p_bin(self, field, galaxy_params, lam_lo, lam_hi, p: ClusterParams = ClusterParams()):
        r""":math:`P(\lambda_{\rm lo} \le \lambda < \lambda_{\rm hi}\,|\,M)` on the mass grid."""
        lam = self.mean_richness(field, galaxy_params, p)
        s = self.sigma_lnlambda(field, galaxy_params, p)
        mu = jnp.log(lam)
        return ndtr((jnp.log(lam_hi) - mu) / s) - ndtr((jnp.log(lam_lo) - mu) / s)

    def number_density(self, field, galaxy_params, lam_lo, lam_hi,
                       p: ClusterParams = ClusterParams()):
        r"""The comoving number density of clusters in the bin [(Mpc/h)^-3]."""
        return field.integrate(field.dndm * self.p_bin(field, galaxy_params, lam_lo, lam_hi, p))

    def weights(self, field, galaxy_params, lam_lo, lam_hi, p: ClusterParams = ClusterParams()):
        """The :class:`TracerWeights` of the clusters of one richness bin."""
        from .protocol import TracerWeights
        w = self.p_bin(field, galaxy_params, lam_lo, lam_hi, p)
        return TracerWeights(w_point=w, w_extended=None, self_pair=w,
                             norm=field.number_density(w), discrete=True,
                             bias_weight=None, name=f"clusters/{lam_lo:g}-{lam_hi:g}")
