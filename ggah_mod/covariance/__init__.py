r"""Layer 6 -- the covariance of a data vector.

Layer 5 turns spectra into what a survey measures; this layer says how those
measurements scatter.  It reads layers 1 to 5 and nothing reads it.

A covariance here is analytic: products of the same spectra the model is made
of, divided by the modes a survey holds, so it is noise-free, invertible at any
length and consistent with the model by construction.  That is what a resampled
covariance cannot be when the data vector is longer than the number of
resamplings -- the LS10 joint vectors are 286 points against 100 jackknife
regions, rank 99 -- and it is the reason this layer exists.

A complete covariance has a Gaussian term, a connected non-Gaussian term, a
super-sample term and noise (Krause & Eifler 2017; Friedrich et al. 2021;
Reischke et al. 2025).  Here they are named terms of :func:`covariance` --
``gaussian``, ``shared_pair``, ``one_halo``, ``one_halo_shared``,
``multi_halo`` and ``super_sample`` -- for :math:`C_\ell`, :math:`w_p` and
:math:`\Delta\Sigma`, and every result lists the ones it lacks.

Modules
-------

:mod:`~ggah_mod.covariance.geometry`
    Footprints (the extragalactic sky :math:`|b| > 20^\circ`, the LS10 samples,
    any HEALPix mask), slab volumes, mask power and the projected super-sample
    variance.
:mod:`~ggah_mod.covariance.gaussian`
    Mode counting, bin-averaged Bessel kernels, the :math:`C_\ell` and
    :math:`w_p` Gaussian covariances.
:mod:`~ggah_mod.covariance.lensing`
    Comoving :math:`\Sigma_{\rm crit}`, source samples and shape noise, the
    :math:`\Delta\Sigma` and :math:`w_p\times\Delta\Sigma` Gaussian covariances
    (Singh et al. 2017).
:mod:`~ggah_mod.covariance.trispectrum`
    The four-point rule, the one-halo connected covariance and its
    shared-object terms.
:mod:`~ggah_mod.covariance.multihalo`
    The two-, three- and four-halo trispectrum with tree-level PT.
:mod:`~ggah_mod.covariance.supersample`
    The halo-model response to a long mode and the slab's variance.
:mod:`~ggah_mod.covariance.spec`
    :func:`covariance`: an ``ObservableSpec`` in, ``params -> Cov`` out.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

from .geometry import (
    DEG2_PER_SR, FULL_SKY_DEG2, LS10_SAMPLES, SurveyGeometry, from_area,
    from_healpix_mask, from_randoms, galactic_latitude_cut, ls10_sample, projected_sigma2_b,
    spherical_cap,
)
from .gaussian import (
    annulus_j0, annulus_j2, annulus_overlap, band_modes, cl_gaussian,
    make_wp_kernel, wp_gaussian, wp_noise_term,
)
from .lensing import (
    LensingQuadrature, SourceSample, delta_sigma_gaussian,
    effective_sigma_crit, make_lensing_quadrature, sigma_crit_comoving,
    wp_delta_sigma_gaussian,
)
from .spec import (
    NOT_IMPLEMENTED, TERMS, Covariance, covariance, gaussian_covariance,
)
from .multihalo import multi_halo_trispectrum, pt_kernels
from .supersample import response, slab_sigma2_b
from .trispectrum import (
    RealKernel, contract_cng, interpolate_factor, one_halo_factors,
    one_halo_trispectrum, real_kernel, shared_object_cng,
)

__all__ = [
    "SurveyGeometry", "galactic_latitude_cut", "spherical_cap", "from_area",
    "from_healpix_mask", "from_randoms", "ls10_sample", "LS10_SAMPLES", "projected_sigma2_b",
    "DEG2_PER_SR", "FULL_SKY_DEG2",
    "band_modes", "cl_gaussian", "annulus_j0", "annulus_j2", "make_wp_kernel",
    "wp_gaussian", "wp_noise_term", "annulus_overlap",
    "sigma_crit_comoving", "effective_sigma_crit", "SourceSample",
    "LensingQuadrature", "make_lensing_quadrature", "delta_sigma_gaussian",
    "wp_delta_sigma_gaussian",
    "one_halo_factors", "one_halo_trispectrum", "interpolate_factor",
    "contract_cng", "RealKernel", "real_kernel", "shared_object_cng",
    "response", "slab_sigma2_b", "multi_halo_trispectrum", "pt_kernels",
    "Covariance", "covariance", "gaussian_covariance", "TERMS",
    "NOT_IMPLEMENTED",
]
