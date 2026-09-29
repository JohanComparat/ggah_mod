r"""Layer 5 -- projections: :math:`w_p`, :math:`\Delta\Sigma`, :math:`C_\ell`,
:math:`w(\theta)`.

Layer 4 gives a 3D spectrum for any pair of tracers; this layer turns one into
something a survey measures.  Two kinds of projection, and one observation that
makes both cheap:

**Every node, weight and coefficient depends only on static grid choices.**
Ogata's abscissae are the zeros of :math:`J_\nu`; FFTLog's coefficients are
ratios of :math:`\Gamma` at complex argument.  None of them sees a traced value,
so all of them are built once in numpy and the traced path is an ``einsum`` or
an ``rfft``.  That is why there is no Bessel function on the differentiable
path, and why two independent engines cost little more than one.

**A spec, not a script.**  :class:`~ggah_mod.observables.spec.ObservableSpec` is
a list of wanted statistics, and :func:`~ggah_mod.observables.spec.make_model`
compiles it into one ``jax.jit``-able function of the parameters — resolving,
on the way, which :math:`P_{ab}(k, z)` are needed so each is computed exactly
once.  The spec is entirely static, which is what fixes the output shapes, which
is what makes ``jax.jacfwd`` of a data vector possible.

The dependency runs one way: this layer reads layers 1-4, and only layer 6
(:mod:`~ggah_mod.covariance`) reads it, to compile the spectra a covariance
needs through the same resolver a data vector uses.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

from .beams import gaussian, king
from .kernels import (
    RadialKernel, cmb_lensing, dispersion_measure, lensing_efficiency,
    limber_grid, mean_dispersion_measure, mean_electron_density,
    number_counts, thermal_sz,
)
from .limber import c_ell, limber_k
from .spec import (
    Cl, DeltaSigma, ObservableSpec, Plan, Statistic, TauKsz, WTheta, Wp,
    Xi,
    make_model,
)
from .real_space import (
    compensated_aperture, delta_sigma, delta_sigma_via_abel, sigma,
    sigma_electrons, tau_ksz, tau_ksz_cap, CAP_OUTER,
    w_theta, wp, xi,
)
from .transforms import (
    ENGINES, FFTLogRule, HankelRule, bessel_k, cl_to_wtheta, fftlog, hankel,
    make_fftlog, make_hankel, pk_to_delta_sigma, pk_to_sigma, pk_to_xi,
)

__all__ = [
    "HankelRule", "FFTLogRule", "make_hankel", "make_fftlog", "hankel",
    "fftlog", "ENGINES", "bessel_k",
    "pk_to_xi", "pk_to_sigma", "pk_to_delta_sigma", "cl_to_wtheta",
    # real space
    "xi", "wp", "sigma", "delta_sigma", "delta_sigma_via_abel", "w_theta",
    "compensated_aperture", "tau_ksz_cap", "CAP_OUTER",
    # the angular side
    "RadialKernel", "limber_grid", "number_counts", "lensing_efficiency",
    "cmb_lensing", "thermal_sz", "c_ell", "limber_k",
    "sigma_electrons", "tau_ksz",
    "dispersion_measure", "mean_dispersion_measure",
    "mean_electron_density",
    "gaussian", "king",
    # the declarative spec
    "Statistic", "Wp", "DeltaSigma", "Xi", "TauKsz", "Cl", "WTheta",
    "ObservableSpec", "Plan", "make_model",
]
