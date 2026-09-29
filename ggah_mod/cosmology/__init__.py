"""Layer 1 -- the cosmology.

Everything here depends only on the cosmological parameters: the background
expansion, the linear power spectrum, the growth of structure, and the derived
amplitude diagnostics.  Nothing in this subpackage knows about halos, galaxies,
gas or AGN.

The two paths through layer 1
-----------------------------

Everything in this subpackage is pure JAX and traceable **except** the
solver-backed pieces, which are named here so the boundary is not something to
be discovered:

===================================  ===========================================
Accurate path only                   why
===================================  ===========================================
``power.ClassPk``                    CLASS is a C Boltzmann solver
``power.CambPk``                     CAMB is Fortran
``amplitude.ln10A_s_for_sigma8``     a one-time Newton solve, run before a fit
===================================  ===========================================

Everything else -- :mod:`~ggah_mod.cosmology.background`,
:mod:`~ggah_mod.cosmology.amplitude`, :mod:`~ggah_mod.cosmology.growth`,
and the ``GgahEmuPk`` backend -- is differentiable, and differentiable in the
*redshift* as well as the cosmology.

Several functions skip their input validation under tracing, because ``float()``
on a tracer raises.  That is safe only because the skipped code checks and never
computes; ``tests/test_paths.py`` asserts jit and eager agree to round-off, and
that the checks still fire when they can.

A gradient taken through a solver backend **raises** rather than returning a
number.  It would otherwise be missing the whole dependence of P(k) on the
cosmology: finite, smooth, and wrong.
"""

from . import constants
from .parameters import Cosmology, PLANCK18, RETIRED_KEYS
from .amplitude import (
    sigma8, s8, sigma2_tophat, sigma_tophat, sigma_v, tophat_window,
    ln10A_s_for_sigma8,
)
from .growth import growth_factor, growth_scale_spread
from .power import make_pk, LinearPowerSpectrum, PK_BACKENDS
from .background import (
    hubble_e,
    nu_density_shape,
    comoving_distance,
    comoving_distance_z1z2,
    angular_diameter_distance,
    luminosity_distance,
    comoving_volume_element,
    distance_modulus,
)

__all__ = [
    "constants", "Cosmology", "PLANCK18", "RETIRED_KEYS",
    "hubble_e", "nu_density_shape", "comoving_distance",
    "comoving_distance_z1z2", "angular_diameter_distance",
    "luminosity_distance", "comoving_volume_element", "distance_modulus",
    "sigma8", "s8", "sigma2_tophat", "sigma_tophat", "sigma_v",
    "tophat_window",
    "ln10A_s_for_sigma8", "growth_factor", "growth_scale_spread",
    "make_pk", "LinearPowerSpectrum", "PK_BACKENDS",
]
