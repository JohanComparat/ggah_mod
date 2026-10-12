"""Layer 1: the cosmology.

This layer takes the cosmological parameters and produces the background
expansion, the linear power spectrum, the growth of structure, and the derived
amplitude diagnostics.  Nothing in it knows about haloes, galaxies, gas or AGN.

The two paths through layer 1
-----------------------------

Everything in this subpackage is pure JAX and traceable **except** the
solver-backed pieces:

===================================  ===========================================
Accurate path only                   why
===================================  ===========================================
``power.ClassPk``                    CLASS is a C Boltzmann solver
``power.CambPk``                     CAMB is Fortran
``amplitude.ln10A_s_for_sigma8``     a one-time Newton solve, run before a fit
===================================  ===========================================

Everything else, :mod:`~ggah_mod.cosmology.background`,
:mod:`~ggah_mod.cosmology.drag`, :mod:`~ggah_mod.cosmology.amplitude`,
:mod:`~ggah_mod.cosmology.growth` and the ``GgahEmuPk`` backend, is
differentiable, in the *redshift* as well as in the cosmology.  The drag
redshift is the one fitted quantity among them: a polynomial calibrated against
CLASS (``tools/calibrate_zdrag.py``).

Several functions skip their input validation under tracing, because ``float()``
on a tracer raises.  The skipped code only checks and never computes;
``tests/test_paths.py`` asserts that jit and eager agree to round-off, and that
the checks still fire when they can.

A gradient taken through a solver backend **raises** rather than returning a
number, which would be finite, smooth and missing the whole dependence of P(k)
on the cosmology.
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
    sound_horizon,
)
from .drag import z_drag, r_drag

__all__ = [
    "constants", "Cosmology", "PLANCK18", "RETIRED_KEYS",
    "hubble_e", "nu_density_shape", "comoving_distance",
    "comoving_distance_z1z2", "angular_diameter_distance",
    "luminosity_distance", "comoving_volume_element", "distance_modulus",
    "sound_horizon", "z_drag", "r_drag",
    "sigma8", "s8", "sigma2_tophat", "sigma_tophat", "sigma_v",
    "tophat_window",
    "ln10A_s_for_sigma8", "growth_factor", "growth_scale_spread",
    "make_pk", "LinearPowerSpectrum", "PK_BACKENDS",
]
