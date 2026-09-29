"""ggah_mod -- galaxies, gas, AGN and halos, in layers.

A differentiable halo-model forward model, organised so the layer boundaries
are real:

===  ======================  =================================================
L1   :mod:`ggah_mod.cosmology`  background, P(k), growth, derived amplitudes
L2   :mod:`ggah_mod.halos`      mass function, bias, concentration, profiles
L3   ``ggah_mod.sectors``       galaxy occupation, mis-centering, gas, AGN --
                                peers, none of them the host of the others
L4   :mod:`ggah_mod.spectra`   one 1-halo + 2-halo integral for any tracer pair
L5   ``ggah_mod.observables``   projections: w_p, Delta Sigma, C_ell, w(theta)
===  ======================  =================================================

Two flavours of the *same* assembly are selected by
:class:`ggah_mod.backend.Backend`: ``ACCURATE`` for fitting and ``DIFFERENTIABLE`` for
forecasting.  Their disagreement is measured, not assumed.
"""

__version__ = "1.0.0"

from . import backend, cosmology
from ._wip import WorkInProgressWarning
from .backend import (Backend, ACCURATE, DIFFERENTIABLE,
                      DIFFERENTIABLE_COARSE, resolve_backend)

__all__ = ["backend", "cosmology", "Backend", "ACCURATE", "DIFFERENTIABLE",
           "DIFFERENTIABLE_COARSE", "resolve_backend", "WorkInProgressWarning",
           "__version__"]


def __getattr__(name):
    """Forward the retired flavour names, with their warning.

    Re-exported through here rather than imported at module scope: importing
    them eagerly would fire the ``DeprecationWarning`` on every
    ``import ggah_mod``, which is a warning about nothing the caller did.
    """
    if name in ("FAST", "REFERENCE"):
        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
