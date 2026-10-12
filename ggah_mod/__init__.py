"""Galaxies, gas, AGN and haloes: a layered, differentiable halo-model forward model.

``ggah_mod`` predicts the power spectrum of any pair of tracers, and its
projections, from one set of cosmological and astrophysical parameters.  Each
layer depends only on the ones above it:

===  ===========================  ==================================================
L1   :mod:`ggah_mod.cosmology`    background, P(k), growth, derived amplitudes
L2   :mod:`ggah_mod.halos`        mass function, bias, concentration, profiles
L3   ``ggah_mod.sectors``         galaxy occupation, mis-centring, gas, AGN --
                                  peers, none of them the host of the others
L4   :mod:`ggah_mod.spectra`      one 1-halo + 2-halo integral for any tracer pair
L5   ``ggah_mod.observables``     projections: w_p, Delta Sigma, C_ell, w(theta)
L6   ``ggah_mod.covariance``      covariances of data vectors
===  ===========================  ==================================================

:class:`ggah_mod.backend.Backend` selects one of two flavours of the *same*
assembly: ``ACCURATE`` for fitting and ``DIFFERENTIABLE`` for forecasting.  The
parity budget (``tests/test_parity_budget.py``) measures their disagreement.
"""

__version__ = "1.3.0.dev0"

from . import backend, cosmology
from ._wip import WorkInProgressWarning
from .backend import (Backend, ACCURATE, DIFFERENTIABLE,
                      DIFFERENTIABLE_COARSE, resolve_backend)

__all__ = ["backend", "cosmology", "Backend", "ACCURATE", "DIFFERENTIABLE",
           "DIFFERENTIABLE_COARSE", "resolve_backend", "WorkInProgressWarning",
           "__version__"]


def __getattr__(name):
    """Forward the retired flavour names, with their warning.

    Forwarded here rather than imported at module scope, where they would fire
    the ``DeprecationWarning`` on every ``import ggah_mod``.
    """
    if name in ("FAST", "REFERENCE"):
        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
