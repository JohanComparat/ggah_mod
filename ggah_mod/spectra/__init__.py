r"""Layer 4: one 1-halo + 2-halo integral, for any pair of tracers.

This layer takes the halo field of layer 2 and the tracer weights of layer 3
and produces the power spectrum of a pair of tracers.  It reads layers 1-3 and
knows nothing of :mod:`ggah_mod.observables`.  Every tracer arrives as
:class:`~ggah_mod.sectors.protocol.TracerWeights`, so the layer never learns
which sector it is talking to; ``tests/test_spectra.py`` asserts that
:mod:`~ggah_mod.spectra.pk` names no sector.

Two static facts that :class:`TracerWeights` cannot carry are declared in
:mod:`~ggah_mod.spectra.spec`, and both change the answer: a tracer is a **list
of components** (the X-ray field is gas *plus* AGN), and two discrete tracers
must declare whether they **share objects**.

**One epoch.**  No function in this layer takes a redshift; it is the one the
:class:`~ggah_mod.halos.field.HaloField` was built at.  The integrands here are
``(Nk, NM)`` and a bare ``(NM,)`` weight is told from them by the leading axis,
so a redshift-stacked field would not raise: it would put a redshift where a
wavenumber belongs.  :func:`~ggah_mod.spectra.pair.normalised_parts` refuses a
rank-3 weight for that reason.  To cover a redshift set, use
:func:`~ggah_mod.halos.field.make_fields`, or ``jax.vmap`` the whole
``z -> spectrum`` closure, never a field on its own.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

from .counterterm import bias_consistency, low_mass_counterterm, mass_deficit
from .pair import pair_1h, shot_noise
from .neutrinos import NEUTRINO_TWO_HALO, neutrino_leg, tracer_leg
from .pk import (
    PowerSpectrum, i_of_k, linear_spectrum, pk_1h, pk_2h, pk_cross,
    two_halo_amplitude,
)
from .transition import (
    ONE_HALO_TRANSITION, TRANSITION_CALIBRATION, TransitionParams,
    one_halo_transition,
)
from .spec import (
    OVERLAPS, Band, Component, PkOptions, TracerSpec, overlap_of,
)
from .tracers import (
    TRACERS, build_weights, matter_suppression, resolve, spectrum,
)

__all__ = [
    # the static declarations
    "Band", "Component", "TracerSpec", "PkOptions", "OVERLAPS", "overlap_of",
    # the integral
    "PowerSpectrum", "pk_cross", "pk_1h", "pk_2h", "i_of_k",
    "pair_1h", "shot_noise",
    "linear_spectrum", "one_halo_transition", "TransitionParams",
    "ONE_HALO_TRANSITION", "TRANSITION_CALIBRATION",
    # the two-halo normalisation, and the neutrinos that are in no halo
    "bias_consistency", "mass_deficit", "low_mass_counterterm",
    "two_halo_amplitude", "NEUTRINO_TWO_HALO", "neutrino_leg", "tracer_leg",
    # the registry
    "TRACERS", "build_weights", "resolve", "spectrum", "matter_suppression",
]
