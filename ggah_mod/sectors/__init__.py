r"""Layer 3: the tracers, as peers.

This layer takes the :class:`~ggah_mod.halos.field.HaloField` of layer 2 and
populates it with tracers (galaxies, active nuclei, hot and cold gas, the
matter field), each returning the weights that layer 4 integrates.
Galaxy occupation, AGN, hot gas and the matter field are *peers*: none of them
hosts the others.
The mass grid belongs to the field, so a pure gas or AGN spectrum is computed
without inventing galaxy parameters for it.
In the predecessor, galaxy occupation owned the mass grid (five classes each
built their own ``logspace(10, 16, N)``, with N in {256, 512, 600}), and the
gas, AGN and cluster legs wrapped a galaxy predictor and read its private
attributes.

The field owns the redshift as well: no method of the sector protocol has a
``z`` argument, so a sector is one epoch.
A redshift-stacked field, which ``jax.vmap(make_field)`` legitimately builds,
is not a supported argument here:
:meth:`~ggah_mod.sectors.protocol.TracerWeights.total` refuses the rank-3
weight it would produce, because
:meth:`~ggah_mod.sectors.protocol.TracerWeights.at_large_scales` is
``total()[0]`` and would otherwise return a redshift slice under the name of
the :math:`k \to 0` row.
Cover a redshift set with :func:`~ggah_mod.halos.field.make_fields`, or by
mapping the whole ``z -> spectrum`` closure.

Differentiability
-----------------

Layer 2 is pure JAX with one declared exception, a scikit-learn emulator.
This layer has none: everything here is differentiable.
``tests/test_sectors_paths.py`` parses every module of this package with
``ast``, fails on any numpy call inside a function that is not listed with a
stated reason, and asserts that the set of whole-module exemptions is empty
(the layer-2 equivalent asserts it holds exactly one).

Two conventions make the rule affordable:

* a piece that cannot be made differentiable is excluded, by name, rather than
  admitted behind a flag.
  The predecessor's AGN halo-abundance-matching model is one: its inversion is
  a scipy interpolator over a table built by another interpolator.
  :mod:`.agn` names it.
* a table read off disk is not an exception: loading is construction, and the
  interpolation through it is jnp.

Sector protocol and parameters
------------------------------

Every sector implements :class:`~ggah_mod.sectors.protocol.Sector` and returns
:class:`~ggah_mod.sectors.protocol.TracerWeights`, so layer 4 has one integral
for any pair of tracers, where the predecessor hand-wrote eight.

Parameters are :class:`~ggah_mod.sectors.params.SectorParams`: names in
``_PARAMS`` are pytree leaves, differentiable by construction; names in
``_STATIC`` live in the treedef and cannot be differentiated.
Each parameter carries its bounds, its prior, and the reason for its bounds.
"""

from .._wip import warn_work_in_progress

warn_work_in_progress(__name__)

from .agn import (
    AgnParams, AgnSector, K_BOL, ObscurationParams, XLF, make_xlf,
    obscured_fraction,
)
from .alignments import (
    C1_RHO_CRIT, IaParams, IntrinsicAlignmentSector, alignment_bias,
)
from .census import BaryonCensus, PHASES, census, census_over_mass_ranges
from .clf import CLF, CLF_CALIBRATION, clf_defaults, make_clf
from .coldgas import (
    HI_CALIBRATION, HI_HALO, ColdGasParams, ColdGasSector, make_hi_halo,
)
from .cooling import COOLING, ApecCooling, make_cooling
from .ejecta import ETA_EJ, EjectaParams, EjectaSector
from .energetics import (
    F_GAS, EnergeticsParams, f_retained_energy, gas_concentration_factor,
    make_f_gas, soft_saturate,
)
from .galaxies import (
    GALAXY_MODELS, SATELLITE_PROFILE_PARAMS, GalaxyParams, GalaxySector,
    SatelliteProfileParams, galaxy_defaults,
)
from .gas import DpmParams, HotGasDPM
from .matter import (
    BaryonSplit, MatterField, cosmic_baryon_fraction, f_collisionless,
    matter_weights,
)
from .miscentering import (
    MISCENTERING_CONVENTIONS, MisCentering, MisCenteringParams,
)
from .occupation import (
    DEFAULTS, OCC_CALIBRATION, OCCUPATION, make_occupation,
    occupation_defaults,
)
from .occupation_params import (
    OCCUPATION_PARAMS, PER_MODEL, VOCABULARY, make_occupation_params,
    params_for,
)
from .params import (
    BOUND_KINDS, Flat, Gaussian, LogGaussian, MultivariateGaussian, Param,
    Prior, SectorParams, declared_fields, sector_params,
)
from .protocol import ProfileModifier, Sector, TracerWeights, combine
from .sham import SHMR, SHMR_CALIBRATION, invert_monotone, make_shmr

__all__ = [
    # the contract
    "TracerWeights", "Sector", "ProfileModifier", "combine",
    "Param", "Prior", "Flat", "Gaussian", "LogGaussian",
    "MultivariateGaussian", "SectorParams",
    "sector_params", "declared_fields", "BOUND_KINDS",
    # galaxies
    "OCCUPATION", "OCC_CALIBRATION", "DEFAULTS", "make_occupation",
    "occupation_defaults",
    "CLF", "CLF_CALIBRATION", "make_clf", "clf_defaults",
    "GalaxySector", "GalaxyParams", "GALAXY_MODELS", "galaxy_defaults",
    "EnergeticsParams", "ETA_EJ", "K_BOL", "ObscurationParams",
    "IntrinsicAlignmentSector", "IaParams", "alignment_bias",
    "C1_RHO_CRIT",
    "OCCUPATION_PARAMS", "VOCABULARY", "PER_MODEL",
    "make_occupation_params", "params_for",
    "SatelliteProfileParams", "SATELLITE_PROFILE_PARAMS",
    "SHMR", "SHMR_CALIBRATION", "make_shmr", "invert_monotone",
    # AGN
    "AgnSector", "AgnParams", "XLF", "make_xlf", "obscured_fraction",
    # hot gas
    "HotGasDPM", "DpmParams",
    "ApecCooling", "COOLING", "make_cooling",
    # energetics and the composite
    "EjectaSector", "EjectaParams",
    # neutral gas
    "ColdGasSector", "ColdGasParams", "HI_HALO", "HI_CALIBRATION",
    "make_hi_halo",
    "f_retained_energy", "F_GAS", "make_f_gas", "soft_saturate",
    "gas_concentration_factor",
    "MatterField", "matter_weights", "BaryonSplit", "f_collisionless",
    "cosmic_baryon_fraction",
    # the cosmic budget
    "BaryonCensus", "census", "census_over_mass_ranges", "PHASES",
    "MisCentering", "MisCenteringParams", "MISCENTERING_CONVENTIONS",
]


# The fifteen generated occupation containers, by name.  They are exported
# individually because `tests/test_sectors_paths.py`'s container audits find
# containers by walking this `__all__`, and a container they cannot see is a
# container with no bounds test -- which is exactly how `GalaxyParams` escaped
# `test_every_parameter_has_a_reason_for_its_bound`.
for _cls in OCCUPATION_PARAMS.values():
    globals()[_cls.__name__] = _cls
    if _cls.__name__ not in __all__:
        __all__.append(_cls.__name__)
del _cls
