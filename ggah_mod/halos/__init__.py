r"""Layer 2 -- the halo field.

Depends on the cosmology and on nothing else.  In particular it does **not**
depend on any tracer: the mass grid, the mass function, the bias and the
profiles belong here, so a gas or AGN spectrum can be computed without
inventing galaxy parameters for it.

The two paths through layer 2
-----------------------------

Everything here is pure JAX and differentiable -- the variance, all eighteen
multiplicity functions (sixteen published fits and two recalibrations of
``tinker08``), all six bias fits, and ``dndm`` -- with **one** exception:

=================================  =============================================
Accurate path only                 why
=================================  =============================================
``mass_function.CsstHMF``          a scikit-learn Gaussian process behind numpy
=================================  =============================================

So the halo layer is differentiable exactly when its P(k) backend is.  Both
flavours take their mass function from the same registry entry,
``tinker08_csst`` -- the ``emu_hmf`` recalibration of ``tinker08`` against the
CSST emulator -- so :class:`~ggah_mod.halos.mass_function.CsstHMF` is never on
:func:`~ggah_mod.halos.field.make_field`'s path; it is reachable through
:func:`~ggah_mod.halos.mass_function.make_hmf` for comparison.

``CsstHMF.differentiable`` is ``False`` and ``tests/test_paths.py`` checks that
a gradient through it raises rather than quietly returning one.

What a fit may be used for
--------------------------

:mod:`~ggah_mod.halos.calibration` reads
:data:`~ggah_mod.halos.mass_function.CALIBRATION` and
:data:`~ggah_mod.halos.concentration.CM_CALIBRATION` from
:func:`~ggah_mod.halos.field.make_field` and refuses a mass definition the
chosen fits were not calibrated in.  Both tables predate it by a long way and
neither was read by anything, which is the weaker situation it looks: a table
nobody consults cannot prevent the error it describes.

The same call site now also *hands* the fits the definition the field declares.
Published :math:`\Delta`-dependent fits are indexed against the mean density, so
200c is :math:`\Delta_{\rm m} = 645` at :math:`z = 0` and virial is 331;
``tinker08`` and ``tinker10`` both default to 200, and until this was plumbed a
field declaring 200c had 200c radii, a 200m abundance and a 200m bias.

Validated against an external reference
---------------------------------------

:mod:`~ggah_mod.halos.lensing_profiles` (tNFW, BMO, Hernquist) is no longer the
placeholder this paragraph used to call it, and is no longer unvalidated either
-- which were, for two tiers, two different claims about it made in two files.

Two checks, answering different questions.  *Internally*, BMO is implemented
twice from two papers' algebra -- in Fourier space and in real space -- and the
two agree to 3.0e-7 of the peak through the layer-5 transform on the four radii
the suite pins, 3.4e-7 over a wider forty, sharply enough that a one per cent
error in either shows.  That rules out a transcription slip in one
representation and, structurally, cannot rule out a misreading they share.
*Externally*, the dimensionless kernels now run against the **Oguri et al.
(2026)** implementation at ``mpmath`` 50 digits and agree to **1.2e-13** in
:math:`\Sigma`, 4.0e-10 in :math:`\bar\Sigma`, and 2.8e-9 in the
:math:`P`/:math:`Q` hyperbolic combination that :math:`\bar\Sigma` is built
from.  See that module's own status section, and
``tests/test_lensing_goldens.py`` for the provenance.

Beyond-linear bias
------------------

:mod:`~ggah_mod.halos.beyond_linear_bias` **has landed**, and not in the
restricted form that was planned.  The investigation the port waited on settled
its three open questions, and one of its premises:

* *negative power in the top mass bins* -- real, and it is noise: the published
  errors put :math:`\sigma/|1+\beta|` at 10-100 per cent there, against 0.5-2 per cent
  below nu ~ 2.  ``nu_max_trust`` truncates that corner on request.
* *no genuine k -> 0 limit* -- also real, and the reference implementation never
  used the data there either: it forces beta_NL to zero below
  k = 0.08 h/Mpc.  Below k ~ 0.02 the measurement is consistent with zero
  (SNR 0.2-0.9), so the tabulated values are noise, not a limit.  The module
  tapers rather than stepping, so the correction stays differentiable.
* *one simulation cosmology* -- handled by the Angulo & White (2010) sigma(R)
  rescaling, which is what the reference does by default.
* *z = 0 only* -- a wrong premise.  **35** snapshots are public, z = 2.891 to 0,
  and beta_NL is not redshift-universal: the z = 0 to z = 1 shift is 4-34 sigma
  against the published errors.  All 35 are shipped.

It adds no exception to the rule above: it is pure JAX, it jits, and it
differentiates -- including through the rescaling, whose grid search supplies
only a bracket while the value carries the gradient.  What is deferred is not
the kernel but its wiring: the correction is a 2-halo term, so layer 4 calls it
behind an explicit hook.
"""

from .variance import (
    lagrangian_radius, mass_from_radius, sigma_of_mass, dln_sigma_dln_mass,
    check_k_support, DELTA_C, K_MAX_R_MIN, K_MIN_R_MAX,
)
from .calibration import (MismatchWarning, check_calibration,
                          check_cosmology_support)
from .mass_function import (
    dndm, make_multiplicity, MULTIPLICITY,
    FittingFunctionHMF, CsstHMF, make_hmf, HMF_BACKENDS, CALIBRATION,
    CurvatureResponse, FitCapability, MULTIPLICITY_CAPABILITY,
    VIRIAL_REFERENCED_MULTIPLICITY,
)
from .profiles import (
    g_nfw, si, ci, profile_uk_gl,
    nfw_rho, nfw_mass, nfw_uk, nfw_sigma, nfw_mean_sigma, nfw_delta_sigma,
    nfw_scale_density, einasto_rho, einasto_uk,
    gnfw_shape, gnfw_rho, gnfw_uk, satellite_uk,
)
from .lensing_profiles import (
    tnfw_rho, tnfw_mass, tnfw_sigma, tnfw_mean_sigma, tnfw_delta_sigma, tnfw_uk,
    bmo_rho, bmo_mass, bmo_mass_total, bmo_sigma, bmo_mean_sigma,
    bmo_delta_sigma, bmo_uk,
    hernquist_rho, hernquist_mass, hernquist_sigma, hernquist_mean_sigma,
    hernquist_delta_sigma, HERNQUIST_RB_RE,
)
from .mass_definitions import (
    MassDef, parse_mass_def, rho_reference, delta_vir, translate_mass, VIRIAL,
    nfw_params_from_mass,
)
from .concentration import (
    c_duffy08, c_dutton14, c_klypin16, c_bhattacharya13, c_diemer19, c_seppi21,
    n_eff_from_sigma, CONCENTRATION, make_concentration, CM_CALIBRATION,
    DIEMER19,
    SEPPI21, SEPPI21_A1, SEPPI21_A1_COLUMNS, ANCHORS,
    seppi21_shape, seppi21_table_scale, seppi21_scale,
    seppi21_logpdf, seppi21_pdf, seppi21_quantile,
    sigma_ln_c_seppi21, seppi21_nodes,
    seppi21_shape_translated, seppi21_definition_jacobian,
)
from .linear_bias import (
    make_bias, BIAS, mass_weighted_bias, mass_fraction, Unresolved, unresolved,
    matched_bias_for, MATCHED_BIAS,
)
from .beyond_linear_bias import (
    BetaNLTable, table_at, beta_nl, match, project_weights,
    correction_2h_gg, correction_2h_gm,
    K_MIN_BNL, NU_HARD, R_RESCALE, S_RANGE, MDR1, MDR1_SIGMA8,
)

__all__ = [
    "lagrangian_radius", "mass_from_radius", "sigma_of_mass",
    "dln_sigma_dln_mass", "check_k_support", "DELTA_C",
    "K_MAX_R_MIN", "K_MIN_R_MAX", "MismatchWarning", "check_calibration",
    "check_cosmology_support", "CurvatureResponse", "FitCapability",
    "MULTIPLICITY_CAPABILITY", "VIRIAL_REFERENCED_MULTIPLICITY", "VIRIAL",
    "dndm", "make_multiplicity", "MULTIPLICITY",
    "FittingFunctionHMF", "CsstHMF", "make_hmf", "HMF_BACKENDS",
    "make_bias", "BIAS", "mass_weighted_bias", "mass_fraction",
    "Unresolved", "unresolved",
    "matched_bias_for", "MATCHED_BIAS", "CALIBRATION",
    "g_nfw", "si", "ci", "profile_uk_gl",
    "nfw_rho", "nfw_mass", "nfw_uk", "nfw_sigma", "nfw_mean_sigma",
    "nfw_delta_sigma", "nfw_scale_density", "einasto_rho", "einasto_uk",
    "gnfw_shape", "gnfw_rho", "gnfw_uk", "satellite_uk",
    "MassDef", "parse_mass_def", "rho_reference", "delta_vir", "translate_mass",
    "nfw_params_from_mass",
    "c_duffy08", "c_dutton14", "c_klypin16", "c_bhattacharya13", "c_diemer19",
    "c_seppi21",
    "n_eff_from_sigma", "CONCENTRATION", "make_concentration",
    "CM_CALIBRATION", "DIEMER19",
    "SEPPI21", "SEPPI21_A1", "SEPPI21_A1_COLUMNS", "ANCHORS",
    "seppi21_shape", "seppi21_table_scale", "seppi21_scale",
    "seppi21_logpdf", "seppi21_pdf", "seppi21_quantile",
    "sigma_ln_c_seppi21", "seppi21_nodes",
    "seppi21_shape_translated", "seppi21_definition_jacobian",
    "tnfw_rho", "tnfw_mass", "tnfw_sigma", "tnfw_mean_sigma",
    "tnfw_delta_sigma", "tnfw_uk",
    "bmo_rho", "bmo_mass", "bmo_mass_total", "bmo_sigma", "bmo_mean_sigma",
    "bmo_delta_sigma", "bmo_uk",
    "hernquist_rho", "hernquist_mass", "hernquist_sigma",
    "hernquist_mean_sigma", "hernquist_delta_sigma", "HERNQUIST_RB_RE",
    "BetaNLTable", "table_at", "beta_nl", "match", "project_weights",
    "correction_2h_gg", "correction_2h_gm",
    "K_MIN_BNL", "NU_HARD", "R_RESCALE", "S_RANGE", "MDR1", "MDR1_SIGMA8",
]
