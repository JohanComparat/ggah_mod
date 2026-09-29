r"""Hot gas: one parameter set, four tracer views.

The DPM -- Oppenheimer, Voit & Bahe (2025), MNRAS 543, 2649
(arXiv:2505.14782), *Introducing the Descriptive Parametric Model: gaseous
profiles for galaxies, groups, and clusters* -- describes the hot
circumgalactic and intracluster plasma with three generalised-NFW profiles
sharing one scale radius -- electron density, electron pressure and
metallicity -- each normalised at :math:`0.3\,R_\Delta`.  Here that radius is
the halo's own, :math:`R_s = R_\Delta/c(M,z)` with the concentration the halo
field carries, the one the dark matter's NFW profile uses:

.. math::

    n_e(r|M,z) &= n_{e,0.3}\,\frac{f(x|\alpha^{n})}{f(0.3c)}\,
                  E(z)^{\gamma_n}\,M_{12}^{\beta_n} \\
    P_e(r|M,z) &= P_{0.3}\,\frac{f(x|\alpha^{P})}{f(0.3c)}\,
                  E(z)^{\gamma_P}\,M_{12}^{\beta_P} \\
    Z(r)       &= Z_{0.3}\,f(x|\alpha^{Z})/f(0.3c)

with :math:`x = r/R_s`, :math:`c = c(M,z)` and :math:`f` the gNFW shape.
Everything else follows:
the temperature is :math:`kT = P_e/n_e`, and the X-ray emissivity is
:math:`n_e^2\Lambda(kT, Z)`.

Four views come out of that one parameter set:

=================  ==================================  ======================
view               quantity                            consumer
=================  ==================================  ======================
``mass_uk``        gas mass profile, and `f_gas`       lensing, matter field
``density_uk``     :math:`\tilde n_e(k|M)`             kSZ
``pressure_uk``    :math:`\tilde y(k|M)`               tSZ
``emissivity_uk``  :math:`\tilde X(k|M)`               X-ray brightness
=================  ==================================  ======================

The shape function is **not defined here**.  It is
:func:`ggah_mod.halos.gnfw_shape`, which layer 2 already owns and which
``tests/test_sector_coherence.py`` (``test_the_gnfw_shape_is_not_redefined``)
and ``tests/test_gas.py`` keep to one definition.  The predecessor had six
copies of it and seven literals of ``c_DPM = 2.772``.

Two normalisation errors are corrected on the way in
-----------------------------------------------------

Both were live in the predecessor, and freeing parameters over either lets a
sampler absorb it into an amplitude and report a good fit at wrong physics.

**The pressure normalisation was out by 11.605.**  :math:`P_{0.3}` is published
as :math:`P/k_B` in :math:`{\rm cm^{-3}\,K}`.  The predecessor read the table as
"meV cm^-3" and multiplied by :math:`10^{-6}`; the correct conversion to
:math:`{\rm keV\,cm^{-3}}` is :math:`k_B = 8.617333\times10^{-8}\,{\rm keV/K}`
-- a constant that was *already defined* in that package and never used.  Here
it is :data:`ggah_mod.cosmology.constants.K_B_KEV_PER_K`.

**The mass pivot was in the wrong units.**  :math:`M_{12} \equiv
M_{200}/10^{12}M_\odot` with :math:`M` **physical**; the predecessor divided an
:math:`M_\odot/h` mass by :math:`10^{12}` at four sites, and its own docstrings
contradicted each other about which was meant.  Correcting it makes the gas
profiles depend on :math:`h` for the first time.

**The acceptance test is closed form**, not a golden file: DPM Model 1 is
self-similar *by construction*, so :math:`kT/kT_{\rm vir}` must be flat in mass.
It is 0.74 with the :math:`k_B` fix alone and 0.96 with both, and all three
models then agree on :math:`\approx 6\,{\rm keV}` at :math:`10^{15}M_\odot/h`.
Uncorrected, the profile gives a 70 keV cluster.

**The halo's concentration, for all three profiles.**  The predecessor let the
density use a per-halo :math:`c(M,z)` while the pressure hard-coded
:math:`c_{\rm DPM}`, then divided them at the same radii to get
:math:`T = P_e/n_e` -- so the temperature mixed two conventions.  The published
model fixes :math:`c_{\rm DPM} = 2.772` in :math:`R_{200{\rm c}}`, and this
package carried it for a while as one free parameter, 4.5877 at the shipped
200m definition.  All three profiles now take the halo field's :math:`c(M,z)`
instead, passed in as ``conc`` and never defaulted: a concentration of the
baryons' own that varies with mass is left to the slopes to absorb.  At
Planck 2018 and :math:`z = 0` the halo value runs from 11.3 at
:math:`10^{10}` to 5.3 at :math:`10^{16}\,M_\odot/h` (7.55 at
:math:`10^{14}`), above 4.5877 at every mass of the grid, so the published
slopes now give more concentrated profiles than published everywhere except at
:math:`0.3R_\Delta`, where each profile equals its anchor whatever :math:`c`
is.

Nothing is fixed by hard-coding
-------------------------------

:math:`\gamma_n`, :math:`\gamma_P`, :math:`\alpha_{\rm out}^{\rm var}` (dead at
0 in all three published models), :math:`\sigma_{\rm scatter}` (off everywhere),
:math:`n_H/n_e`, and the six metallicity shape parameters
the predecessor's class took *no constructor arguments at all* for -- every one
is a :class:`~ggah_mod.sectors.params.Param` with a bound and a reason.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..cosmology import constants as C
from ..cosmology.background import hubble_e
from ..halos.profiles import gnfw_shape, profile_uk_gl
from .params import Gaussian, Flat, Param, SectorParams, sector_params

#: Whether the gas amplitudes carry an energy-predicted baryon budget.
#:
#: ``"none"`` is the default and is the model that shipped: the amplitudes
#: are this sector's own free parameters and the energy closure reaches
#: nothing.  ``"closure"`` couples them, and needs the AGN and galaxy
#: sectors because the budget is built from a black-hole mass and a
#: stellar mass.
FEEDBACK_MODES = ("none", "closure")

#: The mass definition used when no field says otherwise.
#:
#: Read from the accurate flavour rather than restated, because the whole
#: point of this change is that the package carries **one** definition and
#: this sector no longer keeps its own.  A bare call with no field behind it
#: still has to pick something; picking what the flavours declare means the
#: two cannot drift apart.
def _default_mdef():
    from ..backend import ACCURATE
    return ACCURATE.mdef

from ..numerics import require_x64
from .protocol import TracerWeights


__all__ = ["DpmParams", "HotGasDPM", "MU_E", "M_PROTON_G", "M_SUN_G",
           "SIGMA_T_CM2", "ME_C2_KEV", "DPM_MODELS", "dpm_model_params"]

# --- microphysical constants -------------------------------------------------
# Not in `cosmology.constants` because nothing above layer 3 needs them.  If a
# later layer does, they move up rather than being written a second time.

#: Proton mass [g].  CODATA 2018.
M_PROTON_G = 1.67262192369e-24
#: Solar mass [g].  IAU 2015 nominal, consistent with `ERG_PER_MSUN_KMS2`.
M_SUN_G = 1.98892e33
#: Mean molecular weight **per electron**, :math:`\rho_{\rm gas} = \mu_e m_p n_e`.
#: 1.14 for a fully ionised solar-abundance plasma.
MU_E = 1.14
#: Thomson cross-section [cm^2].
SIGMA_T_CM2 = 6.6524587e-25

#: One electron density [cm^-3] integrated over a comoving :math:`({\rm Mpc}/h)^3`,
#: expressed in :math:`M_\odot/h`, with the :math:`h` and :math:`(1+z)` factors left
#: outside.  Folded into a single Python float **on purpose**: written as separate
#: factors the conversion cannot survive single precision, because
#: ``MPC_CM**3`` is 2.9e73 (above the largest float32, 3.4e38) and
#: ``M_PROTON_G / M_SUN_G`` is 8.4e-58 (below the smallest normal, 1.2e-38).
#: Either one flushes to ``inf`` or to zero the moment it meets a ``jnp`` array,
#: which is how ``gas_mass`` came to return ``inf`` for every mass under JAX's
#: default dtype.  Their product is 2.8e16 and is representable everywhere.
_NE_INTEGRAL_TO_MSUN_H = MU_E * M_PROTON_G / M_SUN_G * C.MPC_CM ** 3
#: Electron rest energy [keV].
ME_C2_KEV = 510.99895


# =========================================================================
# Parameters
# =========================================================================

@sector_params
class DpmParams(SectorParams):
    """The DPM gas sector's free parameters, with bounds and reasons."""

    # -- the aperture the mass view integrates to -----------------------------
    aperture: float = 1.0

    # -- density -------------------------------------------------------------
    log10_ne_anchor: float = -4.7828
    alpha_in_n: float = 1.0
    alpha_tr_n: float = 1.9
    alpha_out_n: float = 2.7
    beta_n: float = 0.36
    gamma_n: float = 2.0
    # -- pressure ------------------------------------------------------------
    log10_pe_anchor: float = 1.4342
    alpha_in_p: float = 0.3
    alpha_tr_p: float = 1.3
    alpha_out_p: float = 4.1
    alpha_out_var: float = 0.0
    beta_p: float = 0.85
    gamma_p: float = 8.0 / 3.0
    # -- metallicity ---------------------------------------------------------
    z_anchor: float = 0.2508
    alpha_in_z: float = 0.0
    alpha_tr_z: float = 0.5
    alpha_out_z: float = 0.7
    # -- shared --------------------------------------------------------------
    # No concentration: the three profiles sit on the halo's own scale radius,
    # r_delta / c(M,z), read from the field (see `HotGasDPM.n_e`).
    r_max_over_rdelta: float = 1.8127
    sigma_scatter: float = 0.0
    nh_over_ne: float = 0.83

    # No annotation: an annotated assignment in a dataclass body declares a
    # *field*, so `_STATIC: tuple = ()` would make the classification list
    # itself a traced leaf.  Caught by `test_every_declared_field_is_classified`.
    _STATIC = ()

    _PARAMS = {
        "aperture": Param(
            1.0, (0.1, 5.0), Flat(), "R_Delta",
            "the radius the mass view integrates to, in units of R_Delta. It "
            "is a declared parameter and not a call argument because for DPM "
            "Model 3 it is not a convenience: that model's outer density slope "
            "is 0.5, so int r^2 rho dr does not converge and the gas mass is "
            "whatever the aperture says it is. A number the answer depends on "
            "that strongly is a parameter with a bound and a reason, which is "
            "this package's standing rule. Below 0.1 R_Delta the integral is "
            "inside the core and is not a halo gas mass; above ~5 the profile "
            "has merged with the field and the ejected sector is the one "
            "describing it",
            "physical"),
        "log10_ne_anchor": Param(
            -4.7828, (-7.5, -3.5), Flat(), "log10 cm^-3",
            "n_e at 0.3 R_Delta spans the group-to-cluster range within "
            "these decades; outside, the gas mass exceeds the halo's baryons "
            "or underflows the CGM census.  The window is Oppenheimer et "
            "al.'s, shifted with the anchor: -0.42 dex, which is what their "
            "profile does between 0.3 R_200c and 0.3 R_Delta at the shipped "
            "definition", "prior"),
        "alpha_in_n": Param(
            1.0, (0.0, 1.4), Flat(), "",
            "the emission integral int x^{2-2 alpha_in} dx diverges at "
            "alpha_in >= 1.5, so the upper bound is definitional, not a prior",
            "definitional"),
        "alpha_tr_n": Param(1.9, (0.1, 5.0), Flat(), "",
                            "transition sharpness; below 0.1 the two power "
                            "laws never separate and the fit is degenerate",
                            "definitional"),
        "alpha_out_n": Param(
            2.7, (0.3, 4.0), Flat(), "",
            "outer density slope. The gas-mass integral int r^{2-alpha_out} dr "
            "converges only for alpha_out > 3, and the emission integral for "
            "alpha_out > 1.5 -- yet DPM Model 3 publishes 0.5, so the bound "
            "cannot be set by convergence without excluding a published "
            "model. It is set by the fitted range instead, and the mass view "
            "carries an explicit aperture so a divergent profile gives an "
            "aperture-dominated answer rather than a silently infinite one",
            "validity"),
        "beta_n": Param(0.36, (-0.4, 1.2), Flat(), "",
                        "mass scaling; 0 is self-similar, and the measured "
                        "range brackets it generously", "prior"),
        "gamma_n": Param(2.0, (0.0, 4.0), Flat(), "",
                         "redshift scaling; 2 is self-similar. Fixed at 2 in "
                         "the published models and freed here", "prior"),
        "log10_pe_anchor": Param(
            1.4342, (-0.7, 3.3), Flat(), "log10 keV cm^-3 (x 1e-8)",
            "P_0.3/k_B in cm^-3 K; the published models span 71-409, and the "
            "window is set around them rather than around the value the "
            "1e-6 misconversion produced", "prior"),
        "alpha_in_p": Param(0.3, (-3.0, 2.5), Flat(), "",
                            "may be negative: the pressure profile can be "
                            "cored, unlike the density. The floor was -0.7, "
                            "which is 0.1 below DPM model 3's -0.6 and had no "
                            "basis beyond that; it bound **11 of 15** tSZ MAP "
                            "fits in the v0.6.0 campaign. It is not a "
                            "degeneracy: the shape at the fitted radii moves "
                            "12% for a 0.3 change in this parameter and 66% "
                            "for 1.3, because the profile was normalised at "
                            "0.3 c = 1.38, with c then a single parameter at "
                            "4.5877, where the gNFW (1 + x^a_tr) term has not "
                            "yet reached its power-law limit, so the inner "
                            "slope does not cancel out of the ratio. -3.0 is "
                            "wide enough to tell an interior optimum from a "
                            "runaway, which a floor at the published envelope "
                            "cannot. Those two numbers are the outer power-law "
                            "limit, (1 + (0.3c)^-a_tr)^(d a_in/a_tr) - 1. Re-"
                            "measured on the halo's own concentration (0.8.8), "
                            "0.3 c = 1.7-2.2 over 1e13-1e15 Msun/h at z = 0.2: "
                            "8.4% and 42% at 1e14, 35-52% for the 1.3 change "
                            "across the range, so the inner slope still does "
                            "not cancel and the bound stands", "physical"),
        "alpha_tr_p": Param(1.3, (0.1, 10.0), Flat(), "",
                            "transition sharpness of the pressure profile; "
                            "below 0.1 the inner and outer power laws never "
                            "separate and the fit is degenerate. The ceiling "
                            "was 5.0 with no reason attached to it -- the "
                            "sentence above justifies only the floor -- and it "
                            "bound **7 of 15** tSZ MAP fits. 10.0 is where the "
                            "data stops resolving it: the gNFW transition has "
                            "width ~1/a_tr in ln x, and the finest radial "
                            "spacing in the fitted data is dln r = 0.11 (the "
                            "outermost CAP apertures; the y-profile bins are "
                            "0.341 apart), so above a_tr ~ 9 the transition is "
                            "narrower than any bin and the parameter stops "
                            "being constrained rather than becoming "
                            "unphysical", "definitional"),
        "alpha_out_p": Param(4.1, (2.0, 8.0), Flat(), "",
                             "pressure falls faster than density; > 2 keeps "
                             "the thermal energy finite", "definitional"),
        "alpha_out_var": Param(
            0.0, (-1.0, 1.0), Flat(), "per dex",
            "DPM Eq. 5, the mass-dependence of the outer pressure slope. Zero "
            "in all three published models, i.e. dead code there; freed here "
            "because a constant nobody chose is not a measurement",
            "prior"),
        "beta_p": Param(0.85, (0.0, 1.8), Flat(), "",
                        "2/3 is self-similar; the models span 0.67-0.92",
                        "prior"),
        "gamma_p": Param(8.0 / 3.0, (0.0, 5.0), Flat(), "",
                         "8/3 is self-similar, and is what all three models "
                         "adopt", "prior"),
        "z_anchor": Param(0.2508, (0.017, 2.5), Gaussian(0.2508, 0.084),
                      "Z_sun",
                      "metallicity at 0.3 R_Delta.  The cooling table is built "
                      "over 0.02-3 Z_sun in *absolute* metallicity and clamps "
                      "outside it, so that bound moves with the anchor while "
                      "the table does not -- the numbers here are the table's "
                      "range expressed at the new radius.  The prior is "
                      "Oppenheimer et al.'s 0.3 +- 0.1 carried through the "
                      "same shift", "validity"),
        "alpha_in_z": Param(0.0, (-1.0, 1.5), Flat(), "",
                            "metallicity profiles are flat or mildly cored in "
                            "the centre", "physical"),
        "alpha_tr_z": Param(0.5, (0.1, 5.0), Flat(), "",
                            "transition sharpness of the metallicity profile; "
                            "below 0.1 the inner and outer power laws never "
                            "separate and the fit is degenerate",
                            "definitional"),
        "alpha_out_z": Param(0.7, (0.0, 3.0), Flat(), "",
                             "metallicity declines outward; 0 is a flat "
                             "profile, which is the no-gradient limit",
                             "physical"),
        "r_max_over_rdelta": Param(
            1.8127, (0.6, 3.6), Flat(), "R_Delta",
            "outer truncation. Below 1 the profile is cut inside the halo; "
            "above ~6 the DPM calibration has no data", "validity"),
        "sigma_scatter": Param(
            0.0, (0.0, 1.0), Flat(), "dex",
            "DPM Eq. 6, log-normal scatter in n_e boosting <n_e^2>. Off in "
            "every published model. Exactly degenerate with log10_ne_anchor in the "
            "X-ray amplitude alone, so freeing both needs a second observable",
            "prior"),
        "nh_over_ne": Param(
            0.83, (0.7, 1.0), Gaussian(0.83, 0.02), "",
            "n_H/n_e for a fully ionised plasma; depends on abundance, which "
            "is fitted, so it is not a constant", "physical"),
    }


#: The three published DPM models (Oppenheimer et al. 2025 Table 1).
#:
#: ``log10_ne_anchor`` and ``log10_pe_anchor`` are stored as logs of the **published**
#: numbers -- ``n_e`` in cm^-3 and ``P/k_B`` in cm^-3 K.  The conversion of the
#: latter to keV cm^-3 happens once, in :meth:`HotGasDPM.pressure`, through
#: ``K_B_KEV_PER_K``.
#: The three published models, **converted to this package's anchor**.
#:
#: Oppenheimer et al. quote their amplitudes as :math:`n_e` and :math:`P_e` at
#: :math:`0.3R_{200{\rm c}}`, with the mass dependence in
#: :math:`M_{200{\rm c}}`.  This package carries one mass definition end to
#: end, so the anchor is :math:`0.3R_\Delta` and the mass is :math:`M_\Delta`
#: -- and the numbers below are therefore **not the ones in that paper**.  They
#: are its models, re-expressed, and the two shifts that get them here are:
#:
#: * the radius, :math:`R_\Delta = kR_{200{\rm c}}` with :math:`k = 1.655` at
#:   the shipped definition, which moves the normalisation point from
#:   :math:`x = 0.3c` to :math:`0.3ck` and changes the amplitude by the shape
#:   ratio between them;
#: * the mass, :math:`M_\Delta/M_{200{\rm c}} = 1.411` at the
#:   :math:`10^{14}\,M_\odot/h` pivot, which enters as
#:   :math:`-\beta\log_{10}` of that ratio.
#:
#: Neither factor is constant -- both run through the published concentration,
#: :math:`c_{\rm DPM} = 2.772` -- so the conversion is exact at the pivot and
#: drifts to 2--4 per cent by :math:`10^{13}` and :math:`10^{15}`.  That is
#: the price of a single set of numbers covering three decades, and it is why
#: the parameters are renamed rather than merely rescaled: a reader who compares
#: ``log10_ne_anchor`` against the paper's :math:`n_{e,0.3}` should find the
#: names do not match.
#:
#: The profiles themselves no longer use that concentration: they take the
#: halo's own :math:`c(M,z)`, which is above the published value at every mass,
#: so these numbers reproduce a published profile only at :math:`0.3R_\Delta`,
#: where it equals its anchor.  Away from it the shapes are the halo's, and a fit
#: moves the slopes to absorb the difference.
DPM_MODELS = {
    1: dict(log10_ne_anchor=-3.6486, beta_n=0.00,
            alpha_in_n=1.0, alpha_tr_n=1.9, alpha_out_n=2.7,
            log10_pe_anchor=2.0126, beta_p=2.0 / 3.0,
            alpha_in_p=0.3, alpha_tr_p=1.3, alpha_out_p=4.1),
    2: dict(log10_ne_anchor=-4.7828, beta_n=0.36,
            alpha_in_n=1.0, alpha_tr_n=1.9, alpha_out_n=2.7,
            log10_pe_anchor=1.4342, beta_p=0.85,
            alpha_in_p=0.3, alpha_tr_p=1.3, alpha_out_p=4.1),
    3: dict(log10_ne_anchor=-4.4649, beta_n=0.36,
            alpha_in_n=0.4, alpha_tr_n=0.45, alpha_out_n=0.5,
            log10_pe_anchor=1.5586, beta_p=0.92,
            alpha_in_p=-0.6, alpha_tr_p=0.2, alpha_out_p=2.0),
}


def dpm_model_params(model: int = 2, **overrides) -> DpmParams:
    """One of the three published models, as a :class:`DpmParams`."""
    if model not in DPM_MODELS:
        raise ValueError(f"DPM model must be one of {sorted(DPM_MODELS)}, "
                         f"got {model!r}")
    return DpmParams(**{**DPM_MODELS[model], **overrides})


# =========================================================================
# The sector
# =========================================================================


def _lg(amplitude):
    r""":math:`\log_{10}A`, and exactly zero when no amplitude is given.

    Written as a log shift on the anchor rather than a multiplication of the
    profile so that ``amplitude=None`` reproduces the shipped number *bit for
    bit* -- ``x + 0.0`` is ``x``, while ``x * 1.0`` is only almost.
    """
    if amplitude is None:
        return 0.0
    a = jnp.log10(jnp.asarray(amplitude))
    # A per-halo amplitude arrives as (NM,) and the profiles are (NM, Nr) --
    # including inside `_uk`, where the radii are the quadrature's (NM, n_gl).
    # Give it the trailing axis rather than making every caller remember to.
    return a[:, None] if a.ndim == 1 else a


def _every_star(galaxies_params):
    r"""The galaxy parameters with the selection threshold at its box floor.

    For a threshold occupation ``log10m_star_thresh`` says which galaxies a
    *sample* holds; an energy budget wants every star.  Anything without the
    field -- a non-threshold occupation -- is returned unchanged.
    """
    import dataclasses

    from .galaxies import GalaxyParams

    floor = float(GalaxyParams._PARAMS["log10m_star_thresh"].bounds[0])
    if dataclasses.is_dataclass(galaxies_params):
        if hasattr(galaxies_params, "log10m_star_thresh"):
            return dataclasses.replace(galaxies_params,
                                       log10m_star_thresh=floor)
        return galaxies_params
    if isinstance(galaxies_params, dict) and \
            "log10m_star_thresh" in galaxies_params:
        return {**galaxies_params, "log10m_star_thresh": floor}
    return galaxies_params


class HotGasDPM:
    """The hot-gas sector.  Four views, one parameter set."""

    name = "gas"
    differentiable = True

    def __init__(self, cooling=None, n_gl: int | None = None, backend=None,
                 *, feedback: str = "none", feedback_pivot: float = 1e14,
                 agn=None, galaxies=None, energetics=None, coldgas=None):
        """``n_gl`` comes from the backend unless it is given explicitly.

        It used to default to a hard-coded 128, so
        :attr:`~ggah_mod.backend.Backend.n_gl` -- 200 under ``ACCURATE``, 64
        under ``DIFFERENTIABLE`` -- was declared, validated, and read by nothing, next to
        a ``gas_ft`` field that has since been removed for the same reason.  That is
        the same defect layer 3 found in ``cm_model``: a name nothing consumed,
        and therefore a name nothing checked.  An explicit argument still wins,
        because a convergence test has to be able to sweep it.
        """
        from ..backend import resolve_backend
        self._cooling = cooling
        self._backend = resolve_backend(backend)
        self.n_gl = int(self._backend.n_gl if n_gl is None else n_gl)
        if str(feedback) not in FEEDBACK_MODES:
            raise ValueError(f"unknown feedback mode {feedback!r}; expected "
                             f"one of {sorted(FEEDBACK_MODES)}")
        #: Whether the amplitudes carry an energy-predicted baryon budget.
        #: ``"none"`` is the default and asserts no coupling at all.
        self.feedback = str(feedback)
        #: The halo mass at which the fitted anchor keeps its meaning.
        self.feedback_pivot = float(feedback_pivot)
        #: The peers the budget reads; ``None`` unless a closure was asked for.
        self.agn, self.galaxies = agn, galaxies
        #: Optional (0.9.5): the neutral gas the closure may not expel.
        #: Without it the budget treats every non-stellar baryon as expellable.
        self.coldgas = coldgas
        from .energetics import EnergeticsParams
        self.energetics = EnergeticsParams() if energetics is None else energetics

    # -- geometry ------------------------------------------------------------
    @staticmethod
    def _m12(m, cosmo):
        r""":math:`M_{200}/10^{12}M_\odot`, with a **physical** mass.

        ``m`` is in :math:`M_\odot/h`, so the :math:`h` has to be divided out.
        The predecessor did not, at four sites, while its docstrings disagreed
        about which convention was meant.
        """
        return jnp.asarray(m) / (1e12 * cosmo.h)

    @staticmethod
    def _per_halo(conc, like):
        """``conc`` on the mass axis of ``like``: ``(NM, 1)`` against an
        ``(NM, n)`` radius array, unchanged against a 1-D one."""
        c = jnp.atleast_1d(jnp.asarray(conc))
        return c[..., None] if jnp.ndim(like) > 1 else c

    @staticmethod
    def _shape_ratio(x, a_in, a_tr, a_out, c):
        r""":math:`f(x)/f(0.3\,c)` -- the profile normalised at
        :math:`0.3R_\Delta`, which is where these amplitudes are quoted.
        ``c`` is the halo's concentration, shaped like ``x``'s mass axis."""
        return (gnfw_shape(x, a_in, a_tr, a_out)
                / gnfw_shape(0.3 * c, a_in, a_tr, a_out))

    def _x(self, r, r_delta, conc):
        r"""r/R_s with :math:`R_s = R_\Delta/c(M,z)`, the halo's own scale
        radius, broadcasting over the mass axis."""
        r_s = jnp.atleast_1d(r_delta) / jnp.atleast_1d(jnp.asarray(conc))
        return jnp.asarray(r) / r_s[..., None] if jnp.ndim(r) > 1 \
            else jnp.asarray(r) / r_s

    # -- the three profiles --------------------------------------------------
    def n_e(self, r, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
            amplitude=None):
        r""":math:`n_e(r|M,z)` [cm^-3].  ``r`` shape ``(NM, n)``.

        ``conc`` is the halo concentration :math:`c(M,z)`, shape ``(NM,)``, in
        the mass definition ``mdef`` -- a field's ``conc``, which is what
        :meth:`weights` passes.  It is required, and every method below takes it
        the same way: the three profiles share the halo's scale radius
        :math:`R_\Delta/c`, and a default here would be a second convention.
        """
        x = self._x(r, self._r_delta(m, z, cosmo, mdef), conc)
        return (jnp.power(10.0, p.log10_ne_anchor + _lg(amplitude))
                * self._shape_ratio(x, p.alpha_in_n, p.alpha_tr_n,
                                    p.alpha_out_n, self._per_halo(conc, r))
                * jnp.power(hubble_e(z, cosmo), p.gamma_n)
                * jnp.power(self._m12(m, cosmo), p.beta_n)[..., None])

    def pressure(self, r, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                 amplitude=None):
        r""":math:`P_e(r|M,z)` [keV cm^-3].

        The :math:`k_B` conversion is here and nowhere else: the stored
        :math:`P_{0.3}` is :math:`P/k_B` in :math:`{\rm cm^{-3}\,K}`, as
        published.
        """
        m12 = self._m12(m, cosmo)
        # DPM Eq. 5: the outer slope may itself depend on mass.
        a_out = p.alpha_out_p + p.alpha_out_var * jnp.log10(m12)
        x = self._x(r, self._r_delta(m, z, cosmo, mdef), conc)
        return (jnp.power(10.0, p.log10_pe_anchor + _lg(amplitude)) * C.K_B_KEV_PER_K
                * self._shape_ratio(x, p.alpha_in_p, p.alpha_tr_p,
                                    a_out[..., None], self._per_halo(conc, r))
                * jnp.power(hubble_e(z, cosmo), p.gamma_p)
                * jnp.power(m12, p.beta_p)[..., None])

    def metallicity(self, r, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                   amplitude=None):
        r""":math:`Z(r)` [:math:`Z_\odot`].  No mass or redshift dependence in
        the amplitude; the halo's concentration sets the radius."""
        x = self._x(r, self._r_delta(m, z, cosmo, mdef), conc)
        return p.z_anchor * self._shape_ratio(x, p.alpha_in_z, p.alpha_tr_z,
                                          p.alpha_out_z, self._per_halo(conc, r))

    def temperature(self, r, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                    amplitude=None):
        r""":math:`kT = P_e/n_e` [keV].

        *Derived*, not parameterised: there is no polytropic index in the DPM,
        the radial temperature shape follows from the two profiles' ratio, and
        both use the same halo concentration so the ratio is meaningful.
        """
        # The amplitude cancels here and that is the point of passing it to
        # both: kT is invariant under a shared rescaling, so an energy-driven
        # gas mass cannot silently de-calibrate the kT-M relation.
        return (self.pressure(r, m, z, cosmo, p, mdef, conc=conc,
                              amplitude=amplitude)
                / jnp.maximum(
                    self.n_e(r, m, z, cosmo, p, mdef, conc=conc,
                             amplitude=amplitude),
                    1e-40))

    def emissivity(self, r, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                   amplitude=None):
        r""":math:`\varepsilon = n_e^2\,\Lambda(kT, Z)\,b_\sigma`."""
        ne = self.n_e(r, m, z, cosmo, p, mdef, conc=conc, amplitude=amplitude)
        kt = (self.pressure(r, m, z, cosmo, p, mdef, conc=conc,
                            amplitude=amplitude)
              / jnp.maximum(ne, 1e-40))
        zz = self.metallicity(r, m, z, cosmo, p, mdef, conc=conc)
        lam = self._lambda(kt, zz, p)
        return ne ** 2 * self._scatter_boost(p) * lam

    @staticmethod
    def _scatter_boost(p):
        r""":math:`\exp[(\sigma\ln 10)^2]`, DPM Eq. 6.

        A ``jnp`` expression of a traced leaf.  The predecessor wrote it as
        ``float(np.exp(...))``, which makes :math:`\partial/\partial\sigma`
        structurally zero.
        """
        return jnp.exp((p.sigma_scatter * jnp.log(10.0)) ** 2)

    def _lambda(self, kt, z_metal, p):
        if self._cooling is None:
            from .cooling import make_cooling
            self._cooling = make_cooling("apec")
        return self._cooling(kt, z_metal, nh_over_ne=p.nh_over_ne)

    # -- radii ---------------------------------------------------------------
    @staticmethod
    def _r_delta(m, z, cosmo, mdef: str = None):
        r""":math:`R_\Delta` [comoving Mpc/h] at the package's mass definition.

        **This used to be :math:`R_{200c}`, hard-coded, whatever the halo field
        declared.**  Oppenheimer et al. quote the DPM against a critical
        overdensity -- their Sec. 2 says so explicitly -- so the formula was
        right and the *mass* was not: a field at 200m hands over an
        :math:`M_{200{\rm m}}`, and computing a critical-overdensity radius
        from it is the same category error the halo layer refuses one rung up.
        At the shipped default that radius was wrong by a factor
        :math:`(M_{200{\rm m}}/M_{200{\rm c}})^{1/3} \simeq 1.12`.

        The package now carries **one** mass definition end to end, and this
        reads it rather than choosing its own.  The consequence is that the
        parameters below are no longer Oppenheimer's, which is why none of them
        is named as his are -- see :class:`DpmParams`.
        """
        from ..halos.mass_definitions import parse_mass_def

        md = parse_mass_def(mdef if mdef is not None else _default_mdef())
        return md.radius(jnp.asarray(m), z, cosmo)

    # -- gas mass and f_gas --------------------------------------------------
    def gas_mass(self, m, z, cosmo, p: DpmParams, aperture=None, *, conc,
                 amplitude=None,
                 mdef=None):
        r""":math:`M_{\rm gas} = \mu_e m_p \int_0^{a R_\Delta} n_e\,4\pi r^2 dr` [Msun/h].

        The integral is in **proper** cm -- a density in cm^-3 multiplied by a
        comoving volume is not a mass.  ``r_delta`` is comoving Mpc/h, so the
        conversion carries both :math:`h` and :math:`(1+z)`.

        This view does not exist anywhere in the predecessor: there is no
        ``mu_e``, ``m_p``, ``M_gas`` or ``gas_mass`` symbol in it.  It is what
        makes the DPM *the* gas sector rather than one of two, because it is
        what lets the lensing leg see the same gas the X-rays do.

        Parameters
        ----------
        aperture : float
            Integration radius in units of :math:`R_\Delta`.  **Defaults to 1,
            not to the profile's truncation radius**, and the distinction is
            not cosmetic: :math:`r_{\max} = 1.81R_\Delta` is the extent over which
            the DPM *profile* is calibrated, but a baryon budget compared
            against the cosmic share,
            :func:`~ggah_mod.sectors.matter.cosmic_baryon_fraction`, has to be
            taken over the halo.

            Measured, at the published Model 1 parameters: within
            :math:`R_\Delta` the gas mass fraction is 0.154 against a cosmic
            0.160 -- 96%, which is what a self-similar model with no feedback
            *should* give, and is the check that this unit chain is right.  Out
            to :math:`1.81R_\Delta` the same profile gives 0.36, i.e. 2.3x the
            available baryons, because it is then counting gas that is not
            bound to the halo.

            For Model 3 the choice is not a refinement but the whole answer:
            its outer slope is 0.5, so :math:`\int r^2\rho\,dr` **diverges**
            and the "gas mass" is whatever the aperture says it is.
        """
        m = jnp.atleast_1d(jnp.asarray(m))
        r_delta = self._r_delta(m, z, cosmo, mdef)
        aperture = p.aperture if aperture is None else aperture
        r_max = aperture * r_delta

        # Gauss-Legendre in comoving Mpc/h, converted once at the end.
        from ..halos.profiles import _leggauss_cached
        x_gl, w_gl = _leggauss_cached(self.n_gl)
        half = 0.5 * r_max
        r_nodes = half[:, None] * (jnp.asarray(x_gl) + 1.0)[None, :]
        ne = self.n_e(r_nodes, m, z, cosmo, p, mdef, conc=conc,
                      amplitude=amplitude)
        integral = jnp.sum(half[:, None] * jnp.asarray(w_gl)[None, :]
                           * ne * r_nodes ** 2, axis=-1) * 4.0 * jnp.pi

        # (Mpc/h)^3 comoving -> proper cm^3, then g -> Msun/h.  The whole
        # conversion is one pre-folded constant (see
        # `_NE_INTEGRAL_TO_MSUN_H`); forming `(MPC_CM / (h(1+z)))**3` on the
        # way, as this did, overflows float32 and returns `inf`.  The result,
        # ~1e13 Msun/h, is representable in single precision -- only the
        # intermediate was not, so this needs no precision guard.
        return (_NE_INTEGRAL_TO_MSUN_H * integral
                / (cosmo.h ** 2 * (1.0 + z) ** 3))

    def f_gas(self, m, z, cosmo, p: DpmParams, aperture=None, *, conc,
              amplitude=None,
              mdef=None):
        r""":math:`M_{\rm gas}(M)/M` within ``aperture`` :math:`\times R_\Delta`.

        A **diagnostic** when the energy closure is in use: the closure defines
        :math:`f_{\rm gas}`, and the ratio of the two is the recorded statement
        about whether the X-ray amplitude and the feedback budget agree.  See
        :mod:`ggah_mod.sectors.energetics`.

        Nothing here forces :math:`f_{\rm gas} \le f_b^{\rm cosmic}`, and
        nothing should: a clip would give a plausible number with a dead
        gradient, and exceeding the budget is information -- it says the
        density normalisation and the aperture are inconsistent with the
        cosmology.  That is a fit to reject, not a number to repair.
        """
        aperture = p.aperture if aperture is None else aperture
        return (self.gas_mass(m, z, cosmo, p, aperture, conc=conc,
                              amplitude=amplitude, mdef=mdef)
                / jnp.atleast_1d(jnp.asarray(m)))

    # -- the four Fourier views ---------------------------------------------

    def amplitude_from_budget(self, m, z, cosmo, p: DpmParams, f_budget, *,
                              conc, m_pivot=1e14, aperture=None, mdef=None):
        r"""The shared amplitude :math:`A(M)` an energy-predicted budget asks for.

        .. math::

            A(M) = \frac{f_{\rm gas}^{\rm budget}(M)/f_{\rm gas}^{\rm profile}(M)}
                        {f_{\rm gas}^{\rm budget}(M_{\rm p})/f_{\rm gas}^{\rm profile}(M_{\rm p})}

        **Normalised at a pivot, and that is a decision rather than a detail.**
        The unnormalised ratio would make the profile carry the budget exactly
        at every mass, which sounds stronger and is: it fixes the X-ray
        amplitude to the feedback closure and leaves the data nothing to say
        about the level.  Dividing by the pivot keeps
        :attr:`DpmParams.log10_ne_anchor` and :attr:`DpmParams.log10_pe_anchor`
        meaning what they meant -- the amplitude at :math:`0.3R_\Delta` for a
        :math:`M_{\rm p}` halo, fitted to X-ray data -- and lets the energy
        supply only the **mass dependence**, which is the part a sigmoid was
        standing in for.

        It is also what keeps :math:`\epsilon_{\rm AGN}` and
        :math:`\epsilon_{\rm SN}` identifiable.  Against a free overall
        amplitude they would be degenerate with it; against a *shape* they are
        not, because the two channels scale as :math:`M_{\rm BH}(M)` and
        :math:`M_\star(M)` and those differ by a factor 30 across the grid.

        ``A = 1`` everywhere -- which is what a budget proportional to the
        profile gives -- reproduces the shipped model exactly.

        Returns
        -------
        array (NM,)
            Pass it to :meth:`n_e` and :meth:`pressure` *together*, never to one
            alone: sharing it is what leaves :math:`kT = P_e/n_e` invariant.
        """
        m = jnp.atleast_1d(jnp.asarray(m))
        f_prof = self.f_gas(m, z, cosmo, p, aperture, conc=conc, mdef=mdef)
        ratio = jnp.asarray(f_budget) / f_prof
        # Interpolate the RATIO at the pivot, not the budget -- dividing an
        # interpolated budget by a separately evaluated profile leaves a
        # residual where the two are exactly proportional, and `A = 1` has to
        # be exact for the identity test to mean anything.
        ratio_p = jnp.interp(jnp.log10(jnp.asarray(m_pivot)),
                             jnp.log10(m), ratio)
        return ratio / ratio_p

    def _uk(self, k, m, z, cosmo, p, integrand, mdef=None):
        r_delta = self._r_delta(jnp.atleast_1d(jnp.asarray(m)), z, cosmo, mdef)
        r_max = p.r_max_over_rdelta * r_delta
        return profile_uk_gl(k, r_max, integrand, n_gl=self.n_gl)

    def mass_uk(self, k, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                 amplitude=None):
        r""":math:`\tilde u_{\rm gas}(k|M)`, normalised so :math:`u(k\to0)=1`."""
        return self._uk(k, m, z, cosmo, p, mdef=mdef, integrand=
                        lambda r: self.n_e(r, m, z, cosmo, p, mdef, conc=conc,
                                           amplitude=amplitude))

    #: The gas mass profile and the electron-density profile have the same
    #: *shape*: rho_gas = mu_e m_p n_e, and the constant cancels in a
    #: normalised transform.  One function, two names, so a caller asking for
    #: either gets the same array rather than two that could drift.
    density_uk = mass_uk

    def pressure_uk(self, k, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                 amplitude=None):
        r""":math:`\tilde y(k|M)`, normalised."""
        return self._uk(k, m, z, cosmo, p, mdef=mdef, integrand=
                        lambda r: self.pressure(r, m, z, cosmo, p, mdef,
                                                conc=conc,
                                                amplitude=amplitude))

    def emissivity_uk(self, k, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                      amplitude=None):
        r""":math:`\tilde X(k|M)`, normalised."""
        return self._uk(k, m, z, cosmo, p, mdef=mdef, integrand=
                        lambda r: self.emissivity(r, m, z, cosmo, p, mdef,
                                                  conc=conc,
                                                  amplitude=amplitude))

    # -- amplitudes ----------------------------------------------------------
    def _shell_integral(self, integrand, m, z, cosmo, p, mdef, r_min, r_max):
        r""":math:`4\pi\int_{r_{\min}}^{r_{\max}} f(r)\,r^2\,dr` per halo, by
        Gauss-Legendre on the sector's ``n_gl`` nodes.

        ``r_max`` defaults to ``p.r_max_over_rdelta`` :math:`R_\Delta`, the
        aperture every volume integral of this sector used before 0.9.6;
        ``r_min`` to zero.  Both are comoving Mpc/h, a scalar or one per halo,
        so a caller can integrate to another mass definition's radius
        (:func:`~ggah_mod.halos.mass_definitions.translate_mass`) without a
        Param whose declared box would not hold it.  ``integrand`` maps the
        ``(n_m, n_gl)`` radius nodes to the function's values there.
        """
        from ..halos.profiles import _leggauss_cached
        m = jnp.atleast_1d(jnp.asarray(m))
        r_delta = self._r_delta(m, z, cosmo, mdef)
        hi = (p.r_max_over_rdelta * r_delta if r_max is None
              else jnp.broadcast_to(jnp.asarray(r_max, dtype=float), m.shape))
        lo = (jnp.zeros_like(hi) if r_min is None
              else jnp.broadcast_to(jnp.asarray(r_min, dtype=float), m.shape))
        x_gl, w_gl = _leggauss_cached(self.n_gl)
        half = 0.5 * (hi - lo)
        r_nodes = lo[:, None] + half[:, None] * (jnp.asarray(x_gl) + 1.0)[None, :]
        return jnp.sum(half[:, None] * jnp.asarray(w_gl)[None, :]
                       * integrand(r_nodes) * r_nodes ** 2,
                       axis=-1) * 4.0 * jnp.pi

    def y_amplitude(self, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                    amplitude=None, r_max=None, r_min=None):
        r"""Compton-:math:`y` volume integral, :math:`\int P_e\,dV\,\sigma_T/m_ec^2`
        [(Mpc/h)^2], within ``r_max`` (default ``p.r_max_over_rdelta``
        :math:`R_\Delta`; comoving Mpc/h, see :meth:`_shell_integral`)."""
        m = jnp.atleast_1d(jnp.asarray(m))
        integral = self._shell_integral(
            lambda r: self.pressure(r, m, z, cosmo, p, mdef, conc=conc,
                                    amplitude=amplitude),
            m, z, cosmo, p, mdef, r_min, r_max)
        cm_per_unit = C.MPC_CM / (cosmo.h * (1.0 + z))
        return (SIGMA_T_CM2 / ME_C2_KEV) * integral * cm_per_unit

    def x_ray_luminosity(self, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                         amplitude=None, r_max=None, r_min=None):
        r"""Band luminosity :math:`L_X = \int \varepsilon\,dV` [erg/s], within
        ``r_max`` (default ``p.r_max_over_rdelta`` :math:`R_\Delta`; comoving
        Mpc/h, see :meth:`_shell_integral`).

        Named ``x_ray_luminosity``, never ``lx``: the AGN sector's point
        source is :meth:`~ggah_mod.sectors.agn.AgnSector.l_x_agn`, and a bare
        ``lx`` would not say which of the two X-ray sources it is.
        """
        m = jnp.atleast_1d(jnp.asarray(m))
        integral = self._shell_integral(
            lambda r: self.emissivity(r, m, z, cosmo, p, mdef, conc=conc,
                                      amplitude=amplitude),
            m, z, cosmo, p, mdef, r_min, r_max)
        require_x64("x_ray_luminosity", order="1e44 erg/s",
                    in_range="Densities, pressures, temperatures, `gas_mass`, "
                             "`f_gas` and `y_amplitude` are all in range and do "
                             "not need this.")
        cm_per_unit = C.MPC_CM / (cosmo.h * (1.0 + z))
        return integral * cm_per_unit ** 3

    def x_ray_temperature(self, m, z, cosmo, p: DpmParams, mdef=None, *, conc,
                          amplitude=None, r_max=None, r_min=None):
        r"""The band-luminosity-weighted temperature [keV],

        .. math::

            kT_X = \frac{\int \varepsilon\,kT\,dV}{\int \varepsilon\,dV},

        with :math:`\varepsilon` the emissivity in the sector's cooling band
        (rest-frame 0.5-2 keV for the default ``apec`` table) and :math:`kT`
        the pointwise :math:`P_e/n_e`: the temperature a spectral fit of the
        integrated band emission would return if every shell's spectrum were
        its own temperature's.  ``r_min`` excises a core, as most measured
        :math:`T_X` do (0.15 :math:`R_{500c}`); ``r_max`` defaults to
        ``p.r_max_over_rdelta`` :math:`R_\Delta` (comoving Mpc/h, see
        :meth:`_shell_integral`).

        Weighted by the band emissivity :math:`n_e^2\Lambda(T, Z)` rather than
        the emission measure :math:`n_e^2`: the band's own :math:`\Lambda`
        falls with temperature above a keV, so the two weightings part at
        cluster temperatures.  The shared amplitude cancels in the ratio, as it
        does in :meth:`temperature`, so an energy-driven gas mass cannot
        de-calibrate the :math:`kT`-:math:`M` relation.
        """
        m = jnp.atleast_1d(jnp.asarray(m))

        def eps(r):
            return self.emissivity(r, m, z, cosmo, p, mdef, conc=conc,
                                   amplitude=amplitude)

        def eps_kt(r):
            return eps(r) * self.temperature(r, m, z, cosmo, p, mdef, conc=conc,
                                             amplitude=amplitude)

        num = self._shell_integral(eps_kt, m, z, cosmo, p, mdef, r_min, r_max)
        den = self._shell_integral(eps, m, z, cosmo, p, mdef, r_min, r_max)
        return num / jnp.maximum(den, 1e-300)

    # -- the sector contract -------------------------------------------------

    def feedback_budget(self, field, params: DpmParams, agn_params,
                        galaxies_params, coldgas_params=None):
        r""":math:`f_{\rm gas}` the energy closure predicts, on the mass grid.

        **Gas, not baryons.**  The closure returns the retained *baryon*
        fraction, stars included (:func:`~ggah_mod.sectors.energetics.f_retained_energy`
        says so in its name), and this used to hand that to
        :meth:`amplitude_from_budget` as a gas fraction.  The stars are
        subtracted here, and the closure itself caps what it can expel at the
        non-stellar baryons, so the result is at least the retained floor and
        never negative.  Neutral gas is subtracted only when the sector is
        built with a ``coldgas`` peer (0.9.5, below); without one it is not, and
        at the masses where the amplitude's shape is set it is a few per cent of
        the hot phase.

        None of the chain is new and that is the point: the AGN sector's mean
        black-hole mass gives
        :math:`E_{\rm AGN} = \epsilon_{\rm AGN}\epsilon_{\rm r}M_\bullet c^2`,
        the galaxy sector's stellar mass gives
        :math:`E_{\rm SN} = \epsilon_{\rm SN}M_\star/(1-R)\,e_{\rm SN}`, and
        :func:`~ggah_mod.sectors.energetics.f_retained_energy` balances the pair
        against :math:`\Delta f_b M v_\Delta^2(1-\eta_{\rm ej}^{-1})`.  What was
        missing was the wiring: the closure has been complete, parameterised and
        gradient-safe for a long time with **no production caller at all**.

        **Satellites are in** :math:`E_{\rm SN}`.  A halo's supernova budget is
        every star in it, so the stellar mass is
        ``stellar_fraction(..., satellites=True)`` -- the centrals plus the
        first moment of the same conditional stellar-mass function the AGN
        sector integrates for its own satellites.  There is no duty-cycle
        analogue, and that is exactly where the parallel with the AGN stops: an
        AGN is active or it is not, while every satellite's stars have already
        exploded.  :meth:`~ggah_mod.sectors.agn.AgnSector.omega_bh` sets the
        precedent, counting every satellite black hole whatever ``f_duty_sat``
        is, because a duty cycle is about activity and not about mass.

        **Every star, whatever the selection** (0.9.4).  A threshold
        occupation's parameters carry ``log10m_star_thresh``, the *sample's*
        cut, and :meth:`~ggah_mod.sectors.galaxies.GalaxySector.stellar_fraction`
        counts centrals above it.  Handed a sample's parameters -- which a
        galaxy x tSZ fit must do, for its galaxy leg -- the budget counted only
        the selected galaxies' stars: for M* > 10^11 it lost most of the
        supernova energy of 10^12 haloes and the closure amplitude there came
        out 2.2 times high, a halo's gas depending on which galaxies a survey
        chose.  The budget now reads the threshold at the floor of its declared
        box, 8.5 in h^-2 Msun, where it has converged (it is the same from 9.0
        down), so no caller can pass a selection into it.

        **And no neutral gas** (0.9.5), when the sector is built with a
        ``coldgas`` peer and handed its parameters: the closure is capped at
        the stars plus the neutral gas, and the neutral gas is subtracted from
        what it returns as well as the stars, so the result is the *hot* gas and
        is never negative.  Without the peer the arithmetic is 0.9.4's.

        The satellite term is truncated at the conditional stellar-mass
        function's fitted range, so this is a **lower bound** on
        :math:`E_{\rm SN}` -- a stated and quantified omission rather than a
        hidden one.  The black holes are counted from the AGN sector's own
        stellar-mass cut, :data:`~ggah_mod.sectors.agn.LG_MSTAR_MIN`, which is
        lower: the two terms do not start at the same stellar mass.
        """
        from . import energetics as E
        from .matter import cosmic_baryon_fraction
        agn, gal = self._peer_sectors()
        f_b = float(cosmic_baryon_fraction(field.cosmo))
        # Every star, not the selected sample's: the budget is what the halo's
        # stars and black holes released, whatever a survey later selected.
        everyone = _every_star(galaxies_params)
        f_cen, f_sat = gal.stellar_fraction(field, everyone, satellites=True)
        m_star = (f_cen + f_sat) * field.m
        m_bh = (agn.mean_mbh(jnp.log10(field.m), agn_params, everyone,
                             h=field.cosmo.h, z=field.z)
                + agn.mean_mbh_satellites(field, agn_params, everyone))
        m_cold = self._neutral_gas(field, coldgas_params, everyone)
        f_ret = E.f_retained_energy(field.m, field.z, field.cosmo, m_star, f_b,
                                    m_bh=m_bh, params=self.energetics,
                                    mdef=field.mdef, m_cold=m_cold)
        hot = f_ret - (f_cen + f_sat)
        return hot if m_cold is None else hot - m_cold / field.m

    def _neutral_gas(self, field, coldgas_params, galaxies_params):
        """Neutral gas per halo [Msun/h] for the closure, or ``None``.

        **The peer opts in.**  Built without ``coldgas`` the sector ignores
        cold-gas parameters -- layer 4 hands them to every gas component of a
        spectrum that carries them, a matter spectrum with neutral gas in it
        among others, and that must not change a gas sector that was not asked
        to see it.  Built with one, the parameters are required: a peer with no
        parameters would silently mean no neutral gas.
        """
        if self.coldgas is None:
            return None
        if coldgas_params is None:
            raise ValueError(
                "this gas sector was built with a cold-gas peer, so its closure "
                "floors the retained baryons at stars + neutral gas and needs "
                "the cold-gas parameters: pass `coldgas_params=`, or put a "
                "'coldgas' block in the spectrum's parameters")
        return self.coldgas.m_neutral(field, coldgas_params, galaxies_params)

    def _peer_sectors(self):
        """The two sectors the budget reads, refused when they were not given."""
        if self.agn is None or self.galaxies is None:
            raise ValueError(
                "feedback='closure' needs the AGN and galaxy *sectors*, not "
                "only their parameters: the budget is built from a black-hole "
                "mass and a stellar mass, which are the sectors' to compute.  "
                "Build it as HotGasDPM(feedback='closure', agn=..., "
                "galaxies=...), with the same instances the spectrum uses.")
        return self.agn, self.galaxies

    def _feedback_amplitude(self, field, params, agn_params, galaxies_params,
                            coldgas_params=None):
        """``A(M)`` from the feedback closure, or ``None`` when it is off.

        ``None`` rather than an array of ones: a log shift of exactly zero
        leaves every shipped number bit for bit, where multiplying by 1.0 is
        only almost that.
        """
        if self.feedback == "none":
            return None
        if agn_params is None or galaxies_params is None:
            raise ValueError(
                "feedback='closure' needs both the AGN and the galaxy "
                "parameters: the budget is the energy the black holes and the "
                "stars released, so it cannot be formed from the gas sector's "
                "own numbers.  Pass them through layer 4 -- they are declared "
                "in `spectra.tracers.OPTIONAL_PEERS` -- or build the sector "
                "with feedback='none', which asserts no coupling.")
        return self.amplitude_from_budget(
            field.m, field.z, field.cosmo, params,
            self.feedback_budget(field, params, agn_params, galaxies_params,
                                 coldgas_params),
            conc=field.conc, m_pivot=self.feedback_pivot, mdef=field.mdef)

    def weights(self, field, params: DpmParams,
                view: str = "pressure", *, agn_params=None,
                galaxies_params=None, coldgas_params=None) -> TracerWeights:
        """One of the four views, as :class:`TracerWeights`.

        Continuous: the gas is a field, so its one-halo auto-spectrum has no
        self-pair to exclude.

        **The peers are optional, and their absence is the default.**  With
        neither, the amplitudes are this sector's own free parameters and every
        shipped number is what it was.  With both, and a sector built with
        ``feedback="closure"``, the energy the AGN and the stars released sets
        the halo's baryon budget and the budget sets the amplitude's *mass
        dependence* through :meth:`amplitude_from_budget` -- the level staying
        with the fitted anchor, which is what keeps the two efficiencies
        identifiable rather than degenerate with an overall normalisation.

        A gas spectrum with no galaxy sector in it is the case this package
        exists to make possible, so requiring peers here would take that away
        to buy a coupling nobody asked for on that call.
        """
        views = {"pressure": (self.pressure_uk, self.y_amplitude),
                 "xray": (self.emissivity_uk, self.x_ray_luminosity),
                 "mass": (self.mass_uk, self.gas_mass),
                 "density": (self.density_uk, self.gas_mass)}
        # The mass view's amplitude is taken over R_Delta, its profile over
        # r_max: the first is a budget, the second is a shape.
        if view not in views:
            raise ValueError(f"unknown gas view {view!r}; expected one of "
                             f"{sorted(views)}")
        uk_fn, amp_fn = views[view]
        m, z, cosmo = field.m, field.z, field.cosmo
        a = self._feedback_amplitude(field, params, agn_params, galaxies_params,
                                     coldgas_params)
        # The field's own definition and concentration, not this sector's idea
        # of either: the profiles sit on the halo's scale radius r_delta/c.
        amp = amp_fn(m, z, cosmo, params, mdef=field.mdef, conc=field.conc,
                     amplitude=a)
        uk = uk_fn(field.k, m, z, cosmo, params, mdef=field.mdef,
                   conc=field.conc, amplitude=a)
        # The baryons below the mass grid are not in a hot atmosphere -- haloes
        # under 1e10 Msun/h hold none (Okamoto et al. 2008) -- so none of this
        # sector's four views continues below it.  The ejecta sector books those
        # baryons as diffuse gas; without this the counterterm would extrapolate
        # the DPM's pressure, emissivity and mass per unit halo mass to masses it
        # was never fitted at.
        return TracerWeights(w_point=None, w_extended=amp[None, :] * uk,
                             norm=jnp.asarray(1.0), discrete=False,
                             bias_weight=None, name=f"gas:{view}",
                             w_unresolved=jnp.zeros(uk.shape[0]))
