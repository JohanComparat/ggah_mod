# Changelog

## 1.1.0.dev2 (unreleased)

**Changed: the eight fitted `AgnParams` defaults are the MAP of a fit to
data** -- eROSITA AGN in DESI Legacy Survey hosts at 0.2 < z < 0.5 (ggah_cal's
eROSITA c030 fit: six flux-limited samples, F(0.2-2.3 keV) > 2.5e-14, in three
bins of intrinsic 2-10 keV luminosity, their n_bar, w(theta), w_p and Delta
Sigma, and the luminosity function of all AGN in the same bins; 70 points),
fitted on 1.1.0.dev1's galaxies: chi2 103.2 for 62 dof. They replace the MAP
of a fit to a synthetic vector (0.8.7-1.1.0.dev1).

| | `mu_bh` | `al_bh` | `sig_bh` | `log10_lstar` | `delta1` | `delta2` | `log10_ferdf` | `f_duty_sat` |
|---|---|---|---|---|---|---|---|---|
| 1.1.0.dev2 | 7.954 | 0.842 | 0.280 | -0.798 | 0.232 | 3.749 | -1.640 | 0.296 |
| 0.8.7-1.1.0.dev1 | 7.561 | 0.974 | 0.086 | -0.981 | 0.198 | 2.376 | -1.000 | 0.052 |

- The data constrain `al_bh` (0.82 +- 0.07 in the posterior), `log10_ferdf`
  (-1.64 +- 0.19) and `f_duty_sat` (0.33 +- 0.16); the other five stay within
  one standard deviation of their Gaussian priors, the published values.
- What it moves, on the 1.1.0 galaxies at z = 0.135 and L_X > 1e42:
  satellites are 13.4% of the AGN (2.2%), b_eff 0.918 against 0.789 for
  centrals alone (0.925, 0.904). Omega_BH at z = 0 is 3.57e-6, 0.80 of
  Fukugita & Peebles (2004) as a share of Omega_b, against 9.22e-7 (0.21);
  the satellites add 50% of it.
- Through the feedback budget (Soltan channel, E_AGN of every black hole):
  E_AGN/E_SN is 2.42 at 1e11 Msun/h and 0.99 at 1e15 (0.49, 0.42), and the
  two channels reach 3.59 times the binding energy (1.30). On the census's
  chain at z = 0 the hot gas falls from 26.6% to 23.2% of Omega_b and the
  ejected gas rises from 20.6% to 24.0%; S(k) with the ejected shell is 0.964
  at k = 1 h/Mpc and 0.781 at k = 10 (0.969, 0.784). The joint gas fits run
  with feedback off and do not see it.
- Tests that pinned numbers at the old defaults are re-measured.

**Changed: `GalaxyParams.fc` is bounded by one, (0.1, 3) -> (0.1, 1), and its
kind is definitional.** `fc` is the high-mass limit of the zumandelbaum
central occupation, a completeness, and a mean number of centrals cannot
exceed one; the (0.1, 3) box was Zu & Mandelbaum's prior on their own `f_c`,
a satellite concentration ratio. Every other central amplitude in the
registry was already at most one. The default, 0.874, is inside the new box,
so the 1.1.0.dev1 MAP is unchanged and no default moves; ggah_cal's
`PARAM_BOX` follows.

**Added: the galaxy registries of layer 3 are documented** on a docs page,
`layer3_galaxies`, and its notebook: the occupation models besides
`zumandelbaum15`, the two conditional luminosity functions, the stellar-mass
relations written forwards and the calibration tables. It replaces Appendix A
of the technical paper. The supernova constants of `energetics` (`e_SN`, `R`,
`eps_sn`) now cite their sources; no value changes.

## 1.1.0.dev1 (unreleased)

For the joint tSZ + X-ray fit of the hot gas: two bugs fixed, the published DPM
parameter sets removed, and the gas defaults calibrated on observations.
**Breaking:** `dpm_model_params` and `DPM_MODELS` are gone, and `DpmParams()`
no longer returns the same numbers; nor does `GalaxyParams()`, refitted on
sum_stat 0.6.0. Every other addition defaults to the shipped behaviour.

- **Removed: the three published DPM parameter sets** (`dpm_model_params`,
  `DPM_MODELS`). The DPM *form* (Oppenheimer et al. 2025) stays; its paper's
  Models 1-3 are tuned to one sample of low-mass systems and are no longer a
  default, a base or a benchmark anywhere. A caller that used them passes a
  `DpmParams` of its own.
- **Changed: `DpmParams()` defaults are an observational calibration**, the
  posterior mean of every parameter but `aperture` fitted to X-COP density,
  pressure, temperature, Fe and gas-fraction measurements, Arnaud et al.
  (2010), the Planck Y-M relation, group and cluster gas fractions and
  temperatures, and L_X from galaxy haloes to clusters (ggah_cal
  `scripts/47_gas_prior_from_observations.py`), with every effective Eq. 5
  slope inside its physical limits and the gas within the cosmic baryon share
  over 1e10-1e16 Msun/h as one-sided walls, and one hydrostatic mass bias on
  the hydrostatic-mass targets (1-b = 0.70 +- 0.02). chi2 178 over 83
  measurements at the mean, against 73 with neither constraint nor bias; the
  budget is exceeded only at 10^16 Msun/h (1.03 of the cosmic share). At the
  defaults kT(0.3 R_Delta) is 4.1 keV at 10^15 Msun/h and the gas fraction
  within R_Delta rises from 0.33 to 0.67 of the cosmic share over 10^12-10^15;
  in groups and clusters the pressure peaks at 0.03-0.04 R500c and falls
  inward.
- **Changed (1.1.0.dev1): the `zumandelbaum` iHOD defaults are refitted on sum_stat 0.6.0**
  (`GalaxyParams`, `DEFAULTS["zumandelbaum15"]` and both `zumandelbaum16`
  colours, through `occupation._ZU15_LS10`). sum_stat 0.6.0 re-measured the
  LS10 products: its survey area raises every n_bar by 11-12 per cent and its
  jackknife makes the w_p errors 0.2-0.7 times smaller. Against them the 0.8.5
  defaults give chi2 = 445 for 105 dof (17.98 on the old products with this
  code, so the code is not what moved). The new values are the MAP of ggah_cal
  `massbins_zu15_gt10.5_nbar-wp`, n_bar + w_p of five bins, 10.6-12.0 log10
  Msun, 0.05 < z < 0.18, from the old MAP: chi2 = 152.91 for 105 dof, n_bar of
  every bin to 1.3 per cent. Rounded to three decimals: lg_m1h 12.289,
  lg_m0star 10.331, beta 0.765, delta 0.781, gamma 0.496, sigma_lnmstar
  0.538, eta -0.125, fc 0.874, bsat 12.53, beta_sat 0.856, bcut 0.750,
  beta_cut 0.669, alpha_sat 1.083; threshold 10.157 as before. In units of the
  new Laplace errors the largest moves are bcut (-3.8 sigma) and bsat and
  alpha_sat (+1.9). The 0.8.5 values are `occupation.ZU15_LS10_085`; the
  quenching defaults stay Paper III's.

- **Fixed: `hankel` on a spectrum that changes sign.** The log-cubic
  interpolation floored non-positive nodes at `peak * 1e-300` and overshot the
  690-e-fold step: on a pressure profile with a central depression
  (`alpha_in_p < 0`), whose `P_gy` goes negative at high k, `w(1.9')` came out
  as -6.0e3 in float64 against a brute-force 4.1e-7 (float32 survived on its
  smaller floor). Queries whose cubic stencil touches a non-positive node now
  take the cubic of the value (`transforms._signed_fallback`), good to 6e-4
  against a closed-form sign-changing pair on `DEFAULT_WTHETA_ELL`. A positive
  spectrum is unchanged bit for bit.
- **Added: the five other DPM Eq. 5 mass slopes**, `alpha_{in,tr,out}_n_var`
  and `alpha_{in,tr}_p_var`, beside `alpha_out_var`: each shape parameter of
  the density and pressure profiles may now run linearly in lg M12.
- **Fixed: `limber.c_ell` on a spectrum that changes sign**, the same defect
  as `hankel`'s one step earlier: `log P` interpolated through the floor. An
  X-ray emissivity whose temperature rises outward through the steep low-T
  part of Lambda(T) is hollow, its cross power spectrum goes negative at high
  k, and ggah_cal's galaxy x X-ray `w(theta)` came out at +-1e14. It now takes
  `_signed_fallback` too; a closed-form test with a sign change past one
  multipole fails on 1.0.0 and passes here.
- **Added: `DpmParams.log10_conc_ratio` and `log10_conc_ratio_var`** (default
  0): the gas scale radius against the halo's, `c_gas = c(M,z)
  10^(r + r_var lg M12)`. Observed cluster pressure profiles have
  `c500 ~ 1.2`, about a quarter of the halo's; with the gas pinned to the
  dark matter's scale radius a calibration on observations railed nine
  parameters faking it. The six slope-variation boxes are widened to +-2 per
  dex for the same calibration.
- **Added: `make_cooling("apec_wide")`** (`ApecCoolingWide`, built by
  `python -m ggah_mod.sectors.cooling --wide`): the shipped 0.5-2 keV table's
  nodes plus 22 below 0.08 keV, so identical above 0.088 keV and following APEC
  below instead of clamping. The clamp overstated Lambda by 8, 480 and 1e8
  times at 0.06, 0.04 and 0.02 keV.
- **Added: `HotGasDPM(scatter="isobaric")`**, the published DPM's log-normal of
  density at the local pressure, summed by Gauss-Hermite in the X-ray
  emissivity and its emission-weighted temperature. Pressure, density and so
  y and the gas mass are untouched; with Lambda constant it is the shipped
  `"constant"` boost, at sigma = 0 the unscattered emissivity.


- **The BAO sound horizon.**
  - `sound_horizon(z, cosmo)` integrates $r_s(z)$ exactly over the package's own
    $E(z)$; given CLASS's $z_d$ it is CLASS's `rs_d` to $1\times10^{-7}$.
  - `z_drag(cosmo)` is a cubic fit to CLASS 3.3.4's drag redshift, accurate to 0.017
    on held-out points and 0.029 at the box's corners; it is calibrated by
    `tools/calibrate_zdrag.py` over the `emu_pk` box with $T_{\rm CMB}\pm1\%$.
  - `r_drag(cosmo)` combines the two, in $h^{-1}$Mpc, and is differentiable in every
    parameter. In flat cosmologies it is CLASS's `rs_d` to $4.5\times10^{-6}$.
  - All three are in `ggah_mod.cosmology` (new module `ggah_mod.cosmology.drag`).
  - `r_drag` refuses cosmologies outside the calibration box, and those where dark
    energy is more than $10^{-3}$ of the density at the drag epoch.
  - The BBN assumptions behind the helium fraction are recorded with the
    coefficients (`_zdrag_coefficients.BBN`) and guarded by a slow test.
- The cobaya theory provides `rdrag`, in Mpc, which is what cobaya's BAO
  likelihoods (`bao.desi_dr2` and the rest) request.
- The `ClassPk.SBBN_FILE` comment now gives the measured agreement of the 2017 and
  2025 BBN tables, $1.2\times10^{-4}$ in $Y_{\rm He}$; it said $4\times10^{-4}$.

- **Fixed: wrong arctangents on jaxlib 0.10.** jaxlib 0.10.2, the newest that
  supports Python 3.11, miscompiles `jnp.arctan` on a CPU with vector
  instructions: from 64 elements on, half the outputs come back zero and most of
  the rest are wrong, and inside a fused kernel it fails on a dozen. The
  Hernquist $\Sigma$ and $\Delta\Sigma$ came out negative, off by up to $10^3$;
  the NFW and BMO lensing kernels (`_nfw_f_of_x`, `_bmo_ff`, `_m_bmo_dl`), the
  CLF faint-end slope (`alpha_faint_cacciato09`) and the AGN
  `compton_thick_fraction` call the same primitive. All seven call sites now use
  `numerics.arctan`, built from `arcsin` and `arccos` and equal to numpy's to
  round-off in value and gradient; `tests/test_numerics.py` refuses `jnp.arctan`
  anywhere in the package. On jaxlib 0.11 nothing changes beyond round-off.

## 1.0.0 (2026-09-29)

The first public release.

- **Six layers**: the cosmology (`ggah_mod.cosmology`), the haloes
  (`ggah_mod.halos`), the tracers (`ggah_mod.sectors`), the power spectra of
  pairs of tracers (`ggah_mod.spectra`), the projected observables
  (`ggah_mod.observables`) and their covariances (`ggah_mod.covariance`), with
  adapters in `ggah_mod.interfaces`.
- **Two flavours** of one model, `ACCURATE` (CLASS or CAMB) and
  `DIFFERENTIABLE` (the `emu_pk` emulator, differentiable end to end), with
  `DIFFERENTIABLE_COARSE`.
- **Documentation** at <https://ggah-mod.readthedocs.io> for layers 1 and 2: an
  overview, one page and one executed notebook per layer, and their API.
  Layers 3 to 6 and the interfaces are released as code and are work in
  progress: importing them emits a `ggah_mod.WorkInProgressWarning` once.
- Requires `emu_pk >= 2.0.1`, which restates this package's neutrino
  convention. The `reference` extra installs CLASS 3.3 (`classy >=3.3.4,<3.4`),
  the version every accurate-path measurement was made with.
- **Installation from PyPI**, `pip install ggah_mod`, with the extras
  `reference`, `backends`, `cobaya`, `tables`, `dev` and `docs`.
- Docstring corrections in layers 1 and 2: the enclosed-mass formula of
  `profiles.ejected_mass_within` restored, and statements that had gone stale
  about the `emu_pk` curvature axis and the number of multiplicity functions
  corrected.

Known limitations:

- `emu_pk` is scored for $k \ge 10^{-3}\,h\,{\rm Mpc}^{-1}$ while the
  `DIFFERENTIABLE` grid starts at $10^{-4}$; in that decade the transfer function
  recovered from its $P_{\rm cb}$ is not monotone, at the $2\times10^{-3}$ level.
  The test that checks it is marked as an expected failure until the emulator's
  range covers it.
- 0.9.8 raised the `ACCURATE` flavour's CLASS precision while `emu_pk` reproduces
  CLASS at its defaults, so the flavours differ by that precision's error. Twelve
  rows of the parity budget between them, measured before, are marked as
  expected failures until they are re-measured.
