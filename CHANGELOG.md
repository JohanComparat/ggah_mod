# Changelog

## 1.4.0 (2026-10-13)

**Changed: the AGN sector takes the central's stellar-mass scatter from the
galaxy sector.**  `AgnParams.sigma_ms`, the scatter of a central's
log10 M_* at fixed halo mass held at 0.20 dex, is removed.  The AGN sector
already read the galaxy sector's mean relation M_*^c(M_h); it now reads the
scatter about it too, `GalaxySector.log10_mstar_scatter` (new): for
`zumandelbaum15` and `zumandelbaum16_*`, sigma_lnM*(M_h)/ln 10 of Zu &
Mandelbaum (2015, Eq. 20), 0.32 dex up to M_1 and 0.20 dex at 10^14 Msun/h at
the 1.2.0 galaxy defaults; `sigma_logmstar` for `leauthaud12`; and
`width_logmstar`/sqrt(2) for `zacharegkas25`.  `AgnSector.sigma_mstar` returns
it, and it sets the central fraction above the stellar-mass cut and the widths
of log M_BH (`_sigma_lm`) and of the luminosity kernel (`_sigma_lx`) at each
halo mass, so the central's kernel is one per halo mass (`_kernel` takes a width
per halo mass; one matrix product convolves the Eddington-ratio distribution
with all of them).

At the option-B MAPs of the eROSITA x DESI YR3 fits chi2 changes by -0.006
(Fit 1) and +0.004 (Fit 2): every luminosity observable reads the scatter
through (al_bh + log10_lstar_mslope)^2 = 0.004.  The quantities read from M_BH
move: the lognormal mean of a central's black hole is up to 1.20 times the
1.3 one, and the satellites' share of Omega_BH falls from 29.6 to 25.6 per cent.

## 1.3.1 (2026-10-13)

**Fixed: the AGN luminosity kernel runs on every GPU.**
`AgnSector._kernel` convolved the Eddington-ratio distribution with its
Gaussian through `jnp.convolve`, which lowers to a cuDNN convolution; in
float64, cuDNN 9.24 fails to execute it on a V100 (status 5003, with or without
autotuning).  The convolution is now a matrix product (`_convolve_full`), which
agrees with `numpy.convolve` to 6e-16; the log posterior of the eROSITA x DESI
YR3 fit at its MAP changes by 1e-8.

## 1.3.0 (2026-10-13)

**Changed: the AGN defaults are the MAP of the eROSITA x DESI YR3 fit.**
The fit (ggah_cal, option B of `docs/agn_compare.rst`) uses the photometric
eROSITA AGN samples in three bins of 2-10 keV luminosity (the Aird et al. 2015
XLF in their observed luminosity, n_bar and w(theta)), their cross-correlations
with Legacy Survey galaxies of M_* > 10^10 Msun at 0.05 < z < 0.18, and
p(lambda_SAR | M_*, z) of Igo et al. (2026) with a 0.30 dex floor, on the 1.2.0
galaxy defaults: 135 points, chi2 = 115.5 for 120 degrees of freedom, fifteen
free parameters, with the duty cycle capped at one (below) and both stellar-mass
slopes free. They replace the eROSITA c030 MAP of 1.1.0.dev2 (DESI DR1
redshifts, unobscured AGN only, chi2 = 103.2 for 62 dof).

| parameter | 1.1.0.dev2-1.2.0 | 1.3.0.dev0 |
|---|---|---|
| `mu_bh` | 7.954 | 7.601 |
| `al_bh` | 0.842 | 1.061 |
| `sig_bh` | 0.280 | 0.597 |
| `log10_lstar` | -0.798 | -0.999 |
| `log10_lstar_mslope` | 0 | -1.123 |
| `delta1` | 0.232 | 0.370 |
| `delta2` | 3.749 | 2.581 |
| `log10_ferdf` | -1.640 | -0.700 |
| `log10_ferdf_mslope` | 0 | 1.401 |
| `gamma_x` | 1.8 | 1.723 |
| `f_duty_sat` | 0.296 | 0.321 |
| `ObscurationParams.f_faint_norm` | 0.9 | 1.0 |
| `ObscurationParams.l_blend_0` | 43.2 | 43.184 |
| `ObscurationParams.l_blend_width` | 0.6 | 0.691 |
| `ObscurationParams.l_blend_m_amp` | 0 | 0.119 |

- **The prior of `al_bh` is now Reines & Volonteri (2015)'s, N(1.05, 0.11)**,
  for AGN hosts and their total stellar mass, in place of Powell et al. (2022)
  Model 1's N(0.67, 0.24); `mu_bh` and `sig_bh` keep Powell's. Every luminosity
  observable reads `al_bh` and `log10_lstar_mslope` only through their sum, so
  the prior sets the black-hole relation's slope and the data the break's.
  Holding `log10_lstar_mslope` at zero instead (option C) gives al_bh = 0.37,
  flatter than every measured relation, and fits worse (-2 ln P 132.1 against
  121.6).
- The same fit to the spectroscopic samples (delta2 free) has chi2 = 98.6 for
  108 dof and agrees with it within 1.1 sigma on each of the fourteen shared
  parameters (D^2 = 6.5, p = 0.95).
- `GAMMA_X` stays 1.8: it is the catalogue's flux-to-luminosity convention
  (`catalogue_gamma`), not the AGN spectrum.
- `OBSCURATION_PUBLISHED` (new) holds Comparat et al. (2019)'s nine
  coefficients; `ObscurationParams(**OBSCURATION_PUBLISHED)` is their Fig. 6.
- At the new defaults and the `zumandelbaum15` galaxy defaults (z = 0.135,
  L_X > 10^42), satellites are 9.96% of the AGN (7.2% before) and b_eff is
  1.109 (0.895), 1.015 (0.822) for centrals alone; Omega_BH at z = 0 is
  1.87e-6 on a 10^10-10^16 Msun/h halo field (2.63e-6 at the old defaults).
  The duty cycle reaches one at M* = 10^11.0 Msun, and the central AGN
  occupation peaks, at 0.28 near 10^13 Msun/h: the luminosity at fixed
  Eddington ratio falls slowly with stellar mass (al_bh + log10_lstar_mslope =
  -0.06).
- Earlier defaults: the MAP of a fit to a synthetic vector
  (`agn_mock_z0p202_g086`) in 0.8.7-1.1.0.dev1, the published values up to 0.8.6.

**Changed: the AGN duty cycle is capped at one.** `agn.duty` is now the smooth
minimum -ln(exp(-k f) + exp(-k))/k of the uncapped fraction f and one, with
k = `DUTY_CEILING_SHARPNESS` = 64 (new, exported); `agn.log10_duty` is its
logarithm. With a slope the uncapped fraction exceeds one away from the pivot:
at the eROSITA x DESI YR3 MAP of the uncapped model (log10_ferdf = -0.779,
slope 0.934) above log M* = 11.33, reaching 3.2 at the top of the central grid;
at the defaults above 11.00.
k is a power of two, so the products by k are exact and the capped fraction is
the uncapped one to the last bit wherever that is below 0.469: at slope zero and
log10_ferdf < -0.33 every output is unchanged bit for bit. At f = 1 the cap gives
0.989, and one to double precision above f = 2. `log10_ferdf` is now read by
`duty` alone (a test walks the syntax tree to keep it so).

**Added: a stellar-mass-dependent Eddington-ratio break, `AgnParams.log10_lstar_mslope`.**
The break of the ERDF was one number at each redshift, `lstar_of_z`; it is now

    log10 lambda*(M*, z) = log10_lstar(z) + log10_lstar_mslope * (log10 M* - 10.5)

(`agn.lstar_of_mstar`, also `AgnSector.lstar_of_mstar`), with M* physical in
Msun and the duty cycle's pivot (`LG_MSTAR_LSTAR_PIVOT` = `LG_MSTAR_DUTY_PIVOT`
= 10.5): a central's at its halo's mean stellar mass M*_c(M_h), the one
`mean_log10_mbh` reads, and a satellite's at its own stellar mass. Neutral at 0,
bounds (-3, 3), Flat, kind "physical". The fits to eROSITA and to Igo et al.
(2026) constrain mu_bh + log10_lstar + al_bh (log10 M* - 11), and al_bh was
carrying the mass dependence of the break (0.37 at the YR3 MAPs, flatter than
every measured M_BH-M* relation); the slope lets the break move with stellar
mass while al_bh follows its prior.

The ERDF is a function of lambda / lambda*, normalised on the log-lambda grid at
the pivot, so the slope is a shift of the luminosity kernel by
`log10_lstar_mslope * (log10 M* - 10.5)` at each stellar mass:
`AgnSector._shift(t_grid, k, p, mean_log10_mbh, log10_mstar)` now takes the
stellar mass, and a caller that places the kernel itself must pass it (ggah_cal's
`igo26_p` and scripts 37 and 47 do). The central's luminosity kernel has a width
of its own, `AgnSector._sigma_lx`, with variance (al_bh + log10_lstar_mslope)^2
sigma_ms^2 (1 - rho) + sig_bh^2, because the stellar-mass scatter at fixed halo
mass moves the black-hole mass and the break together; `_sigma_lm`, the width of
log10 M_BH, stays that of `mean_mbh` and the black-hole mass function. The shift
reaches every luminosity output: the occupation, satellite occupation, effective
bias, weights, emission weights, the XLF (intrinsic, `observed=True` and
`observed="aird15"`) and `validity_cost` of `n_agn` and `l_x`. `mean_mbh`,
`mean_mbh_satellites`, the black-hole mass function, `omega_bh` and the
energetics' Soltan E_AGN read M_BH itself and not the slope. One helper,
`_lstar_tilt`, reads the slope and its pivot; a test walks the syntax tree to
keep it so.

**`al_bh` and `log10_lstar_mslope` are exactly degenerate in every luminosity
observable**: the luminosity reads them only through al_bh +
log10_lstar_mslope and mu_bh - 11 al_bh - 10.5 log10_lstar_mslope, so al_bh +
d, log10_lstar_mslope - d and mu_bh + d/2 leave every luminosity output
unchanged (to 4e-14 relative at d = 0.3, z = 0.135, 0.4 and 0.8, sigma_ms =
0.3, rho = 0.2), while the black-hole quantities move. A fit that frees both
leaves al_bh to its prior and the M_BH-based quantities to al_bh. **At zero
slope every output is bit for bit the mass-independent one**: the shift
subtracts an exact zero and the two widths are one expression, checked against
1.3.0.dev0 before the change on 33 outputs at four parameter points and two
redshifts, eager and jitted (528 arrays).

**Added: a stellar-mass-dependent AGN duty cycle, `AgnParams.log10_ferdf_mslope`.**
The active fraction was one constant, `10**log10_ferdf`; it is now

    log10 f_ERDF(M*) = log10_ferdf + log10_ferdf_mslope * (log10 M* - 10.5)

with M* physical in Msun (pivot `LG_MSTAR_DUTY_PIVOT` = 10.5): a central's at
its halo's mean stellar mass M*_c(M_h), the one `mean_log10_mbh` reads, and a
satellite's at its own stellar mass on the satellite grid. Neutral at 0, bounds
(-3, 3), Flat, kind "physical". Motivated by Igo et al. (2026), whose
p(lambda_SAR|M*) the constant duty cycle over-predicts by 0.3-0.6 dex at
10^11-10^12 Msun. Every duty cycle the sector applies now goes through one
helper, `agn.duty` / `agn.log10_duty` (also `AgnSector.duty`,
`AgnSector.log10_duty`), through `AgnSector.central_duty` and the new
`AgnSector.p_active_loglx_given_m` for centrals and `_satellite_kernel` for
satellites: the occupation, satellite occupation, effective bias, weights,
emission weights, the XLF (intrinsic, `observed=True` and `observed="aird15"`)
and `validity_cost`; a test asserts no other function of the module reads
`log10_ferdf`. `p_loglx_given_m` is unchanged (the distribution of a shining
black hole, without the duty cycle). The black-hole masses, `omega_bh` and the
energetics' E_AGN (the Soltan channel, from `mean_mbh`) read no duty cycle and
are unaffected. **At zero slope every output is bit for bit the
constant-duty one** (the helper multiplies `10**log10_ferdf` by an exact
`10**0`), checked against 1.3.0.dev0 before the change on 27 outputs, eager and
jitted.

**Changed: the four `zumandelbaum16_red` / `_blue` quenching defaults are fitted to
LS10.** They were Zu & Mandelbaum (2016, Paper II) Table 2's SDSS values, now
`ZU16_PUBLISHED`. The new ones are the maximum of ggah_cal's
`massbins_zu15_gt10.0_sys0.05_redblue_free-lg_mh_qc-mu_c-lg_mh_qs-mu_s_pin_nbar-wp`:
n_bar and w_p of the red and blue galaxies of the eight LS10 stellar-mass bins
(sum_stat 0.6.0, two classes cut 0.185 mag below the red sequence), the thirteen
iHOD parameters held at the 1.2.0 defaults, joint red-blue jackknife covariance
plus 5 per cent per point; chi2 = 765.07 for 335 dof, against 14312 at Paper II's
values. Rounded to three decimals (`_ZU16_LS10`, `GalaxyParams`):

| parameter | Paper II | LS10 | Laplace sigma (conditional on the thirteen) |
|---|---|---|---|
| lg_mh_qc | 12.20 | 11.845 | 0.006 |
| mu_c | 0.38 | 1.531 | 0.067 |
| lg_mh_qs | 12.17 | 12.844 | 0.032 |
| mu_s | 0.15 | 0.621 | 0.056 |

The thirteen iHOD defaults do not change, so `zumandelbaum15` and everything
built on it are unchanged. A fit with all seventeen free fits the colours better
(chi2 = 475.1) but moves the thirteen by up to 3.8 sigma and raises the
eight-bin fit's chi2 from 195.6 to 625.8; it was not adopted (ggah_cal
`docs/galaxy_quenching.rst`). The colour model leaves the lowest bin too red
(0.477 against a measured 0.316 at 10^10.0-10.2 Msun).

**Fixed: Zu & Mandelbaum's published values, and the citations of the galaxy
registries.**

- `ZU15_PUBLISHED["fc"]` is 1.0, was 0.86. Paper I's central occupation (their
  Eq. 21) has no amplitude; their f_c = 0.86 is the satellite-to-matter
  concentration ratio, now `ZU15_PUBLISHED_B_SAT_CONC` (`b_sat_conc` of
  `SatelliteProfileParams`). Everything evaluated at Paper I's iHOD has 1/0.86 =
  1.16 times the centrals and satellites it had.
- Zu & Mandelbaum (2016) is their Paper II, not Paper III (comments,
  docstrings, the `zumandelbaum16_*` calibration notes).
- `leauthaud12`: the relation and occupations are Leauthaud et al. (2011)
  Eqs. 2, 8 and 12, the model Leauthaud et al. (2012) fitted (their Eqs. 13-15
  in arXiv:1104.0928v1); the docstrings cited "Leauthaud et al. (2012) Eq. 3,
  8, 12".
- `zacharegkas25`: Zacharegkas et al. (2025, DES Y3), not "Zacharegkas &
  Chang"; its calibration range is 0.20 < z < 0.85 (was 1.0), and its note says
  that the Table 3 relation is a fit to the centrals' mean halo mass at fixed
  stellar mass, in M_200c.

**Added: `AgnSector(selection="catalogue")`, the AGN a soft-band X-ray
catalogue selects -- unobscured, Compton-thin and Compton-thick together.**
A catalogue such as eROSITA's converts the observed 0.2-2.3 keV flux to rest
2-10 keV with an *unabsorbed* Gamma = 1.8 power law, so an absorbed source is
catalogued below its intrinsic luminosity, at
log L_cat = log L_X + Delta_b(z), Delta_b = log10[K(z; gamma_x, N_H,b) /
K(z; 1.8, 0)], K the energy arriving in the observed band per unit intrinsic
2-10 keV luminosity (`absorbed_band_energy` over `band_energy`, the
K-correction in the limits). The selection is
S(L_X) = sum_b f_b(L_X, z) * sigmoid((log L_X + Delta_b - l_min) / lx_sel_width),
inside the L_X integral, with `l_min` a threshold on the catalogue luminosity.
It needs `obscuration="split"`. Every other mode is unchanged bit for bit
except through the obscured-fraction fix below, which moves every number that
reads the split.

- **Three branches, Comparat et al. (2019)'s columns** (`obscuration_branches`,
  `CATALOGUE_BRANCHES`, `CATALOGUE_NH_CLASSES`): `catalogue_nh="comparat19"`,
  the default, gives every AGN a log N_H uniform in its class -- unobscured
  [20, 22), Compton-thin [22, 24), Compton-thick [24, 26), their Sec. 3.2 --
  and averages each branch's selection over its class by Gauss-Legendre in
  log N_H (`catalogue_nh_nodes`, 8: within 3e-5 of 32 nodes on the selected
  counts); `log10_nh` is not read. `catalogue_nh="single"` puts the branches
  at one column each -- zero (Delta = 0 exactly at gamma_x = 1.8), `log10_nh`,
  and the new constant `LOG10_NH_CT` = 24.5 -- and is kept for
  reproducibility: with its thin column free, the first YR3 MAPs pushed it to
  10^23.7-10^24.2 and removed every obscured AGN from the samples, an edge
  the uniform classes do not admit. `catalogue_columns` gives the nodes,
  `catalogue_nh_classes` replaces the intervals.
  f_CT = softmin(`compton_thick_fraction` = 0.3, f_obs), f_thin = f_obs - f_CT,
  f_unobs = 1 - f_obs -- their Eqs. 4-15: the obscured fraction is of every
  column above 1e22 and its bright branch already carries the Compton-thick
  population, so the thick share is taken out of it, not added. The soft
  minimum (`_smooth_min`, sharpness `CT_SOFTMIN_SHARPNESS` = 1000) is a guard:
  at the published coefficients f_obs >= 0.31 and the thick share is 0.3 to
  4.5e-8.
- **Scattered light** (`catalogue_scatter`, default `F_SCATTER` = 0.02 at
  `catalogue_nh="comparat19"`, 0 at `"single"`): the thin and thick classes
  send 2 per cent of their intrinsic power law into the band unabsorbed
  (log-add-exp), the spectrum the Aird et al. (2015) observed XLF below uses,
  so a Compton-thick AGN is catalogued 1.7 dex below its intrinsic luminosity
  rather than never; the band factors of `emission_weights` carry it too.
- **What it gives at the defaults** (0.2-2.3 keV, z = 0.3, L_cat > 1e42), on
  the fixed Comparat et al. (2019) fraction. The unobscured class is
  catalogued 0.03-0.54 dex low across 10^20-10^22, the thin one 0.55 dex low at
  10^22 down to the scattered-light floor, 1.70 dex, and the thick one at that
  floor: 33 per cent of the selected central AGN are obscured, Compton-thick
  ones 3.8 per cent of the selected counts (26 per cent obscured without the
  scattered light). At `catalogue_nh="single"` the thin branch is 0.94 dex low
  (1.04 at z = 0.2, 0.80 at z = 0.5) and the thick one 23.
- **Constructor options**: `catalogue_band=(0.2, 2.3)` and
  `catalogue_gamma=1.8`, the catalogue's own conversion (not `gamma_x`), and
  `catalogue_flux_limit=False`; with it on, `log10_flux_min` in that band is a
  second threshold on the same catalogue luminosity and the larger applies
  (`catalogue_threshold`, `catalogue_flux_luminosity`).
- **Where it reaches**: `occupation` (inside the `tanh`), `satellite_occupation`,
  `effective_bias`, `weights`, `emission_weights` (each branch with its own
  band factor and column, the fraction applied once) and `xlf(observed=True)`,
  the luminosity function of the catalogue's luminosities. The intrinsic XLF
  stays the default. `branch=None` is the observed mixed sample;
  `"unobscured"` and `"obscured"` (thin plus thick) add to it.
- `log10_absorbed_band_energy`: the band integral as a log-sum-exp, finite at
  a Compton-thick column in any precision, where the linear one is 1e-30 or
  less and its logarithm's gradient in Gamma would be nan.

**Added: the XLF in Aird et al. (2015)'s observed hard-band luminosity**,
`AgnSector.xlf_observed_hard` (and `xlf(..., observed="aird15")`, which needs
`obscuration="split"`). Their LADE fit neglects absorption -- the hard-band
XLF "can be described as tracing the 'observed' luminosity, corrected to
2-10 keV rest-frame values" (their Sec. 4) -- so comparing it with the
intrinsic XLF compares two quantities. Each Comparat et al. (2019) class's
intrinsic XLF is moved by the shift of each of its columns,
Delta = log10[(K(2-7 keV obs; gamma_x, N_H) + f_sc K(gamma_x, 0)) /
<K(2-7 keV obs; 1.9, N_H')>_{log N_H' in [20, 21)}] (`observed_hard_shift`),
with `F_SCATTER` = 0.02 scattered light on the thin and thick classes (Aird
et al.'s spectral model; reflection is not modelled), the conventions
`AIRD15_BAND`, `AIRD15_GAMMA`, `AIRD15_LOG_NH`. At z = 0.3 and the defaults
the unobscured class moves by +0.004 to -0.019 dex, the thin one by -0.01 to
-0.94, the Compton-thick one by -1.05 to -1.70 (the scattered light), and the
observed XLF is 0.52, 0.50 and 0.64 of the intrinsic one in 42-43, 43-44 and
above 44. The soft-band catalogue selection carries the same scattered light
(`catalogue_scatter`). `TestTheObservedHardBandXlf`.

**Fixed: `obscured_fraction` is Comparat et al. (2019) Eqs. 4-11 as
published.** Three changes, each moving every number that reads the split
(branch and band views of the AGN emission, the catalogue selection, the
observed XLF; not the luminosity-mode counts, which read no fraction):

- **Its ceiling no longer compresses it.** It ended in `soft_saturate(f, 1)`
  = 1 - exp(-f), which is f only for f << 1, so their faint-end 0.889 came out
  as 0.589 and 0.5 as 0.39 (`obscured_fraction(42, 0.19)` was 0.589); the
  docstring called it exact. It is now a smooth minimum with one,
  -ln(e^{-kf} + e^{-k})/k at k = `F_OBS_CEILING_SHARPNESS` = 100: within 7e-5
  of f for f <= 0.95, C-infinity, increasing. `soft_saturate` itself and its
  other callers (`energetics.f_gas` and the expelled fraction) are unchanged.
- **f_thick = 0.3, constant** (`F_THICK`, their Eq. 4), in place of this
  package's luminosity- and redshift-dependent `compton_thick_fraction`, which
  now returns the constant.
- **`ObscurationParams.f_bright_amp` = 0.4**, their Eq. 8 (was 0.3); the other
  seven defaults were theirs already.

The curve is their Fig. 6: 0.935 at log L 38, falling at the bright end to
0.31 at z = 0 and 0.70 at z = 6; it matches a transcription of their
equations to 1.5e-5 (`TestTheObscuredFractionIsComparat19`). 0.89 of AGN at
1e42 erg/s and z = 0.2 are obscured (0.59 before). Re-measured at the 1.2.0
defaults: the obscured-to-unobscured bias ratio at z = 0.135, L_X > 1e42 is
0.95 (0.98); with the halo-mass tilt at 1.5, 1.20 at Petter et al.'s z = 1,
L_X > 1e44 (1.33), biases 1.96 and 1.63 (2.46, 1.85), and 1.10 locally (1.13);
the rest-frame 0.5-2 keV emission is 0.15-0.65 of the unabsorbed hard-band
ratio, median 0.55 (0.83). The tests pinning the old values are re-measured.

**Fixed: the obscuration coefficients reach the spectra.**
`spectra.tracers` called the AGN sector without `obsc`, so every band and
branch view of the AGN emission was built at the default `ObscurationParams`
whatever a fit set them to. A spectrum now passes `params["obscuration"]`
(`SECTOR_EXTRAS`) to the counts and emission adapters, and
`AgnSector.weights` takes `obsc`. Absent, it is the defaults, as before.

- **Tests:** `TestTheCatalogueSelection` and
  `TestTheObscurationReachesTheSpectrum` in `test_agn.py`: with f_obs = 0 the
  catalogue counts are the luminosity- and flux-threshold counts to 1e-10;
  Delta_unobs = 0 exactly at Gamma = 1.8 and Delta_thin < 0, falling with the
  column; the branches add to the mixed sample; the gradient reaches all nine
  `ObscurationParams`, `gamma_x` and `log10_nh` (zero for the satellites'
  halo-mass tilt, which they do not carry); the Compton-thick branch is below
  1e-3 of the soft-band counts. `TestComparat19Columns`: every class at a
  negligible column is the luminosity threshold to 1e-10, every class at
  10^20 a threshold shifted by that column's dimming; more column selects
  fewer; the shifts fall across each class; `log10_nh` is not read; 8
  Gauss-Legendre nodes match 32 to 1e-4; branches add; the gradient reaches
  all nine coefficients and `gamma_x`.

**Fixed: every multiplicity function evaluates through `make_hmf`.**
`FittingFunctionHMF.dndm` passed `delta=` to every fit, so all of them but
`tinker08` raised through `make_hmf` (`TypeError` on `delta`; the two
`tinker08_csst` fits on the missing `cosmo`). The mass definition now reaches
the fit through `mass_function.multiplicity_kwargs`: `delta` for `tinker08`,
`delta_ratio` for `despali16`, `cosmo` for the cosmology-dependent fits, nothing
for the others. It is the dispatch `make_field` uses (`field._delta_kw` now
calls it), so the two paths cannot diverge. `FittingFunctionHMF` takes
`calibration` (default `"strict"`) and runs `check_cosmology_support` and
`check_calibration` as `make_field` does: `make_hmf("watson13", pk)` at the
default 200m is now refused as a definition mismatch, as in `make_field`, and
`calibration="warn"` or `"off"` takes it knowingly.

- **Tests:** `test_make_hmf.py`: all eighteen fits are finite and positive
  through `make_hmf`; `tinker08` and `despali16` at 200m, 200c and vir equal
  `make_field`'s `dndm` to 1e-6; the mismatch is refused by default; an
  unknown policy is refused.

## 1.2.0 (2026-10-08)

The galaxy defaults the next model layers build on.  1.1.0 was never
released: this release carries its three development steps, the entries
headed 1.1.0.dev3, dev2 and dev1 below (the first is 1.2.0's own).

- **Tests:** three cold-gas predictions are marked expected failures with
  their 1.2.0 numbers (`test_coldgas_galaxies.py`, strict): Omega_HI is 0.52
  of the measured budget and the HI mass function 0.54 of ALFALFA at 10^9
  Msun/h, because the 1.1.0.dev1 galaxies' agreement came from a stellar-mass
  function 1.5-4.5 times GAMA below 10^10 Msun.  The AGN numbers on the new
  galaxies are re-measured: satellites 7.2% of the AGN, b_eff 0.895 (0.822
  for centrals), satellites 28% of Omega_BH.

## 1.1.0.dev3

**Changed: the thirteen fitted `zumandelbaum` iHOD defaults are the MAP of all
eight LS10 stellar-mass bins, 10^10-10^12 Msun, with a 5 per cent systematic
error** (ggah_cal `massbins_zu15_gt10.0_sys0.05_nbar-wp`, `--sys-frac 0.05`):
n_bar and w_p of each bin on sum_stat 0.6.0, the jackknife covariance plus 5
per cent of every point in quadrature, for the photometric-redshift and
stellar-mass errors that move galaxies between bins; chi2 = 195.61 for 192 dof
(195.68 at the rounded values). They replace the fit of the five bins above
10^10.6 Msun with the jackknife alone (chi2 = 152.91 for 105 dof), which
extrapolated to 1.62 times the abundance of the 10^10-10^10.2 bin.

| | `lg_m1h` | `lg_m0star` | `beta` | `delta` | `gamma` | `sigma_lnmstar` | `eta` | `fc` | `bsat` | `beta_sat` | `bcut` | `beta_cut` | `alpha_sat` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1.0.dev3 | 12.131 | 10.238 | 0.605 | 0.684 | 2.177 | 0.734 | -0.143 | 0.726 | 11.28 | 0.658 | 0.716 | 0.161 | 1.057 |
| 1.1.0.dev1-dev2 | 12.289 | 10.331 | 0.765 | 0.781 | 0.496 | 0.538 | -0.125 | 0.874 | 12.53 | 0.856 | 0.750 | 0.669 | 1.083 |

- The data now reach below the knee of the relation: gamma = 2.18 +- 0.17
  (0.50 +- 0.63) and lg_m0star to +-0.03 (+-0.17) in the Laplace
  approximation; no parameter is prior-dominated.
- The previous values are kept as `occupation.ZU15_LS10_110`; the calibration
  record says eight bins and `mstar_range=(10.0, 12.0)`.
- The threshold default is unchanged, 10^10.5 Msun at h = 0.6736.
- Every number computed at the galaxy defaults moves: the AGN sector's hosts,
  the energy closure, the census. The AGN and gas defaults were fitted on the
  previous galaxies and are not refitted here.

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
