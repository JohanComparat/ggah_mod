# Changelog

## Unreleased (1.1.0)

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
