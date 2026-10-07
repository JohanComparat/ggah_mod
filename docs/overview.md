# Overview

`ggah_mod` is a halo-model forward model for galaxies, gas, AGN and haloes.
It predicts the power spectrum of any pair of tracers, and its projections, from
one set of cosmological and astrophysical parameters, so that constraints from
different surveys bear on the same numbers.
The code is written in JAX, so every prediction can be differentiated with
respect to every parameter, exactly and at a cost that does not grow with the
number of parameters.
Background on the halo model is in the reviews of {cite:t}`CooraySheth_2002PhR...372....1C`
and {cite:t}`AsgariMeadHeymans_2023OJAp....6E..39A`; the technical paper
(Comparat, in prep.) documents the implementation and its accuracy.

## Six layers

The package is built in six layers, each depending only on the ones above it.

| layer | subpackage | computes |
|---|---|---|
| 1 | `ggah_mod.cosmology` | parameters and density budget, background expansion and distances, linear $P(k)$, amplitude and growth |
| 2 | `ggah_mod.halos` | variance and peak height, mass function, bias, concentration, profiles, the halo field |
| 3 | `ggah_mod.sectors` | the tracers (work in progress) |
| 4 | `ggah_mod.spectra` | power spectra of pairs of tracers (work in progress) |
| 5 | `ggah_mod.observables` | projected observables (work in progress) |
| 6 | `ggah_mod.covariance` | covariances of data vectors (work in progress) |

`ggah_mod.interfaces` holds adapters that let other packages drive this one, and
`ggah_mod.backend` holds the flavours described below.
This documentation covers layers 1 and 2, {doc}`layer1_cosmology` and
{doc}`layer2_halos`, and the galaxy registries of layer 3,
{doc}`layer3_galaxies`.
:::{warning}
Layers 3 to 6 and `ggah_mod.interfaces` are **work in progress**: released as
tested code ahead of their verification and their documentation, so their
interface and results may change. Importing any of them emits a
`ggah_mod.WorkInProgressWarning` once per session, which can be silenced on its
own:

```python
import warnings
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)
```
:::

## Two flavours of one model

The same code runs in two flavours, selected by a
{py:class}`~ggah_mod.backend.Backend`.
`ACCURATE` is the fitting path: a Boltzmann solver for the linear power spectrum
(CLASS by default, CAMB as the alternative) and fine grids.
`DIFFERENTIABLE` is the forecasting path: the `emu_pk` emulator of CLASS, which is
the only differentiable spectrum here, on coarser grids, differentiable end to
end.
`DIFFERENTIABLE_COARSE` is the same flavour on smaller grids.
The flavour fixes the grids and the four halo-model choices once, so everything
downstream reads them from one object:

```{include} _generated/flavours.md
```

The two flavours are one implementation.
Only the linear power spectrum forks, in `cosmology/power.py`; the background,
the variance, the growth and everything built on them are written once, and are
differentiable exactly when the power spectrum driving them is.
A flavour is chosen by passing it, or by the environment variable
`GGAH_BACKEND`, which {py:func}`~ggah_mod.backend.resolve_backend` reads when no
flavour is passed.

## Decisions fixed once

- **One amplitude.** $\ln(10^{10}A_{\rm s})$ is the input; $\sigma_8$ and $S_8$
  are outputs, computed from the power spectrum. Passing either as an input
  raises.
- **$\Omega_{\rm m}$ contains the neutrinos' rest mass**, and
  $\Omega_{\rm cb} = \Omega_{\rm b} + \Omega_{\rm cdm}$ is named separately.
  Haloes form from the cold density $\bar\rho_{\rm cb}$; lensing sees the total
  $\bar\rho_{\rm m}$. Neither stands in for the other.
- **One neutrino convention**, that of CLASS: three massive states at
  $T_{\rm ncdm} = 0.71611\,T_{\rm CMB}$ and a massless remainder of
  $N_{\rm eff} = 3.044$.
- **One growth route**, a ratio of $\sigma_8$ of the power spectrum at two
  redshifts, with no fitting formula to fall back on.
- **Nothing is fixed by hard-coding.** A constant a user might want to vary is a
  parameter with a bound and a prior.
- **A request the model cannot honour raises** rather than returning an
  approximation: a cosmology outside an emulator's training box, a mass
  definition a fit was not calibrated in, a non-linear spectrum that does not
  exist here.

## Units and conventions

| quantity | unit |
|---|---|
| distances, radii | $h^{-1}\,{\rm Mpc}$, comoving |
| wavenumbers | $h\,{\rm Mpc}^{-1}$ |
| power spectra | $(h^{-1}\,{\rm Mpc})^3$ |
| masses | $h^{-1}\,{\rm M}_\odot$ |
| densities | $h^{-1}\,{\rm M}_\odot\,(h^{-1}\,{\rm Mpc})^{-3}$ |
| distance modulus | magnitudes, from $D_{\rm L}$ in Mpc (no $h$) |
| neutrino masses | eV |

A density parameter without an argument is its value today,
$\Omega_X \equiv \rho_X(0)/\rho_{\rm crit,0}$; one that depends on redshift
carries it.
Halo masses are spherical-overdensity masses at 200 times the mean matter density
(`200m`) unless a different definition is asked for.

## Installing, and what each path needs

`pip install` gives the differentiable path: `emu_pk` for the power spectrum and
`emu_hmf` for the mass function, both pure JAX, both on PyPI.
It covers all of layers 1 and 2 except the Boltzmann backends.
The accurate path needs CLASS (the `[reference]` extra, which compiles it) or
CAMB (`[backends]`).
The notebooks on these pages run on the pip install alone, and show the CLASS
comparison when `classy` is importable.
Installation is described on the {doc}`front page <index>`.

## Double precision

JAX computes in single precision unless told otherwise, and `ggah_mod` never
changes that setting for you.
Enable 64-bit arithmetic **before the first `ggah_mod` import**, because
module-level arrays take the precision active when they are created:

```python
import jax
jax.config.update("jax_enable_x64", True)

import ggah_mod
```

or set `JAX_ENABLE_X64=1` in the environment.
Layers 1 and 2 run in single precision, but derivatives checked against finite
differences need double precision.

## Citing

Please cite the technical paper (Comparat, in prep.) and the software through its
[`CITATION.cff`](https://github.com/JohanComparat/ggah_mod/blob/main/CITATION.cff).
The references on these pages are collected under {doc}`references`.
