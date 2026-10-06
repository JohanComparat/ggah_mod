# ggah_mod

[![PyPI](https://img.shields.io/pypi/v/ggah_mod.svg)](https://pypi.org/project/ggah_mod/)
[![Python](https://img.shields.io/pypi/pyversions/ggah_mod.svg)](https://pypi.org/project/ggah_mod/)
[![Documentation](https://readthedocs.org/projects/ggah-mod/badge/?version=latest)](https://ggah-mod.readthedocs.io)
[![Tests](https://github.com/JohanComparat/ggah_mod/actions/workflows/ci.yml/badge.svg)](https://github.com/JohanComparat/ggah_mod/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JohanComparat/ggah_mod/blob/main/LICENSE)

**G**alaxies, **g**as, **A**GN and **h**alos: a layered, differentiable
halo-model forward model.

`ggah_mod` predicts the power spectrum of any pair of tracers, and its
projections, from one set of cosmological and astrophysical parameters.
It is written in JAX, so every prediction can be differentiated with respect to
every parameter, exactly.

**Documentation: <https://ggah-mod.readthedocs.io>**

## Install

> **Maintainer setup.** On the development laptop, use the shared `dev` environment
> defined in [`dev_env`](https://github.com/JohanComparat/dev_env), cloned at
> `~/software/dev_env` (`conda activate dev`); this package is already
> installed there in editable mode. Do not create a separate environment for it:
> add missing dependencies to `~/software/dev_env` and rebuild.

<!-- install-start -->
`ggah_mod` needs Python ≥ 3.11:

```bash
pip install ggah_mod
```

or, to work on it, from a clone:

```bash
git clone https://github.com/JohanComparat/ggah_mod.git
cd ggah_mod
pip install -e .
```

That is the differentiable path, and it needs no compiler: numpy, scipy, JAX, and
the two emulators it reads, [`emu_pk`](https://github.com/JohanComparat/emu_pk)
for the linear power spectrum and
[`emu_hmf`](https://github.com/JohanComparat/emu_hmf) for the mass function, both
also from PyPI. Check it with

```bash
python -c "import ggah_mod; print(ggah_mod.__version__)"
```

Optional extras add the rest, e.g. `pip install "ggah_mod[reference]"`:

| extra | installs | for |
|---|---|---|
| `reference` | `classy` | CLASS, the linear spectrum of the `ACCURATE` flavour; pip compiles it, so a C compiler is needed |
| `backends` | `camb` | CAMB, the alternative Boltzmann backend |
| `cobaya` | `cobaya` | `ggah_mod.interfaces.cobaya`, a cobaya `Theory` |
| `tables` | `soxs` | regenerating the shipped APEC cooling table |
| `dev` | `pytest`, `pytest-cov`, `pytest-xdist`, `astropy`, `colossus`, `camb` | the test suite |
| `docs` | `sphinx`, `furo`, `myst-nb`, `sphinx-copybutton`, `sphinxcontrib-bibtex` | building the documentation |

With conda or mamba,
[`environment.yml`](https://github.com/JohanComparat/ggah_mod/blob/main/environment.yml)
creates an environment with `ggah_mod`, the CPU build of `jaxlib`, and
`matplotlib` and a Jupyter kernel for the documentation notebooks:

```bash
mamba env create -f environment.yml
mamba activate ggah_mod
```

JAX installs its CPU build by default; for a GPU, install the matching `jax`
build first, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

To work on `ggah_mod` together with its sibling packages (`emu_pk`, `emu_hmf`,
...), [`dev_env`](https://github.com/JohanComparat/dev_env) defines the conda
environments they are all developed in, with every package installed editable
(Linux x86-64).
<!-- install-end -->

## Quickstart

<!-- quickstart-start -->
```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import numpy as np
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, comoving_distance, make_pk, sigma8
from ggah_mod.halos.field import make_field

# Layer 1: the linear power spectrum, and what is read off it.
pk = make_pk("emu_pk")                        # "class" needs the [reference] extra
k = np.logspace(-4, np.log10(200.0), 512)     # h/Mpc
print(sigma8(pk.pk(k, 0.0, PLANCK18), k))     # an output, never an input
print(comoving_distance(np.array([0.5, 1.0]), PLANCK18))   # h^-1 Mpc

# Layer 2: mass function, bias, concentration and profiles on one mass grid.
field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.5)
print(field.m[::64])                          # h^-1 Msun
print(field.dndm[::64], field.bias[::64])     # dn/dM, b(M)

# Everything is differentiable: d sigma_8 / d Omega_m through the emulator.
d_sigma8 = jax.grad(lambda om: sigma8(pk.pk(k, 0.0, PLANCK18.replace(Omega_m=om)), k))
print(d_sigma8(0.31))
```
<!-- quickstart-end -->

## The six layers

Each layer depends only on the ones above it.

| layer | subpackage | computes | documentation |
|---|---|---|---|
| 1 | `ggah_mod.cosmology` | parameters and density budget, background and distances, linear `P(k)`, amplitude and growth | [layer 1](https://ggah-mod.readthedocs.io/en/latest/layer1_cosmology.html) |
| 2 | `ggah_mod.halos` | variance and peak height, mass function, bias, concentration, profiles, the halo field | [layer 2](https://ggah-mod.readthedocs.io/en/latest/layer2_halos.html) |
| 3 | `ggah_mod.sectors` | the tracers | ⚠️ work in progress |
| 4 | `ggah_mod.spectra` | power spectra of pairs of tracers | ⚠️ work in progress |
| 5 | `ggah_mod.observables` | projected observables | ⚠️ work in progress |
| 6 | `ggah_mod.covariance` | covariances of data vectors | ⚠️ work in progress |
| — | `ggah_mod.interfaces` | adapters that let other packages drive this one | ⚠️ work in progress |

Beside them, `ggah_mod.backend` holds the flavours below, `ggah_mod.numerics` the
numerical helpers several layers share, and `ggah_mod/data` the distilled tables
the package ships.
**Layers 3 to 6 and the interfaces are work in progress**: released as tested
code ahead of their verification and documentation, so their interface and
results may change. Importing any of them emits a `ggah_mod.WorkInProgressWarning`
once; `warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)`
silences it.

## Two flavours of one model

The same code runs in two flavours, selected by a `Backend`:

- **`ACCURATE`**: a Boltzmann solver (CLASS by default, CAMB as the
  alternative), fine grids, exact quadrature. For fitting.
- **`DIFFERENTIABLE`**: the `emu_pk` emulator, coarser grids. Differentiable end
  to end. For forecasting. `DIFFERENTIABLE_COARSE` is the same flavour on
  smaller grids.

They are not two implementations: only the linear power spectrum forks. The
disagreement between them is *measured*, row by row, in the parity budget
(`tests/test_parity_budget.py`), never assumed.

## Decisions fixed once

- **One amplitude.** `ln10A_s`. `sigma8` and `S8` are outputs, computed from the
  spectrum. Passing either as an input raises.
- **`Omega_m` contains the neutrinos.** The budget
  `Omega_b + Omega_cdm + Omega_nu == Omega_m` closes exactly, with `Omega_nu`
  the neutrinos' rest mass.
- **One neutrino convention, CLASS's.** Three massive states at
  `T_ncdm = 0.71611 T_CMB` plus a massless remainder of `N_eff = 3.044`, on the
  exact energy integral over their relic spectrum: the rest mass is
  `Sigma m/(93.143 eV h^2)`, and at 0.06 eV in the degenerate ordering the
  neutrino density today exceeds it by 5.0e-4, which is their kinetic energy and
  the remainder. `CambPk` is handed the same neutrinos, spelt in CAMB's interface.
- **Cold and total densities are separately named.** Halos form from
  `rho_cold`; lensing sees `rho_matter`. Neither ever stands in for the other.
- **One growth route**, with no silent fallback to an approximation.
- **Nothing is fixed by hard-coding.** A constant a user might want to vary is
  a parameter with a prior and a bound.

## Status

Version 1.0 is the first public release. Layers 1 and 2 are verified and
documented. Layers 3 to 6 are work in progress: implemented and tested, with
their verification ongoing and their documentation to follow it. Calibrated values of the
astrophysical parameters are not yet released.

Contributing and the development environment are described in
[`CONTRIBUTING.md`](https://github.com/JohanComparat/ggah_mod/blob/main/CONTRIBUTING.md); changes are listed in
[`CHANGELOG.md`](https://github.com/JohanComparat/ggah_mod/blob/main/CHANGELOG.md).

## Citing

Please cite the technical paper (Comparat, in prep.) and the software, whose
metadata is in [`CITATION.cff`](https://github.com/JohanComparat/ggah_mod/blob/main/CITATION.cff).

## License

MIT; see [`LICENSE`](https://github.com/JohanComparat/ggah_mod/blob/main/LICENSE).
