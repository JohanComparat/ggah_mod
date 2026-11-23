# ggah_mod

**G**alaxies, **g**as, **A**GN and **h**aloes: a layered, differentiable halo-model
forward model.

`ggah_mod` predicts the power spectrum of any pair of tracers, and its
projections, from one set of cosmological and astrophysical parameters, in JAX,
so that every prediction can be differentiated with respect to every parameter.
The {doc}`overview` describes the six layers it is built from and the two
flavours it runs in; the layer pages derive what each layer computes and show how
to call it, and each has a notebook to run.

## Installation

```{include} ../README.md
:start-after: <!-- install-start -->
:end-before: <!-- install-end -->
```

## Quickstart

```{include} ../README.md
:start-after: <!-- quickstart-start -->
:end-before: <!-- quickstart-end -->
```

```{toctree}
:caption: Guide
:maxdepth: 2

overview
layer1_cosmology
layer2_halos
layer3_galaxies
layer3_agn
layer3_hot_gas
closure_options
layer4_spectra
layer5_observables
layer6_covariance
```

```{toctree}
:caption: Notebooks
:maxdepth: 1

notebooks/layer1_cosmology
notebooks/layer2_halos
notebooks/layer3_galaxies
notebooks/layer3_agn
notebooks/layer3_hot_gas
notebooks/layer4_spectra
notebooks/layer5_observables
notebooks/layer6_covariance
notebooks/fit_wp
notebooks/cobaya_bao
```

```{toctree}
:caption: Paper I tutorials
:maxdepth: 1

tutorials/README
tutorials/00_getting_started
tutorials/01_cosmology_background
tutorials/02_power_spectrum
tutorials/03_amplitude_growth
tutorials/04_halo_variance
tutorials/05_halo_abundance_bias
tutorials/06_concentration_profiles
tutorials/07_distance_constraints
tutorials/08_full_shape
tutorials/09_halo_counts
tutorials/10_halo_requirements
tutorials/11_joint_emulator_goal
tutorials/A_emulator_benchmark
tutorials/B_code_comparisons
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
references
```
