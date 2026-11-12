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
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
references
```
