# ggah_mod

**G**alaxies, **g**as, **A**GN and **h**alos: a layered, differentiable halo-model
forward model.

`ggah_mod` predicts the power spectrum of any pair of tracers, and its
projections, from one set of cosmological and astrophysical parameters, in JAX,
so that every prediction can be differentiated with respect to every parameter.
The {doc}`overview` describes the six layers it is built from and the two
flavours it runs in; the layer pages derive what each layer computes and show how
to call it, and each has a notebook to run.
These pages cover layers 1 and 2 and the galaxy registries of layer 3; the
rest follow.

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
```

```{toctree}
:caption: Notebooks
:maxdepth: 1

notebooks/layer1_cosmology
notebooks/layer2_halos
notebooks/layer3_galaxies
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
references
```
