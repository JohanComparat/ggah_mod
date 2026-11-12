# Layer 5: observables

This layer projects the power spectra $P_{ab}(k, z)$ of {doc}`layer4_spectra` onto the statistics a survey measures: the correlation function $\xi(r)$, the projected correlation function $w_p(r_p)$, the excess surface density $\Delta\Sigma(R)$, the kinetic-SZ optical depth, the angular power spectrum $C_\ell$ and the angular correlation function $w(\theta)$.
A static specification of the data vector compiles into one function of the parameters, which `jax.jit` traces and `jax.jacfwd` differentiates.
The technical paper (Comparat, in prep., Sec. 6) documents the layer; this page follows it.
Distances and radii are comoving, in $h^{-1}{\rm Mpc}$, and angles in radians.

:::{warning}
Layer 5 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.observables` emits a `ggah_mod.WorkInProgressWarning` once per
session, which the snippets below silence.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.observables.real_space` | {py:func}`~ggah_mod.observables.real_space.xi`, {py:func}`~ggah_mod.observables.real_space.wp`, {py:func}`~ggah_mod.observables.real_space.sigma`, {py:func}`~ggah_mod.observables.real_space.delta_sigma`, {py:func}`~ggah_mod.observables.real_space.tau_ksz_cap`, {py:func}`~ggah_mod.observables.real_space.w_theta` |
| {py:mod}`ggah_mod.observables.transforms` | the Hankel transforms, by quadrature or FFTLog |
| {py:mod}`ggah_mod.observables.limber` | {py:func}`~ggah_mod.observables.limber.c_ell`, the Limber projection |
| {py:mod}`ggah_mod.observables.kernels` | the radial kernels: number counts, lensing, CMB lensing, thermal SZ, dispersion measure |
| {py:mod}`ggah_mod.observables.beams` | Gaussian and King beams |
| {py:mod}`ggah_mod.observables.spec` | {py:class}`~ggah_mod.observables.spec.ObservableSpec` and {py:func}`~ggah_mod.observables.spec.make_model` |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

import jax.numpy as jnp
import ggah_mod.observables as OB
import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field, make_fields

pk = make_pk("emu_pk")
field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.25)
K = np.asarray(field.k)                                    # concrete wavenumbers
gal, gp = S.GalaxySector("zheng07", backend=DIFFERENTIABLE), S.galaxy_defaults("zheng07")
split = S.BaryonSplit.from_hot(S.cosmic_baryon_fraction(PLANCK18), jnp.full(field.n_m, 0.1))
sectors = {"galaxies": gal, "matter": S.MatterField()}
params = {"galaxies": gp, "matter": {"split": split}}
p_gg = SP.spectrum(field, "galaxies", "galaxies", sectors, params)
p_gm = SP.spectrum(field, "galaxies", "matter", sectors, params)
```

The {doc}`notebook <notebooks/layer5_observables>` draws every statistic below, and a {doc}`second one <notebooks/fit_wp>` fits $w_p(r_p)$.

## Hankel transforms

Four statistics are the same integral of a spectrum against a Bessel function:

$$
\begin{aligned}
\xi(r) &= \frac{1}{2\pi^2}\int{\rm d}k\,k^2P(k)\,j_0(kr), &
\Sigma(R) &= \frac{\bar\rho_{\rm m}}{2\pi}\int{\rm d}k\,k\,P(k)\,J_0(kR),\\
\Delta\Sigma(R) &= \frac{\bar\rho_{\rm m}}{2\pi}\int{\rm d}k\,k\,P(k)\,J_2(kR), &
w(\theta) &= \frac{1}{2\pi}\int{\rm d}\ell\,\ell\,C_\ell\,J_0(\ell\theta).
\end{aligned}
$$ (eq-hankel)

Two engines evaluate them, chosen by the flavour.
`ACCURATE` uses the double-exponential quadrature of {cite:t}`Ogata_2005PRIMS..41..949O`, whose step `hankel_h` sets the accuracy and whose node count sets the reach.
The differentiable flavours use FFTLog {cite:p}`Hamilton_2000MNRAS.312..257H`, low-ringing, on the logarithmic $k$ grid of the field padded by `fftlog_pad_decades`; it needs a concrete NumPy grid.
Beyond the grid the spectrum continues as a power law at its end slope, capped at $-3$, since truncating it would change $w_p$ and $\Delta\Sigma$ by about $20\%$.
The grids of each flavour:

```{include} _generated/projection_grids.md
```

## Real-space statistics

$w_p$ integrates $\xi$ along the line of sight to $\pi_{\max}$,

$$
w_p(r_p) = 2\int_0^{\pi_{\max}}\xi\left(\sqrt{r_p^2 + \pi^2}\right){\rm d}\pi,
$$ (eq-wp)

with $\xi$ tabulated once and the integral taken in ${\rm asinh}(\pi/r_p)$; `pi_max` is a property of the survey and has no default.
$\Delta\Sigma$, in $h\,{\rm M}_\odot\,{\rm pc}^{-2}$, is the direct $J_2$ transform of $P_{\rm gm}$; `comoving=False` with the lens redshift returns the proper surface density, $(1+z)^2$ times the comoving one.
The kinetic-SZ optical depth of a galaxy sample is $\tau = \sigma_{\rm T}/(\mu_{\rm e}m_{\rm p})\,(1+z)^2\,\Sigma_{\rm e}$, from the electron–galaxy cross-spectrum, and {py:func}`~ggah_mod.observables.real_space.tau_ksz_cap` filters it through a compensated aperture of outer radius $\sqrt2$ times the inner one.
A spectrum is passed as a layer-4 {py:class}`~ggah_mod.spectra.pk.PowerSpectrum` or as the pair `(k, P)`.

```python
rp = np.logspace(-1.0, 1.5, 5)                                   # h^-1 Mpc
print(OB.wp(rp, (K, p_gg.total), pi_max=60.0, backend=DIFFERENTIABLE))
print(OB.xi(rp, (K, p_gg.total), backend=DIFFERENTIABLE))
print(OB.delta_sigma(rp, (K, p_gm.total), PLANCK18, backend=DIFFERENTIABLE))   # h Msun pc^-2
```

## Limber projection

The angular power spectrum of two tracers under two radial kernels is, in the extended Limber approximation {cite:p}`Limber_1953ApJ...117..134L,LoVerdeAfshordi_2008PhRvD..78l3506L`,

$$
C_\ell^{ab} = B^a_\ell B^b_\ell\int{\rm d}\chi\,\frac{W_a(\chi)\,W_b(\chi)}{\chi^2}\,
  P_{ab}\left(\frac{\ell + 1/2}{\chi}, z(\chi)\right),
$$ (eq-limber)

with $B_\ell$ the beams.
The spectrum is evaluated on a grid of redshift nodes, one halo field each, and interpolated in $k$ as a cubic in $\log k$ and $\log P$; next to a node where it changes sign, it is interpolated in value.
$w(\theta)$ is the transform of $C_\ell$ in Eq. {eq}`eq-hankel`.

### Radial kernels

Lensing is the matter field under a kernel, not a tracer of its own:

| kernel | $W(\chi)$ |
|---|---|
| {py:func}`~ggah_mod.observables.kernels.number_counts` | ${\rm d}N/{\rm d}z\,{\rm d}z/{\rm d}\chi$, normalised to unit integral |
| {py:func}`~ggah_mod.observables.kernels.lensing_efficiency` | $\tfrac32\Omega_{\rm m}(H_0/c)^2f_K(\chi)(1+z)\int_\chi{\rm d}\chi'\,n(\chi')f_K(\chi'-\chi)/f_K(\chi')$ |
| {py:func}`~ggah_mod.observables.kernels.cmb_lensing` | $\tfrac32\Omega_{\rm m}(H_0/c)^2\chi(1+z)(\chi_\star - \chi)/\chi_\star$ |
| {py:func}`~ggah_mod.observables.kernels.thermal_sz` | $1$, the $\sigma_{\rm T}/m_{\rm e}c^2$ being in the pressure tracer |
| {py:func}`~ggah_mod.observables.kernels.dispersion_measure` | $\bar n_{\rm e,0}(1+z)$, after {cite:t}`MacquartProchaskaMcQuinn_2020Natur.581..391M` |

$\Omega_{\rm m}$ includes the neutrinos.
{py:func}`~ggah_mod.observables.kernels.limber_grid` gives the redshift and distance nodes, linear in $z$.

### Beams

A map's beam multiplies its leg of $C_\ell$.
{py:func}`~ggah_mod.observables.beams.gaussian` takes a full width at half maximum, and {py:func}`~ggah_mod.observables.beams.king` the King profile of an X-ray telescope, differentiable in its two parameters,

$$
B_\ell = \frac{2^{2-\alpha}}{\Gamma(\alpha - 1)}(\ell\theta_{\rm c})^{\alpha-1}K_{\alpha-1}(\ell\theta_{\rm c}),
$$ (eq-king)

with `beams.EROSITA_KING` the eROSITA values.
No pixel window is applied.

## The data vector

An {py:class}`~ggah_mod.observables.spec.ObservableSpec` lists the statistics, each a static declaration: its tracers, its bins, its redshift or its kernels and beams (`Wp`, `DeltaSigma`, `Xi`, `TauKsz`, `Cl`, `WTheta`).
{py:func}`~ggah_mod.observables.spec.make_model` resolves it into the pairs of tracers and the redshifts it needs, builds the fields once per call through `fields_at`, and returns `params -> (dict, vector)`.
The dictionary is keyed by name, and a JAX dictionary is sorted by key, so a Fisher matrix takes the concatenated vector, in the order of the specification (`model.plan.slices`).
The model compiles under `jax.jit` when `k=` gives it concrete wavenumbers.

```python
z, chi = OB.limber_grid(PLANCK18, z_max=1.5, backend=DIFFERENTIABLE)
kernels = {"counts": OB.number_counts(z, chi, np.exp(-0.5 * ((np.asarray(z) - 0.5) / 0.1) ** 2)),
           "cmbk": OB.cmb_lensing(z, chi, PLANCK18)}
ell = np.logspace(1.0, 3.5, 5)
spec = OB.ObservableSpec((
    OB.Wp("wp", "galaxies", "galaxies", rp, pi_max=60.0, z=0.25),
    OB.DeltaSigma("ds", "galaxies", "matter", rp, z=0.25),
    OB.Cl("cl_gk", "galaxies", "matter", ell, kernel_a="counts", kernel_b="cmbk"),
))
model = jax.jit(OB.make_model(
    spec, fields_at=lambda zs: make_fields(PLANCK18, DIFFERENTIABLE, pk, zs),
    sectors=sectors, kernels=kernels, cosmo=PLANCK18, backend=DIFFERENTIABLE,
    z_proj=np.asarray(z), k=K))
out, vector = model(params)
jac = jax.jacfwd(lambda a: model({"galaxies": gp | {"alpha": a}, "matter": params["matter"]})[1])(1.0)
print(vector.size, jac[:3])                                     # d(vector)/d(alpha)
```

## Related pages

- {doc}`notebooks/layer5_observables`: the statistics above, executed, with figures.
- {doc}`notebooks/fit_wp`: a fit of the galaxies and the cosmology to $w_p(r_p)$.
- {doc}`layer6_covariance`: the covariance of the same data vector.
- {doc}`api/observables`: every function and its arguments.
