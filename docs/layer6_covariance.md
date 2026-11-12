# Layer 6: the covariance of a data vector

This layer computes the covariance of a data vector of {doc}`layer5_observables` from the same halo model that predicts it, in the order of the vector:

$$
\mathsf C = \mathsf C^{\rm G} + \mathsf C^{\rm pair} + \mathsf C^{\rm 1h} + \mathsf C^{2\text{–}4{\rm h}} + \mathsf C^{\rm SSC},
$$ (eq-cov-terms)

the Gaussian term, the clustering of a pair shared by two bins, the one-halo and the two- to four-halo trispectra, and the super-sample term.
The sum extends the Gaussian, connected and super-sample decomposition of {cite:t}`KrauseEifler_2017MNRAS.470.2100K`, {cite:t}`FriedrichAndrade-OliveiraCamacho_2021MNRAS.508.3125F` and {cite:t}`ReischkeUnruhAsgari_2025A&A...699A.124R`.
Three statistics are supported: $C_\ell$, $w_p(r_p)$ and $\Delta\Sigma(R)$.
The technical paper (Comparat, in prep., Sec. 6) documents the layer; this page follows it.

:::{warning}
Layer 6 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.covariance` emits a `ggah_mod.WorkInProgressWarning` once per
session, which the snippets below silence.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.covariance.spec` | {py:func}`~ggah_mod.covariance.spec.covariance`, {py:func}`~ggah_mod.covariance.spec.gaussian_covariance`, `TERMS` |
| {py:mod}`ggah_mod.covariance.geometry` | {py:class}`~ggah_mod.covariance.geometry.SurveyGeometry` and its builders |
| {py:mod}`ggah_mod.covariance.gaussian` | the Gaussian terms of $C_\ell$ and $w_p$ |
| {py:mod}`ggah_mod.covariance.lensing` | the Gaussian terms of $\Delta\Sigma$ and $w_p\times\Delta\Sigma$, {py:class}`~ggah_mod.covariance.lensing.SourceSample` |
| {py:mod}`ggah_mod.covariance.trispectrum` | the one-halo trispectrum and the shared-object term |
| {py:mod}`ggah_mod.covariance.multihalo` | the two- to four-halo trispectrum |
| {py:mod}`ggah_mod.covariance.supersample` | the super-sample term |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

import jax.numpy as jnp
import ggah_mod.covariance as CV
import ggah_mod.observables as OB
import ggah_mod.sectors as S
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field, make_fields

pk = make_pk("emu_pk")
gal, gp = S.GalaxySector("zumandelbaum15", calibration="off"), S.galaxy_defaults("zumandelbaum15")
field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.13)
f_cen, _ = gal.stellar_fraction(field, gp)
split = S.BaryonSplit.from_hot(S.cosmic_baryon_fraction(PLANCK18), jnp.full(field.n_m, 0.1),
                               f_star_cen=f_cen)
sectors = {"galaxies": gal, "matter": S.MatterField()}
params = {"galaxies": gp, "matter": {"split": split}}
fields_at = lambda zs: make_fields(PLANCK18, DIFFERENTIABLE, pk, zs)

edges = np.logspace(-1.0, 1.0, 6)                              # h^-1 Mpc
rp = np.sqrt(edges[1:] * edges[:-1])
spec = OB.ObservableSpec((OB.Wp("wp", "galaxies", "galaxies", rp, pi_max=67.4, z=0.13),
                          OB.DeltaSigma("ds", "galaxies", "matter", rp, z=0.13)))
```

The {doc}`notebook <notebooks/layer6_covariance>` draws the covariance below, term by term.

## Survey geometry

A {py:class}`~ggah_mod.covariance.geometry.SurveyGeometry` holds the footprint of a sample: its area, its redshift slab, and, for the super-sample term, the power of its mask, $\sum_m|W_{\ell m}|^2$ in the convention of CCL {cite:p}`ChisariAlonsoKrause_2019ApJS..242....2C`.
Its builders cover a disc (`spherical_cap`), a cut in Galactic latitude (`galactic_latitude_cut`), a HEALPix mask (`from_healpix_mask`), a set of randoms (`from_randoms`), an area alone (`from_area`, which the super-sample term refuses) and the LS10 samples (`ls10_sample`).
The volume of the slab is integrated over the comoving volume element.

```python
area = CV.ls10_sample("10.0")
geo = CV.spherical_cap(area.area_sr / (4 * np.pi), name="LS10 disc",
                       z_min=area.z_min, z_max=area.z_max, ell_max=1024)
overlap = CV.spherical_cap(400.0 / CV.FULL_SKY_DEG2, name="lensing overlap",
                           z_min=area.z_min, z_max=area.z_max, ell_max=2048)
print(f"{geo.area_deg2:.0f} deg^2, f_sky = {geo.f_sky:.3f}, V = {float(geo.volume(PLANCK18)):.2e} h^-3 Mpc^3")
```

## The Gaussian term

For two spectra, Wick's theorem gives ${\rm Cov}[\hat P_{ab}, \hat P_{cd}] = (\tilde P_{ac}\tilde P_{bd} + \tilde P_{ad}\tilde P_{bc})/N_{\rm modes}$, with $\tilde P$ the spectra plus their noise.
For $C_\ell$ the number of modes in a band is $f_{\rm sky}(\ell_{\rm hi}^2 - \ell_{\rm lo}^2)$, the $f_{\rm sky}$ approximation of {cite:t}`FriedrichAndrade-OliveiraCamacho_2021MNRAS.508.3125F`, and the noise of each map is passed as `cl_noise`.
For $w_p$ in annuli of $r_p$,

$$
{\rm Cov}[w_p^i, w_p^j] = \frac{1}{V}\int\frac{k_\perp{\rm d}k_\perp}{2\pi}\,\bar J_0^i\bar J_0^j
  \int\frac{{\rm d}k_\parallel}{2\pi}\,S_\Pi^2(k_\parallel)\left[\tilde P_{ac}\tilde P_{bd} + \tilde P_{ad}\tilde P_{bc}\right],
\qquad S_\Pi = \frac{2\sin(k_\parallel\Pi)}{k_\parallel},
$$ (eq-cov-wp)

with $\bar J_0^i$ the $J_0$ averaged over annulus $i$, $\Pi = \pi_{\max}$ and the noise $1/\bar n$ of each tracer passed as `tracer_noise`; a photometric-redshift dispersion damps the signal along the line of sight (`wp_sigma_los`).
For $\Delta\Sigma$ the term is that of {cite:t}`SinghMandelbaumSeljak_2017MNRAS.471.3827S` (their Eq. 10 and App. A) for the random-subtracted estimator,

$$
{\rm Cov}[\Delta\Sigma_i, \Delta\Sigma_j] = \frac{1}{V_W}\int\frac{k\,{\rm d}k}{2\pi}\,\bar J_2^i\bar J_2^j
  \left[(P_{\rm gg} + N_{\rm g})(P_{\Sigma\Sigma} + N_\Sigma) + \Delta\Pi_2\,\bar\rho_{\rm m}^2P_{\rm gm}^2\right],
$$ (eq-cov-ds)

with $P_{\Sigma\Sigma} = \Sigma_{\rm crit}^2\chi_l^2C_{\kappa\kappa}(k\chi_l)$ the projected matter along the line of sight and $N_\Sigma = \Sigma_{\rm crit}^2\sigma_e^2/n_s$ the shape noise.
The source sample, its $\Sigma_{\rm crit}$, its shape noise, its density and the line-of-sight lengths, is a {py:class}`~ggah_mod.covariance.lensing.SourceSample`, passed per statistic as `sources`; the overlap of lenses and sources is a geometry of its own, `lensing_geometry`.
The term $w_p\times\Delta\Sigma$ follows from the same quadrature.

```python
nbar = float(gal.number_density(field, gp))
src = CV.SourceSample("sources", 5e15, 0.28, 15.0, 700.0, 134.8, None, "illustrative")
common = dict(geometry=geo, edges={"wp": edges, "ds": edges}, fields_at=fields_at,
              sectors=sectors, cosmo=PLANCK18, backend=DIFFERENTIABLE,
              tracer_noise={"galaxies": 1.0 / nbar}, sources={"ds": src},
              lensing_geometry=overlap)
c_g = np.asarray(CV.gaussian_covariance(spec, **common)(params))      # (10, 10)
print(np.sqrt(np.diag(c_g)))
```

## The shared pair

When two bins of $w_p$ count the same pair of galaxies, the clustering of that pair adds to the Poisson noise, which replaces $2\Pi$ by $2\Pi + w_p$ in the noise term of overlapping annuli (`shared_pair`).

## The connected terms

The one-halo trispectrum of four tracers is $T^{1h}_{abcd}(k_1, k_2) = \int{\rm d}M\,({\rm d}n/{\rm d}M)\,\mathcal Q_{abcd}$, with the four-point rule that no two point legs of one discrete sample are the same object; for one sample $\mathcal Q = e_1^2e_2^2 + 2p_1e_1e_2^2 + 2e_1^2p_2e_2$, with $p$ and $e$ the point and extended weights of {doc}`layer4_spectra`.
Its projection onto two bins is

$$
{\rm Cov}^{1h}_{ij} = \frac{1}{V}\sum_\alpha\int{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}
  \left[K_i\cdot F_\alpha\right]\left[K_j\cdot G_\alpha\right],
$$ (eq-cov-1h)

with $K_i$ the bin's real-space kernel and $F_\alpha$, $G_\alpha$ the factors of $\mathcal Q$.
When one object is shared between the two pairs, the term is an integral over the halo's surface density, through Graf's addition theorem {cite:p}`Lacasa_2018A&A...615A...1L` (`one_halo_shared`).
The two- to four-halo terms (`multi_halo`) follow {cite:t}`CoorayHu_2001ApJ...554...56C` and {cite:t}`TakadaHu_2013PhRvD..87l3504T`, in the parallelogram configuration and averaged over angle:

$$
\begin{aligned}
T^{2h}_{22} &= \langle P_L(|\mathbf k_1 + \mathbf k_2|)\rangle\left[I_{ac}I_{bd} + I_{ad}I_{bc}\right],\\
T^{2h}_{13} &= P_L(k_1)\left[I_aI_{bcd} + I_bI_{acd}\right] + P_L(k_2)\left[I_cI_{dab} + I_dI_{cab}\right],\\
T^{3h} &= \langle B^{\rm PT}\rangle\left[I_aI_cI_{bd} + \dots\right],
\qquad T^{4h} = \langle T^{\rm PT}\rangle\,I_aI_bI_cI_d,
\end{aligned}
$$ (eq-cov-multi)

with $I$ the halo integrals of layer 4, the counterterm included, and $B^{\rm PT}$, $T^{\rm PT}$ the tree-level bispectrum and trispectrum.
The one-halo transition and the neutrino leg are not carried here.

## The super-sample term

Modes larger than the survey move every bin together, through the response of each statistic to a background density $\delta_b$ {cite:p}`TakadaHu_2013PhRvD..87l3504T`:

$$
{\rm Cov}^{\rm SSC} = \sigma_b^2\,\frac{\partial X}{\partial\delta_b}\frac{\partial Y}{\partial\delta_b},
\qquad
\frac{\partial P_{ab}}{\partial\delta_b} = \left(\frac{47}{21} - \frac13\frac{{\rm d}\ln P_L}{{\rm d}\ln k}\right)P_LI_aI_b
  + T_{1h} - (b_a + b_b)P_{ab},
$$ (eq-cov-ssc)

with the response of {cite:t}`LiHuTakada_2014PhRvD..89h3519L` and $T_{1h}$ the response of the one-halo term.
$\sigma_b^2$, the variance of the background over the footprint and the slab, needs the mask power of the geometry: exact below $\ell = 40$ and in the Limber approximation above.

## The full covariance

{py:func}`~ggah_mod.covariance.spec.covariance` takes the specification of the data vector, the terms to include, the geometry, the bin edges of every statistic and the noise, and returns a {py:class}`~ggah_mod.covariance.spec.Covariance`, a function of the parameters.
It records the terms it leaves out, `missing`:

```{include} _generated/covariance_terms.md
```

```python
cov = CV.covariance(spec, terms=tuple(CV.TERMS), **common)
c = np.asarray(cov(params))
print(cov.missing)                                              # the terms not yet implemented
print(np.sqrt(np.diag(c) / np.diag(c_g)))                      # error over its Gaussian part
```

The cosmology, the volumes and the geometry are fixed when the covariance is built, so `cov(params)` is a function of the tracer parameters at that cosmology.
A $\xi$, $w(\theta)$ or $\tau_{\rm kSZ}$ statistic, a $C_\ell$ block crossed with a real-space one, several redshifts or several $\pi_{\max}$ in one real-space vector, and several source samples raise `NotImplementedError`.

## Related pages

- {doc}`notebooks/layer6_covariance`: the covariance above, executed, with figures.
- {doc}`notebooks/fit_wp`: a fit to $w_p$ with its Gaussian covariance.
- {doc}`layer5_observables`: the data vector.
- {doc}`api/covariance`: every function and its arguments.
