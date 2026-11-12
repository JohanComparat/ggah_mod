# Layer 3: active galactic nuclei

The AGN sector populates the galaxies of {doc}`layer3_galaxies` with supermassive black holes and decides which of them shine.
Every galaxy, central or satellite, of stellar mass $M_\star \ge 10^{8}\,{\rm M}_\odot$ hosts one black hole, whose mass follows from the galaxy's stellar mass, and that black hole is active for a fraction of the time.
From one set of parameters the sector predicts the X-ray luminosity function, the occupation of haloes by AGNs above a luminosity or flux threshold, their weights as a tracer, the black-hole mass function and $\Omega_{\rm BH}$, and the mean black-hole mass the energy closure of {doc}`layer3_hot_gas` reads.
The technical paper (Comparat, in prep., Sec. 4.2) documents the default chain; this page follows it.
Stellar and black-hole masses are physical, in ${\rm M}_\odot$, and halo masses in $h^{-1}{\rm M}_\odot$.

:::{warning}
Layer 3 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.sectors` emits a `ggah_mod.WorkInProgressWarning` once per
session, which the snippets below silence.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.sectors.agn` | {py:class}`~ggah_mod.sectors.agn.AgnSector`, {py:class}`~ggah_mod.sectors.agn.AgnParams`, {py:class}`~ggah_mod.sectors.agn.ObscurationParams`, {py:func}`~ggah_mod.sectors.agn.obscured_fraction`, the published luminosity functions `XLF` and {py:func}`~ggah_mod.sectors.agn.make_xlf`, and the black-hole relations `BH_CHAINS` |
| {py:mod}`ggah_mod.sectors.xrb` | the X-ray binaries of a galaxy, {py:func}`~ggah_mod.sectors.xrb.lx_xrb` and {py:func}`~ggah_mod.sectors.xrb.stacked_lx` |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import; the emission view needs it

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import (AgnParams, AgnSector, GalaxySector, ObscurationParams,
                              galaxy_defaults)
from ggah_mod.sectors import agn as A

field = make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.2)
gal = GalaxySector("zumandelbaum15", calibration="off")
gp = galaxy_defaults("zumandelbaum15")
agn = AgnSector(gal, calibration="off")
p = AgnParams()
```

The {doc}`notebook <notebooks/layer3_agn>` draws every relation below at the defaults.

## The chain

The sector needs the stellar masses of a galaxy sector whose occupation is a stellar-mass threshold (`zumandelbaum15` and its colour split, `leauthaud12`, `zacharegkas25`), and refuses any other.
From the halo mass to the X-ray luminosity, the chain is

$$
M_h \xrightarrow{\ \rm SHMR\ } M_\star \longrightarrow M_{\rm BH}
  \xrightarrow{\ \xi(\lambda)\ } L_{\rm bol} = \lambda\,L_{\rm Edd}
  \xrightarrow{\ K_X\ } L_X,
\qquad L_{\rm Edd} = 1.26\times10^{38}\,\frac{M_{\rm BH}}{{\rm M}_\odot}\ {\rm erg\,s^{-1}},
$$ (eq-agn-chain)

with $\lambda$ the Eddington ratio and $K_X$ the bolometric correction to the rest-frame $2$–$10$ keV band.
The bolometric luminosity function is then a sum over galaxies,

$$
\begin{aligned}
\Phi(L_{\rm bol}) &= \int{\rm d}\log_{10}M_h\,\frac{{\rm d}n}{{\rm d}\log_{10}M_h}\,
   F_{\rm c}(M_h)\,f_{\rm ERDF}\big(M_\star^{\rm c}(M_h)\big)\,
   \frac{{\rm d}P}{{\rm d}\log L_{\rm bol}}(L_{\rm bol}|M_h) \\
 &\quad + f_{\rm duty}^{\rm sat}\int_{8}^{12.5}{\rm d}\lg M_\star\,\phi_{\rm s}(M_\star)\,
   f_{\rm ERDF}(M_\star)\,\frac{{\rm d}P}{{\rm d}\log L_{\rm bol}}(L_{\rm bol}|M_\star), \\
F_{\rm c}(M_h) &= \tfrac12\,{\rm erfc}\left[
   \frac{8 - \lg M_\star^{\rm c}(M_h)}{\sqrt2\,\sigma_{M_\star}(M_h)}\right],
\end{aligned}
$$ (eq-agn-lbolf)

in $h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}$, with $\phi_{\rm s}$ the stellar-mass function of the satellites, $F_{\rm c}$ the fraction of central galaxies above $10^{8}\,{\rm M}_\odot$, $\sigma_{M_\star}(M_h)$ the scatter of the central's $\lg M_\star$ at fixed halo mass, $f_{\rm ERDF}$ the duty cycle and $f_{\rm duty}^{\rm sat}$ that of satellites relative to central galaxies of the same stellar mass.
The galaxy sector's parameters are an input of every method, and the black-hole parameters are those of {py:class}`~ggah_mod.sectors.agn.AgnParams`:

```{include} _generated/params_agn.md
```

The default parameters were obtained from the abundance, clustering, lensing and luminosity function of X-ray-selected AGNs, their cross-correlation with galaxies, and the Eddington-ratio distributions at fixed stellar mass of {cite:t}`IgoMerloniGeorgakakis_2026A&A...713A.318I`, with the galaxy parameters at their defaults.

```python
lgm = np.log10(np.asarray(field.m))[::50]                      # log10 M_h [h^-1 Msun]
print(agn.log10_mstar(lgm, gp, h=PLANCK18.h))                 # log10 M_*^c [Msun]
print(agn.mean_log10_mbh(lgm, p, gp, h=PLANCK18.h))           # <log10 M_BH> [Msun]
```

## Black-hole mass

The black-hole mass is log-normal about a linear relation in $\log M_\star$, after {cite:t}`PowellAllenCaglar_2022ApJ...938...77P` (their Model 1):

$$
\begin{aligned}
\langle\log M_{\rm BH}\rangle(M_\star) &= \mu_{\rm BH} + \alpha_{\rm BH}\left(\log M_\star - 11\right),\\
\sigma_{\rm lm}^2(M_h) &= \alpha_{\rm BH}^2\,\sigma_{M_\star}^2(M_h)\,(1-\rho) + \sigma_{\rm BH}^2,
\end{aligned}
$$ (eq-mbh)

with $\sigma_{\rm BH}$ the intrinsic scatter at fixed stellar mass, $\sigma_{\rm lm}$ the scatter at fixed halo mass, and $\rho$ the correlation of $M_{\rm BH}$ with halo mass at fixed $M_\star$; $\rho = 0$ is Model 1.
The Gaussian priors on $\mu_{\rm BH}$ and $\sigma_{\rm BH}$ are Powell et al.'s posteriors, and that on $\alpha_{\rm BH}$ the slope of {cite:t}`ReinesVolonteri_2015ApJ...813...82R` for AGN hosts and their total stellar mass.
The relation is extrapolated linearly below the stellar masses it was fitted on, $10^{10}$–$10^{12}\,{\rm M}_\odot$.
`bh_chain="trinity"` selects the TRINITY relation, which has no published coefficients and raises until they are passed:

```{include} _generated/agn_registries.md
```

## The Eddington-ratio distribution

The Eddington ratios follow a broken power law {cite:p}`AnannaWeigelTrakhtenbrot_2022ApJS..261....9A`,

$$
\xi(\lambda) = \frac{{\rm d}P}{{\rm d}\log\lambda} \propto \frac{1}{x^{\delta_1} + x^{\delta_2}},
\qquad x = \frac{\lambda}{\lambda_\star(M_\star, z)},
$$ (eq-erdf)

normalised to one over $-3 \le \log\lambda \le 1.5$, with the break

$$
\log\lambda_\star(M_\star, z) = \log\lambda_\star + \beta_\lambda\left(\lg M_\star - 10.5\right)
  - \log_{10}\frac{f(x)}{f(x_0)}, \qquad
f(u) = u^{\gamma_\lambda} + u^{-\gamma_\lambda^{\rm hi}},
$$ (eq-lstar)

$x = (1+z_\lambda)/(1+z)$ and $x_0 = 1+z_\lambda$.
The normalisation is at $z = 0$, so $\lambda_\star$ is the local break that Ananna et al. measure; $\gamma_\lambda = \gamma_\lambda^{\rm hi} = 0$, the default, is no evolution.
Every luminosity reads $\alpha_{\rm BH}$ and $\beta_\lambda$ only through $\alpha_{\rm BH} + \beta_\lambda$, while $\Omega_{\rm BH}$ and the black-hole mass function read $\alpha_{\rm BH}$ alone, so the prior on $\alpha_{\rm BH}$ separates the two.

The distribution is normalised down to a lower cut-off, and the fraction of black holes that are active is a separate duty cycle, a power law in stellar mass capped smoothly at one:

$$
f_{\rm ERDF}(M_\star) = \mathcal S_{64}\!\left(10^{\log_{10}f_{\rm ERDF} + \beta_f(\lg M_\star - 10.5)}\right),
\qquad
\mathcal S_k(f) = -\frac{1}{k}\ln\left(e^{-kf} + e^{-k}\right).
$$ (eq-agn-duty)

## The luminosity at fixed stellar mass

$\log L_{\rm bol} - \log(1.26\times10^{38})$ is the sum of $\log M_{\rm BH}$ and $\log\lambda$, two independent random variables, so its density is a convolution:

$$
\frac{{\rm d}P}{{\rm d}\log L_{\rm bol}}(L_{\rm bol}|M_\star) = K_{\sigma_{\rm BH}}\big(y - \langle\log M_{\rm BH}\rangle(M_\star)\big),
\qquad K_\sigma = \xi \ast \mathcal N(0, \sigma),
$$ (eq-agn-kernel)

with $y = \log L_{\rm bol} - \log(1.26\times10^{38})$.
The central galaxy of a halo of mass $M_h$ has the same shape with the wider kernel $K_{\sigma_{\rm lx}}$, $\sigma_{\rm lx}^2 = (\alpha_{\rm BH} + \beta_\lambda)^2\sigma_{M_\star}^2(1-\rho) + \sigma_{\rm BH}^2$, since the stellar-mass scatter at fixed halo mass moves the black-hole mass and the break together.
The kernel depends on mass only through a shift, so it is computed once per call and placed at each mass.

The bolometric correction to the rest-frame $2$–$10$ keV band is that of {cite:t}`DurasBongiornoRicci_2020A&A...636A..73D` (their Eq. 2 and Table 1), by default (`kbol="duras20"`):

$$
L_X = \frac{L_{\rm bol}}{K_X(L_{\rm bol})}, \qquad
K_X = a\left[1 + \left(\frac{\log_{10}(L_{\rm bol}/{\rm L}_\odot)}{b}\right)^{c}\right],
\qquad
\Phi_X(L_X) = \Phi\big(L_{\rm bol}(L_X)\big)\frac{{\rm d}\log L_{\rm bol}}{{\rm d}\log L_X},
$$ (eq-kx)

with $a = 10.96$, $b = 11.93$, $c = 17.79$ and ${\rm L}_\odot = 3.828\times10^{33}\,{\rm erg\,s^{-1}}$.
Outside the range they fitted, $7.5 < \log_{10}(L_{\rm bol}/{\rm L}_\odot) < 14.5$, $\log K_X$ continues linearly.
`kbol="constant"` takes $K_X = k_{\rm bol}$ instead.

## Obscuration and absorption

The AGNs are sorted into three classes of hydrogen column density, in ${\rm cm}^{-2}$: unobscured ($20 \le \log N_{\rm H} < 22$), Compton-thin ($22$–$24$) and Compton-thick ($24$–$26$), following {cite:t}`ComparatMerloniSalvato_2019MNRAS.487.2005C` (their Sect. 3.2).
The obscured fraction, of every AGN above $10^{22}\,{\rm cm}^{-2}$, blends a faint and a bright branch:

$$
\begin{aligned}
f_{\rm obs} &= \mathcal S_{100}\!\left[f_{\rm b} + (f_{\rm f} - f_{\rm b})\,
   \tfrac12\left(1 + {\rm erf}\frac{\ell - \log L_X}{w}\right)\right], \\
f_{\rm f} &= a_{\rm f}\sqrt{\ell_{\rm f}/\log L_X}, \qquad
f_{\rm b} = f_{\rm CT} + f_0 + a_{\rm b}\,{\rm erf}(z/z_{\rm b}), \\
\ell &= \ell_0 + \Delta\ell\,{\rm erf}(z) + a_M\log_{10}(M_h/10^{13}\,h^{-1}{\rm M}_\odot),
\end{aligned}
$$ (eq-agn-obsc)

with $f_{\rm CT} = 0.3$ the Compton-thick fraction.
This is Comparat et al.'s Eqs. 4–11 with a tilt $a_M$ in host halo mass added, which lets the obscured and unobscured AGNs differ in bias.
The coefficients are {py:class}`~ggah_mod.sectors.agn.ObscurationParams`, read only by a sector built with `obscuration="split"`:

```{include} _generated/params_obscuration.md
```

One spectrum serves every band.
For a photon spectrum $N(E)\propto E^{-\Gamma}$ absorbed by a column $N_{\rm H}$, the energy a source at redshift $z$ emits into an observed band is

$$
B(E_1, E_2, N_{\rm H}, z) = \int_{E_1(1+z)}^{E_2(1+z)} E^{1-\Gamma}\,e^{-N_{\rm H}\sigma(E)}\,{\rm d}E,
$$ (eq-agn-band)

with $\sigma(E)$ the photoelectric cross-section of {cite:t}`MorrisonMcCammon_1983ApJ...270..119M`; the K-correction is in the limits.
The soft-band luminosity of a class is $10^{\Delta_b}L_X$, with $10^{\Delta_b} = [B(0.5, 2; N_{\rm H}) + f_{\rm sc}B(0.5, 2; 0)]/B(2, 10; 0)$ averaged over the class's column range, and $f_{\rm sc} = 0.02$ the scattered fraction of the obscured classes {cite:p}`AirdCoilGeorgakakis_2015MNRAS.451.1892A`.

```python
from ggah_mod.sectors import obscured_fraction

print(obscured_fraction(np.array([42.0, 44.0, 46.0]), 0.2, ObscurationParams(), log10m=13.0))
print(A.band_energy_ratio(0.5, 2.0, 2.0, 10.0, gamma=1.8))       # soft over hard, no column
```

## Selection, occupation and tracer weights

The probability that the central galaxy of a halo hosts an AGN above the survey threshold is its kernel integrated over a smooth selection in $L_X$:

$$
\langle N_{\rm c}^{\rm AGN}\rangle(M_h) = \tanh\left[f_{\rm ERDF}\big(M_\star^{\rm c}(M_h)\big)
  \int{\rm d}\log L_X\,\frac{{\rm d}P}{{\rm d}\log L_X}\,
  \tfrac12\left(1 + {\rm erf}\frac{\log L_X - \log L_X^{\min}}{\Delta_{\rm sel}}\right)\right],
$$ (eq-agn-occ)

with ${\rm d}P/{\rm d}\log L_X$ the central's kernel times $F_{\rm c}$, and the $\tanh$ a smooth cap at one black hole per central galaxy.
The selection is chosen when the sector is built:

| `selection=` | threshold | reads |
|---|---|---|
| `"luminosity"` (default) | $\log L_X^{\min}$, rest-frame $2$–$10$ keV | `log10lx_min`, `lx_sel_width` |
| `"flux"` | $\log L_X^{\min}(z) = \log[4\pi(D_{\rm L}/h)^2F^{\min}] - \log[B(E_1, E_2)/B_0(E_1, E_2)]$, one column for the sample | `log10_flux_min`, `log10_nh`, `gamma_x` |
| `"none"` | every AGN, as for a photon map | |
| `"catalogue"` | the eRASS-like flux threshold, class by class; needs `obscuration="split"` | `log10_flux_min` and the catalogue arguments of `AgnSector` |

The AGNs of satellite galaxies follow the satellite profile of the galaxies, and a halo holds a Poisson number of them:

$$
\langle N_{\rm s}^{\rm AGN}\rangle = f_{\rm duty}^{\rm sat}\int_8^{12.5}{\rm d}\lg M_\star\,
   \frac{{\rm d}N_{\rm s}}{{\rm d}\lg M_\star}(M_\star|M_h)\,P(>L_X^{\min}|M_\star),
\qquad
w_{\rm point} = \langle N_{\rm c}^{\rm AGN}\rangle, \quad
w_{\rm extended} = \langle N_{\rm s}^{\rm AGN}\rangle\,u_{\rm sat}(k|M), \quad n = \bar n .
$$ (eq-agn-weights)

{py:meth}`~ggah_mod.sectors.agn.AgnSector.weights` returns these counts.
{py:meth}`~ggah_mod.sectors.agn.AgnSector.emission_weights` weights each AGN by its luminosity in a band instead, $w_{\rm point} = \langle N_{\rm c}L\rangle$, with $\langle NL^2\rangle$ for its shot noise, and is the AGN part of an X-ray map; it needs double precision, since $L^2$ reaches $10^{88}$.
With `obscuration="split"`, the obscured fraction of Eq. {eq}`eq-agn-obsc` enters inside the integral of Eq. {eq}`eq-agn-occ`, and `branch=` selects the obscured or unobscured AGNs.

```python
w = agn.weights(field, p, gp)                                  # counts, discrete
print(f"n = {float(w.norm):.3e} h^3 Mpc^-3, b = {float(agn.effective_bias(field, p, gp)):.3f}")

split = AgnSector(gal, calibration="off", obscuration="split")
o = ObscurationParams()
for branch in ("unobscured", "obscured"):
    print(branch, float(split.effective_bias(field, p, gp, branch=branch, obsc=o)))
w_x = split.emission_weights(field, p, gp, band=(0.5, 2.0), obsc=o)  # luminosity-weighted
```

## The X-ray luminosity function

{py:meth}`~ggah_mod.sectors.agn.AgnSector.xlf` integrates Eq. {eq}`eq-agn-lbolf` with Eq. {eq}`eq-kx` into the intrinsic luminosity function in the hard band, or, with `obscuration="split"`, in the soft band class by class.
The X-ray luminosity function is a prediction here rather than an input.
The published fits of {cite:t}`AirdCoilGeorgakakis_2015MNRAS.451.1892A` and {cite:t}`UedaAkiyamaHasinger_2014ApJ...786..104U` are in the registry for comparison, at their $h = 0.7$; {py:func}`~ggah_mod.sectors.agn.xlf_in_h_units` converts them to $h^{-1}$ units.

```python
lx, phi = agn.xlf(field, p, gp)                                # log10 L_X [erg/s], h^3 Mpc^-3 dex^-1
lx_tab = np.array([42.0, 43.0, 44.0, 45.0])
print(np.interp(lx_tab, lx, phi))
print(A.xlf_in_h_units("aird15", lx_tab, 0.2, PLANCK18.h))
```

## Black holes

Every central and satellite galaxy above $10^{8}\,{\rm M}_\odot$ hosts a black hole, active or not.
The same chain, read as a mass rather than a rate, gives the mean black-hole mass of the central galaxy of a halo,

$$
\langle M_{\rm BH}\rangle(M_h) = F_{\rm c}(M_h)\,10^{\langle\log M_{\rm BH}\rangle(M_h)}
  \exp\left[\tfrac12\left(\ln 10\,\sigma_{\rm lm}\right)^2\right],
$$ (eq-mbh-mean)

that of its satellites by the same integral as Eq. {eq}`eq-agn-weights` with $\sigma_{\rm BH}$ in place of $\sigma_{\rm lm}$, the black-hole mass function, and

$$
\Omega_{\rm BH} = \frac{h}{\rho_{\rm c,0}}\int{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}
  \left[\langle M_{\rm BH}\rangle(M) + M_{\rm BH}^{\rm sat}(M)\right].
$$ (eq-omega-bh)

{py:meth}`~ggah_mod.sectors.agn.AgnSector.mean_mbh` and {py:meth}`~ggah_mod.sectors.agn.AgnSector.mean_mbh_satellites` feed the energy the black holes released into the closure of the hot gas.
{py:meth}`~ggah_mod.sectors.agn.AgnSector.active_bhmf` and {py:meth}`~ggah_mod.sectors.agn.AgnSector.joint_mbh_lx` give the mass function of the active black holes and their joint distribution with $L_X$, for a type-1 or flux-limited sample.
{py:meth}`~ggah_mod.sectors.agn.AgnSector.validity_cost` measures what the linear extrapolation of Eq. {eq}`eq-mbh` below $10^{10}\,{\rm M}_\odot$ is worth in $\Omega_{\rm BH}$, $\bar n$ or $L_X$; {cite:t}`FukugitaPeebles_2004ApJ...616..643F` is the budget to compare $\Omega_{\rm BH}$ with.

```python
print(f"Omega_BH = {float(agn.omega_bh(field, p, gp)):.2e}")
lg_mbh = np.linspace(6.0, 10.0, 5)                             # log10 M_BH [Msun]
print(agn.black_hole_mass_function(field, lg_mbh, p, gp))      # h^3 Mpc^-3 dex^-1
```

## X-ray binaries

The X-ray binaries of a galaxy add to its AGN in a stacked X-ray luminosity.
{py:mod}`ggah_mod.sectors.xrb` writes them after {cite:t}`LehmerBasu-ZychMineo_2016ApJ...825....7L` (their Eq. 15), in the $2$–$10$ keV band,

$$
L_X^{\rm XRB} = \left[10^{\log\alpha_0}(1+z)^{\gamma}M_\star + 10^{\log\beta_0}(1+z)^{\delta}\,{\rm SFR}\right]
  \frac{L_{\rm band}}{L_{2-10}},
$$ (eq-xrb)

with the star formation rate of the main sequence of {cite:t}`SpeagleSteinhardtCapak_2014ApJS..214...15S` (their Eq. 28), for a {cite:t}`Kroupa_2001MNRAS.322..231K` initial mass function, and a red fraction that lowers it.
{py:func}`~ggah_mod.sectors.xrb.stacked_lx` adds the mean AGN luminosity of the sector at the same stellar mass:

```{include} _generated/params_xrb.md
```

```python
from ggah_mod.sectors import XrbParams, stacked_lx

lg_ms = np.array([9.5, 10.0, 10.5, 11.0])                      # log10 M_* [Msun]
print(stacked_lx(lg_ms, 0.2, XrbParams(), agn=split, agn_params=p, red_fraction=0.5,
                 obsc=o))                                       # <L_X> [erg/s], 2-10 keV
```

## Refusals and units

- A galaxy sector without a stellar-mass threshold raises, and so does an unknown `bh_chain`, `obscuration`, `selection` or `kbol`.
- `selection="catalogue"`, a `branch=`, a luminosity carried into another band (`emission_weights(band=...)`, `band_factor`), the observed-band luminosity functions and `type1=True` need `obscuration="split"`.
- Stellar and black-hole masses are physical, and the methods that convert from halo masses take `h` explicitly.
- The calibration check compares the field with the range of Powell et al.'s fit, $z \le 0.1$ and $M_\star \ge 10^{10}\,{\rm M}_\odot$, and warns on almost every field; the snippets build the sector with `calibration="off"`.

## Related pages

- {doc}`notebooks/layer3_agn`: the relations above, executed, with figures.
- {doc}`layer3_galaxies`: the stellar masses the chain starts from.
- {doc}`layer3_hot_gas` and {doc}`closure_options`: the energy the black holes release.
- {doc}`layer4_spectra`: the AGN as a tracer, alone and in the X-ray composite.
- {doc}`api/sectors`: every function and its arguments.
