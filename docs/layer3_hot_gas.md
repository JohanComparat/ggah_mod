# Layer 3: hot gas

The hot-gas sector gives every halo of {doc}`layer2_halos` its circumgalactic or intracluster plasma: profiles of the electron density, pressure, temperature and metallicity, their X-ray and Sunyaev–Zel'dovich observables, and their weights as a tracer.
Its amplitudes are free parameters, or, when asked for, set by an energy closure that balances the energy the supernovae and the black holes released against the binding energy of the halo's baryons.
The baryons it does not hold go to an ejected component, and the matter field adds the dark matter, the stars, the neutral gas and the ejected gas to the hot gas, halo by halo.
The technical paper (Comparat, in prep., Sec. 4.3 and 4.4) documents the default; this page follows it.
Radii are comoving, in $h^{-1}{\rm Mpc}$, and halo masses in $h^{-1}{\rm M}_\odot$, at the mass definition of the halo field (`200m` by default).

:::{warning}
Layer 3 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.sectors` emits a `ggah_mod.WorkInProgressWarning` once per
session, which the snippets below silence.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.sectors.gas` | {py:class}`~ggah_mod.sectors.gas.HotGasDPM` and {py:class}`~ggah_mod.sectors.gas.DpmParams` |
| {py:mod}`ggah_mod.sectors.cooling` | the cooling functions `COOLING`, {py:func}`~ggah_mod.sectors.cooling.make_cooling` |
| {py:mod}`ggah_mod.sectors.energetics` | the energy closure {py:func}`~ggah_mod.sectors.energetics.f_retained_energy` and {py:class}`~ggah_mod.sectors.energetics.EnergeticsParams` |
| {py:mod}`ggah_mod.sectors.ejecta` | {py:class}`~ggah_mod.sectors.ejecta.EjectaSector`, the baryons pushed out of haloes |
| {py:mod}`ggah_mod.sectors.matter` | {py:class}`~ggah_mod.sectors.matter.BaryonSplit` and {py:class}`~ggah_mod.sectors.matter.MatterField`, the field lensing sees |
| {py:mod}`ggah_mod.sectors.census` | {py:func}`~ggah_mod.sectors.census.census`, the cosmic baryon budget |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import; L_X needs it

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field
import ggah_mod.sectors as S

field = make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.2)
gas, dp = S.HotGasDPM(backend=DIFFERENTIABLE), S.DpmParams()

m3 = np.array([1e13, 1e14, 1e15])                            # h^-1 Msun
i3 = np.searchsorted(np.asarray(field.m), m3)
on = dict(conc=field.conc[i3], mdef=field.mdef)              # the haloes' own concentrations
```

The {doc}`notebook <notebooks/layer3_hot_gas>` draws the profiles, the scaling relations and the baryon budget at the defaults.

## The profiles

The sector is the Descriptive Parametric Model (DPM) of {cite:t}`OppenheimerVoitBahe_2025MNRAS.543.2649O`: three generalised-NFW profiles, of the electron density, the electron pressure and the metallicity,

$$
\begin{aligned}
n_{\rm e}(r|M,z) &= n_{\rm e}^{\rm a}\,\frac{f(x|\alpha^n)}{f(0.3c_{\rm gas})}\,E(z)^{\gamma_n}M_{12}^{\beta_n},\\
P_{\rm e}(r|M,z) &= P_{\rm e}^{\rm a}k_{\rm B}\,\frac{f(x|\alpha^P)}{f(0.3c_{\rm gas})}\,E(z)^{\gamma_P}M_{12}^{\beta_P},\\
Z(r) &= Z^{\rm a}\,\frac{f(x|\alpha^Z)}{f(0.3c_{\rm gas})},
\end{aligned}
$$ (eq-dpm)

with $f(x|\alpha_{\rm in}, \alpha_{\rm tr}, \alpha_{\rm out}) = x^{-\alpha_{\rm in}}(1 + x^{\alpha_{\rm tr}})^{(\alpha_{\rm in} - \alpha_{\rm out})/\alpha_{\rm tr}}$ the shape of {doc}`layer2_halos`, $x = r\,c_{\rm gas}/R_\Delta$, $M_{12} = M_\Delta/(10^{12}h^{-1}\,{\rm M}_\odot)$ in physical solar masses, and $E(z) = H(z)/H_0$.
The amplitudes are quoted at $0.3R_\Delta$, $n_{\rm e}$ in ${\rm cm^{-3}}$ and $P_{\rm e}/k_{\rm B}$ in ${\rm cm^{-3}\,K}$.
The three profiles share one scale radius, tied to the halo's concentration $c(M,z)$, and their slopes vary with mass:

$$
c_{\rm gas} = c(M,z)\,10^{\lambda_c + \lambda_c^{\rm var}\log_{10}M_{12}},
\qquad
\alpha(M) = \alpha + \alpha^{\rm var}\log_{10}M_{12}.
$$ (eq-dpm-var)

The temperature is their ratio, $kT = P_{\rm e}/n_{\rm e}$ in keV.
Every profile method takes the halo concentration `conc` explicitly, from the halo field, and the radii as an array of shape `(NM, n)`:

```{include} _generated/params_dpm.md
```

The defaults are a fit to the soft X-ray profile of LS10 galaxies above $10^{10.5}\,{\rm M}_\odot$ from eROSITA and to measurements of the gas in haloes from the Milky Way to clusters: the X-COP profiles and gas fractions {cite:p}`GhirardiniEckertEttori_2019A&A...621A..41G,EckertGhirardiniEttori_2019A&A...621A..40E,GhizzardiMolendivanderBurg_2021A&A...646A..92G`, the pressure profile of {cite:t}`ArnaudPrattPiffaretti_2010A&A...517A..92A`, the $Y$–$M$ relations of {cite:t}`PlanckCollaborationAdeAghanim_2011A&A...536A..11P` and {cite:t}`HiltonSifonNaess_2021ApJS..253....3H`, the gas fractions and temperatures of groups and clusters {cite:p}`LovisariReiprichSchellenberger_2015A&A...573A.118L,LovisariSchellenbergerSereno_2020ApJ...892..102L,MantzAllenMorris_2016MNRAS.463.3582M,BulbulLiuKluge_2024A&A...685A.106B`, the hot halo of the Milky Way {cite:p}`PontiZhengLocatelli_2023A&A...674A.195P`, and the $L_X$ of galaxy haloes {cite:p}`ZhangComparatPonti_2024A&A...690A.268Z`.
They were fitted with the `apec_wide` cooling function and the `isobaric` scatter, whereas the sector's own defaults are `apec` and `constant`; build it with `make_cooling("apec_wide")` and `scatter="isobaric"` to predict what the fit saw.

```python
r = np.asarray(field.r_delta)[i3, None] * np.array([0.1, 0.3, 1.0])   # (NM, n) h^-1 Mpc
print(gas.n_e(r, m3, field.z, field.cosmo, dp, **on))         # cm^-3
print(gas.temperature(r, m3, field.z, field.cosmo, dp, **on))  # keV
```

## The cooling function and the emissivity

The gas emits as an optically thin plasma in collisional ionisation equilibrium, with the spectrum of APEC {cite:p}`SmithBrickhouseLiedahl_2001ApJ...556L..91S` integrated over a band.
The ratio $n_{\rm H}/n_{\rm e}$ is folded into the cooling function, so the profile carries one $n_{\rm e}^2$:

$$
\Lambda_{\rm APEC}(kT, Z) = \int_{E_1(1+z)}^{E_2(1+z)}\frac{{\rm d}\Lambda_{\rm APEC}}{{\rm d}E}\,{\rm d}E,
\qquad
\Lambda \equiv \frac{n_{\rm H}}{n_{\rm e}}\Lambda_{\rm APEC},
\qquad
\varepsilon = n_{\rm e}^2\,\Lambda(kT, Z)\,b_\sigma,
$$ (eq-emissivity)

in ${\rm erg\,cm^3\,s^{-1}}$, with $[E_1, E_2]$ the band as observed from redshift $z$ ($z = 0$ for a band in the rest frame).
The density scatters about the profile, log-normally with a dispersion $\sigma_{\rm scatter}$ in dex.
With `scatter="constant"` (the default) the boost $b_\sigma = \exp[(\sigma_{\rm scatter}\ln 10)^2]$ multiplies $n_{\rm e}^2$, as in {cite:t}`OppenheimerVoitBahe_2025MNRAS.543.2649O` (their Eq. 6), and is one at the default $\sigma_{\rm scatter} = 0$.
With `scatter="isobaric"` every phase sits at the local pressure, so a denser phase is a cooler one, and $\varepsilon$ is a Gauss–Hermite sum over the phases.
APEC is tabulated once on a $(\log T, \log Z)$ grid, built with `soxs`, and interpolated:

```{include} _generated/cooling.md
```

`apec` and `apec_wide` are the rest-frame $0.5$–$2$ keV band, the second clamped at a lower temperature; their tables ship with the package.
`apec_band` takes any band in the observer's frame and its table is built once, with `python -m ggah_mod.sectors.cooling --band` (it needs `soxs`), and raises until then.
The power law is a bolometric, bremsstrahlung-like null test; the sector never falls back to it.

## Integrated quantities

Two volume integrals give the observables of a halo, over the sphere $r < r_{\max}$ or over a shell chosen per halo:

$$
Y = \frac{\sigma_{\rm T}}{m_{\rm e}c^2}\int P_{\rm e}\,{\rm d}V,
\qquad
L_X^{\rm gas} = \int\varepsilon\,{\rm d}V,
\qquad
kT_X = \frac{\int\varepsilon\,kT\,{\rm d}V}{\int\varepsilon\,{\rm d}V},
$$ (eq-gas-integrals)

with $Y$ in $(h^{-1}{\rm Mpc})^2$, $L_X$ in ${\rm erg\,s^{-1}}$ in the band of the cooling function, and $T_X$ weighted by the band emissivity rather than by $n_{\rm e}^2$.
The gas mass integrates the density within an aperture $a = R_{\rm ap}/R_\Delta$,

$$
M_{\rm gas} = \mu_{\rm e}m_{\rm p}\int_0^{aR_\Delta}n_{\rm e}\,4\pi r^2\,{\rm d}r,
\qquad f_{\rm gas} = M_{\rm gas}/M,
$$ (eq-mgas)

with $\mu_{\rm e} = 1.14$.
The aperture is set to that of the measurement, not fitted; a gas fraction within $R_{500{\rm c}}$ is predicted at $a = R_{500{\rm c}}/R_\Delta$.
$f_{\rm gas}$ is not capped at the cosmic baryon fraction: an outer slope below three lets it grow with the aperture.

```python
print(gas.f_gas(m3, field.z, field.cosmo, dp, **on))                     # within R_Delta
print(gas.y_amplitude(m3, field.z, field.cosmo, dp, **on))               # (h^-1 Mpc)^2
print(gas.x_ray_luminosity(m3, field.z, field.cosmo, dp, **on))          # erg/s, 0.5-2 keV rest
print(gas.x_ray_temperature(m3, field.z, field.cosmo, dp, **on))         # keV
```

## Fourier profiles and tracer views

The hot gas is a continuous tracer, with no self-pair to exclude, and its weight is an amplitude times the normalised Fourier transform of the profile,

$$
w_{\rm extended} = A(M)\,u(k|M),
\qquad
u(k|M) = \frac{\int_0^{r_{\max}}g(r)\,j_0(kr)\,r^2{\rm d}r}{\int_0^{r_{\max}}g(r)\,r^2{\rm d}r},
$$ (eq-gas-uk)

with $r_{\max}$ the truncation radius `r_max_over_rdelta` in units of $R_\Delta$.
{py:meth}`~ggah_mod.sectors.gas.HotGasDPM.weights` reads it four ways:

| `view=` | profile $g(r)$ | amplitude $A(M)$ | observable |
|---|---|---|---|
| `"pressure"` (default) | $P_{\rm e}$ | $Y$ | thermal SZ |
| `"xray"` | $\varepsilon$ | $L_X^{\rm gas}$ | X-ray surface brightness |
| `"mass"`, `"density"` | $n_{\rm e}$ | $M_{\rm gas}$ | gas mass, kinetic SZ, dispersion measure |

{py:meth}`~ggah_mod.sectors.gas.HotGasDPM.bound_uk` is the density profile cut at $R_\Delta$, for a matter budget that carries the displaced gas in its own ejected component; {doc}`closure_options` gives its effect on $P_{\rm mm}$.

```python
for view in ("pressure", "xray", "mass"):
    w = gas.weights(field, dp, view=view)
    print(view, w.w_extended.shape, w.discrete)                # (Nk, NM), continuous
```

## The energy closure

A halo loses gas once the energy injected into it exceeds its binding energy, the argument of {cite:t}`DekelSilk_1986ApJ...303...39D` for supernovae and of {cite:t}`SilkRees_1998A&A...331L...1S` for black holes.
With `feedback="closure"`, built with an AGN and a galaxy sector as peers, the sector compares the energies the stars and the black holes have released by the redshift of the field with the binding energy at that redshift.
Moving a fraction $\Delta f_{\rm b}$ of the halo mass from $R_\Delta$ to $\eta_{\rm ej}R_\Delta$, in the potential of a point mass, costs

$$
E_{\rm bind} = \Delta f_{\rm b}\,M\,v_\Delta^2\left(1 - \eta_{\rm ej}^{-1}\right),
\qquad
v_\Delta^2 = \frac{GM}{R_\Delta/(1+z)},
$$ (eq-ebind)

and the energies are those of the AGN sector's black holes and of the galaxy sector's stars,

$$
E_{\rm AGN} = \epsilon_{\rm AGN}(M)\,\epsilon_{\rm r}\left[\langle M_{\rm BH}\rangle + M_{\rm BH}^{\rm sat}\right]c^2,
\qquad
E_{\rm SN} = \epsilon_{\rm SN}\,\frac{M_\star}{1 - R}\,e_{\rm SN},
$$ (eq-energies)

with $\epsilon_{\rm r}$ the radiative efficiency, $R$ the return fraction, $e_{\rm SN} = 5\times10^5\,({\rm km\,s^{-1}})^2$, and $M_\star$ the stars of the central galaxy and of the satellites.
Equating the two gives the expelled fraction, which a soft saturation caps at the baryons that are neither stars nor neutral gas:

$$
x = \frac{E_{\rm AGN} + E_{\rm SN}}{M v_\Delta^2(1 - \eta_{\rm ej}^{-1})},
\qquad
\Delta f_{\rm b} = c\left(1 - e^{-x/c}\right),
\qquad
c = f_{\rm b} - f_\star - f_{\rm cold} - f_{\rm ret}^{\min},
$$ (eq-gas-closure)

with $f_{\rm b} = \Omega_{\rm b}/\Omega_{\rm cb}$ and $f_{\rm ret}^{\min} = 0.01$.
The retained hot gas $f_{\rm gas}^{\rm budget} = f_{\rm b} - \Delta f_{\rm b} - f_\star - f_{\rm cold}$ rescales both amplitudes of Eq. {eq}`eq-dpm` by one factor per mass, $A(M) = [f_{\rm gas}^{\rm budget}/f_{\rm gas}](M) / [f_{\rm gas}^{\rm budget}/f_{\rm gas}](M_{\rm p})$, so the closure moves $M_{\rm gas}$, $Y$ and $L_X$ but not $kT$.
The coupling $\epsilon_{\rm AGN}(M)$ is a double power law in halo mass by default; {doc}`closure_options` gives it, the alternatives and their parameters:

```{include} _generated/params_energetics.md
```

```python
gal = S.GalaxySector("zumandelbaum15", calibration="off")
gp = S.galaxy_defaults("zumandelbaum15")
agn = S.AgnSector(gal, calibration="off")
closed = S.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure", agn=agn, galaxies=gal)
f_budget = closed.feedback_budget(field, dp, S.AgnParams(), gp)           # on field.m
print(np.interp(m3, np.asarray(field.m), np.asarray(f_budget)))
w_y = closed.weights(field, dp, view="pressure", agn_params=S.AgnParams(), galaxies_params=gp)
```

## Ejected baryons

The baryons the hot gas does not hold are a separate component, as in {cite:t}`SchneiderTeyssier_2015JCAP...12..049S`, spread over a Gaussian around the halo:

$$
\tilde u_{\rm ej}(k|M) = \exp\left(-\tfrac12k^2r_{\rm ej}^2\right),
\qquad r_{\rm ej} = \eta_{\rm ej}R_\Delta,
\qquad W_{\rm ej} = f_{\rm ej}\,M\,\tilde u_{\rm ej}.
$$ (eq-uej)

At $\eta_{\rm ej} = 2$, $90\%$ of the ejected gas lies within $5R_\Delta$.
{py:meth}`~ggah_mod.sectors.ejecta.EjectaSector.split_from_gas` builds the split whose hot gas is the DPM's own, so the ejected gas is exactly what the hot atmosphere does not hold; the `electrons` tracer of {doc}`layer4_spectra`, the DPM density plus the ejected mass, then closes on the budget:

```{include} _generated/params_ejecta.md
```

## The matter field

The matter field, the field that gravitational lensing sees, is one tracer with six components per halo:

$$
W_{\rm m}(k|M) = \frac{M}{\bar\rho_{\rm m}}\left[
   f_{\rm coll}\tilde u_{\rm DM} + f_{\rm hot}\tilde u_{\rm gas}
 + f_\star^{\rm cen} + f_\star^{\rm sat}\tilde u_{\rm sat}
 + f_{\rm cold}\tilde u_{\rm HI} + f_{\rm ej}\tilde u_{\rm ej}\right],
\qquad f_{\rm coll} = 1 - \Omega_{\rm b}/\Omega_{\rm cb},
$$ (eq-wm)

with $\bar\rho_{\rm m} = \Omega_{\rm m}\rho_{\rm c,0}$, neutrinos included, and $f_{\rm ej} = f_{\rm b} - f_{\rm hot} - f_\star^{\rm cen} - f_\star^{\rm sat} - f_{\rm cold}$.
The central galaxy's stars are a point mass, the satellites' stars sit on the satellite profile, and the neutral gas is that of the cold-gas sector ({py:mod}`ggah_mod.sectors.coldgas`), zero unless it is supplied.
The massive neutrinos free-stream and are in no halo: they enter only through $\bar\rho_{\rm m}$, and layer 4 adds them back on large scales.
Each $\tilde u \to 1$ as $k \to 0$ and the six fractions sum to one, so $W_{\rm m}(k\to0|M) = M/\bar\rho_{\rm m}$.
{py:class}`~ggah_mod.sectors.matter.BaryonSplit` holds the six fractions, built from a retained baryon fraction (`from_retained`) or from a hot-gas fraction (`from_hot`), and {py:class}`~ggah_mod.sectors.matter.MatterField` reads it from its parameters.

```python
f_cen, f_sat = gal.stellar_fraction(field, gp, satellites=True)
split = S.EjectaSector.split_from_gas(field, gas, dp, f_star_cen=f_cen, f_star_sat=f_sat)
w_m = S.MatterField().weights(field, {"split": split})
print(w_m.w_extended.shape)
```

## The cosmic baryon budget

Integrating the per-halo split over the mass function gives the share of the cosmic baryons in each phase,

$$
\Omega_i = \frac{1}{\rho_{\rm c,0}}\int_{M_{\min}}{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}\,M\,f_i(M),
\qquad
\Omega_{\rm b} = \sum_i\Omega_i + \Omega_{\rm outside},
$$ (eq-census)

with $\Omega_{\rm outside}$ the baryons the mass grid does not hold, in unresolved haloes or in no halo, which the halo model cannot separate.
It is set by where the mass grid stops rather than by any baryon parameter.
{py:func}`~ggah_mod.sectors.census.census` returns the budget by phase (`PHASES`), with ${\rm d}\Omega_i/{\rm d}\log_{10}M$ and the fraction of the cold matter the grid holds; the stars, neutral gas and hot gas compare with the rows of {cite:t}`FukugitaPeebles_2004ApJ...616..643F`.

```python
budget = S.census(field, split)
for phase, omega in budget.omega.items():
    print(f"{phase:9s} {float(omega) / float(budget.omega_b):.3f} of Omega_b")
print(f"outside   {float(budget.omega_outside) / float(budget.omega_b):.3f}")
```

## Refusals and units

- `conc` is required by every profile, integral and Fourier method, so that the profile shares the scale radius of the field it is used with.
- `x_ray_luminosity` and the `xray` view need double precision; the other quantities run in single precision.
- An unknown `feedback`, `scatter` or view raises, and so does `feedback="closure"` without its AGN and galaxy peers or their parameters.
- In the default `agn_energy="eps_dpl"` mode, `f_retained_energy` refuses a single coupling `log10_eps_agn`, passed, set or traced.
- `EjectaSector` and `MatterField` raise without a `split`.

## Related pages

- {doc}`notebooks/layer3_hot_gas`: the profiles, scaling relations and budget above, executed, with figures.
- {doc}`closure_options`: the AGN coupling of the closure and its options.
- {doc}`layer3_galaxies` and {doc}`layer3_agn`: the stars and black holes the closure reads.
- {doc}`layer4_spectra`: the gas, the electrons, the X-ray composite and the baryonic suppression of $P_{\rm mm}$.
- {doc}`api/sectors`: every function and its arguments.
