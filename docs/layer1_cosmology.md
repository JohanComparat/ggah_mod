# Layer 1: the cosmology

The cosmology layer is everything that depends on the cosmological parameters
alone.
It holds the parameter container and its density budget, the background
expansion and the distances, the linear power spectrum, and the amplitude and
growth read off it.

| module | provides |
|---|---|
| {py:mod}`ggah_mod.cosmology.parameters`, {py:mod}`~ggah_mod.cosmology.constants` | {py:class}`~ggah_mod.cosmology.parameters.Cosmology`, `PLANCK18`, the derived densities |
| {py:mod}`ggah_mod.cosmology.background` | $E(z)$, $\chi$, $f_K$, $D_{\rm A}$, $D_{\rm L}$, ${\rm d}V_{\rm c}/{\rm d}z\,{\rm d}\Omega$, $\mu$ |
| {py:mod}`ggah_mod.cosmology.power` | the linear $P_{\rm m}(k,z)$ and $P_{\rm cb}(k,z)$ from CLASS, CAMB or `emu_pk` |
| {py:mod}`ggah_mod.cosmology.amplitude` | $\sigma^2(R)$, $\sigma_8$, $S_8$ |
| {py:mod}`ggah_mod.cosmology.growth` | $D(z)$, $f(z)$, $f\sigma_8$, $\mathcal{S}(z)$ |

The layer runs in either flavour of the {doc}`overview`, and only
{py:mod}`~ggah_mod.cosmology.power` differs between them.
The background, the variance and the growth are written once, and are
differentiable exactly when the power spectrum driving them is.
Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import numpy as np
from ggah_mod.cosmology import PLANCK18, Cosmology
```

The {doc}`notebook <notebooks/layer1_cosmology>` works through the same material
with figures.

## Parameters and the density budget

The cosmology is an eleven-field frozen dataclass,
{py:class}`~ggah_mod.cosmology.parameters.Cosmology`, registered as a JAX pytree.
Its ten numerical fields are differentiable leaves, so `jax.grad` of anything
downstream flows through them; the eleventh, the neutrino ordering, is static.
The shipped `PLANCK18` is the *Planck* 2018 TT,TE,EE+lowE+lensing best fit
{cite:p}`PlanckCollaborationAghanimAkrami_2020A&A...641A...6P` at the minimal
neutrino mass, with the CMB temperature of {cite:t}`Fixsen_2009ApJ...707..916F`
as a parameter rather than a constant:

```{include} _generated/planck18.md
```

$\ln(10^{10}A_{\rm s})$ is the amplitude.
$\sigma_8$ and $S_8$ are outputs, integrals over the power spectrum given by
Eq. {eq}`eq-sigma8`, and supplying either to the container raises, since a
container holding both an amplitude and a $\sigma_8$ is over-determined.
To start from a $\sigma_8$, solve for the amplitude once with
{py:func}`~ggah_mod.cosmology.amplitude.ln10A_s_for_sigma8` (below) and carry the
amplitude afterwards.

```python
cosmo = Cosmology.create(Omega_m=0.30, sum_mnu=0.10, Omega_k=0.0)
try:
    Cosmology.create(sigma8=0.81)
except TypeError as err:
    print(err)
```

Build a cosmology from user input with `Cosmology.create` or `.replace`, which
check the neutrino masses against the ordering and refuse the retired keys.
The plain constructor is branch-free, for use inside `jax.jit`, and checks
nothing.

The comoving densities are measured against the critical density today,

$$
\rho_{\rm crit,0} = \frac{3H_{100}^2}{8\pi G}
= 2.77537\times10^{11}\,h^{-1}{\rm M}_\odot\,(h^{-1}{\rm Mpc})^{-3},
$$ (eq-rho)

and every density the model contains enters one budget,

$$
\Omega_\gamma + \Omega_{\rm b} + \Omega_{\rm cdm} + \Omega_\nu^{\rm nr}
+ \Omega_\nu^{\rm r} + \Omega_k + \Omega_{\rm DE} = 1 ,
$$ (eq-closure)

where $\Omega_\nu^{\rm nr}$ and $\Omega_\nu^{\rm r}$ are the non-relativistic and
relativistic parts of the neutrino density.
The closure fixes $\Omega_{\rm DE}$, the one density the container does not set.
$\Omega_{\rm m}$ contains the neutrinos' rest mass, and the cold matter is named
separately,

$$
\Omega_{\rm m} \equiv \Omega_{\rm b} + \Omega_{\rm cdm} + \Omega_\nu^{\rm nr},
\qquad
\Omega_{\rm cb} \equiv \Omega_{\rm b} + \Omega_{\rm cdm} ,
$$ (eq-cb-m)

so the *Planck* $\Omega_{\rm m} = 0.3100$ gives $\Omega_{\rm cdm} = 0.2593$ rather
than $\Omega_{\rm m} - \Omega_{\rm b}$.
The photons are a blackbody at $T_{\rm CMB}$,
$\Omega_\gamma = 2.47298\times10^{-5}\,h^{-2}\,(T_{\rm CMB}/2.7255\,{\rm K})^4$.

```python
c = PLANCK18
budget = (c.Omega_gamma + c.Omega_b + c.Omega_cdm + c.Omega_nu + c.Omega_nu_r
          + c.Omega_k + c.Omega_de)
print(budget)                         # 1 to round-off
print(c.Omega_cdm, c.Omega_cb)        # the cold matter excludes the neutrinos
print(c.rho_matter, c.rho_cold)       # h^-1 Msun (h^-1 Mpc)^-3
```

## Neutrinos

Two fields describe the neutrinos: the mass sum $\sum m_\nu$, a differentiable
leaf, and the ordering `nu_hierarchy`, which divides it among the three mass
eigenstates.
`massless` takes $\sum m_\nu = 0$ alone; `degenerate` gives $\sum m_\nu/3$ to each
state; `normal` and `inverted` split the sum with the measured oscillation
splittings {cite:p}`EstebanGonzalez-GarciaMaltoni_2024JHEP...12..216E` and admit
it only above their floors, 0.0590 and 0.0994 eV.
The ordering is static: large-scale structure constrains the sum and not its
division, so there is no gradient to take.
`PLANCK18` declares `degenerate`; a cosmology built from scratch defaults to
`normal`. Neither reproduces the *Planck* 2018 baseline exactly, which assumes the
normal ordering at its minimal mass but approximates it as two massless states
and a single massive one of 0.06 eV; `degenerate` is the convention *Planck*
adopts when it varies $\sum m_\nu$, neglecting the splittings at its sensitivity
({cite:t}`PlanckCollaborationAghanimAkrami_2020A&A...641A...6P`, Sects. 2.1 and 7.5.1).

The neutrinos keep the Fermi–Dirac momentum spectrum they decoupled with
{cite:p}`LesgourguesPastor_2006PhR...429..307L`, and each state's energy density
relative to massless is

$$
F(y) = \frac{120}{7\pi^4}\int_0^\infty {\rm d}x\,
\frac{x^2\sqrt{x^2+y^2}}{e^x+1},\qquad y = \frac{m}{k_{\rm B}T_{\rm ncdm}},
$$ (eq-komatsu)

which is 1 for a massless state and $\kappa y$, with
$\kappa = 180\zeta(3)/7\pi^4$, once it is cold
{cite:p}`KomatsuSmithDunkley_2011ApJS..192...18K`.
This package integrates $F$ rather than using the Komatsu et al. fit.
The massive states are relics at $T_{\rm ncdm} = 0.71611\,T_{\rm CMB}$, and a
massless remainder $\Delta N_{\rm ur} = 0.0044$ carries the rest of
$N_{\rm eff} = 3.044$, as CLASS divides them.
The rest mass is then

$$
\Omega_\nu^{\rm nr} = \frac{\sum m_\nu}{93.14\,{\rm eV}\,h^2},
$$ (eq-nu-budget)

and $\Omega_\nu^{\rm r}$ is what the neutrinos carry beyond it: the kinetic
energy of the massive states and the massless remainder.
$\Omega_\nu^{\rm nr}$ depends on the sum alone, so $\Omega_{\rm cb}$, $f_\nu$,
$\bar\rho_{\rm cb}$ and all of layer 2 are bit-for-bit independent of the
ordering; only $\Omega_\nu^{\rm r}$ feels it.

| symbol | property |
|---|---|
| $\Omega_\nu^{\rm nr}$, the rest mass | `Omega_nu`, `Omega_nu_matter` |
| $\Omega_\nu^{\rm r}$, the relativistic part today | `Omega_nu_r` |
| $\Omega_\nu^{\rm nr} + \Omega_\nu^{\rm r}$, what $E(z)$ carries | `Omega_nu_today` |
| the density of three massless species | `Omega_nu_rel` |
| the massless remainder $\Omega_{\rm ur}$ | `Omega_ur` |
| $f_\nu = \Omega_\nu^{\rm nr}/\Omega_{\rm m}$ | `f_nu` |
| the three masses, eV | `nu_masses` |

```python
for ordering in ("degenerate", "normal"):
    c = PLANCK18.replace(nu_hierarchy=ordering)
    print(ordering, np.asarray(c.nu_masses), c.Omega_nu, c.Omega_nu_r)
```

## Background expansion and distances

The expansion rate carries the complete density budget,

$$
\begin{aligned}
E^2(z) ={}& \Omega_\gamma(1+z)^4 + \Omega_{\rm cb}(1+z)^3
+ \big[(\Omega_\nu^{\rm ur} - \Omega_{\rm ur})\,S_\nu(z) + \Omega_{\rm ur}\big](1+z)^4\\
&+ \Omega_k(1+z)^2 + \Omega_{\rm DE}\,f_{\rm DE}(z),
\end{aligned}
$$ (eq-Ez)

where $S_\nu(z) = \tfrac13\sum_j F\big(y_j/(1+z)\big)$ turns the neutrinos from
radiation into matter as they cool, and dark energy follows the
$w(a) = w_0 + w_a(1-a)$ equation of state of
{cite:t}`ChevallierPolarski_2001IJMPD..10..213C` and
{cite:t}`Linder_2003PhRvL..90i1301L`,

$$
f_{\rm DE}(z) = (1+z)^{3(1+w_0+w_a)}\exp\!\left(\frac{-3w_a z}{1+z}\right).
$$ (eq-fde)

The comoving distance, in $h^{-1}$Mpc, is the *radial* distance,

$$
\chi(z) = \frac{c}{H_0}\int_0^z\frac{{\rm d}z'}{E(z')} ,
$$ (eq-chiz)

and curvature enters only through the transverse distance,

$$
f_K(\chi) = \chi\,s(x),\qquad x = \Omega_k\left(\frac{\chi}{D_H}\right)^2,\qquad
s(x) = \frac{\sinh\sqrt{x}}{\sqrt{x}} = \sum_{n\ge0}\frac{x^n}{(2n+1)!},
$$ (eq-fk)

with $D_H = c/H_0$.
This is Eq. (16) of {cite:t}`Hogg_1999astro.ph..5116H` written once for open,
flat and closed geometries.
Because $s$ is entire, $\Omega_k = 0$ is an interior point: $f_K = \chi$ holds
exactly there, and the derivative with respect to $\Omega_k$ is smooth through
it.
The other distances follow,

$$
\begin{aligned}
D_{\rm A} &= \frac{f_K}{1+z}, &
D_{\rm L} &= (1+z)f_K,\\
\frac{{\rm d}V_{\rm c}}{{\rm d}z\,{\rm d}\Omega} &= \frac{c}{H_0}\frac{f_K^2}{E(z)}, &
\mu &= 5\log_{10}\frac{D_{\rm L}/h}{10\,{\rm pc}} ,
\end{aligned}
$$ (eq-distances)

all in $h^{-1}$Mpc except $\mu$, an observable, which carries no $h$.

```python
from ggah_mod.cosmology import (angular_diameter_distance, comoving_distance,
                                distance_modulus, hubble_e)
from ggah_mod.cosmology.background import transverse_distance

z = np.array([0.1, 0.5, 1.0, 2.0])
E = hubble_e(z, PLANCK18)
chi = comoving_distance(z, PLANCK18)            # h^-1 Mpc; at least 1-D
DA = angular_diameter_distance(z, PLANCK18)
mu = distance_modulus(z, PLANCK18)

assert np.all(transverse_distance(chi, PLANCK18) == chi)   # flat: exact

curved = PLANCK18.replace(Omega_k=0.05)
chi_c = comoving_distance(z, curved)
print(transverse_distance(chi_c, curved) / chi_c)          # > 1 when open
```

## The sound horizon at the drag epoch

The BAO scale is the comoving distance sound travels in the photon-baryon fluid
before the baryons are released from the photons' drag,

$$
r_s(z) = \int_z^\infty \frac{c_s\,{\rm d}z'}{H(z')},\qquad
c_s = \frac{c}{\sqrt{3(1+R)}},\qquad
R = \frac{3\rho_b}{4\rho_\gamma} = \frac34\frac{\Omega_b}{\Omega_\gamma}\frac{1}{1+z}
$$ (eq-rs)

{cite:p}`EisensteinHu_1998ApJ...496..605E`, at the drag redshift $z_d$ where the
baryons' drag depth reaches one {cite:p}`HuSugiyama_1996ApJ...471..542H`:

$$
\tau_d(z_d) = \int_0^{z_d}\frac{{\rm d}\tau_T}{{\rm d}z'}\,\frac{{\rm d}z'}{R(z')} = 1 ,
$$ (eq-zd)

with $\tau_T$ the Thomson optical depth. CLASS and CAMB both find $z_d$ this way.

**The integral is exact; the drag redshift is fitted.**
`sound_horizon` integrates Eq. {eq}`eq-rs` over this package's own $E(z)$, so the
neutrinos, the massless remainder of $N_{\rm eff}$ and $T_{\rm CMB}$ are the ones
every distance here uses. Given CLASS's $z_d$ it reproduces CLASS's `rs_d` to
$1\times10^{-7}$ at the fiducial.
Eq. {eq}`eq-zd` needs the free-electron history, so a recombination code and a
helium fraction, which this package does not carry.
`z_drag` is therefore a cubic polynomial in
$(\ln\omega_b, \ln\omega_{cb}, \ln T_{\rm CMB}, \Sigma m_\nu)$, fitted to CLASS 3.3.4
with HyRec-2 recombination
{cite:p}`LeeAli-Haimoud_2020PhRvD.102h3517L,Ali-HaimoudHirata_2011PhRvD..83d3513A`.
It was fitted on 1915 cosmologies across the `emu_pk` training box, with
$T_{\rm CMB}$ allowed $\pm1\%$, by `tools/calibrate_zdrag.py`.
`r_drag` is $r_s(z_d)$, in $h^{-1}$Mpc like every distance here, so $D_M/r_d$
needs no $h$.

```python
from ggah_mod.cosmology import r_drag, sound_horizon, z_drag

rd = r_drag(PLANCK18)                           # h^-1 Mpc
print(float(z_drag(PLANCK18)), float(rd / PLANCK18.h))   # ~1059.7, ~147.74 Mpc
DM = transverse_distance(comoving_distance(0.51, PLANCK18), PLANCK18)
print(float(DM[0] / rd))                        # D_M/r_d at DESI's LRG1, no h
```

| Comparison (largest difference) | $z_d$ | $r_d$, relative |
|---|---|---|
| fit against CLASS, 485 held-out cosmologies | 0.017 | |
| fit against CLASS, 192 box corners | 0.029 | |
| `r_drag` against CLASS, flat cosmologies | | $4.5\times10^{-6}$ |
| CAMB against CLASS, across the box | 0.75 | $6.6\times10^{-4}$ |
| {cite:t}`AubourgBaileyBautista_2015PhRvD..92l3516A` Eq. 16, at the fiducial | | $3.5\times10^{-4}$ |
| {cite:t}`EisensteinHu_1998ApJ...496..605E` Eq. 4 for $z_d$, at the fiducial | 39 | $2.5\times10^{-2}$ |

Since $\partial\ln r_d/\partial z_d = -6\times10^{-4}$, the fit's 0.03 in $z_d$ is
$2\times10^{-5}$ in $r_d$: tens of times below the spread between the two
Boltzmann codes,
which comes from their recombination codes (HyRec-2 against RECFAST
{cite:p}`SeagerSasselovScott_1999ApJ...523L...1S`) and their helium tables.
The Aubourg et al. formula is a different trade.
It is closed-form and accurate to 0.021 per cent near the *Planck* cosmology at
$N_{\rm eff} = 3.046$, but it has no $T_{\rm CMB}$ dependence and leaves that
range inside the box.

**Curvature.**
CLASS multiplies ${\rm d}r_s$ by $\sqrt{1-Kr_s^2}$; CAMB does not, and neither
does this package, because $r_s$ is a comoving length and curvature enters where
it becomes an angle, in $f_K$.
The two conventions differ by up to $6.6\times10^{-5}$ at $|\Omega_k| = 0.15$.

**Where it refuses.**
Outside the calibration box, and wherever dark energy is more than $10^{-3}$ of
the density at the drag epoch.
That happens when $1+w_0+w_a > 0$, where dark energy grows into the past.
At $w_0 = -0.5$, $w_a = 0.6$ it is 70 per cent of the density at recombination
and moves $z_d$ by tens.
The bound excludes 4.25 per cent of the `emu_pk` box; DESI-like values such as
$w_0 = -0.75$, $w_a = -0.86$ are well inside it.

**The helium fraction, and every assumption behind it.**
$z_d$ depends on the helium fraction through the electron density:
$\partial z_d/\partial Y_{\rm He} = 35$, $\partial\ln r_d/\partial Y_{\rm He} = -0.021$.
The fit inherits CLASS's.
The assumptions are recorded, as read from CLASS and its table, in
`ggah_mod.cosmology._zdrag_coefficients.BBN`, and a slow test fails if they move:

- $Y_{\rm He}$ is not a parameter. It is CLASS's standard-BBN interpolation at
  $\omega_b$, so a function of $\omega_b$ alone here, because $N_{\rm eff} = 3.044$ is
  fixed: 0.24537 at the fiducial, 0.2427–0.2475 across the box.
- The table is `sBBN_2017.dat`, which this package pins for its range in $\omega_b$
  (to 0.0399).
  - It was computed with PArthENoPE {cite:p}`PisantiCirilloEsposito_2008CoPhC.178..956P`
    for a neutron lifetime of 880.2 s, "identical to standard assumptions of
    *Planck* 2017 papers" (its header); see Sect. 2.1 of
    {cite:t}`PlanckCollaborationAghanimAkrami_2020A&A...641A...6P`.
  - CLASS's own default, `sBBN_2025.dat` (878.4 s), changes $Y_{\rm He}$ by
    $-9\times10^{-5}$ and $r_d$ by $+2\times10^{-6}$.
- CLASS looks the table up at $\Delta N = N_{\rm eff} - 3.046 = -0.002$, because the
  table's reference is 3.046; this is worth $-2.7\times10^{-5}$ in $Y_{\rm He}$.
- CLASS does not rescale $\omega_b$ by $(2.7255\,{\rm K}/T_{\rm CMB})^3$ when it reads
  the table (CAMB does), so $Y_{\rm He}$ does not follow $T_{\rm CMB}$ in the fit.
  This matters only away from 2.7255 K.
- The table's column is the helium mass fraction, used without conversion.
- Standard BBN: no electron-neutrino chemical potential, constants not varied.
- CAMB, as this package calls it, uses PRIMAT
  {cite:p}`PitrouCocUzan_2018PhR...754....1P` with a neutron lifetime of 879.4 s:
  $Y_{\rm He} = 0.24586$ at the fiducial, $+4.9\times10^{-4}$, which is $-1\times10^{-5}$
  of its $-4.4\times10^{-5}$ difference from CLASS in $r_d$ there.

A free $Y_{\rm He}$ or $N_{\rm eff}$ would need a refit with those axes.

## The linear power spectrum

Each backend returns the total-matter spectrum $P_{\rm m}(k,z)$ and the
cold (CDM + baryon) spectrum $P_{\rm cb}(k,z)$, in $(h^{-1}{\rm Mpc})^3$ with
$k$ in $h\,{\rm Mpc}^{-1}$.
They are not interchangeable.
Haloes form out of the cold field, so $\sigma(M)$, ${\rm d}n/{\rm d}M$ and $b(M)$
are built from $P_{\rm cb}$; lensing sees all the matter and uses $P_{\rm m}$.
{py:func}`~ggah_mod.cosmology.power.make_pk` builds a backend by name:

```{include} _generated/pk_backends.md
```

CLASS {cite:p}`BlasLesgourguesTram_2011JCAP...07..034B` is the default of
`ACCURATE`, because `emu_pk` was trained on it, so both flavours rest on one
solver; CAMB {cite:p}`LewisChallinorLasenby_2000ApJ...538..473L` is the
alternative.
CLASS integrates the massive-neutrino hierarchy until a mode is well inside the
horizon and switches to the fluid approximation of
{cite:t}`LesgourguesTram_2011JCAP...09..032L` from there on.
Both run at raised precision (`CLASS_PRECISION`, `CAMB_PRECISION`), sample
$k\in[10^{-5}, 300]\,h\,{\rm Mpc}^{-1}$ and interpolate log–log, and memoise each
solve on the cosmology and the tuple of redshifts, so pass all redshifts in one
call rather than looping.
CLASS answers up to $z = 5$ (`Z_MAX_PK`) and refuses beyond.
`emu_pk` is a neural-network emulator of CLASS's linear $P_{\rm m}$ and
$P_{\rm cb}$ with the architecture of CosmoPower
{cite:p}`SpurioManciniPiras_2022MNRAS.511.1771S`, trained on CLASS at its default
precision for $z \le 5$ and $k \le 200\,h\,{\rm Mpc}^{-1}$.
It restores $n_{\rm s}$ and $\ln(10^{10}A_{\rm s})$ in closed form rather than
learning them, so the derivatives with respect to those two are exact.
It refuses a cosmology outside its training box rather than extrapolating.

```python
from ggah_mod.cosmology import make_pk

pk = make_pk("emu_pk")          # make_pk("class") needs the [reference] extra
k = np.logspace(-4, np.log10(200.0), 512)
z = np.array([0.0, 0.5, 1.0])
P_m = pk.pk(k, z, PLANCK18)     # shape (3, 512): total matter
P_cb = pk.pk_cb(k, z, PLANCK18) # cold dark matter and baryons

try:
    pk.pk(k, 0.0, PLANCK18.replace(h=0.9))
except ValueError as err:
    print(err)                  # outside the emulator's box
```

## Amplitude, variance and growth

The variance of the linear field in a real-space top hat of radius $R$ is the one
integral this layer exports; layer 2 calls it with a mass-dependent radius.
The window's closed form cancels catastrophically at small argument, so it is
evaluated in two branches,

$$
W(x) = \begin{cases} 1 - x^2/10 + x^4/280, & x < 0.1,\\
3(\sin x - x\cos x)/x^3, & x \ge 0.1,\end{cases}
\qquad
\sigma^2(R) = \frac{1}{2\pi^2}\int P(k)\,W^2(kR)\,k^3\,{\rm d}\ln k ,
$$ (eq-sigma2)

with the integral a trapezoid in $\ln k$.
The amplitude diagnostics are taken on the total-matter spectrum, since that is
what the priors quoted on them assume,

$$
\sigma_8 = \sigma(8\,h^{-1}{\rm Mpc})\ \text{on}\ P_{\rm m},\qquad
S_8 = \sigma_8\sqrt{\Omega_{\rm m}/0.3} .
$$ (eq-sigma8)

There is one growth factor and one route to it, the ratio of $\sigma_8$ at two
redshifts, with no fitting formula and no separate differential equation:

$$
D_X(z) = \frac{\sigma_8[P_X(k,z)]}{\sigma_8[P_X(k,0)]},\qquad X\in\{{\rm m},{\rm cb}\},
\qquad
f(z) = \frac{{\rm d}\ln D}{{\rm d}\ln a},\qquad f\sigma_8 = f(z)\,\sigma_8(z) .
$$ (eq-growth)

The growth rate is a second-order difference of $D$ in $\ln(1+z)$ from one
backend call.
Once neutrinos have mass, free streaming makes $\sqrt{P(k,z)/P(k,0)}$ depend on
$k$ and a scalar $D$ is ambiguous;
{py:func}`~ggah_mod.cosmology.growth.growth_scale_spread` reports the size of
that ambiguity,
$\mathcal{S}(z) = \max_{k\in[10^{-3},10]}\big|\sqrt{P(k,z)/P(k,0)}/D(z) - 1\big|$.

```python
from ggah_mod.cosmology import growth_factor, ln10A_s_for_sigma8, s8, sigma8
from ggah_mod.cosmology.growth import f_sigma8, growth_rate

P0 = pk.pk(k, 0.0, PLANCK18)
print(sigma8(P0, k), s8(P0, k, PLANCK18))

z = np.array([0.0, 0.5, 1.0, 2.0])
D = growth_factor(z, PLANCK18, pk)            # variant="cold" grows P_cb instead
f = growth_rate(z, PLANCK18, pk)
fs8 = f_sigma8(z, PLANCK18, pk)

lnA = ln10A_s_for_sigma8(0.81, PLANCK18, pk, k=k)   # once, up front
print(sigma8(pk.pk(k, 0.0, PLANCK18.replace(ln10A_s=lnA)), k))
```

## Differentiating

A `Cosmology` is a pytree, so `jax.grad` with respect to it returns a
`Cosmology` whose fields are the derivatives.
Every function on this page is written in `jax.numpy` and can be differentiated
and compiled; with `emu_pk` the power spectrum can be too.

```python
import jax.numpy as jnp

grad = jax.grad(lambda c: comoving_distance(1.0, c)[0])(PLANCK18)
print(grad.Omega_m, grad.h, grad.Omega_k)       # d chi(z=1) / d theta

def ln_sigma8(Omega_m):
    c = PLANCK18.replace(Omega_m=Omega_m)
    return jnp.log(sigma8(pk.pk(k, 0.0, c), k))

print(jax.grad(ln_sigma8)(0.31))
```

The ordering, a string, is not a leaf and has no derivative.
`ln10A_s_for_sigma8` is a root-finder in plain Python and is not traceable; it
is meant to be called once, before any differentiation.

## Accuracy

The technical paper (Comparat, in prep.) compares every quantity on this page,
values and derivatives, against CLASS, CAMB, CCL
{cite:p}`ChisariAlonsoKrause_2019ApJS..242....2C`, `astropy`
{cite:p}`AstropyCollaborationPriceWhelanLim_2022ApJ...935..167A`, `jax_cosmo`
{cite:p}`CampagneLanusseZuntz_2023OJAp....6E..15C`, Cobaya
{cite:p}`TorradoLewis_2021JCAP...05..057T`, and the differentiable Boltzmann
solvers DISCO-EB {cite:p}`HahnListPorqueres_2024JCAP...06..063H` and SymBoltz.jl
{cite:p}`Sletmoen_2026A&A...707A.128S`.
`emu_pk` reproduces CLASS at CLASS's default precision, and so carries that
precision's error; the {doc}`notebook <notebooks/layer1_cosmology>` measures its
ratio to CLASS at the fiducial cosmology when `classy` is installed.

## Next

- {doc}`notebooks/layer1_cosmology`: this page, executed, with figures.
- {doc}`layer2_halos`: what the halo layer builds on these quantities.
- {doc}`api/cosmology`: every function and its arguments.
