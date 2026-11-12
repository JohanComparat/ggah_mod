# Layer 3: galaxies

Layer 3 populates the halo field of {doc}`layer2_halos` with tracers: galaxies,
active nuclei, hot and cold gas.
Each tracer returns its weight per halo, $W(k|M)$, which layer 4 integrates over the mass function into a power spectrum.
This page covers the galaxies: their occupation, their weights as a tracer, and the modifiers and samples built on them (mis-centring, satellite kinematics, richness-selected clusters, intrinsic alignments).
The active nuclei and the hot gas have pages of their own, {doc}`layer3_agn` and {doc}`layer3_hot_gas`.

The technical paper (Comparat, in prep., Sec. 4.1) documents one galaxy model,
the default: the occupation of
{cite:t}`ZuMandelbaum_2015MNRAS.454.1161Z`, `zumandelbaum15`, with its colour
split by the halo quenching of {cite:t}`ZuMandelbaum_2016MNRAS.457.4360Z`,
`zumandelbaum16_red` and `zumandelbaum16_blue`.
The registries it belongs to hold ten other occupations, two conditional
luminosity functions, and seven stellar-mass relations, five of which are
written forwards and so reach the halo-mass grid without a numerical inversion.
Every entry carries a row of a calibration table recording what it was fitted
to; the three tables below are generated from the code on every build.

:::{warning}
Layer 3 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.sectors` emits a `ggah_mod.WorkInProgressWarning` once per
session, which can be silenced on its own, as the snippets below do and as the
{doc}`overview` shows.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.sectors.occupation` | the thirteen occupation models: `OCCUPATION`, `OCC_CALIBRATION`, `DEFAULTS`, {py:func}`~ggah_mod.sectors.occupation.make_occupation` |
| {py:mod}`ggah_mod.sectors.clf` | the two conditional luminosity functions: `CLF`, `CLF_CALIBRATION`, `CLF_DEFAULTS`, {py:func}`~ggah_mod.sectors.clf.make_clf` |
| {py:mod}`ggah_mod.sectors.sham` | the seven stellar-mass relations: `SHMR`, `SHMR_CALIBRATION`, `SHMR_MASS_UNITS`, {py:func}`~ggah_mod.sectors.sham.make_shmr` |
| {py:mod}`ggah_mod.numerics` | {py:func}`~ggah_mod.numerics.invert_monotone`, the inverse that keeps its gradient |
| {py:mod}`ggah_mod.sectors.galaxies` | {py:class}`~ggah_mod.sectors.galaxies.GalaxySector`, which reaches any occupation or luminosity function by name, and {py:func}`~ggah_mod.sectors.galaxies.galaxy_defaults` |
| {py:mod}`ggah_mod.sectors.protocol` | {py:class}`~ggah_mod.sectors.protocol.TracerWeights`, what every sector returns to layer 4 |
| {py:mod}`ggah_mod.sectors.miscentering` | {py:class}`~ggah_mod.sectors.miscentering.MisCentering`, the displacement of the central galaxy |
| {py:mod}`ggah_mod.sectors.kinematics` | {py:func}`~ggah_mod.sectors.kinematics.satellite_velocity_dispersion` |
| {py:mod}`ggah_mod.sectors.clusters` | {py:class}`~ggah_mod.sectors.clusters.ClusterSector`, richness-selected clusters |
| {py:mod}`ggah_mod.sectors.alignments` | {py:class}`~ggah_mod.sectors.alignments.IntrinsicAlignmentSector` and {py:func}`~ggah_mod.sectors.alignments.alignment_bias` |
| {py:mod}`ggah_mod.sectors.params` and {py:mod}`ggah_mod.sectors.occupation_params` | the parameter containers, their bounds and their priors |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import GalaxySector, galaxy_defaults
from ggah_mod.sectors import clf as C, occupation as O, sham as SH

lm = np.array([11.0, 12.0, 13.0, 14.0, 15.0])     # log10 M [h^-1 Msun]
field = make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.2)
```

The {doc}`notebook <notebooks/layer3_galaxies>` draws every model below at its
defaults, and the derivative the inversion keeps.

## Model selection and calibration

An occupation or a conditional luminosity function is selected by its name, as
`GalaxySector(model=...)` or through `make_occupation` and `make_clf`, each of
which returns the pair of functions
$(\langle N_{\rm c}\rangle, \langle N_{\rm s}\rangle)$.
The relations sit in a registry of their own, `sham.SHMR`, reached through
`make_shmr`, which `GalaxySector` calls for its `shmr=` argument and for the
relation a threshold occupation counts its centrals with.
A calibration row records what the entry was fitted to and the redshift range;
the occupation and luminosity-function rows also record the selection, which the
relation rows leave empty.
Applying a model outside its own row is a systematic error rather than a
tolerance, so none of them is a fallback for the default.
`GalaxySector` compares the field's redshift with the row when it computes its
weights, and warns outside it (`calibration="warn"`, the default; `"strict"`
raises and `"off"` is silent).
It hands each function exactly the parameters that function's signature names,
so one set of defaults serves both terms of a pair.

```python
for name in ("zheng07", "more15", "leauthaud12", "zacharegkas25", "vanuitert16"):
    sector = GalaxySector(name)
    n_cen, n_sat = sector.occupation(field, galaxy_defaults(name))   # on field.m
    nbar = float(sector.number_density(field, galaxy_defaults(name)))
    print(f"{name:14s} n = {nbar:.3e} h^3 Mpc^-3")
```

## The default model

The models below are written against the default, and several are its shape in
another convention; the technical paper derives its equations and records the
fit its parameters default to.
Throughout, $\log$ is $\log_{10}$ and halo masses are in $h^{-1}{\rm M}_\odot$;
this relation's stellar masses are in $h^{-2}{\rm M}_\odot$.
It is written backwards, as halo mass against stellar mass, with a scatter that
depends on halo mass:

$$
\begin{aligned}
\log_{10}M_h &= \log_{10}M_1 + \beta x
  + \frac{1}{\ln 10}\left[\frac{t^{\delta}}{1 + t^{-\gamma}} - \tfrac12\right],\\
\sigma_{\ln M_\star}(M_h) &= \max\left\{\sigma_{\rm floor},\ \sigma_0
   + \eta\max\left[0,\ \log_{10}(M_h/M_1)\right]\right\},
\end{aligned}
$$ (eq-zm-shmr)

with $x = \log_{10}(M_\star/M_\star^0)$, $t = 10^{x}$ and
$\sigma_{\rm floor} = 0.01$.
The occupation needs $M_\star^{\rm c}(M_h)$ on the halo-mass grid, so the
relation is inverted numerically:
{py:func}`~ggah_mod.numerics.invert_monotone` brackets the root by bisection
under a `stop_gradient` and takes one Newton step,

$$
x_0 = \mathrm{stop\_grad}\left[\tfrac12(a_N + b_N)\right], \qquad
x = x_0 - \frac{f(x_0) - y}{f'(x_0)}, \qquad
f' = \partial_x f \ \text{by autodiff},
$$ (eq-invert)

with $[a_N, b_N]$ the bracket after $N$ bisection steps and $y$ the target value.
Centrals above a stellar-mass threshold $M_\star^{\rm t}$ are

$$
\langle N_{\rm c}\rangle = \frac{f_{\rm c}}{2}\,{\rm erfc}\!\left[
  \frac{\left(\log M_\star^{\rm t} - \log M_\star^{\rm c}(M_h)\right)\ln 10}
       {\sqrt2\,\sigma_{\ln M_\star}(M_h)}\right],
$$ (eq-zm-cen)

and satellites

$$
\begin{aligned}
\langle N_{\rm s}\rangle &= \langle N_{\rm c}\rangle
   \left(\frac{M_h}{M_{\rm sat}}\right)^{\alpha_{\rm sat}}e^{-M_{\rm cut}/M_h},\\
M_{\rm sat} &= b_{\rm sat}\,\hat m^{\beta_{\rm sat}}\,10^{12}, \qquad
M_{\rm cut} = b_{\rm cut}\,\hat m^{\beta_{\rm cut}}\,10^{12}, \qquad
\hat m = M_{\min}/10^{12},
\end{aligned}
$$ (eq-zm-sat)

where $M_{\min}$ is Eq. {eq}`eq-zm-shmr` evaluated at the threshold, the halo
mass whose mean central stellar mass is $M_\star^{\rm t}$; the relation is used
forwards there, so the satellites need no inversion.
The colour split multiplies each term by a red fraction, and the blue
populations are the complements, so the two colours sum to the total at every
mass:

$$
\begin{aligned}
\langle N_{\rm c}^{\rm red}\rangle &= f(M_h; M_h^{qc}, \mu_{\rm c})\,\langle N_{\rm c}\rangle,
\qquad
\langle N_{\rm s}^{\rm red}\rangle = f(M_h; M_h^{qs}, \mu_{\rm s})\,\langle N_{\rm s}\rangle,\\
f(M_h; M_h^{q}, \mu) &= 1 - \exp\left[-\left(M_h/M_h^{q}\right)^{\mu}\right],
\end{aligned}
$$ (eq-zm16-quench)

with $(M_h^{qc}, \mu_{\rm c})$ for centrals and $(M_h^{qs}, \mu_{\rm s})$ for
satellites.

```python
p = galaxy_defaults("zumandelbaum15")
shape = {k: p[k] for k in ("lg_m1h", "lg_m0star", "beta", "delta", "gamma")}
lms = SH.mstar_from_mh_zu15(lm, **shape)           # log10 M_*^c [h^-2 Msun]
print(lms, SH.mh_from_mstar_zu15(lms, **shape) - lm)   # the round trip: ~1e-15
```

## Occupations

The kernel everything else decorates is
{cite:t}`ZhengCoilZehavi_2007ApJ...667..760Z`, `zheng07`:

$$
\langle N_{\rm c}\rangle = \tfrac12\left[1 + {\rm erf}\frac{\log M - \log M_{\min}}
                                         {\sigma_{\log M}}\right]
      = \tfrac12\,{\rm erfc}\frac{\log M_{\min} - \log M}{\sigma_{\log M}},
$$ (eq-zheng-cen)

$$
\langle N_{\rm s}\rangle = \langle N_{\rm c}\rangle\left(\frac{M - M_0}{M_1}\right)^{\alpha},
\quad M > M_0,
$$ (eq-zheng-sat)

and zero below $M_0$.
That cut-off is written as a `where` on the *mass* rather than a `clip` on the
ratio: below $M_0$ the bracket is negative and raising it to a fractional power
gives `nan`, and a `clip` at zero would put a tie exactly at $M_0$ and split the
gradient there.
{cite:t}`ConroyWechslerKravtsov_2006ApJ...647..201C`, after
{cite:t}`TinkerWeinbergZheng_2005ApJ...631...41T`, replace the kink with an
exponential cut-off and drop the $-M_0$ from the power law; gated by the central
and with $\alpha$ free, as in Eq. {eq}`eq-zm-sat`, that is

$$
\langle N_{\rm s}\rangle = \langle N_{\rm c}\rangle\left(\frac{M}{M_1}\right)^{\alpha}e^{-M_0/M} .
$$ (eq-kravtsov-sat)

Its key, `kravtsov04`, is historical:
{cite:t}`KravtsovBerlindWechsler_2004ApJ...609...35K` roll the power law off as
$(M/M_1 - C)^{\beta}$, with no exponential.
{cite:t}`LangeWellsHearin_2025arXiv251215962L`, `lange25`, multiply the central
of Eq. {eq}`eq-zheng-cen` by a completeness $f_\Gamma$, the fraction of centrals
above threshold that enter the catalogue, and leave it off the satellites, which
are those of Eq. {eq}`eq-zheng-sat` without the $\langle N_{\rm c}\rangle$
factor.
This package uses the satellites of Eq. {eq}`eq-kravtsov-sat` instead, so at
$f_\Gamma = 1$ the model has the `kravtsov04` form.

```python
p = O.occupation_defaults("zheng07")
print(O.n_cen_zheng07(lm, p["log10mmin"], p["sigma_logm"]))
print(O.n_sat_zheng07(lm, **p))
print(O.n_sat_kravtsov04(lm, **p))                 # the same parameters, the smooth cut-off
```

### Incompleteness

{cite:t}`MoreMiyatakeMandelbaum_2015ApJ...806....2M`, `more15`, add an
incompleteness ramp, which multiplies the centrals of Eq. {eq}`eq-zheng-cen`,
and tie the satellite threshold to the central cut-off mass rather than letting
it float:

$$
\begin{aligned}
f_{\rm inc}(M) &= \min\left\{1, \max\left[0,\
   1 + \alpha_{\rm inc}(\log M - \log M_{\rm inc})\right]\right\}, \\
\langle N_{\rm s}\rangle &= f_{\rm inc}\langle N_{\rm c}\rangle
   \left(\frac{M - \kappa M_{\min}}{M_1}\right)^{\alpha},
   \quad M > \kappa M_{\min},
\end{aligned}
$$ (eq-more-finc)

and zero below $\kappa M_{\min}$, with the same `where` as
Eq. {eq}`eq-zheng-sat`.
The clip here saturates a physical fraction, and the model is flat on both sides
of it, so the zero derivative there is correct.
A variant, `more15_const`, replaces the ramp with a constant $f_{\rm inc}$.

### Completeness and quenching

{cite:t}`GuoYangLu_2018ApJ...858...30G`, `guo18`, work through a
broken-power-law stellar-mass relation and a completeness function:

$$
\begin{aligned}
\log_{10}\langle M_\star\rangle &= \log_{10}M_\star^0 + (\alpha+\beta)x
   - \beta\log_{10}\left(1 + 10^{x}\right), \\
c(M_\star) &= \frac{f}{2}\left[1 + {\rm erf}
   \frac{\log M_\star - \log M_\star^{\min}}{\sigma_{\rm c}}\right],
\end{aligned}
$$ (eq-guo-shmr)

with $x = \log_{10}(M/M_1)$.
The centrals are
$c(\langle M_\star\rangle)\,\tfrac12{\rm erfc}[(\log M_\star^{\min} -
\log\langle M_\star\rangle)/w_{\log M_\star}]$ and the satellites
$c(\langle M_\star\rangle)\,(M/M_1^{\rm sat})^{\alpha_{\rm sat}}$, each with its
own $f$, $M_\star^{\min}$ and $\sigma_{\rm c}$.
The completeness is evaluated at the *mean* relation rather than integrated over
its scatter, which is this package's analytic approximation and keeps the
occupation in closed form.
{cite:t}`GuoYangLu_2018ApJ...858...30G` integrate the log-normal times the
completeness over $M_\star$ (their Eqs. 13–14), and the two agree only where the
completeness varies slowly across the scatter.
{cite:t}`GuoYangRaichoor_2019ApJ...871..147G`, `guo19`, multiply both components
by a star-forming fraction,

$$
f_{\rm sf} = 1 - f_{\rm q}, \qquad f_{\rm q} = \frac{1}{1 + M_{\rm q}/M},
$$ (eq-guo-quench)

which suppresses the occupation in massive haloes and makes `guo19` the model
for emission-line galaxies.
Their printed Eq. 10 has $M/M_{\rm q}$ in place of $M_{\rm q}/M$, which would
make $f_{\rm q}$ fall with mass; their Fig. 5 has it rise, as here.

### A relation written backwards

{cite:t}`LeauthaudTinkerBundy_2012ApJ...744..159L`, `leauthaud12`, is the same
shape as the default's Eq. {eq}`eq-zm-cen`, written in $\log_{10}M_\star$ rather
than $\ln M_\star$, so it carries the $\sqrt2$ and no $\ln 10$:

$$
\langle N_{\rm c}\rangle = \tfrac12\,{\rm erfc}\left[
  \frac{\log M_\star^{\rm t} - \log M_\star^{\rm c}(M_h)}
       {\sqrt2\,\sigma_{\log M_\star}}\right].
$$ (eq-leau-cen)

Its mean relation is written backwards too, and reaches the halo-mass grid
through Eq. {eq}`eq-invert`:

$$
\log_{10}M_h = \log_{10}M_1 + \beta x
  + \frac{t^{\delta}}{1 + t^{-\gamma}} - \tfrac12 ,
$$ (eq-leau-shmr)

with $x$ and $t$ as in Eq. {eq}`eq-zm-shmr`.
It is that relation without the $1/\ln 10$ on the non-linear term, a factor
$2.3$ in the high-mass turnover, so $\beta$, $\delta$ and $\gamma$ fitted under
one are not the parameters of the other; the two are separate functions rather
than one with a flag.
Its satellites have the form of Eq. {eq}`eq-zm-sat`, with $M_{\rm sat}$ and
$M_{\rm cut}$ free rather than scaled from $M_{\min}$.

### Ungated satellites and stellar-mass bins

{cite:t}`ZacharegkasChangPrat_2025arXiv250622367Z`, `zacharegkas25`, build on
the {cite:t}`KravtsovVikhlininMeshcheryakov_2018AstL...44....8K` relation,
Eq. {eq}`eq-krav-shmr`:

$$
\begin{aligned}
\langle N_{\rm c}\rangle &= \frac{f_{\rm c}}{2}\left[1 + {\rm erf}
   \frac{\log M_\star(M_h) - \log M_\star^{\rm t}}{w_{\log M_\star}}\right], \\
\langle N_{\rm s}\rangle &= f_{\rm s}\left(\frac{M_h - \kappa M_{\min}}{M_{\rm sat}}
   \right)^{\alpha_{\rm sat}}e^{-M_{\rm cut}/M_h} .
\end{aligned}
$$ (eq-zach-cen)

$M_{\min}$ is the halo mass at which Eq. {eq}`eq-krav-shmr` reaches
$M_\star^{\rm t}$, found with Eq. {eq}`eq-invert`, and $M_{\rm sat}$ and
$M_{\rm cut}$ scale with it as in Eq. {eq}`eq-zm-sat`.
Unlike Eqs. {eq}`eq-zheng-sat`, {eq}`eq-more-finc` and {eq}`eq-zm-sat`, this
satellite term is not gated by the central occupation, so it takes neither
$w_{\log M_\star}$ nor $f_{\rm c}$.
They are left out of the signature rather than accepted and ignored, so that a
fit cannot report a constraint on a parameter the model never uses.

{cite:t}`vanUitertCacciatoHoekstra_2016MNRAS.459.3251V`, `vanuitert16`, count
galaxies in a stellar-mass *bin* rather than above a threshold, so the centrals
are a difference of two cumulative distributions, clipped into $[0,1]$ because a
central is either in the bin or not:

$$
\begin{aligned}
\mu &= \log_{10}M_\star^0 + \beta_1 x
   - (\beta_1-\beta_2)\log_{10}\left(1 + 10^{x}\right), \\
\langle N_{\rm c}\rangle &= {\rm clip}\left\{\tfrac12\left[{\rm erf}
   \frac{\log M_\star^{\rm hi} - \mu}{\sqrt2\sigma_{\rm c}} - {\rm erf}
   \frac{\log M_\star^{\rm lo} - \mu}{\sqrt2\sigma_{\rm c}}\right],0,1\right\},
\end{aligned}
$$ (eq-vu-cen)

with $x = \log_{10}(M_h/M_{h,1})$ and $\beta_2$ sampled as $\log_{10}\beta_2$ so
it cannot go negative and turn the high-mass end over.
Its satellites integrate a modified Schechter conditional stellar-mass function
over the bin:

$$
\begin{aligned}
\Phi_{\rm s}(M_\star|M_h)\,{\rm d}M_\star &= \frac{\phi_{\rm s}}{M_\star^{\rm s}}
   \left(\frac{M_\star}{M_\star^{\rm s}}\right)^{\alpha_{\rm s}}
   e^{-(M_\star/M_\star^{\rm s})^{2}}{\rm d}M_\star, \\
\langle N_{\rm s}\rangle &= \phi_{\rm s}\ln 10 \int u^{\alpha_{\rm s}+1}e^{-u^{2}}
   \,{\rm d}\log_{10}M_\star,
\end{aligned}
$$ (eq-vu-sat)

with $u = M_\star/M_\star^{\rm s}$, $M_\star^{\rm s} = 0.56\,M_\star^{\rm c}$ and
$\log_{10}\phi_{\rm s} = b_0 + b_1(\log_{10}M_h - 13)$.
The measure is ${\rm d}M_\star$ rather than ${\rm d}M_\star/M_\star$, and the
$\alpha_{\rm s}+1$ is the Jacobian of the substitution.
Reading the exponent off a ${\rm d}M_\star/M_\star$ statement of the same
equation drops it and leaves the satellite occupation wrong by a factor of order
$u$, which varies across the bin and so cannot be absorbed into $\phi_{\rm s}$.
The integral has no closed form for general $\alpha_{\rm s}$ and is a fixed
128-node trapezoid: the node count sets an array shape, and a shape that
depended on the parameters would not compile.

## Width conventions of the central occupation

Eqs. {eq}`eq-zheng-cen`, {eq}`eq-zm-cen` and {eq}`eq-leau-cen` differ by a
$\sqrt2$ and a $\ln 10$ because their widths follow three conventions.
{cite:t}`ZhengCoilZehavi_2007ApJ...667..760Z` define $\sigma_{\log M}$ as the
width parameter of the error function itself, so there is no $\sqrt2$.
{cite:t}`ZuMandelbaum_2015MNRAS.454.1161Z` has a log-normal scatter in
$\ln M_\star$, so the $\sqrt2$ is that of a Gaussian cumulative distribution
written with `erfc`, and the $\ln 10$ converts the threshold.
{cite:t}`LeauthaudTinkerBundy_2012ApJ...744..159L` is the same log-normal in
$\log_{10}M_\star$, hence $\sqrt2$ and no $\ln 10$.
At a $1\sigma$ offset, an `erf` width and a Gaussian $\sigma$ differ by a factor
two in the central occupation: $\tfrac12{\rm erfc}(1) = 0.0786$ against
$\tfrac12{\rm erfc}(1/\sqrt2) = 0.1587$.
Eq. {eq}`eq-zach-cen`, after {cite:t}`ZacharegkasChangPrat_2025arXiv250622367Z`,
adds a fourth convention: the same log-normal in $\log_{10}M_\star$ with the
$\sqrt2$ absorbed into the symbol, so its width is $\sqrt2$ times Leauthaud's
$\sigma$; the two models are numerically identical, and their fitted widths
differ by $\sqrt2$.
The $\sigma_{\rm c} = 0.173$ of {cite:t}`GuoYangLu_2018ApJ...858...30G` is a
Gaussian $\sigma$, in Leauthaud's convention; the `guo18` and `guo19` centrals
take a width, so it enters them as $0.245$.

So the parameters are named apart, one name per (variable, convention) pair:

| name | variable | convention | taken by |
|---|---|---|---|
| `sigma_logm` | $\log_{10}M_h$ | `erf` width, no $\sqrt2$ | Eq. {eq}`eq-zheng-cen` and its four relatives: `zheng07`, `kravtsov04`, `lange25`, `more15`, `more15_const` |
| `sigma_lnmstar` | $\ln M_\star$ | Gaussian $\sigma$ | Eq. {eq}`eq-zm-cen`: `zumandelbaum15` and its colour split |
| `sigma_logmstar` | $\log_{10}M_\star$ | Gaussian $\sigma$ | Eq. {eq}`eq-leau-cen`: `leauthaud12` |
| `width_logmstar` | $\log_{10}M_\star$ | `erf` width, $\sqrt2\,\sigma$ | Eq. {eq}`eq-zach-cen`, `guo18` and `guo19` |

The fourth name is `width_` rather than `sigma_logm_star` because that would
differ from `sigma_logmstar` by a single underscore.
The rule covers these four and no other width: the $\sigma_{\rm c}$ of
Eq. {eq}`eq-vu-cen` and the $\sigma_{\rm c}$ in $\log_{10}L$ of
Eq. {eq}`eq-clf-cen` keep their papers' names, and the completeness widths
`sigma_c_cen` and `sigma_c_sat` of Eq. {eq}`eq-guo-shmr` keep names close to
theirs, $\sigma_{\rm I}$ and $\sigma_{\rm II}$ in
{cite:t}`GuoYangLu_2018ApJ...858...30G` and a single $\sigma_{\rm c}$ in
{cite:t}`GuoYangRaichoor_2019ApJ...871..147G`.
A fit that swaps two of the four is refused: a model function, or its parameter
container, raises a `TypeError` on a name it does not take, and `GalaxySector`
raises a `ValueError` on the missing one, although it drops an extra name
silently.

```python
try:
    O.n_cen_zheng07(lm, log10mmin=11.35, sigma_lnmstar=0.25)
except TypeError as err:
    print(err)

swapped = galaxy_defaults("zheng07")
swapped["sigma_lnmstar"] = swapped.pop("sigma_logm")
try:
    GalaxySector("zheng07").occupation(field, swapped)
except ValueError as err:
    print(err)
```

## The registry at its defaults

The {doc}`notebook <notebooks/layer3_galaxies>` draws all thirteen occupations at
their defaults, $\langle N_{\rm c}\rangle$ and $\langle N_{\rm s}\rangle$ side by
side, with the two conditional luminosity functions of the next section in a
third panel.
The defaults of `zumandelbaum15` and its colour split are the LS10 fit of the
technical paper; those of the other ten are not a complete published fit.
`zheng07` is the $M_r < -18$ fit of {cite:t}`ZhengCoilZehavi_2007ApJ...667..760Z`
with $\alpha = 1$ for their $0.83$; `leauthaud12` and `zacharegkas25` carry their
papers' stellar-mass relations with illustrative occupation parameters;
`vanuitert16` takes the means of its paper's priors for $\beta_1$,
$\alpha_{\rm s}$, $b_0$ and $b_1$ and illustrative values otherwise; and the
remaining six are illustrative throughout: `kravtsov04`, for which no parameter
set of this form is published, `lange25`, published only as posteriors, `more15`,
`more15_const`, `guo18` and `guo19`.
Each model was fitted to a different survey with a different selection, which
the registry records, so the spread of the curves reflects their samples rather
than a disagreement.
The centrals of the binned model saturate below one by construction,
Eq. {eq}`eq-vu-cen`.

An occupation calibrated on a threshold sample and evaluated as a binned one
gives a different number, so `OCC_CALIBRATION` carries the survey, the selection
and the redshift range for every key; a test asserts that its keys and the
registry's are the same set:

```{include} _generated/occupation_calibration.md
```

## Conditional luminosity functions

A conditional luminosity function gives the number of galaxies at each
luminosity, and its integral above a luminosity threshold is an occupation.
The central luminosity is a broken power law
{cite:p}`CacciatovandenBoschMore_2009MNRAS.394..929C`,

$$
\log_{10}L_{\rm c} = \log_{10}L_0 + \alpha x
   + (\beta - \alpha)\log_{10}\left(1 + 10^{x}\right),
$$ (eq-clf-lc)

with $x = \log_{10}(M/M_1)$, and the centrals are log-normal about it:

$$
\langle N_{\rm c}\rangle(>L) = \tfrac12\,{\rm erfc}\frac{\log L_{\rm lim} - \log L_{\rm c}}
                                     {\sqrt2\,\sigma_{\rm c}}, \qquad
\phi^\star_{\rm s} = \frac{A_\phi}{\sqrt{2\pi}\,\sigma_{\rm c}}\frac{M}{M_1} ,
$$ (eq-clf-cen)

with $A_\phi$ the amplitude `phi_s_amp`, a simplification of
{cite:author}`CacciatovandenBoschMore_2009MNRAS.394..929C`'s normalisation
(below).
The satellite integral has a closed form:

$$
\langle N_{\rm s}\rangle(>L) = \frac{\phi^\star_{\rm s}(M)}{2}\,
   \Gamma\left(\frac{\alpha_{\rm s}+1}{2},\ (L/L_{\rm c})^{2}\right),
$$ (eq-clf-sat)

with $\Gamma$ the unregularised upper incomplete gamma function.
It needs no quadrature, which over $L^{\alpha_{\rm s}}$ would lose precision near
zero for a negative $\alpha_{\rm s}$, and it is differentiable in $\alpha_{\rm s}$.

Every published fit has $\alpha_{\rm s} < -1$, so the first argument
$(\alpha_{\rm s}+1)/2$ is negative, and the available regularised incomplete
gamma is defined only for a positive first argument.
The recurrence

$$
\Gamma(a, x) = \frac{\Gamma(a+1, x) - x^{a}e^{-x}}{a}
$$ (eq-upper-gamma)

moves the argument into the supported range and is selected with a `where`
({py:func}`~ggah_mod.sectors.clf.upper_gamma`).
A `where` evaluates both branches, and a `nan` in the unused one still poisons
the gradient of the used one, so each branch is guarded independently at $a$, at
$a+1$, at $x$ and at the denominator.
Both default parameter sets go through the recurrence branch rather than the
direct one.

The default, `cacciato09`, simplifies
{cite:t}`CacciatovandenBoschMore_2009MNRAS.394..929C` in three ways:
*(i)* the satellite normalisation of Eq. {eq}`eq-clf-cen` is linear in $M$ where
theirs is a quadratic in $\log_{10}(M/10^{12}\,h^{-1}{\rm M}_\odot)$ (their
Eq. 40), *(ii)* the Schechter cut-off sits at $L_{\rm c}$ rather than at their
$L^\star_{\rm s} = 0.562\,L_{\rm c}$ (Eq. 38), and *(iii)* the faint-end slope is
a constant rather than their

$$
\alpha_{\rm s}(M) = -2 + a_1\left[1 - \frac{2}{\pi}
   \arctan\left(a_2(\log_{10}M - \log M_2)\right)\right]
$$ (eq-clf-alphasat)

(Eq. 39), which {py:func}`~ggah_mod.sectors.clf.alpha_faint_cacciato09`
defines and no model calls, so that its absence from the defaults is visible.
The default's parameters are illustrative, not those of their Table 3:
$\log M_1 = 11.0$, $\alpha = 2.95$ and $\beta = 0.18$ in Eq. {eq}`eq-clf-lc`
against their $11.07$, $3.273$ and $0.255$.
`vandenbosch13` is the published form:
{cite:t}`vandenBoschMoreCacciato_2013MNRAS.430..725V` keep the quadratic,
$\log_{10}\phi^\star_{\rm s} = b_0 + b_1 y + b_2 y^2$ with
$y = \log_{10}(M/10^{12}\,h^{-1}{\rm M}_\odot)$, and the $0.562\,L_{\rm c}$, and
hold $\alpha_{\rm s}$ constant (their Eqs. 77–79).
Its parameters are the conditional luminosity function they populate their mocks
with (their Sect. 4.1), not a fit.

```{include} _generated/clf_calibration.md
```

```python
cen, sat = C.make_clf("vandenbosch13")
q = C.clf_defaults("vandenbosch13")
q_sat = {k: v for k, v in q.items() if k != "sigma_c"}   # the satellites do not take it
print(sat(lm, **q_sat))                             # the closed form: no quadrature
d_sat = jax.grad(lambda a: sat(13.0, **(q_sat | {"alpha_faint": a})))
print(d_sat(q["alpha_faint"]))                      # through the recurrence branch
```

## The relations that need no inversion

Eq. {eq}`eq-zm-shmr` is written backwards and reaches the halo-mass grid through
Eq. {eq}`eq-invert`, and so is
{cite:author}`LeauthaudTinkerBundy_2012ApJ...744..159L`'s, Eq. {eq}`eq-leau-shmr`.
The other five of the registry's seven relations are written forwards and reach
the grid directly.
They are four functional forms rather than five, since
{cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B` is
{cite:t}`KravtsovVikhlininMeshcheryakov_2018AstL...44....8K`'s with redshift
terms on top.
Every relation carries a calibration row, and a test checks the relation and
calibration dictionaries against each other key for key.
Each relation is evaluated in its paper's mass units, which `SHMR_MASS_UNITS`
records, and `GalaxySector` converts the halo mass in and the stellar mass out at
its boundary ({py:func}`~ggah_mod.sectors.sham.halo_mass_to_relation` and
{py:func}`~ggah_mod.sectors.sham.stellar_mass_to_msun`):

```{include} _generated/shmr_calibration.md
```

### Kravtsov et al., and Behroozi et al. (2013)

The first of the four forms is

$$
\begin{aligned}
f(x) &= -\log_{10}\left(10^{\alpha x}+1\right)
   + \delta\,\frac{\left[\log_{10}(1+e^{x})\right]^{\gamma}}{1+e^{10^{-x}}}, \\
\log_{10}M_\star &= \log_{10}\epsilon + \log_{10}M_1 + f(x) - f(0),
\end{aligned}
$$ (eq-krav-shmr)

with $x = \log_{10}(M_h/M_1)$.
Both logarithms go through a `logaddexp`, so that neither overflows, and the
inner exponent is capped far inside the region where the term it guards has
saturated.
The denominator of $f(0)$ is $1 + e^{10^{0}} = 1 + e$, not $2$.
Using $2$ lowers the whole relation by
$\delta(\log_{10}2)^{\gamma}\left[\tfrac12 - 1/(1+e)\right]$, an offset uniform in
mass but not in redshift: $0.43$ dex at the `zacharegkas25` defaults, and
$0.555$, $0.563$, $0.473$ and $0.235$ dex for
{cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B` at $z = 0$, $1$, $2$
and $4$.
So the code evaluates $f$ at zero rather than writing the special case out.
{cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B` adds redshift terms scaled
by $\nu(a) = e^{-4a^{2}}$, which rises to one at high redshift and shuts them off
towards low redshift, to $e^{-4}$ at $z = 0$, so that the well-constrained
low-redshift parameters stand on their own.

### Two double power laws

The next two forward relations are double power laws:

$$
\frac{M_\star}{M_h} = \frac{2N(z)}
   {\left(M_h/M_1\right)^{-\beta} + \left(M_h/M_1\right)^{\gamma}},
$$ (eq-moster-girelli)

with every coefficient linear in $z/(1+z)$ for
{cite:t}`MosterNaabWhite_2013MNRAS.428.3121M`, and $\log_{10}M_1 = B + z\mu$,
$N = C(1+z)^{\nu}$, $\gamma = D(1+z)^{\eta}$, $\beta = Fz + E$ for
{cite:t}`GirelliPozzettiBolzonella_2020A&A...634A.135G`.
Swapping the two exponents leaves a relation that still rises monotonically but
puts the steeper index, $\beta$, on the high-mass side.
On {cite:author}`GirelliPozzettiBolzonella_2020A&A...634A.135G`'s $z = 0$ relation
at $10^{15}\,h^{-1}{\rm M}_\odot$ ($1.49\times10^{15}\,h_{67}^{-1}{\rm M}_\odot$ in
their units), the swap gives $M_\star/M_h = 5.2\times10^{-5}$ where the relation
gives $3.7\times10^{-4}$, a stellar fraction seven times too small at the cluster
scale, which would enter the stellar part of the matter field's per-halo mass
budget.

### UniverseMachine

The fourth form is the median stellar mass–halo mass fit of UniverseMachine
(Appendix J of {cite:t}`BehrooziWechslerHearin_2019MNRAS.488.3143B`), the one
empirical backbone in the set, written as its authors wrote it:

$$
\begin{aligned}
\log_{10}\frac{M_\star}{M_1} &= \epsilon
  - \log_{10}\left(10^{-\alpha x} + 10^{-\beta x}\right)
  + \gamma\,e^{-\frac12 (x/\delta)^{2}},\\
x &= \log_{10}\frac{M_{\rm peak}}{M_1},
\end{aligned}
$$ (eq-universemachine)

with $\log_{10}M_1$, $\epsilon$ and $\alpha$ evolving as
$p_0 + p_a(a-1) - p_{\ln a}\ln a + p_z z$, $\beta$ without the $\ln a$ term,
$\delta$ constant, and $\log_{10}\gamma$ as $\gamma_0 + \gamma_a(a-1) + \gamma_z z$.
It differs from {cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B`, which
shares its three authors and a purpose: that one is a single power law with a
$\delta/\gamma$ term and an $e^{-4a^{2}}$ damping, and this one a double power
law with a Gaussian term and no damping.
Both are kept, since an abundance-matching fit and a merger-tree backbone are
different models of the same relation.
It is the one relation that asks for $h$, with no default, because its fit is in
physical solar masses and the package's halo masses are not.

```python
h = PLANCK18.h
for name in ("moster13", "behroozi13", "girelli20", "universemachine"):
    units = SH.SHMR_MASS_UNITS[name]
    kw = {"h": h} if name == "universemachine" else {}
    ms = SH.make_shmr(name)(SH.halo_mass_to_relation(lm, units, h), z=0.0, **kw)
    print(f"{name:16s}", np.round(SH.stellar_mass_to_msun(ms, units, h), 3))  # log10 M_* [Msun]
```

### The gradient the inversion keeps

The {doc}`notebook <notebooks/layer3_galaxies>` draws the seven stellar-mass
relations at $z = 0$, each converted from its paper's units to
$h^{-1}{\rm M}_\odot$, the two written backwards through Eq. {eq}`eq-invert`,
and the derivative of one of them with respect to a shape parameter, three ways.
A bare bisection returns a derivative of exactly zero at every mass, since it
depends on its inputs only through the sign of a residual, while its value is
correct to machine precision.
Bracketing under a `stop_gradient` and taking one Newton step (Eq. {eq}`eq-invert`)
leaves the value unchanged and recovers the implicit-function derivative
${\rm d}x/{\rm d}\theta = -(\partial_\theta F)/(\partial_x F)$, with $F = f - y$,
which agrees with a central difference.
A forecast that varies this parameter through the bisection inherits a column of
zeros in its information matrix and reads it as infinite confidence.

```python
zm = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)  # their fit
mstar = lambda beta: SH.mstar_from_mh_zu15(13.0, **(zm | {"beta": beta}))
print(jax.grad(mstar)(0.33), (mstar(0.33 + 1e-5) - mstar(0.33 - 1e-5)) / 2e-5)
```

## Galaxies as a tracer

The occupation $(\langle N_{\rm c}\rangle, \langle N_{\rm s}\rangle)$ becomes a tracer through {py:meth}`~ggah_mod.sectors.galaxies.GalaxySector.weights`, which layer 4 reads.
Every sector returns its weight per halo in two parts, at the centre and extended,

$$
W(k|M) = \frac{w_{\rm point} + w_{\rm extended}}{n},
$$ (eq-wtotal)

because the one-halo term of a *discrete* tracer must not pair a central galaxy with itself: it carries $\langle N(N-1)\rangle$ rather than $\langle N\rangle^2$ ({doc}`layer4_spectra`).
The weights exclude ${\rm d}n/{\rm d}M$, which enters once, in layer 4.
The galaxy sector reads its occupation three ways (or views):

$$
w_{\rm point} = \langle N_{\rm c}\rangle, \qquad
w_{\rm extended} = \langle N_{\rm s}\rangle\,u_{\rm sat}(k|M), \qquad
n = \bar n,
$$ (eq-gal-weights)

with $w_{\rm extended}$ left out in the `cen` view and $w_{\rm point}$ in the `sat` view.
The tracer is discrete in all three, so its shot noise is $1/\bar n$.
Each view is normalised by its own $\bar n$, so a satellite-only spectrum is per satellite rather than per galaxy, and the `cen` and `sat` tracers do not add up to `total`.
{py:meth}`~ggah_mod.sectors.galaxies.GalaxySector.effective_bias` returns $\int{\rm d}M\,({\rm d}n/{\rm d}M)\,b(M)\langle N\rangle/\bar n$ in the same three views.

```python
gal = GalaxySector("zumandelbaum15", calibration="off")
gp = galaxy_defaults("zumandelbaum15")
for view in ("total", "cen", "sat"):
    w = gal.weights(field, gp, view=view)                 # TracerWeights
    b = gal.effective_bias(field, gp, view=view)
    print(f"{view:5s} n = {float(w.norm):.3e} h^3 Mpc^-3, b = {float(b):.3f}")
```

Assembly bias is not modelled: the keys `a_cen` and `a_sat` of an earlier decoration raise, since its kernel had no published source.

### The satellite profile

The satellites follow the halo's NFW profile {cite:p}`NavarroFrenkWhite_1997ApJ...490..493N` by default, with three freedoms of their own, separate from the occupation:

$$
\rho_{\rm sat}(r) \propto \frac{x^{-\gamma_{\rm in}}}{x(1+x)^2}\,e^{-f\,r/R_\Delta},
\qquad x = \frac{b_{\rm c}\,c\,r}{R_\Delta},
$$ (eq-sat-profile)

with $c$ the halo concentration, $b_{\rm c}$ (`b_sat_conc`) its multiplier, $f$ (`f_cut_inv`) an outer truncation and $\gamma_{\rm in}$ (`gamma_inner`) an extra inner slope.
Its transform {py:func}`~ggah_mod.halos.profiles.satellite_uk` is normalised to $u(0) = 1$.
The three parameters are keys of the galaxy parameters, and their defaults give the NFW profile exactly:

```{include} _generated/params_satellite_profile.md
```

```python
k = np.asarray(field.k)
i, j = np.searchsorted(k, 5.0), np.searchsorted(np.asarray(field.m), 1e14)
for b_c in (1.0, 2.0):
    u = gal.satellite_uk(field, gp | {"b_sat_conc": b_c})   # (Nk, NM)
    print(f"b_sat_conc = {b_c}: u(k = 5 h/Mpc | 1e14) = {float(u[i, j]):.3f}")
```

### Mis-centring

The central galaxy does not always sit at the halo's centre.
The correction is a *modifier* rather than a sector: it moves one object without changing how many a halo holds, so it multiplies $w_{\rm point}$ and leaves $\bar n$ alone.
A fraction $p_{\rm off}$ of central galaxies is displaced by a Gaussian of width $R_{\rm off}$, and the rest sit at the centre:

$$
h(k|M) = (1 - p_{\rm off}) + p_{\rm off}\,e^{-k^2R_{\rm off}^2/2},
\qquad
R_{\rm off} = \begin{cases}
  r_{\rm off}\,r_{\rm s}(M) & \texttt{more15},\\
  \sigma_{\rm off} & \texttt{johnston07},
\end{cases}
$$ (eq-miscen)

with $r_{\rm s}$ the scale radius of the halo.
`more15` {cite:p}`MoreMiyatakeMandelbaum_2015ApJ...806....2M` is the default convention and `johnston07` {cite:p}`JohnstonSheldonWechsler_2007arXiv0709.1159J` the alternative.
$p_{\rm off}$ defaults to zero, so mis-centring has to be asked for; it only suppresses small-scale power (degenerate with concentration).
`GalaxySector.weights` applies the `more15` form whenever the galaxy parameters carry a non-zero `p_off`:

```{include} _generated/params_miscentering.md
```

```python
from ggah_mod.sectors import MisCentering, MisCenteringParams

h_k = MisCentering("more15").h_k(field, MisCenteringParams(p_off=0.3))   # (Nk, NM)
print(float(h_k[0, j]), float(h_k[i, j]))            # 1 at k -> 0, suppressed at k = 5
w_off = gal.weights(field, gp | {"p_off": 0.3, "r_off": 0.25}, view="cen")
print(w_off.w_point.shape)                           # now (Nk, NM): w_point times h(k|M)
```

## Stellar masses

A sector whose occupation is a stellar-mass threshold (`zumandelbaum15` and its colour split, `leauthaud12`, `zacharegkas25`) carries its stellar-mass relation, and exposes it on the halo-mass grid.
{py:meth}`~ggah_mod.sectors.galaxies.GalaxySector.log10_mstar_msun` gives the mean central stellar mass in ${\rm M}_\odot$, which the AGN sector reads, and {py:meth}`~ggah_mod.sectors.galaxies.GalaxySector.log10_mstar_scatter` its scatter in dex.
Any other occupation needs a relation from the registry passed as `shmr=`, and raises without one.

The conditional stellar-mass functions are the derivatives of the threshold occupations with respect to the threshold,

$$
\frac{{\rm d}N_x}{{\rm d}\log_{10}M_\star}
  = -\frac{\partial\langle N_x\rangle(>M_\star^{\rm t})}{\partial\log_{10}M_\star^{\rm t}},
\qquad x \in \{{\rm c}, {\rm s}\},
$$ (eq-csmf)

taken by forward-mode differentiation on a fixed grid of stellar masses, and their integral over the mass function is the stellar-mass function,
$\Phi_x(M_\star) = \int{\rm d}M\,({\rm d}n/{\rm d}M)\,{\rm d}N_x/{\rm d}\log_{10}M_\star$, in $h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}$.
{py:meth}`~ggah_mod.sectors.galaxies.GalaxySector.stellar_fraction` returns the central and satellite stellar mass per halo mass, $f_\star^{\rm c} = \langle N_{\rm c}\rangle M_\star/M$ and the integral of Eq. {eq}`eq-csmf` weighted by $M_\star$, which the matter field of {doc}`layer3_hot_gas` takes as its stellar parts.

```python
print(gal.log10_mstar_msun(lm, gp, h=PLANCK18.h))      # log10 M_* [Msun] at lm
lg_ms, phi_cen, phi_sat = gal.stellar_mass_function(field, gp, msun=True)
f_cen, f_sat = gal.stellar_fraction(field, gp, satellites=True)
print(f"peak M_*/M: centrals {float(f_cen.max()):.4f}, satellites {float(f_sat.max()):.4f}")
```

## Satellite kinematics

The velocity dispersion of satellite galaxies around central galaxies of a given stellar mass is a second probe of the halo mass {cite:p}`MorevandenBoschCacciato_2011MNRAS.410..210M`.
The dispersion of a halo follows from the isotropic Jeans equation for an NFW profile ({cite:t}`LokasMamon_2001MNRAS.321..155L`, their Eq. 14),

$$
\sigma_v^2(M) = b_v^2\,K(c)\,V_\Delta^2, \qquad
K(c) = \frac{c\,J(c)}{m(c)^2},
$$ (eq-sigma-v)

with $m(y) = \ln(1+y) - y/(1+y)$, $J(x) = \int_0^x u^2 I(u)\,{\rm d}u$, $I(x) = \int_x^\infty m(y)\,y^{-3}(1+y)^{-2}\,{\rm d}y$, $V_\Delta^2 = GM/R_\Delta$ and $b_v$ (`b_v`) a velocity bias.
{py:func}`~ggah_mod.sectors.kinematics.satellite_velocity_dispersion` averages it over the haloes that host a central galaxy in each stellar-mass bin, weighted by their satellites (`weighting="satellite"`, as a stack of satellite velocities measures it) or by host (`"host"`):

$$
\langle\sigma^2\rangle = \frac{\int{\rm d}M\,({\rm d}n/{\rm d}M)\,P(M|M_\star^{\rm c})\,\langle N_{\rm s}\rangle\,\sigma_v^2}
                             {\int{\rm d}M\,({\rm d}n/{\rm d}M)\,P(M|M_\star^{\rm c})\,\langle N_{\rm s}\rangle},
$$ (eq-sigma-sat)

with $P(M|M_\star^{\rm c})$ from the central conditional stellar-mass function of Eq. {eq}`eq-csmf` and $\langle N_{\rm s}\rangle$ the satellites above `lg_mstar_sat_min`.
It is an observable, not a tracer: it returns one dispersion per bin, in ${\rm km\,s^{-1}}$.
The default $b_v = 1$ sits inside the subhalo velocity bias of $0.87$–$1.07$ that {cite:t}`WuHahnEvrard_2013MNRAS.436..460W` find in simulated clusters:

```{include} _generated/params_kinematics.md
```

```python
from ggah_mod.sectors import satellite_velocity_dispersion

edges = np.array([10.5, 11.0, 11.5])                   # log10 M_*^c [Msun]
print(satellite_velocity_dispersion(field, gal, gp, edges, lg_mstar_sat_min=9.5))
```

## Richness-selected clusters

{py:class}`~ggah_mod.sectors.clusters.ClusterSector` turns the red satellites of a threshold sector into a cluster richness, after {cite:t}`RykoffRozoBusha_2014ApJ...785..104R`:

$$
\bar\lambda(M) = 1 + A_\lambda\,\langle N_{\rm s}^{\rm red}\rangle(>M_\star^\lambda),
\qquad
s^2 = \sigma_{\ln\lambda}^2 + \frac{\bar\lambda - 1}{\bar\lambda^2},
$$ (eq-richness)

with $M_\star^\lambda$ the stellar mass of a member (`lg_mstar_member_min`, $10^{10.2}\,{\rm M}_\odot$ by default), $\sigma_{\ln\lambda}$ an intrinsic scatter and the second term the Poisson scatter in its log-normal limit ({cite:t}`CostanziRozoRykoff_2019MNRAS.482..490C`, their Eq. 12).
A richness bin $[\lambda_1, \lambda_2)$ selects a halo with probability

$$
P(\lambda_1 \le \lambda < \lambda_2\,|\,M) =
  \Phi\!\left(\frac{\ln\lambda_2 - \ln\bar\lambda}{s}\right)
  - \Phi\!\left(\frac{\ln\lambda_1 - \ln\bar\lambda}{s}\right),
$$ (eq-richness-bin)

with $\Phi$ the standard normal cumulative distribution.
The sector's weights are this probability at the centre, one cluster per halo, so a stacked $\Delta\Sigma$ is its cross-spectrum with the matter field through {py:func}`~ggah_mod.spectra.pk.pk_cross`.
It is built on `zumandelbaum16_red` and refuses a sector without a stellar-mass threshold.
The default scatter is close to the rough estimate of about $25\%$ that {cite:t}`McClintockVargaGruen_2019MNRAS.482.1352M` adopt (their Sect. 3.2.4), and $A_\lambda = 1$ counts every red satellite within $R_\Delta$:

```{include} _generated/params_clusters.md
```

```python
from ggah_mod.sectors import ClusterParams, ClusterSector

red = GalaxySector("zumandelbaum16_red", calibration="off")
clusters = ClusterSector(red)
n_cl = clusters.number_density(field, galaxy_defaults("zumandelbaum16_red"), 20.0, 40.0,
                               ClusterParams())
print(f"20 <= lambda < 40: n = {float(n_cl):.2e} h^3 Mpc^-3")
```

## Intrinsic alignments

A weak-lensing likelihood needs a model of intrinsic alignments.
This package writes the alignment as a tracer, a peer of the others, so it can be crossed with any of them, rather than as a correction to $C_\ell^{\kappa\kappa}$.
To linear order the intrinsic shear traces the tidal field, and so the matter: the non-linear alignment (NLA) model {cite:p}`HirataSeljak_2004PhRvD..70f3526H,BridleKing_2007NJPh....9..444B` multiplies the matter weights of {doc}`layer3_hot_gas` by

$$
b_I(z) = -A_{\rm IA}\,f_{\rm red}\left(\frac{\langle M_h\rangle}{M_{h,{\rm piv}}}\right)^{\beta_M}
   \frac{C_1\rho_{\rm crit}\,\Omega_{\rm m}^{\rm ref}}{D(z)}
   \left(\frac{1+z}{1+z_0}\right)^{\eta_{\rm IA}}
   \left(\frac{L}{L_0}\right)^{\beta_{\rm IA}},
$$ (eq-nla)

with $D(z)$ the growth factor normalised to one today.
By default only red galaxies align, the NLA-M model of {cite:t}`WrightStolznerAsgari_2025A&A...703A.158W` (their Eq. 11): $f_{\rm red}$ is the red fraction of the shape sample and $\langle M_h\rangle$ the mean host halo mass of its red galaxies.
{py:meth}`~ggah_mod.sectors.alignments.IntrinsicAlignmentSector.red_sample` computes both on the halo field, as $\bar n$ of `zumandelbaum16_red` over that of `zumandelbaum15`, and as the mean of $\log_{10}M$ over the red occupation.
`population="all"` gives the plain NLA, $f_{\rm red} = 1$ with no mass term.
$b_I$ is negative for a positive $A_{\rm IA}$, since galaxies align along the stretching axis of the tidal field and lensing shears them across it, so $P_{gI}$ subtracts from galaxy–galaxy lensing.
There is no one-halo alignment term: $P_{II}$ inherits the one-halo term of the matter field.

The defaults $A_{\rm IA} = 5.74$ and $\beta_M = 0.44$ are the early-type fit of {cite:t}`FortunaDvornikHoekstra_2025A&A...694A.322F`.
$\Omega_{\rm m}^{\rm ref}$, $z_0$, $M_{h,{\rm piv}}$ and $L/L_0$ are conventions of the published amplitudes rather than physical parameters.
$C_1\rho_{\rm crit} = 0.0134$ is a constant of the module, the literal of {cite:t}`JoachimiMandelbaumAbdalla_2011A&A...527A..26J` after {cite:t}`BrownTaylorHambly_2002MNRAS.333..501B`; CCL computes $0.0138768$ from $\rho_{\rm c,0}$ and scales by the sampled $\Omega_{\rm m}$, so an $A_{\rm IA}$ moved between the two codes changes by $3.6\%$.

```{include} _generated/params_alignments.md
```

| model of {cite:t}`WrightStolznerAsgari_2025A&A...703A.158W` | settings in this package |
|---|---|
| NLA | `population="all"`, $\eta_{\rm IA} = \beta_{\rm IA} = 0$ |
| NLA-z | `population="all"`, $z_0 = 0.3$, $\eta_{\rm IA} = -B_{\rm IA}/A_{\rm IA}$, to first order |
| NLA-k | not implemented (the density weighting of {cite:t}`BlazekMacCrannTroxel_2019PhRvD.100j3506B`) |
| NLA-M | the default, `population="red"` |

```python
from ggah_mod.cosmology import growth_factor
from ggah_mod.sectors import IaParams, IntrinsicAlignmentSector, alignment_bias

f_red, lg_mh = IntrinsicAlignmentSector(calibration="off").red_sample(field)
d_z = growth_factor(0.2, PLANCK18, make_pk("emu_pk"))
b_i = alignment_bias(0.2, d_z, IaParams(), f_red=f_red, log10_mh=lg_mh)
print(f"f_red = {float(f_red):.3f}, log10 <M_h> = {float(lg_mh):.2f}, b_I = {float(b_i):.4f}")
```

## Parameter containers and priors

The parameters of a sector are a frozen dataclass registered as a JAX pytree ({py:class}`~ggah_mod.sectors.params.SectorParams`), so a fit differentiates through them and `jax.jit` traces them.
Each traced parameter is a {py:class}`~ggah_mod.sectors.params.Param`: a default, inclusive bounds, a prior (flat, Gaussian or log-Gaussian), a unit, and the reason for its bound.
Names that select a model rather than a value are static, so changing one recompiles and a gradient with respect to it cannot be asked for.
`log_prior` sums the one-parameter priors, and a {py:class}`~ggah_mod.sectors.params.MultivariateGaussian` passed as `joint=` replaces those of the parameters it covers; bounds belong to the sampler and are not applied.
`OCCUPATION_PARAMS` holds one generated container per occupation and luminosity function, with the bounds of its parameters:

```python
from ggah_mod.sectors import OCCUPATION_PARAMS

zp = OCCUPATION_PARAMS["zheng07"]()                  # Zheng07Params, at its defaults
print(zp.as_dict(), float(zp.log_prior()), zp.within_bounds())
print(zp.replace(alpha=1.1).alpha)
```

## Related pages

- {doc}`notebooks/layer3_galaxies`: the registries above, executed, with
  figures.
- {doc}`notebooks/fit_wp`: the `zheng07` occupation and two cosmological
  parameters fitted to a mock $w_p(r_p)$ alone.
- {doc}`layer2_halos`: the halo field these models populate.
- {doc}`layer3_agn` and {doc}`layer3_hot_gas`: the sectors that read the stellar masses.
- {doc}`layer4_spectra`: how the weights become power spectra.
- {doc}`api/sectors`: every function and its arguments.
