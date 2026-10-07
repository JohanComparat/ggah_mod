# Layer 3: the galaxy registries

Layer 3 populates the halo field of {doc}`layer2_halos` with tracers: galaxies,
active nuclei, hot and cold gas.
The technical paper (Comparat, in prep., Sec. 4.1) documents one galaxy model,
the one the package defaults to: the occupation of
{cite:t}`ZuMandelbaum_2015MNRAS.454.1161Z`, `zumandelbaum15`, with its colour
split by the halo quenching of {cite:t}`ZuMandelbaum_2016MNRAS.457.4360Z`,
`zumandelbaum16_red` and `zumandelbaum16_blue`.
This page documents the rest of the registries that model is one entry of: the
ten other occupations, the two conditional luminosity functions, and the five of
the seven stellar-mass relations that are written forwards and so reach the
halo-mass grid without a numerical inversion.
Every entry carries a row of a calibration table recording what it was fitted
to; the three tables below are generated from the code on every build.

:::{warning}
Layer 3 is **work in progress**: released as tested code ahead of its
verification and its documentation, so its interface and results may change.
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

The {doc}`notebook <notebooks/layer3_galaxies>` draws every model of this page at
its defaults, and the derivative the inversion keeps.

## How a model is reached

An occupation or a conditional luminosity function is selected by its name, as
`GalaxySector(model=...)` or through `make_occupation` and `make_clf`, each of
which returns the pair of functions
$(\langle N_{\rm c}\rangle, \langle N_{\rm s}\rangle)$.
The relations sit in a registry of their own, `sham.SHMR`, reached through
`make_shmr`, which `GalaxySector` calls for its `shmr=` argument and for the
relation a threshold occupation counts its centrals with.
Every entry carries a calibration row recording what it was fitted to and the
redshift range; the occupation and luminosity-function rows also record the
selection, which the relation rows leave empty.
None of them is a fallback for the default: applying one outside its own row is a
systematic error rather than a tolerance, and that is exactly what the
calibration tables exist to say.
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

## The default, in brief

The models below are written against the default, and several are its shape in
another convention, so its equations are restated here; the paper derives them
and records the fit its parameters default to.
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

which leaves the value unchanged to machine precision and makes the derivative
the implicit-function result
${\rm d}x/{\rm d}\theta = -(\partial_\theta F)/(\partial_x F)$.
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
ratio, and the choice is deliberate in both directions: below $M_0$ the bracket
is negative and raising it to a fractional power gives `nan`, while a `clip` at
zero would put a tie exactly at $M_0$ and split the gradient there.
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
The clip here is a genuine saturation of a physical fraction and the model is
flat on both sides of it, so the zero derivative there is the right answer
rather than a lost one, which is the distinction the technical paper's appendix
on table interpolation draws for a clip that was neither.
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

which suppresses the occupation in massive haloes, the direction that makes this
the emission-line-galaxy flavour.
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

### Two more, and a Jacobian

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
Note what is *absent*: unlike Eqs. {eq}`eq-zheng-sat`, {eq}`eq-more-finc`
and {eq}`eq-zm-sat`, this satellite term is not gated by the central occupation,
so it takes neither $w_{\log M_\star}$ nor $f_{\rm c}$.
They are left out of the signature rather than accepted and ignored: a parameter
a function quietly does nothing with is how a fit ends up reporting a constraint
on something it never used.

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
**The measure is ${\rm d}M_\star$, not ${\rm d}M_\star/M_\star$**, and the
$\alpha_{\rm s}+1$ is the Jacobian of the substitution, not a typographical slip.
Reading the exponent off a "${\rm d}M_\star/M_\star$" statement of the same
equation drops it and leaves the satellite occupation wrong by a factor of order
$u$, which varies across the bin, and so cannot be absorbed into $\phi_{\rm s}$.
The integral has no closed form for general $\alpha_{\rm s}$ and is a fixed
128-node trapezoid, fixed rather than adaptive because the node count sets an
array shape and a shape that depended on the parameters would not compile.

## Three sigmas, and a fourth

It is tempting to read Eq. {eq}`eq-zheng-cen` next to Eqs. {eq}`eq-zm-cen`
and {eq}`eq-leau-cen` and conclude that one of them has lost a $\sqrt2$.
They have not, and unifying them would be the bug.
{cite:t}`ZhengCoilZehavi_2007ApJ...667..760Z` *define* $\sigma_{\log M}$ as the
width parameter of the error function itself, so there is no $\sqrt2$ by
definition of the symbol.
{cite:t}`ZuMandelbaum_2015MNRAS.454.1161Z` has a genuine log-normal scatter in
$\ln M_\star$, so the $\sqrt2$ is the one that always accompanies a Gaussian
cumulative distribution written with `erfc`, and the $\ln 10$ converts the
threshold.
{cite:t}`LeauthaudTinkerBundy_2012ApJ...744..159L` is the same log-normal in
$\log_{10}M_\star$, hence $\sqrt2$ and no $\ln 10$.
Three symbols spelled the same way, meaning three things, and a $1\sigma$ offset
separates $\tfrac12{\rm erfc}(1) = 0.0786$ from
$\tfrac12{\rm erfc}(1/\sqrt2) = 0.1587$, a factor two in the central occupation.
And there is a fourth convention: Eq. {eq}`eq-zach-cen`, after
{cite:t}`ZacharegkasChangPrat_2025arXiv250622367Z`, writes the same log-normal in
$\log_{10}M_\star$ with the $\sqrt2$ absorbed into the symbol, so what it calls a
width is $\sqrt2$ times Leauthaud's $\sigma$: numerically identical models, and
not the same number to fit.
The $\sigma_{\rm c} = 0.173$ of {cite:t}`GuoYangLu_2018ApJ...858...30G` is a
Gaussian $\sigma$, in Leauthaud's convention; the `guo18` and `guo19` centrals
take a width all the same, so it enters them as $0.245$.

The parameters are therefore named apart, one name per (variable, convention)
pair and four names for the four:

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
raises a `ValueError` on the missing one, although it drops an extra name without
a word.

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

The {doc}`notebook <notebooks/layer3_galaxies>` draws all thirteen at their
defaults, $\langle N_{\rm c}\rangle$ and $\langle N_{\rm s}\rangle$ side by side,
with the two conditional luminosity functions of the next section in a third
panel: the LS10 fit of the paper for `zumandelbaum15` and its colour split, and
for none of the other ten a complete published fit.
`zheng07` is the $M_r < -18$ fit of {cite:t}`ZhengCoilZehavi_2007ApJ...667..760Z`
with $\alpha = 1$ for their $0.83$; `leauthaud12` and `zacharegkas25` carry their
papers' stellar-mass relations with illustrative occupation parameters;
`vanuitert16` takes the means of its paper's priors for $\beta_1$,
$\alpha_{\rm s}$, $b_0$ and $b_1$ and illustrative values otherwise; and the
remaining six are illustrative throughout: `kravtsov04`, for which no parameter
set of this form is published, `lange25`, published only as posteriors, `more15`,
`more15_const`, `guo18` and `guo19`.
Each model was fitted to a different survey with a different selection, and the
registry records both, so the spread is not a disagreement to reconcile.
The centrals of the binned model saturate below one by construction,
Eq. {eq}`eq-vu-cen`.

What the figure cannot show is the selection each was fitted to: an occupation
calibrated on a threshold sample and evaluated as a binned one is a different
number rather than a tolerance.
`OCC_CALIBRATION` carries the survey, the selection and the redshift range for
every key, and a test asserts that its keys and the registry's are the same set:

```{include} _generated/occupation_calibration.md
```

## Conditional luminosity functions

An occupation answers "how many galaxies above a threshold?".
A conditional luminosity function answers the finer question, how many at each
luminosity, and integrates it.
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
The satellite integral is closed form, which is what makes the family worth
having:

$$
\langle N_{\rm s}\rangle(>L) = \frac{\phi^\star_{\rm s}(M)}{2}\,
   \Gamma\left(\frac{\alpha_{\rm s}+1}{2},\ (L/L_{\rm c})^{2}\right),
$$ (eq-clf-sat)

with $\Gamma$ the unregularised upper incomplete gamma function: no quadrature,
and differentiable in $\alpha_{\rm s}$, which matters because $\alpha_{\rm s}$ is
negative in every published fit and a numerical integral over
$L^{\alpha_{\rm s}}$ near zero is where precision goes.

That negativity is the implementation's one difficulty.
$(\alpha_{\rm s}+1)/2$ is negative for $\alpha_{\rm s} < -1$, which every
published fit has, and the available regularised incomplete gamma is defined only
for a positive first argument.
The recurrence

$$
\Gamma(a, x) = \frac{\Gamma(a+1, x) - x^{a}e^{-x}}{a}
$$ (eq-upper-gamma)

moves the argument into the supported range and is selected with a `where`
({py:func}`~ggah_mod.sectors.clf.upper_gamma`).
Both branches are therefore evaluated, so each is guarded independently at $a$,
at $a+1$, at $x$ and at the denominator: a `nan` in an unused `where` branch
still poisons the gradient of the used one.
Both default parameter sets exercise the recurrence branch in production rather
than the direct one.

The default, `cacciato09`, simplifies
{cite:t}`CacciatovandenBoschMore_2009MNRAS.394..929C` in three ways: the
satellite normalisation of Eq. {eq}`eq-clf-cen` is linear in $M$ where theirs is
a quadratic in $\log_{10}(M/10^{12}\,h^{-1}{\rm M}_\odot)$ (their Eq. 40), the
Schechter cut-off sits at $L_{\rm c}$ rather than at their
$L^\star_{\rm s} = 0.562\,L_{\rm c}$ (Eq. 38), and the faint-end slope is a
constant rather than their

$$
\alpha_{\rm s}(M) = -2 + a_1\left[1 - \frac{2}{\pi}
   \arctan\left(a_2(\log_{10}M - \log M_2)\right)\right]
$$ (eq-clf-alphasat)

(Eq. 39), which is defined here, as
{py:func}`~ggah_mod.sectors.clf.alpha_faint_cacciato09`, and wired into
nothing, named so that its absence from the defaults is visible.
The default's parameters are illustrative and not their Table 3:
$\log M_1 = 11.0$, $\alpha = 2.95$ and $\beta = 0.18$ in Eq. {eq}`eq-clf-lc`
against their $11.07$, $3.273$ and $0.255$.
`vandenbosch13` is the published form:
{cite:t}`vandenBoschMoreCacciato_2013MNRAS.430..725V` keep the quadratic,
$\log_{10}\phi^\star_{\rm s} = b_0 + b_1 y + b_2 y^2$ with
$y = \log_{10}(M/10^{12}\,h^{-1}{\rm M}_\odot)$, and the $0.562\,L_{\rm c}$, and
hold $\alpha_{\rm s}$ constant (their Eqs. 77–79), and its parameters are the
conditional luminosity function they populate their mocks with (their
Sect. 4.1), not a fit.

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
Both logarithms go through a `logaddexp`, which is the difference between this
evaluating and it overflowing, and the inner exponent is capped far inside the
region where the term it guards has saturated.
Note $f(0)$: its denominator is $1 + e^{10^{0}} = 1 + e$, not $2$.
Using $2$ lowers the whole relation by
$\delta(\log_{10}2)^{\gamma}\left[\tfrac12 - 1/(1+e)\right]$, an offset uniform in
mass, which reads as a calibration choice, but not in redshift: $0.43$ dex at the
`zacharegkas25` defaults, and $0.555$, $0.563$, $0.473$ and $0.235$ dex for
{cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B` at $z = 0$, $1$, $2$
and $4$.
Evaluating $f$ at zero rather than writing the special case out is what makes
that impossible.
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
{cite:t}`GirelliBolzonellaCimatti_2020A&A...634A.135G`.
Swapping the two exponents leaves a relation that still rises monotonically but
puts the steeper index, $\beta$, on the high-mass side: on
{cite:author}`GirelliBolzonellaCimatti_2020A&A...634A.135G`'s $z = 0$ relation at
$10^{15}\,h^{-1}{\rm M}_\odot$, which is
$1.49\times10^{15}\,h_{67}^{-1}{\rm M}_\odot$ in their units, it gives
$M_\star/M_h = 5.2\times10^{-5}$ where the relation gives $3.7\times10^{-4}$, a
stellar fraction seven times too small at the cluster scale, which would go
straight into the stellar part of the matter field's per-halo mass budget.

### UniverseMachine

The fourth form is the one empirical backbone in the set, and it is written as
its authors wrote it:

$$
\begin{aligned}
\log_{10}\frac{M_\star}{M_1} &= \epsilon
  - \log_{10}\left(10^{-\alpha x} + 10^{-\beta x}\right)
  + \gamma\,e^{-\frac12 (x/\delta)^{2}},\\
x &= \log_{10}\frac{M_{\rm peak}}{M_1},
\end{aligned}
$$ (eq-universemachine)

UniverseMachine's median stellar mass–halo mass fit (Appendix J of
{cite:t}`BehrooziWechslerHearin_2019MNRAS.488.3143B`), with $\log_{10}M_1$,
$\epsilon$ and $\alpha$ evolving as $p_0 + p_a(a-1) - p_{\ln a}\ln a + p_z z$,
$\beta$ without the $\ln a$ term, $\delta$ constant, and $\log_{10}\gamma$ as
$\gamma_0 + \gamma_a(a-1) + \gamma_z z$.
It is not the same function as
{cite:t}`BehrooziWechslerConroy_2013ApJ...770...57B`, which shares its three
authors and a purpose: that one is a single power law with a $\delta/\gamma$ term
and an $e^{-4a^{2}}$ damping, and this one a double power law with a Gaussian
term and no damping.
Both are kept, since an abundance-matching fit and a merger-tree backbone are
different claims about the same axis.
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
A bare bisection returns *exactly zero at every mass*, since it depends on its
inputs only through the sign of a residual, while its value is correct to
machine precision.
Bracketing under a `stop_gradient` and taking one Newton step leaves the value
unchanged and recovers the implicit-function derivative, which agrees with a
central difference.
A forecast that varies this parameter through the bisection inherits a column of
zeros in its information matrix and reads it as infinite confidence.

```python
zm = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42, gamma=1.21)  # their fit
mstar = lambda beta: SH.mstar_from_mh_zu15(13.0, **(zm | {"beta": beta}))
print(jax.grad(mstar)(0.33), (mstar(0.33 + 1e-5) - mstar(0.33 - 1e-5)) / 2e-5)
```

## Next

- {doc}`notebooks/layer3_galaxies`: the registries of this page, executed, with
  figures.
- {doc}`layer2_halos`: the halo field these models populate.
