# Layer 4: power spectra

This layer takes the weight per halo that layer 3 gives each tracer and turns any two of them into a power spectrum, through one halo-model integral over the mass grid of the halo field.
The same integral serves the galaxies, the AGNs, the hot gas, the free electrons, the X-ray sky, the matter and the intrinsic alignments, and their cross-spectra.
The technical paper (Comparat, in prep., Sec. 5) documents it; this page follows it.
The spectra are at the single redshift of the halo field, in $(h^{-1}{\rm Mpc})^3$ times the units of the two tracers.

:::{warning}
Layer 4 is **work in progress**: released as tested code ahead of its
verification, so its interface and results may change.
Importing `ggah_mod.spectra` emits a `ggah_mod.WorkInProgressWarning` once per
session, which the snippets below silence.
:::

| module | provides |
|---|---|
| {py:mod}`ggah_mod.spectra.tracers` | {py:func}`~ggah_mod.spectra.tracers.spectrum`, the named tracers, {py:func}`~ggah_mod.spectra.tracers.matter_suppression` |
| {py:mod}`ggah_mod.spectra.spec` | {py:class}`~ggah_mod.spectra.spec.PkOptions`, {py:class}`~ggah_mod.spectra.spec.TracerSpec`, {py:class}`~ggah_mod.spectra.spec.Component`, {py:class}`~ggah_mod.spectra.spec.Overlap`, {py:class}`~ggah_mod.spectra.spec.Band` |
| {py:mod}`ggah_mod.spectra.pk` | {py:func}`~ggah_mod.spectra.pk.pk_cross`, the integral for two weights, and {py:class}`~ggah_mod.spectra.pk.PowerSpectrum` |
| {py:mod}`ggah_mod.spectra.pair` | the pair rule of the one-halo term and the shot noise |
| {py:mod}`ggah_mod.spectra.counterterm` | the low-mass counterterm |
| {py:mod}`ggah_mod.spectra.neutrinos` | the neutrino leg of the two-halo term |
| {py:mod}`ggah_mod.spectra.bnl` | the beyond-linear halo bias |
| {py:mod}`ggah_mod.spectra.transition` | the optional damping of the one-halo term |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import; the X-ray sky needs it

import warnings
import numpy as np
import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.halos.profiles import ejected_uk

field = make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.25)
k = np.asarray(field.k)
gal, gp = S.GalaxySector("zumandelbaum15", calibration="off"), S.galaxy_defaults("zumandelbaum15")
gas, dp = S.HotGasDPM(backend=DIFFERENTIABLE), S.DpmParams()
f_cen, f_sat = gal.stellar_fraction(field, gp, satellites=True)
split = S.EjectaSector.split_from_gas(field, gas, dp, f_star_cen=f_cen, f_star_sat=f_sat)

sectors = {"galaxies": gal, "gas": gas, "agn": S.AgnSector(gal, calibration="off"),
           "matter": S.MatterField(), "ejecta": S.EjectaSector()}
params = {"galaxies": gp, "gas": dp, "agn": S.AgnParams(),
          "matter": {"split": split}, "ejecta": {"split": split}}
options = SP.PkOptions.from_backend(DIFFERENTIABLE)
```

The {doc}`notebook <notebooks/layer4_spectra>` draws the spectra below and the effect of each option.

## One-halo and two-halo terms

Every tracer arrives as a {py:class}`~ggah_mod.sectors.protocol.TracerWeights`, its weight per halo $W_a(k|M) = (w_{\rm point} + w_{\rm extended})/n_a$ ({doc}`layer3_galaxies`), with $n_a$ the mean of the tracer, its number density for a discrete one.
The halo model is two integrals over one mass grid:

$$
\begin{aligned}
P^{1h}_{ab}(k) &= \int{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}\,\Pi_{ab}(k|M),\\
P^{2h}_{ab}(k) &= P_{cb}(k)\left[\mathcal I_a(k)\,\mathcal I_b(k) + I^{\rm NL}_{ab}(k)\right],\\
\mathcal I_a(k) &= \int{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}\,b_a(M)\,W_a(k|M) + \Delta I_a(k) + \frac{\nu_a}{n_a}L(k),
\end{aligned}
$$ (eq-pab)

with $\Pi_{ab}$ the pair function, $b_a$ the halo bias (or the tracer's own, `bias_weight`), $\Delta I_a$ the low-mass counterterm, $I^{\rm NL}_{ab}$ the beyond-linear term and $L$ the neutrino leg carried with the weight $\nu_a$.
The total is

$$
P_{ab}(k) = T(k)\,P^{1h}_{ab}(k) + P^{2h}_{ab}(k),
$$ (eq-pk-total)

with $T = 1$ by default.
The linear spectrum is $P_{cb}$, that of the cold matter, because $\sigma(M)$, ${\rm d}n/{\rm d}M$ and $b(M)$ are built from it.
{py:class}`~ggah_mod.spectra.spec.PkOptions` holds the choices, which each flavour sets:

```{include} _generated/pk_options.md
```

## Named tracers and composites

{py:func}`~ggah_mod.spectra.tracers.spectrum` takes two tracers by name, the sector instances and their parameters, and sums over their components, $P_{ab} = \sum_{i\in a}\sum_{j\in b}P_{ij}$, each pair under its own rule:

```{include} _generated/tracers.md
```

The X-ray sky is hot gas plus point sources, $P_{XX} = P_{\rm gas,gas} + 2P_{\rm gas,agn} + P_{\rm agn,agn}$, and the free electrons, which the kinetic SZ effect and the dispersion measure see, are the gas on its profile plus the ejected gas, which counts every electron once only when the split is the gas model's own ({py:meth}`~ggah_mod.sectors.ejecta.EjectaSector.split_from_gas`).
A sector that needs another (the AGNs need the galaxies, the closure of the hot gas the AGNs and galaxies) reads it from the same dictionaries, and must be the instance it was built with.
Lensing is not a tracer: it is the matter field under a radial kernel, in {doc}`layer5_observables`.
A tracer outside the registry, such as the clusters of {doc}`layer3_galaxies`, goes through {py:func}`~ggah_mod.spectra.pk.pk_cross` with its weights directly.

```python
for a, b in (("galaxies", "galaxies"), ("galaxies", "matter"), ("matter", "matter"),
             ("galaxies", "pressure"), ("xray", "xray"), ("electrons", "electrons")):
    p_ab = SP.spectrum(field, a, b, sectors, params, options=options)   # PowerSpectrum
    print(f"{a:9s} x {b:9s} P(k = 0.1) = {np.interp(0.1, k, p_ab.total):.3e}")
```

## Discrete tracers and the pair rule

For the auto-spectrum of a discrete tracer the pair function counts distinct pairs of objects, and otherwise it is the product of the two weights:

$$
\Pi_{ab}(k|M) =
\begin{cases}
2\,w^a_{\rm p}w^a_{\rm e} + \left(w^a_{\rm e}\right)^2 & a \equiv b,\ \text{discrete},\\
\left(w^a_{\rm p} + w^a_{\rm e}\right)\left(w^b_{\rm p} + w^b_{\rm e}\right) & \text{otherwise},
\end{cases}
$$ (eq-pair)

with $w_{\rm p}$ the central object and $w_{\rm e}$ the satellites on their profile, both divided by $n$.
The first line is $\langle N(N-1)\rangle$ rather than $\langle N\rangle^2$ {cite:p}`AsgariMeadHeymans_2023OJAp....6E..39A`: a central object is not paired with itself and the satellites are Poisson about it.
A tracer of central galaxies only has no one-halo auto-term.

### Overlap of two samples

The second line holds only when no object is in both samples.
Each component carries a population label, a path such as `galaxies/cen`, and the overlap is declared rather than inferred: a continuous tracer overlaps nothing, equal labels are identical, a label that begins another (`galaxies` and `galaxies/cen`) is *nested*, and siblings (`galaxies/cen` and `galaxies/sat`, or `galaxies` and `agn`) are disjoint.
A nested pair raises, because a label does not say which components the samples share; an {py:class}`~ggah_mod.spectra.spec.Overlap` passed through `overlaps=` declares whether the two point populations are the same object and whether the samples share objects.
AGNs that sit inside a galaxy sample are labelled under it, `galaxies/agn`.

### Shot noise

The shot noise of a discrete tracer is

$$
P^{\rm shot}_{aa} = \frac{1}{n_a^2}\int{\rm d}M\,\frac{{\rm d}n}{{\rm d}M}\,\langle N\rangle\,\ell^2,
$$ (eq-shot)

with $\ell$ the weight per object: $1/\bar n$ for counts, the second moment of the luminosity for the AGN emission.
A cross-spectrum has shot noise only when its samples share objects.
It is kept apart from `PowerSpectrum.total`, since whether it belongs in a prediction depends on the estimator; `with_shot()` adds it.

```python
p_gg = SP.spectrum(field, "galaxies", "galaxies", sectors, params, options=options)
print(f"shot noise {float(p_gg.shot):.1f} (h^-1 Mpc)^3 = 1/n = {1 / float(gal.number_density(field, gp)):.1f}")
p_cs = SP.spectrum(field, "centrals", "satellites", sectors, params, options=options)   # siblings
try:
    SP.spectrum(field, "galaxies", "centrals", sectors, params, options=options)    # nested
except NotImplementedError as err:
    print(type(err).__name__)
```

## Normalisation of the two-halo term

The mass function and the bias are calibrated separately, and on a grid that stops at $M_{\min}$ the bias-weighted mass $C = \int{\rm d}M\,({\rm d}n/{\rm d}M)\,b\,M/\bar\rho_{cb}$ is below one ({py:func}`~ggah_mod.spectra.counterterm.bias_consistency`).
The missing bias-weighted mass is put back at the bottom of the grid with $b = 1$ {cite:p}`Schmidt_2016PhRvD..93f3512S`, as in {cite:t}`MeadVerde_2021MNRAS.503.3095M` (their App. A):

$$
\Delta I_a(k) = (1 - C)\,\hat W_a(k),
\qquad
\hat W_a(k) = \begin{cases}
W_a(k|M_{\min})\,\bar\rho_{cb}/M_{\min} & \text{by default},\\
w^a_{\rm u}(k) & \text{if the tracer declares the composition of the unresolved matter}.
\end{cases}
$$ (eq-counterterm)

It is zero for a discrete tracer, since a threshold sample does not continue below its threshold, and it is not clipped.
For the matter it makes $\mathcal I_{\rm m}(k\to0) = \bar\rho_{cb}/\bar\rho_{\rm m} = 1 - f_\nu$ exactly.

## Massive neutrinos

The neutrinos are in no halo: above the free-streaming scale they follow the linear fluctuation of the cold field, so the two linear spectra fix their amplitude,

$$
L(k) = f_\nu\frac{T_\nu(k)}{T_{cb}(k)} = \sqrt{\frac{P_{\rm m}(k)}{P_{cb}(k)}} - \frac{\bar\rho_{cb}}{\bar\rho_{\rm m}},
$$ (eq-nu-leg)

which tends to $f_\nu$ on large scales and to zero below the free-streaming scale {cite:p}`MassaraVillaescusa-NavarroViel_2014JCAP...12..053M`.
The matter carries it with $\nu = 1$, the intrinsic alignments with $b_I$, and the tracers of the cold field with zero.
With the counterterm, $P^{2h}_{\rm mm} \to P_{\rm m}$ on large scales.
`neutrino_two_halo="none"` drops it, and a two-halo term on the total linear spectrum (`two_halo_spectrum="total"`) refuses it, which would count the neutrinos twice.

## Beyond-linear halo bias

Haloes are not linearly biased where the one-halo and two-halo terms cross.
{cite:t}`MeadVerde_2021MNRAS.503.3095M` measure the residual in MultiDark as $P_{hh} = b_1b_2P_{\rm lin}[1 + \beta^{\rm NL}(k, \nu_1, \nu_2)]$, which layer 2 tabulates and rescales to the cosmology with {cite:t}`AnguloWhite_2010MNRAS.405..143A` ({doc}`layer2_halos`).
The two-halo term gains

$$
I^{\rm NL}_{ab}(k) = \sum_i\sum_j\tilde w^a_i(k)\,\tilde w^b_j(k)\,\beta^{\rm NL}(k, \nu_i, \nu_j),
\qquad
\tilde w^a_i = \left(\frac{{\rm d}n}{{\rm d}M}\right)_i b_a(M_i)\,W_a(k|M_i)\,\Delta M_i,
$$ (eq-bnl-2h)

with $\Delta M_i$ the quadrature weights of the mass grid, and the counterterm $\Delta I_a$ added to the first node, the low-mass completion of their Eqs. A7–A10.
It is on by default (`bnl=True`).
For the matter, it brings the spectrum within a few per cent of HMcode-2020 {cite:p}`MeadBriedenTroster_2021MNRAS.502.1401M` at $0.05 < k < 1\,h\,{\rm Mpc}^{-1}$ and $z \ge 0.5$; the paper's comparison gives the numbers.

## The one-halo plateau

The one-halo term tends to a constant on large scales, which is kept by default ($T = 1$).
It is the integral of a $\xi^{1h}$ with compact support, so $\xi$, $w_p$ and $\Delta\Sigma$ at $r > 0$ do not see it, and for a tracer of rare haloes it is their Poisson power, the large-scale thermal-SZ signal.
Only the matter, whose halo contribution must go as $k^4$, has a reason to lose it; for $P_{\rm mm}$ in Fourier space, `one_halo_transition="mead20"` damps it with the form of HMcode-2020,

$$
T(k) = \frac{(k/k_*)^4}{1 + (k/k_*)^4},
\qquad k_* = c_*\,\sigma_{8,cb}(z)^{-1.0131066},
$$ (eq-mead20)

with $c_*$ free, since the published value was fitted with the two-halo damping of HMcode, which this package does not have:

```{include} _generated/params_transition.md
```

```python
for opts in (SP.PkOptions.from_backend(DIFFERENTIABLE, bnl=False), options,
             SP.PkOptions.from_backend(DIFFERENTIABLE, one_halo_transition="mead20")):
    p_mm = SP.spectrum(field, "matter", "matter", sectors, params, options=opts)
    print(np.interp([0.01, 0.5], k, p_mm.total))
```

## Baryonic suppression

The suppression of the matter power spectrum,

$$
S(k) = \frac{P_{\rm mm}}{P_{\rm mm}^{\rm DMO}},
$$ (eq-suppression)

is built on the same weights.
The dark-matter-only reference keeps every component's mass and puts it back on $\tilde u_{\rm DM}$, so the two spectra differ only in where the baryons are.
The matter parameters carry the profiles: without them every component sits on $\tilde u_{\rm DM}$ and $S = 1$.
The hot gas cut at $R_\Delta$ (`bound_uk`) and the ejected shell count the displaced gas once ({doc}`closure_options`):

```python
u_gas = gas.bound_uk(field.k, field.m, field.z, field.cosmo, dp, field.mdef, conc=field.conc)
u_ej = ejected_uk(field.k, field.r_delta, eta_ej=2.0)
params["matter"] = {"split": split, "u_gas": u_gas, "u_ej": u_ej}
k_s, s_k = SP.matter_suppression(field, sectors, params, options=options)
print(np.interp([0.1, 1.0, 10.0], k_s, s_k))
```

## Scale-dependent bias from primordial non-Gaussianity

A local $f_{\rm NL}$ adds a scale-dependent term to the bias of every halo {cite:p}`DalalDoreHuterer_2008PhRvD..77l3514D`,

$$
\Delta b(k, M) = 3f_{\rm NL}\left[b(M) - p\right]\delta_c\,\frac{\Omega_{\rm m}}{T(k)\,D(z)}\left(\frac{H_0}{ck}\right)^2,
$$ (eq-png)

with $p = 1$ for a universal mass function and $1.6$ for recent mergers {cite:p}`SlosarHirataSeljak_2008JCAP...08..031S`, $T(k)$ the transfer function of $P_{cb}$ and $D$ the growth factor.
{py:func}`~ggah_mod.sectors.png.apply_png_bias` sets it as the tracer's `bias_weight`, so it reaches the two-halo term and not the counterterm; `p` and `growth` have no default.

```python
from ggah_mod.cosmology import growth_factor
from ggah_mod.sectors.png import apply_png_bias

w_g = gal.weights(field, gp)
w_png = apply_png_bias(w_g, field, 10.0, p=1.0, growth=growth_factor(0.25, PLANCK18, make_pk("emu_pk")))
ratio = (SP.pk_cross(field, w_png, w_png, overlap="identical", options=options).total
         / SP.pk_cross(field, w_g, w_g, overlap="identical", options=options).total)
print(np.interp([0.001, 0.01, 0.1], k, ratio))                 # f_NL = 10
```

## One redshift

Every spectrum is at the redshift of its field: no function takes a redshift, and a weight stacked over redshifts raises.
Projections over redshift are layer 5's, which builds one field per redshift node.

## Related pages

- {doc}`notebooks/layer4_spectra`: the spectra above, executed, with figures.
- {doc}`layer3_galaxies`, {doc}`layer3_agn` and {doc}`layer3_hot_gas`: the weights.
- {doc}`layer5_observables`: the projections of these spectra.
- {doc}`api/spectra`: every function and its arguments.
