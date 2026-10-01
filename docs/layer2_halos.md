# Layer 2: the haloes

The halo layer turns the outputs of {doc}`layer1_cosmology` into the four
quantities a halo model needs: how many haloes there are at each mass, how
strongly they cluster, how concentrated they are, and what they look like in
Fourier space.
It depends on the cosmology layer and on nothing else; in particular it depends
on no tracer, so the mass grid, the mass function and the profiles are shared by
every tracer built on it.
Most of the layer is calibrated fits: eighteen multiplicity functions (sixteen
published fits and two recalibrations of one of them), six bias fits and six
concentration relations, each behind one contract.

| module | provides |
|---|---|
| {py:mod}`ggah_mod.halos.variance` | $\sigma(M)$, ${\rm d}\ln\sigma/{\rm d}\ln M$, $\delta_{\rm c}$ |
| {py:mod}`ggah_mod.halos.mass_definitions` | {py:class}`~ggah_mod.halos.mass_definitions.MassDef`: 200m, 200c, 500c, vir, and conversions |
| {py:mod}`ggah_mod.halos.mass_function` | ${\rm d}n/{\rm d}M$ and the multiplicity registry |
| {py:mod}`ggah_mod.halos.linear_bias` | $b(\nu)$, the bias registry, the peak-background-split integrals |
| {py:mod}`ggah_mod.halos.concentration` | $c(M)$ and its registry |
| {py:mod}`ggah_mod.halos.profiles`, {py:mod}`~ggah_mod.halos.lensing_profiles` | $\rho(r)$, $u(k\vert M)$, $\Sigma(R)$, $\Delta\Sigma(R)$ |
| {py:mod}`ggah_mod.halos.beyond_linear_bias` | $\beta^{\rm NL}(k,\nu_1,\nu_2)$ |
| {py:mod}`ggah_mod.halos.calibration` | which fits may be used at which mass definition |
| {py:mod}`ggah_mod.halos.field` | {py:class}`~ggah_mod.halos.field.HaloField`, which holds all of the above on one grid |

Every snippet below runs on a `pip install`, in order, after

```python
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

import numpy as np
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field, make_fields

pk = make_pk("emu_pk")
```

The {doc}`notebook <notebooks/layer2_halos>` works through the same material
with figures.

## The halo field

{py:func}`~ggah_mod.halos.field.make_field` runs the whole chain of this page for
one cosmology, one power spectrum and one redshift, and returns a
{py:class}`~ggah_mod.halos.field.HaloField`.
The flavour supplies the mass and wavenumber grids and the four model choices
(mass definition, multiplicity, bias, concentration), so a flavour is selected
once; an explicit argument overrides any of them.
The field is a frozen JAX pytree whose differentiable leaves are

$$
\left\{M,\ k,\ z,\ \sigma,\ \frac{{\rm d}\ln\sigma}{{\rm d}\ln M},\ \nu,\
\frac{{\rm d}n}{{\rm d}M},\ b,\ c,\ r_\Delta,\ r_{\rm s},\ P_{\rm cb},\ P_{\rm lin},\
\theta\right\},
$$ (eq-halofield)

with $\theta$ the cosmology, and whose four model names are static entries of
its tree definition.
It adds no physics to the loose functions described below: the suite checks that
its tables equal the hand-assembled chain.

```python
field = make_field(PLANCK18, DIFFERENTIABLE, pk, z=0.0)
print(field.mdef, field.hmf_model, field.bias_model, field.cm_model)
print(field.m.shape, field.k.shape, field.u_nfw().shape)   # (NM,), (Nk,), (Nk, NM)
```

`make_field` requires a power spectrum and refuses a vector redshift; for several
redshifts, {py:func}`~ggah_mod.halos.field.make_fields` returns one field per
redshift from a single spectrum call, which matters for a Boltzmann backend that
solves once per call.

## Variance and peak height

The chain starts from the top-hat variance of layer 1, evaluated at the
Lagrangian radius of the sphere that contained the halo's mass before collapse,

$$
\sigma^2(M) = \frac{1}{2\pi^2}\int P_{\rm cb}(k)\,W^2\!\big(kR(M)\big)\,k^2\,{\rm d}k,
\qquad R(M) = \left(\frac{3M}{4\pi\bar\rho_{\rm cb}}\right)^{1/3} ,
$$ (eq-sigma-m)

on the cold spectrum and the cold density, because haloes form out of the
CDM + baryon field.
Its logarithmic slope, which the mass function needs, is taken by automatic
differentiation rather than by differencing,
${\rm d}\ln\sigma/{\rm d}\ln M = \tfrac13\,{\rm d}\ln\sigma/{\rm d}\ln R$, and the
peak height is

$$
\nu(M,z) = \frac{\delta_{\rm c}}{\sigma(M,z)},\qquad \delta_{\rm c} = 1.686 ,
$$ (eq-nu)

with $\delta_{\rm c}$ held at the value the fits written in $\nu$ use, except
`despali16`, which was calibrated with a redshift-dependent $\delta_{\rm c}(z)$;
the fits written in $\sigma$ do not depend on it.
Two densities appear in this layer and are not interchangeable: $\bar\rho_{\rm cb}$
sets $R(M)$, and $\bar\rho_{\rm m}$ sets the spherical-overdensity boundaries,
which are defined against all the matter.
Once neutrinos have mass the quantities split as follows.

| built from the cold field ($P_{\rm cb}$, $\bar\rho_{\rm cb}$) | built from the total matter ($P_{\rm m}$, $\bar\rho_{\rm m}$) |
|---|---|
| $\sigma(M)$, ${\rm d}\ln\sigma/{\rm d}\ln M$, $R(M)$, $\nu$ | $\sigma_8$, $S_8$ |
| ${\rm d}n/{\rm d}M$, $b(M)$, $c(M)$, $\beta^{\rm NL}$ | lensing and the matter two-halo term |
| the mass-weighted bias and mass fraction | the $\Delta\times\bar\rho$ mass definitions and $r_\Delta$ |

```python
i_star = np.argmin(np.abs(field.nu - 1.0))
print(f"nu = 1 at M* = {field.m[i_star]:.2e} h^-1 Msun")
print(field.rho_cold / field.rho_matter)          # 1 - f_nu
```

## Mass definitions

A halo mass is the mass inside the radius $r_\Delta$ within which the mean
density is $\Delta$ times a reference density: the mean matter density for
`200m`, the critical density for `200c` and `500c`, and the $\Delta_{\rm vir}(z)$
fit of {cite:t}`BryanNorman_1998ApJ...495...80B` for `vir`.
{py:class}`~ggah_mod.halos.mass_definitions.MassDef` computes the radius, the
overdensity with respect to the mean density that $\Delta$-dependent fits are
indexed by, and conversions between definitions for an NFW profile.
Both flavours use `200m`.

```python
from ggah_mod.halos import MassDef, translate_mass

for name in ("200m", "200c", "vir"):
    print(name, MassDef.from_string(name).delta_mean(0.0, PLANCK18))
m200c, r200c, c200c = translate_mass(1e14, 5.0, "200m", "200c", 0.0, PLANCK18)
```

## Abundance

The mass function is assembled from a multiplicity function $f(\sigma)$,

$$
\frac{{\rm d}n}{{\rm d}M} = f(\sigma)\,\frac{\bar\rho_{\rm cb}}{M^2}
\left|\frac{{\rm d}\ln\sigma}{{\rm d}\ln M}\right| .
$$ (eq-dndm)

Sixteen published fits are re-implemented in JAX, following the registry of
`colossus` {cite:p}`Diemer_2018ApJS..239...35D`, each with the halo definition and
redshift range it was calibrated on (`CALIBRATION`):

```{include} _generated/multiplicity.md
```

The {cite:t}`TinkerKravtsovKlypin_2008ApJ...688..709T` fit,

$$
f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]e^{-c/\sigma^2} ,
$$ (eq-tinker08)

is tabulated at nine overdensities and interpolated in $\log\Delta$.
A fit of this form depends on the cosmology only through $\sigma$, which a
simulation-based emulator does not assume, but the published emulators
{cite:p}`McClintockRozoBecker_2019ApJ...872...53M,NishimichiTakadaTakahashi_2019ApJ...884...29N,BocquetHeitmannHabib_2020ApJ...901....5B`
are not differentiable.
`tinker08_csst`, the default of both flavours, keeps the form and rescales its
four coefficients by a factor $1 + g(\theta, z)$ predicted by `emu_hmf`, a
network trained on the CSST emulator of {cite:t}`ChenYu_2025SCPMA..6809513C`;
`tinker08_csst_vir` is the same at the virial definition.
Both refuse a cosmology outside the CSST training box, and warn outside
$0 \le z \le 3$.

```python
from ggah_mod.halos import dndm

f_bare = dndm(field.m, field.sigma, field.dlns, PLANCK18.rho_cold,
              model="tinker08", delta=200.0)
print(field.dndm[::64] / f_bare[::64] - 1)        # the recalibration, per mass
```

## Bias, and why it is not a free choice

Six fits give the large-scale bias as a function of peak height, $b = b(\nu)$.
A mass function and a bias calibrated together are a pair, and `MATCHED_BIAS`
records the published pairs.
For `press74`, `sheth99`, `bhattacharya11` and `comparat17` the bias follows from
the mass function by a peak-background split; `tinker10` is a direct fit to the
measured bias, normalised against the Appendix C mass function of
{cite:t}`TinkerKravtsovKlypin_2008ApJ...688..709T`; and `despali16` is paired
with `sheth99`, the same functional form with Sheth and Tormen's own
coefficients, because {cite:t}`DespaliGiocoliAngulo_2016MNRAS.456.2486D` publish
no bias.

```{include} _generated/bias.md
```

If the abundance and the bias are consistent, weighting the mass function by the
bias recovers the mean density,

$$
\int b(M)\,\frac{M}{\bar\rho_{\rm cb}}\,\frac{{\rm d}n}{{\rm d}M}\,{\rm d}M
\longrightarrow 1 ,
$$ (eq-pbs)

and the same integral without $b$ is the fraction of the mass bound in haloes.
Over any finite mass range both fall short of one and lose the same mass, so
their ratio, the mass-weighted mean bias, isolates the pairing from the range.
The mass definition matters more than the partner: using the coefficients of the
wrong definition moves these integrals by tens of per cent, and both halves of
the pair must carry the same definition.
{py:func}`~ggah_mod.halos.calibration.check_calibration`, called by `make_field`,
therefore refuses a fit at a definition it was not calibrated in (the policy is
`calibration="strict"`, `"warn"` or `"off"`).
The package pairs `tinker08_csst` with `tinker10`, both at `200m`.

```python
from ggah_mod.halos import mass_fraction, mass_weighted_bias

bound = mass_fraction(field.m, field.dndm, field.rho_cold)
weighted = mass_weighted_bias(field.m, field.dndm, field.bias, field.rho_cold)
print(bound, weighted, weighted / bound)

try:
    make_field(PLANCK18, DIFFERENTIABLE, pk, mdef="200c")
except ValueError as err:
    print(err)                                     # tinker08_csst is a 200m fit
```

## Concentration

Six concentration relations come in two families with different signatures:
power laws in mass and redshift, $c(M, z, \Delta)$, and relations in peak height,
$c(\sigma, \cdot)$, which read the cosmology through $\sigma(M)$ and, for some,
the growth factor or the slope of the spectrum.
They keep separate interfaces because a power law in $(M, z)$ cannot respond to
a cosmology, and one that accepted a cosmology and ignored it would be
indistinguishable from one that used it.
`CM_CALIBRATION` records the simulations, mass definitions and redshift range of
each:

```{include} _generated/concentration.md
```

Both flavours use `bhattacharya13` {cite:p}`BhattacharyaHabibHeitmann_2013ApJ...766...32B`,
which covers `200m` and reads the growth factor off the power spectrum; it
therefore needs a backend that knows its own redshift dependence.
The scale radius follows as $r_{\rm s} = r_\Delta/c$.

```python
i13 = np.argmin(np.abs(field.m - 1e13))
print(field.conc[i13], field.r_delta[i13], field.r_s[i13])   # h^-1 Mpc, comoving
```

## Profiles

The halo model needs the normalised Fourier transform of each halo's density
profile,

$$
u(k\vert M) = \frac{4\pi}{M}\int_0^{r_\Delta}\rho(r)\,\frac{\sin kr}{kr}\,r^2\,{\rm d}r,
\qquad u(k\to0\vert M) = 1 .
$$ (eq-uk)

NFW {cite:p}`NavarroFrenkWhite_1997ApJ...490..493N`, Einasto, a generalised NFW,
a satellite profile (NFW with a concentration factor, an outer cut-off and an
inner slope, each neutral at its default) and a Gaussian ejected-gas profile are
provided, and {py:func}`~ggah_mod.halos.profiles.profile_uk_gl` is the
Gauss–Legendre transform any other profile gets its $u(k\vert M)$ from.
The generalised form

$$
f(x) = x^{-\alpha_{\rm in}}\left(1 + x^{\alpha_{\rm tr}}\right)^{(\alpha_{\rm in}-\alpha_{\rm out})/\alpha_{\rm tr}}
$$ (eq-gnfw)

recovers NFW at $(\alpha_{\rm in}, \alpha_{\rm tr}, \alpha_{\rm out}) = (1, 1, 3)$
and is the one definition of that shape in the package.
For NFW the transform is analytic in the sine and cosine integrals,

$$
u(k) = \frac{1}{g(c)}\Big\{\sin\kappa\big[{\rm Si}((1+c)\kappa) - {\rm Si}(\kappa)\big]
+ \cos\kappa\big[{\rm Ci}((1+c)\kappa) - {\rm Ci}(\kappa)\big]
- \frac{\sin c\kappa}{(1+c)\kappa}\Big\},
$$ (eq-nfw)

with $\kappa = k r_{\rm s}$ and $g(c) = \ln(1+c) - c/(1+c)$.
{py:mod}`~ggah_mod.halos.lensing_profiles` adds the projected profiles lensing
needs, for a truncated NFW, the Baltz–Marshall–Oguri profile and a Hernquist
stellar profile.
Their dimensionless kernels agree with the Oguri et al. (2026) implementation
([arXiv:2512.13954](https://arxiv.org/abs/2512.13954)), evaluated at 50 digits,
to 1.2e-13 in $\Sigma$; the two implementations of the Baltz–Marshall–Oguri
profile here, one in Fourier space and one in real space, agree to 3.0e-7 of the
peak once the first is transformed to real space.

```python
from ggah_mod.halos import nfw_delta_sigma, nfw_params_from_mass

u = field.u_nfw()                                  # (Nk, NM), from each halo's c and r_s
rho_s, r_s, r_200m = nfw_params_from_mass(1e13, field.conc[i13], 0.0, PLANCK18,
                                          mdef="200m")   # the default is 200c
R = np.logspace(-2, 0.5, 30)                       # h^-1 Mpc
dsigma = nfw_delta_sigma(R, rho_s, r_s)            # h Msun Mpc^-2
```

## Beyond-linear halo bias

Haloes are not linearly biased tracers of the linear field on the scales where
the one-halo and two-halo terms meet.
{cite:t}`MeadVerde_2021MNRAS.503.3095M` tabulate the residual from the MultiDark
simulation,

$$
P_{hh}(\nu_1,\nu_2,k) = b(\nu_1)\,b(\nu_2)\,P_{\rm lin}(k)
\left[1 + \beta^{\rm NL}(k,\nu_1,\nu_2)\right] .
$$ (eq-bnl)

It is computed in this layer because it is a function of peak height, and is
applied to the two-halo term of the power spectra, where it is on by default.
$\beta^{\rm NL}$ is not redshift-universal, so all 35 public MultiDark snapshots
are shipped, and the snapshot and length scaling that best match the target
cosmology are chosen by the rescaling of {cite:t}`AnguloWhite_2010MNRAS.405..143A`,
matching $\sigma(R)$ over $1$–$10\,h^{-1}$Mpc.
Below $k = 0.08\,h\,{\rm Mpc}^{-1}$ the correction is tapered to zero, since the
measurement there is consistent with zero, and beyond the table's last wavenumber
it is held at its last value, where the one-halo term dominates.
The table is read with monotone cubic Hermite interpolation
{cite:p}`FritschCarlson_1980SJNA...17..238F` in wavenumber, in peak height and
across snapshots, so its derivatives are continuous.

```python
from ggah_mod.halos.beyond_linear_bias import beta_nl, table_at

table = table_at(field.k, k=field.k, pk_cb=field.pk_cb)
nu = np.array([1.0, 2.0])
beta = beta_nl(field.k, nu, nu, table)             # (Nk, 2, 2)
```

## Curvature

$E(z)$ is the only route curvature takes into this layer.
The `vir` definition uses the flat-universe coefficients of
{cite:t}`BryanNorman_1998ApJ...495...80B`, and no published pair covers a
universe with both curvature and dark energy, so `make_field` refuses `vir`, and
the `despali16` fit indexed by it, at $\Omega_k \ne 0$; `200m` and `200c` stay
exact.
The CSST suite behind `tinker08_csst` is flat, so that fit warns at any
$\Omega_k \ne 0$ and refuses beyond the $|\Omega_k|$ at which its correction
stops improving on the uncorrected `tinker08`.
Only `tinker08_csst` and `tinker08_csst_vir` were calibrated on simulations
with massive neutrinos: the CSST suite's box has a neutrino-sum axis, run with
three degenerate masses. No bias or concentration fit was, and nothing here was
calibrated on curved simulations; until such simulations exist we recommend
using this layer at $\Omega_k = 0$.

## Redshifts and derivatives

Every leaf of a field is differentiable whenever its power spectrum is, which
with `emu_pk` means the whole layer.
A field can be compiled, and differentiated with respect to the cosmology or the
redshift:

```python
fields = make_fields(PLANCK18, DIFFERENTIABLE, pk, np.array([0.0, 0.5, 1.0]))
print([float(f.z) for f in fields])

above = (field.m > 1e13).astype(float)             # an occupation: every halo above 1e13

def b_eff(Omega_m):
    f = make_field(PLANCK18.replace(Omega_m=Omega_m), DIFFERENTIABLE, pk, z=0.5)
    return f.effective_bias(above)

print(b_eff(0.31), jax.grad(b_eff)(0.31))
```

## Next

- {doc}`notebooks/layer2_halos`: this page, executed, with figures.
- {doc}`api/halos`: every function and its arguments.
