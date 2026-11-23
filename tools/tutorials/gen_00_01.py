"""Write tutorials/00 and 01 as notebooks (sources only; executed separately)."""
import os
import nbformat as nbf

OUT = os.path.expanduser("~/software/ggah_mod/tutorials")


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}}
    book.cells = [nbf.v4.new_markdown_cell(s.strip("\n")) if kind == "md"
                  else nbf.v4.new_code_cell(s.strip("\n")) for kind, s in cells]
    return book


SETUP = r'''
import jax
jax.config.update("jax_enable_x64", True)   # before the first ggah_mod import

from importlib.metadata import version

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import _helpers as T
T.paper_style()
for name in ("ggah_mod", "emu_pk", "jax", "numpy"):
    print(f"{name:9s} {version(name)}")
'''

# ---------------------------------------------------------------- 00
n00 = [
("md", r'''
# 0. Getting started

These notebooks reproduce each section and each figure of Paper I, the cosmology and halo layers of `ggah_mod` and the constraints they give.
Each notebook follows one section: the equations, the calls that compute them step by step, the figure of the paper, a check of its numbers against the text, and exercises.
This one sets up the environment and introduces the two objects every later notebook uses: the cosmology container and the backend that says how to compute with it.

| Notebook | Paper I | Figures |
|---|---|---|
| 01 Cosmology and background | Secs. 2.1–2.2 | distance gradients |
| 02 Power spectrum | Sec. 2.3 | $P(k)$ gradients |
| 03 Amplitude and growth | Secs. 2.4–2.5 | amplitude gradients |
| 04 Variance and peak height | Sec. 3.1 | variance |
| 05 Abundance and bias | Secs. 3.2–3.3 | abundance |
| 06 Concentration and profiles | Secs. 3.4–3.7 | concentration, profile |
| 07 Distance constraints | Sec. 4.1 | distance Fisher |
| 08 Full shape | Sec. 4.2 | $P(k)$ Fisher, with BAO and BBN |
| 09 Halo counts | Sec. 4.3 | mass-function Fisher |
| 10 Halo-layer requirements | Sec. 4.4 | halo requirements |
| 11 A joint halo emulator | Secs. 4.5–4.6 | suite errors |
| A Emulator benchmark | App. A | emulator benchmark |
| B Code comparisons | App. B | comparison tables |

Notebooks 01–09 compute everything from a `pip install`.
Notebooks 10, 11, A and B also read precomputed results from the tutorial data bundle (see the end of this notebook).
'''),
("md", r'''
## Installation

```
pip install "ggah_mod[tutorials]"
```

installs `ggah_mod`, `emu_pk`, `matplotlib`, `pooch` and `jupyter`.
CLASS (`pip install "ggah_mod[reference]"`) is needed only for the cells that compare against it, which are skipped otherwise.
The notebooks import `_helpers.py` from their own directory, so run them from `tutorials/`.

`ggah_mod` assumes double precision.
JAX defaults to single precision, so every notebook starts by switching it on, before the first `ggah_mod` import.
'''),
("code", SETUP),
("md", r'''
## The cosmology container

A `Cosmology` holds eleven fields.
Ten are differentiable leaves of a JAX pytree, so `jax.grad` of anything downstream flows through them; the eleventh, the neutrino ordering, is static.
`PLANCK18` is the default of the paper (Eq. 1): the *Planck* 2018 TT,TE,EE+lowE+lensing means at the minimal neutrino mass.
'''),
("code", r'''
from ggah_mod.cosmology import PLANCK18, Cosmology

print(PLANCK18)
c = PLANCK18.replace(Omega_m=0.30, w0=-0.9)   # a new container; PLANCK18 is unchanged
print(c.Omega_m, c.w0, PLANCK18.Omega_m)
'''),
("md", r'''
$\sigma_8$ and $S_8$ are outputs of the power spectrum, not inputs; asking for one raises.
`ln10A_s_for_sigma8` returns the amplitude that gives a requested $\sigma_8$ (notebook 03).
'''),
("code", r'''
try:
    Cosmology.create(sigma8=0.8)
except TypeError as err:
    print(err)
'''),
("md", r'''
## Two flavours

A backend says how to compute: which power spectrum, how fine the grids, which halo fits.
Two are provided.
`ACCURATE` is the fitting path: a Boltzmann solver (CLASS by default) on fine grids.
`DIFFERENTIABLE` is the forecasting path: the `emu_pk` emulator, the only differentiable backend, on coarser grids, traced end to end.
'''),
("code", r'''
from ggah_mod import ACCURATE, DIFFERENTIABLE

print(ACCURATE)
print(DIFFERENTIABLE)
'''),
("md", r'''
Every notebook uses `DIFFERENTIABLE`, since the paper's figures are derivatives and forecasts.
A gradient through a Boltzmann solver raises `NotDifferentiableError` rather than returning zero.
'''),
("code", r'''
from ggah_mod.cosmology import make_pk, sigma8
from ggah_mod.cosmology.power import NotDifferentiableError

k = np.logspace(-4, np.log10(50.0), 512)
pk = make_pk("emu_pk")


def sigma8_of(omega_m, pk):
    return sigma8(pk.pk(k, 0.0, PLANCK18.replace(Omega_m=omega_m)), k)


print("sigma8 =", float(sigma8_of(PLANCK18.Omega_m, pk)))
print("d sigma8 / d Omega_m =", float(jax.grad(sigma8_of)(PLANCK18.Omega_m, pk)))

try:
    jax.grad(sigma8_of)(PLANCK18.Omega_m, make_pk("class"))
except NotDifferentiableError as err:
    print("CLASS:", str(err).splitlines()[0])
except ImportError:
    print("classy is not installed; the CLASS backend is optional")
'''),
("md", r'''
## The tutorial data

Notebooks 07–11, A and B read files from a data bundle on Zenodo ([doi:10.5281/zenodo.24469507](https://doi.org/10.5281/zenodo.24469507)): published DESI posteriors drawn over the forecasts, and results that took cluster time to compute.
`_helpers.data_path(name)` returns the local path of a file.
It looks in the directory named by `GGAH_TUTORIALS_DATA`, then in `~/.cache/ggah_mod/tutorials`, and downloads the bundle once otherwise.
Each file is checked against its SHA-256 in `data_registry.txt`, so a notebook reads exactly the file the paper was made from.
'''),
("code", r'''
for name in list(T._registry())[:4]:
    print(name)
print("...", len(T._registry()), "files")
'''),
("md", r'''
## Run time

Each notebook has a `FAST` switch, set from the environment variable `GGAH_TUTORIALS_FAST` (default on).
With `FAST` on, grids are reduced so that a notebook runs in a few minutes on a laptop CPU; with it off, the notebook reproduces the figure of the paper exactly.
The first call of a function compiles it, so the second is the one to time.
'''),
("code", r'''
import time

f = jax.jit(lambda om: sigma8_of(om, pk))
for label in ("first call (compiles)", "second call"):
    t0 = time.perf_counter()
    f(0.31).block_until_ready()
    print(f"{label:22s} {1e3 * (time.perf_counter() - t0):8.2f} ms")
'''),
("md", r'''
## Exercises

1. Build a cosmology with `nu_hierarchy="inverted"` and `sum_mnu=0.06`, read the error, and find the smallest sum it accepts.
2. Compute $\partial\sigma_8/\partial h$ with `jax.grad`, and check it against a central finite difference with a step of $10^{-4}$.
3. Time `sigma8_of` with and without `jax.jit` for a vector of 100 values of $\Omega_{\rm m}$, using `jax.vmap`.
'''),
]

# ---------------------------------------------------------------- 01
n01 = [
("md", r'''
# 1. Cosmology and background

This notebook follows Secs. 2.1 and 2.2 of Paper I.
It builds the density budget of the cosmology container, the neutrino terms of Table 1, the expansion $E(z)$, the distances and the sound horizon, and their derivatives with respect to the ten parameters of Eq. (1).
It ends on the figure of the distance gradients.
Everything here depends on the cosmological parameters alone and costs milliseconds.
'''),
("code", SETUP + r'''
from ggah_mod.cosmology import (PLANCK18, angular_diameter_distance, comoving_distance,
                                comoving_volume_element, distance_modulus, hubble_e,
                                luminosity_distance, nu_density_shape, r_drag, z_drag)
from ggah_mod.cosmology.background import transverse_distance
'''),
("md", r'''
## Parameters and the density budget

The comoving densities are fractions of the critical density today, $\rho_{\rm crit,0} = 2.77537\times10^{11}\,h^2{\rm M}_\odot\,{\rm Mpc}^{-3}$.
Every density enters one budget (Eq. 3),

$$\Omega_\gamma + \Omega_{\rm b} + \Omega_{\rm cdm} + \Omega_\nu^{\rm nr} + \Omega_\nu^{\rm r} + \Omega_k + \Omega_{\rm DE} = 1,$$

which fixes $\Omega_{\rm DE}$.
$\Omega_{\rm m}$ contains the neutrinos' rest mass, $\Omega_{\rm m} = \Omega_{\rm b} + \Omega_{\rm cdm} + \Omega_\nu^{\rm nr}$, and $\Omega_{\rm cb} = \Omega_{\rm b} + \Omega_{\rm cdm}$ is the cold part the halo layer uses.
The photons are a blackbody at $T_{\rm CMB}$, $\Omega_\gamma = 2.47298\times10^{-5}\,h^{-2}\,(T_{\rm CMB}/2.7255\,{\rm K})^4$ (Eq. 5).
'''),
("code", r'''
c = PLANCK18
terms = {"Omega_gamma": c.Omega_gamma, "Omega_b": c.Omega_b, "Omega_cdm": c.Omega_cdm,
         "Omega_nu (rest mass)": c.Omega_nu, "Omega_nu_r": c.Omega_nu_r,
         "Omega_k": c.Omega_k, "Omega_DE": c.Omega_de}
for name, value in terms.items():
    print(f"{name:22s} {float(value):.8e}")
print(f"{'sum - 1':22s} {float(sum(terms.values())) - 1:.1e}")
print(f"Omega_m - Omega_cb = {float(c.Omega_m - c.Omega_cb):.5e}  (the neutrino rest mass)")
print(f"Omega_gamma h^2 = {float(c.Omega_gamma * c.h**2):.5e}")
'''),
("md", r'''
## Neutrinos

The rest mass is linear in the mass sum, $\Omega_\nu^{\rm nr} = \sum m_\nu/(93.14\,{\rm eV}\,h^2)$ (Eq. 11), so it does not depend on how the sum is divided among the three states.
The relativistic part $\Omega_\nu^{\rm r}$ is the kinetic energy the states still carry today, plus a massless remainder $\Omega_{\rm ur}$ that holds the rest of $N_{\rm eff} = 3.044$.
Each state's energy relative to massless is the Fermi–Dirac integral $F(y)$ of Eq. (9), with $y = m/k_{\rm B}T_{\rm ncdm}$; it is 1 for a massless state and tends to $\kappa y$, $\kappa = 0.31732$, for a cold one.

We compare the degenerate ordering of `PLANCK18` with the normal one at the same sum, as in Table 1 of the paper.
'''),
("code", r'''
# Table 1 of Paper I: the neutrino terms at sum m_nu = 0.06 eV, h = 0.6736
paper = {"degenerate": dict(rest=1.41970e-3, kin=6.49e-7, ur=5.44e-8, rel=7.04e-7, total=1.42040e-3),
         "normal":     dict(rest=1.41970e-3, kin=4.48e-6, ur=5.44e-8, rel=4.54e-6, total=1.42423e-3)}

print(f"{'':11s}{'masses [eV]':>27s}{'rest mass':>12s}{'kinetic':>10s}{'remainder':>11s}"
      f"{'relativ.':>10s}{'total':>12s}")
for ordering in ("degenerate", "normal"):
    cc = PLANCK18.replace(nu_hierarchy=ordering)
    row = dict(rest=float(cc.Omega_nu), kin=float(cc.Omega_nu_r - cc.Omega_ur),
               ur=float(cc.Omega_ur), rel=float(cc.Omega_nu_r), total=float(cc.Omega_nu_today))
    masses = np.array2string(np.asarray(cc.nu_masses), precision=5)
    print(f"{ordering:11s}{masses:>27s}{row['rest']:12.5e}{row['kin']:10.2e}{row['ur']:11.2e}"
          f"{row['rel']:10.2e}{row['total']:12.5e}")
    for key, value in paper[ordering].items():          # check against the paper
        assert abs(row[key] / value - 1) < 5e-3, (ordering, key, row[key], value)
print("Table 1 reproduced")
'''),
("md", r'''
The ordering does not change the rest mass, nor the massless remainder.
It changes only the kinetic energy: the lightest state of the normal ordering, at $y = 5.6$, is still partly relativistic, so its kinetic energy is seven times the degenerate one.

The mean energy of the three states, $S_\nu(z) = \tfrac13\sum_j F\big(y_j/(1+z)\big)$ (Eq. 15), is 1 while they are relativistic and $\kappa\bar y/(1+z)$ once they are cold, which turns the neutrinos' $(1+z)^4$ into the $(1+z)^3$ of matter.
'''),
("code", r'''
z = np.logspace(-2, 5, 300)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for sum_mnu, ordering in ((0.06, "degenerate"), (0.06, "normal"), (0.3, "degenerate")):
    cc = PLANCK18.replace(sum_mnu=sum_mnu, nu_hierarchy=ordering)
    ax.loglog(1 + z, nu_density_shape(z, cc), label=f"{sum_mnu} eV, {ordering}")
ax.set(xlabel="$1+z$", ylabel=r"$S_\nu(z)$")
ax.legend();
'''),
("md", r'''
## Expansion and distances

The expansion rate with the complete budget reads (Eq. 13):

$$E^2(z) = \Omega_\gamma(1+z)^4 + \Omega_{\rm cb}(1+z)^3 + \big[(\Omega_\nu^{\rm ur}-\Omega_{\rm ur})S_\nu(z) + \Omega_{\rm ur}\big](1+z)^4 + \Omega_k(1+z)^2 + \Omega_{\rm DE}f_{\rm DE}(z),$$

with $f_{\rm DE}(z) = (1+z)^{3(1+w_0+w_a)}\exp[-3w_a z/(1+z)]$ for $w(a) = w_0 + w_a(1-a)$.
The comoving distance is its quadrature, in $h^{-1}$Mpc (Eq. 16),

$$\chi(z) = 2997.92458\int_0^z \frac{{\rm d}z'}{E(z')},$$

evaluated on 256 Gauss–Legendre nodes.
Curvature enters only through the transverse distance $f_K(\chi) = \chi\,s(x)$, $x = \Omega_k(\chi/D_H)^2$, $s(x) = \sinh\sqrt{x}/\sqrt{x}$ (Eq. 17).
$s$ is an entire function, so the flat case is an interior point and $f_K = \chi$ exactly at $\Omega_k = 0$.
The other distances follow from $f_K$ (Eq. 18).
'''),
("code", r'''
z = np.linspace(0.01, 5.0, 200)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.0, 3.0))
a0.plot(z, hubble_e(z, PLANCK18))
a0.set(xlabel="$z$", ylabel="$E(z)$")
for f, label in ((comoving_distance, r"$\chi$"), (angular_diameter_distance, r"$D_{\rm A}$"),
                 (luminosity_distance, r"$D_{\rm L}$")):
    a1.semilogy(z, f(z, PLANCK18), label=label)
a1.set(xlabel="$z$", ylabel=r"distance [$h^{-1}$Mpc]")
a1.legend()
fig.tight_layout()

chi = comoving_distance(z, PLANCK18)
print("flat: f_K == chi to the last bit:", bool(np.all(transverse_distance(chi, PLANCK18) == chi)))
print(f"chi(z=1) = {float(comoving_distance(1.0, PLANCK18)[0]):.3f} Mpc/h,  "
      f"mu(z=1) = {float(distance_modulus(1.0, PLANCK18)[0]):.4f} mag")
'''),
("md", r'''
### Exercise: the quadrature

Integrate $1/E$ yourself with `scipy.integrate.quad` at $z = 1$ and $z = 1100$ and compare with `comoving_distance`.
The paper quotes $10^{-15}$ up to $z = 10$ and $5\times10^{-13}$ at $z = 1100$.
'''),
("code", r'''
from scipy.integrate import quad

for zz in (1.0, 10.0, 1100.0):
    ref = 2997.92458 * quad(lambda x: 1.0 / float(hubble_e(x, PLANCK18)), 0.0, zz,
                            epsabs=0, epsrel=1e-13, limit=500)[0]
    print(f"z = {zz:6g}: chi / quad - 1 = {float(comoving_distance(zz, PLANCK18)[0]) / ref - 1:+.1e}")
'''),
("md", r'''
## The sound horizon

The BAO scale is the comoving sound horizon at the drag epoch, $r_{\rm d} = \int_{z_{\rm d}}^\infty c_s\,{\rm d}z/H$ (Eq. 19), integrated over the same $E(z)$ as every distance here.
The drag redshift needs a recombination history, which this package does not carry; it is a cubic polynomial fitted to CLASS 3.3.4 with HyRec-2 on 1915 cosmologies across the `emu_pk` box, accurate to 0.017 in $z_{\rm d}$.
Outside that box the fit refuses; `in_domain(cosmo)` returns the box as a boolean for a sampler's prior.
'''),
("code", r'''
from ggah_mod.cosmology.domain import in_domain

print(f"z_d = {float(z_drag(PLANCK18)):.3f},  r_d = {float(r_drag(PLANCK18)):.4f} Mpc/h "
      f"= {float(r_drag(PLANCK18)) / PLANCK18.h:.3f} Mpc")
print("PLANCK18 in the calibrated box:", bool(in_domain(PLANCK18)))
'''),
("md", r'''
## Derivatives

The derivatives of $\ln E$ are known in closed form (Eq. 22).
For instance, substituting the closure into $E^2$ gives

$$\frac{\partial\ln E}{\partial\ln\Omega_{\rm m}} = \frac{\Omega_{\rm m}(x^3-f)}{2E^2},\qquad
\frac{\partial\ln E}{\partial w_0} = \frac{3\,\Omega_{\rm DE} f\ln x}{2E^2},$$

with $x = 1+z$ and $f = f_{\rm DE}(z)$.
`ggah_mod` does not use these formulas: `jax.jacfwd` differentiates the code of $E(z)$ itself.
We check that the two agree.
'''),
("code", r'''
z = jnp.linspace(0.1, 3.0, 30)


def lnE(value, name):
    return jnp.log(hubble_e(z, PLANCK18.replace(**{name: value})))


c = PLANCK18
x, E2 = 1 + z, hubble_e(z, c) ** 2
f = x ** (3 * (1 + c.w0 + c.wa)) * jnp.exp(-3 * c.wa * z / x)

auto_om = jax.jacfwd(lnE)(c.Omega_m, "Omega_m") * c.Omega_m      # logarithmic derivative
auto_w0 = jax.jacfwd(lnE)(c.w0, "w0")
closed_om = c.Omega_m * (x ** 3 - f) / (2 * E2)
closed_w0 = 3 * c.Omega_de * f * jnp.log(x) / (2 * E2)
print(f"Omega_m: max |autodiff - closed form| = {float(jnp.max(jnp.abs(auto_om - closed_om))):.1e}")
print(f"w0     : max |autodiff - closed form| = {float(jnp.max(jnp.abs(auto_w0 - closed_w0))):.1e}")
'''),
("md", r'''
The distance's derivative is the integral of the one of $E$, with a minus sign (Eq. 23), so every direction the background sees moves $E$ and $\chi$ with opposite signs.

We now differentiate $\ln E$ and $\ln\chi$ with respect to all ten parameters.
A positive scale ($\Omega_{\rm m}$, $\Omega_{\rm b}$, $h$, $\sum m_\nu$, $T_{\rm CMB}$) gets a logarithmic derivative; a location, or a parameter that is already a logarithm, gets an absolute one.
One `jax.jacfwd` per parameter is cheaper than a reverse pass, because the output is a whole redshift grid and the input a scalar.
'''),
("code", r'''
z = np.geomspace(0.01, 5.0, 240)
zj = jnp.asarray(z)
ge, gc = {}, {}
for p in T.DPARAMS:
    v0 = float(getattr(PLANCK18, p))
    scale = v0 if p in T.DLOG else 1.0
    ge[p] = np.asarray(jax.jacfwd(lambda v: jnp.log(hubble_e(zj, PLANCK18.replace(**{p: v}))))(v0)) * scale
    gc[p] = np.asarray(jax.jacfwd(lambda v: jnp.log(comoving_distance(zj, PLANCK18.replace(**{p: v}))))(v0)) * scale

print(f"{'':10s}{'dlnE at z=1':>14s}{'dlnchi at z=1':>15s}")
for p in T.DPARAMS:
    print(f"{p:10s}{np.interp(1.0, z, ge[p]):+14.4g}{np.interp(1.0, z, gc[p]):+15.4g}")
'''),
("md", r'''
## The figure

Fig. 2 of the paper draws these twenty curves.
The vertical axis is `symlog`: linear inside the shaded band, where the derivative is below $10^{-7}$ in magnitude, and logarithmic outside it, because the curves span eight decades.
'''),
("code", r'''
LIN = 1e-7
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
for axis, g, sym, title in ((ax[0], ge, "E", "(a) the expansion"),
                            (ax[1], gc, r"\chi", "(b) the distance")):
    axis.axhspan(-LIN, LIN, color="0.90", zorder=0, lw=0)
    exact = [p for p in T.DPARAMS if np.all(g[p] == 0.0)]
    for p in T.DPARAMS:
        if p not in exact:
            colour, ls = T.DSTYLE[p]
            axis.plot(z, g[p], color=colour, ls=ls, lw=1.3, label=T.TEX[p])
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=LIN)
    axis.axhline(0.0, color="k", lw=0.8)
    axis.annotate(", ".join(T.TEX[p] for p in exact) + ":  exactly $0$",
                  xy=(0.98, 0.0), xycoords=("axes fraction", "data"), xytext=(0, 2.5),
                  textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color="0.25")
    axis.set_ylim(-1.0, 1.0)
    axis.set_yticks([-1e-1, -1e-3, -1e-5, 0.0, 1e-5, 1e-3, 1e-1])
    axis.set(xlabel="$z$", title=title, ylabel=rf"$\partial\ln {sym}/\partial\theta$")
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6.6, ncol=len(labels), loc="lower center",
           bbox_to_anchor=(0.5, 0.0), frameon=False, handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
T.save_figure(fig, "fig_distance_gradients")
'''),
("md", r'''
## Check against the paper

The three exact zeros are the structural statement of the figure: the background sees the total matter density and not its split, and it never sees the primordial spectrum.
The numbers below are those quoted in Sec. 2.2.
'''),
("code", r'''
for g in (ge, gc):
    assert {p for p in T.DPARAMS if np.all(g[p] == 0.0)} == {"Omega_b", "n_s", "ln10A_s"}
for p in T.DPARAMS:                     # E and chi move with opposite signs
    if np.any(ge[p] != 0.0):
        assert np.all(np.sign(ge[p]) == -np.sign(gc[p])), p

at = lambda g, p, zz: float(np.interp(zz, z, g[p]))
quoted = [("dlnE/dOmega_k, z=1", at(ge, "Omega_k", 1.0), 0.47),
          ("dlnchi/dOmega_k, z=1", at(gc, "Omega_k", 1.0), -0.29),
          ("dlnE/dln sum_mnu, z=1", at(ge, "sum_mnu", 1.0), -3.1e-6),
          ("dlnchi/dln sum_mnu, z=1", at(gc, "sum_mnu", 1.0), 1.2e-6),
          ("dlnE/dln Omega_m, z=5", at(ge, "Omega_m", 5.0), 0.49)]
print(f"{'':26s}{'here':>11s}{'paper':>10s}")
for label, here, paper in quoted:
    print(f"{label:26s}{here:+11.3g}{paper:+10.2g}")
    assert abs(here / paper - 1) < 0.05, label
print("Sec. 2.2 reproduced")
'''),
("md", r'''
## Exercises

1. Redo the figure at $\Omega_k = 0.05$. Which curves change, and why does the $\Omega_k$ derivative of $\chi$ grow faster with redshift than that of $E$?
2. Swap `nu_hierarchy` to `"normal"` and plot the ratio of the $\sum m_\nu$ derivatives of $E$ in the two orderings. Where does the splitting matter?
3. Differentiate $r_{\rm d}$ with respect to $\Omega_{\rm b}$, $h$ and $\Omega_{\rm m}$ with `jax.grad`. Which combination of them does the BAO ratio $\chi(z)/r_{\rm d}$ constrain? (Notebook 07 answers it with a Fisher matrix.)

Next: notebook 02, the linear power spectrum.
'''),
]

nbf.write(nb(n00), os.path.join(OUT, "00_getting_started.ipynb"))
nbf.write(nb(n01), os.path.join(OUT, "01_cosmology_background.ipynb"))
print("written")
