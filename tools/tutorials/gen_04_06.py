"""Write tutorials/04, 05 and 06 (sources only; executed separately)."""
import sys

from _nb import SETUP, write

HALO_SETUP = SETUP + r'''
import ggah_mod.halos as H
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.halos.field import make_field

pk = make_pk("emu_pk")
c = PLANCK18
M = np.logspace(10.0, 16.0, 80 if FAST else 160)     # h^-1 Msun
K = np.logspace(-4.0, 2.0, 512)                      # h Mpc^-1
mj, kj = jnp.asarray(M), jnp.asarray(K)
near = lambda arr, value: int(np.argmin(np.abs(np.asarray(arr) - value)))
'''

FIELD = r'''
#: The four model choices of Sec. 3: the defaults of the DIFFERENTIABLE backend.
CHOICE = dict(mdef="200m", hmf_model="tinker08", bias_model="tinker10", cm_model="bhattacharya13")


def field(cosmo, **kw):
    """The halo field of Eq. (36) at ``cosmo`` on this notebook's grids."""
    return make_field(cosmo, DIFFERENTIABLE, pk, z=0.0, m=M, k=K, **{**CHOICE, **kw})


def dln(read):
    """``{param: d ln read(field) / d theta}`` by forward autodiff of the whole field."""
    out = {}
    for p in T.DPARAMS:
        v0 = float(getattr(c, p))
        scale = v0 if p in T.DLOG else 1.0
        out[p] = np.asarray(jax.jacfwd(lambda v, p=p: jnp.log(read(field(c.replace(**{p: v})))))(v0)) * scale
    return out


#: The directions the left-hand panels move, either side of the fiducial.
VARIANTS = (("Omega_m", 0.28), ("Omega_m", 0.34), ("h", 0.65), ("h", 0.70),
            ("Omega_k", -0.05), ("Omega_k", 0.05))


def variant_lines(axis, of, x):
    for p, v in VARIANTS:
        ls = (0, (5, 1.6)) if v < float(getattr(c, p)) else (0, (1, 1.4))
        axis.plot(x, of(c.replace(**{p: v})), color=T.DSTYLE[p][0], ls=ls, lw=1.1,
                  label=T.TEX[p][:-1] + rf" = {v:g}$")


def gradient_panel(axis, x, g, xlabel, ylabel, title, lin=1e-2):
    axis.grid(False)
    axis.axhspan(-lin, lin, color="0.90", zorder=0)
    for p in T.DPARAMS:
        if not np.all(g[p] == 0.0):
            colour, ls = T.DSTYLE[p]
            axis.plot(x, g[p], color=colour, ls=ls, lw=1.3, label=T.TEX[p])
    axis.axhline(0.0, color="k", lw=0.8)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=lin)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.set_xlim(float(np.min(x)), float(np.max(x)))


def mass_axis_on_sigma(axis, sigma, masses=(1e10, 1e12, 1e14, 1e16)):
    """Masses along the top of a panel drawn against sigma, at the fiducial."""
    top = axis.twiny()
    top.grid(False)
    top.set_xscale("log")
    top.set_xlim(axis.get_xlim())
    top.set_xticks([float(np.exp(np.interp(np.log(mm), np.log(M), np.log(sigma)))) for mm in masses])
    top.set_xticklabels([rf"$10^{{{int(np.log10(mm))}}}$" for mm in masses])
    top.set_xlabel(r"$M\ [h^{-1}M_\odot]$ at the fiducial")


def bottom_legend(fig, axis):
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.6, ncol=10, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               frameon=False, handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.055, 1, 1))


fid = field(c)
sigma = np.asarray(fid.sigma)
'''

# ------------------------------------------------------------------ 04
n04 = [
("md", r'''
# 4. Variance and peak height

This notebook follows Sec. 3.1 of Paper I, the first rung of the halo layer.
Every fit of the layer — sixteen multiplicity functions, six bias fits, the peak-height concentrations — reads the cosmology through one curve, $\sigma(M)$.
We build it, its slope and the peak height, and differentiate it with respect to the ten parameters, ending on the figure of the variance.
'''),
("code", HALO_SETUP),
("md", r'''
## From a radius to a mass

The variance of notebook 03 is specialised to a mass through the Lagrangian radius of the sphere that contained it before collapse (Eq. 31):

$$\sigma^2(M) = \frac{1}{2\pi^2}\int P_{\rm cb}(k)\,W^2\big(kR(M)\big)k^2\,{\rm d}k,\qquad R(M) = \left(\frac{3M}{4\pi\bar\rho_{\rm cb}}\right)^{1/3}.$$

Haloes form out of the CDM+baryon field, so both the spectrum and the density are the cold ones.
The halo boundaries of a spherical-overdensity mass definition use the total matter density instead; the two densities differ by 0.46 per cent at $\sum m_\nu = 0.06$ eV.
'''),
("code", r'''
print(f"rho_cb = {float(c.rho_cold):.5e},  rho_m = {float(c.rho_matter):.5e} (h^-1 Msun)/(h^-1 Mpc)^3, "
      f"ratio - 1 = {float(c.rho_matter / c.rho_cold) - 1:.2%}")

pk_cb = pk.pk_cb(kj, 0.0, c)
H.check_k_support(M, K, c.rho_cold)            # refuses a k grid too narrow for these masses
sig = np.asarray(H.sigma_of_mass(mj, kj, pk_cb, c.rho_cold))
R = np.asarray(H.lagrangian_radius(mj, c.rho_cold))
nu = H.DELTA_C / sig
for mm in (1e10, 1e12, 1e14, 1e16):
    i = near(M, mm)
    print(f"M = {M[i]:8.2e}  R = {R[i]:7.3f} Mpc/h  sigma = {sig[i]:6.4f}  nu = {nu[i]:6.3f}")
'''),
("md", r'''
## The slope and the peak height

The mass function needs ${\rm d}\ln\sigma/{\rm d}\ln M = \tfrac13\,{\rm d}\ln\sigma/{\rm d}\ln R$ (Eq. 32), which `dln_sigma_dln_mass` takes by automatic differentiation rather than by differencing $\sigma(M)$.
The peak height is $\nu = \delta_{\rm c}/\sigma$ with $\delta_{\rm c} = 1.686$ (Eq. 33); $\nu = 1$ defines the characteristic mass $M_\star$.
'''),
("code", r'''
dlns = np.asarray(H.dln_sigma_dln_mass(mj, kj, pk_cb, c.rho_cold))
fd = np.gradient(np.log(sig), np.log(M))
print(f"max |autodiff - finite difference| of dln sigma/dln M = {np.max(np.abs(dlns - fd)[2:-2]):.1e}")
M_star = float(np.interp(H.DELTA_C, sig[::-1], M[::-1]))
print(f"M_star = {M_star:.2e} h^-1 Msun   (paper: 4.6e12)")

from ggah_mod.cosmology import sigma8, sigma_tophat
s8_cold = float(sigma_tophat(pk_cb, kj, 8.0))
s8_tot = float(sigma8(pk.pk(kj, 0.0, c), kj))
print(f"at R = 8 Mpc/h: sigma on P_cb = {s8_cold:.3f}, sigma_8 on P_m = {s8_tot:.3f}   (paper: 0.817, 0.813)")
'''),
("md", r'''
## Derivatives

The mass is held fixed, so the cosmology moves the radius $R(M)$ through $\bar\rho_{\rm cb}$ as well as the spectrum.
Both are inside the function we differentiate.
'''),
("code", r'''
g = {}
for p in T.DPARAMS:
    v0 = float(getattr(c, p))
    scale = v0 if p in T.DLOG else 1.0

    def lnsig(v, p=p):
        cc = c.replace(**{p: v})
        return jnp.log(H.sigma_of_mass(mj, kj, pk.pk_cb(kj, 0.0, cc), cc.rho_cold))

    g[p] = np.asarray(jax.jacfwd(lnsig)(v0)) * scale

lo, hi = near(M, 1e12), near(M, 1e15)
print(f"{'':10s}{'1e12':>10s}{'1e15':>10s}")
for p in T.DPARAMS:
    print(f"{p:10s}{g[p][lo]:+10.4g}{g[p][hi]:+10.4g}")
'''),
("md", r'''
## The figure

Fig. 5 of the paper.
Panel (a) is $\sigma(M)$ at the fiducial in black, with $\Omega_{\rm m}$ and $\Omega_k$ moved either side; the top axis is the Lagrangian radius and the right axis the peak height.
Panel (b) is $\partial\ln\sigma/\partial\theta$ for the ten parameters.
'''),
("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1))
ax[0].grid(False)
ax[1].grid(False)
ax[0].loglog(M, sig, "k", lw=1.6, zorder=3, label="fiducial")
for p, v, ls in (("Omega_m", 0.28, (0, (5, 1.6))), ("Omega_m", 0.34, (0, (1, 1.4))),
                 ("Omega_k", -0.05, (0, (5, 1.6))), ("Omega_k", 0.05, (0, (1, 1.4)))):
    cv = c.replace(**{p: v})
    ax[0].loglog(M, H.sigma_of_mass(mj, kj, pk.pk_cb(kj, 0.0, cv), cv.rho_cold),
                 color=T.DSTYLE[p][0], ls=ls, lw=1.3, label=T.TEX[p][:-1] + rf" = {v:g}$")
ax[0].legend(fontsize=6.0, loc="lower left", frameon=False, handlelength=2.2, labelspacing=0.35)
rho = float(c.rho_cold)
ax[0].set_xlim(M[0], M[-1])
sec = ax[0].secondary_xaxis("top", functions=(lambda m: (3 * np.maximum(m, 1e-300) / (4 * np.pi * rho)) ** (1 / 3),
                                              lambda r: 4 / 3 * np.pi * np.maximum(r, 1e-300) ** 3 * rho))
sec.set_xlabel(r"$R_{\rm Lagrange}(M)\ [h^{-1}\mathrm{Mpc}]$")
sec.grid(False)
nax = ax[0].secondary_yaxis("right", functions=(lambda s: H.DELTA_C / np.maximum(s, 1e-300),
                                                lambda n: H.DELTA_C / np.maximum(n, 1e-300)))
nax.set_ylabel(r"$\nu = \delta_{\rm c}/\sigma$")
nax.grid(False)
ax[0].set(xlabel=r"$M\ [h^{-1}M_\odot]$", ylabel=r"$\sigma(M)$", title="(a) the variance")
ax[0].set_yticks([0.3, 0.5, 1.0, 2.0, 3.0, 4.0])
ax[0].set_yticks([], minor=True)
ax[0].set_yticklabels(["0.3", "0.5", "1", "2", "3", "4"])
nax.set_yticks([0.5, 1.0, 2.0, 3.0, 5.0])
nax.set_yticklabels(["0.5", "1", "2", "3", "5"])

LIN = 1e-2
ax[1].axhspan(-LIN, LIN, color="0.90", zorder=0)
for p in T.DPARAMS:
    colour, ls = T.DSTYLE[p]
    ax[1].plot(M, g[p], color=colour, ls=ls, lw=1.3, label=T.TEX[p])
ax[1].axhline(0.0, color="k", lw=0.8)
ax[1].set_xscale("log")
ax[1].set_yscale("symlog", linthresh=LIN)
ax[1].set(xlabel=r"$M\ [h^{-1}M_\odot]$", ylabel=r"$\partial\ln\sigma/\partial\theta$",
          title="(b) what it responds to", xlim=(M[0], M[-1]))
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6.6, ncol=10, loc="lower center", bbox_to_anchor=(0.5, -0.015),
           frameon=False, handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.055, 1, 1))
T.save_figure(fig, "fig_variance")
'''),
("md", r'''
## Exercises

1. Recompute $\sigma(M)$ on $P_{\rm m}$ and $\bar\rho_{\rm m}$ instead of the cold field, at $\sum m_\nu = 0.06$ and 0.3 eV. How large is the difference at $10^{14}\,h^{-1}{\rm M}_\odot$, and how would a fit to cluster counts absorb it?
2. Shrink the $k$ grid to $10^{-2} < k < 10\,h\,{\rm Mpc}^{-1}$ and call `H.check_k_support`. Which masses does it refuse, and why?
3. Plot $M_\star(z)$ for $0 < z < 3$. How fast does the characteristic mass fall?

Next: notebook 05, the abundance and the bias of haloes.
'''),
]

# ------------------------------------------------------------------ 05
n05 = [
("md", r'''
# 5. Abundance and bias of haloes

This notebook follows Secs. 3.2 and 3.3 of Paper I.
It assembles the mass function from a multiplicity function, compares the sixteen fits the package provides, differentiates the default `tinker08` with respect to the ten parameters, and checks the peak-background split of the bias fits (Table 4).
It ends on the figure of the abundance.
'''),
("code", HALO_SETUP + FIELD),
("md", r'''
## The mass function

The mass function is a multiplicity function $f(\sigma)$ times the Jacobian from $\sigma$ to mass (Eq. 34):

$$\frac{{\rm d}n}{{\rm d}M} = f(\sigma)\,\frac{\bar\rho_{\rm cb}}{M^2}\left|\frac{{\rm d}\ln\sigma}{{\rm d}\ln M}\right|.$$

The default, `tinker08`, is (Eq. 35)

$$f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]e^{-c/\sigma^2},$$

with parameters interpolated in $\log\Delta$ between the nine overdensities Tinker et al. tabulated.
The halo field holds the result; we rebuild it by hand from its own $\sigma$ and slope.
'''),
("code", r'''
by_hand = H.make_multiplicity("tinker08")(fid.sigma, 0.0) * c.rho_cold / M**2 * jnp.abs(fid.dlns)
print(f"max |by hand / field - 1| = {float(jnp.max(jnp.abs(by_hand / fid.dndm - 1))):.1e}")

fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.loglog(M, M**2 / c.rho_cold * np.asarray(fid.dndm))
ax.set(xlabel=r"$M\ [h^{-1}M_\odot]$", ylabel=r"$(M^2/\bar\rho_{\rm cb})\,{\rm d}n/{\rm d}M$");
'''),
("md", r'''
## Sixteen fits

The package re-implements sixteen multiplicity functions in JAX, following `colossus`.
`CALIBRATION` records, for each, the simulation suite, the halo finder, the overdensity and the redshift range it was fitted on.
At fixed $\sigma$ they disagree by tens of per cent, which is the size of the systematic a choice of fit carries.
'''),
("code", r'''
import inspect

lo, hi = near(M, 1e12), near(M, 1e15)
vals = {}
for name, fn in H.MULTIPLICITY.items():
    if name.startswith("tinker08_csst"):
        continue                                   # optional emulator, needs emu_hmf
    kw = {"cosmo": c} if "cosmo" in inspect.signature(fn).parameters else {}
    vals[name] = np.asarray(fn(fid.sigma, 0.0, **kw))
print(f"{len(vals)} fits;  f(sigma) at 1e12 and 1e15:")
for name, v in vals.items():
    print(f"  {name:18s} {v[lo]:.4f}  {v[hi]:.5f}")
v12 = np.array([v[lo] for v in vals.values()])
v15 = np.array([v[hi] for v in vals.values()])
print(f"spread: {np.ptp(v12) / np.median(v12):.0%} of the median at 1e12, {np.ptp(v15) / np.median(v15):.0%} at 1e15")
'''),
("md", r'''
## Derivatives

A multiplicity function is a function of $\sigma$ alone, so at fixed $\sigma$ it has no cosmological derivative.
What the cosmology moves is $\sigma(M)$, so we differentiate at fixed mass, through the whole field.
'''),
("code", r'''
fit = H.make_multiplicity("tinker08")
g = dln(lambda f: fit(f.sigma, 0.0))
print(f"{'':10s}{'1e12':>10s}{'1e15':>10s}")
for p in T.DPARAMS:
    print(f"{p:10s}{g[p][lo]:+10.4g}{g[p][hi]:+10.4g}")
'''),
("md", r'''
## The figure

Fig. 6 of the paper, against $\sigma$ with the mass the fiducial assigns it on the top axis.
Panel (a) is `tinker08` at the fiducial in black, with $\Omega_{\rm m}$, $h$ and $\Omega_k$ moved either side, at fixed mass.
Panel (b) is $\partial\ln f/\partial\theta$ at fixed mass.
'''),
("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1))
ax[0].grid(False)
ax[0].loglog(sigma, np.asarray(fit(fid.sigma, 0.0)), "k", lw=1.8, zorder=3, label="fiducial")
variant_lines(ax[0], lambda cc: np.asarray(fit(field(cc).sigma, 0.0)), sigma)
ax[0].set(xlabel=r"$\sigma$", ylabel=r"$f(\sigma)$", title="(a) tinker08", xlim=(sigma.min(), sigma.max()))
ax[0].legend(fontsize=5.6, ncol=2, loc="lower right", frameon=False, handlelength=2.2, labelspacing=0.3, columnspacing=1.0)
mass_axis_on_sigma(ax[0], sigma)
gradient_panel(ax[1], sigma, g, r"$\sigma$", r"$\partial\ln f/\partial\theta$", "(b) its cosmological response")
mass_axis_on_sigma(ax[1], sigma)
bottom_legend(fig, ax[1])
T.save_figure(fig, "fig_abundance")
'''),
("md", r'''
## Large-scale bias

Six bias fits are provided as functions of $\nu = \delta_{\rm c}/\sigma$ (Eq. 37).
A mass function and a bias fit calibrated together are a pair; `MATCHED_BIAS` records them.
If the two are consistent, the bias-weighted mass function recovers the mean (Eq. 38),

$$\int b(M)\,\frac{M}{\bar\rho_{\rm cb}}\frac{{\rm d}n}{{\rm d}M}\,{\rm d}M \to 1,$$

and the same integral without $b$ is the fraction of mass in haloes.
Over a finite mass range both fall short of one, so Table 4 also gives their ratio, the mean bias of the mass counted.
We recompute the table over $10^8$–$10^{16}\,h^{-1}{\rm M}_\odot$ with `emu_pk`.
The mass definition must be converted to an overdensity with respect to the mean density before it is passed to `tinker08`: $200{\rm c}$ is $\Delta\simeq 640$ mean at $z = 0$.
'''),
("code", r'''
print("matched pairs:", H.MATCHED_BIAS)
mm = np.logspace(8.0, 16.0, 1024)
s_m = H.sigma_of_mass(mm, kj, pk_cb := pk.pk_cb(kj, 0.0, c), c.rho_cold)
dl_m = H.dln_sigma_dln_mass(mm, kj, pk_cb, c.rho_cold)

paper = {("press74", "press74", "200m"): (0.900, 0.764), ("tinker08", "tinker10", "200c"): (0.604, 0.508),
         ("tinker08", "tinker10", "200m"): (0.767, 0.623), ("despali16", "sheth99", "vir"): (0.669, 0.559),
         ("comparat17", "comparat17", "vir"): (0.656, 0.540), ("tinker08", "sheth01", "200c"): (0.593, 0.508)}
print(f"{'dn/dM':11s}{'b':11s}{'mdef':6s}{'int b df':>10s}{'int df':>9s}{'ratio':>8s}   paper")
for (hmf, bias, mdef), (pb, pf) in paper.items():
    delta = float(H.MassDef.from_string(mdef).delta_mean(0.0, c))
    kw = {"delta": delta} if hmf == "tinker08" else {}
    if hmf == "despali16":
        kw = {"delta_ratio": delta / float(H.MassDef("vir").delta_mean(0.0, c))}
    n_m = H.dndm(mm, s_m, dl_m, c.rho_cold, model=hmf, z=0.0, **kw)
    b_m = H.make_bias(bias)(s_m, delta)
    bdf = float(H.mass_weighted_bias(mm, n_m, b_m, c.rho_cold))
    df = float(H.mass_fraction(mm, n_m, c.rho_cold))
    print(f"{hmf:11s}{bias:11s}{mdef:6s}{bdf:10.3f}{df:9.3f}{bdf / df:8.3f}   {pb:.3f} {pf:.3f}")
    assert abs(bdf / pb - 1) < 0.02 and abs(df / pf - 1) < 0.02
'''),
("md", r'''
Evaluating `tinker08`–`tinker10` at $200{\rm c}$ rather than $200{\rm m}$ moves the bias integral from 0.767 to 0.604, a 21 per cent effect; pairing `tinker08` with the wrong bias fit at a fixed definition moves it by 2 per cent.
Choosing the mass definition wrongly costs an order of magnitude more than choosing the partner wrongly.

## Exercises

1. Plot ${\rm d}n/{\rm d}M$ for all sixteen fits at $z = 0$ and $z = 1$, as a ratio to `tinker08`. Where do they agree best, and why there?
2. Differentiate the number of haloes above $10^{14}\,h^{-1}{\rm M}_\odot$ per $(h^{-1}{\rm Mpc})^3$ with respect to $\Omega_{\rm m}$ and $\ln(10^{10}A_{\rm s})$. Which combination do cluster counts constrain? (Notebook 09 answers it.)
3. Extend the integrals of Table 4 to $\nu = 10^{-8}$ for `press74` and for `tinker08`–`tinker10`. Which one closes, and why can the other not?

Next: notebook 06, concentrations and profiles.
'''),
]

# ------------------------------------------------------------------ 06
n06 = [
("md", r'''
# 6. Concentration, profiles and the halo field

This notebook follows Secs. 3.4–3.7 of Paper I.
It shows which concentration relations a $200{\rm m}$ field admits, the Fourier-space NFW profile and its cosmological response, the beyond-linear halo bias, and the `HaloField` that holds every rung of the layer.
It ends on the figures of the concentration and the profile.
'''),
("code", HALO_SETUP + FIELD),
("md", r'''
## Concentration

Six relations are provided, in two families (Eq. 39): empirical power laws $c(M, z, \Delta)$ (`duffy08`, `dutton14`, `klypin16`), which cannot see a cosmology, and peak-height relations $c(\sigma, \cdot)$ (`bhattacharya13`, `diemer19`, `seppi21`), which see it through $\sigma(M)$.
`CM_CALIBRATION` records the simulation, the mass definitions and the redshift range of each.
We ask for each of the six at $200{\rm m}$; the ones calibrated at another mass definition are refused rather than converted.
'''),
("code", r'''
cm = {}
for name in H.CONCENTRATION:
    try:
        cm[name] = np.asarray(field(c, cm_model=name).conc)
    except Exception as err:
        print(f"{name:15s} refused at 200m: {type(err).__name__}")
lo, hi = near(M, 1e12), near(M, 1e14)
for name, v in cm.items():
    print(f"{name:15s} c(1e12) = {v[lo]:6.3f}   c(1e14) = {v[hi]:6.3f}")
d12 = cm["duffy08"][lo] / cm["bhattacharya13"][lo] - 1
d14 = cm["duffy08"][hi] / cm["bhattacharya13"][hi] - 1
print(f"duffy08 / bhattacharya13 - 1: {d12:+.0%} at 1e12, {d14:+.0%} at 1e14   (paper: 11%, 3%)")

same = np.array_equal(np.asarray(field(c, cm_model="duffy08").conc),
                      np.asarray(field(c.replace(Omega_m=0.34), cm_model="duffy08").conc))
print("duffy08 identical at Omega_m = 0.3153 and 0.34:", same)
'''),
("md", r'''
`bhattacharya13` is $c = K D^{\alpha}\nu^{\beta}$ with $\beta = -0.29$ at $200{\rm m}$ in the form used here.
At $z = 0$ the growth factor is one and carries no derivative, so every cosmological direction of the concentration is $0.29$ times the corresponding direction of $\sigma(M)$.
'''),
("code", r'''
g_c = dln(lambda f: f.conc)
g_s = dln(lambda f: f.sigma)
worst = max(float(np.max(np.abs(g_c[p] - 0.29 * g_s[p]))) for p in T.DPARAMS)
print(f"d ln c / d theta = 0.29 d ln sigma / d theta to {worst:.1e}")
assert worst < 1e-8
'''),
("md", r'''
Fig. 7 of the paper.
'''),
("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1))
ax[0].grid(False)
ax[0].loglog(sigma, cm["bhattacharya13"], "k", lw=1.8, zorder=3, label="fiducial")
variant_lines(ax[0], lambda cc: np.asarray(field(cc).conc), sigma)
ax[0].set(xlabel=r"$\sigma$", ylabel=r"$c$", title=r"(a) bhattacharya13 at $200\mathrm{m}$",
          xlim=(sigma.min(), sigma.max()))
ax[0].set_yticks([3, 5, 10, 20])
ax[0].set_yticks([], minor=True)
ax[0].set_yticklabels(["3", "5", "10", "20"])
ax[0].legend(fontsize=5.6, ncol=2, loc="upper left", frameon=False, handlelength=2.2, labelspacing=0.3, columnspacing=1.0)
mass_axis_on_sigma(ax[0], sigma)
gradient_panel(ax[1], sigma, g_c, r"$\sigma$", r"$\partial\ln c/\partial\theta$", "(b) what bhattacharya13 responds to")
mass_axis_on_sigma(ax[1], sigma)
bottom_legend(fig, ax[1])
T.save_figure(fig, "fig_concentration")
'''),
("md", r'''
## Profiles

The halo model needs the normalised Fourier transform of the radial profile (Eq. 40),

$$u(k|M) = \frac{4\pi}{M}\int_0^{r_\Delta}\rho(r)\frac{\sin kr}{kr}r^2\,{\rm d}r,\qquad u(k\to0|M) = 1.$$

For NFW truncated at $r_\Delta = c\,r_{\rm s}$ the integral is analytic in sine and cosine integrals (Eq. 43), with $\kappa = kr_{\rm s}$ and $g(c) = \ln(1+c) - c/(1+c)$:

$$u(k) = \frac{1}{g(c)}\Big\{\sin\kappa\,[{\rm Si}((1+c)\kappa) - {\rm Si}(\kappa)] + \cos\kappa\,[{\rm Ci}((1+c)\kappa) - {\rm Ci}(\kappa)] - \frac{\sin c\kappa}{(1+c)\kappa}\Big\}.$$

Einasto, generalised NFW and others are provided too; `profile_uk_gl` is the Gauss–Legendre transform any new profile gets its $u(k)$ from.
We check the analytic NFW transform against it.
'''),
("code", r'''
i13 = near(M, 1e13)
rs, cc13 = fid.r_s[i13], fid.conc[i13]
kk = np.logspace(-1, 2, 50)
u_nfw = np.ravel(H.nfw_uk(jnp.asarray(kk), rs, cc13))
nfw_shape = lambda r: 1.0 / ((r / rs) * (1.0 + r / rs) ** 2)      # rho_s cancels in u(k)
u_gl = np.ravel(H.profile_uk_gl(jnp.asarray(kk), jnp.array([cc13 * rs]), nfw_shape, n_gl=400))
print(f"M = 1e13: max |Gauss-Legendre / analytic - 1| on 0.1 < k < 100 = {np.max(np.abs(u_gl / u_nfw - 1)):.1e}")
'''),
("code", r'''
KP = np.logspace(-3.0, 3.0, 160 if FAST else 320)
kp = jnp.asarray(KP)
MASSES = (1e12, 1e13, 1e14)
idx = [near(M, m) for m in MASSES]


def read_uk(f):
    return jnp.stack([jnp.ravel(H.nfw_uk(kp, f.r_s[i], f.conc[i])) for i in idx])


u = np.asarray(read_uk(fid))
for j, i in enumerate(idx):
    print(f"M = {MASSES[j]:.0e}: c = {float(fid.conc[i]):.2f}  r_s = {float(fid.r_s[i]):.4f} Mpc/h  "
          f"1/r_s = {1 / float(fid.r_s[i]):.1f} h/Mpc")
g_u = dln(read_uk)
g_mid = {p: v[1] for p, v in g_u.items()}
'''),
("md", r'''
Fig. 8 of the paper.
Panel (a) is $u(k|M)$ at three masses, each turning over at its own $1/r_{\rm s}$ (dotted); the top axis is $1/k$.
Panel (b) is $\partial\ln u/\partial\theta$ at $10^{13}\,h^{-1}{\rm M}_\odot$.
'''),
("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1))
ax[0].grid(False)
for j, i in enumerate(idx):
    col = "k" if j == 1 else f"C{j}"
    ax[0].loglog(KP, u[j], color=col, lw=1.8 if j == 1 else 1.3, zorder=3,
                 label=rf"$M = 10^{{{int(np.log10(MASSES[j]))}}}$,  $c = {float(fid.conc[i]):.1f}$")
    ax[0].axvline(1.0 / float(fid.r_s[i]), color=col, lw=0.7, ls=(0, (1, 2)))
ax[0].set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$", ylabel=r"$u(k|M)$", title="(a) the profile at three masses",
          xlim=(0.1, KP[-1]), ylim=(1e-4, 2.0))
ax[0].legend(fontsize=6.0, loc="lower left", frameon=False, handlelength=2.2, labelspacing=0.3)
top = ax[0].twiny()
top.grid(False)
top.set_xscale("log")
top.set_xlim(ax[0].get_xlim())
radii = [r for r in (10.0, 1.0, 0.1, 0.01, 0.001) if 0.1 <= 1 / r <= KP[-1]]
top.set_xticks([1 / r for r in radii])
top.set_xticklabels([f"{r:g}" for r in radii])
top.set_xlabel(r"$1/k\ [h^{-1}\mathrm{Mpc}]$")
gradient_panel(ax[1], KP, g_mid, r"$k\ [h\,\mathrm{Mpc}^{-1}]$", r"$\partial\ln u/\partial\theta$",
               r"(b) what it responds to, at $10^{13}$")
ax[1].set_xlim(0.1, KP[-1])
bottom_legend(fig, ax[1])
T.save_figure(fig, "fig_profile")
'''),
("md", r'''
## Beyond-linear halo bias

Haloes are not linearly biased tracers of the linear field at intermediate scales.
Mead & Verde (2021) tabulate the residual from the MultiDark simulation (Eq. 44),

$$P_{hh}(\nu_1,\nu_2,k) = b(\nu_1)b(\nu_2)P_{\rm lin}(k)\big[1 + \beta^{\rm NL}(k,\nu_1,\nu_2)\big].$$

All 35 public snapshots are shipped, and the one matching the user's cosmology and redshift is selected by the Angulo & White (2010) rescaling.
It is computed in this layer and consumed by the two-halo term of layer 4.
'''),
("code", r'''
kb = np.logspace(-2, 0, 60)
nus = np.array([1.0, 1.5, 2.0])
# the MultiDark snapshot is chosen by matching sigma(R) of this spectrum (Angulo & White 2010)
table = H.beyond_linear_bias.table_at(kb, k=kj, pk_cb=pk.pk_cb(kj, 0.0, c))
beta = np.asarray(H.beta_nl(kb, nus, nus, table=table))   # shape (Nk, N1, N2)
fig, ax = plt.subplots(figsize=(4.5, 3.0))
for j, n in enumerate(nus):
    ax.semilogx(kb, beta[:, j, j], label=rf"$\nu_1 = \nu_2 = {n:g}$")
ax.axhline(0.0, color="k", lw=0.8)
ax.set(xlabel=r"$k\ [h\,\mathrm{Mpc}^{-1}]$", ylabel=r"$\beta^{\rm NL}$")
ax.legend();
'''),
("md", r'''
## The halo field

`HaloField` holds every rung of the layer on shared grids, as differentiable leaves of one pytree (Eq. 36): $M$, $k$, $z$, $\sigma$, ${\rm d}\ln\sigma/{\rm d}\ln M$, $\nu$, ${\rm d}n/{\rm d}M$, $b$, $c$, $r_\Delta$, $r_{\rm s}$, $P_{\rm cb}$, $P_{\rm lin}$ and the cosmology.
The four model choices are static.
`make_field` takes its grids and choices from the backend, so a flavour is selected once.
Anything computed from a field is differentiable with respect to the cosmology, as every derivative in notebooks 04–06 was.
'''),
("code", r'''
default = make_field(c, DIFFERENTIABLE, pk)
print(type(default).__name__, "on", default.m.shape[0], "masses and", default.k.shape[0], "wavenumbers")
print("choices:", default.mdef, default.hmf_model, default.bias_model, default.cm_model)


def mean_bias_above(om, m_min=1e13):
    f = make_field(c.replace(Omega_m=om), DIFFERENTIABLE, pk)
    w = jnp.where(f.m > m_min, f.dndm * f.m, 0.0)
    return jnp.sum(w * f.bias) / jnp.sum(w)


print(f"mass-weighted bias above 1e13: {float(mean_bias_above(c.Omega_m)):.4f}; "
      f"d/dOmega_m = {float(jax.grad(mean_bias_above)(c.Omega_m)):+.4f}")
'''),
("md", r'''
## Exercises

1. Replace NFW by an Einasto profile with the same $r_{\rm s}$ and $c$ and plot the ratio of the two $u(k)$. At which $k$ does the choice of profile matter at $10^{14}\,h^{-1}{\rm M}_\odot$?
2. Build the field at $z = 1$ and compare the concentrations of `bhattacharya13` with those at $z = 0$. Through which term does redshift enter?
3. Compute $\partial\ln b/\partial\theta$ for `tinker10` at fixed mass, and check that it factors through $\sigma(M)$ as the multiplicity function did in notebook 05.

Next: notebook 07, cosmological constraints from distance measurements.
'''),
]

which = sys.argv[1:] or ["04", "05", "06"]
if "04" in which:
    write("04_halo_variance", n04)
if "05" in which:
    write("05_halo_abundance_bias", n05)
if "06" in which:
    write("06_concentration_profiles", n06)
