"""Write tutorials/02 and 03 (sources only; executed separately)."""
import sys

from _nb import SETUP, write

n02 = [
("md", r'''
# 2. The linear power spectrum

This notebook follows Sec. 2.3 of Paper I.
It shows the two spectra every backend returns, the box `emu_pk` was trained on, the agreement of the emulator with CLASS, and the derivatives of $\ln P$ with respect to the ten parameters, ending on the figure of the $P(k)$ gradients.
'''),
("code", SETUP + r'''
import time

from ggah_mod.cosmology import PLANCK18, make_pk
'''),
("md", r'''
## Two backends, two spectra

`ACCURATE` takes its spectrum from a Boltzmann solver, CLASS by default and CAMB as the alternative; `DIFFERENTIABLE` takes it from `emu_pk`, a neural-network emulator of the linear spectrum written in JAX.
`emu_pk` was trained on 150 000 CAMB 1.6.6 solves over $\omega_{\rm b}$, $\omega_{\rm cdm}$, $h$, $w_0$, $w_a$, $\Omega_k$ and $\sum m_\nu$ with two mass ratios, for $z \le 5$ and $k \le 300\,h\,{\rm Mpc}^{-1}$.
$n_{\rm s}$ and $\ln(10^{10}A_{\rm s})$ are restored in closed form rather than learned.

Each backend returns the total-matter spectrum $P_{\rm m}(k,z)$ and the CDM+baryon spectrum $P_{\rm cb}(k,z)$, in $(h^{-1}{\rm Mpc})^3$ with $k$ in $h\,{\rm Mpc}^{-1}$.
Haloes form out of the CDM+baryon field, so the halo layer uses $P_{\rm cb}$; lensing sees all the matter and uses $P_{\rm m}$.
'''),
("code", r'''
import emu_pk.box

pk = make_pk("emu_pk")
print("emu_pk network inputs:", emu_pk.box.PARAMS)
print("differentiable:", pk.differentiable)

k = np.logspace(-4, np.log10(200.0), 512)
zs = np.array([0.0, 1.0, 3.0])
P_m = pk.pk(k, zs, PLANCK18)          # several redshifts in one call: shape (3, 512)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.0, 3.0))
for zi, row in zip(zs, P_m):
    a0.loglog(k, row, label=f"$z = {zi:g}$")
a0.set(xlabel=r"$k$ [$h$ Mpc$^{-1}$]", ylabel=r"$P_{\rm m}(k)$ [$(h^{-1}{\rm Mpc})^3$]")
a0.legend()
for sum_mnu in (0.06, 0.15, 0.3):
    cc = PLANCK18.replace(sum_mnu=sum_mnu)
    a1.semilogx(k, pk.pk_cb(k, 0.0, cc) / pk.pk(k, 0.0, cc), label=rf"$\sum m_\nu = {sum_mnu}$ eV")
a1.set(xlabel=r"$k$ [$h$ Mpc$^{-1}$]", ylabel=r"$P_{\rm cb}/P_{\rm m}$")
a1.legend()
fig.tight_layout()
'''),
("md", r'''
The emulator refuses a cosmology outside its training box rather than extrapolating.
'''),
("code", r'''
try:
    pk.pk(k, 0.0, PLANCK18.replace(sum_mnu=0.8))
except ValueError as err:
    print(err)
'''),
("md", r'''
## The control at zero mass

With massless neutrinos the two fields are one, so $P_{\rm cb}/P_{\rm m} - 1$ must vanish.
CLASS and CAMB return zero identically.
`emu_pk` has one output head per spectrum and nothing ties them together, so it returns a small residual.
'''),
("code", r'''
c0 = PLANCK18.replace(sum_mnu=0.0)
kk = np.logspace(-2, np.log10(0.5), 200)
resid = float(np.max(np.abs(pk.pk_cb(kk, 0.0, c0) / pk.pk(kk, 0.0, c0) - 1)))
print(f"max |P_cb/P_m - 1| at zero mass = {resid:.1e}   (paper: 3.0e-4)")
assert 1e-4 < resid < 1e-3
'''),
("md", r'''
## The emulator against CLASS

`emu_pk` inherits the accuracy of its training set, CAMB at raised precision, which is $3.2\times10^{-5}$ from CAMB's converged answer at the fiducial.
Against held-out spectra it is accurate to 0.064 per cent at the median and 0.18 per cent at most.
The comparison below runs only if `classy` is installed (`pip install "ggah_mod[reference]"`); a CLASS solve takes about 20 s.
The difference is then the emulator's error plus the distance between CLASS and CAMB at their precision settings, which the paper puts at $6.1\times10^{-4}$.
'''),
("code", r'''
try:
    pk_class = make_pk("class")
    kc = np.logspace(-2, np.log10(0.5), 200)
    for zi in (0.0, 1.0):
        r = pk.pk(kc, zi, PLANCK18) / pk_class.pk(kc, zi, PLANCK18) - 1
        print(f"z = {zi:g}: max |emu_pk/CLASS - 1| on 0.01 < k < 0.5 = {np.max(np.abs(r)):.1e}")
except ImportError:
    print("classy is not installed: skipped")
'''),
("md", r'''
## Derivatives

`jax.jacfwd` differentiates $\ln P_{\rm m}(k)$ through the network, one parameter at a time, in the convention of notebook 01 (logarithmic for the five positive scales, absolute for the other five).
Two directions are known in closed form, which checks that they are applied and not fitted.
The amplitude enters $\ln P$ as an additive constant, so $\partial\ln P/\partial\ln(10^{10}A_{\rm s}) = 1$.
The tilt gives $\partial\ln P/\partial n_{\rm s} = \ln(k/k_{\rm pivot})$, with $k_{\rm pivot} = 0.05\,{\rm Mpc}^{-1} = 0.05/h\;h\,{\rm Mpc}^{-1}$.
'''),
("code", r'''
k = np.logspace(-3, np.log10(200.0), 200 if FAST else 400)
kj = jnp.asarray(k)
c = PLANCK18

g, gr = {}, {}
for p in T.DPARAMS:
    v0 = float(getattr(c, p))
    scale = v0 if p in T.DLOG else 1.0
    lnP = lambda v, p=p: jnp.log(pk.pk(kj, 0.0, c.replace(**{p: v})))
    ratio = lambda v, p=p: (jnp.log(pk.pk_cb(kj, 0.0, c.replace(**{p: v})))
                            - jnp.log(pk.pk(kj, 0.0, c.replace(**{p: v}))))
    g[p] = np.asarray(jax.jacfwd(lnP)(v0)) * scale
    gr[p] = np.asarray(jax.jacfwd(ratio)(v0)) * scale

print(f"{'':10s}{'dlnP at k=0.1':>15s}{'dln(Pcb/Pm) at k=1':>21s}")
for p in T.DPARAMS:
    print(f"{p:10s}{np.interp(0.1, k, g[p]):+15.4g}{np.interp(1.0, k, gr[p]):+21.4g}")
'''),
("code", r'''
amp = float(np.max(np.abs(g["ln10A_s"] - 1.0)))
k_piv = 0.05 / c.h
tilt = float(np.max(np.abs(g["n_s"] - np.log(k / k_piv))))
print(f"d lnP / d ln10A_s = 1 to {amp:.1e}")
print(f"d lnP / d n_s = ln(k/k_pivot) to {tilt:.1e}   (paper: 5e-13)")
assert amp < 1e-12 and tilt < 1e-11
'''),
("md", r'''
A central finite difference checks the automatic derivative.
'''),
("code", r'''
for p, step in (("Omega_m", 1e-4), ("h", 1e-4), ("w0", 1e-4), ("sum_mnu", 1e-3)):
    v0 = float(getattr(c, p))
    scale = v0 if p in T.DLOG else 1.0
    lnP = lambda v: np.log(np.asarray(pk.pk(kj, 0.0, c.replace(**{p: v}))))
    fd = (lnP(v0 + step) - lnP(v0 - step)) / (2 * step) * scale
    print(f"{p:8s} max |autodiff - finite difference| / max |derivative| = "
          f"{np.max(np.abs(g[p] - fd)) / np.max(np.abs(g[p])):.1e}")
'''),
("md", r'''
## The figure

Fig. 3 of the paper.
Panel (a) is $\partial\ln P_{\rm m}/\partial\theta$; the wiggles of $\Omega_{\rm m}$ and $h$ around $k\sim0.1\,h\,{\rm Mpc}^{-1}$ are the acoustic scale moving with the parameter.
Panel (b) is the same derivatives of the ratio $P_{\rm cb}/P_{\rm m}$: insensitive to $n_{\rm s}$ and the amplitude, and responding to $\Omega_{\rm m}$ and $h$ only through the neutrino fraction $f_\nu$.
'''),
("code", r'''
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9))
for p in T.DPARAMS:
    colour, ls = T.DSTYLE[p]
    ax[0].semilogx(k, g[p], color=colour, ls=ls, lw=1.3, label=T.TEX[p])
    ax[1].semilogx(k, gr[p], color=colour, ls=ls, lw=1.3)
for a in ax:
    a.axhline(0.0, color="k", lw=0.8)
    a.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
ax[0].set_ylabel(r"$\partial\ln P_{\rm m}/\partial\theta$")
ax[0].legend(fontsize=6.3, ncol=2, loc="upper left", framealpha=0.9, handlelength=1.7, columnspacing=0.9)
ax[0].set_title("(a) the ten emu_pk responds to")
ax[1].set_ylabel(r"$\partial\ln(P_{\rm cb}/P_{\rm m})/\partial\theta$")
ax[1].annotate("colours and dashes as in (a)", xy=(0.5, 0.03), xycoords="axes fraction",
               ha="center", va="bottom", fontsize=6.6, color="0.25")
ax[1].set_title(r"(b) the $cb$ head against the total")
fig.tight_layout()

assert not any(np.all(g[p] == 0.0) for p in T.DPARAMS)   # none of the ten is an exact zero
T.save_figure(fig, "fig_pk_gradients")
'''),
("md", r'''
## Cost

A forecast wants a gradient cheap enough to take many times.
The paper measures 0.47 ms for one spectrum and 0.84–0.85 ms for its gradient, on one core of an Intel Xeon Gold 6244; a laptop gives the same order.
A CLASS solve takes about 20 s, so a Fisher matrix of thousands of evaluations is seconds through the emulator and hours through the solver.
'''),
("code", r'''
k200 = jnp.logspace(-3, 1, 200)
f_val = jax.jit(lambda om: pk.pk(k200, 0.0, c.replace(Omega_m=om)))
f_grad = jax.jit(jax.jacfwd(lambda om: jnp.log(pk.pk(k200, 0.0, c.replace(Omega_m=om)))))
for label, f in (("value", f_val), ("gradient", f_grad)):
    f(0.31).block_until_ready()                     # compile
    t0 = time.perf_counter()
    for _ in range(200):
        f(0.31).block_until_ready()
    print(f"{label:9s} {(time.perf_counter() - t0) / 200 * 1e3:.2f} ms")
'''),
("md", r'''
## Exercises

1. Plot $\partial\ln P_{\rm m}/\partial\theta$ at $z = 0$, 1 and 3 for $\Omega_{\rm m}$. Why does the derivative at large scales change with redshift?
2. Take the Jacobian of $\ln P_{\rm m}$ with respect to all ten parameters at once, with `jax.jacfwd` and with `jax.jacrev`, on 31 redshifts. Compare the time and the memory; which mode does a Fisher matrix want, and why?
3. Move $\sum m_\nu$ from 0.06 to 0.3 eV and plot the change in $P_{\rm cb}/P_{\rm m}$. At which wavenumber does free streaming set in?

Next: notebook 03, the amplitude and the growth.
'''),
]

n03 = [
("md", r'''
# 3. Amplitude, variance and growth

This notebook follows Secs. 2.4 and 2.5 of Paper I.
It builds the top-hat variance $\sigma(R)$, the amplitudes $\sigma_8$ and $S_8$, the growth factor and the growth rate, and the scale dependence massive neutrinos give the growth.
It ends on the figure of the amplitude gradients.
'''),
("code", SETUP + r'''
from ggah_mod.cosmology import (PLANCK18, growth_factor, growth_scale_spread,
                                ln10A_s_for_sigma8, make_pk, s8, sigma2_tophat,
                                sigma8, tophat_window)
from ggah_mod.cosmology.growth import f_sigma8, growth_rate

pk = make_pk("emu_pk")
c = PLANCK18
'''),
("md", r'''
## The top-hat window

The variance of the linear field smoothed on a scale $R$ uses the Fourier transform of a real-space top hat, evaluated in two branches (Eq. 25):

$$W(x) = \begin{cases} 1 - x^2/10 + x^4/280, & x < 0.1,\\ 3(\sin x - x\cos x)/x^3, & x \ge 0.1.\end{cases}$$

The series is a requirement: the closed form cancels catastrophically at small $x$.
Both branches are evaluated and selected with `where`, the unused one held finite, so the gradient is clean at the switch.
'''),
("code", r'''
x = np.logspace(-6, 0, 400)
closed = 3 * (np.sin(x) - x * np.cos(x)) / x**3
fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.loglog(x, np.abs(closed / tophat_window(x) - 1) + 1e-18, label="closed form")
ax.axvline(0.1, color="0.5", ls=":")
ax.set(xlabel="$x = kR$", ylabel=r"$|W_{\rm closed}/W - 1|$", ylim=(1e-17, 1))
ax.legend()

dW = jax.vmap(jax.grad(tophat_window))(jnp.array([0.0999999, 0.1, 0.1000001]))
print("dW/dx either side of the switch:", np.asarray(dW))
'''),
("md", r'''
The closed form loses about twelve significant digits at $x\sim10^{-4}$; the series is accurate to machine precision below the switch.

## The variance and the amplitudes

The variance is a trapezoid in $\ln k$ (Eq. 26),

$$\sigma^2(R) = \frac{1}{2\pi^2}\int P(k)\,W^2(kR)\,k^3\,{\rm d}\ln k,$$

and the amplitudes follow, $\sigma_8 = \sigma(8\,h^{-1}{\rm Mpc})$ on $P_{\rm m}$ and $S_8 = \sigma_8\sqrt{\Omega_{\rm m}/0.3}$ (Eq. 27).
Both take the total-matter spectrum, as the priors quoted on them assume.
'''),
("code", r'''
k = np.logspace(-4, 2, 512)
P0 = pk.pk(k, 0.0, c)
print(f"sigma8 = {float(sigma8(P0, k)):.5f},  S8 = {float(s8(P0, k, c)):.5f}")
print(f"sigma(8 Mpc/h) on P_cb = {float(sigma8(pk.pk_cb(k, 0.0, c), k)):.5f}")

# the trapezoid on 512 wavenumbers against a 4096-point grid
kf = np.logspace(-4, 2, 4096)
Pf = pk.pk(kf, 0.0, c)
for R in (1.0, 8.0, 30.0):
    r = np.sqrt(sigma2_tophat(P0, k, R) / sigma2_tophat(Pf, kf, R)) - 1
    print(f"R = {R:4g} Mpc/h: sigma(512) / sigma(4096) - 1 = {float(r):+.1e}")
'''),
("md", r'''
$\sigma_8$ is an output.
To start from a requested $\sigma_8$, solve once for the amplitude and carry it.
'''),
("code", r'''
lnA = ln10A_s_for_sigma8(0.78, c, pk, k=k)
print(f"ln10A_s = {lnA:.5f} gives sigma8 = {float(sigma8(pk.pk(k, 0.0, c.replace(ln10A_s=lnA)), k)):.6f}")
'''),
("md", r'''
## Growth

There is one growth factor in this package, the ratio of the top-hat amplitudes of the spectrum at two redshifts (Eq. 28),

$$D_X(z) = \frac{\sigma_8[P_X(k,z)]}{\sigma_8[P_X(k,0)]},\qquad X \in \{{\rm m}, {\rm cb}\}.$$

No fitting formula and no separate ODE.
The growth rate $f = {\rm d}\ln D/{\rm d}\ln a$ is a second-order difference in $\ln(1+z)$ with a step of $10^{-3}$ (Eq. 29).
'''),
("code", r'''
z = np.linspace(0.0, 3.0, 31)
D, f, fs8 = growth_factor(z, c, pk, k=k), growth_rate(z, c, pk, k=k), f_sigma8(z, c, pk, k=k)
fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.plot(z, D, label="$D(z)$")
ax.plot(z, f, label="$f(z)$")
ax.plot(z, fs8, label=r"$f\sigma_8(z)$")
omz = c.Omega_m * (1 + z) ** 3 / (c.Omega_m * (1 + z) ** 3 + c.Omega_de)
ax.plot(z, omz ** 0.55, "k:", lw=1, label=r"$\Omega_{\rm m}(z)^{0.55}$")
ax.set(xlabel="$z$")
ax.legend();
'''),
("md", r'''
The dotted line is the familiar approximation $f \simeq \Omega_{\rm m}(z)^{0.55}$, for comparison.

## Scale-dependent growth

With massive neutrinos, free streaming suppresses small scales more than large ones, so $\sqrt{P(k,z)/P(k,0)}$ depends on $k$ and a scalar $D(z)$ is ambiguous.
`growth_scale_spread` measures the ambiguity (Eq. 30),
$\mathcal{S}(z) = \max_k |\sqrt{P(k,z)/P(k,0)}/D(z) - 1|$ over $10^{-3} < k < 10$.
The paper, with CLASS, finds $5.6\times10^{-5}$ at $z = 0.5$ with massless neutrinos, $6.8\times10^{-4}$ at 0.06 eV and $3.0\times10^{-3}$ at 0.30 eV.
The emulator's own error sets a floor of a few $10^{-4}$ here.
'''),
("code", r'''
for sum_mnu in (0.0, 0.06, 0.3):
    cc = c.replace(sum_mnu=sum_mnu)
    print(f"sum m_nu = {sum_mnu:4.2f} eV: S(z=0.5) = {float(growth_scale_spread(0.5, cc, pk)):.1e}")
'''),
("md", r'''
## Derivatives and the figure

Fig. 4 of the paper differentiates $\ln\sigma_8(z)$ and $\ln D(z)$ with respect to the ten parameters, both off one total-matter spectrum on one $k$ grid.
Nothing on $\sigma_8$ is exactly zero, unlike the background of notebook 01: $\sigma_8$ integrates the spectrum, so every parameter that moves $P(k)$ moves it.
Two exact statements check the figure: $\partial\ln\sigma_8/\partial\ln(10^{10}A_{\rm s}) = 1/2$ at every redshift, since $\sigma_8 \propto \sqrt{A_{\rm s}}$, and the amplitude cancels from $D$, a ratio of two $\sigma_8$ at one amplitude.
'''),
("code", r'''
z = np.geomspace(0.01, 3.0, 20 if FAST else 60)
zj, kj = jnp.asarray(z), jnp.asarray(k)
gs, gd = {}, {}
for p in T.DPARAMS:
    v0 = float(getattr(c, p))
    scale = v0 if p in T.DLOG else 1.0

    def lns8(v, p=p):
        cc = c.replace(**{p: v})
        return jnp.log(jnp.array([sigma8(pk.pk(kj, zz, cc), kj) for zz in z]))

    def lnD(v, p=p):
        return jnp.log(growth_factor(zj, c.replace(**{p: v}), pk, k=k))

    gs[p] = np.asarray(jax.jacfwd(lns8)(v0)) * scale
    gd[p] = np.asarray(jax.jacfwd(lnD)(v0)) * scale

LIN = 1e-2
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
for axis, g, name, title in ((ax[0], gs, r"\sigma_8", "(a) the amplitude"),
                             (ax[1], gd, "D", "(b) the growth factor")):
    axis.axhspan(-LIN, LIN, color="0.90", zorder=0)
    for p in T.DPARAMS:
        colour, ls = T.DSTYLE[p]
        axis.plot(z, g[p], color=colour, ls=ls, lw=1.3, label=T.TEX[p])
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=LIN)
    axis.axhline(0.0, color="k", lw=0.8)
    axis.set(xlabel="$z$", title=title, ylabel=rf"$\partial\ln {name}/\partial\theta$")
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=6.6, ncol=10, loc="lower center", bbox_to_anchor=(0.5, -0.015),
           frameon=False, handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.055, 1, 1))
T.save_figure(fig, "fig_amplitude_gradients")
'''),
("md", r'''
## Exercises

1. Replace the total-matter spectrum by `pk.pk_cb` in $D(z)$ (`variant="cold"`) and compare at $\sum m_\nu = 0.3$ eV. Which one should a halo mass function use?
2. The growth rate is a finite difference in $\ln(1+z)$. Differentiate $\ln D$ with respect to $\ln(1+z)$ with `jax.grad` instead and compare with `growth_rate`.
3. Compute $\partial S_8/\partial\Omega_{\rm m}$ at fixed amplitude, and at fixed $\sigma_8$ (re-solving the amplitude with `ln10A_s_for_sigma8` at each step). Why do they differ?

Next: notebook 04, the variance as a function of halo mass.
'''),
]

which = sys.argv[1:] or ["02", "03"]
if "02" in which:
    write("02_power_spectrum", n02)
if "03" in which:
    write("03_amplitude_growth", n03)
