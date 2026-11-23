"""Write tutorials/A and B (sources only; executed separately)."""
import sys

from _nb import SETUP, write

nA = [
("md", r'''
# A. Linear power-spectrum emulators: accuracy and speed

This notebook follows Appendix A of Paper I, which redoes Fig. 10 of Sui et al. (2025) with `emu_pk` in it.
Each emulator is scored by the mean of $|P/P_{\rm true}-1|$ over 200 wavenumbers log-spaced in $[0.009, 9]\,h\,{\rm Mpc}^{-1}$ at $z = 0$, and timed for one evaluation at $z = 1$, at the four Quijote cosmologies.
We recompute `emu_pk`'s errors and time it on this machine; the other codes need their own environments, so their errors and timings, and the CLASS and CAMB spectra used as truths, come from the tutorial data bundle.
'''),
("code", SETUP + r'''
import json
import time

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from emu_pk import PkEmulator

load = lambda name: json.load(open(T.data_path(name)))
'''),
("md", r'''
## The four cosmologies and the two truths

Fid has $\Omega_{\rm m} = 0.3175$, $\Omega_{\rm b} = 0.049$, $h = 0.6711$, $n_{\rm s} = 0.9624$, $\sigma_8 = 0.834$, $w = -1$ and massless neutrinos.
$w^\pm$ change only $w$, to $-0.95$ and $-1.05$, at the same $\sigma_8$.
$M_\nu^+$ adds $\sum m_\nu = 0.1$ eV in three equal masses at the $A_{\rm s}$ and $\Omega_{\rm m}$ of Fid.
The first truth is CLASS 3.3.4 at its default precision, set up as Sui et al. (2025) do; the second is converged CLASS.
'''),
("code", r'''
cos = load("emulators/cosmologies.json")
truth = np.load(T.data_path("emulators/truth.npz"))
ref = np.load(T.data_path("emulators/ref.npz"))
K = np.logspace(np.log10(9e-3), np.log10(9.0), 200)
assert np.allclose(truth["k"], K)
for name, cc in cos.items():
    print(f"{name:5s} " + "  ".join(f"{k}={v:.4g}" for k, v in cc.items()))
'''),
("md", r'''
## Scoring `emu_pk`

`emu_pk` takes $(\omega_{\rm b}, \omega_{\rm cdm}, h, n_{\rm s}, \ln10^{10}A_{\rm s}, \sum m_\nu, w_0, w_a, \Omega_k)$ and two mass ratios ($1/3$ each for degenerate masses); $\Omega_{\rm m}$ contains the neutrinos' rest mass, $\Omega_\nu h^2 = \sum m_\nu/93.14$ eV.
'''),
("code", r'''
emu = PkEmulator()


def theta(c):
    ocdm = (c["Om"] - c["Ob"] - c["mnu"] / 93.14 / c["h"] ** 2) * c["h"] ** 2
    return np.array([c["Ob"] * c["h"] ** 2, ocdm, c["h"], c["ns"], np.log(1e10 * c["As"]), c["mnu"],
                     c["w0"], c["wa"], 0.0, 1 / 3, 1 / 3])


err = {n: float(np.mean(np.abs(np.asarray(emu.pk(K, 0.0, theta(c))) / truth[f"class/{n}/0"] - 1))) for n, c in cos.items()}
err_ref = {n: float(np.mean(np.abs(np.asarray(emu.pk(K, 0.0, theta(c))) / ref[n] - 1))) for n, c in cos.items()}
cpu = load("emulators/cpu.json")
print(f"{'':6s}{'vs CLASS default':>18s}{'(bundle)':>10s}{'vs converged':>14s}{'(bundle)':>10s}")
for n in cos:
    print(f"{n:6s}{err[n]:18.2e}{cpu['error_z0']['emu_pk'][n]:10.2e}{err_ref[n]:14.2e}{cpu['error_ref_z0']['emu_pk'][n]:10.2e}")
'''),
("md", r'''
At $M_\nu^+$ the default-precision CLASS is itself $2.1\times10^{-3}$ from converged CLASS, so the offset in the first column is that of the truth, not of the emulator.

## Timing

The paper times one core of an Intel Xeon Gold 6244 with one thread, the JAX function compiled, as the mean over 1 000 calls.
A laptop is usually faster.
'''),
("code", r'''
kj = jnp.asarray(K)
f = jax.jit(lambda th: emu.pk(kj, 1.0, th))
th = jnp.asarray(theta(cos["Fid"]))
f(th).block_until_ready()
n = 200 if FAST else 1000
t0 = time.perf_counter()
for _ in range(n):
    f(th).block_until_ready()
t_here = (time.perf_counter() - t0) / n
print(f"emu_pk on this machine: {1e3 * t_here:.2f} ms;  paper (Xeon Gold 6244): {1e3 * cpu['time_cpu_z1']['emu_pk']:.2f} ms")
'''),
("md", r'''
## The figure

Fig. B.1 of the paper (App. A).
Top: against CLASS at its default precision, the truth of Sui et al. (2025); the dotted vertical is the time of that CLASS.
Bottom: against converged CLASS.
Left: one CPU core.
Right: GPU, in batches of 2 000 (filled) and 20 000 (open), in single precision.
'''),
("code", r'''
gpu = load("emulators/gpu.json")
ttr = load("emulators/timing_truth.json")
b = load("emulators/bacco_benchmark.json")
COLOURS = {"emu_pk": "#1f6f8b", "SYREN-NEW": "#d4a017", "SYREN LCDM": "#7a5c00", "CosmoPower": "#c0392b",
           "Eisenstein & Hu": "#8c8c8c", "CAMB": "#8e44ad", "CAMB (emu_pk precision)": "#d291e6",
           "CLASS": "#2e8b57", "BACCO": "#e7298a"}
LABELS = {"SYREN LCDM": r"SYREN $\Lambda$CDM", "CAMB (emu_pk precision)": "CAMB, emu_pk precision",
          "CLASS": "CLASS, default precision"}
MARKERS = {"Fid": "o", "Mnu+": "s", "w+": "^", "w-": "v"}
MLAB = {"Fid": "Fid", "Mnu+": r"$M_\nu^+$", "w+": r"$w^+$", "w-": r"$w^-$"}

tcpu = dict(cpu["time_cpu_z1"], **ttr)
tcpu.pop("SYREN-NEW (JAX)")
tgpu = dict(gpu["time_gpu_z1"])
tgpu["SYREN-NEW"] = tgpu.pop("SYREN-NEW (JAX)")
tbig = dict(gpu["time_gpu_z1_batch20000"])
tbig["SYREN-NEW"] = tbig.pop("SYREN-NEW (JAX)")
e0, eref = dict(cpu["error_z0"]), dict(cpu["error_ref_z0"])
e0["BACCO"], eref["BACCO"], tcpu["BACCO"] = b["err_vs_class_default"], b["err_vs_class_converged"], b["cpu_time_s"]

# The style of the script that made the paper's figure: matplotlib's defaults at 9 pt, cropped tight.
plt.rcdefaults()
plt.rcParams.update({"savefig.bbox": "tight", "font.size": 9})
fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.6), sharey=True, sharex="col", gridspec_kw={"wspace": 0.05, "hspace": 0.08})
for r, (errs, lab) in enumerate(((e0, "against CLASS, default precision"), (eref, "against converged CLASS"))):
    for i, times in enumerate((tcpu, tgpu)):
        for name, t in times.items():
            for cname, e in errs.get(name, {}).items():
                ax[r, i].scatter(t, 100 * e, marker=MARKERS[cname], s=26, zorder=3, facecolor=COLOURS[name],
                                 edgecolor="k", lw=0.7)
        ax[r, i].set_xscale("log")
        ax[r, i].grid(alpha=0.3)
    for name, t in tbig.items():
        for cname, e in errs.get(name, {}).items():
            ax[r, 1].scatter(t, 100 * e, marker=MARKERS[cname], s=26, zorder=3, facecolor="white",
                             edgecolor=COLOURS[name], lw=1.2)
    ax[r, 0].set_ylabel(r"mean $|\Delta P/P|$ [%]" + "\n" + lab, fontsize=8.5)
ax[0, 0].set_yscale("log")
ax[0, 0].axvline(ttr["CLASS"], color=COLOURS["CLASS"], ls=":", lw=1)
ax[0, 0].text(ttr["CLASS"] * 0.75, 0.96, "CLASS\n(truth)", ha="right", va="top", fontsize=7,
              transform=ax[0, 0].get_xaxis_transform())
ax[0, 0].set_title("CPU, one core", fontsize=8.5)
ax[0, 1].set_title("GPU, batch of 2000 (filled)\nand 20 000 (open)", fontsize=8.5)
ax[1, 0].set_xlabel("time per evaluation [s]")
ax[1, 1].set_xlabel("time per evaluation [s]")
ax[1, 1].set_xlim(5e-8, 5e-6)
import matplotlib.ticker
ax[1, 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
hand = [Patch(color=COLOURS[nm], label=LABELS.get(nm, nm)) for nm in COLOURS]
hand += [Line2D([], [], ls="", marker=m, mfc="white", mec="k", label=MLAB[cn]) for cn, m in MARKERS.items()]
fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=7.5, frameon=False,
           columnspacing=1.0, handlelength=1.2);
T.save_figure(fig, "fig_emulator_benchmark")
T.paper_style()
'''),
("md", r'''
## Check against the paper
'''),
("code", r'''
quoted = [("emu_pk Fid vs CLASS default", err["Fid"], 7.3e-4), ("emu_pk Mnu+ vs CLASS default", err["Mnu+"], 2.3e-3),
          ("emu_pk Fid vs converged", err_ref["Fid"], 8.3e-4), ("emu_pk Mnu+ vs converged", err_ref["Mnu+"], 2.7e-4),
          ("SYREN-NEW CPU [ms]", 1e3 * cpu["time_cpu_z1"]["SYREN-NEW"], 0.17), ("emu_pk CPU [ms]", 1e3 * cpu["time_cpu_z1"]["emu_pk"], 0.42),
          ("CLASS CPU [s]", ttr["CLASS"], 0.57), ("emu_pk GPU 20000 [s]", gpu["time_gpu_z1_batch20000"]["emu_pk"], 1.9e-7)]
bad = []
for label, here, paper in quoted:
    ok = abs(here / paper - 1) < 0.05
    print(f"{label:30s} here {here:.3g}   paper {paper:.3g}" + ("" if ok else "   <-- differs"))
    bad += [] if ok else [label]
assert not bad, bad
assert all(abs(err[n] / cpu["error_z0"]["emu_pk"][n] - 1) < 1e-3 for n in cos)
print("App. A reproduced")
'''),
("md", r'''
## Exercises

1. Score `emu_pk` through `ggah_mod` instead (`make_pk("emu_pk")` on a `Cosmology` built from each Quijote cosmology). Which of the two inputs, $\Omega_{\rm m}$ or $\omega_{\rm cdm}$, must you convert, and does the score change?
2. Time the gradient of $\ln P$ with respect to the nine parameters with `jax.jacfwd`. How many spectra does one gradient cost?
3. Batch 2 000 cosmologies with `jax.vmap` and time them on your CPU. How does the time per spectrum compare with one call?
'''),
]

nB = [
("md", r'''
# B. Code comparisons

This notebook follows Appendix B of Paper I and the comparisons of Secs. 2.2 and 2.3.
Every code is compared with every other, one parameter varied at a time about `PLANCK18` over a five-point grid: the background $E(z)$ and $\chi(z)$, their derivatives, and the linear $P_{\rm m}$ and $P_{\rm cb}$ with their derivatives.
We recompute one row against CAMB if it is installed, then rebuild the four tables of the appendix from `accuracy.json`, which holds every number the paper quotes from these comparisons.
'''),
("code", SETUP + r'''
import json
import re

import pandas as pd

from ggah_mod.cosmology import PLANCK18, comoving_distance, hubble_e

A = json.load(open(T.data_path("comparisons/accuracy.json")))
print(len(A), "measurements")
'''),
("md", r'''
## One row, recomputed

For each pair of codes and each parameter, the cell is the largest fractional deviation over the five-point grid and over $z\in[0.01, 3]$, for $E$ and for $\chi$ separately.
We recompute this package against CAMB for the $\Omega_{\rm m}$ scan.
`camb_input` translates a `Cosmology` into CAMB's parameters with the same neutrino convention.
'''),
("code", r'''
z = np.linspace(0.01, 3.0, 60)
GRID = {"Omega_m": (0.24, 0.28, 0.31, 0.35, 0.40), "w0": (-1.3, -1.15, -1.0, -0.85, -0.7)}
try:
    import camb
    from ggah_mod.cosmology.power import camb_input
    for p, grid in GRID.items():
        de = dx = 0.0
        for v in grid:
            c = PLANCK18.replace(**{p: v})
            bg = camb.get_background(camb_input(c))
            e_c = bg.hubble_parameter(z) / bg.hubble_parameter(0.0)
            x_c = bg.comoving_radial_distance(z) * c.h
            de = max(de, float(np.max(np.abs(np.asarray(hubble_e(z, c)) / e_c - 1))))
            dx = max(dx, float(np.max(np.abs(np.asarray(comoving_distance(z, c)) / x_c - 1))))
        print(f"{p:8s} E: {de:.1e} (paper {A[f'scan {p}: E, ggah vs CAMB']:.1e})   "
              f"chi: {dx:.1e} (paper {A[f'scan {p}: chi, ggah vs CAMB']:.1e})")
except ImportError:
    print("camb is not installed (pip install camb): skipped")
'''),
("md", r'''
## The four tables

A cell holds two numbers, $E$/$\chi$ for the background and $P_{\rm m}$/$P_{\rm cb}$ for the power spectrum; a dash is a parameter one of the pair does not take.
The derivative tables start with a tier of self-checks, each differentiable code against its own finite differences.
'''),
("code", r'''
def fmt(x):
    return "-" if x is None or x != x else f"{x:.1e}"


def table(prefix, params, pairs, q):
    """Rows: pairs; columns: params; cell 'q0 / q1' from keys '<prefix> <p>: <q>, <pair>'."""
    rows = {}
    for pair in pairs:
        rows[pair] = [fmt(A.get(f"{prefix} {p}: {q[0]}, {pair}")) + " / " + fmt(A.get(f"{prefix} {p}: {q[1]}, {pair}"))
                      for p in params]
    return pd.DataFrame.from_dict(rows, orient="index", columns=params)


def pairs_of(prefix, q):
    seen = []
    for k in A:
        m = re.match(rf"{re.escape(prefix)} \S+: {q}, (.+)$", k)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


BG = ["Omega_m", "Omega_b", "h", "sum_mnu", "w0", "wa", "Omega_k", "T_cmb"]
PK = ["Omega_m", "Omega_b", "h", "n_s", "ln10A_s", "sum_mnu", "w0", "wa", "Omega_k", "T_cmb"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("Table B.1: background values, E / chi")
print(table("scan", BG, pairs_of("scan", "E"), ("E", "chi")).to_string())
'''),
("code", r'''
print("Table B.2: background derivatives, E / chi")
print(table("dscan", BG, pairs_of("dscan", "E"), ("E", "chi")).to_string())
'''),
("code", r'''
for prefix, title in (("pkscan", "Table B.3: P(k) values at the fiducial, Pm / Pcb"),
                      ("pkscan-base", "Table B.3, lower tier: P(k) values at the common baseline, Pm / Pcb"),
                      ("pkdscan", "Table B.4: P(k) derivatives at the fiducial, Pm / Pcb"),
                      ("pkdscan-base", "Table B.4, lower tier: P(k) derivatives at the baseline, Pm / Pcb")):
    print(title)
    print(table(prefix, PK, pairs_of(prefix, "Pm"), ("Pm", "Pcb")).to_string())
    print()
'''),
("md", r'''
## Check against the paper

Sec. 2.2 quotes from Table B.1 that this package, CAMB and CLASS agree to $2.3\times10^{-9}$–$2.9\times10^{-8}$, since they carry one neutrino convention, and that in $w_a$ CLASS departs from the other two by up to $1.3\times10^{-7}$; the derivative in $\Omega_{\rm b}$ is the control, identically zero in every code.
'''),
("code", r'''
three = ["ggah vs CAMB", "ggah vs CLASS", "CAMB vs CLASS"]
rng = lambda ps: [A[f"scan {p}: {q}, {pair}"] for p in ps for q in ("E", "chi") for pair in three]
vals, no_wa = rng(BG), rng([p for p in BG if p != "wa"])
print(f"ggah, CAMB, CLASS agree to {min(vals):.1e}-{max(vals):.1e}; without the w_a column "
      f"{min(no_wa):.1e}-{max(no_wa):.1e}   (paper text: 2.3e-09-2.9e-08, w_a 1.3e-07)")
assert abs(min(no_wa) / 2.3e-9 - 1) < 0.05 and abs(max(no_wa) / 2.9e-8 - 1) < 0.05 and abs(max(vals) / 1.3e-7 - 1) < 0.05      # one neutrino convention: agreement at 1e-8, w_a at 1e-7
ctrl = max(A[k] for k in A if k.startswith("dscan Omega_b: ") and " vs " in k and "AD vs FD" not in k)
print(f"Omega_b derivative control, largest pairwise difference: {ctrl:.1e}")
print("App. B reproduced")
'''),
("md", r'''
## Exercises

1. Install `astropy` and add it to the recomputed row through `astropy.cosmology.w0waCDM`. Why does it sit $10^{-5}$ from the others in $\sum m_\nu$ (Sec. 2.1, the neutrino convention)?
2. Recompute the finite-difference self-check of this package (the `AD vs FD` rows) for $\Omega_{\rm m}$ with a step of $10^{-4}$, and then $10^{-6}$. Which step does the table use, and why does the smaller one not do better?
3. Pick the worst cell of Table B.3 that involves `emu_pk` and explain it from Sec. 2.3.
'''),
]

which = sys.argv[1:] or ["A", "B"]
if "A" in which:
    write("A_emulator_benchmark", nA)
if "B" in which:
    write("B_code_comparisons", nB)
