"""Calibrate the drag-redshift fit of :mod:`ggah_mod.cosmology.drag` against CLASS.

    JAX_ENABLE_X64=1 python tools/calibrate_zdrag.py [--n-fit 2000] [--n-val 500]

Writes ``ggah_mod/cosmology/_zdrag_coefficients.py``.  Needs ``classy`` (the
``reference`` extra).

**What it fits.**  CLASS's ``z_d`` -- drag depth one, HyRec-2 recombination,
helium from the BBN table :class:`~ggah_mod.cosmology.power.ClassPk` pins -- as
a polynomial in :data:`~ggah_mod.cosmology.drag.FEATURES`, over the ``emu_pk``
training box widened by :math:`T_{\\rm CMB}\\pm1\\%`.  Every CLASS call goes
through :func:`~ggah_mod.cosmology.power.class_input`, so the neutrinos, the BBN
table and the precision settings are the package's own and not restated here.

**How the degree is chosen.**  The smallest with ``max |dz_d| <= TARGET_DZ``
over the held-out points and the four neutrino orderings at the fiducial, and
``<= TARGET_DZ_CORNERS`` at the box's corners.  ``TARGET_DZ`` = 0.02 is 1.2e-5
in :math:`r_d`, a third of the CLASS-CAMB difference.

**Where it does not fit.**  Where dark energy grows into the past
(:math:`1+w_0+w_a > 0`) it can dominate the density at recombination, which
none of the fit's inputs describe; CLASS's :math:`z_d` then moves by tens.  The
calibration excludes :math:`|f_{\\rm DE}(z_d)| \\ge` ``F_DE_MAX``, measures the
fit there anyway, and writes both numbers into the coefficient file.

**What it records.**  ``VALIDATION`` (the accuracy actually reached) and
``BBN`` (every assumption behind the helium fraction, read from CLASS and its
table rather than typed), so the file carries its own provenance.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import pprint
import re
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "ggah_mod" / "cosmology" / "_zdrag_coefficients.py"

#: Acceptance on the held-out set: 0.02 in z_d is 1.2e-5 in r_d.
TARGET_DZ = 0.02
#: Acceptance at the box's corners, where the polynomial is at the edge of its
#: data in every axis at once: 0.05 is 3e-5 in r_d, still below CLASS-CAMB.
TARGET_DZ_CORNERS = 0.05
#: Bound on dark energy's share of the density at the drag epoch; see
#: :func:`ggah_mod.cosmology.drag.dark_energy_fraction_at_drag`.
F_DE_MAX = 1.0e-3
#: Feature centres: round numbers near the fiducial, in FEATURES order.
CENTRE = (0.0224, 0.143, 2.7255, 0.06)


def box() -> dict:
    """The calibration box: ``emu_pk.box.BOX`` on its background axes, plus T_CMB +-1%."""
    from emu_pk.box import BOX
    from ggah_mod.cosmology import constants as C
    out = {p: tuple(float(v) for v in BOX[p])
           for p in ("omega_b", "omega_cdm", "h", "sum_mnu", "w0", "wa", "Omega_k")}
    out["T_cmb"] = (round(C.T_CMB * 0.99, 6), round(C.T_CMB * 1.01, 6))
    return out


def cosmology(p: dict, nu_hierarchy: str = "degenerate"):
    """A :class:`Cosmology` at physical densities ``p`` (keys of :func:`box`)."""
    from ggah_mod.cosmology import PLANCK18
    h = p["h"]
    c = PLANCK18.replace(nu_hierarchy=nu_hierarchy, h=h, sum_mnu=p["sum_mnu"],
                         T_cmb=p["T_cmb"], w0=p["w0"], wa=p["wa"],
                         Omega_k=p["Omega_k"], Omega_b=p["omega_b"] / h ** 2)
    c = c.replace(Omega_m=p["omega_cdm"] / h ** 2 + p["omega_b"] / h ** 2
                  + float(c.Omega_nu_matter))
    assert abs(float(c.Omega_cdm) * h ** 2 / p["omega_cdm"] - 1.0) < 1e-12
    return c


def run_class(c, **over):
    """``(z_d, rs_d [Mpc], YHe)`` from CLASS through ``class_input``, or None."""
    from classy import Class
    from ggah_mod.cosmology.power import class_input
    d = class_input(c, output="")
    d.update(over)
    cl = Class()
    cl.set(d)
    try:
        cl.compute()
        r = cl.get_current_derived_parameters(["z_d", "rs_d", "YHe"])
        return r["z_d"], r["rs_d"], r["YHe"]
    except Exception:
        return None
    finally:
        cl.struct_cleanup()
        cl.empty()


def lhs(n: int, b: dict, rng) -> list[dict]:
    names = list(b)
    u = (np.argsort(rng.random((len(names), n)), axis=1).T + rng.random((n, len(names)))) / n
    lo = np.array([b[p][0] for p in names])
    hi = np.array([b[p][1] for p in names])
    return [dict(zip(names, row)) for row in lo + u * (hi - lo)]


def evaluate(points, label, cache=None):
    """Run CLASS on every point; keep the ones it solves, report the rest.

    With ``cache`` (an ``.npz`` path), CLASS's answers are saved on the first
    run and read back on later ones, keyed by ``label`` and checked against the
    points -- so refitting does not re-solve 2760 cosmologies.
    """
    from ggah_mod.cosmology.drag import features
    names = list(points[0].keys() - {"_hierarchy"})
    names.sort()
    hier = [p.get("_hierarchy", "degenerate") for p in points]
    grid = np.array([[p[k] for k in names] for p in points])
    stored = dict(np.load(cache)) if cache and os.path.exists(cache) else {}
    t0 = time.perf_counter()
    if f"{label}/grid" in stored:
        assert np.array_equal(stored[f"{label}/grid"], grid), f"{label}: cache is of other points"
        out = stored[f"{label}/out"]
    else:
        out = np.full((len(points), 3), np.nan)
        for i, (row, hh) in enumerate(zip(grid, hier)):
            try:
                r = run_class(cosmology(dict(zip(names, row)), hh))
            except ValueError:
                r = None
            if r is not None:
                out[i] = r
        if cache:
            stored.update({f"{label}/grid": grid, f"{label}/out": out})
            np.savez(cache, **stored)
    rows, failed = [], []
    for row, hh, r in zip(grid, hier, out):
        p = dict(zip(names, row))
        if np.isnan(r[0]):
            failed.append(p)
            continue
        c = cosmology(p, hh)
        rows.append(dict(p=p, hierarchy=hh, cosmo=c, z_d=r[0], rs_d=r[1], YHe=r[2],
                         x=np.asarray(features(c, jnp.asarray(CENTRE)))))
    print(f"  {label}: {len(rows)} solved, {len(failed)} refused by CLASS, "
          f"{time.perf_counter() - t0:.0f} s")
    for p in failed[:5]:
        print("    refused:", {k: round(v, 5) for k, v in p.items()})
    return rows, failed


def bbn_record(fid_row) -> dict:
    """Every BBN assumption behind CLASS's helium fraction, read from source."""
    import classy
    from ggah_mod.cosmology import PLANCK18, constants as C
    from ggah_mod.cosmology.power import ClassPk
    table = pathlib.Path(classy.__file__).parent / ClassPk.SBBN_FILE.lstrip("/")
    header = [ln[1:].strip() for ln in table.read_text().splitlines()
              if ln.startswith("#") and ln[1:].strip()][:6]
    text = " ".join(header)
    code = re.search(r"(PArthENoPE v[0-9.]+)", text)
    tau_n = re.search(r"neutron\s+lifetime of ([0-9.]+)\s*s", text)
    # The response of z_d and r_d to the helium fraction, at the fiducial.
    y0 = float(fid_row["YHe"])
    up = run_class(PLANCK18, YHe=y0 + 1e-3)
    dn = run_class(PLANCK18, YHe=y0 - 1e-3)
    return {
        "class_version": str(getattr(classy, "__version__", "unknown")),
        "recombination": "CLASS default (HyRec-2: Lee & Ali-Haimoud 2020; "
                         "Ali-Haimoud & Hirata 2011)",
        "sbbn_file": ClassPk.SBBN_FILE,
        "sbbn_header": header,
        "bbn_code": code.group(1) if code else None,
        "neutron_lifetime_s": float(tau_n.group(1)) if tau_n else None,
        "helium_quantity": "mass fraction YHe, read from the table's third "
                           "column with no Y_p^BBN conversion",
        "n_eff": float(C.N_EFF),
        "delta_n_looked_up": round(float(C.N_EFF) - 3.046, 6),
        "delta_n_convention": "CLASS measures N_eff at T = 0.1 MeV from the "
                              "background and subtracts 3.046, the table's "
                              "reference (thermodynamics_helium_from_bbn)",
        "t_cmb_rescaling": "none: CLASS interpolates the table at omega_b as "
                           "given; CAMB rescales omega_b by (2.7255/T_cmb)^3",
        "standard_bbn": "no electron-neutrino chemical potential; constants "
                        "not varied",
        "YHe_fiducial": y0,
        "dzd_dYHe": (up[0] - dn[0]) / 2e-3,
        "dlnrd_dYHe": float(np.log(up[1] / dn[1]) / 2e-3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--max-degree", type=int, default=6)
    ap.add_argument("--cache", default=None,
                    help=".npz of CLASS's answers, written on the first run and "
                         "reused after")
    args = ap.parse_args(argv)

    from ggah_mod.cosmology import PLANCK18, constants as C
    from ggah_mod.cosmology.background import sound_horizon
    from ggah_mod.cosmology.drag import (Z_DE_REF, dark_energy_fraction_at_drag,
                                         design, terms)

    b = box()
    rng = np.random.default_rng(args.seed)
    print("calibration box:", b)

    fit_rows, _ = evaluate(lhs(args.n_fit, b, rng), "fit sample", args.cache)
    val_rows, _ = evaluate(lhs(args.n_val, b, rng), "held-out sample", args.cache)
    corners = [dict(zip(b, v)) for v in itertools.product(*b.values())]
    corner_rows, corner_fail = evaluate(corners, "box corners", args.cache)
    fid = {"omega_b": float(PLANCK18.Omega_b * PLANCK18.h ** 2),
           "omega_cdm": float(PLANCK18.Omega_cdm * PLANCK18.h ** 2),
           "h": float(PLANCK18.h), "w0": -1.0, "wa": 0.0, "Omega_k": 0.0,
           "T_cmb": float(C.T_CMB)}
    orderings = [dict(fid, sum_mnu=m, _hierarchy=hh) for hh, m in
                 (("degenerate", 0.06), ("normal", 0.06), ("inverted", 0.12),
                  ("massless", 0.0))]
    hier_rows, _ = evaluate(orderings, "neutrino orderings at the fiducial", args.cache)

    # The package builds CLASS at its own precision; the drag epoch must not
    # depend on that choice, or the fit would be of a setting.
    from classy import Class
    from ggah_mod.cosmology.power import class_input
    for r in fit_rows[:10]:
        cl = Class()
        cl.set(class_input(r["cosmo"], output="", precision={}))
        cl.compute()
        z_default = cl.get_current_derived_parameters(["z_d"])["z_d"]
        cl.struct_cleanup()
        cl.empty()
        assert z_default == r["z_d"], (z_default, r["z_d"])
    print("  z_d is identical at the package's CLASS precision and CLASS's defaults")

    # The domain: dark energy negligible at the drag epoch.  Above F_DE_MAX the
    # fit's inputs no longer describe H(z ~ 1000) -- measured, not assumed:
    # the excluded points are fitted and reported below.
    for r in fit_rows + val_rows + corner_rows + hier_rows:
        r["f_de"] = float(dark_energy_fraction_at_drag(r["cosmo"]))
    inside = lambda rows: [r for r in rows if abs(r["f_de"]) < F_DE_MAX]
    fit_in, held_in, corner_in = inside(fit_rows), inside(val_rows + hier_rows), inside(corner_rows)
    excluded = [r for r in val_rows if abs(r["f_de"]) >= F_DE_MAX]
    frac_out = 1.0 - len(fit_in) / len(fit_rows)
    print(f"  |f_DE(z={Z_DE_REF:g})| < {F_DE_MAX:g}: {len(fit_in)} fit, {len(held_in)} "
          f"held-out, {len(corner_in)} corners inside; {100 * frac_out:.1f}% of the box excluded")

    X = lambda rows: np.array([r["x"] for r in rows])
    Y = lambda rows: np.array([r["z_d"] for r in rows])
    pred = lambda rows, deg, coef: np.asarray(design(jnp.asarray(X(rows)), deg)) @ coef

    chosen = None
    for deg in range(1, args.max_degree + 1):
        coef, *_ = np.linalg.lstsq(np.asarray(design(jnp.asarray(X(fit_in)), deg)),
                                   Y(fit_in), rcond=None)
        rh = pred(held_in, deg, coef) - Y(held_in)
        rc = pred(corner_in, deg, coef) - Y(corner_in)
        print(f"  degree {deg} ({len(terms(deg))} terms): held-out max |dz_d| = "
              f"{np.max(np.abs(rh)):.4f} (rms {np.sqrt(np.mean(rh ** 2)):.4f}), "
              f"corners max {np.max(np.abs(rc)):.4f}")
        if (chosen is None and np.max(np.abs(rh)) <= TARGET_DZ
                and np.max(np.abs(rc)) <= TARGET_DZ_CORNERS):
            chosen = (deg, coef, rh, rc)
    if chosen is None:
        raise SystemExit(f"no degree <= {args.max_degree} reaches |dz_d| <= "
                         f"{TARGET_DZ} held-out and {TARGET_DZ_CORNERS} at the corners")
    deg, coef, rh, rc = chosen
    rx = pred(excluded, deg, coef) - Y(excluded) if excluded else np.array([np.nan])

    # What that is in r_d: the package's own integral at the fitted z_d,
    # against CLASS's rs_d -- and, separately, at CLASS's own z_d, which
    # isolates the integral (and CLASS's curvature factor) from the fit.
    test = held_in + corner_in
    d_fit, d_int, ok = [], [], []
    for r, z in zip(test, pred(test, deg, coef)):
        c, h = r["cosmo"], float(r["cosmo"].h)
        d_fit.append(np.log(float(sound_horizon(z, c)[0]) / h / r["rs_d"]))
        d_int.append(np.log(float(sound_horizon(r["z_d"], c)[0]) / h / r["rs_d"]))
        ok.append(r["p"]["Omega_k"])
    d_fit, d_int, ok = map(np.asarray, (d_fit, d_int, ok))
    flat = np.abs(ok) < 0.01
    print(f"  chosen degree {deg}: max |d ln r_d| against CLASS = {np.max(np.abs(d_fit)):.2e} "
          f"(|Omega_k| < 0.01: {np.max(np.abs(d_fit[flat])):.2e}); integral alone, at "
          f"CLASS's z_d: {np.max(np.abs(d_int)):.2e} (|Omega_k| < 0.01: "
          f"{np.max(np.abs(d_int[flat])):.2e}); excluded held-out points: max "
          f"|dz_d| = {np.nanmax(np.abs(rx)):.1f}")

    fid_row = hier_rows[0]
    bbn = bbn_record(fid_row)
    bbn["YHe_range_in_box"] = [float(min(r["YHe"] for r in fit_in)),
                               float(max(r["YHe"] for r in fit_in))]
    validation = {
        "target_abs_dz_heldout": TARGET_DZ,
        "target_abs_dz_corners": TARGET_DZ_CORNERS,
        "n_fit": len(fit_in), "n_heldout": len(held_in), "n_corners": len(corner_in),
        "n_corners_refused_by_class": len(corner_fail),
        "fraction_of_box_excluded_by_f_de": frac_out,
        "max_abs_dz_heldout": float(np.max(np.abs(rh))),
        "rms_dz_heldout": float(np.sqrt(np.mean(rh ** 2))),
        "max_abs_dz_corners": float(np.max(np.abs(rc))),
        "max_abs_dz_excluded_heldout": float(np.nanmax(np.abs(rx))),
        "max_abs_dlnrd_vs_class": float(np.max(np.abs(d_fit))),
        "max_abs_dlnrd_vs_class_flat": float(np.max(np.abs(d_fit[flat]))),
        "max_abs_dlnrd_integral_at_class_zd": float(np.max(np.abs(d_int))),
        "max_abs_dlnrd_integral_at_class_zd_flat": float(np.max(np.abs(d_int[flat]))),
        "rs_d_fiducial_class_mpc": float(fid_row["rs_d"]),
        "z_d_fiducial_class": float(fid_row["z_d"]),
    }
    print("BBN record:")
    pprint.pprint(bbn, sort_dicts=False, width=100)
    print("VALIDATION:")
    pprint.pprint(validation, sort_dicts=False, width=100)

    body = f'''"""Coefficients of :func:`ggah_mod.cosmology.drag.z_drag`.

Generated by ``tools/calibrate_zdrag.py`` (seed {args.seed}); do not edit.
Rerun that script after any change to CLASS, to its BBN table, or to
:func:`~ggah_mod.cosmology.power.class_input`.
"""

#: Total degree of the polynomial in ``drag.FEATURES``.
DEGREE = {deg}

#: Feature centres, in ``drag.FEATURES`` order.
CENTRE = {CENTRE!r}

#: Coefficients, in ``drag.terms(DEGREE)`` order.
COEFFS = {pprint.pformat([float(c) for c in coef], width=88)}

#: The box the fit was calibrated and validated on; ``drag.check_drag_domain``
#: refuses outside it.
BOX = {pprint.pformat(b, sort_dicts=False, width=88)}

#: And inside it, only where dark energy is below this share of the density at
#: z = {Z_DE_REF:g} (``drag.dark_energy_fraction_at_drag``).  Beyond it the fit
#: misses CLASS by up to {np.nanmax(np.abs(rx)):.0f} in z_d; it excludes
#: {100 * frac_out:.1f}% of the box above.
F_DE_MAX = {F_DE_MAX!r}

#: The accuracy reached, against CLASS {bbn["class_version"]}.
VALIDATION = {pprint.pformat(validation, sort_dicts=False, width=88)}

#: Every assumption behind the helium fraction CLASS used, read from CLASS and
#: its table at calibration time.
BBN = {pprint.pformat(bbn, sort_dicts=False, width=88)}
'''
    OUT.write_text(body)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
