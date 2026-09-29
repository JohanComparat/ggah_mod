"""Tables the documentation reads off the code at build time.

A table typed into a page is true on the day it is typed.  These are written
from the objects themselves on every build -- the three flavours, the fiducial
cosmology, the power-spectrum backends and the three layer-2 registries -- into
``docs/_generated/``, which the pages pull in with ``{include}``.  A registry
entry without a citation in :mod:`registry_citations` stops the build.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib

from sphinx.errors import ExtensionError

import registry_citations as CITE

OUT = pathlib.Path(__file__).resolve().parents[1] / "_generated"


def _cite(keys) -> str:
    return "; ".join(f"{{cite:t}}`{k}`" for k in keys)


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _code(x) -> str:
    return f"`{x}`"


def _zrange(z) -> str:
    lo, hi = z
    hi = "∞" if math.isinf(hi) else f"{hi:g}"
    return f"{lo:g}–{hi}"


def _require(registry: str, names, cited: dict) -> None:
    missing = sorted(set(names) - set(cited))
    stale = sorted(set(cited) - set(names))
    if missing or stale:
        raise ExtensionError(
            f"docs/_ext/registry_citations.py::{registry} is out of step with the "
            f"code: uncited entries {missing}, cited entries that no longer exist "
            f"{stale}.")


def flavours() -> str:
    from ggah_mod.backend import ACCURATE, DIFFERENTIABLE, DIFFERENTIABLE_COARSE

    flav = (ACCURATE, DIFFERENTIABLE, DIFFERENTIABLE_COARSE)
    rows = [
        ("differentiable (`traced`)", *(_code(b.traced) for b in flav)),
        ("linear $P(k)$ backend (`pk`)", *(_code(b.pk) for b in flav)),
        ("$k$ grid [$h\\,{\\rm Mpc}^{-1}$]",
         *(f"{b.n_k} points, {b.k_min:g}–{b.k_max:g}" for b in flav)),
        ("mass grid [$h^{-1}{\\rm M}_\\odot$]",
         *(f"{b.n_m} points, {b.m_min:.0e}–{b.m_max:.0e}" for b in flav)),
        ("mass definition (`mdef`)", *(_code(b.mdef) for b in flav)),
        ("multiplicity (`hmf_model`)", *(_code(b.hmf_model) for b in flav)),
        ("linear bias (`bias_model`)", *(_code(b.bias_model) for b in flav)),
        ("concentration (`cm_model`)", *(_code(b.cm_model) for b in flav)),
        ("beyond-linear bias (`bnl`)", *(_code(b.bnl) for b in flav)),
        ("Hankel engine (`hankel`)", *(_code(b.hankel) for b in flav)),
    ]
    return _table(("", "`ACCURATE`", "`DIFFERENTIABLE`", "`DIFFERENTIABLE_COARSE`"),
                  rows)


def planck18() -> str:
    from ggah_mod.cosmology import PLANCK18, Cosmology

    default = Cosmology.create()
    rows = [(_code(f.name), _code(getattr(PLANCK18, f.name)),
             _code(getattr(default, f.name)))
            for f in dataclasses.fields(PLANCK18)]
    return _table(("field", "`PLANCK18`", "`Cosmology.create()`"), rows)


def pk_backends() -> str:
    from ggah_mod.cosmology.power import PK_BACKENDS

    rows = [(_code(name), f"{{py:class}}`~{cls.__module__}.{cls.__name__}`",
             _code(cls.differentiable), _code(cls.supports_curvature),
             _code(cls.has_native_z))
            for name, cls in PK_BACKENDS.items()]
    return _table(("`make_pk(...)`", "class", "`differentiable`",
                   "`supports_curvature`", "`has_native_z`"), rows)


def multiplicity() -> str:
    from ggah_mod.halos.mass_function import (
        CALIBRATION, COSMOLOGY_DEPENDENT_MULTIPLICITY, MULTIPLICITY)

    _require("MULTIPLICITY", MULTIPLICITY, CITE.MULTIPLICITY)
    rows = []
    for name in MULTIPLICITY:
        halos, z = CALIBRATION[name]
        rows.append((_code(name), _cite(CITE.MULTIPLICITY[name]), halos,
                     _zrange(z),
                     "yes" if name in COSMOLOGY_DEPENDENT_MULTIPLICITY else ""))
    return _table(("`hmf_model`", "reference", "calibrated on", "$z$",
                   "reads the cosmology"), rows)


def bias() -> str:
    from ggah_mod.halos.linear_bias import BIAS, MATCHED_BIAS

    _require("BIAS", BIAS, CITE.BIAS)
    partners: dict[str, list[str]] = {}
    for hmf, b in MATCHED_BIAS.items():
        partners.setdefault(b, []).append(hmf)
    rows = [(_code(name), _cite(CITE.BIAS[name]),
             ", ".join(_code(h) for h in partners.get(name, [])))
            for name in BIAS]
    return _table(("`bias_model`", "reference", "peak-background-split partner of"),
                  rows)


def concentration() -> str:
    from ggah_mod.halos.concentration import (
        CM_CALIBRATION, CONCENTRATION, PEAK_HEIGHT_MODELS)

    _require("CONCENTRATION", CONCENTRATION, CITE.CONCENTRATION)
    rows = []
    for name in CONCENTRATION:
        sim, mdefs, z, _ = CM_CALIBRATION[name]
        family = "peak height" if name in PEAK_HEIGHT_MODELS else "power law in $(M, z)$"
        rows.append((_code(name), _cite(CITE.CONCENTRATION[name]), family, sim,
                     mdefs, _zrange(z)))
    return _table(("`cm_model`", "reference", "family", "simulations", "definitions",
                   "$z$"), rows)


TABLES = {
    "flavours.md": flavours,
    "planck18.md": planck18,
    "pk_backends.md": pk_backends,
    "multiplicity.md": multiplicity,
    "bias.md": bias,
    "concentration.md": concentration,
}


def write_all(app=None) -> None:
    OUT.mkdir(exist_ok=True)
    for name, make in TABLES.items():
        text = make()
        path = OUT / name
        # Rewriting an unchanged file would mark every including page stale.
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", write_all)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
