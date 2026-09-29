r"""Verification: layer 5's numpy is all construction, and all of it is named.

Layer 4's allow-list is **empty** -- it builds nothing and reads nothing.  Layer
5 cannot make that claim, and should not pretend to: Ogata's abscissae are the
zeros of :math:`J_\nu`, FFTLog's coefficients are ratios of :math:`\Gamma` at
complex argument, and the double-exponential nodes behind :math:`K_\nu` are
``sinh``/``cosh`` of a fixed grid.  All three are ``scipy`` and ``numpy``.

The claim that *is* made, and checked here, is narrower and more useful: **every
one of them is a function of static grid choices alone**, so each is built once,
before anything is traced, and none is ever evaluated at a traced argument.  The
allow-list therefore has exactly one category -- ``construction`` -- and every
entry must say what makes it one.

That is the whole reason this layer needs no Bessel function on the
differentiable path, and why two independent transform engines cost little more
than one.
"""
import ast
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.observables
from tests.test_paths import _numpy_uses

OBS_DIR = pathlib.Path(ggah_mod.observables.__file__).parent

#: Every symbol in `ggah_mod/observables` allowed to call numpy in a function.
#:
#: One category only -- **construction** -- and each entry has to say why it is
#: one.  There is no "table I/O" entry, because this layer reads nothing.
OBSERVABLES_NUMPY_ALLOWED: dict[str, str] = {
    "transforms._ogata_nodes":
        "construction: the zeros of J_nu and the double-exponential weights, "
        "functions of (nu, N, h) alone -- memoised, and a constant thereafter",
    "transforms._fftlog_setup":
        "construction: the u_m coefficients and the reciprocal output grid, "
        "functions of (nu, kernel, q, N, Delta, x_0) alone",
    "transforms._de_nodes":
        "construction: the double-exponential abscissae behind bessel_k, a "
        "function of (n, h) alone",
    "transforms._mellin_bessel_j":
        "construction: called only from _fftlog_setup, at fixed complex z",
    "transforms._mellin_spherical_bessel_j":
        "construction: called only from _fftlog_setup, at fixed complex z",
    "transforms.make_fftlog":
        "construction: validates the input grid is log-spaced and computes its "
        "spacing and its dtype, once, from concrete floats -- the dtype "
        "deliberately before the widening to float64, since that is the "
        "precision the tolerance has to be set from",
    "transforms._log_spacing_tol":
        "construction: eps of the input grid's dtype, a function of that dtype "
        "and the grid's static geometry alone -- no array reaches it",
}


class TestEveryNumpyCallIsConstruction:
    MODULES = sorted(OBS_DIR.glob("*.py"))

    def test_there_are_modules_to_check(self):
        assert len(self.MODULES) >= 3

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
    def test_every_numpy_call_is_accounted_for(self, path):
        unexplained = [
            f"{path.stem}.{qual}:{line} -> np.{attr}"
            for qual, line, attr in _numpy_uses(path)
            if qual and f"{path.stem}.{qual}" not in OBSERVABLES_NUMPY_ALLOWED]
        assert not unexplained, (
            "numpy inside a traced layer-5 function, with no entry in "
            "OBSERVABLES_NUMPY_ALLOWED:\n  " + "\n  ".join(unexplained) +
            "\n\nLayer 5 admits exactly one category, `construction`: a "
            "quantity that is a function of static grid choices alone and is "
            "built before anything is traced.  If it is not one of those, it "
            "is a hole in the differentiable path.")

    def test_the_allow_list_has_no_dead_entries(self):
        seen = set()
        for path in self.MODULES:
            for qual, _, _ in _numpy_uses(path):
                if qual:
                    seen.add(f"{path.stem}.{qual}")
        assert not (set(OBSERVABLES_NUMPY_ALLOWED) - seen), (
            "a stale exemption is how a real hole gets waved through later: "
            f"{sorted(set(OBSERVABLES_NUMPY_ALLOWED) - seen)}")

    def test_every_entry_says_construction(self):
        """One category, and the reason has to name it."""
        for name, why in OBSERVABLES_NUMPY_ALLOWED.items():
            assert why.startswith("construction:"), name

    def test_the_real_space_module_touches_no_numpy_at_all(self):
        """It composes transforms; it computes nothing of its own."""
        assert not [q for q, _, _ in _numpy_uses(OBS_DIR / "real_space.py") if q]


class TestNothingStaticIsBuiltPerCall:
    """Construction means *once*, and memoisation is what makes that true."""

    def test_the_ogata_rule_is_memoised(self):
        from ggah_mod.observables.transforms import _ogata_nodes
        assert _ogata_nodes(0.5, 32, 0.01)[0] is _ogata_nodes(0.5, 32, 0.01)[0]

    def test_the_fftlog_coefficients_are_memoised(self):
        from ggah_mod.observables.transforms import _fftlog_setup
        a = _fftlog_setup(0.0, "bessel", 2.0, 1.0, 32, 0.2, -4.0)
        assert a[1] is _fftlog_setup(0.0, "bessel", 2.0, 1.0, 32, 0.2, -4.0)[1]

    def test_no_module_imports_plain_scipy_at_module_scope(self):
        """Every ``scipy`` import sits inside a construction helper, so
        importing the layer does not put one anywhere a trace could reach.

        ``jax.scipy`` is a different thing and is fine at module scope -- it is
        JAX's own reimplementation, traced and differentiable, and
        :mod:`~ggah_mod.observables.beams` uses its ``gammaln``.  The test
        compares the *root* module, so ``jax.scipy.special`` does not read as
        ``scipy``: a substring check flags it, which is how a real rule turns
        into a rule nobody trusts.
        """
        def root(name):
            return (name or "").split(".")[0]

        for path in sorted(OBS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    assert root(node.module) != "scipy", path.name
                elif isinstance(node, ast.Import):
                    assert all(root(a.name) != "scipy"
                               for a in node.names), path.name


class TestTheLayerBoundaryRunsOneWay:
    def test_nothing_imports_the_observables_layer(self):
        """Layers 1-4 must not reach up into layer 5."""
        root = pathlib.Path(ggah_mod.observables.__file__).parent.parent
        for sub in ("cosmology", "halos", "sectors", "spectra"):
            for path in (root / sub).glob("*.py"):
                body = path.read_text().split('"""')[-1]
                assert "observables" not in body, f"{sub}/{path.name}"

    def test_it_owns_no_mass_grid(self):
        """The grids belong to `HaloField`; this layer builds r, chi and pi
        grids of its own and must build no others."""
        for path in sorted(OBS_DIR.glob("*.py")):
            code = ast.dump(ast.parse(path.read_text()))
            assert "logspace(10" not in code, path.name
