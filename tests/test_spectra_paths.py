r"""Verification: layer 4 is differentiable everywhere, and **computes nothing**.

Layer 2 permits exactly one non-differentiable piece and pins it with an AST
audit.  Layer 3 permits none, and asserts its whole-module exemption set is
empty.  Layer 4 makes the strongest claim of the three: its allow-list has no
entries **at all**, not even the construction-time ones layer 3 admits, because
there is nothing here to construct.  It reads no table, memoises no quadrature
rule and holds no data -- it integrates arrays other layers built.

That is worth asserting rather than assuming.  The first numpy call to appear in
this layer will be a convenience during a debugging session, and it will sever
every gradient that passes through it while returning a number that looks right.
"""
import ast
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.spectra
from tests.test_paths import _numpy_uses

SPECTRA_DIR = pathlib.Path(ggah_mod.spectra.__file__).parent

#: Empty, and it must stay empty.
#:
#: Layer 3's equivalent has four entries, all in ``cooling``, and all of one of
#: two kinds: a constant built once at construction, or a table read off disk.
#: Layer 4 has neither kind of thing in it.  An entry appearing here means
#: either that the layer has grown a responsibility that belongs elsewhere, or
#: that a gradient has been broken.
SPECTRA_NUMPY_ALLOWED: dict[str, str] = {}


class TestNoNumpyOnTheTracedPath:
    MODULES = sorted(SPECTRA_DIR.glob("*.py"))

    def test_there_are_modules_to_check(self):
        """Guards the parametrised loop against checking nothing."""
        assert len(self.MODULES) >= 5

    @pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
    def test_every_numpy_call_is_accounted_for(self, path):
        unexplained = [f"{path.stem}.{qual}:{line} -> np.{attr}"
                       for qual, line, attr in _numpy_uses(path)
                       if qual and f"{path.stem}.{qual}" not in SPECTRA_NUMPY_ALLOWED]
        assert not unexplained, (
            "numpy inside a traced layer-4 function:\n  "
            + "\n  ".join(unexplained))

    def test_the_layer_has_no_exemptions_at_all(self):
        """The headline claim, as one line.

        Stronger than layer 3's: this layer builds nothing and reads nothing,
        so it has no legitimate use for numpy even at construction.
        """
        assert SPECTRA_NUMPY_ALLOWED == {}

    def test_the_layer_does_not_import_numpy(self):
        """Structural, and it catches the module-level constant an AST walk
        over *functions* would let through."""
        for path in self.MODULES:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(n.split(".")[0] == "numpy" for n in names), (
                    f"{path.name} imports numpy")


class TestTheLayerOwnsNoGrid:
    """It integrates over grids the halo field owns, and builds none.

    Five classes in the predecessor each built their own ``logspace(10, 16, N)``
    with ``N`` in {256, 512, 600}.  ``tests/test_sector_coherence.py`` asserts
    layer 3 does not; the same has to be true one layer up, where the temptation
    is a private k grid for a transform.
    """

    def test_no_module_builds_a_mass_or_wavenumber_grid(self):
        for path in sorted(SPECTRA_DIR.glob("*.py")):
            code = ast.dump(ast.parse(path.read_text()))
            for maker in ("logspace", "linspace", "geomspace"):
                assert maker not in code, f"{path.name} builds a grid"

    def test_the_mass_integral_has_one_implementation(self):
        """`HaloField.integrate` or nothing.  A second `trapezoid` written at a
        call site is how the `dM` versus `dlnM` choice gets made twice."""
        for path in sorted(SPECTRA_DIR.glob("*.py")):
            code = ast.dump(ast.parse(path.read_text()))
            assert "trapezoid" not in code, (
                f"{path.name} integrates over mass itself; use "
                f"HaloField.integrate, which is the one implementation")


class TestItSurvivesTheTransformations:
    """Declarations checked by trying them."""

    def test_the_spectrum_is_a_pytree_with_only_array_leaves(self):
        from ggah_mod.spectra.pk import PowerSpectrum
        p = PowerSpectrum(k=jnp.ones(3), one_halo=jnp.ones(3),
                          two_halo=jnp.ones(3), shot=jnp.asarray(0.1),
                          a="x", b="y")
        leaves = jax.tree_util.tree_leaves(p)
        assert len(leaves) == 4
        assert all(not isinstance(x, (str, bool)) for x in leaves)
        rebuilt = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(p), leaves)
        assert rebuilt.a == "x" and rebuilt.b == "y"

    def test_the_static_declarations_expose_no_numeric_leaves(self):
        """`Band`, `Component`, `TracerSpec` and `PkOptions` live in a treedef.

        None of them is a registered pytree node, so JAX treats each as opaque
        -- the object is its own single leaf, and its float fields are *not*
        reachable.  That is the property being asserted, and it is the safe
        one: a band edge that leaked out as a leaf would become differentiable,
        and an `ell` array would become a traced value, either of which breaks
        `jacfwd`'s fixed output shape.  Putting one of these into a traced
        pytree by mistake fails loudly at the boundary rather than quietly.
        """
        from ggah_mod.spectra.spec import (
            Band, Component, PkOptions, TracerSpec,
        )
        for obj in (Band(0.5, 2.0), Component("gas", "xray"),
                    TracerSpec.one("matter"), PkOptions()):
            leaves = jax.tree_util.tree_leaves(obj)
            assert leaves in ([], [obj]), type(obj).__name__
            assert not any(isinstance(x, (float, int, jnp.ndarray, np.ndarray))
                           for x in leaves), type(obj).__name__

    def test_the_static_declarations_are_hashable(self):
        """They are dict keys in the resolver, and treedef entries under jit."""
        from ggah_mod.spectra.spec import (
            Band, Component, PkOptions, TracerSpec,
        )
        for obj in (Band(0.5, 2.0), Component("gas", "xray"),
                    TracerSpec.one("matter"), PkOptions()):
            assert isinstance(hash(obj), int)


class TestLayerFourIsOneEpoch:
    r"""A redshift-stacked weight is refused rather than absorbed.

    Layer 4's contractions are ``(Nk, NM)``, and `normalised_parts` tells that
    from a bare ``(NM,)`` by the leading axis: ``atleast_2d`` then
    ``shape[0] == 1``.  Give it a ``(Nz, Nk, NM)`` weight -- which is what a
    sector built on ``jax.vmap(make_field)``'s output produces -- and the test
    reads ``Nz != 1``, declines to broadcast, and hands the pair rule an array
    whose first axis is a redshift being treated as a wavenumber.  Nothing
    downstream looks wrong.
    """

    @staticmethod
    def _w(shape):
        import ggah_mod.sectors as S
        return S.TracerWeights(
            w_point=jnp.full(shape, 0.3), w_extended=None,
            norm=jnp.asarray(2.0), discrete=True, bias_weight=None,
            name="stacked")

    def test_a_rank_three_weight_is_refused(self):
        from ggah_mod.spectra.pair import normalised_parts
        with pytest.raises(ValueError, match="redshift axis|\\(NM,\\)"):
            normalised_parts(self._w((3, 4, 6)), n_k=4)

    def test_the_two_supported_ranks_still_work(self):
        """Guards against over-refusing: both legal shapes must still pass, and
        the bare one must still be broadcast up."""
        from ggah_mod.spectra.pair import normalised_parts
        point, _ = normalised_parts(self._w((6,)), n_k=4)
        assert point.shape == (4, 6)
        point, _ = normalised_parts(self._w((4, 6)), n_k=4)
        assert point.shape == (4, 6)
