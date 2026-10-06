r"""``numerics.arctan``: the package's arctangent, and why it is not ``jnp.arctan``.

jaxlib 0.10.2 -- what Python 3.11 resolves to, and what the shared development
environment is held at -- miscompiles the ``atan`` primitive on a CPU with
vector instructions.  On 64 elements or more half the outputs come back zero;
inside a fused kernel it fails on a dozen, which is how it was found: the
Hernquist lensing profile came out negative.  These tests are written so that
they fail on that jaxlib if ``numerics.arctan`` were ever the primitive again.
"""
import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ggah_mod
from ggah_mod.numerics import arctan


def _inputs():
    """1000+ values over twelve decades, both signs, and the edges."""
    rng = np.random.default_rng(0)
    mag = 10 ** rng.uniform(-6, 6, 500)
    return np.concatenate([mag, -mag[::-1], [0.0, 1.0, -1.0, 1e200, -1e200]])


class TestAgreesWithNumpy:

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
    def test_value(self, compiled):
        x = _inputs()
        f = jax.jit(arctan) if compiled else arctan
        got = np.asarray(f(jnp.asarray(x)))
        np.testing.assert_allclose(got, np.arctan(x), rtol=1e-15, atol=1e-300)

    def test_infinities(self):
        got = np.asarray(jax.jit(arctan)(jnp.asarray([np.inf, -np.inf])))
        np.testing.assert_array_equal(got, [np.pi / 2, -np.pi / 2])

    def test_gradient(self):
        """Exactly :math:`1/(1+x^2)`, with no ``nan`` leaking in from the
        branch that is not taken."""
        x = _inputs()[:-2]                       # 1e200 squared underflows the answer
        g = np.asarray(jax.jit(jax.vmap(jax.grad(arctan)))(jnp.asarray(x)))
        np.testing.assert_allclose(g * (1.0 + x * x), 1.0, rtol=1e-15)

    def test_inside_a_fused_kernel(self):
        """The case that failed: the arctangent in an expression XLA fuses."""
        x = _inputs()
        x = x[(x > 1.0) & (x < 1e100)]
        a = np.sqrt(x ** 2 - 1.0)
        want = ((2.0 + x ** 2) * np.arctan(a) / a - 3.0) / (x ** 2 - 1.0) ** 2

        @jax.jit
        def kernel(xx):
            aa = jnp.sqrt(xx ** 2 - 1.0)
            return ((2.0 + xx ** 2) * arctan(aa) / aa - 3.0) / (xx ** 2 - 1.0) ** 2

        np.testing.assert_allclose(np.asarray(kernel(jnp.asarray(x))), want,
                                   rtol=1e-12)

    def test_single_precision(self):
        x = _inputs()[:-4]
        got = np.asarray(jax.jit(arctan)(jnp.asarray(x, jnp.float32)))
        np.testing.assert_allclose(got, np.arctan(x.astype(np.float32)),
                                   rtol=3e-7, atol=1e-30)


class TestThePrimitiveIsNotUsed:
    """``jnp.arctan`` anywhere in the package is a result that is wrong on
    jaxlib 0.10.2 and right everywhere the tests usually run.

    The one-argument forms only.  ``jnp.arctan2(y, x)`` of two arrays compiles
    correctly; ``jnp.arctan2(y, 1.0)`` is the bug again, and is for review to
    catch.
    """

    BANNED = {("jnp", "arctan"), ("jnp", "atan"), ("lax", "atan")}

    def test_no_module_calls_it(self):
        root = pathlib.Path(ggah_mod.__file__).parent
        bad = []
        for f in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text())):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and (node.value.id, node.attr) in self.BANNED):
                    bad.append(f"{f.relative_to(root)}:{node.lineno} "
                               f"{node.value.id}.{node.attr}")
        assert not bad, (
            "the atan primitive, which jaxlib 0.10.2 miscompiles; use "
            "ggah_mod.numerics.arctan:\n  " + "\n  ".join(bad))
