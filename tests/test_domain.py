"""What the differentiable flavour does at its boxes' edges, and what ACCURATE does under JAX.

Eagerly, emu_pk, the drag-redshift fit and, when installed, the optional
emu_hmf raise outside their boxes.
Under ``jax.jit`` / ``jax.grad`` the checks are skipped and each fit
extrapolates: a finite value and a finite gradient, never ``nan`` and never a
clamp.  :func:`ggah_mod.cosmology.domain.in_domain` gives the box as a traced
boolean for a log-posterior to use as a prior boundary.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggah_mod.cosmology import PLANCK18, make_pk
from ggah_mod.cosmology.domain import domain_margins, in_domain
from ggah_mod.cosmology.drag import r_drag
from ggah_mod.halos.mass_function import fsigma_tinker08_csst

try:
    import emu_hmf  # noqa: F401  (the optional `hmf` extra)
    HAS_HMF = True
except ImportError:
    HAS_HMF = False

K = np.logspace(-3, 0, 20)
SIGMA = jnp.array([0.5, 1.0, 2.0])

#: (what, eager call, traced function of one scalar, a value outside, a value inside)
CASES = {
    "emu_pk": (lambda c: make_pk("emu_pk").pk(K, 0.0, c),
               lambda w0: jnp.sum(make_pk("emu_pk").pk(K, 0.0, PLANCK18.replace(w0=w0))),
               dict(w0=-0.4), -0.4, -1.0),
    "emu_hmf": (lambda c: fsigma_tinker08_csst(SIGMA, 0.0, c),
                lambda w0: jnp.sum(fsigma_tinker08_csst(SIGMA, 0.0, PLANCK18.replace(w0=w0))),
                dict(w0=-1.45), -1.45, -1.0),
    "z_drag": (lambda c: r_drag(c),
               lambda m: r_drag(PLANCK18.replace(sum_mnu=m)),
               dict(sum_mnu=2.0), 2.0, 0.06),
}


def _need(name):
    if name == "emu_hmf" and not HAS_HMF:
        pytest.skip("emu_hmf (the optional `hmf` extra) is not installed")


@pytest.mark.parametrize("name", CASES)
def test_eager_refuses_outside(name):
    _need(name)
    eager, _, outside, _, _ = CASES[name]
    with pytest.raises(ValueError, match="outside"):
        eager(PLANCK18.replace(**outside))


@pytest.mark.parametrize("name", CASES)
def test_traced_extrapolates_finite(name):
    """No nan, no clamp: the value moves and the gradient is not zero."""
    _need(name)
    _, traced, _, x_out, _ = CASES[name]
    f = jax.jit(traced)
    v, g = f(x_out), jax.grad(traced)(x_out)
    assert np.isfinite(float(v)) and np.isfinite(float(g))
    assert float(g) != 0.0
    assert float(f(x_out * 1.01)) != float(v)


@pytest.mark.parametrize("name", CASES)
def test_in_domain_flags_the_edge_under_jit(name):
    _need(name)
    _, _, outside, _, _ = CASES[name]
    (field, x_out), = outside.items()
    hmf = name == "emu_hmf"
    flag = jax.jit(lambda x: in_domain(PLANCK18.replace(**{field: x}), hmf=hmf))
    assert bool(flag(x_out)) is False
    assert bool(in_domain(PLANCK18)) is True


def test_margins_are_signed_fractions_of_the_box():
    m = domain_margins(PLANCK18.replace(w0=-0.4), pk=True, hmf=False, drag=False)
    assert float(m["pk_w0"]) == pytest.approx(-0.1, abs=1e-3)


def test_prior_boundary_gradient_is_clean():
    """`jnp.where(in_domain, ...)` keeps the gradient finite on both sides."""
    def log_post(w0):
        return jnp.where(in_domain(PLANCK18.replace(w0=w0)), -0.5 * (w0 + 1.0) ** 2, -jnp.inf)
    assert float(jax.grad(log_post)(-0.9)) == pytest.approx(-0.1)
    assert float(jax.grad(log_post)(-0.4)) == 0.0
    assert float(log_post(-0.4)) == -np.inf


@pytest.mark.parametrize("transform", [jax.grad, jax.jit])
def test_accurate_under_jax_names_the_flavour(transform):
    pytest.importorskip("classy")
    from ggah_mod.cosmology.power import NotDifferentiableError
    pk = make_pk("class")
    f = lambda h: jnp.sum(jnp.asarray(pk.pk(K, 0.0, PLANCK18.replace(h=h))))
    with pytest.raises(NotDifferentiableError, match="DIFFERENTIABLE"):
        transform(f)(0.67)


def test_not_differentiable_is_a_type_error():
    from ggah_mod.cosmology.power import NotDifferentiableError
    assert issubclass(NotDifferentiableError, TypeError)
