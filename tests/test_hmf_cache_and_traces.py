"""The recalibration must not cache a value made inside a trace.

`halos/mass_function.py::_hmf_correction` is `lru_cache`d on
`(mdef, check_box)`, which is right: the weights are read from disk and the
object is stateless once built.  What was not right is *where* it got built.
`HmfCorrection.__init__` does `jnp.asarray` over the weight arrays, and the
first call reaches it from inside a `jit` the moment a traced forward model asks
for the default mass function.  Cache a value made under a trace, and the next
trace leaks it.

It needs an **abandoned** trace to bite, which is why it survived: one, two or
three ordinary traces in a fresh process are fine.  This package abandons traces
routinely -- it refuses rather than degrades, and a refusal inside a trace is an
exception inside a trace.

The symptom was `pytest tests/test_field.py -n auto` failing where the same file
passes serially, because serially some earlier test warms the cache from outside
a trace first.  That is a bad thing to rely on and a worse thing to test, so
this looks at what was cached instead.

Verified sharp by reverting the fix: `test_the_cached_object_holds_no_tracer`
turns red.  The end-to-end scenario below it does not, at this size, and says so
-- a test whose docstring overstates what it catches is worse than no test,
because the next person trusts it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.halos.mass_function import _hmf_correction


@pytest.fixture
def cold_cache():
    """A cache in the state a fresh process has, restored afterwards."""
    _hmf_correction.cache_clear()
    yield
    _hmf_correction.cache_clear()


def _sum_dndm(pk, om):
    return jnp.sum(make_field(PLANCK18.replace(Omega_m=om),
                              DIFFERENTIABLE, pk, z=0.0).dndm)


def test_a_trace_that_dies_does_not_poison_the_cache(cold_cache):
    """The scenario, end to end.

    Honest about what it is worth: with the fix reverted this one still passes,
    because two traces in one process are not enough to make the leak surface --
    which is exactly why the defect survived.  It is here because it is the
    sequence that actually happens, and because if the leak ever *does* start
    biting at this size, this is where it will show.  The sharp check is the
    next test, which looks at what was cached instead of waiting for it to hurt.
    """
    pk = make_pk("emu_pk")

    @jax.jit
    def doomed(om):
        out = _sum_dndm(pk, om)
        raise RuntimeError("this trace is abandoned, as a refusal would be")

    with pytest.raises(RuntimeError):
        doomed(0.31)

    # The cache now holds whatever that trace built.  A fresh trace must still
    # work: in the suite, before the fix, this raised UnexpectedTracerError
    # pointing at `_hmf_correction` -- from a line that only reads weights off
    # disk.
    assert float(jax.jit(lambda om: _sum_dndm(pk, om))(0.32)) > 0.0


def test_the_cached_object_holds_no_tracer(cold_cache):
    """Directly: build it under a trace, then look at what was cached."""
    pk = make_pk("emu_pk")

    @jax.jit
    def build(om):
        return _sum_dndm(pk, om)

    build(0.31)
    corr = _hmf_correction("200m", True)
    leaked = [k for k, v in corr._p.items() if isinstance(v, jax.core.Tracer)]
    assert not leaked, (
        f"the cached correction holds tracers for {leaked}; it was built inside "
        "a trace, so the next trace will leak it")


def test_the_cache_is_still_a_cache(cold_cache):
    """The fix must not turn a cached load into a per-call one."""
    a = _hmf_correction("200m", True)
    b = _hmf_correction("200m", True)
    assert a is b


def test_check_box_still_keys_the_cache(cold_cache):
    """The property the second cache key exists for."""
    assert _hmf_correction("200m", True) is not _hmf_correction("200m", False)
