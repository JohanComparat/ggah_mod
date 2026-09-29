"""Shared fixtures.  x64 is enabled before any ggah_mod import.

Module-level ``jnp`` constants (the Gauss-Legendre nodes in
``cosmology.background``) are materialised at first import with whatever
precision is active, so flipping the flag afterwards silently leaves them at
float32 and degrades every finite-difference comparison in the suite.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import pytest

from ggah_mod.cosmology import Cosmology, PLANCK18


@pytest.fixture(scope="session")
def cosmo():
    return PLANCK18


@pytest.fixture(scope="session")
def massless():
    return Cosmology.create(sum_mnu=0.0)


# ==========================================================================
# A cheap analytic spectrum, for tests only
# ==========================================================================
class AnalyticPk:
    """A differentiable :math:`P(k)` with no data file and no solver.

    Replaces ``Eh98Pk()`` throughout the suite.  The EH98 *backend* is
    gone -- an order of magnitude less accurate than everything else, and a
    faithful transcription of a paper, so it earned neither its place in the
    package nor its pages in the document -- but a large part of what it was
    doing here was being the cheapest differentiable spectrum available: no
    network to load, no table to read, microseconds per call.

    This is that, and nothing else.  Its shape is BBKS (Bardeen et al. 1986,
    Eq. G3), not anything EH98 left behind: the no-wiggle fit went with the
    backend once ``diemer19`` replaced ``diemer15`` and removed its last
    consumer, so :mod:`ggah_mod.cosmology.transfer_eh98` no longer exists.  It
    is **not** registered in ``PK_BACKENDS``, and it makes no accuracy claim --
    no test should ever compare it to CLASS.

    ``has_native_z = False``, deliberately: that is a capability some code has
    to refuse rather than work around, and with the EH98 backend gone this is
    the only object in the tree that declares it.
    """

    name = "analytic"
    has_native_z = False
    differentiable = True

    @staticmethod
    def _require_z0(z):
        import jax.numpy as jnp
        from ggah_mod.cosmology.power import concrete
        zc = concrete(jnp.max(jnp.abs(jnp.asarray(z))))
        if zc is not None and zc > 0.0:
            raise ValueError(
                "AnalyticPk has no redshift dependence (has_native_z = False); "
                "it can only return z = 0.")

    #: sigma_8 the spectrum is normalised to at the fiducial cosmology.
    #:
    #: Not cosmetic.  Several tests assert that the tables built on this
    #: spectrum are *physical* -- that the peak height crosses unity somewhere
    #: inside the mass grid, that the effective bias is of order one.  An
    #: arbitrary amplitude makes those assertions fail for a reason that has
    #: nothing to do with what they are testing: at the first guess here,
    #: nu(10^10 Msun) came out at 3900 instead of ~0.5.
    SIGMA8 = 0.8111

    _norm = None

    @classmethod
    def _normalisation(cls):
        """Scale the shape to :data:`SIGMA8`, once, at the **fiducial**.

        Always at ``PLANCK18``, never at the cosmology being evaluated: this is
        called from inside ``jax.grad``, where the cosmology is a tracer and
        ``np.asarray`` of it raises.  A constant is also what it should be --
        a normalisation that moved with the cosmology would silently cancel the
        amplitude dependence the gradient tests are checking.
        """
        if cls._norm is None:
            import numpy as np
            from ggah_mod.cosmology.amplitude import sigma8
            # Materialised outside whatever trace is running.  Under `jit`
            # every `jnp` operation yields a tracer -- constants included --
            # so computing this lazily meant computing it inside a trace
            # whenever a traced test happened to be the first to ask, and
            # `np.asarray` of a tracer raises.  Serially something always
            # warmed it from outside first; under `-n auto` a worker reaches it
            # cold, which is how `test_beyond_linear_bias` came to fail in
            # parallel and pass serially.
            #
            # The same defect, and the same fix, as
            # `halos/mass_function.py::_hmf_correction`.  A lazily cached
            # constant is unsafe exactly when its first use can be inside a
            # trace, and in this package it always can.
            with jax.ensure_compile_time_eval():
                k = np.logspace(-4.0, 2.0, 2048)
                shape = np.asarray(cls._shape(k, PLANCK18))
                cls._norm = float(cls.SIGMA8 ** 2 / sigma8(shape, k) ** 2)
        return cls._norm

    @staticmethod
    def _shape(k, cosmo):
        """``k^n_s T_BBKS^2(k)`` -- Bardeen et al. (1986), Eq. G3.

        Smooth, monotonic in its log-slope, and analytic, which is all this
        stub needs.  A_s enters so that a gradient with respect to the
        amplitude is non-zero.
        """
        import jax.numpy as jnp
        k = jnp.asarray(k)
        gamma = cosmo.Omega_m * cosmo.h          # shape parameter
        q = k / gamma                            # k in h/Mpc, so h cancels
        t = (jnp.log(1.0 + 2.34 * q) / (2.34 * q)
             * (1.0 + 3.89 * q + (16.1 * q) ** 2
                + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25)
        return jnp.exp(cosmo.ln10A_s) * 1e-10 * k ** cosmo.n_s * t ** 2

    def _pk(self, k, cosmo):
        return self._shape(k, cosmo) * self._normalisation()

    def pk(self, k, z, cosmo):
        self._require_z0(z)
        return self._pk(k, cosmo)

    def pk_cb(self, k, z, cosmo):
        self._require_z0(z)
        return self._pk(k, cosmo)


class CountingPk(AnalyticPk):
    r"""`AnalyticPk` with a redshift dependence and a solve counter.

    Two things `AnalyticPk` deliberately lacks, and a redshift-stack test needs
    both.

    ``has_native_z = True``, because the growth-parameterised concentration
    relations refuse a backend without one --- and those are exactly the
    relations where batching pays, since they make layer-1 calls of their own on
    top of the field's two.

    :attr:`solves` records the distinct redshift **tuples** asked for.  That is
    what ``_BoltzmannBase._solve_cached`` is keyed on, so ``len(pk.solves)`` is
    the number of Boltzmann solves the same call sequence would have cost ---
    measured in milliseconds, without a Boltzmann solver.  :attr:`calls` records
    every call in order, which is what catches a loop that a cache would hide.

    The growth is scale-independent by construction, :math:`D(z) = (1+z)^{-1}`,
    so the parity tests have a closed form to check against rather than a second
    numerical answer.
    """

    name = "counting"
    has_native_z = True

    def __init__(self):
        self.solves = []                 # distinct z-tuples, in first-seen order
        self.calls = []                  # every call's z shape, in order

    def _record(self, z):
        import jax.numpy as jnp
        import numpy as np
        self.calls.append(jnp.shape(z))
        try:
            key = tuple(np.round(np.atleast_1d(np.asarray(z, dtype=float)), 8))
        except Exception:                # traced: no value to key on
            return
        if key not in self.solves:
            self.solves.append(key)

    def _at(self, k, z, cosmo):
        import jax.numpy as jnp
        z = jnp.asarray(z, dtype=float)
        p = self._pk(k, cosmo)
        if jnp.ndim(z) == 0:
            return p / (1.0 + z) ** 2
        return p[None, :] / (1.0 + z[:, None]) ** 2

    def pk(self, k, z, cosmo):
        self._record(z)
        return self._at(k, z, cosmo)

    def pk_cb(self, k, z, cosmo):
        self._record(z)
        return self._at(k, z, cosmo)


@pytest.fixture(scope="session")
def analytic_pk():
    return AnalyticPk()


@pytest.fixture
def counting_pk():
    """Function-scoped: every test gets its own counter."""
    return CountingPk()


#: The concentration relation :class:`AnalyticPk` can drive.
#:
#: Not a workaround -- a property of the stub, and one worth stating.  The
#: default flavours declare ``bhattacharya13``, which is parameterised by the
#: growth factor and therefore refuses a backend with ``has_native_z = False``;
#: the stub has no redshift dependence at all, by design, because most of this
#: suite is testing plumbing and a Boltzmann solve per test would cost minutes.
#: So a stub-driven field has to name a relation from the *empirical* family,
#: and at the default ``200m`` that is ``duffy08``: ``dutton14`` and
#: ``klypin16`` are calibrated in ``200c/vir`` and the calibration guard
#: refuses them there, correctly.
STUB_CM_MODEL = "duffy08"


@pytest.fixture
def stub_field(analytic_pk):
    """A field on the cheap stub, at the default definition.

    Used by every test that needs *a* field rather than a particular one.
    Going through one fixture means the next time the default set moves, this
    is the single place that has to follow it.
    """
    from ggah_mod import DIFFERENTIABLE
    from ggah_mod.cosmology import PLANCK18
    from ggah_mod.halos.field import make_field

    return make_field(PLANCK18, DIFFERENTIABLE, analytic_pk, z=0.0,
                      cm_model=STUB_CM_MODEL)

def constructible(name):
    """``make_pk(name)``, or skip with the reason it could not be built.

    A backend in the registry whose *artefact* is absent -- weights not yet
    trained, an optional dependency not installed -- is a legitimate state of
    the working tree and not a failing test.  What must not be legitimate is a
    backend that constructs and then misbehaves, which is what every test using
    this helper goes on to check.
    """
    import pytest
    from ggah_mod.cosmology.power import make_pk
    try:
        return make_pk(name)
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
