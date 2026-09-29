r"""Verification: which code is on the accurate path, which is differentiable.

The package offers two flavours of one assembly.  That only means anything if
each piece's declared capability is *true*, so this module checks the
declarations against reality rather than trusting them.

Three failure modes are being guarded against, in increasing order of nastiness:

1. **A backend declared differentiable that is not.**  Loud: the gradient
   raises.
2. **A backend declared non-differentiable that silently returns a number
   anyway.**  This is the dangerous one.  A gradient taken through a Boltzmann
   solver would be missing the whole dependence of :math:`P(k)` on the
   cosmology, and there is no symptom -- the number is finite, smooth and
   wrong.
3. **A validation branch that changes the answer.**  Several functions skip
   their input checks under tracing, because ``float()`` on a tracer raises.
   That is correct, but only if the skipped code is *checking* and never
   *computing*: jitted and eager evaluation must agree bit for bit.

The map
-------

Accurate path only -- numpy, Fortran, C or scikit-learn inside::

    cosmology.power.ClassPk           CLASS, the reference
    cosmology.power.CambPk            CAMB, production fitting
    cosmology.amplitude.ln10A_s_for_sigma8    a one-time Newton solve
    halos.mass_function.CsstHMF       scikit-learn GP emulator

Both paths -- pure JAX, differentiable whenever the P(k) backend is::

    cosmology.background              E(z), distances, the neutrino transition
    cosmology.amplitude               top-hat variance, sigma8, S8
    cosmology.growth                  D(z) and its scale-dependence
    cosmology.power.GgahEmuPk         the emulator network
    halos.variance                    sigma(M,z) and its log-derivative
    halos.mass_function               all 16 multiplicity fits, dndm
    halos.linear_bias                      all 6 bias fits
    halos.beyond_linear_bias          the table, the AW10 rescaling, the
                                      correction integrals
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod
from ggah_mod import ACCURATE, DIFFERENTIABLE
from ggah_mod.cosmology import (
    Cosmology, PLANCK18, background, amplitude, growth,
)
from ggah_mod.cosmology.power import make_pk, PK_BACKENDS

from conftest import AnalyticPk
from ggah_mod.halos import (
    MULTIPLICITY, BIAS, make_multiplicity, make_bias, make_hmf,
    sigma_of_mass, dln_sigma_dln_mass, dndm, table_at,
)

_K = np.logspace(-3, 1, 40)
_M = np.logspace(11, 15, 32)

#: Everything that is deliberately *not* differentiable, and why.
NOT_DIFFERENTIABLE = {
    "class": "CLASS is a C Boltzmann solver",
    "camb": "CAMB is a Fortran Boltzmann solver",
}


#: Every symbol in `ggah_mod/halos` allowed to call numpy, and why.
#:
#: The layer docstring claims it is pure JAX with one exception.  That claim was
#: prose until this table existed: nothing checked it, and a numpy call added to
#: a traced function would have produced a silent hole in the differentiable
#: path -- the gradient still returns, and is missing a term.
#:
#: Module-level numpy is never listed here.  A constant built once at import is
#: not on the traced path at all, and every place that does it says so in a
#: comment (the Gauss-Legendre nodes in `profiles`, and in `concentration` the
#: klypin16 and Seppi Table A.1 coefficient tables and the double-exponential
#: quadrature nodes).
#: Everything below is numpy *inside a function*, which needs a reason.
NUMPY_ALLOWED = {
    # --- the declared exception, and its shim -----------------------------
    "mass_function.CsstHMF._set_cosmology":
        "the CSST emulator is a scikit-learn GP behind numpy",
    "mass_function.CsstHMF.dndm":
        "same; CsstHMF.differentiable is False and a gradient through it raises",
    "_cemulator_compat":
        "patches the vendored CSSTemu, which is numpy throughout",

    # --- construction, not evaluation -------------------------------------
    "mass_function.FittingFunctionHMF.__init__":
        "builds the mass grid once, at construction",
    "profiles._leggauss_cached":
        "Gauss-Legendre nodes, memoised; numpy so importing does not start JAX",

    # --- table construction and I/O: accurate path by definition -----------
    "beyond_linear_bias.build":
        "regenerates the beta^NL table from upstream ascii and a Boltzmann solve",
    "beyond_linear_bias._read_bnl": "parses the upstream ascii",
    "beyond_linear_bias.load": "reads the .npz off disk",
    "beyond_linear_bias._extrapolation":
        "fits the end slopes once, on the static table; stored as constants",
    "beyond_linear_bias._ln_sigma_spline":
        "spline coefficients of the static sigma_MD table, once; stored as constants",
    "beyond_linear_bias._g_slopes":
        "Hermite slopes of the static table along the snapshots, once; stored as constants",

    # --- shapes and indices, never values ---------------------------------
    # These two do run inside a traced call.  What they compute from is the
    # loaded table and Python arguments -- both compile-time constants -- and
    # what they return is a bin count and a snapshot index, used to slice.  No
    # traced value reaches them, so no gradient can be lost in them.
    "beyond_linear_bias._trusted_bins":
        "median fractional error of the static table -> a bin count",
    "beyond_linear_bias.table_at":
        "locates a pinned snapshot number in the static table -> an index",

    # --- a check that steps aside rather than concretising ----------------
    # `check_k_support` compares the *grids* -- static choices, not traced
    # quantities -- and `_concrete` returns None the moment any of them is a
    # tracer, at which point the check returns without touching a value.  So
    # numpy here can only ever see a concrete array, and the traced path runs
    # straight past it.
    "variance._concrete":
        "converts to numpy, or gives up if the argument is a tracer",
    "variance.check_k_support":
        "compares concrete grid extents; returns immediately if any is traced",
}


def _numpy_uses(path: pathlib.Path):
    """(qualname, lineno, attr) for every numpy attribute access in a module.

    Read with `ast`, importing nothing -- the same tool and the same reason as
    the paper's `checksrc.py`.  Local aliases count: `import numpy as _np`
    inside a function body is still numpy.
    """
    tree = ast.parse(path.read_text())
    aliases = {a.asname or a.name
               for n in ast.walk(tree) if isinstance(n, ast.Import)
               for a in n.names if a.name == "numpy"}
    if not aliases:
        return []

    out, stack = [], []

    def walk(node, qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                walk(child, qual + [child.name])
            else:
                if (isinstance(child, ast.Attribute)
                        and isinstance(child.value, ast.Name)
                        and child.value.id in aliases):
                    out.append((".".join(qual), child.lineno, child.attr))
                walk(child, qual)

    walk(tree, stack)
    return out


class TestNoNumpyOnTheTracedPath:
    """The halo layer is pure JAX except where this file says otherwise.

    Deliberately structural rather than behavioural.  A gradient test can only
    catch a numpy island that happens to sit on the path it differentiates; this
    catches one anywhere, and names the symbol.
    """

    HALOS = sorted((pathlib.Path(ggah_mod.__file__).parent / "halos").glob("*.py"))

    def test_there_are_modules_to_check(self):
        """Guards the loop below against silently checking nothing."""
        assert len(self.HALOS) >= 9

    @pytest.mark.parametrize("path", HALOS, ids=lambda p: p.stem)
    def test_every_numpy_call_is_accounted_for(self, path):
        unexplained = []
        for qual, lineno, attr in _numpy_uses(path):
            if not qual:
                continue                        # module level: a constant
            if path.stem in NUMPY_ALLOWED:
                continue                        # whole module is declared
            if f"{path.stem}.{qual}" in NUMPY_ALLOWED:
                continue
            unexplained.append(f"{path.stem}.{qual}:{lineno} -> np.{attr}")

        assert not unexplained, (
            "numpy inside a traced halo function, with no entry in "
            "NUMPY_ALLOWED:\n  " + "\n  ".join(unexplained) +
            "\n\nEither rewrite it in jnp, or add it to NUMPY_ALLOWED in "
            "tests/test_paths.py with the reason it cannot lose a gradient.")

    def test_the_allow_list_has_no_dead_entries(self):
        """A stale exemption is how a real hole gets waved through later."""
        seen = set()
        for path in self.HALOS:
            for qual, _, _ in _numpy_uses(path):
                seen.add(path.stem)
                if qual:
                    seen.add(f"{path.stem}.{qual}")
        assert not (set(NUMPY_ALLOWED) - seen), (
            f"NUMPY_ALLOWED names symbols that no longer use numpy: "
            f"{sorted(set(NUMPY_ALLOWED) - seen)}")

    def test_csst_is_the_only_whole_module_exempted(self):
        """The layer docstring's "one exception" claim, as an assertion."""
        whole_modules = {k for k in NUMPY_ALLOWED if "." not in k}
        assert whole_modules == {"_cemulator_compat"}


class TestDeclarationsAreTrue:
    """Every backend's `differentiable` flag, checked by trying it."""

    @pytest.mark.parametrize("name", sorted(PK_BACKENDS))
    def test_flag_matches_reality(self, name):
        from conftest import constructible
        pk = constructible(name)
        z = 0.0
        f = lambda v: jnp.sum(jnp.asarray(
            pk.pk(_K, z, PLANCK18.replace(ln10A_s=v))))
        if pk.differentiable:
            g = float(jax.grad(f)(PLANCK18.ln10A_s))
            assert np.isfinite(g) and g != 0.0
        else:
            with pytest.raises(Exception):
                jax.grad(f)(PLANCK18.ln10A_s)

    @pytest.mark.parametrize("name", sorted(NOT_DIFFERENTIABLE))
    def test_solver_backends_refuse_rather_than_mislead(self, name):
        """The dangerous direction.

        A gradient through a Boltzmann solver must **raise**.  If it returned a
        number it would be missing the entire dependence of P(k) on the
        cosmology -- finite, smooth, and wrong, with nothing to show for it.
        """
        pk = make_pk(name)
        f = lambda v: jnp.sum(jnp.asarray(pk.pk(_K, 0.0, PLANCK18.replace(Omega_m=v))))
        with pytest.raises(Exception):
            jax.grad(f)(PLANCK18.Omega_m)

    def test_every_backend_declares_both_capabilities(self):
        """On the *class*, not on an instance.

        A capability is a property of the backend, not of a particular object,
        and reading it off the class is what a caller choosing between them
        does -- and what lets the check run for a backend whose artefact is not
        on this machine.  `test_flag_matches_reality` is where the declaration
        is held to the behaviour.
        """
        for name, klass in PK_BACKENDS.items():
            assert isinstance(klass.differentiable, bool), name
            assert isinstance(klass.has_native_z, bool), name

    @pytest.mark.slow
    def test_beyond_linear_bias_adds_no_exception(self):
        """The correction differentiates -- it is not a second CsstHMF.

        A zero gradient would mean the amplitude dependence had been dropped
        somewhere in the rescaling, not that there is none: a more clustered
        cosmology genuinely sits at a different point on the MultiDark
        sequence.
        """
        pk = AnalyticPk()
        k = np.logspace(-4, 2, 512)
        k_out = np.logspace(-1.2, -0.2, 12)

        def f(lnA):
            c = PLANCK18.replace(ln10A_s=lnA)
            return jnp.sum(table_at(k_out, k, 0.35 * pk.pk_cb(k, 0.0, c),
                                    check=False).beta)

        g = float(jax.grad(f)(3.044))
        assert np.isfinite(g)
        assert abs(g) > 1e-3

    def test_csst_mass_function_declares_itself_correctly(self):
        # CSSTemu is optional and installed from git; `find_spec` rather than an
        # import, because CEmulator imports only after the shim has run.
        import importlib.util
        if importlib.util.find_spec("CEmulator") is None:
            pytest.skip("CSSTemu (CEmulator) is not installed")
        hmf = make_hmf("csst")
        assert hmf.differentiable is False
        f = lambda v: jnp.sum(jnp.asarray(hmf.dndm(_M, 0.0, PLANCK18.replace(Omega_m=v))))
        with pytest.raises(Exception):
            jax.grad(f)(PLANCK18.Omega_m)


class TestValidationBranchesDoNotChangeTheAnswer:
    """The `concrete()` guards skip input checks under tracing.

    That is necessary -- `float()` on a tracer raises -- but it means some code
    runs eagerly and not under `jit`.  The invariant that makes it safe is that
    the skipped code only ever *checks*, never computes.

    The bar is agreement to floating-point round-off, not bit-identity: XLA
    fuses and reorders arithmetic under `jit`, so the two paths legitimately
    accumulate rounding differently.  Measured, that is 3.3e-15 for the
    emulator and 1.0e-15 for EH98 against a float64 eps of 2.2e-16, while
    repeated eager evaluation is bit-identical -- so the residual is reordering,
    not nondeterminism, and anything structural would be orders of magnitude
    larger.
    """

    #: A few tens of float64 eps: far above the measured reordering residual,
    #: far below any difference that could come from skipping real work.
    RTOL = 1e-13

    @pytest.mark.parametrize("name", ["emu_pk"])
    def test_pk_jit_equals_eager(self, name):
        pk = make_pk(name)
        eager = np.asarray(pk.pk(_K, 0.0, PLANCK18))
        jitted = np.asarray(jax.jit(lambda c: pk.pk(_K, 0.0, c))(PLANCK18))
        np.testing.assert_allclose(jitted, eager, rtol=self.RTOL)

    def test_nu_correction_jit_equals_eager(self):
        """The correction lives in `emu_pk` now, and its validity check calls
        `float()` -- which raises on a tracer, so it is *skipped* under jit
        rather than attempted.  That is safe only because the skipped code
        checks and never computes, which is what this asserts."""
        from emu_pk import ratio

        def corr(c):
            return ratio.suppression_m(_K, c.f_nu, 0.0, c.w0, c.wa)

        eager = np.asarray(corr(PLANCK18))
        jitted = np.asarray(jax.jit(corr)(PLANCK18))
        np.testing.assert_allclose(jitted, eager, rtol=self.RTOL)

    def test_growth_jit_equals_eager(self):
        pk = make_pk("emu_pk")
        eager = float(growth.growth_factor(1.0, PLANCK18, pk))
        jitted = float(jax.jit(lambda c: growth.growth_factor(1.0, c, pk))(PLANCK18))
        assert jitted == pytest.approx(eager, rel=self.RTOL)

    def test_beyond_linear_bias_jit_equals_eager(self):
        """beta^NL is on the differentiable path, rescaling included.

        The rescaling contains an ``argmin``, which is why this is worth
        checking rather than assuming: the grid search supplies only a bracket,
        under ``stop_gradient``, while the cost values it selects carry the
        value and the gradient.  A structural failure would show up here as an
        outright trace error, not a small residual.
        """
        pk = AnalyticPk()
        k = np.logspace(-4, 2, 512)
        k_out = np.logspace(-1.2, -0.2, 12)

        def f(c):
            # scaled to sit inside the MultiDark sequence rather than on its
            # clamp, where the answer would be trivially constant
            return table_at(k_out, k, 0.35 * pk.pk_cb(k, 0.0, c),
                            check=False).beta

        eager = np.asarray(f(PLANCK18))
        jitted = np.asarray(jax.jit(f)(PLANCK18))
        # Peak-normalised, and three orders looser than `RTOL`, because the
        # rescaling amplifies round-off and this is the measurement of by how
        # much.  `D_0` is a 1024-step RK4 solve through `lax.scan`, which XLA
        # fuses differently under `jit`: a few ulp.
        # The AW10 cost is then matched to a length rescaling `s`, and `s` is
        # read back as `k_MD = s k` off a tabulated `beta`, so the two stages
        # multiply -- x400 into `s`, x620 out of the table, x1e6 overall.
        # Nothing here is unstable; `beta^NL` is simply reproducible to 1e-9
        # rather than to 1e-15, and quoting the tighter number would be
        # quoting a property of the inputs rather than of the method.
        peak = np.max(np.abs(eager))
        assert np.max(np.abs(jitted - eager)) / peak < 1e-9

    def test_the_residual_really_is_only_round_off(self):
        """Guards the tolerance above: eager evaluation must be bit-repeatable,
        so any jit/eager difference is reordering rather than nondeterminism."""
        pk = make_pk("emu_pk")
        a = np.asarray(pk.pk(_K, 0.0, PLANCK18))
        b = np.asarray(pk.pk(_K, 0.0, PLANCK18))
        np.testing.assert_array_equal(a, b)

    def test_checks_still_fire_when_they_can(self):
        """Skipping under trace must not mean skipping always."""
        cp = make_pk("emu_pk")
        with pytest.raises(ValueError, match="training box"):
            cp.pk(_K, 0.0, PLANCK18.replace(h=0.95))
        from emu_pk import ratio
        far = PLANCK18.replace(sum_mnu=0.9)
        with pytest.raises(ValueError, match="outside the distilled table"):
            ratio.suppression_m(_K, far.f_nu, 0.0, far.w0, far.wa)


class TestRedshiftIsTraceable:
    """`vmap`, not a Python loop, over the redshift axis.

    A Python loop needs `np.asarray(z)`, which forces z concrete.  The
    limitation hides behind the obvious test: `jax.grad` with respect to a
    *cosmological* parameter still works, because z stays a constant there.
    Only `jit` over the whole model, or a gradient in z, exposes it.
    """

    def test_gradient_with_respect_to_redshift(self):
        pk = make_pk("emu_pk")
        g = float(jax.grad(lambda zz: jnp.sum(pk.pk(_K, zz, PLANCK18)))(0.5))
        assert np.isfinite(g) and g != 0.0

    def test_growth_gradient_with_respect_to_redshift(self):
        pk = make_pk("emu_pk")
        g = float(jax.grad(lambda zz: growth.growth_factor(zz, PLANCK18, pk))(1.0))
        assert np.isfinite(g) and g < 0.0        # growth falls with redshift

    def test_vector_redshift_under_jit(self):
        pk = make_pk("emu_pk")
        out = jax.jit(lambda c: pk.pk(_K, jnp.array([0.0, 1.0]), c))(PLANCK18)
        assert np.asarray(out).shape == (2, len(_K))

    # `growth_rate` and `f_sigma8` below use the counting stub rather than
    # `emu_pk`: they are about the *redshift* path, and a stub with a closed-form
    # z dependence makes the assertion exact and the test cheap.

    def test_growth_rate_gradient_with_respect_to_redshift(self):
        """`growth_rate` built its finite-difference stencil in numpy, which
        forced z concrete: this raised `TracerArrayConversionError`, and so did
        `jax.vmap(make_field)` with `cm_model="diemer19"`, which reads it.  The
        stencil is still chosen by value with `jnp.where` over both branches --
        nothing here is control flow."""
        from conftest import CountingPk
        pk = CountingPk()
        pk.pk_cb(_K, 0.0, PLANCK18)                       # warm the stub
        g = jax.grad(lambda zz: growth.growth_rate(zz, PLANCK18, pk))(1.0)
        assert np.isfinite(float(g))

    def test_f_sigma8_gradient_with_respect_to_redshift(self):
        from conftest import CountingPk
        pk = CountingPk()
        pk.pk_cb(_K, 0.0, PLANCK18)
        g = jax.grad(lambda zz: growth.f_sigma8(zz, PLANCK18, pk))(1.0)
        assert np.isfinite(float(g)) and g != 0.0

    def test_growth_rate_under_vmap(self):
        from conftest import CountingPk
        pk = CountingPk()
        pk.pk_cb(_K, 0.0, PLANCK18)
        out = jax.vmap(lambda zz: growth.growth_rate(zz, PLANCK18, pk))(
            jnp.array([0.0, 0.5, 1.0]))
        assert np.all(np.isfinite(np.asarray(out)))

    def test_the_stencil_is_selected_by_value_not_by_a_branch(self):
        """Structural, in the idiom of `tests/test_limber.py`: the property that
        makes the redshift traceable is that both stencils are formed."""
        body = inspect.getsource(growth.growth_rate)
        assert "jnp.where" in body
        assert "np.where" not in body.replace("jnp.where", "")
        assert "lax.cond" not in body
        # Whether numpy still touches `z` is an AST question, not a substring
        # one -- `jnp.asarray(z` contains `np.asarray(z`.  See the test below.

    def test_no_numpy_call_in_growth_takes_the_redshift(self):
        """The narrow form of the numpy audit, for the one module where a
        numpy call on `z` is the whole defect.  `np.logspace` for a static k
        grid is fine and stays; `np.asarray(z)` is not."""
        src = pathlib.Path(growth.__file__).read_text()
        offenders = []
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "np"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in ("z", "z_arr"):
                    offenders.append(f"line {node.lineno}: np.{f.attr}({arg.id})")
        assert not offenders, offenders


class TestPureLayerIsAlwaysTraceable:
    """Nothing in the layers below the P(k) backend may need concrete values."""

    def test_background(self):
        for fn in (background.hubble_e, background.comoving_distance,
                   background.angular_diameter_distance,
                   background.luminosity_distance,
                   background.comoving_volume_element):
            v = jax.jit(lambda c, fn=fn: jnp.sum(jnp.asarray(fn(jnp.array([0.5]), c))))(PLANCK18)
            assert np.isfinite(float(v)), fn.__name__

    def test_amplitude(self):
        p = np.logspace(3, -1, len(_K))
        assert np.isfinite(float(jax.jit(lambda pp: amplitude.sigma8(pp, _K))(p)))

    @pytest.mark.parametrize("name", sorted(MULTIPLICITY))
    def test_every_multiplicity_function(self, name):
        """Both families, and the second one needs an argument the first refuses.

        Sixteen of these are fits to sigma and z.  One is recalibrated per
        cosmology and takes one, which is the distinction
        ``COSMOLOGY_DEPENDENT_MULTIPLICITY`` exists to record -- so this test
        reads that set rather than assuming a uniform signature, in the same
        way ``make_field`` does.
        """
        from ggah_mod.halos.mass_function import (
            COSMOLOGY_DEPENDENT_MULTIPLICITY)

        fn = make_multiplicity(name)
        kw = {"cosmo": PLANCK18} if name in COSMOLOGY_DEPENDENT_MULTIPLICITY \
            else {}
        g = jax.grad(lambda s: jnp.sum(
            jnp.atleast_1d(fn(jnp.array([s]), 0.0, **kw))))(1.0)
        assert np.isfinite(float(g)) and float(g) != 0.0

    @pytest.mark.parametrize("name", sorted(BIAS))
    def test_every_bias_function(self, name):
        fn = make_bias(name)
        g = jax.grad(lambda s: jnp.sum(jnp.atleast_1d(fn(jnp.array([s])))))(1.0)
        assert np.isfinite(float(g)) and float(g) != 0.0
        jax.jit(lambda s: fn(s))(jnp.array([1.0]))


class TestWholeChain:
    """Cosmology to b_eff, both flavours."""

    @staticmethod
    def _b_eff(cosmo, pk, k, m):
        p = pk.pk_cb(k, 0.0, cosmo)
        s = sigma_of_mass(m, k, p, cosmo.rho_cold)
        d = dln_sigma_dln_mass(m, k, p, cosmo.rho_cold)
        n = dndm(m, s, d, cosmo.rho_cold, model="tinker08")
        b = make_bias("tinker10")(s)
        return jnp.trapezoid(n * b * m, jnp.log(m)) / jnp.trapezoid(n * m, jnp.log(m))

    def test_fast_chain_jits_and_differentiates(self):
        pk = make_pk(DIFFERENTIABLE.pk)
        k = np.logspace(-4, 2, 256)
        f = lambda v: jnp.log(self._b_eff(PLANCK18.replace(ln10A_s=v), pk, k, _M))
        ad = float(jax.grad(f)(PLANCK18.ln10A_s))
        step = 1e-5
        fd = float((f(PLANCK18.ln10A_s + step) - f(PLANCK18.ln10A_s - step)) / (2 * step))
        assert ad == pytest.approx(fd, rel=1e-4)
        assert np.isfinite(float(jax.jit(lambda c: self._b_eff(c, pk, k, _M))(PLANCK18)))

    @pytest.mark.slow
    def test_accurate_chain_runs_and_agrees_in_value(self):
        """The two flavours must give the *same physics*; how far apart they are
        is the parity budget's job, not this test's."""
        k = np.logspace(-4, 2, 256)
        fast = float(self._b_eff(PLANCK18, make_pk(DIFFERENTIABLE.pk), k, _M))
        acc = float(self._b_eff(PLANCK18, make_pk(ACCURATE.pk), k, _M))
        assert fast == pytest.approx(acc, rel=0.02)

    @pytest.mark.slow
    def test_accurate_chain_refuses_a_gradient(self):
        pk = make_pk(ACCURATE.pk)
        k = np.logspace(-4, 2, 256)
        with pytest.raises(Exception):
            jax.grad(lambda v: self._b_eff(PLANCK18.replace(ln10A_s=v), pk, k, _M)
                     )(PLANCK18.ln10A_s)


class TestBackendFlavourConsistency:
    def test_fast_pk_is_differentiable_and_accurate_is_not(self):
        assert make_pk(DIFFERENTIABLE.pk).differentiable is True
        assert make_pk(ACCURATE.pk).differentiable is False
        assert make_pk(ACCURATE.pk).differentiable is False

    def test_traced_flavour_cannot_be_given_a_solver(self):
        with pytest.raises(ValueError, match="traced"):
            DIFFERENTIABLE.with_(pk="class")

    def test_accurate_flavour_cannot_claim_to_be_traced(self):
        with pytest.raises(ValueError, match="traced"):
            ACCURATE.with_(traced=True)

    def test_the_map_in_this_module_covers_every_backend(self):
        """If a backend is added, it must be classified here."""
        declared = {n for n, k in PK_BACKENDS.items() if not k.differentiable}
        assert declared == set(NOT_DIFFERENTIABLE)


class TestTheSourceCompilesClean:
    r"""A docstring warning is a docstring that is not saying what it reads as.

    Every equation in this package is written in reStructuredText maths, so
    docstrings are full of ``\Omega``, ``\omega``, ``\delta``.  In a non-raw
    string Python reads those as escape sequences, keeps them verbatim because
    they are not recognised, and emits a ``DeprecationWarning`` -- today.  The
    warning is scheduled to become a ``SyntaxError``, at which point the module
    stops importing; and in the meantime ``\n``, ``\t`` and ``\b`` are all
    plausible things to write next to a subscript and all silently *not*
    verbatim.  Cheaper to refuse the class than to find the one that bites.
    """

    def test_no_module_warns_on_compile(self):
        import pathlib
        import warnings

        root = pathlib.Path(ggah_mod.__file__).parent
        bad = []
        for f in sorted(root.rglob("*.py")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile(f.read_text(), str(f), "exec")
            bad += [f"{f.relative_to(root)}:{w.lineno} {w.message}"
                    for w in caught]
        assert not bad, "\n".join(bad)


class TestTheEmulatorPinsAgreeEverywhere:
    r"""One floor per emulator, declared once.

    ``pyproject.toml`` and ``environment.yml`` both pinned ``emu_pk`` and
    ``emu_hmf`` until 1.0.0, and on 2026-09-10 they disagreed for several hours:
    the cap on ``emu_pk`` was lifted in the first and left at ``<2`` in the
    second.  That is the same drift that let a cluster clone measure an
    eight-parameter network while the package described an eleven-parameter
    one.  Since 1.0.0 ``environment.yml`` installs ``ggah_mod`` through pip and
    restates nothing, so the floors live in ``pyproject.toml`` alone and this
    class checks that they stay there.

    Both floors are **correctness** floors rather than convenience ones.  Below
    ``emu_pk`` 2.0.1 the emulator restates the rounded 93.14 eV rather than this
    package's derived rest-mass denominator (X47), and below 2.0.0 its box has no
    curvature axis while ``GgahEmuPk`` declares one, which answers a curved
    cosmology from a flat network without raising.  Below ``emu_hmf`` 1.2.0 the
    two conversions are not exact inverses and the four curvature constants this
    package restates do not exist, so their drift check skips.
    """

    @staticmethod
    def _pins():
        import re
        root = pathlib.Path(__file__).resolve().parent.parent
        pj = (root / "pyproject.toml").read_text()
        ev = (root / "environment.yml").read_text()
        # Only the requirement lines, never the prose: an earlier version of
        # this check matched a pin quoted inside a comment and reported a
        # disagreement that was not one.
        env = dict(re.findall(r"^\s+- (emu[-_]\w+) ?(\S*)\s*(?:#.*)?$", ev, re.M))
        proj = {p: re.search(rf'"{p} (>=[^"]+)"', pj).group(1)
                for p in ("emu_pk", "emu_hmf")}
        return proj, env

    def test_the_floors_are_declared_once(self):
        proj, env = self._pins()
        assert not env, (
            f"environment.yml pins {sorted(env)} again; it installs ggah_mod "
            f"through pip, so the floors belong to pyproject.toml alone")

    def test_the_floors_are_the_releases_the_code_needs(self):
        """Pinned, because each floor is the release that made a silent failure
        impossible rather than the newest thing available."""
        proj, _ = self._pins()
        assert proj["emu_pk"] == ">=2.0.1,<3"
        assert proj["emu_hmf"] == ">=1.2.0,<2"

    def test_what_is_installed_satisfies_them(self):
        """A floor nothing checks is a sentence.  This is the environment
        actually resolving, which is the thing that decides what runs.  Read
        from the distributions' metadata: ``emu_pk`` 2.0.1 shipped with its
        ``__version__`` still reading 2.0.0."""
        from importlib.metadata import version

        def release(dist):
            return tuple(int(x) for x in version(dist).split(".")[:3])
        assert release("emu_pk") >= (2, 0, 1)
        assert release("emu_hmf") >= (1, 2, 0)
