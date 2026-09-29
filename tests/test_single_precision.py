"""Single precision is the default; the package must not answer `inf` in it.

`tests/conftest.py` turns x64 on for every other test in this suite, and the
paper's `measure_accuracy.py` does the same, so both ran in double precision
while an ordinary ``import ggah_mod`` did not.  That blind spot let
`gas_mass`, `f_gas` and `x_ray_luminosity` return ``inf`` for every mass under
JAX's default dtype.  These tests run in a **subprocess** with x64 off, which
is the only way to check it: `jax_enable_x64` must be set before the first JAX
call, so it cannot be toggled inside a session conftest has already configured.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def _run_float32(src: str) -> str:
    """Run `src` in a fresh interpreter with x64 off; return its stdout."""
    # Inherit the environment -- `emu_pk` and friends are resolved by
    # PYTHONPATH here -- but pin x64 off and make `ggah_mod` importable
    # regardless of pytest's working directory.
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "0"
    root = str(pathlib.Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def _in_float32(body: str) -> str:
    """Run `body` against the hot-gas preamble, with x64 off."""
    src = textwrap.dedent("""
        import numpy as np, jax, jax.numpy as jnp
        assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
        from ggah_mod.sectors import gas as G
        from ggah_mod.cosmology.parameters import Cosmology
        from ggah_mod import DIFFERENTIABLE
        c = Cosmology()
        g = G.HotGasDPM(backend=DIFFERENTIABLE)
        p = G.dpm_model_params(1)
        m = jnp.asarray(np.logspace(13.0, 15.0, 8))
        z = 0.25
        # The halo concentration the profiles require, from the suite's stub
        # relation: empirical, so the subprocess builds no spectrum for it.
        from ggah_mod.halos.concentration import c_duffy08
        cc = c_duffy08(m, z, "200m")
    """) + textwrap.dedent(body)
    return _run_float32(src)


def _in_float32_agn(body: str) -> str:
    r"""Run `body` against an AGN-sector preamble, with x64 off.

    The AGN sector has the same range problem as the hot gas and did not have
    the guard: :math:`L_X \sim 10^{44}` erg/s, and ``emission_weights`` squares
    it to :math:`10^{88}`.  Building the field here is the cost of the check --
    the sector reads a galaxy sector's stellar masses, so there is no cheaper
    object that reaches the luminosity.
    """
    src = textwrap.dedent("""
        import numpy as np, jax, jax.numpy as jnp
        assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
        from ggah_mod import DIFFERENTIABLE
        from ggah_mod.cosmology import PLANCK18
        from ggah_mod.cosmology.power import make_pk
        from ggah_mod.halos.field import make_field
        from ggah_mod.sectors import agn as A
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        gal = GalaxySector("zumandelbaum15")
        gp = galaxy_defaults("zumandelbaum15")
        a = A.AgnSector(gal)
        ap = A.AgnParams()
        f = make_field(PLANCK18, DIFFERENTIABLE.with_(m_min=1e10, m_max=1e16),
                       make_pk("emu_pk"), z=0.135)
    """) + textwrap.dedent(body)
    return _run_float32(src)


class TestQuantitiesThatFitInFloat32:
    """These are in range, so they must simply work -- no guard, no `inf`."""

    @pytest.mark.parametrize("call, lo, hi", [
        ("g.gas_mass(m, z, c, p, conc=cc)", 1e11, 1e15),
        ("g.f_gas(m, z, c, p, conc=cc)", 1e-3, 1.0),
        ("g.y_amplitude(m, z, c, p, conc=cc)", 1e-10, 1e-3),
        ("g.n_e(g._r_delta(m, z, c, None)[:, None] * 0.5, m, z, c, p, None,"
         " conc=cc)", 1e-8, 1.0),
        ("g.pressure(g._r_delta(m, z, c, None)[:, None] * 0.5, m, z, c, p,"
         " None, conc=cc)", 1e-8, 1e2),
    ])
    def test_finite_and_in_range(self, call, lo, hi):
        out = _in_float32(f"""
            v = np.asarray({call})
            print(bool(np.all(np.isfinite(v))), float(np.min(np.abs(v))),
                  float(np.max(np.abs(v))))
        """)
        finite, vmin, vmax = out.split()
        assert finite == "True", f"{call} is not finite in float32"
        assert lo <= float(vmin) and float(vmax) <= hi, (call, vmin, vmax)

    def test_gas_mass_matches_double_precision(self):
        """The regrouped conversion is algebraically exact, not an approximation.

        Single against double must agree to float32's own resolution, which is
        the statement that the fix changed the arithmetic's *range* and not its
        value.
        """
        out = _in_float32("""
            print(" ".join(repr(float(x)) for x in np.asarray(
                g.gas_mass(m, z, c, p, conc=cc))))
        """)
        import numpy as _np
        got = _np.asarray([float(x) for x in out.split()])
        from ggah_mod.sectors import gas as G
        from ggah_mod.cosmology.parameters import Cosmology
        from ggah_mod import DIFFERENTIABLE
        from ggah_mod.halos.concentration import c_duffy08
        m_ref = _np.logspace(13.0, 15.0, 8)
        ref = _np.asarray(G.HotGasDPM(backend=DIFFERENTIABLE).gas_mass(
            m_ref, 0.25, Cosmology(), G.dpm_model_params(1),
            conc=c_duffy08(m_ref, 0.25, "200m")))
        assert _np.max(_np.abs(got / ref - 1.0)) < 1e-5


class TestQuantitiesThatDoNotFit:

    def test_x_ray_luminosity_refuses_rather_than_returning_inf(self):
        """~1e44 erg/s against a float32 ceiling of 3.4e38: no rearrangement helps.

        The contrast with `gas_mass` is the point.  There only an intermediate
        left the range and regrouping was enough; here the answer itself is out
        of range, so the only honest options are to refuse or to change the
        units, and changing the units would make every published `L_X` mean
        something else.
        """
        out = _in_float32("""
            try:
                g.x_ray_luminosity(m, z, c, p, conc=cc)
                print("RETURNED")
            except RuntimeError as e:
                print("REFUSED", "x64" in str(e), "3.4e38" in str(e))
        """)
        assert out.split()[0] == "REFUSED", "returned a value instead of refusing"
        assert out.split()[1:] == ["True", "True"], \
            "the refusal must name x64 and the float32 ceiling"

    def test_the_agn_luminosity_refuses_rather_than_returning_inf(self):
        """`l_x_agn` is the hot gas's problem in another sector, and was unguarded.

        The AGN sector formed :math:`10^{44}` and squared it to :math:`10^{88}`
        with no guard at all, while `gas.py` had carried one since the same
        defect was found there.  One implementation now, in `numerics`.
        """
        out = _in_float32_agn("""
            try:
                a.l_x_agn(f, ap, gp)
                print("RETURNED")
            except RuntimeError as e:
                print("REFUSED", "x64" in str(e), "3.4e38" in str(e))
        """)
        assert out.split()[0] == "REFUSED", "returned a value instead of refusing"
        assert out.split()[1:] == ["True", "True"], \
            "the refusal must name x64 and the float32 ceiling"

    def test_the_agn_emission_weights_refuse(self):
        """The self-pair squares L_X, so this one is out of range by 1e50."""
        out = _in_float32_agn("""
            try:
                a.emission_weights(f, ap, gp)
                print("RETURNED")
            except RuntimeError as e:
                print("REFUSED", "1e88" in str(e))
        """)
        assert out.split()[0] == "REFUSED", "returned a value instead of refusing"
        assert out.split()[1] == "True", "the refusal must name the squared range"

    def test_the_agn_counts_view_still_works_in_single_precision(self):
        """The guard must not be in `occupation`, and this is what says so.

        `weights` calls `occupation` and throws the luminosity away, so the
        counts view is in range and has to keep working.  A guard placed where
        the number is *formed* rather than where it *escapes* would fail here.
        """
        out = _in_float32_agn("""
            w = a.weights(f, ap, gp)
            print("FINITE", bool(np.all(np.isfinite(np.asarray(w.w_point)))),
                  bool(np.isfinite(float(w.norm))))
        """)
        assert out.split() == ["FINITE", "True", "True"], out


class TestLayerFiveRunsInSinglePrecision:
    r"""The transforms, in the dtype the package actually ships in.

    Two unrelated defects met here, and between them ``w_p``, ``xi``,
    ``Sigma``, ``Delta Sigma`` and ``w(theta)`` were unusable in float32 on
    **both** flavours:

    * ``DIFFERENTIABLE`` is the only flavour with ``hankel="fftlog"``, and
      ``make_fftlog``'s log-spacing check was ``rtol=1e-6`` *relative to*
      ``Delta``.  The deviation a float32 grid carries is absolute in ``ln k``
      and does not move with ``Delta``, so every grid this package builds was
      refused -- on the one flavour a forecast can differentiate.
    * ``_safe_log``'s floor was the literal ``1e-300``, which **is** ``0.0`` in
      float32 (smallest normal 1.2e-38).  So the floor was not too low, it was
      absent: ``log(0) = -inf``, and the quadrature engine returned ``nan`` at
      every radius -- on ``ACCURATE``, the default for a plain import.

    ``tests/conftest.py`` turns x64 on before the first import, which is why
    the rest of this suite saw neither.
    """

    #: The Gaussian pair's width [Mpc/h], as in ``tests/test_transforms.py``.
    A = 5.0

    _PREAMBLE = """
        import numpy as np, jax, jax.numpy as jnp
        assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
        from jax.scipy.special import erf
        from ggah_mod.backend import ACCURATE, DIFFERENTIABLE
        A = 5.0
        k = jnp.logspace(jnp.log10(1e-4), jnp.log10(200.0), 512)
        pk = (2 * jnp.pi * A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * A) ** 2)
    """

    def _gaussian_f32(self, body):
        return _run_float32(textwrap.dedent(self._PREAMBLE)
                            + textwrap.dedent(body))

    def test_wp_answers_its_closed_form_in_float32(self):
        """The headline, and note the grid: ``jnp.logspace``, the *old*
        construction.  Relaxing the tolerance is what unblocks this -- building
        the grid in float64 first is neither necessary nor sufficient, so this
        test would still pass if ``log_grid`` were reverted and still fail if
        the tolerance were.

        Measured: **2.4e-7**, and 3.0e-7 on the asinh line-of-sight grid of
        0.9.7 with the table from 1e-3 Mpc/h.  The threshold is 1e-5 rather than that, because
        this is a tripwire against a defect that returned a hard refusal, not a
        calibration of the FFT -- but the number is the point.  There was no
        accuracy catastrophe hiding behind the refusal."""
        out = self._gaussian_f32("""
            from ggah_mod.observables.real_space import wp
            rp = jnp.asarray([1.0, 2.0, 5.0])
            got = np.asarray(wp(rp, (k, pk), pi_max=40.0, backend=DIFFERENTIABLE))
            want = np.asarray(jnp.sqrt(2 * jnp.pi) * A
                              * jnp.exp(-0.5 * (rp / A) ** 2)
                              * erf(40.0 / (A * jnp.sqrt(2.0))))
            print(k.dtype, np.max(np.abs(got / want - 1.0)))
        """)
        dtype, err = out.split()
        assert dtype == "float32", "x64 leaked; this would prove nothing"
        assert float(err) < 1e-5, f"w_p is {err} off its closed form in float32"

    def test_the_rescaling_solves_in_float32(self):
        """0.9.7's Newton polish of the AW10 ``s``, in float32.

        A target built from the MDR1 table itself, whose answer is ``s = 0.8557``
        exactly.  The Newton step divides C' by C''; in float32 C' is noise at
        ~1e-7 against C'' ~ 0.04, so ``s`` stays within a few 1e-6 -- better
        than the grid vertex alone -- and its derivative stays finite."""
        out = _run_float32(textwrap.dedent("""
            import numpy as np, jax, jax.numpy as jnp
            import ggah_mod.halos.beyond_linear_bias as B
            tab = B.load()
            nodes = jnp.linspace(jnp.log(B.R_RESCALE[0]), jnp.log(B.R_RESCALE[1]), B.N_R_COST)
            def solve(lam):
                return B._solve_s(nodes, B._ln_sigma_md(nodes - jnp.log(lam), tab)
                                  + jnp.log(1.02), tab)
            s = solve(jnp.float32(0.8557))
            d = jax.grad(solve)(jnp.float32(0.8557))
            print(s.dtype, abs(float(s) - 0.8557), float(d))
        """))
        dtype, err, d = out.split()
        assert dtype == "float32", "x64 leaked; this would prove nothing"
        assert float(err) < 1e-4, f"s is {err} off in float32"
        assert np.isfinite(float(d)) and abs(float(d) - 1.0) < 1e-2

    def test_the_quadrature_engine_does_not_return_nan_in_float32(self):
        """``_safe_log``'s whole purpose, defeated by its own literal.  The
        Gaussian underflows to exactly zero at high k, which is the input the
        closed-form validation uses; before the clamp this printed
        ``nan nan nan nan``."""
        out = self._gaussian_f32("""
            from ggah_mod.observables.transforms import pk_to_xi
            R = jnp.asarray([1.0, 2.0, 5.0, 10.0])
            got = np.asarray(pk_to_xi(R, k, pk, backend=ACCURATE))
            want = np.asarray(jnp.exp(-0.5 * (R / A) ** 2))
            # measured 1.9e-6 with the clamp; `nan` at every radius without it
            print(float(jnp.min(pk)), bool(np.all(np.isfinite(got))),
                  np.max(np.abs(got / want - 1.0)))
        """)
        pk_min, finite, err = out.split()
        assert float(pk_min) == 0.0, "the spectrum must underflow for this to bite"
        assert finite == "True", "the quadrature engine returned nan in float32"
        assert float(err) < 1e-5

    @pytest.mark.parametrize("flavour", ["ACCURATE", "DIFFERENTIABLE"])
    def test_xi_matches_double_precision(self, flavour):
        """Single against double to float32's own resolution: the statement
        that the fix changed which grids are *admitted*, not what the transform
        answers.

        Measured against the closed form in float32: 1.9e-6 on the quadrature
        engine, 4.0e-6 on FFTLog -- float32's own noise level, and both a long
        way from the ~1e-8 the float64 path reaches."""
        out = self._gaussian_f32(f"""
            from ggah_mod.observables.transforms import pk_to_xi
            R = jnp.asarray([1.0, 2.0, 5.0, 10.0])
            print(" ".join(repr(float(v)) for v in
                           np.asarray(pk_to_xi(R, k, pk, backend={flavour}))))
        """)
        got = np.asarray([float(v) for v in out.split()])

        import jax.numpy as jnp
        from ggah_mod.backend import BACKENDS
        from ggah_mod.observables.transforms import pk_to_xi
        k = jnp.logspace(jnp.log10(1e-4), jnp.log10(200.0), 512)
        pk = (2 * jnp.pi * self.A ** 2) ** 1.5 * jnp.exp(-0.5 * (k * self.A) ** 2)
        ref = np.asarray(pk_to_xi(jnp.asarray([1.0, 2.0, 5.0, 10.0]), k, pk,
                                  backend=BACKENDS[flavour.lower()]))
        assert np.max(np.abs(got / ref - 1.0)) < 1e-5

    def test_wp_differentiates_and_jits_in_float32(self):
        """What the flavour is *for*, in the dtype it ships in.  A refusal in
        `make_fftlog` is a Python-level raise, so this failed at trace time --
        no gradient, not a wrong one."""
        out = self._gaussian_f32("""
            from ggah_mod.observables.real_space import wp
            rp = jnp.asarray([1.0, 2.0, 5.0])
            f = lambda a: jnp.sum(wp(rp, (k, a * pk), pi_max=40.0,
                                     backend=DIFFERENTIABLE))
            g = float(jax.grad(f)(1.0))
            print(np.isfinite(g), abs(g - float(jax.jit(f)(1.0))) / abs(g))
        """)
        finite, rel = out.split()
        assert finite == "True", "the gradient is not finite in float32"
        # d/da of a linear-in-a functional is the functional itself.
        assert float(rel) < 1e-5

    def test_the_default_multipole_grid_is_the_same_in_both_precisions(self):
        """``DEFAULT_WTHETA_ELL`` is materialised at *import* time, so its
        values used to depend on ``JAX_ENABLE_X64`` at that moment.  Exact
        equality is the sharpest available statement of dtype-independence,
        and it fails against the old construction."""
        out = _run_float32(textwrap.dedent("""
            import jax
            assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
            from ggah_mod.observables.spec import DEFAULT_WTHETA_ELL
            print(" ".join(repr(v) for v in DEFAULT_WTHETA_ELL))
        """))
        from ggah_mod.observables.spec import DEFAULT_WTHETA_ELL
        assert [float(v) for v in out.split()] == list(DEFAULT_WTHETA_ELL)


def test_the_conversion_constant_is_folded_in_python_floats():
    """Written as separate factors it cannot survive float32 either way.

    ``MPC_CM**3`` overflows and ``M_PROTON_G / M_SUN_G`` flushes to zero, so
    the constant has to be formed in Python before it meets a `jnp` array.
    Pinning both bounds is what stops a later edit from splitting it again.
    """
    from ggah_mod.sectors.gas import _NE_INTEGRAL_TO_MSUN_H
    from ggah_mod.cosmology import constants as C
    from ggah_mod.sectors.gas import MU_E, M_PROTON_G, M_SUN_G
    f32_max, f32_min_normal = 3.4028235e38, 1.1754944e-38
    assert C.MPC_CM ** 3 > f32_max
    assert M_PROTON_G / M_SUN_G < f32_min_normal
    assert f32_min_normal < _NE_INTEGRAL_TO_MSUN_H < f32_max
    assert _NE_INTEGRAL_TO_MSUN_H == MU_E * M_PROTON_G / M_SUN_G * C.MPC_CM ** 3


class TestEverySectorAnswersOrRefuses:
    """The general form of the two defects, not just the two.

    `gas_mass` and `x_ray_luminosity` were found by checking one number that
    looked wrong.  What the package actually owes a caller is broader: in the
    default dtype every sector either returns finite weights or raises saying
    why.  ``inf`` is the third option and it is the one that propagates into a
    spectrum, a likelihood and a posterior without anything raising.
    """

    def test_all_sector_weights_are_finite_or_refuse(self):
        out = _in_float32("""
            from ggah_mod.cosmology import PLANCK18
            from ggah_mod.cosmology.power import make_pk
            from ggah_mod.halos.field import make_field
            from ggah_mod.sectors import galaxies as GAL, agn as AGN, gas as GS

            c = PLANCK18
            fl = make_field(c, DIFFERENTIABLE, make_pk("emu_pk"), z=0.25)
            todo = [("galaxies", lambda: GAL.GalaxySector().weights(
                        fl, GAL.GalaxyParams())),
                    ("agn", lambda: AGN.AgnSector(GAL.GalaxySector()).weights(
                        fl, AGN.AgnParams(), GAL.GalaxyParams()))]
            for view in ("pressure", "mass", "xray"):
                todo.append(("gas:" + view, lambda v=view: GS.HotGasDPM(
                    backend=DIFFERENTIABLE).weights(
                        fl, GS.dpm_model_params(1), view=v)))

            for name, fn in todo:
                try:
                    w = fn()
                except RuntimeError as e:
                    # A refusal is a pass, provided it says what to do.
                    print(name, "REFUSED", "x64" in str(e))
                    continue
                bad = 0
                for f in ("w_point", "w_extended", "norm", "bias_weight"):
                    v = getattr(w, f, None)
                    if v is not None:
                        bad += int((~np.isfinite(np.asarray(v))).sum())
                print(name, "FINITE" if bad == 0 else "NONFINITE", bad)
        """)
        seen = {}
        for line in out.strip().splitlines():
            parts = line.split()
            seen[parts[0]] = parts[1:]
        assert seen, out

        nonfinite = [k for k, v in seen.items() if v[0] == "NONFINITE"]
        assert not nonfinite, f"returned inf/nan in float32: {nonfinite}"

        # And the one that cannot fit must not have quietly started answering.
        assert seen["gas:xray"][0] == "REFUSED", (
            "L_X is 1e44 erg/s against a float32 ceiling of 3.4e38; if this "
            "now returns, it is returning `inf`")
        assert seen["gas:xray"][1] == "True", "the refusal must name x64"
        for name in ("galaxies", "agn", "gas:pressure", "gas:mass"):
            assert seen[name][0] == "FINITE", (name, seen[name])


def test_the_concentration_distribution_survives_float32():
    """The quantile inversion reaches a tail probability of 3e-15.

    Getting there means evaluating ``P(k, exp(tau))`` at ``tau`` far below
    zero, and in float32 ``exp(-700)`` is not a small number, it is **zero** --
    so ``P`` and its derivative both vanish and `invert_monotone`'s Newton
    polish divides by that derivative.  The symptom was a NaN in one node of
    one mass bin, which is the easiest kind to miss.  `_ln_x_bracket` raises
    the floor to the working dtype's smallest normal; this pins that it is
    still raised, and that the answer is right to float32's own resolution.
    """
    out = _run_float32(textwrap.dedent("""
        import numpy as np, jax, jax.numpy as jnp
        assert not jax.config.jax_enable_x64, "x64 leaked into the subprocess"
        import ggah_mod.halos.concentration as CM
        lo, hi = CM._ln_x_bracket()
        s = np.array([0.45, 0.8, 1.3], dtype=np.float32)
        ok, mean = True, []
        for z in (0.0, 0.52, 1.03, 1.43):
            c, w = CM.seppi21_nodes(6.0, s, z)
            ok = ok and bool(jnp.all(jnp.isfinite(c)))
            mean.extend(float(v) for v in jnp.sum(w * c, -1))
        print(ok, lo, " ".join(repr(v) for v in mean))
    """))
    fields = out.split()
    assert fields[0] == "True", "the node table is not finite in float32"
    assert -90.0 < float(fields[1]) < -80.0, "the bracket floor was not raised"
    mean = np.asarray([float(v) for v in fields[2:]])
    assert np.max(np.abs(mean / 6.0 - 1.0)) < 1e-5
