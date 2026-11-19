"""Float32 behaviour of the removable singularities and the Si/Ci NFW transform.

The thresholds of the series branches (``|x| < 1e-3`` for s(x), ``x < 0.1``
for W(x)) are tuned for float64, which is the precision the package is
validated in; ``jax_enable_x64`` is not set by ``import ggah_mod``.  These
tests state what float32 costs, in a subprocess with x64 off (it must be set
before the first JAX call, and ``tests/conftest.py`` turns it on).
"""
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _float32(src: str) -> dict:
    env = dict(os.environ, JAX_ENABLE_X64="0")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(src)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


mp.mp.dps = 40
X_W = [1e-6, 1e-4, 1e-3, 0.05, 0.0999, 0.1001, 0.5, 1.0, 10.0, 100.0]
X_S = [-0.5, -1e-2, -1.001e-3, -9.99e-4, -1e-6, 1e-6, 9.99e-4, 1.001e-3, 1e-2, 0.5, 5.0]
KAPPA = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 100.0, 1000.0]
C_NFW = 5.0


def _w_ref(x):
    x = mp.mpf(x)
    return float(3 * (mp.sin(x) - x * mp.cos(x)) / x ** 3)


def _s_ref(x):
    r = mp.sqrt(abs(mp.mpf(x)))
    return float(mp.sinh(r) / r if x > 0 else mp.sin(r) / r)


def _u_ref(k):
    k, c = mp.mpf(k), mp.mpf(C_NFW)
    m = mp.log(1 + c) - c / (1 + c)
    return float((mp.sin(k) * (mp.si((1 + c) * k) - mp.si(k))
                  - mp.sin(c * k) / ((1 + c) * k)
                  + mp.cos(k) * (mp.ci((1 + c) * k) - mp.ci(k))) / m)


@pytest.fixture(scope="module")
def f32():
    xw = [float(np.float32(v)) for v in X_W]
    xs = [float(np.float32(v)) for v in X_S]
    xk = [float(np.float32(v)) for v in KAPPA]
    got = _float32(f"""
        import json, jax, jax.numpy as jnp, numpy as np
        assert not jax.config.jax_enable_x64
        from ggah_mod.cosmology.amplitude import tophat_window
        from ggah_mod.cosmology.background import _sinhc
        from ggah_mod.halos.profiles import si, ci, nfw_uk
        xw = jnp.asarray({xw}, dtype=jnp.float32)
        xs = jnp.asarray({xs}, dtype=jnp.float32)
        xk = jnp.asarray({xk}, dtype=jnp.float32)
        out = dict(dtype=str(tophat_window(xw).dtype),
                   w=np.asarray(tophat_window(xw)).tolist(),
                   s=np.asarray(_sinhc(xs)).tolist(),
                   si=np.asarray(si(xk)).tolist(), ci=np.asarray(ci(xk)).tolist(),
                   u=[float(np.asarray(nfw_uk(k, jnp.float32(1.0), jnp.float32({C_NFW}))).ravel()[0]) for k in xk],
                   ds0=float(jax.grad(_sinhc)(jnp.float32(0.0))),
                   dw_series=float(jax.grad(tophat_window)(jnp.float32(0.05))))
        print(json.dumps(out))
    """)
    got.update(xw=xw, xs=xs, xk=xk)
    return got


def test_float32_stays_float32(f32):
    assert f32["dtype"] == "float32"


def test_tophat_window_float32(f32):
    err = {x: abs(w / _w_ref(x) - 1) for x, w in zip(f32["xw"], f32["w"])}
    # The series branch and the closed form well above the switch: float32 eps.
    assert max(e for x, e in err.items() if x < 0.1 or x >= 0.5) < 1e-6, err
    # Just above the switch the closed form cancels: eps_32 / x^2.
    assert err[f32["xw"][5]] < 5e-5, err
    assert f32["dw_series"] == pytest.approx(-0.05 / 5 + 4 * 0.05 ** 3 / 280, rel=1e-5)


def test_sinhc_float32(f32):
    err = [abs(s / _s_ref(x) - 1) for x, s in zip(f32["xs"], f32["s"])]
    assert max(err) < 5e-7, err
    assert f32["ds0"] == pytest.approx(1 / 6, rel=1e-6)


def test_sici_float32(f32):
    si_err = max(abs(v - float(mp.si(x))) for x, v in zip(f32["xk"], f32["si"]))
    ci_err = max(abs(v - float(mp.ci(x))) for x, v in zip(f32["xk"], f32["ci"]))
    assert si_err < 1e-6 and ci_err < 1e-6, (si_err, ci_err)


def test_nfw_transform_float32(f32):
    """Good to 3e-7 for kappa <= 1; the bracketed differences of Si and Ci then
    cancel, and the error grows as kappa: 3e-5 at 10, 5e-4 at 100, 1.5e-2 at
    1000, where u ~ 1e-6.  In float64 it is below 1.3e-10 throughout."""
    err = {k: abs(u / _u_ref(k) - 1) for k, u in zip(f32["xk"], f32["u"])}
    for k, e in err.items():
        bound = 1e-6 if k <= 1.0 else 2e-5 * k
        assert e < bound, (k, e)


def test_nfw_transform_float64():
    import jax.numpy as jnp
    from ggah_mod.halos.profiles import nfw_uk
    for k in KAPPA:
        u = float(np.asarray(nfw_uk(jnp.asarray(k), 1.0, C_NFW)).ravel()[0])
        assert abs(u / _u_ref(k) - 1) < 1e-9, k
