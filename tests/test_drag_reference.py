"""The drag epoch against CLASS and CAMB themselves (slow).

Skipped, with a reason, where a solver is absent: **a skip here is not a pass**
-- nothing then checks that :func:`~ggah_mod.cosmology.drag.r_drag` is
CLASS's or CAMB's rather than merely self-consistent.

Which code each comparison uses is a statement, not a convenience:

* :math:`z_d` and flat :math:`r_d` against **CLASS**, which the fit was
  calibrated on, through the package's own
  :func:`~ggah_mod.cosmology.power.class_input`;
* curved :math:`r_d` against **CAMB**, whose convention this package follows:
  CLASS multiplies :math:`dr_s` by :math:`\\sqrt{1-Kr_s^2}` (``background.c``,
  "TBC"), worth 6.6e-5 at :math:`|\\Omega_k| = 0.15` and low :math:`h`.

CAMB and CLASS themselves differ by up to 6.6e-4 in :math:`r_d` and 0.75 in
:math:`z_d` across the box, measured when these tests were written; that, not
the fit's 1.7e-5, is the floor any statement about "the" sound horizon has.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18, r_drag, sound_horizon, z_drag
from ggah_mod.cosmology import _zdrag_coefficients as K
from ggah_mod.cosmology.drag import drag_domain_violations
from ggah_mod.cosmology.power import camb_input, class_input


def _points(n, seed, flat):
    """``n`` cosmologies drawn uniformly in the calibration box and inside the
    dark-energy bound, as physical densities -- the axes the box is in."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = {k: rng.uniform(*v) for k, v in K.BOX.items()}
        if flat:
            p["Omega_k"] = 0.0
        h = p["h"]
        c = PLANCK18.replace(h=h, sum_mnu=p["sum_mnu"], T_cmb=p["T_cmb"],
                             w0=p["w0"], wa=p["wa"], Omega_k=p["Omega_k"],
                             Omega_b=p["omega_b"] / h ** 2)
        c = c.replace(Omega_m=(p["omega_cdm"] + p["omega_b"]) / h ** 2
                      + float(c.Omega_nu_matter))
        if not drag_domain_violations(c):
            out.append(c)
    return out


def _class(c, **over):
    classy = pytest.importorskip(
        "classy", reason="CLASS absent: the drag epoch is not checked against "
                         "the code it was calibrated on, and a skip is not a pass")
    d = class_input(c, output="")
    d.update(over)
    cl = classy.Class()
    cl.set(d)
    cl.compute()
    try:
        r = cl.get_current_derived_parameters(["z_d", "rs_d", "YHe"])
    finally:
        cl.struct_cleanup()
        cl.empty()
    return r


def _camb(c):
    pytest.importorskip("camb", reason="CAMB absent: curved r_d is not checked "
                                       "against the convention it follows")
    import camb
    return camb.get_background(camb_input(c)).get_derived_params()


FLAT = _points(12, 11, flat=True)
CURVED = _points(6, 12, flat=False)


@pytest.mark.slow
@pytest.mark.parametrize("c", FLAT, ids=[f"flat{i}" for i in range(len(FLAT))])
def test_flat_cosmologies_are_classs(c):
    """:math:`z_d` to the held-out tolerance and :math:`r_d` to 1.5e-5 --
    the calibration measured 4.5e-6 flat."""
    r = _class(c)
    assert float(z_drag(c)) == pytest.approx(r["z_d"], abs=K.VALIDATION["target_abs_dz_heldout"])
    assert float(r_drag(c)) / float(c.h) == pytest.approx(r["rs_d"], rel=1.5e-5)


@pytest.mark.slow
@pytest.mark.parametrize("c", CURVED, ids=[f"curved{i}" for i in range(len(CURVED))])
def test_curved_cosmologies_follow_camb(c):
    """Each code for what it is the reference for.  Given CAMB's :math:`z_d`
    the integral is CAMB's to 5e-6 -- the curvature convention, isolated -- and
    :math:`z_d` is still CLASS's, which curvature does not redefine.  Against
    CAMB's own ``rdrag`` only a coarse bound is asserted: CAMB and CLASS
    themselves differ by up to 6.6e-4 in :math:`r_d` at these points (Recfast
    against HyRec, PRIMAT against PArthENoPE helium), so a tighter bound would
    test CAMB against CLASS, not this package."""
    d = _camb(c)
    h = float(c.h)
    at_camb_zd = float(sound_horizon(d["zdrag"], c)[0]) / h
    assert at_camb_zd == pytest.approx(d["rdrag"], rel=5e-6)
    assert float(z_drag(c)) == pytest.approx(_class(c)["z_d"],
                                             abs=K.VALIDATION["target_abs_dz_heldout"])
    assert float(r_drag(c)) / h == pytest.approx(d["rdrag"], rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("p,step", [("Omega_m", 1e-3), ("Omega_b", 1e-3), ("h", 1e-3),
                                    ("sum_mnu", 5e-2), ("T_cmb", 1e-3)])
def test_the_gradient_is_classs(p, step):
    r""":math:`\partial\ln r_d/\partial\theta` by autodiff against CLASS's
    central difference.  Measured to 6e-4; halving the step moves CLASS's own
    difference by as much, so 1e-3 is CLASS's noise and not a bias here."""
    v0 = float(getattr(PLANCK18, p))
    dv = v0 * step

    def ln_rd(v):
        c = PLANCK18.replace(**{p: v})
        return np.log(_class(c)["rs_d"] * float(c.h))

    fd = (ln_rd(v0 + dv) - ln_rd(v0 - dv)) / (2.0 * dv)
    ours = float(getattr(jax.grad(lambda c: jnp.log(r_drag(c)))(PLANCK18), p))
    assert ours == pytest.approx(fd, rel=1e-3)


@pytest.mark.slow
def test_the_bbn_assumptions_have_not_moved():
    """The helium fraction CLASS uses at the fiducial, and the header of the
    table it reads, are the ones the calibration recorded.  A new CLASS, a new
    table or a new ``class_input`` fails here before it silently shifts
    :math:`z_d`."""
    import pathlib

    import classy
    from ggah_mod.cosmology.power import ClassPk

    assert _class(PLANCK18)["YHe"] == pytest.approx(K.BBN["YHe_fiducial"], abs=1e-6)
    table = pathlib.Path(classy.__file__).parent / ClassPk.SBBN_FILE.lstrip("/")
    header = [ln[1:].strip() for ln in table.read_text().splitlines()
              if ln.startswith("#") and ln[1:].strip()][:len(K.BBN["sbbn_header"])]
    assert header == K.BBN["sbbn_header"]
    assert ClassPk.SBBN_FILE == K.BBN["sbbn_file"]
