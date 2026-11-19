"""The peak-background-split budget over all peak heights (PLAN G2, referee R2.3)."""
import numpy as np
import pytest

from ggah_mod.cosmology import PLANCK18
from ggah_mod.halos import linear_bias as LB
from ggah_mod.halos import mass_function as MF

NU_MIN = 0.3        # about 1e8 Msun/h at z = 0 for Planck 2018


def _budget(f, b):
    return LB.pbs_budget(f, lambda s: LB.BIAS[b](s, 200.0), NU_MIN)


def test_press_schechter_is_exactly_consistent():
    r = _budget(MF.fsigma_press74, "press74")
    assert r.converged
    assert float(r.norm_all) == pytest.approx(1.0, abs=1e-4)
    assert float(r.bias_all) == pytest.approx(1.0, abs=1e-4)
    assert float(r.b_unresolved) == pytest.approx(float(r.b_unresolved_fit), rel=1e-4)


def test_sheth_tormen_is_consistent():
    r = _budget(MF.fsigma_sheth99, "sheth99")
    assert r.converged
    assert abs(float(r.defect)) < 5e-4
    assert float(r.b_unresolved) == pytest.approx(float(r.b_unresolved_fit), rel=1e-3)


@pytest.mark.parametrize("f", [lambda s: MF.fsigma_tinker08(s, 0.0),
                               lambda s: MF.fsigma_tinker08_csst(s, 0.0, PLANCK18)],
                         ids=["tinker08", "tinker08_csst"])
def test_tinker08_does_not_normalise(f):
    """f(sigma) -> A as sigma -> infinity, so the all-nu integral diverges."""
    assert not _budget(f, "tinker10").converged


def test_recalibration_against_tinker10():
    """The default pair is not a PBS pair: tinker10 was normalised to the
    original tinker08.  Over nu >= 0.3 the recalibration raises B by 1.2 per
    cent and lowers the bias the unresolved matter must carry by 4 per cent."""
    r08 = _budget(lambda s: MF.fsigma_tinker08(s, 0.0), "tinker10")
    rcs = _budget(lambda s: MF.fsigma_tinker08_csst(s, 0.0, PLANCK18), "tinker10")
    assert float(rcs.bias_res / r08.bias_res - 1) == pytest.approx(0.012, abs=0.003)
    assert float(rcs.b_unresolved / r08.b_unresolved - 1) == pytest.approx(-0.04, abs=0.01)


def test_budget_agrees_with_unresolved_on_a_mass_grid():
    """The same F and B as :func:`unresolved` when the grid is written in nu."""
    nu = np.exp(np.linspace(np.log(NU_MIN), np.log(12.0), 4096))
    sigma = LB.DELTA_C / nu
    f = np.asarray(MF.fsigma_sheth99(sigma))
    b = np.asarray(LB.bias_sheth99(sigma))
    # A fake mass grid with M proportional to nu, rho = 1: dn/dM M^2 = f.
    m = nu
    u = LB.unresolved(m, f / m ** 2, b, 1.0)
    r = _budget(MF.fsigma_sheth99, "sheth99")
    assert float(u.mass) == pytest.approx(1 - float(r.norm_res), rel=1e-9)
    assert float(u.bias) == pytest.approx(float(r.b_unresolved), rel=1e-9)
