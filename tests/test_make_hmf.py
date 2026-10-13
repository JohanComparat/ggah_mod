"""Every multiplicity function, reached through ``make_hmf``.

``FittingFunctionHMF`` once passed ``delta=`` to every fit, so all of them but
``tinker08`` raised ``TypeError`` through ``make_hmf``.  The mass definition now
reaches the fit through ``multiplicity_kwargs``, the dispatch ``make_field``
uses, and the two paths are compared here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import ggah_mod.halos as H
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.halos.mass_function import MULTIPLICITY, make_hmf

_M = np.logspace(12.0, 15.0, 7)
_K = np.logspace(-4, 2, 512)


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.mark.parametrize("model", sorted(MULTIPLICITY))
def test_every_fit_evaluates_through_make_hmf(pk, model):
    hmf = make_hmf(model, pk=pk, calibration="off")
    n = np.asarray(hmf.dndm(_M, 0.0, PLANCK18))
    assert n.shape == _M.shape
    assert np.all(np.isfinite(n)) and np.all(n > 0.0)


@pytest.mark.parametrize("model", ["tinker08", "despali16"])
@pytest.mark.parametrize("mdef", ["200m", "200c", "vir"])
def test_make_hmf_agrees_with_make_field(pk, model, mdef):
    delta_m = H.MassDef.from_string(mdef).delta_mean(0.0, PLANCK18)
    hmf = make_hmf(model, pk=pk, delta=float(delta_m), calibration="off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        field = make_field(PLANCK18, pk=pk, z=0.0, mdef=mdef, hmf_model=model,
                           m=_M, k=_K, calibration="off")
    np.testing.assert_allclose(np.asarray(hmf.dndm(_M, 0.0, PLANCK18)),
                               np.asarray(field.dndm), rtol=1e-6)


def test_a_fit_at_another_definition_is_refused_by_default(pk):
    # watson13 is calibrated on FoF haloes, so a 200m request is a mismatch,
    # refused under the default policy exactly as make_field refuses it.
    with pytest.raises(ValueError, match="watson13"):
        make_hmf("watson13", pk=pk).dndm(_M, 0.0, PLANCK18)


def test_an_unknown_policy_is_refused(pk):
    with pytest.raises(ValueError, match="calibration must be one of"):
        make_hmf("tinker08", pk=pk, calibration="sometimes")
