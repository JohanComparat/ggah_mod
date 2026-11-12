r"""The cobaya adapter, run without cobaya.

``tests/test_cobaya_interface.py`` checks the adapter inside a real cobaya
``Model`` and skips where cobaya is absent, which is everywhere CI runs.  The
adapter needs nothing from cobaya but the ``Theory`` base class and the two
attributes cobaya sets on an instance (``current_state``, ``output_params``),
so here a minimal ``cobaya.theory`` stands in for it and the conversion, the
refusals and every accessor run in the fast suite.
"""
import sys
import types
import warnings

import numpy as np
import pytest

import ggah_mod
warnings.filterwarnings("ignore", category=ggah_mod.WorkInProgressWarning)

from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.background import comoving_distance, hubble_e
from ggah_mod.cosmology.constants import C_KM_S

#: Planck 2018 in cobaya's spelling.
P18 = dict(H0=67.66, ombh2=0.02242, omch2=0.11933, ns=0.9665, logA=3.044,
           mnu=0.06)


@pytest.fixture(scope="module")
def adapter():
    """``ggah_mod.interfaces.cobaya`` imported against a stub ``cobaya``, and
    the modules restored afterwards so a real cobaya is not shadowed."""
    saved = {k: sys.modules.get(k) for k in
             ("cobaya", "cobaya.theory", "ggah_mod.interfaces.cobaya")}
    theory = types.ModuleType("cobaya.theory")

    class Theory:                                   # the base cobaya provides
        output_params: list = []

    theory.Theory = Theory
    root = types.ModuleType("cobaya")
    root.theory = theory
    sys.modules["cobaya"], sys.modules["cobaya.theory"] = root, theory
    sys.modules.pop("ggah_mod.interfaces.cobaya", None)
    import ggah_mod.interfaces.cobaya as module
    yield module
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@pytest.fixture(scope="module")
def theory(adapter):
    """A theory at Planck 2018 on the coarse flavour, after one ``calculate``."""
    t = adapter.GgahMod()
    t.backend, t.pk = "differentiable_coarse", None
    t.output_params = []
    t.initialize()
    t.must_provide(Pk_grid={"z": [0.0, 0.5], "k_max": 1.0}, rdrag=None,
                   ggah_fields={"z": [0.2]})
    state = {}
    assert t.calculate(state, want_derived=True, **P18) is True
    t.current_state = state
    return t


class TestTheConversion:
    def test_omega_m_contains_the_neutrinos(self, adapter):
        c = adapter.cosmology_from_cobaya(**P18)
        h = P18["H0"] / 100.0
        naive = (P18["ombh2"] + P18["omch2"]) / h ** 2
        assert float(c.Omega_m) - naive == pytest.approx(
            float(c.Omega_nu_matter), rel=1e-12)

    def test_the_round_trip_is_exact(self, adapter):
        c = adapter.cosmology_from_cobaya(**P18)
        h = float(c.h)
        assert float(c.Omega_b) * h ** 2 == pytest.approx(P18["ombh2"], rel=1e-12)
        assert float(c.Omega_cdm) * h ** 2 == pytest.approx(P18["omch2"], rel=1e-12)

    @pytest.mark.parametrize("bad", ["As", "sigma8", "S8"])
    def test_a_derived_amplitude_is_refused(self, adapter, bad):
        with pytest.raises(ValueError, match=bad):
            adapter.cosmology_from_cobaya(**(P18 | {bad: 1.0}))

    def test_curvature_and_dark_energy_are_carried(self, adapter):
        c = adapter.cosmology_from_cobaya(**(P18 | {"omk": 0.01, "w": -0.9, "wa": 0.1}))
        assert float(c.Omega_k) == 0.01
        assert (float(c.w0), float(c.wa)) == (-0.9, 0.1)


class TestTheTheory:
    def test_it_provides_what_it_declares(self, theory):
        assert "Pk_grid" in theory.get_can_provide()
        assert theory.get_can_provide_params() == ["rdrag"]

    def test_the_derived_parameters(self, theory):
        d = theory.current_state["derived"]
        assert 0.79 < d["sigma8"] < 0.83
        assert d["Omega_m"] == pytest.approx(float(theory.get_ggah_cosmology().Omega_m))
        assert 140.0 < d["rdrag"] < 155.0                        # Mpc

    def test_the_background_is_in_cobayas_units(self, theory):
        c = theory.get_ggah_cosmology()
        z = np.array([0.5, 1.0])
        h_z = theory.get_Hubble(z)
        np.testing.assert_allclose(h_z, 100.0 * float(c.h) * np.asarray(hubble_e(z, c)),
                                   rtol=1e-12)
        np.testing.assert_allclose(theory.get_Hubble(z, units="1/Mpc"), h_z / C_KM_S,
                                   rtol=1e-12)
        with pytest.raises(ValueError, match="unknown units"):
            theory.get_Hubble(z, units="km/s")
        chi = theory.get_comoving_radial_distance(z)
        np.testing.assert_allclose(chi, np.asarray(comoving_distance(z, c)) / float(c.h),
                                   rtol=1e-12)
        np.testing.assert_allclose(theory.get_angular_diameter_distance(z),
                                   chi / (1.0 + z), rtol=1e-10)

    def test_the_spectrum_is_at_the_likelihoods_redshifts(self, theory):
        k, z, p = theory.get_Pk_grid()
        np.testing.assert_array_equal(z, [0.0, 0.5])
        assert p.shape == (2, k.size)
        assert np.all(p[1] < p[0])
        _, _, p_cb = theory.get_Pk_grid(var_pair=("delta_nonu", "delta_nonu"))
        assert np.all(p_cb[0] >= p[0] * (1 - 1e-12))            # P_cb >= P_m
        with pytest.raises(ValueError, match="non-linear"):
            theory.get_Pk_grid(nonlinear=True)

    def test_sigma8_fsigma8_and_growth(self, theory):
        s8 = theory.get_sigma8_z([0.0, 1.0])
        assert s8[1] < s8[0]
        assert s8[0] == pytest.approx(theory.current_state["derived"]["sigma8"], rel=1e-12)
        assert np.all(theory.get_fsigma8([0.5]) > 0.0)
        assert float(theory.get_growth_factor(0.0)) == pytest.approx(1.0, rel=1e-10)

    def test_the_halo_fields_come_from_the_requirement(self, theory):
        fields = theory.get_ggah_fields()
        assert len(fields) == 1 and float(fields[0].z) == pytest.approx(0.2)
        assert len(theory.get_ggah_fields(z=[0.1, 0.3])) == 2

    def test_no_redshift_no_fields(self, adapter, theory):
        t = adapter.GgahMod()
        t.current_state = {"ggah_cosmology": theory.get_ggah_cosmology()}
        with pytest.raises(ValueError, match="no redshifts"):
            t.get_ggah_fields()

    def test_a_k_max_beyond_the_flavour_is_refused(self, adapter):
        t = adapter.GgahMod()
        t.backend, t.pk = "differentiable_coarse", None
        t.initialize()
        with pytest.raises(ValueError, match="stops at"):
            t.must_provide(Pk_grid={"k_max": 1e6})
