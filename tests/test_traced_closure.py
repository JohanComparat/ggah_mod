"""1.5.1: the energy closure is traced in the cosmology and in its couplings, and an
empty emission profile has a finite derivative."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.halos.profiles import profile_uk_gl
from ggah_mod.sectors import gas as G


def test_an_empty_profile_has_unit_shape_and_a_finite_derivative():
    k = jnp.geomspace(1e-2, 10.0, 5)

    def u(a):
        return profile_uk_gl(k, jnp.array([1.0, 2.0]),
                             lambda r: a * jnp.where(r > 0.0, 0.0, 1.0) * r, n_gl=16)

    val, der = jax.jvp(u, (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    np.testing.assert_array_equal(np.asarray(val), 1.0)
    assert np.all(np.isfinite(np.asarray(der)))


@pytest.mark.slow
def test_the_hard_band_of_a_cool_sky_differentiates_at_z3():
    """The 2-10 keV band is 8-40 keV in the rest frame at z = 3, and empty for the
    cool haloes: their emissivity is zero, and the normalisation was 0/0."""
    from ggah_mod.sectors.cooling import BandCooling

    gas = G.HotGasDPM(cooling=BandCooling(band=(2.0, 10.0)), scatter="isobaric",
                      backend=DIFFERENTIABLE)
    pk = make_pk("emu_pk")

    def f(om):
        fl = make_field(PLANCK18.replace(Omega_m=om), DIFFERENTIABLE, pk, z=3.0)
        return jnp.sum(gas.weights(fl, G.DpmParams(), view="xray").w_extended)

    _, d = jax.jvp(f, (jnp.asarray(PLANCK18.Omega_m),), (jnp.asarray(1.0),))
    assert np.isfinite(float(d))


class TestTheClosureIsTraced:
    """``feedback_budget`` took the cosmic baryon fraction through ``float``, so a
    traced cosmology either refused or lost that dependence."""

    @staticmethod
    def _setup():
        from ggah_mod.sectors import agn as A
        from ggah_mod.sectors import energetics as E
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        gal = GalaxySector("zumandelbaum15", calibration="off")
        agn = A.AgnSector(gal, calibration="off")
        return gal, agn, galaxy_defaults("zumandelbaum15"), A.AgnParams(), E

    def _budget(self, ob, log10_eps):
        gal, agn, gp, ap, E = self._setup()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure", agn=agn, galaxies=gal,
                        energetics=E.EnergeticsParams(log10_eps_agn=log10_eps))
        fl = make_field(PLANCK18.replace(Omega_b=ob), DIFFERENTIABLE, make_pk("emu_pk"),
                        z=0.2)
        return g.feedback_budget(fl, G.DpmParams(), ap, gp)

    @pytest.mark.slow
    @pytest.mark.parametrize("which", ["Omega_b", "log10_eps_agn"])
    def test_autodiff_matches_central_differences(self, which):
        from ggah_mod.sectors import energetics as E

        x0 = {"Omega_b": float(PLANCK18.Omega_b),
              "log10_eps_agn": float(E.EnergeticsParams().log10_eps_agn)}

        def f(x):
            ob = x if which == "Omega_b" else x0["Omega_b"]
            le = x if which == "log10_eps_agn" else x0["log10_eps_agn"]
            return self._budget(ob, le)

        _, ad = jax.jvp(f, (jnp.asarray(x0[which]),), (jnp.asarray(1.0),))
        h = 1e-4 if which == "Omega_b" else 1e-3
        fd = (np.asarray(f(x0[which] + h)) - np.asarray(f(x0[which] - h))) / (2 * h)
        ad = np.asarray(ad)
        assert np.all(np.isfinite(ad))
        big = np.abs(fd) > 1e-6 * np.abs(fd).max()
        np.testing.assert_allclose(ad[big], fd[big], rtol=2e-3)
        assert np.abs(fd).max() > 0.0
