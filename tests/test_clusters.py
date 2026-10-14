"""Richness-selected clusters on the red galaxy sector."""

import numpy as np
import pytest

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module")
def setup():
    from ggah_mod import ACCURATE
    from ggah_mod.cosmology import PLANCK18
    from ggah_mod.cosmology.power import make_pk
    from ggah_mod.halos.field import make_field
    from ggah_mod.sectors.clusters import ClusterSector
    from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

    field = make_field(PLANCK18, ACCURATE, make_pk("emu_pk"), z=0.2)
    red = GalaxySector("zumandelbaum16_red")
    return field, ClusterSector(red), galaxy_defaults("zumandelbaum16_red")


def test_richness_rises_with_mass_and_is_one_without_satellites(setup):
    field, cl, gp = setup
    lam = np.asarray(cl.mean_richness(field, gp))
    big = np.log10(np.asarray(field.m)) > 13.0
    assert np.all(np.diff(lam[big]) > 0) and lam.min() >= 1.0
    assert lam[np.argmin(np.abs(np.log10(np.asarray(field.m)) - 14.5))] > 20.0


def test_the_bins_partition_the_haloes(setup):
    field, cl, gp = setup
    edges = [1e-8, 5.0, 20.0, 60.0, 1e4]
    total = sum(np.asarray(cl.p_bin(field, gp, a, b)) for a, b in zip(edges[:-1], edges[1:]))
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_the_mass_of_a_richness_bin_is_redmapper_like(setup):
    """Simet et al. (2017): M200m = 10^14.344 (lambda/40)^1.33 Msun/h, so
    20 <= lambda < 30 sits near 10^14.0.  Reported in a broad window: the
    calibration amplitude is a free parameter of the sector."""
    field, cl, gp = setup
    w = np.asarray(field.dndm) * np.asarray(cl.p_bin(field, gp, 20.0, 30.0))
    m = np.asarray(field.m)
    lgm = float(np.log10(np.trapezoid(w * m * m, np.log(m)) / np.trapezoid(w * m, np.log(m))))
    assert 13.6 < lgm < 14.6


def test_the_stacked_lensing_of_a_bin_is_positive_and_falls(setup):
    import jax.numpy as jnp
    from ggah_mod.observables.real_space import delta_sigma
    from ggah_mod.sectors import BaryonSplit, MatterField
    from ggah_mod.sectors.matter import cosmic_baryon_fraction
    from ggah_mod.spectra.pk import pk_cross
    from ggah_mod.spectra.spec import PkOptions

    field, cl, gp = setup
    w = cl.weights(field, gp, 20.0, 30.0)
    assert float(w.norm) > 0
    f_b = cosmic_baryon_fraction(field.cosmo)
    wm = MatterField().weights(field, {"split": BaryonSplit.from_hot(f_b, f_b)})
    p = pk_cross(field, w, wm, overlap="none", options=PkOptions())
    rp = jnp.logspace(-1, 1, 8)
    ds = np.asarray(delta_sigma(rp, (field.k, p.one_halo + p.two_halo), field.cosmo))
    assert np.all(ds > 0) and np.all(np.diff(ds) < 0)


def test_the_gradient_reaches_the_cluster_and_galaxy_parameters(setup):
    import jax
    from ggah_mod.sectors.clusters import ClusterParams

    field, cl, gp = setup

    def n(amp, s, mu_s):
        return cl.number_density(field, dict(gp, mu_s=mu_s), 20.0, 30.0,
                                 ClusterParams(lambda_amp=amp, sigma_lnlambda=s))

    g = jax.grad(n, argnums=(0, 1, 2))(1.0, 0.2, float(gp["mu_s"]))
    assert all(np.isfinite(float(x)) and float(x) != 0.0 for x in g)
