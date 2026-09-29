"""The seam between ``ggah_mod`` and ``emu_pk``, measured through ``dn/dM``.

There used to be a distilled table here.  The emulator was trained on massless
LambdaCDM, so massive neutrinos and CPL dark energy arrived through
:mod:`emu_pk.ratio` -- a correction applied on top -- and this file existed to
check that the seam between the two was sound: that the correction was reached
at all, with the right arguments, and that its C1 continuity in redshift
survived the halo sector built on it.

``emu_pk``'s network carries both natively, so there is no correction to reach
and no table nodes to kink at.  Nothing in ``ggah_mod`` reads
:mod:`emu_pk.ratio` any more.

Two things still belong here, and they are the two that were never about the
table.  The redshift derivative of ``dn/dM`` has to be smooth -- an emulator
can put a kink in it as surely as an interpolant can, and this is where such a
kink would show.  And the two packages have to agree on what a cosmology *is*:
``emu_pk`` cannot import ``ggah_mod`` without a cycle, so it restates the
neutrino convention, and a network trained under a different one is wrong in a
way neither package's own tests can see.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import Cosmology, PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field

_K = np.logspace(-3, 1, 40)

#: What were the nodes of the retired correction table's redshift axis.
#: Ordinary redshifts now, kept as sample points: if a backend ever
#: interpolates in z again, a C0 interpolant would kink here first.
Z_NODES = (0.25, 0.5, 1.0, 2.0)


@pytest.fixture(scope="module")
def dndm_of_z():
    pk = make_pk("emu_pk")
    return lambda zz: jnp.log(jnp.sum(
        make_field(PLANCK18, DIFFERENTIABLE, pk, z=zz).dndm))


class TestTheDerivativeIsSmoothInRedshift:
    """Measured through `dn/dM`, which is what a forecast differentiates.

    `Z_NODES` were the nodes of the retired table's redshift axis, and are kept
    as sample points rather than as suspects: they are ordinary redshifts now,
    and the assertion is simply that autodiff and a central difference agree
    everywhere.  If a future backend ever interpolates in z again, this is the
    test that fails.
    """

    @pytest.mark.x64
    @pytest.mark.parametrize("z0", Z_NODES)
    def test_autodiff_matches_a_difference_at_every_node(self, dndm_of_z, z0):
        ad = float(jax.grad(dndm_of_z)(z0))
        fd = float((dndm_of_z(z0 + 1e-5) - dndm_of_z(z0 - 1e-5)) / 2e-5)
        assert ad == pytest.approx(fd, rel=1e-6), f"kink at z = {z0}"

    @pytest.mark.x64
    @pytest.mark.parametrize("z0", [0.37, 0.73, 1.5])
    def test_and_off_node_too(self, dndm_of_z, z0):
        """Off-node was always fine; if it were not, the problem would be
        somewhere other than the interpolant."""
        ad = float(jax.grad(dndm_of_z)(z0))
        fd = float((dndm_of_z(z0 + 1e-5) - dndm_of_z(z0 - 1e-5)) / 2e-5)
        assert ad == pytest.approx(fd, rel=1e-6)

    @pytest.mark.x64
    def test_a_node_is_no_worse_than_an_off_node_point(self, dndm_of_z):
        """The statement the fix is actually making: nodes stop being special."""
        def err(z0):
            ad = float(jax.grad(dndm_of_z)(z0))
            fd = float((dndm_of_z(z0 + 1e-5) - dndm_of_z(z0 - 1e-5)) / 2e-5)
            return abs(ad / fd - 1.0)
        assert max(err(0.5), err(1.0)) < 100.0 * max(err(0.37), err(0.73))


class TestDarkEnergyReachesTheSpectrum:
    def test_dark_energy_reaches_the_spectrum(self):
        """Now a network input; once a gap the four-axis table had to close.

        Through a LambdaCDM-trained network this derivative was not small, it
        was identically zero -- which a Fisher matrix reads as a flat direction
        rather than as an error.  The assertion costs nothing and is the one
        that would notice a backend regressing to that.
        """
        pk = make_pk("emu_pk")

        def amp(w0):
            c = PLANCK18.replace(w0=w0)
            return jnp.log(jnp.sum(pk.pk(_K, 0.0, c)))

        g = float(jax.grad(amp)(-1.0))
        assert abs(g) > 1e-3, "dP/dw0 is still absent through this backend"


class TestTheConventionsMatchAcrossThePackages:
    def test_emu_pk_restates_ggah_mod_densities_exactly(self):
        """`emu_pk` cannot import `ggah_mod` -- it would be a cycle -- so it
        restates the neutrino convention.  A table built under a different one
        is wrong in a way neither package's own tests can see."""
        from emu_pk import cosmo as E
        import ggah_mod.cosmology.constants as C
        assert E.NU_DENOM_EV == C.NU_DENOM_EV
        assert E.N_EFF == C.N_EFF and E.N_NU_MASSIVE == C.N_NU_MASSIVE
        for mnu in (0.0, 0.06, 0.3):
            c = Cosmology.create(sum_mnu=mnu)
            # One pressureless density on both sides since 0.9.8: `emu_pk`'s
            # `f_nu` is `Cosmology.f_nu`, the fraction `Omega_cb` excludes.
            assert E.f_nu(mnu, float(c.h), float(c.Omega_m)) == pytest.approx(
                float(c.f_nu), rel=1e-12)

    def test_the_emulator_and_this_package_mean_one_f_nu(self):
        """**The divergence of X13, closed by a convention rather than a retrain.**

        Until 0.9.8 this package's `f_nu` was 0.46 per cent above `emu_pk`'s:
        `Omega_cb` subtracted the Komatsu fit's rest mass at :math:`N_{\rm
        eff}/3` degeneracy while `emu_pk` -- and CLASS, which trained it --
        counted :math:`\Sigma m_\nu/93.14`.  The massive states now sit at
        CLASS's temperature, so the rest mass here *is* CLASS's, and `emu_pk`
        2.0.1 restates the derived denominator: the two fractions are one
        number.  Recorded as `CROSS_REPO.md` X13, applied.
        """
        from emu_pk import cosmo as E
        gaps = []
        for mnu in (0.06, 0.3, 0.6):
            c = Cosmology.create(sum_mnu=mnu)
            theirs = E.f_nu(mnu, float(c.h), float(c.Omega_m))
            gaps.append(float(c.f_nu) / theirs - 1.0)
        assert all(abs(g) < 1e-12 for g in gaps), gaps
