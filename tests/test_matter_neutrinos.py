r"""Where the neutrinos are in the matter field: in no halo.

``dn/dM`` counts cold mass, so every per-halo fraction is of the cold mass and
the baryons' ceiling is :math:`\Omega_b/\Omega_{cb}`.  The neutrinos enter the
matter weights only through the :math:`M/\bar\rho_m` prefactor.  These tests
hold the convention in both places it can be broken: the fraction itself, and
the cosmic census, which is the one check that sees it.
"""
import pytest

import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import Cosmology, PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors.census import census
from ggah_mod.sectors.matter import (
    BaryonSplit, cosmic_baryon_fraction, f_collisionless,
)


class TestTheFractionIsOfTheColdMass:

    def test_the_ceiling_is_against_the_cold_density(self):
        c = PLANCK18
        assert float(cosmic_baryon_fraction(c)) == pytest.approx(
            float(c.Omega_b / c.Omega_cb), rel=1e-15)

    def test_the_old_convention_is_short_by_one_minus_f_nu(self):
        """And the shortfall is really there at the fiducial, so the identity
        is not being checked at a point where both sides are one."""
        c = PLANCK18
        ratio = float((c.Omega_b / c.Omega_m) / cosmic_baryon_fraction(c))
        assert ratio == pytest.approx(1.0 - float(c.f_nu), rel=1e-14)
        assert ratio < 0.996

    def test_the_dark_matter_share_has_no_neutrinos(self):
        c = PLANCK18
        assert float(f_collisionless(cosmic_baryon_fraction(c))) == \
            pytest.approx(float(c.Omega_cdm / c.Omega_cb), rel=1e-14)

    def test_without_neutrinos_the_two_conventions_are_one_number(self):
        c = Cosmology.create(sum_mnu=0.0)
        assert float(cosmic_baryon_fraction(c)) == float(c.Omega_b / c.Omega_m)

    def test_a_split_built_from_it_carries_no_neutrinos(self):
        c = PLANCK18
        f_b = cosmic_baryon_fraction(c)
        split = BaryonSplit.from_hot(f_b, jnp.full(4, 0.1), f_star_cen=0.02)
        assert float(split.f_collisionless[0]) == pytest.approx(
            float(c.Omega_cdm / c.Omega_cb), rel=1e-14)
        assert float(split.f_baryon[0]) == pytest.approx(float(f_b), rel=1e-14)


@pytest.mark.slow
class TestTheCensusSeesTheConvention:
    """The per-halo budget closes either way; the cosmic one does not."""

    @pytest.fixture(scope="class")
    def field(self):
        return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)

    @staticmethod
    def _census(field, f_b):
        lm = jnp.log10(field.m)
        return census(field, BaryonSplit.from_hot(f_b, E.f_gas_sigmoid(lm, f_b)))

    def test_the_cold_convention_closes(self, field):
        c = self._census(field, cosmic_baryon_fraction(PLANCK18))
        assert abs(float(c.closure_residual())) < 1e-12

    def test_the_closed_form_has_no_neutrino_factor(self, field):
        c = self._census(field, cosmic_baryon_fraction(PLANCK18))
        f = float(c.mass_fraction_in_halos)
        assert float(c.omega_outside_direct) == pytest.approx(
            PLANCK18.Omega_b * (1.0 - f), rel=1e-15)

    def test_the_total_matter_convention_is_caught_by_its_size(self, field):
        r"""Built from :math:`\Omega_b/\Omega_m`, the resolved baryons are
        :math:`\Omega_b(1-f_\nu)F`, so the residual is
        :math:`f_\nu F/(1-F)` -- compared as a value, not as "non-zero"."""
        c = self._census(field, PLANCK18.Omega_b / PLANCK18.Omega_m)
        f, f_nu = float(c.mass_fraction_in_halos), float(PLANCK18.f_nu)
        assert float(c.closure_residual()) == pytest.approx(
            f_nu * f / (1.0 - f), rel=1e-6)
        assert float(c.closure_residual()) > 1e-3
