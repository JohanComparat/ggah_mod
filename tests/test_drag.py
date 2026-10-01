"""The sound horizon and the drag epoch, without a Boltzmann code.

Each test states one property :func:`~ggah_mod.cosmology.drag.r_drag` must
keep.  The comparisons against CLASS and CAMB themselves are in
``test_drag_reference.py`` (slow); here the CLASS numbers are the ones the
calibration stored, which is what makes these fast and what makes them fail if
the fit is regenerated against a different CLASS without anyone looking.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import PLANCK18, Cosmology, r_drag, sound_horizon, z_drag
from ggah_mod.cosmology import _zdrag_coefficients as K
from ggah_mod.cosmology.drag import (
    FEATURES, check_drag_domain, dark_energy_fraction_at_drag,
    drag_domain_violations, terms,
)

LEAVES = ("Omega_m", "Omega_b", "h", "n_s", "ln10A_s", "sum_mnu", "w0", "wa",
          "Omega_k", "T_cmb")


class TestTheSoundHorizon:
    def test_it_falls_with_redshift(self):
        z = jnp.array([0.0, 10.0, 300.0, 1060.0, 3e3, 1e5, 1e8])
        rs = np.asarray(sound_horizon(z, PLANCK18))
        assert np.all(np.diff(rs) < 0.0) and np.all(rs > 0.0)

    def test_deep_in_radiation_it_is_the_radiation_integral(self):
        r""":math:`D_H/\sqrt{3\Omega_r}(1+z)` at :math:`z = 10^9`, where matter
        is 3e-6 of the density and :math:`R` is 7e-7: the quadrature and the
        closed-form tail together, checked where both matter."""
        c = PLANCK18
        z = 1e9
        expect = float(c.hubble_distance) / (
            np.sqrt(3.0 * float(c.Omega_gamma + c.Omega_nu_rel)) * (1.0 + z))
        assert float(sound_horizon(z, c)[0]) == pytest.approx(expect, rel=1e-5)

    def test_it_is_in_mpc_per_h(self):
        """Like every distance here: dividing by h gives CLASS's Mpc."""
        rs_mpc = float(r_drag(PLANCK18)) / float(PLANCK18.h)
        assert 140.0 < rs_mpc < 155.0


class TestTheDragEpoch:
    def test_it_reproduces_the_class_numbers_the_calibration_stored(self):
        """At the fiducial: z_d to the held-out tolerance, r_d to 2e-5."""
        v = K.VALIDATION
        assert float(z_drag(PLANCK18)) == pytest.approx(
            v["z_d_fiducial_class"], abs=v["target_abs_dz_heldout"])
        assert float(r_drag(PLANCK18)) / float(PLANCK18.h) == pytest.approx(
            v["rs_d_fiducial_class_mpc"], rel=2e-5)

    def test_the_basis_is_the_one_the_coefficients_were_fitted_in(self):
        assert len(terms(K.DEGREE)) == len(K.COEFFS)
        assert len(K.CENTRE) == len(FEATURES)

    def test_the_calibration_met_its_own_targets(self):
        v = K.VALIDATION
        assert v["max_abs_dz_heldout"] <= v["target_abs_dz_heldout"]
        assert v["max_abs_dz_corners"] <= v["target_abs_dz_corners"]


class TestTheGradient:
    @pytest.mark.parametrize("cosmo", [
        PLANCK18,
        Cosmology.create(sum_mnu=0.0, nu_hierarchy="massless"),
        Cosmology.create(sum_mnu=0.12, nu_hierarchy="inverted", Omega_k=-0.05),
    ], ids=["fiducial", "massless", "inverted-closed"])
    def test_it_is_finite_and_blind_to_the_primordial_spectrum(self, cosmo):
        """Finite in every leaf, and exactly zero in the two the background
        never sees.

        One exception, and it is not this module's: at the massless point
        :func:`~ggah_mod.cosmology.background.hubble_e` itself has a ``nan``
        derivative in ``sum_mnu`` -- the edge of the mass domain, shared by
        :func:`~ggah_mod.cosmology.background.comoving_distance` -- so
        :math:`r_s` inherits it there.  The fit's own derivative must still be
        finite at ``sum_mnu = 0``, where a monomial written as ``x ** 0`` would
        not be.
        """
        g = jax.grad(r_drag)(cosmo)
        massless = cosmo.nu_hierarchy == "massless"
        for p in LEAVES:
            if massless and p == "sum_mnu":
                continue
            assert np.isfinite(float(getattr(g, p))), p
        assert float(g.n_s) == 0.0 and float(g.ln10A_s) == 0.0
        gz = jax.grad(z_drag)(cosmo)
        for p in LEAVES:
            assert np.isfinite(float(getattr(gz, p))), p

    def test_it_jits(self):
        f = jax.jit(r_drag)
        assert float(f(PLANCK18)) == pytest.approx(float(r_drag(PLANCK18)), rel=1e-14)


class TestTheDomain:
    def test_the_fiducial_is_inside(self):
        assert drag_domain_violations(PLANCK18) == {}
        check_drag_domain(PLANCK18)

    def test_a_baryon_density_off_the_box_is_refused_by_name(self):
        c = PLANCK18.replace(Omega_b=0.032 / float(PLANCK18.h) ** 2)
        with pytest.raises(ValueError, match="omega_b"):
            z_drag(c)

    def test_dark_energy_at_recombination_is_refused(self):
        r"""Inside the ``emu_pk`` box, but with :math:`1+w_0+w_a = 1.1` dark
        energy is 70 per cent of the density at the drag epoch and moves
        :math:`z_d` by tens -- a regime the fit was never asked to cover."""
        c = PLANCK18.replace(w0=-0.5, wa=0.6)
        assert abs(float(dark_energy_fraction_at_drag(c))) > K.F_DE_MAX
        with pytest.raises(ValueError, match="dark_energy_fraction_at_drag"):
            r_drag(c)

    def test_desi_like_dark_energy_is_inside(self):
        """:math:`w_0 = -0.75`, :math:`w_a = -0.86`: the region a BAO analysis
        is likely to be in, and well inside the bound."""
        check_drag_domain(PLANCK18.replace(w0=-0.75, wa=-0.86))

    def test_under_a_trace_the_check_steps_aside(self):
        """A check that raised on a tracer would break the gradient it guards."""
        c = PLANCK18.replace(Omega_b=0.032 / float(PLANCK18.h) ** 2)
        assert np.isfinite(float(jax.jit(r_drag)(c)))


class TestTheBBNRecord:
    """The helium fraction behind the fit is CLASS's, from a named table; the
    record says which, and these say the record is whole."""

    def test_every_assumption_is_recorded(self):
        for key in ("class_version", "recombination", "sbbn_file", "sbbn_header",
                    "bbn_code", "neutron_lifetime_s", "helium_quantity", "n_eff",
                    "delta_n_looked_up", "delta_n_convention", "t_cmb_rescaling",
                    "standard_bbn", "YHe_fiducial", "dzd_dYHe", "dlnrd_dYHe",
                    "YHe_range_in_box"):
            assert K.BBN.get(key) is not None, key

    def test_it_is_the_table_the_package_pins(self):
        from ggah_mod.cosmology.power import ClassPk
        assert K.BBN["sbbn_file"] == ClassPk.SBBN_FILE

    def test_the_numbers_are_the_measured_ones(self):
        """PArthENoPE at 880.2 s, Delta N = -0.002, Y_He ~ 0.2454, and a
        sensitivity that makes the table choice a 1e-6 question."""
        b = K.BBN
        assert b["neutron_lifetime_s"] == 880.2
        assert b["delta_n_looked_up"] == pytest.approx(-0.002, abs=1e-9)
        assert b["YHe_fiducial"] == pytest.approx(0.2454, abs=5e-4)
        assert -0.03 < b["dlnrd_dYHe"] < -0.01
