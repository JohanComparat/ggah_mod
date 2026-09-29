r"""Verification: ``ggah_mod`` driven as a cobaya ``Theory``.

``PLAN.md`` item **E1**.  The benchmark scored this *low* cost because "the
parameter and unit mapping is written and tested in
``ggah_bench/adapters/cobaya.py``" -- but that adapter drives cobaya's Boltzmann
wrappers *from* the benchmark, and this is the other direction.  The mapping is
not the same one reversed, because the conversions that matter here are the two
this package is opinionated about: what :math:`\Omega_m` contains, and which
spelling of the amplitude is an input.

The conversion tests do not need cobaya and do not skip.  Only the ones that put
a real ``Model`` together do.
"""
import numpy as np
import pytest

from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.constants import NU_DENOM_EV

cobaya = pytest.importorskip("cobaya", reason="cobaya is an optional extra")

from ggah_mod.interfaces.cobaya import (            # noqa: E402
    REFUSED, GgahMod, cosmology_from_cobaya,
)

#: Planck 2018 in cobaya's spelling.
P18 = dict(H0=67.66, ombh2=0.02242, omch2=0.11933, ns=0.9665, logA=3.044,
           mnu=0.06)


class TestTheConversion:
    r"""The one line that matters, and the two this package refuses."""

    def test_omega_m_contains_the_neutrinos(self):
        r"""``omch2`` is **cold dark matter only**, and this package's
        :math:`\Omega_m` contains the neutrinos, so their density has to be
        added.  Every other interface writes
        :math:`(\omega_b+\omega_c)/h^2`, which omits them.

        At :math:`\Sigma m_\nu = 0.06` eV that is 0.0014 in :math:`\Omega_m` --
        half a per cent: far too small to look like a bug and far too large to
        ignore in a posterior.
        """
        c = cosmology_from_cobaya(**P18)
        h = P18["H0"] / 100.0
        naive = (P18["ombh2"] + P18["omch2"]) / h ** 2
        assert float(c.Omega_m) > naive
        # The added density is `Omega_nu_matter`, the matter-like part of the
        # relic energy integral, because that is what `Omega_cdm` subtracts on
        # the way back out.  It sits 0.46 per cent above the 93.14 convention,
        # which is why the tolerance is loose enough to admit either and the
        # round-trip test below is what pins which one.
        assert float(c.Omega_m) - naive == pytest.approx(
            float(c.Omega_nu_matter), rel=1e-12)
        assert float(c.Omega_m) - naive == pytest.approx(
            P18["mnu"] / NU_DENOM_EV / h ** 2, rel=1e-2)
        assert float(c.Omega_m) - naive == pytest.approx(0.00141, abs=1e-5)

    def test_the_round_trip_is_exact(self):
        """cobaya's physical densities out again: the check that the addition
        above went into the right term rather than merely somewhere."""
        c = cosmology_from_cobaya(**P18)
        h2 = float(c.h) ** 2
        assert float(c.Omega_b) * h2 == pytest.approx(P18["ombh2"], rel=1e-12)
        assert float(c.Omega_cdm) * h2 == pytest.approx(P18["omch2"], rel=1e-12)
        assert float(c.sum_mnu) == pytest.approx(P18["mnu"], rel=1e-12)
        assert float(c.h) == pytest.approx(P18["H0"] / 100.0, rel=1e-12)

    def test_it_reproduces_the_shipped_fiducial_to_the_rounding_in_it(self):
        r"""Planck 2018 in cobaya's parameters is PLANCK18 -- to 0.34 per cent,
        and the discrepancy is in the fiducial rather than in the conversion.

        :data:`~ggah_mod.cosmology.PLANCK18` carries a **round**
        :math:`\Omega_m = 0.31`; deriving it from the published physical
        densities gives 0.31105.  So the two are not the same cosmology to
        better than a third of a per cent, which matters when a cobaya chain
        started from :math:`\omega_b`, :math:`\omega_c` is compared against a
        number this package computed at its own fiducial.  Recorded here rather
        than hidden in a loose tolerance, because it is a property of the
        fiducial and would otherwise read as an error in the mapping.
        """
        c = cosmology_from_cobaya(**P18)
        assert float(c.Omega_m) == pytest.approx(0.31105, abs=1e-5)
        assert float(PLANCK18.Omega_m) == 0.31
        assert float(c.Omega_m) / float(PLANCK18.Omega_m) - 1.0 == \
            pytest.approx(3.4e-3, abs=2e-4)
        # The amplitude, which is the same number in both spellings.
        assert float(c.ln10A_s) == pytest.approx(float(PLANCK18.ln10A_s),
                                                 rel=1e-6)

    @pytest.mark.parametrize("bad", sorted(REFUSED))
    def test_the_refused_inputs_are_refused_with_a_reason(self, bad):
        """``As``, ``sigma8`` and ``S8`` are decision 1 -- one amplitude, and
        the rest derived from the spectrum."""
        with pytest.raises(ValueError, match=bad):
            cosmology_from_cobaya(**dict(P18, **{bad: 0.5}))

    def test_curvature_is_not_on_the_refusal_list_any_more(self):
        r"""It was, for one commit, and the refusal was right at the time.

        ``omk`` was refused because ``Omega_de`` was derived from flatness, so
        the package would have taken the parameter and ignored it -- a posterior
        on something nothing reads is worse than no posterior.  ``PLAN.md`` item
        **E2** landed the commit after this module and made ``Omega_de`` close
        against :math:`\Omega_k`, so the refusal expired.

        Pinned in both directions because the failure here is silent: a
        curvature run whose ``omk`` never reached ``Cosmology`` would return
        flat numbers under a curved label.
        """
        assert "omk" not in REFUSED
        for omk in (-0.05, 0.0, 0.05):
            c = cosmology_from_cobaya(**dict(P18, omk=omk))
            assert float(c.Omega_k) == pytest.approx(omk, abs=1e-15)
        # And it reaches the geometry rather than merely the container.
        from ggah_mod.cosmology.background import hubble_e
        flat = cosmology_from_cobaya(**dict(P18, omk=0.0))
        open_ = cosmology_from_cobaya(**dict(P18, omk=0.05))
        assert float(hubble_e(1.0, open_)) > float(hubble_e(1.0, flat))

    def test_omk_is_declared_optional_rather_than_demanded(self):
        """``None`` here would make cobaya require every flat analysis to say
        so; a default of ``0.0`` makes curvature something a run opts into."""
        from ggah_mod.interfaces.cobaya import GgahMod
        assert GgahMod.params["omk"] == 0.0

    def test_a_massless_cosmology_still_works(self):
        c = cosmology_from_cobaya(**dict(P18, mnu=0.0))
        assert float(c.sum_mnu) == 0.0
        assert float(c.Omega_m) == pytest.approx(
            (P18["ombh2"] + P18["omch2"]) / (P18["H0"] / 100.0) ** 2, rel=1e-12)


@pytest.fixture(scope="module")
def model():
    from cobaya.model import get_model

    return get_model({
        "params": dict(P18),
        "theory": {"ggah_mod.interfaces.cobaya.GgahMod": {
            "backend": "differentiable", "pk": "emu_pk",
                         "nu_hierarchy": "degenerate"}},
        "likelihood": {"one": None},
    })


@pytest.fixture(scope="module")
def provider(model):
    model.add_requirements({
        "Hubble": {"z": [0.0, 0.5, 1.0]},
        "comoving_radial_distance": {"z": [0.5]},
        "angular_diameter_distance": {"z": [0.5]},
        "Pk_grid": {"z": [0.0, 0.5], "k_max": 10.0},
        "sigma8_z": {"z": [0.0]},
        "fsigma8": {"z": [0.5]},
        "ggah_fields": {"z": [0.3]},
    })
    model.logposterior({})
    return model.provider


@pytest.mark.slow
class TestItRunsUnderCobaya:
    """A real ``Model``, resolved the way a likelihood would resolve it."""

    def test_the_background_comes_back_in_cobayas_units(self, provider):
        r"""``H`` in km/s/Mpc and distances in **Mpc**, not Mpc/h -- the
        opposite of this package's convention, and the conversion the benchmark
        calls "a factor of :math:`h` between two codes is a 30 per cent error
        that reads as a physics result"."""
        h_z = np.asarray(provider.get_Hubble([0.0, 0.5, 1.0]))
        assert h_z[0] == pytest.approx(P18["H0"], rel=1e-6)
        assert np.all(np.diff(h_z) > 0.0)

        chi = float(np.asarray(provider.get_comoving_radial_distance([0.5]))[0])
        assert 1800.0 < chi < 2100.0                     # Mpc, not Mpc/h
        d_a = float(np.asarray(
            provider.get_angular_diameter_distance([0.5]))[0])
        assert d_a == pytest.approx(chi / 1.5, rel=1e-6)

    def test_sigma8_and_fsigma8_are_the_planck_values(self, provider):
        assert float(np.asarray(provider.get_sigma8_z([0.0]))[0]) == \
            pytest.approx(0.810, abs=0.01)
        assert float(np.asarray(provider.get_fsigma8([0.5]))[0]) == \
            pytest.approx(0.472, abs=0.02)

    def test_the_spectrum_takes_its_redshifts_from_the_requirement(self, provider):
        """Not from the theory block's ``z_halo``.

        cobaya resolves its graph once and a quantity's arguments arrive at
        ``must_provide``, so getting this backwards gives a likelihood that
        asked for one redshift an array of the right shape at the wrong epoch.
        """
        k, z, p = provider.get_Pk_grid()
        assert list(np.asarray(z)) == pytest.approx([0.0, 0.5])
        assert p.shape == (2, k.size)
        assert np.all(p[0] > p[1])                        # growth, at fixed k

    def test_the_spectrum_is_in_cobayas_units(self, provider):
        r""":math:`k` in 1/Mpc and :math:`P` in Mpc^3, so both carry an
        :math:`h` relative to this package's."""
        from ggah_mod.backend import DIFFERENTIABLE

        k, _, _ = provider.get_Pk_grid()
        h = P18["H0"] / 100.0
        assert k[0] == pytest.approx(DIFFERENTIABLE.k_min * h, rel=1e-9)
        assert k[-1] == pytest.approx(DIFFERENTIABLE.k_max * h, rel=1e-9)

    def test_a_nonlinear_request_is_refused(self, provider):
        """There is no halofit here. Returning the linear spectrum for a
        ``nonlinear=True`` request is the silent kind of wrong."""
        with pytest.raises(ValueError, match="no non-linear"):
            provider.get_Pk_grid(nonlinear=True)

    def test_the_halo_fields_are_what_make_the_adapter_worth_having(self, provider):
        """A likelihood wanting only a background and a linear spectrum is
        better served by CAMB. One call gives the halo chain at the requested
        redshifts, from the same cosmology the background came from."""
        fields = provider.get_result("ggah_fields")
        assert len(fields) == 1
        f = fields[0]
        assert float(f.z) == pytest.approx(0.3, abs=1e-6)
        assert f.dndm.shape == (f.n_m,) and np.all(np.asarray(f.dndm) > 0.0)

    def test_a_k_max_beyond_the_flavour_is_refused(self):
        """Widening the grid here would change ``sigma(M)`` too, because
        ``make_field``'s k grid is also its quadrature."""
        from cobaya.model import get_model

        m = get_model({
            "params": dict(P18),
            "theory": {"ggah_mod.interfaces.cobaya.GgahMod": {
                "backend": "differentiable", "pk": "emu_pk",
                         "nu_hierarchy": "degenerate"}},
            "likelihood": {"one": None},
        })
        with pytest.raises(ValueError, match="stops at"):
            m.add_requirements({"Pk_grid": {"z": [0.0], "k_max": 1e4}})
