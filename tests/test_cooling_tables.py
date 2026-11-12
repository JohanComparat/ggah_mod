r"""The cooling-table builders and readers, run without ``soxs``.

``build``, ``build_wide`` and ``build_band`` call ``soxs.ApecGenerator`` and
nothing else from it.  A stand-in generator with an exponential spectrum,
:math:`d\Lambda/dE = A\,Z\,e^{-E/kT}`, has exact band integrals, so the tables
it produces can be read back through :class:`ApecCooling` and
:class:`BandCooling` and compared with the closed form: the layout the builders
write is the one the readers expect, and the K-correction of an observed band is
the rest-frame band at :math:`E(1+z)`.
"""
import sys
import types

import numpy as np
import pytest

from ggah_mod.sectors import cooling as CL

AMP = 1e-23


def _band(kt, z, e1, e2):
    """:math:`\\int_{e_1}^{e_2} A Z e^{-E/kT} dE`."""
    return AMP * z * kt * (np.exp(-e1 / kt) - np.exp(-e2 / kt))


class _Quantity:
    def __init__(self, value):
        self.value = value


class _Spectrum:
    def __init__(self, ebins, kt, z):
        self.ebins = _Quantity(ebins)
        flux = 1e14 * _band(kt, z, ebins[:-1], ebins[1:])   # APEC's norm
        self.binned_energy_flux = _Quantity(flux)
        self.total_energy_flux = _Quantity(flux.sum())


class _Generator:
    def __init__(self, emin, emax, nbins, binscale="linear", broadening=True):
        space = np.geomspace if binscale == "log" else np.linspace
        self.ebins = space(emin, emax, nbins + 1)

    def get_spectrum(self, kt, z, redshift=0.0, norm=1.0):
        return _Spectrum(self.ebins, kt, z)


@pytest.fixture
def soxs(monkeypatch):
    stub = types.ModuleType("soxs")
    stub.ApecGenerator = _Generator
    stub.__version__ = "stub"
    monkeypatch.setitem(sys.modules, "soxs", stub)
    return stub


class TestTheFixedBandTable:
    def test_built_and_read_back_it_is_the_band_integral(self, soxs, tmp_path):
        path = CL.build(out=tmp_path / "apec.npz", n_t=12, n_z=4, nbins=400)
        t = CL.load(path)
        assert (float(t["emin"]), float(t["emax"])) == (0.5, 2.0)
        cool = CL.ApecCooling(t, nh_over_ne=1.0)
        kt, z = 10.0 ** t["log10_kt"][5], 10.0 ** t["log10_z"][2]
        assert float(cool(kt, z)) == pytest.approx(_band(kt, z, 0.5, 2.0), rel=1e-10)
        assert float(cool(kt, z, nh_over_ne=0.8)) == pytest.approx(
            0.8 * _band(kt, z, 0.5, 2.0), rel=1e-10)

    def test_the_wide_table_extends_the_shipped_nodes_below(self, soxs, tmp_path):
        kt_min, n_t = CL._wide_grid()
        lo, hi, n = CL.SHIPPED_KT
        assert n_t == n + CL.WIDE_EXTRA
        step = (np.log10(hi) - np.log10(lo)) / (n - 1)
        assert np.log10(kt_min) == pytest.approx(np.log10(lo) - CL.WIDE_EXTRA * step)
        t = CL.load(CL.build_wide(out=tmp_path / "wide.npz", n_z=2, nbins=20))
        assert t["log10_kt"].size == n_t
        np.testing.assert_allclose(t["log10_kt"][CL.WIDE_EXTRA:],
                                   np.linspace(np.log10(lo), np.log10(hi), n), atol=1e-12)
        assert isinstance(CL.ApecCoolingWide(t), CL.ApecCooling)

    def test_a_missing_table_raises_with_the_command_that_builds_it(self, tmp_path,
                                                                     monkeypatch):
        with pytest.raises(FileNotFoundError, match="python -m ggah_mod.sectors.cooling"):
            CL.load(tmp_path / "absent.npz")
        monkeypatch.setattr(CL, "WIDE_TABLE", tmp_path / "absent_wide.npz")
        with pytest.raises(FileNotFoundError, match="--wide"):
            CL.ApecCoolingWide()


class TestTheBandTable:
    @pytest.fixture
    def table(self, soxs, tmp_path):
        path = CL.build_band(out=tmp_path / "band.npz", n_t=10, n_z=3,
                             n_e=600, e_range=(0.1, 40.0))
        return CL.load_band.__wrapped__(path)

    def test_a_rest_frame_band_is_its_integral(self, table):
        cool = CL.BandCooling(table, nh_over_ne=1.0, band=(0.5, 2.0))
        kt, z = 10.0 ** table["log10_kt"][6], 10.0 ** table["log10_z"][1]
        assert float(cool(kt, z)) == pytest.approx(_band(kt, z, 0.5, 2.0), rel=2e-3)

    def test_an_observed_band_is_the_rest_frame_band_moved_by_one_plus_z(self, table):
        cool = CL.BandCooling(table, nh_over_ne=1.0, band=(0.5, 2.0))
        kt, z = 10.0 ** table["log10_kt"][6], 10.0 ** table["log10_z"][1]
        assert float(cool(kt, z, z_obs=1.0)) == pytest.approx(
            _band(kt, z, 1.0, 4.0), rel=2e-3)
        assert float(cool(kt, z, band=(2.0, 10.0))) == pytest.approx(
            _band(kt, z, 2.0, 10.0), rel=2e-3)

    def test_it_reports_which_bands_it_covers(self, table):
        cool = CL.BandCooling(table, band=(0.5, 2.0))
        assert cool.covers()
        assert cool.covers(band=(2.0, 10.0), z_obs=2.0)
        assert not cool.covers(band=(10.0, 30.0), z_obs=1.0)

    def test_a_per_halo_redshift_is_refused(self, table):
        cool = CL.BandCooling(table)
        with pytest.raises(ValueError, match="scalars"):
            cool(1.0, 0.3, z_obs=np.array([0.1, 0.2]))

    def test_a_float32_cumulative_table_is_refused(self, table):
        old = {k: v for k, v in table.items() if k != "upper"}
        with pytest.raises(ValueError, match="float64 upper-tail"):
            CL.BandCooling(old)

    def test_a_missing_band_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--band"):
            CL.load_band.__wrapped__(tmp_path / "absent.npz")
