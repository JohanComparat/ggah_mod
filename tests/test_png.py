r"""Verification: scale-dependent bias from primordial non-Gaussianity.

``PLAN.md`` item **E4**.  The benchmark scored it *low* -- "a k-dependent
additive term on b(M), given the existing b(M) and transfer function" -- which
was right about the size and slightly wrong about the shape, because this
package stores no transfer function.  It works from :math:`P(k)` throughout, so
:math:`T` is recovered from the spectrum rather than read off, and that recovery
is the piece worth checking.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.sectors as S
import ggah_mod.spectra as SP
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.growth import growth_factor
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors.png import (
    P_MERGER, P_UNIVERSAL, apply_png_bias, png_bias_shift,
    transfer_from_spectrum,
)


@pytest.fixture(scope="module")
def pk():
    return make_pk("emu_pk")


@pytest.fixture(scope="module")
def field(pk):
    return make_field(PLANCK18, DIFFERENTIABLE, pk=pk, z=0.0,
                      calibration="off")


@pytest.fixture(scope="module")
def growth(pk):
    return float(growth_factor(0.0, PLANCK18, pk))


class TestTheTransferFunction:
    r"""Recovered from :math:`P(k)`, because nothing here stores one."""

    @pytest.mark.xfail(strict=True, reason=(
        "CROSS_REPO X16: emu_pk 2.0 is scored for k >= 1e-3 h/Mpc and the "
        "DIFFERENTIABLE grid starts at 1e-4, where the transfer function "
        "recovered from its P_cb is not monotone at the 2e-3 level.  Strict, "
        "so this turns red when the emulator's range covers the grid."))
    def test_it_tends_to_one_at_large_scales_and_falls_at_small(self, field):
        t = np.asarray(transfer_from_spectrum(field.k, field.pk_cb,
                                              PLANCK18.n_s))
        assert t[0] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(t) < 1e-12)              # monotone falling
        assert t[-1] < 1e-3

    def test_the_growth_factor_divides_out(self, pk):
        r"""Which is why :func:`png_bias_shift` takes :math:`D` separately and
        cannot take it from the spectrum: :math:`P(k,z) \propto T^2D^2`, and
        normalising at the first node removes :math:`D` exactly.  A spectrum at
        any redshift gives the same :math:`T`."""
        a = make_field(PLANCK18, DIFFERENTIABLE, pk=pk, z=0.0,
                       calibration="off")
        b = make_field(PLANCK18, DIFFERENTIABLE, pk=pk, z=1.0,
                       calibration="off")
        ta = np.asarray(transfer_from_spectrum(a.k, a.pk_cb, PLANCK18.n_s))
        tb = np.asarray(transfer_from_spectrum(b.k, b.pk_cb, PLANCK18.n_s))
        assert np.allclose(ta, tb, rtol=2e-3)

        # ...and the spectra themselves are *not* equal, so the agreement above
        # is the normalisation working rather than the two being the same array.
        assert not np.allclose(np.asarray(a.pk_cb), np.asarray(b.pk_cb))


class TestTheShift:
    r"""Dalal et al. (2008): :math:`\Delta b \propto f_{\rm NL}(b-p)/k^2`."""

    def test_it_vanishes_at_zero_f_nl(self, field, growth):
        db = np.asarray(png_bias_shift(field.k, 2.0, 0.0, PLANCK18,
                                       field.pk_cb, p=1.0, growth=growth))
        assert np.all(db == 0.0)

    def test_it_is_linear_in_f_nl(self, field, growth):
        a = np.asarray(png_bias_shift(field.k, 2.0, 10.0, PLANCK18,
                                      field.pk_cb, p=1.0, growth=growth))
        b = np.asarray(png_bias_shift(field.k, 2.0, 100.0, PLANCK18,
                                      field.pk_cb, p=1.0, growth=growth))
        assert np.allclose(b, 10.0 * a, rtol=1e-12)

    def test_it_goes_as_one_over_k_squared_where_the_transfer_is_flat(
            self, field, growth):
        r"""At :math:`k \ll k_{\rm eq}`, :math:`T \simeq 1` and the whole
        k-dependence is the :math:`1/k^2` -- which is the signature the
        measurement looks for, so it is worth pinning rather than assuming."""
        k = np.asarray(field.k)
        db = np.asarray(png_bias_shift(field.k, 2.0, 100.0, PLANCK18,
                                       field.pk_cb, p=1.0, growth=growth))
        sel = k < 1e-3
        slope = np.polyfit(np.log(k[sel]), np.log(db[sel]), 1)[0]
        assert slope == pytest.approx(-2.0, abs=0.02)

    def test_the_merger_convention_gives_more_signal(self, field, growth):
        """p = 1.6 against p = 1 is a 60 per cent change at b = 2, which is why
        neither is a default."""
        uni = np.asarray(png_bias_shift(field.k, 2.0, 100.0, PLANCK18,
                                        field.pk_cb, p=P_UNIVERSAL,
                                        growth=growth))
        mrg = np.asarray(png_bias_shift(field.k, 2.0, 100.0, PLANCK18,
                                        field.pk_cb, p=P_MERGER,
                                        growth=growth))
        assert np.allclose(mrg / uni, (2.0 - P_MERGER) / (2.0 - P_UNIVERSAL))
        assert mrg[0] / uni[0] == pytest.approx(0.4, rel=1e-9)

    def test_p_and_growth_are_required(self, field, growth):
        """Neither has a default, and the module says why: a silent p = 1 in a
        quasar analysis is a 60 per cent error, and the two growth conventions
        differ by about 25 per cent."""
        with pytest.raises(TypeError):
            png_bias_shift(field.k, 2.0, 10.0, PLANCK18, field.pk_cb,
                           growth=growth)
        with pytest.raises(TypeError):
            png_bias_shift(field.k, 2.0, 10.0, PLANCK18, field.pk_cb, p=1.0)

    def test_a_per_mass_bias_gives_a_per_mass_shift(self, field, growth):
        db = np.asarray(png_bias_shift(field.k, field.bias, 100.0, PLANCK18,
                                       field.pk_cb, p=1.0, growth=growth))
        assert db.shape == (field.n_k, field.n_m)
        # More biased haloes respond more, which is the (b - p) factor.
        i = int(np.argmin(np.abs(np.asarray(field.k) - 1e-3)))
        assert db[i, -1] > db[i, 0]

    def test_it_carries_a_gradient_in_f_nl(self, field, growth):
        """It is the parameter the whole term exists to constrain."""
        g = jax.grad(lambda f: jnp.sum(
            png_bias_shift(field.k, 2.0, f, PLANCK18, field.pk_cb, p=1.0,
                           growth=growth)))(10.0)
        assert np.isfinite(float(g)) and float(g) > 0.0


class TestItReachesLayerFourThroughTheExistingChannel:
    """No new plumbing, which was the claim -- verified rather than assumed."""

    @pytest.fixture(scope="class")
    def weights(self, field):
        return S.GalaxySector("zheng07", shmr="zu15",
                              backend=DIFFERENTIABLE).weights(
            field, S.galaxy_defaults("zheng07"))

    def test_the_override_is_k_dependent_and_the_integral_takes_it(
            self, field, weights, growth):
        r"""``bias_weight`` is documented as :math:`(N_M,)` and assembly bias
        supplies one; this supplies :math:`(N_k, N_M)`, and
        :func:`~ggah_mod.spectra.pk.i_of_k` broadcasts over both because it
        multiplies by a :math:`(N_k, N_M)` weight before integrating over mass.
        """
        w = apply_png_bias(weights, field, 100.0, p=1.0, growth=growth)
        assert w.bias_weight.shape == (field.n_k, field.n_m)
        i = np.asarray(SP.i_of_k(field, w))
        assert i.shape == (field.n_k,) and np.all(np.isfinite(i))

    def test_only_the_two_halo_term_moves(self, field, weights, growth):
        """A within-halo pair count knows nothing about the long-wavelength
        potential, which is why this lives on ``bias_weight`` and not on the
        profile."""
        w = apply_png_bias(weights, field, 100.0, p=1.0, growth=growth)
        plain = SP.pk_cross(field, weights, weights, overlap="identical")
        png = SP.pk_cross(field, w, w, overlap="identical")
        assert np.allclose(np.asarray(plain.one_halo),
                           np.asarray(png.one_halo), rtol=1e-12)
        assert float(png.shot) == pytest.approx(float(plain.shot), rel=1e-12)

    def test_the_effect_is_large_at_large_scales_and_negligible_at_small(
            self, field, weights, growth):
        """Measured, at ``f_NL = 100``: a factor of 9 at k = 1e-3 and 7e-4 at
        k = 1.  That shape is the whole observable -- a PNG signal that showed
        up at k = 1 would be a bug."""
        k = np.asarray(field.k)
        w = apply_png_bias(weights, field, 100.0, p=1.0, growth=growth)
        r = (np.asarray(SP.pk_cross(field, w, w, overlap="identical").two_halo)
             / np.asarray(SP.pk_cross(field, weights, weights,
                                      overlap="identical").two_halo))
        assert r[int(np.argmin(np.abs(k - 1e-3)))] > 5.0
        assert r[int(np.argmin(np.abs(k - 0.1)))] == pytest.approx(1.0, abs=0.01)
        assert r[int(np.argmin(np.abs(k - 1.0)))] == pytest.approx(1.0, abs=1e-3)

    def test_stacking_two_k_dependent_decorations_is_refused(
            self, field, weights, growth):
        """There is no general rule for which of two bias decorations should
        see the other, and guessing one would put an unstated model into every
        spectrum -- the same refusal ``pair_1h`` makes about nested pairs."""
        once = apply_png_bias(weights, field, 100.0, p=1.0, growth=growth)
        with pytest.raises(ValueError, match="already k-dependent"):
            apply_png_bias(once, field, 100.0, p=1.0, growth=growth)

    def test_it_does_not_reach_the_counterterm(self, field, weights, growth):
        r"""And must not: the counterterm's bracket is a statement about the
        (mass function, bias) pairing, and a :math:`1/k^2` divergence entering
        a normalisation condition would be a :math:`k\to0` limit that does not
        exist.  ``PLAN.md`` item **A3** settled the asymmetry before this
        arrived."""
        w = apply_png_bias(weights, field, 1000.0, p=1.0, growth=growth)
        a = np.asarray(SP.low_mass_counterterm(field, weights, field.n_k))
        b = np.asarray(SP.low_mass_counterterm(field, w, field.n_k))
        assert np.array_equal(a, b)
