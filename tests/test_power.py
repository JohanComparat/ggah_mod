"""The linear-P(k) backends and the contract they share."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.cosmology import Cosmology, PLANCK18
from ggah_mod.cosmology.parameters import Cosmology as _C, LEAF_FIELDS
from ggah_mod.cosmology.power import make_pk, PK_BACKENDS, LinearPowerSpectrum

pytestmark = pytest.mark.slow

_K = np.logspace(-3, 1, 60)


@pytest.fixture(scope="module")
def klass():
    return make_pk("class")


@pytest.fixture(scope="module")
def camb():
    return make_pk("camb")


class TestRegistry:
    def test_make_pk_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown P\\(k\\) backend"):
            make_pk("halofit-ish")

    @pytest.mark.parametrize("name", sorted(PK_BACKENDS))
    def test_satisfies_the_protocol(self, name):
        from conftest import constructible
        pk = constructible(name)
        assert isinstance(pk, LinearPowerSpectrum)
        assert isinstance(pk.has_native_z, bool)
        assert isinstance(pk.differentiable, bool)


class TestColdVersusTotal:
    def test_identical_when_massless(self, klass):
        """No mass, no free-streaming: the cold field IS all the matter."""
        c = Cosmology.create(sum_mnu=0.0)
        np.testing.assert_array_equal(np.asarray(klass.pk(_K, 0.0, c)),
                                      np.asarray(klass.pk_cb(_K, 0.0, c)))

    @pytest.mark.parametrize("mnu", [0.06, 0.30])
    def test_cold_exceeds_total_when_massive(self, klass, mnu):
        c = Cosmology.create(sum_mnu=mnu)
        ratio = np.asarray(klass.pk_cb(_K, 0.0, c)) / np.asarray(klass.pk(_K, 0.0, c))
        assert np.all(ratio >= 1.0 - 1e-12)
        assert ratio[-1] > 1.0 + 1e-3           # suppression grows to small scales
        assert ratio[0] == pytest.approx(1.0, abs=5e-3)   # ...and vanishes on large

    def test_suppression_grows_with_mass(self, klass):
        r = []
        for mnu in (0.06, 0.30):
            c = Cosmology.create(sum_mnu=mnu)
            r.append(float(np.asarray(klass.pk_cb(np.array([1.0]), 0.0, c))[0]
                           / np.asarray(klass.pk(np.array([1.0]), 0.0, c))[0]))
        assert r[1] > r[0] > 1.0


class TestSolversAgree:
    """CLASS and CAMB are independent implementations; disagreement is a bug in
    one of them, so this is the strongest check available without data."""

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_shape_agreement(self, klass, camb, mnu):
        c = Cosmology.create(sum_mnu=mnu)
        a = np.asarray(klass.pk(_K, 0.0, c))
        b = np.asarray(camb.pk(_K, 0.0, c))
        r = (a / b) / np.median(a / b)
        assert np.max(np.abs(r - 1.0)) < 0.01, "shape"
        assert abs(np.median(a / b) - 1.0) < 0.01, "amplitude"

    def test_cold_shape_agreement(self, klass, camb):
        c = Cosmology.create(sum_mnu=0.06)
        a = np.asarray(klass.pk_cb(_K, 0.0, c))
        b = np.asarray(camb.pk_cb(_K, 0.0, c))
        r = (a / b) / np.median(a / b)
        assert np.max(np.abs(r - 1.0)) < 0.01


class TestTheVarianceIntegralIsSupportedToKMax:
    r"""The spectrum has to be right where :math:`\sigma(M)` actually looks.

    The differentiable flavour's ``k_max`` is 200, and the :math:`\sigma(M)`
    quadrature runs the whole way, so a backend that is excellent to
    :math:`k = 10` and wrong above it produces a variance that is wrong at
    every mass while looking perfectly well supported.

    That is not hypothetical.  The predecessor network stopped at
    :math:`9.81\,\mathrm{Mpc}^{-1}` --- :math:`14.6\,h/`Mpc at the fiducial
    :math:`h` --- and ``jnp.interp`` does not fail outside its grid, it
    answers: it returns the last value, unchanged, for every mode above the
    last one.  Every mass integrated a plateau across a decade where the
    spectrum should have fallen by three, which made :math:`\sigma` at
    :math:`10^{8}\,\msunh` too large by 60 per cent.  A power-law continuation
    was fitted over the gap; ``emu_pk`` was instead trained to
    :math:`k = 200`, so there is no gap to continue across.

    What is pinned here is the *consequence*, not the mechanism: the variance
    a caller gets agrees with a Boltzmann solver over the whole band the
    integral uses.  That statement outlives whichever backend is in place, and
    it is the one that failed when the mechanism was wrong.
    """

    _M = np.array([1e8, 1e10, 1e12, 1e14])

    @staticmethod
    def _sigma(pk, k, cosmo, m):
        from ggah_mod.cosmology.amplitude import sigma_tophat
        from ggah_mod.halos.variance import lagrangian_radius

        r = np.asarray(lagrangian_radius(m, cosmo.rho_cold))
        return np.asarray(sigma_tophat(np.asarray(pk.pk(k, 0.0, cosmo)), k, r))

    def test_the_variance_agrees_with_class_over_the_whole_band(self, klass):
        cp = make_pk("emu_pk")
        # emu_pk was trained on degenerate species and refuses a split; what
        # this class measures is its k-support, not an ordering.
        c = Cosmology.create(nu_hierarchy="degenerate")
        k = np.logspace(-4, np.log10(200.0), 1024)
        ref = self._sigma(klass, k, c, self._M)
        got = self._sigma(cp, k, c, self._M)
        assert np.max(np.abs(got / ref - 1.0)) < 0.01

    def test_the_spectrum_is_still_falling_at_k_max(self):
        """A clamp reads as slope 0; CDM is about -3 with a log correction."""
        cp = make_pk("emu_pk")
        # emu_pk was trained on degenerate species and refuses a split; what
        # this class measures is its k-support, not an ordering.
        c = Cosmology.create(nu_hierarchy="degenerate")
        k = np.array([50.0, 100.0, 200.0])
        p = np.asarray(cp.pk(k, 0.0, c))
        slope = np.log(p[-1] / p[-2]) / np.log(2.0)
        assert -3.5 < slope < -2.0, slope
        assert np.all(np.diff(p) < 0.0)


class TestTheTwoAccurateBackendsShareOneDomain:
    r"""The ceiling nobody wrote down.

    ``ACCURATE`` offers CLASS and CAMB as interchangeable solvers, which is
    only true if they accept the same cosmologies.  CLASS 3.3's default BBN
    table stops at :math:`\omega_b = 0.03289` and refuses everything above it;
    CAMB's own interpolator runs past 0.045.  That is 15 per cent of the CSST
    emulator's training box -- the whole high-:math:`\Omega_b`,
    high-:math:`H_0` corner -- available through one backend and not the other,
    with nothing in the package saying so.

    These tests pin the fix at both ends: the corner is reachable, and buying
    it did not move the spectrum anywhere it was already reachable.
    """

    #: Omega_b h^2 = 0.0382 -- the top corner of the CSST box, and above the
    #: default table's ceiling by a sixth.
    HIGH_OMEGA_B = dict(Omega_b=0.06, h=0.8)

    def test_the_high_baryon_corner_is_reachable(self, klass):
        c = Cosmology.create(**self.HIGH_OMEGA_B)
        assert float(c.Omega_b * c.h ** 2) > 0.0329, "not actually past it"
        pk = np.asarray(klass.pk(_K, 0.0, c))
        assert np.all(np.isfinite(pk)) and np.all(pk > 0.0)

    def test_camb_reaches_it_too_and_they_agree(self, klass, camb):
        c = Cosmology.create(**self.HIGH_OMEGA_B)
        a = np.asarray(klass.pk(_K, 0.0, c))
        b = np.asarray(camb.pk(_K, 0.0, c))
        r = (a / b) / np.median(a / b)
        assert np.max(np.abs(r - 1.0)) < 0.01, "shape"

    def test_the_table_swap_is_invisible_where_both_were_defined(self, klass):
        """4e-4 in Y_He, and this is what that is worth downstream.

        Reachability was bought with a different BBN table, so the price has to
        be stated: at the fiducial, where the old table was perfectly happy,
        the change must be far below every tolerance in this package.
        """
        from ggah_mod.cosmology.power import ClassPk

        c = Cosmology.create()
        a = np.asarray(klass.pk(_K, 0.0, c))
        try:
            ClassPk.SBBN_FILE = "/external/bbn/sBBN_2025.dat"
            b = np.asarray(make_pk("class", k_max=klass.k_max).pk(_K, 0.0, c))
        finally:
            ClassPk.SBBN_FILE = "/external/bbn/sBBN_2017.dat"
        assert np.max(np.abs(a / b - 1.0)) < 1e-4

    def test_a_missing_table_is_named_not_guessed(self, monkeypatch):
        """The failure mode this replaces.

        CLASS reports a missing table as a thermodynamics error deep in the
        solve, which reads like a physics problem.  Silently falling back to
        the default would be worse still: it would restore the ceiling and
        change Y_He, both invisibly.
        """
        from ggah_mod.cosmology.power import ClassPk

        monkeypatch.setattr(ClassPk, "SBBN_FILE", "/external/bbn/nope.dat")
        with pytest.raises(FileNotFoundError, match="nope.dat"):
            ClassPk._sbbn_path()


class TestCacheKeyCoversEveryParameter:
    """The predecessor keyed its P(k) cache on a hand-picked subset, so a chain
    varying anything else was served a spectrum from a different cosmology --
    and the symptom, a direction with no response, reads as a physics result.

    Every field must move the spectrum, or be a documented exception."""

    # `LEAF_FIELDS` rather than every dataclass field: `nu_hierarchy` is a
    # static string, so "perturb it by 2 per cent" has no meaning.  That it
    # reaches the cache key is asserted separately below, where the test can
    # move it between two *values* instead of along an axis.
    @pytest.mark.parametrize("field", [
        f for f in LEAF_FIELDS if f != "T_cmb"])
    def test_perturbing_any_parameter_changes_the_spectrum(self, klass, field):
        base = Cosmology.create()
        val = float(getattr(base, field))
        step = 0.02 * abs(val) if val != 0.0 else 0.02
        other = base.replace(**{field: val + step})
        a = np.asarray(klass.pk(_K, 0.0, base))
        b = np.asarray(klass.pk(_K, 0.0, other))
        assert np.max(np.abs(b / a - 1.0)) > 1e-6, (
            f"changing {field} left P(k) untouched -- the cache key is "
            f"missing it, or the backend ignores it")

    def test_the_mass_ordering_reaches_the_cache_key_too(self, klass):
        """The static field is in the key, not only the ten floats.

        It cannot be swept like the others -- "perturb the ordering by 2 per
        cent" means nothing -- so it is moved between two *values* instead.
        The failure this rules out is specific and silent: `_as_key` builds
        the key from `LEAF_FIELDS + STATIC_FIELDS`, and a key that had kept
        only the floats would serve a normal-ordering spectrum for an inverted
        cosmology, at 2.1e-4 in P(k), with nothing to see.

        At 0.15 eV, which is above both floors so all three orderings exist.
        """
        base = Cosmology.create(sum_mnu=0.15, nu_hierarchy="normal")
        for other in ("inverted", "degenerate"):
            a = np.asarray(klass.pk(_K, 0.0, base))
            b = np.asarray(klass.pk(_K, 0.0, base.replace(nu_hierarchy=other)))
            assert np.max(np.abs(b / a - 1.0)) > 1e-6, (
                f"normal and {other} returned the same spectrum -- the cache "
                f"key is missing nu_hierarchy, or the backend ignores it")

    def test_same_cosmology_is_bit_identical(self, klass):
        c = Cosmology.create()
        np.testing.assert_array_equal(np.asarray(klass.pk(_K, 0.0, c)),
                                      np.asarray(klass.pk(_K, 0.0, c)))


class TestTheSpectrumDoesNotDependOnTheRequest:
    r"""P(k, z) is a function of the cosmology and the redshift, and of nothing else.

    It was not.  ``ClassPk`` set ``z_max_pk = max(z.max(), 1.0)`` --- a *solver*
    setting, taken from the *request* --- and ``z_max_pk`` decides the time
    sampling of CLASS's output.  So the same redshift came back different
    depending on which other redshifts were asked for in the same call: 9.6e-8
    in :math:`\sigma_8`, and 1.3e-5 in ``dn/dM`` once the mass function's
    exponential had magnified it.  A batched result and a looped one disagreed,
    which is what surfaced it --- see
    :func:`~ggah_mod.halos.field.make_fields`.

    Parametrised over both Boltzmann backends rather than CLASS alone: CAMB
    takes its redshifts as an explicit output list and has no ``z_max_pk``, so
    it is *expected* to be request-independent already --- but that is a
    property to check, not to assume, and this is the check.
    """

    REQUESTS = [(0.5,), (0.5, 2.0), (0.0, 0.5, 1.0, 3.0), (0.5, 4.9)]

    @pytest.mark.parametrize("name", ["class", "camb"])
    def test_one_redshift_is_one_spectrum_however_it_was_asked_for(self, name):
        # `constructible` cannot cover this one: `CambPk.__init__` imports
        # nothing, so a backend whose solver is not installed constructs
        # perfectly and fails on first *use*.  The skip therefore has to sit
        # around the call rather than around the construction.
        from conftest import constructible
        pk = constructible(name)
        ref = None
        for req in self.REQUESTS:
            try:
                rows = np.asarray(pk.pk(_K, np.array(req), PLANCK18))
            except ImportError as exc:
                pytest.skip(f"{name}: {exc}")
            row = rows[req.index(0.5)]
            if ref is None:
                ref = row
                continue
            # CLASS is bit-for-bit; **CAMB is not, and never was**.  Asked
            # for `z = 0.5` alone and as part of `(0.5, 2.0)` it returns
            # spectra differing by ~2e-7 at every `omch2` tried, including the
            # one this package used before `Omega_cdm` was corrected -- where
            # it happened to land on 1.9e-8 and pass an equality test.  That
            # is CAMB's own output-list handling, which Sec. 2.4.2 names, and
            # a tolerance is the honest way to hold it: the claim is that the
            # spectrum does not *move*, not that a Boltzmann solver is
            # deterministic to the last bit across different requests.
            if name == "camb":
                np.testing.assert_allclose(
                    row, ref, rtol=1e-6, atol=0.0,
                    err_msg=f"{name}: z = 0.5 moved when asked for as "
                            f"part of {req}")
            else:
                np.testing.assert_array_equal(
                    row, ref, err_msg=f"{name}: z = 0.5 moved when asked for "
                                      f"as part of {req}")

    def test_the_ceiling_is_refused_rather_than_raised(self):
        """Extending ``z_max_pk`` on demand is the request-dependence this
        constant exists to remove, so above the ceiling is an error."""
        from conftest import constructible
        from ggah_mod.cosmology.power import Z_MAX_PK
        pk = constructible("class")
        with pytest.raises(ValueError, match="answers up to z"):
            pk.pk(_K, Z_MAX_PK + 1.0, PLANCK18)

    def test_the_ceiling_admits_the_projection_grids(self):
        """A ceiling below what the package's own defaults ask for would be a
        refusal waiting to happen: `limber_grid` defaults to z_max = 3.0."""
        import inspect
        from ggah_mod.cosmology.power import Z_MAX_PK
        from ggah_mod.observables.kernels import limber_grid
        default_z_max = inspect.signature(limber_grid).parameters["z_max"].default
        assert Z_MAX_PK >= default_z_max


class TestRedshift:
    def test_power_decreases_with_redshift(self, klass, cosmo):
        a = np.asarray(klass.pk(_K, 0.0, cosmo))
        b = np.asarray(klass.pk(_K, 1.0, cosmo))
        assert np.all(b < a)

    def test_vector_z_matches_scalar_z(self, klass, cosmo):
        """Exactly, not to a tolerance -- and it used to pass by luck.

        ``z_max_pk`` was ``max(z.max(), 1.0)``, so both requests here landed on
        the **floor** of 1.0 and got the same solver sampling.  That is why this
        test was green while `tests/test_growth.py` -- which straddles the floor
        with 0.5 against (0.5, 2.0) -- would not have been.  With the ceiling
        now a constant the agreement is structural rather than a coincidence of
        the two redshifts chosen here, so it is asserted as equality.
        """
        vec = np.asarray(klass.pk(_K, np.array([0.0, 1.0]), cosmo))
        assert vec.shape == (2, len(_K))
        np.testing.assert_array_equal(vec[1], np.asarray(klass.pk(_K, 1.0, cosmo)))


# The growth-normalisation class that stood here tested `_growth_today`, the
# RK4 solve that normalised `pk_eh98`.  Both are gone with the EH98 backend, and
# with them the second defect the cross-code benchmark found -- the
# Peebles-Heath integral fed a w != -1 background, which moves with w0 smoothly,
# plausibly, and with a slope twice the truth.  The finding is reported in
# Sec. 8 of the paper, where it belongs: it is a benchmark result about code
# this package no longer contains.
