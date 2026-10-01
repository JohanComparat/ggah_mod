r"""Verification: the cooling table, which had no test module at all.

``PLAN.md`` item **G2**.  ``sectors/cooling.py`` makes four claims -- that it is
:math:`C^1` in :math:`\log T`, that its interpolation is monotone and cannot
overshoot, that it refuses to degrade silently, and that its distillation
carries a factor of :math:`10^{14}` from APEC's normalisation -- and every one
was exercised only indirectly, through whichever gas test happened to call it.

Indirect coverage catches a table that fails to load.  It does not catch a
:math:`C^1` claim that is only :math:`C^0`, which is the defect this package
has already found twice on other axes (``nu_ratio`` in :math:`f_\nu`, then the
same on :math:`z`) and whose signature is a derivative wrong at a *grid node*
and right between them.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.sectors import cooling as CL


@pytest.fixture(scope="module")
def cool():
    return CL.ApecCooling()


class TestTheInterpolationIsC1:
    r"""The claim ``ApecCooling``'s docstring makes, on the axis it makes it.

    A kink is invisible in the value and obvious in the derivative -- and only
    *at* a node, which is why a random-point check would pass.  So the test
    walks the table's own nodes and compares autodiff against a central
    difference straddling each one, the way ``test_backends.py`` pins the
    neutrino table.
    """

    @staticmethod
    def _d_ln_lambda(cool, kt, z_metal=0.3):
        f = lambda t: jnp.log(cool(t, z_metal))
        return float(jax.grad(f)(kt))

    def test_the_derivative_agrees_at_every_node(self, cool):
        r"""Compared against a *scale*, not against the local value.

        A relative comparison of derivatives is meaningless wherever the
        derivative vanishes, and this one does: the band-integrated cooling
        function has a **local maximum near 0.7 keV**, where line emission
        stops gaining on the shrinking fraction of flux inside the 0.5--2 keV
        band.  At that node autodiff gives -5.2e-13 and a central difference
        gives 1.0e-4 -- both zero to any purpose, and a ratio of one.

        So the tolerance is absolute, set by the typical slope across the
        table.  Written down because the first version of this test reported a
        kink at exactly the one node where there is none, which is the same
        trap the parity budget hit on `xi` crossing zero.
        """
        nodes = np.power(10.0, np.asarray(cool._lt))
        # Interior nodes only: the ends clamp, which is a separate claim.
        slopes = [abs(self._d_ln_lambda(cool, float(k))) for k in nodes[2:-2]]
        scale = float(np.median(slopes))
        assert scale > 0.1, "the table is flat; this test would prove nothing"

        def worst_at(frac):
            out = 0.0
            for kt in nodes[2:-2]:
                kt = float(kt)
                h = frac * kt
                auto = self._d_ln_lambda(cool, kt)
                fd = (float(jnp.log(cool(kt + h, 0.3)))
                      - float(jnp.log(cool(kt - h, 0.3)))) / (2.0 * h)
                out = max(out, abs(auto - fd) / scale)
            return out

        coarse, fine = worst_at(1e-3), worst_at(2.5e-4)
        assert coarse < 5e-2 and fine < 2e-2

        # **Step-dependence is the test**, and the exponent identifies which
        # smoothness the table actually has.  Quartering the step:
        #
        #   error ~ h^0  -- the derivative itself is discontinuous: not C1,
        #                   which is the defect this package found twice on the
        #                   neutrino table;
        #   error ~ h^1  -- the *second* derivative jumps at the node: C1 and
        #                   not C2, which is precisely what a Fritsch--Carlson
        #                   monotone cubic is, and what is claimed;
        #   error ~ h^2  -- smooth through the node, which this is not and does
        #                   not claim to be.
        #
        # Measured: 3.4e-2 at h and 8.5e-3 at h/4, a ratio of 3.9.  That is
        # h^1, so the claim in `ApecCooling`'s docstring is exactly right --
        # C1 in log T, and no more than C1.
        ratio = coarse / max(fine, 1e-30)
        assert ratio > 2.5, (
            f"the disagreement is {fine:.2e} at h/4 against {coarse:.2e} at h, "
            f"a ratio of {ratio:.1f}. Flat in the step means the derivative "
            f"itself jumps: the interpolation is not C1")
        assert ratio < 8.0, (
            f"ratio {ratio:.1f} looks like h^2, i.e. smooth through the node. "
            f"A monotone cubic is C1 and not C2, so either the table changed "
            f"or this is no longer Fritsch--Carlson")

    def test_the_cooling_function_really_does_turn_over(self, cool):
        """The reason the test above cannot use a relative tolerance, measured
        rather than assumed: the log-slope changes sign inside the table."""
        nodes = np.power(10.0, np.asarray(cool._lt))[2:-2]
        slopes = np.asarray([self._d_ln_lambda(cool, float(k)) for k in nodes])
        assert slopes.min() < 0.0 < slopes.max()
        turn = float(nodes[int(np.argmin(np.abs(slopes)))])
        assert 0.3 < turn < 2.0, turn

    def test_a_c0_interpolation_would_fail_this(self, cool):
        """The control: linear interpolation of the same table is C0, and the
        check above catches it.  Without this, a passing test proves only that
        the test is weak."""
        lt, loglam = np.asarray(cool._lt), np.asarray(cool._loglam)
        j = loglam.shape[1] // 2

        def linear(kt):
            return jnp.interp(jnp.log10(kt), lt, loglam[:, j])

        node = float(10.0 ** lt[len(lt) // 2])
        h = 1e-4 * node
        auto = float(jax.grad(linear)(node))
        fd = (float(linear(node + h)) - float(linear(node - h))) / (2.0 * h)
        assert abs(auto - fd) / abs(fd) > 1e-3, (
            "the C0 control did not show a kink, so the C1 test above is not "
            "sensitive to one")


class TestItIsMonotoneAndDoesNotOvershoot:
    """Fritsch--Carlson, and the property it is chosen for.

    An ordinary cubic through a table that falls steeply can overshoot between
    nodes, and a negative cooling function is not a small error: it is an
    emissivity with the wrong sign, and it would appear only between the points
    anyone plotted.
    """

    def test_lambda_is_positive_everywhere_between_the_nodes(self, cool):
        lt = np.asarray(cool._lt)
        kt = np.power(10.0, np.linspace(lt[0], lt[-1], 4001))
        lam = np.asarray(cool(jnp.asarray(kt), 0.3))
        assert np.all(np.isfinite(lam)) and np.all(lam > 0.0)

    def test_it_stays_inside_the_node_values_it_interpolates(self, cool):
        """No overshoot: on each interval the curve is bounded by its ends.
        That is what monotone buys and what a plain cubic does not give."""
        lt = np.asarray(cool._lt)
        j = np.asarray(cool._lz)
        z_mid = float(10.0 ** j[len(j) // 2])
        worst = 0.0
        for i in range(len(lt) - 1):
            a, b = float(10.0 ** lt[i]), float(10.0 ** lt[i + 1])
            ends = np.asarray(cool(jnp.asarray([a, b]), z_mid))
            inner = np.asarray(cool(jnp.asarray(
                np.geomspace(a, b, 32)[1:-1]), z_mid))
            lo, hi = ends.min(), ends.max()
            over = max(float(inner.max() - hi), float(lo - inner.min()), 0.0)
            worst = max(worst, over / max(hi, 1e-40))
        assert worst < 1e-9, f"overshoot of {worst:.2e} of the node value"


class TestItRefusesRatherThanDegrades:
    """The third claim: it does not quietly return something plausible."""

    def test_an_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown cooling"):
            CL.make_cooling("nope")

    def test_the_query_is_clamped_and_not_the_fraction(self, cool):
        """``lin_weights`` carries the reason: clamping the *fraction* puts a
        tie at the edge and splits the gradient, where clamping the query
        leaves a finite one-sided derivative."""
        lo = float(10.0 ** np.asarray(cool._lt)[0])
        below = float(cool(lo * 0.1, 0.3))
        at = float(cool(lo, 0.3))
        assert below == pytest.approx(at, rel=1e-10)
        g = float(jax.grad(lambda t: cool(t, 0.3))(lo * 0.1))
        assert np.isfinite(g)

    def test_the_band_is_carried_and_is_a_single_one(self, cool):
        """``emin``/``emax`` are scalars in the shipped table, so the sector has
        exactly one band and cannot answer for another.  Recorded because a
        band-integrated cooling function silently used outside its band is the
        error the AGN sector's `k_h2s` exists to make explicit."""
        assert len(cool.band) == 2 and cool.band[0] < cool.band[1]
        assert cool.band == (0.5, 2.0)


class TestTheDistilledTable:
    """The factor of 1e14, and the shape the sector relies on."""

    def test_the_table_has_the_shape_the_interpolator_assumes(self, cool):
        assert cool._loglam.shape == (cool._lt.size, cool._lz.size)
        assert np.all(np.diff(np.asarray(cool._lt)) > 0)
        assert np.all(np.diff(np.asarray(cool._lz)) > 0)

    def test_lambda_is_of_the_order_a_cooling_function_is(self, cool):
        r"""A factor of :math:`10^{14}` is APEC's normalisation convention, and
        getting it wrong moves :math:`\Lambda` by fourteen orders -- which no
        plot hides but which nothing here was checking.  Band-integrated
        :math:`\Lambda` for a 1 keV, third-solar plasma sits near
        :math:`10^{-23}` erg cm^3/s."""
        lam = float(cool(1.0, 0.3))
        assert 1e-25 < lam < 1e-21, lam

    def test_the_power_law_stand_in_is_a_different_quantity(self, cool):
        r"""And the ratio says so, which is worth recording rather than
        bounding loosely.

        ``lambda_powerlaw`` is bremsstrahlung-like and effectively bolometric;
        ``ApecCooling`` is integrated over 0.5--2 keV.  So the ratio is not a
        constant "factor of a few" -- it falls monotonically as the plasma gets
        hotter and more of its emission leaves the band:

        =========  ==========
        kT [keV]   apec/power
        =========  ==========
        0.2        0.292
        1.0        0.256
        3.0        0.095
        15.0       0.036
        =========  ==========

        A test asserting they agree to a factor of a few would pass at 1 keV
        and fail at 8, for a reason that is physics rather than error.  What is
        checkable is the trend and the band's fingerprint on it.
        """
        kt = np.asarray([0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0])
        ratio = np.asarray([float(cool(float(k), 0.3))
                            / float(CL.lambda_powerlaw(float(k), 0.3))
                            for k in kt])
        assert np.all(ratio > 0.0)
        # Monotone decline above the line-cooling peak: the band loses flux.
        hot = ratio[kt >= 1.0]
        assert np.all(np.diff(hot) < 0.0)
        assert ratio[kt == 1.0][0] == pytest.approx(0.256, abs=0.02)
        assert ratio[-1] == pytest.approx(0.036, abs=0.005)


class TestItDifferentiatesInBoth:
    """Both arguments, because a fit varies the metallicity too."""

    @pytest.mark.parametrize("kt", [0.3, 1.0, 5.0])
    def test_the_temperature_gradient_is_finite_and_negative_ish(self, cool, kt):
        g = float(jax.grad(lambda t: jnp.log(cool(t, 0.3)))(kt))
        assert np.isfinite(g)

    def test_the_metallicity_gradient_is_finite_and_positive(self, cool):
        """More metals, more line cooling."""
        g = float(jax.grad(lambda zz: jnp.log(cool(1.0, zz)))(0.3))
        assert np.isfinite(g) and g > 0.0


# ==========================================================================
# The band-resolved route
# ==========================================================================

@pytest.fixture(scope="module")
def band():
    """The band table, or a skip saying how to build it.

    Not shipped -- 23 MB against 9 kB for the fixed-band table beside it -- so
    its absence is a legitimate state of a fresh checkout and not a failing
    test.  The same shape as ``conftest.constructible``: skip on the artefact
    being absent, and hold everything that follows to a real standard.
    """
    try:
        return CL.BandCooling(band=(0.5, 2.0))
    except FileNotFoundError as exc:
        pytest.skip(str(exc).splitlines()[0])


class TestTheTwoRoutesDescribeOnePlasma:
    r"""``ApecCooling`` and ``BandCooling`` over the same band.

    The only check that says the band-resolved table is the same physics rather
    than a second, plausible one.  They are built from the same APEC version on
    the same :math:`(T, Z)` grid, so agreement is not a coincidence -- but they
    reach the band differently, one at build time and one at read time, and it
    is that difference the numbers below measure.

    **The bound is looser off-node than on it, and that ordering is the point.**
    On a node the band table has interpolated only in energy; between nodes it
    has also interpolated in :math:`T` and :math:`Z`, where the fixed table
    interpolates a quantity that was integrated first.  Both bounds are what was
    measured, set about threefold above it.
    """

    #: Measured 5.7e-4 on-node, 1.9e-3 off-node.
    ON_NODE = 2e-3
    OFF_NODE = 6e-3

    @pytest.mark.x64
    def test_they_agree_on_the_shared_grid_nodes(self, cool, band):
        t = CL.load()
        kt, zm = 10.0 ** t["log10_kt"], 10.0 ** t["log10_z"]
        worst = max(
            abs(float(band(kt[i], zm[j])) / float(cool(kt[i], zm[j])) - 1.0)
            for i in range(0, len(kt), 3) for j in range(0, len(zm), 3))
        assert worst < self.ON_NODE

    @pytest.mark.x64
    def test_they_agree_between_the_nodes_too(self, cool, band):
        r"""Where the first attempt was **4.5 per cent** out.

        Recorded because the cause is not the one anybody would guess and the
        module's own comment now carries it: reading the cumulative at each
        band edge and interpolating *each* in :math:`(T, Z)` before subtracting
        amplifies a per-mille interpolation error into a seven per cent one,
        because at 0.1 keV the 0.5--2 keV band is 1.4 per cent of the tabulated
        range and the two numbers being subtracted are both near unity.
        Forming the band at each node *first* removes it.

        The temperatures below deliberately sit between grid nodes, including
        in the first cell, which is where the steepest part of
        :math:`\Lambda(T)` is and where the defect was largest.
        """
        kt = np.logspace(np.log10(0.085), np.log10(28.0), 61)
        worst = max(abs(float(band(t, z)) / float(cool(t, z)) - 1.0)
                    for t in kt for z in (0.05, 0.3, 1.0))
        assert worst < self.OFF_NODE


class TestTheKCorrectionIsTheLimits:
    r"""An observed band at :math:`z` is the rest-frame band at
    :math:`E(1+z)`.

    The whole reason to tabulate in energy: there is no K-correction to apply,
    only limits to move, so it cannot be applied twice or forgotten.
    """

    KT, ZM = 3.0, 0.3

    @pytest.mark.x64
    @pytest.mark.parametrize("z", [0.0, 0.3, 1.0, 2.0, 4.0])
    def test_an_observed_band_is_a_shifted_rest_frame_band(self, band, z):
        """True by construction, asserted anyway -- it is the definition the
        rest of the class is measured against, and a sign slip in ``1 + z``
        would satisfy every other test here."""
        obs = float(band(self.KT, self.ZM, band=(0.5, 2.0), z_obs=z))
        rest = float(band(self.KT, self.ZM,
                          band=(0.5 * (1 + z), 2.0 * (1 + z)), z_obs=0.0))
        assert obs == pytest.approx(rest, rel=1e-12)

    @pytest.mark.slow
    @pytest.mark.x64
    @pytest.mark.parametrize("z", [0.0, 1.0, 4.0])
    def test_against_a_direct_soxs_calculation(self, band, z):
        r"""The real check, and it needs the right reference.

        Comparing against ``get_spectrum(..., redshift=z)`` does **not** work
        and the reason is instructive: soxs holds ``norm`` fixed while
        redshifting, and ``norm`` carries a :math:`(1+z)^{-2}`, so the ratio
        comes out near :math:`(1+z)^2` -- with a further few per cent that
        grows with :math:`z` and belongs to *soxs* rebinning a shifted spectrum
        onto a fixed output grid.  Against the **rest-frame** band at
        ``redshift=0`` instead, the comparison is clean: measured 6.0e-4 flat
        from z = 0 to 4, with no drift, which is the table's own accuracy and
        says the redshift handling contributes nothing of its own.
        """
        soxs = pytest.importorskip("soxs")
        lo, hi = 0.5 * (1 + z), 2.0 * (1 + z)
        try:
            direct = float(soxs.ApecGenerator(lo, hi, 4000, broadening=False)
                           .get_spectrum(self.KT, self.ZM, redshift=0.0,
                                         norm=1.0).total_energy_flux.value) / 1e14
        except (OSError, IOError) as exc:            # atomic data absent
            pytest.skip(f"soxs atomic data: {exc}")
        mine = float(band(self.KT, self.ZM, band=(0.5, 2.0),
                          z_obs=z)) / CL.NH_OVER_NE
        assert mine == pytest.approx(direct, rel=3e-3)


class TestTheBandEdgesCarryAGradient:
    """The capability a build-time band cannot have.

    ``ApecCooling`` has a structural zero here -- ``emin``/``emax`` are
    properties of a file -- so a fit marginalising over a band-edge calibration
    would read a flat direction rather than an error.
    """

    KT, ZM = 3.0, 0.3

    @pytest.mark.x64
    def test_gradient_in_the_upper_edge(self, band):
        f = lambda e: jnp.sum(band(self.KT, self.ZM, band=(0.5, e)))
        ad = float(jax.grad(f)(2.0))
        h = 1e-4
        fd = float((f(2.0 + h) - f(2.0 - h)) / (2 * h))
        assert np.isfinite(ad) and ad > 0.0
        assert ad == pytest.approx(fd, rel=1e-3)

    @pytest.mark.x64
    def test_gradient_in_the_source_redshift(self, band):
        f = lambda z: jnp.sum(band(self.KT, self.ZM, z_obs=z))
        ad = float(jax.grad(f)(0.5))
        h = 1e-4
        fd = float((f(0.5 + h) - f(0.5 - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-3)

    @pytest.mark.x64
    def test_it_still_differentiates_in_temperature(self, band):
        """The band route must not lose what the fixed one had."""
        f = lambda t: jnp.sum(band(t, self.ZM))
        ad = float(jax.grad(f)(3.0))
        h = 1e-5
        assert ad == pytest.approx(float((f(3.0 + h) - f(3.0 - h)) / (2 * h)),
                                   rel=1e-3)


class TestTheBandBehavesLikeABand:
    KT, ZM = 3.0, 0.3

    @pytest.mark.x64
    def test_a_wider_band_holds_more(self, band):
        narrow = float(band(self.KT, self.ZM, band=(0.5, 2.0)))
        wide = float(band(self.KT, self.ZM, band=(0.5, 7.0)))
        assert wide > narrow > 0.0

    @pytest.mark.x64
    def test_adjacent_bands_add(self, band):
        """`Lambda` is an integral, so it is additive in the band -- which a
        cumulative table gets right identically and a per-band table could
        not."""
        a = float(band(self.KT, self.ZM, band=(0.5, 2.0)))
        b = float(band(self.KT, self.ZM, band=(2.0, 7.0)))
        both = float(band(self.KT, self.ZM, band=(0.5, 7.0)))
        assert a + b == pytest.approx(both, rel=1e-10)

    @pytest.mark.x64
    def test_a_zero_width_band_is_zero_to_the_floor(self, band):
        r"""8e-301, not 0.0, and the difference is deliberate.

        :meth:`BandCooling._band_nodes` floors the band at 1e-300 before taking
        its log, because an exact zero puts :math:`-\infty` into the Hermite --
        and a ``nan`` or an infinity in a cell the query does not use still
        poisons the gradient of one it does.  So an empty band returns the
        floor rather than zero, which is numerically zero for every purpose and
        finite for the one that matters.

        Asserted against the floor rather than against a tolerance, so the day
        somebody changes the floor this says so instead of passing quietly.
        """
        got = float(band(self.KT, self.ZM, band=(2.0, 2.0)))
        assert 0.0 < got < 1e-295
        # And it does not poison the gradient, which is the reason for it.
        g = float(jax.grad(lambda e: jnp.sum(band(self.KT, self.ZM,
                                                  band=(2.0, e))))(2.0))
        assert np.isfinite(g)

    def test_it_says_whether_it_covers_a_band(self, band):
        """Concrete, so a caller can ask before a fit rather than inside one."""
        assert band.covers((0.5, 2.0), z_obs=0.0)
        assert band.covers((0.5, 2.0), z_obs=4.0)          # 2.5-10 keV emitted
        assert not band.covers((0.5, 2.0), z_obs=30.0)     # past the ceiling
        assert not band.covers((0.01, 2.0))                # below the floor

    @pytest.mark.x64
    def test_it_is_in_the_registry_and_not_substituted(self):
        assert CL.COOLING["apec_band"] is CL.BandCooling
        with pytest.raises(ValueError, match="unknown cooling function"):
            CL.make_cooling("apec_bands")


class TestTheBandTableRefusesRatherThanDegrades:
    def test_a_missing_cache_names_the_command_that_builds_it(self, tmp_path):
        """The same refusal :func:`load` makes, and for the same reason: a
        cooling function that quietly downgrades is how an atomic-physics error
        ends up absorbed into a fitted X-ray amplitude."""
        with pytest.raises(FileNotFoundError, match="--band"):
            CL.load_band(tmp_path / "absent.npz")

    def test_the_cache_path_is_overridable(self, monkeypatch, tmp_path):
        """So a build on a shared machine does not write to a home directory,
        and so this suite can point at a scratch copy."""
        monkeypatch.setenv("GGAH_APEC_CACHE", str(tmp_path))
        assert CL.band_table_path() == tmp_path / "apec_band.npz"


class TestTheWideTable:
    r"""``apec_wide``: the shipped nodes, plus 22 below 0.08 keV.

    The shipped table clamps its query at 0.08 keV.  In the rest-frame 0.5--2
    keV band :math:`\Lambda` falls by orders of magnitude below that, so the
    clamp made cool gas as bright as 0.08 keV gas -- a flat direction for a fit
    that lowers :math:`kT = P_e/n_e`.  The wide table must be the shipped one
    wherever both interpolations see the same nodes, and APEC below.
    """

    @pytest.fixture(scope="class")
    def wide(self):
        if not CL.WIDE_TABLE.exists():
            pytest.skip("wide table not built: python -m ggah_mod.sectors.cooling --wide")
        return CL.ApecCoolingWide()

    def test_its_upper_nodes_are_the_shipped_table(self, wide):
        a, w = CL.load(), CL.load(CL.WIDE_TABLE)
        n = a["log10_kt"].size
        assert w["log10_kt"].size == n + CL.WIDE_EXTRA
        np.testing.assert_allclose(w["log10_kt"][-n:], a["log10_kt"], atol=1e-14)
        np.testing.assert_allclose(w["log10_lambda"][-n:], a["log10_lambda"],
                                   atol=1e-13)

    def test_it_is_apec_above_the_second_shipped_node(self, cool, wide):
        """The monotone-cubic slope at a node reads its two neighbours, so
        only the first shipped interval can differ."""
        kt = jnp.logspace(np.log10(0.0879), np.log10(30.0), 200)
        for z in (0.1, 0.3, 1.0):
            np.testing.assert_allclose(np.asarray(wide(kt, z)),
                                       np.asarray(cool(kt, z)), rtol=1e-12)

    def test_below_the_old_edge_it_follows_apec_down(self, cool, wide):
        """Measured: 0.12, 2.1e-3 and 5.7e-9 of the clamped value at 0.06,
        0.04 and 0.02 keV for a third-solar plasma."""
        r = [float(wide(t, 0.3)) / float(cool(t, 0.3)) for t in (0.06, 0.04, 0.02)]
        assert 0.05 < r[0] < 0.3
        assert 5e-4 < r[1] < 1e-2
        assert r[2] < 1e-7
        kt = jnp.logspace(np.log10(0.0101), np.log10(0.08), 50)
        # non-decreasing: the coolest nodes sit on the table's 1e-40 floor
        assert np.all(np.diff(np.asarray(wide(kt, 0.3))) >= 0.0)

    def test_it_is_selected_by_name(self, wide):
        assert CL.make_cooling("apec_wide").name == "apec_wide"
