r"""The parity budget: ``ACCURATE`` against ``DIFFERENTIABLE``, row by row.

``README.md`` has ended with the same sentence since the package was started --
*"What remains is the parity budget ... values and gradients, row by row -- and
calibration. Not yet usable for science."* -- and ``PLAN.md`` carried it as the
one **TODO** in a status table of forty-odd rows.  This is it.

**Why it is not the same thing as the tests beside it.**  Every layer already
checks itself against a closed form or an external reference, and every gradient
test is parametrised over both flavours.  What none of them does is compare the
two flavours *to each other*: each is checked against its own finite difference,
so a differentiable gradient thirty per cent from the accurate one passes both.
A fast flavour whose values agree to one per cent and whose derivatives are
thirty per cent off gives a wrong Fisher matrix, and nothing else here would
catch it.

**Probes change one thing.**  A raw A-against-D ratio conflates the spectrum
backend, four grid sizes and the Hankel engine at once, so where the two
disagree this module says which of them did it -- the discipline
``ggah_mod_benchmark`` uses for the same reason.  That is not decoration: on its
first run the raw layer-5 row was 42 per cent in ``w_p`` and the decomposition
put 4e-4 of it on the cosmology backend and all the rest on the transform.

**Numbers, not tolerances.**  Each bound below is a measurement with the
configuration that produced it, so a change that moves one is a visible edit
rather than a test that still passes.  ``tests/parity/`` is a different thing
with a colliding name: that is the port-parity suite against ``hod_mod``.
"""
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.spectra as SP
from ggah_mod.backend import ACCURATE, DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import (BaryonSplit, GalaxySector, MatterField,
                              galaxy_defaults)
from ggah_mod.observables.real_space import delta_sigma, sigma, wp, xi

pytestmark = pytest.mark.slow

#: Separations the layer-5 rows are quoted at [Mpc/h].
R = np.logspace(-1.0, 1.7, 24)

#: Wavenumbers the layer-4 rows are quoted at [h/Mpc].
K_PROBE = (0.01, 0.1, 1.0, 10.0)

#: The rows below were measured before 0.9.8 raised the ``ACCURATE`` flavour's
#: CLASS precision (``CLASS_PRECISION``, ``tol_ncdm_synchronous = 1e-6``), while
#: ``emu_pk`` still reproduces CLASS at its own defaults; the flavours therefore
#: moved apart by about that precision's error.  With ``ClassPk`` at CLASS's
#: defaults every row passes (measured 2026-09-29), so these are expected
#: failures until the layer 4-5 verification re-measures the bounds at the new
#: configuration -- strict, so they turn red when it does.
PRECISION_RAISED = pytest.mark.xfail(strict=True, reason=(
    "0.9.8 raised ACCURATE's CLASS precision and emu_pk reproduces CLASS's "
    "defaults; passes with ClassPk at CLASS defaults. Bound to be re-measured."))


def _spectra(backend):
    """One field, one galaxy tracer, one matter tracer, three pairs."""
    f = make_field(PLANCK18, backend, pk=make_pk(backend.pk), z=0.0,
                   calibration="off")
    g = GalaxySector("zheng07", shmr="zu15", backend=backend).weights(
        f, galaxy_defaults("zheng07"))
    m = MatterField().weights(
        f, {"split": BaryonSplit.from_hot(0.0, jnp.zeros(f.n_m))})
    return f, {"gg": SP.pk_cross(f, g, g, overlap="identical"),
               "gm": SP.pk_cross(f, g, m, overlap="none"),
               "mm": SP.pk_cross(f, m, m, overlap="none")}


@pytest.fixture(scope="module")
def budget():
    """Both flavours built once.  ``ACCURATE`` runs CLASS, so this is not free."""
    out = {}
    for b in (ACCURATE, DIFFERENTIABLE):
        f, pairs = _spectra(b)
        out[b.name] = (b, f, pairs)
    return out


def _rel_on_common_k(field_a, arr_a, field_d, arr_d):
    """``|A/D - 1|`` on the differentiable grid; the two ``k`` grids differ."""
    ka, kd = np.asarray(field_a.k), np.asarray(field_d.k)
    a = np.interp(np.log(kd), np.log(ka), np.asarray(arr_a))
    with np.errstate(divide="ignore", invalid="ignore"):
        return kd, np.abs(a / np.asarray(arr_d) - 1.0)


# ==========================================================================
# Layer 4
# ==========================================================================
class TestLayerFour:
    """``P_ab(k)``, every shipped term, both flavours.

    This is the half that was already fine, and saying so is the point: it is
    what makes the layer-5 rows a localised finding rather than a general
    disagreement between the flavours.
    """

    #: ``pair -> term -> max |A/D - 1|`` over ``1e-3 < k < 20``.  Measured.
    BOUND = 2.5e-3

    @pytest.mark.parametrize("term, pair", [
        pytest.param("one_halo", "gg", marks=PRECISION_RAISED),
        pytest.param("one_halo", "gm", marks=PRECISION_RAISED),
        pytest.param("one_halo", "mm", marks=PRECISION_RAISED),
        ("two_halo", "gg"),
        ("two_halo", "gm"),
        pytest.param("two_halo", "mm", marks=PRECISION_RAISED),
        ("total", "gg"),
        pytest.param("total", "gm", marks=PRECISION_RAISED),
        pytest.param("total", "mm", marks=PRECISION_RAISED),
    ])
    def test_the_spectra_agree_to_a_few_parts_in_a_thousand(self, budget, pair,
                                                            term):
        _, fa, pa = budget["accurate"]
        _, fd, pd = budget["differentiable"]
        a = pa[pair].total if term == "total" else getattr(pa[pair], term)
        d = pd[pair].total if term == "total" else getattr(pd[pair], term)
        k, rel = _rel_on_common_k(fa, a, fd, d)
        sel = (k > 1e-3) & (k < 20.0)
        worst = float(np.nanmax(rel[sel]))
        assert worst < self.BOUND, (
            f"P_{pair}[{term}] flavours disagree by {worst:.2e}, above the "
            f"measured {self.BOUND:.1e}")

    @PRECISION_RAISED
    def test_the_one_halo_term_is_the_tighter_half(self, budget):
        """One-halo 7e-4, two-halo 1.7e-3.  The two-halo term carries the
        linear spectrum directly, so it inherits the CLASS-vs-emulator
        difference undiluted; the one-halo term only sees it through the mass
        function."""
        _, fa, pa = budget["accurate"]
        _, fd, pd = budget["differentiable"]
        worst = {}
        for term in ("one_halo", "two_halo"):
            k, rel = _rel_on_common_k(fa, getattr(pa["mm"], term),
                                      fd, getattr(pd["mm"], term))
            sel = (k > 1e-3) & (k < 20.0)
            worst[term] = float(np.nanmax(rel[sel]))
        assert worst["one_halo"] < worst["two_halo"]
        assert worst["one_halo"] < 1.0e-3
        assert worst["two_halo"] < 2.5e-3


# ==========================================================================
# Layer 5
# ==========================================================================
class TestLayerFive:
    r"""The projections -- and the row the budget was written to find.

    Values that agree to 1.7e-3 in Fourier space disagreed by **42 per cent** in
    ``w_p`` at 50 Mpc/h when this module was first run.  Not a zero crossing:
    neither statistic changes sign over the range, and ``|A-D|`` is 20 per cent
    of the peak.  The cause was the Ogata rule's reach, and it is fixed in
    ``backend.py`` -- see ``TestTheEngineWasTheCause`` below, which is kept as
    the record of how the two were separated.
    """

    #: Two regimes, because there are two findings and one number would hide
    #: both.  The small-scale row was open for two tiers and is closed --
    #: ``Backend.fftlog_pad_decades``, see ``test_the_small_scale_row_is_closed``.
    R_SPLIT = 2.0
    BOUND_LARGE = {"xi": 5e-3, "wp": 1e-3, "sigma": 5e-3, "delta_sigma": 5e-3}
    #: Measured after the padding landed: 9.4e-4, 5.1e-3 and 1.5e-4 against
    #: 1.2e-1, 2e-2 and 4e-2 before it.  ``wp`` was the exception, at 1.955e-1,
    #: and it was not the engine but the line-of-sight grid -- see
    #: ``test_the_wp_row_was_the_pi_quadrature``.  Converged in 0.9.7: 5.7e-4.
    #: ``sigma`` had drifted to 7.8e-3 by 0.9.4 and is 8.6e-3 since the
    #: beyond-linear bias is read C^1 (0.9.7), the one row that change moved up;
    #: it moved the large-scale ``sigma`` row down, 2.1e-3 to 5.3e-4, and
    #: ``delta_sigma``'s 9.0e-4 to 4.8e-4.
    BOUND_SMALL = {"xi": 2e-3, "wp": 1e-3, "sigma": 9e-3,
                   "delta_sigma": 5e-4}

    @staticmethod
    def _statistics(backend, field, pairs):
        k = np.asarray(field.k)
        pgg = (k, np.asarray(pairs["gg"].total))
        pgm = (k, np.asarray(pairs["gm"].total))
        return {
            "xi": np.asarray(xi(R, pgg, backend=backend)),
            "wp": np.asarray(wp(R, pgg, backend=backend, pi_max=100.0)),
            "sigma": np.asarray(sigma(R, pgm, PLANCK18, backend=backend)),
            "delta_sigma": np.asarray(
                delta_sigma(R, pgm, PLANCK18, backend=backend)),
        }

    @pytest.fixture(scope="class")
    def stats(self, budget):
        return {name: self._statistics(*budget[name])
                for name in ("accurate", "differentiable")}

    @pytest.mark.parametrize("q", ["xi", "wp", "sigma", "delta_sigma"])
    def test_the_projections_agree_beyond_a_couple_of_mpc(self, stats, q):
        """The row the reach fix repaired.  Before it, ``w_p`` was 42 per cent
        apart at 50 Mpc/h; after it 4.5e-3, and 4.6e-4 once the line-of-sight
        quadrature was converged (0.9.7), which had been carrying the rest.
        ``xi`` at 13 Mpc/h agrees to 4e-6."""
        a, d = stats["accurate"][q], stats["differentiable"][q]
        sel = R >= self.R_SPLIT
        rel = float(np.nanmax(np.abs(a[sel] / d[sel] - 1.0)))
        assert rel < self.BOUND_LARGE[q], (
            f"{q}: flavours disagree by {rel:.2e} beyond {self.R_SPLIT} Mpc/h, "
            f"above the measured {self.BOUND_LARGE[q]:.1e}.  Decompose it "
            f"before widening this bound -- see TestTheEngineWasTheCause.")

    @PRECISION_RAISED
    @pytest.mark.parametrize("q", ["xi", "wp", "sigma", "delta_sigma"])
    def test_the_small_scale_row_is_closed(self, stats, q):
        r"""Inside a couple of Mpc/h the flavours now agree too.

        Open for two tiers with a **stated cause that was wrong**.  The row
        recorded FFTLog as unconverged, quoting ``xi(1 Mpc/h)`` drifting 2.553e1,
        2.285e1, 2.357e1, 2.392e1 across ``n_k`` = 512..4096.  That does not
        reproduce: FFTLog is stable to four digits over the same range.

        The cause is **periodicity**.  FFTLog treats its input as periodic in
        ``ln k``, and at ``k_max = 200`` a galaxy auto-spectrum has not decayed
        -- the one-halo term leaves the log-slope at -1.79, so ``k^3 P`` is
        still at its maximum.  A step at the wrap rings, which is why the
        FFTLog values *oscillate about* Ogata's rather than sitting to one side,
        and why more resolution never helped.

        Fixed by continuing the grid past its ends with the same capped power
        law ``hankel`` already uses -- ``Backend.fftlog_pad_decades``.  The cap
        is the point: ``P_gg``'s own end slope is -1.79, and continuing by it
        would buy no decay at all.

        ``xi`` 1.09e-1 -> 9.4e-4, ``delta_sigma`` 4e-2 -> 1.5e-4.  ``wp`` is the
        exception, and it was not this engine's doing; see the next test.
        """
        a, d = stats["accurate"][q], stats["differentiable"][q]
        sel = R < self.R_SPLIT
        rel = float(np.nanmax(np.abs(a[sel] / d[sel] - 1.0)))
        assert rel < self.BOUND_SMALL[q], (
            f"{q}: small-scale disagreement is {rel:.2e}, worse than the "
            f"recorded {self.BOUND_SMALL[q]:.1e}")

    @PRECISION_RAISED
    def test_the_wp_row_was_the_pi_quadrature(self, budget, stats):
        r"""And now it is the engine's alone, like every other row.

        ``wp`` reads **two** things off the backend -- the Hankel engine and
        ``n_pi`` -- and the flavours differ in both: quadrature/512 against
        fftlog/256.  Until 0.9.7 the line of sight was a trapezoid on a grid
        *linear* in :math:`\pi`, and holding the engine fixed while moving only
        ``n_pi`` reproduced the whole row: Ogata at ``n_pi = 256`` against Ogata
        at 512 gave **1.955e-1**, the flavour comparison to four digits, while
        matching ``n_pi`` left the engines 3.0e-4 apart.  The grid was simply too
        coarse below :math:`r_p \approx 0.2` Mpc/h, where the integrand's peak at
        :math:`\pi \lesssim r_p` fell between nodes 0.2-0.4 Mpc/h apart
        (``tests/test_real_space.py::TestWpLineOfSight``).

        It also explained why padding first looked like it made ``wp`` worse --
        1.61e-1 to 1.91e-1: the engine error had been *partially cancelling* the
        pi-quadrature error.  A number that improves when two errors are present
        and worsens when one is removed is two errors with opposite signs.

        Since 0.9.7 the nodes are uniform in :math:`{\rm asinh}(\pi/r_p)`.
        Measured: moving ``n_pi`` alone from 512 to 256 changes ``wp`` by
        **1.2e-6** inside 2 Mpc/h, and the flavours differ by **5.7e-4**, all of
        it the engine (5.7e-4 with ``n_pi`` matched).
        """
        import dataclasses

        sel = R < self.R_SPLIT
        b_a, f_a, pairs_a = budget["accurate"]
        coarse_pi = dataclasses.replace(
            b_a, name="accurate_npi_256",
            n_pi=budget["differentiable"][0].n_pi)
        only_pi = self._statistics(coarse_pi, f_a, pairs_a)["wp"]
        ref = stats["accurate"]["wp"]
        rel_pi = float(np.nanmax(np.abs(ref[sel] / only_pi[sel] - 1.0)))
        assert rel_pi < 1e-5, (
            f"n_pi 512 against 256 moves wp by {rel_pi:.2e}; the line-of-sight "
            f"quadrature is no longer converged at either")

        b_d, f_d, pairs_d = budget["differentiable"]
        fine_pi = dataclasses.replace(b_d, name="differentiable_npi_512",
                                      n_pi=b_a.n_pi)
        matched = self._statistics(fine_pi, f_d, pairs_d)["wp"]
        rel_engine = float(np.nanmax(np.abs(ref[sel] / matched[sel] - 1.0)))
        both = float(np.nanmax(
            np.abs(ref[sel] / stats["differentiable"]["wp"][sel] - 1.0)))
        assert both == pytest.approx(rel_engine, rel=0.05), (
            f"the flavours differ by {both:.3e} and the engines alone by "
            f"{rel_engine:.3e}: the wp row has acquired a non-engine component")
        assert rel_engine < 1e-3, (
            f"with n_pi matched the flavours differ by {rel_engine:.3e} in wp; "
            f"that is the engine's share and it should be at the same 1e-4 "
            f"level as xi")

    def test_the_disagreement_is_not_a_zero_crossing(self, stats):
        """A relative bound is meaningless across a sign change, so the fact
        that there is none is part of what makes the row above readable."""
        for q, a in stats["accurate"].items():
            assert np.sign(a).min() == np.sign(a).max(), (
                f"{q} changes sign over the quoted range; quote |A-D| against "
                f"the peak instead")


# ==========================================================================
# What the budget found, kept as the record of how
# ==========================================================================
class TestTheEngineWasTheCause:
    r"""Ogata's reach, and the threshold that was documented and unenforced.

    ``Backend.hankel_h`` has said since it was written that ``n_hankel`` sets
    the *reach* and that the rule "saturates once ``h*N`` exceeds ~2.5".
    ``ACCURATE`` shipped ``h = 0.001, N = 512`` -- ``hN = 0.51``, a fifth of its
    own stated threshold -- and nothing read the number.  That is the same shape
    of defect as a ``Backend`` field with no reader, which this package already
    has a test for; it just had not occurred to anyone that a *pair* of fields
    could be unread in combination.

    The consequence was that the flavour named for accuracy was the less
    accurate one wherever it mattered: 27 per cent low in ``xi`` at 50 Mpc/h
    against a value the other flavour got right.
    """

    @pytest.fixture(scope="class")
    def spectrum(self):
        f, pairs = _spectra(DIFFERENTIABLE)
        return f, (np.asarray(f.k), np.asarray(pairs["gg"].total))

    def test_the_reach_threshold_is_enforced_not_merely_written_down(self):
        from ggah_mod.backend import MIN_OGATA_REACH

        assert ACCURATE.hankel * 1 == "quadrature"
        assert ACCURATE.hankel_h * ACCURATE.n_hankel >= MIN_OGATA_REACH
        with pytest.raises(ValueError, match="hankel_h \\* n_hankel"):
            ACCURATE.with_(n_hankel=512)

    def test_below_the_threshold_the_rule_is_wrong_not_merely_coarse(
            self, spectrum):
        """The number that names the defect.  ``h*N = 0.51`` against a converged
        reference: a quarter of the answer gone at 50 Mpc/h, which no reading of
        "accuracy setting" covers."""
        field, pk = spectrum
        ref = np.asarray(xi(R, pk, backend=DIFFERENTIABLE))     # fftlog
        i50 = int(np.argmin(np.abs(R - 50.0)))

        # The rule cannot be built through `Backend` any more, which is the
        # point of the guard, so the knobs go straight to the transform.
        from ggah_mod.observables.transforms import make_hankel

        bad = np.asarray(xi(R, pk, backend=DIFFERENTIABLE.with_(
            hankel="quadrature", hankel_h=0.005, n_hankel=512)))
        # hN = 2.56, converged: 5.0e-3 measured on the 0.8.0 default spectrum
        # (beyond-linear bias on, no damping), 4.9e-3 before it.
        assert abs(bad[i50] / ref[i50] - 1.0) < 6e-3

        rule = make_hankel(0.5, engine="quadrature", backend=DIFFERENTIABLE,
                           n=512, h=0.001)                # hN = 0.51
        assert float(jnp.max(rule.x)) < 1200.0            # the truncated reach

    def test_the_two_engines_agree_once_the_reach_is_adequate(self, spectrum):
        """Which is what says fftlog was right rather than merely different:
        Ogata converges *to* it from several ``(h, N)`` directions."""
        field, pk = spectrum
        ref = np.asarray(xi(R, pk, backend=DIFFERENTIABLE))
        for h, n in ((0.005, 512), (0.002, 2048), (0.001, 4096)):
            got = np.asarray(xi(R, pk, backend=DIFFERENTIABLE.with_(
                hankel="quadrature", hankel_h=h, n_hankel=n)))
            sel = R >= 2.0          # small r is the other finding, above
            worst = float(np.nanmax(np.abs(got[sel] / ref[sel] - 1.0)))
            assert worst < 1.5e-2, f"h={h}, N={n} disagrees by {worst:.2e}"

    def test_the_transition_did_not_cause_it(self, spectrum):
        """The control that mattered when this was first measured: the engine
        gap is identical with the one-halo damping on and off, so it is not a
        consequence of the transition that landed just before it."""
        field, _ = spectrum
        g = GalaxySector("zheng07", shmr="zu15",
                         backend=DIFFERENTIABLE).weights(
            field, galaxy_defaults("zheng07"))
        k = np.asarray(field.k)
        og = DIFFERENTIABLE.with_(hankel="quadrature", hankel_h=0.005,
                                  n_hankel=512)
        gap = {}
        for tr in ("none", "mead20"):
            p = (k, np.asarray(SP.pk_cross(
                field, g, g, overlap="identical",
                options=SP.PkOptions(one_halo_transition=tr)).total))
            a = np.asarray(xi(R, p, backend=og))
            d = np.asarray(xi(R, p, backend=DIFFERENTIABLE))
            gap[tr] = float(np.abs(a - d).max() / np.abs(a).max())
        assert gap["none"] == pytest.approx(gap["mead20"], rel=5e-2)
        # ...and the illegal pairing that caused the original finding can no
        # longer be built at all, which is the guard doing its job.
        with pytest.raises(ValueError, match="hankel_h \\* n_hankel"):
            DIFFERENTIABLE.with_(hankel="quadrature", hankel_h=0.001,
                                 n_hankel=512)


# ==========================================================================
# The gradient rows -- the one blank row of the derivative table
# ==========================================================================
class TestTheGradientRows:
    r"""``jacfwd`` of the differentiable flavour against a *difference of the
    accurate one*.

    The gradient tests elsewhere are parametrised over both flavours and each is
    checked against its own finite difference, so a differentiable derivative
    thirty per cent from the accurate one passes both.  This is the comparison
    that does not.

    ``ACCURATE`` cannot be differentiated -- ``traced=True`` on a Boltzmann
    backend is refused, which is the whole point of that guard -- so its column
    is a central difference, two CLASS solves per parameter.
    """

    STEP = {"ln10A_s": 0.02, "Omega_m": 0.005, "h": 0.005}
    BOUND = 5e-2

    @staticmethod
    def _sigma8_like(cosmo, backend):
        """One scalar the whole layer-1-to-2 chain reaches: ``sigma(M)`` at a
        fixed mass, which is what ``dn/dM`` and ``b(M)`` are both built from."""
        f = make_field(cosmo, backend, pk=make_pk(backend.pk), z=0.0,
                       calibration="off")
        i = int(np.argmin(np.abs(np.asarray(f.m) - 1e13)))
        return f.sigma[i] if hasattr(f, "sigma") else f.pk_cb[100]

    @pytest.mark.parametrize("name", ["ln10A_s", "Omega_m", "h"])
    def test_the_response_agrees_between_the_flavours(self, name):
        step = self.STEP[name]
        base = float(getattr(PLANCK18, name))

        def accurate_at(value):
            c = PLANCK18.replace(**{name: value})
            return float(self._sigma8_like(c, ACCURATE))

        hi, lo = accurate_at(base + step), accurate_at(base - step)
        d_accurate = (hi - lo) / (2.0 * step)

        def differentiable_at(value):
            c = PLANCK18.replace(**{name: value})
            return self._sigma8_like(c, DIFFERENTIABLE)

        d_diff = float(jax.grad(differentiable_at)(base))

        scale = max(abs(d_accurate), abs(d_diff))
        assert scale > 0.0, f"{name}: both responses are zero"
        rel = abs(d_accurate - d_diff) / scale
        assert rel < self.BOUND, (
            f"d/d{name}: accurate {d_accurate:.4e} against differentiable "
            f"{d_diff:.4e}, {rel:.1%} apart -- a Fisher matrix built on the "
            f"differentiable flavour would inherit that")


# ==========================================================================
# The structural zero the budget was expected to find
# ==========================================================================
class TestTheConcentrationResponse:
    r"""``dc/dtheta`` on the shipped flavours -- and it is **not** zero.

    ``PLAN.md`` carried this as item **B3**, from the technical paper: *"the one
    it ships is an empirical power law in mass and redshift, so
    ``dc/dtheta = 0`` exactly on the single rung through which the cosmology
    reaches the profile ... the flavour that exists to be differentiated has a
    structural zero where a forecast expects a response."*

    Measured, that is stale twice over.  It was written about ``dutton14``,
    which neither flavour has declared since the concentration work landed --
    ``PLAN.md``'s own corrections table already records the finding as stale in
    name and value -- and the relation both flavours *do* declare,
    ``bhattacharya13``, is a power law in **peak height**, not in mass.  Peak
    height is ``delta_c/sigma(M)``, so the cosmology reaches it through the
    variance and the derivative is ordinary and finite.

    Measured at ``M = 1e13 Msun/h``, z = 0, PLANCK18, by ``jax.grad`` through
    ``make_field``:

    ==================  =========  ==================
    relation            ``c``      ``dln c/dln10A_s``
    ==================  =========  ==================
    ``bhattacharya13``  8.6200     **1.45e-1**
    ``duffy08``         8.9202     **0.0**
    ==================  =========  ==================

    So the structural zero is real, and it belongs to ``duffy08`` -- which is a
    power law in mass and redshift alone, exactly as the paper described, and
    which neither flavour ships.  The other four relations in the registry are
    refused at ``mdef = "200m"`` by the calibration guard rather than evaluated,
    which is that guard working.

    **B3 is therefore closed by measurement rather than by work**, and the test
    below is what stops it re-opening silently: it fails if the default acquires
    a zero response, and it fails if ``duffy08`` stops having one.
    """

    def test_the_shipped_relation_responds_to_the_cosmology(self):
        assert ACCURATE.cm_model == DIFFERENTIABLE.cm_model == "bhattacharya13"

        def c_at(ln10a, cm="bhattacharya13"):
            c = PLANCK18.replace(ln10A_s=ln10a)
            f = make_field(c, DIFFERENTIABLE.with_(cm_model=cm),
                           pk=make_pk("emu_pk"), z=0.0, calibration="off")
            return f.conc[int(np.argmin(np.abs(np.asarray(f.m) - 1e13)))]

        base = float(PLANCK18.ln10A_s)
        c0 = float(c_at(base))
        g = float(jax.grad(c_at)(base))
        assert c0 == pytest.approx(8.62, abs=0.05)
        assert g / c0 == pytest.approx(0.145, rel=0.05), (
            f"dln c/dln10A_s is {g / c0:.4e}; it was 1.45e-1 when the parity "
            f"budget was written.  A forecast's concentration response moved")

    def test_the_structural_zero_is_real_and_belongs_to_a_relation_we_do_not_ship(
            self):
        """The paper's concern was not imaginary -- it was about a different
        relation.  ``duffy08`` is a power law in mass and redshift alone, so a
        forecast run on it gets no concentration response at all and a Fisher
        matrix reads that as a flat direction."""
        def c_at(ln10a):
            c = PLANCK18.replace(ln10A_s=ln10a)
            f = make_field(c, DIFFERENTIABLE.with_(cm_model="duffy08"),
                           pk=make_pk("emu_pk"), z=0.0, calibration="off")
            return f.conc[int(np.argmin(np.abs(np.asarray(f.m) - 1e13)))]

        assert float(jax.grad(c_at)(float(PLANCK18.ln10A_s))) == 0.0
        assert DIFFERENTIABLE.cm_model != "duffy08"
