r"""Verification: the occupation models, and that every parameter is differentiable.

Faithfulness to the published equations is checked in ``tests/parity/`` against
the predecessor.  What is checked here is what parity cannot see: that the
models behave, and that a gradient reaches **every** parameter -- including the
four that sit behind a numerical inversion, where the predecessor's gradient was
structurally zero.
"""
import inspect

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod.sectors import occupation as O
from ggah_mod.sectors import sham as SH

LOG10M = jnp.asarray(np.linspace(10.5, 15.5, 25))

#: Multiple of the finite-difference noise floor to allow.
#:
#: A central difference of a float64 function carries an error of about
#: ``eps * |f| / h`` from cancellation alone.  For these models that is not a
#: rounding detail: ``n_sat`` for van Uitert peaks near 9e3, so at ``h = 5e-6``
#: the floor is ~4e-7 while the derivative being measured is 6e-4 -- a 0.1%
#: apparent disagreement that is entirely the *reference*, not the autodiff.
#:
#: Comparing without it forces a choice between a tolerance loose enough to hide
#: a real error and a test that fails on arithmetic.  With it, the comparison is
#: "AD agrees with FD to within FD's own resolution", which is the strongest
#: statement a finite difference can support.
FD_FLOOR_FACTOR = 64.0

def _params(name, fn):
    """The default parameters this function actually takes."""
    d = O.DEFAULTS[name]
    return {k: d[k] for k in inspect.signature(fn).parameters if k in d}


def _all_cases():
    out = []
    for name, (cen, sat) in O.OCCUPATION.items():
        for kind, fn in (("cen", cen), ("sat", sat)):
            for key in _params(name, fn):
                out.append((name, kind, fn, key))
    return out


CASES = _all_cases()


class TestEveryParameterIsDifferentiable:
    """The layer's requirement, parameter by parameter.

    Parametrised over the **cross product** of model and parameter rather than
    over models: a per-model test that sums a gradient would pass while one
    parameter of the ten contributed nothing, which is exactly the failure the
    predecessor had -- `A_cen` and `A_sat` read through `float()`, in a model
    whose other five parameters were fine.
    """

    def test_there_are_cases_to_check(self):
        """Guards the parametrisation against silently checking nothing."""
        assert len(CASES) > 80

    @pytest.mark.x64
    @pytest.mark.parametrize("name,kind,fn,key", CASES,
                             ids=[f"{n}-{k}-{p}" for n, k, _, p in CASES])
    def test_gradient_matches_a_central_difference(self, name, kind, fn, key):
        p = _params(name, fn)

        def f(v):
            q = dict(p)
            q[key] = v
            return jnp.sum(jnp.asarray(fn(LOG10M, **q)))

        x = float(p[key])
        h = 1e-6 * max(abs(x), 1.0)
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad), f"{name}.{kind}.{key} gradient is not finite"

        floor = (FD_FLOOR_FACTOR * np.finfo(float).eps
                 * max(abs(float(f(x))), 1.0) / h)

        if abs(ad) <= floor and abs(fd) <= floor:
            # Below the finite difference's own resolution: the parameter does
            # not measurably move this model on this mass grid.  Recorded as
            # "unresolved", not as agreement -- there is nothing here to agree
            # about, and calling it a pass would let a genuine structural zero
            # hide behind a saturated model.  (`kravtsov04`'s `sigma_logm` is
            # the real instance: `log10mmin = 13` puts the error function's
            # transition inside the grid but saturated at both ends.)
            return

        assert abs(ad - fd) <= max(1e-4 * max(abs(ad), abs(fd)), floor), (
            f"{name}.{kind}.{key}: AD {ad:.6g} vs FD {fd:.6g} "
            f"(fd noise floor {floor:.2g})")
        assert ad != 0.0, (
            f"{name}.{kind}.{key}: autodiff returned exactly zero while the "
            f"finite difference did not.  That is the signature of a lost "
            f"dependence -- a `float()` on the traced path, or a bare "
            f"bisection -- not of a flat model.")

    @pytest.mark.parametrize("name", sorted(O.OCCUPATION))
    def test_jits(self, name):
        cen, sat = O.OCCUPATION[name]
        for fn in (cen, sat):
            p = _params(name, fn)
            out = jax.jit(lambda m, g=fn, q=p: g(m, **q))(LOG10M)
            assert np.all(np.isfinite(np.asarray(out))), name


class TestTheInversionsKeptTheirGradient:
    """The predecessor's referee defect A7, as a reproduction test.

    `d log10 M* / d beta` was autodiff `0.0` against a central difference of
    `0.105`.  A bisection depends on its inputs only through the *sign* of the
    residual, and a boolean carries no derivative, so the gradient was not
    small -- it was structurally absent, and silently so.
    """

    ZM15 = dict(lg_m1h=12.10, lg_m0star=10.31, beta=0.33, delta=0.42,
                gamma=1.21)

    @pytest.mark.x64
    def test_the_inverse_is_an_inverse(self):
        lm = jnp.asarray(np.linspace(11.0, 15.0, 9))
        back = SH.mh_from_mstar_zu15(
            SH.mstar_from_mh_zu15(lm, **self.ZM15), **self.ZM15)
        np.testing.assert_allclose(np.asarray(back), np.asarray(lm), atol=1e-12)

    @pytest.mark.x64
    @pytest.mark.parametrize("key", sorted(ZM15))
    def test_gradient_through_the_inversion(self, key):
        def f(v):
            p = dict(self.ZM15)
            p[key] = v
            return float(jnp.sum(SH.mstar_from_mh_zu15(jnp.asarray(13.0), **p)))

        x = self.ZM15[key]
        ad = float(jax.grad(lambda v: jnp.sum(
            SH.mstar_from_mh_zu15(jnp.asarray(13.0),
                                  **{**self.ZM15, key: v})))(x))
        fd = (f(x + 1e-6) - f(x - 1e-6)) / 2e-6
        assert ad != 0.0, key
        assert ad == pytest.approx(fd, rel=1e-5), key

    @pytest.mark.x64
    def test_a_bare_bisection_would_have_returned_zero(self):
        """The contrast, run rather than asserted from memory.

        Without this the fix above looks like a tolerance choice.  With it, the
        two implementations are side by side and the difference is 0 versus
        -0.58.
        """
        def bare(beta, target=13.0, n=60):
            p = dict(self.ZM15, beta=beta)
            g = lambda x: SH.mh_from_mstar_zu15(x, **p)

            def body(_, ab):
                a, b = ab
                mid = 0.5 * (a + b)
                right = g(mid) < target
                return jnp.where(right, mid, a), jnp.where(right, b, mid)

            a, b = jax.lax.fori_loop(0, n, body,
                                     (jnp.asarray(4.0), jnp.asarray(13.0)))
            return 0.5 * (a + b)

        polished = float(jax.grad(lambda v: jnp.sum(
            SH.mstar_from_mh_zu15(jnp.asarray(13.0),
                                  **{**self.ZM15, "beta": v})))(0.33))
        assert float(jax.grad(bare)(0.33)) == 0.0     # the predecessor
        assert abs(polished) > 0.1                    # the fix
        # and the two agree in *value* to machine precision, which is what
        # makes the Newton polish free rather than a different answer
        assert float(bare(0.33)) == pytest.approx(
            float(SH.mstar_from_mh_zu15(jnp.asarray(13.0), **self.ZM15)),
            abs=1e-12)


class TestOccupationsAreSensible:
    @pytest.mark.parametrize("name", sorted(O.OCCUPATION))
    def test_non_negative_and_finite(self, name):
        cen, sat = O.OCCUPATION[name]
        for fn in (cen, sat):
            v = np.asarray(fn(LOG10M, **_params(name, fn)))
            assert np.all(np.isfinite(v)), name
            assert np.all(v >= -1e-12), name

    @pytest.mark.parametrize("name", sorted(O.OCCUPATION))
    def test_a_central_occupation_never_exceeds_one(self, name):
        """A halo has at most one central.  A model that returns 1.4 is not
        slightly wrong, it is a different quantity."""
        cen, _ = O.OCCUPATION[name]
        v = np.asarray(cen(LOG10M, **_params(name, cen)))
        assert np.max(v) <= 1.0 + 1e-9, f"{name}: max N_cen = {np.max(v)}"

    @pytest.mark.parametrize("name", sorted(O.OCCUPATION))
    def test_satellites_rise_with_halo_mass(self, name):
        _, sat = O.OCCUPATION[name]
        v = np.asarray(sat(LOG10M, **_params(name, sat)))
        assert v[-1] > v[len(v) // 2], name

    def test_the_registry_and_its_calibration_table_agree(self):
        """If a model is added, it must be classified -- the same key-parity
        rule layer 2 applies to its mass functions and bias fits."""
        assert set(O.OCCUPATION) == set(O.OCC_CALIBRATION)
        assert set(O.OCCUPATION) == set(O.DEFAULTS)

    def test_every_default_is_accepted_by_its_model(self):
        """A default that no longer matches a signature is dead weight that
        looks like configuration."""
        for name, (cen, sat) in O.OCCUPATION.items():
            taken = set()
            for fn in (cen, sat):
                taken |= set(inspect.signature(fn).parameters)
            unused = set(O.DEFAULTS[name]) - taken
            assert not unused, f"{name}: unused defaults {sorted(unused)}"

    def test_make_occupation_rejects_an_unknown_name(self):
        with pytest.raises(ValueError, match="unknown occupation model"):
            O.make_occupation("nope")


class TestTheFourSigmasAreNotTheSameQuantity:
    """Named apart on purpose; the module docstring explains why.

    Reading them as one symbol and "fixing" the missing sqrt2 would change the
    central occupation by tens of per cent.

    **This class used to assert the defect it is named after.** It checked
    ``"sigma_logm" in leau`` -- that Leauthaud's scatter in ``log10 M*`` carried
    the *halo*-mass name, the one it shared with ``zheng07`` -- and called that
    "no accidental sharing".  It was the accidental sharing.  PLAN.md item
    **G5**; the names are ``sigma_logm``, ``sigma_logmstar``, ``sigma_lnmstar``
    and ``width_logmstar`` now, one per (variable, convention) pair.
    """

    #: ``model -> (parameter, variable, convention)``.  Four rows, no two of
    #: which may share a name.
    EXPECTED = {
        "n_cen_zheng07": "sigma_logm",
        "n_cen_leauthaud12": "sigma_logmstar",
        "n_cen_zu15": "sigma_lnmstar",
        "n_cen_zacharegkas25": "width_logmstar",
    }

    @pytest.mark.parametrize("fn,param", sorted(EXPECTED.items()))
    def test_each_model_takes_its_own_name(self, fn, param):
        pars = set(inspect.signature(getattr(O, fn)).parameters)
        assert param in pars
        others = set(self.EXPECTED.values()) - {param}
        assert not (others & pars), (
            f"{fn} takes {sorted(others & pars)} as well as {param!r}; four "
            f"conventions, one name each")

    def test_no_two_conventions_share_a_name(self):
        assert len(set(self.EXPECTED.values())) == len(self.EXPECTED)

    def test_the_two_stellar_mass_names_differ_by_more_than_punctuation(self):
        """``sigma_logmstar`` against ``sigma_logm_star`` would have replaced a
        collision with a typo, so the erf-width one is ``width_`` instead."""
        a, b = "sigma_logmstar", "width_logmstar"
        assert a.replace("_", "") != b.replace("_", "")

    def test_the_sqrt2_matters(self):
        """A 1-sigma offset, with and without: 0.159 against 0.241."""
        from jax.scipy.special import erfc
        arg = 1.0
        assert float(0.5 * erfc(arg)) == pytest.approx(0.0786, abs=1e-3)
        assert float(0.5 * erfc(arg / np.sqrt(2.0))) == pytest.approx(
            0.1587, abs=1e-3)


class TestLange25:
    r"""Lange et al. (2025), DESI DR1: Zheng07 centrals x completeness,
    Kravtsov04 satellites.

    Added to reproduce ``hod_mod``'s ``benchmark_lange2025_*`` configs, which
    could not be built at all before -- ``build_benchmark_fit`` refused them
    with *"no ggah_mod occupation for 'Lange25HODModel'"*.

    **Their assembly bias is not implemented.**  Lange et al. fit the Hearin
    et al. (2016) decorated HOD on AbacusSummit.  The analytic kernel this
    package once applied on ``a_cen``/``a_sat`` was not that model, and it was
    removed.
    """

    P = dict(log10mmin=13.0, sigma_logm=0.3, log10m0=13.5, log10m1=14.0,
             alpha=1.0, f_gamma=1.0)

    def test_the_undecorated_limit_is_exactly_kravtsov04(self):
        """f_gamma = 1 is not *approximately* the base model."""
        lm = jnp.asarray(np.linspace(11.0, 15.5, 60))
        nc_l = O.n_cen_lange25(lm, self.P["log10mmin"], self.P["sigma_logm"],
                               self.P["f_gamma"])
        ns_l = O.n_sat_lange25(lm, self.P["log10mmin"], self.P["sigma_logm"],
                               self.P["log10m0"], self.P["log10m1"],
                               self.P["alpha"], self.P["f_gamma"])
        nc_k = O.n_cen_zheng07(lm, self.P["log10mmin"], self.P["sigma_logm"])
        ns_k = O.n_sat_kravtsov04(lm, self.P["log10mmin"], self.P["sigma_logm"],
                                  self.P["log10m0"], self.P["log10m1"],
                                  self.P["alpha"])
        assert np.array_equal(np.asarray(nc_l), np.asarray(nc_k))
        assert np.array_equal(np.asarray(ns_l), np.asarray(ns_k))

    def test_completeness_scales_centrals_and_not_satellites(self):
        r"""Applying :math:`f_\Gamma` twice would put it squared into every
        satellite pair count, which is the error this asymmetry prevents."""
        lm = jnp.asarray(np.linspace(11.0, 15.5, 60))
        args = (self.P["log10mmin"], self.P["sigma_logm"])
        sat = (self.P["log10m0"], self.P["log10m1"], self.P["alpha"])
        for f in (0.5, 0.75, 1.0):
            nc = np.asarray(O.n_cen_lange25(lm, *args, f))
            ns = np.asarray(O.n_sat_lange25(lm, *args, *sat, f))
            assert np.allclose(nc, f * np.asarray(O.n_cen_zheng07(lm, *args)))
            assert np.allclose(ns, np.asarray(
                O.n_sat_kravtsov04(lm, *args, *sat)))

    def test_the_completeness_box_stops_at_one(self):
        """A completeness above 1 says the sample holds centrals the haloes do
        not have.

        Declared in ``VOCABULARY``, the per-quantity registry, because
        ``f_gamma`` is new.
        """
        from ggah_mod.sectors.occupation_params import VOCABULARY
        assert VOCABULARY["f_gamma"]["bounds"] == (0.5, 1.0)

    def test_no_assembly_bias_amplitude_is_declared_anywhere(self):
        """The decoration was removed, so its amplitudes must not survive as a
        declared parameter a sampler could free and the model would ignore."""
        from ggah_mod.sectors.galaxies import GalaxyParams
        from ggah_mod.sectors.occupation_params import VOCABULARY
        for n in ("a_cen", "a_sat"):
            assert n not in GalaxyParams._PARAMS
            assert n not in VOCABULARY


class TestTheScatterFloor:
    r""":func:`~ggah_mod.sectors.sham.scatter_zu15` is floored at the prior's
    lower bound, because a steep negative ``eta`` otherwise takes the width
    through zero and :func:`~ggah_mod.sectors.occupation.n_cen_zu15` to zero
    with it -- the LS10 mass-bin MAP of ggah_cal's v0.8.0 campaign emptied every
    halo above :math:`10^{14.955}`."""

    #: ggah_cal ``massbins_zu15_gt10.5_nbar-wp``, v0.8.0 campaign, rounded.
    LS10_V080 = dict(lg_m1h=12.491, lg_m0star=10.150, beta=0.728, delta=0.580,
                     gamma=0.845, sigma_lnmstar=0.848, eta=-0.344, fc=1.702,
                     bsat=15.14, beta_sat=0.690, bcut=1.722, beta_cut=0.932,
                     alpha_sat=1.165)
    LOG10M_WIDE = jnp.asarray(np.linspace(10.5, 16.0, 56))

    def test_the_floor_is_the_prior_lower_bound(self):
        from ggah_mod.sectors.galaxies import GalaxyParams
        assert SH.SIGMA_LNMSTAR_FLOOR == \
            GalaxyParams._PARAMS["sigma_lnmstar"].bounds[0]

    def test_the_published_values_never_reach_the_floor(self):
        """Where the floor does not bind the function is the published one bit
        for bit, so at Paper I's values it changes nothing.  The reference is
        the unfloored line jitted the same way: against numpy the two differ by
        an ulp of fused multiply-add, which is not the floor."""
        d = O.ZU15_PUBLISHED
        args = (self.LOG10M_WIDE, d["sigma_lnmstar"], d["eta"], d["lg_m1h"])
        got = np.asarray(SH.scatter_zu15(*args))
        raw = np.asarray(jax.jit(
            lambda m, s, e, l: s + e * jnp.maximum(m - l, 0.0))(*args))
        assert np.min(raw) > SH.SIGMA_LNMSTAR_FLOOR
        assert np.array_equal(got, raw)

    @pytest.mark.parametrize(
        "name", ["zumandelbaum15", "zumandelbaum16_red", "zumandelbaum16_blue"])
    def test_the_defaults_reach_it_only_at_the_top_of_the_grid(self, name):
        r"""The LS10 defaults (0.8.5) take the line to the floor at
        :math:`\lg M_h = \lg M_1 + (\sigma_{\rm floor} - \sigma_0)/\eta = 15.730`,
        inside the field's grid (to :math:`10^{16}`) but above any halo the fit
        saw in number."""
        d = O.DEFAULTS[name]
        cross = d["lg_m1h"] + (SH.SIGMA_LNMSTAR_FLOOR - d["sigma_lnmstar"]) / d["eta"]
        assert cross == pytest.approx(15.730, abs=1e-3)
        s = np.asarray(SH.scatter_zu15(self.LOG10M_WIDE, d["sigma_lnmstar"],
                                       d["eta"], d["lg_m1h"]))
        below = np.asarray(self.LOG10M_WIDE) < cross
        assert np.all(s[below] > SH.SIGMA_LNMSTAR_FLOOR)
        assert np.all(s[~below] == SH.SIGMA_LNMSTAR_FLOOR)

    def test_a_steep_eta_keeps_the_clusters_populated(self):
        p = dict(self.LS10_V080, log10m_star_thresh=10.2)
        cen = {k: p[k] for k in inspect.signature(O.n_cen_zu15).parameters
               if k in p}
        sat = {k: p[k] for k in inspect.signature(O.n_sat_zu15).parameters
               if k in p}
        sigma = np.asarray(SH.scatter_zu15(self.LOG10M_WIDE, p["sigma_lnmstar"],
                                           p["eta"], p["lg_m1h"]))
        assert np.min(sigma) == pytest.approx(SH.SIGMA_LNMSTAR_FLOOR, abs=0)
        nc = np.asarray(O.n_cen_zu15(self.LOG10M_WIDE, **cen))
        ns = np.asarray(O.n_sat_zu15(self.LOG10M_WIDE, **sat))
        above = np.asarray(self.LOG10M_WIDE) > 15.0
        assert np.allclose(nc[above], p["fc"], rtol=1e-9), nc[above]
        assert np.all(np.diff(ns[np.asarray(self.LOG10M_WIDE) > 13.0]) > 0), ns


class TestVanUitert16SHMR:
    r"""van Uitert et al. (2016) Eq. 16, checked against the paper rather than
    against the predecessor, which carried the same error as this package did
    up to 1.0.0:

    .. math::

        M_*^c = M_{*,0}\,\frac{(M_h/M_{h,1})^{\beta_1}}
                              {\left[1 + M_h/M_{h,1}\right]^{\beta_1-\beta_2}}

    The exponent :math:`\beta_1-\beta_2` is outside the bracket.  Inside it,
    :math:`\log_{10}[1 + 10^{(\beta_1-\beta_2)x}]`, the two asymptotic slopes
    survive and the turnover does not -- 1.1 dex high at :math:`M_{h,1}` at the
    defaults, which no test of the slopes alone would see.
    """

    #: The defaults, and van Uitert et al.'s Table 3 "All" and "Cen" medians.
    CASES = [
        dict(O.DEFAULTS["vanuitert16"]),
        dict(log10m_h1=10.97, log10m_star0=10.58, beta1=7.5,
             log10_beta2=float(np.log10(0.25))),
        dict(log10m_h1=12.06, log10m_star0=11.16, beta1=5.4,
             log10_beta2=float(np.log10(0.15))),
    ]
    LOG10M = np.array([10.0, 10.8, 11.5, 12.0, 12.6, 13.3, 14.0, 15.0])

    @staticmethod
    def _eq16(log10m, log10m_h1, log10m_star0, beta1, log10_beta2):
        """Eq. 16 as printed, in linear mass, by numpy."""
        r = 10.0 ** (np.asarray(log10m) - log10m_h1)
        beta2 = 10.0 ** log10_beta2
        return np.log10(10.0 ** log10m_star0 * r ** beta1
                        / (1.0 + r) ** (beta1 - beta2))

    @staticmethod
    def _shmr(p):
        keys = ("log10m_h1", "log10m_star0", "beta1", "log10_beta2")
        return {k: p[k] for k in keys}

    @pytest.mark.parametrize("case", range(3))
    def test_it_is_eq16_as_published(self, case):
        p = self._shmr(self.CASES[case])
        got = np.asarray(O.shmr_vanuitert16(jnp.asarray(self.LOG10M), **p))
        want = self._eq16(self.LOG10M, **p)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("case", range(3))
    def test_the_turnover_is_beta1_minus_beta2_times_log2(self, case):
        r"""At :math:`M_h = M_{h,1}` the relation sits
        :math:`(\beta_1-\beta_2)\log_{10}2` below :math:`M_{*,0}`; the 1.0.0
        form put it :math:`\log_{10}2` below, whatever the slopes."""
        p = self._shmr(self.CASES[case])
        beta2 = 10.0 ** p["log10_beta2"]
        got = float(O.shmr_vanuitert16(p["log10m_h1"], **p))
        assert got == pytest.approx(
            p["log10m_star0"] - (p["beta1"] - beta2) * np.log10(2.0), abs=1e-12)

    def test_the_asymptotic_slopes_are_beta1_and_beta2(self):
        p = self._shmr(self.CASES[0])
        beta2 = 10.0 ** p["log10_beta2"]
        slope = jax.grad(lambda m: O.shmr_vanuitert16(m, **p))
        assert float(slope(p["log10m_h1"] - 6.0)) == pytest.approx(
            p["beta1"], rel=1e-4)
        assert float(slope(p["log10m_h1"] + 6.0)) == pytest.approx(
            beta2, rel=1e-4)

    def test_it_stays_finite_far_from_the_knee(self):
        """``logaddexp`` rather than ``log10(1 + 10**x)``, which overflows."""
        p = self._shmr(self.CASES[1])
        lm = jnp.asarray([p["log10m_h1"] - 400.0, p["log10m_h1"] + 400.0])
        val = np.asarray(O.shmr_vanuitert16(lm, **p))
        grad = np.asarray(jax.vmap(jax.grad(
            lambda m: O.shmr_vanuitert16(m, **p)))(lm))
        assert np.all(np.isfinite(val)) and np.all(np.isfinite(grad))
