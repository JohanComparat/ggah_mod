r"""Verification: the halo layer's JAX implementation, and its coherence with
the cosmology layer.

Two questions, both answered by checking rather than by convention:

**Is it actually JAX?**  Every public function in :mod:`ggah_mod.halos` must be
traceable and differentiable, or declared not to be.  The predecessor had numpy
functions sitting on its traced path -- ``einasto_uk`` and ``satellite_nfw_uk``
-- which never failed because the one gradient test differentiated an amplitude
parameter that does not reach the halo radius or concentration.  So the test
here differentiates with respect to :math:`\Omega_m`, which reaches everything.

**Is it coherent with the cosmology layer?**  Three ways it could stop being:

* an **import cycle** -- the halo layer may depend on the cosmology, never the
  reverse;
* a **duplicated constant** -- the predecessor had ``rho_crit,0`` written as a
  literal in one module while three others called the function, ``delta_c`` at
  eleven sites, and :math:`g(y)` eight times;
* a **confused density** -- ``rho_cold`` and ``rho_matter`` answer different
  questions and differ by 0.46% at the minimal neutrino mass.  Halos form from
  the cold field; a halo boundary is defined against all the matter.
"""
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod
from ggah_mod import ACCURATE
from ggah_mod.cosmology import PLANCK18, Cosmology
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.power import make_pk
from ggah_mod import halos as H

_K = np.logspace(-4, 2, 256)
_M = np.logspace(11, 15, 24)


# =========================================================================
# Coherence: layering
# =========================================================================

class TestLayering:
    def test_cosmology_does_not_import_halos(self):
        """The dependency runs one way.  A cycle here would mean the cosmology
        layer could not be reasoned about on its own."""
        import ggah_mod.cosmology as cosmo_pkg
        for mod in pkgutil.iter_modules(cosmo_pkg.__path__):
            m = importlib.import_module(f"ggah_mod.cosmology.{mod.name}")
            src = inspect.getsource(m)
            assert "from ..halos" not in src, mod.name
            assert "import ggah_mod.halos" not in src, mod.name

    def test_halos_uses_the_cosmology_layer_rather_than_reimplementing_it(self):
        """Anything the halo layer needs about the background must come from
        the cosmology layer, not be recomputed locally."""
        import ggah_mod.halos.mass_definitions as md
        src = inspect.getsource(md)
        assert "hubble_e" in src            # E(z) from the cosmology layer
        assert "RHO_CRIT0" in src           # and its critical density

    def test_profiles_depend_on_nothing_but_jax(self):
        """Pure shapes: a profile takes r_s and rho_s, not a cosmology.  That
        is what lets mass_definitions import it without a cycle."""
        src = inspect.getsource(H.profiles if hasattr(H, "profiles")
                                else importlib.import_module("ggah_mod.halos.profiles"))
        assert "from ..cosmology" not in src


class TestNoDuplicatedConstants:
    """Each physical constant has exactly one definition in the package."""

    @staticmethod
    def _sources():
        out = {}
        for pkg in ("ggah_mod.cosmology", "ggah_mod.halos"):
            p = importlib.import_module(pkg)
            for mod in pkgutil.iter_modules(p.__path__):
                name = f"{pkg}.{mod.name}"
                out[name] = inspect.getsource(importlib.import_module(name))
        return out

    def test_delta_c_defined_once(self):
        """The predecessor hard-coded 1.686 at about eleven sites."""
        hits = [n for n, s in self._sources().items()
                if "1.686" in s and "DELTA_C = " not in s]
        assert not hits, f"delta_c literal outside variance.py: {hits}"

    def test_critical_density_defined_once(self):
        hits = [n for n, s in self._sources().items()
                if "2.775" in s and "RHO_CRIT0 = " not in s]
        assert not hits, f"rho_crit,0 literal outside constants.py: {hits}"

    def test_nfw_mass_shape_defined_once(self):
        """g(y) = ln(1+y) - y/(1+y).  Eight copies in the predecessor, one of
        them dead code."""
        hits = [n for n, s in self._sources().items()
                if "log1p(y) - y / (1.0 + y)" in s]
        assert hits == ["ggah_mod.halos.profiles"], hits

    def test_bryan_norman_defined_once(self):
        hits = [n for n, s in self._sources().items() if "82.0 * x - 39.0" in s]
        assert hits == ["ggah_mod.halos.mass_definitions"], hits


class TestColdVersusTotalDensity:
    """The two densities that get confused, and which uses which."""

    def test_they_are_different(self):
        assert PLANCK18.rho_cold < PLANCK18.rho_matter
        assert PLANCK18.rho_matter / PLANCK18.rho_cold - 1.0 > 4e-3

    def test_variance_uses_the_cold_field(self):
        """Halos form out of the matter that collapses, and neutrinos do not."""
        assert "rho_cold" in inspect.signature(H.sigma_of_mass).parameters
        assert "rho_cold" in inspect.signature(H.lagrangian_radius).parameters

    def test_mass_definitions_use_total_matter(self):
        """A halo boundary is defined against the mean density of the universe,
        which neutrinos contribute to."""
        _, rho = H.MassDef.from_string("200m").delta_rho(0.0, PLANCK18)
        assert float(rho) == pytest.approx(PLANCK18.rho_matter, rel=1e-12)

    def test_swapping_them_is_a_measurable_error(self):
        """Not a rounding difference: 0.15% in radius, 0.46% in enclosed mass."""
        r_cold = float(H.lagrangian_radius(1e13, PLANCK18.rho_cold))
        r_tot = float(H.lagrangian_radius(1e13, PLANCK18.rho_matter))
        assert abs(r_cold / r_tot - 1.0) > 1e-3

    def test_massless_collapses_the_distinction(self):
        m0 = Cosmology.create(sum_mnu=0.0)
        assert m0.rho_cold == m0.rho_matter


# =========================================================================
# JAX implementation
# =========================================================================

def _b_eff(cosmo, pk, model="tinker08", bias="tinker10"):
    """The whole halo chain: cosmology -> P_cb -> sigma -> dn/dM -> b_eff."""
    p = pk.pk_cb(_K, 0.0, cosmo)
    s = H.sigma_of_mass(_M, _K, p, cosmo.rho_cold)
    d = H.dln_sigma_dln_mass(_M, _K, p, cosmo.rho_cold)
    n = H.dndm(_M, s, d, cosmo.rho_cold, model=model)
    b = H.make_bias(bias)(s)
    return jnp.trapezoid(n * b * _M, jnp.log(_M)) / jnp.trapezoid(n * _M, jnp.log(_M))


@pytest.fixture(scope="module")
def fast_pk():
    return make_pk("emu_pk")


class TestWholeChainIsTraceable:
    def test_jit(self, fast_pk):
        got = float(jax.jit(lambda c: _b_eff(c, fast_pk))(PLANCK18))
        assert np.isfinite(got) and got > 0.0

    @pytest.mark.x64
    @pytest.mark.parametrize("field", ["Omega_m", "ln10A_s", "h", "n_s", "sum_mnu"])
    def test_gradient_reaches_every_cosmological_parameter(self, fast_pk, field):
        f = lambda v: jnp.log(_b_eff(PLANCK18.replace(**{field: v}), fast_pk))
        x = float(getattr(PLANCK18, field))
        step = 1e-6 * max(abs(x), 1e-3)
        ad = float(jax.grad(f)(x))
        fd = float((f(x + step) - f(x - step)) / (2 * step))
        assert np.isfinite(ad)
        assert ad == pytest.approx(fd, rel=5e-3, abs=1e-8), field


def _chain(cosmo, pk, z=0.0, model="tinker08", bias="tinker10", k_out=None):
    """Every rung of the cosmology -> halos ladder, kept rather than collapsed.

    ``_b_eff`` above runs the same path but returns one number, so a hop that
    goes wrong in a way another hop compensates for stays invisible.  This
    returns the intermediates so each can be checked against something it must
    satisfy on its own.
    """
    k_out = jnp.asarray(np.logspace(-2, 0.5, 16)) if k_out is None else k_out
    m = jnp.asarray(_M)

    p_cb = pk.pk_cb(_K, z, cosmo)
    sigma = H.sigma_of_mass(m, _K, p_cb, cosmo.rho_cold)
    dlns = H.dln_sigma_dln_mass(m, _K, p_cb, cosmo.rho_cold)
    nu = H.DELTA_C / sigma
    n_m = H.dndm(m, sigma, dlns, cosmo.rho_cold, model=model, z=z)
    b_m = H.make_bias(bias)(sigma)

    mdef = H.MassDef.from_string("200m")
    r_delta = mdef.radius(m, z, cosmo)
    # diemer19 rather than an empirical power law, deliberately: it is the
    # relation that takes sigma, so the cosmology reaches the concentration and
    # therefore the profile.  The empirical fits take (m, z, mdef) and cannot
    # respond to a cosmology at all -- which is the point of their having a
    # different signature, and would make the gradient through this rung
    # identically zero for an uninteresting reason.
    n_eff = H.n_eff_from_sigma(
        m, cosmo, lambda mm: H.dln_sigma_dln_mass(mm, _K, p_cb, cosmo.rho_cold))
    c = H.c_diemer19(sigma, n_eff, 0.52)
    u_k = H.nfw_uk(k_out, r_delta / c, c)                       # (Nk, NM)

    w = jnp.trapezoid(n_m * m, jnp.log(m))
    b_eff = jnp.trapezoid(n_m * b_m * m, jnp.log(m)) / w
    return dict(p_cb=p_cb, sigma=sigma, dlns=dlns, nu=nu, dndm=n_m, bias=b_m,
                r_delta=r_delta, c=c, u_k=u_k, b_eff=b_eff, m=m, k_out=k_out)


class TestFullChainCosmologyToHalos:
    """The ladder, rung by rung: values, then jit, then a gradient.

    Traceability is already covered above.  What is not, and is what a reader of
    the paper would want established, is that the chain computes the right
    *numbers* -- each rung against an invariant it must satisfy regardless of
    which fit or backend is selected.
    """

    @pytest.fixture(scope="class")
    def chain(self, fast_pk):
        return _chain(PLANCK18, fast_pk)

    # -- values, rung by rung ---------------------------------------------

    def test_sigma_falls_with_mass_and_brackets_collapse(self, chain):
        s = np.asarray(chain["sigma"])
        assert np.all(np.diff(s) < 0)
        # sigma = delta_c somewhere inside this mass range: M_star is ~1e12-1e13
        assert s[0] > H.DELTA_C > s[-1]

    def test_peak_height_crosses_unity_inside_the_mass_range(self, chain):
        nu = np.asarray(chain["nu"])
        assert np.all(np.diff(nu) > 0)
        assert nu[0] < 1.0 < nu[-1]

    def test_log_derivative_is_negative_and_matches_a_difference(self, chain):
        d = np.asarray(chain["dlns"])
        assert np.all(d < 0)
        # Against a second-order difference on the 24-point chain grid.  The
        # tolerance is set by that difference, not by the derivative: `dndm` is
        # proportional to this quantity, which is why the code computes it by AD
        # rather than differencing an integral in the first place.
        lnm, lns = np.log(np.asarray(chain["m"])), np.log(np.asarray(chain["sigma"]))
        np.testing.assert_allclose(d[2:-2], np.gradient(lns, lnm)[2:-2], rtol=1e-2)

    def test_mass_function_is_positive_and_steeply_falling(self, chain):
        n = np.asarray(chain["dndm"])
        assert np.all(n > 0) and np.all(np.diff(n) < 0)
        # four decades in mass must cost far more than four in abundance
        assert n[0] / n[-1] > 1e6

    def test_bias_rises_through_unity(self, chain):
        b = np.asarray(chain["bias"])
        assert np.all(np.diff(b) > 0)
        assert b[0] < 1.0 < b[-1]

    def test_the_mass_fraction_matches_its_closed_form(self, fast_pk):
        """Press-Schechter end to end, against the analytic answer.

        Not against unity: the integral reaches 1 only over *all* mass, and the
        deficit on a finite k grid is the mass below the smallest resolved
        scale rather than an error -- `test_halos.py::TestPressSchechterIsExact`
        makes the same point.  What can be asserted end to end is that the
        chain's own sigma and its own dn/dM agree with the closed form for the
        range they actually cover.
        """
        from scipy.special import erfc

        m = np.logspace(2, 17, 1200)
        p_cb = fast_pk.pk_cb(_K, 0.0, PLANCK18)
        sigma = H.sigma_of_mass(m, _K, p_cb, PLANCK18.rho_cold)
        dlns = H.dln_sigma_dln_mass(m, _K, p_cb, PLANCK18.rho_cold)
        n = H.dndm(m, sigma, dlns, PLANCK18.rho_cold, model="press74")

        got = float(H.mass_fraction(m, n, PLANCK18.rho_cold))
        nu0 = H.DELTA_C / float(np.max(np.asarray(sigma)))
        assert got == pytest.approx(erfc(nu0 / np.sqrt(2.0)), rel=2e-3)
        assert got < 1.0

    def test_concentration_and_radius_are_physical(self, chain):
        c, r = np.asarray(chain["c"]), np.asarray(chain["r_delta"])
        assert np.all((c > 1.0) & (c < 30.0))
        assert np.all(np.diff(c) < 0)                  # falls with mass
        assert np.all(np.diff(r) > 0)                  # rises with mass
        # r ~ M^(1/3) exactly, for a fixed overdensity
        lm, lr = np.log(np.asarray(chain["m"])), np.log(r)
        assert np.polyfit(lm, lr, 1)[0] == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_profile_transform_has_the_right_limits(self, chain):
        u = np.asarray(chain["u_k"])
        assert np.all(u <= 1.0 + 1e-9)
        assert np.all(np.diff(u, axis=0) < 0)          # falls with k, every mass
        # u -> 1 as k -> 0, and the most massive halo departs first
        u0 = np.asarray(H.nfw_uk(jnp.asarray([1e-4]),
                                 chain["r_delta"] / chain["c"], chain["c"]))
        np.testing.assert_allclose(u0[0], 1.0, atol=1e-6)
        assert u[-1, -1] < u[-1, 0]

    def test_b_eff_is_of_order_unity(self, chain):
        assert 0.5 < float(chain["b_eff"]) < 3.0

    # -- jit, gradient ------------------------------------------------------

    def test_jit_is_value_identical(self, fast_pk):
        """Not merely finite: the same numbers to round-off.

        `TestWholeChainIsTraceable.test_jit` checks the chain compiles; a jitted
        path that silently took a different branch would pass that and fail
        this.
        """
        eager = _chain(PLANCK18, fast_pk)
        jitted = jax.jit(lambda c: _chain(c, fast_pk))(PLANCK18)
        for key in ("sigma", "dndm", "bias", "c", "u_k", "b_eff"):
            np.testing.assert_allclose(np.asarray(jitted[key]),
                                       np.asarray(eager[key]), rtol=1e-12,
                                       err_msg=key)

    @pytest.mark.x64
    @pytest.mark.parametrize("rung", ["sigma", "dndm", "bias", "c", "u_k", "b_eff"])
    def test_gradient_reaches_every_rung(self, fast_pk, rung):
        """Omega_m, which is the parameter that reaches the halo radius."""
        f = lambda om: jnp.log(jnp.sum(jnp.abs(
            _chain(PLANCK18.replace(Omega_m=om), fast_pk)[rung])))
        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0, rung
        assert ad == pytest.approx(fd, rel=1e-4), rung

    # -- the two flavours ---------------------------------------------------

    def test_both_flavours_run_the_same_chain(self, fast_pk):
        """DIFFERENTIABLE against ACCURATE, in value.

        Derivative parity is deliberately not asserted here: it is the standing
        parity-budget item, and claiming it from a value comparison is exactly
        the substitution this package exists to avoid.
        """
        # CLASS is the ACCURATE flavour's default and is an *extra*
        # (`pip install ggah_mod[reference]`).  Skipped rather than failed
        # when it is absent, so the suite passes in the environment a plain
        # `pip install ggah_mod` creates -- which is what CI checks, and
        # which was not true until this guard existed.
        pytest.importorskip("classy")
        accurate = make_pk(ACCURATE.pk)
        fast = _chain(PLANCK18, fast_pk)
        acc = _chain(PLANCK18, accurate)
        # c and u_k are not independent tolerances, and `c` is the loosest row
        # here for a reason worth writing down.
        #
        # `diemer19` takes n_eff = -6 dln sigma/dln M - 3: a slope of a slope.
        # The two flavours' sigma(M) agree to 6.6e-4, but their *log-derivative*
        # agrees only to 5.0e-3 -- differentiating amplifies the emulator's
        # shape error -- and n_eff multiplies that by six before dc/dn_eff ~ 3.6
        # turns it into 3.2e-2 in the concentration.
        #
        # `diemer15` looked far better here, and the reason it looked better is
        # the reason it was replaced: its n_eff came from an analytic no-wiggle
        # fit, which is *identical for both flavours* whatever backend is
        # driving them.  The parity was a property of the fitting function, not
        # of the chain.  This number is larger and it is real.
        for key, tol in (("sigma", 3e-3), ("bias", 3e-3), ("c", 5e-2),
                         ("u_k", 5e-2), ("b_eff", 5e-3)):
            np.testing.assert_allclose(np.asarray(fast[key]),
                                       np.asarray(acc[key]), rtol=tol,
                                       err_msg=f"DIFFERENTIABLE vs ACCURATE: {key}")


class TestChainReachesTheNewestPieces:
    """The three things no chain test touched before this round."""

    def test_beyond_linear_bias_closes_the_two_halo_correction(self, fast_pk):
        """From a cosmology to a beta^NL correction, in one path."""
        ch = _chain(PLANCK18, fast_pk)
        tab = H.table_at(ch["k_out"], _K, ch["p_cb"], check=False)

        dm = jnp.gradient(ch["m"])
        w = ch["dndm"] * ch["bias"] * dm
        w = w / jnp.trapezoid(ch["dndm"] * dm, ch["m"])
        delta = H.correction_2h_gg(ch["nu"], w, ch["u_k"], tab)

        assert delta.shape == ch["k_out"].shape
        assert np.all(np.isfinite(np.asarray(delta)))
        # zero below the taper, non-zero above it
        below = H.table_at(jnp.asarray([1e-3]), _K, ch["p_cb"], check=False)
        assert float(jnp.sum(jnp.abs(below.beta))) == 0.0
        assert float(jnp.max(jnp.abs(delta))) > 0.0

    @pytest.mark.x64
    def test_beyond_linear_bias_gradient_reaches_the_cosmology(self, fast_pk):
        r"""The gradient is connected, checked where it can actually be checked.

        This used to compare autodiff against a central difference on
        ``sum(tab.beta)``, and that comparison is not one this quantity can
        support.  :math:`\beta^{\rm NL}` is a *shape* statistic: the amplitude
        divides out of it almost exactly, leaving
        ``dln(sum beta)/dln10A_s ~ 1e-12``.  The sum itself is -211, so a
        central difference subtracts two numbers agreeing to their twelfth
        significant digit -- at float64 eps that leaves a noise floor of
        ~2e-11 against a true derivative of ~2e-10, one digit of signal.  The
        difference does not converge as the step shrinks; it changes *sign*
        between h = 2e-3 and h = 1e-3.  It passed against the previous
        spectrum by luck, not by measurement.

        What "reaches the cosmology" actually asserts is that the chain does
        not cut the gradient, and that is checked twice here.  Upstream, on a
        quantity with a derivative worth resolving: :math:`P \propto A_s`, so
        ``dln(sum p_cb)/dln10A_s`` must be exactly the number of modes.
        Downstream, that the tapered table still carries a finite, non-zero
        derivative rather than a severed or NaN one.
        """
        def p_cb_of(lnA):
            return _chain(PLANCK18.replace(ln10A_s=lnA), fast_pk)["p_cb"]

        # Upstream: exact, and the tightest statement available on this chain.
        g = float(jax.grad(lambda a: jnp.sum(jnp.log(p_cb_of(a))))(PLANCK18.ln10A_s))
        assert g == pytest.approx(float(len(_K)), rel=1e-10), \
            "the amplitude stopped reaching p_cb"

        # Downstream: connected, finite, and not identically zero.
        def f(lnA):
            c = PLANCK18.replace(ln10A_s=lnA)
            ch = _chain(c, fast_pk)
            tab = H.table_at(ch["k_out"], _K, ch["p_cb"], check=False)
            return jnp.sum(tab.beta)

        ad = float(jax.grad(f)(PLANCK18.ln10A_s))
        assert np.isfinite(ad)
        assert ad != 0.0, "beyond_linear_bias severed the gradient"

    @pytest.mark.x64
    @pytest.mark.parametrize("which", ["tnfw", "bmo", "hernquist"])
    def test_lensing_profiles_are_on_the_chain_and_externally_checked(self, which):
        """Reached, differentiated -- and now checked against another code.

        This assertion used to read ``"no external reference" in doc``, and it
        was right to: skipping these would have let a transcription error sit
        behind a status note, and asserting agreement with a reference that did
        not exist would have overstated what was known.  The reference exists
        now (``tests/test_lensing_goldens.py``, Oguri et al. 2026 at 50
        digits), so the assertion flips rather than being deleted -- what it
        pins is that the module's stated status matches the tests that exist,
        in whichever direction that is.
        """
        import ggah_mod.halos.lensing_profiles as LP
        doc = inspect.getdoc(LP).lower()
        assert "oguri" in doc
        assert "no external reference" not in doc

        def f(om):
            cosmo = PLANCK18.replace(Omega_m=om)
            r_d = H.MassDef.from_string("200m").radius(1e14, 0.0, cosmo)
            c = H.c_duffy08(1e14, 0.0, "200m")
            r_s, R = r_d / c, jnp.asarray(np.logspace(-1.5, 0.5, 8))
            if which == "tnfw":
                return jnp.sum(H.tnfw_delta_sigma(R, 1.0, r_s, 2.0 * r_d))
            if which == "bmo":
                return jnp.sum(H.bmo_delta_sigma(R, 1.0, r_s, 2.0 * r_d / r_s))
            return jnp.sum(H.hernquist_delta_sigma(R, 1e14, r_s))

        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        fd = float((f(PLANCK18.Omega_m + 1e-6) - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0, which
        assert ad == pytest.approx(fd, rel=1e-4), which

    @pytest.mark.x64
    def test_generic_gauss_legendre_transform_is_differentiable(self):
        """`profile_uk_gl`, the fallback any new profile gets its u(k) from."""
        def f(om):
            cosmo = PLANCK18.replace(Omega_m=om)
            r_d = H.MassDef.from_string("200m").radius(1e13, 0.0, cosmo)
            k = jnp.asarray(np.logspace(-1, 0.5, 6))
            rho = lambda r: jnp.exp(-r / (0.2 * r_d))
            return jnp.sum(H.profile_uk_gl(k, r_d, rho, n_gl=64))

        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        fd = float((f(PLANCK18.Omega_m + 1e-6) - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)


class TestProfilesAreTraceableInTheHaloRadius:
    """The predecessor's blind spot, exactly.

    ``einasto_uk`` and ``satellite_nfw_uk`` were numpy while sitting on the
    traced path.  They never failed because the only gradient test
    differentiated an amplitude parameter, which does not reach ``r_delta`` or
    ``c``.  Differentiating with respect to ``Omega_m`` does.
    """

    @staticmethod
    def _uk_from_cosmology(om, which):
        cosmo = PLANCK18.replace(Omega_m=om)
        r_d = H.MassDef.from_string("200m").radius(1e13, 0.0, cosmo)
        c = H.c_duffy08(1e13, 0.0, "200m")
        r_s = r_d / c
        k = jnp.asarray(np.logspace(-2, 1, 12))
        if which == "nfw":
            return jnp.sum(H.nfw_uk(k, jnp.atleast_1d(r_s), jnp.atleast_1d(c)))
        if which == "einasto":
            return jnp.sum(H.einasto_uk(k, jnp.atleast_1d(r_s), jnp.atleast_1d(c)))
        if which == "gnfw":
            return jnp.sum(H.gnfw_uk(k, jnp.atleast_1d(r_s), jnp.atleast_1d(r_d)))
        return jnp.sum(H.satellite_uk(k, jnp.atleast_1d(r_d), jnp.atleast_1d(c)))

    @pytest.mark.x64
    @pytest.mark.parametrize("which", ["nfw", "einasto", "gnfw", "satellite"])
    def test_gradient_with_respect_to_omega_m(self, which):
        f = lambda om: self._uk_from_cosmology(om, which)
        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        fd = float((f(PLANCK18.Omega_m + 1e-6) - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0, which
        assert ad == pytest.approx(fd, rel=1e-4), which

    @pytest.mark.parametrize("which", ["nfw", "einasto", "gnfw", "satellite"])
    def test_jit(self, which):
        v = float(jax.jit(lambda om: self._uk_from_cosmology(om, which))(PLANCK18.Omega_m))
        assert np.isfinite(v)


class TestEveryPublicFunctionIsTraceable:
    """No numpy may hide in the halo layer's per-call path."""

    @pytest.mark.parametrize("name", sorted(H.MULTIPLICITY))
    def test_multiplicity(self, name):
        """Both families.  The recalibrated fit takes a cosmology; the sixteen
        others refuse one, which is the distinction
        ``COSMOLOGY_DEPENDENT_MULTIPLICITY`` records."""
        from ggah_mod.cosmology import PLANCK18
        from ggah_mod.halos.mass_function import (
            COSMOLOGY_DEPENDENT_MULTIPLICITY)

        fn = H.make_multiplicity(name)
        kw = {"cosmo": PLANCK18} if name in COSMOLOGY_DEPENDENT_MULTIPLICITY \
            else {}
        g = jax.grad(lambda s: jnp.sum(
            jnp.atleast_1d(fn(jnp.array([s]), 0.0, **kw))))(1.0)
        assert np.isfinite(float(g))

    @pytest.mark.parametrize("name", sorted(H.BIAS))
    def test_bias(self, name):
        fn = H.make_bias(name)
        g = jax.grad(lambda s: jnp.sum(jnp.atleast_1d(fn(jnp.array([s])))))(1.0)
        assert np.isfinite(float(g))

    #: Extra arguments of each peak-height relation, after ``sigma``.  They are
    #: **not** the same quantities and there are not the same number of them:
    #: ``diemer19`` takes the effective slope of sigma(M) (negative) *and* the
    #: growth rate, ``bhattacharya13`` takes the growth factor (positive), and
    #: ``seppi21`` takes the redshift.  Passing the slope where the growth
    #: belongs gives NaN, because the growth is raised to a fractional power --
    #: which is the right answer for nonsense input, and is how this test
    #: caught its own first version.  Passing z where the growth belongs does
    #: not: both are positive and order unity, so that one comes back smooth
    #: and wrong, which is why the dispatch in `field._concentration` names all
    #: three rather than letting one fall through.
    _EXTRA_ARGS = {"diemer19": (-2.0, 0.52), "bhattacharya13": (1.0,),
                   "seppi21": (0.5,)}

    @pytest.mark.parametrize("name", sorted(H.CONCENTRATION))
    def test_concentration(self, name):
        """Two families, two signatures: the empirical fits take a mass, the
        peak-height relations take sigma and one physical second argument."""
        fn = H.make_concentration(name)
        if name in H.concentration.PEAK_HEIGHT_MODELS:
            extra = self._EXTRA_ARGS[name]
            g = jax.grad(lambda s: jnp.sum(jnp.atleast_1d(
                fn(jnp.array([s]), *extra))))(1.0)
        else:
            g = jax.grad(lambda m: jnp.sum(jnp.atleast_1d(fn(jnp.array([m]), 0.0))))(1e13)
        assert np.isfinite(float(g)) and float(g) != 0.0

    def test_peak_height_models_are_registered(self):
        """If a relation is added, it must be classified -- otherwise the test
        above silently calls it with the wrong second argument."""
        assert H.concentration.PEAK_HEIGHT_MODELS <= set(H.CONCENTRATION)
        assert set(self._EXTRA_ARGS) == set(H.concentration.PEAK_HEIGHT_MODELS)

    @pytest.mark.x64
    def test_mass_translation(self):
        """A bare bisection returns a zero gradient, silently."""
        f = lambda om: jnp.log(H.translate_mass(
            1e14, 6.0, "200m", "500c", 0.0, PLANCK18.replace(Omega_m=om))[0])
        ad = float(jax.grad(f)(PLANCK18.Omega_m))
        assert ad != 0.0
        fd = float((f(PLANCK18.Omega_m + 1e-6) - f(PLANCK18.Omega_m - 1e-6)) / 2e-6)
        assert ad == pytest.approx(fd, rel=1e-5)


class TestDeclaredCapabilities:
    def test_csst_is_the_only_non_differentiable_piece(self):
        """And it says so, and a gradient through it raises."""
        assert H.CsstHMF.differentiable is False
        assert H.FittingFunctionHMF.differentiable is True

    def test_lensing_profiles_are_present_and_validated_both_ways(self):
        """Two checks that answer different questions, and the module says so.

        Cross-validated between two representations to 3.0e-7, which rules out
        a transcription slip in one of them; and against an external code at
        1.2e-13, which is the half the first structurally could not do.  The
        assertion keys on both numbers rather than on a status word -- keying
        it on "placeholder" is how this module and ``README.md`` came to say
        opposite things for two tiers.
        """
        import ggah_mod.halos.lensing_profiles as LP
        doc = inspect.getdoc(LP)
        assert "3.0e-7" in doc, "the internal cross-check's number"
        assert "1.2e-13" in doc, "the external reference's number"

    def test_beyond_linear_bias_ships_every_snapshot(self):
        """It landed, and not in the restricted form that was planned.

        "z = 0 only" was a premise, not a finding: 35 snapshots are public, and
        beta^NL is not redshift-universal -- so shipping one of them would be
        the bug, not the caution.
        """
        import ggah_mod.halos.beyond_linear_bias as BNL
        tab = BNL.load()
        assert tab["snap"].shape == (35,)
        assert float(tab["z"][0]) > 2.8 and float(tab["z"][-1]) == 0.0

    def test_beyond_linear_bias_is_not_a_second_exception(self):
        """The layer docstring promises exactly one non-differentiable piece."""
        import ggah_mod.halos.beyond_linear_bias as BNL
        doc = inspect.getdoc(BNL).lower()
        assert "not present" not in doc
        assert not hasattr(BNL, "differentiable")   # nothing to declare

    def test_beyond_linear_bias_rescaling_is_documented_as_derived(self):
        """The k direction is upstream's one genuine ambiguity.

        Three upstream statements of it disagree, so the module must not claim
        to be following any of them -- it is pinned by a test instead, and says
        so where a reader will look.
        """
        import ggah_mod.halos.beyond_linear_bias as BNL
        assert "construction" in inspect.getsource(BNL.table_at)
