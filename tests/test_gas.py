r"""Verification: the DPM gas sector, and the two normalisations it corrects.

The headline test is closed-form and needs no reference data.  **A parameter
set with** :math:`\beta_n = 0` **and** :math:`\beta_P = 2/3` **is self-similar by
construction**, so its temperature at the anchor radius must be a fixed
multiple of the virial temperature at every mass.  It is, to machine precision
-- but only with the mass pivot in physical units.  With the pressure
conversion left as the predecessor had it, the calibrated profile puts a
:math:`10^{15}M_\odot/h` cluster at 47 keV.

The second closed-form check is the unit chain of the gas mass: ``gas_mass``
against a brute-force integral of the sector's own ``n_e``, written out here
in cgs.

No published DPM parameter set is used anywhere in this file.  The parameters
are the sector's defaults, which are calibrated on observations, or a set built
here for a stated property.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology import constants as C
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import gas as G
from ggah_mod.sectors.energetics import v200_squared
from conftest import AnalyticPk, STUB_CM_MODEL

pytestmark = pytest.mark.slow          # the APEC table is read on first use

M = jnp.asarray(np.logspace(12, 15, 7))
Z = 0.0
F_B = PLANCK18.Omega_b / PLANCK18.Omega_m

#: Mean molecular weight of a fully ionised solar-abundance plasma.
MU = 0.6
#: (km/s)^2 -> keV for one proton mass.
_KMS2_TO_KEV = 1e10 / 1.602176634e-9


@pytest.fixture(scope="module")
def gas():
    return G.HotGasDPM()


#: A concentration that does not run with mass, so a self-similar parameter set
#: is self-similar in its gas fraction as well as in its temperature.
C_CONST = 5.0

#: What makes a parameter set self-similar: no mass dependence beyond
#: :math:`n_e \propto M^0` and :math:`P_e \propto M^{2/3}`.  Applied on top of the
#: calibrated defaults, which are not self-similar.
SELF_SIMILAR = dict(beta_n=0.0, beta_p=2.0 / 3.0, alpha_in_n_var=0.0,
                    alpha_tr_n_var=0.0, alpha_out_n_var=0.0,
                    alpha_in_p_var=0.0, alpha_tr_p_var=0.0, alpha_out_var=0.0,
                    log10_conc_ratio_var=0.0)

_FIELDS = {}


def _c(m, z=Z, mdef=None):
    r"""The halo concentration :math:`c(M,z)` a field carries, on the masses ``m``.

    The profiles take it as a required argument, so a call with no field
    behind it reads one rather than inventing a value: one field per redshift
    and definition, on the differentiable flavour's own spectrum and relation,
    interpolated in :math:`\log M` so any grid inside it -- a single mass
    included -- gets the value a spectrum would have passed.
    """
    key = (float(z), mdef)
    if key not in _FIELDS:
        # `calibration="off"` only for a definition the flavour does not
        # declare: the mass function is fitted at 200m and refuses 200c, and
        # nothing here reads it -- only the concentration is taken.
        _FIELDS[key] = make_field(
            PLANCK18, DIFFERENTIABLE, make_pk(DIFFERENTIABLE.pk), z=z,
            mdef=mdef, calibration="strict" if mdef is None else "off")
    f = _FIELDS[key]
    return jnp.power(10.0, jnp.interp(jnp.log10(jnp.asarray(m)),
                                      jnp.log10(f.m), jnp.log10(f.conc)))


def _kt_vir(m, z, cosmo):
    r""":math:`kT_{\rm vir} = \tfrac12\mu m_p v_{200}^2` [keV]."""
    return 0.5 * MU * G.M_PROTON_G * v200_squared(m, z, cosmo) * _KMS2_TO_KEV


def _params(self_similar=False, **over):
    return G.DpmParams(**{**(SELF_SIMILAR if self_similar else {}), **over})


def _kt_at(gas, m, self_similar=False, cosmo=PLANCK18, **over):
    p = _params(self_similar, **over)
    r = (0.3 * gas._r_delta(m, Z, cosmo))[:, None]
    return np.asarray(gas.temperature(r, m, Z, cosmo, p, conc=_c(m)))[:, 0]


class TestSelfSimilarityIsTheAcceptanceTest:
    """beta_n = 0 and beta_P = 2/3 make a parameter set self-similar."""

    def test_kt_over_kt_vir_is_flat_in_mass(self, gas):
        ratio = _kt_at(gas, M, self_similar=True) / np.asarray(
            _kt_vir(M, Z, PLANCK18))
        assert ratio.max() / ratio.min() - 1.0 < 1e-6, (
            f"self-similar by construction; kT/kT_vir spread "
            f"{ratio.max() / ratio.min() - 1:.4f}")

    def test_the_defaults_give_a_cluster_temperature(self, gas):
        """4.1 keV at 0.3 R_Delta of a 1e15 Msun/h halo at z = 0: the defaults
        are calibrated on X-COP, and this is that scale."""
        kt = float(_kt_at(gas, jnp.asarray([1e15]))[0])
        assert 3.0 < kt < 10.0

    def test_the_defaults_are_not_self_similar(self, gas):
        """Or the flatness test above is checking arithmetic, not physics.
        kT/kT_vir varies by a factor 4.7 over 1e12-1e15 at the defaults."""
        ratio = _kt_at(gas, M) / np.asarray(_kt_vir(M, Z, PLANCK18))
        assert ratio.max() / ratio.min() - 1.0 > 0.2


class TestTheTwoNormalisationFixes:
    def test_the_pressure_conversion_is_boltzmann_not_a_micro(self, gas):
        """P_0.3 is published as P/k_B in cm^-3 K.

        Reading it as "meV cm^-3" and multiplying by 1e-6 is a factor
        1e-6 / 8.617333e-8 = 11.605 -- reproduced here, so the fix is a
        measured difference rather than an assertion about the past.
        """
        good = _kt_at(gas, M)
        bad = good * (1e-6 / C.K_B_KEV_PER_K)
        # The ratio is the whole content and is anchor-independent: it is two
        # readings of one number, not a temperature.
        np.testing.assert_allclose(bad / good, 11.605, rtol=1e-4)
        # At 1e15 Msun/h the calibrated defaults read 4.1 keV, and 47 under
        # the wrong constant.
        assert bad[-1] > 40.0                  # absurd, which is the point
        assert good[-1] < 12.0                 # and a plausible cluster

    def test_the_constant_used_is_the_packages_own(self):
        """It was already defined in the predecessor, and never used."""
        assert C.K_B_KEV_PER_K == pytest.approx(8.617333e-8, rel=1e-6)

    def test_the_mass_pivot_makes_the_profile_depend_on_h(self, gas):
        """M_12 = M_200/1e12 Msun with M *physical*.

        Dividing an Msun/h mass by 1e12, as the predecessor did at four sites,
        makes the gas profile independent of h.  With the pivot right, it is
        not -- and this is the assertion that says so.
        """
        p = G.DpmParams()
        # One concentration for both, so the pivot is the only path to h here.
        c = _c(M)
        a = np.asarray(gas.f_gas(M, Z, PLANCK18, p, conc=c))
        b = np.asarray(gas.f_gas(M, Z, PLANCK18.replace(h=0.70), p, conc=c))
        assert np.max(np.abs(a / b - 1.0)) > 1e-3

    def test_the_halo_concentration_feeds_all_three_profiles(self, gas):
        """The predecessor let the density use c(M,z) and the pressure use a
        literal, then divided them for T = P/n_e -- mixing two conventions
        inside one temperature.  All three now take the halo's c(M,z), passed
        in, so there is only one to pass.

        Evaluated at 0.6 R_Delta, **not** at 0.3.  Every profile is normalised
        at 0.3 R_Delta, so the shape ratio there is 1 by construction and the
        temperature is independent of c *whatever* convention is used.
        Testing at the normalisation point would pass with the profiles
        disagreeing everywhere else -- which is exactly the shape the
        predecessor's defect had.
        """
        c, p = _c(M), G.DpmParams()
        r = (0.6 * gas._r_delta(M, Z, PLANCK18))[:, None]
        base = np.asarray(gas.temperature(r, M, Z, PLANCK18, p, conc=c))
        hot = np.asarray(gas.temperature(r, M, Z, PLANCK18, p, conc=0.5 * c))
        assert not np.allclose(base, hot)

        at_norm = (0.3 * gas._r_delta(M, Z, PLANCK18))[:, None]
        np.testing.assert_allclose(
            np.asarray(gas.temperature(at_norm, M, Z, PLANCK18, p, conc=c)),
            np.asarray(gas.temperature(at_norm, M, Z, PLANCK18, p,
                                       conc=0.5 * c)),
            rtol=1e-12)

    def test_the_concentration_is_the_callers_and_never_defaulted(self, gas):
        """A default would be a second convention, which is the defect."""
        r = (0.6 * gas._r_delta(M, Z, PLANCK18))[:, None]
        with pytest.raises(TypeError, match="conc"):
            gas.n_e(r, M, Z, PLANCK18, G.DpmParams())
        assert "c_gas" not in G.DpmParams._PARAMS
        assert not hasattr(G.DpmParams(), "c_gas")

    def test_each_halo_takes_its_own_concentration(self, gas):
        """Row i of an (NM, n) profile is halo i evaluated alone.

        Five radii against seven masses, so a concentration broadcast along the
        radius axis instead of the mass axis fails to broadcast rather than
        passing by a coincidence of shapes.
        """
        c, p = _c(M), G.DpmParams()
        r = (jnp.asarray([0.1, 0.3, 0.6, 1.0, 1.5])[None, :]
             * gas._r_delta(M, Z, PLANCK18)[:, None])
        together = np.asarray(gas.pressure(r, M, Z, PLANCK18, p, conc=c))
        for i in range(M.shape[0]):
            alone = np.asarray(gas.pressure(r[i:i + 1], M[i:i + 1], Z,
                                            PLANCK18, p, conc=c[i:i + 1]))
            np.testing.assert_allclose(together[i], alone[0], rtol=1e-13)

    def test_the_weights_read_the_fields_concentration(self, gas):
        """``weights`` is the path every spectrum takes, so it is checked
        against the field's ``conc`` explicitly -- and against another
        concentration, so the first check cannot pass on an argument that is
        read and not used."""
        field = make_field(PLANCK18, DIFFERENTIABLE, AnalyticPk(), z=0.0,
                           cm_model=STUB_CM_MODEL)
        p = G.DpmParams()
        w = np.asarray(gas.weights(field, p, view="pressure").w_extended)

        def by_hand(conc):
            amp = gas.y_amplitude(field.m, field.z, PLANCK18, p,
                                  mdef=field.mdef, conc=conc)
            uk = gas.pressure_uk(field.k, field.m, field.z, PLANCK18, p,
                                 mdef=field.mdef, conc=conc)
            return np.asarray(amp[None, :] * uk)
        np.testing.assert_allclose(w, by_hand(field.conc), rtol=1e-13,
                                   atol=1e-30)
        assert not np.allclose(w, by_hand(1.3 * field.conc))


class TestTheBaryonBudget:
    def test_a_self_similar_profile_has_a_flat_gas_fraction(self, gas):
        """At a *constant* concentration, because that is what makes a
        self-similar parameter set self-similar in its gas fraction as well as
        in its temperature; the halo's own c(M,z) changes the shape with mass,
        and the test after the next measures by how much."""
        f = np.asarray(gas.f_gas(M, Z, PLANCK18, _params(self_similar=True),
                                 conc=jnp.full_like(M, C_CONST)))
        assert f.max() / f.min() - 1.0 < 1e-6

    @pytest.mark.x64
    def test_the_gas_mass_unit_chain(self, gas):
        r"""``gas_mass`` against :math:`\mu_e m_p \int 4\pi r^2 n_e\,dr` in cgs.

        The check that the chain from cm^-3 through comoving Mpc/h to Msun/h
        is right, with h and (1+z) both in the right place: at z = 0.5 a
        missing (1+z)^3 is a factor 3.4 and a wrong power of h a factor 1.5.
        Agrees to 1.3e-6, the brute-force trapezoid's own error.
        """
        z = 0.5
        m = jnp.asarray([1e13, 1e14, 1e15])
        c = _c(m, z)
        p = G.DpmParams()
        r = (jnp.asarray(np.logspace(-5, 0, 4001))[None, :]
             * gas._r_delta(m, z, PLANCK18)[:, None])
        ne = np.asarray(gas.n_e(r, m, z, PLANCK18, p, conc=c))
        h = PLANCK18.h
        r_cm = np.asarray(r) * C.MPC_CM / h / (1.0 + z)
        brute = (G.MU_E * G.M_PROTON_G / G.M_SUN_G * h
                 * np.trapezoid(4.0 * np.pi * r_cm ** 2 * ne, r_cm, axis=1))
        got = np.asarray(gas.gas_mass(m, z, PLANCK18, p, aperture=1.0, conc=c))
        np.testing.assert_allclose(brute, got, rtol=1e-5)

    def test_the_halo_concentration_makes_the_gas_fraction_run(self, gas):
        r"""The same self-similar set on the halo's :math:`c(M,z)`: no longer
        flat.

        Its temperature at :math:`0.3R_\Delta` is still exactly self-similar
        (the acceptance test), because every profile equals its anchor there.
        Its gas fraction is an integral over the shape, and the shape now
        follows the halo's concentration, which falls with mass.
        """
        f = np.asarray(gas.f_gas(M, Z, PLANCK18, _params(self_similar=True),
                                 conc=_c(M)))
        assert f.max() / f.min() - 1.0 > 1e-2

    def test_the_aperture_is_explicit_and_matters(self, gas):
        """The profile's extent is not a baryon budget: at the defaults a
        1e12 halo holds 3.7 times more gas out to r_max than inside R_Delta."""
        p = G.DpmParams()
        inner = float(gas.f_gas(M, Z, PLANCK18, p, aperture=1.0,
                                conc=_c(M))[0])
        outer = float(gas.f_gas(M, Z, PLANCK18, p,
                                aperture=p.r_max_over_rdelta, conc=_c(M))[0])
        assert inner < outer
        assert outer / inner > 1.3

    def test_nothing_clips_f_gas_to_the_budget(self, gas):
        """A clip would give a plausible number with a dead gradient.  Exceeding
        the budget is information: it says the normalisation and the aperture
        disagree with the cosmology."""
        f = float(gas.f_gas(M, Z, PLANCK18, G.DpmParams(),
                            aperture=3.0, conc=_c(M))[0])
        assert f > F_B

    def test_a_shallow_outer_slope_has_a_divergent_mass_integral(self, gas):
        """alpha_out = 0.5 < 3, inside the box, so int r^2 rho dr does not
        converge and the "gas mass" is whatever the aperture says.  Recorded,
        because a parameter value outside its own convergence condition is
        exactly the thing a bound table exists to surface."""
        p = _params(alpha_out_n=0.5, alpha_out_n_var=0.0,
                    r_max_over_rdelta=1.0)
        a = float(gas.f_gas(M, Z, PLANCK18, p, aperture=1.0, conc=_c(M))[0])
        b = float(gas.f_gas(M, Z, PLANCK18, p, aperture=3.0, conc=_c(M))[0])
        assert b / a > 10.0


class TestTheFourViews:
    VIEWS = ["mass", "density", "pressure", "xray"]

    @pytest.mark.parametrize("view", VIEWS)
    def test_u_tends_to_one_at_large_scales(self, gas, view):
        """By construction in `profile_uk_gl`, so this checks the plumbing:
        that each view reaches it with a positive-definite integrand."""
        fn = {"mass": gas.mass_uk, "density": gas.density_uk,
              "pressure": gas.pressure_uk, "xray": gas.emissivity_uk}[view]
        u0 = np.asarray(fn(jnp.asarray([1e-6]), M, Z, PLANCK18,
                           G.DpmParams(), conc=_c(M)))
        np.testing.assert_allclose(u0[0], 1.0, atol=1e-9)

    @pytest.mark.parametrize("view", VIEWS)
    def test_u_falls_monotonically_with_k(self, gas, view):
        fn = {"mass": gas.mass_uk, "density": gas.density_uk,
              "pressure": gas.pressure_uk, "xray": gas.emissivity_uk}[view]
        k = jnp.asarray(np.logspace(-2, 1, 12))
        u = np.asarray(fn(k, M, Z, PLANCK18, G.DpmParams(),
                          conc=_c(M)))
        assert np.all(np.diff(u, axis=0) < 0), view

    def test_mass_and_density_are_the_same_shape(self, gas):
        """rho_gas = mu_e m_p n_e, and the constant cancels in a normalised
        transform -- so they must be the *same array*, not two that agree."""
        assert type(gas).mass_uk is type(gas).density_uk

    def test_the_views_are_genuinely_different(self, gas):
        """Or "four views from one parameter set" is one view four times."""
        k = jnp.asarray(np.logspace(-1, 0.5, 8))
        p = G.DpmParams()
        a = np.asarray(gas.mass_uk(k, M, Z, PLANCK18, p, conc=_c(M)))
        b = np.asarray(gas.pressure_uk(k, M, Z, PLANCK18, p, conc=_c(M)))
        c = np.asarray(gas.emissivity_uk(k, M, Z, PLANCK18, p, conc=_c(M)))
        assert np.max(np.abs(a - b)) > 0.05
        assert np.max(np.abs(a - c)) > 0.05
        assert np.max(np.abs(b - c)) > 0.05

    def test_weights_are_continuous_and_named(self, gas):
        field = make_field(PLANCK18, DIFFERENTIABLE, AnalyticPk(), z=0.0,
                      cm_model=STUB_CM_MODEL)
        w = gas.weights(field, G.DpmParams(), view="pressure")
        assert w.discrete is False and w.name == "gas:pressure"
        assert w.w_point is None
        assert w.w_extended.shape == (field.n_k, field.n_m)

    def test_an_unknown_view_raises(self, gas):
        field = make_field(PLANCK18, DIFFERENTIABLE, AnalyticPk(), z=0.0,
                      cm_model=STUB_CM_MODEL)
        with pytest.raises(ValueError, match="unknown gas view"):
            gas.weights(field, G.DpmParams(), view="nope")


class TestNothingIsFixedByHardCoding:
    def test_every_parameter_the_predecessor_froze_is_free(self):
        """The DPM requirement, as a list."""
        frozen = ["gamma_n", "gamma_p", "alpha_out_var", "sigma_scatter",
                  "nh_over_ne", "z_anchor", "alpha_in_z", "alpha_tr_z",
                  "alpha_out_z"]
        for name in frozen:
            assert name in G.DpmParams._PARAMS, name

    def test_every_parameter_carries_a_reason(self):
        for name, p in G.DpmParams._PARAMS.items():
            assert p.why and len(p.why) > 20, name

    def test_the_shape_function_is_not_redefined_here(self):
        """Six copies in the predecessor.  This module must import the one
        layer 2 owns rather than add a seventh."""
        import inspect
        src = inspect.getsource(G)
        assert "from ..halos.profiles import" in src
        assert "def gnfw_shape" not in src

    def test_no_concentration_is_written_into_the_module(self):
        """The predecessor had `c_DPM = 2.772` at seven independent sites, so
        changing it moved some profiles and not others.  The sector now has no
        concentration of its own at all: it reads the halo's.

        Structural rather than textual: neither the published value nor the
        4.5877 it became at 200m may appear as a number anywhere in the
        module's code -- prose describing them is a string, not a constant, and
        is allowed.
        """
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(G))
        found = [child.lineno for child in ast.walk(tree)
                 if isinstance(child, ast.Constant)
                 and isinstance(child.value, float)
                 and child.value in (2.772, 4.5877)]
        assert not found, f"a concentration is written in at lines {found}"
        assert "c_gas" not in G.DpmParams._PARAMS


class TestDifferentiability:
    @pytest.mark.x64
    @pytest.mark.parametrize("key", ["log10_ne_anchor", "beta_n", "alpha_in_n",
                                     "alpha_out_n", "log10_pe_anchor",
                                     "beta_p", "alpha_out_var", "z_anchor",
                                     "nh_over_ne", "gamma_n", "gamma_p"])
    def test_gradient_through_the_x_ray_luminosity(self, gas, key):
        """At z = 0.3, not z = 0.

        `gamma_n` and `gamma_p` enter only as `E(z)^gamma`, and E(0) = 1
        identically in this package -- so at z = 0 their derivative is
        `ln(E) E^gamma = 0` exactly.  A correct structural zero, and testing
        there would assert the wrong thing about a right answer.
        """
        p = G.DpmParams()
        z = 0.3

        c = _c(M, z)

        def f(v):
            return jnp.sum(gas.x_ray_luminosity(M, z, PLANCK18,
                                                p.replace(**{key: v}), conc=c))
        x = float(getattr(p, key))
        h = 1e-6 * max(abs(x), 1.0)
        ad = float(jax.grad(f)(x))
        fd = float((f(x + h) - f(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0, key
        assert ad == pytest.approx(fd, rel=1e-4), key

    @pytest.mark.x64
    def test_gradient_through_the_concentration(self, gas):
        """The path by which the cosmology now reaches the profile *shape*:
        c(M,z) is a function of sigma(M) and the growth, so a spectrum's
        derivative runs through it.  Checked on a common rescaling of it."""
        p, z = G.DpmParams(), 0.3
        c = _c(M, z)
        f = lambda s: jnp.sum(gas.x_ray_luminosity(M, z, PLANCK18, p,
                                                    conc=s * c))
        ad = float(jax.grad(f)(1.0))
        fd = float((f(1.0 + 1e-6) - f(1.0 - 1e-6)) / 2e-6)
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    @pytest.mark.x64
    def test_the_redshift_exponents_are_unconstrained_at_z_zero(self, gas):
        """The companion to the test above, so its z = 0.3 is not read as a
        tolerance dodge: E(0) = 1 makes these exactly flat."""
        p = G.DpmParams()
        for key in ("gamma_n", "gamma_p"):
            f = lambda v, k=key: jnp.sum(gas.x_ray_luminosity(
                M, 0.0, PLANCK18, p.replace(**{k: v}), conc=_c(M)))
            assert float(jax.grad(f)(float(getattr(p, key)))) == 0.0

    def test_the_scatter_boost_is_differentiable_but_stationary_at_zero(self, gas):
        """exp[(sigma ln10)^2] is *even* in sigma, so its derivative at the
        fiducial sigma = 0 is exactly 0 -- a stationary point, not a lost
        gradient.  The distinction matters: the predecessor computed this with
        `float(np.exp(...))`, which is zero everywhere.
        """
        p = G.DpmParams()
        f = lambda v: jnp.sum(gas.x_ray_luminosity(
            M, Z, PLANCK18, p.replace(sigma_scatter=v), conc=_c(M)))
        assert float(jax.grad(f)(0.0)) == 0.0
        assert float(jax.grad(f)(0.3)) > 0.0        # live away from it

    @pytest.mark.x64
    def test_f_gas_reaches_the_cosmology_through_h(self, gas):
        p = G.DpmParams()
        c = _c(M)
        f = lambda v: jnp.sum(gas.f_gas(M, Z, PLANCK18.replace(h=v), p,
                                        conc=c))
        ad = float(jax.grad(f)(PLANCK18.h))
        fd = float((f(PLANCK18.h + 1e-6) - f(PLANCK18.h - 1e-6)) / 2e-6)
        assert ad != 0.0 and ad == pytest.approx(fd, rel=1e-4)

    def test_the_sector_declares_itself_differentiable(self, gas):
        assert G.HotGasDPM.differentiable is True
        assert G.HotGasDPM.name == "gas"


class TestTheAmplitudeIsShared:
    r"""Scaling both normalisations, which is what makes the coupling safe.

    :mod:`~ggah_mod.sectors.energetics` refuses to tie the gas amplitude to an
    energy-predicted :math:`f_{\rm gas}`, and gives a reason: rescaling
    :math:`n_{e,0.3}` alone moves every halo's temperature as :math:`1/A`,
    because :math:`kT = P_e/n_e` and :math:`P_e` does not depend on the density
    normalisation, de-calibrating :math:`kT`--:math:`M` and making the X-ray
    emissivity scale as :math:`A^{2-p}` rather than :math:`A^2`.

    That is an argument against moving **one** amplitude.  These tests are the
    other half: shifting both by the same :math:`\log A` leaves :math:`kT`
    exactly invariant and the emissivity exactly :math:`A^2`, so the refusal
    becomes a rule about *how* to couple rather than whether to.
    """

    @staticmethod
    def _setup():
        g = G.HotGasDPM(backend=DIFFERENTIABLE)
        p = G.DpmParams()
        m = jnp.asarray(np.logspace(13.0, 15.0, 6))
        z = 0.25
        r = (jnp.asarray(np.logspace(-2, 0.3, 40))[None, :]
             * g._r_delta(m, z, PLANCK18, None)[:, None])
        return (g, p, m, z, r, jnp.asarray(np.linspace(0.3, 2.0, 6)),
                _c(m, z))

    def test_none_is_the_shipped_number_bit_for_bit(self):
        """A log shift of exactly zero, not a multiplication by almost one."""
        g, p, m, z, r, _, c = self._setup()
        assert np.array_equal(
            np.asarray(g.n_e(r, m, z, PLANCK18, p, conc=c, amplitude=None)),
            np.asarray(g.n_e(r, m, z, PLANCK18, p, conc=c)))

    def test_the_temperature_is_invariant(self):
        """The answer to the objection `energetics` records."""
        g, p, m, z, r, a, c = self._setup()
        kt0 = np.asarray(g.temperature(r, m, z, PLANCK18, p, conc=c))
        kt1 = np.asarray(g.temperature(r, m, z, PLANCK18, p, conc=c,
                                       amplitude=a))
        assert np.max(np.abs(kt1 / kt0 - 1.0)) < 1e-13

    def test_the_density_scales_and_the_emissivity_squares(self):
        g, p, m, z, r, a, c = self._setup()
        n0 = np.asarray(g.n_e(r, m, z, PLANCK18, p, conc=c))
        n1 = np.asarray(g.n_e(r, m, z, PLANCK18, p, conc=c, amplitude=a))
        assert np.allclose(n1 / n0, np.asarray(a)[:, None], rtol=1e-13)
        e0 = np.asarray(g.emissivity(r, m, z, PLANCK18, p, conc=c))
        e1 = np.asarray(g.emissivity(r, m, z, PLANCK18, p, conc=c,
                                     amplitude=a))
        assert np.allclose(e1 / e0, np.asarray(a)[:, None] ** 2, rtol=1e-12)

    def test_the_gas_fraction_scales_by_it(self):
        r"""Which is what lets :math:`A = f^{\rm budget}/f^{\rm profile}` close the loop.

        The gas sector's own ``f_gas`` is a profile integral and the energy
        closure predicts a budget; today their ratio is computed and reported.
        Because ``f_gas`` is exactly linear in the amplitude, setting :math:`A`
        to that ratio makes the two agree by construction rather than by a fit.
        """
        g, p, m, z, _, a, c = self._setup()
        f0 = np.asarray(g.f_gas(m, z, PLANCK18, p, conc=c))
        f1 = np.asarray(g.f_gas(m, z, PLANCK18, p, conc=c, amplitude=a))
        assert np.allclose(f1 / f0, np.asarray(a), rtol=1e-13)

    def test_the_amplitudes_move_by_it(self):
        """Y by A and L_X by A squared, as the budget requires.

        ``y_amplitude`` used to accept the amplitude and not pass it to the
        pressure, so under ``feedback="closure"`` the tSZ amplitude did not
        move while the gas mass did.  Its mass definition was dropped the same
        way, harmless only while every flavour declares 200m.
        """
        g, p, m, z, _, a, c = self._setup()
        y0 = np.asarray(g.y_amplitude(m, z, PLANCK18, p, conc=c))
        y1 = np.asarray(g.y_amplitude(m, z, PLANCK18, p, conc=c, amplitude=a))
        assert np.allclose(y1 / y0, np.asarray(a), rtol=1e-12)
        l0 = np.asarray(g.x_ray_luminosity(m, z, PLANCK18, p, conc=c))
        l1 = np.asarray(g.x_ray_luminosity(m, z, PLANCK18, p, conc=c,
                                           amplitude=a))
        assert np.allclose(l1 / l0, np.asarray(a) ** 2, rtol=1e-12)
        for f in (g.y_amplitude, g.x_ray_luminosity):
            # Each definition with its own concentration, as a field would
            # hand them: a 200m c on a 200c radius is two conventions again.
            m_def = np.asarray(f(m, z, PLANCK18, p, mdef="200c",
                                 conc=_c(m, z, "200c")))
            m_def_m = np.asarray(f(m, z, PLANCK18, p, mdef="200m", conc=c))
            assert np.max(np.abs(m_def / m_def_m - 1.0)) > 1e-3, (
                f"{f.__name__} does not read its mass definition")

    @pytest.mark.parametrize("view", ["mass_uk", "pressure_uk", "emissivity_uk"])
    def test_the_transforms_do_not_move(self, view):
        r"""The claim that the coupling cannot reshape a spectrum.

        :func:`~ggah_mod.halos.profiles.profile_uk_gl` normalises every
        transform by its own :math:`k\to0` integral, and :math:`A` is constant
        in :math:`r` at fixed :math:`M`, so it cancels.  Only the amplitudes --
        ``gas_mass``, ``y_amplitude``, ``x_ray_luminosity`` -- move, which is
        where a baryon budget belongs.  It is threaded into these three
        deliberately so the claim is measured rather than assumed.
        """
        g, p, m, z, _, a, c = self._setup()
        k = jnp.asarray(np.logspace(-2, 1, 12))
        f = getattr(g, view)
        u0 = np.asarray(f(k, m, z, PLANCK18, p, conc=c))
        u1 = np.asarray(f(k, m, z, PLANCK18, p, conc=c, amplitude=a))
        # Of the peak: these transforms cross zero.
        assert np.max(np.abs(u1 - u0)) / np.max(np.abs(u0)) < 1e-12

    def test_a_budget_proportional_to_the_profile_is_the_identity(self):
        """Exactly, which is what makes it safe to land switched off."""
        g, p, m, z, _, _, _ = self._setup()
        mm = jnp.asarray(np.logspace(12.0, 15.5, 40))
        cc = _c(mm, z)
        f_prof = g.f_gas(mm, z, PLANCK18, p, conc=cc)
        a = np.asarray(g.amplitude_from_budget(mm, z, PLANCK18, p,
                                               0.37 * f_prof, conc=cc))
        assert np.max(np.abs(a - 1.0)) < 1e-14

    def test_the_energy_supplies_the_shape_and_the_anchor_keeps_the_level(self):
        r"""What the pivot normalisation is for, measured.

        Applying :math:`A(M)` makes the profile's gas fraction track the
        budget's **mass dependence** exactly -- their ratio becomes constant --
        while the constant itself is still whatever the fitted anchor says.
        That is the arrangement that keeps :math:`\epsilon_{\rm AGN}` and
        :math:`\epsilon_{\rm SN}` identifiable: against a free overall
        amplitude they would be degenerate with it, against a shape they are
        not.

        Checked here with the FLAMINGO-calibrated sigmoid standing in for an
        energy closure, because the two enter `BaryonSplit` the same way.  Its
        ratio to the profile runs 1.03, 0.55, 1.06 across three decades before
        and is flat after.
        """
        from ggah_mod.sectors import energetics as E
        g, p, _, z, _, _, _ = self._setup()
        mm = jnp.asarray(np.logspace(12.0, 15.5, 40))
        f_b = float(PLANCK18.Omega_b / PLANCK18.Omega_m)
        budget = E.f_gas_sigmoid(jnp.log10(mm), f_b)
        cc = _c(mm, z)
        a = g.amplitude_from_budget(mm, z, PLANCK18, p, budget, conc=cc)
        after = (np.asarray(g.f_gas(mm, z, PLANCK18, p, conc=cc, amplitude=a))
                 / np.asarray(budget))
        before = (np.asarray(g.f_gas(mm, z, PLANCK18, p, conc=cc))
                  / np.asarray(budget))
        assert np.ptp(after) / np.mean(after) < 1e-10, \
            "after the coupling the profile must follow the budget's shape"
        assert np.ptp(before) / np.mean(before) > 0.5, \
            "and before it, the two disagreed about the mass trend"

    def test_the_pivot_is_where_the_anchor_still_means_what_it_meant(self):
        from ggah_mod.sectors import energetics as E
        g, p, _, z, _, _, _ = self._setup()
        mm = jnp.asarray(np.logspace(12.0, 15.5, 40))
        f_b = float(PLANCK18.Omega_b / PLANCK18.Omega_m)
        a = np.asarray(g.amplitude_from_budget(
            mm, z, PLANCK18, p, E.f_gas_sigmoid(jnp.log10(mm), f_b),
            conc=_c(mm, z), m_pivot=1e14))
        assert float(np.interp(14.0, np.log10(np.asarray(mm)), a)) == \
            pytest.approx(1.0, abs=1e-3)


class TestTheClosureReachesTheAmplitudes:
    r"""The energy budget wired to the gas profile, which had never been.

    :mod:`~ggah_mod.sectors.energetics` has carried a complete, parameterised,
    gradient-safe closure with **no production caller**: ``f_retained_energy``
    was invoked by one test and nothing else, while the live path used
    ``BaryonSplit.from_hot`` and an energy-free gas fraction.  This is the
    wiring, and the mode that keeps it optional.
    """

    @pytest.fixture(scope="class")
    def field(self):
        return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=0.0)

    @staticmethod
    def _sectors():
        from ggah_mod.sectors import agn as A
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        gal = GalaxySector("zumandelbaum15")
        gp = galaxy_defaults("zumandelbaum15")
        agn = A.AgnSector(gal, calibration="off")
        # The published AGN parameters (the defaults up to 0.8.6): at the 0.8.7
        # defaults the closure keeps 0.32 of the gas at 1e12 rather than < 0.3.
        return agn, gal, A.AgnParams(**A.AGN_PUBLISHED), gp

    def test_the_default_asserts_no_coupling(self):
        assert G.HotGasDPM().feedback == "none"

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="feedback mode"):
            G.HotGasDPM(feedback="sort-of")

    def test_a_closure_without_its_sectors_is_refused(self, field):
        """Parameters are not enough: a budget is built from masses."""
        _, _, ap, gp = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure")
        with pytest.raises(ValueError, match="sectors"):
            g.weights(field, G.DpmParams(), agn_params=ap,
                      galaxies_params=gp)

    def test_a_closure_without_its_parameters_is_refused(self, field):
        agn, gal, _, _ = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        with pytest.raises(ValueError, match="AGN and the galaxy"):
            g.weights(field, G.DpmParams())

    def test_a_gas_spectrum_still_needs_no_peers_at_all(self, field):
        """The case this package exists to make possible, kept.

        Requiring peers here would buy the coupling by taking away a gas
        spectrum with no galaxy sector in it -- which is why they are in
        ``OPTIONAL_PEERS`` and not ``PEERS``.
        """
        g = G.HotGasDPM(backend=DIFFERENTIABLE)
        w = g.weights(field, G.DpmParams(), view="pressure")
        assert np.all(np.isfinite(np.asarray(w.w_extended)))

    def test_the_budget_counts_every_star_whatever_the_selection(self, field):
        """0.9.4.  A galaxy x tSZ fit hands the budget its sample's parameters,
        threshold included; the budget used to count only the selected
        galaxies' stars, and at M* > 10^11 the closure amplitude at 1e12 came
        out 2.2 times high.  Now one budget, bit for bit, at any threshold."""
        agn, gal, ap, gp = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        p = G.DpmParams()
        ref = np.asarray(g.feedback_budget(
            field, p, ap, dict(gp, log10m_star_thresh=8.5)))
        for thr in (9.0, 10.157, 10.66, 11.16, 11.9):
            b = np.asarray(g.feedback_budget(
                field, p, ap, dict(gp, log10m_star_thresh=thr)))
            np.testing.assert_array_equal(b, ref)

    @staticmethod
    def _cold(gal):
        from ggah_mod.sectors.coldgas import ColdGasParams, ColdGasSector
        return ColdGasSector("catinella18", galaxies=gal), ColdGasParams()

    def test_without_the_neutral_gas_the_closure_is_094(self, field):
        """``m_cold=None`` and no cold-gas peer leave the arithmetic as it was,
        and a sector built without the peer ignores cold-gas parameters."""
        from ggah_mod.sectors import energetics as E
        agn, gal, ap, gp = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        p = G.DpmParams()
        _, cp = self._cold(gal)
        a = np.asarray(g.feedback_budget(field, p, ap, gp))
        b = np.asarray(g.feedback_budget(field, p, ap, gp, cp))
        np.testing.assert_array_equal(a, b)
        m = np.asarray(field.m)
        ms = np.full_like(m, 0.01) * m
        f1 = np.asarray(E.f_retained_energy(m, 0.0, PLANCK18, ms, 0.157,
                                            m_bh=1e-4 * m))
        f2 = np.asarray(E.f_retained_energy(m, 0.0, PLANCK18, ms, 0.157,
                                            m_bh=1e-4 * m, m_cold=None))
        np.testing.assert_array_equal(f1, f2)

    @pytest.mark.parametrize("agn_defaults", [True, False])
    def test_the_hot_gas_is_never_negative_with_the_neutral_gas_in(
            self, field, agn_defaults):
        r"""0.9.5.  The closure cannot expel the neutral gas either.  Counted as
        expellable, the hot gas it left was negative where the closure keeps
        less than stars + neutral gas -- -0.06 of f_b at 1e10.6-1e11.3 Msun/h
        with ``catinella18`` at the LS10 defaults, z = 0.  With the cold-gas
        peer the budget is the hot gas and is at least ``f_retained_min``
        everywhere on the grid, 1e10-1e16."""
        from ggah_mod.sectors import agn as A
        from ggah_mod.sectors.energetics import EnergeticsParams
        agn, gal, ap, gp = self._sectors()
        if agn_defaults:
            ap = A.AgnParams()
        cold, cp = self._cold(gal)
        with_hi = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                              agn=agn, galaxies=gal, coldgas=cold)
        hot = np.asarray(with_hi.feedback_budget(field, G.DpmParams(),
                                                 ap, gp, cp))
        assert hot.min() >= EnergeticsParams().f_retained_min * (1 - 1e-9)
        assert np.all(np.isfinite(hot))

    def test_the_old_arithmetic_was_negative_where_it_was_said_to_be(
            self, field):
        """What the floor is for, pinned: 0.9.4's budget minus the neutral gas
        goes below zero, at 1e10.6-1e11.3, at the LS10 defaults."""
        from ggah_mod.sectors import agn as A
        agn, gal, _, gp = self._sectors()
        cold, cp = self._cold(gal)
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        old = np.asarray(g.feedback_budget(field, G.DpmParams(),
                                           A.AgnParams(), gp))
        f_cold = np.asarray(cold.f_cold(field, cp, G._every_star(gp)))
        lg = np.log10(np.asarray(field.m))
        neg = (old - f_cold) < 0
        assert neg.any()
        assert 10.3 < lg[neg].min() and lg[neg].max() < 11.6

    def test_a_cold_gas_peer_without_its_parameters_is_refused(self, field):
        agn, gal, ap, gp = self._sectors()
        cold, _ = self._cold(gal)
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal, coldgas=cold)
        with pytest.raises(ValueError, match="cold-gas parameters"):
            g.feedback_budget(field, G.DpmParams(), ap, gp)

    def test_every_star_puts_the_threshold_at_its_floor(self):
        import dataclasses

        from ggah_mod.sectors.galaxies import GalaxyParams
        floor = GalaxyParams._PARAMS["log10m_star_thresh"].bounds[0]
        assert floor == 8.5
        d = G._every_star({"log10m_star_thresh": 11.2, "lg_m1h": 12.0})
        assert d == {"log10m_star_thresh": 8.5, "lg_m1h": 12.0}
        dc = G._every_star(dataclasses.replace(GalaxyParams(),
                                               log10m_star_thresh=11.2))
        assert dc.log10m_star_thresh == 8.5
        other = {"log10mmin": 12.0}
        assert G._every_star(other) is other

    def test_the_budget_falls_with_halo_mass_the_way_feedback_does(self, field):
        r"""A cluster keeps nearly its cosmic share of gas; a group does not.

        Against the non-stellar share, :math:`f_b - f_\star`, because the
        budget is gas: the stars are subtracted, and they cannot be expelled."""
        from ggah_mod.sectors.matter import cosmic_baryon_fraction
        agn, gal, ap, gp = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        fb = np.asarray(g.feedback_budget(field, G.DpmParams(), ap, gp))
        f_cen, f_sat = gal.stellar_fraction(field, gp, satellites=True)
        gas_share = (float(cosmic_baryon_fraction(PLANCK18))
                     - np.asarray(f_cen + f_sat))
        lg = np.log10(np.asarray(field.m))
        at = lambda x: float(np.interp(x, lg, fb / gas_share))
        assert at(12.0) < at(13.0) < at(14.0) < at(15.0)
        assert at(15.0) > 0.9, "feedback cannot unbind a cluster"
        assert at(12.0) < 0.3, "and it empties a small halo"

    def test_the_budget_is_gas_and_never_negative(self, field):
        """It used to be the retained baryons, stars included, handed on as
        gas.  Subtracting the stars after the fact went negative on 14 of 256
        nodes; with the closure capped at the non-stellar baryons it stays at
        or above the retained floor."""
        from ggah_mod.sectors.energetics import EnergeticsParams
        agn, gal, ap, gp = self._sectors()
        g = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                        agn=agn, galaxies=gal)
        fb = np.asarray(g.feedback_budget(field, G.DpmParams(), ap, gp))
        floor = EnergeticsParams().f_retained_min
        assert np.all(fb > floor * (1.0 - 1e-9))

    def test_the_satellites_are_in_the_supernova_budget(self, field):
        r"""Every star in the halo, not only the central's.

        There is no duty-cycle analogue and that is where the parallel with the
        AGN stops: an AGN is active or it is not, while every satellite's stars
        have already exploded.  ``omega_bh`` sets the precedent by counting
        every satellite black hole whatever ``f_duty_sat`` is.
        """
        import inspect
        src = inspect.getsource(G.HotGasDPM.feedback_budget)
        assert "satellites=True" in src
        agn, gal, _, gp = self._sectors()
        f_cen, f_sat = gal.stellar_fraction(field, gp, satellites=True)
        assert float(jnp.max(f_sat)) > 0.0, "the satellite term must be real"

    def test_switching_it_on_moves_the_budget_and_not_the_shape(self, field):
        r"""The whole claim, end to end.

        ``gas_mass`` moves because a baryon budget is what the energy sets;
        ``pressure_uk`` does not, because :math:`A` is constant in :math:`r` at
        fixed :math:`M` and the transform is normalised by its own
        :math:`k\to0` integral.
        """
        agn, gal, ap, gp = self._sectors()
        p = G.DpmParams()
        off = G.HotGasDPM(backend=DIFFERENTIABLE)
        on = G.HotGasDPM(backend=DIFFERENTIABLE, feedback="closure",
                         agn=agn, galaxies=gal)
        a = on._feedback_amplitude(field, p, ap, gp)
        assert np.ptp(np.asarray(a)) > 0.1, "the coupling must do something"
        u0 = np.asarray(off.pressure_uk(field.k, field.m, field.z, PLANCK18, p,
                                        mdef=field.mdef, conc=field.conc))
        u1 = np.asarray(on.pressure_uk(field.k, field.m, field.z, PLANCK18, p,
                                       mdef=field.mdef, conc=field.conc,
                                       amplitude=a))
        # Against the peak, not point by point: u(k) crosses zero, where a
        # ratio divides two roundoffs and says nothing about agreement.
        assert np.max(np.abs(u1 - u0)) / np.max(np.abs(u0)) < 1e-12
        m0 = np.asarray(off.gas_mass(field.m, field.z, PLANCK18, p,
                                     mdef=field.mdef, conc=field.conc))
        m1 = np.asarray(on.gas_mass(field.m, field.z, PLANCK18, p,
                                    mdef=field.mdef, conc=field.conc,
                                    amplitude=a))
        assert np.max(np.abs(m1 / m0 - 1.0)) > 0.1


class TestTheIsobaricScatter:
    r"""``scatter="isobaric"``: the published DPM's log-normal of density at
    the local pressure, against the shipped constant boost.

    Three identities pin it.  At :math:`\sigma = 0` there is nothing to
    scatter, so it is the unscattered emissivity.  With :math:`\Lambda`
    constant the phase sum is :math:`\bar n^2 e^{s^2}`, which is exactly the
    ``"constant"`` mode's boost.  And it moves only the X-ray: pressure and
    density -- so :math:`y` and the gas mass -- never see it.
    """

    @staticmethod
    def _grid():
        m = jnp.asarray([1e12, 1e13, 1e14])
        r = jnp.outer(jnp.ones(3), jnp.logspace(-2, 0.2, 40))
        return m, r

    def test_zero_scatter_is_the_unscattered_emissivity(self, gas):
        m, r = self._grid()
        p = G.DpmParams(sigma_scatter=0.0)
        iso = G.HotGasDPM(scatter="isobaric")
        np.testing.assert_allclose(
            np.asarray(iso.emissivity(r, m, Z, PLANCK18, p, conc=_c(m))),
            np.asarray(gas.emissivity(r, m, Z, PLANCK18, p, conc=_c(m))),
            rtol=1e-13)

    def test_a_constant_lambda_reduces_it_to_the_boost(self):
        flat = lambda kt, z, nh_over_ne=None: 1e-23 * jnp.ones_like(kt * z)
        m, r = self._grid()
        p = G.DpmParams(sigma_scatter=0.3)
        a = G.HotGasDPM(cooling=flat).emissivity(r, m, Z, PLANCK18, p, conc=_c(m))
        b = G.HotGasDPM(cooling=flat, scatter="isobaric").emissivity(
            r, m, Z, PLANCK18, p, conc=_c(m))
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-9)

    def test_it_moves_the_x_ray_and_nothing_else(self, gas):
        m, r = self._grid()
        p = G.DpmParams(sigma_scatter=0.15)
        iso = G.HotGasDPM(scatter="isobaric")
        for view in ("pressure", "n_e"):
            np.testing.assert_array_equal(
                np.asarray(getattr(iso, view)(r, m, Z, PLANCK18, p, conc=_c(m))),
                np.asarray(getattr(gas, view)(r, m, Z, PLANCK18, p, conc=_c(m))))
        ratio = (np.asarray(iso.emissivity(r, m, Z, PLANCK18, p, conc=_c(m)))
                 / np.asarray(gas.emissivity(r, m, Z, PLANCK18, p, conc=_c(m))))
        assert np.all(np.isfinite(ratio)) and np.ptp(ratio) > 0.05

    def test_it_differentiates_in_sigma(self):
        m, r = self._grid()
        iso = G.HotGasDPM(scatter="isobaric")
        f = lambda s: jnp.sum(iso.emissivity(
            r, m, Z, PLANCK18, G.DpmParams(sigma_scatter=s), conc=_c(m)))
        assert bool(jnp.isfinite(jax.grad(f)(0.15)))

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="scatter mode"):
            G.HotGasDPM(scatter="clumpy")
