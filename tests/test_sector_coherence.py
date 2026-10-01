r"""Verification: the sectors are peers, and the graph between them is a DAG.

Layer 3's central claim is that **no sector is the host of the others**.  In the
predecessor it was false -- galaxy occupation owned the mass grid, and the gas,
AGN and cluster legs existed only by wrapping a galaxy predictor and reading its
privates, so a pure pressure auto-spectrum still had to be handed an HOD
parameter dict.

The claim is checkable, and this file checks it three ways:

1. **A pure gas spectrum needs no galaxy parameters.**  Run it.
2. **No sector owns a mass grid.**  Structural: none of them builds a
   ``logspace``.
3. **The dependency graph is acyclic**, and the four places a cycle could form
   are each closed by a naming or an ownership rule.
"""
import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import ggah_mod.sectors as S
from ggah_mod.sectors.galaxies import GalaxySector as _GS, galaxy_defaults as _gd
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18
from ggah_mod.cosmology.power import make_pk
from ggah_mod.halos.field import make_field
from ggah_mod.sectors import agn as A
from ggah_mod.sectors import energetics as E
from ggah_mod.sectors import gas as G
from ggah_mod.sectors import matter as MT
from ggah_mod.sectors import occupation as O
from ggah_mod.sectors import sham as SH

Z = 0.135


@pytest.fixture(scope="module")
def field():
    return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=Z)


#: The AGN chain reads a threshold occupation's stellar masses.
_GAL = _GS("zumandelbaum15")
_GP = _gd("zumandelbaum15")


class TestTheSectorsArePeers:
    def test_a_gas_spectrum_needs_no_galaxy_parameters(self, field):
        """The predecessor could not do this.  It is the whole point."""
        gas = G.HotGasDPM()
        w = gas.weights(field, G.DpmParams(), view="pressure")
        assert np.all(np.isfinite(np.asarray(w.w_extended)))
        assert float(jnp.max(jnp.abs(w.w_extended))) > 0.0

    def test_an_agn_spectrum_needs_the_galaxy_sector(self, field):
        """The one sector that is **not** a peer of the galaxies.  The AGN
        chain starts from a stellar mass, and the galaxy occupation is where
        one lives -- so an AGN sector without one is refused rather than handed
        a private stellar-mass relation nobody chose."""
        with pytest.raises(ValueError, match="needs a galaxy sector"):
            A.AgnSector()
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults
        w = A.AgnSector(GalaxySector("zumandelbaum15")).weights(
            field, A.AgnParams(), galaxy_defaults("zumandelbaum15"))
        assert np.all(np.isfinite(np.asarray(w.w_point)))

    def test_no_sector_builds_a_mass_grid(self):
        """Five classes did in the predecessor, with three different sizes.

        A sector that builds one is a sector that has stopped taking the halo
        field's -- which is how the mass range silently stopped agreeing
        between the legs of one analysis.
        """
        import ggah_mod.sectors as pkg
        offenders = []
        for mod in pkgutil.iter_modules(pkg.__path__):
            src = inspect.getsource(
                importlib.import_module(f"ggah_mod.sectors.{mod.name}"))
            for pattern in ("logspace(10", "logspace(11", "logspace(12"):
                if pattern in src:
                    offenders.append(f"{mod.name}: {pattern}")
        assert not offenders, f"a sector owns a mass grid: {offenders}"

    def test_every_sector_takes_a_halo_field(self):
        for cls in (G.HotGasDPM, A.AgnSector, MT.MatterField):
            params = list(inspect.signature(cls.weights).parameters)
            assert params[1] == "field", cls.__name__


class TestTheGraphIsADag:
    """`field -> occupation -> agn -> energetics -> gas -> matter`."""

    def test_the_chain_evaluates_in_order(self, field):
        lm = jnp.log10(field.m)
        # occupation -> M_*
        m_star = jnp.power(10.0, _GAL.log10_mstar(lm, _GP, h=field.cosmo.h))
        # agn -> L_X, from the occupation's own stellar masses
        l_x = A.AgnSector(_GAL).l_x_agn(field, A.AgnParams(), _GP)
        # energetics -> f_retained (BARYONS, stars included)
        f_b = PLANCK18.Omega_b / PLANCK18.Omega_m
        m_bh = A.AgnSector(_GAL).mean_mbh(lm, A.AgnParams(), _GP,
                                         h=field.cosmo.h)
        f_ret = E.f_retained_energy(field.m, Z, PLANCK18, m_star, f_b,
                                    m_bh=m_bh)
        # matter -> W_m.  `from_retained`, because that is what the closure
        # predicts: it balances the energy to displace baryons, so the stars
        # are already inside `f_ret` and adding them again is the double count
        # this constructor pair exists to make unsayable.
        split = MT.BaryonSplit.from_retained(f_b, f_ret,
                                             f_star_cen=m_star / field.m)
        w = MT.matter_weights(field, split)
        res = np.abs(np.asarray(
            MT.MatterField.mass_conservation_residual(field, w)))
        assert np.max(res) < 1e-6
        assert np.all(np.asarray(f_ret) > 0.0)
        assert np.max(np.abs(np.asarray(split.residual()))) < 5e-16

    def test_the_agn_and_gas_luminosities_are_named_apart(self):
        """Wiring the *gas* luminosity into the feedback budget would close
        `gas -> energetics -> gas`.  Neither is called `lx`, so the mistake
        cannot be made by autocomplete."""
        assert hasattr(A.AgnSector, "l_x_agn")
        assert hasattr(G.HotGasDPM, "x_ray_luminosity")
        for cls in (A.AgnSector, G.HotGasDPM):
            assert not hasattr(cls, "lx")

    def test_the_energy_closure_takes_the_black_hole_mass_explicitly(self):
        """It is an *argument*, not something the closure fetches: a sector
        that reached out to another would make the order implicit.

        And it takes no luminosity at all since 0.8.8: the luminosity channel
        is gone, so neither `l_x_agn` nor a channel switch may come back as a
        parameter nothing would read."""
        params = list(inspect.signature(E.f_retained_energy).parameters)
        assert "m_star" in params and "m_bh" in params
        for gone in ("l_x_agn", "agn_channel", "f_duty", "k_bol"):
            assert gone not in params, gone
        for gone in ("e_agn_luminosity", "hubble_time_s", "AGN_CHANNELS"):
            assert not hasattr(E, gone), gone

    def test_the_halo_field_stays_dark_matter_only(self):
        """If `f_gas` ever reached back into c(M), r_delta or dn/dM, that is
        `matter -> field -> gas` and the evaluation order stops existing."""
        import ast

        import ggah_mod.halos.field as F
        # Structural, not textual: the module docstring *says* it is not
        # baryonic, so grepping the source finds the words and fails.
        tree = ast.parse(inspect.getsource(F))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("f_gas", "f_star", "f_hot", "f_ejected",
                       "f_collisionless", "matter_weights"):
            assert banned not in names, f"the halo field computes {banned!r}"
        imported = {n.module for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module}
        assert not any("sectors" in (m or "") for m in imported), (
            "the halo field imports layer 3; the dependency runs one way")

    def test_layer_three_does_not_import_layer_four_or_five(self):
        """The dependency runs one way."""
        import ggah_mod.sectors as pkg
        for mod in pkgutil.iter_modules(pkg.__path__):
            src = inspect.getsource(
                importlib.import_module(f"ggah_mod.sectors.{mod.name}"))
            assert "from ..spectra" not in src, mod.name
            assert "from ..observables" not in src, mod.name


class TestNoDuplicatedPhysics:
    """Each shape has one definition, as layer 2 enforces for its own."""

    @staticmethod
    def _sources():
        import ggah_mod.sectors as pkg
        return {m.name: inspect.getsource(
            importlib.import_module(f"ggah_mod.sectors.{m.name}"))
            for m in pkgutil.iter_modules(pkg.__path__)}

    def test_the_gnfw_shape_is_not_redefined(self):
        """Six copies in the predecessor."""
        hits = [n for n, s in self._sources().items() if "def gnfw_shape" in s]
        assert hits == [], f"gnfw_shape redefined in {hits}"

    def test_the_shmr_inverses_are_not_reimplemented(self):
        """Three bare bisections in the predecessor, each with a zero
        gradient.  One helper now, and nothing else may roll its own."""
        hits = [n for n, s in self._sources().items()
                if "fori_loop" in s and n != "sham"]
        assert hits == [], f"a hand-rolled root solve in {hits}"

    def test_the_baryon_fraction_has_one_home(self):
        hits = [n for n, s in self._sources().items() if "def f_gas_" in s]
        assert hits == ["energetics"], hits

    def test_the_boltzmann_constant_comes_from_layer_one(self):
        """It was defined and never used in the predecessor, while the
        conversion it should have done was written as 1e-6."""
        assert "K_B_KEV_PER_K" in self._sources()["gas"]
        assert "1e-6" not in self._sources()["gas"].split('"""')[-1]


class TestEveryRegistryHasItsCalibrationTable:
    """The key-parity rule layer 2 applies to its fits."""

    @pytest.mark.parametrize("registry,table", [
        (O.OCCUPATION, O.OCC_CALIBRATION),
        (O.OCCUPATION, O.DEFAULTS),
        (E.F_GAS, E.F_GAS_CALIBRATION),
        (SH.SHMR, SH.SHMR_CALIBRATION),
    ])
    def test_keys_match(self, registry, table):
        assert set(registry) == set(table)

    def test_the_clf_registry_too(self):
        from ggah_mod.sectors import clf as C
        assert set(C.CLF) == set(C.CLF_CALIBRATION) == set(C.CLF_DEFAULTS)


class TestTheWholeLayerUnderJitAndGrad:
    """One gradient, from a cosmological parameter through every sector."""

    @pytest.mark.x64
    def test_gradient_reaches_the_cosmology_through_the_whole_chain(self):
        pk = make_pk("emu_pk")

        def forward(omega_m):
            cosmo = PLANCK18.replace(Omega_m=omega_m)
            fl = make_field(cosmo, DIFFERENTIABLE, pk, z=Z)
            lm = jnp.log10(fl.m)
            m_star = jnp.power(10.0, _GAL.log10_mstar(lm, _GP, h=cosmo.h))
            l_x = A.AgnSector(_GAL).l_x_agn(fl, A.AgnParams(), _GP)
            f_b = cosmo.Omega_b / cosmo.Omega_m
            m_bh = A.AgnSector(_GAL).mean_mbh(lm, A.AgnParams(), _GP,
                                             h=cosmo.h)
            f_ret = E.f_retained_energy(fl.m, Z, cosmo, m_star, f_b,
                                        m_bh=m_bh)
            split = MT.BaryonSplit.from_retained(f_b, f_ret,
                                                 f_star_cen=m_star / fl.m)
            w = MT.matter_weights(fl, split)
            return jnp.log(jnp.sum(w.total()))

        x, h = PLANCK18.Omega_m, 1e-6
        ad = float(jax.grad(forward)(x))
        fd = float((forward(x + h) - forward(x - h)) / (2 * h))
        assert np.isfinite(ad) and ad != 0.0
        assert ad == pytest.approx(fd, rel=1e-4)

    def test_the_chain_jits(self):
        pk = make_pk("emu_pk")

        def forward(cosmo):
            fl = make_field(cosmo, DIFFERENTIABLE, pk, z=Z)
            lm = jnp.log10(fl.m)
            m_star = jnp.power(10.0, _GAL.log10_mstar(lm, _GP, h=cosmo.h))
            l_x = A.AgnSector(_GAL).l_x_agn(fl, A.AgnParams(), _GP)
            f_b = cosmo.Omega_b / cosmo.Omega_m
            m_bh = A.AgnSector(_GAL).mean_mbh(lm, A.AgnParams(), _GP,
                                             h=cosmo.h)
            f_ret = E.f_retained_energy(fl.m, Z, cosmo, m_star, f_b,
                                        m_bh=m_bh)
            split = MT.BaryonSplit.from_retained(f_b, f_ret,
                                                 f_star_cen=m_star / fl.m)
            return jnp.sum(MT.matter_weights(fl, split).total())

        assert np.isfinite(float(jax.jit(forward)(PLANCK18)))


class TestTheSectorRegistriesAreRead:
    r"""The unclosed half of the benchmark's calibration finding.

    Layer 2's fits got a mechanism -- `halos/calibration.py`, read by
    `make_field` -- and the report recorded what was left: *"the five sector
    registries are still unread, so the wider point stands."*  A table nobody
    consults cannot prevent the error it describes, which is the whole argument,
    and it applied here unchanged.

    What is pinned below is that the check *fires*, that it fires on the right
    side of each boundary, and that it stays out of the way of a gradient.  The
    registries' key parity is `TestEveryRegistryHasItsCalibrationTable` above;
    this is the half that was missing.
    """

    @staticmethod
    def _at(z):
        return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=z)

    def test_every_registry_now_uses_one_record(self):
        """Four tuple shapes became one, so a reader learns the fields once."""
        from ggah_mod.sectors import clf as C
        from ggah_mod.sectors import coldgas as CG
        from ggah_mod.sectors.calibration import Calibration

        for table in (O.OCC_CALIBRATION, C.CLF_CALIBRATION, SH.SHMR_CALIBRATION,
                      CG.HI_CALIBRATION, E.F_GAS_CALIBRATION):
            for name, cal in table.items():
                assert isinstance(cal, Calibration), name
                assert cal.fit, name

    def test_an_occupation_outside_its_redshift_range_warns(self):
        """`zumandelbaum15` is an LS10 fit over 0.05 < z < 0.18."""
        from ggah_mod.sectors.calibration import MismatchWarning
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        g = GalaxySector("zumandelbaum15")
        p = galaxy_defaults("zumandelbaum15")
        with pytest.warns(MismatchWarning, match="zumandelbaum15"):
            g.weights(self._at(1.5), p)

    def test_and_is_silent_inside_it(self):
        """The other side of the boundary, so the test is evidence of a check
        rather than of a warning that always fires."""
        import warnings

        from ggah_mod.sectors.calibration import MismatchWarning
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        g = GalaxySector("zumandelbaum15")
        p = galaxy_defaults("zumandelbaum15")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.weights(self._at(0.135), p)
        assert not [w for w in caught
                    if issubclass(w.category, MismatchWarning)]

    def test_one_snapshot_is_a_range_of_one_point(self):
        r"""`villaescusa18` is TNG100 at :math:`z = 0` and nothing else.

        Its notes column always said "one simulation, one redshift, one
        cosmology"; lifting that into ``z_range=(0, 0)`` is what turns the
        sentence into something that fires.  A relation fitted at one snapshot
        and evaluated at another is the case this registry was recording.
        """
        from ggah_mod.sectors.calibration import MismatchWarning
        from ggah_mod.sectors.coldgas import ColdGasParams, ColdGasSector

        cg, cp = ColdGasSector("villaescusa18"), ColdGasParams()
        with pytest.warns(MismatchWarning, match="villaescusa18"):
            cg.weights(self._at(0.5), cp)

    def test_a_registry_with_no_fitted_range_never_warns(self):
        """`F_GAS_CALIBRATION` carries `z_range=None` throughout.

        `powerlaw` is a null test rather than a fit, so it has no range to
        leave; the other two have one that was never recorded here.  Both come
        out as "nothing to check", and the distinction lives in the notes.  A
        `None` range must not be read as a permissive one -- that would be the
        check reporting success where it has no information.
        """
        import warnings

        from ggah_mod.sectors.calibration import MismatchWarning
        from ggah_mod.sectors.energetics import make_f_gas

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for name in E.F_GAS_CALIBRATION:
                make_f_gas(name, z=99.0)
        assert not [w for w in caught
                    if issubclass(w.category, MismatchWarning)]
        assert all(c.z_range is None for c in E.F_GAS_CALIBRATION.values())

    def test_the_policy_word_is_the_layer_two_one(self):
        """One vocabulary for the package: strict / warn / off."""
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        p = galaxy_defaults("zumandelbaum15")
        with pytest.raises(ValueError, match="was fitted to"):
            GalaxySector("zumandelbaum15",
                         calibration="strict").weights(self._at(1.5), p)
        with pytest.raises(ValueError, match="calibration must be one of"):
            GalaxySector("zumandelbaum15",
                         calibration="lenient").weights(self._at(1.5), p)

    def test_a_traced_redshift_steps_aside_rather_than_raising(self):
        r"""``float()`` on a tracer raises and takes the gradient with it.

        The check only ever reports, so skipping it cannot change a number --
        which is what makes stepping aside the right move rather than a
        compromise.  Same contract as ``halos.variance.check_k_support``.
        """
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        g = GalaxySector("zumandelbaum15")
        p = galaxy_defaults("zumandelbaum15")

        def f(z):
            fld = make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=z)
            return jnp.sum(g.weights(fld, p).w_point)

        assert np.isfinite(float(jax.grad(f)(0.2)))

    def test_the_worst_point_of_a_grid_is_the_one_reported(self):
        r"""A grid is out of range at whichever end leaves the fit first.

        Reporting the maximum would miss it: `leauthaud12` starts at
        :math:`z = 0.22`, so a grid running from 0 is out of range at its
        *smallest* value, not its largest.
        """
        from ggah_mod.sectors.calibration import _worst

        assert _worst(np.array([0.0, 0.3, 0.5]), (0.22, 1.0)) == 0.0
        assert _worst(np.array([0.3, 0.5, 2.0]), (0.22, 1.0)) == 2.0


class TestTheSectorCheckHandlesItsEdges:
    """The branches the first round of tests left unexecuted.

    Written after measuring rather than guessed at: coverage put four
    statements of `sectors/calibration.py` outside the suite, and each is a
    contract rather than a defensive shrug.
    """

    @staticmethod
    def _at(z):
        return make_field(PLANCK18, DIFFERENTIABLE, make_pk("emu_pk"), z=z)

    def test_off_skips_the_check_entirely(self):
        """`off` is the escape hatch, so it has to actually escape."""
        import warnings

        from ggah_mod.sectors.calibration import MismatchWarning
        from ggah_mod.sectors.galaxies import GalaxySector, galaxy_defaults

        g = GalaxySector("zumandelbaum15", calibration="off")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g.weights(self._at(1.5), galaxy_defaults("zumandelbaum15"))
        assert not [w for w in caught
                    if issubclass(w.category, MismatchWarning)]

    def test_a_traced_redshift_array_steps_aside_too(self):
        r"""The scalar case was covered; the array case was not.

        `_worst` reaches for `float(v)` on each element, which raises under a
        trace exactly as the scalar path does -- but through a different
        branch, and a check that survived a traced scalar and died on a traced
        vector would be a poor guarantee.
        """
        from ggah_mod.sectors.calibration import check_sector_calibration
        from ggah_mod.sectors.coldgas import HI_CALIBRATION

        # `villaescusa18` is fitted at z = 0 alone, so 0.9 is far outside it:
        # eagerly this warns, and under `jit` it must not raise instead.
        @jax.jit
        def f(z):
            check_sector_calibration("HI-halo relation", "villaescusa18",
                                     HI_CALIBRATION, z)
            return jnp.sum(z ** 2)

        got = float(f(jnp.asarray([0.1, 0.9])))
        assert np.isfinite(got) and got == pytest.approx(0.82, rel=1e-6)

    def test_an_empty_redshift_grid_says_nothing(self):
        """No values means no worst value, and no claim to make about them."""
        from ggah_mod.sectors.calibration import _worst

        assert _worst(np.array([]), (0.0, 1.0)) is None
