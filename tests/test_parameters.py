"""The cosmology container: one amplitude, a budget that closes, a real pytree."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggah_mod.cosmology import Cosmology, PLANCK18, RETIRED_KEYS
from ggah_mod.cosmology.parameters import LEAF_FIELDS, STATIC_FIELDS
from ggah_mod.cosmology import constants as C


class TestDensityBudget:
    """Omega_m contains the neutrinos, and every derived density follows."""

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.15, 0.30])
    def test_matter_budget_closes_exactly(self, mnu):
        """It closes on `Omega_nu_matter`, not on the 93.14 convention.

        The two differ by 0.46 per cent, and the budget has to close on the
        one the expansion carries: closing it on the convention while
        `hubble_e` sums the integral is what put `6.5e-5` of extra total
        matter into the model at `0.6 eV`.  `Omega_nu` remains what
        Eq. (nu-budget) reports and is checked separately below.
        """
        c = Cosmology.create(sum_mnu=mnu)
        assert c.Omega_b + c.Omega_cdm + c.Omega_nu_matter == \
            pytest.approx(c.Omega_m, abs=1e-15)

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30])
    def test_flatness_closes_against_what_the_expansion_sums(self, mnu):
        """Not against the mass budget -- against `Omega_nu_today`.

        `hubble_e` sums the *relativistic* neutrino amplitude times the Komatsu
        factor, so that is what flatness has to close against if `E(0) = 1` is
        to hold identically.  Closing against `Omega_m` instead is what let a
        massless model carry no neutrino radiation for as long as nobody
        evaluated `E(z)` there.
        """
        c = Cosmology.create(sum_mnu=mnu)
        total = c.Omega_gamma + c.Omega_cb + c.Omega_nu_today + c.Omega_de
        assert total == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30, 0.60])
    def test_omega_m_is_the_total_matter(self, mnu):
        """`Omega_cb` and the expansion's neutrinos add back to `Omega_m`.

        The two neutrino densities are not the same number and cannot be: one
        is a convention chosen so the mass budget closes, the other is the
        integral the expansion needs.  They differ by 0.53 per cent.  **The
        one that may be subtracted from `Omega_m` is the integral**, because
        that is the one `hubble_e` adds back -- subtract the convention and the
        model carries `Omega_m + 6.5e-5` of total matter at 0.6 eV while every
        reference code is handed the same `Omega_m` and uses it.

        That was worth 1.0e-4 on `E(z=3)`, four times the spread between CCL,
        CAMB, CLASS and cloelib, rising linearly with the neutrino mass and
        exactly zero without one.  This is the invariant that rules it out.
        """
        c = Cosmology.create(sum_mnu=mnu)
        assert float(c.Omega_cb + c.Omega_nu_matter - c.Omega_m) == \
            pytest.approx(0.0, abs=1e-15)
        assert float(c.Omega_cdm + c.Omega_b + c.Omega_nu_matter
                     - c.Omega_m) == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("mnu", [0.0, 0.06, 0.30, 0.60])
    def test_dark_energy_carries_only_the_relativistic_remainder(self, mnu):
        """`Omega_de` sheds exactly the neutrino energy that is not matter.

        `Omega_cb` subtracts :attr:`Omega_nu_matter` from `Omega_m` and
        `hubble_e` adds :attr:`Omega_nu_today` back, so closure leaves
        `Omega_de` short by the difference -- which is the *relativistic*
        part of the neutrino density, and belongs out of the matter budget.

        At `sum_mnu = 0` that difference is the whole `3.8e-5` of massless
        neutrino radiation, and dark energy is right to carry none of it.  At
        `0.6 eV` it is `1.5e-7`.  It held a `7e-6` excess of a different kind
        while `Omega_cb` subtracted the 93.14 convention instead, and that
        excess was the whole of the background's disagreement with CCL, CAMB
        and CLASS.
        """
        c = Cosmology.create(sum_mnu=mnu)
        naive = 1.0 - c.Omega_gamma - c.Omega_m - c.Omega_k
        shed = float(c.Omega_nu_today - c.Omega_nu_matter)
        assert float(naive - c.Omega_de) == pytest.approx(shed, rel=1e-9)
        assert 0.0 < shed <= float(c.Omega_nu_rel) * 1.0001
        # the two mass conventions still differ; using the wrong one here is
        # what this whole invariant exists to prevent
        assert 0.0 <= float(c.Omega_nu_matter - c.Omega_nu) < 1e-4

    def test_omega_nu_follows_the_stated_convention(self, cosmo):
        assert cosmo.Omega_nu == pytest.approx(
            cosmo.sum_mnu / (C.NU_DENOM_EV * cosmo.h ** 2), rel=1e-15)

    def test_cold_and_total_are_different_and_both_available(self, cosmo):
        """The two densities that get confused.  Halos form from one, lensing
        sees the other; the package exposes them under distinct names."""
        assert cosmo.rho_cold < cosmo.rho_matter
        assert cosmo.rho_matter / cosmo.rho_cold - 1.0 == pytest.approx(
            cosmo.f_nu / (1.0 - cosmo.f_nu), rel=1e-12)

    def test_massless_collapses_the_two(self, massless):
        assert massless.rho_cold == massless.rho_matter
        assert massless.Omega_nu == 0.0
        assert massless.f_nu == 0.0


class TestOneAmplitude:
    @pytest.mark.parametrize("key", RETIRED_KEYS)
    def test_retired_keys_are_refused(self, key):
        with pytest.raises(TypeError, match="not an input"):
            Cosmology.create(**{key: 0.81})

    @pytest.mark.parametrize("key", ["sigma8", "S8"])
    def test_replace_also_refuses(self, cosmo, key):
        with pytest.raises(TypeError, match="not an input"):
            cosmo.replace(**{key: 0.81})

    def test_unknown_parameter_is_refused(self):
        with pytest.raises(TypeError, match="unknown parameter"):
            Cosmology.create(Omeag_m=0.3)

    def test_the_amplitude_is_the_only_amplitude(self):
        fields = set(Cosmology.__dataclass_fields__)
        assert "ln10A_s" in fields
        assert not (fields & set(RETIRED_KEYS))


class TestPytree:
    def test_round_trip(self, cosmo):
        leaves, treedef = jax.tree_util.tree_flatten(cosmo)
        assert len(leaves) == len(LEAF_FIELDS)
        assert jax.tree_util.tree_unflatten(treedef, leaves) == cosmo

    def test_the_leaves_and_the_statics_are_all_of_it(self):
        """No field may be silently neither.

        ``LEAF_FIELDS`` is derived by subtracting ``STATIC_FIELDS`` from the
        dataclass, so a new parameter becomes a leaf without anyone
        remembering to add it -- and this asserts the subtraction covers the
        whole, which is what lets `_as_key` build a cache key from the two
        lists instead of a hand-picked subset.
        """
        assert LEAF_FIELDS + STATIC_FIELDS == tuple(Cosmology.__dataclass_fields__)
        assert set(LEAF_FIELDS).isdisjoint(STATIC_FIELDS)

    def test_the_hierarchy_is_a_choice_and_not_a_coordinate(self):
        """It rides in ``aux_data``, and that is a statement about physics.

        Large-scale structure responds to :math:`\\Sigma m_\\nu` and not to how
        the sum is divided, so the ordering carries no likelihood gradient:
        two discrete hypotheses, chosen once at the start of a run, rather than
        two points on an axis a chain explores.  It also indexes a dict by name
        inside a ``jit`` trace, which a tracer cannot do.

        Both halves are asserted here -- it survives the round trip, and two
        cosmologies differing only in it are not equal.
        """
        n = Cosmology.create(sum_mnu=0.15, nu_hierarchy="normal")
        i = Cosmology.create(sum_mnu=0.15, nu_hierarchy="inverted")
        assert n != i
        for c in (n, i):
            leaves, treedef = jax.tree_util.tree_flatten(c)
            back = jax.tree_util.tree_unflatten(treedef, leaves)
            assert back == c and back.nu_hierarchy == c.nu_hierarchy
        # the ordering moves the expansion but not the matter budget
        assert float(n.Omega_nu_today) != float(i.Omega_nu_today)
        assert float(n.Omega_cb) == float(i.Omega_cb)

    def test_survives_jit(self, cosmo):
        @jax.jit
        def f(c):
            return c.Omega_cb + c.Omega_gamma
        assert float(f(cosmo)) == pytest.approx(
            cosmo.Omega_cb + cosmo.Omega_gamma, rel=1e-12)

    def test_every_field_is_a_differentiable_leaf(self, cosmo):
        """A field that is not a leaf is a parameter no forecast can vary.

        Which is the rule ``nu_hierarchy`` satisfies rather than breaks, and so
        it is excluded here by being a *static* field rather than by name.  No
        forecast varies the mass ordering: large-scale structure constrains
        :math:`\\Sigma m_\\nu` and not its division, so there is no gradient to
        take and nothing for a chain to explore.  Comparing the two orderings
        means running twice.
        """
        for name in LEAF_FIELDS:
            g = jax.grad(lambda v, n=name: cosmo.replace(**{n: v}).Omega_de
                         )(float(getattr(cosmo, name)))
            assert np.isfinite(float(g)), f"{name} produced a non-finite gradient"

    def test_frozen(self, cosmo):
        with pytest.raises(Exception):
            cosmo.Omega_m = 0.4
