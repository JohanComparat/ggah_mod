r"""beta^NL parity: ``ggah_mod`` against the ``hod_mod`` reference.

Both packages read the same 35 MultiDark snapshots, but they choose between
them differently on purpose -- ``hod_mod`` matches clustering amplitude
(sigma_8), ``ggah_mod`` matches sigma(R) shape via the Angulo & White (2010)
rescaling.  Comparing the two matchers would measure that intended difference
and nothing else.

So both are **pinned to the same snapshot** here.  What is then compared is the
part that is supposed to agree exactly: the distilled table, the k and nu
interpolation, the four edge rules, and the correction integrals.  A difference
in any of those is a porting error; a difference in the matching is a design
decision, and is measured in the main suite instead.

**The k reading is the reference's when asked for.**  Since 0.9.7 ggah_mod
reads beta between tabulated wavenumbers and snapshots with a C^1 cubic
(``K_INTERP``/``G_INTERP``), so that the cosmology derivatives are smooth; the
reference reads it linearly.  Every table here is built with
``interp="linear"``, and the peak-height projection is set to the reference's
hat functions (``NU_INTERP``): this suite checks the port, and the C^1 defaults
are a design decision measured in ``tests/test_beyond_linear_bias.py``.
"""

import numpy as np
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.parity

from ggah_mod.halos import beyond_linear_bias as B


@pytest.fixture(autouse=True)
def _reference_reading(monkeypatch):
    """The reference projects a peak height with hat functions; so does this suite."""
    monkeypatch.setattr(B, "NU_INTERP", "linear")

hod_bnl = pytest.importorskip(
    "hod_mod.core.beyond_linear_bias",
    reason="hod_mod is a test-only dependency; install it to run parity")


SNAPS = (85, 74, 52, 36)


@pytest.fixture(scope="module")
def k():
    # Inside the taper and below the top of the table, where both packages are
    # doing ordinary interpolation rather than edge handling.  The edges are
    # compared separately below.
    return np.logspace(np.log10(0.11), np.log10(0.70), 24)


@pytest.mark.parametrize("snap", SNAPS)
def test_table_agrees_snapshot_for_snapshot(k, snap):
    ours = B.table_at(jnp.asarray(k), snap=snap, interp="linear")
    theirs = hod_bnl.BeyondLinearBiasMead21(snap=snap)

    ref = theirs.table_at(k)
    np.testing.assert_allclose(np.asarray(ours.nu),
                               np.asarray(ref.nu_ref), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ours.beta),
                               np.asarray(ref.beta), rtol=1e-10)


@pytest.mark.parametrize("snap", SNAPS)
def test_nu_interpolation_agrees(k, snap):
    """Including both edge rules: the sample straddles the tabulated range."""
    ours = B.table_at(jnp.asarray(k), snap=snap, interp="linear")
    theirs = hod_bnl.BeyondLinearBiasMead21(snap=snap)
    nu = np.linspace(float(ours.nu[0]) - 0.4, float(ours.nu[-1]) + 0.6, 37)

    np.testing.assert_allclose(
        np.asarray(B.beta_nl(k, nu, nu, table=ours)),
        np.asarray(theirs.beta_nl(k, nu, nu)), rtol=1e-9, atol=1e-12)


def test_k_edge_rules_agree():
    """Below the taper, and above the top of the table."""
    k_edge = np.array([1e-4, 0.02, 0.05, 0.064, 0.08, 0.10, 0.5, 0.74, 5.0])
    ours = B.table_at(jnp.asarray(k_edge), snap=85, interp="linear")
    theirs = hod_bnl.BeyondLinearBiasMead21(snap=85)
    np.testing.assert_allclose(np.asarray(ours.beta),
                               np.asarray(theirs.beta_nl_matrix(k_edge)),
                               rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("snap", SNAPS)
def test_correction_integrals_agree(k, snap):
    ours = B.table_at(jnp.asarray(k), snap=snap, interp="linear")
    theirs = hod_bnl.BeyondLinearBiasMead21(snap=snap)

    rng = np.random.default_rng(0xB0DE)
    nu = np.sort(rng.uniform(0.5, 4.5, 80))
    w_a = rng.lognormal(size=80) / 80.0
    w_b = rng.lognormal(size=80) / 80.0
    uk = np.exp(-np.outer(k, np.linspace(0.0, 3.0, 80)))

    np.testing.assert_allclose(
        np.asarray(B.correction_2h_gg(nu, w_a, uk, ours)),
        np.asarray(theirs.correction_2h_gg(k, nu, w_a, uk, table=_pin(theirs, k))),
        rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(B.correction_2h_gm(nu, w_a, w_b, uk, uk, ours)),
        np.asarray(theirs.correction_2h_gm(k, nu, w_a, w_b, uk, uk,
                                           table=_pin(theirs, k))),
        rtol=1e-9)


def _pin(model, k):
    """The hod_mod table object for its pinned snapshot."""
    return model.table_at(k)


def test_the_two_packages_ship_the_same_measurement():
    """Snapshot for snapshot, the distilled tables are the same numbers.

    Both were distilled from the upstream ascii independently, and both keep
    beta^NL in float64 for exactly this comparison.
    """
    ours = B.load()
    theirs = hod_bnl.BeyondLinearBiasMead21()
    np.testing.assert_array_equal(np.asarray(ours["snap"]), theirs._snaps)
    np.testing.assert_array_equal(np.asarray(ours["beta"]), theirs._beta_tab)
    np.testing.assert_array_equal(np.asarray(ours["nu"]), theirs._nu_tab)
