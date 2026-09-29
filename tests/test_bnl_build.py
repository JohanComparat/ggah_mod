r"""The beyond-linear bias table rebuilds from the upstream files, offline.

:func:`~ggah_mod.halos.beyond_linear_bias.build` reads the Mead & Verde ascii
tables and computes MultiDark's own :math:`\sigma(R, a)`.  Its downloads are
skipped for any file already in the cache, so a cache seeded with small
synthetic files in the upstream layout runs the whole build without a network,
and ``emu_pk`` stands in for the Boltzmann solver.  What is checked is the part
most likely to go wrong silently: the triple loop over (bin, bin, k) landing on
the right axes, and the consistency checks refusing inconsistent inputs.
"""
import numpy as np
import pytest

import ggah_mod.halos.beyond_linear_bias as B

K = np.logspace(np.log10(0.0063), np.log10(0.738), B._NK)


def _beta(snap, i, j, kk):
    """A synthetic value that encodes its own position."""
    return 1e-3 * (10 * i + j) + 1e-5 * kk + 1e-7 * snap


def _write(cache, snap, *, errors_offset=0.0, k=K):
    stem = f"MDR1_rockstar_{snap}"
    nu = np.linspace(0.5, 4.0, B._NBIN)
    bins = np.column_stack([np.full(B._NBIN, 11.0), np.full(B._NBIN, 12.0),
                            np.linspace(11, 15, B._NBIN), nu - 0.1, nu + 0.1, nu,
                            np.linspace(1.0, 5.0, B._NBIN), np.ones(B._NBIN)])
    rows, rows_err = [], []
    for i in range(B._NBIN):                # upstream: bin, bin, k
        for j in range(B._NBIN):
            for kk in range(B._NK):
                one_plus = 1.0 + _beta(snap, i, j, kk)
                rows.append((k[kk], one_plus))
                rows_err.append((k[kk], one_plus + errors_offset, 0.01))
    for sub, data in ((f"BNL/M512/{stem}_binstats.dat", bins),
                      (f"BNL/M512/{stem}_bnl.dat", rows),
                      (f"BNL_errors/M512/{stem}_bnl.dat", rows_err)):
        path = cache / sub
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.asarray(data))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cache = tmp_path_factory.mktemp("bnl-cache")
    for snap in B.SNAPS:
        _write(cache, snap)
    out = B.build(out=cache / "rebuilt.npz", cache=cache, pk_name="emu_pk")
    return dict(np.load(out))


def test_the_rebuild_puts_each_value_on_its_axes(built):
    beta = built["beta"]
    assert beta.shape == (len(B.SNAPS), B._NK, B._NBIN, B._NBIN)
    for s_i, i, j, kk in ((0, 0, 0, 0), (3, 2, 5, 7), (len(B.SNAPS) - 1, 7, 1, 24)):
        assert beta[s_i, kk, i, j] == pytest.approx(_beta(B.SNAPS[s_i], i, j, kk),
                                                    abs=1e-12)
    assert np.allclose(built["k"], K)


def test_the_rebuild_carries_the_shipped_tables_fields(built):
    shipped = B.load()
    assert set(shipped) <= set(built)
    assert built["g_residual"] < 1e-3
    assert np.all(np.diff(built["g_md"]) > 0)


def test_the_two_upstream_variants_must_agree(tmp_path):
    _write(tmp_path, B.SNAPS[0], errors_offset=0.01)
    with pytest.raises(ValueError, match="disagree"):
        B.build(out=tmp_path / "x.npz", cache=tmp_path, pk_name="emu_pk")


def test_every_snapshot_must_share_the_k_grid(tmp_path):
    _write(tmp_path, B.SNAPS[0])
    _write(tmp_path, B.SNAPS[1], k=K * 1.01)
    with pytest.raises(ValueError, match="k grid differs"):
        B.build(out=tmp_path / "x.npz", cache=tmp_path, pk_name="emu_pk")


def test_a_file_of_the_wrong_shape_is_refused(tmp_path):
    path = tmp_path / "short_bnl.dat"
    np.savetxt(path, np.ones((10, 2)))
    with pytest.raises(ValueError, match="expected"):
        B._read_bnl(path, 2)
