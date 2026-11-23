"""The forecast machinery of Paper I, Secs. 4.1-4.3: BAO ratios, full shape, halo counts.

Ported from the paper's figure script so that notebooks 07-09 compute exactly
what the paper's figures show.  Notebooks 08 and 09 walk through each piece;
this file holds the parts too long for a notebook cell.
"""
from __future__ import annotations

import ggah_mod.halos as H
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from ggah_mod import DIFFERENTIABLE
from ggah_mod.cosmology import PLANCK18, make_pk

import _helpers as T

_DTEX = T.TEX
_HALO_CHOICE = dict(mdef="200m", hmf_model="tinker08", bias_model="tinker10", cm_model="bhattacharya13")


class _DFProxy:
    """DESI DR1 full-shape posteriors from the tutorial data bundle."""

    _post = None

    def __getattr__(self, name):
        if _DFProxy._post is None:
            _DFProxy._post = T.DesiPosteriors("desi/desi_dr1_fs_contours.json")
        return getattr(_DFProxy._post, name)


DF = _DFProxy()


#: The mock BAO survey of :func:`fig_distance_fisher`: the seven effective
#: redshifts of DESI DR2's BAO measurements (Abdul Karim et al. 2025, Table 4
#: -- BGS, LRG1, LRG2, LRG3+ELG1, ELG2, QSO, Ly-alpha), measured to the four
#: fractional precisions the figure compares, as ratios to the sound horizon.
#: As in DR2, BGS gives the isotropic D_V alone (``_BAO_ISO``) and the other
#: six both D_M and D_H.  ``sum_mnu`` and ``T_cmb`` are held fixed: their
#: derivatives (Fig. distance-gradients) are too small for these ratios to see.
_BAO_Z = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
_BAO_TRACER = ("BGS", "LRG1", "LRG2", "LRG3+ELG1", "ELG2", "QSO", "Lya")
_BAO_ISO = np.array([True, False, False, False, False, False, False])
_BAO_SIG = (0.0003, 0.001, 0.003, 0.01)
_BAO_P = ("Omega_m", "Omega_b", "h", "w0", "wa", "Omega_k")
_BAO_COLOUR = ("#0b2545", "#1f6fb2", "#2ca58d", "#d1495b")
#: The exact Delta chi^2 at a Fisher 1-sigma point above which that precision's
#: ellipse is not drawn as a contour: 2 puts the true 1-sigma surface at 0.7 of
#: the drawn one.
_BAO_CHI2_BAD = 2.0
#: The BBN prior on omega_b DESI DR2 uses (their Eq. 14): Schoneberg (2024)
#: Eq. 5.1, PRyMordial with NACRE II rates and PDG-recommended abundances, in
#: LambdaCDM.  Only its width enters the forecast, centred on the fiducial
#: omega_b (0.02237), which a Fisher matrix needs and the prior's mean is not.
_BBN_OMEGA_B = (0.02218, 0.00055)
#: DESI DR2's own posteriors (``desi_contours.py``), drawn as outlines over the
#: forecast's filled ellipses: ``bao`` is BAO alone in this model, ``bbn`` BAO
#: with ``_BBN_OMEGA_B`` in flat LambdaCDM.  Purple clears every forecast
#: colour by OKLab Delta E >= 17.8 (6.4 deuteranope, against the blue -- legal
#: because outline-against-fill is the second encoding); orange is the one
#: orange that keeps >= 15 from the red, and its 2.8:1 on white is why it is
#: also dash-dotted and labelled.
_DESI_STYLE = {"bao": dict(color="#7b3294", lw=1.2, ls="-"),
               "bbn": dict(color="#e08214", lw=1.2, ls="-.")}
_DESI_LABEL = {"bao": "DESI DR2 BAO alone, sampled",
               "bbn": r"DESI DR2 BAO + BBN, flat $\Lambda$CDM"}


def _ln_bao(t, z, params=_BAO_P, cosmo=PLANCK18):
    r"""``[ln f_K/r_d, ln D_H/(E r_d)]`` at the values ``t`` of ``params``.

    The two distances a BAO measurement returns, the transverse
    :math:`f_K(\chi)` of Eq. (eq:fk) and the radial :math:`c/H(z) = D_H/E`, as
    ratios to the sound horizon :math:`r_d` of Eq. (eq:rd) -- the quantity BAO
    measures.  Both are in Mpc/h, so the ratio carries no unit and :math:`h`
    reaches it only through the physics: through :math:`r_d`'s
    :math:`\omega_b` and :math:`\omega_{cb}`, and the radiation.

    :func:`~ggah_mod.cosmology.drag.r_drag` checks its calibration domain on
    concrete values and steps aside under a trace, so a Jacobian at the
    fiducial differentiates it and a point outside the domain raises.
    """
    from ggah_mod.cosmology import r_drag
    from ggah_mod.cosmology.background import (
        comoving_distance,
        hubble_e,
        transverse_distance,
    )
    c = cosmo.replace(**dict(zip(params, t)))
    zj = jnp.asarray(z)
    rd = r_drag(c)
    fk = transverse_distance(comoving_distance(zj, c), c)
    dh = c.hubble_distance / hubble_e(zj, c)
    return jnp.concatenate([jnp.log(fk / rd), jnp.log(dh / rd)])


def _bao_observed(z=_BAO_Z, iso=_BAO_ISO):
    r"""``(M, b)``: the measured log-ratios are ``M @ _ln_bao(...) + b``.

    :func:`_ln_bao` returns both ratios at every redshift; DESI measures both
    only where the clustering is anisotropic enough to separate them, and
    elsewhere :math:`D_V = [z\,D_M^2\,D_H/E]^{1/3}`, whose log ratio to
    :math:`r_d` is a fixed linear combination of the two plus :math:`\ln z/3`
    (the weights sum to one, so :math:`r_d` divides out once).  Rows:
    :math:`f_K` at the anisotropic redshifts, then :math:`D_H/E` there, then
    :math:`D_V`.
    """
    n = len(z)
    rows, b = [], []
    for block in (0, 1):
        for i in np.flatnonzero(~iso):
            r = np.zeros(2 * n)
            r[block * n + i] = 1.0
            rows.append(r)
            b.append(0.0)
    for i in np.flatnonzero(iso):
        r = np.zeros(2 * n)
        r[i], r[n + i] = 2.0 / 3.0, 1.0 / 3.0
        rows.append(r)
        b.append(np.log(z[i]) / 3.0)
    return np.array(rows), np.array(b)


def _bbn_prior(t, params=_BAO_P):
    r"""The BBN prior on :math:`\omega_b = \Omega_b h^2` as a Fisher matrix in
    ``params``: :math:`g g^{\rm T}/\sigma^2`, :math:`g = \partial\omega_b/\partial\theta`."""
    t = dict(zip(params, t))
    g = np.zeros(len(params))
    if "Omega_b" in params:
        g[params.index("Omega_b")] = t["h"] ** 2
    if "h" in params:
        g[params.index("h")] = 2.0 * t["Omega_b"] * t["h"]
    return np.outer(g, g) / _BBN_OMEGA_B[1] ** 2


def _fisher_ellipse(ax, mean, cov, *, k=np.sqrt(2.30), **kw):
    """The 68% contour of a 2-D Gaussian (Delta chi2 = 2.30), as a patch.

    From the eigenvectors of ``cov``, as ``ggah_cal/scripts/13_cosmo_fisher.py``
    draws it; a :class:`~matplotlib.patches.Polygon` so it can be filled.
    """
    from matplotlib.patches import Polygon
    w, v = np.linalg.eigh(cov)
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    pts = v @ (k * np.sqrt(np.clip(w, 0.0, None))[:, None]
               * np.array([np.cos(t), np.sin(t)]))
    ax.add_patch(Polygon(np.column_stack([mean[0] + pts[0], mean[1] + pts[1]]),
                         closed=True, **kw))





# ---- Sec. 2.6: what a full-shape measurement constrains ---------------------

#: The mock full-shape survey of :func:`fig_pk_fisher`: DESI DR1's six
#: full-shape samples (DESI 2024 V, Table 1; ELG1 is not one of them), each
#: measured as the monopole and quadrupole on 36 bins of 0.005 h/Mpc over
#: 0.02 < k < 0.20 (V Sect. 4.7, VII Sect. 2.1.2) -- 432 points, DESI's count.
_FS_TRACER = ("BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO")
_FS_Z = np.array([0.295, 0.510, 0.706, 0.919, 1.317, 1.491])
#: V Table 1's monopole at k = 0.14 h/Mpc, (Mpc/h)^3, quoted there to two
#: figures as approximate.  It fixes each sample's b1 through the Kaiser
#: monopole, and with ``_FS_INV_NBAR`` the shot noise's share of the variance.
_FS_P0_014 = np.array([9.2e3, 8.9e3, 8.9e3, 8.4e3, 2.9e3, 5.0e3])
#: V Table 1's effective volumes in Gpc^3 (no h): the DR1 anchor, nothing else.
_FS_VEFF = np.array([1.7, 2.6, 4.0, 5.0, 2.7, 1.5])
#: 1/nbar in (Mpc/h)^3, from Ivanov et al. (2026), Table 1, same six samples.
_FS_INV_NBAR = np.array([5.2e3, 4.2e3, 4.7e3, 8.8e3, 9.6e3, 4.5e4])
_FS_DK = 0.005
_FS_K = 0.02 + _FS_DK * (np.arange(36) + 0.5)
#: The wavenumber of a 40 Mpc/h separation, 2 pi / 40, where each precision
#: is set: the monopole's diagonal error there is ``sigma`` in every sample.
_FS_K40 = 2.0 * np.pi / 40.0
_FS_SIG = (0.001, 0.003, 0.01, 0.03)
#: The precisions Fig. pk-fisher draws: alone, the 3% ellipse leaves the
#: emulator's box at 1 sigma (printed), so only Fig. pk-bao-fisher keeps it.
#: The first three of ``_FS_COLOUR``, so a precision has one colour in both
#: figures.
_FS_SIG_ALONE = _FS_SIG[:3]
_FS_P = ("Omega_m", "Omega_b", "h", "n_s", "ln10A_s", "w0", "wa", "Omega_k")
#: Sec. 2.2.3's four, in the same order of precision.
_FS_COLOUR = _BAO_COLOUR
#: Gauss-Legendre nodes on 0 <= mu <= 1; the multipoles move by < 1e-8 when
#: doubled, at the worst 1% point (asserted).
_FS_NMU = 24
#: The precision whose 1-sigma ellipses set each figure's axes, before they
#: widen to hold DESI's contours: the loosest drawn, 1% alone (an outline) and
#: 3% with BAO (a contour).
_FS_FRAME = {"fs": 0.01, "fsbao": 0.03}
#: DESI's ``ns10`` prior on n_s (DESI 2024 VII, Eq. 2.1), used only to put the
#: forecast on the footing of DESI's own full-shape-alone result.
_NS10 = (0.9649, 0.042)
#: The BAO of the combination: Sec. 2.2.3's thirteen ratios, at 1 per cent.
_FS_BAO_SIG = 0.01
#: DESI DR1's own posteriors (``desi_fs_contours.py``), as outlines over the
#: forecast's filled ellipses -- the purple of Fig. distance-fisher's DESI BAO.
_FS_DESI_STYLE = dict(color="#7b3294", lw=1.2, ls="-")
_FS_DESI_LABEL = {
    "fs": r"DESI DR1 FS + BBN + $n_{\rm s}$ prior, flat $\Lambda$CDM",
    "fsbao": r"DESI DR1 FS + BAO + BBN + $n_{\rm s}$ prior, flat $\Lambda$CDM"}
_FS_CACHE = {}


def _mu_nodes(n):
    """Gauss-Legendre on [0, 1], weights summing to one."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _fs_fid():
    r"""What every evaluation is compared against, computed once.

    The emulator; :math:`f_K` and :math:`E` at the six redshifts, which define
    the Alcock-Paczynski fiducial; the growth rate of the cold field, Eq.
    (eq:growth-rate) on :math:`P_{\rm cb}`; and each sample's :math:`b_1`, the
    positive root of :math:`(b^2 + 2bf/3 + f^2/5)\,P_{\rm cb}(0.14) = P_0`
    with :math:`P_0` from ``_FS_P0_014``.
    """
    if "fid" not in _FS_CACHE:
        from ggah_mod.cosmology.background import (
            comoving_distance,
            hubble_e,
            transverse_distance,
        )
        from ggah_mod.cosmology.growth import growth_rate
        emu = make_pk("emu_pk")
        c, zj = PLANCK18, jnp.asarray(_FS_Z)
        f = np.asarray(growth_rate(zj, c, emu, variant="cold"))
        p14 = np.array([float(emu.pk_cb(jnp.asarray([0.14]), float(z), c)[0])
                        for z in _FS_Z])
        _FS_CACHE["fid"] = dict(
            emu=emu, f=f, p14=p14,
            fk=np.asarray(transverse_distance(comoving_distance(zj, c), c)),
            e=np.asarray(hubble_e(zj, c)),
            b=-f / 3.0 + np.sqrt(f ** 2 / 9.0 - f ** 2 / 5.0 + _FS_P0_014 / p14))
    return _FS_CACHE["fid"]


def _pk_nowiggle(emu, z, c, kq, width=0.25):
    r"""P_cb without its acoustic wiggles: ln P smoothed by a Gaussian of
    ``width`` in log10 k over 1e-4 to 10 h/Mpc, interpolated to ``kq``.
    Traceable (the referee round's double-counting test, R5.5)."""
    lk = jnp.linspace(-4.0, 1.0, 1001)
    lp = jnp.log(emu.pk_cb(10.0 ** lk, z, c))
    w = jnp.exp(-0.5 * ((lk[:, None] - lk[None, :]) / width) ** 2)
    sm = (w @ lp) / w.sum(1)
    # the smoothing bends a curved ln P near the ends: correct by the smoothed
    # power-law-free part away from the wiggles, i.e. keep only the ratio's
    # smooth trend over 0.005-0.6 h/Mpc
    ratio = lp - sm
    rs = (w @ ratio) / w.sum(1)
    lnw = lp - (ratio - rs)
    return jnp.exp(jnp.interp(jnp.log10(kq), lk, lnw))


def _fs_basis(t, params=_FS_P, nmu=_FS_NMU, k=_FS_K, ap=True, nowiggle=False):
    r"""``(I, f)``: the observed multipoles' three :math:`\mu` moments per
    sample, shape ``(6, 3, 2, nk)``, and :math:`f`, such that

    .. math::  P_\ell = b_1^2 I_{0\ell} + 2 b_1 f I_{1\ell} + f^2 I_{2\ell},
               \quad I_{n\ell} = (2\ell+1)\int_0^1 \dd\mu\,L_\ell(\mu)\,
               \mu_t^{2n}\,\frac{P_{\rm cb}(k_t)}{\alpha_\perp^2\alpha_\parallel}.

    The Kaiser power spectrum :math:`(b_1 + f\mu^2)^2 P_{\rm cb}` seen through the
    fiducial distances: :math:`\alpha_\perp = f_K/f_K^{\rm fid}`,
    :math:`\alpha_\parallel = E^{\rm fid}/E`, both in Mpc/h so neither carries
    :math:`h`, and :math:`(k_t, \mu_t)` the true wavevector behind the observed
    :math:`(k, \mu)`.  Kept as moments because the model is exactly quadratic
    in each sample's :math:`b_1`, which is what lets :func:`_fs_chi2_profiled`
    minimise over the six of them exactly.

    Traceable; called eagerly, :mod:`emu_pk` checks its box and raises
    ``ValueError`` outside it, as ``r_drag`` does in :func:`_ln_bao`.
    """
    from ggah_mod.cosmology.background import (
        comoving_distance,
        hubble_e,
        transverse_distance,
    )
    from ggah_mod.cosmology.growth import growth_rate
    fid = _fs_fid()
    emu = fid["emu"]
    c = PLANCK18.replace(**dict(zip(params, t)))
    zj = jnp.asarray(_FS_Z)
    a_perp = transverse_distance(comoving_distance(zj, c), c) / fid["fk"]
    a_par = fid["e"] / hubble_e(zj, c)
    if not ap:      # the distances held at the fiducial: the shape alone
        a_perp, a_par = jnp.ones_like(a_perp), jnp.ones_like(a_par)
    f = growth_rate(zj, c, emu, variant="cold")
    mu, w = _mu_nodes(nmu)
    wl = np.stack([w, 5.0 * w * 0.5 * (3.0 * mu ** 2 - 1.0)]).T     # (nmu, 2)
    kj = jnp.asarray(k)
    out = []
    for i, z in enumerate(_FS_Z):
        r = a_par[i] / a_perp[i]
        s = jnp.sqrt(1.0 + mu ** 2 * (1.0 / r ** 2 - 1.0))
        mt2 = (mu / (r * s)) ** 2
        kt = kj[:, None] * s[None, :] / a_perp[i]
        pk = (_pk_nowiggle(emu, float(z), c, kt) if nowiggle
              else emu.pk_cb(kt, float(z), c))
        p = pk / (a_perp[i] ** 2 * a_par[i])
        out.append(jnp.stack([(p * mt2 ** n) @ wl for n in range(3)])
                   .transpose(0, 2, 1))
    return jnp.stack(out), f


def _fs_multipoles(t, b, params=_FS_P, nmu=_FS_NMU, k=_FS_K, ap=True, nowiggle=False):
    """``(6, 2, nk)``: each sample's monopole and quadrupole at ``t``, ``b``."""
    I, f = _fs_basis(t, params, nmu, k, ap=ap, nowiggle=nowiggle)
    b, f = jnp.asarray(b)[:, None, None], f[:, None, None]
    return b ** 2 * I[:, 0] + 2.0 * b * f * I[:, 1] + f ** 2 * I[:, 2]


def _fs_cov_unit(k, nmu=48):
    r"""The Gaussian covariance of :math:`(P_0, P_2)` at each ``k`` for a
    volume of one (Mpc/h)^3, ``(6, nk, 2, 2)``, at the fiducial:

    .. math::  C_{\ell\ell'} = \frac{(2\ell+1)(2\ell'+1)}{N_k}\int_{-1}^{1}
               \dd\mu\,L_\ell L_{\ell'}\Big[P_g(k,\mu) + \frac{1}{\bar n}\Big]^2,
               \quad N_k = \frac{V k^2\Delta k}{2\pi^2}.

    Diagonal in :math:`k`, the two multipoles correlated at each :math:`k`.
    """
    fid = _fs_fid()
    mu, w = _mu_nodes(nmu)
    L = np.stack([np.ones_like(mu), 0.5 * (3.0 * mu ** 2 - 1.0)])
    k = np.atleast_1d(k)
    nk = k ** 2 * _FS_DK / (2.0 * np.pi ** 2)
    g = np.array([1.0, 5.0])
    out = []
    for i, z in enumerate(_FS_Z):
        pcb = np.asarray(fid["emu"].pk_cb(jnp.asarray(k), float(z), PLANCK18))
        pt = ((fid["b"][i] + fid["f"][i] * mu ** 2) ** 2 * pcb[:, None]
              + _FS_INV_NBAR[i])
        # The integral over -1..1 is twice that over 0..1, where sum(w) = 1.
        out.append(np.einsum("a,b,kj,aj,bj,j->kab", g, g, pt ** 2, L, L, 2.0 * w)
                   / nk[:, None, None])
    return np.array(out)


def _fs_setup():
    r"""The forecast both figures share, built and checked once.

    Returns ``t0``, ``b0``, the fiducial multipoles ``m0``, the Jacobian ``J``
    ``(6, 2, nk, 14)`` in ``_FS_P`` then the six :math:`b_1`, the inverse
    covariances at unit volume ``cinv1`` and ``vs2`` = :math:`V\sigma^2` per
    sample, the volume that puts the monopole's error at ``_FS_K40`` at
    :math:`\sigma`.
    """
    if "setup" in _FS_CACHE:
        return _FS_CACHE["setup"]
    fid = _fs_fid()
    P, c, emu = _FS_P, PLANCK18, fid["emu"]
    npar, ntr = len(P), len(_FS_Z)
    t0 = np.array([float(getattr(c, p)) for p in P])
    b0, f0 = fid["b"], fid["f"]
    m0 = np.asarray(_fs_multipoles(jnp.asarray(t0), jnp.asarray(b0)))

    # At the fiducial alpha = 1, and the quadrature must reproduce Kaiser's
    # closed forms -- the check on the mu integral and on the moments.
    dev = 0.0
    for i, z in enumerate(_FS_Z):
        pcb = np.asarray(emu.pk_cb(jnp.asarray(_FS_K), float(z), c))
        want = ((b0[i] ** 2 + 2 * b0[i] * f0[i] / 3 + f0[i] ** 2 / 5) * pcb,
                (4 * b0[i] * f0[i] / 3 + 4 * f0[i] ** 2 / 7) * pcb)
        for ell in (0, 1):
            assert np.allclose(m0[i, ell], want[ell], rtol=1e-12, atol=0.0)
            dev = max(dev, np.max(np.abs(m0[i, ell] / want[ell] - 1.0)))
    T0 = np.concatenate([t0, b0])
    J = np.asarray(jax.jacfwd(
        lambda T: _fs_multipoles(T[:npar], T[npar:]))(jnp.asarray(T0)))
    # The amplitude enters every multipole as exp(ln10A_s): its column is the
    # model itself.  f is a ratio of two sigma8, so it cannot carry A_s.
    ia = P.index("ln10A_s")
    amp = float(np.max(np.abs(J[..., ia] / m0 - 1.0)))
    assert amp < 1e-10, amp
    # Each b1 reaches its own sample only, as 2 b I0 + 2 f I1.
    I0, _ = _fs_basis(jnp.asarray(t0))
    I0 = np.asarray(I0)
    for i in range(ntr):
        want = 2 * b0[i] * I0[i, 0] + 2 * f0[i] * I0[i, 1]
        assert np.allclose(J[i, ..., npar + i], want, rtol=1e-12, atol=0.0)
        others = np.delete(J[..., npar + i], i, axis=0)
        assert not np.any(others), i

    cinv1 = np.linalg.inv(_fs_cov_unit(_FS_K))
    c40 = _fs_cov_unit(np.array([_FS_K40]))[:, 0, 0, 0]
    p40 = np.asarray(_fs_multipoles(jnp.asarray(t0), jnp.asarray(b0),
                                    k=np.array([_FS_K40])))[:, 0, 0]
    vs2 = c40 / p40 ** 2
    _FS_CACHE["setup"] = dict(t0=t0, b0=b0, f0=f0, m0=m0, J=J, cinv1=cinv1,
                              vs2=vs2, T0=T0, kaiser_dev=dev, amp_dev=amp)
    return _FS_CACHE["setup"]


def _fs_fisher(J, cinv1, vs2, sig):
    """Sum over samples and k of J^T C^-1 J, at precision ``sig``."""
    F = 0.0
    for i in range(len(J)):
        F = F + np.einsum("lkp,klm,mkq->pq", J[i], cinv1[i] * vs2[i] / sig ** 2,
                          J[i])
    return F


def _fs_chi2_profiled(t, sig, setup, params=_FS_P):
    r"""``(chi2, b1)``: the exact :math:`\chi^2` of the multipoles at
    cosmology ``t``, minimised over each sample's :math:`b_1`.

    Exact rather than iterated, because :math:`P_\ell` is quadratic in
    :math:`b_1` and so the :math:`\chi^2` of each sample a quartic: its
    minimum over :math:`b_1 > 0` is a root of a cubic.  Eager, so a cosmology
    outside :mod:`emu_pk`'s box raises ``ValueError``.
    """
    I, f = _fs_basis(jnp.asarray(t), params)
    I, f = np.asarray(I), np.asarray(f)
    tot, b = 0.0, []
    for i in range(len(I)):
        Ci = setup["cinv1"][i] * setup["vs2"][i] / sig ** 2

        def q(u, v, _C=Ci):
            return float(np.einsum("lk,klm,mk->", u, _C, v))
        a2, a1 = I[i, 0], 2.0 * f[i] * I[i, 1]
        a0 = f[i] ** 2 * I[i, 2] - setup["m0"][i]
        poly = [q(a2, a2), 2 * q(a2, a1), q(a1, a1) + 2 * q(a2, a0),
                2 * q(a1, a0), q(a0, a0)]
        r = np.roots(np.polyder(poly))
        r = r[(np.abs(r.imag) < 1e-9) & (r.real > 0)].real
        v = np.polyval(poly, r)
        tot += float(v.min())
        b.append(float(r[np.argmin(v)]))
    return tot, np.array(b)


def _refusal(err):
    """A refusal's message, cut to what is out of bounds: emu_hmf and emu_pk
    name every parameter outside the box, and all of them are printed."""
    msg = str(err).splitlines()[0]
    if "training data: " in msg:
        return msg.split("training data: ", 1)[1].split(".  ", 1)[0]
    return msg[:160]


def _fs_validity(cov, sigs, chi2_of, P, t0, tag,
                 fmt=lambda s: f"{100 * s:g}%", note=", b1 profiled,"):
    r"""Exact :math:`\chi^2` at the :math:`\pm1\sigma` points along each
    principal direction of the cosmological covariance ``cov[s]``, where a
    Gaussian likelihood gives 1; the biases are profiled, as they are
    marginalised in :math:`F^{-1}`.  Returns ``(chi2max, refused)``.

    ``fmt`` names a level (a precision by default; a mass threshold for
    Fig. hmf-fisher), ``note`` what was done to the nuisance parameters."""
    print(f"  exact chi^2 ({tag}){note} at the +-1 sigma points along "
          "the principal directions (Fisher: 1):")
    chi2max, refused_at = {}, {}
    for s in sigs:
        w, v = np.linalg.eigh(cov[s])
        vals, refused = [], 0
        for kk in range(len(P)):
            for sgn in (1.0, -1.0):
                t = t0 + sgn * np.sqrt(w[kk]) * v[:, kk]
                try:
                    vals.append((chi2_of(t, s), kk, sgn))
                except ValueError as err:
                    refused += 1
                    step = t - t0
                    print(f"      refused at {fmt(s)}, step in marginal "
                          "sigma: " + ", ".join(
                              f"{p} {step[i] / np.sqrt(cov[s][i, i]):+.2f}"
                              for i, p in enumerate(P))
                          + f" -- {_refusal(err)}")
        lo, hi = min(vals), max(vals)
        chi2max[s], refused_at[s] = hi[0], refused
        print(f"    {fmt(s):>6s}  min {lo[0]:.3f}  max {hi[0]:.3f}  ({refused} "
              f"of {2 * len(P)} points outside the box; max at the "
              f"{'+' if hi[2] > 0 else '-'} end of the step "
              + ", ".join(f"{p} {hi[2] * np.sqrt(w[hi[1]]) * v[i, hi[1]]:+.3g}"
                          for i, p in enumerate(P)) + ")")
    return chi2max, refused_at


def _fs_lims(t0, cov_frame, P, chain):
    """Each axis: 1.7 sigma of ``cov_frame``, widened to hold every DESI
    contour and marginal in it."""
    half = 1.7 * np.sqrt(np.diag(cov_frame))
    lims = []
    for i, p in enumerate(P):
        lo, hi = t0[i] - half[i], t0[i] + half[i]
        v = [poly[:, 0] for q in DF.parameters(chain) if q != p
             for poly in DF.contours(chain, p, q)]
        if v:
            v = np.concatenate(v)
            pad = 0.05 * (v.max() - v.min())
            lo, hi = min(lo, v.min() - pad), max(hi, v.max() + pad)
        lims.append((lo, hi))
    print("  axes: " + "  ".join(f"{p} [{lo:.4g}, {hi:.4g}]"
                                 for p, (lo, hi) in zip(P, lims)))
    return lims


def _fisher_corner(fig, gs, P, t0, cov, sigs, colours, invalid, lims,
                   chain=None, ref=None):
    r"""The lower triangle of a Fisher forecast: Gaussian marginals on the
    diagonal and 68% ellipses (:func:`_fisher_ellipse`) below it, one colour
    per precision, an outline for a precision outside Fisher validity.

    ``chain`` names a :mod:`desi_fs_contours` posterior drawn as outlines in
    the panels it constrains; ``ref = (cov, params, style)`` a reference
    forecast over a subset of ``P``, drawn as an outline.
    """
    npar = len(P)
    # One colour per drawn precision, in order: the reversed loops below
    # would otherwise pair a short ``sigs`` with the palette's last colours.
    colours = tuple(colours)[:len(sigs)]
    assert len(colours) == len(sigs), (sigs, colours)
    ax = {}
    for i in range(npar):
        for j in range(i + 1):
            a = fig.add_subplot(gs[i, j])
            ax[i, j] = a
            a.set_xlim(lims[j])
            a.tick_params(labelsize=5.8, length=2.2)
            # Ticks within 8% of a column's edge go: rotated, their labels run
            # into the neighbouring column's.
            lo, hi = lims[j]
            a.set_xticks([x for x in matplotlib.ticker.MaxNLocator(3).tick_values(lo, hi)
                          if lo + 0.08 * (hi - lo) <= x <= hi - 0.08 * (hi - lo)])
            if i == j:
                a.set_yticks([])
                a.set_ylim(0.0, 1.08)
                x = np.linspace(*lims[j], 400)
                if ref is not None and P[j] in ref[1]:
                    k = ref[1].index(P[j])
                    sd = np.sqrt(ref[0][k, k])
                    a.plot(x, np.exp(-0.5 * ((x - t0[j]) / sd) ** 2), **ref[2])
                for s, col in zip(sigs[::-1], colours[::-1]):
                    sd = np.sqrt(cov[s][j, j])
                    a.plot(x, np.exp(-0.5 * ((x - t0[j]) / sd) ** 2), color=col,
                           lw=1.0, ls="--" if s in invalid else "-")
                if chain and P[j] in DF.parameters(chain):
                    a.plot(*DF.marginal(chain, P[j]), **_FS_DESI_STYLE)
            else:
                a.set_ylim(lims[i])
                a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                if ref is not None and P[i] in ref[1] and P[j] in ref[1]:
                    kk = [ref[1].index(P[j]), ref[1].index(P[i])]
                    _fisher_ellipse(a, (t0[j], t0[i]), ref[0][np.ix_(kk, kk)],
                                    fill=False, ec=ref[2]["color"],
                                    lw=ref[2]["lw"], ls=ref[2]["ls"])
                for s, col in zip(sigs[::-1], colours[::-1]):
                    c2 = cov[s][np.ix_([j, i], [j, i])]
                    if s in invalid:
                        _fisher_ellipse(a, (t0[j], t0[i]), c2, fill=False,
                                        ec=col, lw=1.0, ls="--")
                    else:
                        _fisher_ellipse(a, (t0[j], t0[i]), c2, fc=col,
                                        alpha=0.55, ec=col, lw=0.6)
                if chain:
                    for poly in DF.contours(chain, P[j], P[i]):
                        a.plot(poly[:, 0], poly[:, 1], **_FS_DESI_STYLE)
                a.plot(t0[j], t0[i], "+", color="k", ms=4, mew=0.6)
            if i == npar - 1:
                a.set_xlabel(_DTEX[P[j]], fontsize=7.5)
                for lab in a.get_xticklabels():
                    lab.set_rotation(40)
            else:
                a.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                a.set_ylabel(_DTEX[P[i]], fontsize=7.5)
            else:
                a.tick_params(labelleft=False)
    return ax


def _fs_print_errors(cov, sigs, P, t0, title):
    print(f"  marginal 1-sigma errors, {title}:")
    for s in sigs:
        sd = np.sqrt(np.diag(cov[s]))
        print(f"    {100 * s:5g}%  " + "  ".join(
            f"{p}={x:.4g}" + (f" ({100 * x / abs(v):.3g}%)" if v else "")
            for p, x, v in zip(P, sd, t0)))


def _fs_pivot(c, P, label):
    iw0, iwa = P.index("w0"), P.index("wa")
    c2 = c[np.ix_([iw0, iwa], [iw0, iwa])]
    one_m_a = -c2[0, 1] / c2[1, 1]
    print(f"  pivot {label}: z_p = {1.0 / (1.0 - one_m_a) - 1.0:.3f}, "
          f"sigma(w_p) = {np.sqrt(c2[0, 0] - c2[0, 1] ** 2 / c2[1, 1]):.4g}")


def _fs_sigma8_gradient(t0, P=_FS_P, emu=None):
    """d sigma8 / d theta at z = 0, on the total-matter spectrum (Eq. sigma8)."""
    from ggah_mod.cosmology.amplitude import sigma8
    emu = _fs_fid()["emu"] if emu is None else emu
    kk = jnp.asarray(np.logspace(-4, 2, 512))
    return np.asarray(jax.grad(lambda t: sigma8(
        emu.pk(kk, 0.0, PLANCK18.replace(**dict(zip(P, t)))), kk))(
            jnp.asarray(t0)))


def _fs_match(err_of, target):
    """The precision at which ``err_of(sigma)`` equals ``target``, by
    log-log interpolation on a fine grid (``err_of`` grows with sigma)."""
    g = np.geomspace(3e-4, 0.3, 600)
    e = np.array([err_of(s) for s in g])
    assert np.all(np.diff(e) > 0)
    return float(np.exp(np.interp(np.log(target), np.log(e), np.log(g))))


def _fs_lcdm(setup, F1, prior_fn, g8):
    """``err(sigma) -> {param: sd}`` in flat LambdaCDM (the first five of
    ``_FS_P`` and the six b1) with ``prior_fn``'s Fisher matrix added."""
    npar = len(_FS_P)
    cols = list(range(5)) + list(range(npar, npar + len(_FS_Z)))

    def err(s):
        F = F1 * (0.01 / s) ** 2 + prior_fn()
        C = np.linalg.inv(F[np.ix_(cols, cols)])[:5, :5]
        sd = dict(zip(_FS_P[:5], np.sqrt(np.diag(C))))
        sd["H0"] = 100.0 * sd["h"]
        sd["sigma8"] = float(np.sqrt(g8[:5] @ C @ g8[:5]))
        return sd
    return err


# ---- Sec. 4.3: what counting every halo constrains ---------------------------

#: The mock halo survey of :func:`fig_hmf_fisher`: every halo above a mass
#: threshold, counted with perfect completeness and purity and its M200m known
#: exactly, in 0.1-dex bins from the threshold to 10^15.5 Msun/h, and in four
#: redshift shells over 20,000 deg^2.  CsstHMF (no longer used: the default is tinker08)
#: against, tabulates its simulations' M200m function only to 10^15.2 (z <= 0.1),
#: 10^15.1 (0.25) and 10^14.9 (0.5) -- CEmulator HMFRockstarM200m_gp.mhmax_ind
#: -- and extrapolates above.
_HMF_AREA_DEG2 = 20000.0
_HMF_SHELLS = ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5))
_HMF_THRESH = (5e13, 1e14, 3e14, 5e14)
_HMF_DLOGM = 0.1
_HMF_LOGM_MAX = 15.5
#: Gauss-Legendre nodes per shell in z and per bin in ln M; doubling both moves
#: every forecast error by < 1e-3 (asserted).
_HMF_NZ, _HMF_NM = 6, 4
#: Flat LambdaCDM, where the counts are forecast (with the BBN and ns10 priors);
#: the Jacobian also carries w0 and wa, which are freed only as a printed test.
_HMF_P = ("Omega_m", "Omega_b", "h", "n_s", "ln10A_s")
_HMF_PX = _HMF_P + ("w0", "wa")
#: The corner's axes: sigma8 in place of ln10A_s, through its Jacobian.
_HMF_PS8 = ("Omega_m", "Omega_b", "h", "n_s", "sigma8")
#: Sec. 2.2.3's four, tightest (lowest threshold) first.
_HMF_COLOUR = _BAO_COLOUR
#: Multipoles of the cross-shell super-sample matrix, summed exactly; its
#: diagonal agrees with ggah_mod's slab_sigma2_b (exact + Limber) to < 1e-3
#: (asserted).
_HMF_ELL = 200
#: Its wavenumber grid: to 1 h/Mpc, 32 nodes per period of the squared radial
#: integral of the thinnest shell.  ggah_mod's slab default (0.5 and 16) is
#: up to 1.5e-3 low here, so the reference is recomputed at these settings;
#: 64 nodes and 1.5 h/Mpc move S by < 2e-3 (measured once, 2026-10-02, 400 s).
_HMF_SSC_KMAX, _HMF_SSC_KPP = 1.0, 32
_HMF_CACHE = {}


def _hmf_label(th):
    """``5e13`` -> ``"5e13"``, the name of a threshold in the printout."""
    m, e = f"{th:.0e}".split("e")
    return f"{m}e{int(e)}"


def _hmf_bins(threshold, nm=_HMF_NM):
    r"""``(lo, nodes, weights)``: the lower edges of the 0.1-dex bins in
    :math:`\log_{10}M`, and Gauss-Legendre nodes (in :math:`\log_{10}M`) and
    weights (in :math:`\ln M`) inside each, shape ``(nbin, nm)``."""
    lt = np.log10(threshold)
    nb = int(np.floor((_HMF_LOGM_MAX - lt) / _HMF_DLOGM + 1e-9))
    lo = lt + _HMF_DLOGM * np.arange(nb)
    x, w = np.polynomial.legendre.leggauss(nm)
    half = 0.5 * _HMF_DLOGM
    nodes = lo[:, None] + half * (x[None, :] + 1.0)
    return lo, nodes, np.broadcast_to(half * w * np.log(10.0), nodes.shape).copy()


def _hmf_znodes(nz=_HMF_NZ):
    """Gauss-Legendre nodes and weights in z, shell after shell."""
    x, w = np.polynomial.legendre.leggauss(nz)
    return (np.concatenate([0.5 * (b - a) * (x + 1.0) + a for a, b in _HMF_SHELLS]),
            np.concatenate([0.5 * (b - a) * w for a, b in _HMF_SHELLS]))


def _hmf_model(emu, nz=_HMF_NZ, nm=_HMF_NM):
    r"""``(counts, per_z)``.

    ``counts(t)`` returns, per threshold, the expected number of haloes
    (shells x bins) and their count-weighted bias at the values ``t`` of
    ``_HMF_PX``,

    .. math:: N_{ai} = \Omega\int_a dz\,\frac{dV}{dz\,d\Omega}
                       \int_i d\ln M\,M\frac{dn}{dM},

    with :math:`dn/dM` of Eq. (eq:dndm), ``tinker08``,
    on emu_pk's cold spectrum, and ``tinker10``'s bias at 200m (Sec. 3.3's
    pair).  ``per_z(cosmo, z)`` is the chain of `make_field` at one redshift
    on every mass node -- loose, so that `jax.vmap` takes it over the redshift
    nodes; :func:`fig_hmf_fisher` asserts it equal to `make_fields`.  The k
    grid is the backend's, so the two read the same quadrature.
    """
    from ggah_mod.cosmology.background import comoving_volume_element
    from ggah_mod.halos.linear_bias import bias_tinker10
    from ggah_mod.numerics import log_grid
    B = DIFFERENTIABLE
    k = jnp.asarray(log_grid(B.k_min, B.k_max, B.n_k))
    zn, wz = _hmf_znodes(nz)
    bins = [_hmf_bins(th, nm) for th in _HMF_THRESH]
    m = jnp.asarray(10.0 ** np.concatenate([b[1].ravel() for b in bins]))
    omega = _HMF_AREA_DEG2 * (np.pi / 180.0) ** 2

    def per_z(c, z):
        pcb = emu.pk_cb(k, z, c)
        s = H.sigma_of_mass(m, k, pcb, c.rho_cold)
        d = H.dln_sigma_dln_mass(m, k, pcb, c.rho_cold)
        dn = H.dndm(m, s, d, c.rho_cold, model=_HALO_CHOICE["hmf_model"], z=z)
        return m * dn, bias_tinker10(s, 200.0)

    def counts(t):
        c = PLANCK18.replace(**dict(zip(_HMF_PX, t)))
        dn, b = jax.vmap(lambda z: per_z(c, z))(jnp.asarray(zn))
        w = (omega * jnp.asarray(wz)
             * comoving_volume_element(jnp.asarray(zn), c))[:, None]
        shells = len(_HMF_SHELLS)
        n_out, b_out, off = [], [], 0
        for lo, nodes, wm in bins:
            sl = slice(off, off + nodes.size)
            off += nodes.size
            wmf = jnp.asarray(wm.ravel())
            nn = jnp.sum((dn[:, sl] * w).reshape(shells, nz, -1), axis=1) * wmf
            bb = jnp.sum((b[:, sl] * dn[:, sl] * w).reshape(shells, nz, -1),
                         axis=1) * wmf
            n = nn.reshape(shells, len(lo), nm).sum(-1)
            n_out.append(n)
            b_out.append(bb.reshape(shells, len(lo), nm).sum(-1) / n)
        return n_out, b_out
    return counts, dict(per_z=per_z, k=k, m=m, zn=zn, bins=bins, omega=omega)


def _hmf_refuse(c, emu):
    """Raise ``ValueError`` where the model is not defined: outside emu_pk's
    box (a concrete check, not under a trace).  tinker08 has no box."""
    emu.pk_cb(jnp.asarray([0.1]), 0.3, c)


def _hmf_ssc(emu, ell_max=_HMF_ELL, n_chi=96, k_max=_HMF_SSC_KMAX,
             k_per_period=_HMF_SSC_KPP):
    r"""``(S, geometry)``: the covariance of the mean linear density contrast
    of the cold field over the footprint in shells ``a`` and ``b``,

    .. math::

        S_{ab} = \frac{2}{\pi}\sum_\ell W_\ell\int dk\,k^2
                 \sqrt{P_{\rm L}(k, z_a)P_{\rm L}(k, z_b)}\,
                 \frac{R_a R_b}{v_a v_b},\qquad
        R_a = \int_a d\chi\,\chi^2 \frac{D(\chi)}{D(z_a)}\,j_\ell(k\chi),

    :math:`z_a` the middle of shell ``a``, :math:`W_\ell` the mask power of a
    disc of the survey's area (`spherical_cap`), :math:`v_a = \int_a\chi^2d\chi`.
    The diagonal is ggah_mod's `slab_sigma2_b` -- the spectrum taken inside
    each shell, because the cold field's growth is scale-dependent with
    neutrinos -- which this extends to pairs of shells; it is summed exactly
    to ``ell_max``, since Limber's approximation has no cross-shell term at
    all.  Fiducial only, as the covariance is.
    """
    from ggah_mod.cosmology.background import comoving_distance
    from ggah_mod.covariance.geometry import spherical_cap
    from scipy import special
    fsky = _HMF_AREA_DEG2 * (np.pi / 180.0) ** 2 / (4.0 * np.pi)
    geo = spherical_cap(fsky)
    wl = np.asarray(geo.mask_wl)
    assert wl.size > ell_max, (wl.size, ell_max)
    c = PLANCK18
    zz = np.linspace(0.0, 0.6, 601)
    chiz = np.asarray(comoving_distance(jnp.asarray(zz), c))
    kref = jnp.asarray([1e-4])
    pz = np.asarray(jax.vmap(lambda z: emu.pk_cb(kref, z, c)[0])(jnp.asarray(zz)))
    dz = np.sqrt(pz / pz[0])
    # the squared radial integral oscillates with period pi / (shell depth)
    depth = min(float(np.diff(np.interp(s, zz, chiz))[0]) for s in _HMF_SHELLS)
    dk = np.pi / (k_per_period * depth)
    kg = np.concatenate([np.geomspace(1e-4, 1e-3, 24, endpoint=False),
                         np.arange(1e-3, k_max + dk, dk)])
    kern = (2.0 / np.pi) * kg ** 2 * np.gradient(kg)
    x, w = np.polynomial.legendre.leggauss(n_chi)
    chis, wts, vols, amp = [], [], [], []
    for a, b in _HMF_SHELLS:
        c1, c2 = np.interp([a, b], zz, chiz)
        chi = 0.5 * (c2 - c1) * (x + 1.0) + c1
        wc = 0.5 * (c2 - c1) * w
        zr = 0.5 * (a + b)
        chis.append(chi)
        wts.append(wc * chi ** 2 * np.interp(chi, chiz, dz)
                   / np.interp(zr, zz, dz))
        vols.append(float(np.sum(wc * chi ** 2)))
        amp.append(np.sqrt(np.asarray(emu.pk_cb(jnp.asarray(kg), zr, c))))
    amp = np.stack(amp)
    S = np.zeros((len(_HMF_SHELLS),) * 2)
    for ell in range(ell_max + 1):
        R = np.stack([special.spherical_jn(ell, np.outer(kg, chi)) @ wt
                      for chi, wt in zip(chis, wts)]) * amp
        S += wl[ell] * (R * kern) @ R.T
    return S / np.outer(vols, vols), geo


def _hmf_cov(n, b, S):
    """``(Poisson, Poisson + super-sample)`` covariance of one threshold's
    counts, flattened shell after shell; the response of a count to a mode
    longer than the survey is its bias times itself."""
    r = (b * n).ravel()
    sh = np.repeat(np.arange(n.shape[0]), n.shape[1])
    cp = np.diag(n.ravel())
    return cp, cp + np.outer(r, r) * S[np.ix_(sh, sh)]


def _hmf_prior(P, t):
    """BBN on omega_b and ns10 on n_s, as a Fisher matrix in ``P``."""
    F = _bbn_prior(t, P)
    if "n_s" in P:
        F[P.index("n_s"), P.index("n_s")] += 1.0 / _NS10[1] ** 2
    return F


def _hmf_setup():
    """Counts, Jacobian, covariances and Fisher matrices, computed once."""
    if "setup" in _HMF_CACHE:
        return _HMF_CACHE["setup"]
    emu = make_pk("emu_pk")
    counts, aux = _hmf_model(emu)
    t0 = np.array([float(getattr(PLANCK18, p)) for p in _HMF_PX])
    n0, b0 = counts(jnp.asarray(t0))
    n0 = [np.asarray(x) for x in n0]
    b0 = [np.asarray(x) for x in b0]
    J = jax.jacfwd(lambda t: counts(t)[0])(jnp.asarray(t0))
    J = [np.asarray(x).reshape(-1, len(_HMF_PX)) for x in J]
    S, geo = _hmf_ssc(emu)
    C = [_hmf_cov(n, b, S) for n, b in zip(n0, b0)]
    F = {th: J[i].T @ np.linalg.solve(C[i][1], J[i])
         for i, th in enumerate(_HMF_THRESH)}
    Fp = {th: J[i].T @ (J[i] / n0[i].ravel()[:, None])
          for i, th in enumerate(_HMF_THRESH)}
    out = dict(emu=emu, counts=counts, aux=aux, t0=t0, n0=n0, b0=b0, J=J, S=S,
               geo=geo, C=C, F=F, Fp=Fp)
    _HMF_CACHE["setup"] = out
    return out


def _hmf_errors(F, P, t0, prior=True):
    """The covariance over ``P`` (a prefix-free subset of ``_HMF_PX``) of the
    Fisher matrix ``F`` over ``_HMF_PX``, with the BBN and ns10 priors."""
    idx = [_HMF_PX.index(p) for p in P]
    Fs = F[np.ix_(idx, idx)]
    if prior:
        Fs = Fs + _hmf_prior(P, t0[idx])
    return np.linalg.inv(Fs)


