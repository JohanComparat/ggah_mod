r"""X-ray cooling: :math:`\Lambda(T, Z)`, tabulated once and read differentiably.

The X-ray emissivity of a hot plasma is

.. math::  \varepsilon(r) = n_e(r)\,n_H(r)\,\Lambda_{\rm APEC}(T, Z)
                          = n_e^2(r)\,\Lambda(T, Z),
           \quad \Lambda \equiv (n_H/n_e)\,\Lambda_{\rm APEC}

with the ratio :math:`n_H/n_e \simeq 0.83` for a fully ionised solar-abundance
plasma.  Folding it into :math:`\Lambda` is what lets the profile carry a single
:math:`n_e^2`, and it is a **parameter** here rather than a literal: abundance
is one of the things a gas fit varies, and the predecessor had it fixed at 0.83
in a private constant.

APEC itself is an atomic-physics calculation, not something to differentiate
through.  So it is evaluated once onto a :math:`(\log T, \log Z)` grid, stored,
and interpolated -- and the interpolation is where the care goes.

Why the interpolation is not bilinear
-------------------------------------

The predecessor interpolated the stored table bilinearly, with ``jnp.clip`` on
both axes to hold queries inside the grid.  Two problems, and the second is the
one that matters here:

* ``clip`` puts a **tie** at each grid edge, and JAX splits the gradient of a
  ``minimum`` 50/50 across a tie.  This package has already been bitten by
  exactly that -- the neutrino-mass derivative was wrong by a factor of two for
  the same reason -- and :func:`ggah_mod.numerics.lin_weights` exists to hold
  the fix.
* bilinear interpolation is :math:`C^0`: the derivative **jumps** at every node.
  A tabulated :math:`\Lambda` is smooth in reality, so those jumps are pure
  artefact, and they land in :math:`\partial P_{XX}/\partial\theta` wherever a
  halo's temperature happens to sit on a node.

So the temperature axis -- the one the emissivity is most sensitive to, and the
one a halo's mass maps onto -- is interpolated with the **monotone cubic**
(Fritsch--Carlson) route, which is :math:`C^1` and cannot overshoot into a
negative emissivity.  Monotone rather than natural-cubic for that reason: a
cooling function must stay positive, and a spline that rings does not.

Falling back
------------

When the table has not been built, :func:`lambda_powerlaw` is available and says
so.  It is a two-parameter fit, good to a factor of a few, and it is **not**
silently substituted: :func:`make_cooling` returns it only when asked by name.
A cooling function that quietly degrades is how an X-ray amplitude ends up
absorbing an atomic-physics error.
"""

from __future__ import annotations

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics import hermite as _hermite, lin_weights

__all__ = ["lambda_powerlaw", "ApecCooling", "ApecCoolingWide", "BandCooling",
           "load", "load_band", "build", "build_wide", "build_band",
           "make_cooling", "COOLING", "NH_OVER_NE", "DATA", "WIDE_TABLE",
           "N_E", "E_RANGE", "band_table_path"]

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "apec"


def band_table_path() -> pathlib.Path:
    """Where the band-resolved table is cached.

    ``$GGAH_APEC_CACHE`` if set, else ``$XDG_CACHE_HOME/ggah_mod``, else
    ``~/.cache/ggah_mod``.  A cache and not ``data/``: :class:`BandCooling`
    carries the reason this one table is not shipped.
    """
    import os
    env = os.environ.get("GGAH_APEC_CACHE")
    if env:
        return pathlib.Path(env) / "apec_band.npz"
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = pathlib.Path(xdg) if xdg else pathlib.Path.home() / ".cache"
    return root / "ggah_mod" / "apec_band.npz"

#: :math:`n_H/n_e` for a fully ionised solar-abundance plasma.
#:
#: A default, not a constant: it depends on abundance, which a gas fit varies.
#: The predecessor had it as a private literal inside the cooling class, which
#: is why it could not be varied.
NH_OVER_NE = 0.83


# =========================================================================
# The analytic fallback
# =========================================================================

@jax.jit
def lambda_powerlaw(kt_kev, z_metal, lambda_0=3.0e-23, alpha_t=0.5,
                    alpha_z=1.0, z_ref=0.3):
    r""":math:`\Lambda = \Lambda_0 (kT/{\rm keV})^{\alpha_T}(Z/Z_{\rm ref})^{\alpha_Z}`.

    Bremsstrahlung-like, :math:`\alpha_T = 1/2`.  Good to a factor of a few over
    0.5--10 keV and wrong below ~1 keV where line emission dominates and the
    true :math:`\Lambda` *rises* as :math:`T` falls.  Useful as a null test and
    for a run with no atomic data available; never as a silent substitute.
    """
    kt = jnp.maximum(jnp.asarray(kt_kev), 1e-3)
    z = jnp.maximum(jnp.asarray(z_metal), 1e-3)
    return lambda_0 * jnp.power(kt, alpha_t) * jnp.power(z / z_ref, alpha_z)


# =========================================================================
# The tabulated version
# =========================================================================

def _pchip_slopes(x, y):
    """Fritsch--Carlson monotone-cubic slopes along the leading axis.

    Numpy, at construction: the table is static and this runs once when it is
    loaded, so nothing here is on the traced path.
    """
    h = np.diff(x)
    shape = (-1,) + (1,) * (y.ndim - 1)
    delta = np.diff(y, axis=0) / h.reshape(shape)
    m = np.zeros_like(y)
    m[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    m[0], m[-1] = delta[0], delta[-1]
    # Zero the slope wherever the data turn over, and limit it to three times
    # the secant: this is what makes the spline monotone, hence unable to
    # overshoot to a negative cooling rate.
    sign_change = delta[:-1] * delta[1:] <= 0.0
    m[1:-1] = np.where(sign_change, 0.0, m[1:-1])
    for i in range(len(h)):
        lo, hi = m[i], m[i + 1]
        d = delta[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(d != 0.0, lo / np.where(d != 0.0, d, 1.0), 0.0)
            b = np.where(d != 0.0, hi / np.where(d != 0.0, d, 1.0), 0.0)
        s = np.sqrt(a ** 2 + b ** 2)
        tau = np.where(s > 3.0, 3.0 / np.maximum(s, 1e-30), 1.0)
        m[i] = np.where(s > 3.0, tau * a * d, m[i])
        m[i + 1] = np.where(s > 3.0, tau * b * d, m[i + 1])
    return m


class ApecCooling:
    r"""Band-integrated :math:`\Lambda(T, Z)` from an APEC table.

    Differentiable in :math:`T` and :math:`Z`, :math:`C^1` in :math:`\log T`.
    """

    name = "apec"
    differentiable = True

    def __init__(self, table=None, nh_over_ne: float = NH_OVER_NE):
        t = load() if table is None else table
        self._lt = jnp.asarray(t["log10_kt"])
        self._lz = jnp.asarray(t["log10_z"])
        # log10 Lambda, so the interpolation is in the variable that is smooth
        # over four decades rather than in one that is not.
        self._loglam = jnp.asarray(t["log10_lambda"])
        self._slopes = jnp.asarray(_pchip_slopes(
            np.asarray(t["log10_kt"]), np.asarray(t["log10_lambda"])))
        self._nh_over_ne = nh_over_ne
        self.band = (float(t["emin"]), float(t["emax"]))

    def __call__(self, kt_kev, z_metal, nh_over_ne=None):
        r""":math:`\Lambda(T,Z)` such that :math:`\varepsilon = n_e^2\Lambda`."""
        lt = jnp.log10(jnp.maximum(jnp.asarray(kt_kev), 1e-30))
        lz = jnp.log10(jnp.maximum(jnp.asarray(z_metal), 1e-30))
        lt, lz = jnp.broadcast_arrays(lt, lz)

        # Monotone cubic in log T (C1), linear in log Z.  Both clamp the
        # *query*, not the fraction -- `lin_weights` carries the reason.
        i, t = lin_weights(lt, self._lt)
        j, u = lin_weights(lz, self._lz)

        h = self._lt[i + 1] - self._lt[i]

        def col(jj):
            return _hermite(t, self._loglam[i, jj], self._loglam[i + 1, jj],
                            self._slopes[i, jj], self._slopes[i + 1, jj], h)

        lam = jnp.power(10.0, (1.0 - u) * col(j) + u * col(j + 1))
        ratio = self._nh_over_ne if nh_over_ne is None else nh_over_ne
        return ratio * lam


@functools.lru_cache(maxsize=1)
def load(path=None):
    """Read the distilled table.

    Kept as **numpy**, deliberately.  A ``jnp`` array cached during a ``jit``
    trace carries that trace with it, and the next transformation dies with an
    ``UnexpectedTracerError`` -- the same reason
    :func:`emu_pk.ratio.load` is numpy.
    """
    p = pathlib.Path(path) if path is not None else DATA / "apec_cooling.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"the APEC cooling table is not present at {p}.  Regenerate it "
            f"with:\n    python -m ggah_mod.sectors.cooling\n"
            f"which needs `soxs` and its atomic data "
            f"(soxs.download_spectrum_tables('apec')).")
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def build(out=None, emin=0.5, emax=2.0, n_t=64, kt_min=0.08, kt_max=30.0,
          n_z=16, z_min=0.02, z_max=3.0, nbins=1000):
    r"""Regenerate the table from APEC.  Accurate path: numpy and soxs.

    The grid is wider than the predecessor's on both axes -- 0.08--30 keV
    rather than 0.08--20, and down to 0.02 :math:`Z_\odot` rather than 0.05 --
    because the clamp at the edge is where a derivative dies, and the massive
    end of the halo mass function reaches past 20 keV once the temperature
    normalisation is right.
    """
    import soxs

    kt = np.logspace(np.log10(kt_min), np.log10(kt_max), n_t)
    zz = np.logspace(np.log10(z_min), np.log10(z_max), n_z)
    agen = soxs.ApecGenerator(emin, emax, nbins, broadening=False)

    lam = np.zeros((n_t, n_z))
    for i, t in enumerate(kt):
        for j, z in enumerate(zz):
            sp = agen.get_spectrum(t, z, redshift=0.0, norm=1.0)
            # The APEC normalisation is `norm = 1e-14 EM / (4 pi D_A^2)`.  At
            # norm = 1, D_A = 1 cm and z = 0 that is EM = 4 pi x 1e14 cm^-5, so
            # the band-integrated energy flux is `Lambda_APEC x 1e14` and the
            # 1e14 has to come back out.  Dropping it is a factor 1e14 that no
            # shape comparison would reveal -- the ratio stays flat in T and Z,
            # so every *relative* test passes.
            lam[i, j] = float(sp.total_energy_flux.value) / 1e14

    # `n_H/n_e` is deliberately NOT folded in here, unlike the predecessor's
    # table: it depends on abundance, which a gas fit varies, and a table with
    # it baked in cannot be re-weighted without regenerating APEC.

    out = pathlib.Path(out) if out is not None else DATA / "apec_cooling.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, log10_kt=np.log10(kt), log10_z=np.log10(zz),
        log10_lambda=np.log10(np.maximum(lam, 1e-40)),
        emin=emin, emax=emax, apec=str(soxs.__version__))
    return out


#: The shipped table's temperature nodes, extended **downwards** with the same
#: log step: ``WIDE_EXTRA`` more nodes take 0.08 keV to 0.0101 keV.
SHIPPED_KT = (0.08, 30.0, 64)
WIDE_EXTRA = 22
WIDE_TABLE = DATA / "apec_cooling_wide.npz"


def _wide_grid():
    """``(kt_min, n_t)`` for the wide table: the shipped nodes plus 22 below."""
    lo, hi, n = SHIPPED_KT
    step = (np.log10(hi) - np.log10(lo)) / (n - 1)
    return 10.0 ** (np.log10(lo) - WIDE_EXTRA * step), n + WIDE_EXTRA


def build_wide(out=None, **kw):
    r"""The 0.5--2 keV table down to 0.0101 keV, on the shipped nodes and below.

    **Why it exists.**  :class:`ApecCooling` clamps its query at 0.08 keV, so
    gas cooler than that keeps the 0.08 keV emissivity.  In the rest-frame
    0.5--2 keV band APEC falls by orders of magnitude below it: on soxs 5.3.0,
    :math:`\log_{10}\Lambda` is -24.8 at 0.08 keV and -33.3 at 0.02 keV, and
    :math:`d\ln\Lambda/d\ln T` is already +6 at the edge.  The clamp is a flat
    direction for any fit that lowers :math:`kT = P_e/n_e` -- a joint tSZ +
    X-ray fit can trade density against pressure there and the X-ray never
    objects.

    A **new named table**, not a rebuilt ``apec``: regenerating the shipped
    file would move the monotone-cubic slope at its first node and change every
    stored result that touched 0.08--0.088 keV.  The nodes are the shipped ones
    plus ``WIDE_EXTRA`` below, so the two tables agree exactly wherever both
    interpolations are defined by the same four nodes.
    """
    kt_min, n_t = _wide_grid()
    return build(out=WIDE_TABLE if out is None else out, kt_min=kt_min,
                 kt_max=SHIPPED_KT[1], n_t=n_t, **kw)


class ApecCoolingWide(ApecCooling):
    r""":class:`ApecCooling` on the wide table: clamps at 0.0101 keV, not 0.08.

    Identical to ``apec`` above :math:`kT \approx 0.088` keV, the shipped
    table's second node, and below it follows APEC down instead of holding the
    0.08 keV value.  Selected by name, ``make_cooling("apec_wide")``.
    """

    name = "apec_wide"

    def __init__(self, table=None, nh_over_ne: float = NH_OVER_NE):
        if table is None:
            if not WIDE_TABLE.exists():
                raise FileNotFoundError(
                    f"the wide APEC table is not present at {WIDE_TABLE}.  "
                    f"Build it with:\n    python -m ggah_mod.sectors.cooling "
                    f"--wide\nwhich needs `soxs` and its atomic data.")
            table = load(WIDE_TABLE)
        super().__init__(table=table, nh_over_ne=nh_over_ne)


# =========================================================================
# Any band, in the observer's frame
# =========================================================================
#
# :class:`ApecCooling` above is band-integrated *at build time*, so ``emin`` and
# ``emax`` are properties of the file, and the observer frame has never been
# handled at all: :func:`build` calls ``get_spectrum(..., redshift=0.0)``.  Two
# consequences that look unrelated and are one gap seen from two layers.  Here,
# a caller wanting 0.1--2.4 keV has to regenerate the table.  In
# :mod:`ggah_mod.spectra.tracers`, an AGN emissivity in a *named* band raises,
# because there is no object in this package that could answer for one.
#
# What follows is that object.
#
# Which function to tabulate
# --------------------------
#
# ``PLAN.md``'s Tier 7 entry asks for a differential :math:`\Lambda(T, Z, E)`
# integrated at runtime.  A **cumulative** one is strictly better, for a reason
# worth writing down: store
#
# .. math::  C(<E \mid T, Z) = \int_{E_0}^{E} \frac{d\Lambda}{dE'}\,dE'
#
# and a band is the *exact* difference :math:`C(<E_2) - C(<E_1)` at the nodes,
# where a differential table would be integrating an already-binned spectrum a
# second time.  :math:`C` is also monotone by construction, so the
# monotone-cubic route this module already owns is the natural interpolant here
# rather than a compromise.
#
# The K-correction falls out rather than being applied
# ----------------------------------------------------
#
# A band :math:`[E_1, E_2]` observed from a source at redshift :math:`z` is the
# rest-frame band :math:`[E_1(1+z), E_2(1+z)]`.  An observer-frame band is a
# change of *integration limits*, so there is no K-correction to apply, get
# wrong, or forget.  That is the whole argument for tabulating in energy: the
# alternative is one table per band per redshift.
#
# Three things that could have set the accuracy, and the one that did
# -------------------------------------------------------------------
#
# Each was measured rather than reasoned about, and the first two -- the
# obvious two -- turned out not to matter:
#
# * **The energy grid.**  At :data:`N_E` = 8192 log-spaced nodes the raw
#   cumulative reproduces a direct ``soxs`` band to **7e-5**, worst case, at
#   0.1 keV and solar abundance where the spectrum is most line-dominated.
#   Not the limit.
# * **float32 storage** of the fraction: **1e-6**.  Not the limit either, and
#   this one was a wrong guess that a measurement corrected -- the reasoning
#   *"a thin band is a difference of two numbers near unity, so single
#   precision will cancel"* is sound arithmetic about the wrong step.
# * **The order of subtraction and interpolation**, which is the whole
#   algorithm.  Reading the cumulative at :math:`E_1` and :math:`E_2` and
#   interpolating *each* in :math:`(T, Z)` before subtracting was wrong by
#   **4.5 per cent**.  At 0.1 keV the 0.5--2 keV share of 0.05--50 keV is 1.4
#   per cent, because nearly all the flux is softer than the band -- so a
#   per-mille interpolation error on each of two numbers near unity is a seven
#   per cent error on their difference.
#
# So: form the band at each surrounding node, where it is *exact*, and
# interpolate that.  The subtraction happens before any approximation instead
# of after it, and the quantity carried across the cell is the one the caller
# asked for.  On-node the two routes then agree to 5.7e-4 and off-node to the
# same, which is the band treatment alone.
#
# Why this table is cached and not shipped
# ----------------------------------------
#
# 23 MB, against 448 kB for the largest table this package ships and 9 kB for
# the fixed-band one beside it -- and it builds in **23 seconds**, because the
# atomic data is already on disk by the time anyone asks.  A 23 MB wheel to
# save 23 seconds is the wrong trade, so :func:`load_band` refuses with the
# command to build it, exactly as :func:`load` does.


def _band_slopes(log10_e, frac):
    """Fritsch--Carlson slopes along the trailing (energy) axis."""
    flat = np.moveaxis(frac, -1, 0).reshape(frac.shape[-1], -1)
    return np.moveaxis(_pchip_slopes(log10_e, flat), 0, -1).reshape(frac.shape)


#: Energy nodes for the band table.  A line-resolution requirement, not an
#: interpolation-order one -- see above.
N_E = 8192

#: The tabulated energy range, keV.  Wider than any band a halo model asks for,
#: because the *rest-frame* limits move up by :math:`(1+z)`: a 0.5--2 keV
#: observed band at z = 3 is 2--8 keV emitted, and a range stopping at 10 would
#: silently truncate the hard end at high redshift.
E_RANGE = (0.05, 50.0)


class BandCooling:
    r""":math:`\Lambda(T, Z)` in **any** band, in the observer's frame.

    ``band`` is :math:`(E_{\min}, E_{\max})` in keV **as observed** and
    ``z_obs`` is the source redshift; the rest-frame limits are :math:`E(1+z)`.
    Both may be traced, which is the point of tabulating in energy at all -- a
    fit marginalising over a calibration uncertainty in a band edge needs
    :math:`\partial\Lambda/\partial E`, and a table built at one band has a
    structural zero there.

    Both must be **scalars**, and that is a property of the thing rather than a
    limitation: a band belongs to an instrument and ``z_obs`` to the population
    a statistic is measured for, so each is one number per statistic.  A
    per-halo redshift is a different call, not a broadcast.

    Differentiable in :math:`T`, :math:`Z`, both band edges and :math:`z_{\rm
    obs}`; :math:`C^1` in :math:`\log T` and in :math:`\log E`.
    """

    name = "apec_band"
    differentiable = True

    def __init__(self, table=None, nh_over_ne: float = NH_OVER_NE,
                 band=(0.5, 2.0), z_obs=0.0):
        t = load_band() if table is None else table
        self._lt = jnp.asarray(t["log10_kt"])
        self._lz = jnp.asarray(t["log10_z"])
        self._le = jnp.asarray(t["log10_e"])
        #: ``C(<E)/C(<inf)`` on the (T, Z, E) grid, **not** in logs.
        #:
        #: The fixed-band table interpolates ``log10 Lambda`` because it spans
        #: four decades and is smooth in the log.  A cumulative is the opposite
        #: case: it starts at zero, where the log does not exist.
        frac = np.asarray(t["fraction"], dtype=np.float64)
        self._frac = jnp.asarray(frac)
        #: ``log10 C(<inf)``.  The steep part -- all of the temperature
        #: dependence -- kept in float64 while the *shape* is stored in
        #: float32, which is what puts the precision where the dynamic range
        #: is and the cache at 23 MB rather than 100.
        self._logtot = jnp.asarray(
            np.log10(np.maximum(np.asarray(t["total"]), 1e-300)))
        #: Fritsch--Carlson slopes along E, rebuilt here rather than stored.
        #:
        #: Storing them doubled the cache for a pass costing a fraction of a
        #: second, run once per process behind :func:`load_band`'s cache.  It
        #: is also the only place they can be built in float64 *from* float32
        #: data, which is the right order -- the slopes of a rounded array, not
        #: a rounded copy of the slopes.
        self._slopes = jnp.asarray(
            _band_slopes(np.asarray(t["log10_e"]), frac))
        self._nh_over_ne = nh_over_ne
        self.band = (float(band[0]), float(band[1]))
        self.z_obs = float(z_obs)
        self.e_range = (float(10.0 ** np.asarray(t["log10_e"])[0]),
                        float(10.0 ** np.asarray(t["log10_e"])[-1]))

    def _band_nodes(self, le_lo, le_hi):
        r"""``log10 Lambda`` over the band at **every** node.  Shape (NT, NZ).

        Depends on the band alone, so it is one vectorised pass per call and
        not one per query point.
        """
        def cum(le):
            m, c = lin_weights(le, self._le)
            h = self._le[m + 1] - self._le[m]
            return _hermite(c, self._frac[:, :, m], self._frac[:, :, m + 1],
                            self._slopes[:, :, m], self._slopes[:, :, m + 1], h)

        share = jnp.clip(cum(le_hi) - cum(le_lo), 0.0, 1.0)
        # Floored rather than guarded: a band entirely outside the tabulated
        # range is legitimately zero, and the log has to stay finite because a
        # nan in an unused cell still poisons the gradient of a used one.
        return jnp.log10(jnp.maximum(
            jnp.power(10.0, self._logtot) * share, 1e-300))

    @staticmethod
    def _slopes_along_t(lt_grid, y):
        r"""Three-point Hermite slopes along axis 0, in JAX.

        **Unlimited**, where the stored energy slopes are Fritsch--Carlson, and
        :func:`ggah_mod.numerics.hermite_slopes` gives the reason: a monotone
        limiter selects branches with ``min``/``max`` on *values*, and every
        such tie is a gradient split 50/50.  The energy slopes are limited
        safely because they are built once from a static table; these are built
        from a **traced** array -- it moves with the band edges -- so a limiter
        would be active at the fiducial rather than at an edge case.

        Positivity survives dropping it: this interpolates the *log* of the
        band, and exponentiating afterwards is a stronger guarantee than
        monotonicity was.
        """
        h = jnp.diff(lt_grid)[:, None]
        d = jnp.diff(y, axis=0) / h
        interior = (h[1:] * d[:-1] + h[:-1] * d[1:]) / (h[:-1] + h[1:])
        return jnp.concatenate([d[:1], interior, d[-1:]], axis=0)

    def __call__(self, kt_kev, z_metal, nh_over_ne=None, band=None, z_obs=None):
        r""":math:`\Lambda(T,Z)` over the band, so :math:`\varepsilon = n_e^2\Lambda`."""
        e_lo, e_hi = self.band if band is None else band
        one_z = 1.0 + jnp.asarray(self.z_obs if z_obs is None else z_obs)

        lt = jnp.log10(jnp.maximum(jnp.asarray(kt_kev), 1e-30))
        lz = jnp.log10(jnp.maximum(jnp.asarray(z_metal), 1e-30))
        lt, lz = jnp.broadcast_arrays(lt, lz)
        # The rest frame.  This is the K-correction, and it is two products.
        le_lo = jnp.log10(jnp.asarray(e_lo) * one_z)
        le_hi = jnp.log10(jnp.asarray(e_hi) * one_z)

        lgb = self._band_nodes(le_lo, le_hi)
        slopes = self._slopes_along_t(self._lt, lgb)

        # Monotone cubic in log T, linear in log Z: the same interpolant
        # `ApecCooling` uses on the same axes, so the two routes differ in
        # their band treatment and in nothing else -- which is what makes
        # comparing them a test rather than a coincidence.
        i, a = lin_weights(lt, self._lt)
        j, b = lin_weights(lz, self._lz)
        h = self._lt[i + 1] - self._lt[i]

        def col(jj):
            return _hermite(a, lgb[i, jj], lgb[i + 1, jj],
                            slopes[i, jj], slopes[i + 1, jj], h)

        lam = jnp.power(10.0, (1.0 - b) * col(j) + b * col(j + 1))
        ratio = self._nh_over_ne if nh_over_ne is None else nh_over_ne
        return ratio * lam

    def covers(self, band=None, z_obs=None) -> bool:
        """Whether the *rest-frame* band lies inside the tabulated range.

        Concrete-only, deliberately: a caller can ask this before a fit and get
        an answer, and cannot ask it inside ``jit`` and get a silent one.
        """
        e_lo, e_hi = self.band if band is None else band
        one_z = 1.0 + float(self.z_obs if z_obs is None else z_obs)
        return (self.e_range[0] <= float(e_lo) * one_z
                and float(e_hi) * one_z <= self.e_range[1])


@functools.lru_cache(maxsize=1)
def load_band(path=None):
    """Read the cached band table.  Numpy, for the reason :func:`load` gives."""
    p = pathlib.Path(path) if path is not None else band_table_path()
    if not p.exists():
        raise FileNotFoundError(
            f"the band-resolved APEC table is not present at {p}.  Unlike the "
            f"fixed-band table it is not shipped: it is 23 MB, and building it "
            f"takes about 23 seconds once the atomic data is on disk.  Build "
            f"it with:\n    python -m ggah_mod.sectors.cooling --band\n"
            f"which needs `soxs` and its atomic data "
            f"(soxs.download_spectrum_tables('apec')).")
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def build_band(out=None, n_t=64, kt_min=0.08, kt_max=30.0,
               n_z=16, z_min=0.02, z_max=3.0, n_e=N_E, e_range=E_RANGE):
    r"""Tabulate :math:`C(<E \mid T, Z)` and cache it.  numpy and soxs.

    The :math:`(T, Z)` grid is :func:`build`'s, so the two tables can be
    compared node for node -- which ``tests/test_cooling.py`` does, and is the
    only check that says the two routes describe one plasma.
    """
    import soxs

    kt = np.logspace(np.log10(kt_min), np.log10(kt_max), n_t)
    zz = np.logspace(np.log10(z_min), np.log10(z_max), n_z)
    agen = soxs.ApecGenerator(e_range[0], e_range[1], n_e, binscale="log",
                              broadening=False)

    cum = np.zeros((n_t, n_z, n_e + 1))
    for i, t in enumerate(kt):
        for j, z in enumerate(zz):
            sp = agen.get_spectrum(t, z, redshift=0.0, norm=1.0)
            # The 1e14 is the APEC normalisation coming back out -- see
            # `build`, where dropping it was a factor no shape test would show.
            cum[i, j, 1:] = np.cumsum(
                np.asarray(sp.binned_energy_flux.value)) / 1e14
    ebins = np.asarray(agen.get_spectrum(kt[0], zz[0], redshift=0.0,
                                         norm=1.0).ebins.value)

    # Stored as a *fraction* of the total in float32, with the total kept
    # separately in float64.  The fraction is a spectral shape and O(1) across
    # the whole grid, where the cumulative spans the four decades `Lambda`
    # itself does -- so the split puts float64 where the dynamic range is and
    # takes the cache from 100 MB to 23.  Measured cost of the float32 half:
    # 1e-6, two decades below the grid's own error.
    total = cum[:, :, -1].copy()
    safe = np.where(total > 0.0, total, 1.0)
    frac = (cum / safe[:, :, None]).astype(np.float32)

    out = pathlib.Path(out) if out is not None else band_table_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, log10_kt=np.log10(kt), log10_z=np.log10(zz),
        log10_e=np.log10(ebins), fraction=frac, total=total,
        e_min=e_range[0], e_max=e_range[1], apec=str(soxs.__version__))
    return out


#: Cooling functions.  ``apec`` is a class because it owns a table; the
#: power law is a free function.  Selected by name and never substituted.
COOLING = {"apec": ApecCooling, "apec_wide": ApecCoolingWide,
           "apec_band": BandCooling, "powerlaw": lambda_powerlaw}


def make_cooling(name: str = "apec", **kw):
    """Look up a cooling function by name.

    ``apec`` raises if its table is missing rather than degrading to the power
    law.  A cooling function that quietly downgrades is how an atomic-physics
    error ends up absorbed into a fitted X-ray amplitude.
    """
    key = str(name).lower()
    if key not in COOLING:
        raise ValueError(f"unknown cooling function {name!r}; expected one of "
                         f"{sorted(COOLING)}")
    if key in ("apec", "apec_wide", "apec_band"):
        return COOLING[key](**kw)
    return lambda_powerlaw


if __name__ == "__main__":       # pragma: no cover
    import sys
    args = sys.argv[1:]
    print(build_band() if "--band" in args
          else build_wide() if "--wide" in args else build())
