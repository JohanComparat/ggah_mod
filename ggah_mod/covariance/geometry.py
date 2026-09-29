r"""Survey geometry: how much sky, which sky, and how deep.

A covariance is a statement about a survey, not about a model, and the survey
enters it in exactly two ways.  The **amount** of sky and volume sets the number
of independent modes, which every Gaussian term divides by.  The **shape** of
the footprint sets how strongly modes larger than the survey couple into it,
which is the super-sample term, through the power spectrum of the mask.

Both are carried here and nowhere else, so that two blocks of one covariance
cannot be computed for two different surveys.

Two footprints ship
-------------------

:func:`galactic_latitude_cut`
    The complete extragalactic sky, :math:`|b| > b_{\min}`.  Its area is
    :math:`4\pi(1 - \sin b_{\min})`, so :math:`f_{\rm sky} = 0.6580` at
    :math:`20^\circ`, and because the mask depends on Galactic latitude alone
    its harmonic coefficients are analytic -- no pixelisation, no ``anafast``.

:func:`ls10_sample`
    The LS10 volume-limited samples ``ggah_cal`` fits, with the area and
    redshift slab each one's files record.  The areas differ between samples
    (18 703 to 18 978 deg\ :sup:`2`), because each is counted from that
    sample's own randoms, so there is one geometry per sample rather than one
    for the survey.

Mask power, in one convention
-----------------------------

:attr:`SurveyGeometry.mask_wl` is :math:`\sum_m |W_{\ell m}|^2` with
:math:`W_{\ell m} = \Omega^{-1}\int d\Omega\,W(\hat n)Y^*_{\ell m}(\hat n)`, the
coefficients normalised by the footprint's solid angle :math:`\Omega`.  That is
CCL's ``mask_wl`` convention, chosen so the projected super-sample variance
here and CCL's ``sigma2_B_from_mask`` can be compared number for number.  Its
monopole is :math:`1/(4\pi)` for every footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from ..cosmology import background

__all__ = ["SurveyGeometry", "galactic_latitude_cut", "spherical_cap",
           "from_area", "from_healpix_mask", "ls10_sample", "LS10_SAMPLES",
           "DEG2_PER_SR", "FULL_SKY_DEG2", "projected_sigma2_b", "from_randoms"]

#: Square degrees per steradian.
DEG2_PER_SR = (180.0 / np.pi) ** 2
#: The whole sphere, in square degrees.
FULL_SKY_DEG2 = 4.0 * np.pi * DEG2_PER_SR

#: Default multipole reach of an analytic mask power spectrum.
MASK_ELL_MAX = 4096

#: The LS10 x DESI-BGS volume-limited samples, ``M*min -> (area_deg2, z_min,
#: z_max, n_gal)``.
#:
#: Read from the ``joint_covariance`` group attributes of the ``sum_stat`` v2.0
#: joint files (``area_deg2``, ``n_gal``) and ``ggah_cal.datasets.sumstat``'s
#: sample table (the redshift slab), 2026-09-14.  The area is the number of
#: HEALPix ``NSIDE = 64`` pixels holding a random, times the pixel area
#: (``sum_stat/scripts/common.py::area_from_randoms``), so it is the footprint
#: at 0.84 deg resolution and not a mask.  Only the five samples with a joint
#: file are listed: the other four record no area this table could copy.
LS10_SAMPLES = {
    "9.0":  (18887.464, 0.05, 0.08, 523486),
    "10.0": (18973.911, 0.05, 0.18, 2759238),
    "10.5": (18978.108, 0.05, 0.26, 3263228),
    "11.0": (18955.447, 0.05, 0.35, 1619838),
    "11.5": (18702.820, 0.05, 0.35, 120882),
}


@dataclass(frozen=True)
class SurveyGeometry:
    r"""A footprint and, optionally, a redshift slab.

    Frozen and array-free: a geometry is a static choice, and a traced area
    would make the number of modes -- and so the shape of nothing, but the
    meaning of everything -- depend on a parameter.

    Parameters
    ----------
    name : str
    area_sr : float
        Solid angle of the footprint.
    z_min, z_max : float, optional
        The redshift slab, needed by any real-space statistic, whose covariance
        divides by a volume.  A projected statistic takes its depth from its
        kernels instead, and needs neither.
    mask_wl : tuple of float, optional
        :math:`\sum_m|W_{\ell m}|^2` from :math:`\ell = 0`, in the convention of
        the module docstring.  Needed only by the super-sample term; ``None``
        means "not known", and the super-sample term refuses rather than
        assuming a disc.
    provenance : str
        Where the numbers came from.  Required in spirit: a covariance from an
        unattributed area is a covariance nobody can check.
    """

    name: str
    area_sr: float
    z_min: float | None = None
    z_max: float | None = None
    mask_wl: tuple[float, ...] | None = None
    provenance: str = ""

    def __post_init__(self):
        if not 0.0 < float(self.area_sr) <= 4.0 * np.pi * (1.0 + 1e-12):
            raise ValueError(
                f"{self.name}: area_sr = {self.area_sr} is not a solid angle on "
                f"the sphere (0, 4 pi].")
        if (self.z_min is None) != (self.z_max is None):
            raise ValueError(
                f"{self.name}: give both z_min and z_max or neither; half a "
                f"slab has no volume.")
        if self.z_min is not None and not 0.0 <= self.z_min < self.z_max:
            raise ValueError(
                f"{self.name}: z_min = {self.z_min}, z_max = {self.z_max} is "
                f"not a slab.")

    @property
    def f_sky(self) -> float:
        """Fraction of the sphere."""
        return float(self.area_sr) / (4.0 * np.pi)

    @property
    def area_deg2(self) -> float:
        return float(self.area_sr) * DEG2_PER_SR

    def volume(self, cosmo, n: int = 128):
        r"""Comoving volume of the slab, :math:`\Omega\int dz\,dV/(dz\,d\Omega)`
        [(Mpc/h)\ :sup:`3`].

        Integrated over the volume element rather than written as
        :math:`\tfrac43\pi f_{\rm sky}(\chi_{\max}^3-\chi_{\min}^3)`, which holds
        only for a flat cosmology: the element carries the transverse distance,
        so a curved model gets the right volume without a branch here.
        """
        if self.z_min is None:
            raise ValueError(
                f"{self.name} has no redshift slab, so it has no volume. A "
                f"real-space covariance divides by one; build the geometry "
                f"with z_min and z_max.")
        x, w = np.polynomial.legendre.leggauss(n)
        z = 0.5 * (self.z_max - self.z_min) * (x + 1.0) + self.z_min
        dv = background.comoving_volume_element(jnp.asarray(z), cosmo)
        return (float(self.area_sr) * 0.5 * (self.z_max - self.z_min)
                * jnp.sum(jnp.asarray(w) * dv))

    def with_slab(self, z_min: float, z_max: float) -> "SurveyGeometry":
        """The same footprint over another redshift slab."""
        return SurveyGeometry(self.name, self.area_sr, float(z_min),
                              float(z_max), self.mask_wl, self.provenance)


# --------------------------------------------------------------------------
# analytic footprints
# --------------------------------------------------------------------------

def _legendre_up_to(mu: float, ell_max: int) -> np.ndarray:
    r""":math:`P_\ell(\mu)` for :math:`\ell = 0 \ldots \ell_{\max}+1`, by the
    three-term recurrence, which is stable upwards at fixed :math:`\mu`."""
    p = np.empty(ell_max + 2)
    p[0] = 1.0
    p[1] = mu
    for ell in range(1, ell_max + 1):
        p[ell + 1] = ((2 * ell + 1) * mu * p[ell] - ell * p[ell - 1]) / (ell + 1)
    return p


def _cap_integral(mu0: float, ell_max: int) -> np.ndarray:
    r""":math:`I_\ell = \int_{\mu_0}^1 P_\ell(\mu)\,d\mu`, from
    :math:`(2\ell+1)P_\ell = P'_{\ell+1} - P'_{\ell-1}` and
    :math:`P_\ell(1) = 1`."""
    p = _legendre_up_to(mu0, ell_max)
    out = np.empty(ell_max + 1)
    out[0] = 1.0 - mu0
    ell = np.arange(1, ell_max + 1)
    out[1:] = (p[ell - 1] - p[ell + 1]) / (2 * ell + 1)
    return out


def _azimuthal_mask_wl(integral: np.ndarray, area_sr: float) -> tuple:
    r"""Mask power of a footprint symmetric about the pole.

    Only :math:`m = 0` survives, :math:`a_{\ell 0} =
    2\pi\sqrt{(2\ell+1)/4\pi}\int W P_\ell\,d\mu`, and
    :math:`\sum_m|W_{\ell m}|^2` is rotation invariant, so the frame the
    symmetry holds in does not matter.
    """
    ell = np.arange(integral.size)
    a_l0 = 2.0 * np.pi * np.sqrt((2 * ell + 1) / (4.0 * np.pi)) * integral
    return tuple(float(v) for v in (a_l0 / area_sr) ** 2)


def spherical_cap(f_sky: float, *, name: str = "disc", z_min=None, z_max=None,
                  ell_max: int = MASK_ELL_MAX,
                  provenance: str = "") -> SurveyGeometry:
    r"""A disc covering ``f_sky``: :math:`\mu \ge 1 - 2f_{\rm sky}`.

    The shape CCL's ``sigma2_B_disc`` assumes, and so the reference the
    footprints that are not discs are measured against.
    """
    if not 0.0 < f_sky <= 1.0:
        raise ValueError(f"f_sky = {f_sky} is not a fraction of the sky")
    mu0 = 1.0 - 2.0 * f_sky
    area = 4.0 * np.pi * f_sky
    return SurveyGeometry(name, area, z_min, z_max,
                          _azimuthal_mask_wl(_cap_integral(mu0, ell_max), area),
                          provenance or f"analytic spherical cap, f_sky={f_sky}")


def galactic_latitude_cut(b_min_deg: float = 20.0, *, z_min=None, z_max=None,
                          ell_max: int = MASK_ELL_MAX) -> SurveyGeometry:
    r"""The extragalactic sky, :math:`|b| > b_{\min}`, with no other mask.

    With :math:`s = \sin b_{\min}` the footprint is :math:`|\mu| \ge s` in
    Galactic coordinates, so :math:`\Omega = 4\pi(1-s)` and

    .. math::

        \int W P_\ell\,d\mu = \left[1 + (-1)^\ell\right]\int_s^1 P_\ell\,d\mu,

    which vanishes for odd :math:`\ell`: the two caps are mirror images, and a
    mask symmetric under :math:`\mu \to -\mu` has no odd multipoles.

    This is the upper bound on what a ground-based or all-sky extragalactic
    survey can use, not a survey: star masks, bright-source masks and the
    ecliptic are not in it.
    """
    if not 0.0 <= b_min_deg < 90.0:
        raise ValueError(f"b_min = {b_min_deg} deg is not a latitude cut")
    s = float(np.sin(np.deg2rad(b_min_deg)))
    area = 4.0 * np.pi * (1.0 - s)
    parity = 1.0 + (-1.0) ** np.arange(ell_max + 1)
    integral = parity * _cap_integral(s, ell_max)
    return SurveyGeometry(
        f"|b|>{b_min_deg:g}deg", area, z_min, z_max,
        _azimuthal_mask_wl(integral, area),
        f"analytic Galactic latitude cut |b| > {b_min_deg:g} deg, "
        f"f_sky = 1 - sin(b_min) = {1.0 - s:.6f}")


def from_area(area_deg2: float, *, name: str, z_min=None, z_max=None,
              provenance: str) -> SurveyGeometry:
    """A footprint known only by its area.

    Enough for every Gaussian term, and deliberately not enough for the
    super-sample one: ``mask_wl`` is ``None``, so that term refuses rather than
    assuming the area is a disc.
    """
    return SurveyGeometry(name, float(area_deg2) / DEG2_PER_SR, z_min, z_max,
                          None, provenance)


def from_healpix_mask(mask, *, name: str, z_min=None, z_max=None,
                      ell_max: int | None = None,
                      provenance: str) -> SurveyGeometry:
    r"""A footprint from a HEALPix map with values in :math:`[0, 1]`.

    Requires ``healpy``, imported here rather than at module scope because
    nothing else in the package needs it.  ``anafast`` returns
    :math:`C_\ell = \sum_m|a_{\ell m}|^2/(2\ell+1)`, so
    ``mask_wl`` is :math:`(2\ell+1)C_\ell/\Omega^2`.
    """
    import healpy as hp

    mask = np.asarray(mask, dtype=float)
    if mask.min() < 0.0 or mask.max() > 1.0:
        raise ValueError(f"{name}: a mask takes values in [0, 1]")
    nside = hp.npix2nside(mask.size)
    area = float(mask.sum()) * hp.nside2pixarea(nside)
    lmax = 3 * nside - 1 if ell_max is None else int(ell_max)
    cl = hp.anafast(mask, lmax=lmax)
    ell = np.arange(cl.size)
    return SurveyGeometry(name, area, z_min, z_max,
                          tuple(float(v) for v in (2 * ell + 1) * cl / area ** 2),
                          provenance)


def from_randoms(ra, dec, *, nside: int, name: str, z_min=None, z_max=None,
                 min_fraction: float = 0.1, ell_max: int | None = None,
                 provenance: str) -> SurveyGeometry:
    r"""A footprint from a random catalogue, as a fractional HEALPix map.

    Each pixel's coverage is its random count over the count of a fully
    covered pixel, taken as the median over pixels whose eight neighbours are
    all occupied; pixels below ``min_fraction`` of that are dropped, the rule
    ``sys_mapping`` uses to define the footprint.  A catalogue with holes
    inside the footprint (star masks) has a median below a hole-free pixel's,
    so its area is an upper bound by the holes' share of the median pixel.

    The randoms' own shot noise adds a white term
    :math:`\Omega_{\rm pix}A/(4\pi\bar n_{\rm pix})` to the mask power,
    which is subtracted.  It matters only where the mask power itself has
    fallen that low, far above the multipoles a super-sample term reads, and
    it sets how coarse ``nside`` must be for a given random density.
    """
    import healpy as hp

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    counts = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
    occupied = counts > 0
    neighbours = hp.get_all_neighbours(nside, np.flatnonzero(occupied))
    interior = np.flatnonzero(occupied)[np.all(occupied[neighbours], axis=0)]
    if interior.size < 10:
        raise ValueError(f"{name}: too few fully covered pixels at nside "
                         f"{nside} to calibrate the coverage; use a coarser map")
    full = float(np.median(counts[interior]))
    # Not clipped at one: the randoms' Poisson scatter would then be cut on
    # one side only, and the area biased low by about half its amplitude.
    coverage = counts / full
    coverage[coverage < min_fraction] = 0.0
    area = float(coverage.sum()) * hp.nside2pixarea(nside)
    lmax = 3 * nside - 1 if ell_max is None else int(ell_max)
    cl = hp.anafast(coverage, lmax=lmax)
    noise = hp.nside2pixarea(nside) * area / (4.0 * np.pi * full)
    ell = np.arange(cl.size)
    wl = (2 * ell + 1) * np.maximum(cl - noise, 0.0) / area ** 2
    wl[0] = (cl[0]) / area ** 2
    return SurveyGeometry(name, area, z_min, z_max,
                          tuple(float(v) for v in wl), provenance)


def ls10_sample(mstar_min: str) -> SurveyGeometry:
    """One LS10 volume-limited sample: its own area and redshift slab.

    Area only, no mask power: the files record a pixel count, not a map.  The
    super-sample term therefore needs the footprint map, or a stated stand-in
    such as :func:`spherical_cap` of the same area.
    """
    if mstar_min not in LS10_SAMPLES:
        raise KeyError(f"unknown LS10 sample {mstar_min!r}; known: "
                       f"{', '.join(LS10_SAMPLES)}")
    area, z_min, z_max, n_gal = LS10_SAMPLES[mstar_min]
    return from_area(
        area, name=f"LS10 M*>{mstar_min}", z_min=z_min, z_max=z_max,
        provenance=(f"sum_stat v2.0 joint file, joint_covariance.attrs "
                    f"area_deg2 = {area} (NSIDE 64 pixels holding a random), "
                    f"{z_min} < z < {z_max}, N_gal = {n_gal}"))


# --------------------------------------------------------------------------
# the projected super-sample variance
# --------------------------------------------------------------------------

def projected_sigma2_b(geometry: SurveyGeometry, chi, k, p_lin):
    r"""Variance of the linear density field projected over the footprint.

    .. math::

        \sigma_b^2(\chi) = \frac{1}{\chi^2}\sum_\ell
            P_{\rm lin}\!\left(\frac{\ell+\tfrac12}{\chi}\right)
            \sum_m |W_{\ell m}|^2 ,

    per unit comoving distance, which is how the super-sample term of a
    projected statistic consumes it.  CCL's ``sigma2_B_from_mask``, with
    :math:`\chi` in Mpc rather than Mpc/h.

    ``chi`` [Mpc/h], ``k`` [h/Mpc] and ``p_lin`` [(Mpc/h)\ :sup:`3`], one row
    of ``p_lin`` per ``chi``.  The spectrum is interpolated in log-log.
    """
    if geometry.mask_wl is None:
        raise ValueError(
            f"{geometry.name} carries no mask power spectrum, so its "
            f"super-sample variance is not known.  A disc of the same area is "
            f"a choice, not a default: pass `spherical_cap(f_sky)` explicitly "
            f"if that is the approximation wanted.")
    wl = jnp.asarray(np.asarray(geometry.mask_wl))
    ell = jnp.arange(wl.size)
    chi = jnp.atleast_1d(jnp.asarray(chi))
    log_k = jnp.log(jnp.asarray(k))
    log_p = jnp.log(jnp.atleast_2d(jnp.asarray(p_lin)))
    kq = (ell[None, :] + 0.5) / chi[:, None]
    inside = (kq >= jnp.exp(log_k[0])) & (kq <= jnp.exp(log_k[-1]))
    kq_c = jnp.clip(kq, jnp.exp(log_k[0]), jnp.exp(log_k[-1]))
    pq = jnp.stack([jnp.exp(jnp.interp(jnp.log(kq_c[i]), log_k, log_p[i]))
                    for i in range(chi.size)])
    return jnp.sum(jnp.where(inside, pq, 0.0) * wl[None, :], axis=1) / chi ** 2
