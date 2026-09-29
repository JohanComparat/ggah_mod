r"""Real-space projections: :math:`\xi`, :math:`w_p`, :math:`\Sigma`,
:math:`\Delta\Sigma`, :math:`w(\theta)`.

One implementation of each.  The predecessor has **four** ``w_p`` and **three**
:math:`\Delta\Sigma`, and they differ: one integrates the line of sight to
:math:`\chi = 300` Mpc/h, another to a survey :math:`\pi_{\max}`, and a third
adds a central point mass after the fact, outside the halo model.

Two of these are Hankel transforms and one is not
--------------------------------------------------

:math:`\xi`, :math:`\Sigma`, :math:`\Delta\Sigma` and :math:`w(\theta)` are
transforms of a spectrum, and :mod:`~ggah_mod.observables.transforms` does them
in one step each.

:math:`w_p(r_p;\pi_{\max})` is **not**.  It is

.. math::  w_p(r_p) = 2\int_0^{\pi_{\max}} \xi\big(\sqrt{r_p^2+\pi^2}\big)\,d\pi

with a *finite, survey-defined* :math:`\pi_{\max}` -- 40 to 100 Mpc/h in real
data, not infinity -- and the residual redshift-space distortions live in
exactly that truncation.  Letting :math:`\pi_{\max}\to\infty` turns it into a
clean transform and into a different statistic.  So :math:`\pi_{\max}` is a
**required argument with no default**: it is a property of the measurement, not
of the model, and the predecessor's habit of integrating to 300 Mpc/h and
calling the result :math:`w_p` is how a survey's number silently becomes a
modelling choice.

:math:`\Delta\Sigma` in one step, not four
-------------------------------------------

:math:`J_2` *is* what :math:`\bar\Sigma(<R) - \Sigma(R)` encodes, so
:func:`~ggah_mod.observables.transforms.pk_to_delta_sigma` takes it directly.
The predecessor's route -- :math:`\xi \to` Abel :math:`\to \bar\Sigma` by
cumulative trapezoid -- carries three error sources this does not: the
line-of-sight truncation, the interpolation onto :math:`\sqrt{R^2+\chi^2}`, and
a cumulative integral from :math:`R = 0` that needs :math:`\Sigma` below the
smallest tabulated radius.  It records one of them, a linear :math:`\chi` grid,
as a ten-fold overestimate.

That route is kept, in :func:`delta_sigma_via_abel`, as an **independent
cross-check** -- the disagreement between them is a measured row, not an
assumption -- and never as a second production path.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..backend import resolve_backend
from ..cosmology import constants as C
from ..numerics import interp_cubic
from .transforms import (
    cl_to_wtheta, pk_to_delta_sigma, pk_to_sigma, pk_to_xi,
)

__all__ = ["xi", "wp", "sigma", "delta_sigma", "delta_sigma_via_abel",
           "w_theta", "SIGMA_UNIT", "sigma_electrons", "tau_ksz",
           "compensated_aperture", "tau_ksz_cap", "CAP_OUTER",
           "SIGMA_THOMSON_CM2"]

#: :math:`(M_\odot/h)(\mathrm{Mpc}/h)^{-2} \to (M_\odot/h)(\mathrm{pc}/h)^{-2}`.
#:
#: The unit lensing measurements are quoted in.  One constant, named, because
#: the predecessor wrote ``1e-12`` inline at three call sites.
SIGMA_UNIT = 1.0e-12

#: Thomson cross-section [cm^2].  CODATA 2018.
SIGMA_THOMSON_CM2 = 6.6524587321e-25


def _spectrum(pk):
    """Accept a :class:`~ggah_mod.spectra.pk.PowerSpectrum` or ``(k, P)``."""
    if hasattr(pk, "k") and hasattr(pk, "total"):
        return pk.k, pk.total
    k, p = pk
    return jnp.asarray(k), jnp.asarray(p)


def xi(r, pk, *, backend=None):
    r""":math:`\xi(r)`, the 3D correlation function."""
    k, p = _spectrum(pk)
    return pk_to_xi(r, k, p, backend=backend)


def wp(rp, pk, *, pi_max, backend=None):
    r""":math:`w_p(r_p;\pi_{\max})`, in Mpc/h.

    Parameters
    ----------
    pi_max : float [Mpc/h]
        **Required.**  A survey property; see the module docstring.

    Notes
    -----
    :math:`\xi` is tabulated once on a log grid (``n_r_tab`` points from
    :math:`10^{-3}` to :math:`10^{2.5}` Mpc/h) and read off it at
    :math:`\sqrt{r_p^2+\pi^2}`, rather than transformed afresh at every
    :math:`(r_p,\pi)`: the transform is the expensive part and the tabulation
    is exact to the interpolation, which is :math:`C^1` cubic.  The table starts
    a decade below the smallest separation any data set asks for (LS10 reaches
    :math:`r_p = 0.0078` Mpc/h), so no :math:`r_p` is read off its end.

    The line of sight is integrated in :math:`u = {\rm asinh}(\pi/r_p)`, by
    :func:`_los_integral`: ``n_pi`` nodes per :math:`r_p`, linear in
    :math:`\pi` below :math:`r_p` and logarithmic above.

    **Until 0.9.7 the** :math:`\pi` **grid was linear**, ``n_pi`` nodes from 0 to
    :math:`\pi_{\max}` -- 0.196 Mpc/h apart at :math:`\pi_{\max} = 100` on
    ``ACCURATE``.  The integrand peaks at :math:`\pi \lesssim r_p`, so below
    :math:`r_p \approx 0.2` Mpc/h the peak fell between nodes and the trapezoid
    overestimated :math:`w_p`: by 1.9-3.9x at :math:`r_p = 0.011`, 1.1-1.45x at
    0.045 and 2-9 per cent at 0.089 Mpc/h for galaxy samples at
    :math:`\pi_{\max} = 100`, and by 31-75 per cent at 0.02 Mpc/h at the LS10
    :math:`\pi_{\max} = 67.4`.  The same grid was the whole of the
    ``ACCURATE``/``DIFFERENTIABLE`` ``w_p`` disagreement (512 against 256 nodes;
    ``tests/test_parity_budget.py``).
    """
    b = resolve_backend(backend)
    if pi_max is None:
        raise ValueError(
            "wp needs an explicit `pi_max` [Mpc/h].  There is no default "
            "because it is a property of the measurement, not of the model: "
            "real w_p data uses 40-100 Mpc/h, and integrating to 300 and "
            "calling the result w_p -- which the predecessor did -- turns a "
            "survey's number into a silent modelling choice.")
    k, p = _spectrum(pk)
    rp = jnp.atleast_1d(jnp.asarray(rp))

    r_tab = jnp.logspace(-3.0, 2.5, b.n_r_tab)
    xi_tab = pk_to_xi(r_tab, k, p, backend=b)
    return _los_integral(rp, r_tab, xi_tab, float(pi_max), b.n_pi)


def _los_integral(rp, r_tab, xi_tab, pi_max: float, n: int):
    r""":math:`2\int_0^{\pi_{\max}}\xi(\sqrt{r_p^2+\pi^2})\,d\pi` from a
    tabulated :math:`\xi`, in :math:`u = {\rm asinh}(\pi/r_p)`.

    With :math:`\pi = r_p\sinh u`, :math:`\sqrt{r_p^2+\pi^2} = r_p\cosh u
    \equiv r` and :math:`d\pi = r\,du`, so

    .. math::  w_p = 2\int_0^{{\rm asinh}(\pi_{\max}/r_p)} \xi(r)\,r\,du .

    The nodes are uniform in :math:`u`: linear in :math:`\pi` where
    :math:`\pi < r_p` and the integrand is flat, logarithmic beyond, where it
    falls as a power law -- one grid that resolves every :math:`r_p` with the
    same ``n``.  The integrand is even in :math:`u`, so the trapezoid rule's
    end correction at :math:`u = 0` vanishes; Gauss-Legendre does worse here,
    because the cubic :math:`\xi` interpolant is only :math:`C^1`.  Measured
    against a 32,000-node reference on a galaxy spectrum: 5.8e-7 at ``n = 512``,
    2.3e-6 at 256, 4.2e-5 at 128 (``tests/test_real_space.py``).

    The node positions depend on ``rp`` and ``pi_max`` but the shape does not,
    so this traces and differentiates like the grid it replaced.
    """
    u = jnp.arcsinh(pi_max / rp)[:, None] * jnp.linspace(0.0, 1.0, n)[None, :]
    r = rp[:, None] * jnp.cosh(u)                                # (Nrp, n)
    xi_of = interp_cubic(jnp.log(r), jnp.log(r_tab), xi_tab)
    return 2.0 * jnp.trapezoid(xi_of * r, u, axis=-1)


def _area_factor(comoving: bool, z, what: str):
    r"""``1`` for a comoving surface density, :math:`(1+z)^2` for a proper one.

    Shared by :func:`sigma`, :func:`delta_sigma` and
    :func:`delta_sigma_via_abel` because they are the same quantity under three
    routes: each is :math:`\bar\rho_m` (comoving) integrated against a comoving
    :math:`k`, so each carries the same conversion.  Writing it once is not
    only tidiness -- three copies of :math:`(1+z)^2` is three places for one to
    be dropped, and the resulting disagreement between the production
    :math:`\Delta\Sigma` and its own cross-check would read as a defect in the
    transform rather than in the units.

    Both misuses raise rather than being absorbed.  ``z`` missing when it is
    needed cannot be guessed; ``z`` supplied when it is not would be a
    parameter accepted and unread, which is the exact defect ``sigma`` carried
    before this.
    """
    if comoving and z is not None:
        raise ValueError(
            f"{what}() was given a `z` with comoving=True, where it would have "
            f"no effect: the comoving surface density does not depend on the "
            f"redshift the spectrum was evaluated at.  Either drop `z`, or "
            f"pass comoving=False to ask for the proper surface density it is "
            f"needed for.  It is refused rather than ignored because a "
            f"parameter that is accepted and unread is exactly the defect "
            f"`sigma` used to have.")
    if not comoving and z is None:
        raise ValueError(
            f"{what}(comoving=False) needs a `z`: the proper surface density "
            f"is the comoving one times (1 + z)^2, and this function has no "
            f"other way to learn the redshift -- the spectrum carries k and "
            f"the halo terms but no z, and `cosmo.rho_matter` is comoving by "
            f"construction.  Pass the redshift the spectrum was evaluated at.")
    return 1.0 if comoving else (1.0 + jnp.asarray(z)) ** 2


def sigma(rp, pk, cosmo, *, backend=None, comoving: bool = True, z=None):
    r""":math:`\Sigma(R)`, in :math:`(M_\odot/h)(\mathrm{pc}/h)^{-2}`.

    Comoving by default.  ``comoving=False`` returns the **proper** surface
    density and then ``z`` is required, because the conversion is a function of
    redshift and this function has no other way to learn one.

    What the conversion is, and what it is not
    ------------------------------------------

    :func:`~ggah_mod.observables.transforms.pk_to_sigma` integrates a comoving
    :math:`\bar\rho_m` against a comoving :math:`k`, so its answer is a mass per
    *comoving* area.  The same physical mass occupies a proper area smaller by
    :math:`(1+z)^2`, so

    .. math::  \Sigma_{\rm proper}(R) = (1+z)^2\,\Sigma_{\rm comoving}(R)

    **Only the area units change.  The radius does not.**  ``rp`` is the
    comoving projected radius on the way in and still labels the same comoving
    annulus on the way out; what differs is the area that annulus is divided
    by.  A caller who wants :math:`\Sigma` at a *proper* radius converts the
    abscissa themselves -- ``rp_comoving = rp_proper * (1 + z)`` -- before
    calling, rather than having this function move their radii under them.
    That distinction is the one worth being explicit about: the two readings
    differ by :math:`(1+z)^2` in the value *and* by :math:`(1+z)` in where it
    is evaluated, and nothing in a returned array says which was meant.

    Why ``z`` is refused when ``comoving=True``
    -------------------------------------------

    Because it would do nothing.  This function's previous defect was a
    ``comoving`` flag whose branches both read ``cosmo.rho_matter``, so the
    parameter was accepted and ignored; accepting a ``z`` that no arithmetic
    reads would be the same mistake in a new place.

    Parameters
    ----------
    rp : array [comoving Mpc/h]
    comoving : bool
        ``True`` (default) for a mass per comoving area, ``False`` for proper.
    z : float, optional
        **Required** when ``comoving=False``, **refused** otherwise.  May be
        traced: it enters as an arithmetic factor and nothing branches on it.
    """
    f = _area_factor(comoving, z, "sigma")
    k, p = _spectrum(pk)
    return pk_to_sigma(rp, k, p, cosmo.rho_matter,
                       backend=backend) * SIGMA_UNIT * f


def sigma_electrons(rp, pk, *, backend=None):
    r"""Projected electron mass column :math:`\Sigma_e(R)`
    [:math:`(M_\odot/h)(\mathrm{Mpc}/h)^{-2}`].

    .. math::  \Sigma_e(R) = \frac{1}{2\pi}\int dk\,k\,P_{eg}(k)\,J_0(kR)

    The primitive quantity; :func:`tau_ksz` is this times
    :math:`\sigma_T/\mu_e m_p` and the unit conversions, which is the form the
    measurements are quoted in.  Both exist because the two names appear in the
    literature for one thing, and having only the derived one would leave a
    reader converting back.

    **No** :math:`\bar\rho` **is multiplied in.**  See :func:`tau_ksz` for why,
    and for the ten orders of magnitude that follow from getting it wrong.
    """
    k, p = _spectrum(pk)
    return pk_to_sigma(rp, k, p, 1.0, backend=backend)


def tau_ksz(rp, pk, cosmo, z=0.0, *, backend=None):
    r"""Stacked kinetic-SZ optical depth :math:`\tau(R)`, dimensionless.

    .. math::

        \tau(R) = \frac{\sigma_T}{\mu_e m_p}\,(1+z)^2\,\Sigma_e(R),
        \qquad
        \Sigma_e(R) = \frac{1}{2\pi}\int dk\,k\,P_{eg}(k)\,J_0(kR)

    The *excess* electron column around the stacked objects, which is what a
    kinetic-SZ stack measures once the velocity weighting has been divided out.
    ``pk`` should be an **electron-galaxy cross-spectrum** -- the ``electrons``
    tracer of :mod:`~ggah_mod.spectra.tracers` against a galaxy sample -- for
    the reason :func:`delta_sigma` gives about its own argument: an
    auto-spectrum silently answers a different question.

    **No mean density is multiplied in, and that is the whole unit chain.**
    :func:`~ggah_mod.observables.transforms.pk_to_sigma` takes a
    :math:`\bar\rho` because the matter field's weight is
    :math:`M/\bar\rho_m`, i.e. dimensionless, so the transform of
    :math:`P_{mm}` needs restoring to a density.  The gas and ejecta sectors
    weight by **mass** instead, so :math:`P_{eg}` already carries it and
    passing a density again is a second factor of :math:`\bar\rho`.  Measured,
    doing so put :math:`\tau` at :math:`10^{6}` -- ten orders high, and a
    number large enough that no plot would have hidden it, which is the only
    reason it was cheap to find.

    **Not** :math:`C_\ell^{\rm kSZ}`.  The kinetic SZ signal is proportional to
    the line-of-sight *velocity*, so its angular power spectrum needs the
    velocity field and cannot be had from :math:`P_{eg}` alone.  The stacked
    profile can, because the velocity weighting is what the estimator divides
    out -- and it is the measurement that localises the ejected baryons rather
    than counting them.  Naming here the thing this function does not do is
    cheaper than someone inferring it from a wrong amplitude.

    The :math:`(1+z)^2` is two of the dispersion measure's three factors: the
    proper electron density goes as :math:`(1+z)^3` and the proper path length
    as :math:`(1+z)^{-1}`.  There is no third -- an optical depth is not
    redshifted the way an *observed* dispersion measure is.

    Parameters
    ----------
    rp : array
        Projected radius [comoving Mpc/h].
    pk : PowerSpectrum or (k, P)
        The electron-galaxy cross-spectrum, in
        :math:`(M_\odot/h)(\mathrm{Mpc}/h)^3`.
    cosmo : Cosmology
        For :math:`h`, which the comoving-to-proper area conversion carries.
    z : float
        Redshift of the stack.

    Returns
    -------
    array
        Dimensionless.  Measured for a ``zheng07`` sample at :math:`z = 0`:
        :math:`1.1\times10^{-4}` at :math:`R = 0.05` Mpc/h falling to
        :math:`1.6\times10^{-5}` at 3 Mpc/h, which is the range stacked
        measurements report.  The hot gas alone supplies 81% of that column at
        0.05 Mpc/h and 40% at 3 Mpc/h -- the ejected component taking over with
        radius is the thing a kSZ stack is used to see, and it is why the
        composite ``electrons`` tracer exists rather than ``gas:density``.

    Notes
    -----
    **This is a profile, and the measurements are apertures.**  Stacked kSZ is
    reported through compensated aperture photometry -- a disc minus a
    surrounding annulus -- which is not :math:`\tau(R)` at a radius but an
    integral of it against a filter, and the filter subtracts a background this
    function does not know about.  Comparing the two needs that convention
    applied, and it is not applied here.  Named rather than left for whoever
    first overlays this on a data point.
    """
    from ..sectors.gas import M_PROTON_G, M_SUN_G, MU_E

    sigma_e = sigma_electrons(rp, pk, backend=backend)     # (Msun/h)(Mpc/h)^-2
    # (Msun/h)(Mpc/h)^-2 -> electrons cm^-2: one solar mass is M_SUN_G/h grams
    # and one (Mpc/h)^2 is (MPC_CM/h)^2 cm^2, so the h's do not cancel.
    per_cm2 = (M_SUN_G / cosmo.h) / (C.MPC_CM / cosmo.h) ** 2 / (MU_E * M_PROTON_G)
    return SIGMA_THOMSON_CM2 * sigma_e * per_cm2 * (1.0 + z) ** 2


def delta_sigma(rp, pk, cosmo, *, backend=None, comoving: bool = True, z=None):
    r""":math:`\Delta\Sigma(R)`, in :math:`(M_\odot/h)(\mathrm{pc}/h)^{-2}`.

    The direct :math:`J_2` transform of :math:`P_{gm}`.  ``pk`` should be a
    galaxy-matter cross-spectrum: :math:`\Delta\Sigma` is the *excess* surface
    density around the lenses, and handing it an auto-spectrum silently answers
    a different question.

    Comoving by default; ``comoving=False`` needs a ``z`` and returns the
    proper excess surface density, :math:`(1+z)^2` times the comoving one.
    Same convention as :func:`sigma`, and the same limit: **only the area units
    change**.  ``rp`` is the comoving projected radius on the way in and still
    labels the same comoving annulus on the way out, so a caller wanting a
    proper abscissa converts it themselves.

    This is the one of the two most likely to be *fitted* -- it is what a
    lensing survey reports -- so a redshift-binned fit is exactly where the
    proper/comoving choice has to be stated rather than defaulted.  See
    :func:`_area_factor` for why both misuses raise.
    """
    f = _area_factor(comoving, z, "delta_sigma")
    k, p = _spectrum(pk)
    return pk_to_delta_sigma(rp, k, p, cosmo.rho_matter,
                             backend=backend) * SIGMA_UNIT * f


def delta_sigma_via_abel(rp, pk, cosmo, *, backend=None,
                         comoving: bool = True, z=None):
    r"""The same, the long way: :math:`\xi \to \Sigma \to \bar\Sigma-\Sigma`.

    **A cross-check, not a production path.**  Kept because two independent
    routes to one number is how this package measures rather than asserts, and
    because it is the route the predecessor's AUM comparison was made through.

    .. math::

        \Sigma(R) = 2\bar\rho_m\!\int_0^{\chi_{\max}}\!\xi\big(\sqrt{R^2+\chi^2}\big)d\chi,
        \qquad
        \bar\Sigma(<R) = \frac{2}{R^2}\int_0^R \Sigma(R')R'\,dR'

    The :math:`\chi` grid is hybrid log-plus-linear, and that is not a flourish:
    a purely linear grid with :math:`\Delta\chi \gg R_{\min}` overestimates
    :math:`\Sigma` at small :math:`R` by about ten times, which the predecessor
    records having discovered.

    **How well it agrees, measured.**  Against the closed-form BMO
    :math:`\Delta\Sigma` on ``ACCURATE``, this route is 0.08-0.22% off at
    :math:`R = 0.2`-2 Mpc/h (2.2e-3 of the peak) and 0.03-0.15% at 0.01-0.5,
    while :func:`delta_sigma` reaches :math:`2\times10^{-8}` and
    :math:`3\times10^{-6}` of the peak on the same radii -- two to five orders
    of magnitude apart.  The gap is this route's own error sources, which the
    transform does not carry: the line-of-sight truncation, the interpolation
    onto :math:`\sqrt{R^2+\chi^2}`, and a cumulative integral started below the
    smallest tabulated radius.  ``tests/test_real_space.py`` asserts the direct
    route is within 3e-4 of the peak and the gap at least twentyfold.  (The
    direct route was :math:`1.2\times10^{-4}` until the Ogata rule's reach was
    fixed; see :attr:`~ggah_mod.backend.Backend.hankel_h`.)  That gap is the
    argument for which one is production, and it is a number rather than a
    preference.

    **Through 0.9.8 the line of sight started at a fixed** :math:`10^{-3}`
    **Mpc/h**, whatever radii were asked for.  The missing
    :math:`\int_0^{10^{-3}}\xi\,d\chi` made this route read low by about
    :math:`10^{-3}/R`: 18% at :math:`R = 0.01`, 4.2% at 0.05, 1.3% at
    0.2 Mpc/h.  Those were the numbers quoted here, and they were attributed
    to the route rather than to the grid.  The :math:`\chi` grid, the
    :math:`\xi` table and the :math:`\bar\Sigma` integral now share one floor,
    three decades below the smallest ``rp``.
    """
    # The same convention as the production route, and it has to be: this
    # exists to be compared against `delta_sigma`, and a cross-check that could
    # only be run in comoving units would silently disagree by (1+z)^2 the
    # moment anyone asked either route for proper ones -- which reads as a
    # defect in the transform rather than in the units.
    f = _area_factor(comoving, z, "delta_sigma_via_abel")
    b = resolve_backend(backend)
    k, p = _spectrum(pk)
    rp = jnp.atleast_1d(jnp.asarray(rp))

    # Three decades below the smallest requested radius, not one, and one
    # floor for all three grids below.  Each integral starts at its grid's first
    # node and therefore *drops* everything inside it.  For the cumulative
    # Sigma_bar integral, with Sigma ~ 1/R, that missing annulus scales as
    # r_min: at one decade it costs 10% at R = 0.2 Mpc/h.  Measured: 1 decade
    # -> 0.90 of the closed form, 3 -> 0.986, 5 -> no further change.
    # The line of sight drops the same way: its log half used to start at a
    # fixed 1e-3 Mpc/h, which leaves out int_0^1e-3 xi dchi, about 1e-3/R of
    # Sigma.  Measured against BMO: -18% at R = 0.01, -4.2% at 0.05, -1.3% at
    # 0.2 and -0.6% at 2 Mpc/h; with the chi grid tied to the same floor
    # the error drops to 0.1% at every one of those radii.  The xi table
    # starts there too, so sqrt(R^2+chi^2) -- never below chi -- is never
    # read off its end, where `interp_cubic` would extrapolate a cubic.
    log_r_min = jnp.log10(jnp.min(rp)) - 3.0

    r_tab = jnp.logspace(log_r_min, 2.5, b.n_r_tab)
    xi_tab = pk_to_xi(r_tab, k, p, backend=b)

    half = b.n_chi // 2
    chi = jnp.sort(jnp.concatenate([
        jnp.logspace(log_r_min, jnp.log10(b.chi_max_los), half),
        jnp.linspace(1.0, b.chi_max_los, b.n_chi - half)]))

    # Four times `n_r_tab`, because this grid carries *two* jobs: it is the
    # abscissa of a cumulative trapezoid as well as an interpolation grid, and
    # the first is first-order accurate where the second is fourth.
    r_tab_out = jnp.logspace(log_r_min, jnp.log10(jnp.max(rp)), 4 * b.n_r_tab)
    r_of = jnp.sqrt(r_tab_out[:, None] ** 2 + chi[None, :] ** 2)
    sigma_tab = 2.0 * cosmo.rho_matter * jnp.trapezoid(
        interp_cubic(jnp.log(r_of), jnp.log(r_tab), xi_tab), chi, axis=-1)

    # Sigma_bar by a cumulative trapezoid of `Sigma R dR`, from the innermost
    # tabulated radius.  The r grid starts a decade below the smallest requested
    # R precisely so this integral is not started at a radius that matters.
    integrand = sigma_tab * r_tab_out
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid * jnp.diff(r_tab_out))])
    sigma_bar = 2.0 * cum / r_tab_out ** 2

    out = interp_cubic(jnp.log(rp), jnp.log(r_tab_out), sigma_bar - sigma_tab)
    return out * SIGMA_UNIT * f



#: The compensated aperture's outer radius, in units of the inner one.
#:
#: :math:`\sqrt2` and not a choice: it is what makes the disc and the annulus
#: equal in area, so the filter subtracts the local background with unit weight
#: and returns zero on any profile that is flat across the aperture.  A
#: different ratio is a different statistic, not a tuned one.
#:
#: ``math.sqrt`` and not ``jnp.sqrt``.  Written with the latter it evaluated at
#: import time in whatever dtype JAX was configured for and came out
#: **1.4142135381698608** -- float32, widened into a float64 Python float, with
#: nothing recording where the digits went.  That is the trap
#: :data:`~ggah_mod.observables.spec.DEFAULT_WTHETA_ELL` already documents, hit
#: a second time in the same layer within one module of the paragraph warning
#: about it.
CAP_OUTER = math.sqrt(2.0)


def compensated_aperture(r_ap, rp, profile, *, outer: float = CAP_OUTER):
    r"""Compensated aperture photometry of a projected profile.

    .. math::

        {\rm CAP}(R_{\rm ap}) = 2\pi\left[\int_0^{R_{\rm ap}}\!\!f(R)\,R\,dR
        \;-\;\int_{R_{\rm ap}}^{\sqrt2 R_{\rm ap}}\!\!f(R)\,R\,dR\right]

    **What a stacked kinetic-SZ measurement actually reports.**  A kSZ stack
    cannot quote :math:`\tau` at a radius: the signal sits under a primary-CMB
    fluctuation and a beam, and the estimator that removes both is this filter.
    The disc and the annulus have equal area by construction -- that is what the
    :math:`\sqrt2` buys -- so any component flat across the aperture cancels,
    which is the point: the CMB is flat on these scales and the halo is not.

    So :func:`tau_ksz` was a profile and the measurements were apertures, and
    comparing them needed a conversion nobody had written.  ``PLAN.md`` recorded
    that as the reason the statistic was *computable and not comparable*.

    Parameters
    ----------
    r_ap : array (Nap,)
        Aperture radii [comoving Mpc/h].  The **inner** radius; the annulus runs
        from here to ``outer * r_ap``.
    rp : array (Nr,)
        The grid ``profile`` is tabulated on, ascending.  It must reach
        ``outer * max(r_ap)`` or the annulus is truncated, and it should start
        well inside ``min(r_ap)`` because the disc integral runs from zero.
    profile : array (Nr,)
        :math:`f(R)` on ``rp`` -- a :math:`\tau(R)` or a :math:`\Sigma_e(R)`.

    Returns
    -------
    array (Nap,), in ``profile``'s units times :math:`(\mathrm{Mpc}/h)^2`.

    Notes
    -----
    Implemented as a cumulative trapezoid in :math:`R` interpolated at the two
    edges, so it is one pass over the grid however many apertures are asked
    for, and differentiable in all of ``profile``, ``rp`` and ``r_ap`` -- the
    last mattering because an aperture is a survey property a fit may want to
    marginalise over.

    The inner integral starts at ``rp[0]`` rather than at zero, so the disc is
    missing :math:`\int_0^{r_0}`.  That is the same approximation
    :func:`delta_sigma_via_abel` documents and it is bounded the same way: for a
    profile going as :math:`R^{-1}` the missing annulus scales as
    :math:`r_0`, so a grid starting a decade inside the smallest aperture costs
    about a per cent of it.  A grid that does not is the caller's error and this
    function cannot see it, which is why the requirement is written down here.
    """
    rp = jnp.asarray(rp)
    profile = jnp.asarray(profile)
    r_ap = jnp.atleast_1d(jnp.asarray(r_ap))

    integrand = profile * rp
    mid = 0.5 * (integrand[:-1] + integrand[1:])
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mid * jnp.diff(rp))])

    inner = jnp.interp(r_ap, rp, cum)
    outer_ = jnp.interp(outer * r_ap, rp, cum)
    # disc  = cum(R_ap);  annulus = cum(sqrt2 R_ap) - cum(R_ap)
    return 2.0 * jnp.pi * (2.0 * inner - outer_)


def tau_ksz_cap(r_ap, pk, cosmo, z=0.0, *, backend=None, rp=None,
                n_rp: int = 256, outer: float = CAP_OUTER):
    r"""Stacked kSZ optical depth through a compensated aperture.

    :func:`tau_ksz` evaluated on an internal grid and filtered by
    :func:`compensated_aperture` -- the form a measurement is quoted in, so a
    model prediction and a data point are finally the same quantity.

    The internal grid spans a decade inside the smallest aperture to
    ``outer * max(r_ap)``, for the reason
    :func:`compensated_aperture` gives about each end.  Pass ``rp`` to override
    it; the override is checked rather than trusted, because a grid that stops
    short truncates the annulus and returns a number that is too large in a way
    no plot shows.
    """
    r_ap = jnp.atleast_1d(jnp.asarray(r_ap))
    if rp is None and isinstance(r_ap, jax.core.Tracer):
        raise TypeError(
            "tau_ksz_cap builds its profile grid from the apertures, so it "
            "cannot do that when they are traced: a grid is a *static* choice "
            "-- the same reason `Statistic.x` is a tuple of floats and not an "
            "array -- and deriving one from a tracer would make the output "
            "shape depend on a value. Pass `rp=` explicitly and the result "
            "differentiates in `r_ap` through the interpolation, which is "
            "what `compensated_aperture` is for.")
    traced = isinstance(r_ap, jax.core.Tracer)
    if rp is None:
        need = float(outer) * float(jnp.max(r_ap))
        rp = jnp.geomspace(float(jnp.min(r_ap)) / 10.0, need, n_rp)
    else:
        rp = jnp.asarray(rp)
        # The reach check needs a concrete aperture.  With a traced one it is
        # skipped rather than approximated: the caller supplied the grid, and a
        # check that cannot run is better absent than guessed at.
        need = None if traced else float(outer) * float(jnp.max(r_ap))
        if need is not None and float(jnp.max(rp)) < need * (1.0 - 1e-9):
            raise ValueError(
                f"the profile grid reaches {float(jnp.max(rp)):.4g} Mpc/h and "
                f"the compensated aperture needs {need:.4g} -- "
                f"`outer` x the largest aperture. A short grid truncates the "
                f"annulus, which makes the filter subtract less background "
                f"than it should and returns a CAP that is too large.")
    return compensated_aperture(
        r_ap, rp, tau_ksz(rp, pk, cosmo, z, backend=backend), outer=outer)


def w_theta(theta, cl, ell, *, backend=None):
    r""":math:`w(\theta) = \frac{1}{2\pi}\int d\ell\,\ell\,C_\ell\,J_0(\ell\theta)`.

    ``theta`` in radians.
    """
    return cl_to_wtheta(theta, ell, cl, backend=backend)
