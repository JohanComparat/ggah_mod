r"""Derived amplitudes: :math:`\sigma_8`, :math:`S_8`, and the top-hat variance.

:math:`\sigma_8` is an **output** of this package.  The amplitude is
:math:`\ln 10^{10}A_s`, carried by
:class:`~ggah_mod.cosmology.parameters.Cosmology`; everything here reads a
spectrum and reports a number about it.

We read :math:`\sigma_8` off rather than take it as an input because an input
:math:`\sigma_8` needs something to renormalise the spectrum to it, and with
several P(k) backends that renormalisation gets written more than once, in more
than one convention.

:func:`sigma2_tophat` is the single implementation of the top-hat variance.
Nothing else in the package integrates :math:`P(k)W^2(kR)k^2`: the halo mass
function calls this function with a mass-dependent radius, and
:mod:`ggah_mod.cosmology.growth` calls it at :math:`R=8`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import constants as C

__all__ = ["tophat_window", "sigma2_tophat", "sigma_tophat", "sigma_v",
           "sigma8", "s8", "ln10A_s_for_sigma8"]


def tophat_window(x):
    r"""Fourier transform of a real-space top-hat, :math:`W(x) = 3(\sin x - x\cos x)/x^3`.

    Series-expanded near zero, where the closed form is a catastrophic
    cancellation: at :math:`x \sim 10^{-4}` the numerator loses about twelve
    significant digits, and in float32 the result is noise.  The expansion
    :math:`1 - x^2/10 + x^4/280` is accurate to machine precision below
    :math:`x = 0.1`, the large-scale end of every :math:`\sigma(M)` integral.
    """
    x = jnp.asarray(x)
    small = x < 0.1
    x_safe = jnp.where(small, 1.0, x)          # keep the unused branch finite
    exact = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / x_safe ** 3
    series = 1.0 - x ** 2 / 10.0 + x ** 4 / 280.0
    return jnp.where(small, series, exact)


def sigma2_tophat(pk, k, R):
    r""":math:`\sigma^2(R) = \frac{1}{2\pi^2}\int P(k)\,W^2(kR)\,k^2\,dk`.

    Integrated in :math:`\ln k` by the trapezoid rule, suited to a log-spaced
    grid and a smooth integrand.

    Parameters
    ----------
    pk : array, shape (Nk,)
        :math:`P(k)` [(Mpc/h)^3] on the grid ``k``.
    k : array, shape (Nk,)
        Wavenumbers [h/Mpc], ascending and log-spaced.
    R : float or array, shape (NR,)
        Top-hat radius/radii [Mpc/h].

    Returns
    -------
    Scalar if ``R`` is scalar, else shape ``(NR,)``.

    Raises
    ------
    ValueError
        If ``pk`` is not one-dimensional.  Handed a ``(Nz, Nk)`` stack,
        ``pk[None, :]`` would become ``(1, Nz, Nk)`` and the ``axis=1``
        trapezoid would integrate **over redshift**, returning a smooth,
        positive, correctly shaped number that is not a variance.  Everything
        the halo layer builds from a spectrum comes through here
        (:func:`~ggah_mod.halos.variance.sigma_of_mass`,
        :func:`~ggah_mod.halos.variance.dln_sigma_dln_mass`), so this one check
        covers them all.  The shape is static, so the check costs nothing under
        ``jit`` and is never skipped under trace.
    """
    k = jnp.asarray(k)
    pk = jnp.asarray(pk)
    R = jnp.asarray(R)
    if pk.ndim != 1:
        raise ValueError(
            f"sigma2_tophat takes one spectrum, shape (Nk,); got {pk.shape}.  "
            f"A stack of spectra over redshift is not a valid argument here: "
            f"the k integral would run over the wrong axis and come back "
            f"looking like a variance.  Layers 2 to 4 of this package are one "
            f"epoch per object -- pass one row, and stack the results.")
    scalar = R.ndim == 0
    R = jnp.atleast_1d(R)
    w = tophat_window(k[None, :] * R[:, None])
    integrand = pk[None, :] * w ** 2 * k[None, :] ** 3      # extra k from dlnk
    out = jnp.trapezoid(integrand, jnp.log(k), axis=1) / (2.0 * jnp.pi ** 2)
    return out[0] if scalar else out


def sigma_tophat(pk, k, R):
    r""":math:`\sigma(R)`."""
    return jnp.sqrt(sigma2_tophat(pk, k, R))


def sigma_v(pk, k):
    r""":math:`\sigma_v^2 = \frac{1}{3}\int \frac{dk}{2\pi^2}\,P(k)`, in Mpc/h.

    The **1D** linear-theory displacement dispersion.  The Zel'dovich
    displacement is :math:`\psi(\mathbf k) = i\mathbf k\delta/k^2`, so

    .. math::

        \langle\psi^2\rangle = \int\frac{d^3k}{(2\pi)^3}\frac{P(k)}{k^2}
                             = \frac{1}{2\pi^2}\int dk\,P(k)

    in three dimensions, and this is a third of it.  The factor of three is 1.7
    in :math:`\sigma_v`, and HMcode-2015's one-halo damping scale,
    :math:`k_* = 0.584/\sigma_v`, was fitted with this convention.  This
    package's damping is HMcode-2020's, which reads :math:`\sigma_8` instead,
    so the function is kept as a layer-1 moment rather than for a consumer.

    Integrated in :math:`\ln k` by the trapezoid rule, as
    :func:`sigma2_tophat` is, so the two moments of one spectrum share one
    convention.

    It lives here, beside the other moments of :math:`P(k)`, rather than in
    :mod:`ggah_mod.spectra.transition`, its former consumer: a moment of the
    linear spectrum is a layer-1 quantity whichever layer wants it, and
    ``tests/test_spectra_paths.py`` refuses a ``trapezoid`` in layer 4.

    **The grid is part of the answer.**  :math:`\int dk\,P(k)` converges at both
    ends, as :math:`k` for :math:`P \sim k^{n_s}` below the turnover and as
    :math:`k^{-2}` above it, so a finite grid is legitimate rather than a
    truncation, but it is not exact.  ``HaloField.k`` is guarded by
    :func:`~ggah_mod.halos.variance.check_k_support` for the range
    :math:`\sigma(M)` needs, which is wider than this integral needs, so the
    guard covers this consumer too.

    Parameters
    ----------
    pk : array, shape (Nk,)
        :math:`P(k)` [(Mpc/h)^3] on ``k``.
    k : array, shape (Nk,)
        Wavenumbers [h/Mpc], ascending and log-spaced.

    Raises
    ------
    ValueError
        If ``pk`` is not one-dimensional, the boundary
        :func:`~ggah_mod.cosmology.amplitude.sigma2_tophat` draws, for the
        same reason: handed a ``(Nz, Nk)`` stack the trapezoid would integrate
        over redshift and return a positive, correctly shaped number that is not
        a dispersion.  Layers 2 to 4 are one epoch per object.
    """
    k = jnp.asarray(k)
    pk = jnp.asarray(pk)
    if pk.ndim != 1:
        raise ValueError(
            f"sigma_v takes one spectrum, shape (Nk,); got {pk.shape}.  A "
            f"stack over redshift is not a valid argument here: the k integral "
            f"would run over the wrong axis and come back looking like a "
            f"dispersion.  Pass one row, and stack the results.")
    integral = jnp.trapezoid(pk * k, jnp.log(k))      # extra k from dlnk
    return jnp.sqrt(integral / (6.0 * jnp.pi ** 2))


def sigma8(pk, k):
    r""":math:`\sigma_8` from a spectrum.

    Pass the **total**-matter spectrum: :math:`\sigma_8` conventionally refers
    to all the matter, and priors quoted in the literature assume that.  The
    cold-field variance is a different number and belongs to the halo sector.
    """
    return sigma_tophat(pk, k, C.R8)


def s8(pk, k, cosmo):
    r""":math:`S_8 = \sigma_8\sqrt{\Omega_m/0.3}`."""
    return sigma8(pk, k) * jnp.sqrt(cosmo.Omega_m / C.S8_PIVOT_OMEGA_M)


def ln10A_s_for_sigma8(target_sigma8: float, cosmo, pk_backend, k=None,
                       tol: float = 1e-8, max_iter: int = 40) -> float:
    r"""Solve for the :math:`\ln 10^{10}A_s` that yields ``target_sigma8``.

    The migration helper, and the only supported way to start from a
    :math:`\sigma_8`.  Call it **once, up front**, and carry the amplitude
    afterwards: a cosmology holding both is over-determined and is refused.

    Exact in one step up to the linearity of :math:`P \propto A_s`: since
    :math:`\sigma_8^2 \propto A_s`, one evaluation fixes the answer, and the
    iteration only absorbs any residual non-linearity a backend introduces.
    """
    import numpy as np

    k = np.logspace(-4, 2, 512) if k is None else np.asarray(k)
    lnA = float(cosmo.ln10A_s)
    for _ in range(max_iter):
        c = cosmo.replace(ln10A_s=lnA)
        got = float(sigma8(np.asarray(pk_backend.pk(k, 0.0, c)), k))
        if abs(got / target_sigma8 - 1.0) < tol:
            return lnA
        lnA += 2.0 * np.log(target_sigma8 / got)    # sigma8^2 ~ A_s
    raise RuntimeError(
        f"ln10A_s_for_sigma8 did not converge: wanted {target_sigma8}, "
        f"reached {got} after {max_iter} iterations")
