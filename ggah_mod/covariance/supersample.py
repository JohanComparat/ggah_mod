r"""The super-sample covariance: the response to a mode larger than the survey.

.. math::

    {\rm Cov}^{\rm SSC}[X_i, Y_j] = \sigma_b^2\,
        \frac{\partial X_i}{\partial\delta_b}\,\frac{\partial Y_j}{\partial\delta_b}

(Takada & Hu 2013).  A density mode :math:`\delta_b` longer than the footprint
is a change of background inside it, and every statistic measured there
responds coherently; :math:`\sigma_b^2` is that mode's variance over the
survey window.

The response
------------

The halo-model response of Li, Hu & Takada (2014), as CCL writes it in
``halomod_Tk3D_SSC``, for one pair of components:

.. math::

    \frac{\partial P_{ab}}{\partial\delta_b} =
        \Big(\frac{47}{21} - \frac13\frac{d\ln P_{\rm L}}{d\ln k}\Big)
        P_{\rm L}\,I_a I_b
      + T_{1h}(k)\int dM\,\frac{dn}{dM}\,b(M)\,{\rm pair}_{ab}(k|M)
      - (b_a + b_b)\,P_{ab} .

The first term is growth and dilation of the two-halo term; the second is halo
sample variance, the one-halo term weighted by the bias of the halos that carry
it, with the same one-halo transition :math:`T_{1h}` as :math:`P^{1h}`; the last
is the response of a mean density that is measured inside the survey, and is
present only for a discrete component normalised by its own :math:`\bar n`.
There :math:`b_a = I_a(k\to0)`, the large-scale bias, where CCL takes the
scale-dependent :math:`I_a(k)` -- the mean density is a single number, and it
responds with a single bias.  A multi-component tracer is summed over
component pairs, as :func:`~ggah_mod.spectra.tracers.spectrum` sums its
spectrum.

The halo bias is the field's :math:`b(M)`.  A ``bias_weight`` override (a
generic channel; the scale-dependent shift of the unexported
:mod:`ggah_mod.sectors.png` is one) reaches the two-halo integrals but not the halo sample variance, whose
bias is that of the halos themselves.

The variance
------------

For a :math:`C_\ell` block the response is projected with the mode's variance
per unit distance, :math:`\sigma_b^2(\chi)` of
:func:`~ggah_mod.covariance.geometry.projected_sigma2_b`:

.. math::

    {\rm Cov}\big[C^{ab}_{\ell_1}, C^{cd}_{\ell_2}\big] = \int d\chi\,
        \frac{W_aW_bW_cW_d}{\chi^4}\,\sigma_b^2(\chi)\,
        \frac{\partial P_{ab}}{\partial\delta_b}\Big(\frac{\ell_1+\tfrac12}{\chi}\Big)
        \frac{\partial P_{cd}}{\partial\delta_b}\Big(\frac{\ell_2+\tfrac12}{\chi}\Big) .

For a slab statistic the variance is that of the footprint times the slab,
:func:`slab_sigma2_b`, exact in the radial direction at the multipoles where
Limber's approximation fails for a thin, wide slab.

The cross block of two footprints, one inside the other, takes the larger
one's :math:`\sigma_b^2`: the modes they share are the ones longer than the
larger footprint, over which the smaller one's window is already unity.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..cosmology import background
from ..spectra.pair import pair_1h
from ..spectra.pk import linear_spectrum, pk_cross, two_halo_amplitude
from ..spectra.spec import overlap_of
from ..spectra.tracers import build_weights, resolve
from ..spectra.transition import one_halo_transition
from .geometry import SurveyGeometry, projected_sigma2_b

__all__ = ["response", "slab_sigma2_b"]


def response(field, a, b, sectors: dict, params: dict, *, options,
             overlaps=None, transition_params=None, bnl_table=None):
    r""":math:`\partial P_{ab}/\partial\delta_b` on the field's grid, as two
    non-negative parts ``(gain, loss)`` with the response ``gain - loss``.

    Two parts because the response changes sign -- a biased tracer's
    mean-density term overtakes its growth on large scales -- and the
    real-space transforms of layer 5 interpolate in :math:`\log P`.  Each part
    transforms as a spectrum, and the difference is taken afterwards.

    The beyond-linear two-halo term responds with the same growth-and-dilation
    factor as the linear one: it is :math:`P_{\rm lin}` times a ratio that a
    long mode leaves unchanged at this order, which is the latitude CCL takes
    with its own non-linear two-halo spectrum.  ``bnl_table`` is built from
    the field when not passed.
    """
    from ..spectra.bnl import table_for, two_halo_correction
    table = bnl_table if bnl_table is not None else table_for(field, options)
    overlaps = overlaps or {}
    a, b = resolve(a), resolve(b)
    n_k = int(jnp.atleast_1d(field.k).shape[-1])
    p_lin = linear_spectrum(field, options)
    growth = 47.0 / 21.0 - jnp.gradient(jnp.log(p_lin), jnp.log(field.k)) / 3.0
    t1h = one_halo_transition(p_lin, field.k,
                              transition=options.one_halo_transition,
                              params=transition_params)
    gain = jnp.zeros(n_k)
    loss = jnp.zeros(n_k)
    for ca in a.components:
        wa = build_weights(ca, field, sectors, params)
        ia = two_halo_amplitude(field, wa, options)
        for cb in b.components:
            wb = wa if cb == ca else build_weights(cb, field, sectors, params)
            ib = ia if cb == ca else two_halo_amplitude(field, wb, options)
            key = (ca.population, cb.population)
            ov = overlaps.get(key) or overlaps.get(key[::-1]) or overlap_of(
                ca, cb, a_discrete=wa.discrete, b_discrete=wb.discrete)
            gain = gain + growth * p_lin * ia * ib
            if table is not None:
                gain = gain + growth * two_halo_correction(
                    field, wa, wb, table, options=options)
            pair = pair_1h(wa, wb, n_k, overlap=ov)
            if pair is not None:
                gain = gain + t1h * field.integrate(field.dndm * field.bias * pair,
                                                    axis=-1)
            mean = (ia[0] if wa.discrete else 0.0) + (ib[0] if wb.discrete else 0.0)
            if wa.discrete or wb.discrete:
                p_ab = pk_cross(field, wa, wb, overlap=ov, options=options,
                                bnl_table=table,
                                transition_params=transition_params).total
                loss = loss + mean * p_ab
    return gain, loss


def slab_sigma2_b(geometry: SurveyGeometry, cosmo, fields_at, options, *,
                  ell_exact: int = 40, n_chi: int = 192, k_max: float = 0.5,
                  k_per_period: int = 16, n_growth: int = 16):
    r""":math:`\sigma_b^2` of the footprint times its redshift slab.

    .. math::

        \sigma_b^2 = \frac{2}{\pi}\Big(\frac{\Omega}{V}\Big)^2
            \sum_\ell W_\ell \int dk\,k^2
            \Big[\int d\chi\,\chi^2 D(\chi)\,j_\ell(k\chi)\Big]^2
            P_{\rm L}(k, z_{\rm ref}),

    :math:`W_\ell` the mask power of
    :attr:`~ggah_mod.covariance.geometry.SurveyGeometry.mask_wl`.  Exact below
    ``ell_exact``, where a slab thin compared to its width breaks Limber's
    approximation -- by 38% for the LS10 slabs over :math:`f_{\rm sky} = 0.46`
    -- and Limber above, where the radial integral has converged to it (5% per
    multipole at 40, and a total within :math:`10^{-4}` of the exact sum).

    Everything that depends only on the geometry and the background -- the
    distances, :math:`j_\ell` and the wavenumber grid -- is built in numpy
    once; the parameters reach the result through :math:`P_{\rm L}` and the
    growth :math:`D(\chi)`, read from ``fields_at`` on ``n_growth`` nodes.  A
    flat background only: the radial functions of a curved one are
    hyperspherical.
    """
    from scipy import special

    if geometry.z_min is None:
        raise ValueError(f"{geometry.name} has no redshift slab")
    if geometry.mask_wl is None:
        projected_sigma2_b(geometry, 1.0, [1.0, 2.0], [1.0, 1.0])  # raises
    if float(getattr(cosmo, "Omega_k", 0.0)) != 0.0:
        raise NotImplementedError(
            "the exact slab variance uses spherical Bessel functions, which "
            "are the radial functions of a flat background only")
    wl = np.asarray(geometry.mask_wl, float)
    lo_ell = min(int(ell_exact), wl.size)

    # growth on a few nodes, from the linear spectrum's large-scale amplitude
    xg, wg = np.polynomial.legendre.leggauss(n_growth)
    half = 0.5 * (geometry.z_max - geometry.z_min)
    zg = half * (xg + 1.0) + geometry.z_min
    fields = tuple(fields_at(tuple(float(v) for v in zg)))
    k_f = fields[0].k
    p_g = jnp.stack([linear_spectrum(f, options) for f in fields])  # (Ng, Nk)
    ref = n_growth // 2
    d_g = jnp.sqrt(p_g[:, 0] / p_g[ref, 0])

    # exact part, static radial nodes
    chi_lo, chi_hi = (float(np.asarray(background.comoving_distance(z, cosmo))[0])
                      for z in (geometry.z_min, geometry.z_max))
    xc, wc = np.polynomial.legendre.leggauss(n_chi)
    chi = 0.5 * (chi_hi - chi_lo) * (xc + 1.0) + chi_lo
    wchi = 0.5 * (chi_hi - chi_lo) * wc
    z_of_chi = np.interp(chi, np.asarray(background.comoving_distance(
        jnp.asarray(np.linspace(geometry.z_min, geometry.z_max, 512)), cosmo)),
        np.linspace(geometry.z_min, geometry.z_max, 512))
    v_over_omega = float(np.sum(wchi * chi ** 2))
    # the squared radial integral oscillates with period pi / (chi_hi - chi_lo)
    dk = np.pi / (k_per_period * (chi_hi - chi_lo))
    k_log = np.geomspace(1e-4, 1e-3, 24, endpoint=False)
    k_lin = np.arange(1e-3, k_max + dk, dk)
    k = np.concatenate([k_log, k_lin])
    wk = np.gradient(k)
    ells = np.arange(lo_ell)
    a = special.spherical_jn(ells[:, None, None], k[None, :, None] * chi[None, None, :]) \
        * (wchi * chi ** 2)[None, None, :]                           # (L, Nk, Nchi)
    kernel = (2.0 / np.pi) * wl[:lo_ell, None] * (k ** 2 * wk)[None, :] \
        / v_over_omega ** 2                                            # (L, Nk)
    d_chi = jnp.interp(jnp.asarray(z_of_chi), jnp.asarray(zg), d_g)
    radial = jnp.einsum("lkc,c->lk", jnp.asarray(a), d_chi)
    p_ref = jnp.exp(jnp.interp(jnp.log(jnp.asarray(k)), jnp.log(k_f),
                               jnp.log(p_g[ref])))
    exact = jnp.sum(jnp.asarray(kernel) * radial ** 2 * p_ref[None, :])

    # Limber above ell_exact, on the growth nodes
    high = SurveyGeometry(geometry.name, geometry.area_sr, geometry.z_min,
                          geometry.z_max, np.concatenate([np.zeros(lo_ell),
                                                          wl[lo_ell:]]),
                          geometry.provenance)
    chi_g = background.comoving_distance(jnp.asarray(zg), cosmo)
    s2 = projected_sigma2_b(high, chi_g, k_f, p_g)
    dchi_dz = cosmo.hubble_distance / background.hubble_e(jnp.asarray(zg), cosmo)
    limber = jnp.sum(jnp.asarray(wg) * half * dchi_dz * chi_g ** 4 * s2) \
        / v_over_omega ** 2
    return exact + limber
