r"""Where the calibrated parts of the differentiable flavour are valid, as a traced value.

Two parts of the ``DIFFERENTIABLE`` flavour are fits over a box: the
``emu_pk`` linear spectrum and the polynomial for the drag redshift
(:func:`~ggah_mod.cosmology.drag.z_drag`).  A third, the ``emu_hmf``
recalibration of the mass function (``tinker08_csst``), is an option and is
checked only when asked for.  Called with concrete values, each raises outside
its box and names the offending axes.

**Under** ``jax.jit``, ``jax.grad`` **or** ``jax.vmap`` **those checks are
skipped**, because a Python ``raise`` cannot depend on a traced value.  Inside
a trace the fits do not return ``nan`` and do not clamp: they
*extrapolate*.  The network or the polynomial is evaluated at the requested
point and returns a finite value and a finite gradient, with no accuracy
guarantee.  ``tests/test_domain.py`` pins this.  A gradient-based sampler
(HMC, NUTS) therefore neither crashes nor sticks at a box edge: it walks
through it, and the posterior beyond the edge is the extrapolation's.

The functions here give the box as a traced value, so a log-posterior can
treat it as a prior boundary:

.. code-block:: python

    from ggah_mod.cosmology.domain import in_domain

    def log_post(theta):
        cosmo = PLANCK18.replace(Omega_m=theta[0], ln10A_s=theta[1])
        return jnp.where(in_domain(cosmo), log_like(cosmo), -jnp.inf)

A ``-inf`` log-density is rejected by the Metropolis step of HMC and NUTS in
``blackjax`` and ``numpyro``, so the chain stays inside the box without a
``nan`` in the gradient: ``jnp.where`` selects the value, and the gradient of
the rejected branch is not used.  This is the prior-boundary answer to "what
should a sampler do at the edge"; the other two are to extend the fit or to
fall back to the ``ACCURATE`` flavour, which has no box but no gradient either.
"""

from __future__ import annotations

import jax.numpy as jnp

from .parameters import Cosmology

__all__ = ["domain_margins", "in_domain"]


def _box_margin(value, lo, hi):
    """Signed distance inside ``[lo, hi]`` as a fraction of its width: >= 0 inside."""
    width = hi - lo
    return jnp.minimum(value - lo, hi - value) / width


def domain_margins(cosmo: Cosmology, z=0.0, *, pk: bool = True,
                   hmf: bool = False, drag: bool = True) -> dict:
    r"""``{name: margin}`` for every box axis, traced; a margin ``>= 0`` is inside.

    Each margin is the distance to the nearer bound in units of the axis's
    width, so ``-0.1`` is a tenth of the box beyond an edge.  The axes are those
    the eager checks use: ``emu_pk``'s eleven inputs, its redshift range and its
    ``w0 + wa < 0`` condition (``pk_*``); ``emu_hmf``'s eight (``hmf_*``); the
    drag fit's eight and its bound on the dark-energy fraction at the drag
    epoch (``drag_*``).  ``z`` may be an array; its extreme values are checked.
    Skip a part with ``pk=False`` or ``drag=False`` when the model does not use
    it (``drag`` without BAO).  ``hmf=True`` adds the box of ``emu_hmf``, for a
    model run with ``hmf_model="tinker08_csst"`` (the ``hmf`` extra); the default
    mass function, ``tinker08``, has no box.
    """
    out = {}
    if pk:
        from emu_pk import box as pk_box
        from emu_pk import grid as pk_grid
        from .power import GgahEmuPk

        theta = GgahEmuPk._params(None, cosmo)
        for i, name in enumerate(pk_box.PARAMS):
            lo, hi = pk_box.BOX[name]
            # The eager check's tolerance, which admits the degenerate split's
            # nu_r1, one ulp above 1/3 (see `power._nu_ratios`).
            out[f"pk_{name}"] = _box_margin(theta[i], lo, hi) + pk_box._SLACK
        zz = jnp.atleast_1d(jnp.asarray(z, dtype=float))
        out["pk_z"] = _box_margin(jnp.stack([zz.min(), zz.max()]),
                                  pk_grid.Z_MIN, pk_grid.Z_MAX).min()
        out["pk_w0_plus_wa"] = -(jnp.asarray(cosmo.w0) + jnp.asarray(cosmo.wa))
    if hmf:
        from emu_hmf import box as hmf_box
        from emu_hmf.target import theta_from_cosmology

        theta = theta_from_cosmology(cosmo)
        for i, name in enumerate(hmf_box.PARAMS):
            lo, hi = hmf_box.BOX[name]
            out[f"hmf_{name}"] = _box_margin(theta[i], lo, hi)
    if drag:
        from . import _zdrag_coefficients as K
        from .drag import dark_energy_fraction_at_drag

        h2 = cosmo.h ** 2
        values = {"omega_b": cosmo.Omega_b * h2, "omega_cdm": cosmo.Omega_cdm * h2,
                  "h": cosmo.h, "sum_mnu": cosmo.sum_mnu, "w0": cosmo.w0,
                  "wa": cosmo.wa, "Omega_k": cosmo.Omega_k, "T_cmb": cosmo.T_cmb}
        for name, (lo, hi) in K.BOX.items():
            out[f"drag_{name}"] = _box_margin(jnp.asarray(values[name]), lo, hi)
        f = dark_energy_fraction_at_drag(cosmo)
        out["drag_dark_energy_fraction"] = (K.F_DE_MAX - jnp.abs(f)) / K.F_DE_MAX
    return out


def in_domain(cosmo: Cosmology, z=0.0, **parts):
    """``True`` (traced) when ``cosmo`` is inside every box :func:`domain_margins` checks.

    Keyword arguments as for :func:`domain_margins`.
    """
    margins = domain_margins(cosmo, z, **parts)
    return jnp.all(jnp.stack([jnp.min(m) >= 0.0 for m in margins.values()]))
