r"""The Limber projection, and the loop it does not contain.

.. math::

    C_\ell^{ab} = \int d\chi\,\frac{W_a(\chi)W_b(\chi)}{\chi^2}\,
                  P_{ab}\Big(k = \frac{\ell+\tfrac12}{\chi},\, z(\chi)\Big)

The **extended** Limber relation, :math:`k = (\ell+\tfrac12)/\chi` rather than
:math:`\ell/\chi`: the half is a genuine :math:`O(\ell^{-2})` correction, and it
costs nothing.  The predecessor uses it in its :math:`C_\ell` and drops it in
the beam it applies to :math:`\Sigma_y` in the *same file*, so one analysis
carries both conventions.

Not a Python loop over redshift
-------------------------------

Every ``angular_cl_*`` in the predecessor builds its :math:`P(k,z)` stack as
``[self._pk_tables_gy(zi, ...) for zi in z_arr]``, optionally threaded.  That is
why :math:`\partial/\partial z` does not exist there: ``z`` is a concrete Python
float throughout.  Here the stack arrives as one ``(Nz, Nk)`` array and the
projection is two ``vmap``\ s and a ``trapezoid``; there is no loop in this
file.

Where the stack comes from
--------------------------

Not from here, and not from one call: a :class:`~ggah_mod.halos.field.HaloField`
is one epoch, because layers 2 to 4 read the redshift off the field rather than
carrying an axis for it.  Two routes build the rows, and
:func:`~ggah_mod.observables.spec.make_model` takes the builder as an argument
rather than choosing:

``make_fields(cosmo, backend, pk, z_array, ...)``
    one linear-P(k) call for the whole grid, then one field per redshift.  The
    only cheap route for CLASS or CAMB, whose solve is memoised on the redshift
    *tuple* -- 64 projection nodes asked for one at a time are 64 solves.
``jax.vmap(lambda zz: ...)``
    for a differentiable backend, mapping the whole ``z -> spectrum`` closure.
    Note *closure*: ``jax.vmap(make_field)`` alone returns a field with ``Nz``
    on the leading axis of every leaf, which layers 3 and 4 must not be handed
    -- they tell a ``(NM,)`` weight from a ``(Nk, NM)`` one by that axis.  This
    module docstring used to recommend exactly that, and it was wrong twice
    over: it also raised for ``cm_model="diemer19"``, whose growth rate was
    computed in numpy and so could not be traced in the redshift at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .transforms import _log_interp_with_tail, _safe_log, _signed_fallback

__all__ = ["c_ell", "limber_k"]


def limber_k(ell, chi):
    r""":math:`k = (\ell+\tfrac12)/\chi`, shape ``(Nell, Nz)``."""
    ell = jnp.atleast_1d(jnp.asarray(ell))
    chi = jnp.atleast_1d(jnp.asarray(chi))
    return (ell[:, None] + 0.5) / chi[None, :]


def c_ell(ell, k, pk_stack, kernel_a, kernel_b, *, beam_a=None, beam_b=None):
    r""":math:`C_\ell^{ab}`, shape ``(Nell,)``.

    Parameters
    ----------
    ell : array (Nell,)
    k : array (Nk,)
        The wavenumber grid ``pk_stack`` is given on.
    pk_stack : array (Nz, Nk)
        :math:`P_{ab}(k, z_i)`, one row per redshift of the kernels' grid.
    kernel_a, kernel_b : RadialKernel
        Must share a :math:`\chi` grid; a pair integrated on different grids has
        an inconsistent cross-covariance and nothing would say so.
    beam_a, beam_b : array (Nell,), optional
        :math:`B_\ell`, **one per leg** -- so an auto-spectrum built from one
        field passed twice carries :math:`B_\ell^2`, which is the rule the
        predecessor states and then breaks.

    Notes
    -----
    The spectrum is interpolated in log-log, :math:`C^1` cubic, at every
    :math:`(\ell, z)`, and **continued as a power law beyond the k grid** by
    the same rule the Hankel transforms use.  Both halves matter: a clamp would
    put a flat spectrum where there should be a falling one, and a bare cubic
    extrapolation -- which is what a plain interpolator does past its last knot
    -- is a cubic in :math:`\log P` and diverges.
    """
    chi = jnp.asarray(kernel_a.chi)
    if kernel_b.chi.shape != chi.shape:
        raise ValueError(
            f"kernels {kernel_a.name!r} and {kernel_b.name!r} are on different "
            f"chi grids ({chi.shape} against {kernel_b.chi.shape}).  Build one "
            f"grid with `limber_grid` and give it to both: a pair projected on "
            f"different grids has an inconsistent cross-covariance, and nothing "
            f"downstream can tell.")

    log_k = jnp.log(jnp.asarray(k))
    log_pk = _safe_log(jnp.asarray(pk_stack))              # (Nz, Nk)
    log_kl = jnp.log(limber_k(ell, chi))                   # (Nell, Nz)

    # One redshift's row, read at that redshift's k.  vmap over z, then over
    # ell: no Python loop anywhere, so d/dz exists.
    #
    # `_log_interp_with_tail`, not a bare `interp_cubic`.  At high ell and small
    # chi, k = (ell+1/2)/chi leaves the grid -- at ell = 5000 and z = 1e-3 it is
    # already 1700 h/Mpc against a grid ending at 200 -- and a cubic Hermite
    # extrapolated a decade past its last knot is a *cubic in log P*, which
    # blows up.  Measured, with the bare interpolation: C_l^gg came out 4e22 at
    # ell = 5000 while ell = 100 and 1000 were correct, which is exactly the
    # shape of a bug that a plot of the first two points would have missed.
    #
    # And a spectrum that changes sign -- a cross power spectrum with a hollow
    # profile, e.g. an X-ray emissivity whose temperature rises outward through
    # the steep low-T part of Lambda(T) -- takes the cubic of the *value* next
    # to its non-positive nodes (`transforms._signed_fallback`), as `hankel`
    # does.  Measured on L3-5's M*>11.0 vector at a merged tSZ/X-ray point:
    # P_g,gas went negative at the top of the k grid in every node and w(theta)
    # came out at +-1e14.  A positive spectrum is unchanged, bit-identical.
    pk_raw = jnp.asarray(pk_stack)

    def one_z(log_k_query, row, raw):
        v = jnp.exp(_log_interp_with_tail(log_k_query, log_k, row,
                                          slope_cap=-3.0))
        return _signed_fallback(v, log_k_query, log_k, raw)

    over_z = jax.vmap(one_z, in_axes=(0, 0, 0))            # (Nz,)
    over_ell = jax.vmap(over_z, in_axes=(0, None, None))   # (Nell, Nz)
    pk = over_ell(log_kl, log_pk, pk_raw)

    integrand = (jnp.asarray(kernel_a.w) * jnp.asarray(kernel_b.w)
                 / chi ** 2)[None, :] * pk
    out = jnp.trapezoid(integrand, chi, axis=-1)

    for beam in (beam_a, beam_b):
        if beam is not None:
            out = out * jnp.asarray(beam)
    return out
