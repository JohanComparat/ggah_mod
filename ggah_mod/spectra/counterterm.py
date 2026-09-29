r"""The low-mass deficit, and why the two-halo term needs one.

The halo model's two-halo term is :math:`P_{\rm lin}(k)I_a(k)I_b(k)` with

.. math::  I_a(k) = \int dM\,\frac{dn}{dM}\,b(M)\,W_a(k|M)

For the matter field, :math:`W_m = (M/\bar\rho_m)\tilde u`, so

.. math::

    I_m(k\to0) = \frac{\bar\rho_{cb}}{\bar\rho_m}
        \underbrace{\int dM\,\frac{dn}{dM}\,b(M)\,\frac{M}{\bar\rho_{cb}}}_{\to 1}

The bracket is the bias-consistency integral, and it equals one only when the
mass range covers every halo.  **Measured on the shipped grid** -- the
``ACCURATE`` flavour (CLASS, 512 mass nodes to :math:`10^{16}`), 200m,
``tinker08_csst`` x ``tinker10``, PLANCK18, :math:`z = 0`, with ``k_max``
widened to 800 at ``m_min = 1e6`` as ``check_k_support`` requires:

Columns are :math:`\int b\,(M/\bar\rho_{cb})\,dn/dM\,dM` -- the bracket above,
:func:`bias_consistency` -- and its unweighted partner
:math:`\int (M/\bar\rho_{cb})\,dn/dM\,dM`, which is
:func:`~ggah_mod.halos.linear_bias.mass_fraction`:

=========  =============  ==========
``m_min``  bias-weighted  unweighted
=========  =============  ==========
1e10       **0.7154**     0.5248
1e8        0.7762         0.6231
1e6        0.8145         0.6873
=========  =============  ==========

``DIFFERENTIABLE`` (``emu_pk``, 256 nodes, the same pairing) agrees to the
fourth decimal: 0.7154/0.5247, 0.7761/0.6230, 0.8143/0.6871.  The table used to
read 0.7020/0.5185 at ``1e10``, overshooting to 1.0648/1.1151 at ``1e6``; that
was plain ``tinker08`` evaluated at 200 whatever the field's mass definition,
which is neither the shipped pairing nor anything the package computes now
(``tests/test_spectra.py`` records the history).

So an uncorrected :math:`P_{mm}^{2h}` is short by :math:`0.715^2 = 0.51` -- **a
factor of two** -- at every k.  And widening the mass range is not the fix: at
``m_min = 1e6`` the integral has reached only 0.81, in a regime where the
multiplicity function was never calibrated, and the ``k`` grid has had to grow
with it.

Why nothing caught this before
------------------------------

The predecessor did not have the problem because it did not do the integral:
``forward_jax`` sets :math:`P_{mm}^{2h} = P_{\rm lin}` outright and the galaxy
leg uses :math:`b_{\rm eff}` with :math:`\tilde u \to 1` -- which is precisely
the "2-halo hack" that :mod:`~ggah_mod.sectors.matter`'s docstring calls out and
removes.  Removing the hack without supplying the correction leaves the deficit
in the open, where it looks like a physics result.

Layer 2 anticipated it exactly.
:func:`~ggah_mod.halos.linear_bias.mass_weighted_bias` says how close the
integral gets "is a property of the (mass function, bias) *pairing* and of the
mass range integrated over; both matter, and both are the caller's choice, so
this reports rather than asserts."  **Layer 4 is that caller.**

The correction
--------------

Schmidt (2016); the same idea in HMcode and CCL.  The mass missing from the grid
is put back at :math:`M_{\min}` with :math:`b = 1`:

.. math::

    \Delta I_a(k) = \Big[1 - \int dM\,\frac{dn}{dM}\,b\,\frac{M}{\bar\rho_{cb}}\Big]
                    \,W_a(k|M_{\min})\,\frac{\bar\rho_{cb}}{M_{\min}}

For the matter field this makes :math:`I_m(k\to0) = \bar\rho_{cb}/\bar\rho_m`
identically -- **not** 1, and that is the correct cold-halo-model answer:
:math:`b(M)` is defined against the field whose variance set it, which is the
cold one, so the halo integral gives :math:`P_{mm}^{2h} \to (1-f_\nu)^2P_{cb}`.
The remaining neutrino terms are additive and linear, and
:mod:`~ggah_mod.spectra.neutrinos` supplies them as a leg whose constant is
this same :math:`\bar\rho_{cb}/\bar\rho_m` -- which is why, with both on,
:math:`P_{mm}^{2h} \to P_m` exactly.

It applies to **continuous** tracers only.  It extrapolates a tracer's weight
*per unit halo mass* below the grid, which is a statement about a field that
pervades halos; a discrete tracer's count is not a mass and there is no reason a
threshold sample continues below its own threshold.  That is an assumption for
gas and pressure as much as for matter -- the sub-grid halos really do hold gas,
but this says their gas looks like the smallest resolved halo's -- and it is why
the whole thing is switchable rather than always on.

How much of this depends on :math:`M_{\min}`, measured
-------------------------------------------------------

The question is fair -- both the deficit and the stand-in weight
:math:`W_a(k|M_{\min})` are read off the grid's edge -- and it has two answers,
not one.

**The normalisation does not depend on it at all.**  That is by construction and
it is exact: whatever the deficit is, adding it back at :math:`M_{\min}` with
:math:`b = 1` puts :math:`I_m(k\to0)` at :math:`\bar\rho_{cb}/\bar\rho_m`
identically.  Measured on the differentiable flavour at PLANCK18, z = 0, with
``k_max`` widened as ``check_k_support`` requires:

=========  =====  ===========  ==========  ==============
``m_min``  k_max  consistency  I_m(0) raw  I_m(0) + Delta
=========  =====  ===========  ==========  ==============
1e10       200    0.71543      0.712157    0.9954201720
1e8        200    0.77616      0.772603    0.9954201720
1e6        800    0.81437      0.810643    0.9954201720
=========  =====  ===========  ==========  ==============

against :math:`\bar\rho_{cb}/\bar\rho_m = 0.9954201720`.  Four decades of
:math:`M_{\min}`, agreeing to **4.3e-10** -- which is the quadrature, not the
prescription.

**The shape depends on it, and hardly.**  :math:`W_a(k|M_{\min})` is a real
grid dependence because the smallest resolved halo's profile is what stands in
for everything below it, and a smaller halo is more concentrated.  Relative to
the ``m_min = 1e10`` grid, :math:`I_m(k)` moves by 4.9e-6 at
:math:`k = 0.1`, 5.5e-4 at :math:`k = 1` and 1.2e-3 at :math:`k = 5` h/Mpc.
Twelve parts in ten thousand at the small-scale end, over four decades of mass.

So the correction is converged in the sense that matters, and the honest
statement of what it is conditional on is the second table rather than the
first.

**The census and this counterterm book one object, with two weights.**  The
census puts :math:`1 - F = 0.48` of the cold mass outside the grid; this puts
back :math:`1 - B = 0.28` of the *bias-weighted* mass.  They are not the same
number because the missing matter is not unbiased: its effective bias is
:math:`b_u = (1-B)/(1-F) = 0.60`
(:func:`~ggah_mod.halos.linear_bias.unresolved`), the smooth component of
Asgari, Mead & Heymans (2023, Sec. 5.1) written in the two numbers that fix it.
What the object is *made of* is declared once and read by both: a tracer's
:attr:`~ggah_mod.sectors.protocol.TracerWeights.w_unresolved` here, and
:attr:`~ggah_mod.sectors.census.BaryonCensus.omega_outside_by_phase` there --
the stars and neutral gas the smallest resolved halo holds, and every other
baryon as diffuse, ionised gas, which the hot-gas views declare zero for and
the ejecta sector carries.  For the matter field the composition does not
matter at :math:`k\to0` (the parts sum to one), so the stand-in is kept.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["bias_consistency", "mass_deficit", "low_mass_counterterm"]


def bias_consistency(field):
    r""":math:`\int dM\,(dn/dM)\,b(M)\,M/\bar\rho_{cb}`.  Should be 1.

    Reported, not asserted -- it is 0.715 on the shipped grid, and how far it
    misses is information about the (mass function, bias, mass range) triple
    rather than an error to hide.

    **It reads ``field.bias`` unconditionally, where**
    :func:`~ggah_mod.spectra.pk.i_of_k` **honours a tracer's ``bias_weight``
    override.  That asymmetry is deliberate and it is the correct way round.**
    This integral is the statement *the mass-weighted bias of all matter is one*
    -- a property of the (mass function, bias) pairing and of the mass range,
    and of nothing else.  A bias decoration re-weights one tracer's
    bias at fixed halo mass; it does not change how much bias-weighted **mass**
    the grid is missing, and there is no reason for a decorated
    :math:`\int b_a(M)(M/\bar\rho_{cb})\,dn/dM\,dM` to equal one.  Reading
    the override here would replace a known limit with an unknown one.

    Today the two cannot even meet: no shipped sector sets ``bias_weight`` on
    its own weights -- the one override,
    :func:`ggah_mod.sectors.png.apply_png_bias`, an unexported extension, is
    applied by the caller --
    and :func:`low_mass_counterterm` returns zero for a discrete tracer.  That is a contingent fact about which sectors decorate,
    not a structural one, so ``tests/test_spectra.py`` pins both halves: that
    the deficit ignores a decoration, and that the pair is unreachable by the
    shipped sectors.
    """
    return field.integrate(field.dndm * field.bias * field.m) / field.rho_cold


def mass_deficit(field):
    r""":math:`1 - ` :func:`bias_consistency`: the bias-weighted mass missing.

    Not clipped at zero, and clipping would be a **defect** rather than a
    tidy-up.  A (mass function, bias) pair integrated over a wide enough range
    can **overshoot** -- this docstring used to quote 1.065 for plain
    ``tinker08`` x ``tinker10`` at ``m_min = 1e6``, a configuration that
    ignored the mass definition and is gone -- and there a negative deficit is
    the honest statement that the pair is not consistent.  The counterterm's whole guarantee is that
    :math:`I_m(k\to0) = \bar\rho_{cb}/\bar\rho_m` *identically*; a floor at
    zero would hold in exactly the configurations where the integral overshoots
    and would leave the limit wrong by the amount it refused to subtract.  So
    the sign is carried, and the overshoot is reported by
    :func:`bias_consistency` rather than absorbed here.

    Whether the overshoot happens at all is a property of the pairing, not of
    the mass range alone: on the pairing both flavours ship (``tinker08_csst``
    x ``tinker10`` at 200m) the consistency integral rises to 0.814 at
    ``m_min = 1e6`` and does not cross 1.
    """
    return 1.0 - bias_consistency(field)


def low_mass_counterterm(field, w, n_k: int):
    r""":math:`\Delta I_a(k)`, shape ``(Nk,)``, for a continuous tracer.

    Zero for a discrete one, by return rather than by branch at the call site.
    The tracer's declared
    :attr:`~ggah_mod.sectors.protocol.TracerWeights.w_unresolved` is used when
    it has one; otherwise the smallest resolved halo stands in.
    """
    if w.discrete:
        return jnp.zeros((n_k,))
    if w.w_unresolved is not None:
        return mass_deficit(field) * jnp.broadcast_to(
            jnp.asarray(w.w_unresolved), (n_k,))
    from .pair import normalised_parts, _total

    point, ext = normalised_parts(w, n_k)
    total = _total(point, ext)                       # (Nk, NM) = W_a(k|M)
    if total is None:
        return jnp.zeros((n_k,))
    # The smallest resolved halo stands in for everything below it.
    w_at_min = total[:, 0]                           # (Nk,)
    return mass_deficit(field) * w_at_min * field.rho_cold / field.m[0]
